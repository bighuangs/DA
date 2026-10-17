import asyncio
import unittest
from types import SimpleNamespace

from app.infrastructure.clients.embedding.scheduler import (
    EmbeddingCapacityError,
    EmbeddingDeadlineExceeded,
    EmbeddingRuntimeMetrics,
    MicroBatchEmbeddingScheduler,
)


def _config(**overrides):
    values = {
        "queue_capacity": 16,
        "batch_window_ms": 10,
        "max_batch_texts": 8,
        "max_batch_tokens": 64,
        "request_deadline_ms": 1000,
        "online_query_priority": 0,
        "offline_corpus_priority": 10,
        "offline_corpus_quota": 2,
        "shutdown_drain_timeout_ms": 1000,
        "metrics_window_size": 64,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class EmbeddingRuntimeMetricsEpochTests(unittest.TestCase):
    def test_reset_excludes_slow_warmup_but_counts_suite_sample(self):
        metrics = EmbeddingRuntimeMetrics("test", window_size=64)
        metrics.submitted_request(1)
        metrics.record_batch(
            request_count=1,
            text_count=1,
            queue_wait_ms=[2.0],
            inference_ms=8000.0,
            total_ms=[8002.0],
        )
        self.assertEqual(
            metrics.snapshot()["latency_ms"]["inference"]["p95"],
            8000.0,
        )

        reset = metrics.reset_epoch(reason="evaluation_repeat_1_suite_boundary")
        after_reset = metrics.snapshot()
        self.assertEqual(after_reset["throughput"]["model_invocations"], 0)
        self.assertEqual(after_reset["latency_ms"]["inference"]["p95"], 0.0)
        self.assertNotEqual(
            reset["previous_epoch"]["id"],
            reset["active_epoch"]["id"],
        )

        metrics.submitted_request(1)
        metrics.record_batch(
            request_count=1,
            text_count=1,
            queue_wait_ms=[1.0],
            inference_ms=25.0,
            total_ms=[26.0],
        )
        suite = metrics.snapshot()
        self.assertEqual(suite["throughput"]["model_invocations"], 1)
        self.assertEqual(suite["latency_ms"]["inference"]["p95"], 25.0)

    def test_three_repeat_epochs_do_not_accumulate_prior_latency(self):
        metrics = EmbeddingRuntimeMetrics("test", window_size=64)
        metrics.record_batch(
            request_count=1,
            text_count=1,
            queue_wait_ms=[0],
            inference_ms=8000,
            total_ms=[8000],
        )

        for repeat_index, latency in ((1, 31.0), (2, 42.0), (3, 53.0)):
            reset = metrics.reset_epoch(
                reason=f"evaluation_repeat_{repeat_index}_suite_boundary"
            )
            self.assertEqual(reset["active_epoch"]["index"], repeat_index)
            self.assertEqual(
                metrics.snapshot()["latency_ms"]["inference"]["p95"],
                0.0,
            )
            metrics.record_batch(
                request_count=1,
                text_count=1,
                queue_wait_ms=[1.0],
                inference_ms=latency,
                total_ms=[latency + 1],
            )
            self.assertEqual(
                metrics.snapshot()["latency_ms"]["inference"]["p95"],
                latency,
            )


class EmbeddingSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_queries_are_merged_and_order_is_reversible(self):
        calls = []

        async def dispatch(texts, mode):
            calls.append((mode, list(texts)))
            await asyncio.sleep(0.005)
            return [f"vector:{text}" for text in texts]

        scheduler = MicroBatchEmbeddingScheduler(
            config=_config(),
            dispatcher=dispatch,
            token_estimator=len,
        )
        try:
            outputs = await asyncio.gather(
                *(scheduler.submit([f"q-{index}"], "query") for index in range(6))
            )
            self.assertEqual(
                outputs, [[f"vector:q-{index}"] for index in range(6)]
            )
            self.assertEqual(calls, [("query", [f"q-{index}" for index in range(6)])])
            status = scheduler.status()
            self.assertEqual(status["metrics"]["batching"]["merged_batch_count"], 1)
            self.assertEqual(status["metrics"]["throughput"]["model_invocations"], 1)
        finally:
            await scheduler.close()

    async def test_query_and_document_requests_never_share_a_batch(self):
        calls = []

        async def dispatch(texts, mode):
            calls.append((mode, list(texts)))
            return list(texts)

        scheduler = MicroBatchEmbeddingScheduler(
            config=_config(batch_window_ms=2),
            dispatcher=dispatch,
            token_estimator=len,
        )
        try:
            query, document = await asyncio.gather(
                scheduler.submit(["query"], "query"),
                scheduler.submit(["doc-a", "doc-b"], "document"),
            )
            self.assertEqual(query, ["query"])
            self.assertEqual(document, ["doc-a", "doc-b"])
            self.assertEqual({mode for mode, _ in calls}, {"query", "document"})
            self.assertTrue(all(
                values == ["query"] if mode == "query" else values == ["doc-a", "doc-b"]
                for mode, values in calls
            ))
        finally:
            await scheduler.close()

    async def test_online_priority_yields_to_waiting_document_after_quota(self):
        calls = []
        first_started = asyncio.Event()
        release_first = asyncio.Event()

        async def dispatch(texts, mode):
            calls.append((mode, list(texts)))
            if texts == ["q-1"]:
                first_started.set()
                await release_first.wait()
            return list(texts)

        scheduler = MicroBatchEmbeddingScheduler(
            config=_config(
                batch_window_ms=0,
                max_batch_texts=1,
                offline_corpus_quota=2,
            ),
            dispatcher=dispatch,
            token_estimator=len,
        )
        first = asyncio.create_task(scheduler.submit(["q-1"], "query"))
        await asyncio.wait_for(first_started.wait(), timeout=1)
        pending = [
            asyncio.create_task(scheduler.submit(["q-2"], "query")),
            asyncio.create_task(scheduler.submit(["q-3"], "query")),
            asyncio.create_task(scheduler.submit(["doc"], "document")),
        ]
        await asyncio.sleep(0)
        release_first.set()
        await asyncio.gather(first, *pending)
        self.assertEqual(
            calls[:3],
            [
                ("query", ["q-1"]),
                ("query", ["q-2"]),
                ("document", ["doc"]),
            ],
        )
        await scheduler.close()

    async def test_queue_capacity_rejects_instead_of_growing_without_bound(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def dispatch(texts, mode):
            started.set()
            await release.wait()
            return list(texts)

        scheduler = MicroBatchEmbeddingScheduler(
            config=_config(queue_capacity=1, batch_window_ms=0),
            dispatcher=dispatch,
            token_estimator=len,
        )
        first = asyncio.create_task(scheduler.submit(["first"], "query"))
        await asyncio.wait_for(started.wait(), timeout=1)
        second = asyncio.create_task(scheduler.submit(["second"], "query"))
        await asyncio.sleep(0)
        with self.assertRaises(EmbeddingCapacityError):
            await scheduler.submit(["third"], "query")
        release.set()
        self.assertEqual(await first, ["first"])
        self.assertEqual(await second, ["second"])
        self.assertEqual(scheduler.status()["metrics"]["requests"]["rejected"], 1)
        await scheduler.close()

    async def test_queued_cancellation_does_not_enter_model(self):
        started = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def dispatch(texts, mode):
            calls.extend(texts)
            if "first" in texts:
                started.set()
                await release.wait()
            return list(texts)

        scheduler = MicroBatchEmbeddingScheduler(
            config=_config(batch_window_ms=0),
            dispatcher=dispatch,
            token_estimator=len,
        )
        first = asyncio.create_task(scheduler.submit(["first"], "query"))
        await asyncio.wait_for(started.wait(), timeout=1)
        queued = asyncio.create_task(scheduler.submit(["cancel-me"], "query"))
        await asyncio.sleep(0)
        queued.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await queued
        release.set()
        await first
        await asyncio.sleep(0.02)
        self.assertNotIn("cancel-me", calls)
        self.assertEqual(scheduler.status()["metrics"]["requests"]["cancelled"], 1)
        await scheduler.close()

    async def test_deadline_is_typed_and_scheduler_recovers(self):
        async def dispatch(texts, mode):
            await asyncio.sleep(0.05)
            return list(texts)

        scheduler = MicroBatchEmbeddingScheduler(
            config=_config(batch_window_ms=0, request_deadline_ms=10),
            dispatcher=dispatch,
            token_estimator=len,
        )
        with self.assertRaises(EmbeddingDeadlineExceeded):
            await scheduler.submit(["slow"], "query")
        await asyncio.sleep(0.06)
        status = scheduler.status()
        self.assertEqual(status["metrics"]["requests"]["timed_out"], 1)
        self.assertEqual(status["queue"]["depth"], 0)
        await scheduler.close()

    async def test_close_cancels_queued_requests_and_drains_running_batch(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def dispatch(texts, mode):
            started.set()
            await release.wait()
            return list(texts)

        scheduler = MicroBatchEmbeddingScheduler(
            config=_config(batch_window_ms=0),
            dispatcher=dispatch,
            token_estimator=len,
        )
        running = asyncio.create_task(scheduler.submit(["running"], "query"))
        await asyncio.wait_for(started.wait(), timeout=1)
        queued = asyncio.create_task(scheduler.submit(["queued"], "query"))
        await asyncio.sleep(0)
        close_task = asyncio.create_task(scheduler.close())
        await asyncio.sleep(0)
        release.set()
        self.assertEqual(await running, ["running"])
        with self.assertRaises(asyncio.CancelledError):
            await queued
        status = await close_task
        self.assertEqual(status["state"], "closed")
        self.assertEqual(status["queue"]["depth"], 0)

    async def test_close_reclaims_an_already_cancelled_worker(self):
        async def dispatch(texts, mode):
            return list(texts)

        scheduler = MicroBatchEmbeddingScheduler(
            config=_config(),
            dispatcher=dispatch,
            token_estimator=len,
        )
        worker = asyncio.create_task(asyncio.sleep(10))
        scheduler._worker = worker
        worker.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await worker

        status = await scheduler.close()

        self.assertEqual(status["state"], "closed")
        self.assertIsNone(scheduler._worker)

    async def test_close_propagates_a_completed_worker_exception(self):
        async def dispatch(texts, mode):
            return list(texts)

        scheduler = MicroBatchEmbeddingScheduler(
            config=_config(),
            dispatcher=dispatch,
            token_estimator=len,
        )

        async def fail_worker():
            raise RuntimeError("worker lifecycle fault")

        scheduler._worker = asyncio.create_task(fail_worker())
        await asyncio.sleep(0)

        with self.assertRaisesRegex(RuntimeError, "worker lifecycle fault"):
            await scheduler.close()
        self.assertFalse(scheduler._closed)
        self.assertIsNotNone(scheduler._worker)


if __name__ == "__main__":
    unittest.main()
