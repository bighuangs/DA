import asyncio
import math
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.infrastructure.clients.embedding_client_manager import EmbeddingClientManager
from app.infrastructure.clients.hybrid_embedding_client import (
    HybridEmbeddingClient,
    HybridEmbeddingRuntimeError,
)
from app.agent_system.governance.retrieval_budget import embedding_runtime_evidence
from app.domain.retrieval.hybrid_embedding import HybridEmbedding


def _config(
    path: str,
    *,
    runtime_mode: str = "serial",
    shutdown_drain_timeout_ms: int = 2000,
):
    return SimpleNamespace(
        model="BAAI/bge-m3",
        model_path=path,
        revision="unit-test",
        dense_size=1024,
        max_length=64,
        batch_size=2,
        device="cpu",
        use_fp16=False,
        normalize_dense=True,
        runtime_mode=runtime_mode,
        scheduler=SimpleNamespace(
            queue_capacity=32,
            batch_window_ms=8,
            max_batch_texts=16,
            max_batch_tokens=256,
            request_deadline_ms=2000,
            online_query_priority=0,
            offline_corpus_priority=10,
            offline_corpus_quota=4,
            shutdown_drain_timeout_ms=shutdown_drain_timeout_ms,
            metrics_window_size=64,
        ),
    )


class _FakeBGE:
    def __init__(self):
        self.query_calls = []
        self.corpus_calls = []
        self.active = 0
        self.max_active = 0

    def _encode(self, texts):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        self.active -= 1
        return {
            "dense_vecs": [[1.0] + [0.0] * 1023 for _ in texts],
            "lexical_weights": [{7: 0.8, 2: 0.3} for _ in texts],
        }

    def encode_queries(self, texts, **kwargs):
        self.query_calls.append((list(texts), kwargs))
        return self._encode(texts)

    def encode_corpus(self, texts, **kwargs):
        self.corpus_calls.append((list(texts), kwargs))
        return self._encode(texts)


class _BlockingBGE(_FakeBGE):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def _encode(self, texts):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            if not self.release.wait(timeout=5):
                raise TimeoutError("test model was not released")
            return {
                "dense_vecs": [[1.0] + [0.0] * 1023 for _ in texts],
                "lexical_weights": [{7: 0.8, 2: 0.3} for _ in texts],
            }
        finally:
            self.active -= 1
            self.finished.set()


class HybridEmbeddingValueTests(unittest.TestCase):
    def test_dense_is_normalized_and_sparse_is_sorted(self):
        embedding = HybridEmbedding.from_raw(
            [2.0, 0.0],
            {"9": 0.2, "3": 0.5, "4": 0.0},
            dense_size=2,
            normalize_dense=True,
        )
        self.assertAlmostEqual(math.sqrt(sum(v * v for v in embedding.dense)), 1.0)
        self.assertEqual(embedding.sparse_indices, (3, 9))
        self.assertEqual(embedding.sparse_values, (0.5, 0.2))

    def test_invalid_dense_or_empty_sparse_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "维度异常"):
            HybridEmbedding.from_raw(
                [1.0], {1: 0.2}, dense_size=2, normalize_dense=True
            )
        with self.assertRaisesRegex(ValueError, "sparse 为空"):
            HybridEmbedding.from_raw(
                [1.0, 0.0], {1: 0.0}, dense_size=2, normalize_dense=True
            )


class HybridEmbeddingClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        Path(self.temp.name, "config.json").write_text("{}", encoding="utf-8")
        self.model = _FakeBGE()
        self.factory_calls = 0

        def factory():
            self.factory_calls += 1
            return self.model

        self.client = HybridEmbeddingClient(
            _config(self.temp.name), model_factory=factory
        )

    async def asyncTearDown(self):
        await self.client.close()
        self.temp.cleanup()

    async def test_query_and_document_use_distinct_bge_contracts(self):
        query = await self.client.aembed_query("order_amount")
        documents = await self.client.aembed_documents(["订单金额", "销售额"])
        self.assertEqual(len(query.dense), 1024)
        self.assertEqual(query.sparse_indices, (2, 7))
        self.assertEqual(len(documents), 2)
        self.assertEqual(self.model.query_calls[0][0], ["order_amount"])
        self.assertEqual(self.model.corpus_calls[0][0], ["订单金额", "销售额"])
        self.assertEqual(self.factory_calls, 1)

    async def test_query_batch_uses_encode_queries_once(self):
        outputs = await self.client.aembed_queries(["GMV", "order_amount"])
        self.assertEqual(len(outputs), 2)
        self.assertEqual(self.model.query_calls[0][0], ["GMV", "order_amount"])
        self.assertEqual(self.model.corpus_calls, [])

    async def test_metrics_epoch_reset_keeps_model_and_binds_suite_evidence(self):
        await self.client.warmup()
        warmup_status = self.client.status()
        self.assertEqual(
            warmup_status["scheduling"]["metrics"]["throughput"][
                "model_invocations"
            ],
            1,
        )

        reset = self.client.reset_runtime_metrics_epoch(
            reason="evaluation_repeat_1_suite_boundary"
        )
        self.assertTrue(reset["only_observability_reset"])
        self.assertTrue(reset["model_state_preserved"])
        self.assertTrue(self.client.status()["loaded"])
        self.assertEqual(
            self.client.status()["scheduling"]["metrics"]["throughput"][
                "model_invocations"
            ],
            0,
        )

        await self.client.aembed_query("suite case")
        evidence = embedding_runtime_evidence(self.client)
        self.assertEqual(evidence["metrics_epoch_id"], reset["active_epoch"]["id"])
        self.assertEqual(
            evidence["metrics_epoch_index"], reset["active_epoch"]["index"]
        )
        self.assertGreater(evidence["inference_p95_ms"], 0)

    async def test_metrics_epoch_reset_rejects_active_native_work(self):
        await self.client.close()
        model = _BlockingBGE()
        self.client = HybridEmbeddingClient(
            _config(self.temp.name),
            model_factory=lambda: model,
        )
        pending = asyncio.create_task(self.client.aembed_query("running"))
        self.assertTrue(await asyncio.to_thread(model.started.wait, 1))

        with self.assertRaisesRegex(
            HybridEmbeddingRuntimeError,
            "仍有活动任务",
        ):
            self.client.reset_runtime_metrics_epoch(reason="must-fail")

        model.release.set()
        await asyncio.wait_for(pending, timeout=1)

    async def test_first_load_and_encoding_are_serialized(self):
        await asyncio.gather(
            self.client.aembed_query("GMV"),
            self.client.aembed_query("AOV"),
            self.client.aembed_documents(["customer_id"]),
        )
        self.assertEqual(self.factory_calls, 1)
        self.assertEqual(self.model.max_active, 1)

    async def test_cancelled_waiter_close_drains_active_inference(self):
        model = _BlockingBGE()
        self.model = model
        pending = asyncio.create_task(self.client.aembed_query("cancelled"))
        self.assertTrue(await asyncio.to_thread(model.started.wait, 1))

        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertEqual(
            self.client.status()["lifecycle"]["active_inference_futures"],
            1,
        )

        close_task = asyncio.create_task(self.client.close())
        await asyncio.sleep(0.03)
        self.assertFalse(close_task.done())
        model.release.set()
        await asyncio.wait_for(close_task, timeout=1)

        lifecycle = self.client.status()["lifecycle"]
        self.assertEqual(lifecycle["state"], "closed")
        self.assertEqual(lifecycle["active_inference_futures"], 0)
        self.assertFalse(lifecycle["accepting_requests"])

    async def test_microbatch_close_normally_drains_scheduler_and_inference(self):
        await self.client.close()
        model = _BlockingBGE()
        self.model = model
        self.client = HybridEmbeddingClient(
            _config(self.temp.name, runtime_mode="microbatch"),
            model_factory=lambda: model,
        )
        pending = asyncio.create_task(self.client.aembed_query("running"))
        self.assertTrue(await asyncio.to_thread(model.started.wait, 1))

        close_task = asyncio.create_task(self.client.close())
        await asyncio.sleep(0.03)
        self.assertFalse(close_task.done())
        model.release.set()

        result = await asyncio.wait_for(pending, timeout=1)
        await asyncio.wait_for(close_task, timeout=1)
        self.assertEqual(len(result.dense), 1024)
        status = self.client.status()
        self.assertEqual(status["lifecycle"]["state"], "closed")
        self.assertEqual(status["scheduling"]["state"], "closed")
        self.assertEqual(status["scheduling"]["queue"]["depth"], 0)

    async def test_close_timeout_is_observable_and_can_be_retried(self):
        await self.client.close()
        model = _BlockingBGE()
        self.model = model
        self.client = HybridEmbeddingClient(
            _config(self.temp.name, shutdown_drain_timeout_ms=20),
            model_factory=lambda: model,
        )
        pending = asyncio.create_task(self.client.aembed_query("stuck"))
        self.assertTrue(await asyncio.to_thread(model.started.wait, 1))
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending

        with self.assertRaisesRegex(
            HybridEmbeddingRuntimeError,
            "fail-close",
        ):
            await self.client.close()
        lifecycle = self.client.status()["lifecycle"]
        self.assertEqual(lifecycle["state"], "close_failed")
        self.assertTrue(lifecycle["close_failed"])
        self.assertFalse(lifecycle["accepting_requests"])
        self.assertEqual(lifecycle["active_inference_futures"], 1)
        self.assertIn("排空超时", lifecycle["last_close_error"])
        with self.assertRaisesRegex(HybridEmbeddingRuntimeError, "正在关闭"):
            await self.client.aembed_query("must-not-start")

        model.release.set()
        self.assertTrue(await asyncio.to_thread(model.finished.wait, 1))
        for _ in range(20):
            if not self.client.status()["lifecycle"][
                "active_inference_futures"
            ]:
                break
            await asyncio.sleep(0.01)
        await self.client.close()
        self.assertEqual(
            self.client.status()["lifecycle"]["state"],
            "closed",
        )

    async def test_output_shape_violation_is_not_silently_accepted(self):
        self.model.encode_queries = lambda texts, **kwargs: {
            "dense_vecs": [[1.0, 0.0]],
            "lexical_weights": [{1: 0.3}],
        }
        with self.assertRaises(HybridEmbeddingRuntimeError):
            await self.client.aembed_query("bad")

    async def test_microbatch_profile_merges_calls_without_model_concurrency(self):
        await self.client.close()
        self.client = HybridEmbeddingClient(
            _config(self.temp.name, runtime_mode="microbatch"),
            model_factory=lambda: self.model,
        )
        outputs = await asyncio.gather(
            *(self.client.aembed_query(f"query-{index}") for index in range(6))
        )
        self.assertEqual(len(outputs), 6)
        self.assertLess(len(self.model.query_calls), 6)
        self.assertEqual(
            [text for call, _ in self.model.query_calls for text in call],
            [f"query-{index}" for index in range(6)],
        )
        self.assertEqual(self.model.max_active, 1)
        status = self.client.status()
        self.assertEqual(status["runtime_profile"], "microbatch_inprocess_v1")
        self.assertGreaterEqual(
            status["scheduling"]["metrics"]["batching"]["merged_batch_count"], 1
        )
        self.assertEqual(status["model_instances"], 1)

    async def test_runtime_profile_does_not_change_vector_profile_fingerprint(self):
        serial = HybridEmbeddingClient(
            _config(self.temp.name, runtime_mode="serial"),
            model_factory=lambda: _FakeBGE(),
        )
        microbatch = HybridEmbeddingClient(
            _config(self.temp.name, runtime_mode="microbatch"),
            model_factory=lambda: _FakeBGE(),
        )
        try:
            self.assertEqual(
                serial.profile_fingerprint, microbatch.profile_fingerprint
            )
            self.assertNotEqual(
                serial.runtime_profile_fingerprint,
                microbatch.runtime_profile_fingerprint,
            )
        finally:
            await serial.close()
            await microbatch.close()


class HybridEmbeddingClientCrossLoopTests(unittest.TestCase):
    def test_cancelled_microbatch_worker_is_reclaimed_from_a_later_loop(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "config.json").write_text("{}", encoding="utf-8")
            model = _BlockingBGE()
            config = _config(temp_dir, runtime_mode="microbatch")
            client = HybridEmbeddingClient(
                config,
                model_factory=lambda: model,
            )
            manager = EmbeddingClientManager(config)
            manager.client = client

            async def cancel_waiter() -> None:
                pending = asyncio.create_task(
                    client.aembed_query("first-loop")
                )
                for _ in range(100):
                    if model.started.is_set():
                        break
                    await asyncio.sleep(0.005)
                self.assertTrue(model.started.is_set())
                pending.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await pending

            asyncio.run(cancel_waiter())
            self.assertEqual(
                client.status()["lifecycle"]["active_inference_futures"],
                1,
            )
            self.assertTrue(client._scheduler._worker.cancelled())
            model.release.set()
            self.assertTrue(model.finished.wait(timeout=1))
            try:
                asyncio.run(manager.close())
            finally:
                model.release.set()
                if manager.client is not None:
                    asyncio.run(manager.close())
            self.assertEqual(client.status()["lifecycle"]["state"], "closed")
            self.assertIsNone(manager.client)

            replacement = manager.init()
            self.assertIsNot(replacement, client)
            asyncio.run(manager.close())

    def test_close_waits_for_native_result_async_delivery_before_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "config.json").write_text("{}", encoding="utf-8")
            model = _FakeBGE()
            client = HybridEmbeddingClient(
                _config(temp_dir),
                model_factory=lambda: model,
            )
            original_run_serial = client._run_serial
            native_delivered = threading.Event()
            release_delivery = threading.Event()
            query_errors: list[BaseException] = []
            close_errors: list[BaseException] = []

            async def delayed_run_serial(function, *args):
                result = await original_run_serial(function, *args)
                native_delivered.set()
                while not release_delivery.is_set():
                    await asyncio.sleep(0.005)
                return result

            client._run_serial = delayed_run_serial

            def run_query() -> None:
                try:
                    asyncio.run(client.aembed_query("delivery-race"))
                except BaseException as exc:  # noqa: BLE001
                    query_errors.append(exc)

            def run_close() -> None:
                try:
                    asyncio.run(client.close())
                except BaseException as exc:  # noqa: BLE001
                    close_errors.append(exc)

            query_thread = threading.Thread(target=run_query)
            close_thread = threading.Thread(target=run_close)
            query_thread.start()
            try:
                self.assertTrue(native_delivered.wait(timeout=1))
                self.assertEqual(
                    client.status()["lifecycle"][
                        "active_inference_futures"
                    ],
                    0,
                )
                self.assertEqual(
                    client.status()["lifecycle"][
                        "active_async_operations"
                    ],
                    1,
                )

                close_thread.start()
                time.sleep(0.05)
                self.assertTrue(close_thread.is_alive())
            finally:
                release_delivery.set()
                query_thread.join(timeout=1)
                if close_thread.ident is not None:
                    close_thread.join(timeout=1)
            try:
                self.assertFalse(query_thread.is_alive())
                self.assertFalse(close_thread.is_alive())
                self.assertEqual(close_errors, [])
                self.assertEqual(len(query_errors), 1)
                self.assertIsInstance(
                    query_errors[0], HybridEmbeddingRuntimeError
                )
                status = client.status()
                self.assertEqual(status["lifecycle"]["state"], "closed")
                self.assertEqual(
                    status["lifecycle"]["active_async_operations"], 0
                )
                self.assertFalse(status["loaded"])
                self.assertEqual(status["model_instances"], 0)
            finally:
                release_delivery.set()
                if client.status()["lifecycle"]["state"] != "closed":
                    asyncio.run(client.close())


if __name__ == "__main__":
    unittest.main()
