import unittest
from unittest.mock import Mock, patch

from app.config.app_config import app_config
from evals.legacy.embedding_session import (
    EVALUATION_EMBEDDING_WARMUP_SCHEMA,
    EvaluationEmbeddingSession,
    EvaluationEmbeddingWarmupError,
    evaluation_embedding_evidence_checks,
)
from evals.legacy.run_eval import (
    apply_evaluation_cache_control,
    embedding_runtime_health_audit,
    evaluation_baseline_eligible,
    evaluation_continue_signal,
)


class _FakeEmbeddingClient:
    def __init__(self, *, sparse_nonzero: int = 3, failure: Exception | None = None):
        self.loaded = False
        self.sparse_nonzero = sparse_nonzero
        self.failure = failure
        self.warmup_calls = 0
        self.reset_calls = 0
        self.epoch_index = 0
        self.epoch_id = "epoch-0"

    def status(self):
        return {
            "loaded": self.loaded,
            "model_instances": int(self.loaded),
            "profile_fingerprint": "profile-1",
            "runtime_profile_fingerprint": "runtime-1",
            "runtime_profile": "microbatch_inprocess_v1",
            "lifecycle": {
                "accepting_requests": True,
                "active_inference_futures": 0,
                "active_async_operations": 0,
            },
            "scheduling": {
                "state": "idle",
                "queue": {"depth": 0, "running_requests": 0},
                "metrics": {
                    "epoch": {
                        "schema": "embedding-runtime-metrics-epoch-v1",
                        "id": self.epoch_id,
                        "index": self.epoch_index,
                        "started_at": f"epoch-start-{self.epoch_index}",
                    },
                    "requests": {
                        "submitted": 0,
                        "completed": 0,
                        "rejected": 0,
                        "cancelled": 0,
                        "timed_out": 0,
                        "failed": 0,
                    },
                    "throughput": {"model_invocations": 0},
                    "latency_ms": {},
                    "batching": {},
                },
            },
        }

    async def warmup(self):
        self.warmup_calls += 1
        if self.failure is not None:
            raise self.failure
        self.loaded = True
        return {
            "dense_size": app_config.embedding.dense_size,
            "sparse_nonzero": self.sparse_nonzero,
        }

    def reset_runtime_metrics_epoch(self, *, reason):
        previous = {
            "id": self.epoch_id,
            "index": self.epoch_index,
            "started_at": f"epoch-start-{self.epoch_index}",
        }
        self.reset_calls += 1
        self.epoch_index += 1
        self.epoch_id = f"epoch-{self.epoch_index}"
        active = {
            "schema": "embedding-runtime-metrics-epoch-v1",
            "id": self.epoch_id,
            "index": self.epoch_index,
            "started_at": f"epoch-start-{self.epoch_index}",
        }
        return {
            "schema": "embedding-runtime-metrics-reset-v1",
            "status": "pass",
            "reason": reason,
            "previous_epoch": previous,
            "active_epoch": active,
            "active_observations": {
                "requests": {
                    "submitted": 0,
                    "completed": 0,
                    "rejected": 0,
                    "cancelled": 0,
                    "timed_out": 0,
                    "failed": 0,
                },
                "throughput": {"model_invocations": 0},
            },
            "queue_state_preserved": True,
            "model_state_preserved": True,
            "active_inference_before": 0,
            "active_async_before": 0,
            "only_observability_reset": True,
        }


class _FakeEmbeddingManager:
    def __init__(self, client):
        self.client = client
        self.close_calls = 0

    def init(self):
        return self.client

    async def close(self):
        self.close_calls += 1
        self.client = None


class _FailingInitManager(_FakeEmbeddingManager):
    def init(self):
        raise RuntimeError("init failed after allocation")


class EvaluationEmbeddingSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_warmup_is_reused_and_records_cold_boundary(self):
        client = _FakeEmbeddingClient()
        manager = _FakeEmbeddingManager(client)
        with (
            patch(
                "evals.legacy.embedding_session.capture_retrieval_cache_generation",
                side_effect=[40, 41],
            ),
            patch(
                "evals.legacy.embedding_session.invalidate_retrieval_caches"
            ) as invalidate,
        ):
            session = EvaluationEmbeddingSession(manager)
            await session.start()
            cold = session.evidence_for_run(cache_state="cold")
            warm = session.evidence_for_run(cache_state="warm")
            warm_again = session.evidence_for_run(cache_state="warm")

        self.assertEqual(client.warmup_calls, 1)
        invalidate.assert_called_once_with()
        self.assertEqual(cold["schema"], EVALUATION_EMBEDDING_WARMUP_SCHEMA)
        self.assertEqual(cold["session_run_index"], 1)
        self.assertFalse(cold["resident_reused"])
        self.assertEqual(warm["session_run_index"], 2)
        self.assertTrue(warm["resident_reused"])
        self.assertEqual(warm_again["session_run_index"], 3)
        self.assertEqual(
            [
                cold["suite_metrics_epoch"]["index"],
                warm["suite_metrics_epoch"]["index"],
                warm_again["suite_metrics_epoch"]["index"],
            ],
            [1, 2, 3],
        )
        self.assertEqual(client.reset_calls, 3)
        self.assertEqual(cold["session_id"], warm["session_id"])
        self.assertTrue(cold["retrieval_caches_invalidated_after_warmup"])
        self.assertIs(manager.client, client)

        checks = evaluation_embedding_evidence_checks(
            cold,
            {
                "requested_state": "cold",
                "result_cache_cleared": True,
                "retrieval_caches_cleared": True,
                "retrieval_cache_generation_before": 41,
                "retrieval_cache_generation_after": 42,
            },
        )
        self.assertTrue(checks["complete"])

        final = session.finalize_run_evidence(warm_again)
        self.assertEqual(final["status"], "pass")
        self.assertEqual(
            final["epoch"]["id"],
            warm_again["suite_metrics_epoch"]["id"],
        )

        await session.close()
        self.assertEqual(manager.close_calls, 1)

    async def test_invalid_sparse_probe_fails_before_suite_and_closes_client(self):
        manager = _FakeEmbeddingManager(_FakeEmbeddingClient(sparse_nonzero=0))
        with (
            patch(
                "evals.legacy.embedding_session.capture_retrieval_cache_generation",
                return_value=50,
            ),
            patch("evals.legacy.embedding_session.invalidate_retrieval_caches"),
            self.assertRaisesRegex(
                EvaluationEmbeddingWarmupError,
                "付费 suite 前终止",
            ),
        ):
            await EvaluationEmbeddingSession(manager).start()

        self.assertEqual(manager.close_calls, 1)
        self.assertIsNone(manager.client)

    async def test_warmup_exception_fails_closed(self):
        manager = _FakeEmbeddingManager(
            _FakeEmbeddingClient(failure=RuntimeError("model unavailable"))
        )
        with patch(
            "evals.legacy.embedding_session.capture_retrieval_cache_generation",
            return_value=60,
        ), self.assertRaisesRegex(
            EvaluationEmbeddingWarmupError,
            "model unavailable",
        ):
            await EvaluationEmbeddingSession(manager).start()

        self.assertEqual(manager.close_calls, 1)

    async def test_init_and_generation_preflight_failures_close_manager(self):
        init_manager = _FailingInitManager(_FakeEmbeddingClient())
        with self.assertRaisesRegex(EvaluationEmbeddingWarmupError, "init failed"):
            await EvaluationEmbeddingSession(init_manager).start()
        self.assertEqual(init_manager.close_calls, 1)

        generation_manager = _FakeEmbeddingManager(_FakeEmbeddingClient())
        with (
            patch(
                "evals.legacy.embedding_session.capture_retrieval_cache_generation",
                side_effect=RuntimeError("generation unavailable"),
            ),
            self.assertRaisesRegex(
                EvaluationEmbeddingWarmupError,
                "generation unavailable",
            ),
        ):
            await EvaluationEmbeddingSession(generation_manager).start()
        self.assertEqual(generation_manager.close_calls, 1)


class EvaluationCacheControlTests(unittest.TestCase):
    def test_cold_clears_result_and_all_retrieval_caches(self):
        result_cache = Mock()
        result_cache.stats.return_value = {"entries": 7, "hits": 1, "misses": 2}
        with (
            patch("evals.legacy.run_eval.result_cache", result_cache),
            patch(
                "evals.legacy.run_eval.capture_retrieval_cache_generation",
                side_effect=[70, 71],
            ),
            patch("evals.legacy.run_eval.invalidate_retrieval_caches") as invalidate,
        ):
            evidence = apply_evaluation_cache_control({"cache_state": "cold"})

        result_cache.clear.assert_called_once_with()
        invalidate.assert_called_once_with()
        self.assertTrue(evidence["result_cache_cleared"])
        self.assertTrue(evidence["retrieval_caches_cleared"])
        self.assertEqual(evidence["retrieval_cache_generation_before"], 70)
        self.assertEqual(evidence["retrieval_cache_generation_after"], 71)
        self.assertEqual(evidence["effective_policy"], "cleared_all")
        self.assertFalse(evidence["cache_hit_not_guaranteed"])
        self.assertFalse(evidence["latency_attribution_allowed"])

    def test_warm_preserves_result_and_retrieval_caches(self):
        result_cache = Mock()
        result_cache.stats.return_value = {"entries": 7, "hits": 1, "misses": 2}
        with (
            patch("evals.legacy.run_eval.result_cache", result_cache),
            patch(
                "evals.legacy.run_eval.capture_retrieval_cache_generation",
                side_effect=[80, 80],
            ),
            patch("evals.legacy.run_eval.invalidate_retrieval_caches") as invalidate,
        ):
            evidence = apply_evaluation_cache_control({"cache_state": "warm"})

        result_cache.clear.assert_not_called()
        invalidate.assert_not_called()
        self.assertFalse(evidence["result_cache_cleared"])
        self.assertFalse(evidence["retrieval_caches_cleared"])
        self.assertEqual(evidence["retrieval_cache_generation_before"], 80)
        self.assertEqual(evidence["retrieval_cache_generation_after"], 80)
        self.assertEqual(evidence["effective_policy"], "retained_ttl_best_effort")
        self.assertTrue(evidence["cache_hit_not_guaranteed"])
        self.assertEqual(evidence["ttl_seconds"]["result"], 30)
        self.assertEqual(evidence["ttl_seconds"]["retrieval"], 120)


class EvaluationEmbeddingRuntimeHealthTests(unittest.TestCase):
    @staticmethod
    def _final_runtime(*, cancelled=0):
        return {
            "status": "pass",
            "checks": {
                "epoch_stable": True,
                "queue_idle": True,
                "native_idle": True,
                "model_resident": True,
                "profile_stable": True,
            },
            "requests": {
                "rejected": 0,
                "cancelled": cancelled,
                "timed_out": 0,
                "failed": 0,
            },
        }

    @staticmethod
    def _result(degradations=None):
        return {
            "id": "easy_001",
            "degradations": list(degradations or []),
            "stage_evidence": {
                "retrieval_budget": [{
                    "route": "column",
                    "embedding_runtime": {
                        "metrics_epoch_id": "epoch-1",
                        "metrics_epoch_index": 1,
                        "metrics_epoch_started_at": "start-1",
                    },
                }],
            },
        }

    def test_only_typed_core_retrieval_and_scheduler_failures_block(self):
        epoch = {"id": "epoch-1", "index": 1, "started_at": "start-1"}
        correction_recovery = {
            "node_id": "correct_sql",
            "reason_code": "timeout",
            "affects_core_path": True,
        }
        clean = embedding_runtime_health_audit(
            [self._result([correction_recovery])],
            self._final_runtime(),
            epoch,
        )
        self.assertTrue(clean["eligible"])

        retrieval_timeout = {
            "node_id": "recall_column",
            "reason_code": "timeout",
            "affects_core_path": True,
        }
        degraded = embedding_runtime_health_audit(
            [self._result([retrieval_timeout])],
            self._final_runtime(),
            epoch,
        )
        self.assertFalse(degraded["eligible"])
        self.assertFalse(degraded["checks"]["typed_core_retrieval_clean"])

        cancelled = embedding_runtime_health_audit(
            [self._result()],
            self._final_runtime(cancelled=1),
            epoch,
        )
        self.assertFalse(cancelled["eligible"])
        self.assertFalse(cancelled["checks"]["scheduler_failure_delta_zero"])

    def test_experiment_stops_on_ineligible_repeat_but_standalone_keeps_valid_semantics(self):
        self.assertFalse(evaluation_continue_signal(
            experiment={"experiment_id": "exp-1"},
            valid=True,
            baseline_eligible=False,
        ))
        self.assertTrue(evaluation_continue_signal(
            experiment=None,
            valid=True,
            baseline_eligible=False,
        ))

    def test_llm_release_failure_stops_experiment_without_changing_validity(self):
        base = {
            "failed_core_llm_calls": 0,
            "llm_attempts_timed_out": 0,
        }
        common = {
            "valid": True,
            "full_dataset": True,
            "pipeline_evidence_eligible": True,
            "embedding_warmup_eligible": True,
            "embedding_runtime_eligible": True,
        }
        self.assertTrue(evaluation_baseline_eligible(
            **common, aggregate=base,
        ))
        for raw_field in base:
            with self.subTest(raw_field=raw_field):
                aggregate = {**base, raw_field: 1}
                baseline_eligible = evaluation_baseline_eligible(
                    **common, aggregate=aggregate,
                )
                self.assertFalse(baseline_eligible)
                self.assertFalse(evaluation_continue_signal(
                    experiment={"experiment_id": "exp-1"},
                    valid=True,
                    baseline_eligible=baseline_eligible,
                ))


if __name__ == "__main__":
    unittest.main()
