"""评测进程内 BGE-M3 预热与驻留生命周期合同。"""

from __future__ import annotations

import asyncio
import copy
import time
import uuid
from typing import Any, Self

from app.infrastructure.clients.embedding_client_manager import (
    EmbeddingClientManager,
    embedding_client_manager,
)
from app.config.app_config import app_config
from app.agent_system.governance.retrieval_budget import (
    capture_retrieval_cache_generation,
    invalidate_retrieval_caches,
)

EVALUATION_EMBEDDING_WARMUP_SCHEMA = "evaluation-embedding-warmup-v1"


class EvaluationEmbeddingWarmupError(RuntimeError):
    """评测 BGE 预热或驻留合同未通过。"""


def evaluation_embedding_evidence_checks(
    warmup: dict[str, Any] | None,
    cache_control: dict[str, Any] | None,
) -> dict[str, bool]:
    """重新校验持久化预热/缓存证据，防止展示字段绕过治理。"""
    warmup = dict(warmup or {})
    cache_control = dict(cache_control or {})
    checks_payload = dict(warmup.get("checks") or {})
    requested_state = cache_control.get("requested_state")
    warmup_generation_advanced = (
        int(warmup.get("retrieval_cache_generation_after") or 0)
        > int(warmup.get("retrieval_cache_generation_before") or 0)
    )
    run_generation_before = int(
        cache_control.get("retrieval_cache_generation_before") or 0
    )
    run_generation_after = int(
        cache_control.get("retrieval_cache_generation_after") or 0
    )
    cold = requested_state == "cold"
    checks = {
        "schema": warmup.get("schema") == EVALUATION_EMBEDDING_WARMUP_SCHEMA,
        "status": warmup.get("status") == "pass",
        "session_identity": bool(warmup.get("session_id")),
        "profile_identity": bool(warmup.get("profile_fingerprint")),
        "probe_contract": bool(checks_payload) and all(
            bool(value) for value in checks_payload.values()
        ),
        "model_resident": bool(
            warmup.get("model_resident_for_suite")
            and warmup.get("resident_loaded_at_suite_start")
            and warmup.get("resident_accepting_at_suite_start")
        ),
        "warmup_cache_boundary": bool(
            warmup.get("retrieval_caches_invalidated_after_warmup")
            and warmup_generation_advanced
        ),
        "timing_boundary": bool(
            warmup.get("warmup_excluded_from_case_timing")
            and warmup.get("suite_started_only_after_pass")
        ),
        "metrics_epoch_reset": bool(
            (warmup.get("metrics_epoch_reset") or {}).get("status") == "pass"
            and (warmup.get("metrics_epoch_reset") or {}).get(
                "only_observability_reset"
            )
            and (warmup.get("metrics_epoch_reset") or {}).get(
                "queue_state_preserved"
            )
            and (warmup.get("metrics_epoch_reset") or {}).get(
                "model_state_preserved"
            )
        ),
        "metrics_epoch_bound": bool(
            (warmup.get("suite_metrics_epoch") or {}).get("id")
            and (warmup.get("suite_metrics_epoch") or {}).get("id")
            == (warmup.get("metrics_epoch_reset") or {}).get(
                "active_epoch", {}
            ).get("id")
            and (warmup.get("suite_metrics_epoch") or {}).get("started_at")
        ),
        "cache_state_bound": (
            warmup.get("requested_cache_state") == requested_state
        ),
        "result_cache_protocol": (
            bool(cache_control.get("result_cache_cleared")) == cold
        ),
        "retrieval_cache_protocol": (
            bool(cache_control.get("retrieval_caches_cleared")) == cold
            and (
                run_generation_after > run_generation_before
                if cold
                else run_generation_after == run_generation_before
            )
        ),
    }
    return {**checks, "complete": all(checks.values())}


class EvaluationEmbeddingSession:
    """在一次评测进程会话内预热并复用同一个 BGE 客户端。

    预热发生在任何评测用例之前，耗时不计入逐题延迟。预热成功后清空
    exact-vector、candidate 与 context 检索缓存，但保留已加载模型；调用方再按
    cold/warm 协议决定是否在每个 repeat 前清缓存。
    """

    def __init__(
        self,
        manager: EmbeddingClientManager = embedding_client_manager,
    ) -> None:
        self._manager = manager
        self._client: Any | None = None
        self._warmup_evidence: dict[str, Any] | None = None
        self._session_id = uuid.uuid4().hex
        self._run_index = 0
        self._active_epoch_id: str | None = None

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def _close_after_failed_start(self) -> str | None:
        try:
            await self._manager.close()
        except Exception as exc:  # noqa: BLE001  # manager 需保留 poisoned client
            return f"{type(exc).__name__}: {exc}"
        return None

    async def start(self) -> dict[str, Any]:
        """执行一次有界预热；失败时终止会话且不得进入付费 suite。"""
        if self._warmup_evidence is not None:
            self._assert_resident()
            return copy.deepcopy(self._warmup_evidence)

        started = time.perf_counter()
        timeout_seconds = max(
            1.0,
            float(app_config.query.hybrid_retrieval_timeout_seconds),
        )
        client: Any | None = None
        try:
            # init/status/generation 均属于付费前置合同；任一失败都必须走同一关闭路径。
            client = self._manager.init()
            before = client.status()
            generation_before = capture_retrieval_cache_generation()
            async with asyncio.timeout(timeout_seconds):
                probe = await client.warmup()
            after = client.status()
            checks = {
                "dense_size": int(probe.get("dense_size") or 0)
                == int(app_config.embedding.dense_size),
                "sparse_nonzero": int(probe.get("sparse_nonzero") or 0) > 0,
                "model_loaded": bool(after.get("loaded")),
                "single_model_instance": int(after.get("model_instances") or 0) == 1,
                "accepting_requests": bool(
                    (after.get("lifecycle") or {}).get("accepting_requests")
                ),
                "profile_stable": (
                    before.get("profile_fingerprint")
                    == after.get("profile_fingerprint")
                ),
            }
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                raise EvaluationEmbeddingWarmupError(
                    "BGE-M3 评测预热合同未通过：" + ",".join(failed)
                )

            # warmup 只应留下模型驻留，不能让任何检索缓存看起来是 warm。
            invalidate_retrieval_caches()
            generation_after = capture_retrieval_cache_generation()
            if generation_after <= generation_before:
                raise EvaluationEmbeddingWarmupError(
                    "BGE-M3 预热后检索缓存 generation 未前进"
                )
        except BaseException as exc:
            close_error = await self._close_after_failed_start()
            if isinstance(exc, asyncio.CancelledError):
                raise
            detail = f"{type(exc).__name__}: {exc}"
            if close_error:
                detail += f"；fail-close 关闭失败：{close_error}"
            raise EvaluationEmbeddingWarmupError(
                "BGE-M3 评测预热失败，已在付费 suite 前终止：" + detail
            ) from exc

        self._client = client
        self._warmup_evidence = {
            "schema": EVALUATION_EMBEDDING_WARMUP_SCHEMA,
            "status": "pass",
            "scope": "single_process_evaluation_session",
            "session_id": self._session_id,
            "warmup_invocations": 1,
            "warmup_duration_ms": round(
                (time.perf_counter() - started) * 1000,
                1,
            ),
            "timeout_seconds": timeout_seconds,
            "checks": checks,
            "dense_size": int(probe["dense_size"]),
            "sparse_nonzero": int(probe["sparse_nonzero"]),
            "profile_fingerprint": after.get("profile_fingerprint"),
            "runtime_profile_fingerprint": after.get(
                "runtime_profile_fingerprint"
            ),
            "runtime_profile": after.get("runtime_profile"),
            "warmup_metrics_epoch": dict(
                ((after.get("scheduling") or {}).get("metrics") or {}).get(
                    "epoch"
                )
                or {}
            ),
            "model_loaded_before": bool(before.get("loaded")),
            "model_loaded_after": bool(after.get("loaded")),
            "model_resident_for_suite": True,
            "retrieval_caches_invalidated_after_warmup": True,
            "retrieval_cache_generation_before": generation_before,
            "retrieval_cache_generation_after": generation_after,
            "warmup_excluded_from_case_timing": True,
            "suite_started_only_after_pass": True,
            "raw_probe_text_persisted": False,
        }
        return copy.deepcopy(self._warmup_evidence)

    def _assert_resident(self) -> dict[str, Any]:
        if self._client is None or self._warmup_evidence is None:
            raise EvaluationEmbeddingWarmupError("BGE-M3 评测会话尚未完成预热")
        if self._manager.client is not self._client:
            raise EvaluationEmbeddingWarmupError(
                "BGE-M3 评测会话的驻留客户端已被替换"
            )
        status = self._client.status()
        lifecycle = status.get("lifecycle") or {}
        if not status.get("loaded") or not lifecycle.get("accepting_requests"):
            raise EvaluationEmbeddingWarmupError(
                "BGE-M3 评测会话未保持可接收请求的模型驻留状态"
            )
        if (
            status.get("profile_fingerprint")
            != self._warmup_evidence.get("profile_fingerprint")
        ):
            raise EvaluationEmbeddingWarmupError(
                "BGE-M3 评测会话的模型 profile 在 repeat 间发生漂移"
            )
        if (
            status.get("runtime_profile_fingerprint")
            != self._warmup_evidence.get("runtime_profile_fingerprint")
        ):
            raise EvaluationEmbeddingWarmupError(
                "BGE-M3 评测会话的 runtime profile 在 repeat 间发生漂移"
            )
        current_epoch = dict(
            ((status.get("scheduling") or {}).get("metrics") or {}).get("epoch")
            or {}
        )
        if self._active_epoch_id and current_epoch.get("id") != self._active_epoch_id:
            raise EvaluationEmbeddingWarmupError(
                "BGE-M3 评测会话的 metrics epoch 在 repeat 内发生漂移"
            )
        return status

    @property
    def client(self) -> Any:
        """返回经过本 session 预热和身份校验的唯一客户端。"""
        self._assert_resident()
        return self._client

    def evidence_for_run(self, *, cache_state: str | None) -> dict[str, Any]:
        """为每个 repeat 建立独立 metrics epoch，不改变模型或业务缓存。"""
        self._assert_resident()
        next_run_index = self._run_index + 1
        reset = self._client.reset_runtime_metrics_epoch(
            reason=f"evaluation_repeat_{next_run_index}_suite_boundary"
        )
        status = self._client.status()
        epoch = dict(
            ((status.get("scheduling") or {}).get("metrics") or {}).get("epoch")
            or {}
        )
        active_epoch = dict(reset.get("active_epoch") or {})
        reset_checks = {
            "status": reset.get("status") == "pass",
            "queue_preserved": bool(reset.get("queue_state_preserved")),
            "model_preserved": bool(reset.get("model_state_preserved")),
            "observability_only": bool(reset.get("only_observability_reset")),
            "idle_before_reset": (
                int(reset.get("active_inference_before") or 0) == 0
                and int(reset.get("active_async_before") or 0) == 0
            ),
            "epoch_bound": bool(
                epoch.get("id")
                and epoch.get("id") == active_epoch.get("id")
                and epoch.get("index") == active_epoch.get("index")
                and epoch.get("started_at") == active_epoch.get("started_at")
            ),
            "observations_empty": all(
                int(value or 0) == 0
                for value in (
                    ((reset.get("active_observations") or {}).get("requests") or {}).values()
                )
            ) and int(
                ((reset.get("active_observations") or {}).get("throughput") or {}).get(
                    "model_invocations"
                )
                or 0
            ) == 0,
        }
        failed = [name for name, passed in reset_checks.items() if not passed]
        if failed:
            raise EvaluationEmbeddingWarmupError(
                "BGE-M3 repeat metrics epoch 合同未通过：" + ",".join(failed)
            )
        self._run_index = next_run_index
        self._active_epoch_id = str(epoch["id"])
        evidence = copy.deepcopy(self._warmup_evidence)
        evidence.update({
            "session_run_index": self._run_index,
            "requested_cache_state": cache_state,
            "resident_reused": self._run_index > 1,
            "resident_loaded_at_suite_start": bool(status.get("loaded")),
            "resident_accepting_at_suite_start": bool(
                (status.get("lifecycle") or {}).get("accepting_requests")
            ),
            "metrics_epoch_reset": reset,
            "metrics_epoch_reset_checks": reset_checks,
            "suite_metrics_epoch": epoch,
            "runtime_evidence_epoch_id": epoch.get("id"),
        })
        evidence["checks"] = {
            **dict(evidence.get("checks") or {}),
            "metrics_epoch_reset": all(reset_checks.values()),
        }
        return evidence

    def finalize_run_evidence(self, run_evidence: dict[str, Any]) -> dict[str, Any]:
        """冻结 repeat 结束快照并证明它仍属于启动时绑定的 epoch。"""
        status = self._assert_resident()
        scheduling = dict(status.get("scheduling") or {})
        queue = dict(scheduling.get("queue") or {})
        metrics = dict(scheduling.get("metrics") or {})
        epoch = dict(metrics.get("epoch") or {})
        expected_epoch = dict(run_evidence.get("suite_metrics_epoch") or {})
        lifecycle = dict(status.get("lifecycle") or {})
        checks = {
            "epoch_stable": bool(
                epoch.get("id")
                and epoch.get("id") == expected_epoch.get("id")
                and epoch.get("index") == expected_epoch.get("index")
                and epoch.get("started_at") == expected_epoch.get("started_at")
            ),
            "queue_idle": (
                int(queue.get("depth") or 0) == 0
                and int(queue.get("running_requests") or 0) == 0
            ),
            "native_idle": (
                int(lifecycle.get("active_inference_futures") or 0) == 0
                and int(lifecycle.get("active_async_operations") or 0) == 0
            ),
            "model_resident": bool(
                status.get("loaded")
                and lifecycle.get("accepting_requests")
                and int(status.get("model_instances") or 0) == 1
            ),
            "profile_stable": (
                status.get("profile_fingerprint")
                == run_evidence.get("profile_fingerprint")
                and status.get("runtime_profile_fingerprint")
                == run_evidence.get("runtime_profile_fingerprint")
            ),
        }
        return {
            "schema": "evaluation-embedding-runtime-epoch-v1",
            "status": "pass" if all(checks.values()) else "fail",
            "checks": checks,
            "epoch": epoch,
            "requests": dict(metrics.get("requests") or {}),
            "throughput": dict(metrics.get("throughput") or {}),
            "latency_ms": dict(metrics.get("latency_ms") or {}),
            "batching": dict(metrics.get("batching") or {}),
            "queue": queue,
            "profile_fingerprint": status.get("profile_fingerprint"),
            "runtime_profile_fingerprint": status.get(
                "runtime_profile_fingerprint"
            ),
            "raw_text_persisted": False,
        }

    async def close(self) -> None:
        """关闭会话拥有的驻留模型；manager 负责失败后的 poisoned 状态。"""
        if self._client is None:
            return
        await self._manager.close()
        self._client = None
