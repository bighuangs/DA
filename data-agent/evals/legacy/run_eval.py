"""
NL2SQL 评测 CLI（TODO-23）。

一句话总述：加载显式评测套件 → 驱动 graph 逐用例跑 → 聚合多维指标 → 输出可治理制品。

用法：
    python -m evals.legacy.run_eval
    python -m evals.legacy.run_eval --only <case-id-1>,<case-id-2>  # 只跑指定用例
"""
import argparse
import asyncio
import datetime as dt
import json
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from app.agent_system.agents.nl2sql.context import DataAgentContext
from app.adapters.outbound.agent.llm_provider import llm_route_manifest
from app.infrastructure.clients.es_client_manager import es_client_manager
from app.infrastructure.clients.mysql_client_manager import (
    dw_mysql_client_manager,
    meta_mysql_client_manager,
)
from app.infrastructure.clients.qdrant_client_manager import qdrant_client_manager
from app.config.app_config import app_config
from app.domain.access.policy import PolicySnapshot
from app.agent_system.agents.nl2sql.services.candidate_confidence import (
    SEMANTIC_REVIEW_POLICY_VERSION,
    compact_candidate_decision,
)
from app.domain.data_sources.context import stable_fingerprint
from app.agent_system.runtime.llm_attempts import (
    LLM_ATTEMPT_SCHEMA,
    attempt_accounting,
    normalize_degradation,
)
from app.infrastructure.clients.llm.capabilities import (
    TODO97_CANDIDATE_SET,
    candidate_capability,
)
from app.infrastructure.clients.llm.invocation import FATAL_PROVIDER_REASON_CODES
from app.infrastructure.observability.logging import logger
from app.infrastructure.cache.result_cache import result_cache
from app.agent_system.governance.retrieval_budget import (
    capture_retrieval_cache_generation,
    invalidate_retrieval_caches,
)
from evals.legacy.attribution import summarize_attributions
from evals.legacy.dataset_suite import (
    SUITE_PATHS,
    SUITE_SPECS,
    dataset_suite_report,
    load_suite_cases,
)
from evals.legacy.embedding_session import (
    EvaluationEmbeddingSession,
    evaluation_embedding_evidence_checks,
)
from evals.legacy.evaluator import run_case, semantic_review_case_metrics
from evals.legacy.governance import (
    build_experiment_metadata,
    collect_run_fingerprint,
    experiment_report,
    llm_release_evidence_checks,
    pipeline_evidence_checks,
    resolve_baseline_artifact,
)
from evals.legacy.model_invocation_contract import (
    GENERATE_SQL_PHYSICAL_ATTEMPT_PER_CASE,
    evaluate_llm_invocation_contract,
    validate_reference_result_contract,
)
from evals.legacy.profiles import EVALUATION_PROFILES, resolve_evaluation_profile
from evals.adapters.outbound.nl2sql import build_nl2sql_evaluation_context
from app.infrastructure.persistence.mysql.records.metric_info_mysql import MetricInfoMySQL
from app.infrastructure.persistence.mysql.records.training_info_mysql import TrainingInfoMySQL
from app.infrastructure.persistence.mysql.records.workspace_mysql import WorkspaceMySQL
from app.infrastructure.search.elasticsearch.repositories.value_es_repository import ValueESRepository
from app.infrastructure.persistence.mysql.repositories.dw_mysql_repository import DWMysqlRepository
from evals.infrastructure.persistence.mysql.evaluation_baseline_repository import (
    EvaluationBaselineRepository,
)
from app.infrastructure.persistence.mysql.repositories.meta_mysql_repository import MetaMysqlRepository
from app.infrastructure.search.qdrant.repositories.column_qdrant_repository import ColumnQdrantRepository
from app.infrastructure.search.qdrant.repositories.metric_qdrant_repository import MetricQdrantRepository
from app.infrastructure.search.qdrant.repositories.training_qdrant_repository import TrainingQdrantRepository
from evals.paths import REPORTS_ROOT, RUNS_ROOT

REPORT_DIR = REPORTS_ROOT / "nl2sql"
RUNS_DIR = RUNS_ROOT


def targeted_report_path(
    dataset_suite: str,
    report_dir: Path = REPORT_DIR,
) -> Path:
    if dataset_suite not in SUITE_SPECS:
        raise ValueError(f"未知数据集套件：{dataset_suite}")
    return report_dir / f"targeted_{dataset_suite}_latest.md"


def targeted_run_path(
    dataset_suite: str,
    runs_dir: Path = RUNS_DIR,
) -> Path:
    if dataset_suite not in SUITE_SPECS:
        raise ValueError(f"未知数据集套件：{dataset_suite}")
    return runs_dir / f"targeted_{dataset_suite}_latest.json"


def apply_evaluation_cache_control(experiment: dict | None) -> dict[str, Any]:
    """按 repeat 合同切换缓存状态，并返回可持久化的 generation 证据。"""
    requested_state = (experiment or {}).get("cache_state")
    cache_cleared = requested_state == "cold"
    result_before = result_cache.stats()
    retrieval_generation_before = capture_retrieval_cache_generation()
    if cache_cleared:
        result_cache.clear()
        invalidate_retrieval_caches()
    return {
        "requested_state": requested_state,
        "effective_policy": (
            "cleared_all" if cache_cleared else "retained_ttl_best_effort"
        ),
        "result_cache_cleared": cache_cleared,
        "retrieval_caches_cleared": cache_cleared,
        "ttl_seconds": {
            "result": int(app_config.query.cache_ttl_seconds),
            "retrieval": int(app_config.query.retrieval_cache_ttl_seconds),
        },
        "cache_hit_not_guaranteed": not cache_cleared,
        "latency_attribution_allowed": False,
        "retrieval_cache_generation_before": retrieval_generation_before,
        "retrieval_cache_generation_after": (
            capture_retrieval_cache_generation()
        ),
        "before": result_before,
        "provider_cache_control": "unavailable",
    }


def evaluation_continue_signal(
    *,
    experiment: dict | None,
    valid: bool,
    baseline_eligible: bool,
) -> bool:
    """正式 experiment 只在本 repeat 证据可准入时继续产生后续费用。"""
    return bool(baseline_eligible) if experiment is not None else bool(valid)


def evaluation_baseline_eligible(
    *,
    valid: bool,
    full_dataset: bool,
    pipeline_evidence_eligible: bool,
    embedding_warmup_eligible: bool,
    embedding_runtime_eligible: bool,
    aggregate: dict,
) -> bool:
    """组合正式 baseline 门禁，并从 raw LLM 计数保持失败关闭。"""

    return bool(
        valid
        and full_dataset
        and pipeline_evidence_eligible
        and embedding_warmup_eligible
        and embedding_runtime_eligible
        and all(llm_release_evidence_checks(aggregate).values())
    )


def existing_experiment_runs(
    experiment_id: str,
    runs_dir: Path = RUNS_DIR,
    dataset_suite: str | None = None,
) -> list[dict]:
    runs: list[dict] = []
    if not runs_dir.exists():
        return runs
    for path in sorted(runs_dir.glob("run_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (payload.get("experiment") or {}).get("experiment_id") != experiment_id:
            continue
        payload_suite = str(payload.get("dataset_suite") or "")
        if dataset_suite and payload_suite != dataset_suite:
            raise ValueError(
                f"experiment {experiment_id} 已绑定套件 "
                f"{payload_suite or 'missing'}，不能用于 {dataset_suite}"
            )
        payload["_artifact"] = path.name
        runs.append(payload)
    return runs


def ensure_experiment_repeat_available(
    experiment_id: str,
    repeat_index: int,
    runs_dir: Path = RUNS_DIR,
    dataset_suite: str | None = None,
) -> None:
    duplicate = next((
        run for run in existing_experiment_runs(
            experiment_id, runs_dir, dataset_suite,
        )
        if int((run.get("experiment") or {}).get("repeat_index") or 0) == repeat_index
    ), None)
    if duplicate:
        raise ValueError(
            f"experiment {experiment_id} 的 repeat {repeat_index} 已由 run "
            f"{duplicate.get('run_id')} 占用；请勿覆盖不可变评测制品"
        )


def load_cases(
    only: str | None = None, tier: str | None = None, suite: str = "easy_sql_v1",
) -> list[dict]:
    """Load one governed suite or an explicit diagnostic slice."""
    return load_suite_cases(suite=suite, only=only, tier=tier)


def _todo97_expected_route_identity(candidate: dict[str, Any]) -> dict[str, Any]:
    """Translate registry names to the public route-manifest contract."""
    return {
        "candidate_id": candidate.get("candidate_id"),
        "model_name": candidate.get("model_id"),
        "provider": candidate.get("provider"),
        "model_provider": candidate.get("model_provider"),
        "access_mode": candidate.get("access_mode"),
    }


async def collect_context_fingerprint(cases: list[dict], all_cases: list[dict],
                                      context: DataAgentContext,
                                      evaluation_profile: str = "core-default") -> dict:
    """Bridge TypedDict agent dependencies into governance fingerprint collection."""
    return await collect_run_fingerprint(
        selected_cases=cases,
        all_cases=all_cases,
        dw_repo=context["dw_mysql_repo"],
        meta_repo=context["meta_mysql_repo"],
        value_repo=context.get("value_es_repo"),
        evaluation_profile=evaluation_profile,
    )


def _pct(n: int, d: int) -> str:
    return f"{(n / d * 100):.1f}%" if d else "N/A"


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0
    import math
    k = max(0, math.ceil(p * len(sorted_vals)) - 1)
    return sorted_vals[k]


TIMING_PROTOCOL_VERSION = "timing-v2"
PROVIDER_FAILURE_AUDIT_SCHEMA = "provider-failure-audit-v1"
EVAL_EXECUTION_AUDIT_SCHEMA = "eval-execution-audit-v1"
CORE_RETRIEVAL_NODES = {
    "recall_column",
    "recall_metric",
    "recall_training",
    "recall_value",
}
CORE_RETRIEVAL_FAILURE_REASONS = {"timeout", "dependency_unavailable"}


def embedding_runtime_health_audit(
    results: list[dict],
    final_runtime: dict[str, Any],
    expected_epoch: dict[str, Any],
) -> dict[str, Any]:
    """门禁本 repeat 的召回降级、scheduler 增量与 epoch 绑定。"""
    retrieval_failures = []
    observed_epochs = []
    for result in results:
        case_id = str(result.get("id") or "")
        for degradation in result.get("degradations") or []:
            if (
                str(degradation.get("node_id") or "") in CORE_RETRIEVAL_NODES
                and str(degradation.get("reason_code") or "")
                in CORE_RETRIEVAL_FAILURE_REASONS
                and bool(degradation.get("affects_core_path"))
            ):
                retrieval_failures.append({
                    "case_id": case_id,
                    "node_id": str(degradation.get("node_id") or ""),
                    "reason_code": str(degradation.get("reason_code") or ""),
                })
        retrieval = dict(result.get("stage_evidence") or {}).get(
            "retrieval_budget"
        ) or []
        for route in retrieval:
            runtime = dict((route or {}).get("embedding_runtime") or {})
            if runtime.get("metrics_epoch_id"):
                observed_epochs.append({
                    "case_id": case_id,
                    "route": str((route or {}).get("route") or ""),
                    "epoch_id": str(runtime.get("metrics_epoch_id")),
                    "epoch_index": runtime.get("metrics_epoch_index"),
                    "epoch_started_at": runtime.get("metrics_epoch_started_at"),
                })

    requests = dict(final_runtime.get("requests") or {})
    scheduler_failure_counts = {
        key: int(requests.get(key) or 0)
        for key in ("rejected", "cancelled", "timed_out", "failed")
    }
    expected_id = str(expected_epoch.get("id") or "")
    expected_index = expected_epoch.get("index")
    expected_started_at = expected_epoch.get("started_at")
    mismatched = [
        item
        for item in observed_epochs
        if (
            item["epoch_id"] != expected_id
            or item["epoch_index"] != expected_index
            or item["epoch_started_at"] != expected_started_at
        )
    ]
    final_checks = dict(final_runtime.get("checks") or {})
    checks = {
        "typed_core_retrieval_clean": not retrieval_failures,
        "scheduler_failure_delta_zero": not any(
            scheduler_failure_counts.values()
        ),
        "runtime_final_clean": (
            final_runtime.get("status") == "pass"
            and bool(final_checks)
            and all(bool(value) for value in final_checks.values())
        ),
        "runtime_epoch_observed": bool(observed_epochs),
        "runtime_epoch_bound": bool(expected_id) and not mismatched,
    }
    return {
        "schema": "evaluation-embedding-runtime-health-v1",
        "eligible": all(checks.values()),
        "checks": checks,
        "expected_epoch": dict(expected_epoch),
        "observed_epoch_count": len(observed_epochs),
        "mismatched_epoch_count": len(mismatched),
        "mismatched_epochs": mismatched,
        "typed_core_retrieval_failures": retrieval_failures,
        "scheduler_failure_counts": scheduler_failure_counts,
        "correction_or_model_recovery_rejected": False,
    }


def provider_failure_audit(results: list[dict]) -> dict[str, Any]:
    """Summarize non-retryable provider failures without provider payloads.

    Authentication, payment and permission failures describe the evaluation
    infrastructure, not model capability.  Persist only the typed boundary
    reason and structural attempt identity; provider response bodies, headers
    and credentials are deliberately excluded.
    """
    failures: list[dict[str, Any]] = []
    for result in results:
        case_id = str(result.get("id") or "")
        for attempt in result.get("llm_attempts") or []:
            reason_code = str(attempt.get("reason_code") or "")
            if (
                reason_code not in FATAL_PROVIDER_REASON_CODES
                or str(attempt.get("status") or "") == "success"
            ):
                continue
            failures.append({
                "case_id": case_id,
                "node_id": str(attempt.get("node_id") or "unknown"),
                "route": str(attempt.get("route") or "unknown"),
                "reason_code": reason_code,
                "error_type": str(attempt.get("error_type") or "UnknownError")[:120],
                "affects_core_path": bool(
                    attempt.get("affects_core_path", True)
                ),
                "retryable": bool(attempt.get("retryable")),
            })
    core_failures = [item for item in failures if item["affects_core_path"]]
    return {
        "schema": PROVIDER_FAILURE_AUDIT_SCHEMA,
        "fatal_failure_count": len(failures),
        "core_fatal_failure_count": len(core_failures),
        "fatal_failures": failures,
        "stop_required": bool(failures),
        "invalidates_run": bool(core_failures),
    }


def build_execution_audit(
    planned_case_ids: list[str],
    results: list[dict],
) -> dict[str, Any]:
    """Describe a complete or provider-fail-fast run without inventing rows."""
    planned = [str(case_id) for case_id in planned_case_ids]
    executed = [str(result.get("id") or "") for result in results]
    executed_set = set(executed)
    not_executed = [case_id for case_id in planned if case_id not in executed_set]
    provider_audit = provider_failure_audit(results)
    first_failure = next(iter(provider_audit["fatal_failures"]), None)
    early_terminated = bool(not_executed and provider_audit["stop_required"])
    return {
        "schema": EVAL_EXECUTION_AUDIT_SCHEMA,
        "planned_count": len(planned),
        "executed_count": len(executed),
        "not_executed_count": len(not_executed),
        "executed_case_ids": executed,
        "not_executed_case_ids": not_executed,
        "early_terminated": early_terminated,
        "termination_reason": (
            "fatal_non_retryable_provider_failure" if early_terminated else None
        ),
        "trigger_case_id": (
            first_failure.get("case_id") if first_failure else None
        ),
        "trigger_reason_code": (
            first_failure.get("reason_code") if first_failure else None
        ),
        "provider_failure_audit": provider_audit,
    }


async def run_cases_fail_fast(
    cases: list[dict],
    context: DataAgentContext,
    dw_repo: Any,
    evaluation_profile: str,
) -> tuple[list[dict], dict[str, Any]]:
    """Run cases sequentially and stop after a typed fatal provider failure."""
    results: list[dict] = []
    for case in cases:
        logger.info(f"运行用例 {case['id']}：{case['question']}")
        result = await run_case(case, context, dw_repo, evaluation_profile)
        results.append(result)
        logger.info(
            f"用例 {case['id']} 完成：exec={result['exec_success']} "
            f"sql_equiv={result['sql_equivalent']} "
            f"result_equiv={result['result_equivalent']} "
            f"{result['latency_ms']}ms"
        )
        case_provider_audit = provider_failure_audit([result])
        if case_provider_audit["stop_required"]:
            failure = case_provider_audit["fatal_failures"][0]
            logger.error(
                "评测检测到不可重试的 Provider 失败，安全停止后续用例："
                f"case={failure['case_id']} node={failure['node_id']} "
                f"reason={failure['reason_code']}"
            )
            break
    return results, build_execution_audit(
        [str(case.get("id") or "") for case in cases],
        results,
    )


def _table_scope_evidence_complete(result: dict[str, Any]) -> bool:
    """Require runtime-bound v2 scope evidence for the exact Final SQL."""
    final = result.get("final_sql_metrics") or {}
    evidence = final.get("table_scope_safety_evidence") or {}
    sql = str(final.get("sql") or result.get("gen_sql") or "").strip()
    executed_authorized_sql = str(
        (result.get("stage_evidence") or {}).get("authorized_sql") or ""
    ).strip()
    return bool(
        sql
        and evidence.get("schema") == "table-scope-safety-v2"
        and evidence.get("authorization_source") == "runtime_table_infos"
        and evidence.get("evidence_complete") is True
        and evidence.get("sql_hash") == stable_fingerprint(sql)
        and evidence.get("request_scope_fingerprint")
        and evidence.get("authorized_sql_hash")
        and (
            not executed_authorized_sql
            or evidence.get("authorized_sql_hash")
            == stable_fingerprint(executed_authorized_sql)
        )
    )


def aggregate(results: list[dict]) -> dict:
    n = len(results)
    timing_measurement_complete = bool(results) and all(
        "tt_complete_ms" in result and float(result.get("tt_complete_ms") or 0) > 0
        for result in results
    )
    # TTComplete is a case-level wall-clock metric.  Keep it separate from model
    # route latency: reusing this variable used to let the last route overwrite
    # the full-run P95 and made an experiment appear much faster than it was.
    tt_complete = sorted(
        float(r.get("tt_complete_ms") or r.get("latency_ms") or 0)
        for r in results
    )
    retry_cases = [r for r in results if r["retry_count"] > 0]
    tt_sql = sorted(float(r.get("tt_sql_ms") or 0) for r in results)
    tt_result = sorted(float(r.get("tt_result_ms") or 0) for r in results)
    llm_calls = sorted(float(r.get("llm_calls") or 0) for r in results)
    total_tokens = sorted(float(r.get("total_tokens") or 0) for r in results)
    token_peak = max(results, key=lambda row: float(row.get("total_tokens") or 0), default={})
    prompt_peak = max(results, key=lambda row: float(row.get("prompt_tokens") or 0), default={})
    route_groups: dict[tuple[str, str], dict] = {}
    prompt_before = prompt_after = prompt_payloads = 0
    candidate_extra_calls = selected_b = unsafe_selected = 0
    candidate_valid_b = candidate_failed = candidate_repaired = 0
    candidate_waits: list[float] = []
    adaptive_triggered = adaptive_extra_calls = adaptive_selected_b = 0
    adaptive_valid_b = adaptive_failed = adaptive_unsafe_selected = 0
    adaptive_high_confidence_extra = 0
    adaptive_waits: list[float] = []
    review_eligible = review_triggered = review_attempts = review_extra_calls = 0
    review_selected_b = review_valid_b = review_failed = review_failed_core = 0
    review_unsafe_selected = review_harmful = review_fixed_result = 0
    review_first_correct_final_incorrect = 0
    review_same_sql_execution_loss = review_selected_b_no_sql_delta = 0
    review_budget_skipped = review_decision_cases = 0
    review_evidence_complete = True
    review_coverage_complete = True
    review_at_most_one = True
    review_attribution_complete = True
    review_waits: list[float] = []
    structured_fallbacks = 0
    all_attempts: list[dict] = []
    semantic_coverage_scores: list[float] = []
    semantic_unresolved_cases = 0
    semantic_unsafe_compile_cases = 0
    deadline_remaining: list[float] = []
    deadline_contract_complete = True
    degradation_evidence_complete = True
    attempt_accounting_complete = True
    query_signal_profile_counts: Counter[str] = Counter()
    query_signal_route_policy_counts: Counter[str] = Counter()
    query_signal_dictionary_version_counts: Counter[str] = Counter()
    query_signal_route_counts: Counter[str] = Counter()
    query_signal_encode_mode_counts: Counter[str] = Counter()
    query_signal_remote_signal_counts: Counter[str] = Counter()
    query_signal_consumer_signal_counts: Counter[str] = Counter()
    query_signal_extractor_kind_counts: Counter[str] = Counter()
    query_signal_consumer_kind_counts: Counter[str] = Counter()
    query_signal_evidence_count = query_signal_evidence_complete_count = 0
    query_signal_route_evidence_count = query_signal_dictionary_fallback_count = 0
    planning_profile_counts: Counter[str] = Counter()
    planning_configured_profile_counts: Counter[str] = Counter()
    planning_fallback_profile_counts: Counter[str] = Counter()
    planning_fallback_class_counts: Counter[str] = Counter()
    planning_route_counts: Counter[str] = Counter()
    planning_plan_calls = planning_physical_llm_calls = 0
    planning_fallback_count = planning_degraded_count = 0
    planning_policy_fallback_count = planning_unsafe_fallback_count = 0
    planning_evidence_count = planning_evidence_complete_count = 0
    repair_profile_counts: Counter[str] = Counter()
    repair_configured_profile_counts: Counter[str] = Counter()
    repair_attempted_rule_counts: Counter[str] = Counter()
    repair_applied_rule_counts: Counter[str] = Counter()
    repair_changed_count = repair_fallback_count = repair_degraded_count = 0
    repair_evidence_count = repair_evidence_complete_count = 0
    repair_invocation_count = repair_invocation_complete_count = 0
    repair_invocation_profile_counts: Counter[str] = Counter()
    repair_invocation_phase_counts: Counter[str] = Counter()
    repair_selected_invocation_count = repair_invocation_fallback_count = 0
    repair_ledger_complete_cases = 0
    repair_selected_invocation_complete_cases = 0
    for result in results:
        for usage in result.get("llm_routes") or []:
            key = (str(usage.get("node_id") or "unknown"), str(usage.get("route") or "sql"))
            item = route_groups.setdefault(key, {
                "node_id": key[0], "route": key[1], "models": set(), "calls": 0,
                "failed_calls": 0, "total_tokens": 0, "latencies": [],
                "succeeded": 0, "errors": 0, "timed_out": 0,
                "cancelled": 0, "empty_output": 0, "unknown_tokens": 0,
            })
            item["models"].update(usage.get("models") or [])
            item["calls"] += int(usage.get("calls") or 0)
            item["failed_calls"] += int(usage.get("failed_calls") or 0)
            item["succeeded"] += int(
                usage.get("succeeded")
                if usage.get("succeeded") is not None
                else max(0, int(usage.get("calls") or 0) - int(usage.get("failed_calls") or 0))
            )
            item["errors"] += int(usage.get("errors") or 0)
            item["timed_out"] += int(usage.get("timed_out") or 0)
            item["cancelled"] += int(usage.get("cancelled") or 0)
            item["empty_output"] += int(usage.get("empty_output") or 0)
            item["unknown_tokens"] += int(usage.get("unknown_tokens") or 0)
            item["total_tokens"] += int(usage.get("total_tokens") or 0)
            item["latencies"].append(float(usage.get("latency_p95_ms") or 0))
        context = ((result.get("stage_evidence") or {}).get("context_budget") or {})
        for evidence in (context.get("model_prompts") or {}).values():
            prompt_payloads += 1
            prompt_before += int(evidence.get("before_tokens") or 0)
            prompt_after += int(evidence.get("after_tokens") or 0)
        decision = compact_candidate_decision(
            (result.get("stage_evidence") or {}).get("candidate_decision")
        )
        mode = str(decision.get("decision_mode") or "none")
        extra = bool(decision.get("extra_call_used"))
        candidate_extra_calls += int(extra)
        selected_b += int(decision.get("selected") == "B")
        alternative = next((item for item in decision.get("candidates") or [] if item.get("label") == "B"), {})
        candidate_valid_b += int(bool(alternative) and alternative.get("valid") is True)
        candidate_repaired += int(decision.get("selected") == "B" and decision.get("candidate_outcome") == "selected_b")
        selected = next((item for item in decision.get("candidates") or [] if item.get("label") == decision.get("selected")), {})
        unsafe_selected += int(bool(selected) and selected.get("valid") is False)
        primary = next((item for item in decision.get("candidates") or [] if item.get("label") == "A"), {})
        if mode == "adaptive_expansion":
            adaptive_triggered += int(bool(decision.get("adaptive_triggered")))
            adaptive_extra_calls += int(extra)
            adaptive_selected_b += int(decision.get("selected") == "B")
            adaptive_valid_b += int(bool(alternative) and alternative.get("valid") is True)
            adaptive_unsafe_selected += int(
                bool(selected) and selected.get("valid") is False
            )
            adaptive_high_confidence_extra += int(
                extra and primary.get("band") == "high"
            )

        review_metrics = dict(
            result.get("semantic_review_metrics")
            or semantic_review_case_metrics(result)
        )
        review_decision_cases += int(mode == "semantic_review")
        review_eligible += int(bool(review_metrics.get("eligible")))
        review_triggered += int(bool(review_metrics.get("triggered")))
        review_attempts += int(review_metrics.get("attempt_count") or 0)
        review_extra_calls += int(mode == "semantic_review" and extra)
        review_selected_b += int(bool(review_metrics.get("selected_b")))
        review_valid_b += int(
            mode == "semantic_review"
            and bool(alternative)
            and alternative.get("valid") is True
        )
        review_failed_core += int(review_metrics.get("failed_core") or 0)
        review_unsafe_selected += int(bool(review_metrics.get("unsafe_selected")))
        review_harmful += int(bool(review_metrics.get("harmful_modification")))
        review_fixed_result += int(bool(review_metrics.get("fixed_result")))
        review_first_correct_final_incorrect += int(bool(
            review_metrics.get("first_correct_final_incorrect")
        ))
        review_same_sql_execution_loss += int(bool(
            review_metrics.get("same_sql_execution_loss")
        ))
        review_selected_b_no_sql_delta += int(bool(
            review_metrics.get("selected_b_no_sql_delta")
        ))
        review_budget_skipped += int(bool(review_metrics.get("budget_skipped")))
        if review_metrics.get("eligible"):
            review_evidence_complete = review_evidence_complete and bool(
                review_metrics.get("evidence_complete")
            )
            review_coverage_complete = review_coverage_complete and bool(
                review_metrics.get("coverage_complete")
            )
            review_at_most_one = review_at_most_one and bool(
                review_metrics.get("at_most_one_attempt")
            )
            review_attribution_complete = review_attribution_complete and bool(
                review_metrics.get("result_attribution_complete")
            )
        structured_fallbacks += sum(
            str(item.get("reason") or "").startswith("structured_output_fallback")
            for item in result.get("degradations") or []
        )
        attempts = list(result.get("llm_attempts") or [])
        all_attempts.extend(attempts)
        stage_evidence = result.get("stage_evidence") or {}
        query_signals = stage_evidence.get("query_signals") or {}
        if query_signals:
            query_signal_evidence_count += 1
            query_signal_evidence_complete_count += int(
                query_signals.get("evidence_complete") is True
            )
            query_signal_profile_counts.update([
                str(query_signals.get("active_profile") or "unknown")
            ])
            query_signal_route_policy_counts.update([
                str(
                    query_signals.get("configured_route_policy") or "unknown"
                )
            ])
            query_signal_dictionary_version_counts.update([
                str(query_signals.get("dictionary_version") or "unknown")
            ])
            for signal in (query_signals.get("extractor_signals") or []):
                if isinstance(signal, dict):
                    query_signal_extractor_kind_counts.update([
                        str(signal.get("kind") or "unknown")
                    ])
            routes = [
                item for item in (query_signals.get("routes") or [])
                if isinstance(item, dict)
            ]
            query_signal_route_evidence_count += len(routes)
            query_signal_route_counts.update(
                str(item.get("route") or "unknown") for item in routes
            )
            for item in routes:
                route = str(item.get("route") or "unknown")
                encode_mode = str(
                    item.get("query_encode_mode") or "unknown"
                )
                query_signal_encode_mode_counts.update([
                    f"{route}:{encode_mode}"
                ])
                remote_count = item.get("actual_remote_signal_count")
                if (
                    isinstance(remote_count, int)
                    and not isinstance(remote_count, bool)
                    and remote_count >= 0
                ):
                    query_signal_remote_signal_counts[route] += remote_count
                for consumer in ("local", "remote", "rerank"):
                    consumer_count = item.get(
                        f"actual_{consumer}_signal_count"
                    )
                    if (
                        isinstance(consumer_count, int)
                        and not isinstance(consumer_count, bool)
                        and consumer_count >= 0
                    ):
                        query_signal_consumer_signal_counts[
                            f"{route}:{consumer}"
                        ] += consumer_count
                    for signal in (
                        item.get(f"actual_{consumer}_signals") or []
                    ):
                        if isinstance(signal, dict):
                            query_signal_consumer_kind_counts.update([
                                f"{route}:{consumer}:"
                                f"{str(signal.get('kind') or 'unknown')}"
                            ])
            query_signal_dictionary_fallback_count += int(
                query_signals.get("dictionary_fallback_count") or 0
            )
        planning = stage_evidence.get("planning_route") or {}
        if planning:
            planning_evidence_count += 1
            planning_evidence_complete_count += int(
                planning.get("evidence_complete") is True
            )
            planning_profile_counts.update([
                str(planning.get("active_profile") or "unknown")
            ])
            planning_configured_profile_counts.update([
                str(planning.get("configured_profile") or "unknown")
            ])
            planning_fallback_profile_counts.update([
                str(planning.get("fallback_profile") or "unknown")
            ])
            fallback_class = str(
                planning.get("fallback_class") or "unknown"
            )
            planning_fallback_class_counts.update([fallback_class])
            planning_route_counts.update([
                str(planning.get("route") or "unknown")
            ])
            planning_plan_calls += int(planning.get("plan_called") is True)
            planning_physical_llm_calls += int(
                planning.get("physical_llm_call") is True
            )
            planning_fallback_count += int(
                planning.get("fallback_applied") is True
            )
            policy_fallback_valid = bool(
                fallback_class == "policy_abstain"
                and planning.get("policy_fallback_eligible") is True
                and planning.get("fallback_applied") is True
                and planning.get("decision") == "abstain"
                and planning.get("runtime_override_active") is False
                and planning.get("degraded") is False
            )
            no_fallback_valid = bool(
                fallback_class == "none"
                and planning.get("policy_fallback_eligible") is False
                and planning.get("fallback_applied") is False
                and planning.get("decision") != "abstain"
                and planning.get("runtime_override_active") is False
                and planning.get("degraded") is False
            )
            planning_policy_fallback_count += int(
                policy_fallback_valid
            )
            planning_unsafe_fallback_count += int(
                not (policy_fallback_valid or no_fallback_valid)
            )
            planning_degraded_count += int(planning.get("degraded") is True)
        repair = stage_evidence.get("semantic_repair") or {}
        if repair:
            repair_evidence_count += 1
            repair_evidence_complete_count += int(
                repair.get("evidence_complete") is True
            )
            repair_profile_counts.update([
                str(repair.get("active_profile") or "unknown")
            ])
            repair_configured_profile_counts.update([
                str(repair.get("configured_profile") or "unknown")
            ])
            repair_attempted_rule_counts.update(
                str(code) for code in repair.get("attempted_rule_codes") or []
            )
            repair_applied_rule_counts.update(
                str(row.get("rule_code"))
                for row in repair.get("applied_rules") or []
                if isinstance(row, dict) and row.get("rule_code")
            )
            repair_changed_count += int(repair.get("changed") is True)
            repair_fallback_count += int(repair.get("fallback_used") is True)
            repair_degraded_count += int(repair.get("degraded") is True)
            ledger = list(repair.get("ledger") or [])
            declared_invocations = int(repair.get("invocation_count") or 0)
            repair_invocation_count += declared_invocations
            repair_ledger_complete_cases += int(
                repair.get("ledger_evidence_complete") is True
                and declared_invocations == len(ledger)
            )
            selected_invocations = 0
            for invocation in ledger:
                if not isinstance(invocation, dict):
                    continue
                repair_invocation_profile_counts.update([
                    str(invocation.get("profile") or "unknown")
                ])
                repair_invocation_phase_counts.update([
                    str(invocation.get("phase") or "unknown")
                ])
                repair_invocation_complete_count += int(
                    invocation.get("evidence_complete") is True
                )
                selected_invocations += int(
                    invocation.get("selected") is True
                )
                repair_invocation_fallback_count += int(
                    invocation.get("fallback_used") is True
                )
            repair_selected_invocation_count += selected_invocations
            repair_selected_invocation_complete_cases += int(
                repair.get("selected_invocation_count") == 1
                and selected_invocations == 1
            )
        semantic_coverage = stage_evidence.get("semantic_coverage") or {}
        if semantic_coverage:
            semantic_coverage_scores.append(
                float(semantic_coverage.get("coverage_score") or 0)
            )
            semantic_unresolved_cases += int(
                bool(semantic_coverage.get("unresolved_slots"))
            )
            semantic_unsafe_compile_cases += int(
                bool(semantic_coverage.get("unresolved_slots"))
                and bool(
                    (
                        (stage_evidence.get("semantic_ir") or {}).get("compiler")
                        or {}
                    ).get("supported")
                )
            )
        deadline_evidence = stage_evidence.get("request_deadline") or {}
        if deadline_evidence:
            deadline_remaining.append(
                float(deadline_evidence.get("remaining_ms") or 0)
            )
            deadline_contract_complete = deadline_contract_complete and bool(
                deadline_evidence.get("contract_version")
                and deadline_evidence.get("absolute_timestamps_persisted") is False
            )
        candidate_rows = [
            item for item in attempts
            if str(item.get("node_id") or "") == "judge_candidate"
        ]
        candidate_failed += sum(item.get("status") != "success" for item in candidate_rows)
        candidate_waits.extend(float(item.get("latency_ms") or 0) for item in candidate_rows)
        if mode == "adaptive_expansion":
            adaptive_failed += sum(
                item.get("status") != "success" for item in candidate_rows
            )
            adaptive_waits.extend(
                float(item.get("latency_ms") or 0) for item in candidate_rows
            )
        elif mode == "semantic_review":
            review_failed += sum(
                item.get("status") != "success" for item in candidate_rows
            )
            review_waits.extend(
                float(item.get("latency_ms") or 0) for item in candidate_rows
            )
        accounting = result.get("llm_attempt_accounting") or attempt_accounting(attempts)
        coverage_complete = result.get("llm_attempt_accounting_complete") is True
        attempt_accounting_complete = attempt_accounting_complete and bool(
            "llm_attempts" in result
            and
            accounting.get("schema") == LLM_ATTEMPT_SCHEMA
            and accounting.get("reconciled")
            and coverage_complete
            and int(accounting.get("attempts") or 0) == len(attempts)
            and int(result.get("llm_calls") or 0) == len(attempts)
        )
        normalized_degradations = [normalize_degradation(item) for item in result.get("degradations") or []]
        degradation_evidence_complete = degradation_evidence_complete and all(
            item.get("evidence_complete") for item in normalized_degradations
        )
    route_performance = []
    for item in route_groups.values():
        route_latencies = sorted(item.pop("latencies"))
        item["models"] = sorted(item["models"])
        item["tokens_per_call"] = round(item["total_tokens"] / item["calls"], 1) if item["calls"] else 0
        item["latency_p95_ms"] = _percentile(route_latencies, .95)
        route_performance.append(item)
    route_performance.sort(key=lambda item: (item["latency_p95_ms"], item["total_tokens"]), reverse=True)
    nodes: dict[str, list[float]] = {}
    for result in results:
        for item in result.get("node_metrics") or []:
            nodes.setdefault(str(item.get("node_id") or "unknown"), []).append(
                float(item.get("duration_ms") or 0)
            )
    node_performance = []
    for node_id, values in nodes.items():
        ordered = sorted(values)
        node_performance.append({
            "node_id": node_id,
            "calls": len(ordered),
            "mean_ms": round(statistics.mean(ordered), 1),
            "p50_ms": _percentile(ordered, 0.5),
            "p95_ms": _percentile(ordered, 0.95),
            "total_ms": round(sum(ordered), 1),
        })
    node_performance.sort(key=lambda item: item["total_ms"], reverse=True)
    decide_plan_metrics = [
        item
        for result in results
        for item in result.get("node_metrics") or []
        if str(item.get("node_id") or "") == "decide_plan_route"
    ]
    plan_node_metrics = [
        item
        for result in results
        for item in result.get("node_metrics") or []
        if str(item.get("node_id") or "") == "plan_sql"
    ]
    observed_plan_node_calls = len(nodes.get("plan_sql") or [])
    plan_node_successes = sum(
        item.get("status") == "success" for item in plan_node_metrics
    )
    planning_plan_call_accounting_complete = (
        observed_plan_node_calls == planning_plan_calls
        and plan_node_successes == planning_plan_calls
    )
    planning_decision_accounting_complete = (
        len(decide_plan_metrics) == n
        and sum(item.get("status") == "success" for item in decide_plan_metrics)
        == n
    )
    provider_boundary_stages = {
        "provider", "provider_complete", "output_validation",
    }
    planning_physical_attempts = [
        item for item in all_attempts
        if str(item.get("node_id") or "") == "plan_sql"
        and str(item.get("route") or "") == "planning"
        and item.get("pre_invoke_failure") is False
        and int(item.get("physical_attempt_budget_ms") or 0) > 0
        and str(item.get("completion_stage") or "") in provider_boundary_stages
    ]
    planning_physical_call_accounting_complete = (
        len(planning_physical_attempts) == planning_physical_llm_calls
    )
    timing_invariant_violations = []
    for result in results:
        sql_ms = float(result.get("tt_sql_ms") or 0)
        result_ms = float(result.get("tt_result_ms") or 0)
        complete_ms = float(result.get("tt_complete_ms") or 0)
        # A zero intermediate timestamp means that the stage was not reached;
        # failure classification handles that case.  Once all three exist their
        # monotonic ordering is a hard measurement invariant.
        if sql_ms > 0 and result_ms > 0 and complete_ms > 0 and not (
            sql_ms <= result_ms <= complete_ms
        ):
            timing_invariant_violations.append({
                "case_id": str(result.get("id") or "unknown"),
                "tt_sql_ms": sql_ms,
                "tt_result_ms": result_ms,
                "tt_complete_ms": complete_ms,
            })
    complete_p95 = _percentile(tt_complete, 0.95)
    llm_accounting = attempt_accounting(all_attempts)
    queue_waits = sorted(
        float(item.get("queue_wait_ms") or 0) for item in all_attempts
    )
    provider_latencies = sorted(
        float(item.get("provider_ms") or 0) for item in all_attempts
    )
    outcome_counts: dict[str, int] = {}
    for item in all_attempts:
        outcome = str(item.get("outcome_class") or item.get("status") or "unknown")
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
    return {
        "n": n,
        "first_sql_evidence_complete": bool(results) and all(
            bool(r.get("first_sql_metrics"))
            and (r.get("first_sql_metrics") or {}).get("result_evaluated") is True
            for r in results
        ),
        "first_non_empty": sum(
            bool((r.get("first_sql_metrics") or {}).get("non_empty"))
            for r in results
        ),
        "first_syntax_valid": sum(
            bool((r.get("first_sql_metrics") or {}).get("syntax_valid"))
            for r in results
        ),
        "first_guard_passed": sum(
            bool((r.get("first_sql_metrics") or {}).get("guard_passed"))
            for r in results
        ),
        "first_exec_success": sum(
            bool((r.get("first_sql_metrics") or {}).get("exec_success"))
            for r in results
        ),
        "first_structure_equivalent": sum(
            bool((r.get("first_sql_metrics") or {}).get("structure_equivalent"))
            for r in results
        ),
        "first_tables_correct": sum(
            bool((r.get("first_sql_metrics") or {}).get("tables_correct"))
            for r in results
        ),
        "first_table_scope_safe": sum(
            bool((r.get("first_sql_metrics") or {}).get("table_scope_safe"))
            for r in results
        ),
        "first_result_equivalent": sum(
            bool((r.get("first_sql_metrics") or {}).get("result_equivalent"))
            for r in results
        ),
        "correction_triggered": sum(
            bool((r.get("correction_metrics") or {}).get("triggered"))
            for r in results
        ),
        "correction_fixed_execution": sum(
            bool((r.get("correction_metrics") or {}).get("fixed_execution"))
            for r in results
        ),
        "correction_fixed_result": sum(
            bool((r.get("correction_metrics") or {}).get("fixed_result"))
            for r in results
        ),
        "correction_harmful_modification": sum(
            bool((r.get("correction_metrics") or {}).get("harmful_modification"))
            for r in results
        ),
        "correction_unattributable": sum(
            bool((r.get("correction_metrics") or {}).get("triggered"))
            and not bool((r.get("correction_metrics") or {}).get("attribution_valid"))
            for r in results
        ),
        "correction_llm_attempts": sum(
            int((r.get("correction_metrics") or {}).get("llm_attempts") or 0)
            for r in results
        ),
        "correction_additional_tokens": sum(
            int((r.get("correction_metrics") or {}).get("additional_tokens") or 0)
            for r in results
        ),
        "correction_unknown_token_attempts": sum(
            int((r.get("correction_metrics") or {}).get("unknown_token_attempts") or 0)
            for r in results
        ),
        "correction_additional_latency_ms": sum(
            int((r.get("correction_metrics") or {}).get("additional_latency_ms") or 0)
            for r in results
        ),
        "security_guard_bypass_count": sum(
            bool((r.get("first_sql_metrics") or {}).get("execution_attempted"))
            and not bool((r.get("first_sql_metrics") or {}).get("guard_passed"))
            for r in results
        ) + sum(
            bool((r.get("final_sql_metrics") or {}).get("execution_attempted"))
            and not bool((r.get("final_sql_metrics") or {}).get("guard_passed"))
            for r in results
        ),
        "result_evaluated": sum(r.get("result_evaluated") is True for r in results),
        "syntax_valid": sum(r["syntax_valid"] for r in results),
        "exec_success": sum(r["exec_success"] for r in results),
        "sql_equivalent": sum(r["sql_equivalent"] for r in results),
        "structure_equivalent": sum(r["structure_equivalent"] for r in results),
        "tables_correct": sum(r["tables_correct"] for r in results),
        "table_scope_safe": sum(
            bool(
                r.get("table_scope_safe")
                if "table_scope_safe" in r
                else (r.get("final_sql_metrics") or {}).get(
                    "table_scope_safe", False
                )
            )
            for r in results
        ),
        "table_scope_evidence_complete": sum(
            _table_scope_evidence_complete(r) for r in results
        ),
        "result_equivalent": sum(r["result_equivalent"] for r in results),
        "table_f1_mean": round(statistics.mean(r["table_f1"] for r in results), 4) if results else 0,
        "chart_generated": sum(r["chart_generated"] for r in results),
        "analysis_generated": sum(r["analysis_generated"] for r in results),
        "recommend_generated": sum(r["recommend_generated"] for r in results),
        "timing_protocol_version": TIMING_PROTOCOL_VERSION if timing_measurement_complete else "incomplete",
        "timing_measurement_complete": timing_measurement_complete,
        "timing_invariant_violations": timing_invariant_violations,
        "latency_mean": round(statistics.mean(tt_complete), 1) if tt_complete else 0,
        "latency_p50": _percentile(tt_complete, 0.5),
        "latency_p95": complete_p95,
        "latency_max": max(tt_complete) if tt_complete else 0,
        "tt_complete_mean": round(statistics.mean(tt_complete), 1) if tt_complete else 0,
        "tt_complete_p50": _percentile(tt_complete, 0.5),
        "tt_complete_p95": complete_p95,
        "tt_complete_max": max(tt_complete) if tt_complete else 0,
        "tt_sql_mean": round(statistics.mean(tt_sql), 1) if tt_sql else 0,
        "tt_sql_p95": _percentile(tt_sql, 0.95),
        "tt_result_mean": round(statistics.mean(tt_result), 1) if tt_result else 0,
        "tt_result_p95": _percentile(tt_result, 0.95),
        "llm_calls_mean": round(statistics.mean(llm_calls), 2) if llm_calls else 0,
        "llm_calls_p95": _percentile(llm_calls, 0.95),
        "tokens_mean": round(statistics.mean(total_tokens), 1) if total_tokens else 0,
        "tokens_p95": _percentile(total_tokens, 0.95),
        "tokens_max": float(token_peak.get("total_tokens") or 0),
        "tokens_max_case_id": (
            str(token_peak.get("id") or "")
            if float(token_peak.get("total_tokens") or 0) > 0 else ""
        ),
        "prompt_tokens_max": float(prompt_peak.get("prompt_tokens") or 0),
        "prompt_tokens_max_case_id": (
            str(prompt_peak.get("id") or "")
            if float(prompt_peak.get("prompt_tokens") or 0) > 0 else ""
        ),
        "llm_attempt_schema": LLM_ATTEMPT_SCHEMA,
        "llm_attempts": llm_accounting["attempts"],
        "llm_attempts_succeeded": llm_accounting["succeeded"],
        "llm_attempts_failed": llm_accounting["failed"],
        "llm_attempts_timed_out": llm_accounting["timed_out"],
        "llm_attempts_cancelled": llm_accounting["cancelled"],
        "llm_attempts_empty_output": llm_accounting["empty_output"],
        "llm_attempt_unknown_tokens": llm_accounting["unknown_tokens"],
        "llm_callback_missing": llm_accounting.get("callback_missing", 0),
        "llm_pre_invoke_failures": llm_accounting.get("pre_invoke_failures", 0),
        "llm_recoveries": llm_accounting.get("recoveries", 0),
        "llm_provider_request_id_unavailable": llm_accounting.get(
            "provider_request_id_unavailable", 0
        ),
        "llm_token_timing_unavailable": llm_accounting.get(
            "token_timing_unavailable", 0
        ),
        "llm_queue_wait_p95_ms": _percentile(queue_waits, .95),
        "llm_provider_p95_ms": _percentile(provider_latencies, .95),
        "llm_outcome_classes": outcome_counts,
        "llm_attempt_coverage_missing_cases": sum(
            result.get("llm_attempt_accounting_complete") is False
            for result in results
        ),
        "llm_attempt_accounting_complete": bool(results) and attempt_accounting_complete and llm_accounting["reconciled"],
        "failed_llm_calls": llm_accounting["attempts"] - llm_accounting["succeeded"],
        "failed_core_llm_calls": llm_accounting["core_failures"],
        "failed_optional_llm_calls": llm_accounting["optional_failures"],
        "llm_route_performance": route_performance,
        "prompt_payloads": prompt_payloads,
        "prompt_tokens_before": prompt_before,
        "prompt_tokens_after": prompt_after,
        "prompt_compression_rate": round(max(0.0, 1 - prompt_after / prompt_before), 4) if prompt_before else 0.0,
        "adaptive_candidate_count": adaptive_triggered,
        "adaptive_candidate_rate": round(adaptive_triggered / n, 4) if n else 0.0,
        "adaptive_candidate_extra_calls": adaptive_extra_calls,
        "adaptive_candidate_selected_b": adaptive_selected_b,
        "adaptive_candidate_valid_b": adaptive_valid_b,
        "adaptive_candidate_failed": adaptive_failed,
        "adaptive_candidate_unsafe_selected": adaptive_unsafe_selected,
        "adaptive_candidate_high_confidence_extra_calls": adaptive_high_confidence_extra,
        "adaptive_candidate_wait_total_ms": round(sum(adaptive_waits), 1),
        "adaptive_candidate_wait_p95_ms": _percentile(sorted(adaptive_waits), .95),
        "semantic_review_enabled": bool(
            getattr(app_config.query, "sql_semantic_review_enabled", True)
        ),
        "semantic_review_policy_version": SEMANTIC_REVIEW_POLICY_VERSION,
        "semantic_review_policy_mode_cases": review_decision_cases,
        "semantic_review_eligible_count": review_eligible,
        "semantic_review_triggered_count": review_triggered,
        "semantic_review_rate": round(
            review_triggered / review_eligible, 4
        ) if review_eligible else 0.0,
        "semantic_review_attempts": review_attempts,
        "semantic_review_extra_calls": review_extra_calls,
        "semantic_review_selected_b": review_selected_b,
        "semantic_review_valid_b": review_valid_b,
        "semantic_review_failed": review_failed,
        "semantic_review_failed_core": review_failed_core,
        "semantic_review_budget_skipped": review_budget_skipped,
        "semantic_review_unsafe_selected": review_unsafe_selected,
        "semantic_review_harmful_modification": review_harmful,
        "semantic_review_fixed_result": review_fixed_result,
        "semantic_review_first_correct_final_incorrect": (
            review_first_correct_final_incorrect
        ),
        "semantic_review_same_sql_execution_loss": (
            review_same_sql_execution_loss
        ),
        "semantic_review_selected_b_no_sql_delta": (
            review_selected_b_no_sql_delta
        ),
        "semantic_review_evidence_complete": bool(results) and review_evidence_complete,
        "semantic_review_coverage_complete": bool(results) and review_coverage_complete,
        "semantic_review_at_most_one_attempt": bool(results) and review_at_most_one,
        "semantic_review_result_attribution_complete": (
            bool(results) and review_attribution_complete
        ),
        "semantic_review_wait_total_ms": round(sum(review_waits), 1),
        "semantic_review_wait_p95_ms": _percentile(sorted(review_waits), .95),
        "candidate_extra_calls": candidate_extra_calls,
        "candidate_selected_b": selected_b,
        "candidate_valid_b": candidate_valid_b,
        "candidate_repaired": candidate_repaired,
        "candidate_failed": candidate_failed,
        "candidate_wait_total_ms": round(sum(candidate_waits), 1),
        "candidate_wait_p95_ms": _percentile(sorted(candidate_waits), .95),
        "candidate_selected_b_rate": round(
            selected_b / (adaptive_triggered + review_triggered), 4
        ) if (adaptive_triggered + review_triggered) else 0.0,
        "candidate_unsafe_selected": unsafe_selected,
        # Historical generic keys remain readable, while release governance
        # consumes the lane-specific keys above.
        "candidate_high_confidence_extra_calls": adaptive_high_confidence_extra,
        "structured_output_fallbacks": structured_fallbacks,
        "query_signals": {
            "schema_version": "evaluation-query-signal-aggregate-v2",
            "active_profile_counts": dict(
                sorted(query_signal_profile_counts.items())
            ),
            "route_policy_counts": dict(
                sorted(query_signal_route_policy_counts.items())
            ),
            "dictionary_version_counts": dict(
                sorted(query_signal_dictionary_version_counts.items())
            ),
            "route_counts": dict(sorted(query_signal_route_counts.items())),
            "query_encode_mode_counts": dict(
                sorted(query_signal_encode_mode_counts.items())
            ),
            "actual_remote_signal_counts": dict(
                sorted(query_signal_remote_signal_counts.items())
            ),
            "actual_consumer_signal_counts": dict(
                sorted(query_signal_consumer_signal_counts.items())
            ),
            "extractor_kind_counts": dict(
                sorted(query_signal_extractor_kind_counts.items())
            ),
            "consumer_kind_counts": dict(
                sorted(query_signal_consumer_kind_counts.items())
            ),
            "evidence_count": query_signal_evidence_count,
            "evidence_complete_count": query_signal_evidence_complete_count,
            "route_evidence_count": query_signal_route_evidence_count,
            "dictionary_fallback_count": (
                query_signal_dictionary_fallback_count
            ),
            "evidence_complete": bool(results)
            and query_signal_evidence_count == n
            and query_signal_evidence_complete_count == n
            and query_signal_route_evidence_count == n * 4
            and query_signal_dictionary_fallback_count == 0,
        },
        "planning_route": {
            "schema_version": "evaluation-planning-route-aggregate-v1",
            "active_profile_counts": dict(sorted(planning_profile_counts.items())),
            "configured_profile_counts": dict(
                sorted(planning_configured_profile_counts.items())
            ),
            "fallback_profile_counts": dict(
                sorted(planning_fallback_profile_counts.items())
            ),
            "fallback_class_counts": dict(
                sorted(planning_fallback_class_counts.items())
            ),
            "route_counts": dict(sorted(planning_route_counts.items())),
            "plan_calls": planning_plan_calls,
            "decide_node_calls": len(decide_plan_metrics),
            "decide_node_successes": sum(
                item.get("status") == "success" for item in decide_plan_metrics
            ),
            "observed_plan_node_calls": observed_plan_node_calls,
            "plan_node_successes": plan_node_successes,
            "physical_llm_calls": planning_physical_llm_calls,
            "physical_llm_attempts": len(planning_physical_attempts),
            "fallback_count": planning_fallback_count,
            "policy_fallback_count": planning_policy_fallback_count,
            "unsafe_fallback_count": planning_unsafe_fallback_count,
            "degraded_count": planning_degraded_count,
            "evidence_count": planning_evidence_count,
            "evidence_complete_count": planning_evidence_complete_count,
            "plan_call_accounting_complete": (
                planning_plan_call_accounting_complete
            ),
            "decision_accounting_complete": (
                planning_decision_accounting_complete
            ),
            "physical_llm_call_accounting_complete": (
                planning_physical_call_accounting_complete
            ),
            "evidence_complete": bool(results)
            and planning_evidence_count == n
            and planning_evidence_complete_count == n
            and sum(planning_fallback_class_counts.values()) == n
            and planning_fallback_count == planning_policy_fallback_count
            and planning_unsafe_fallback_count == 0
            and planning_degraded_count == 0
            and planning_plan_call_accounting_complete
            and planning_decision_accounting_complete
            and planning_physical_call_accounting_complete,
        },
        "semantic_repair": {
            "schema_version": "evaluation-semantic-repair-aggregate-v1",
            "active_profile_counts": dict(sorted(repair_profile_counts.items())),
            "configured_profile_counts": dict(
                sorted(repair_configured_profile_counts.items())
            ),
            "attempted_rule_counts": dict(
                sorted(repair_attempted_rule_counts.items())
            ),
            "applied_rule_counts": dict(
                sorted(repair_applied_rule_counts.items())
            ),
            "changed_count": repair_changed_count,
            "fallback_count": repair_fallback_count,
            "degraded_count": repair_degraded_count,
            "evidence_count": repair_evidence_count,
            "evidence_complete_count": repair_evidence_complete_count,
            "invocation_count": repair_invocation_count,
            "invocation_evidence_complete_count": (
                repair_invocation_complete_count
            ),
            "invocation_profile_counts": dict(
                sorted(repair_invocation_profile_counts.items())
            ),
            "invocation_phase_counts": dict(
                sorted(repair_invocation_phase_counts.items())
            ),
            "selected_invocation_count": repair_selected_invocation_count,
            "invocation_fallback_count": repair_invocation_fallback_count,
            "ledger_complete_cases": repair_ledger_complete_cases,
            "invocation_evidence_complete": bool(results)
            and repair_ledger_complete_cases == n
            and repair_invocation_count == repair_invocation_complete_count
            and repair_selected_invocation_complete_cases == n
            and repair_selected_invocation_count == n,
            "evidence_complete": bool(results)
            and repair_evidence_count == n
            and repair_evidence_complete_count == n
            and repair_fallback_count == 0
            and repair_degraded_count == 0
            and repair_invocation_fallback_count == 0
            and repair_ledger_complete_cases == n
            and repair_invocation_count == repair_invocation_complete_count
            and repair_selected_invocation_complete_cases == n
            and repair_selected_invocation_count == n,
        },
        "semantic_coverage_cases": len(semantic_coverage_scores),
        "semantic_coverage_mean": round(
            statistics.mean(semantic_coverage_scores), 4
        ) if semantic_coverage_scores else 0.0,
        "semantic_unresolved_cases": semantic_unresolved_cases,
        "semantic_unsafe_compile_cases": semantic_unsafe_compile_cases,
        "request_deadline_contract_complete": bool(results) and deadline_contract_complete,
        "request_deadline_exhausted_cases": sum(value <= 0 for value in deadline_remaining),
        "cost_estimate_available": bool(
            float(app_config.llm.input_cost_per_million or 0) > 0
            or float(app_config.llm.output_cost_per_million or 0) > 0
        ) and int(llm_accounting.get("unknown_tokens") or 0) == 0,
        "estimated_cost_total": round(sum(float(r.get("estimated_cost") or 0) for r in results), 8),
        "cost_currency": next((str(r.get("cost_currency")) for r in results if r.get("cost_currency")), "USD"),
        "degraded_cases": sum(bool(r.get("degradations")) for r in results),
        "core_degradation_count": sum(
            item.get("affects_core_path")
            for result in results for item in map(normalize_degradation, result.get("degradations") or [])
        ),
        "optional_degradation_count": sum(
            not item.get("affects_core_path")
            for result in results for item in map(normalize_degradation, result.get("degradations") or [])
        ),
        "degradation_evidence_complete": degradation_evidence_complete,
        "node_performance": node_performance,
        "retry_triggered": len(retry_cases),
        "retry_success": sum(1 for r in retry_cases if r["exec_success"]),
        "infrastructure_errors": sum(
            1 for r in results if r.get("failure_type") == "infrastructure"
        ),
    }


def results_form_valid_run(results: list[dict], aggregate_data: dict | None = None) -> bool:
    """Only infrastructure-clean runs may become comparison/UI baselines.

    Pipeline or semantic failures are valid evaluation evidence.  Infrastructure
    failures are different: they mean one or more cases were never evaluated, so
    accepting such a batch would silently replace a complete baseline with a
    partial outage artifact.
    """
    infrastructure_clean = bool(results) and all(
        result.get("failure_type") != "infrastructure" for result in results
    )
    provider_clean = not provider_failure_audit(results)["invalidates_run"]
    timing_clean = (
        aggregate_data is None
        or (
            aggregate_data.get("timing_protocol_version") == TIMING_PROTOCOL_VERSION
            and not aggregate_data.get("timing_invariant_violations")
        )
    )
    evidence_clean = (
        aggregate_data is None
        or (
            aggregate_data.get("llm_attempt_schema") == LLM_ATTEMPT_SCHEMA
            and aggregate_data.get("llm_attempt_accounting_complete") is True
            and aggregate_data.get("degradation_evidence_complete") is True
        )
    )
    return infrastructure_clean and provider_clean and timing_clean and evidence_clean


async def resolve_workspace_id(meta_session, explicit: str | None = None) -> str:
    """Resolve the tenant used by an evaluation without leaking across workspaces.

    Priority: CLI argument -> EVAL_WORKSPACE_ID -> workspace owning most enabled
    metric/training metadata -> oldest workspace.  The selected id is persisted in
    the run artifact so that a result can be reproduced and audited.
    """
    requested = (explicit or os.getenv("EVAL_WORKSPACE_ID") or "").strip()
    if requested:
        return requested

    for model in (MetricInfoMySQL, TrainingInfoMySQL):
        statement = (
            select(model.workspace_id)
            .where(model.workspace_id.is_not(None))
            .group_by(model.workspace_id)
            .order_by(func.count().desc(), model.workspace_id)
        )
        workspace_id = (await meta_session.execute(statement)).scalars().first()
        if workspace_id:
            return str(workspace_id)

    statement = select(WorkspaceMySQL.id).order_by(WorkspaceMySQL.created_at, WorkspaceMySQL.id)
    workspace_id = (await meta_session.execute(statement)).scalars().first()
    if workspace_id:
        return str(workspace_id)
    raise RuntimeError(
        "评测无法确定工作空间：请先创建工作空间，或在 .env 配置 EVAL_WORKSPACE_ID"
    )


def aggregate_by_category(results: list[dict]) -> list[dict]:
    cats: dict[str, list[dict]] = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r)
    rows = []
    for cat, rs in cats.items():
        n = len(rs)
        rows.append({
            "category": cat,
            "n": n,
            "exec_success": sum(r["exec_success"] for r in rs),
            "sql_equivalent": sum(r["sql_equivalent"] for r in rs),
            "result_equivalent": sum(r["result_equivalent"] for r in rs),
        })
    return sorted(rows, key=lambda x: x["category"])


def render_report(
    results: list[dict],
    agg: dict,
    cat_rows: list[dict],
    attribution_summary: dict | None = None,
    invocation_audit: dict | None = None,
    execution_audit: dict | None = None,
) -> str:
    ts = dt.datetime.now(dt.UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    n = agg["n"]
    lines = []
    lines.append("# NL2SQL 评测报告\n")
    lines.append(f"> 生成时间：{ts} ｜ 黄金集用例数：{n}\n")
    if execution_audit:
        planned = int(execution_audit.get("planned_count") or n)
        executed = int(execution_audit.get("executed_count") or n)
        not_executed = int(execution_audit.get("not_executed_count") or 0)
        lines.append(
            f"> 执行审计：计划 {planned} ｜ 实际执行 {executed} ｜ "
            f"未执行 {not_executed}；所有正确率分母仅包含实际执行题。\n"
        )
        if execution_audit.get("early_terminated"):
            lines.append(
                "> **本次运行无效：检测到不可重试的模型服务基础设施失败，"
                "后续题已安全停止且不计为模型错题。** "
                f"触发题：`{execution_audit.get('trigger_case_id')}` ｜ "
                f"原因：`{execution_audit.get('trigger_reason_code')}`\n"
            )
    lines.append(
        f"> Evaluation Profile：`{(results[0].get('evaluation_profile') if results else 'unknown')}` ｜ "
        f"Analysis：`{'on' if results and results[0].get('analysis_enabled') else 'off'}`\n"
    )
    lines.append("## 整体指标\n")
    lines.append("| 指标 | 命中/总数 | 比率 |")
    lines.append("|---|---|---|")
    lines.append(f"| 语法正确率 | {agg['syntax_valid']}/{n} | {_pct(agg['syntax_valid'], n)} |")
    lines.append(f"| 执行成功率 | {agg['exec_success']}/{n} | {_pct(agg['exec_success'], n)} |")
    lines.append(f"| SQL 等价率（近似） | {agg['sql_equivalent']}/{n} | {_pct(agg['sql_equivalent'], n)} |")
    lines.append(f"| SQL 结构一致率 | {agg['structure_equivalent']}/{n} | {_pct(agg['structure_equivalent'], n)} |")
    lines.append(f"| 表选择准确率 | {agg['tables_correct']}/{n} | {_pct(agg['tables_correct'], n)} |")
    result_denominator = int(agg.get("result_evaluated", n))
    lines.append(f"| 结果一致率 | {agg['result_equivalent']}/{result_denominator} | {_pct(agg['result_equivalent'], result_denominator)} |")
    lines.append(f"| 图表生成率 | {agg['chart_generated']}/{n} | {_pct(agg['chart_generated'], n)} |")
    lines.append(f"| 分析生成率 | {agg['analysis_generated']}/{n} | {_pct(agg['analysis_generated'], n)} |")
    lines.append(f"| 推荐生成率 | {agg['recommend_generated']}/{n} | {_pct(agg['recommend_generated'], n)} |")
    lines.append(f"| 纠错触发率 | {agg['retry_triggered']}/{n} | {_pct(agg['retry_triggered'], n)} |")
    retry_d = agg["retry_triggered"]
    lines.append(f"| 纠错成功率 | {agg['retry_success']}/{retry_d} | {_pct(agg['retry_success'], retry_d)} |")
    lines.append("")
    lines.append("## First SQL / Correction / Final SQL\n")
    lines.append("| 层次 | 语法 | Guard | 执行 | 表选择 | 结构 | 严格结果 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| First SQL | {agg.get('first_syntax_valid', 0)}/{n} | "
        f"{agg.get('first_guard_passed', 0)}/{n} | {agg.get('first_exec_success', 0)}/{n} | "
        f"{agg.get('first_tables_correct', 0)}/{n} | "
        f"{agg.get('first_structure_equivalent', 0)}/{n} | "
        f"{agg.get('first_result_equivalent', 0)}/{n} |"
    )
    lines.append(
        f"| Final SQL | {agg['syntax_valid']}/{n} | — | {agg['exec_success']}/{n} | "
        f"{agg['tables_correct']}/{n} | {agg['structure_equivalent']}/{n} | "
        f"{agg['result_equivalent']}/{int(agg.get('result_evaluated', n))} |"
    )
    correction_d = int(agg.get("correction_triggered") or 0)
    lines.extend([
        "",
        (
            f"- Correction：触发 {correction_d}/{n}；修复执行 "
            f"{agg.get('correction_fixed_execution', 0)}；修复严格结果 "
            f"{agg.get('correction_fixed_result', 0)}；有害修改 "
            f"{agg.get('correction_harmful_modification', 0)}；不可直接归因 "
            f"{agg.get('correction_unattributable', 0)}；"
            "额外 Attempt/Token/未知Token尝试/延迟 "
            f"{agg.get('correction_llm_attempts', 0)}/"
            f"{agg.get('correction_additional_tokens', 0)}/"
            f"{agg.get('correction_unknown_token_attempts', 0)}/"
            f"{agg.get('correction_additional_latency_ms', 0)} ms。"
        ),
        f"- First SQL 证据：{'完整' if agg.get('first_sql_evidence_complete') else '不完整'}。",
        "",
    ])
    lines.append("## 延迟分布（ms）\n")
    lines.append(
        f"> 协议：`{agg.get('timing_protocol_version', 'unknown')}` ｜ "
        f"时序不变量异常：{len(agg.get('timing_invariant_violations') or [])}\n"
    )
    lines.append("| 平均 | P50 | P95 | 最大 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| {agg['latency_mean']} | {agg['latency_p50']} | {agg['latency_p95']} | {agg['latency_max']} |")
    lines.append("")
    lines.append("## 节点级性能、Token 与成本\n")
    lines.append("| TTSQL 平均/P95 | TTResult 平均/P95 | LLM 调用平均/P95 | Token 平均/P95 | 估算总成本 | 降级用例 |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(
        f"| {agg['tt_sql_mean']} / {agg['tt_sql_p95']} ms | "
        f"{agg['tt_result_mean']} / {agg['tt_result_p95']} ms | "
        f"{agg['llm_calls_mean']} / {agg['llm_calls_p95']} | "
        f"{agg['tokens_mean']} / {agg['tokens_p95']} | "
        f"{(str(agg['estimated_cost_total']) + ' ' + agg['cost_currency']) if agg.get('cost_estimate_available') else '未知（未配置价格）'} | "
        f"{agg['degraded_cases']}/{n} |"
    )
    lines.append(
        f"\n- 模型尝试：{agg.get('llm_attempts', 0)} = 成功 {agg.get('llm_attempts_succeeded', 0)} + "
        f"失败 {agg.get('llm_attempts_failed', 0)} + 超时 {agg.get('llm_attempts_timed_out', 0)} + "
        f"取消 {agg.get('llm_attempts_cancelled', 0)}；核心/可选失败 "
        f"{agg.get('failed_core_llm_calls', 0)}/{agg.get('failed_optional_llm_calls', 0)}；"
        f"结构化输出降级：{agg['structured_output_fallbacks']}；"
        f"Prompt payload：{agg['prompt_tokens_before']} → {agg['prompt_tokens_after']} tokens "
        f"（压缩 {agg['prompt_compression_rate']:.1%}）；旧自适应扩展触发率："
        f"{agg['adaptive_candidate_rate']:.1%}。"
    )
    lines.append(
        f"- 自适应扩展：触发 {agg.get('adaptive_candidate_count', 0)}，"
        f"有效 B {agg.get('adaptive_candidate_valid_b', 0)}，"
        f"选择 B {agg.get('adaptive_candidate_selected_b', 0)}，"
        f"失败 {agg.get('adaptive_candidate_failed', 0)}。"
    )
    lines.append(
        f"- 语义复核：eligible/triggered/attempt "
        f"{agg.get('semantic_review_eligible_count', 0)}/"
        f"{agg.get('semantic_review_triggered_count', 0)}/"
        f"{agg.get('semantic_review_attempts', 0)}；选择 B "
        f"{agg.get('semantic_review_selected_b', 0)}；核心失败/有害修改/预算跳过 "
        f"{agg.get('semantic_review_failed_core', 0)}/"
        f"{agg.get('semantic_review_harmful_modification', 0)}/"
        f"{agg.get('semantic_review_budget_skipped', 0)}；"
        f"首对终错/同 SQL 执行损失/选 B 无 SQL 变化 "
        f"{agg.get('semantic_review_first_correct_final_incorrect', 0)}/"
        f"{agg.get('semantic_review_same_sql_execution_loss', 0)}/"
        f"{agg.get('semantic_review_selected_b_no_sql_delta', 0)}。"
    )
    lines.append(
        f"- 单题 Token 最大值：{agg.get('tokens_max', 0):.0f}（{agg.get('tokens_max_case_id') or 'N/A'}）；"
        f"单题 Prompt Token 最大值：{agg.get('prompt_tokens_max', 0):.0f}"
        f"（{agg.get('prompt_tokens_max_case_id') or 'N/A'}）。"
    )
    lines.append(
        f"- 语义覆盖：{agg.get('semantic_coverage_cases', 0)}/{n} 题有证据，"
        f"平均 {float(agg.get('semantic_coverage_mean') or 0):.1%}，"
        f"未闭合 {agg.get('semantic_unresolved_cases', 0)} 题。"
    )
    lines.append(
        f"- 调用治理：恢复调用 {agg.get('llm_recoveries', 0)}，"
        f"排队/Provider P95 {agg.get('llm_queue_wait_p95_ms', 0)} / "
        f"{agg.get('llm_provider_p95_ms', 0)} ms；"
        f"总截止时间耗尽 {agg.get('request_deadline_exhausted_cases', 0)} 题，"
        f"相对时间证据{'完整' if agg.get('request_deadline_contract_complete') else '不完整'}。"
    )
    lines.append("")
    lines.append("| 节点 | 调用数 | 平均 ms | P50 ms | P95 ms | 累计 ms |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for node in agg["node_performance"]:
        lines.append(
            f"| {node['node_id']} | {node['calls']} | {node['mean_ms']} | "
            f"{node['p50_ms']} | {node['p95_ms']} | {node['total_ms']} |"
        )
    if agg.get("llm_route_performance"):
        lines.append("\n| 模型节点 | Route / 模型 | 调用 / 失败 | Token/调用 | 模型 P95 ms |")
        lines.append("|---|---|---:|---:|---:|")
        for node in agg["llm_route_performance"]:
            lines.append(
                f"| {node['node_id']} | {node['route']} / {', '.join(node['models'])} | "
                f"{node['calls']} / {node['failed_calls']} | {node['tokens_per_call']} | {node['latency_p95_ms']} |"
            )
    lines.append("")
    lines.append("## 分类别指标\n")
    lines.append("| 类别 | 用例数 | 执行成功 | SQL 等价 | 结果一致 |")
    lines.append("|---|---|---|---|---|")
    for c in cat_rows:
        lines.append(f"| {c['category']} | {c['n']} | {_pct(c['exec_success'], c['n'])} | {_pct(c['sql_equivalent'], c['n'])} | {_pct(c['result_equivalent'], c['n'])} |")
    lines.append("")
    if invocation_audit and invocation_audit.get("required"):
        lines.append("## 模型物理调用门禁\n")
        lines.append(
            f"> 合同：`{invocation_audit.get('contract')}` ｜ "
            f"状态：`{'PASS' if invocation_audit.get('passed') else 'FAIL'}`\n"
        )
        lines.append("| 期望用例 | 模型路由用例 | generate_sql 物理调用用例 | 物理尝试 | 回退用例 |")
        lines.append("|---:|---:|---:|---:|---:|")
        lines.append(
            f"| {invocation_audit.get('expected_cases', 0)} | "
            f"{invocation_audit.get('policy_model_cases', 0)} | "
            f"{invocation_audit.get('physical_generate_sql_cases', 0)} | "
            f"{invocation_audit.get('physical_generate_sql_attempts', 0)} | "
            f"{len(invocation_audit.get('fallback_cases') or [])} |"
        )
        lines.append("")
        lines.append("| 节点 / Route | 尝试 | 唯一用例 | 状态 | Token | Token 未知 |")
        lines.append("|---|---:|---:|---|---:|---:|")
        for row in invocation_audit.get("per_node") or []:
            statuses = ", ".join(
                f"{name}={count}"
                for name, count in (row.get("statuses") or {}).items()
            )
            lines.append(
                f"| {row.get('node_id')} / {row.get('route')} | "
                f"{row.get('attempts', 0)} | {row.get('unique_cases', 0)} | "
                f"{statuses or '—'} | {row.get('total_tokens', 0)} | "
                f"{row.get('unknown_tokens', 0)} |"
            )
        lines.append("")
    attribution_summary = attribution_summary or summarize_attributions(results)
    lines.append("## 正确性归因与定向回归\n")
    if attribution_summary["root_causes"]:
        lines.append("| 根因 | 所属阶段 | 失败数 | 用例 | 定向动作 |")
        lines.append("|---|---|---:|---|---|")
        for cause in attribution_summary["root_causes"]:
            lines.append(
                f"| {cause['label']} | {cause['stage']} | {cause['count']} | "
                f"{', '.join(cause['case_ids'])} | {cause['recommendation']} |"
            )
        lines.append("")
        for action in attribution_summary["targeted_actions"]:
            lines.append(f"- `{action['regression_command']}`")
    else:
        lines.append("全部用例结果一致，无失败根因。")
    lines.append("")
    lines.append("## 逐用例明细\n")
    lines.append("| ID | 问题 | 类别 | First结果 | Final语法 | Final执行 | SQL等价 | 表选择 | Final结果 | 归因 | 延迟ms | 纠错次数 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['id']} | {r['question']} | {r['category']} | "
            f"{'✅' if (r.get('first_sql_metrics') or {}).get('result_equivalent') else '❌'} | "
            f"{'✅' if r['syntax_valid'] else '❌'} | "
            f"{'✅' if r['exec_success'] else '❌'} | "
            f"{'✅' if r['sql_equivalent'] else '❌'} | "
            f"{'✅' if r['tables_correct'] else '❌'} | "
            f"{'✅' if r['result_equivalent'] else '❌'} | "
            f"{(r.get('attribution') or {}).get('label', '—')} | "
            f"{r['latency_ms']} | {r['retry_count']} |"
        )
    # 失败用例详情
    failed = [r for r in results if not r["exec_success"] or not r["result_equivalent"]]
    if failed:
        lines.append("\n## 失败/不一致用例详情\n")
        for r in failed:
            lines.append(f"### {r['id']} — {r['question']}")
            if r["error"]:
                lines.append(f"- 错误：{r['error']}")
            lines.append(f"- 生成 SQL：`{r['gen_sql']}`")
            lines.append(f"- 期望 SQL：`{r['expected_sql']}`")
            lines.append("")
    return "\n".join(lines)


async def main(only: str | None = None, tier: str | None = None,
               workspace_id: str | None = None, *, experiment_id: str | None = None,
               candidate_label: str | None = None, repeat_index: int | None = None,
               cache_state: str | None = None,
               evaluation_profile: str = "core-default",
               dataset_suite: str = "easy_sql_v1",
               embedding_session: EvaluationEmbeddingSession | None = None) -> bool:
    profile = resolve_evaluation_profile(evaluation_profile)
    cases = load_cases(only, tier, dataset_suite)
    all_cases = load_cases(suite=dataset_suite)
    selected_ids = {str(item.get("id")) for item in cases}
    all_ids = {str(item.get("id")) for item in all_cases}
    full_dataset = len(cases) == len(all_cases) and selected_ids == all_ids
    suite_spec = SUITE_SPECS[dataset_suite]
    force_model_sql = bool(
        suite_spec.get("llm_invocation_contract")
        == GENERATE_SQL_PHYSICAL_ATTEMPT_PER_CASE
    )
    model_sql_fallback_enabled = bool(
        app_config.query.deterministic_sql_model_fallback_enabled
        and not force_model_sql
    )
    governance_snapshot = dataset_suite_report(dataset_suite)
    experiment = None
    if any(value is not None for value in (experiment_id, candidate_label, repeat_index, cache_state)):
        if not all(value is not None for value in (experiment_id, candidate_label, repeat_index, cache_state)):
            raise ValueError(
                "experiment-id、candidate-label、repeat-index、cache-state 必须同时提供"
            )
        route_manifest = llm_route_manifest()
        sql_route = dict(route_manifest.get("sql") or {})
        correction_route = dict(route_manifest.get("correction") or {})
        normalized_candidate = str(candidate_label).strip().upper()
        if normalized_candidate in TODO97_CANDIDATE_SET:
            expected_candidate = candidate_capability(normalized_candidate)
            route_identity = {
                "candidate_id": sql_route.get("candidate_id"),
                "model_name": sql_route.get("model_name"),
                "provider": sql_route.get("provider"),
                "model_provider": sql_route.get("model_provider"),
                "access_mode": sql_route.get("access_mode"),
            }
            expected_identity = _todo97_expected_route_identity(expected_candidate)
            recipe = expected_candidate.get("phase1") or {}
            if (
                route_identity != expected_identity
                or str(sql_route.get("thinking_mode") or "auto")
                != str(recipe.get("thinking_mode") or "auto")
                or float(sql_route.get("temperature") or 0)
                != float(recipe.get("temperature") or 0)
            ):
                raise ValueError(
                    f"TODO-97 {normalized_candidate} 与生效 SQL route 身份/配方不一致"
                )
            d0 = candidate_capability("D0")
            if not all((
                correction_route.get("candidate_id") == d0["candidate_id"],
                correction_route.get("model_name") == d0["model_id"],
                correction_route.get("provider") == d0["provider"],
                correction_route.get("thinking_mode")
                == d0["phase1"]["thinking_mode"],
            )):
                raise ValueError(
                    "TODO-97 Phase 1 必须固定 correction route 为 D0"
                )
        experiment = build_experiment_metadata(
            experiment_id=str(experiment_id), candidate_label=str(candidate_label),
            repeat_index=int(repeat_index), cache_state=str(cache_state),
            provider_parameters={
                # 记录真实解析后的 SQL/Correction 部署，而不是全局默认模型。
                # endpoint 和 credential source 在 manifest 中均为不可逆指纹。
                "model_name": sql_route.get("model_name"),
                "provider": sql_route.get("provider"),
                "model_provider": sql_route.get("model_provider"),
                "temperature": sql_route.get("temperature"),
                "thinking_mode": sql_route.get("thinking_mode"),
                "reasoning_effort": sql_route.get("reasoning_effort"),
                "sql_route": sql_route,
                "correction_route": correction_route,
                "seed": "unavailable",
                "timeout_seconds": sql_route.get("timeout_seconds"),
                "max_retries": sql_route.get("max_retries"),
                "max_completion_tokens": sql_route.get("max_completion_tokens"),
                "timeout_recovery_enabled": True,
                "force_model_sql": force_model_sql,
                "model_sql_fallback_enabled": model_sql_fallback_enabled,
                "model_routing_contract": (
                    suite_spec.get("llm_invocation_contract")
                    or "automatic_policy"
                ),
                "recovery_reserve_ms": (
                    app_config.query.llm_recovery_reserve_ms
                ),
                "recovery_min_budget_ms": (
                    app_config.query.llm_recovery_min_budget_ms
                ),
            },
            evaluation_profile=profile["name"],
        )
        ensure_experiment_repeat_available(
            str(experiment_id),
            int(repeat_index),
            RUNS_DIR,
            dataset_suite,
        )
    owns_embedding_session = embedding_session is None
    active_embedding_session = embedding_session or EvaluationEmbeddingSession()
    if owns_embedding_session:
        await active_embedding_session.start()
    try:
        warmup_evidence = active_embedding_session.evidence_for_run(
            cache_state=str(cache_state) if cache_state is not None else None,
        )
        cache_control = apply_evaluation_cache_control(experiment)
        embedding_warmup_checks = evaluation_embedding_evidence_checks(
            warmup_evidence,
            cache_control,
        )
        if not embedding_warmup_checks["complete"]:
            failed = [
                name
                for name, passed in embedding_warmup_checks.items()
                if name != "complete" and not passed
            ]
            raise RuntimeError(
                "评测 BGE 预热/缓存证据合同未通过：" + ",".join(failed)
            )
    except BaseException:
        if owns_embedding_session:
            try:
                await active_embedding_session.close()
            except Exception:  # noqa: BLE001  # 清理失败不能覆盖前置根因
                logger.exception("评测前置失败后关闭 BGE 驻留会话失败")
        raise
    try:
        es_client_manager.init()
        dw_mysql_client_manager.init()
        meta_mysql_client_manager.init()
        qdrant_client_manager.init()
        async with (dw_mysql_client_manager.session_factory() as dw_session,
                    meta_mysql_client_manager.session_factory() as meta_session):
            selected_workspace_id = await resolve_workspace_id(meta_session, workspace_id)
            logger.info(f"评测工作空间：{selected_workspace_id}")
            dw_repo = DWMysqlRepository(dw_session)
            meta_repo = MetaMysqlRepository(meta_session, selected_workspace_id)
            context = build_nl2sql_evaluation_context(
                dw_mysql_repo=dw_repo,
                meta_mysql_repo=meta_repo,
                value_es_repo=ValueESRepository(es_client_manager.client),
                column_qdrant_repo=ColumnQdrantRepository(qdrant_client_manager.client),
                metric_qdrant_repo=MetricQdrantRepository(qdrant_client_manager.client),
                training_qdrant_repo=TrainingQdrantRepository(qdrant_client_manager.client),
                embedding_client=active_embedding_session.client,
                access_policy=PolicySnapshot(
                    policies=(),
                    user_id="eval-user",
                    username="eval",
                    workspace_id=selected_workspace_id,
                    role="owner",
                ),
                workspace_id=selected_workspace_id,
                force_model_sql=force_model_sql,
                force_model_sql_reason=(
                    "evaluation_physical_generate_sql_contract"
                    if force_model_sql else ""
                ),
                model_sql_fallback_enabled=model_sql_fallback_enabled,
            )
            reference_audit = await validate_reference_result_contract(
                all_cases,
                dw_repo,
                suite_spec.get("reference_result_contract"),
            )
            if not reference_audit.get("passed"):
                raise ValueError(
                    f"suite {dataset_suite} Reference SQL 非空/投影门禁失败；"
                    "已在任何 Provider 调用前终止"
                )
            fingerprint = await collect_context_fingerprint(
                cases, all_cases, context, profile["name"],
            )
            logger.info(f"评测启动：suite={dataset_suite}，共 {len(cases)} 条用例")
            results, execution_audit = await run_cases_fail_fast(
                cases,
                context,
                dw_repo,
                profile["name"],
            )
            embedding_runtime_final = (
                active_embedding_session.finalize_run_evidence(
                    warmup_evidence
                )
            )
            embedding_runtime_health = embedding_runtime_health_audit(
                results,
                embedding_runtime_final,
                dict(warmup_evidence.get("suite_metrics_epoch") or {}),
            )
            agg = aggregate(results)
            governance_snapshot["runtime_reference_validation"] = reference_audit
            reference_observed = bool(
                (
                    reference_audit.get("required")
                    and reference_audit.get("passed")
                    and int(reference_audit.get("executed_cases") or 0)
                    == len(all_cases)
                )
                or (
                    not reference_audit.get("required")
                    and int(agg.get("result_evaluated") or 0) == len(results)
                )
            )
            governance_snapshot["runtime_validation"] = {
                **dict(governance_snapshot.get("runtime_validation") or {}),
                "reference_sql_observed": reference_observed,
                "result_equivalence_observed": bool(
                    int(agg.get("result_evaluated") or 0) == len(results)
                ),
            }
            runtime_validation = governance_snapshot["runtime_validation"]
            runtime_validation["ready"] = bool(
                runtime_validation.get("reference_sql_observed")
                and runtime_validation.get("result_equivalence_observed")
                and not execution_audit.get("early_terminated")
                and not (
                    execution_audit.get("provider_failure_audit") or {}
                ).get("invalidates_run")
            )
            suite_spec = SUITE_SPECS[dataset_suite]
            invocation_audit = evaluate_llm_invocation_contract(
                results,
                suite_spec.get("llm_invocation_contract"),
                expected_case_ids=[str(item.get("id")) for item in all_cases],
            )
            invocation_audit["suite_expected_cases"] = len(all_cases)
            invocation_audit["full_suite"] = full_dataset
            if invocation_audit.get("required") and not full_dataset:
                invocation_audit["passed"] = False
                invocation_audit["failure_reason"] = "partial_suite"
            base_run_valid = results_form_valid_run(results, agg)
            base_run_valid = bool(
                base_run_valid
                and not execution_audit.get("early_terminated")
            )
            valid = bool(base_run_valid and invocation_audit.get("passed"))
            cat_rows = aggregate_by_category(results)
            attribution_summary = summarize_attributions(results, selected_workspace_id)
            report = render_report(
                results,
                agg,
                cat_rows,
                attribution_summary,
                invocation_audit,
                execution_audit,
            )
            report_dir = REPORT_DIR
            report_dir.mkdir(parents=True, exist_ok=True)
            now = dt.datetime.now(dt.UTC).astimezone()
            ts = now.strftime("%Y%m%d_%H%M%S_%f")
            immutable = full_dataset or experiment is not None
            out = (
                report_dir / f"report_{dataset_suite}_{ts}.md"
                if immutable
                else targeted_report_path(dataset_suite, report_dir)
            )
            out.write_text(report, encoding="utf-8")
            if full_dataset:
                (report_dir / f"report_{dataset_suite}_latest.md").write_text(
                    report, encoding="utf-8",
                )
            # 完整基线/实验保留不可变时间戳；定向诊断只覆盖各 suite
            # 自己的 targeted_<suite>_latest，防止跨难度覆盖或让 1～2 题
            # 试跑持续污染正式 runs、reports 与候选基线注册表。
            RUNS_DIR.mkdir(parents=True, exist_ok=True)
            provider_request_ids = sorted({
                str(attempt.get("provider_request_id"))
                for case_result in results
                for attempt in (case_result.get("llm_attempts") or [])
                if attempt.get("provider_request_id_available") is True
                and attempt.get("provider_request_id")
                and attempt.get("provider_request_id") != "unavailable"
            })
            provider_attempt_count = sum(
                1 for case_result in results
                for _ in (case_result.get("llm_attempts") or [])
            )
            pipeline_checks = pipeline_evidence_checks(
                fingerprint, agg, results,
            )
            pipeline_evidence_eligible = bool(
                pipeline_checks["query_signal_evidence_complete"]
                and pipeline_checks["planning_route_evidence_complete"]
                and pipeline_checks["semantic_repair_evidence_complete"]
            )
            run_payload = {
                "run_id": ts,
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                "valid": valid,
                "base_run_valid": base_run_valid,
                "workspace_id": selected_workspace_id,
                "dataset_suite": dataset_suite,
                "artifact_scope": "release_candidate",
                "baseline_eligible": evaluation_baseline_eligible(
                    valid=valid,
                    full_dataset=full_dataset,
                    pipeline_evidence_eligible=pipeline_evidence_eligible,
                    embedding_warmup_eligible=embedding_warmup_checks[
                        "complete"
                    ],
                    embedding_runtime_eligible=embedding_runtime_health[
                        "eligible"
                    ],
                    aggregate=agg,
                ),
                "embedding_warmup_eligible": embedding_warmup_checks[
                    "complete"
                ],
                "pipeline_evidence_eligible": pipeline_evidence_eligible,
                "pipeline_evidence_checks": pipeline_checks,
                "execution_audit": execution_audit,
                "provider_failure_audit": execution_audit.get(
                    "provider_failure_audit"
                ),
                "llm_invocation_contract": suite_spec.get(
                    "llm_invocation_contract"
                ),
                "llm_invocation_audit": invocation_audit,
                "experiment": experiment,
                "evaluation_profile": profile,
                "cache_control": {
                    **cache_control,
                    "after": result_cache.stats(),
                },
                "embedding_warmup": warmup_evidence,
                "embedding_warmup_checks": embedding_warmup_checks,
                "embedding_runtime_final": embedding_runtime_final,
                "embedding_runtime_health": embedding_runtime_health,
                "embedding_runtime_eligible": embedding_runtime_health[
                    "eligible"
                ],
                "provider_request_ids": provider_request_ids,
                "provider_request_ids_available": bool(provider_request_ids),
                "provider_request_id_coverage": {
                    "available": len(provider_request_ids),
                    "attempts": provider_attempt_count,
                    "complete": len(provider_request_ids) == provider_attempt_count,
                },
                "dataset_governance": governance_snapshot,
                "fingerprint": fingerprint,
                "n": agg["n"],
                "aggregate": agg,
                "by_category": cat_rows,
                "attribution_summary": attribution_summary,
                "cases": results,
            }
            run_json = (
                RUNS_DIR / f"run_{dataset_suite}_{ts}.json"
                if immutable
                else targeted_run_path(dataset_suite, RUNS_DIR)
            )
            run_json.write_text(
                json.dumps(run_payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            baseline_repo = EvaluationBaselineRepository(meta_session)
            if full_dataset and run_payload["baseline_eligible"]:
                await baseline_repo.register_candidate(run_payload, run_json.name)
            print(report)
            print(f"\n报告已写入：{out}")
            print(f"结构化结果已写入：{run_json}")
            if not full_dataset:
                print(
                    "本次为定向诊断切片：仅覆盖 "
                    f"targeted_{dataset_suite}_latest，"
                    "未注册正式基线候选。"
                )
            if experiment:
                approved_entry = await baseline_repo.approved(
                    selected_workspace_id, dataset_suite,
                )
                baseline_selection = resolve_baseline_artifact(
                    workspace_id=selected_workspace_id,
                    approved_entry=approved_entry,
                    expected_dataset_suite=dataset_suite,
                )
                group = experiment_report(
                    existing_experiment_runs(
                        str(experiment_id), RUNS_DIR, dataset_suite,
                    ),
                    baseline_run=baseline_selection.get("payload"),
                    experiment_id=str(experiment_id),
                )
                print(
                    f"实验组 {experiment_id}：{group.get('repeat_count', 0)}/"
                    f"{group.get('required_repeats', 3)} 次，状态 {group.get('status')}"
                )
                if group.get("failed_checks"):
                    print("尚未通过：" + "、".join(group["failed_checks"]))
            if not valid:
                print(
                    "评测运行无效：基础设施/证据合同或模型物理调用门禁未通过，"
                    "不得作为有效边界结论或正式基线。"
                )
            if experiment and not run_payload["baseline_eligible"]:
                print(
                    "实验 repeat 未通过正式准入证据，已阻止后续付费 repeat。"
                )
            return evaluation_continue_signal(
                experiment=experiment,
                valid=valid,
                baseline_eligible=bool(run_payload["baseline_eligible"]),
            )
    finally:
        try:
            await es_client_manager.close()
            await dw_mysql_client_manager.close()
            await meta_mysql_client_manager.close()
            await qdrant_client_manager.close()
        finally:
            if owns_embedding_session:
                await active_embedding_session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="只跑指定用例ID，逗号分隔")
    parser.add_argument("--tier", choices=("smoke", "regression"), default=None, help="按层级运行黄金集")
    parser.add_argument(
        "--suite", choices=tuple(SUITE_PATHS), default="easy_sql_v1",
        help=(
            "选择简单、中等或困难 SQL 基线集；具体题数由对应 manifest 决定；"
            "困难集保留逐题物理模型调用合同。"
        ),
    )
    parser.add_argument("--workspace-id", default=None, help="指定评测使用的工作空间 ID")
    parser.add_argument("--experiment-id", default=None, help="三次可比评测共用的 experiment ID")
    parser.add_argument("--candidate-label", default=None, help="候选版本名称，三次必须一致")
    parser.add_argument("--repeat-index", type=int, choices=(1, 2, 3), default=None, help="实验重复序号")
    parser.add_argument("--cache-state", choices=("cold", "warm"), default=None, help="repeat 1 用 cold，2/3 用 warm")
    parser.add_argument(
        "--evaluation-profile", choices=tuple(EVALUATION_PROFILES), default="core-default",
        help="core-default 与网页默认一致；enhanced-analysis 单独评估可选解读",
    )
    args = parser.parse_args()
    try:
        passed = asyncio.run(main(
            args.only, args.tier, args.workspace_id,
            experiment_id=args.experiment_id, candidate_label=args.candidate_label,
            repeat_index=args.repeat_index, cache_state=args.cache_state,
            evaluation_profile=args.evaluation_profile,
            dataset_suite=args.suite,
        ))
    except ValueError as exc:
        parser.error(str(exc))
    raise SystemExit(0 if passed else 2)
