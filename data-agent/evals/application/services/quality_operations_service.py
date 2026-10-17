"""Workspace-scoped, privacy-minimized quality operations aggregation (TODO-37)."""
from __future__ import annotations

import math
import re
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from app.domain.query.safety.sql_security import check_sql_read
from app.domain.privacy.masking import MaskingPolicy, mask_sql_literals
from app.foundation.audit_compaction import (
    compact_query_contract,
    compact_relation_closure,
)
from app.foundation.llm_evidence import attempt_accounting, normalize_llm_attempt
from evals.legacy.dataset_suite import SUITE_SPECS
from evals.application.dto.quality_operations import FeedbackRecord, TraceRecord
from evals.application.ports.quality_operations import (
    EvaluationCatalogPort,
    QualityOperationsStore,
)
from evals.legacy.candidate_confidence_eval import build_report as build_candidate_confidence_report


REVIEW_CATEGORIES = {
    "invalid_feedback", "data_issue", "retrieval_issue", "sql_issue",
    "metric_issue", "presentation_issue",
}
REVIEW_STATUSES = {"pending", "reviewed", "dismissed"}
CANDIDATE_TYPES = {"golden", "training"}
_MASKING_POLICY = MaskingPolicy()

_TRACE_OUTCOME_SUCCESS = "success"
_TRACE_OUTCOME_EXPECTED_TERMINAL = "expected_terminal"
_TRACE_OUTCOME_FAILURE = "failure"
_TRACE_OUTCOME_RUNNING = "running"


def _trace_outcome(status: object) -> str:
    """把持久化状态归一为互斥质量统计桶。"""

    normalized = str(status or "").strip().lower()
    if normalized == "success":
        return _TRACE_OUTCOME_SUCCESS
    if normalized == "handled":
        return _TRACE_OUTCOME_EXPECTED_TERMINAL
    if normalized == "running":
        return _TRACE_OUTCOME_RUNNING
    # 显式 failed 与未知历史状态都属于失败，避免未知状态被静默忽略。
    return _TRACE_OUTCOME_FAILURE


def _mask_sql(sql: str) -> str:
    return mask_sql_literals(sql, policy=_MASKING_POLICY)


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _redact_text(value: str | None, limit: int = 180) -> str:
    text = str(value or "").replace("\x00", " ").strip()
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "<email>", text)
    text = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "<phone>", text)
    text = re.sub(r"(?<!\d)\d{6,}(?!\d)", "<number>", text)
    text = re.sub(r"(['\"]).*?\1", "?", text)
    return text[:limit]


def _percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return float(ordered[max(0, math.ceil(len(ordered) * ratio) - 1)])


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


class QualityOperationsService:
    def __init__(
        self,
        store: QualityOperationsStore | None,
        workspace_id: str,
        evaluation_catalog: EvaluationCatalogPort | None = None,
    ):
        self.store = store
        self.workspace_id = workspace_id
        self.evaluation_catalog = evaluation_catalog

    async def overview(self, *, days: int = 7, version: str | None = None,
                       category: str | None = None, status: str | None = None) -> dict:
        if self.store is None or self.evaluation_catalog is None:
            raise RuntimeError("质量运营查询需要存储与评测目录端口")
        days = max(1, min(int(days), 90))
        cutoff = _now() - timedelta(days=days)
        snapshot = await self.store.load_overview(
            cutoff=cutoff,
            version=version,
            category=category,
            status=status,
        )
        coverage = {
            "exact": (
                snapshot.trace_count <= snapshot.trace_limit
                and snapshot.feedback_count <= snapshot.feedback_limit
            ),
            "truncated": (
                snapshot.trace_count > snapshot.trace_limit
                or snapshot.feedback_count > snapshot.feedback_limit
            ),
            "trace": {
                "matched": snapshot.trace_count,
                "sampled": len(snapshot.traces),
                "limit": snapshot.trace_limit,
            },
            "feedback": {
                "matched": snapshot.feedback_count,
                "sampled": len(snapshot.feedback),
                "limit": snapshot.feedback_limit,
            },
            "strategy": "newest_bounded_window",
            "coverage_from": (
                str(snapshot.traces[0].started_at) if snapshot.traces else None
            ),
            "coverage_to": (
                str(snapshot.traces[-1].started_at) if snapshot.traces else None
            ),
        }
        approved = await self.evaluation_catalog.approved_entries(self.workspace_id)
        evaluation_runs = await self.evaluation_catalog.list_runs(
            self.workspace_id, 20
        )
        return self.aggregate(
            list(snapshot.traces),
            list(snapshot.steps),
            list(snapshot.usages),
            list(snapshot.feedback),
            days=days, filters={"version": version, "category": category, "status": status},
            # Feedback is already workspace/time scoped and, when filters are
            # active, SQL-scoped to the complete matching trace set.
            metric_chat_ids=None,
            approved=approved, evaluation_runs=evaluation_runs,
            coverage=coverage,
        )

    def aggregate(self, traces: list, steps: list, usages: list, feedback: list,
                  *, days: int, filters: dict, metric_chat_ids: set[str] | None = None,
                  approved: list[dict] | None = None,
                  evaluation_runs: list[dict] | None = None,
                  coverage: dict | None = None) -> dict:
        total = len(traces)
        trace_outcomes = [
            (trace, _trace_outcome(getattr(trace, "status", None)))
            for trace in traces
        ]
        successful = sum(
            outcome == _TRACE_OUTCOME_SUCCESS for _, outcome in trace_outcomes
        )
        expected_terminal = sum(
            outcome == _TRACE_OUTCOME_EXPECTED_TERMINAL
            for _, outcome in trace_outcomes
        )
        failed = sum(
            outcome == _TRACE_OUTCOME_FAILURE for _, outcome in trace_outcomes
        )
        running = sum(
            outcome == _TRACE_OUTCOME_RUNNING for _, outcome in trace_outcomes
        )
        terminal_total = successful + expected_terminal + failed
        sql_terminal_total = successful + failed
        sql_terminal_traces = [
            trace
            for trace, outcome in trace_outcomes
            if outcome in {_TRACE_OUTCOME_SUCCESS, _TRACE_OUTCOME_FAILURE}
        ]
        sql_terminal_trace_ids = {
            str(getattr(trace, "id", "")) for trace in sql_terminal_traces
        }
        metric_feedback = feedback if metric_chat_ids is None else [
            item for item in feedback if item.chat_id in metric_chat_ids
        ]
        negative_chat_ids = {item.chat_id for item in metric_feedback if item.score == -1 and item.chat_id}
        positive = sum(item.score == 1 for item in metric_feedback)
        negative = sum(item.score == -1 for item in metric_feedback)
        feedback_total = positive + negative
        durations = [float(item.tt_complete_ms or item.duration_ms or 0) for item in traces]
        corrected_trace_ids = {
            item.trace_id for item in steps
            if str(item.step_name).lower() in {"correct_sql", "纠错sql", "纠错 sql"}
            and str(item.trace_id) in sql_terminal_trace_ids
        }
        failed_by_code: dict[str, int] = defaultdict(int)
        categories: dict[str, int] = defaultdict(int)
        versions: dict[str, dict] = {}
        daily: dict[str, dict] = {}
        trace_by_chat: dict[str, TraceRecord] = {}
        semantic_profile_counts: dict[str, int] = defaultdict(int)
        semantic_rule_counts: dict[str, int] = defaultdict(int)
        semantic_violation_counts: dict[str, int] = defaultdict(int)
        semantic_repair_latencies: list[float] = []
        semantic_changed = 0
        semantic_fallbacks = 0
        semantic_coverage_scores: list[float] = []
        semantic_unresolved = 0
        semantic_unsafe = 0
        deadline_observed = 0
        deadline_exhausted = 0
        deadline_contract_complete = True
        for trace, outcome in trace_outcomes:
            if trace.chat_id:
                trace_by_chat[trace.chat_id] = trace
            if outcome == _TRACE_OUTCOME_FAILURE:
                failure_code = str(trace.failure_code or "pipeline_unknown")
                if failure_code == "none":
                    failure_code = "pipeline_unknown"
                failed_by_code[failure_code] += 1
            categories[str(trace.query_category or "general")] += 1
            version_key = str(trace.run_fingerprint or "unversioned")
            version_row = versions.setdefault(version_key, {
                "run_fingerprint": version_key, "version_run_id": trace.version_run_id,
                "version_complete": bool(trace.version_complete), "traces": 0,
                "terminal": 0, "sql_terminal": 0, "success": 0,
                "expected_terminal": 0, "failed": 0, "running": 0,
                "tokens": 0, "durations": [],
            })
            version_row["traces"] += 1
            version_row["terminal"] += outcome != _TRACE_OUTCOME_RUNNING
            version_row["sql_terminal"] += outcome in {
                _TRACE_OUTCOME_SUCCESS,
                _TRACE_OUTCOME_FAILURE,
            }
            version_row["success"] += outcome == _TRACE_OUTCOME_SUCCESS
            version_row["expected_terminal"] += (
                outcome == _TRACE_OUTCOME_EXPECTED_TERMINAL
            )
            version_row["failed"] += outcome == _TRACE_OUTCOME_FAILURE
            version_row["running"] += outcome == _TRACE_OUTCOME_RUNNING
            version_row["tokens"] += int(trace.total_tokens or 0)
            version_row["durations"].append(float(trace.tt_complete_ms or trace.duration_ms or 0))
            date = trace.started_at.date().isoformat() if trace.started_at else "unknown"
            day = daily.setdefault(date, {
                "date": date, "traces": 0, "terminal": 0,
                "sql_terminal": 0, "success": 0, "expected_terminal": 0,
                "failed": 0, "running": 0, "tokens": 0,
                "cost": 0.0, "durations": [],
            })
            day["traces"] += 1
            day["terminal"] += outcome != _TRACE_OUTCOME_RUNNING
            day["sql_terminal"] += outcome in {
                _TRACE_OUTCOME_SUCCESS,
                _TRACE_OUTCOME_FAILURE,
            }
            day["success"] += outcome == _TRACE_OUTCOME_SUCCESS
            day["expected_terminal"] += (
                outcome == _TRACE_OUTCOME_EXPECTED_TERMINAL
            )
            day["failed"] += outcome == _TRACE_OUTCOME_FAILURE
            day["running"] += outcome == _TRACE_OUTCOME_RUNNING
            day["tokens"] += int(trace.total_tokens or 0)
            day["cost"] += float(trace.estimated_cost or 0)
            day["durations"].append(float(trace.tt_complete_ms or trace.duration_ms or 0))
            sql_terminal = outcome in {
                _TRACE_OUTCOME_SUCCESS,
                _TRACE_OUTCOME_FAILURE,
            }
            repair = (
                self._semantic_repair_evidence(
                    getattr(trace, "semantic_repair_summary", None)
                )
                if sql_terminal
                else None
            )
            if repair:
                semantic_profile_counts[str(repair.get("profile") or "unknown")] += 1
                semantic_changed += bool(repair.get("changed"))
                semantic_fallbacks += bool(repair.get("fallback_used"))
                semantic_repair_latencies.append(float(repair.get("repair_ms") or 0))
                violation = repair.get("violation") or {}
                if violation.get("violation_code"):
                    semantic_violation_counts[str(violation["violation_code"])] += 1
                for row in repair.get("applied_rules") or []:
                    if isinstance(row, dict) and row.get("rule_code"):
                        semantic_rule_counts[str(row["rule_code"])] += 1
            semantic_coverage = (
                self._semantic_coverage_evidence(
                    getattr(trace, "semantic_coverage_summary", None)
                )
                if sql_terminal
                else None
            )
            if semantic_coverage:
                semantic_coverage_scores.append(
                    float(semantic_coverage.get("coverage_score") or 0)
                )
                semantic_unresolved += bool(
                    semantic_coverage.get("unresolved_slots")
                )
                semantic_unsafe += bool(
                    semantic_coverage.get("compiler_supported")
                    and semantic_coverage.get("unresolved_slots")
                )
            deadline = self._request_deadline_evidence(
                getattr(trace, "request_deadline_summary", None)
            )
            if deadline:
                deadline_observed += 1
                deadline_exhausted += int(deadline.get("remaining_ms") or 0) <= 0
                deadline_contract_complete = (
                    deadline_contract_complete
                    and deadline.get("contract_version") == "request-deadline-v1"
                    and deadline.get("absolute_timestamps_persisted") is False
                )

        negative_trace_count = sum(
            1
            for trace, outcome in trace_outcomes
            if outcome == _TRACE_OUTCOME_SUCCESS
            and trace.chat_id
            and trace.chat_id in negative_chat_ids
        )
        adaptive_count = sum(
            bool(getattr(item, "adaptive_candidate_triggered", False))
            for item in sql_terminal_traces
        )
        low_confidence_count = sum(
            str(getattr(item, "candidate_band", "") or "") == "low"
            for item in sql_terminal_traces
        )
        adaptive_success = sum(
            bool(getattr(item, "adaptive_candidate_triggered", False))
            and str(getattr(item, "candidate_selected", "A") or "A") == "B"
            and _trace_outcome(item.status) == _TRACE_OUTCOME_SUCCESS
            for item in sql_terminal_traces
        )
        node_rows: dict[str, dict] = {}
        token_by_node: dict[str, int] = defaultdict(int)
        normalized_usages: list[dict] = []
        for usage in usages:
            token_by_node[str(usage.step_name or "llm")] += int(usage.total_tokens or 0)
            normalized_usages.append(normalize_llm_attempt({
                "attempt_id": getattr(usage, "attempt_id", None),
                "node_id": getattr(usage, "step_name", None),
                "route": getattr(usage, "route", None),
                "model": getattr(usage, "model", None),
                "status": getattr(usage, "status", None),
                "token_known": bool(getattr(usage, "token_known", False)),
                "timeout_budget_ms": getattr(usage, "timeout_budget_ms", None),
                "affects_core_path": bool(
                    getattr(usage, "affects_core_path", True)
                ),
                "error_type": getattr(usage, "error_type", None),
                "reason_code": getattr(usage, "reason_code", None),
                "retry_index": getattr(usage, "retry_index", 0),
                "callback_observed": bool(
                    getattr(usage, "callback_observed", False)
                ),
                "pre_invoke_failure": bool(
                    getattr(usage, "pre_invoke_failure", False)
                ),
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
                "latency_ms": getattr(usage, "latency_ms", 0),
                "estimated_cost": getattr(usage, "estimated_cost", 0),
                "request_deadline_version": getattr(
                    usage, "request_deadline_version", None
                ),
                "deadline_before_ms": getattr(usage, "deadline_before_ms", None),
                "deadline_after_ms": getattr(usage, "deadline_after_ms", None),
                "queue_wait_ms": getattr(usage, "queue_wait_ms", 0),
                "provider_ms": getattr(usage, "provider_ms", 0),
                "time_to_first_token_ms": getattr(
                    usage, "time_to_first_token_ms", None
                ),
                "time_to_last_token_ms": getattr(
                    usage, "time_to_last_token_ms", None
                ),
                "token_timing_available": bool(
                    getattr(usage, "token_timing_available", False)
                ),
                "completion_stage": getattr(usage, "completion_stage", None),
                "cancel_stage": getattr(usage, "cancel_stage", None),
                "provider_request_id": getattr(
                    usage, "provider_request_id", None
                ),
                "provider_request_id_available": bool(
                    getattr(usage, "provider_request_id_available", False)
                ),
                "provider_request_id_unavailable_reason": getattr(
                    usage,
                    "provider_request_id_unavailable_reason",
                    None,
                ),
                "token_timing_unavailable_reason": getattr(
                    usage,
                    "token_timing_unavailable_reason",
                    None,
                ),
                "outcome_class": getattr(usage, "outcome_class", None),
                "recovery": bool(getattr(usage, "recovery", False)),
                "retryable": bool(getattr(usage, "retryable", False)),
                "recovery_reserve_ms": getattr(
                    usage, "recovery_reserve_ms", 0
                ),
                "physical_attempt_budget_ms": getattr(
                    usage, "physical_attempt_budget_ms", None
                ),
                "source": "persisted",
            }))
        for step in steps:
            name = str(step.step_name or "unknown")
            row = node_rows.setdefault(name, {"node_id": name, "calls": 0, "failed": 0, "durations": []})
            row["calls"] += 1
            row["failed"] += step.status == "failed"
            row["durations"].append(float(step.duration_ms or 0))

        approved_by_run: dict[str, set[str]] = {}
        for item in approved or []:
            dataset_suite = str(item.get("dataset_suite") or "")
            run_id = str(item.get("run_id") or "")
            if not run_id or dataset_suite not in SUITE_SPECS:
                continue
            approved_by_run.setdefault(run_id, set()).add(dataset_suite)
        difficulty_by_suite = {
            name: str(spec.get("difficulty") or "unknown")
            for name, spec in SUITE_SPECS.items()
        }
        version_output = []
        for row in versions.values():
            baseline_suites = approved_by_run.get(
                str(row["version_run_id"] or ""), set(),
            )
            version_output.append({
                "run_fingerprint": row["run_fingerprint"],
                "version_run_id": row["version_run_id"],
                "version_complete": row["version_complete"],
                "baseline_bound": bool(baseline_suites),
                "baseline_difficulties": sorted({
                    difficulty_by_suite[suite]
                    for suite in baseline_suites
                }),
                "traces": row["traces"],
                "terminal_traces": row["terminal"],
                "sql_terminal_traces": row["sql_terminal"],
                "expected_terminal_traces": row["expected_terminal"],
                "failed_traces": row["failed"],
                "running_traces": row["running"],
                "success_rate": _rate(row["success"], row["sql_terminal"]),
                "expected_terminal_rate": _rate(
                    row["expected_terminal"], row["terminal"]
                ),
                "p95_ms": _percentile(row["durations"], .95),
                "tokens_mean": round(row["tokens"] / row["traces"], 1) if row["traces"] else 0,
            })
        trend_output = [{
            "date": row["date"], "traces": row["traces"],
            "terminal_traces": row["terminal"],
            "sql_terminal_traces": row["sql_terminal"],
            "expected_terminal_traces": row["expected_terminal"],
            "failed_traces": row["failed"],
            "running_traces": row["running"],
            "success_rate": _rate(row["success"], row["sql_terminal"]),
            "expected_terminal_rate": _rate(
                row["expected_terminal"], row["terminal"]
            ),
            "p50_ms": _percentile(row["durations"], .5),
            "p95_ms": _percentile(row["durations"], .95),
            "tokens_mean": round(row["tokens"] / row["traces"], 1) if row["traces"] else 0,
            "estimated_cost": round(row["cost"], 8),
        } for row in sorted(daily.values(), key=lambda item: item["date"])]
        slow_nodes = [{
            "node_id": row["node_id"], "calls": row["calls"], "failed": row["failed"],
            "mean_ms": round(sum(row["durations"]) / row["calls"], 1) if row["calls"] else 0,
            "p95_ms": _percentile(row["durations"], .95),
            "total_ms": round(sum(row["durations"]), 1),
            "tokens": token_by_node.get(row["node_id"], 0),
        } for row in node_rows.values()]
        slow_nodes.sort(key=lambda item: item["total_ms"], reverse=True)

        feedback_queue = [self._feedback_view(item, trace_by_chat.get(item.chat_id)) for item in feedback if item.score == -1]
        recent_traces = [{
            "trace_id": trace.id, "chat_id": trace.chat_id,
            "query_summary": _redact_text(trace.query), "status": trace.status,
            "failure_code": trace.failure_code, "query_category": trace.query_category,
            "tt_complete_ms": trace.tt_complete_ms or trace.duration_ms,
            "total_tokens": trace.total_tokens, "estimated_cost": trace.estimated_cost,
            "cache_hit": bool(trace.cache_hit), "version_run_id": trace.version_run_id,
            "candidate_confidence": getattr(trace, "candidate_confidence", None),
            "candidate_band": getattr(trace, "candidate_band", None),
            "adaptive_candidate_triggered": bool(getattr(trace, "adaptive_candidate_triggered", False)),
            "semantic_repair": self._semantic_repair_evidence(
                getattr(trace, "semantic_repair_summary", None)
            ),
            "semantic_coverage": self._semantic_coverage_evidence(
                getattr(trace, "semantic_coverage_summary", None)
            ),
            "request_deadline": self._request_deadline_evidence(
                getattr(trace, "request_deadline_summary", None)
            ),
            "version_complete": bool(trace.version_complete),
            "started_at": str(trace.started_at) if trace.started_at else None,
        } for trace in reversed(traces[-50:])]
        return {
            "available": bool(traces or feedback), "days": days, "filters": filters,
            "coverage": coverage or {
                "exact": True, "truncated": False,
                "trace": {"matched": len(traces), "sampled": len(traces), "limit": None},
                "feedback": {"matched": len(feedback), "sampled": len(feedback), "limit": None},
                "strategy": "complete_input", "coverage_from": None, "coverage_to": None,
            },
            "summary": {
                "traces": total,
                "terminal_traces": terminal_total,
                "sql_terminal_traces": sql_terminal_total,
                "successful_traces": successful,
                "expected_terminal_traces": expected_terminal,
                "failed_traces": failed,
                "running_traces": running,
                "success_rate": _rate(successful, sql_terminal_total),
                "expected_terminal_rate": _rate(
                    expected_terminal, terminal_total
                ),
                "negative_rate": _rate(negative, feedback_total),
                "result_proxy_rate": _rate(
                    max(0, successful - negative_trace_count),
                    sql_terminal_total,
                ),
                "p50_ms": _percentile(durations, .5), "p95_ms": _percentile(durations, .95),
                "tokens_mean": round(sum(int(item.total_tokens or 0) for item in traces) / total, 1) if total else 0,
                "estimated_cost": round(sum(float(item.estimated_cost or 0) for item in traces), 8),
                "correction_rate": _rate(
                    len(corrected_trace_ids), sql_terminal_total
                ),
                "cache_hit_rate": _rate(
                    sum(bool(item.cache_hit) for item in sql_terminal_traces),
                    sql_terminal_total,
                ),
                "low_confidence_rate": _rate(
                    low_confidence_count, sql_terminal_total
                ),
                "second_candidate_rate": _rate(
                    adaptive_count, sql_terminal_total
                ),
                "adaptive_repair_rate": _rate(adaptive_success, adaptive_count),
                "feedback_total": feedback_total, "negative_feedback": negative,
                # 显式返回每类比率分母，避免 expected terminal 被当成 SQL 失败。
                "rate_denominators": {
                    "success_rate": sql_terminal_total,
                    "expected_terminal_rate": terminal_total,
                    "result_proxy_rate": sql_terminal_total,
                    "negative_rate": feedback_total,
                    "correction_rate": sql_terminal_total,
                    "cache_hit_rate": sql_terminal_total,
                    "low_confidence_rate": sql_terminal_total,
                    "second_candidate_rate": sql_terminal_total,
                    "adaptive_repair_rate": adaptive_count,
                },
            },
            "semantic_repair": {
                "observed_traces": sum(semantic_profile_counts.values()),
                "changed_traces": semantic_changed,
                "change_rate": _rate(semantic_changed, sum(semantic_profile_counts.values())),
                "fallback_count": semantic_fallbacks,
                "fallback_rate": _rate(semantic_fallbacks, sum(semantic_profile_counts.values())),
                "repair_p95_ms": _percentile(semantic_repair_latencies, .95),
                "profiles": [
                    {"profile": key, "count": value}
                    for key, value in sorted(
                        semantic_profile_counts.items(), key=lambda item: item[1], reverse=True
                    )
                ],
                "rules": [
                    {"rule_code": key, "count": value}
                    for key, value in sorted(
                        semantic_rule_counts.items(), key=lambda item: item[1], reverse=True
                    )
                ],
                "violations": [
                    {"violation_code": key, "count": value}
                    for key, value in sorted(
                        semantic_violation_counts.items(), key=lambda item: item[1], reverse=True
                    )
                ],
            },
            "runtime_governance": {
                "semantic_coverage_observed": len(semantic_coverage_scores),
                "semantic_coverage_rate": _rate(
                    len(semantic_coverage_scores), sql_terminal_total
                ),
                "semantic_coverage_denominator": sql_terminal_total,
                "semantic_coverage_mean": round(
                    sum(semantic_coverage_scores) / len(semantic_coverage_scores),
                    4,
                ) if semantic_coverage_scores else 0.0,
                "semantic_unresolved_traces": semantic_unresolved,
                "semantic_unsafe_compile_traces": semantic_unsafe,
                "deadline_observed": deadline_observed,
                "deadline_coverage_rate": _rate(deadline_observed, total),
                "deadline_exhausted_traces": deadline_exhausted,
                "deadline_contract_complete": (
                    bool(deadline_observed) and deadline_contract_complete
                ),
                "llm_attempt_accounting": attempt_accounting(normalized_usages),
                "llm_queue_wait_p95_ms": _percentile(
                    [float(item.get("queue_wait_ms") or 0)
                     for item in normalized_usages],
                    .95,
                ),
                "llm_provider_p95_ms": _percentile(
                    [float(item.get("provider_ms") or 0)
                     for item in normalized_usages],
                    .95,
                ),
                "llm_outcome_classes": {
                    key: sum(
                        str(item.get("outcome_class") or "unknown") == key
                        for item in normalized_usages
                    )
                    for key in sorted({
                        str(item.get("outcome_class") or "unknown")
                        for item in normalized_usages
                    })
                },
            },
            "trends": trend_output,
            "versions": sorted(version_output, key=lambda item: str(item.get("version_run_id") or ""), reverse=True),
            "categories": [{"category": key, "count": value} for key, value in sorted(categories.items())],
            "failure_funnel": [
                {"failure_code": key, "count": value}
                for key, value in sorted(failed_by_code.items(), key=lambda item: item[1], reverse=True)
                if value
            ],
            "slow_nodes": slow_nodes[:12],
            "feedback_queue": feedback_queue[:100],
            "recent_traces": recent_traces,
            "evaluation_runs": evaluation_runs or [],
            "candidate_policy": build_candidate_confidence_report(write=False),
        }

    async def trace_detail(self, trace_id: str) -> dict:
        if self.store is None:
            raise RuntimeError("质量运营查询需要存储端口")
        snapshot = await self.store.load_trace(trace_id)
        if snapshot is None:
            raise ValueError("当前工作空间不存在该 trace")
        trace = snapshot.trace
        steps = snapshot.steps
        usages = snapshot.usages
        llm_attempts = [
            normalize_llm_attempt({
                "attempt_id": item.attempt_id,
                "node_id": item.step_name,
                "route": item.route,
                "model": item.model,
                "status": item.status,
                "token_known": bool(item.token_known),
                "timeout_budget_ms": item.timeout_budget_ms,
                "affects_core_path": bool(item.affects_core_path),
                "error_type": item.error_type,
                "reason_code": item.reason_code,
                "retry_index": item.retry_index,
                "callback_observed": bool(item.callback_observed),
                "pre_invoke_failure": bool(item.pre_invoke_failure),
                "request_deadline_version": item.request_deadline_version,
                "deadline_before_ms": item.deadline_before_ms,
                "deadline_after_ms": item.deadline_after_ms,
                "queue_wait_ms": item.queue_wait_ms,
                "provider_ms": item.provider_ms,
                "time_to_first_token_ms": item.time_to_first_token_ms,
                "time_to_last_token_ms": item.time_to_last_token_ms,
                "token_timing_available": bool(item.token_timing_available),
                "completion_stage": item.completion_stage,
                "cancel_stage": item.cancel_stage,
                "provider_request_id": item.provider_request_id,
                "provider_request_id_available": bool(
                    item.provider_request_id_available
                ),
                "provider_request_id_unavailable_reason": (
                    item.provider_request_id_unavailable_reason
                ),
                "token_timing_unavailable_reason": (
                    item.token_timing_unavailable_reason
                ),
                "outcome_class": item.outcome_class,
                "recovery": bool(item.recovery),
                "retryable": bool(item.retryable),
                "recovery_reserve_ms": item.recovery_reserve_ms,
                "physical_attempt_budget_ms": (
                    item.physical_attempt_budget_ms
                ),
                "prompt_tokens": item.prompt_tokens,
                "completion_tokens": item.completion_tokens,
                "total_tokens": item.total_tokens,
                "latency_ms": item.latency_ms,
                "estimated_cost": item.estimated_cost,
                "source": "persisted",
            })
            for item in usages
        ]
        llm_accounting = attempt_accounting(llm_attempts)
        return {
            "trace_id": trace.id, "query_summary": _redact_text(trace.query),
            "sql_summary": _mask_sql(trace.sql) if trace.sql else None,
            "status": trace.status, "failure_code": trace.failure_code,
            "query_category": trace.query_category, "version_run_id": trace.version_run_id,
            "run_fingerprint": trace.run_fingerprint, "version_complete": bool(trace.version_complete),
            "cache_hit": bool(trace.cache_hit), "row_count": trace.row_count,
            "follow_up_action": trace.follow_up_action,
            "follow_up_summary": _redact_text(trace.follow_up_summary, 500),
            "candidate_decision": self._candidate_decision(getattr(trace, "candidate_summary", None)),
            "planning_route": self._candidate_decision(
                getattr(trace, "planning_summary", None)
            ),
            "retrieval_evidence": self._retrieval_evidence(
                getattr(trace, "retrieval_summary", None)
            ),
            "semantic_repair": self._semantic_repair_evidence(
                getattr(trace, "semantic_repair_summary", None)
            ),
            "semantic_coverage": self._semantic_coverage_evidence(
                getattr(trace, "semantic_coverage_summary", None)
            ),
            "request_deadline": self._request_deadline_evidence(
                getattr(trace, "request_deadline_summary", None)
            ),
            "tt_complete_ms": trace.tt_complete_ms or trace.duration_ms,
            "total_tokens": trace.total_tokens, "estimated_cost": trace.estimated_cost,
            "llm_attempts": llm_attempts,
            "llm_attempt_accounting": llm_accounting,
            "steps": [{
                "step_name": item.step_name, "status": item.status,
                "duration_ms": item.duration_ms, "attempt": item.attempt,
                "error_summary": _redact_text(item.error),
            } for item in steps],
        }

    @staticmethod
    def _candidate_decision(value: str | None) -> dict:
        try:
            item = json.loads(value or "{}")
        except (TypeError, ValueError):
            return {}
        return item if isinstance(item, dict) else {}

    @staticmethod
    def _retrieval_evidence(value: str | None) -> list[dict]:
        """Return the already-whitelisted, privacy-safe retrieval trace rows."""
        try:
            item = json.loads(value or "{}")
        except (TypeError, ValueError):
            return []
        rows = item.get("routes") if isinstance(item, dict) else []
        return [dict(row) for row in (rows or []) if isinstance(row, dict)][:24]

    @staticmethod
    def _semantic_repair_evidence(value: str | None) -> dict:
        """Return the versioned, privacy-safe TODO-64 repair evidence only."""
        try:
            item = json.loads(value or "{}")
        except (TypeError, ValueError):
            return {}
        if not isinstance(item, dict):
            return {}
        allowed = {
            "contract_version", "registry_version", "rule_set_version", "profile",
            "changed", "original_sql_hash", "repaired_sql_hash", "applied_rules",
            "skipped_rules", "shadow_rules", "attempted_rule_codes", "repair_ms",
            "fallback_used", "fallback_reason", "invariants", "raw_sql_logged",
            "violation",
        }
        return {key: item.get(key) for key in allowed if key in item}

    @staticmethod
    def _semantic_coverage_evidence(value: str | None) -> dict:
        """Return structural coverage only; question and filter literals are forbidden."""
        try:
            item = json.loads(value or "{}")
        except (TypeError, ValueError):
            return {}
        if not isinstance(item, dict):
            return {}
        allowed = {
            "schema_version", "registry_version", "question_fingerprint",
            "coverage_score", "compiler_mode", "compiler_supported",
            "raw_question_persisted", "filter_values_persisted",
            "detected_slots", "represented_slots", "unresolved_slots",
            "abstain_reasons", "query_contract", "relation_closure",
        }
        output = {key: item.get(key) for key in allowed if key in item}
        if isinstance(output.get("query_contract"), dict):
            output["query_contract"] = compact_query_contract(
                output["query_contract"]
            )
        if isinstance(output.get("relation_closure"), dict):
            output["relation_closure"] = compact_relation_closure(
                output["relation_closure"]
            )
        return output

    @staticmethod
    def _request_deadline_evidence(value: str | None) -> dict:
        """Return relative budgets only; absolute timestamps never leave the collector."""
        try:
            item = json.loads(value or "{}")
        except (TypeError, ValueError):
            return {}
        if not isinstance(item, dict):
            return {}
        allowed = {
            "contract_version", "budget_ms", "remaining_ms",
            "execution_reserve_ms", "phase_budgets_ms",
            "absolute_timestamps_persisted",
        }
        return {key: item.get(key) for key in allowed if key in item}

    async def review_feedback(self, feedback_id: int, *, status: str, category: str | None,
                              note: str, reviewer_id: str) -> dict:
        if self.store is None:
            raise RuntimeError("质量运营写入需要存储端口")
        if status not in REVIEW_STATUSES:
            raise ValueError("无效审核状态")
        if status == "reviewed" and category not in REVIEW_CATEGORIES:
            raise ValueError("已审核差评必须选择有效分类")
        item = await self.store.review_feedback(
            feedback_id,
            status=status,
            category=category,
            note=note,
            reviewer_id=reviewer_id,
            reviewed_at=_now(),
        )
        if item is None:
            raise ValueError("当前工作空间不存在该反馈")
        return self._feedback_view(item, None)

    async def draft_candidate(self, feedback_id: int, *, candidate_type: str,
                              question: str, sql: str) -> dict:
        if self.store is None:
            raise RuntimeError("质量运营写入需要存储端口")
        if candidate_type not in CANDIDATE_TYPES:
            raise ValueError("candidate_type 只允许 golden 或 training")
        item = await self.store.get_feedback(feedback_id)
        if item is None:
            raise ValueError("当前工作空间不存在该反馈")
        if item.score != -1 or item.review_status != "reviewed":
            raise ValueError("只有已审核差评才能生成候选")
        check_sql_read(sql)
        item = await self.store.draft_candidate(
            feedback_id,
            candidate_type=candidate_type,
            question=question,
            sql=sql,
        )
        if item is None:
            raise ValueError("当前工作空间不存在该反馈")
        return self._feedback_view(item, None)

    async def confirm_candidate(self, feedback_id: int, *, confirmer_id: str) -> dict:
        if self.store is None:
            raise RuntimeError("质量运营写入需要存储端口")
        item = await self.store.get_feedback(feedback_id, for_update=True)
        if item is None:
            raise ValueError("当前工作空间不存在该反馈")
        # HTTP timeout/retry after a successful commit is idempotent.
        if item.candidate_status == "confirmed":
            return self._feedback_view(item, None)
        if item.candidate_status != "draft" or item.candidate_type not in CANDIDATE_TYPES:
            raise ValueError("当前反馈没有待确认候选")
        if not item.candidate_question or not item.candidate_sql:
            raise ValueError("候选问题和 SQL 不完整")
        check_sql_read(item.candidate_sql)
        item = await self.store.confirm_candidate(
            feedback_id,
            confirmer_id=confirmer_id,
            confirmed_at=_now(),
        )
        if item is None:
            raise ValueError("当前工作空间不存在该反馈")
        return self._feedback_view(item, None)

    @staticmethod
    def _feedback_view(
        item: FeedbackRecord,
        trace: TraceRecord | None,
    ) -> dict:
        return {
            "id": item.id, "chat_id": item.chat_id,
            "query_summary": _redact_text(item.query),
            "sql_summary": _mask_sql(item.sql) if item.sql else None,
            "score": item.score, "comment_summary": _redact_text(item.comment),
            "created_at": str(item.created_at) if item.created_at else None,
            "review_status": item.review_status or "pending",
            "review_category": item.review_category, "review_note": _redact_text(item.review_note, 300),
            "source_trace_id": item.source_trace_id or (trace.id if trace else None),
            "candidate_type": item.candidate_type, "candidate_status": item.candidate_status or "none",
            "candidate_question": _redact_text(item.candidate_question, 300),
            "candidate_sql_summary": _mask_sql(item.candidate_sql) if item.candidate_sql else None,
            "candidate_training_id": item.candidate_training_id,
            "trace": ({
                "trace_id": trace.id, "status": trace.status, "failure_code": trace.failure_code,
                "tt_complete_ms": trace.tt_complete_ms or trace.duration_ms,
                "total_tokens": trace.total_tokens, "version_run_id": trace.version_run_id,
            } if trace else None),
        }
