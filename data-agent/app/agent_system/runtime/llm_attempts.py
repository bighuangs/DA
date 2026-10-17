"""Canonical, privacy-safe LLM attempt and degradation evidence.

The callback path is the preferred source for provider usage.  Nodes that own an
external timeout additionally emit an explicit attempt so cancellation/timeout
cannot disappear when a provider callback is skipped.  ``reconcile_attempts``
merges both views without double counting the same node invocation.
"""
from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from typing import Any

from app.agent_system.agents.nl2sql.llm import route_for_node
from app.foundation.llm_evidence import (
    ATTEMPT_STATUSES,
    LLM_ATTEMPT_SCHEMA,
    LLM_INVOCATION_CONTRACT,
    make_llm_attempt as _foundation_make_llm_attempt,
)

OPTIONAL_POSTPROCESS_NODES = {"generate_chart", "generate_analysis", "generate_recommend"}
PRODUCTION_LLM_NODES = {
    "recall_column",
    "recall_metric",
    "recall_value",
    "filter_table",
    "filter_metric",
    "plan_sql",
    "generate_sql",
    "correct_sql",
    "judge_candidate",
    *OPTIONAL_POSTPROCESS_NODES,
}


def _route_for_node(node_id: str | None) -> str:
    return route_for_node(node_id)


def affects_core_path(node_id: str | None) -> bool:
    return str(node_id or "unknown") not in OPTIONAL_POSTPROCESS_NODES


def classify_attempt_status(error: BaseException) -> str:
    if isinstance(error, asyncio.CancelledError):
        return "cancelled"
    if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        return "timeout"
    return "error"


def reason_code_for_status(status: str) -> str:
    return {
        "success": "llm.success",
        "timeout": "llm.timeout",
        "cancelled": "llm.cancelled",
        "empty_output": "llm.empty_output",
        "error": "llm.provider_error",
    }.get(str(status), "llm.unknown_failure")


# 公共 API 直接复用 Foundation 的唯一证据构造器，避免证据 schema 出现双源。
make_llm_attempt = _foundation_make_llm_attempt


def normalize_llm_attempt(value: Mapping[str, Any]) -> dict[str, Any]:
    return make_llm_attempt(
        node_id=str(value.get("node_id") or value.get("step_name") or "unknown"),
        route=value.get("route"),
        model=value.get("model"),
        status=str(value.get("status") or "error"),
        latency_ms=value.get("latency_ms") or 0,
        timeout_budget_ms=value.get("timeout_budget_ms"),
        prompt_tokens=value.get("prompt_tokens"),
        completion_tokens=value.get("completion_tokens"),
        total_tokens=value.get("total_tokens"),
        estimated_cost=value.get("estimated_cost"),
        token_known=value.get("token_known"),
        error_type=value.get("error_type"),
        reason_code=value.get("reason_code"),
        retry_index=value.get("retry_index") or 0,
        callback_observed=value.get("callback_observed"),
        pre_invoke_failure=bool(value.get("pre_invoke_failure")),
        core_path=value.get("affects_core_path"),
        attempt_id=value.get("attempt_id"),
        source=str(value.get("source") or "normalized"),
        deadline_before_ms=value.get("deadline_before_ms"),
        deadline_after_ms=value.get("deadline_after_ms"),
        queue_wait_ms=value.get("queue_wait_ms"),
        provider_ms=value.get("provider_ms"),
        time_to_first_token_ms=value.get("time_to_first_token_ms"),
        time_to_last_token_ms=value.get("time_to_last_token_ms"),
        token_timing_available=bool(value.get("token_timing_available")),
        completion_stage=value.get("completion_stage"),
        cancel_stage=value.get("cancel_stage"),
        provider_request_id=value.get("provider_request_id"),
        provider_request_id_available=value.get("provider_request_id_available"),
        provider_request_id_unavailable_reason=value.get(
            "provider_request_id_unavailable_reason"
        ),
        token_timing_unavailable_reason=value.get(
            "token_timing_unavailable_reason"
        ),
        outcome_class=value.get("outcome_class"),
        recovery=bool(value.get("recovery")),
        retryable=bool(value.get("retryable")),
        recovery_reserve_ms=value.get("recovery_reserve_ms"),
        physical_attempt_budget_ms=value.get("physical_attempt_budget_ms"),
        request_deadline_version=str(
            value.get("request_deadline_version") or "request-deadline-v1"
        ),
    )


def reconcile_attempts(
    callback_attempts: Iterable[Mapping[str, Any]] | None,
    explicit_attempts: Iterable[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Replace a callback view with the node-owned view for the same invocation.

    A node-owned attempt is authoritative for timeout/cancel/empty-output status;
    callback token metadata remains authoritative when it is available.
    """
    callbacks = [normalize_llm_attempt(item) for item in (callback_attempts or [])]
    consumed: set[int] = set()
    merged: list[dict[str, Any]] = []
    for raw in explicit_attempts or []:
        explicit = normalize_llm_attempt(raw)
        match_index = next((
            index for index, callback in enumerate(callbacks)
            if index not in consumed
            and (
                callback["attempt_id"] == explicit["attempt_id"]
                or (
                    callback["node_id"] == explicit["node_id"]
                    and callback["route"] == explicit["route"]
                    and int(callback.get("retry_index") or 0)
                    == int(explicit.get("retry_index") or 0)
                )
            )
        ), None)
        if match_index is not None:
            consumed.add(match_index)
            callback = callbacks[match_index]
            if callback.get("token_known"):
                explicit.update({
                    "prompt_tokens": callback.get("prompt_tokens"),
                    "completion_tokens": callback.get("completion_tokens"),
                    "total_tokens": callback.get("total_tokens"),
                    "estimated_cost": callback.get("estimated_cost"),
                    "token_known": True,
                })
            if callback.get("provider_request_id_available"):
                explicit.update({
                    "provider_request_id": callback.get(
                        "provider_request_id"
                    ),
                    "provider_request_id_available": True,
                    "provider_request_id_unavailable_reason": None,
                })
            elif not explicit.get(
                "provider_request_id_unavailable_reason"
            ):
                explicit["provider_request_id_unavailable_reason"] = (
                    callback.get("provider_request_id_unavailable_reason")
                    or "provider_callback_did_not_expose_request_id"
                )
            explicit["model"] = callback.get("model") or explicit.get("model")
            # The node-owned id is the stable invocation id.  Callback ids are
            # transport observations and must not replace it.
            explicit["callback_observed"] = True
            explicit["callback_missing"] = False
        merged.append(explicit)
    merged.extend(item for index, item in enumerate(callbacks) if index not in consumed)
    return merged


def attempt_accounting(attempts: Iterable[Mapping[str, Any]] | None) -> dict[str, Any]:
    rows = [normalize_llm_attempt(item) for item in (attempts or [])]
    counts = {status: sum(item["status"] == status for item in rows) for status in ATTEMPT_STATUSES}
    terminal = sum(counts.values())
    attempt_ids = [str(item.get("attempt_id") or "") for item in rows]
    unique_ids = all(attempt_ids) and len(attempt_ids) == len(set(attempt_ids))
    evidence_complete = unique_ids and all(
        item.get("node_id")
        and item.get("route")
        and item.get("model")
        and item.get("reason_code")
        and item.get("timeout_budget_ms") is not None
        and item.get("deadline_before_ms") is not None
        and item.get("deadline_after_ms") is not None
        and item.get("completion_stage")
        and item.get("provider_request_id")
        and item.get("provider_request_id_available") is not None
        and (
            bool(item.get("provider_request_id_available"))
            or bool(item.get("provider_request_id_unavailable_reason"))
        )
        and (
            bool(item.get("token_timing_available"))
            or bool(item.get("token_timing_unavailable_reason"))
        )
        and item.get("queue_wait_ms") is not None
        and item.get("provider_ms") is not None
        and item.get("physical_attempt_budget_ms") is not None
        for item in rows
    )
    return {
        "schema": LLM_ATTEMPT_SCHEMA,
        "attempts": len(rows),
        "succeeded": counts["success"],
        "failed": counts["error"] + counts["empty_output"],
        "timed_out": counts["timeout"],
        "cancelled": counts["cancelled"],
        "empty_output": counts["empty_output"],
        "core_failures": sum(
            item["status"] != "success" and item.get("affects_core_path") for item in rows
        ),
        "optional_failures": sum(
            item["status"] != "success" and not item.get("affects_core_path") for item in rows
        ),
        "unknown_tokens": sum(not item.get("token_known") for item in rows),
        "callback_missing": sum(bool(item.get("callback_missing")) for item in rows),
        "pre_invoke_failures": sum(bool(item.get("pre_invoke_failure")) for item in rows),
        "recoveries": sum(bool(item.get("recovery")) for item in rows),
        "provider_request_id_unavailable": sum(
            not bool(item.get("provider_request_id_available")) for item in rows
        ),
        "token_timing_unavailable": sum(
            not bool(item.get("token_timing_available")) for item in rows
        ),
        "provider_request_id_unavailable_reason_missing": sum(
            not bool(item.get("provider_request_id_available"))
            and not bool(item.get("provider_request_id_unavailable_reason"))
            for item in rows
        ),
        "token_timing_unavailable_reason_missing": sum(
            not bool(item.get("token_timing_available"))
            and not bool(item.get("token_timing_unavailable_reason"))
            for item in rows
        ),
        "duplicate_attempt_ids": len(attempt_ids) - len(set(attempt_ids)),
        "evidence_complete": evidence_complete,
        "reconciled": len(rows) == terminal and unique_ids,
    }


def attempt_coverage(
    attempts: Iterable[Mapping[str, Any]] | None,
    *,
    failed_nodes: Iterable[str] | None = None,
    output_nodes: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Check observation coverage in addition to internal attempt arithmetic."""
    rows = [normalize_llm_attempt(item) for item in (attempts or [])]
    observed = {str(item.get("node_id") or "") for item in rows}
    expected_failures = {
        str(item) for item in (failed_nodes or []) if str(item) in PRODUCTION_LLM_NODES
    }
    expected_outputs = {
        str(item) for item in (output_nodes or []) if str(item) in PRODUCTION_LLM_NODES
    }
    missing_failure = sorted(expected_failures - observed)
    missing_success = sorted(expected_outputs - observed)
    typed_failure_complete = all(
        item.get("reason_code")
        and item.get("error_type")
        and item.get("timeout_budget_ms") is not None
        for item in rows
        if item.get("status") != "success"
    )
    return {
        "contract_version": LLM_INVOCATION_CONTRACT,
        "complete": not missing_failure and not missing_success and typed_failure_complete,
        "failed_nodes_expected": sorted(expected_failures),
        "output_nodes_expected": sorted(expected_outputs),
        "missing_failure_attempts": missing_failure,
        "missing_success_attempts": missing_success,
        "typed_failure_complete": typed_failure_complete,
    }


def normalize_degradation(value: Mapping[str, Any] | None) -> dict[str, Any]:
    item = dict(value or {})
    original_node = item.get("node_id") or item.get("node")
    node = str(original_node or "unknown")
    route = str(item.get("route") or _route_for_node(node))
    reason = str(item.get("reason_code") or item.get("reason") or "unknown")[:120]
    display_reason = str(
        item.get("reason")
        or (reason.split(".", 1)[1] if reason.startswith("llm.") else reason)
    )[:120]
    return {
        "node_id": node,
        "route": route,
        "reason_code": reason,
        # Public degradation DTO field consumed by the current frontend.
        "reason": display_reason,
        "severity": str(item.get("severity") or "warning")[:32],
        "affects_core_path": (
            affects_core_path(node)
            if item.get("affects_core_path") is None else bool(item.get("affects_core_path"))
        ),
        "message": str(item.get("message") or item.get("detail") or "")[:300] or None,
        "duration_ms": (
            max(0, int(float(item.get("duration_ms") or 0)))
            if item.get("duration_ms") is not None else None
        ),
        "evidence_complete": bool(original_node and reason != "unknown"),
    }
