"""不依赖 Agent 运行时的 LLM 调用证据构造器。"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterable, Mapping
from typing import Any

from app.foundation.deadlines import REQUEST_DEADLINE_VERSION


LLM_ATTEMPT_SCHEMA = "llm-attempt-v2"
LLM_INVOCATION_CONTRACT = "llm-invocation-v2"
ATTEMPT_STATUSES = ("success", "error", "timeout", "cancelled", "empty_output")
OPTIONAL_POSTPROCESS_NODES = {
    "generate_chart",
    "generate_analysis",
    "generate_recommend",
}


def _reason_code_for_status(status: str) -> str:
    return {
        "success": "llm.success",
        "timeout": "llm.timeout",
        "cancelled": "llm.cancelled",
        "empty_output": "llm.empty_output",
        "error": "llm.provider_error",
    }.get(str(status), "llm.unknown_failure")


def make_llm_attempt(
    *,
    node_id: str,
    status: str,
    latency_ms: int | float,
    route: str | None = None,
    model: str | None = None,
    timeout_budget_ms: int | float | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    estimated_cost: float | None = None,
    token_known: bool | None = None,
    error_type: str | None = None,
    reason_code: str | None = None,
    retry_index: int = 0,
    callback_observed: bool | None = None,
    pre_invoke_failure: bool = False,
    core_path: bool | None = None,
    attempt_id: str | None = None,
    source: str = "node",
    deadline_before_ms: int | float | None = None,
    deadline_after_ms: int | float | None = None,
    queue_wait_ms: int | float | None = None,
    provider_ms: int | float | None = None,
    time_to_first_token_ms: int | float | None = None,
    time_to_last_token_ms: int | float | None = None,
    token_timing_available: bool = False,
    completion_stage: str | None = None,
    cancel_stage: str | None = None,
    provider_request_id: str | None = None,
    provider_request_id_available: bool | None = None,
    provider_request_id_unavailable_reason: str | None = None,
    token_timing_unavailable_reason: str | None = None,
    outcome_class: str | None = None,
    recovery: bool = False,
    retryable: bool = False,
    recovery_reserve_ms: int | float | None = None,
    physical_attempt_budget_ms: int | float | None = None,
    request_deadline_version: str = REQUEST_DEADLINE_VERSION,
) -> dict[str, Any]:
    """构造有界且不包含 Prompt/响应正文的物理调用证据。"""

    node = str(node_id or "unknown")
    resolved_route = str(route or "unknown")
    resolved_status = status if status in ATTEMPT_STATUSES else "error"
    known = bool(token_known) if token_known is not None else all(
        value is not None for value in (prompt_tokens, completion_tokens, total_tokens)
    )
    provider_id = str(provider_request_id or "").strip()
    provider_id_available = (
        bool(provider_request_id_available)
        if provider_request_id_available is not None
        else bool(provider_id)
    )
    if not provider_id:
        provider_id = "unavailable"
    provider_id_unavailable_reason = (
        None
        if provider_id_available
        else str(
            provider_request_id_unavailable_reason
            or "provider_response_did_not_expose_request_id"
        )[:120]
    )
    timing_unavailable_reason = (
        None
        if token_timing_available
        else str(
            token_timing_unavailable_reason
            or "non_stream_invocation_does_not_expose_token_timing"
        )[:120]
    )
    resolved_reason = str(
        reason_code or _reason_code_for_status(resolved_status)
    )[:120]
    resolved_outcome = str(
        outcome_class
        or (
            "success"
            if resolved_status == "success"
            else resolved_reason.removeprefix("llm.")
        )
    )[:64]
    budget_before = (
        max(0, int(deadline_before_ms))
        if deadline_before_ms is not None
        else (
            max(0, int(timeout_budget_ms))
            if timeout_budget_ms is not None
            else None
        )
    )
    budget_after = (
        max(0, int(deadline_after_ms))
        if deadline_after_ms is not None
        else (
            max(0, int((budget_before or 0) - max(0, int(latency_ms or 0))))
            if budget_before is not None
            else None
        )
    )
    return {
        "schema": LLM_ATTEMPT_SCHEMA,
        "contract_version": LLM_INVOCATION_CONTRACT,
        "attempt_id": str(attempt_id or uuid.uuid4().hex),
        "node_id": node,
        "route": resolved_route,
        "model": str(model or "unregistered"),
        "status": resolved_status,
        "latency_ms": max(0, int(latency_ms or 0)),
        "timeout_budget_ms": (
            max(0, int(timeout_budget_ms)) if timeout_budget_ms is not None else None
        ),
        "prompt_tokens": max(0, int(prompt_tokens or 0)) if known else None,
        "completion_tokens": max(0, int(completion_tokens or 0)) if known else None,
        "total_tokens": max(0, int(total_tokens or 0)) if known else None,
        "estimated_cost": max(0.0, float(estimated_cost or 0)),
        "token_known": known,
        "affects_core_path": (
            node not in OPTIONAL_POSTPROCESS_NODES
            if core_path is None
            else bool(core_path)
        ),
        "error_type": str(error_type or "")[:120] or None,
        "reason_code": resolved_reason,
        "retry_index": max(0, int(retry_index or 0)),
        "callback_observed": (
            bool(callback_observed)
            if callback_observed is not None
            else str(source or "").lower() == "callback"
        ),
        "callback_missing": (
            not bool(callback_observed)
            if callback_observed is not None
            else str(source or "").lower() != "callback"
        ),
        "pre_invoke_failure": bool(pre_invoke_failure),
        "source": str(source or "unknown")[:32],
        "request_deadline_version": str(request_deadline_version)[:64],
        "deadline_before_ms": budget_before,
        "deadline_after_ms": budget_after,
        "queue_wait_ms": max(0, int(queue_wait_ms or 0)),
        "provider_ms": max(0, int(provider_ms or 0)),
        "time_to_first_token_ms": (
            max(0, int(time_to_first_token_ms))
            if token_timing_available and time_to_first_token_ms is not None
            else None
        ),
        "time_to_last_token_ms": (
            max(0, int(time_to_last_token_ms))
            if token_timing_available and time_to_last_token_ms is not None
            else None
        ),
        "token_timing_available": bool(token_timing_available),
        "completion_stage": str(
            completion_stage
            or ("provider_complete" if resolved_status == "success" else "provider")
        )[:64],
        "cancel_stage": str(cancel_stage or "")[:64] or None,
        "provider_request_id": provider_id[:160],
        "provider_request_id_available": provider_id_available,
        "provider_request_id_unavailable_reason": provider_id_unavailable_reason,
        "token_timing_unavailable_reason": timing_unavailable_reason,
        "outcome_class": resolved_outcome,
        "recovery": bool(recovery),
        "retryable": bool(retryable),
        "recovery_reserve_ms": (
            max(0, int(recovery_reserve_ms))
            if recovery_reserve_ms is not None
            else 0
        ),
        "physical_attempt_budget_ms": (
            max(0, int(physical_attempt_budget_ms))
            if physical_attempt_budget_ms is not None
            else (
                max(0, int(timeout_budget_ms))
                if timeout_budget_ms is not None
                else None
            )
        ),
    }


def classify_attempt_status(error: BaseException) -> str:
    """把异常归一化为有限的 attempt 终态。"""

    if isinstance(error, asyncio.CancelledError):
        return "cancelled"
    if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        return "timeout"
    return "error"


def normalize_llm_attempt(value: Mapping[str, Any]) -> dict[str, Any]:
    """把历史或回调 attempt 重新收敛到当前证据合同。"""

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
            value.get("request_deadline_version") or REQUEST_DEADLINE_VERSION
        ),
    )


def reconcile_attempts(
    callback_attempts: Iterable[Mapping[str, Any]] | None,
    explicit_attempts: Iterable[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """以节点事实为终态，并合并同一次回调提供的 token 元数据。"""

    callbacks = [normalize_llm_attempt(item) for item in (callback_attempts or [])]
    consumed: set[int] = set()
    merged: list[dict[str, Any]] = []
    for raw in explicit_attempts or []:
        explicit = normalize_llm_attempt(raw)
        match_index = next(
            (
                index
                for index, callback in enumerate(callbacks)
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
            ),
            None,
        )
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
                    "provider_request_id": callback.get("provider_request_id"),
                    "provider_request_id_available": True,
                    "provider_request_id_unavailable_reason": None,
                })
            elif not explicit.get("provider_request_id_unavailable_reason"):
                explicit["provider_request_id_unavailable_reason"] = (
                    callback.get("provider_request_id_unavailable_reason")
                    or "provider_callback_did_not_expose_request_id"
                )
            explicit["model"] = callback.get("model") or explicit.get("model")
            explicit["callback_observed"] = True
            explicit["callback_missing"] = False
        merged.append(explicit)
    merged.extend(
        item for index, item in enumerate(callbacks) if index not in consumed
    )
    return merged


def attempt_accounting(
    attempts: Iterable[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """核对 attempt 终态、标识唯一性与审计字段完整性。"""

    rows = [normalize_llm_attempt(item) for item in (attempts or [])]
    counts = {
        status: sum(item["status"] == status for item in rows)
        for status in ATTEMPT_STATUSES
    }
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
            item["status"] != "success" and item.get("affects_core_path")
            for item in rows
        ),
        "optional_failures": sum(
            item["status"] != "success" and not item.get("affects_core_path")
            for item in rows
        ),
        "unknown_tokens": sum(not item.get("token_known") for item in rows),
        "callback_missing": sum(bool(item.get("callback_missing")) for item in rows),
        "pre_invoke_failures": sum(
            bool(item.get("pre_invoke_failure")) for item in rows
        ),
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


__all__ = [
    "ATTEMPT_STATUSES",
    "LLM_ATTEMPT_SCHEMA",
    "LLM_INVOCATION_CONTRACT",
    "OPTIONAL_POSTPROCESS_NODES",
    "attempt_accounting",
    "classify_attempt_status",
    "make_llm_attempt",
    "normalize_llm_attempt",
    "reconcile_attempts",
]
