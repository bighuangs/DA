"""Monotonic request deadline and phase-budget helpers.

The contract stores only process-local monotonic timestamps in graph state.
Persisted evidence is produced through :func:`compact_deadline_evidence` and
never includes those absolute values.
"""
from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from app.foundation.deadlines import REQUEST_DEADLINE_VERSION

MODEL_PHASE_BUDGET_VERSION = "model-phase-budget-v1"
DEFAULT_REQUEST_DEADLINE_MS = 60_000
DEFAULT_PHASE_BUDGETS_MS = {
    "retrieval": 7_000,
    "planning": 6_000,
    "generation": 22_000,
    "correction": 16_000,
    "execution": 6_000,
}
DEFAULT_EXECUTION_RESERVE_MS = 4_000

_ROUTE_PHASE = {
    "keyword": "retrieval",
    "selection": "retrieval",
    "planning": "planning",
    "sql": "generation",
    "candidate": "generation",
    "correction": "correction",
    "postprocess": "postprocess",
    "recommend": "postprocess",
}


def initialize_request_deadline(
    started_at: float | None = None,
    *,
    budget_ms: int | None = None,
    phase_budgets_ms: Mapping[str, int] | None = None,
    execution_reserve_ms: int | None = None,
) -> dict[str, Any]:
    start = float(started_at if started_at is not None else time.perf_counter())
    total = max(
        1,
        int(
            DEFAULT_REQUEST_DEADLINE_MS
            if budget_ms is None
            else budget_ms
        ),
    )
    phases = {
        str(key): max(1, int(value))
        for key, value in (
            phase_budgets_ms or DEFAULT_PHASE_BUDGETS_MS
        ).items()
    }
    return {
        "contract_version": REQUEST_DEADLINE_VERSION,
        "started_at": start,
        "deadline_at": start + total / 1000,
        "budget_ms": total,
        "phase_budgets_ms": phases,
        "execution_reserve_ms": max(
            0,
            int(
                DEFAULT_EXECUTION_RESERVE_MS
                if execution_reserve_ms is None
                else execution_reserve_ms
            ),
        ),
    }


def ensure_request_deadline(state: Mapping[str, Any] | None) -> dict[str, Any]:
    existing = dict((state or {}).get("request_deadline") or {})
    if (
        existing.get("contract_version") == REQUEST_DEADLINE_VERSION
        and existing.get("deadline_at") is not None
    ):
        return existing
    return initialize_request_deadline(
        (state or {}).get("performance_started_at"),
    )


def remaining_ms(
    deadline: Mapping[str, Any] | None,
    *,
    now: float | None = None,
) -> int:
    current = time.perf_counter() if now is None else float(now)
    try:
        absolute = float((deadline or {}).get("deadline_at"))
    except (TypeError, ValueError):
        return 0
    return max(0, int((absolute - current) * 1000))


def route_deadline_at(
    state: Mapping[str, Any] | None,
    route: str,
    *,
    timeout_seconds: float | None = None,
    now: float | None = None,
) -> float:
    current = time.perf_counter() if now is None else float(now)
    contract = ensure_request_deadline(state)
    request_deadline = float(contract["deadline_at"])
    phase = _ROUTE_PHASE.get(str(route or ""), "generation")
    phase_budget = int(
        (contract.get("phase_budgets_ms") or {}).get(
            phase,
            max(1, int(float(timeout_seconds or 30) * 1000)),
        )
    )
    local_budget = phase_budget
    if timeout_seconds is not None:
        local_budget = min(
            local_budget,
            max(1, int(float(timeout_seconds) * 1000)),
        )
    if phase not in {"execution", "postprocess"}:
        request_deadline -= (
            max(0, int(contract.get("execution_reserve_ms") or 0)) / 1000
        )
    return min(request_deadline, current + local_budget / 1000)


def route_timeout_seconds(
    state: Mapping[str, Any] | None,
    route: str,
    *,
    timeout_seconds: float | None = None,
    now: float | None = None,
) -> float:
    """Return the remaining duration inside the configured route boundary.

    Dependency wrappers consume a relative timeout while the request contract
    stores an absolute monotonic deadline.  Converting in one place prevents a
    nominal 45-second dependency timeout from silently bypassing the existing
    7-second retrieval phase and starving a required downstream model call.
    """

    current = time.perf_counter() if now is None else float(now)
    deadline = route_deadline_at(
        state,
        route,
        timeout_seconds=timeout_seconds,
        now=current,
    )
    return max(0.001, deadline - current)


def model_phase_budget(
    state: Mapping[str, Any] | None,
    route: str,
    *,
    minimum_start_ms: int,
    timeout_seconds: float | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Return a privacy-safe go/no-go decision before a physical model call.

    The decision is based on the same phase boundary and execution reserve as
    :func:`route_deadline_at`.  It never persists monotonic timestamps and lets
    callers skip calls that cannot receive a useful completion window.
    """

    current = time.perf_counter() if now is None else float(now)
    contract = ensure_request_deadline(state)
    phase_deadline = route_deadline_at(
        state,
        route,
        timeout_seconds=timeout_seconds,
        now=current,
    )
    available = max(0, int((phase_deadline - current) * 1000))
    minimum = max(1, int(minimum_start_ms))
    allowed = available >= minimum
    return {
        "contract_version": MODEL_PHASE_BUDGET_VERSION,
        "request_deadline_version": contract.get("contract_version"),
        "route": str(route or ""),
        "available_ms": available,
        "minimum_start_ms": minimum,
        "allowed": allowed,
        "reason_code": (
            "model_phase.budget_available"
            if allowed
            else "model_phase.insufficient_budget"
        ),
        "absolute_timestamps_persisted": False,
    }


def execution_timeout_seconds(
    state: Mapping[str, Any] | None,
    *,
    now: float | None = None,
) -> float:
    current = time.perf_counter() if now is None else float(now)
    contract = ensure_request_deadline(state)
    phase_budget = int(
        (contract.get("phase_budgets_ms") or {}).get("execution", 2500)
    )
    available_ms = min(remaining_ms(contract, now=current), phase_budget)
    return max(0.001, available_ms / 1000)


def compact_deadline_evidence(
    deadline: Mapping[str, Any] | None,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    item = dict(deadline or {})
    return {
        "contract_version": item.get("contract_version"),
        "budget_ms": int(item.get("budget_ms") or 0),
        "remaining_ms": remaining_ms(item, now=now),
        "phase_budgets_ms": {
            str(key): int(value)
            for key, value in (item.get("phase_budgets_ms") or {}).items()
        },
        "execution_reserve_ms": int(item.get("execution_reserve_ms") or 0),
        "absolute_timestamps_persisted": False,
    }
