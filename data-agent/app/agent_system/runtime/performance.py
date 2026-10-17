"""Node-level performance instrumentation and post-processing budgets."""
from __future__ import annotations

import time
import inspect
from functools import wraps
from typing import Any, Callable

from app.foundation.runtime_context import current_node_id


def elapsed_ms(state: dict) -> int:
    started = float(state.get("performance_started_at") or time.perf_counter())
    return max(0, int((time.perf_counter() - started) * 1000))


def postprocess_budget_exhausted(
    state: dict,
    *,
    budget_ms: int = 60_000,
) -> bool:
    budget = int(budget_ms or 0)
    return budget > 0 and elapsed_ms(state) >= budget


def instrument_node(
    node_id: str,
    func: Callable[..., Any],
):
    """Wrap a LangGraph node with stable, parallel-safe timing evidence."""
    @wraps(func)
    async def wrapped(state, runtime):
        started = time.perf_counter()
        request_started = float(state.get("performance_started_at") or started)
        start_ms = max(0, int((started - request_started) * 1000))
        runtime.stream_writer({
            "trace_event": {"event": "node_start", "node_id": node_id, "start_ms": start_ms}
        })
        token = current_node_id.set(node_id)
        status = "success"
        error = None
        try:
            output = func(state, runtime)
            if inspect.isawaitable(output):
                output = await output
            return {
                **(output or {}),
                "node_metrics": [{
                    "node_id": node_id,
                    "start_ms": start_ms,
                    "end_ms": max(0, int((time.perf_counter() - request_started) * 1000)),
                    "duration_ms": max(0, int((time.perf_counter() - started) * 1000)),
                    "status": status,
                }],
            }
        except Exception as exc:
            status = "failed"
            error = str(exc)[:500]
            try:
                setattr(exc, "node_id", node_id)
            except Exception:
                pass
            raise
        finally:
            ended = time.perf_counter()
            runtime.stream_writer({
                "trace_event": {
                    "event": "node_end",
                    "node_id": node_id,
                    "start_ms": start_ms,
                    "end_ms": max(0, int((ended - request_started) * 1000)),
                    "duration_ms": max(0, int((ended - started) * 1000)),
                    "status": status,
                    "error": error,
                }
            })
            current_node_id.reset(token)

    return wrapped
