"""支持端到端截止时间、取消与流式输出的 Graph Runner。"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Any, TypeAlias, cast

from app.agent_system.contracts import (
    Error,
    ExecutionSnapshot,
    ExecutionStatus,
    InvocationContext,
    OutputEvent,
    OutputEventType,
    QueryInput,
    StateV2,
    new_state,
)
from app.agent_system.governance import BudgetPolicy
from app.agent_system.runtime.graph import GraphFactory, RunnableGraph


StateFactory: TypeAlias = Callable[[QueryInput], StateV2]
EventMapper: TypeAlias = Callable[[object, str, int, str], OutputEvent]


class CancellationToken:
    """允许调用边界在不取消所属 Task 的情况下请求终止。"""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason = "caller_requested"

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    def cancel(self, reason: str = "caller_requested") -> None:
        normalized = str(reason or "").strip()
        self._reason = normalized or "caller_requested"
        self._event.set()

    async def wait(self) -> str:
        await self._event.wait()
        return self._reason


class _ExecutionCancelled(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_value(value: object, *, field_name: str) -> object:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是 JSON 值") from exc


def _default_event_mapper(
    raw: object,
    request_id: str,
    sequence: int,
    emitted_at: str,
) -> OutputEvent:
    if isinstance(raw, OutputEvent):
        return OutputEvent(
            request_id=request_id,
            sequence=sequence,
            type=raw.type,
            emitted_at=emitted_at,
            payload=raw.payload,
            error=raw.error,
        )
    copied = _json_value(raw, field_name="Graph stream event")
    if isinstance(copied, dict):
        payload = {"state": copied}
    else:
        payload = {"value": copied}
    return OutputEvent(
        request_id=request_id,
        sequence=sequence,
        type=OutputEventType.STATE,
        emitted_at=emitted_at,
        payload=payload,
    )


async def _await_or_cancel(
    awaitable: Awaitable[Any],
    cancellation: CancellationToken | None,
) -> Any:
    if cancellation is None:
        return await awaitable
    if cancellation.is_cancelled:
        if inspect.iscoroutine(awaitable):
            awaitable.close()
        raise _ExecutionCancelled(cancellation.reason)
    operation = asyncio.ensure_future(awaitable)
    cancel_wait = asyncio.create_task(cancellation.wait())
    try:
        done, _ = await asyncio.wait(
            {operation, cancel_wait},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if operation in done:
            return await operation
        operation.cancel()
        await asyncio.gather(operation, return_exceptions=True)
        raise _ExecutionCancelled(cancel_wait.result())
    finally:
        if not cancel_wait.done():
            cancel_wait.cancel()
        await asyncio.gather(cancel_wait, return_exceptions=True)


async def _await_before_deadline(
    awaitable: Awaitable[Any],
    *,
    deadline_at: float,
    cancellation: CancellationToken | None,
) -> Any:
    async with asyncio.timeout_at(deadline_at):
        return await _await_or_cancel(awaitable, cancellation)


async def _open_stream(
    graph: RunnableGraph,
    state: Mapping[str, object],
    context: object,
    *,
    stream_mode: str,
    config: Mapping[str, object] | None,
    deadline_at: float,
    cancellation: CancellationToken | None,
) -> AsyncIterator[object]:
    if config is None:
        stream = graph.astream(state, context=context, stream_mode=stream_mode)
    else:
        stream = graph.astream(
            state,
            context=context,
            stream_mode=stream_mode,
            config=config,
        )
    if inspect.isawaitable(stream):
        stream = await _await_before_deadline(
            stream,
            deadline_at=deadline_at,
            cancellation=cancellation,
        )
    if not hasattr(stream, "__anext__"):
        raise TypeError("Graph astream() 必须返回异步迭代器")
    return cast(AsyncIterator[object], stream)


async def _close_stream(stream: AsyncIterator[object] | None) -> None:
    close = getattr(stream, "aclose", None)
    if not callable(close):
        return
    try:
        async with asyncio.timeout(1):
            await close()
    except (TimeoutError, RuntimeError):
        return


class GraphRunner:
    """把 Graph 的内部输出转换为稳定合同，并统一治理终态。"""

    def __init__(
        self,
        *,
        factory: GraphFactory,
        agent_id: str = "data-agent.nl2sql",
        budget_policy: BudgetPolicy | None = None,
        state_factory: StateFactory = new_state,
        event_mapper: EventMapper = _default_event_mapper,
        stream_mode: str = "values",
    ) -> None:
        self._factory = factory
        self._agent_id = str(agent_id or "").strip()
        if not self._agent_id:
            raise ValueError("agent_id 不能为空")
        self._budget_policy = budget_policy or BudgetPolicy()
        self._state_factory = state_factory
        self._event_mapper = event_mapper
        self._stream_mode = str(stream_mode or "").strip()
        if not self._stream_mode:
            raise ValueError("stream_mode 不能为空")

    @property
    def factory(self) -> GraphFactory:
        """返回所属 GraphFactory，供组合根观测惰性编译状态。"""

        return self._factory

    async def stream(
        self,
        query_input: QueryInput,
        context: InvocationContext,
        *,
        initial_state: Mapping[str, object] | None = None,
        graph_context: object | None = None,
        graph_config: Mapping[str, object] | None = None,
        timeout_seconds: float | None = None,
        cancellation: CancellationToken | None = None,
    ) -> AsyncIterator[OutputEvent]:
        """流式运行一次 Graph；deadline 覆盖构建、编译和全部节点。"""

        if query_input.request_id != context.request_id:
            raise ValueError("QueryInput 与 InvocationContext 的 request_id 必须一致")
        effective_timeout = (
            self._budget_policy.total_timeout_seconds
            if timeout_seconds is None
            else float(timeout_seconds)
        )
        if effective_timeout <= 0:
            raise ValueError("timeout_seconds 必须大于 0")

        started_at = _utc_now()
        started_clock = time.perf_counter()
        deadline_at = asyncio.get_running_loop().time() + effective_timeout
        state = dict(initial_state or self._state_factory(query_input))
        sequence = 0
        stream: AsyncIterator[object] | None = None
        error: Error | None = None
        error_emitted = False
        status = ExecutionStatus.RUNNING

        yield OutputEvent(
            request_id=query_input.request_id,
            sequence=sequence,
            type=OutputEventType.STARTED,
            emitted_at=started_at,
            payload={"agent_id": self._agent_id},
        )
        sequence += 1

        try:
            graph = await _await_before_deadline(
                self._factory.get_graph(),
                deadline_at=deadline_at,
                cancellation=cancellation,
            )
            stream = await _open_stream(
                graph,
                state,
                context.to_dict() if graph_context is None else graph_context,
                stream_mode=self._stream_mode,
                config=graph_config,
                deadline_at=deadline_at,
                cancellation=cancellation,
            )
            while True:
                try:
                    raw = await _await_before_deadline(
                        stream.__anext__(),
                        deadline_at=deadline_at,
                        cancellation=cancellation,
                    )
                except StopAsyncIteration:
                    break
                event = self._event_mapper(
                    raw,
                    query_input.request_id,
                    sequence,
                    _utc_now(),
                )
                event_state = event.payload.get("state")
                if isinstance(event_state, dict):
                    state = dict(event_state)
                if event.type is OutputEventType.ERROR:
                    error = event.error
                    error_emitted = True
                    status = ExecutionStatus.FAILED
                yield event
                sequence += 1

            if status is ExecutionStatus.RUNNING:
                state_error = state.get("error")
                state_status = str(state.get("status") or "")
                if state_error or state_status in {
                    "failed",
                    "cancelled",
                    "timed_out",
                }:
                    if isinstance(state_error, dict) and state_error.get("version"):
                        error = Error.from_dict(state_error)
                    else:
                        error = Error(
                            code="agent.reported_failure",
                            message="Agent 返回了失败终态",
                            stage=str(state.get("phase") or "graph"),
                        )
                    status = {
                        "cancelled": ExecutionStatus.CANCELLED,
                        "timed_out": ExecutionStatus.TIMED_OUT,
                    }.get(state_status, ExecutionStatus.FAILED)
                else:
                    status = ExecutionStatus.SUCCEEDED
        except TimeoutError:
            status = ExecutionStatus.TIMED_OUT
            error = Error(
                code="agent.deadline_exceeded",
                message="Agent 执行超过端到端截止时间",
                retryable=True,
                stage="runtime",
            )
        except _ExecutionCancelled as exc:
            status = ExecutionStatus.CANCELLED
            error = Error(
                code="agent.cancelled",
                message="Agent 执行已取消",
                stage="runtime",
                details={"reason": exc.reason},
            )
        except asyncio.CancelledError:
            await _close_stream(stream)
            raise
        except Exception as exc:
            status = ExecutionStatus.FAILED
            error = Error(
                code="agent.execution_failed",
                message="Agent 执行失败",
                stage="runtime",
                details={"exception_type": type(exc).__name__},
            )
        finally:
            await _close_stream(stream)

        state["status"] = status.value
        if error is not None:
            state["error"] = error.to_dict()
            if not error_emitted:
                yield OutputEvent(
                    request_id=query_input.request_id,
                    sequence=sequence,
                    type=OutputEventType.ERROR,
                    emitted_at=_utc_now(),
                    error=error,
                )
                sequence += 1

        finished_at = _utc_now()
        snapshot = ExecutionSnapshot(
            request_id=query_input.request_id,
            agent_id=self._agent_id,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            last_sequence=max(0, sequence - 1),
            state=cast(dict[str, Any], _json_value(state, field_name="execution state")),
            error=error,
            metrics={
                "elapsed_ms": max(0, int((time.perf_counter() - started_clock) * 1_000)),
                "stream_event_count": sequence,
            },
        )
        yield OutputEvent(
            request_id=query_input.request_id,
            sequence=sequence,
            type=OutputEventType.SNAPSHOT,
            emitted_at=finished_at,
            payload={"snapshot": snapshot.to_dict()},
        )
        sequence += 1
        yield OutputEvent(
            request_id=query_input.request_id,
            sequence=sequence,
            type=OutputEventType.COMPLETED,
            emitted_at=_utc_now(),
            payload={"status": status.value},
        )

    async def execute(
        self,
        query_input: QueryInput,
        context: InvocationContext,
        *,
        initial_state: Mapping[str, object] | None = None,
        graph_context: object | None = None,
        graph_config: Mapping[str, object] | None = None,
        timeout_seconds: float | None = None,
        cancellation: CancellationToken | None = None,
    ) -> ExecutionSnapshot:
        """消费流并返回最终快照，适合非 SSE 调用方与评测 Harness。"""

        snapshot: ExecutionSnapshot | None = None
        async for event in self.stream(
            query_input,
            context,
            initial_state=initial_state,
            graph_context=graph_context,
            graph_config=graph_config,
            timeout_seconds=timeout_seconds,
            cancellation=cancellation,
        ):
            if event.type is OutputEventType.SNAPSHOT:
                raw_snapshot = event.payload.get("snapshot")
                if isinstance(raw_snapshot, dict):
                    snapshot = ExecutionSnapshot.from_dict(raw_snapshot)
        if snapshot is None:
            raise RuntimeError("GraphRunner 未产生 ExecutionSnapshot")
        return snapshot
