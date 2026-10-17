import asyncio

import pytest

from app.agent_system.contracts import (
    ExecutionStatus,
    InvocationContext,
    OutputEventType,
    QueryInput,
)
from app.agent_system.runtime import AgentCapabilities, CancellationToken, GraphFactory, GraphRunner


class _UnusedCapability:
    pass


def _capabilities() -> AgentCapabilities:
    unused = _UnusedCapability()
    return AgentCapabilities(
        embedding=unused,  # type: ignore[arg-type]
        retrieval=unused,  # type: ignore[arg-type]
        metadata=unused,  # type: ignore[arg-type]
        model=unused,  # type: ignore[arg-type]
        sql_execution=unused,  # type: ignore[arg-type]
    )


class _FakeGraph:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.delay = delay
        self.closed = False

    async def astream(self, state, *, context, stream_mode):
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield {
                **state,
                "status": "succeeded",
                "phase": "completed",
                "analysis": f"actor={context['actor_id']};mode={stream_mode}",
            }
        finally:
            self.closed = True


class _FakeDefinition:
    def __init__(self, graph: _FakeGraph, counters: dict[str, int]) -> None:
        self.graph = graph
        self.counters = counters

    def compile(self) -> _FakeGraph:
        self.counters["compile"] += 1
        return self.graph


def _input_and_context() -> tuple[QueryInput, InvocationContext]:
    query_input = QueryInput(
        request_id="req-runtime",
        query="查询销售额",
        actor_id="user-1",
        workspace_id="workspace-1",
    )
    context = InvocationContext(
        request_id=query_input.request_id,
        actor_id=query_input.actor_id,
        workspace_id=query_input.workspace_id,
        allowed_data_sources=("sales",),
    )
    return query_input, context


@pytest.mark.asyncio
async def test_graph_factory_compiles_lazily_and_only_once() -> None:
    counters = {"build": 0, "compile": 0}
    graph = _FakeGraph()

    def build(_capabilities):
        counters["build"] += 1
        return _FakeDefinition(graph, counters)

    factory = GraphFactory(builder=build, capabilities=_capabilities())
    assert factory.is_compiled is False
    assert counters == {"build": 0, "compile": 0}

    first, second = await asyncio.gather(factory.get_graph(), factory.get_graph())

    assert first is graph
    assert second is graph
    assert factory.is_compiled is True
    assert factory.compile_count == 1
    assert counters == {"build": 1, "compile": 1}


@pytest.mark.asyncio
async def test_runner_enforces_end_to_end_deadline() -> None:
    graph = _FakeGraph(delay=1.0)
    factory = GraphFactory(
        builder=lambda _: _FakeDefinition(graph, {"compile": 0}),
        capabilities=_capabilities(),
    )
    runner = GraphRunner(factory=factory)
    query_input, context = _input_and_context()

    events = [
        event
        async for event in runner.stream(
            query_input,
            context,
            timeout_seconds=0.02,
        )
    ]

    assert [event.type for event in events] == [
        OutputEventType.STARTED,
        OutputEventType.ERROR,
        OutputEventType.SNAPSHOT,
        OutputEventType.COMPLETED,
    ]
    assert events[1].error is not None
    assert events[1].error.code == "agent.deadline_exceeded"
    assert events[-1].payload["status"] == ExecutionStatus.TIMED_OUT.value
    assert graph.closed is True


@pytest.mark.asyncio
async def test_runner_interrupts_blocked_stream_when_token_is_cancelled() -> None:
    graph = _FakeGraph(delay=1.0)
    factory = GraphFactory(
        builder=lambda _: _FakeDefinition(graph, {"compile": 0}),
        capabilities=_capabilities(),
    )
    runner = GraphRunner(factory=factory)
    query_input, context = _input_and_context()
    cancellation = CancellationToken()

    async def cancel_soon() -> None:
        await asyncio.sleep(0.02)
        cancellation.cancel("http_disconnect")

    cancel_task = asyncio.create_task(cancel_soon())
    snapshot = await runner.execute(
        query_input,
        context,
        timeout_seconds=1.0,
        cancellation=cancellation,
    )
    await cancel_task

    assert snapshot.status is ExecutionStatus.CANCELLED
    assert snapshot.error is not None
    assert snapshot.error.code == "agent.cancelled"
    assert snapshot.error.details == {"reason": "http_disconnect"}
    assert graph.closed is True
