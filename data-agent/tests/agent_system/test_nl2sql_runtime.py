from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.agent_system.agents.nl2sql.runtime import create_nl2sql_graph_runner
from app.agent_system.contracts import (
    ExecutionStatus,
    InvocationContext,
    OutputEventType,
    QueryInput,
)
from app.agent_system.runtime import GraphFactory


class _RecordingGraph:
    def __init__(self) -> None:
        self.context: object | None = None
        self.config: object | None = None

    async def astream(self, state, *, context, stream_mode, config=None):
        self.context = context
        self.config = config
        assert stream_mode == "custom"
        yield {"stage": "测试阶段"}
        yield {"error": "测试失败"}


class _Definition:
    def __init__(self, graph: _RecordingGraph) -> None:
        self.graph = graph

    def compile(self) -> _RecordingGraph:
        return self.graph


def _contracts() -> tuple[QueryInput, InvocationContext]:
    query_input = QueryInput(
        request_id="trace-1",
        query="查询销售额",
        actor_id="user-1",
        workspace_id="workspace-1",
    )
    context = InvocationContext(
        request_id=query_input.request_id,
        actor_id=query_input.actor_id,
        workspace_id=query_input.workspace_id,
    )
    return query_input, context


def test_nl2sql_runtime_does_not_build_or_compile_during_creation() -> None:
    runner = create_nl2sql_graph_runner()

    assert runner.factory.is_compiled is False
    assert runner.factory.compile_count == 0

    graph_path = (
        Path(__file__).resolve().parents[2]
        / "app"
        / "agent_system"
        / "agents"
        / "nl2sql"
        / "graph.py"
    )
    tree = ast.parse(graph_path.read_text(encoding="utf-8"))
    module_assignments = {
        target.id
        for statement in tree.body
        if isinstance(statement, (ast.Assign, ast.AnnAssign))
        for target in (
            statement.targets
            if isinstance(statement, ast.Assign)
            else (statement.target,)
        )
        if isinstance(target, ast.Name)
    }
    assert "graph" not in module_assignments
    assert "graph_builder" not in module_assignments


@pytest.mark.asyncio
async def test_nl2sql_runner_preserves_custom_chunks_and_emits_failed_snapshot() -> None:
    graph = _RecordingGraph()
    factory = GraphFactory(builder=lambda _: _Definition(graph))
    runner = create_nl2sql_graph_runner(factory=factory)
    query_input, context = _contracts()
    graph_context = {"repository": "injected-per-request"}
    graph_config = {"callbacks": ["collector"], "metadata": {"trace_id": "trace-1"}}

    events = [
        event
        async for event in runner.stream(
            query_input,
            context,
            initial_state={"query": query_input.query},
            graph_context=graph_context,
            graph_config=graph_config,
            timeout_seconds=1,
        )
    ]

    assert [event.type for event in events] == [
        OutputEventType.STARTED,
        OutputEventType.MESSAGE,
        OutputEventType.ERROR,
        OutputEventType.SNAPSHOT,
        OutputEventType.COMPLETED,
    ]
    assert events[1].payload["chunk"] == {"stage": "测试阶段"}
    assert events[2].payload["chunk"] == {"error": "测试失败"}
    assert events[2].error is not None
    assert events[2].error.message == "测试失败"
    snapshot = events[3].payload["snapshot"]
    assert isinstance(snapshot, dict)
    assert snapshot["status"] == ExecutionStatus.FAILED.value
    assert events[-1].payload["status"] == ExecutionStatus.FAILED.value
    assert graph.context is graph_context
    assert graph.config is graph_config
    assert factory.compile_count == 1
