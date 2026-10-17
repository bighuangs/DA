"""NL2SQL Application Runtime 出站适配器测试。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from app.adapters.outbound.agent.nl2sql_runtime import Nl2SqlApplicationRuntime
from app.application.ports.agent_runtime import AgentExecutionPlan, AgentExecutionRequest


class _RecordingRunner:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}
        self.calls: list[dict[str, Any]] = []

    async def stream(self, *args: object, **kwargs: object) -> AsyncIterator[object]:
        self.kwargs = dict(kwargs)
        self.calls.append(self.kwargs)
        if False:
            yield object()


@pytest.mark.asyncio
async def test_runtime_merges_fixed_and_request_capabilities() -> None:
    runner = _RecordingRunner()
    repository = object()
    runtime = Nl2SqlApplicationRuntime(
        runner,
        {
            "repository": repository,
            "access_policy": "fixed-policy",
        },
    )
    request = AgentExecutionRequest(
        request_id="request-1",
        query="查询销售额",
        actor_id="user-1",
        workspace_id="workspace-1",
        conversation_id="chat-1",
        trace_id="trace-1",
        roles=("analyst",),
        allowed_data_sources=("official",),
        analysis_enabled=False,
        plan=AgentExecutionPlan(
            initial_state={"query": "查询销售额"},
            deadline_evidence={},
            timeout_seconds=1.0,
        ),
        capabilities={
            "access_policy": "request-policy",
            "workspace": "workspace-context",
            "metadata_call_gate": "request-must-not-override-gate",
        },
    )

    events = [event async for event in runtime.stream(request)]

    assert events == []
    graph_context = runner.kwargs["graph_context"]
    assert graph_context["repository"] is repository
    assert graph_context["access_policy"] == "request-policy"
    assert graph_context["workspace"] == "workspace-context"
    gate = graph_context["metadata_call_gate"]
    assert gate != "request-must-not-override-gate"
    assert callable(gate.run)


@pytest.mark.asyncio
async def test_runtime_creates_an_isolated_metadata_gate_per_execution() -> None:
    runner = _RecordingRunner()
    runtime = Nl2SqlApplicationRuntime(runner, {})
    request = AgentExecutionRequest(
        request_id="request-1",
        query="查询销售额",
        actor_id="user-1",
        workspace_id="workspace-1",
        conversation_id="chat-1",
        trace_id="trace-1",
        roles=("analyst",),
        allowed_data_sources=("official",),
        analysis_enabled=False,
        plan=AgentExecutionPlan(
            initial_state={"query": "查询销售额"},
            deadline_evidence={},
            timeout_seconds=1.0,
        ),
        capabilities={},
    )

    for _ in range(2):
        assert [event async for event in runtime.stream(request)] == []

    first = runner.calls[0]["graph_context"]["metadata_call_gate"]
    second = runner.calls[1]["graph_context"]["metadata_call_gate"]
    assert first is not second
