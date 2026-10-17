"""把通用 GraphRunner 适配为 Application 的 NL2SQL 运行时端口。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping

from app.agent_system.contracts import InvocationContext, QueryInput
from app.agent_system.runtime import GraphRunner
from app.agent_system.governance.deadline import (
    compact_deadline_evidence,
    initialize_request_deadline,
    remaining_ms,
)
from app.application.ports.agent_runtime import (
    AgentEventKind,
    AgentExecutionPlan,
    AgentExecutionRequest,
    AgentStreamEvent,
)


class _MetadataCallGate:
    """把同一次执行中的 Meta 查询串行化，避免压垮请求级 Session。"""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def run(self, operation):
        async with self._lock:
            return await operation()


class Nl2SqlApplicationRuntime:
    """隐藏 NL2SQL State、Context 与 GraphRunner 的 Application 适配器。"""

    def __init__(
        self,
        runner: GraphRunner,
        capabilities: Mapping[str, object],
    ) -> None:
        self._runner = runner
        # 固定能力由 Bootstrap 一次性装配；请求只能补充请求级治理上下文。
        self._capabilities = dict(capabilities)

    def prepare(
        self,
        *,
        performance_started_at: float,
        initial_state: Mapping[str, object],
    ) -> AgentExecutionPlan:
        """冻结请求截止时间，并返回可审计的执行计划。"""

        state = dict(initial_state)
        settings = self._capabilities.get("settings")
        query_settings = getattr(settings, "query", None)
        deadline = initialize_request_deadline(
            performance_started_at,
            budget_ms=getattr(query_settings, "request_deadline_ms", None),
            phase_budgets_ms=getattr(
                query_settings,
                "request_phase_budgets_ms",
                None,
            ),
            execution_reserve_ms=getattr(
                query_settings,
                "request_execution_reserve_ms",
                None,
            ),
        )
        state["request_deadline"] = deadline
        return AgentExecutionPlan(
            initial_state=state,
            deadline_evidence=compact_deadline_evidence(deadline),
            timeout_seconds=max(0.001, remaining_ms(deadline) / 1000),
        )

    async def stream(
        self,
        request: AgentExecutionRequest,
    ) -> AsyncIterator[AgentStreamEvent]:
        """把应用合同转换为 Runtime 合同，并规范化输出事件。"""

        query_input = QueryInput(
            request_id=request.request_id,
            query=request.query,
            actor_id=request.actor_id,
            workspace_id=request.workspace_id,
            conversation_id=request.conversation_id,
            trace_id=request.trace_id,
            analysis_enabled=request.analysis_enabled,
        )
        invocation_context = InvocationContext(
            request_id=request.request_id,
            actor_id=request.actor_id,
            workspace_id=request.workspace_id,
            roles=request.roles,
            allowed_data_sources=request.allowed_data_sources,
            conversation_id=request.conversation_id,
            trace_id=request.trace_id,
            analysis_enabled=request.analysis_enabled,
        )
        graph_config: dict[str, object] = {
            "metadata": {"trace_id": request.trace_id},
        }
        if request.callbacks:
            graph_config["callbacks"] = list(request.callbacks)

        async for event in self._runner.stream(
            query_input,
            invocation_context,
            initial_state=request.plan.initial_state,
            graph_context={
                **self._capabilities,
                **request.capabilities,
                "metadata_call_gate": _MetadataCallGate(),
            },
            graph_config=graph_config,
            timeout_seconds=request.plan.timeout_seconds,
        ):
            error_message = event.error.message if event.error is not None else None
            yield AgentStreamEvent(
                kind=AgentEventKind(event.type.value),
                payload=event.payload,
                error_message=error_message,
            )


__all__ = ["Nl2SqlApplicationRuntime"]
