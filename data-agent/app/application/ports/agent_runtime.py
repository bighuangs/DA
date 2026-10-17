"""Application 驱动 Agent 时使用的稳定端口与数据合同。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class AgentEventKind(StrEnum):
    """Application 关心的最小流式事件分类。"""

    STARTED = "started"
    STATE = "state"
    MESSAGE = "message"
    RESULT = "result"
    ERROR = "error"
    SNAPSHOT = "snapshot"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class AgentStreamEvent:
    """隔离具体 Agent Runtime 合同后的应用层事件。"""

    kind: AgentEventKind
    payload: Mapping[str, object]
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class AgentExecutionPlan:
    """一次受截止时间治理的 Agent 执行计划。"""

    initial_state: Mapping[str, object]
    deadline_evidence: Mapping[str, object]
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class AgentExecutionRequest:
    """由 Application 交给 Agent 适配器的一次调用。"""

    request_id: str
    query: str
    actor_id: str
    workspace_id: str
    conversation_id: str
    trace_id: str
    roles: tuple[str, ...]
    allowed_data_sources: tuple[str, ...]
    analysis_enabled: bool
    plan: AgentExecutionPlan
    capabilities: Mapping[str, object]
    callbacks: tuple[object, ...] = ()


class QueryAgentRuntime(Protocol):
    """流式 NL2SQL 用例依赖的 Agent Runtime 端口。"""

    def prepare(
        self,
        *,
        performance_started_at: float,
        initial_state: Mapping[str, object],
    ) -> AgentExecutionPlan: ...

    def stream(
        self,
        request: AgentExecutionRequest,
    ) -> AsyncIterator[AgentStreamEvent]: ...


__all__ = [
    "AgentEventKind",
    "AgentExecutionPlan",
    "AgentExecutionRequest",
    "AgentStreamEvent",
    "QueryAgentRuntime",
]
