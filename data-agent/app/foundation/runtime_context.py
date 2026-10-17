"""跨层共享的请求运行上下文变量。"""

from __future__ import annotations

from contextvars import ContextVar


current_node_id: ContextVar[str | None] = ContextVar(
    "data_agent_node_id",
    default=None,
)

__all__ = ["current_node_id"]
