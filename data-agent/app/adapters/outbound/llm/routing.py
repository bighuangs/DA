"""NL2SQL 模型路由的 Application Port 适配器。"""

from __future__ import annotations

from collections.abc import Mapping

from app.agent_system.agents.nl2sql.llm import route_for_node
from app.adapters.outbound.agent.llm_provider import task_profile
from app.application.ports.llm_routing import LLMRouteCatalog


class Nl2SqlRouteCatalog(LLMRouteCatalog):
    """复用 NL2SQL 权威路由清单，不向 Application 暴露 Agent 模块。"""

    def route_for_node(self, node_id: str | None) -> str:
        return route_for_node(node_id)

    def profile_for_route(self, route: str | None) -> Mapping[str, object]:
        return task_profile(route)


__all__ = ["Nl2SqlRouteCatalog"]
