"""生产 NL2SQL profile 控制端口的出站适配器。"""

from __future__ import annotations

from app.agent_system.agents.nl2sql.services import planning_router, semantic_repair


class PlanningRuntimeProfileControl:
    def status(self) -> dict:
        return planning_router.runtime_profile_status()

    def rollback(self) -> dict:
        return planning_router.rollback_to_fallback()


class SemanticRepairRuntimeProfileControl:
    def status(self) -> dict:
        return semantic_repair.runtime_profile_status()

    def rollback(self) -> dict:
        return semantic_repair.rollback_to_fallback()
