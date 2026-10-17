"""Explicit LangGraph node for TODO-63 SQL-planning routing."""

from langgraph.runtime import Runtime

from app.agent_system.agents.nl2sql.context import DataAgentContext
from app.agent_system.agents.nl2sql.state import DataAgentState
from app.agent_system.agents.nl2sql.services.planning_router import decide_planning_route


async def decide_plan_route(
    state: DataAgentState,
    runtime: Runtime[DataAgentContext],
):
    """Emit one privacy-safe actual decision plus deterministic shadows."""

    runtime.stream_writer({"stage": "评估规划必要性"})
    decision = decide_planning_route(
        state,
        settings=runtime.context["settings"],
    )
    payload = decision.privacy_summary()
    runtime.stream_writer({"planning_route": payload})
    return {"planning_route": payload}
