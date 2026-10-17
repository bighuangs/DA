"""LLM Invocation 内层合同兼容性测试。"""

from __future__ import annotations

import pytest

from app.agent_system.agents.nl2sql.llm import (
    NODE_TASK_ROUTES,
    route_for_node as agent_route_for_node,
)
from app.adapters.outbound.agent.llm_provider import task_profile as provider_task_profile
from app.agent_system.governance.deadline import (
    REQUEST_DEADLINE_VERSION as LEGACY_DEADLINE_VERSION,
)
from app.agent_system.runtime.llm_attempts import (
    make_llm_attempt as legacy_make_llm_attempt,
)
from app.foundation.deadlines import REQUEST_DEADLINE_VERSION
from app.foundation.llm_evidence import make_llm_attempt
from app.infrastructure.clients.llm.invocation import (
    invoke_llm,
    route_for_node,
    task_profile,
)


class _Runnable:
    async def ainvoke(self, payload: object, config: object = None) -> str:
        del payload, config
        return "SELECT 1"


class _RouteCatalog:
    def route_for_node(self, node_id: str | None) -> str:
        assert node_id == "custom_node"
        return "custom_route"

    def profile_for_route(self, route: str | None) -> dict:
        assert route == "custom_route"
        return {
            "model_name": "custom-model",
            "timeout_seconds": 0.2,
            "max_retries": 0,
            "max_concurrency": 1,
        }


def test_foundation_attempt_matches_legacy_runtime_for_explicit_identity() -> None:
    values = {
        "node_id": "generate_sql",
        "route": "sql",
        "model": "test-model",
        "status": "timeout",
        "latency_ms": 17,
        "timeout_budget_ms": 20,
        "token_known": False,
        "error_type": "TimeoutError",
        "reason_code": "llm.timeout",
        "retry_index": 1,
        "callback_observed": False,
        "core_path": True,
        "attempt_id": "fixed-attempt",
        "source": "unified_boundary",
        "deadline_before_ms": 20,
        "deadline_after_ms": 3,
        "queue_wait_ms": 2,
        "provider_ms": 15,
        "completion_stage": "provider",
        "outcome_class": "timeout",
        "recovery": True,
        "retryable": False,
        "recovery_reserve_ms": 0,
        "physical_attempt_budget_ms": 20,
    }
    assert make_llm_attempt(**values) == legacy_make_llm_attempt(**values)
    assert REQUEST_DEADLINE_VERSION == LEGACY_DEADLINE_VERSION


def test_default_route_profiles_match_agent_execution_fields() -> None:
    nodes = {None, "unknown", *NODE_TASK_ROUTES}
    routes = {route_for_node(node) for node in nodes}
    assert all(route_for_node(node) == agent_route_for_node(node) for node in nodes)
    for route in routes:
        expected = provider_task_profile(route)
        actual = task_profile(route)
        assert actual == {
            key: expected[key]
            for key in (
                "task",
                "model_name",
                "timeout_seconds",
                "max_retries",
                "max_concurrency",
            )
        }


@pytest.mark.asyncio
async def test_invocation_accepts_injected_route_catalog() -> None:
    result = await invoke_llm(
        _Runnable(),
        {},
        node_id="custom_node",
        route_catalog=_RouteCatalog(),
    )
    assert result.attempts[0]["route"] == "custom_route"
    assert result.attempts[0]["model"] == "custom-model"
