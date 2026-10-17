from types import SimpleNamespace
import json

import pytest

from app.adapters.outbound.agent.nl2sql_capabilities import build_nl2sql_settings
from app.agent_system.agents.nl2sql.nodes.plan_sql import build_structured_plan, plan_sql
from app.agent_system.agents.nl2sql.settings import override_settings
from app.agent_system.governance.deadline import initialize_request_deadline


def _state() -> dict:
    return {
        "query": "按地区汇总金额",
        "table_infos": [{"name": "fact_order"}, {"name": "dim_region"}],
        "semantic_ir": {
            "semantic_family": "aggregate",
            "required_tables": ["fact_order", "dim_region"],
            "semantic_coverage": {
                "detected_slots": [
                    "dimension:region_name",
                    "measure:sum:order_amount:all",
                ],
                "unresolved_slots": [],
            },
        },
        "request_deadline": initialize_request_deadline(),
    }


def test_structured_plan_uses_semantic_contract_without_query_text():
    plan = build_structured_plan(_state())
    assert "structured-plan-v1" in plan
    assert "fact_order" in plan
    assert "按地区汇总金额" not in plan


def test_model_route_structured_plan_is_guided_and_hides_parser_slots():
    state = _state()
    state["semantic_ir"] = {
        **state["semantic_ir"],
        "query_contract": {"authority": "guided"},
        "model_sql_policy": {"use_model": True},
    }

    plan = json.loads(build_structured_plan(state))

    assert plan["authority"] == "guided"
    assert plan["required_tables"] == ["fact_order", "dim_region"]
    assert plan["detected_slots"] == []
    assert plan["unresolved_slots"] == []
    assert any(
        "用户问题是最终语义权威" in item
        for item in plan["constraints"]
    )


def test_model_route_rebuilds_a_stale_closed_contract_before_planning():
    state = _state()
    state["metric_infos"] = []
    state["semantic_ir"] = {
        **state["semantic_ir"],
        "query_contract": {
            "authority": "closed",
            "output_grain": [{"logical": "wrong_dimension"}],
            "operations": ["share"],
        },
        "model_sql_policy": {"use_model": True},
    }

    plan = json.loads(build_structured_plan(state))

    assert plan["authority"] == "guided"
    assert plan["query_contract"]["authority"] == "guided"
    assert plan["query_contract"]["output_grain"] == []
    assert plan["query_contract"]["operations"] == []
    assert plan["detected_slots"] == []


def test_deterministic_ablation_preserves_a_closed_structured_plan():
    state = _state()
    state["semantic_ir"] = {
        **state["semantic_ir"],
        "query_contract": {
            "authority": "closed",
            "output_grain": [{"logical": "region_name"}],
            "operations": [],
        },
        "model_sql_policy": {"use_model": False},
    }

    plan = json.loads(build_structured_plan(state))

    assert plan["authority"] == "closed"
    assert plan["query_contract"]["authority"] == "closed"
    assert plan["query_contract"]["output_grain"] == [
        {"logical": "region_name"}
    ]


@pytest.mark.asyncio
async def test_optional_planning_model_is_disabled_by_default():
    events = []
    settings = override_settings(
        build_nl2sql_settings(),
        "query",
        planning_llm_enabled=False,
    )
    result = await plan_sql(
        _state(),
        SimpleNamespace(
            stream_writer=events.append,
            context={"settings": settings},
        ),
    )
    assert result["planning_evidence"]["physical_llm_call"] is False
    assert result["planning_evidence"]["source"] == (
        "deterministic_semantic_contract"
    )
    assert result["plan"]
    assert not result.get("llm_attempts")
