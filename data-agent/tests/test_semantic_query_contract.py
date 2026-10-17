"""Shared query-contract tests for current deterministic difficulty cases."""
import pytest

from app.agent_system.agents.nl2sql.semantic_contract import (
    SEMANTIC_QUERY_CONTRACT_VERSION,
    audit_sql_against_query_contract,
    build_query_contract,
    compact_query_contract,
    render_query_contract,
)
from app.agent_system.agents.nl2sql.semantic_ir import build_semantic_ir, compile_semantic_ir
from app.agent_system.agents.nl2sql.services.model_sql_policy import model_semantic_summary
from tests.semantic_fixtures import closed_semantic_ir_cases, semantic_metadata


def _contract(question: str, metrics: list[dict] | None = None) -> dict:
    metadata = semantic_metadata()
    metric_infos = metrics or []
    ir = build_semantic_ir(question, metadata, metric_infos, [])
    return build_query_contract(question, metadata, metric_infos, ir)


def test_direct_registered_share_owns_formula_and_denominator_scope():
    metrics = [{
        "name": "收入构成率",
        "alias": ["成交贡献占比"],
        "relevant_columns": ["fact_order.order_amount"],
        "formula_rule": {
            "type": "share",
            "function": "SUM",
            "column": "fact_order.order_amount",
            "denominator_scope": (
                "all_groups_after_shared_filters_before_rank_or_limit"
            ),
            "zero_denominator": "null",
        },
    }]
    contract = _contract("按商品类别给出成交贡献占比", metrics)

    assert contract["schema_version"] == SEMANTIC_QUERY_CONTRACT_VERSION
    registered = [
        item for item in contract["metric_contracts"]
        if item["source"] == "registered_metric_direct"
    ]
    assert len(registered) == 1
    assert registered[0]["name"] == "收入构成率"
    assert registered[0]["denominator_scope"] == (
        "all_groups_after_shared_filters_before_rank_or_limit"
    )
    assert contract["operator_contract"]["ratio_required"] is True


def test_model_summary_carries_same_compact_query_contract():
    question = "各地区的销售总额"
    metadata = semantic_metadata()
    ir = build_semantic_ir(question, metadata, [], [])
    ir["query_contract"] = build_query_contract(question, metadata, [], ir)

    summary = model_semantic_summary(ir)

    assert summary["schema_version"] == "model-semantic-summary-v4"
    assert summary["query_contract"] == compact_query_contract(
        ir["query_contract"]
    )
    rendered = render_query_contract(ir["query_contract"])
    assert "最终可见粒度" in rendered
    assert "计算阶段必须按顺序" in rendered


def test_contract_contains_no_filter_literals_or_raw_question():
    contract = _contract("2021年只统计华中大区各品牌的订单笔数")
    encoded = str(compact_query_contract(contract))

    assert "华中" not in encoded
    assert "2021" not in encoded
    assert contract["raw_question_persisted"] is False
    assert contract["filter_values_persisted"] is False


def test_model_route_downgrades_self_certified_ir_to_guided_contract():
    """An abstaining compiler cannot make its own parsed shape authoritative."""

    metadata = semantic_metadata()
    ir = build_semantic_ir("按地区统计成交金额", metadata, [], [])
    # Simulate a parser that confidently missed the requested region grain.
    ir["dimensions"] = []
    ir["required_columns"] = {"fact_order": ["order_amount"]}
    ir["semantic_coverage"] = {
        **dict(ir.get("semantic_coverage") or {}),
        "coverage_score": 1.0,
        "unresolved_slots": [],
    }
    ir["model_sql_policy"] = {
        "decision": "model_sql",
        "use_model": True,
        "reason_codes": ["correctness_first_model_route"],
    }

    contract = build_query_contract(
        "按地区统计成交金额", metadata, [], ir,
    )

    assert contract["authority"] == "guided"
    assert contract["output_grain"] == []
    assert not any(
        item.get("source") == "semantic_ir"
        for item in contract["metric_contracts"]
    )


def test_model_route_wrong_ir_metric_cannot_regain_registered_formula_authority():
    """A registry match selected only by an untrusted IR stays advisory."""

    metadata = semantic_metadata()
    wrong_metrics = [{
        "name": "商品件数",
        "alias": ["购买件数"],
        "relevant_columns": ["fact_order.order_quantity"],
        "formula_rule": {
            "type": "aggregate",
            "function": "SUM",
            "column": "fact_order.order_quantity",
        },
    }]
    ir = build_semantic_ir("按地区统计成交金额", metadata, wrong_metrics, [])
    ir["metric"] = {"name": "商品件数"}
    ir["semantic_coverage"] = {
        **dict(ir.get("semantic_coverage") or {}),
        "detected_slots": ["operator:share"],
        "unresolved_slots": [],
    }
    ir["model_sql_policy"] = {
        "decision": "model_sql",
        "use_model": True,
        "reason_codes": ["correctness_first_model_route"],
    }

    contract = build_query_contract(
        "按地区统计成交金额", metadata, wrong_metrics, ir,
    )
    audit = audit_sql_against_query_contract(
        "SELECT r.region_name, SUM(f.order_amount) AS total_amount "
        "FROM fact_order f JOIN dim_region r ON r.region_id = f.region_id "
        "GROUP BY r.region_name",
        contract,
    )

    assert contract["authority"] == "guided"
    assert not any(
        item.get("source") == "registered_metric_ir"
        for item in contract["metric_contracts"]
    )
    assert contract["operations"] == []
    assert "query_contract.metric_formula_missing" not in {
        item["code"] for item in audit["violations"]
    }
    assert not {
        "query_contract.ratio_shape_missing",
        "query_contract.zero_denominator_guard_missing",
    } & {item["code"] for item in audit["violations"]}

    ir["query_contract"] = contract
    summary = model_semantic_summary(ir)
    assert summary["ir_hints_withheld"] is True
    assert summary["dimensions"] == []
    assert summary["measures"] == []
    assert summary["semantic_slots"] == {}
    assert summary["detected_slots"] == []
    assert summary["sql_blueprint"] == {}


@pytest.mark.parametrize(
    "case", closed_semantic_ir_cases(), ids=lambda item: item["id"],
)
def test_current_closed_cases_satisfy_shared_query_contract(case):
    metadata = semantic_metadata()
    ir = build_semantic_ir(case["question"], metadata, [], case.get("history") or [])
    contract = build_query_contract(case["question"], metadata, [], ir)
    audit = audit_sql_against_query_contract(compile_semantic_ir(ir), contract)

    assert audit["valid"], audit["violations"]


def test_count_star_rejects_unqualified_non_unique_derived_rows():
    contract = _contract("有过下单行为的客户有多少")
    audit = audit_sql_against_query_contract(
        "SELECT COUNT(*) AS customer_count FROM ("
        "SELECT order_id FROM fact_order) raw_events",
        contract,
    )

    assert audit["valid"] is False
    assert "query_contract.metric_formula_missing" in {
        item["code"] for item in audit["violations"]
    }
