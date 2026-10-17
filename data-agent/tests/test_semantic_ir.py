import copy
from types import SimpleNamespace

import pytest
from app.adapters.outbound.agent.nl2sql_capabilities import build_nl2sql_settings
import sqlglot

from app.agent_system.agents.nl2sql.follow_up import semantic_state_from_sql
from app.agent_system.agents.nl2sql.nodes.filter_metric import filter_metric
from app.agent_system.agents.nl2sql.semantic_ir import (
    IR_VERSION,
    build_semantic_ir,
    compile_semantic_ir,
    restrict_table_infos_to_ir,
)
from app.domain.query.safety.query_guard import check_metric_formulas
from app.agent_system.agents.nl2sql.services.semantic_guard import check_semantic_ir

TABLES = [
    {
        "name": "dim_region", "role": "dim", "description": "地区维度表",
        "columns": [
            {"name": "region_id", "role": "primary_key", "description": "地区ID", "alias": ["地区ID"]},
            {"name": "region_name", "role": "dimension", "description": "大区名称", "alias": ["地区", "区域"]},
        ],
    },
    {
        "name": "dim_product", "role": "dim", "description": "商品维度表",
        "columns": [
            {"name": "product_id", "role": "primary_key", "description": "商品ID", "alias": ["商品ID"]},
            {"name": "brand", "role": "dimension", "description": "商品品牌名称", "alias": ["品牌", "品牌名称"]},
            {"name": "category", "role": "dimension", "description": "商品品类", "alias": ["品类"]},
        ],
    },
    {
        "name": "dim_customer", "role": "dim", "description": "客户维度表",
        "columns": [
            {"name": "customer_id", "role": "primary_key", "description": "客户ID", "alias": ["客户ID"]},
            {"name": "customer_name", "role": "dimension", "description": "客户名称", "alias": ["客户名称"]},
        ],
    },
    {
        "name": "fact_order", "role": "fact", "description": "订单事实表",
        "columns": [
            {"name": "order_id", "role": "primary_key", "description": "订单ID", "alias": ["订单ID"]},
            {"name": "region_id", "role": "foreign_key", "description": "地区外键", "alias": ["地区ID"]},
            {"name": "product_id", "role": "foreign_key", "description": "商品外键", "alias": ["商品ID"]},
            {"name": "customer_id", "role": "foreign_key", "description": "客户外键", "alias": ["客户ID"]},
            {"name": "order_amount", "role": "measure", "description": "订单金额", "alias": ["销售额", "收入"]},
            {"name": "order_quantity", "role": "measure", "description": "购买数量", "alias": ["销量"]},
        ],
    },
]

BIG_CUSTOMER = {
    "name": "大客户", "alias": ["重点客户"],
    "formula_rule": {
        "type": "ranking", "dimension": "dim_customer.customer_name",
        "function": "SUM", "column": "fact_order.order_amount",
        "direction": "DESC", "default_limit": 10, "result_alias": "total_amount",
    },
}
REPEAT_RATE = {
    "name": "复购率", "alias": ["回购率"],
    "formula_rule": {
        "type": "repeat_rate", "source_table": "fact_order",
        "entity_column": "fact_order.customer_id", "event_column": "fact_order.order_id",
        "repeat_threshold": 2, "result_alias": "repurchase_rate",
    },
}


def _normalized(sql: str) -> str:
    return sqlglot.parse_one(sql, read="mysql").sql(dialect="mysql")


def test_exact_metadata_aggregate_compiles_region_amount_without_llm():
    ir = build_semantic_ir("各地区的销售总额", TABLES, [])
    sql = compile_semantic_ir(ir)
    assert ir["schema_version"] == IR_VERSION
    assert ir["compiler"]["supported"] is True
    assert _normalized(sql) == _normalized(
        "SELECT r.region_name, SUM(f.order_amount) AS total_amount "
        "FROM fact_order f JOIN dim_region r ON f.region_id = r.region_id "
        "GROUP BY r.region_name ORDER BY total_amount DESC"
    )
    check_semantic_ir(sql, ir)


def test_brand_exact_term_wins_over_unrequested_category():
    ir = build_semantic_ir("各品牌的销售额排名", TABLES, [])
    sql = compile_semantic_ir(ir)
    assert ir["dimensions"][0] == {
        "table": "dim_product",
        "column": "brand",
        "logical": "brand",
        "source": "semantic_slot_v2",
    }
    assert "brand" in sql and "category" not in sql
    assert ir["limit"] is None


def test_sold_item_quantity_cue_compiles_sum_quantity_instead_of_order_count():
    ir = build_semantic_ir("各品牌共售出多少件商品", TABLES, [])
    sql = compile_semantic_ir(ir)
    assert ir["compiler"]["supported"] is True
    assert ir["measures"][0] == {
        "table": "fact_order",
        "column": "order_quantity",
        "logical": "order_quantity",
        "function": "SUM",
        "alias": "total_quantity",
        "distinct": False,
        "source": "semantic_slot_v2",
    }
    assert "SUM(f.order_quantity) AS total_quantity" in sql
    assert "COUNT(" not in sql
    check_semantic_ir(sql, ir)


def test_registered_ranking_metric_compiles_dimension_measure_and_limit():
    ir = build_semantic_ir("找出大客户前十名", TABLES, [BIG_CUSTOMER])
    sql = compile_semantic_ir(ir)
    assert ir["metric"] == {"name": "大客户", "rule_type": "ranking"}
    assert "customer_name" in sql
    assert "SUM(f.order_amount) AS total_amount" in sql
    assert sql.endswith("LIMIT 10")
    check_metric_formulas(sql, "找出大客户前十名", [BIG_CUSTOMER])


def test_explicit_order_count_overrides_big_customer_amount_ranking():
    ir = build_semantic_ir("大客户中谁的订单数量最多", TABLES, [BIG_CUSTOMER])
    sql = compile_semantic_ir(ir)
    assert ir["metric"] is None
    assert ir["measures"][0] == {
        "table": "fact_order",
        "column": "order_id",
        "logical": "order_id",
        "function": "COUNT",
        "alias": "order_count",
        "distinct": True,
        "source": "semantic_slot_v2",
    }
    assert ir["limit"] == 1
    assert "COUNT(DISTINCT f.order_id) AS order_count" in sql
    assert "SUM(f.order_amount)" not in sql
    check_semantic_ir(sql, ir)


@pytest.mark.asyncio
async def test_explicit_measure_override_removes_conflicting_metric_guard():
    ir = build_semantic_ir("大客户中谁的订单数量最多", TABLES, [BIG_CUSTOMER])
    output = await filter_metric(
        {"query": "大客户中谁的订单数量最多", "metric_infos": [BIG_CUSTOMER], "semantic_ir": ir},
        SimpleNamespace(
            stream_writer=lambda event: None,
            context={"settings": build_nl2sql_settings()},
        ),
    )
    assert output["metric_infos"] == []


def test_registered_repeat_rate_compiles_null_safe_cte_formula():
    ir = build_semantic_ir("复购率是多少", TABLES, [REPEAT_RATE])
    sql = compile_semantic_ir(ir)
    expected = (
        "WITH customer_orders AS (SELECT customer_id, COUNT(order_id) AS order_count "
        "FROM fact_order GROUP BY customer_id) "
        "SELECT SUM(CASE WHEN order_count >= 2 THEN 1 ELSE 0 END) / "
        "NULLIF(COUNT(*), 0) AS repurchase_rate FROM customer_orders"
    )
    assert _normalized(sql) == _normalized(expected)
    check_semantic_ir(sql, ir)
    check_metric_formulas(sql, "复购率是多少", [REPEAT_RATE])


def test_time_filter_falls_back_but_closed_history_compiles_without_phantom_filter():
    timed = build_semantic_ir("2025年各地区的销售总额", TABLES, [])
    previous_query = "销售额"
    previous_sql = "SELECT SUM(order_amount) FROM fact_order"
    follow_up = build_semantic_ir(
        "那再按地区看",
        TABLES,
        [],
        [{
            "query": previous_query,
            "standalone_question": previous_query,
            "sql": previous_sql,
            "semantic_state": semantic_state_from_sql(
                previous_query,
                previous_sql,
            ),
        }],
    )
    assert timed["compiler"]["supported"] is False
    assert follow_up["compiler"]["supported"] is True
    assert follow_up["semantic_coverage"]["follow_up_operations"] == [
        "append:dimension"
    ]
    follow_up_sql = compile_semantic_ir(follow_up)
    assert "GROUP BY r.region_name" in follow_up_sql
    assert "WHERE" not in follow_up_sql
    assert compile_semantic_ir(timed) is None


def test_complete_metric_question_still_compiles_when_session_has_history():
    ir = build_semantic_ir(
        "复购率是多少",
        TABLES,
        [REPEAT_RATE],
        [{"query": "各地区销售额", "sql": "SELECT region_name FROM dim_region"}],
    )
    assert ir["intent"] == "repeat_rate"
    assert ir["compiler"]["supported"] is True
    assert "NULLIF(COUNT(*), 0)" in compile_semantic_ir(ir)


def test_ir_restriction_removes_noisy_table_and_unrequested_dimension():
    ir = build_semantic_ir("各品牌的销售额排名", TABLES, [])
    restricted = restrict_table_infos_to_ir(TABLES, ir)
    names = {table["name"] for table in restricted}
    columns = {column["name"] for table in restricted for column in table["columns"]}
    assert names == {"dim_product", "fact_order"}
    assert columns == {"product_id", "brand", "order_amount"}


def test_schema_sync_dimension_role_on_fact_measure_uses_narrow_compatibility_rule():
    tables = copy.deepcopy(TABLES)
    order_amount = next(
        column for table in tables if table["name"] == "fact_order"
        for column in table["columns"] if column["name"] == "order_amount"
    )
    order_amount["role"] = "dimension"
    ir = build_semantic_ir("各地区的销售总额", tables, [])
    assert ir["compiler"]["supported"] is True
    assert "SUM(f.order_amount)" in compile_semantic_ir(ir)
