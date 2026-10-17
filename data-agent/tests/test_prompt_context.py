import yaml

from app.agent_system.agents.nl2sql.services.prompt_context import (
    build_candidate_prompt_payload,
    build_model_prompt_payload,
)


def test_model_prompt_payload_removes_examples_and_limits_history_training():
    state = {
        "table_infos": [{
            "name": "fact_order", "description": "事实订单表" * 100,
            "columns": [{
                "name": "order_amount", "type": "decimal", "role": "measure",
                "description": "订单金额", "examples": [100, 200, 300], "unused": "noise" * 100,
            }],
        }],
        "metric_infos": [{"name": "销售额", "formula_rule": {"function": "SUM"}, "unused": "noise"}],
        "recall_training": [
            {"question": f"q{i}", "sql": "SELECT 1", "score": .9, "embedding": [0] * 50}
            for i in range(5)
        ],
        "chat_history": [{"query": f"q{i}", "sql": "SELECT 1", "result": [{"secret": i}]} for i in range(5)],
        "context_budget": {"after_tokens": {"total": 1000}},
    }
    payload, budget = build_model_prompt_payload(state, "generate_sql")
    assert "examples" not in payload["table_infos"]
    assert "unused" not in payload["table_infos"]
    assert "secret" not in payload["chat_history"]
    evidence = budget["model_prompts"]["generate_sql"]
    assert evidence["training_items"] == 2
    assert evidence["history_items"] == 2
    assert evidence["after_tokens"] < evidence["before_tokens"]


def test_model_prompt_preserves_only_values_repeated_by_the_user():
    state = {
        "query": "比较华东和华南两个大区的成交金额",
        "table_infos": [{
            "name": "dim_region",
            "columns": [{
                "name": "region_name",
                "type": "varchar",
                "role": "dimension",
                "examples": ["华东", "华南", "华北"],
            }],
        }],
    }

    payload, _budget = build_model_prompt_payload(state, "generate_sql")
    tables = yaml.safe_load(payload["table_infos"])

    assert tables[0]["columns"][0]["examples"] == ["华东", "华南"]
    assert "华北" not in payload["table_infos"]


def test_model_prompt_preserves_privacy_safe_value_match_shape():
    state = {
        "query": "统计测试品牌的不同品牌数",
        "table_infos": [{
            "name": "dim_product",
            "columns": [{
                "name": "brand",
                "role": "dimension",
                "examples": ["测试品牌甲", "测试品牌乙"],
                "value_match_hints": [{
                    "query_term": "测试品牌",
                    "operator": "prefix",
                    "matched_distinct_lower_bound": 2,
                    "stored_values_exposed": False,
                }],
            }],
        }],
    }
    payload, _ = build_model_prompt_payload(state, "generate_sql")
    tables = yaml.safe_load(payload["table_infos"])
    column = tables[0]["columns"][0]
    assert column["value_match_hints"][0]["operator"] == "prefix"
    assert "examples" not in column
    assert "测试品牌甲" not in payload["table_infos"]


def test_semantic_review_sees_all_controlled_tables_and_columns_not_only_primary_sql_or_ir():
    state = {
        "query": "按会员等级统计客户数和销售额",
        "sql": "SELECT SUM(order_amount) AS total_amount FROM fact_order",
        "table_infos": [
            {
                "name": "fact_order",
                "columns": [
                    {"name": "customer_id", "role": "foreign_key"},
                    {"name": "order_amount", "role": "measure"},
                ],
            },
            {
                "name": "dim_customer",
                "columns": [
                    {"name": "customer_id", "role": "primary_key"},
                    {"name": "member_level", "role": "dimension"},
                    {"name": "customer_name", "role": "dimension", "is_display": True},
                ],
            },
        ],
        # Guided IR 故意只覆盖 A 中的金额列；复核上下文不得被它限定。
        "semantic_ir": {
            "status": "partial",
            "measures": [{"table": "fact_order", "column": "order_amount"}],
        },
        "metric_infos": [
            {"name": "客户数", "formula": "COUNT(DISTINCT fact_order.customer_id)"},
            {"name": "销售额", "formula": "SUM(fact_order.order_amount)"},
        ],
        "db_info": {"dialect": "mysql"},
    }

    payload, budget = build_candidate_prompt_payload(state, [])
    tables = yaml.safe_load(payload["table_infos"])
    metrics = yaml.safe_load(payload["metric_infos"])
    semantic_ir = yaml.safe_load(payload["semantic_ir"])

    assert [table["name"] for table in tables] == ["fact_order", "dim_customer"]
    assert {column["name"] for column in tables[1]["columns"]} == {
        "customer_id", "member_level", "customer_name",
    }
    assert {metric["name"] for metric in metrics} == {"客户数", "销售额"}
    assert semantic_ir["ir_hints_withheld"] is True
    assert semantic_ir["dimensions"] == []
    assert semantic_ir["measures"] == []
    evidence = budget["model_prompts"]["judge_candidate"]
    assert evidence["profile"] == "candidate-semantic-review"
    assert evidence["tables"] == 2
    assert evidence["ir_hints_withheld"] is True
