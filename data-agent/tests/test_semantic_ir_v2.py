import pytest

from app.agent_system.agents.nl2sql.follow_up import resolve_follow_up
from app.agent_system.agents.nl2sql.semantic_coverage import audit_compiled_semantic_coverage
from app.agent_system.agents.nl2sql.semantic_ir_v2 import (
    FIELD_SPECS,
    IR_VERSION_V2,
    build_semantic_ir_v2,
    compile_semantic_ir_v2,
    semantic_metadata_requirements,
    transaction_item_quantity_requested,
    validate_compiled_semantic_ir,
)
from tests.semantic_fixtures import difficulty_sql_cases


def _metadata() -> list[dict]:
    """Build a complete metadata catalog from the production logical contract."""
    tables: dict[str, dict] = {}
    for spec in FIELD_SPECS.values():
        table = tables.setdefault(
            spec["table"],
            {"name": spec["table"], "role": "fact" if spec["table"] == "fact_order" else "dim", "columns": []},
        )
        if any(column["name"] == spec["column"] for column in table["columns"]):
            continue
        table["columns"].append({
            "name": spec["column"],
            "description": " ".join(spec.get("terms") or ()),
            "alias": list(spec.get("terms") or ()),
        })
    return list(tables.values())


def _standalone(case: dict) -> str:
    if not case.get("history"):
        return case["question"]
    resolution = resolve_follow_up(case["question"], case["history"])
    assert resolution["needs_clarification"] is False
    return resolution["standalone_question"]


def _v2_supported_cases() -> list[dict]:
    metadata = _metadata()
    selected: list[dict] = []
    for case in difficulty_sql_cases():
        ir = build_semantic_ir_v2(_standalone(case), metadata, [])
        if ir is not None and ir["compiler"]["supported"]:
            selected.append(case)
    return selected


@pytest.mark.parametrize(
    "case", _v2_supported_cases(), ids=lambda item: item["id"],
)
def test_current_v2_supported_cases_have_closed_auditable_ir(case):
    ir = build_semantic_ir_v2(_standalone(case), _metadata(), [])
    assert ir is not None
    assert ir["schema_version"] == IR_VERSION_V2
    assert ir["status"] == "complete"
    assert ir["compiler"]["supported"] is True
    assert ir["compiler"]["mode"] == "deterministic_sqlglot_v2"

    sql = compile_semantic_ir_v2(ir)
    assert sql
    validate_compiled_semantic_ir(sql, ir)

    assert ir["required_tables"]
    assert ir["required_columns"]
    assert ir["semantic_family"]


@pytest.mark.parametrize(
    ("question", "family", "slot", "expected"),
    [
        ("2023年和2026年的销售额分别是多少", "period_compare", "comparison_periods", [2023, 2026]),
        ("2026年每月销售额及其环比增长率", "window_mom", "window_function", "lag"),
        ("各品牌销售额及其占总销售额的比例", "group_share", "denominator", "post_filter_total_sales"),
        ("每个地区销售额最高的三个商品", "top_per_group", "top_n", 3),
        ("按首次下单月份统计2024年的新客户数", "cohort_new", "observation_year", 2024),
        ("各会员级别的客户复购率", "repeat_rate", "repeat_threshold", 2),
        ("累计消费高于客户平均累计消费的客户", "above_average_customer", "outer_filter", "group_total_gt_average_group_total"),
        ("2026年哪些日期没有订单", "anti_join", "observation_year", 2026),
    ],
)
def test_semantic_families_generalize_across_entities_and_periods(
    question, family, slot, expected,
):
    ir = build_semantic_ir_v2(question, _metadata(), [])
    assert ir["semantic_family"] == family
    assert ir["semantic_slots"][slot] == expected
    sql = compile_semantic_ir_v2(ir)
    validate_compiled_semantic_ir(sql, ir)


def test_missing_required_field_stays_partial_instead_of_emitting_simplified_sql():
    metadata = _metadata()
    date = next(table for table in metadata if table["name"] == "dim_date")
    date["columns"] = [column for column in date["columns"] if column["name"] != "month"]

    ir = build_semantic_ir_v2(
        "2025年每月销售额及其环比增长率",
        metadata,
        [],
    )
    assert ir["status"] == "partial"
    assert ir["compiler"]["supported"] is False
    assert "month" in ir["compiler"]["missing_slots"]
    assert compile_semantic_ir_v2(ir) is None


def test_antijoin_contract_includes_nullable_fact_event():
    ir = build_semantic_ir_v2("从未售出的商品有哪些", _metadata(), [])
    assert "order_id" in ir["required_columns"]["fact_order"]
    sql = compile_semantic_ir_v2(ir)
    assert "f.order_id IS NULL" in sql
    validate_compiled_semantic_ir(sql, ir)


def test_explicit_member_level_value_requests_member_level_metadata():
    requirements = semantic_metadata_requirements(
        "黄金会员2025年每个月贡献了多少销售额",
    )

    assert "member_level" in requirements["dim_customer"]


def test_repeat_rate_keeps_explicit_gender_and_time_metadata():
    requirements = semantic_metadata_requirements(
        "2025年男女客户各自的复购率",
    )

    assert "gender" in requirements["dim_customer"]
    assert {"date_id", "year"}.issubset(requirements["dim_date"])
    assert {"customer_id", "order_id"}.issubset(
        requirements["fact_order"]
    )


def test_monthly_average_and_gender_shorthand_close_metadata_before_rag():
    monthly = semantic_metadata_requirements("各类别的月均成交额")
    gender = semantic_metadata_requirements(
        "2023年西北地区女顾客的成交额按省份统计"
    )

    assert {"date_id", "year", "month"}.issubset(monthly["dim_date"])
    assert "gender" in gender["dim_customer"]


@pytest.mark.parametrize("amount_term", ["成交金额", "单笔金额"])
def test_governed_amount_aliases_drive_family_measure_and_recall(amount_term):
    question = f"按客户性别汇总{amount_term}"
    ir = build_semantic_ir_v2(question, _metadata(), [])
    requirements = semantic_metadata_requirements(
        question,
        table_infos=_metadata(),
    )

    assert ir["semantic_family"] == "aggregate"
    assert [item["logical"] for item in ir["measures"]] == ["order_amount"]
    assert "order_amount" in requirements["fact_order"]


@pytest.mark.parametrize(
    ("question", "operator", "value"),
    [
        ("2024年购买件数不低于4件时各产品的单笔金额合计", "gte", 4),
        ("2023年件数达到2件时每个商品的成交金额", "gte", 2),
    ],
)
def test_filter_only_quantity_keeps_product_grain_and_amount_measure(
    question, operator, value,
):
    ir = build_semantic_ir_v2(question, _metadata(), [])
    sql = compile_semantic_ir_v2(ir)

    assert [item["logical"] for item in ir["dimensions"]] == ["product_name"]
    assert [item["logical"] for item in ir["measures"]] == ["order_amount"]
    assert {
        (item["logical"], item["operator"], item["value"])
        for item in ir["filters"]
    } >= {("order_quantity", operator, value)}
    assert {"product_name", "product_id"}.issubset(
        ir["required_columns"]["dim_product"]
    )
    assert "SUM(f.order_amount)" in sql
    assert "f.order_quantity" in sql


def test_quantity_without_comparison_remains_the_requested_measure():
    ir = build_semantic_ir_v2("各商品的购买件数", _metadata(), [])

    assert [item["logical"] for item in ir["measures"]] == ["order_quantity"]
    assert not any(item["logical"] == "order_quantity" for item in ir["filters"])


@pytest.mark.parametrize(
    "question",
    [
        "华东地区累计成交了多少件商品",
        "西南地区一共销售了几件产品",
        "各品牌购买累计多少件商品",
        "各品类共售出了几件",
        "按会员等级累计购买了几件",
    ],
)
def test_transaction_item_quantity_closes_recall_ir_and_coverage(question):
    metadata = _metadata()
    requirements = semantic_metadata_requirements(
        question,
        table_infos=metadata,
    )
    ir = build_semantic_ir_v2(question, metadata, [])
    sql = compile_semantic_ir_v2(ir)
    coverage = audit_compiled_semantic_coverage(question, metadata, sql)

    assert transaction_item_quantity_requested(question, metadata)
    assert "order_quantity" in requirements["fact_order"]
    assert [item["logical"] for item in ir["measures"]] == ["order_quantity"]
    assert "SUM(f.order_quantity)" in sql
    assert coverage["coverage_score"] == 1.0
    assert coverage["unresolved_slots"] == []


@pytest.mark.parametrize(
    "question",
    [
        "商品目录有多少商品",
        "商品目录中共有几件商品",
        "产品目录累计有多少件产品",
        "按品牌统计商品目录中的商品件数",
    ],
)
def test_catalog_cardinality_does_not_become_transaction_quantity(question):
    metadata = _metadata()
    requirements = semantic_metadata_requirements(
        question,
        table_infos=metadata,
    )
    ir = build_semantic_ir_v2(question, metadata, [])

    assert not transaction_item_quantity_requested(question, metadata)
    assert "order_quantity" not in requirements.get("fact_order", set())
    assert not ir or not any(
        item.get("logical") == "order_quantity"
        for item in ir.get("measures") or []
    )


@pytest.mark.parametrize(
    "question",
    [
        "按性别分别统计金额达到800元和低于800元的订单量",
        "各会员等级中订单金额不低于1200元及小于1200元的订单笔数",
    ],
)
def test_conditional_count_family_generalizes_across_group_and_wording(question):
    ir = build_semantic_ir_v2(question, _metadata(), [])
    sql = compile_semantic_ir_v2(ir)

    assert ir["semantic_family"] == "conditional_aggregate"
    assert len(ir["semantic_slots"]["conditional_predicates"]) == 2
    assert sql.count("CASE WHEN") == 2
    assert "COUNT(DISTINCT" in sql


def test_single_numeric_condition_is_a_row_filter_not_conditional_buckets():
    ir = build_semantic_ir_v2(
        "按性别统计金额至少1000元的订单数",
        _metadata(),
        [],
    )
    sql = compile_semantic_ir_v2(ir)

    assert ir["semantic_family"] == "aggregate"
    assert "f.order_amount >= 1000" in sql
    assert "CASE WHEN" not in sql


def test_adjacent_numeric_fields_do_not_steal_each_others_predicates():
    ir = build_semantic_ir_v2(
        "各商品中订单金额超过100元且购买件数至少3件的订单数",
        _metadata(),
        [],
    )

    assert {
        (item["logical"], item["operator"], item["value"])
        for item in ir["filters"]
    } >= {
        ("order_amount", "gt", 100),
        ("order_quantity", "gte", 3),
    }


@pytest.mark.parametrize(
    ("question", "grain", "threshold"),
    [
        ("共有多少品牌的累计单笔金额不低于20万元", "brand", 200_000),
        ("有多少省份的合计成交金额超过300万元", "province", 3_000_000),
    ],
)
def test_qualifying_group_count_has_scalar_outer_contract(
    question, grain, threshold,
):
    ir = build_semantic_ir_v2(question, _metadata(), [])
    sql = compile_semantic_ir_v2(ir)

    assert ir["semantic_family"] == "qualifying_group_count"
    assert ir["dimensions"] == []
    assert ir["semantic_slots"]["aggregate_grain"] == grain
    assert ir["semantic_slots"]["having_threshold"] == threshold
    assert ir["semantic_slots"]["output_shape"] == "scalar"
    assert sql.startswith("SELECT COUNT(*)")
    assert " HAVING SUM(" in sql


def test_qualified_group_list_is_not_misclassified_as_scalar_count():
    ir = build_semantic_ir_v2(
        "列出各省份合计成交金额超过300万元的结果",
        _metadata(),
        [],
    )
    sql = compile_semantic_ir_v2(ir)

    assert ir["semantic_family"] == "having_aggregate"
    assert [item["logical"] for item in ir["dimensions"]] == ["province"]
    assert not sql.startswith("SELECT COUNT(*)")


@pytest.mark.parametrize(
    ("question", "expected_dimension", "expected_function"),
    [
        ("2028年各产品类别的平均每笔订单金额", "category", "AVG"),
        ("2027年按商品品类统计平均单笔成交金额", "category", "AVG"),
    ],
)
def test_compound_category_and_per_order_average_are_semantic_roles(
    question, expected_dimension, expected_function,
):
    """Compound entity nouns must not become product names or literal filters."""

    ir = build_semantic_ir_v2(question, _metadata(), [])
    sql = compile_semantic_ir_v2(ir)

    assert [item["logical"] for item in ir["dimensions"]] == [expected_dimension]
    assert [item["function"] for item in ir["measures"]] == [expected_function]
    assert not any(item["logical"] == "category" for item in ir["filters"])
    assert "d.year" in sql
    assert "AVG(f.order_amount)" in sql


@pytest.mark.parametrize(
    "question",
    [
        "有多少种品牌的不同购买客户数至少达到800名",
        "共有多少个商品类别的购买客户数不少于1200名",
    ],
)
def test_distinct_entity_count_threshold_uses_inner_having_and_outer_scalar_count(
    question,
):
    ir = build_semantic_ir_v2(question, _metadata(), [])
    sql = compile_semantic_ir_v2(ir)

    assert ir["semantic_family"] == "qualifying_group_count"
    assert ir["semantic_slots"]["qualifying_measure"] == "customer_id"
    assert ir["semantic_slots"]["qualifying_aggregate"] == "COUNT_DISTINCT"
    assert sql.startswith("SELECT COUNT(*)")
    assert "HAVING COUNT(DISTINCT f.customer_id)" in sql


def test_grouped_distinct_dimension_count_keeps_dimension_ownership():
    ir = build_semantic_ir_v2(
        "按销售区域统计产生过交易的省份数量",
        _metadata(),
        [],
    )
    sql = compile_semantic_ir_v2(ir)

    assert [item["logical"] for item in ir["dimensions"]] == ["region_name"]
    assert ir["measures"] == [{
        "table": "dim_region",
        "column": "province",
        "logical": "province",
        "function": "COUNT",
        "alias": "province_count",
        "distinct": True,
        "source": "semantic_slot_v2",
    }]
    assert "COUNT(DISTINCT r.province)" in sql


def test_three_period_month_compare_preserves_textual_period_order_and_wide_shape():
    ir = build_semantic_ir_v2(
        "按月份分别给出2022、2024和2027年的订单金额",
        _metadata(),
        [],
    )
    sql = compile_semantic_ir_v2(ir)

    assert ir["semantic_family"] == "period_compare"
    assert ir["semantic_slots"]["comparison_periods"] == [2022, 2024, 2027]
    assert [item["logical"] for item in ir["dimensions"]] == ["month"]
    assert sql.count("CASE WHEN") == 3
    assert "sales_2022" in sql and "sales_2027" in sql


@pytest.mark.parametrize(
    "question",
    [
        "按会员级别计算每位客户累计订单金额的平均值",
        "按客户性别统计客户合计成交额的均值",
    ],
)
def test_average_of_entity_totals_has_explicit_inner_and_outer_grains(question):
    ir = build_semantic_ir_v2(question, _metadata(), [])
    sql = compile_semantic_ir_v2(ir)

    assert ir["semantic_family"] == "entity_aggregate_average"
    assert ir["semantic_slots"]["aggregate_grain"] == "customer_id"
    assert ir["semantic_slots"]["inner_aggregate"] == "SUM"
    assert ir["semantic_slots"]["outer_aggregate"] == "AVG"
    assert "GROUP BY c.member_level, f.customer_id" in sql or (
        "GROUP BY c.gender, f.customer_id" in sql
    )
    assert "AVG(entity_total)" in sql


def test_partitioned_percentile_closes_recall_but_stays_on_model_route():
    question = "2024年计算每个产品在所属品类内按单笔金额从高到低的百分位排名"
    ir = build_semantic_ir_v2(question, _metadata(), [])
    requirements = semantic_metadata_requirements(
        question,
        table_infos=_metadata(),
    )

    assert ir["semantic_family"] == "distribution_rank"
    assert ir["semantic_slots"]["partition_keys"] == ["category"]
    assert ir["semantic_slots"]["aggregate_grain"] == "product_name"
    assert ir["compiler"]["supported"] is False
    assert {"category", "product_name"}.issubset(requirements["dim_product"])
    assert "order_amount" in requirements["fact_order"]


@pytest.mark.parametrize(
    ("question", "bad_sql", "missing_prefixes"),
    [
        (
            "按性别分别统计金额达到900元和低于900元的订单数",
            "SELECT c.gender, COUNT(DISTINCT f.order_id) FROM fact_order f "
            "JOIN dim_customer c ON f.customer_id=c.customer_id GROUP BY c.gender",
            ("comparison:row:order_amount", "operator:conditional_aggregate"),
        ),
        (
            "2022年件数不少于5件时各产品的成交金额",
            "SELECT SUM(f.order_quantity) FROM fact_order f JOIN dim_date d "
            "ON f.date_id=d.date_id WHERE d.year=2022",
            ("dimension:product_name", "measure:sum:order_amount", "comparison:row:order_quantity"),
        ),
        (
            "有多少省份的累计成交金额超过400万元",
            "SELECT r.province FROM fact_order f JOIN dim_region r "
            "ON f.region_id=r.region_id GROUP BY r.province "
            "HAVING SUM(f.order_amount)>4000000",
            ("output:scalar", "operator:count_qualified_groups"),
        ),
        (
            "2024年每个商品在所属类别内按成交金额从低到高的百分位排名",
            "WITH s AS (SELECT p.category, p.product_name, SUM(f.order_amount) a "
            "FROM fact_order f JOIN dim_product p ON f.product_id=p.product_id "
            "JOIN dim_date d ON f.date_id=d.date_id WHERE d.year=2024 "
            "GROUP BY p.category,p.product_name) SELECT category,product_name,a,"
            "PERCENT_RANK() OVER (ORDER BY a) pr FROM s",
            ("operator:partitioned_rank",),
        ),
    ],
)
def test_independent_coverage_rejects_semantically_incomplete_sql(
    question, bad_sql, missing_prefixes,
):
    coverage = audit_compiled_semantic_coverage(question, _metadata(), bad_sql)

    for prefix in missing_prefixes:
        assert any(
            item.startswith(prefix) for item in coverage["unresolved_slots"]
        ), (prefix, coverage)
    assert coverage["coverage_score"] < 1.0
