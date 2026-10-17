from app.agent_system.agents.nl2sql.nodes.filter_table import stabilize_table_selection
from app.agent_system.agents.nl2sql.relation_graph import (
    RELATION_CLOSURE_VERSION,
    build_relation_edges,
    close_relation_requirements,
)


def _snowflake_metadata() -> list[dict]:
    return [
        {
            "name": "fact_sales", "role": "fact", "columns": [
                {"name": "sale_id", "role": "primary_key"},
                {"name": "product_id", "role": "foreign_key"},
                {"name": "store_id", "role": "foreign_key"},
                {"name": "amount", "role": "measure", "alias": ["成交金额"]},
            ],
        },
        {
            "name": "dim_product", "role": "dim", "columns": [
                {"name": "product_id", "role": "primary_key"},
                {"name": "category_id", "role": "foreign_key"},
                {"name": "product_name", "role": "dimension", "is_display": True},
            ],
        },
        {
            "name": "dim_category", "role": "dim", "columns": [
                {"name": "category_id", "role": "primary_key"},
                {"name": "category_name", "role": "dimension", "alias": ["品类"]},
            ],
        },
        {
            "name": "dim_store", "role": "dim", "columns": [
                {"name": "store_id", "role": "primary_key"},
                {"name": "region_id", "role": "foreign_key"},
            ],
        },
        {
            "name": "dim_region", "role": "dim", "columns": [
                {"name": "region_id", "role": "primary_key"},
                {"name": "region_name", "role": "dimension", "alias": ["大区"]},
            ],
        },
    ]


def test_relation_closure_adds_unique_four_hop_snowflake_bridge():
    metadata = _snowflake_metadata()
    closure = close_relation_requirements(
        metadata,
        {
            "dim_category": {"category_name"},
            "dim_region": {"region_name"},
        },
    )

    assert closure["schema_version"] == RELATION_CLOSURE_VERSION
    assert closure["status"] == "closed"
    assert closure["unresolved_tables"] == []
    assert set(closure["added_tables"]) == {
        "fact_sales", "dim_product", "dim_store",
    }
    assert len(closure["joins"]) == 4
    assert {
        (item["left_table"], item["left_column"], item["right_table"], item["right_column"])
        for item in closure["joins"]
    } == {
        ("dim_product", "category_id", "dim_category", "category_id"),
        ("fact_sales", "product_id", "dim_product", "product_id"),
        ("fact_sales", "store_id", "dim_store", "store_id"),
        ("dim_store", "region_id", "dim_region", "region_id"),
    }


def test_filter_selection_restores_bridge_tables_and_both_join_keys():
    selected = stabilize_table_selection(
        "按品类和大区查看成交金额",
        _snowflake_metadata(),
        {
            "dim_category": ["category_name"],
            "dim_region": ["region_name"],
            "fact_sales": ["amount"],
        },
    )
    mapping = {
        table["name"]: {column["name"] for column in table["columns"]}
        for table in selected
    }

    assert set(mapping) == {
        "fact_sales", "dim_product", "dim_category", "dim_store", "dim_region",
    }
    assert {"product_id", "store_id", "amount"}.issubset(mapping["fact_sales"])
    assert {"product_id", "category_id"}.issubset(mapping["dim_product"])
    assert {"store_id", "region_id"}.issubset(mapping["dim_store"])


def test_equal_strength_ambiguous_bridge_abstains():
    metadata = [
        {
            "name": "dim_left", "role": "dim", "columns": [
                {"name": "left_id", "role": "primary_key"},
                {"name": "left_name", "role": "dimension"},
            ],
        },
        {
            "name": "dim_right", "role": "dim", "columns": [
                {"name": "right_id", "role": "primary_key"},
                {"name": "right_name", "role": "dimension"},
            ],
        },
        *[
            {
                "name": fact, "role": "fact", "columns": [
                    {"name": "left_id", "role": "foreign_key"},
                    {"name": "right_id", "role": "foreign_key"},
                ],
            }
            for fact in ("fact_a", "fact_b")
        ],
    ]

    assert len(build_relation_edges(metadata)) == 4
    closure = close_relation_requirements(
        metadata,
        {"dim_left": {"left_name"}, "dim_right": {"right_name"}},
    )

    assert closure["status"] == "partial"
    assert closure["ambiguous_paths"] == [{
        "target_table": "dim_right",
        "candidate_path_count": 2,
        "hop_count": 2,
    }]
    assert closure["unresolved_tables"] == ["dim_right"]
