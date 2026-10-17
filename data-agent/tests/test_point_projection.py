from types import SimpleNamespace
import uuid

from app.infrastructure.search.projection import (
    build_column_projection_records,
    build_metric_projection_records,
    projection_manifest,
)
from app.infrastructure.search.index_version import column_index_version
from evals.legacy.point_projection import select_strategy


def _manifest(*rows):
    return {
        "version": "approved-alias-v1",
        "approved_aliases": list(rows),
    }


def _table_rows():
    table = SimpleNamespace(id="fact_order", name="fact_order")
    amount = SimpleNamespace(
        id="fact_order.order_amount",
        name="order_amount",
        type="decimal",
        role="measure",
        examples=["Alice", "100.00"],
        description="订单金额",
        alias=["销售额", "收入"],
        table_id="fact_order",
        is_display=False,
    )
    quantity = SimpleNamespace(
        id="fact_order.order_quantity",
        name="order_quantity",
        type="int",
        role="measure",
        examples=[1, 2],
        description="购买数量",
        alias=["销量"],
        table_id="fact_order",
        is_display=False,
    )
    return [(table, [amount, quantity])]


def _version(rows, strategy):
    columns = [column for _, items in rows for column in items]
    return column_index_version(
        columns,
        point_strategy=strategy,
        alias_manifest_hash="sha256:test",
    )


def test_three_projection_profiles_have_expected_physical_shape():
    rows = _table_rows()
    alias_manifest = _manifest({
        "entity_type": "column",
        "business_key": "column:fact_order.order_amount",
        "alias": "销售额",
    })

    canonical = build_column_projection_records(
        rows,
        index_version=_version(rows, "single_canonical"),
        strategy="single_canonical",
        alias_manifest=alias_manifest,
    )
    multi = build_column_projection_records(
        rows,
        index_version=_version(rows, "multi_all"),
        strategy="multi_all",
        alias_manifest=alias_manifest,
    )
    selective = build_column_projection_records(
        rows,
        index_version=_version(rows, "mixed_selective"),
        strategy="mixed_selective",
        alias_manifest=alias_manifest,
    )

    assert len(canonical) == 2
    assert len(multi) == 7  # 2 names + 2 descriptions + 3 aliases
    assert len(selective) == 3
    assert projection_manifest(canonical)["points_per_entity"] == 1.0
    assert projection_manifest(multi)["points_per_entity"] == 3.5
    assert projection_manifest(selective)["point_roles"] == {
        "approved_alias": 1,
        "canonical": 2,
    }


def test_canonical_projection_is_stable_and_excludes_sensitive_examples():
    rows = _table_rows()
    manifest = _manifest()
    first = build_column_projection_records(
        rows,
        index_version=_version(rows, "single_canonical"),
        strategy="single_canonical",
        alias_manifest=manifest,
    )
    table, columns = rows[0]
    reversed_rows = [(table, list(reversed(columns)))]
    second = build_column_projection_records(
        reversed_rows,
        index_version=_version(reversed_rows, "single_canonical"),
        strategy="single_canonical",
        alias_manifest=manifest,
    )

    assert {item.point_id for item in first} == {item.point_id for item in second}
    assert all(uuid.UUID(item.point_id).version == 5 for item in first)
    assert all("Alice" not in item.text for item in first)
    assert all("examples" not in item.payload for item in first)
    assert all("Alice" not in str(item.payload) for item in first)
    assert all(item.payload["point_identity_version"] == "point-identity-v2" for item in first)
    assert projection_manifest(first)["stable_ids_valid"] is True


def test_canonical_id_is_stable_across_text_and_profile_changes():
    rows = _table_rows()
    manifest = _manifest({
        "entity_type": "column",
        "business_key": "column:fact_order.order_amount",
        "alias": "销售额",
    })
    original = build_column_projection_records(
        rows,
        index_version=_version(rows, "single_canonical"),
        strategy="single_canonical",
        alias_manifest=manifest,
    )[0]
    rows[0][1][0].description = "更新后的订单成交金额定义"
    changed_text = build_column_projection_records(
        rows,
        index_version=_version(rows, "single_canonical"),
        strategy="single_canonical",
        alias_manifest=manifest,
    )[0]
    mixed_canonical = next(
        item for item in build_column_projection_records(
            rows,
            index_version=_version(rows, "mixed_selective"),
            strategy="mixed_selective",
            alias_manifest=manifest,
        )
        if item.payload["business_key"] == "column:fact_order.order_amount"
        and item.point_role == "canonical"
    )

    assert original.point_id == changed_text.point_id == mixed_canonical.point_id
    assert original.payload["projection_hash"] != changed_text.payload["projection_hash"]


def test_selective_profile_only_materializes_governed_existing_aliases():
    rows = _table_rows()
    manifest = _manifest(
        {
            "entity_type": "column",
            "business_key": "column:fact_order.order_amount",
            "alias": "销售额",
        },
        {
            "entity_type": "column",
            "business_key": "column:fact_order.order_amount",
            "alias": "不存在于元知识的注入词",
        },
        {
            "entity_type": "column",
            "business_key": "column:fact_order.order_quantity",
            "alias": "销量",
        },
    )
    records = build_column_projection_records(
        rows,
        index_version=_version(rows, "mixed_selective"),
        strategy="mixed_selective",
        alias_manifest=manifest,
    )
    approved = [
        item.payload["projected_alias"]
        for item in records
        if item.point_role == "approved_alias"
    ]

    assert approved == ["销售额", "销量"]
    assert "不存在于元知识的注入词" not in approved
    assert max(
        sum(
            item.point_role == "approved_alias"
            for item in records
            if item.payload["entity_id"] == entity_id
        )
        for entity_id in {"fact_order.order_amount", "fact_order.order_quantity"}
    ) == 1


def test_metric_profiles_preserve_business_payload_and_projection_evidence():
    metric = SimpleNamespace(
        id="GMV",
        name="GMV",
        description="成交金额总和",
        relevant_columns=["fact_order.order_amount"],
        alias=["成交总额"],
        formula_rule={"type": "aggregate", "function": "SUM"},
    )
    records = build_metric_projection_records(
        [metric],
        workspace_id="workspace-a",
        index_version="sha256:index",
        strategy="mixed_selective",
        alias_manifest=_manifest({
            "entity_type": "metric",
            "business_key": "metric:GMV",
            "alias": "成交总额",
        }),
    )

    assert [item.point_role for item in records] == ["canonical", "approved_alias"]
    assert all(item.payload["workspace_id"] == "workspace-a" for item in records)
    assert all(item.payload["business_key"] == "metric:GMV" for item in records)
    assert all(item.payload["projection_hash"].startswith("sha256:") for item in records)


def _evaluation_profile(
    *,
    recall: float = 1.0,
    precision: float = 0.8,
    points: int = 10,
    entities: int = 10,
    approved_alias_points: int = 0,
):
    return {
        "summary": {
            "passed_cases": 28 if recall == 1.0 else 27,
            "routes": {
                route: {
                    "recall_at_k": recall,
                    "precision_at_k": precision,
                    "mrr": recall,
                    "ndcg": recall,
                }
                for route in ("column", "metric", "training")
            },
            "base_column_recall_at_k": recall,
            "base_column_precision_at_k": precision,
            "identifier_subset": {"recall_at_k": recall, "mrr": recall},
            "tenant_leakage": 0,
            "degradation_count": 0,
        },
        "manifest": {
            "point_count": points,
            "entity_count": entities,
            "points_per_entity": points / entities,
            "approved_alias_points": approved_alias_points,
            "stable_ids_valid": True,
            "stable_rebuild_ids": True,
            "vector_integrity": {"valid": True},
            "routes": {
                route: {
                    "collection_compatible": True,
                    "collection_points_count": 1,
                    "point_count": 1,
                }
                for route in ("column", "metric", "training")
            },
        },
        "ambiguous_false_positives": 0,
    }


def test_selection_prefers_canonical_when_quality_is_equal():
    profiles = {
        "single_canonical": _evaluation_profile(points=10),
        "multi_all": _evaluation_profile(points=35),
        "mixed_selective": _evaluation_profile(
            points=14,
            approved_alias_points=4,
        ),
    }

    decision = select_strategy(profiles, full_contract=True)

    assert decision["selected_strategy"] == "single_canonical"
    assert decision["gate_passed"] is True


def test_selection_uses_selective_only_to_close_canonical_gap():
    profiles = {
        "single_canonical": _evaluation_profile(recall=0.9, points=10),
        "multi_all": _evaluation_profile(points=35),
        "mixed_selective": _evaluation_profile(
            points=14,
            approved_alias_points=4,
        ),
    }

    decision = select_strategy(profiles, full_contract=True)

    assert decision["selected_strategy"] == "mixed_selective"
    assert "route_recall_non_regression" in decision["rejected"]["single_canonical"]
