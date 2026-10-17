"""“继续分析”推荐的分析主体连续性 Contract 测试。"""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from app.agent_system.agents.nl2sql.services.recommendation_options import (
    build_contextual_recommendations,
    build_recommendation_context,
    normalize_model_recommendations,
)


def _state(
    *,
    query: str,
    dimensions: Iterable[str],
    rows: list[dict[str, object]],
    filters: Iterable[tuple[str, object]] = (),
    measure: str = "order_amount",
    limit: int | None = None,
) -> dict[str, object]:
    """按统一语义 Contract 构造场景，不为任何样例增加专有分支。"""

    semantic_ir = {
        "dimensions": [{"logical": logical} for logical in dimensions],
        "measures": [{"logical": measure}],
        "filters": [
            {"logical": logical, "operator": "eq", "value": value}
            for logical, value in filters
        ],
    }
    if limit is not None:
        semantic_ir["limit"] = limit
    return {
        "query": query,
        "result": rows,
        "semantic_ir": semantic_ir,
    }


def _assert_continuity(
    state: dict[str, object],
    *,
    scope_tokens: Iterable[str],
    focus_labels: Iterable[str],
) -> None:
    options = build_contextual_recommendations(state)
    scope_tokens = tuple(scope_tokens)
    focus_labels = tuple(focus_labels)

    assert len(options) == 4
    assert len({option["kind"] for option in options}) == 4
    assert all(
        all(token in option["query"] for token in scope_tokens)
        for option in options
    )
    assert all(
        any(label in option["query"] for label in focus_labels)
        for option in options
    )


@pytest.mark.parametrize(
    ("state", "scope_tokens", "focus_labels"),
    [
        pytest.param(
            _state(
                query="华中地区各品类销售额，按销售额降序",
                dimensions=["category_name"],
                filters=[("region_name", "华中")],
                rows=[
                    {"category_name": "户外", "total_sales_amount": 8100},
                    {"category_name": "家居", "total_sales_amount": 6200},
                ],
            ),
            ("华中",),
            ("品类",),
            id="region-scope-category-focus",
        ),
        pytest.param(
            _state(
                query="2024年各地区销量",
                dimensions=["region_name"],
                filters=[("year", 2024)],
                rows=[
                    {"region_name": "西南", "total_quantity": 120},
                    {"region_name": "东北", "total_quantity": 95},
                ],
                measure="order_quantity",
            ),
            ("2024",),
            ("地区",),
            id="time-scope-region-focus",
        ),
        pytest.param(
            _state(
                query="品牌远山各商品销售额",
                dimensions=["product_name"],
                filters=[("brand_name", "远山")],
                rows=[
                    {"product_name": "冲锋衣", "total_sales_amount": 3000},
                    {"product_name": "登山包", "total_sales_amount": 2200},
                ],
            ),
            ("远山",),
            ("商品",),
            id="brand-scope-product-focus",
        ),
        pytest.param(
            _state(
                query="各客户销售额",
                dimensions=["customer_name"],
                rows=[
                    {"customer_name": "客户甲", "total_sales_amount": 5000},
                    {"customer_name": "客户乙", "total_sales_amount": 4300},
                ],
            ),
            (),
            ("客户",),
            id="unscoped-customer-focus",
        ),
    ],
)
def test_recommendations_preserve_scope_and_current_analysis_focus(
    state: dict[str, object],
    scope_tokens: tuple[str, ...],
    focus_labels: tuple[str, ...],
) -> None:
    _assert_continuity(
        state,
        scope_tokens=scope_tokens,
        focus_labels=focus_labels,
    )


def test_multi_dimension_result_preserves_at_least_one_current_focus() -> None:
    state = _state(
        query="各品类各品牌销售额",
        dimensions=["category_name", "brand_name"],
        rows=[
            {"category_name": "户外", "brand_name": "远山", "total_sales_amount": 3000},
            {"category_name": "家居", "brand_name": "木棉", "total_sales_amount": 2600},
        ],
    )

    _assert_continuity(state, scope_tokens=(), focus_labels=("品类", "品牌"))


def test_context_versions_and_exposes_current_grouping_focus() -> None:
    state = _state(
        query="北部地区各商品销量",
        dimensions=["product_name"],
        filters=[("region_name", "北部")],
        rows=[
            {"product_name": "商品一", "total_quantity": 18},
            {"product_name": "商品二", "total_quantity": 13},
        ],
        measure="order_quantity",
    )

    context = build_recommendation_context(state)

    assert context["schema_version"] == "recommendation-context/v2"
    assert context["focus_policy_version"] == "recommendation-focus-continuity/v1"
    assert context["analysis_focus"] == [
        {
            "logical": "product_name",
            "label": "商品",
            "source": "current_grouping",
        }
    ]
    assert context["analysis_focus_text"] == "商品"


def test_single_row_rank_result_becomes_explicit_scope_anchor() -> None:
    state = _state(
        query="销售额第二高的地区",
        dimensions=["region_name"],
        rows=[{"region_name": "华北", "total_sales_amount": 17000}],
        limit=1,
    )

    context = build_recommendation_context(state)
    options = build_contextual_recommendations(state)

    assert any(
        item["source"] == "single_row_result" and item["value"] == "华北"
        for item in context["scope"]
    )
    assert context["analysis_focus"] == []
    assert len(options) == 4
    assert all("华北" in option["query"] for option in options)
    assert all("该地区" not in option["query"] for option in options)


def test_single_row_without_single_entity_selector_remains_analysis_focus() -> None:
    state = _state(
        query="各品类销售额",
        dimensions=["category_name"],
        rows=[{"category_name": "办公", "total_sales_amount": 4200}],
    )

    context = build_recommendation_context(state)
    options = build_contextual_recommendations(state)

    assert not any(item["source"] == "single_row_result" for item in context["scope"])
    assert context["analysis_focus"] == [
        {
            "logical": "category_name",
            "label": "品类",
            "source": "current_grouping",
        }
    ]
    assert all("品类" in option["query"] for option in options)


def test_model_candidate_that_keeps_scope_but_drops_focus_is_rejected() -> None:
    state = _state(
        query="西南地区各品类销售额",
        dimensions=["category_name"],
        filters=[("region_name", "西南")],
        rows=[
            {"category_name": "食品", "total_sales_amount": 9800},
            {"category_name": "服饰", "total_sales_amount": 7600},
        ],
    )
    fallback = build_contextual_recommendations(state)
    drifted_query = "查询西南地区各品牌销售额"

    options = normalize_model_recommendations(
        [
            {
                "label": "分析品牌",
                "query": drifted_query,
                "kind": "breakdown",
            },
            {
                "label": "查看品类趋势",
                "query": "用折线图展示西南地区各品类每月销售额趋势",
                "kind": "trend",
            },
        ],
        state=state,
        fallback=fallback,
    )

    assert drifted_query not in {option["query"] for option in options}
    assert len(options) == 4
    assert all("西南" in option["query"] for option in options)
    assert all("品类" in option["query"] for option in options)


def test_time_axis_only_is_an_adjacent_exploration_guardrail() -> None:
    state = _state(
        query="每月销售额趋势",
        dimensions=["month"],
        rows=[
            {"month": 1, "total_sales_amount": 5100},
            {"month": 2, "total_sales_amount": 5400},
            {"month": 3, "total_sales_amount": 5200},
        ],
    )

    options = build_contextual_recommendations(state)

    assert len(options) == 4
    assert len({option["kind"] for option in options}) == 4
    assert any(
        any(label in option["query"] for label in ("品类", "品牌", "地区", "客户", "商品"))
        for option in options
    )
    assert all(
        not any(reference in option["query"] for reference in ("该结果", "上述数据", "它们"))
        for option in options
    )
