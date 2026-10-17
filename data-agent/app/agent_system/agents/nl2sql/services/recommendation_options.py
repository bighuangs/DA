"""为“继续分析”生成带上下文、可独立执行的推荐项。

推荐项把界面展示文案和真正提交的问题分开。执行问题必须自行包含当前查询
已经确定的业务范围，不能依赖“该地区”“刚才结果”等脆弱指代。
"""

from __future__ import annotations

import hashlib
import math
import re
from decimal import Decimal
from typing import Any, Literal, Mapping, Sequence, TypedDict


RECOMMENDATION_CONTRACT_VERSION = "recommendation-options/v2"
RECOMMENDATION_CONTEXT_VERSION = "recommendation-context/v2"
RECOMMENDATION_FOCUS_POLICY_VERSION = "recommendation-focus-continuity/v1"
RecommendationKind = Literal[
    "breakdown",
    "trend",
    "ranking",
    "composition",
    "comparison",
    "exploration",
]


class RecommendationOption(TypedDict):
    """前端可直接展示并提交的版本化推荐项。"""

    id: str
    label: str
    query: str
    kind: RecommendationKind


_LOGICAL_LABELS = {
    "region": "地区",
    "region_name": "地区",
    "category": "品类",
    "category_name": "品类",
    "brand": "品牌",
    "brand_name": "品牌",
    "product": "商品",
    "product_name": "商品",
    "customer": "客户",
    "customer_name": "客户",
    "month": "月份",
    "year": "年份",
    "quarter": "季度",
    "date": "日期",
    "order_date": "日期",
    "order_amount": "销售额",
    "sales_amount": "销售额",
    "total_amount": "销售额",
    "total_sales_amount": "销售额",
    "revenue": "销售额",
    "order_quantity": "销量",
    "sales_quantity": "销量",
    "total_quantity": "销量",
    "order_count": "订单数",
    "customer_count": "客户数",
    "aov": "客单价",
    "average_order_value": "客单价",
    "profit": "利润",
    "profit_amount": "利润",
}

_MEASURE_HINT = re.compile(
    r"(?:amount|sales|revenue|quantity|count|total|avg|average|rate|ratio|profit|gmv|aov)",
    re.IGNORECASE,
)
_DIMENSION_HINT = re.compile(
    r"(?:name|region|category|brand|product|customer|date|month|year|quarter)",
    re.IGNORECASE,
)
_TIME_DIMENSION_LABELS = {"月份", "年份", "季度", "日期"}
_UNRESOLVED_REFERENCE = re.compile(
    r"(?:该|这个|那个|上述|上面(?:的)?|刚才(?:的)?|前面(?:的)?)"
    r"(?:地区|品类|品牌|商品|客户|结果|数据|指标|时间段)|(?:其中|它们|这些|那些)",
)


def _text(value: object, limit: int) -> str:
    normalized = re.sub(r"\s+", " ", str(value or "")).strip()
    return normalized[: max(0, int(limit))]


def _normalized(value: object) -> str:
    return re.sub(r"[\s，。；、,.!?！？:：'\"“”‘’]+", "", str(value or "")).lower()


def _safe_scalar(value: object) -> str | int | float | bool | None:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, (float, Decimal)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, str):
        return _text(value, 80)
    return None


def _numeric(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _logical_label(logical: object) -> str:
    key = str(logical or "").strip().lower()
    if key in _LOGICAL_LABELS:
        return _LOGICAL_LABELS[key]
    return key.replace("_", "") or "指标"


def _logical_from_column(column: object) -> str:
    key = str(column or "").strip().lower()
    if key in _LOGICAL_LABELS:
        return key
    for logical in _LOGICAL_LABELS:
        if key.endswith(logical) or logical.endswith(key):
            return logical
    return key


def _semantic_logicals(state: Mapping[str, Any], name: str) -> list[str]:
    values = []
    for item in (state.get("semantic_ir") or {}).get(name) or []:
        logical = item.get("logical") if isinstance(item, Mapping) else item
        logical = str(logical or "").strip().lower()
        if logical and logical not in values:
            values.append(logical)
    return values


def _infer_result_shape(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    if not rows:
        return [], []
    columns = list(rows[0].keys())
    dimensions: list[str] = []
    measures: list[str] = []
    for column in columns:
        logical = _logical_from_column(column)
        values = [row.get(column) for row in rows[:20] if row.get(column) is not None]
        numeric_values = [_numeric(value) for value in values]
        numeric = bool(values) and all(value is not None for value in numeric_values)
        if _DIMENSION_HINT.search(logical) and not _MEASURE_HINT.search(logical):
            dimensions.append(logical)
        elif _MEASURE_HINT.search(logical) or numeric:
            measures.append(logical)
        else:
            dimensions.append(logical)
    return dimensions, measures


def _result_key(row: Mapping[str, Any], logical: str) -> str | None:
    logical = str(logical or "").lower()
    for key in row:
        normalized = str(key).lower()
        if normalized == logical or normalized.endswith(logical) or logical.endswith(normalized):
            return str(key)
    expected_label = _logical_label(logical)
    for key in row:
        if _logical_label(_logical_from_column(key)) == expected_label:
            return str(key)
    return None


def _selects_single_entity(state: Mapping[str, Any]) -> bool:
    """判断本轮是否以结构化方式选择了一个实体，而非结果碰巧只有一行。"""

    limit = (state.get("semantic_ir") or {}).get("limit")
    try:
        if int(limit) == 1:
            return True
    except (TypeError, ValueError):
        pass
    sql = str(state.get("sql") or "")
    return bool(
        re.search(
            r"\blimit\s+(?:(?:\d+)\s*,\s*)?1(?:\s+offset\s+\d+)?\b",
            sql,
            re.IGNORECASE,
        )
    )


def _scope_text(logical: str, value: object) -> str:
    scalar = _safe_scalar(value)
    if scalar in (None, ""):
        return ""
    text = str(scalar)
    label = _logical_label(logical)
    if label == "地区":
        return text if text.endswith("地区") else f"{text}地区"
    if label == "年份":
        return text if text.endswith("年") else f"{text}年"
    if label in {"月份", "季度", "日期"}:
        return text
    return f"{label}“{text}”"


def build_recommendation_context(state: Mapping[str, Any]) -> dict[str, Any]:
    """提取当前成功结果的最小权威上下文，不依赖长期记忆投影。"""

    rows = [row for row in list(state.get("result") or []) if isinstance(row, Mapping)]
    inferred_dimensions, inferred_measures = _infer_result_shape(rows)
    dimensions = _semantic_logicals(state, "dimensions") or inferred_dimensions
    measures = _semantic_logicals(state, "measures") or inferred_measures
    primary_measure = measures[0] if measures else "order_amount"

    scope: list[dict[str, Any]] = []
    for item in (state.get("semantic_ir") or {}).get("filters") or []:
        if not isinstance(item, Mapping) or str(item.get("operator") or "eq") != "eq":
            continue
        logical = str(item.get("logical") or "").strip().lower()
        value = _safe_scalar(item.get("value"))
        if not logical or value in (None, "") or _MEASURE_HINT.search(logical):
            continue
        scope.append({
            "logical": logical,
            "label": _logical_label(logical),
            "value": value,
            "source": "semantic_filter",
        })

    # 单行、单实体结果可作为安全的本轮结果锚点。后续执行问题必须显式
    # 写出锚点值，不能依赖“该地区”“这个客户”等脆弱指代。
    if len(rows) == 1 and _selects_single_entity(state):
        row = rows[0]
        scoped_logicals = {str(item["logical"]) for item in scope}
        scoped_labels = {str(item["label"]) for item in scope}
        for logical in dimensions:
            label = _logical_label(logical)
            if (
                logical in scoped_logicals
                or label in scoped_labels
                or label in _TIME_DIMENSION_LABELS
            ):
                continue
            key = _result_key(row, logical)
            value = _safe_scalar(row.get(key)) if key is not None else None
            if value in (None, ""):
                continue
            scope.append({
                "logical": logical,
                "label": label,
                "value": value,
                "source": "single_row_result",
            })
            break

    scope_labels = {str(item.get("label") or "") for item in scope}
    analysis_focus = []
    for logical in dimensions:
        label = _logical_label(logical)
        if label in scope_labels or label in _TIME_DIMENSION_LABELS:
            continue
        analysis_focus.append({
            "logical": logical,
            "label": label,
            "source": "current_grouping",
        })

    result_columns = list(rows[0].keys()) if rows else []
    return {
        "schema_version": RECOMMENDATION_CONTEXT_VERSION,
        "focus_policy_version": RECOMMENDATION_FOCUS_POLICY_VERSION,
        "current_query": _text(
            state.get("query") or state.get("original_query"), 500
        ),
        "sql": _text(state.get("sql"), 4000),
        "dimensions": dimensions[:6],
        "measures": measures[:4],
        "primary_measure": primary_measure,
        "primary_measure_label": _logical_label(primary_measure),
        "scope": scope[:4],
        "scope_text": "".join(
            _scope_text(str(item["logical"]), item["value"]) for item in scope[:4]
        ),
        "analysis_focus": analysis_focus[:4],
        "analysis_focus_text": "、".join(
            str(item["label"]) for item in analysis_focus[:4]
        ),
        "result_columns": [str(column) for column in result_columns[:20]],
        "result_row_count": len(rows),
        "chart_type": str((state.get("chart") or {}).get("type") or ""),
        "analysis_summary": _text(state.get("analysis"), 500),
    }


def _option(kind: RecommendationKind, label: str, query: str) -> RecommendationOption:
    query = _text(query, 300)
    digest = hashlib.sha256(f"{kind}\0{query}".encode("utf-8")).hexdigest()[:12]
    return {
        "id": f"rec-{digest}",
        "label": _text(label, 100),
        "query": query,
        "kind": kind,
    }


def _candidate_dimensions(context: Mapping[str, Any]) -> list[tuple[str, str]]:
    occupied_labels = {
        _logical_label(value) for value in context.get("dimensions") or []
    }
    occupied_labels.update(
        str(item.get("label") or "") for item in context.get("scope") or []
    )
    catalog = [
        ("category", "品类"),
        ("brand", "品牌"),
        ("region_name", "地区"),
        ("customer_name", "客户"),
        ("product_name", "商品"),
    ]
    preferred = [item for item in catalog if item[1] not in occupied_labels]
    return preferred or catalog


def build_contextual_recommendations(
    state: Mapping[str, Any], limit: int = 4
) -> list[RecommendationOption]:
    """基于本轮问题、语义范围和真实结果生成多维度后续分析。"""

    context = build_recommendation_context(state)
    metric = str(context["primary_measure_label"] or "销售额")
    scope = str(context.get("scope_text") or "")
    dimensions = _candidate_dimensions(context)
    breakdown = dimensions[0][1]
    composition = next((label for _, label in dimensions if label != breakdown), breakdown)
    rank_target = "商品" if metric == "销量" else "客户"
    focus_items = [
        item for item in context.get("analysis_focus") or []
        if isinstance(item, Mapping) and str(item.get("label") or "")
    ]
    focus = str(focus_items[0]["label"]) if focus_items else ""

    if focus:
        candidates = [
            _option(
                "breakdown",
                f"按{breakdown}下钻{scope}{focus}",
                f"查询{scope}各{focus}在不同{breakdown}的{metric}，并按{metric}从高到低排序",
            ),
            _option(
                "trend",
                f"查看{scope}各{focus}月度{metric}趋势",
                f"用折线图展示{scope}各{focus}每月{metric}趋势",
            ),
            _option(
                "ranking",
                f"查看{scope}{focus}{metric}排行",
                f"查询{scope}{metric}最高的10个{focus}，并按{metric}从高到低排序",
            ),
            _option(
                "composition",
                f"查看{scope}各{focus}{metric}占比",
                f"用饼图展示{scope}各{focus}的{metric}占比",
            ),
            _option(
                "comparison",
                f"对比{scope}各{focus}{metric}与订单数",
                f"对比{scope}各{focus}的{metric}和订单数",
            ),
        ]
    else:
        candidates = [
            _option(
                "breakdown",
                f"分析{scope}{breakdown}贡献",
                f"查询{scope}各{breakdown}的{metric}，并按{metric}从高到低排序",
            ),
            _option(
                "trend",
                f"查看{scope}月度{metric}趋势",
                f"用折线图展示{scope}每月{metric}趋势",
            ),
            _option(
                "ranking",
                f"查看{scope}{rank_target}{metric}排行",
                f"查询{scope}{metric}最高的10个{rank_target}，并按{metric}从高到低排序",
            ),
            _option(
                "composition",
                f"查看{scope}{composition}{metric}占比",
                f"用饼图展示{scope}各{composition}的{metric}占比",
            ),
            _option(
                "comparison",
                f"对比{scope}{metric}与订单数",
                f"对比{scope}各地区的{metric}和订单数",
            ),
        ]

    current = _normalized(context.get("current_query"))
    current_kind = _infer_kind(context.get("current_query"))
    result: list[RecommendationOption] = []
    seen_queries: set[str] = set()
    for candidate in candidates:
        if (
            candidate["kind"] == current_kind
            and current_kind in {"trend", "ranking", "composition", "comparison"}
        ):
            continue
        normalized = _normalized(candidate["query"])
        if not normalized or normalized == current or normalized in seen_queries:
            continue
        seen_queries.add(normalized)
        result.append(candidate)
        if len(result) >= max(0, int(limit)):
            break
    return result


def _infer_kind(value: object) -> RecommendationKind:
    text = str(value or "")
    if re.search(r"趋势|折线|每月|每日|每年", text):
        return "trend"
    if re.search(r"占比|构成|饼图", text):
        return "composition"
    if re.search(
        r"排名|排行|排序|从高到低|从低到高|降序|升序|最高|最低|前\s*\d+|top",
        text,
        re.IGNORECASE,
    ):
        return "ranking"
    if re.search(r"对比|比较|同比|环比", text):
        return "comparison"
    if re.search(r"按|各|贡献|拆分|分布", text):
        return "breakdown"
    return "exploration"


def options_from_queries(
    queries: Sequence[object], limit: int = 4
) -> list[RecommendationOption]:
    """把旧字符串推荐升级为结构化项，供上传 Schema 路径兼容使用。"""

    result: list[RecommendationOption] = []
    for value in queries:
        query = _text(value, 300)
        if not query or _UNRESOLVED_REFERENCE.search(query):
            continue
        result.append(_option(_infer_kind(query), query, query))
        if len(result) >= max(0, int(limit)):
            break
    return result


def normalize_model_recommendations(
    values: Sequence[object],
    *,
    state: Mapping[str, Any],
    fallback: Sequence[RecommendationOption],
    limit: int = 4,
) -> list[RecommendationOption]:
    """校验模型推荐并用确定性选项补齐，确保范围闭合和类型多样。"""

    context = build_recommendation_context(state)
    scope_values = [
        str(item.get("value"))
        for item in context.get("scope") or []
        if item.get("value") not in (None, "")
    ]
    focus_labels = [
        str(item.get("label") or "")
        for item in context.get("analysis_focus") or []
        if isinstance(item, Mapping) and str(item.get("label") or "")
    ]
    current = _normalized(context.get("current_query"))
    result: list[RecommendationOption] = []
    seen_queries: set[str] = set()
    seen_kinds: set[str] = set()

    for value in list(values) + list(fallback):
        if isinstance(value, Mapping):
            label = _text(value.get("label") or value.get("query"), 100)
            query = _text(value.get("query") or value.get("label"), 300)
            proposed_kind = str(value.get("kind") or value.get("analysis_type") or "")
        else:
            label = query = _text(value, 300)
            proposed_kind = ""
        if not label or not query or _UNRESOLVED_REFERENCE.search(query):
            continue
        normalized = _normalized(query)
        if not normalized or normalized == current or normalized in seen_queries:
            continue
        # 筛选范围和当前分析主体是两个独立约束：前者保证 WHERE 不丢失，
        # 后者防止推荐被范围值抢占而偏离当前 GROUP BY/分析粒度。
        if scope_values and not all(str(item) in query for item in scope_values):
            continue
        if focus_labels and not any(label in query for label in focus_labels):
            continue
        kind = proposed_kind if proposed_kind in {
            "breakdown", "trend", "ranking", "composition", "comparison", "exploration"
        } else _infer_kind(query)
        if kind in seen_kinds:
            continue
        option = _option(kind, label, query)
        result.append(option)
        seen_queries.add(normalized)
        seen_kinds.add(kind)
        if len(result) >= max(0, int(limit)):
            break
    return result


def recommendation_event(options: Sequence[RecommendationOption]) -> dict[str, Any]:
    """同时发送 v2 DTO 与旧标签数组，避免旧客户端在滚动发布时失效。"""

    items = [dict(item) for item in options]
    return {
        "recommendation_contract": {
            "schema_version": RECOMMENDATION_CONTRACT_VERSION,
            "focus_policy_version": RECOMMENDATION_FOCUS_POLICY_VERSION,
        },
        "recommendations": items,
        "recommend": [item["label"] for item in items],
    }
