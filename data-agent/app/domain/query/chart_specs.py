"""查询结果图表类型与确定性图表规格的纯领域规则。"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence


_EXPLICIT_CHART_TERMS = {
    "column": ("柱状图", "柱形图", "纵向条形图", "column chart"),
    "bar": ("横向柱状图", "条形图", "bar chart"),
    "line": ("折线图", "line chart"),
    "pie": ("圆饼图", "饼状图", "饼图", "pie chart"),
    "table": ("数据表格", "表格", "table view"),
}

_DIMENSION_FIELD_PARTS = frozenset(
    {
        "id",
        "code",
        "name",
        "date",
        "time",
        "day",
        "week",
        "month",
        "quarter",
        "year",
        "category",
        "type",
        "region",
        "area",
        "province",
        "city",
        "country",
        "brand",
        "gender",
        "sex",
        "sku",
    }
)
_MEASURE_FIELD_PARTS = frozenset(
    {
        "total",
        "sum",
        "count",
        "amount",
        "quantity",
        "sales",
        "revenue",
        "avg",
        "average",
        "min",
        "max",
        "rate",
        "ratio",
        "percent",
        "score",
        "price",
        "cost",
        "profit",
    }
)
_DIMENSION_FIELD_HINTS = (
    "名称",
    "编号",
    "编码",
    "日期",
    "时间",
    "年份",
    "月份",
    "季度",
    "类别",
    "品类",
    "地区",
    "区域",
    "省份",
    "城市",
    "国家",
    "品牌",
    "性别",
    "商品",
    "产品",
    "客户",
    "用户",
)
_MEASURE_FIELD_HINTS = (
    "总量",
    "合计",
    "数量",
    "销量",
    "销售额",
    "金额",
    "收入",
    "成本",
    "利润",
    "均值",
    "平均",
    "占比",
    "比例",
    "比率",
    "价格",
    "单价",
    "计数",
    "次数",
)


def explicit_chart_type(query: str) -> str | None:
    """返回用户最后一次未被否定的明确图表类型。"""

    text = str(query or "").lower()
    matches: list[tuple[int, int, str]] = []
    for chart_type, terms in _EXPLICIT_CHART_TERMS.items():
        for term in terms:
            position = text.rfind(term.lower())
            if position < 0:
                continue
            end = position + len(term)
            prefix = text[max(0, position - 6) : position]
            suffix = text[end : end + 4]
            if re.search(
                r"(?:不要|不用|别用|取消|不是|不是要|不想用|不需要|不显示|不画)$",
                prefix,
            ) or re.match(r"(?:不要|不用|取消|去掉)", suffix):
                continue
            matches.append((end, len(term), chart_type))
    if not matches:
        return None
    return max(matches)[2]


def table_chart(columns: Sequence[str], *, title: str = "查询结果") -> dict:
    """构造无可视化维度时使用的表格规格。"""

    return {
        "type": "table",
        "title": title,
        "columns": [{"name": column, "value": column} for column in columns],
    }


def _numeric(value: object) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        float(str(value))
        return True
    except (TypeError, ValueError):
        return False


def _dimension_field(column: str) -> bool:
    normalized = str(column or "").lower()
    parts = {
        part
        for part in re.split(r"[^a-z0-9]+", normalized)
        if part
    }
    if parts & _MEASURE_FIELD_PARTS or any(
        hint in normalized for hint in _MEASURE_FIELD_HINTS
    ):
        return False
    return bool(parts & _DIMENSION_FIELD_PARTS) or any(
        hint in normalized for hint in _DIMENSION_FIELD_HINTS
    )


def build_deterministic_chart(
    query: str,
    rows: Sequence[Mapping[str, object]],
    *,
    requested_type: str | None = None,
) -> dict:
    """仅依据问题与结果形状构造可回放的安全图表规格。"""

    columns = list(rows[0].keys()) if rows else []
    title = (query or "查询结果")[:80]
    explicit_type = requested_type or explicit_chart_type(query)
    if explicit_type == "table":
        return table_chart(columns, title=title)
    if not rows or len(columns) < 2:
        return table_chart(columns, title=title)

    sample = rows[: min(20, len(rows))]
    numeric_candidates = [
        column
        for column in columns
        if any(_numeric(row.get(column)) for row in sample)
    ]
    dimensions = [
        column
        for column in columns
        if column not in numeric_candidates or _dimension_field(column)
    ]
    numeric = [column for column in numeric_candidates if column not in dimensions]
    if (
        not dimensions
        and explicit_type in {"column", "bar", "line", "pie"}
        and len(columns) >= 2
    ):
        dimensions = [columns[0]]
        numeric = [column for column in numeric_candidates if column != columns[0]]
    if not dimensions or not numeric:
        return table_chart(columns, title=title)

    dimension = dimensions[0]
    measures = numeric[:4]
    query_text = (query or "").lower()
    if explicit_type == "pie":
        return {
            "type": "pie",
            "title": title,
            "axis": {
                "series": {"name": dimension, "value": dimension},
                "y": {"name": measures[0], "value": measures[0]},
            },
        }
    if explicit_type in {"column", "bar", "line"}:
        return {
            "type": explicit_type,
            "title": title,
            "axis": {
                "x": {"name": dimension, "value": dimension},
                "y": [
                    {"name": measure, "value": measure}
                    for measure in measures
                ],
            },
        }
    if any(token in query_text for token in ("占比", "比例", "份额")) and len(measures) == 1:
        return {
            "type": "pie",
            "title": title,
            "axis": {
                "series": {"name": dimension, "value": dimension},
                "y": {"name": measures[0], "value": measures[0]},
            },
        }
    chart_type = "line" if any(
        token in query_text
        for token in ("趋势", "每月", "各月", "月度", "季度", "同比", "环比")
    ) else "column"
    return {
        "type": chart_type,
        "title": title,
        "axis": {
            "x": {"name": dimension, "value": dimension},
            "y": [
                {"name": measure, "value": measure}
                for measure in measures
            ],
        },
    }


__all__ = [
    "build_deterministic_chart",
    "explicit_chart_type",
    "table_chart",
]
