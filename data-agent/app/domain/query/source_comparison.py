"""上传来源独立聚合结果的确定性对比器。"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation


_MEASURE_HINTS = (
    "amount",
    "sales",
    "revenue",
    "gmv",
    "total",
    "quantity",
    "count",
    "ratio",
    "rate",
    "value",
    "金额",
    "销售额",
    "数量",
    "占比",
    "比率",
)
_DIMENSION_HINTS = (
    "id",
    "code",
    "year",
    "month",
    "day",
    "date",
    "年份",
    "月份",
    "日期",
)


def _number(value: object) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _measure_column(columns: set[str], rows: list[dict]) -> str | None:
    numeric = [
        column
        for column in sorted(columns)
        if rows
        and all(_number(row.get(column)) is not None for row in rows)
    ]
    hinted = [
        column
        for column in numeric
        if any(hint in column.lower() for hint in _MEASURE_HINTS)
    ]
    if hinted:
        return hinted[0]
    non_dimensions = [
        column
        for column in numeric
        if not any(hint in column.lower() for hint in _DIMENSION_HINTS)
    ]
    return (non_dimensions or numeric or [None])[0]


def _dimension_key(row: dict, dimensions: list[str]) -> tuple[str, ...]:
    # SQL 聚合维度均为标量；转成字符串可让 Decimal、日期和整数稳定对齐，
    # 同时避免把原始行写入日志或依赖不可哈希的应用对象。
    return tuple("" if row.get(column) is None else str(row.get(column)) for column in dimensions)


def _grouped_values(
    rows: list[dict],
    *,
    dimensions: list[str],
    measure: str,
) -> tuple[dict[tuple[str, ...], Decimal], dict[tuple[str, ...], dict]]:
    values: dict[tuple[str, ...], Decimal] = {}
    labels: dict[tuple[str, ...], dict] = {}
    for row in rows:
        key = _dimension_key(row, dimensions)
        value = _number(row.get(measure))
        if value is None:
            continue
        values[key] = values.get(key, Decimal("0")) + value
        labels.setdefault(
            key,
            {column: row.get(column) for column in dimensions},
        )
    return values, labels


def _percentage(delta: Decimal, baseline: Decimal) -> float | None:
    if baseline == 0:
        return None
    return round(float(delta / baseline * Decimal("100")), 4)


def _source_order(source: str) -> tuple[int, str]:
    value = str(source or "").strip().lower()
    if any(token in value for token in ("official", "正式", "dw")):
        return 0, value
    if any(token in value for token in ("upload", "上传")):
        return 1, value
    return 2, value


def deterministic_comparison(rows: list[dict]) -> dict:
    """只基于已执行的聚合结果按维度对齐计算差值，不让模型重算数值。"""
    source_column = next(
        (
            key for key in ("source_label", "source", "数据来源")
            if any(key in row for row in rows)
        ),
        None,
    )
    if not source_column:
        return {
            "contract": "deterministic-source-comparator-v2",
            "status": "not_applicable",
            "reason": "source_label_missing",
        }
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get(source_column) or "unknown"), []).append(row)
    if len(groups) != 2:
        return {
            "contract": "deterministic-source-comparator-v2",
            "status": "not_applicable",
            "reason": "requires_exactly_two_sources",
            "source_count": len(groups),
        }
    # 正式 DW 是稳定基准，上传文件是待比较对象。不能依赖字典序，
    # 否则中文 source_label 会显示成“上传 → 正式”并反转差值含义。
    names = sorted(groups, key=_source_order)
    left_rows, right_rows = groups[names[0]], groups[names[1]]
    all_rows = [*left_rows, *right_rows]
    common_columns = set.intersection(
        *(set(row) for row in all_rows)
    ) - {source_column}
    measure = _measure_column(common_columns, all_rows)
    if not measure:
        return {
            "contract": "deterministic-source-comparator-v2",
            "status": "not_applicable",
            "reason": "common_numeric_measure_missing",
            "sources": names,
        }
    dimensions = sorted(common_columns - {measure})
    left_values, left_labels = _grouped_values(
        left_rows,
        dimensions=dimensions,
        measure=measure,
    )
    right_values, right_labels = _grouped_values(
        right_rows,
        dimensions=dimensions,
        measure=measure,
    )
    keys = sorted(set(left_values) | set(right_values))
    if not keys:
        return {
            "contract": "deterministic-source-comparator-v2",
            "status": "not_applicable",
            "reason": "numeric_values_missing",
            "sources": names,
            "measure": measure,
        }

    comparisons = []
    for key in keys:
        left = left_values.get(key)
        right = right_values.get(key)
        delta = right - left if left is not None and right is not None else None
        comparisons.append({
            "dimensions": left_labels.get(key) or right_labels.get(key) or {},
            "baseline_value": float(left) if left is not None else None,
            "comparison_value": float(right) if right is not None else None,
            "delta": float(delta) if delta is not None else None,
            "delta_pct": _percentage(delta, left) if delta is not None else None,
            "baseline_missing": left is None,
            "comparison_missing": right is None,
        })

    left_total = sum(left_values.values(), Decimal("0"))
    right_total = sum(right_values.values(), Decimal("0"))
    total_delta = right_total - left_total
    grouped = bool(dimensions) or len(keys) > 1
    return {
        "contract": "deterministic-source-comparator-v2",
        "status": "compared",
        "mode": "grouped" if grouped else "scalar",
        "source_column": source_column,
        "measure": measure,
        "dimensions": dimensions,
        "baseline_source": names[0],
        "comparison_source": names[1],
        "baseline_value": float(left_total),
        "comparison_value": float(right_total),
        "delta": float(total_delta),
        "delta_pct": _percentage(total_delta, left_total),
        "compared_key_count": len(keys),
        "matched_key_count": len(set(left_values) & set(right_values)),
        "baseline_missing_count": len(set(right_values) - set(left_values)),
        "comparison_missing_count": len(set(left_values) - set(right_values)),
        "comparisons": comparisons if grouped else [],
        "row_level_join_used": False,
    }


__all__ = ["deterministic_comparison"]
