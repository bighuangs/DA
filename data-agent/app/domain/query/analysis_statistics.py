"""查询结果统计的纯领域规则。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def is_numeric_value(value: object) -> bool:
    """判断值是否可安全转换为统计用浮点数。"""

    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    try:
        float(str(value))
        return True
    except (TypeError, ValueError):
        return False


def compute_analysis_statistics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """按列计算数值聚合或分类频次，供分析模型与确定性摘要复用。"""

    if not rows:
        return {}
    stats: dict[str, dict[str, Any]] = {}
    for column in rows[0]:
        values = [row.get(column) for row in rows]
        non_none = [value for value in values if value is not None]
        numeric = [value for value in non_none if is_numeric_value(value)]
        if numeric and len(numeric) >= max(1, len(non_none) // 2):
            numbers = [float(str(value)) for value in numeric]
            total = sum(numbers)
            stats[column] = {
                "kind": "numeric",
                "min": min(numbers),
                "max": max(numbers),
                "sum": total,
                "avg": round(total / len(numbers), 2),
                "count": len(numbers),
            }
            continue

        frequencies: dict[str, int] = {}
        for value in non_none:
            key = str(value)
            frequencies[key] = frequencies.get(key, 0) + 1
        top = sorted(
            frequencies.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:5]
        stats[column] = {
            "kind": "category",
            "unique": len(frequencies),
            "top": [{"value": key, "count": count} for key, count in top],
        }
    return stats


__all__ = ["compute_analysis_statistics", "is_numeric_value"]
