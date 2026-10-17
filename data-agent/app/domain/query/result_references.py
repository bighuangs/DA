"""把紧邻成功轮的单值结果投影为有界、白名单化的指代锚点。"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import TypedDict


class ResultReference(TypedDict):
    """允许进入 Agent State 的最小结果指代合同。"""

    entity: str
    column: str
    value: str


_REGION_COLUMNS = frozenset(
    {
        "region",
        "region_name",
        "area",
        "area_name",
        "sales_region",
        "地区",
        "区域",
        "大区",
    }
)
_SAFE_VALUE = re.compile(r"^[^\x00-\x1f\x7f]{1,80}$")


def project_single_result_references(
    rows: Sequence[Mapping[str, object]],
    *,
    row_count: int | None = None,
) -> dict[str, ResultReference]:
    """仅从恰好一行结果中提取非敏感的地区指代锚点。

    多行、缺行、字段歧义、空值、复杂值和非白名单字段都返回空映射，
    让上层要求用户澄清，避免从分页、截断或敏感结果中猜测实体。
    """

    if row_count is not None:
        try:
            if int(row_count) != 1:
                return {}
        except (TypeError, ValueError):
            return {}
    if len(rows) != 1 or not isinstance(rows[0], Mapping):
        return {}

    candidates: list[ResultReference] = []
    for raw_column, raw_value in rows[0].items():
        column = str(raw_column or "").strip()
        if column.lower() not in _REGION_COLUMNS:
            continue
        if not isinstance(raw_value, str):
            continue
        value = raw_value.strip()
        if not _SAFE_VALUE.fullmatch(value):
            continue
        candidates.append(
            ResultReference(entity="region", column=column, value=value)
        )

    unique = {(item["column"].lower(), item["value"]) for item in candidates}
    if len(unique) != 1:
        return {}
    return {"region": candidates[0]}


__all__ = ["ResultReference", "project_single_result_references"]
