"""Governed business vocabulary for physical Meta columns.

The YAML metadata remains the place for human-curated descriptions and aliases.
This small registry only supplies broadly reusable business terms that are tied to
stable physical columns.  Callers always keep curated aliases first and append
registry terms with deterministic de-duplication.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Final


BUSINESS_SEMANTIC_ALIASES: Final[Mapping[tuple[str, str], tuple[str, ...]]] = MappingProxyType({
    ("fact_order", "order_amount"): ("成交金额", "单笔金额"),
    ("fact_order", "order_quantity"): ("成交件数", "购买件数"),
    ("fact_order", "order_id"): ("订单数", "订单笔数"),
    ("dim_product", "product_name"): ("商品", "产品"),
})


def business_aliases_for(table_name: str, column_name: str) -> tuple[str, ...]:
    """Return governed aliases for one case-insensitive physical column ref."""

    key = (
        str(table_name or "").strip().casefold(),
        str(column_name or "").strip().casefold(),
    )
    return BUSINESS_SEMANTIC_ALIASES.get(key, ())


def stable_merge_aliases(*groups: Iterable[str] | str | None) -> list[str]:
    """Merge alias groups without changing the first spelling or group order."""

    result: list[str] = []
    seen: set[str] = set()
    for group in groups:
        if group is None:
            continue
        values = (group,) if isinstance(group, str) else group
        for value in values:
            text = str(value or "").strip()
            identity = re.sub(r"\s+", " ", text).casefold()
            if not identity or identity in seen:
                continue
            seen.add(identity)
            result.append(text)
    return result


def merge_business_aliases(
    table_name: str,
    column_name: str,
    *curated_alias_groups: Iterable[str] | str | None,
) -> list[str]:
    """Preserve curated aliases, then append governed defaults exactly once."""

    return stable_merge_aliases(
        *curated_alias_groups,
        business_aliases_for(table_name, column_name),
    )
