"""从已授权 SQL 提取可持久化的多轮追问结构语义。"""

from __future__ import annotations

import sqlglot
from sqlglot import exp


FOLLOW_UP_VERSION = "follow-up-v1"


def semantic_state_from_sql(question: str, sql: str) -> dict:
    """提取脱敏后的分组、聚合、过滤、排序、限制和表集合语义。"""

    # ``question`` 保留在公开合同中，用于兼容现有调用和未来语义扩展。
    _ = question
    state = {
        "version": FOLLOW_UP_VERSION,
        "dimensions": [],
        "measures": [],
        "filters": [],
        "time_grain": None,
        "order": [],
        "limit": None,
        "authorized_tables": [],
    }
    try:
        tree = sqlglot.parse_one(str(sql or ""), read="mysql")
    except Exception:
        return state

    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if select is None:
        return state

    group = select.args.get("group")
    state["dimensions"] = sorted(
        {
            column.name.lower()
            for column in (
                group.find_all(exp.Column) if group is not None else []
            )
        }
    )
    for projection in select.expressions:
        aggregate = projection.find(exp.AggFunc)
        if aggregate is not None:
            state["measures"].append(
                {
                    "function": aggregate.sql_name().upper(),
                    "columns": [
                        column.name.lower()
                        for column in aggregate.find_all(exp.Column)
                    ],
                    "alias": projection.alias_or_name or None,
                }
            )

    where = select.args.get("where")
    if where is not None:
        safe_where = where.copy().transform(
            lambda node: (
                exp.Literal.string("[MASKED]")
                if isinstance(node, exp.Literal) and node.is_string
                else exp.Literal.number(0)
                if isinstance(node, exp.Literal)
                else node
            )
        )
        state["filters"] = [
            {
                "columns": sorted(
                    {
                        column.name.lower()
                        for column in where.find_all(exp.Column)
                    }
                ),
                "shape": safe_where.sql(dialect="mysql")[:1000],
            }
        ]

    order = select.args.get("order")
    if order is not None:
        state["order"] = [
            item.sql(dialect="mysql")[:300] for item in order.expressions
        ]
    limit = select.args.get("limit")
    if limit is not None:
        try:
            state["limit"] = int(limit.expression.this)
        except (AttributeError, TypeError, ValueError):
            pass

    state["authorized_tables"] = sorted(
        {table.name.lower() for table in tree.find_all(exp.Table)}
    )
    dimension_text = " ".join(state["dimensions"])
    for grain, tokens in (
        ("month", ("month", "月份")),
        ("quarter", ("quarter", "季度")),
        ("year", ("year", "年份")),
    ):
        if any(token in dimension_text for token in tokens):
            state["time_grain"] = grain
            break
    return state


__all__ = ["FOLLOW_UP_VERSION", "semantic_state_from_sql"]
