"""Stable semantic fixtures shared by the current difficulty-suite tests."""
from __future__ import annotations

from app.agent_system.agents.nl2sql.semantic_ir import build_semantic_ir
from app.agent_system.agents.nl2sql.semantic_ir_v2 import FIELD_SPECS
from evals.legacy.dataset_suite import load_suite_cases


_EXAMPLES = {
    "region_name": ["华东", "华南", "华北", "西南"],
    "province": ["广东省", "江苏省"],
    "category": ["食品饮料", "家用电器", "手机数码", "办公用品", "户外运动"],
    "brand": ["苹果", "美的"],
    "member_level": ["黄金", "白银"],
    "gender": ["男", "女"],
}


def _column_role(table_name: str, column_name: str) -> str:
    if table_name == "fact_order":
        if column_name == "order_id":
            return "primary_key"
        if column_name.endswith("_id"):
            return "foreign_key"
        return "measure"
    if column_name.endswith("_id"):
        return "primary_key"
    return "dimension"


def semantic_metadata() -> list[dict]:
    """Return the canonical synthetic-DW schema used by semantic unit tests."""
    tables: dict[str, dict] = {}
    for logical, spec in FIELD_SPECS.items():
        table_name = str(spec["table"])
        column_name = str(spec["column"])
        table = tables.setdefault(
            table_name,
            {
                "name": table_name,
                "role": "fact" if table_name == "fact_order" else "dim",
                "description": "canonical synthetic DW table",
                "columns": [],
            },
        )
        if any(column["name"] == column_name for column in table["columns"]):
            continue
        terms = list(spec.get("terms") or ())
        table["columns"].append(
            {
                "name": column_name,
                "type": "",
                "role": _column_role(table_name, column_name),
                "description": " ".join(terms),
                "alias": terms,
                "examples": list(_EXAMPLES.get(logical) or []),
                "is_display": logical in {
                    "region_name",
                    "province",
                    "country",
                    "product_name",
                    "category",
                    "brand",
                    "customer_name",
                    "member_level",
                    "gender",
                    "year",
                    "quarter",
                    "month",
                    "day",
                },
            }
        )
    return list(tables.values())


def difficulty_sql_cases() -> list[dict]:
    """Load all 250 cases governed by the current three difficulty manifests."""
    cases: list[dict] = []
    for suite in ("easy_sql_v1", "medium_sql_v1", "hard_sql_v1"):
        cases.extend(
            {**case, "dataset_suite": suite}
            for case in load_suite_cases(suite)
        )
    return cases


def closed_semantic_ir_cases() -> list[dict]:
    """Return current cases owned by the deterministic compiler contract."""
    metadata = semantic_metadata()
    return [
        case
        for case in difficulty_sql_cases()
        if build_semantic_ir(
            case["question"], metadata, [], case.get("history") or [],
        )["compiler"]["supported"]
    ]
