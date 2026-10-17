"""Structured, privacy-safe degradation responses for unavailable model paths."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


MODEL_CLARIFICATION_VERSION = "model-clarification-v1"


def build_model_clarification(
    *,
    node_id: str,
    reason_code: str,
    semantic_ir: Mapping[str, Any] | None = None,
    retryable: bool,
) -> dict[str, Any]:
    """Build an actionable clarification without copying query text or SQL."""

    coverage = dict((semantic_ir or {}).get("semantic_coverage") or {})
    missing = [
        str(item)
        for item in (coverage.get("unresolved_slots") or [])
        if str(item)
    ][:12]
    suggestions = [
        "明确要统计的指标及聚合方式",
        "明确分组维度、时间范围和筛选条件",
    ]
    if any("operator:" in item for item in missing):
        suggestions.append("把比较、集合、窗口或排序要求单独写明")
    return {
        "schema_version": MODEL_CLARIFICATION_VERSION,
        "node_id": str(node_id),
        "reason_code": str(reason_code),
        "action": "clarify_then_retry",
        "retryable": bool(retryable),
        "message": (
            "当前未能在安全时限内完成完整 SQL。请补充或缩小条件后重试，"
            "系统不会执行丢失语义槽位的简化查询。"
        ),
        "unresolved_slots": missing,
        "suggestions": suggestions,
        "contains_query_or_sql": False,
    }


def clarification_message(value: Mapping[str, Any] | None) -> str:
    item = dict(value or {})
    return str(
        item.get("message")
        or "当前查询需要补充完整的指标、维度、时间或筛选条件。"
    )
