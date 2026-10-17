"""不依赖 Agent 运行时的可观测证据规范化。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def normalize_degradation_evidence(
    value: Mapping[str, Any] | None,
    *,
    default_route: str = "unknown",
    default_affects_core_path: bool = True,
) -> dict[str, Any]:
    """把降级事实转换为有界、可持久化的公共 DTO。"""

    item = dict(value or {})
    original_node = item.get("node_id") or item.get("node")
    node = str(original_node or "unknown")
    route = str(item.get("route") or default_route)
    reason = str(item.get("reason_code") or item.get("reason") or "unknown")[:120]
    display_reason = str(
        item.get("reason")
        or (reason.split(".", 1)[1] if reason.startswith("llm.") else reason)
    )[:120]
    return {
        "node_id": node,
        "route": route,
        "reason_code": reason,
        "reason": display_reason,
        "severity": str(item.get("severity") or "warning")[:32],
        "affects_core_path": (
            default_affects_core_path
            if item.get("affects_core_path") is None
            else bool(item.get("affects_core_path"))
        ),
        "message": str(item.get("message") or item.get("detail") or "")[:300] or None,
        "duration_ms": (
            max(0, int(float(item.get("duration_ms") or 0)))
            if item.get("duration_ms") is not None
            else None
        ),
        "evidence_complete": bool(original_node and reason != "unknown"),
    }


__all__ = ["normalize_degradation_evidence"]
