"""Named evaluation profiles prevent core and optional-analysis evidence mixing."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


EVALUATION_PROFILE_SCHEMA = "evaluation-profile-v1"
EVALUATION_PROFILES: dict[str, dict[str, Any]] = {
    "core-default": {
        "analysis_enabled": False,
        "purpose": "nl2sql_release",
        "release_profile": True,
    },
    "enhanced-analysis": {
        "analysis_enabled": True,
        "purpose": "optional_analysis_quality",
        "release_profile": False,
    },
}


def evaluation_profile_name(value: object, default: str = "unknown") -> str:
    """Read a current, explicitly versioned evaluation-profile object."""
    if not isinstance(value, Mapping):
        return default
    if value.get("schema") != EVALUATION_PROFILE_SCHEMA:
        return default
    resolved = str(value.get("name") or "").strip()
    return resolved if resolved in EVALUATION_PROFILES else default


def resolve_evaluation_profile(name: str | None) -> dict[str, Any]:
    resolved = str(name or "core-default").strip()
    if resolved not in EVALUATION_PROFILES:
        raise ValueError(
            f"未知 evaluation profile：{resolved}；可选值为 "
            + "、".join(EVALUATION_PROFILES)
        )
    return {
        "schema": EVALUATION_PROFILE_SCHEMA,
        "name": resolved,
        **EVALUATION_PROFILES[resolved],
    }
