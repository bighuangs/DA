"""会话 Application 切片的端口与配置边界。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from app.application.settings.conversation import ConversationMemorySettings
from tests.architecture._imports import imports_from


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVICE_ROOT = PROJECT_ROOT / "app" / "application" / "services"
TARGETS = (
    "conversation_data_context_service.py",
    "conversation_history_service.py",
    "conversation_memory_policy.py",
    "conversation_memory_service.py",
    "conversation_summary_service.py",
)
FORBIDDEN_PREFIXES = (
    "app.adapters",
    "app.agent_system",
    "app.config",
    "app.infrastructure",
    "evals",
    "fastapi",
    "langgraph",
    "sqlalchemy",
)


def _is_forbidden(module: str) -> bool:
    return any(
        module == prefix or module.startswith(f"{prefix}.")
        for prefix in FORBIDDEN_PREFIXES
    )


def test_conversation_services_only_depend_on_ports_settings_and_domain() -> None:
    violations = [
        (
            reference.path.relative_to(PROJECT_ROOT).as_posix(),
            reference.line,
            reference.module,
        )
        for name in TARGETS
        for reference in imports_from(PROJECT_ROOT, SERVICE_ROOT / name)
        if _is_forbidden(reference.module)
    ]
    assert violations == []


def test_conversation_settings_are_immutable_snapshots() -> None:
    settings = ConversationMemorySettings()
    with pytest.raises(FrozenInstanceError):
        setattr(settings, "enabled", False)
