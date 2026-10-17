"""Query Trace 应用服务不得依赖配置实现或 Agent 私有模块。"""

from __future__ import annotations

from pathlib import Path

from tests.architecture._imports import imports_from, matches_prefix


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUERY_TRACE_SERVICE = (
    PROJECT_ROOT / "app" / "application" / "services" / "query_trace_service.py"
)


def test_query_trace_service_uses_inner_contracts_only() -> None:
    forbidden = ("app.config", "app.agent_system")
    violations = [
        f"{item.path.relative_to(PROJECT_ROOT).as_posix()}:{item.line} -> {item.module}"
        for item in imports_from(PROJECT_ROOT, QUERY_TRACE_SERVICE)
        if matches_prefix(item.module, forbidden)
    ]

    assert violations == []
