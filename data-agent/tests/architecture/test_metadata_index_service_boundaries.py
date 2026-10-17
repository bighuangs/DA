"""元数据索引 Application 服务只能依赖内层合同。"""

from __future__ import annotations

from pathlib import Path

from tests.architecture._imports import imports_from, matches_prefix


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVICE_ROOT = PROJECT_ROOT / "app" / "application" / "services"
TARGETS = (
    SERVICE_ROOT / "meta_admin_service.py",
    SERVICE_ROOT / "schema_sync_service.py",
    SERVICE_ROOT / "value_index_service.py",
)


def test_metadata_index_services_have_no_outer_imports() -> None:
    forbidden = ("app.config", "app.agent_system", "app.infrastructure")
    violations = [
        f"{item.path.relative_to(PROJECT_ROOT).as_posix()}:{item.line} -> {item.module}"
        for path in TARGETS
        for item in imports_from(PROJECT_ROOT, path)
        if matches_prefix(item.module, forbidden)
    ]

    assert violations == []
