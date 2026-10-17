"""Agent System 到 Application 的单向依赖门禁。"""

from __future__ import annotations

from pathlib import Path

from tests.architecture._imports import imports_from, package_imports


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_SYSTEM_ROOT = PROJECT_ROOT / "app" / "agent_system"


def test_agent_system_does_not_import_application() -> None:
    """扫描所有 AST 节点，函数内惰性导入同样不能绕过边界。"""

    violations = [
        (
            reference.path.relative_to(PROJECT_ROOT).as_posix(),
            reference.line,
            reference.module,
        )
        for reference in package_imports(PROJECT_ROOT, AGENT_SYSTEM_ROOT)
        if reference.module == "app.application"
        or reference.module.startswith("app.application.")
    ]
    assert violations == []


def test_source_comparison_has_single_domain_authority() -> None:
    old_path = (
        PROJECT_ROOT / "app" / "application" / "services" / "source_comparison_service.py"
    )
    new_path = PROJECT_ROOT / "app" / "domain" / "query" / "source_comparison.py"
    assert not old_path.exists()
    assert new_path.is_file()


def test_query_service_uses_domain_follow_up_semantics() -> None:
    query_service = (
        PROJECT_ROOT / "app" / "application" / "services" / "query_service.py"
    )
    agent_imports = [
        reference.module
        for reference in imports_from(PROJECT_ROOT, query_service)
        if reference.module == "app.agent_system"
        or reference.module.startswith("app.agent_system.")
    ]
    assert agent_imports == []

    domain_module = PROJECT_ROOT / "app" / "domain" / "query" / "follow_up_semantics.py"
    assert domain_module.is_file()
