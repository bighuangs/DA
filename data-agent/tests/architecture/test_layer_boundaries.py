"""目标架构边界测试。"""

from __future__ import annotations

import ast
from pathlib import Path

from tests.architecture._imports import (
    imports_from,
    matches_prefix,
    package_imports,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = PROJECT_ROOT / "app"

TARGET_PACKAGES = frozenset({
    "adapters",
    "agent_system",
    "application",
    "bootstrap",
    "config",
    "domain",
    "foundation",
    "infrastructure",
})
REQUIRED_TARGET_PACKAGES = frozenset({
    "adapters",
    "agent_system",
    "application",
    "bootstrap",
})
def _format_violations(violations: list[tuple[Path, str, int]]) -> list[str]:
    return [
        f"{path.relative_to(PROJECT_ROOT).as_posix()}:{line} -> {module}"
        for path, module, line in violations
    ]


def _forbidden_imports(
    package_name: str,
    prefixes: tuple[str, ...],
) -> list[tuple[Path, str, int]]:
    return [
        (reference.path, reference.module, reference.line)
        for reference in package_imports(PROJECT_ROOT, APP_ROOT / package_name)
        if matches_prefix(reference.module, prefixes)
    ]


def _module_scope_compile_paths(package: Path) -> set[str]:
    results: set[str] = set()
    if not package.exists():
        return results
    for path in package.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for statement in tree.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "compile"
                and not (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "re"
                )
                for node in ast.walk(statement)
            ):
                results.add(path.relative_to(PROJECT_ROOT).as_posix())
    return results


def _module_scope_imports(path: Path) -> set[str]:
    """只收集模块顶层导入，函数内的显式惰性桥接不会被计入。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            imports.update(alias.name for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom) and statement.level == 0:
            imports.add(statement.module or "")
    return imports


def test_every_top_level_app_package_is_classified() -> None:
    actual_packages = {
        path.name
        for path in APP_ROOT.iterdir()
        if (
            path.is_dir()
            and path.name != "__pycache__"
            and any(path.rglob("*.py"))
        )
    }
    assert REQUIRED_TARGET_PACKAGES <= actual_packages
    assert actual_packages == TARGET_PACKAGES


def test_domain_has_no_outer_layer_dependencies() -> None:
    violations = _forbidden_imports(
        "domain",
        (
            "app.adapters",
            "app.agent",
            "app.agent_system",
            "app.api",
            "app.application",
            "app.bootstrap",
            "app.infrastructure.clients",
            "app.config",
            "app.core",
            "app.eval",
            "app.infrastructure",
            "app.models",
            "app.repositories",
            "app.services",
            "evals",
            "fastapi",
            "langgraph",
            "sqlalchemy",
        ),
    )
    assert _format_violations(violations) == []


def test_foundation_has_no_dependencies_on_other_application_packages() -> None:
    violations = [
        (reference.path, reference.module, reference.line)
        for reference in package_imports(PROJECT_ROOT, APP_ROOT / "foundation")
        if (
            reference.module.startswith("app.")
            and not matches_prefix(reference.module, ("app.foundation",))
        )
    ]
    assert _format_violations(violations) == []


def test_application_depends_on_domain_and_ports_not_concrete_layers() -> None:
    violations = _forbidden_imports(
        "application",
        (
            "app.adapters",
            "app.agent",
            "app.agent_system",
            "app.api",
            "app.bootstrap",
            "app.infrastructure.clients",
            "app.config",
            "app.core",
            "app.eval",
            "app.infrastructure",
            "app.models",
            "app.repositories",
            "app.services",
            "evals",
            "fastapi",
            "langgraph",
            "sqlalchemy",
        ),
    )
    assert _format_violations(violations) == []


def test_agent_system_has_no_outer_layer_dependencies() -> None:
    violations = _forbidden_imports(
        "agent_system",
        (
            "app.adapters",
            "app.application",
            "app.api",
            "app.bootstrap",
            "app.config",
            "app.infrastructure.clients",
            "app.eval",
            "app.infrastructure",
            "app.repositories",
            "app.services",
            "evals",
            "fastapi",
            "sqlalchemy",
        ),
    )
    assert _format_violations(violations) == []


def test_new_http_routers_do_not_import_repositories_or_infrastructure() -> None:
    http_root = APP_ROOT / "adapters" / "inbound" / "http"
    violations: list[tuple[Path, str, int]] = []
    for path in http_root.rglob("*.py"):
        is_router = path.stem.endswith("router") or "routers" in path.parts
        if not is_router:
            continue
        violations.extend(
            (reference.path, reference.module, reference.line)
            for reference in imports_from(PROJECT_ROOT, path)
            if matches_prefix(reference.module, (
                "app.infrastructure.clients",
                "app.infrastructure",
                "app.repositories",
            ))
        )
    assert _format_violations(violations) == []


def test_new_production_packages_do_not_import_evaluation_code() -> None:
    violations: list[tuple[Path, str, int]] = []
    for package_name in TARGET_PACKAGES - {"bootstrap"}:
        violations.extend(_forbidden_imports(package_name, ("app.eval", "evals")))
    assert _format_violations(violations) == []

    bootstrap_consumers = {
        reference.path.relative_to(PROJECT_ROOT).as_posix()
        for reference in package_imports(PROJECT_ROOT, APP_ROOT / "bootstrap")
        if matches_prefix(reference.module, ("app.eval", "evals"))
    }
    assert bootstrap_consumers == {"app/bootstrap/http_application.py"}


def test_bootstrap_is_the_only_new_root_allowed_to_select_adapters() -> None:
    violations: list[tuple[Path, str, int]] = []
    for package_name in TARGET_PACKAGES - {"bootstrap"}:
        violations.extend(_forbidden_imports(package_name, ("app.bootstrap",)))
    assert _format_violations(violations) == []


def test_http_adapter_declares_dependencies_and_bootstrap_selects_implementations() -> None:
    adapter_dependencies = (
        APP_ROOT / "adapters" / "inbound" / "http" / "dependencies.py"
    )
    bootstrap_dependencies = APP_ROOT / "bootstrap" / "http_dependencies.py"
    adapter_imports = _module_scope_imports(adapter_dependencies)
    bootstrap_imports = _module_scope_imports(bootstrap_dependencies)

    forbidden_adapter_prefixes = (
        "app.agent_system",
        "app.bootstrap",
        "app.config",
        "app.infrastructure",
        "sqlalchemy",
    )
    assert not any(
        matches_prefix(module, forbidden_adapter_prefixes)
        for module in adapter_imports
    )
    assert "app.adapters.inbound.http" in bootstrap_imports
    assert any(module.startswith("app.infrastructure.") for module in bootstrap_imports)


def test_graphs_are_not_compiled_at_import_time() -> None:
    assert _module_scope_compile_paths(APP_ROOT / "agent_system") == set()
