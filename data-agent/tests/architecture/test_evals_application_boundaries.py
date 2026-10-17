"""Eval Application 的端口适配器依赖边界。"""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_APPLICATION_ROOT = PROJECT_ROOT / "evals" / "application"
EVAL_HTTP_ROUTER = PROJECT_ROOT / "evals" / "adapters" / "http" / "router.py"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            modules.add(node.module or "")
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def test_eval_application_has_no_production_concrete_or_framework_imports() -> None:
    forbidden = (
        "app.agent_system",
        "app.infrastructure",
        "app.application",
        "evals.infrastructure",
        "sqlalchemy",
        "fastapi",
    )
    violations = [
        f"{path.relative_to(PROJECT_ROOT).as_posix()} -> {module}"
        for path in sorted(EVAL_APPLICATION_ROOT.rglob("*.py"))
        for module in _imports(path)
        if any(
            module == prefix or module.startswith(f"{prefix}.")
            for prefix in forbidden
        )
    ]

    assert violations == []


def test_eval_http_router_uses_providers_instead_of_constructing_services() -> None:
    tree = ast.parse(
        EVAL_HTTP_ROUTER.read_text(encoding="utf-8"),
        filename=str(EVAL_HTTP_ROUTER),
    )
    constructed_services = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id.endswith("Service")
    ]
    provider_dependencies = {
        node.args[0].id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Depends"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id.startswith("get_")
    }

    assert constructed_services == []
    assert {
        "get_evaluation_service",
        "get_quality_operations_service",
        "get_planning_evaluation_service",
        "get_semantic_repair_evaluation_service",
    } <= provider_dependencies
