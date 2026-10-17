"""Domain 层的独立纯净边界门禁。"""

from __future__ import annotations

from pathlib import Path

from tests.architecture._imports import package_imports


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOMAIN_ROOT = PROJECT_ROOT / "app" / "domain"
ALLOWED_APP_PREFIXES = ("app.domain", "app.foundation")
FORBIDDEN_FRAMEWORK_PREFIXES = (
    "evals",
    "fastapi",
    "langgraph",
    "sqlalchemy",
)


def _is_prefixed(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == prefix or module.startswith(f"{prefix}.") for prefix in prefixes)


def _format(violations: list[tuple[Path, int, str]]) -> list[str]:
    return [
        f"{path.relative_to(PROJECT_ROOT).as_posix()}:{line} -> {module}"
        for path, line, module in violations
    ]


def test_domain_imports_only_domain_or_foundation_from_app() -> None:
    """函数体内的惰性导入也必须遵守同一依赖方向。"""

    violations = [
        (reference.path, reference.line, reference.module)
        for reference in package_imports(PROJECT_ROOT, DOMAIN_ROOT)
        if reference.module.startswith("app.")
        and not _is_prefixed(reference.module, ALLOWED_APP_PREFIXES)
    ]
    assert _format(violations) == []


def test_domain_does_not_import_runtime_frameworks_or_evaluation_code() -> None:
    violations = [
        (reference.path, reference.line, reference.module)
        for reference in package_imports(PROJECT_ROOT, DOMAIN_ROOT)
        if _is_prefixed(reference.module, FORBIDDEN_FRAMEWORK_PREFIXES)
    ]
    assert _format(violations) == []


def test_agent_private_semantics_are_not_classified_as_domain() -> None:
    safety_root = DOMAIN_ROOT / "query" / "safety"
    agent_services = (
        PROJECT_ROOT / "app" / "agent_system" / "agents" / "nl2sql" / "services"
    )
    for name in ("model_sql_policy.py", "semantic_guard.py"):
        assert not (safety_root / name).exists()
        assert (agent_services / name).is_file()
