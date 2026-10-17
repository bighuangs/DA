"""认证 Router 只能调用 Application use case 的架构门禁。"""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUTH_ROUTER = (
    PROJECT_ROOT
    / "app"
    / "adapters"
    / "inbound"
    / "http"
    / "routers"
    / "auth_router.py"
)
FORBIDDEN_AUTH_SYMBOLS = {
    "AuthRepositoryPort",
    "AuthTokenService",
    "PasswordPolicyError",
    "get_auth_repository",
    "get_auth_token_service",
    "hash_password",
    "validate_password",
    "verify_password",
}


def test_auth_router_does_not_import_repository_or_auth_primitives() -> None:
    tree = ast.parse(AUTH_ROUTER.read_text(encoding="utf-8"), filename=str(AUTH_ROUTER))
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        for alias in node.names:
            repository_style = alias.name.endswith(("Repository", "RepositoryPort"))
            if (
                module == "app.application.ports"
                or module.startswith("app.application.ports.")
                or module == "app.infrastructure"
                or module.startswith("app.infrastructure.")
                or repository_style
                or alias.name in FORBIDDEN_AUTH_SYMBOLS
            ):
                violations.append(f"{node.lineno}:{module}:{alias.name}")

    assert violations == []
