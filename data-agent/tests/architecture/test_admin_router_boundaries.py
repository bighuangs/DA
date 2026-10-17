"""Admin Router 只能调用 Application use case。"""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROUTER = PROJECT_ROOT / "app/adapters/inbound/http/routers/admin_router.py"


def test_admin_router_never_reaches_repository_state() -> None:
    tree = ast.parse(ROUTER.read_text(encoding="utf-8"), filename=str(ROUTER))
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    assert "meta_repo" not in attributes
    assert "_repo_from_service" not in names


def test_admin_router_does_not_import_repository_or_cache_ports() -> None:
    tree = ast.parse(ROUTER.read_text(encoding="utf-8"), filename=str(ROUTER))
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert "AccessPolicyAdminPort" not in imported_names
    assert "CacheInvalidationPort" not in imported_names
    assert not {name for name in imported_names if name.endswith("RepositoryPort")}


def test_admin_router_has_no_direct_admin_resource_dependencies() -> None:
    source = ROUTER.read_text(encoding="utf-8")
    assert "get_access_policy_admin_repository" not in source
    assert "get_cache_invalidator" not in source
    assert "get_masking_policy_dependency" not in source
