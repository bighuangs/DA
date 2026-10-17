"""HTTP Router 只能调用 Application use case，不得下钻仓储端口。"""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROUTER_ROOT = PROJECT_ROOT / "app" / "adapters" / "inbound" / "http" / "routers"

FORBIDDEN_PORT_NAMES = {
    "AccessPolicyAdminPort",
    "AuthRepositoryPort",
    "CacheInvalidationPort",
    "ConversationScopePort",
    "DashboardRepositoryPort",
    "FeedbackRepositoryPort",
    "QueryTraceReaderPort",
    "ReadonlySQLPort",
}
FORBIDDEN_DEPENDENCIES = {
    "get_access_policy_admin_repository",
    "get_auth_repository",
    "get_auth_token_service",
    "get_cache_invalidator",
    "get_conversation_scope",
    "get_dashboard_repo",
    "get_dw_mysql_repo",
    "get_feedback_repo",
    "get_query_trace_reader",
}


def test_http_routers_do_not_reach_repository_style_ports() -> None:
    violations: list[str] = []
    for path in sorted(ROUTER_ROOT.glob("*_router.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in FORBIDDEN_PORT_NAMES | FORBIDDEN_DEPENDENCIES:
                        violations.append(
                            f"{path.name}:{node.lineno} imports {alias.name}"
                        )
            elif isinstance(node, ast.Attribute) and node.attr == "meta_repo":
                violations.append(f"{path.name}:{node.lineno} accesses .meta_repo")
    assert violations == []
