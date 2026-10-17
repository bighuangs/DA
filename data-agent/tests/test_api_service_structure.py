"""HTTP Adapter 与 Bootstrap 组合边界的结构回归测试。"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.bootstrap.api import create_app
from app.bootstrap.settings import BootstrapSettings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HTTP_ROOT = PROJECT_ROOT / "app" / "adapters" / "inbound" / "http"


def test_application_factory_registers_protocol_boundaries(monkeypatch) -> None:
    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS",
        "http://frontend-one.example, http://frontend-two.example",
    )

    app = create_app()
    paths = {route.path for route in app.routes}

    assert {
        "/api/query",
        "/api/chats",
        "/api/auth/login",
        "/api/uploaded-datasets",
        "/api/predict",
        "/api/dashboard/tree",
        "/api/feedback",
        "/api/admin/meta/overview",
    } <= paths
    assert "/api/admin/evaluation/quality/overview" not in paths
    assert callable(app.router.lifespan_context)
    assert len(app.dependency_overrides) >= 20

    cors = next(item for item in app.user_middleware if item.cls is CORSMiddleware)
    assert cors.kwargs["allow_origins"] == [
        "http://frontend-one.example",
        "http://frontend-two.example",
    ]
    assert any(item.cls is BaseHTTPMiddleware for item in app.user_middleware)


def test_evaluation_http_router_is_explicit_and_lazily_imported() -> None:
    sys.modules.pop("evals.adapters.http.router", None)

    disabled = create_app(
        settings=BootstrapSettings(evaluation_http_enabled=False)
    )
    disabled_paths = {route.path for route in disabled.routes}

    assert "/api/admin/evaluation/quality/overview" not in disabled_paths
    assert "evals.adapters.http.router" not in sys.modules
    assert disabled.state.evaluation_http_enabled is False

    enabled = create_app(
        settings=BootstrapSettings(evaluation_http_enabled=True)
    )
    enabled_paths = {route.path for route in enabled.routes}

    assert "/api/admin/evaluation/quality/overview" in enabled_paths
    assert "evals.adapters.http.router" in sys.modules
    assert enabled.state.evaluation_http_enabled is True
    eval_provider_names = {
        dependency.__name__
        for dependency in enabled.dependency_overrides
        if dependency.__module__.startswith("evals.adapters.http")
    }
    assert {
        "get_evaluation_service",
        "get_quality_operations_service",
        "get_planning_evaluation_service",
        "get_semantic_repair_evaluation_service",
    } <= eval_provider_names
    eval_routes = {
        (route.path, method)
        for route in enabled.routes
        if route.path.startswith("/api/admin/evaluation")
        for method in getattr(route, "methods", set())
    }
    assert {
        ("/api/admin/evaluation/scenarios/matrix", "GET"),
        ("/api/admin/evaluation/resilience/latest", "GET"),
        ("/api/admin/evaluation/adhoc-isolation/latest", "GET"),
        ("/api/admin/evaluation/rag/latest", "GET"),
        ("/api/admin/evaluation/planning/latest", "GET"),
        ("/api/admin/evaluation/planning/rollback", "POST"),
        ("/api/admin/evaluation/semantic-repair/latest", "GET"),
        ("/api/admin/evaluation/semantic-repair/rollback", "POST"),
        ("/api/admin/evaluation/nl2sql/runs", "GET"),
        ("/api/admin/evaluation/nl2sql/baselines", "GET"),
        ("/api/admin/evaluation/nl2sql/experiments", "GET"),
        ("/api/admin/evaluation/nl2sql/latest", "GET"),
        ("/api/admin/evaluation/nl2sql/governance", "GET"),
        ("/api/admin/evaluation/nl2sql/baselines/{run_id}/review", "POST"),
        ("/api/admin/evaluation/quality/overview", "GET"),
        ("/api/admin/evaluation/quality/traces/{trace_id}", "GET"),
        ("/api/admin/evaluation/quality/feedback/{feedback_id}/review", "PATCH"),
        ("/api/admin/evaluation/quality/feedback/{feedback_id}/candidate", "POST"),
        (
            "/api/admin/evaluation/quality/feedback/{feedback_id}/candidate/confirm",
            "POST",
        ),
    } <= eval_routes


def test_evaluation_http_has_no_top_level_bootstrap_import() -> None:
    path = PROJECT_ROOT / "app" / "bootstrap" / "http_application.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    eager_imports = [
        node.module or ""
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and (node.module or "").startswith("evals")
    ]

    assert eager_imports == []


def test_evaluation_http_environment_flag_is_fail_closed(monkeypatch) -> None:
    monkeypatch.delenv("EVALUATION_HTTP_ENABLED", raising=False)
    assert BootstrapSettings.from_environment().evaluation_http_enabled is False

    monkeypatch.setenv("EVALUATION_HTTP_ENABLED", "true")
    assert BootstrapSettings.from_environment().evaluation_http_enabled is True

    monkeypatch.setenv("EVALUATION_HTTP_ENABLED", "unexpected")
    assert BootstrapSettings.from_environment().evaluation_http_enabled is False


@pytest.mark.asyncio
async def test_disabled_evaluation_http_is_absent_from_no_lifespan_openapi() -> None:
    app = create_app(settings=BootstrapSettings(evaluation_http_enabled=False))
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://bootstrap.test",
    ) as client:
        response = await client.get("/openapi.json")

    assert response.status_code == 200
    assert "/api/admin/evaluation/quality/overview" not in response.json()["paths"]


@pytest.mark.asyncio
async def test_enabled_evaluation_http_contract_builds_without_lifespan() -> None:
    app = create_app(settings=BootstrapSettings(evaluation_http_enabled=True))
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://bootstrap.test",
    ) as client:
        response = await client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "get" in paths["/api/admin/evaluation/quality/overview"]
    assert "post" in paths["/api/admin/evaluation/planning/rollback"]
    assert "post" in paths[
        "/api/admin/evaluation/nl2sql/baselines/{run_id}/review"
    ]


def _absolute_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            imports.add(node.module or "")
        elif isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
    return imports


def test_http_routers_and_dependency_contracts_do_not_compose_infrastructure() -> None:
    forbidden = ("app.infrastructure", "app.config", "sqlalchemy")
    targets = [HTTP_ROOT / "dependencies.py", *sorted((HTTP_ROOT / "routers").glob("*_router.py"))]
    violations = [
        f"{path.relative_to(PROJECT_ROOT).as_posix()} -> {module}"
        for path in targets
        for module in _absolute_imports(path)
        if any(module == prefix or module.startswith(f"{prefix}.") for prefix in forbidden)
    ]

    assert violations == []


def test_frontend_protocol_code_uses_migrated_packages() -> None:
    expected = {
        PROJECT_ROOT / "app" / "bootstrap" / "api.py",
        PROJECT_ROOT / "app" / "bootstrap" / "http_application.py",
        PROJECT_ROOT / "app" / "bootstrap" / "http_dependencies.py",
        PROJECT_ROOT / "app" / "application" / "ports" / "http_runtime.py",
        HTTP_ROOT / "dependencies.py",
        HTTP_ROOT / "middleware.py",
        HTTP_ROOT / "routers" / "query_router.py",
        HTTP_ROOT / "schemas" / "query_schema.py",
    }

    assert all(path.is_file() for path in expected)
