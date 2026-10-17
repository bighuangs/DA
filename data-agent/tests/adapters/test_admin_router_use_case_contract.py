"""Admin Router 到 Application 用例的真实 HTTP 契约回归。"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.adapters.inbound.http.dependencies import (
    get_meta_admin_service,
    get_schema_sync_service,
)
from app.adapters.inbound.http.routers.admin_router import admin_router
from app.application.services.meta_admin_service import AdminConflictError


class _AdminUseCase:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def overview(self) -> dict:
        self.calls.append(("overview", {}))
        return {"metrics": 2, "training": 1}

    async def create_access_policy(self, **values: object) -> dict:
        self.calls.append(("create_access_policy", values))
        return {"id": 7, **values}

    def get_masking_policy(self) -> dict:
        self.calls.append(("get_masking_policy", {}))
        return {"enabled": True, "rules": []}

    async def create_metric(self, **values: object) -> dict:
        self.calls.append(("create_metric", values))
        raise AdminConflictError("指标名称已存在")


class _SchemaSyncUseCase:
    async def preview(self) -> dict:
        return {"source": "dw", "table_count": 3, "column_count": 12, "diff": {}}


def _test_app(admin: _AdminUseCase) -> FastAPI:
    app = FastAPI()
    app.include_router(admin_router)
    app.dependency_overrides[get_meta_admin_service] = lambda: admin
    app.dependency_overrides[get_schema_sync_service] = _SchemaSyncUseCase
    return app


def test_admin_api_delegates_governance_operations_to_application() -> None:
    admin = _AdminUseCase()
    with TestClient(_test_app(admin)) as client:
        overview = client.get("/api/admin/meta/overview")
        access_policy = client.post(
            "/api/admin/meta/access-policies",
            json={"role": "analyst", "table_pattern": "fact_*"},
        )
        masking_policy = client.get("/api/admin/meta/masking/policy")
        schema_preview = client.get("/api/admin/meta/schema/preview")

    assert overview.status_code == 200
    assert overview.json() == {"metrics": 2, "training": 1}
    assert access_policy.status_code == 201
    assert access_policy.json()["id"] == 7
    assert access_policy.json()["role"] == "analyst"
    assert masking_policy.json() == {"enabled": True, "rules": []}
    assert schema_preview.json()["table_count"] == 3
    assert [name for name, _ in admin.calls] == [
        "overview",
        "create_access_policy",
        "get_masking_policy",
    ]


def test_admin_api_maps_application_conflict_without_repository_knowledge() -> None:
    admin = _AdminUseCase()
    with TestClient(_test_app(admin)) as client:
        response = client.post(
            "/api/admin/meta/metrics",
            json={"name": "gmv", "description": "成交金额"},
        )

    assert response.status_code == 409
    assert response.json() == {"detail": "指标名称已存在"}
