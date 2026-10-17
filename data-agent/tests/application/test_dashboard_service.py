"""仪表盘 Application 用例回归。"""

from __future__ import annotations

import pytest

from app.application.services.dashboard_service import (
    DashboardNotFoundError,
    DashboardService,
)


class _Policy:
    def authorize_sql(self, sql: str) -> str:
        return f"authorized:{sql}"


class _Query:
    def __init__(self) -> None:
        self.validated: list[str] = []

    async def validate_sql(self, sql: str) -> None:
        self.validated.append(sql)

    async def execute_sql(self, sql: str):
        if sql.endswith("bad"):
            raise RuntimeError("刷新失败")
        return [{"value": 1}]


class _Store:
    def __init__(self) -> None:
        self.item = {
            "id": "dashboard-1",
            "component_data": [
                {"sql": "good", "x": 0, "y": 0, "h": 6},
                {"sql": "bad", "x": 0, "y": 6, "h": 6},
            ],
        }
        self.updated: list[tuple[str, dict]] = []

    async def list_tree(self):
        return [self.item]

    async def create(self, **values):
        return {"id": "created", **values}

    async def get(self, dashboard_id: str):
        return self.item if dashboard_id == self.item["id"] else None

    async def update(self, dashboard_id: str, **values):
        if dashboard_id != self.item["id"]:
            return None
        self.updated.append((dashboard_id, values))
        return {**self.item, **values}

    async def delete(self, dashboard_id: str):
        return dashboard_id == self.item["id"]


@pytest.mark.asyncio
async def test_dashboard_service_authorizes_and_hydrates_each_block() -> None:
    store = _Store()
    query = _Query()
    service = DashboardService(store=store, query=query, access_policy=_Policy())

    loaded = await service.load_dashboard("dashboard-1")
    created = await service.create_node(
        name="经营看板",
        node_type="dashboard",
        pid=None,
        component_data=[{"sql": "create"}],
        sort=0,
    )

    assert loaded["component_data"][0]["rows"] == [{"value": 1}]
    assert loaded["component_data"][1]["rows"] == []
    assert loaded["component_data"][1]["error"] == "刷新失败"
    assert created["component_data"] == [{"sql": "create"}]
    assert query.validated == [
        "authorized:good",
        "authorized:bad",
        "authorized:create",
    ]


@pytest.mark.asyncio
async def test_dashboard_service_raises_typed_not_found_error() -> None:
    service = DashboardService(
        store=_Store(),
        query=_Query(),
        access_policy=_Policy(),
    )

    with pytest.raises(DashboardNotFoundError):
        await service.update_node(
            "missing",
            name="不存在",
            component_data=None,
        )
