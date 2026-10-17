"""仪表盘资源树与画布刷新用例。"""

from __future__ import annotations

import logging

from app.application.ports.dashboard import DashboardQueryPort, DashboardStore
from app.domain.access.policy import PolicySnapshot


logger = logging.getLogger(__name__)


class DashboardNotFoundError(LookupError):
    """目标仪表盘不存在。"""


class DashboardService:
    """集中执行仪表盘 CRUD、SQL 授权和画布刷新规则。"""

    def __init__(
        self,
        *,
        store: DashboardStore,
        query: DashboardQueryPort,
        access_policy: PolicySnapshot,
    ) -> None:
        self._store = store
        self._query = query
        self._access_policy = access_policy

    async def _authorize_sql(self, sql: str) -> str:
        authorized = self._access_policy.authorize_sql(sql)
        await self._query.validate_sql(authorized)
        return authorized

    async def _authorize_blocks(self, blocks: list | None) -> list | None:
        if blocks is None:
            return None
        prepared: list[dict] = []
        for block in blocks:
            copied = dict(block)
            if copied.get("sql"):
                await self._authorize_sql(str(copied["sql"]))
            prepared.append(copied)
        return prepared

    async def list_tree(self) -> list[dict[str, object]]:
        return await self._store.list_tree()

    async def create_node(
        self,
        *,
        name: str,
        node_type: str,
        pid: str | None,
        component_data: list | None,
        sort: int,
    ) -> dict[str, object]:
        return await self._store.create(
            name=name,
            node_type=node_type,
            pid=pid,
            component_data=await self._authorize_blocks(component_data),
            sort=sort,
        )

    async def update_node(
        self,
        dashboard_id: str,
        *,
        name: str | None,
        component_data: list | None,
    ) -> dict[str, object]:
        result = await self._store.update(
            dashboard_id,
            name=name,
            component_data=await self._authorize_blocks(component_data),
        )
        if result is None:
            raise DashboardNotFoundError("仪表盘不存在")
        return result

    async def delete_node(self, dashboard_id: str) -> None:
        if not await self._store.delete(dashboard_id):
            raise DashboardNotFoundError("仪表盘不存在")

    async def load_dashboard(self, dashboard_id: str) -> dict[str, object]:
        stored = await self._store.get(dashboard_id)
        if stored is None:
            raise DashboardNotFoundError("仪表盘不存在")
        result = dict(stored)
        hydrated: list[dict] = []
        for raw_block in list(stored.get("component_data") or []):
            block = dict(raw_block)
            sql = block.get("sql")
            block["rows"] = []
            block["error"] = None
            if sql:
                try:
                    authorized = await self._authorize_sql(str(sql))
                    block["rows"] = await self._query.execute_sql(authorized)
                except Exception as exc:  # 单块失败不得阻断整个仪表盘
                    logger.warning(
                        "仪表盘块刷新失败 error_type=%s",
                        type(exc).__name__,
                    )
                    block["error"] = str(exc)
            hydrated.append(block)
        result["component_data"] = hydrated
        return result

    async def add_block(
        self,
        dashboard_id: str,
        *,
        chart: dict,
        sql: str,
        query: str,
    ) -> dict:
        data = await self._store.get(dashboard_id)
        if data is None:
            raise DashboardNotFoundError("仪表盘不存在")
        blocks = [dict(item) for item in (data.get("component_data") or [])]
        max_y = max(
            (block.get("y", 0) + block.get("h", 6) for block in blocks),
            default=0,
        )
        await self._authorize_sql(sql)
        new_block = {
            "x": 0,
            "y": max_y,
            "w": 6,
            "h": 6,
            "chart": chart,
            "sql": sql,
            "query": query,
        }
        blocks.append(new_block)
        await self._store.update(dashboard_id, component_data=blocks)
        return new_block


__all__ = ["DashboardNotFoundError", "DashboardService"]
