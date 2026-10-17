"""仪表盘用例的出站端口。"""

from __future__ import annotations

from typing import Protocol


class DashboardStore(Protocol):
    async def list_tree(self) -> list[dict[str, object]]: ...

    async def create(self, **values: object) -> dict[str, object]: ...

    async def get(self, dashboard_id: str) -> dict[str, object] | None: ...

    async def update(
        self,
        dashboard_id: str,
        **values: object,
    ) -> dict[str, object] | None: ...

    async def delete(self, dashboard_id: str) -> bool: ...


class DashboardQueryPort(Protocol):
    async def validate_sql(self, sql: str) -> object: ...

    async def execute_sql(self, sql: str) -> list[dict[str, object]]: ...


__all__ = ["DashboardQueryPort", "DashboardStore"]
