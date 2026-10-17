"""
仪表盘资源树数据访问层（meta 库）。

一句话总述：封装 dashboard 表的 CRUD —— 列树、建节点、改名/改画布、删节点（含子树级联）、取单个仪表盘。

设计说明：
    - 仅做 meta 库的元数据 CRUD，不触碰 DW 库；
    - 画布块中 SQL 的"加载时刷新数据"由路由层调用 dw_mysql_repo 完成，保持本仓储职责单一；
    - 删除文件夹时级联删除其下所有子节点（递归按 pid 收集后代）。
"""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.observability.logging import logger
from app.infrastructure.persistence.mysql.records.dashboard_mysql import DashboardMySQL


class DashboardRepository:
    def __init__(self, session: AsyncSession, owner_id: str | None = None, workspace_id: str | None = None):
        self.session = session
        self.owner_id = owner_id
        self.workspace_id = workspace_id

    async def list_tree(self) -> list[dict]:
        """
        返回全部节点（扁平列表），前端按 pid 自行构建树。

        Returns:
            list[dict]: [{id, name, pid, node_type, sort, created_at}]
        """
        statement = select(DashboardMySQL)
        if self.workspace_id:
            statement = statement.where(DashboardMySQL.workspace_id == self.workspace_id)
        result = await self.session.execute(statement.order_by(DashboardMySQL.sort, DashboardMySQL.created_at))
        rows = result.scalars().all()
        return [
            {
                "id": r.id,
                "name": r.name,
                "pid": r.pid,
                "node_type": r.node_type,
                "sort": r.sort or 0,
                "created_at": str(r.created_at) if r.created_at else None,
            }
            for r in rows
        ]

    async def get(self, dashboard_id: str) -> Optional[dict]:
        """
        取单个仪表盘节点（含 component_data），供加载画布使用。

        Returns:
            dict | None: 节点信息（component_data 反序列化为 list）
        """
        import json
        statement = select(DashboardMySQL).where(DashboardMySQL.id == dashboard_id)
        if self.workspace_id:
            statement = statement.where(DashboardMySQL.workspace_id == self.workspace_id)
        result = await self.session.execute(statement)
        r = result.scalars().first()
        if not r:
            return None
        try:
            blocks = json.loads(r.component_data) if r.component_data else []
        except Exception:
            logger.warning(f"仪表盘 {dashboard_id} 的 component_data 解析失败，按空画布处理")
            blocks = []
        return {
            "id": r.id,
            "name": r.name,
            "pid": r.pid,
            "node_type": r.node_type,
            "sort": r.sort or 0,
            "component_data": blocks,
            "created_at": str(r.created_at) if r.created_at else None,
            "updated_at": str(r.updated_at) if r.updated_at else None,
        }

    async def create(self, name: str, node_type: str, pid: Optional[str] = None,
                     component_data: Optional[list] = None, sort: int = 0) -> dict:
        """
        创建节点（文件夹或仪表盘）。

        业务逻辑：
            1. 生成 uuid4 作为主键
            2. component_data 序列化为 JSON 文本入库
            3. flush+refresh 取回默认值（created_at 等）
        """
        import json
        node = DashboardMySQL(
            id=str(uuid.uuid4()),
            owner_id=self.owner_id,
            workspace_id=self.workspace_id,
            name=name,
            pid=pid,
            node_type=node_type,
            component_data=json.dumps(component_data, ensure_ascii=False) if component_data else None,
            sort=sort,
        )
        self.session.add(node)
        await self.session.flush()
        await self.session.refresh(node)
        await self.session.commit()
        return {
            "id": node.id,
            "name": node.name,
            "pid": node.pid,
            "node_type": node.node_type,
            "sort": node.sort or 0,
        }

    async def update(self, dashboard_id: str, name: Optional[str] = None,
                     component_data: Optional[list] = None) -> Optional[dict]:
        """更新节点名称和/或画布块。"""
        import json
        values: dict = {"updated_at": datetime.now()}
        if name is not None:
            values["name"] = name
        if component_data is not None:
            values["component_data"] = json.dumps(component_data, ensure_ascii=False)
        statement = update(DashboardMySQL).where(DashboardMySQL.id == dashboard_id)
        if self.workspace_id:
            statement = statement.where(DashboardMySQL.workspace_id == self.workspace_id)
        await self.session.execute(statement.values(**values))
        await self.session.commit()
        return await self.get(dashboard_id)

    async def delete(self, dashboard_id: str) -> bool:
        """
        删除节点及其全部后代（递归按 pid 收集子树）。

        业务逻辑：
            1. 从目标节点出发，广度优先收集所有后代 id
            2. 一次性 DELETE WHERE id IN (...)
        """
        ids = {dashboard_id}
        frontier = [dashboard_id]
        while frontier:
            result = await self.session.execute(
                select(DashboardMySQL.id).where(DashboardMySQL.pid.in_(frontier))
                .where(DashboardMySQL.workspace_id == self.workspace_id)
            )
            children = [r for r in result.scalars().all()]
            new_children = [c for c in children if c not in ids]
            ids.update(children)
            frontier = new_children
        statement = delete(DashboardMySQL).where(DashboardMySQL.id.in_(ids))
        if self.workspace_id:
            statement = statement.where(DashboardMySQL.workspace_id == self.workspace_id)
        result = await self.session.execute(statement)
        await self.session.commit()
        return bool(result.rowcount)
