"""
仪表盘路由器：资源树 CRUD + 画布加载（刷新数据）+ 追加图表块。

一句话总述：提供仪表盘资源树（文件夹/仪表盘）的增删改查与画布加载端点；
            加载时按每个画布块的 SQL 在 DW 库重跑，回填最新数据供前端渲染。

设计说明：
    - 资源树 CRUD 仅操作 meta 库（DashboardRepository）；
    - 加载/追加涉及 DW 库执行 SQL（DWMysqlRepository，已带只读校验 + LIMIT 强制）；
    - 追加块端点供问数界面"加入仪表盘"按钮调用，把当前图表作为新块追加到目标仪表盘。
"""
from fastapi import APIRouter, Depends, HTTPException

from app.adapters.inbound.http.dependencies import get_dashboard_service
from app.adapters.inbound.http.schemas.dashboard_schema import AddBlockRequest, DashboardNode, DashboardUpdate
from app.application.services.auth_service import Principal
from app.adapters.inbound.http.dependencies import require_roles
from app.application.services.dashboard_service import (
    DashboardNotFoundError,
    DashboardService,
)

dashboard_router = APIRouter()


@dashboard_router.get("/api/dashboard/tree")
async def get_tree(service: DashboardService = Depends(get_dashboard_service)):
    """返回全部仪表盘节点（扁平列表），前端按 pid 构建树。"""
    return await service.list_tree()


@dashboard_router.post("/api/dashboard")
async def create_node(
    node: DashboardNode,
    service: DashboardService = Depends(get_dashboard_service),
    principal: Principal = Depends(require_roles("owner", "admin", "analyst")),
):
    """创建文件夹或仪表盘节点。"""
    return await service.create_node(
        name=node.name,
        node_type=node.node_type,
        pid=node.pid,
        component_data=node.component_data,
        sort=node.sort,
    )


@dashboard_router.put("/api/dashboard/{dashboard_id}")
async def update_node(
    dashboard_id: str,
    body: DashboardUpdate,
    service: DashboardService = Depends(get_dashboard_service),
    principal: Principal = Depends(require_roles("owner", "admin", "analyst")),
):
    """更新节点名称和/或画布块。"""
    try:
        return await service.update_node(
            dashboard_id,
            name=body.name,
            component_data=body.component_data,
        )
    except DashboardNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@dashboard_router.delete("/api/dashboard/{dashboard_id}")
async def delete_node(
    dashboard_id: str,
    service: DashboardService = Depends(get_dashboard_service),
    principal: Principal = Depends(require_roles("owner", "admin", "analyst")),
):
    """删除节点及其全部后代。"""
    try:
        await service.delete_node(dashboard_id)
    except DashboardNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"success": True}


@dashboard_router.get("/api/dashboard/{dashboard_id}")
async def load_dashboard(
    dashboard_id: str,
    service: DashboardService = Depends(get_dashboard_service),
):
    """
    加载仪表盘：取画布块并按每块的 sql 在 DW 库重跑，回填最新 rows。

    业务逻辑：
        1. 从 meta 库取仪表盘节点（含 component_data）
        2. 遍历每块，调用 dw_repo.execute_sql(sql) 刷新数据
        3. 单块失败不影响整盘，记录 error 字段供前端提示
    """
    try:
        return await service.load_dashboard(dashboard_id)
    except DashboardNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@dashboard_router.post("/api/dashboard/{dashboard_id}/block")
async def add_block(
    dashboard_id: str,
    body: AddBlockRequest,
    service: DashboardService = Depends(get_dashboard_service),
    principal: Principal = Depends(require_roles("owner", "admin", "analyst")),
):
    """
    把一个图表块追加到目标仪表盘的画布末尾（供问数界面"加入仪表盘"按钮调用）。

    业务逻辑：
        1. 取目标仪表盘现有 component_data
        2. 计算新块位置：追加到末尾，y 取已用最大行数，避免重叠
        3. 追加并保存
    """
    try:
        new_block = await service.add_block(
            dashboard_id,
            chart=body.chart,
            sql=body.sql,
            query=body.query or "",
        )
    except DashboardNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"success": True, "block": new_block}
