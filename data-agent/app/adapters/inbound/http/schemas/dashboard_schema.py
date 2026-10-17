from typing import Optional

from pydantic import BaseModel, Field


class DashboardNode(BaseModel):
    """创建/更新节点请求体。"""
    name: str = Field(min_length=1, max_length=128)
    node_type: str = Field(default="dashboard", pattern="^(folder|dashboard)$")
    pid: Optional[str] = Field(default=None, max_length=36)
    component_data: Optional[list[dict]] = Field(default=None, max_length=100)
    sort: int = Field(default=0, ge=-100000, le=100000)


class DashboardUpdate(BaseModel):
    """更新节点请求体（name 与 component_data 均可选）。"""
    name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    component_data: Optional[list[dict]] = Field(default=None, max_length=100)


class AddBlockRequest(BaseModel):
    """把问数图表加入仪表盘：把一个画布块追加到目标仪表盘。"""
    chart: dict
    sql: str = Field(min_length=1, max_length=20000)
    query: Optional[str] = Field(default=None, max_length=2000)
