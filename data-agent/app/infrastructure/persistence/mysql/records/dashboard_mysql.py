from sqlalchemy import String, Text, DateTime, ForeignKey, Integer, func
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.mysql.records.base import Base


class DashboardMySQL(Base):
    """
    仪表盘资源树模型（meta 库）。

    一行可表示「文件夹」或「仪表盘」节点，通过 pid 形成树形结构：
        - node_type='folder'：文件夹，仅承载子节点，component_data 为空
        - node_type='dashboard'：仪表盘，component_data 存放画布块的 JSON
          （每块含 x/y/w/h/chart/sql/query，加载时按 sql 重跑刷新数据）

    资源通过 workspace_id 隔离工作空间，并以 owner_id 记录所有者；写操作仍由
    API/RBAC 层授权，ORM 字段本身不替代权限校验。
    """
    __tablename__ = "dashboard"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        comment="节点ID（uuid4 字符串）"
    )
    owner_id: Mapped[str | None] = mapped_column(ForeignKey("app_user.id", ondelete="SET NULL"), nullable=True, index=True)
    workspace_id: Mapped[str | None] = mapped_column(ForeignKey("workspace.id", ondelete="CASCADE"), nullable=True, index=True)
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="节点名称（文件夹/仪表盘名）"
    )
    pid: Mapped[str] = mapped_column(
        ForeignKey("dashboard.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="父节点ID；根节点为 NULL"
    )
    node_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="节点类型：folder | dashboard"
    )
    component_data: Mapped[str] = mapped_column(
        Text,
        nullable=True,
        comment="画布块 JSON 字符串：[{x,y,w,h,chart,sql,query}]，仅 dashboard 节点使用"
    )
    sort: Mapped[int] = mapped_column(
        Integer,
        nullable=True,
        default=0,
        comment="同级排序序号"
    )
    created_at: Mapped[str] = mapped_column(
        DateTime,
        server_default=func.now(),
        comment="创建时间"
    )
    updated_at: Mapped[str] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        comment="更新时间"
    )
