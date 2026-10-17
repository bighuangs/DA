from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.types import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.mysql.records.base import Base


class AccessPolicyMySQL(Base):
    """工作空间内的表/列/行访问策略。"""

    __tablename__ = "access_policy"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspace.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(20), default="all", index=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("app_user.id", ondelete="CASCADE"), nullable=True, index=True)
    table_pattern: Mapped[str] = mapped_column(String(128), index=True)
    allowed_columns: Mapped[list | None] = mapped_column(JSON, nullable=True)
    row_filter: Mapped[str | None] = mapped_column(Text, nullable=True)
    access: Mapped[bool] = mapped_column(Boolean, default=True, server_default="1")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, server_default="1")
    priority: Mapped[int] = mapped_column(Integer, default=100, server_default="100")
    # 兼容 MySQL/MariaDB 不允许 TEXT DEFAULT 的版本；仓储始终显式写入空串。
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
