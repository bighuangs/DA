from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.mysql.records.base import Base


class MaskingRuleMySQL(Base):
    """后台维护的结果字段脱敏规则（meta 库）。"""

    __tablename__ = "masking_rule"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    workspace_id: Mapped[str | None] = mapped_column(ForeignKey("workspace.id", ondelete="CASCADE"), nullable=True, index=True)
    column_pattern: Mapped[str] = mapped_column(String(255), index=True)
    data_kind: Mapped[str] = mapped_column(String(32), default="generic")
    strategy: Mapped[str] = mapped_column(String(32), default="mask")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    priority: Mapped[int] = mapped_column(Integer, default=100)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
