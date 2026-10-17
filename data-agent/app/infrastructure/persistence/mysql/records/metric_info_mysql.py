from sqlalchemy import String, Text, Boolean
from sqlalchemy.types import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.mysql.records.base import Base


class MetricInfoMySQL(Base):
    __tablename__ = "metric_info"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        comment="指标编码"
    )
    name: Mapped[str] = mapped_column(
        String(128),
        comment="指标名称"
    )
    description: Mapped[str] = mapped_column(
        Text,
        comment="指标描述"
    )
    relevant_columns: Mapped[list] = mapped_column(
        JSON,
        comment="关联字段"
    )
    alias: Mapped[list] = mapped_column(
        JSON,
        comment="指标别名"
    )
    workspace_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    formula_rule: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
        comment="指标计算口径规则"
    )
    enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default="1",
        comment="是否启用（禁用的指标不参与召回与索引构建）"
    )
