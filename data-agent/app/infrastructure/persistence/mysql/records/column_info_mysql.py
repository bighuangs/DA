from sqlalchemy import String, Text, Boolean, ForeignKey
from sqlalchemy.types import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.mysql.records.base import Base


class ColumnInfoMySQL(Base):
    __tablename__ = "column_info"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        comment="列编号"
    )
    name: Mapped[str] = mapped_column(
        String(128),
        comment="列名称"
    )
    type: Mapped[str] = mapped_column(
        String(64),
        comment="数据类型"
    )
    role: Mapped[str] = mapped_column(
        String(32),
        comment="列类型(primary_key,foreign_key,measure,dimension)"
    )
    examples: Mapped[list] = mapped_column(
        JSON,
        comment="数据示例"
    )
    description: Mapped[str] = mapped_column(
        Text,
        comment="列描述"
    )
    alias: Mapped[list] = mapped_column(
        JSON,
        comment="列别名"
    )
    table_id: Mapped[str] = mapped_column(
        ForeignKey("table_info.id", ondelete="CASCADE"),
        comment="所属表编号"
    )
    sync: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default="0",
        comment="是否同步字段值到 ES"
    )
    is_display: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default="0",
        comment="是否为实体默认可读展示字段"
    )
