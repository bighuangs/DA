from sqlalchemy import DateTime, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.mysql.records.base import Base


class SchemaSyncLogMySQL(Base):
    """数据源 Schema 同步快照与差异结果。"""

    __tablename__ = "schema_sync_log"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(128), comment="数据源标识")
    status: Mapped[str] = mapped_column(String(32), comment="success/failed")
    added_tables: Mapped[int] = mapped_column(Integer, default=0)
    removed_tables: Mapped[int] = mapped_column(Integer, default=0)
    changed_tables: Mapped[int] = mapped_column(Integer, default=0)
    added_columns: Mapped[int] = mapped_column(Integer, default=0)
    removed_columns: Mapped[int] = mapped_column(Integer, default=0)
    changed_columns: Mapped[int] = mapped_column(Integer, default=0)
    diff: Mapped[dict] = mapped_column(JSON, default=dict)
    snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(DateTime, server_default=func.now())
