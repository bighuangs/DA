from datetime import datetime

from sqlalchemy import DateTime, Integer, JSON, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.mysql.records.base import Base


class ConversationDataContextMySQL(Base):
    """会话默认数据源上下文；每次更新都递增 revision。"""

    __tablename__ = "conversation_data_context"
    __table_args__ = (
        UniqueConstraint(
            "owner_id", "workspace_id", "chat_id",
            name="uq_conversation_data_context_scope",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_id: Mapped[str] = mapped_column(String(36), index=True)
    workspace_id: Mapped[str] = mapped_column(String(36), index=True)
    chat_id: Mapped[str] = mapped_column(String(64), index=True)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    source_mode: Mapped[str] = mapped_column(String(32), default="dw_only")
    comparison_intent: Mapped[str | None] = mapped_column(String(48), nullable=True)
    dataset_ids: Mapped[list] = mapped_column(JSON, default=list)
    mapping_confirmation_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    invalidation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class ConversationTurnDataContextMySQL(Base):
    """不可变的回合数据源快照，用于重登恢复与审计。"""

    __tablename__ = "conversation_turn_data_context"
    __table_args__ = (
        UniqueConstraint(
            "owner_id", "workspace_id", "turn_id",
            name="uq_turn_data_context_scope",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_id: Mapped[str] = mapped_column(String(36), index=True)
    workspace_id: Mapped[str] = mapped_column(String(36), index=True)
    chat_id: Mapped[str] = mapped_column(String(64), index=True)
    turn_id: Mapped[str] = mapped_column(String(64), index=True)
    revision: Mapped[int] = mapped_column(Integer)
    context_snapshot: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class SourceMappingMySQL(Base):
    """上传表与 DW/另一上传表之间经用户确认的映射证据。"""

    __tablename__ = "source_mapping"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(36), index=True)
    workspace_id: Mapped[str] = mapped_column(String(36), index=True)
    status: Mapped[str] = mapped_column(String(24), default="proposed", index=True)
    intent: Mapped[str] = mapped_column(String(48))
    dataset_version_ids: Mapped[list] = mapped_column(JSON)
    mappings: Mapped[list] = mapped_column(JSON)
    evidence: Mapped[dict] = mapped_column(JSON)
    fingerprint: Mapped[str] = mapped_column(String(80), index=True)
    rejection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class DatasetResourceLedgerMySQL(Base):
    """上传版本物理资源账本；资源释放失败必须保持可重试状态。"""

    __tablename__ = "dataset_resource_ledger"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_version_id: Mapped[str] = mapped_column(String(64), index=True)
    owner_id: Mapped[str] = mapped_column(String(36), index=True)
    workspace_id: Mapped[str] = mapped_column(String(36), index=True)
    resource_kind: Mapped[str] = mapped_column(String(32), default="dw_table")
    resource_ref: Mapped[str] = mapped_column(String(160))
    status: Mapped[str] = mapped_column(String(32), default="allocated", index=True)
    cleanup_attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    released_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
