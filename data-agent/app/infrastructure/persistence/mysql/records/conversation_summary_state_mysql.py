"""Serialized summary coverage watermark for a tenant-scoped chat."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.mysql.records.base import Base


class ConversationSummaryStateMySQL(Base):
    __tablename__ = "conversation_summary_state"
    __table_args__ = (
        UniqueConstraint(
            "owner_id",
            "workspace_id",
            "chat_id",
            name="uq_conversation_summary_state_scope_chat",
        ),
        Index(
            "ix_conversation_summary_state_scope",
            "owner_id",
            "workspace_id",
            "chat_id",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    owner_id: Mapped[str] = mapped_column(String(36), nullable=False)
    workspace_id: Mapped[str] = mapped_column(String(36), nullable=False)
    chat_id: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    covered_through_message_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    active_segment_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_source_hash: Mapped[str | None] = mapped_column(String(80), nullable=True)
    schema_version: Mapped[str] = mapped_column(String(80), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )
