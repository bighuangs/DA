"""Recoverable MySQL checkpoint for Redis working memory."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, JSON, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.mysql.records.base import Base


class ConversationWorkingMemoryCheckpointMySQL(Base):
    __tablename__ = "conversation_working_memory_checkpoint"
    __table_args__ = (
        UniqueConstraint(
            "owner_id", "workspace_id", "chat_id",
            name="uq_conversation_working_memory_scope_chat",
        ),
        Index("ix_conversation_working_memory_expiry", "expires_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    owner_id: Mapped[str] = mapped_column(String(36), nullable=False)
    workspace_id: Mapped[str] = mapped_column(String(36), nullable=False)
    chat_id: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    structured_state: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    rolling_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    recent_message_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    schema_version: Mapped[str] = mapped_column(String(80), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
