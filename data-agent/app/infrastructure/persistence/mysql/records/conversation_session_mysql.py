"""Authoritative conversation directory rows for user-visible chat recovery."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.mysql.records.base import Base


class ConversationSessionMySQL(Base):
    """A lightweight, tenant-scoped directory entry for one chat."""

    __tablename__ = "conversation_session"
    __table_args__ = (
        UniqueConstraint(
            "owner_id",
            "workspace_id",
            "chat_id",
            name="uq_conversation_session_scope_chat",
        ),
        Index(
            "ix_conversation_session_scope_status_order",
            "owner_id",
            "workspace_id",
            "status",
            "last_message_at",
            "id",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    chat_id: Mapped[str] = mapped_column(String(64), nullable=False)
    owner_id: Mapped[str] = mapped_column(String(36), nullable=False)
    workspace_id: Mapped[str] = mapped_column(String(36), nullable=False)
    title: Mapped[str] = mapped_column(String(160), nullable=False)
    preview: Mapped[str] = mapped_column(String(320), nullable=False, default="")
    status: Mapped[str] = mapped_column(
        String(24), nullable=False, default="active"
    )
    last_turn_status: Mapped[str] = mapped_column(
        String(24), nullable=False, default="pending"
    )
    turn_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    context_reset_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    last_message_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    archived_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
