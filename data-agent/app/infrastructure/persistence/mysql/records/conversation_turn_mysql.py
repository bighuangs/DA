"""User-visible transcript turn index backed by existing history snapshots."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.mysql.records.base import Base


class ConversationTurnMySQL(Base):
    """One submitted user turn, including pending and failed attempts."""

    __tablename__ = "conversation_turn"
    __table_args__ = (
        UniqueConstraint(
            "owner_id",
            "workspace_id",
            "turn_id",
            name="uq_conversation_turn_scope_turn",
        ),
        UniqueConstraint(
            "source_history_id",
            name="uq_conversation_turn_source_history",
        ),
        UniqueConstraint(
            "source_message_id",
            name="uq_conversation_turn_source_message",
        ),
        Index(
            "ix_conversation_turn_scope_chat_order",
            "owner_id",
            "workspace_id",
            "chat_id",
            "created_at",
            "id",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    turn_id: Mapped[str] = mapped_column(String(80), nullable=False)
    chat_id: Mapped[str] = mapped_column(String(64), nullable=False)
    owner_id: Mapped[str] = mapped_column(String(36), nullable=False)
    workspace_id: Mapped[str] = mapped_column(String(36), nullable=False)
    trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_history_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    source_message_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    user_text: Mapped[str] = mapped_column(Text, nullable=False)
    assistant_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(24), nullable=False, default="pending"
    )
    replay_fidelity: Mapped[str] = mapped_column(
        String(32), nullable=False, default="message_only"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
