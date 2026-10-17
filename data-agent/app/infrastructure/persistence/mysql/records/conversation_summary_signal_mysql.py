"""Versioned, auditable lexical signals belonging to one summary segment."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Index, Integer, JSON, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.mysql.records.base import Base


class ConversationSummarySignalMySQL(Base):
    __tablename__ = "conversation_summary_signal"
    __table_args__ = (
        UniqueConstraint(
            "summary_id",
            "signal_type",
            "normalized_value",
            "extractor_version",
            name="uq_conversation_summary_signal_identity",
        ),
        Index(
            "ix_conversation_summary_signal_lookup",
            "owner_id",
            "workspace_id",
            "chat_id",
            "status",
            "normalized_value",
        ),
        Index("ix_conversation_summary_signal_summary", "summary_id", "status"),
    )

    signal_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    summary_id: Mapped[str] = mapped_column(String(36), nullable=False)
    owner_id: Mapped[str] = mapped_column(String(36), nullable=False)
    workspace_id: Mapped[str] = mapped_column(String(36), nullable=False)
    chat_id: Mapped[str] = mapped_column(String(64), nullable=False)
    signal_type: Mapped[str] = mapped_column(String(24), nullable=False)
    normalized_value: Mapped[str] = mapped_column(String(255), nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    source_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    source_message_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    extractor_version: Mapped[str] = mapped_column(String(80), nullable=False)
    sensitivity: Mapped[str] = mapped_column(String(24), nullable=False, default="masked")
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
