"""Append-only, tenant-scoped summary segment authority for TODO-96."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, JSON, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.mysql.records.base import Base


class ConversationSummarySegmentMySQL(Base):
    __tablename__ = "conversation_summary_segment"
    __table_args__ = (
        UniqueConstraint(
            "owner_id",
            "workspace_id",
            "chat_id",
            "level",
            "source_content_hash",
            "extractor_version",
            name="uq_conversation_summary_identity",
        ),
        Index(
            "ix_conversation_summary_scope_chat_status",
            "owner_id",
            "workspace_id",
            "chat_id",
            "status",
            "level",
        ),
        Index(
            "ix_conversation_summary_source_range",
            "owner_id",
            "workspace_id",
            "chat_id",
            "source_start_message_id",
            "source_end_message_id",
        ),
    )

    summary_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(36), nullable=False)
    workspace_id: Mapped[str] = mapped_column(String(36), nullable=False)
    chat_id: Mapped[str] = mapped_column(String(64), nullable=False)
    level: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    source_start_message_id: Mapped[int] = mapped_column(Integer, nullable=False)
    source_end_message_id: Mapped[int] = mapped_column(Integer, nullable=False)
    source_message_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    child_summary_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    source_content_hash: Mapped[str] = mapped_column(String(80), nullable=False)
    content_masked: Mapped[str] = mapped_column(Text, nullable=False)
    structured_facts: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    open_items: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    source_token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    summary_token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    summary_schema_version: Mapped[str] = mapped_column(String(80), nullable=False)
    extractor_version: Mapped[str] = mapped_column(String(80), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(80), nullable=True)
    projection_version: Mapped[str] = mapped_column(String(160), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="active")
    superseded_by_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    valid_to: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
