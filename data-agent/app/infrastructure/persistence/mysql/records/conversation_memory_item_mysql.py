"""Authoritative long-term memory item; search stores hold rebuildable projections."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Index, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.mysql.records.base import Base


class ConversationMemoryItemMySQL(Base):
    __tablename__ = "conversation_memory_item"
    __table_args__ = (
        Index(
            "ix_conversation_memory_scope_status_type",
            "owner_id", "workspace_id", "status", "memory_type",
        ),
        Index("ix_conversation_memory_validity", "valid_to", "expires_at"),
        Index("ix_conversation_memory_chat", "owner_id", "workspace_id", "chat_id"),
    )

    memory_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(36), nullable=False)
    workspace_id: Mapped[str] = mapped_column(String(36), nullable=False)
    chat_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    memory_type: Mapped[str] = mapped_column(String(32), nullable=False)
    content_masked: Mapped[str] = mapped_column(Text, nullable=False)
    structured_payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    source_message_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    data_source_id: Mapped[str] = mapped_column(String(80), nullable=False, default="dw-main")
    schema_version: Mapped[str | None] = mapped_column(String(80), nullable=True)
    sql_fingerprint: Mapped[str | None] = mapped_column(String(80), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    verification_status: Mapped[str] = mapped_column(
        String(24), nullable=False, default="verified"
    )
    sensitivity: Mapped[str] = mapped_column(String(24), nullable=False, default="masked")
    valid_from: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    supersedes_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    extractor_version: Mapped[str] = mapped_column(String(80), nullable=False)
    embedding_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    projection_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
