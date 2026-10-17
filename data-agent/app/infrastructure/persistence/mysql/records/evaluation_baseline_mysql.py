from datetime import datetime

from sqlalchemy import Boolean, DateTime, Index, Integer, JSON, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.mysql.records.base import Base


class EvaluationBaselineMySQL(Base):
    """Transactional workspace baseline registry and review audit state."""

    __tablename__ = "evaluation_baseline"
    __table_args__ = (
        UniqueConstraint("workspace_id", "run_id", name="uq_evaluation_baseline_workspace_run"),
        Index(
            "ix_evaluation_baseline_scope_status_reviewed",
            "workspace_id",
            "dataset_suite",
            "status",
            "reviewed_at",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    workspace_id: Mapped[str] = mapped_column(String(36), nullable=False)
    dataset_suite: Mapped[str] = mapped_column(String(64), nullable=False)
    run_id: Mapped[str] = mapped_column(String(32), nullable=False)
    artifact: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="candidate")
    eligible: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="0")
    checklist: Mapped[dict] = mapped_column(JSON, nullable=False)
    run_fingerprint: Mapped[str | None] = mapped_column(String(80), nullable=True)
    reviewed_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    superseded_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
