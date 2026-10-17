"""质量运营 Application 用例依赖的存储与目录端口。"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from evals.application.dto.quality_operations import (
    FeedbackRecord,
    OverviewSnapshot,
    TraceDetailSnapshot,
)


class QualityOperationsStore(Protocol):
    async def load_overview(
        self,
        *,
        cutoff: datetime,
        version: str | None,
        category: str | None,
        status: str | None,
    ) -> OverviewSnapshot: ...

    async def load_trace(self, trace_id: str) -> TraceDetailSnapshot | None: ...

    async def get_feedback(
        self,
        feedback_id: int,
        *,
        for_update: bool = False,
    ) -> FeedbackRecord | None: ...

    async def review_feedback(
        self,
        feedback_id: int,
        *,
        status: str,
        category: str | None,
        note: str,
        reviewer_id: str,
        reviewed_at: datetime,
    ) -> FeedbackRecord | None: ...

    async def draft_candidate(
        self,
        feedback_id: int,
        *,
        candidate_type: str,
        question: str,
        sql: str,
    ) -> FeedbackRecord | None: ...

    async def confirm_candidate(
        self,
        feedback_id: int,
        *,
        confirmer_id: str,
        confirmed_at: datetime,
    ) -> FeedbackRecord | None: ...


class EvaluationCatalogPort(Protocol):
    async def approved_entries(self, workspace_id: str) -> list[dict]: ...

    async def list_runs(
        self,
        workspace_id: str,
        limit: int = 20,
        difficulty: str | None = None,
    ) -> list[dict]: ...
