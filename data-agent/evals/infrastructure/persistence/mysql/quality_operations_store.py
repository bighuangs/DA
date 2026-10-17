"""基于 SQLAlchemy 的质量运营存储适配器。"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.persistence.mysql.records.feedback_mysql import FeedbackMySQL
from app.infrastructure.persistence.mysql.records.query_trace_mysql import (
    QueryStepLogMySQL,
    QueryTraceMySQL,
    TokenUsageMySQL,
)
from app.infrastructure.persistence.mysql.repositories.meta_mysql_repository import (
    MetaMysqlRepository,
)
from evals.application.dto.quality_operations import (
    FeedbackRecord,
    OverviewSnapshot,
    StepRecord,
    TokenUsageRecord,
    TraceDetailSnapshot,
    TraceRecord,
)


def _trace_record(item: QueryTraceMySQL) -> TraceRecord:
    return TraceRecord(
        id=item.id,
        chat_id=item.chat_id,
        query=item.query,
        status=item.status,
        sql=item.sql,
        row_count=int(item.row_count or 0),
        duration_ms=int(item.duration_ms or 0),
        tt_complete_ms=int(item.tt_complete_ms or 0),
        total_tokens=int(item.total_tokens or 0),
        estimated_cost=float(item.estimated_cost or 0),
        run_fingerprint=item.run_fingerprint,
        version_run_id=item.version_run_id,
        version_complete=bool(item.version_complete),
        cache_hit=bool(item.cache_hit),
        failure_code=item.failure_code,
        query_category=item.query_category,
        follow_up_action=item.follow_up_action,
        follow_up_summary=item.follow_up_summary,
        candidate_summary=item.candidate_summary,
        planning_summary=item.planning_summary,
        retrieval_summary=item.retrieval_summary,
        semantic_repair_summary=item.semantic_repair_summary,
        semantic_coverage_summary=item.semantic_coverage_summary,
        request_deadline_summary=item.request_deadline_summary,
        started_at=item.started_at,
    )


def _step_record(item: QueryStepLogMySQL) -> StepRecord:
    return StepRecord(
        id=item.id,
        trace_id=item.trace_id,
        step_name=item.step_name,
        status=item.status,
        duration_ms=int(item.duration_ms or 0),
        attempt=int(item.attempt or 0),
        error=item.error,
    )


def _usage_record(item: TokenUsageMySQL) -> TokenUsageRecord:
    return TokenUsageRecord(
        trace_id=item.trace_id,
        step_name=item.step_name,
        total_tokens=int(item.total_tokens or 0),
        attempt_id=item.attempt_id,
        route=item.route,
        model=item.model,
        status=item.status,
        token_known=bool(item.token_known),
        timeout_budget_ms=item.timeout_budget_ms,
        affects_core_path=bool(item.affects_core_path),
        error_type=item.error_type,
        reason_code=item.reason_code,
        retry_index=int(item.retry_index or 0),
        callback_observed=bool(item.callback_observed),
        pre_invoke_failure=bool(item.pre_invoke_failure),
        request_deadline_version=item.request_deadline_version,
        deadline_before_ms=item.deadline_before_ms,
        deadline_after_ms=item.deadline_after_ms,
        queue_wait_ms=int(item.queue_wait_ms or 0),
        provider_ms=int(item.provider_ms or 0),
        time_to_first_token_ms=item.time_to_first_token_ms,
        time_to_last_token_ms=item.time_to_last_token_ms,
        token_timing_available=bool(item.token_timing_available),
        completion_stage=item.completion_stage,
        cancel_stage=item.cancel_stage,
        provider_request_id=item.provider_request_id,
        provider_request_id_available=bool(item.provider_request_id_available),
        provider_request_id_unavailable_reason=(
            item.provider_request_id_unavailable_reason
        ),
        token_timing_unavailable_reason=item.token_timing_unavailable_reason,
        outcome_class=item.outcome_class,
        recovery=bool(item.recovery),
        retryable=bool(item.retryable),
        recovery_reserve_ms=int(item.recovery_reserve_ms or 0),
        physical_attempt_budget_ms=item.physical_attempt_budget_ms,
        prompt_tokens=int(item.prompt_tokens or 0),
        completion_tokens=int(item.completion_tokens or 0),
        latency_ms=int(item.latency_ms or 0),
        estimated_cost=float(item.estimated_cost or 0),
    )


def _feedback_record(item: FeedbackMySQL) -> FeedbackRecord:
    return FeedbackRecord(
        id=item.id,
        chat_id=item.chat_id,
        query=item.query,
        sql=item.sql,
        score=int(item.score),
        comment=item.comment,
        created_at=item.created_at,
        review_status=item.review_status,
        review_category=item.review_category,
        review_note=item.review_note,
        source_trace_id=item.source_trace_id,
        candidate_type=item.candidate_type,
        candidate_status=item.candidate_status,
        candidate_question=item.candidate_question,
        candidate_sql=item.candidate_sql,
        candidate_training_id=item.candidate_training_id,
    )


class MySQLQualityOperationsStore:
    """把 workspace 作用域 SQL、ORM 映射和事务封装在基础设施层。"""

    def __init__(self, session: AsyncSession, workspace_id: str):
        self._session = session
        self._workspace_id = workspace_id

    async def load_overview(
        self,
        *,
        cutoff: datetime,
        version: str | None,
        category: str | None,
        status: str | None,
    ) -> OverviewSnapshot:
        trace_conditions = [
            QueryTraceMySQL.workspace_id == self._workspace_id,
            QueryTraceMySQL.started_at >= cutoff,
        ]
        if version:
            trace_conditions.append(QueryTraceMySQL.run_fingerprint == version)
        if category:
            trace_conditions.append(QueryTraceMySQL.query_category == category)
        if status:
            trace_conditions.append(QueryTraceMySQL.status == status)

        statement = select(QueryTraceMySQL).where(*trace_conditions)
        trace_limit = 10000
        trace_count = int(
            (
                await self._session.execute(
                    select(func.count()).select_from(
                        statement.order_by(None).subquery()
                    )
                )
            ).scalar()
            or 0
        )
        trace_models = list(
            (
                await self._session.execute(
                    statement.order_by(QueryTraceMySQL.started_at.desc()).limit(
                        trace_limit
                    )
                )
            ).scalars().all()
        )
        trace_models.reverse()
        trace_ids = [item.id for item in trace_models]
        step_models: list[QueryStepLogMySQL] = []
        usage_models: list[TokenUsageMySQL] = []
        if trace_ids:
            step_models = list(
                (
                    await self._session.execute(
                        select(QueryStepLogMySQL).where(
                            QueryStepLogMySQL.trace_id.in_(trace_ids)
                        )
                    )
                ).scalars().all()
            )
            usage_models = list(
                (
                    await self._session.execute(
                        select(TokenUsageMySQL).where(
                            TokenUsageMySQL.trace_id.in_(trace_ids)
                        )
                    )
                ).scalars().all()
            )

        feedback_statement = (
            select(FeedbackMySQL)
            .where(FeedbackMySQL.workspace_id == self._workspace_id)
            .where(FeedbackMySQL.created_at >= cutoff)
        )
        if version or category or status:
            related_chat_ids = (
                select(QueryTraceMySQL.chat_id)
                .where(*trace_conditions)
                .where(QueryTraceMySQL.chat_id.is_not(None))
                .distinct()
            )
            feedback_statement = feedback_statement.where(
                FeedbackMySQL.chat_id.in_(related_chat_ids)
            )
        feedback_limit = 1000
        feedback_count = int(
            (
                await self._session.execute(
                    select(func.count()).select_from(feedback_statement.subquery())
                )
            ).scalar()
            or 0
        )
        feedback_models = list(
            (
                await self._session.execute(
                    feedback_statement.order_by(FeedbackMySQL.created_at.desc()).limit(
                        feedback_limit
                    )
                )
            ).scalars().all()
        )
        return OverviewSnapshot(
            traces=tuple(_trace_record(item) for item in trace_models),
            steps=tuple(_step_record(item) for item in step_models),
            usages=tuple(_usage_record(item) for item in usage_models),
            feedback=tuple(_feedback_record(item) for item in feedback_models),
            trace_count=trace_count,
            feedback_count=feedback_count,
            trace_limit=trace_limit,
            feedback_limit=feedback_limit,
        )

    async def load_trace(self, trace_id: str) -> TraceDetailSnapshot | None:
        trace = (
            await self._session.execute(
                select(QueryTraceMySQL)
                .where(QueryTraceMySQL.id == trace_id)
                .where(QueryTraceMySQL.workspace_id == self._workspace_id)
            )
        ).scalars().first()
        if trace is None:
            return None
        steps = list(
            (
                await self._session.execute(
                    select(QueryStepLogMySQL)
                    .where(QueryStepLogMySQL.trace_id == trace.id)
                    .order_by(QueryStepLogMySQL.id)
                )
            ).scalars().all()
        )
        usages = list(
            (
                await self._session.execute(
                    select(TokenUsageMySQL)
                    .where(TokenUsageMySQL.trace_id == trace.id)
                    .order_by(TokenUsageMySQL.id)
                )
            ).scalars().all()
        )
        return TraceDetailSnapshot(
            trace=_trace_record(trace),
            steps=tuple(_step_record(item) for item in steps),
            usages=tuple(_usage_record(item) for item in usages),
        )

    async def get_feedback(
        self,
        feedback_id: int,
        *,
        for_update: bool = False,
    ) -> FeedbackRecord | None:
        item = await self._feedback_model(feedback_id, for_update=for_update)
        return _feedback_record(item) if item is not None else None

    async def review_feedback(
        self,
        feedback_id: int,
        *,
        status: str,
        category: str | None,
        note: str,
        reviewer_id: str,
        reviewed_at: datetime,
    ) -> FeedbackRecord | None:
        item = await self._feedback_model(feedback_id)
        if item is None:
            return None
        item.review_status = status
        item.review_category = category
        item.review_note = note[:1000] or None
        item.reviewed_by = reviewer_id
        item.reviewed_at = reviewed_at
        if item.chat_id:
            trace = (
                await self._session.execute(
                    select(QueryTraceMySQL)
                    .where(QueryTraceMySQL.workspace_id == self._workspace_id)
                    .where(QueryTraceMySQL.chat_id == item.chat_id)
                    .order_by(QueryTraceMySQL.started_at.desc())
                )
            ).scalars().first()
            item.source_trace_id = trace.id if trace else None
        await self._session.commit()
        return _feedback_record(item)

    async def draft_candidate(
        self,
        feedback_id: int,
        *,
        candidate_type: str,
        question: str,
        sql: str,
    ) -> FeedbackRecord | None:
        item = await self._feedback_model(feedback_id)
        if item is None:
            return None
        item.candidate_type = candidate_type
        item.candidate_status = "draft"
        item.candidate_question = question[:500]
        item.candidate_sql = sql[:10000]
        item.candidate_training_id = None
        item.candidate_confirmed_by = None
        item.candidate_confirmed_at = None
        await self._session.commit()
        return _feedback_record(item)

    async def confirm_candidate(
        self,
        feedback_id: int,
        *,
        confirmer_id: str,
        confirmed_at: datetime,
    ) -> FeedbackRecord | None:
        item = await self._feedback_model(feedback_id, for_update=True)
        if item is None:
            return None
        if item.candidate_status == "confirmed":
            return _feedback_record(item)
        try:
            if item.candidate_type == "training" and item.candidate_training_id is None:
                training = await MetaMysqlRepository(
                    self._session, self._workspace_id
                ).add_training_info(
                    question=str(item.candidate_question or ""),
                    sql=str(item.candidate_sql or ""),
                    enabled=False,
                )
                item.candidate_training_id = training.id
            item.candidate_status = "confirmed"
            item.candidate_confirmed_by = confirmer_id
            item.candidate_confirmed_at = confirmed_at
            await self._session.commit()
        except Exception:
            await self._session.rollback()
            raise
        return _feedback_record(item)

    async def _feedback_model(
        self,
        feedback_id: int,
        *,
        for_update: bool = False,
    ) -> FeedbackMySQL | None:
        statement = (
            select(FeedbackMySQL)
            .where(FeedbackMySQL.id == feedback_id)
            .where(FeedbackMySQL.workspace_id == self._workspace_id)
        )
        if for_update:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalars().first()
