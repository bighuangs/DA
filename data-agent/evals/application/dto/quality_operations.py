"""评测质量运营用例使用的持久化无关 DTO。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class TraceRecord:
    id: str
    chat_id: str | None
    query: str
    status: str
    sql: str | None = None
    row_count: int = 0
    duration_ms: int = 0
    tt_complete_ms: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    run_fingerprint: str | None = None
    version_run_id: str | None = None
    version_complete: bool = False
    cache_hit: bool = False
    failure_code: str | None = None
    query_category: str | None = None
    follow_up_action: str | None = None
    follow_up_summary: str | None = None
    candidate_summary: str | None = None
    planning_summary: str | None = None
    retrieval_summary: str | None = None
    semantic_repair_summary: str | None = None
    semantic_coverage_summary: str | None = None
    request_deadline_summary: str | None = None
    started_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class StepRecord:
    trace_id: str
    step_name: str
    status: str
    duration_ms: int = 0
    attempt: int = 0
    error: str | None = None
    id: int | None = None


@dataclass(frozen=True, slots=True)
class TokenUsageRecord:
    trace_id: str
    step_name: str | None
    total_tokens: int = 0
    attempt_id: str | None = None
    route: str | None = None
    model: str | None = None
    status: str = "success"
    token_known: bool = False
    timeout_budget_ms: int | None = None
    affects_core_path: bool = True
    error_type: str | None = None
    reason_code: str | None = None
    retry_index: int = 0
    callback_observed: bool = False
    pre_invoke_failure: bool = False
    request_deadline_version: str | None = None
    deadline_before_ms: int | None = None
    deadline_after_ms: int | None = None
    queue_wait_ms: int = 0
    provider_ms: int = 0
    time_to_first_token_ms: int | None = None
    time_to_last_token_ms: int | None = None
    token_timing_available: bool = False
    completion_stage: str | None = None
    cancel_stage: str | None = None
    provider_request_id: str | None = None
    provider_request_id_available: bool = False
    provider_request_id_unavailable_reason: str | None = None
    token_timing_unavailable_reason: str | None = None
    outcome_class: str | None = None
    recovery: bool = False
    retryable: bool = False
    recovery_reserve_ms: int = 0
    physical_attempt_budget_ms: int | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    estimated_cost: float = 0.0


@dataclass(frozen=True, slots=True)
class FeedbackRecord:
    id: int
    chat_id: str | None
    query: str
    sql: str | None
    score: int
    comment: str | None = None
    created_at: datetime | None = None
    review_status: str | None = None
    review_category: str | None = None
    review_note: str | None = None
    source_trace_id: str | None = None
    candidate_type: str | None = None
    candidate_status: str | None = None
    candidate_question: str | None = None
    candidate_sql: str | None = None
    candidate_training_id: int | None = None


@dataclass(frozen=True, slots=True)
class OverviewSnapshot:
    traces: tuple[TraceRecord, ...]
    steps: tuple[StepRecord, ...]
    usages: tuple[TokenUsageRecord, ...]
    feedback: tuple[FeedbackRecord, ...]
    trace_count: int
    feedback_count: int
    trace_limit: int
    feedback_limit: int


@dataclass(frozen=True, slots=True)
class TraceDetailSnapshot:
    trace: TraceRecord
    steps: tuple[StepRecord, ...]
    usages: tuple[TokenUsageRecord, ...]
