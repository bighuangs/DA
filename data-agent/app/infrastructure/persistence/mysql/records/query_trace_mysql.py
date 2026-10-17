from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.mysql.records.base import Base


class QueryTraceMySQL(Base):
    """一次问数请求的主链路记录。"""

    __tablename__ = "query_trace"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    chat_id: Mapped[str] = mapped_column(String(64), index=True)
    owner_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    workspace_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    query: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), comment="success/failed")
    sql: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    row_count: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    ttfse_ms: Mapped[int] = mapped_column(Integer, default=0)
    tt_sql_ms: Mapped[int] = mapped_column(Integer, default=0)
    tt_result_ms: Mapped[int] = mapped_column(Integer, default=0)
    tt_complete_ms: Mapped[int] = mapped_column(Integer, default=0)
    llm_calls: Mapped[int] = mapped_column(Integer, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost: Mapped[float] = mapped_column(Float, default=0.0)
    cost_currency: Mapped[str] = mapped_column(String(16), default="USD")
    degraded_count: Mapped[int] = mapped_column(Integer, default=0)
    prompt_version: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    knowledge_version: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    schema_version: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    index_version: Mapped[str | None] = mapped_column(String(80), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    run_fingerprint: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    version_run_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    version_complete: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0", index=True)
    failure_code: Mapped[str] = mapped_column(String(64), default="none", server_default="none", index=True)
    query_category: Mapped[str] = mapped_column(String(64), default="general", server_default="general", index=True)
    follow_up_action: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    follow_up_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    candidate_count: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    candidate_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    candidate_band: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    candidate_selected: Mapped[str | None] = mapped_column(String(8), nullable=True)
    adaptive_candidate_triggered: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0", index=True)
    candidate_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieval_summary: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="隐私安全的 Dense/Sparse/RRF 召回证据"
    )
    planning_summary: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="隐私安全的 SQL 规划路由与分段耗时证据"
    )
    semantic_repair_summary: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="隐私安全的语义 violation、修复规则、hash 与耗时证据"
    )
    semantic_coverage_summary: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="TODO-92 结构槽位覆盖证据，不含问题或筛选字面量"
    )
    request_deadline_summary: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="TODO-93 共享截止时间相对预算，不含绝对单调时钟"
    )
    started_at: Mapped[str] = mapped_column(DateTime)
    finished_at: Mapped[str | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[str] = mapped_column(DateTime, server_default=func.now())


class QueryStepLogMySQL(Base):
    """一次请求中的阶段/节点摘要，不保存完整 Prompt 或结果行。"""

    __tablename__ = "query_step_log"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    step_name: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32), comment="running/success/failed")
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    input_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[str] = mapped_column(DateTime)
    finished_at: Mapped[str | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[str] = mapped_column(DateTime, server_default=func.now())


class TokenUsageMySQL(Base):
    """一次模型尝试及其 token 用量；未知 token 保持显式 unknown。"""

    __tablename__ = "token_usage"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    attempt_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    step_name: Mapped[str] = mapped_column(String(128), nullable=True)
    route: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="success", server_default="success", index=True)
    token_known: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    timeout_budget_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    affects_core_path: Mapped[bool] = mapped_column(Boolean, default=True, server_default="1", index=True)
    error_type: Mapped[str | None] = mapped_column(String(120), nullable=True)
    reason_code: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    retry_index: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    callback_observed: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    pre_invoke_failure: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    request_deadline_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    deadline_before_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    deadline_after_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    queue_wait_ms: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    provider_ms: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    time_to_first_token_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    time_to_last_token_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    token_timing_available: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    completion_stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    cancel_stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    provider_request_id: Mapped[str | None] = mapped_column(String(160), nullable=True)
    provider_request_id_available: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="0"
    )
    provider_request_id_unavailable_reason: Mapped[str | None] = mapped_column(
        String(120), nullable=True
    )
    token_timing_unavailable_reason: Mapped[str | None] = mapped_column(
        String(120), nullable=True
    )
    outcome_class: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    recovery: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    retryable: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    recovery_reserve_ms: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0"
    )
    physical_attempt_budget_ms: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[str] = mapped_column(DateTime, server_default=func.now())
