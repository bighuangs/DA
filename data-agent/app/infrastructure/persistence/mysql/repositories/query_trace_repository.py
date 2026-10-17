import json

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.foundation.tracing import QueryTraceCollectorContract
from app.infrastructure.persistence.mysql.records.query_trace_mysql import QueryStepLogMySQL, QueryTraceMySQL, TokenUsageMySQL
from app.infrastructure.security.masking import mask_sql_literals


def _json_object(value: str | None) -> dict:
    try:
        item = json.loads(value or "{}")
    except (TypeError, ValueError):
        return {}
    return item if isinstance(item, dict) else {}


def classify_failure_code(status: str, error: str | None) -> str:
    if status in {"success", "handled"} and not error:
        return "none"
    message = str(error or "").lower()
    if "timeout" in message or "超时" in message:
        return "timeout"
    if any(key in message for key in ("只读", "权限", "越权", "not allowed", "forbidden")):
        return "security_policy"
    if any(key in message for key in ("成本", "explain", "扫描行数")):
        return "cost_guard"
    if any(key in message for key in ("connect", "connection", "连接", "refused")):
        return "dependency_connection"
    if any(key in message for key in ("syntax", "语法", "sql校验", "sql 校验")):
        return "sql_validation"
    if any(key in message for key in ("execute", "执行sql", "执行 sql", "database")):
        return "sql_execution"
    return "pipeline_unknown"


def classify_query_category(query: str) -> str:
    text = str(query or "").lower()
    if "临时数据集" in text or "文件" in text:
        return "adhoc"
    if any(key in text for key in ("同比", "环比", "趋势", "每月", "季度", "年度")):
        return "time_series"
    if any(key in text for key in ("前十", "前五", "最高", "最低", "top")):
        return "ranking"
    if any(key in text for key in ("gmv", "aov", "复购率", "指标", "平均")):
        return "metric"
    if any(key in text for key in ("多少", "总额", "总数", "合计", "统计")):
        return "aggregation"
    return "general"


class QueryTraceRepository:
    def __init__(self, session: AsyncSession, owner_id: str | None = None, workspace_id: str | None = None):
        self.session = session
        self.owner_id = owner_id
        self.workspace_id = workspace_id

    async def save_trace(self, trace: QueryTraceMySQL,
                         steps: list[QueryStepLogMySQL],
                         usages: list[TokenUsageMySQL]) -> None:
        self.session.add(trace)
        self.session.add_all(steps)
        self.session.add_all(usages)
        await self.session.commit()

    async def list_by_chat_id(self, chat_id: str, limit: int = 20) -> list[dict]:
        statement = select(QueryTraceMySQL).where(QueryTraceMySQL.chat_id == chat_id)
        if self.owner_id:
            statement = statement.where(QueryTraceMySQL.owner_id == self.owner_id)
        if self.workspace_id:
            statement = statement.where(QueryTraceMySQL.workspace_id == self.workspace_id)
        traces = (await self.session.execute(
            statement.order_by(desc(QueryTraceMySQL.started_at)).limit(max(1, min(limit, 100)))
        )).scalars().all()
        result = []
        for trace in traces:
            steps = (await self.session.execute(
                select(QueryStepLogMySQL)
                .where(QueryStepLogMySQL.trace_id == trace.id)
                .order_by(QueryStepLogMySQL.id)
            )).scalars().all()
            usages = (await self.session.execute(
                select(TokenUsageMySQL)
                .where(TokenUsageMySQL.trace_id == trace.id)
                .order_by(TokenUsageMySQL.id)
            )).scalars().all()
            result.append({
                "trace_id": trace.id,
                "chat_id": trace.chat_id,
                "query": trace.query,
                "status": trace.status,
                "sql": trace.sql,
                "error": trace.error,
                "model": trace.model,
                "row_count": trace.row_count,
                "duration_ms": trace.duration_ms,
                "ttfse_ms": trace.ttfse_ms,
                "tt_sql_ms": trace.tt_sql_ms,
                "tt_result_ms": trace.tt_result_ms,
                "tt_complete_ms": trace.tt_complete_ms,
                "llm_calls": trace.llm_calls,
                "prompt_tokens": trace.prompt_tokens,
                "completion_tokens": trace.completion_tokens,
                "total_tokens": trace.total_tokens,
                "estimated_cost": trace.estimated_cost,
                "cost_currency": trace.cost_currency,
                "degraded_count": trace.degraded_count,
                "prompt_version": trace.prompt_version,
                "knowledge_version": trace.knowledge_version,
                "schema_version": trace.schema_version,
                "index_version": trace.index_version,
                "model_version": trace.model_version,
                "run_fingerprint": trace.run_fingerprint,
                "version_run_id": trace.version_run_id,
                "version_complete": bool(trace.version_complete),
                "cache_hit": bool(trace.cache_hit),
                "failure_code": trace.failure_code,
                "query_category": trace.query_category,
                "follow_up_action": trace.follow_up_action,
                "follow_up_summary": trace.follow_up_summary,
                "candidate_count": trace.candidate_count,
                "candidate_confidence": trace.candidate_confidence,
                "candidate_band": trace.candidate_band,
                "candidate_selected": trace.candidate_selected,
                "adaptive_candidate_triggered": bool(trace.adaptive_candidate_triggered),
                "candidate_decision": _json_object(trace.candidate_summary),
                "retrieval_summary": _json_object(trace.retrieval_summary),
                "planning_route": _json_object(
                    getattr(trace, "planning_summary", None)
                ),
                "semantic_repair": _json_object(
                    getattr(trace, "semantic_repair_summary", None)
                ),
                "semantic_coverage": _json_object(
                    getattr(trace, "semantic_coverage_summary", None)
                ),
                "request_deadline": _json_object(
                    getattr(trace, "request_deadline_summary", None)
                ),
                "started_at": str(trace.started_at) if trace.started_at else None,
                "finished_at": str(trace.finished_at) if trace.finished_at else None,
                "steps": [{
                    "step_name": step.step_name,
                    "status": step.status,
                    "duration_ms": step.duration_ms,
                    "attempt": step.attempt,
                    "error": step.error,
                    "started_at": str(step.started_at) if step.started_at else None,
                    "finished_at": str(step.finished_at) if step.finished_at else None,
                } for step in steps],
                "usages": [{
                    "attempt_id": usage.attempt_id,
                    "step_name": usage.step_name,
                    "route": usage.route,
                    "model": usage.model,
                    "status": usage.status,
                    "token_known": bool(usage.token_known),
                    "timeout_budget_ms": usage.timeout_budget_ms,
                    "affects_core_path": bool(usage.affects_core_path),
                    "error_type": usage.error_type,
                    "reason_code": usage.reason_code,
                    "retry_index": usage.retry_index,
                    "callback_observed": bool(usage.callback_observed),
                    "callback_missing": not bool(usage.callback_observed),
                    "pre_invoke_failure": bool(usage.pre_invoke_failure),
                    "request_deadline_version": usage.request_deadline_version,
                    "deadline_before_ms": usage.deadline_before_ms,
                    "deadline_after_ms": usage.deadline_after_ms,
                    "queue_wait_ms": usage.queue_wait_ms,
                    "provider_ms": usage.provider_ms,
                    "time_to_first_token_ms": usage.time_to_first_token_ms,
                    "time_to_last_token_ms": usage.time_to_last_token_ms,
                    "token_timing_available": bool(usage.token_timing_available),
                    "completion_stage": usage.completion_stage,
                    "cancel_stage": usage.cancel_stage,
                    "provider_request_id": usage.provider_request_id,
                    "provider_request_id_available": bool(
                        usage.provider_request_id_available
                    ),
                    "provider_request_id_unavailable_reason": (
                        usage.provider_request_id_unavailable_reason
                    ),
                    "token_timing_unavailable_reason": (
                        usage.token_timing_unavailable_reason
                    ),
                    "outcome_class": usage.outcome_class,
                    "recovery": bool(usage.recovery),
                    "retryable": bool(usage.retryable),
                    "recovery_reserve_ms": usage.recovery_reserve_ms,
                    "physical_attempt_budget_ms": (
                        usage.physical_attempt_budget_ms
                    ),
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "latency_ms": usage.latency_ms,
                    "estimated_cost": usage.estimated_cost,
                } for usage in usages],
            })
        return result


def build_trace_records(collector: QueryTraceCollectorContract, *, query: str, chat_id: str,
                        sql: str | None, status: str, row_count: int,
                        error: str | None = None, owner_id: str | None = None,
                        workspace_id: str | None = None,
                        version_snapshot: dict | None = None,
                        cache_hit: bool = False):
    summary = collector.finish(status)
    versions = version_snapshot or {}
    trace = QueryTraceMySQL(
        id=collector.trace_id,
        chat_id=chat_id,
        owner_id=owner_id,
        workspace_id=workspace_id,
        query=query,
        status=summary["status"],
        sql=mask_sql_literals(sql) if sql else sql,
        error=error or collector.error,
        model=(summary["llm_attempts"][0].get("model") if summary["llm_attempts"] else None),
        row_count=row_count,
        duration_ms=summary["duration_ms"],
        ttfse_ms=summary["ttfse_ms"] or 0,
        tt_sql_ms=summary["tt_sql_ms"] or 0,
        tt_result_ms=summary["tt_result_ms"] or 0,
        tt_complete_ms=summary["tt_complete_ms"] or 0,
        llm_calls=summary["llm_calls"],
        prompt_tokens=summary["prompt_tokens"],
        completion_tokens=summary["completion_tokens"],
        total_tokens=summary["total_tokens"],
        estimated_cost=summary["estimated_cost"],
        cost_currency=summary["cost_currency"],
        degraded_count=summary["degraded_count"],
        prompt_version=versions.get("prompt_version"),
        knowledge_version=versions.get("knowledge_version"),
        schema_version=versions.get("schema_version"),
        index_version=versions.get("index_version"),
        model_version=versions.get("model_version") or (
            summary["llm_attempts"][0].get("model") if summary["llm_attempts"] else None
        ),
        run_fingerprint=versions.get("run_fingerprint"),
        version_run_id=versions.get("version_run_id"),
        version_complete=bool(versions.get("version_complete")),
        cache_hit=bool(cache_hit),
        failure_code=classify_failure_code(status, error or collector.error),
        query_category=classify_query_category(query),
        follow_up_action=collector.follow_up_action,
        follow_up_summary=collector.follow_up_summary,
        candidate_count=summary["candidate_count"],
        candidate_confidence=summary["candidate_confidence"],
        candidate_band=summary["candidate_band"],
        candidate_selected=(summary["candidate_decision"].get("selected") or "A"),
        adaptive_candidate_triggered=bool(summary["candidate_decision"].get("adaptive_triggered")),
        candidate_summary=json.dumps(summary["candidate_decision"], ensure_ascii=False),
        retrieval_summary=json.dumps(
            {"routes": summary.get("retrieval_metrics") or []}, ensure_ascii=False
        ),
        planning_summary=json.dumps(
            summary.get("planning_route") or {}, ensure_ascii=False
        ),
        semantic_repair_summary=json.dumps(
            summary.get("semantic_repair") or {}, ensure_ascii=False
        ),
        semantic_coverage_summary=json.dumps(
            summary.get("semantic_coverage") or {}, ensure_ascii=False
        ),
        request_deadline_summary=json.dumps(
            summary.get("request_deadline") or {}, ensure_ascii=False
        ),
        started_at=collector.started_at,
        finished_at=collector.finished_at,
    )
    attempts: dict[str, int] = {}
    steps = []
    for step in collector.steps:
        attempts[step.name] = attempts.get(step.name, 0) + 1
        steps.append(QueryStepLogMySQL(
            trace_id=collector.trace_id,
            step_name=step.name,
            status=step.status,
            duration_ms=step.duration_ms,
            error=step.error,
            attempt=attempts[step.name],
            started_at=step.started_at,
            finished_at=step.finished_at,
        ))
    usages = [TokenUsageMySQL(
        trace_id=collector.trace_id,
        attempt_id=usage.get("attempt_id"),
        step_name=usage.get("node_id"),
        route=usage.get("route"),
        model=usage.get("model"),
        status=usage.get("status") or "error",
        token_known=bool(usage.get("token_known")),
        timeout_budget_ms=usage.get("timeout_budget_ms"),
        affects_core_path=bool(usage.get("affects_core_path")),
        error_type=usage.get("error_type"),
        reason_code=usage.get("reason_code"),
        retry_index=int(usage.get("retry_index") or 0),
        callback_observed=bool(usage.get("callback_observed")),
        pre_invoke_failure=bool(usage.get("pre_invoke_failure")),
        request_deadline_version=usage.get("request_deadline_version"),
        deadline_before_ms=usage.get("deadline_before_ms"),
        deadline_after_ms=usage.get("deadline_after_ms"),
        queue_wait_ms=int(usage.get("queue_wait_ms") or 0),
        provider_ms=int(usage.get("provider_ms") or 0),
        time_to_first_token_ms=usage.get("time_to_first_token_ms"),
        time_to_last_token_ms=usage.get("time_to_last_token_ms"),
        token_timing_available=bool(usage.get("token_timing_available")),
        completion_stage=usage.get("completion_stage"),
        cancel_stage=usage.get("cancel_stage"),
        provider_request_id=usage.get("provider_request_id"),
        provider_request_id_available=bool(
            usage.get("provider_request_id_available")
        ),
        provider_request_id_unavailable_reason=usage.get(
            "provider_request_id_unavailable_reason"
        ),
        token_timing_unavailable_reason=usage.get(
            "token_timing_unavailable_reason"
        ),
        outcome_class=usage.get("outcome_class"),
        recovery=bool(usage.get("recovery")),
        retryable=bool(usage.get("retryable")),
        recovery_reserve_ms=int(usage.get("recovery_reserve_ms") or 0),
        physical_attempt_budget_ms=usage.get(
            "physical_attempt_budget_ms"
        ),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        total_tokens=int(usage.get("total_tokens") or 0),
        latency_ms=int(usage.get("latency_ms") or 0),
        estimated_cost=float(usage.get("estimated_cost") or 0),
    ) for usage in summary["llm_attempts"]]
    return summary, trace, steps, usages
