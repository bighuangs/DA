"""把生产 NL2SQL Runtime 适配为统一评测 ``EvaluationTarget``。"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any, cast

from langchain_core.callbacks import BaseCallbackHandler

from app.adapters.outbound.agent.nl2sql_capabilities import (
    Nl2SqlIndexVersionAdapter,
    Nl2SqlLLMAdapter,
    build_nl2sql_settings,
)
from app.adapters.outbound.agent.nl2sql_runtime import Nl2SqlApplicationRuntime
from app.agent_system.agents.nl2sql.context import DataAgentContext
from app.agent_system.agents.nl2sql.runtime import create_nl2sql_graph_runner
from app.agent_system.agents.nl2sql.settings import Nl2SqlSettings
from app.agent_system.runtime import GraphRunner
from app.application.ports.agent_runtime import (
    AgentEventKind,
    AgentExecutionRequest,
    QueryAgentRuntime,
)
from app.domain.access.policy import PolicySnapshot
from evals.contracts import (
    EventRecord,
    EventStatus,
    RetrievedEvidence,
    TargetResponse,
)
from evals.datasets import DatasetCase
from evals.tracing import TraceSink


_NODE_ID = re.compile(r"[^A-Za-z0-9._:-]+")


class Nl2SqlEvaluationTargetError(RuntimeError):
    """生产等价 NL2SQL 执行未形成成功终态。"""


class _StateCaptureCallback(BaseCallbackHandler):
    """从 LangGraph callback 收集公开 State delta，不保存私有推理。"""

    def __init__(self) -> None:
        self.state: dict[str, object] = {}

    def on_chain_end(self, outputs: object, **_: object) -> None:
        if isinstance(outputs, Mapping):
            self.state.update({str(key): value for key, value in outputs.items()})


class _NodeEventTranslator:
    """把 Runtime 的 node_start/node_end 配对为 typed eval EventRecord。"""

    def __init__(self, trace: TraceSink, started_at: datetime) -> None:
        self._trace = trace
        self._started_at = started_at
        self._starts: dict[str, list[tuple[int, int]]] = {}
        self._ordinal = 0

    def consume(self, value: Mapping[str, object]) -> None:
        node_id = str(value.get("node_id") or "unknown")
        event = str(value.get("event") or "")
        if event == "node_start":
            self._ordinal += 1
            self._starts.setdefault(node_id, []).append(
                (max(0, int(value.get("start_ms") or 0)), self._ordinal)
            )
            return
        if event != "node_end":
            return
        pending = self._starts.get(node_id) or []
        start_ms, ordinal = (
            pending.pop(0)
            if pending
            else (max(0, int(value.get("start_ms") or 0)), self._next_ordinal())
        )
        self._emit(
            node_id=node_id,
            ordinal=ordinal,
            start_ms=start_ms,
            end_ms=max(start_ms, int(value.get("end_ms") or start_ms)),
            status=str(value.get("status") or "success"),
            error=value.get("error"),
        )

    def fail_pending(self, error_message: str) -> None:
        """执行中断时关闭仍未配对的节点事件，避免 Trace 假装成功。"""

        for node_id, pending in tuple(self._starts.items()):
            while pending:
                start_ms, ordinal = pending.pop(0)
                self._emit(
                    node_id=node_id,
                    ordinal=ordinal,
                    start_ms=start_ms,
                    end_ms=start_ms,
                    status="failed",
                    error=error_message,
                )

    def _next_ordinal(self) -> int:
        self._ordinal += 1
        return self._ordinal

    def _emit(
        self,
        *,
        node_id: str,
        ordinal: int,
        start_ms: int,
        end_ms: int,
        status: str,
        error: object,
    ) -> None:
        normalized_status = status.lower()
        event_status = (
            EventStatus.SUCCEEDED
            if normalized_status in {"success", "succeeded"}
            else EventStatus.CANCELLED
            if normalized_status in {"cancelled", "canceled"}
            else EventStatus.FAILED
        )
        safe_node = _NODE_ID.sub("-", node_id).strip("-")[:72] or "unknown"
        self._trace.emit(
            EventRecord(
                evaluation_id=self._trace.evaluation_id,
                case_id=self._trace.case_id,
                trace_id=self._trace.trace_id,
                event_id=f"node-{ordinal}-{safe_node}",
                event_type=f"agent.node.{node_id}",
                status=event_status,
                started_at=self._started_at + timedelta(milliseconds=start_ms),
                ended_at=self._started_at + timedelta(milliseconds=end_ms),
                error_type=("NodeExecutionError" if event_status is EventStatus.FAILED else None),
                error_message=(str(error)[:500] if error is not None else None),
                attributes={"node_id": node_id},
            )
        )


def build_nl2sql_evaluation_capabilities(
    *,
    dw_mysql_repo: object,
    meta_mysql_repo: object,
    value_es_repo: object,
    column_qdrant_repo: object,
    metric_qdrant_repo: object,
    training_qdrant_repo: object,
    embedding_client: object,
    llm: object | None = None,
    index_versions: object | None = None,
    settings: Nl2SqlSettings | None = None,
) -> dict[str, object]:
    """冻结与生产组合根同构的 NL2SQL 固定能力；本函数不连接外部服务。"""

    return {
        "dw_mysql_repo": dw_mysql_repo,
        "meta_mysql_repo": meta_mysql_repo,
        "value_es_repo": value_es_repo,
        "column_qdrant_repo": column_qdrant_repo,
        "metric_qdrant_repo": metric_qdrant_repo,
        "training_qdrant_repo": training_qdrant_repo,
        "embedding_client": embedding_client,
        "llm": llm if llm is not None else Nl2SqlLLMAdapter(),
        "index_versions": (
            index_versions if index_versions is not None else Nl2SqlIndexVersionAdapter()
        ),
        "settings": settings if settings is not None else build_nl2sql_settings(),
    }


def build_nl2sql_evaluation_context(
    *,
    access_policy: PolicySnapshot,
    workspace_id: str,
    force_model_sql: bool | None = None,
    force_model_sql_reason: str | None = None,
    model_sql_fallback_enabled: bool | None = None,
    **capabilities: object,
) -> DataAgentContext:
    """组合 legacy 节点所需完整 Context，同时保留请求级授权事实。"""

    values = {
        **build_nl2sql_evaluation_capabilities(**capabilities),
        "access_policy": access_policy,
        "workspace_id": str(workspace_id),
    }
    if force_model_sql is not None:
        values["force_model_sql"] = bool(force_model_sql)
    if force_model_sql_reason is not None:
        values["force_model_sql_reason"] = str(force_model_sql_reason)
    if model_sql_fallback_enabled is not None:
        values["model_sql_fallback_enabled"] = bool(model_sql_fallback_enabled)
    return cast(DataAgentContext, values)


def build_nl2sql_evaluation_state(
    query: str,
    *,
    performance_started_at: float | None = None,
    analysis_enabled: bool = False,
) -> dict[str, object]:
    """创建与 QueryService 同构、可安全序列化的单 Case 初始 State。"""

    return {
        "query": str(query),
        "chat_history": [],
        "memory_context": {},
        "analysis_enabled": bool(analysis_enabled),
        "performance_started_at": (
            time.perf_counter()
            if performance_started_at is None
            else float(performance_started_at)
        ),
        "node_metrics": [],
        "degradations": [],
        "retrieval_metrics": [],
        "llm_attempts": [],
        "semantic_repair_ledger": [],
        "repair_attempts": [],
        "data_source_context": {
            "schema_version": "data-source-context-v1",
            "source_mode": "dw_only",
            "datasets": [],
            "revision": 1,
        },
        "source_table_infos": [],
        "upload_profiles": [],
        "source_mapping": {},
        "source_sqls": {},
    }


class Nl2SqlEvaluationTarget:
    """通过生产 Application Runtime 执行冻结 Case 的 concrete Target。"""

    def __init__(
        self,
        runtime: QueryAgentRuntime,
        *,
        workspace_id: str,
        access_policy: PolicySnapshot,
        actor_id: str = "evaluation-user",
        force_model_sql: bool = False,
        model_sql_fallback_enabled: bool = True,
        runner: GraphRunner | None = None,
        fixed_capabilities: Mapping[str, object] | None = None,
    ) -> None:
        self._runtime = runtime
        self._workspace_id = str(workspace_id)
        self._access_policy = access_policy
        self._actor_id = str(actor_id)
        self._force_model_sql = bool(force_model_sql)
        self._model_sql_fallback_enabled = bool(model_sql_fallback_enabled)
        self._runner = runner
        self._fixed_capabilities = MappingProxyType(dict(fixed_capabilities or {}))

    @property
    def runtime(self) -> QueryAgentRuntime:
        return self._runtime

    @property
    def runner(self) -> GraphRunner | None:
        return self._runner

    @property
    def fixed_capabilities(self) -> Mapping[str, object]:
        return self._fixed_capabilities

    async def evaluate(self, case: DatasetCase, trace: TraceSink) -> TargetResponse:
        started_clock = time.perf_counter()
        started_at = datetime.now(UTC)
        initial_state = build_nl2sql_evaluation_state(
            case.query,
            performance_started_at=started_clock,
            analysis_enabled=False,
        )
        plan = self._runtime.prepare(
            performance_started_at=started_clock,
            initial_state=initial_state,
        )
        state_capture = _StateCaptureCallback()
        request_id = f"eval-{case.case_id}-{uuid.uuid4().hex[:12]}"
        request = AgentExecutionRequest(
            request_id=request_id,
            query=case.query,
            actor_id=self._actor_id,
            workspace_id=self._workspace_id,
            conversation_id=f"eval-{case.case_id}",
            trace_id=trace.trace_id,
            roles=(str(self._access_policy.role or "viewer"),),
            allowed_data_sources=(),
            analysis_enabled=False,
            plan=plan,
            capabilities={
                "access_policy": self._access_policy,
                "workspace_id": self._workspace_id,
                "force_model_sql": self._force_model_sql,
                "force_model_sql_reason": (
                    "evaluation_target_contract" if self._force_model_sql else ""
                ),
                "model_sql_fallback_enabled": self._model_sql_fallback_enabled,
            },
            callbacks=(state_capture,),
        )
        translator = _NodeEventTranslator(trace, started_at)
        observed: dict[str, object] = {}
        attempts: list[Mapping[str, object]] = []
        failure: str | None = None
        terminal_status: str | None = None

        async for event in self._runtime.stream(request):
            if event.kind is AgentEventKind.ERROR:
                failure = event.error_message or "Agent Runtime 返回失败事件"
            if event.kind is AgentEventKind.STATE:
                event_state = event.payload.get("state")
                if isinstance(event_state, Mapping):
                    observed.update(event_state)
            chunk = event.payload.get("chunk")
            if isinstance(chunk, Mapping):
                trace_event = chunk.get("trace_event")
                if isinstance(trace_event, Mapping):
                    translator.consume(trace_event)
                llm_attempt = chunk.get("llm_attempt")
                if isinstance(llm_attempt, Mapping):
                    attempts.append(llm_attempt)
                for key in ("sql", "result", "analysis", "error"):
                    if key in chunk:
                        observed[key] = chunk[key]
                if chunk.get("error") is not None:
                    failure = str(chunk["error"])
            if event.kind is AgentEventKind.SNAPSHOT:
                snapshot = event.payload.get("snapshot")
                if isinstance(snapshot, Mapping):
                    terminal_status = str(snapshot.get("status") or "")
                    snapshot_state = snapshot.get("state")
                    if isinstance(snapshot_state, Mapping):
                        observed.update(snapshot_state)
            elif event.kind is AgentEventKind.COMPLETED:
                terminal_status = str(event.payload.get("status") or terminal_status or "")

        captured = dict(state_capture.state)
        captured.update(observed)
        state_attempts = captured.get("llm_attempts")
        if isinstance(state_attempts, Sequence) and not isinstance(
            state_attempts, (str, bytes)
        ):
            attempts.extend(item for item in state_attempts if isinstance(item, Mapping))
        if terminal_status not in {"succeeded", "success"}:
            failure = failure or f"Agent Runtime 终态异常: {terminal_status or 'unknown'}"
        if failure:
            translator.fail_pending(failure)
            raise Nl2SqlEvaluationTargetError(failure)

        token_summary = _token_summary(attempts)
        result = captured.get("result")
        result_rows = result if isinstance(result, list) else []
        sql = str(captured.get("sql") or "")
        analysis = str(captured.get("analysis") or "")
        return TargetResponse(
            answer=analysis or sql,
            retrieved=_retrieved_evidence(captured),
            prompt_tokens=token_summary["prompt_tokens"],
            completion_tokens=token_summary["completion_tokens"],
            cost=token_summary["cost"],
            metadata={
                "target": "nl2sql-application-runtime-v1",
                "execution_status": terminal_status or "succeeded",
                "elapsed_ms": round((time.perf_counter() - started_clock) * 1000, 3),
                "sql_generated": bool(sql),
                "result_row_count": len(result_rows),
                "llm_attempt_count": token_summary["attempt_count"],
                "unknown_token_attempts": token_summary["unknown_token_attempts"],
                "cost_estimate_available": token_summary["cost_estimate_available"],
            },
        )


def create_nl2sql_evaluation_target(
    *,
    workspace_id: str,
    access_policy: PolicySnapshot,
    dw_mysql_repo: object,
    meta_mysql_repo: object,
    value_es_repo: object,
    column_qdrant_repo: object,
    metric_qdrant_repo: object,
    training_qdrant_repo: object,
    embedding_client: object,
    llm: object | None = None,
    index_versions: object | None = None,
    settings: Nl2SqlSettings | None = None,
    runner: GraphRunner | None = None,
    actor_id: str = "evaluation-user",
    force_model_sql: bool = False,
    model_sql_fallback_enabled: bool = True,
) -> Nl2SqlEvaluationTarget:
    """创建 production-equivalent Target；Graph 保持首次执行才编译。"""

    fixed = build_nl2sql_evaluation_capabilities(
        dw_mysql_repo=dw_mysql_repo,
        meta_mysql_repo=meta_mysql_repo,
        value_es_repo=value_es_repo,
        column_qdrant_repo=column_qdrant_repo,
        metric_qdrant_repo=metric_qdrant_repo,
        training_qdrant_repo=training_qdrant_repo,
        embedding_client=embedding_client,
        llm=llm,
        index_versions=index_versions,
        settings=settings,
    )
    resolved_runner = runner or create_nl2sql_graph_runner()
    runtime = Nl2SqlApplicationRuntime(resolved_runner, fixed)
    return Nl2SqlEvaluationTarget(
        runtime,
        workspace_id=workspace_id,
        access_policy=access_policy,
        actor_id=actor_id,
        force_model_sql=force_model_sql,
        model_sql_fallback_enabled=model_sql_fallback_enabled,
        runner=resolved_runner,
        fixed_capabilities=fixed,
    )


def _retrieved_evidence(state: Mapping[str, object]) -> tuple[RetrievedEvidence, ...]:
    rows: list[tuple[str, float | None]] = []

    def add(document_id: str, item: Mapping[str, object]) -> None:
        text = str(document_id or "").strip()
        if not text:
            return
        raw_score = item.get("rerank_score", item.get("score"))
        try:
            score = float(raw_score) if raw_score is not None else None
        except (TypeError, ValueError):
            score = None
        rows.append((text, score))

    for item in _mapping_rows(state.get("table_infos")):
        add(f"table:{item.get('name') or item.get('id') or ''}", item)
    for item in _mapping_rows(state.get("recall_columns")):
        identifier = str(item.get("id") or "")
        if "." not in identifier:
            identifier = ".".join(
                part
                for part in (str(item.get("table_id") or ""), str(item.get("name") or ""))
                if part
            )
        add(f"column:{identifier}", item)
    for item in _mapping_rows(state.get("recall_metrics")):
        add(f"metric:{item.get('name') or item.get('id') or ''}", item)
    for item in _mapping_rows(state.get("recall_values")):
        column_id = str(item.get("column_id") or "")
        if "." not in column_id:
            column_id = ".".join(
                part
                for part in (
                    str(item.get("table_id") or item.get("table_name") or ""),
                    str(item.get("column_name") or ""),
                )
                if part
            )
        add(f"value:{column_id}:{item.get('value') or ''}", item)
    for item in _mapping_rows(state.get("recall_training")):
        add(f"training:{item.get('question') or ''}", item)

    seen: set[str] = set()
    evidence: list[RetrievedEvidence] = []
    for document_id, score in rows:
        if document_id in seen or document_id.endswith(":"):
            continue
        seen.add(document_id)
        evidence.append(
            RetrievedEvidence(
                document_id=document_id,
                chunk_id=None,
                rank=len(evidence) + 1,
                score=score,
            )
        )
    return tuple(evidence)


def _mapping_rows(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _token_summary(attempts: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    unique: dict[str, Mapping[str, object]] = {}
    for index, item in enumerate(attempts):
        key = str(
            item.get("attempt_id")
            or item.get("provider_request_id")
            or f"{item.get('node_id')}:{item.get('route')}:{item.get('retry_index')}:{index}"
        )
        unique[key] = item
    rows = list(unique.values())
    token_known = [item for item in rows if item.get("token_known") is True]
    costs = [
        float(item["estimated_cost"])
        for item in rows
        if isinstance(item.get("estimated_cost"), (int, float))
    ]
    return {
        "attempt_count": len(rows),
        "prompt_tokens": sum(int(item.get("prompt_tokens") or 0) for item in token_known),
        "completion_tokens": sum(
            int(item.get("completion_tokens") or 0) for item in token_known
        ),
        "unknown_token_attempts": len(rows) - len(token_known),
        "cost": round(sum(costs), 8),
        "cost_estimate_available": bool(rows) and len(costs) == len(rows),
    }


__all__ = [
    "Nl2SqlEvaluationTarget",
    "Nl2SqlEvaluationTargetError",
    "build_nl2sql_evaluation_capabilities",
    "build_nl2sql_evaluation_context",
    "build_nl2sql_evaluation_state",
    "create_nl2sql_evaluation_target",
]
