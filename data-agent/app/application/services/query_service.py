"""面向 HTTP/SSE 问数接口的流式应用服务。"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Mapping

from app.application.services.query_trace_service import QueryTraceCollector
from app.application.ports.execution_versions import (
    ExecutionVersionProvider,
    MissingExecutionVersionProvider,
)
from app.application.ports.llm_routing import LLMRouteCatalog
from app.application.ports.agent_runtime import (
    AgentEventKind,
    AgentExecutionRequest,
    QueryAgentRuntime,
)
from app.application.ports.query_persistence import (
    DataWarehouseQueryPort,
    QueryPersistencePort,
)
from app.application.ports.query_settings import QueryServiceSettings
from app.application.ports.query_tracing import QueryTraceLLMSettings
from app.domain.access.policy import PolicySnapshot
from app.domain.query.follow_up_semantics import semantic_state_from_sql
from app.domain.query.result_references import project_single_result_references
from app.application.services.conversation_memory_service import ConversationMemoryService
from app.application.services.data_source_context_service import DataSourceContextService
from app.application.services.conversation_data_context_service import (
    ConversationDataContextService,
)
from app.domain.data_sources.context import (
    DataSourceContextError,
    MappingConfirmationRequired,
)


logger = logging.getLogger(__name__)


def _latest_result_references(history_rows) -> dict:
    """从紧邻成功轮投影可安全用于单数指代的有界结果锚点。"""

    if not history_rows:
        return {}
    latest = history_rows[-1]
    serialized = getattr(latest, "result", None)
    if not serialized:
        return {}
    try:
        rows = json.loads(serialized)
    except (TypeError, ValueError):
        logger.warning("紧邻历史结果无法解析，结果指代将要求澄清")
        return {}
    if not isinstance(rows, list):
        return {}
    return project_single_result_references(
        rows,
        row_count=getattr(latest, "row_count", None),
    )


class QueryService:
    """
    问数业务服务层：组装 LangGraph 运行所需上下文，驱动多轮对话流式问答。

    一句话总述：接收用户提问与会话ID，加载历史对话注入 Agent，流式返回阶段进度、
                SQL 与查询结果，并在结束后把本轮 (query, sql, result) 持久化到 chat_history。

    核心功能：从会话端口读取历史与保存回合，通过 QueryAgentRuntime 驱动受治理的
             Agent 流式执行，并把稳定应用事件转换为既有 SSE 消息。

    业务逻辑：
        1. 若客户端未传 chat_id，则生成一个新会话ID并通过 SSE 推给前端
        2. 若有 chat_id，从 ChatHistoryRepository 加载最近 N 轮历史（旧→新）
        3. 构造应用层执行请求与每次调用的授权事实
        4. QueryAgentRuntime 流式驱动；若 chunk 含 sql/result 则暂存
        5. 流结束后若拿到最终 sql，用独立 meta session 调 save_turn 存回合（含结果）
        6. 任意异常通过 SSE 返回 {"error": ...}

    设计要点：QueryPersistencePort 负责独立写事务；Application 不感知 Session、ORM
              Record 或具体 Repository，因此流式 body 的生命周期不会泄漏到持久层。
    """
    def __init__(self,
                 dw_mysql_repo: DataWarehouseQueryPort,
                 query_persistence: QueryPersistencePort,
                 data_source_context_service: DataSourceContextService,
                 conversation_context_service: ConversationDataContextService,
                 graph_runner: QueryAgentRuntime,
                 settings: QueryServiceSettings,
                 llm_route_catalog: LLMRouteCatalog,
                 trace_llm_settings: QueryTraceLLMSettings,
                 owner_id: str | None = None, workspace_id: str | None = None,
                 access_policy: PolicySnapshot | None = None,
                 memory_service: ConversationMemoryService | None = None,
                 execution_version_provider: ExecutionVersionProvider | None = None):
        """
        构造 QueryService，注入问数用例直接依赖的仓储与运行时端口。

        核心功能：固化 owner / workspace / 访问策略三类租户上下文；具体 ES、Qdrant、
                 embedding 和 GraphRunner 由组合根装配到 QueryAgentRuntime 实现中。

        Args:
            dw_mysql_repo: 数据仓库 MySQL 仓储（执行用户 SQL、读取 schema 快照）
            query_persistence: 会话历史、回合状态与 Trace 的事务端口
            data_source_context_service: 当前调用者作用域的数据源解析用例
            conversation_context_service: 会话默认数据源上下文用例
            graph_runner: Application 侧 QueryAgentRuntime 端口实现
            owner_id: 调用者用户 ID，用于多租户隔离与历史归属
            workspace_id: 工作空间 ID，用于限流、策略与链路归因
            access_policy: 访问策略快照，None 时按 owner/workspace 构造空策略
        """
        self.dw_mysql_repo = dw_mysql_repo
        self.query_persistence = query_persistence
        self.data_source_context_service = data_source_context_service
        self.conversation_context_service = conversation_context_service
        self.agent_runtime = graph_runner
        self.settings = settings
        self.llm_route_catalog = llm_route_catalog
        self.trace_llm_settings = trace_llm_settings
        self.owner_id = owner_id
        self.workspace_id = workspace_id
        # 未显式传入策略时构造一个空策略（无任何 policy），仅承载 owner/workspace 上下文
        self.access_policy = access_policy or PolicySnapshot(
            policies=tuple(), user_id=owner_id or "", workspace_id=workspace_id or ""
        )
        self.memory_service = memory_service
        self.execution_version_provider = (
            execution_version_provider or MissingExecutionVersionProvider()
        )

    async def search(
        self,
        query: str,
        chat_id: str = None,
        dataset_id: str = None,
        dataset_ids: list[str] | None = None,
        compare_dw: bool = False,
        source_mode: str | None = None,
        comparison_intent: str | None = None,
        mapping_confirmation_id: str | None = None,
        data_context_revision: int | None = None,
        enable_analysis: bool | None = None,
    ):
        """
        流式问数主入口（SSE 生成器）。

        业务逻辑：
            1. 确定 chat_id：未传则生成新 uuid，并作为首个 SSE chunk 推给前端
            2. 加载历史：按 chat_id 取最近 history_limit 轮，转为 [{query, sql}, ...]
            3. 构造 State 与 Context，由注入的 GraphRunner 驱动
            4. 转发每个 chunk；若 chunk 含 "sql" 键则暂存为 final_sql，
               若 chunk 含 "result" 键则暂存为 final_result
            5. 流结束后若有 final_sql，用独立 meta session save_turn 持久化本轮（含结果）
            6. 异常通过 SSE {"error": ...} 返回

        Args:
            query: 用户自然语言提问
            chat_id: 会话ID（可选，不传则新建会话）
            dataset_id: 会话临时数据集ID（可选）
            compare_dw: 临时数据查询是否允许与已有数据库对比
            enable_analysis: 是否生成可选 AI 数据解读；不传时读取服务端默认值

        Yields:
            str: 形如 "data: {json}\\n\\n" 的 SSE 消息
        """
        collector = None
        final_sql = None
        final_result = None
        final_follow_up = None
        final_intent_response: dict[str, object] | None = None
        memory_context: dict = {}
        user_message_id: int | None = None
        turn_started = False
        turn_finished = False
        resolved_source = None
        source_context_service = self.data_source_context_service
        conversation_context_service = self.conversation_context_service
        try:
            # =============================================================================
            # 阶段 1：确定会话ID和链路ID
            # =============================================================================
            incoming_chat_id = chat_id
            if not chat_id:
                chat_id = str(uuid.uuid4())
            collector = QueryTraceCollector(
                chat_id=chat_id,
                route_catalog=self.llm_route_catalog,
                llm_settings=self.trace_llm_settings,
            )
            performance_started_at = time.perf_counter()
            yield f"data: {json.dumps({'chat_id': chat_id, 'trace_id': collector.trace_id}, ensure_ascii=False)} \n\n"

            explicit_source_context = bool(
                dataset_id
                or dataset_ids
                or source_mode
                or comparison_intent
                or mapping_confirmation_id
                or compare_dw
            )
            try:
                if explicit_source_context:
                    resolved_source = await source_context_service.resolve(
                        dataset_id=dataset_id,
                        dataset_ids=dataset_ids,
                        source_mode=source_mode,
                        compare_dw=compare_dw,
                        comparison_intent=comparison_intent,
                        mapping_confirmation_id=mapping_confirmation_id,
                        revision=max(1, int(data_context_revision or 1)),
                    )
                elif incoming_chat_id:
                    resolved_source = (
                        await conversation_context_service.resolve_default(chat_id)
                    )
                else:
                    resolved_source = await source_context_service.resolve(
                        source_mode="dw_only"
                    )
            except DataSourceContextError as source_error:
                if explicit_source_context:
                    raise
                # 会话默认上下文引用了已过期/删除/改版资源时，关闭式失效并回到
                # DW-only；旧回合快照仍保留，但不能继续授权新查询。
                previous = await conversation_context_service.get_record(chat_id)
                await conversation_context_service.reset(
                    chat_id,
                    expected_revision=previous.get("revision"),
                )
                resolved_source = await source_context_service.resolve(
                    source_mode="dw_only"
                )
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "source_context_invalidated": {
                                "reason_code": source_error.code,
                                "message": str(source_error),
                                "fallback": "dw_only",
                            }
                        },
                        ensure_ascii=False,
                    )
                    + " \n\n"
                )
            source_public = resolved_source.access.public_dict()
            collector.record_retrieval_metric({
                "responsibility": "data_source_context",
                "source_mode": source_public["source_mode"],
                "dataset_count": len(source_public["datasets"]),
                "context_fingerprint": source_public["context_fingerprint"],
                "revision": source_public["revision"],
                "evidence_complete": True,
            })

            # 权威消息账本先于 Agent 执行写入；若 MySQL 不可用，不能假装本轮
            # 可以可靠继承或持久化，直接进入统一异常路径。
            if self.memory_service is not None and self.settings.memory_enabled:
                user_message_id = await self.memory_service.start_turn(
                    chat_id=chat_id, query=query, trace_id=collector.trace_id
                )
                turn_started = True
                memory_started_ms = int(
                    (time.perf_counter() - performance_started_at) * 1000
                )
                collector.start_node("load_memory", memory_started_ms)
                memory_context = await self.memory_service.load_context(
                    chat_id=chat_id, query=query
                )
                memory_finished_ms = int(
                    (time.perf_counter() - performance_started_at) * 1000
                )
                collector.end_node("load_memory", end_ms=memory_finished_ms)
                memory_evidence = dict(memory_context.get("evidence") or {})
                collector.record_retrieval_metric(memory_evidence)
                for degradation in memory_evidence.get("degradations") or []:
                    collector.add_degradation(degradation)
                # SSE 只公开计数、版本和故障证据；记忆正文仅通过受 scope
                # 保护的管理 API 返回给其所有者，不污染普通对话消息。
                public_memory = {
                    "schema_version": memory_context.get("schema_version"),
                    "working_source": memory_evidence.get("working_source"),
                    "working_version": memory_evidence.get("working_version"),
                    "redis_hit": memory_evidence.get("redis_hit"),
                    "recent_selected": memory_evidence.get("recent_selected"),
                    "preference_count": len(
                        memory_context.get("confirmed_preferences") or []
                    ),
                    "episode_count": len(
                        memory_context.get("relevant_episodes") or []
                    ),
                    "profile_fact_count": len(
                        memory_context.get("confirmed_profile_facts") or []
                    ),
                    "long_term_goal_count": len(
                        memory_context.get("long_term_goals") or []
                    ),
                    "summary_count": len(
                        memory_context.get("selected_summaries") or []
                    ),
                    "summary_status": (
                        memory_evidence.get("summary") or {}
                    ).get("status"),
                    "summary_trigger": (
                        memory_evidence.get("summary") or {}
                    ).get("trigger_reason"),
                    "recent_messages_in_prompt": (
                        memory_evidence.get("summary") or {}
                    ).get("recent_messages_in_prompt", True),
                    "es_candidates": memory_evidence.get("es_candidates"),
                    "qdrant_candidates": memory_evidence.get(
                        "qdrant_candidates"
                    ),
                    "fusion_method": memory_evidence.get("fusion_method"),
                    "authoritative_recheck": memory_evidence.get(
                        "authoritative_recheck"
                    ),
                    "degradations": memory_evidence.get("degradations") or [],
                }
                yield (
                    "data: "
                    + json.dumps(
                        {"memory_context": public_memory},
                        ensure_ascii=False,
                        default=str,
                    )
                    + " \n\n"
                )
            else:
                await self._start_conversation_turn(
                    chat_id=chat_id,
                    turn_id=collector.trace_id,
                    query=query,
                )
                turn_started = True

            await conversation_context_service.snapshot_turn(
                chat_id=chat_id,
                turn_id=collector.trace_id,
                resolved=resolved_source,
            )

            # =============================================================================
            # 阶段 2：加载多轮对话历史（旧→新）
            # =============================================================================
            history_limit = self.settings.history_limit
            history_rows = await self.query_persistence.recent_history(
                chat_id,
                history_limit,
            )
            # 历史上下文只包含问题、SQL 和脱敏结构状态，原始结果行不注入模型。
            # 紧邻单行结果只经白名单投影为最小地区锚点；集合或歧义指代仍要求澄清。
            chat_history = [
                {
                    "query": row.query,
                    "sql": row.sql,
                    "standalone_question": row.standalone_question or row.query,
                    "follow_up_action": row.follow_up_action or "new_topic",
                    "semantic_state": row.semantic_state,
                }
                for row in history_rows
            ]
            result_references = _latest_result_references(history_rows)

            # =============================================================================
            # 阶段 3：构造 State 与 Context，驱动 LangGraph
            # =============================================================================
            source_state = resolved_source.state_dict()
            state: dict[str, object] = dict(
                query=query,
                original_query=query,
                chat_history=chat_history,
                result_references=result_references,
                memory_context=memory_context,
                # enable_analysis 未显式传入时回退到服务端默认配置
                analysis_enabled=(
                    self.settings.analysis_enabled_default
                    if enable_analysis is None else bool(enable_analysis)
                ),
                performance_started_at=performance_started_at,
                node_metrics=[],
                degradations=[],
                retrieval_metrics=[],
                llm_attempts=[],
                semantic_repair_ledger=[],
                **source_state,
            )
            execution_plan = self.agent_runtime.prepare(
                performance_started_at=performance_started_at,
                initial_state=state,
            )
            collector.record_request_deadline(dict(execution_plan.deadline_evidence))
            invocation_facts: dict[str, object] = dict(
                access_policy=self.access_policy,
                workspace_id=str(self.workspace_id or ""),
                asset_access_context=resolved_source.access,
            )

            # =============================================================================
            # 阶段 4：流式驱动，转发 SSE 并采集阶段/Token
            # =============================================================================
            execution_request = AgentExecutionRequest(
                request_id=collector.trace_id,
                query=query,
                actor_id=str(self.owner_id or "anonymous"),
                workspace_id=str(self.workspace_id or "default"),
                conversation_id=chat_id,
                trace_id=collector.trace_id,
                roles=(str(self.access_policy.role or "viewer"),),
                allowed_data_sources=tuple(
                    sorted(resolved_source.access.allowed_physical_tables)
                ),
                analysis_enabled=bool(state.get("analysis_enabled")),
                plan=execution_plan,
                capabilities=invocation_facts,
                callbacks=(collector.callback,),
            )
            async for event in self.agent_runtime.stream(execution_request):
                if event.kind is AgentEventKind.SNAPSHOT:
                    # Runtime 快照属于内部执行合同，不改变既有 SSE 协议。
                    continue
                if event.kind in {
                    AgentEventKind.STARTED,
                    AgentEventKind.COMPLETED,
                }:
                    continue
                chunk = event.payload.get("chunk")
                if chunk is None and event.kind is AgentEventKind.ERROR:
                    chunk = {
                        "error": event.error_message or "Agent 执行失败"
                    }
                """
                chunk: {"stage": "xxx"} / {"sql": "..."} / {"result": [{}, {}]} / {"error": "..."}
                """
                # 捕获最终执行的 SQL 与查询结果，用于流结束后持久化对话历史
                if isinstance(chunk, dict):
                    # ── 分支 A：内部节点事件（trace_event），仅采集不转发 ──
                    trace_event = chunk.get("trace_event")
                    if isinstance(trace_event, dict):
                        collector.record_node_event(trace_event)
                        # 内部节点事件只用于链路采集，不污染前端 SSE 协议。
                        continue
                    # ── 分支 B：内部 LLM attempt，仅采集不转发 ──
                    llm_attempt = chunk.get("llm_attempt")
                    if isinstance(llm_attempt, dict):
                        collector.record_llm_attempt(llm_attempt)
                        # 统一模型 attempt 是内部审计事实，前端只消费 trace 汇总。
                        continue
                    # 业务引导、安全拒绝和补充澄清属于正常终止结果。终止决定一旦
                    # 生效，后续即使 Runtime 错误地产出 SQL/结果，也不能再越过应用边界。
                    intent_response = chunk.get("intent_response")
                    if (
                        isinstance(intent_response, dict)
                        and intent_response.get("decision") == "respond"
                    ):
                        if collector.error is not None:
                            # 已公开的 error 是更早的终态，不能再用晚到的策略响应
                            # 覆盖为 handled，否则 SSE、会话账本与 trace 会相互矛盾。
                            continue
                        if final_intent_response is None:
                            final_intent_response = dict(intent_response)
                            # 预期终态对本轮 SQL/结果拥有最终裁决权。即使上游违反
                            # 事件顺序先发出了查询制品，也不得保存或写入 trace。
                            final_sql = None
                            final_result = None
                            final_follow_up = None
                            collector.mark_first_event()
                            yield (
                                "data: "
                                + json.dumps(
                                    {"intent_response": final_intent_response},
                                    ensure_ascii=False,
                                    default=str,
                                )
                                + " \n\n"
                            )
                        continue
                    if final_intent_response is not None:
                        # handled 是终态；只继续消费内部 trace 事件，不向客户端泄漏
                        # 终态之后的阶段、SQL、结果、图表或错误事件。
                        continue
                    # ── 分支 C：面向前端的阶段/SQL/结果/降级/ follow-up 等事件 ──
                    if "stage" in chunk:
                        collector.mark_first_event()
                    if "error" in chunk:
                        collector.fail(chunk["error"])
                    if "sql" in chunk:
                        final_sql = chunk["sql"]
                        collector.mark_sql()
                    if "result" in chunk:
                        final_result = chunk["result"]
                        collector.mark_result()
                    if "degradation" in chunk and isinstance(chunk["degradation"], dict):
                        collector.add_degradation(chunk["degradation"])
                    if "follow_up" in chunk and isinstance(chunk["follow_up"], dict):
                        final_follow_up = dict(chunk["follow_up"])
                        collector.record_follow_up(final_follow_up)
                    if "candidate_decision" in chunk and isinstance(chunk["candidate_decision"], dict):
                        collector.record_candidate_decision(chunk["candidate_decision"])
                    if "planning_route" in chunk and isinstance(chunk["planning_route"], dict):
                        collector.record_planning_route(chunk["planning_route"])
                    if "planning_timing" in chunk and isinstance(chunk["planning_timing"], dict):
                        collector.record_planning_timing(chunk["planning_timing"])
                    if "semantic_repair" in chunk and isinstance(chunk["semantic_repair"], dict):
                        collector.record_semantic_repair(repair=chunk["semantic_repair"])
                    if "semantic_violation" in chunk and isinstance(chunk["semantic_violation"], dict):
                        collector.record_semantic_repair(violation=chunk["semantic_violation"])
                    if "semantic_coverage" in chunk and isinstance(chunk["semantic_coverage"], dict):
                        collector.record_semantic_coverage(chunk["semantic_coverage"])
                    if "semantic_query_contract" in chunk and isinstance(chunk["semantic_query_contract"], dict):
                        collector.record_semantic_query_contract(chunk["semantic_query_contract"])
                    if "relation_closure" in chunk and isinstance(chunk["relation_closure"], dict):
                        collector.record_relation_closure(chunk["relation_closure"])
                    if "request_deadline" in chunk and isinstance(chunk["request_deadline"], dict):
                        collector.record_request_deadline(chunk["request_deadline"])
                    if "retrieval_budget" in chunk and isinstance(chunk["retrieval_budget"], dict):
                        collector.record_retrieval_metric(chunk["retrieval_budget"])
                yield f"data: {json.dumps(chunk, ensure_ascii=False, default=str)} \n\n"

            # =============================================================================
            # 阶段 5：流结束，持久化本轮对话（含结果）
            # =============================================================================
            # 用独立 session 提交，避免请求级 yield 依赖在 StreamingResponse 下提前关闭
            if (
                final_intent_response is None
                and final_sql
                and collector.error is None
                and final_result is not None
            ):
                try:
                    standalone = str(
                        (final_follow_up or {}).get("standalone_question")
                        or query
                    )
                    follow_up_action = str(
                        (final_follow_up or {}).get("action") or "new_topic"
                    )
                    semantic_state = semantic_state_from_sql(
                        standalone, final_sql
                    )
                    if self.memory_service is not None and user_message_id:
                        version_snapshot = self.execution_version_provider.snapshot(
                            self.workspace_id or ""
                        )
                        write_evidence = await self.memory_service.complete_turn(
                            chat_id=chat_id,
                            query=query,
                            sql=final_sql,
                            row_count=len(final_result),
                            result=final_result,
                            standalone_question=standalone,
                            follow_up_action=follow_up_action,
                            semantic_state=semantic_state,
                            trace_id=collector.trace_id,
                            user_message_id=user_message_id,
                            memory_context=memory_context,
                            schema_version=version_snapshot.get(
                                "schema_version"
                            ),
                        )
                        yield (
                            "data: "
                            + json.dumps(
                                {"memory_write": write_evidence},
                                ensure_ascii=False,
                            )
                            + " \n\n"
                        )
                        turn_finished = True
                    else:
                        await self.query_persistence.save_successful_turn(
                            chat_id=chat_id,
                            turn_id=collector.trace_id,
                            query=query,
                            sql=final_sql,
                            result=final_result,
                            standalone_question=standalone,
                            follow_up_action=follow_up_action,
                            semantic_state=semantic_state,
                        )
                        turn_finished = True
                except Exception as save_err:
                    degradation = {
                        "node_id": "memory_write",
                        "route": "memory",
                        "reason_code": "memory_authority_write_failed",
                        "severity": "warning",
                        "affects_core_path": False,
                        "detail": type(save_err).__name__,
                        "evidence_complete": True,
                    }
                    collector.add_degradation(degradation)
                    logger.error(
                        "保存权威会话记忆失败 error_type=%s",
                        type(save_err).__name__,
                    )
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "memory_write": {
                                    "mysql_committed": False,
                                    "error_type": type(save_err).__name__,
                                },
                                "degradation": degradation,
                            },
                            ensure_ascii=False,
                        )
                        + " \n\n"
                    )
            if final_intent_response is not None and collector.error is None:
                await self._complete_handled_turn(
                    chat_id=chat_id,
                    turn_id=collector.trace_id,
                    user_message_id=user_message_id,
                    response=final_intent_response,
                )
                turn_finished = True
            if turn_started and not turn_finished:
                await self._fail_conversation_turn(
                    chat_id=chat_id,
                    turn_id=collector.trace_id,
                    user_message_id=user_message_id,
                    status="failed",
                    summary=(
                        str(collector.error)
                        if collector.error
                        else "本轮未产生可恢复的查询结果。"
                    ),
                )
                turn_finished = True
            collector.record_request_deadline(
                dict(execution_plan.deadline_evidence)
            )
            try:
                trace_status = (
                    "failed"
                    if collector.error
                    else "handled"
                    if final_intent_response is not None
                    else "success"
                )
                summary = await self._persist_trace(
                    collector, query=query, chat_id=chat_id, sql=final_sql,
                    status=trace_status,
                    row_count=len(final_result or []), error=collector.error,
                )
            except Exception as trace_err:
                # 链路日志失败不影响已经返回的查询结果。
                logger.error(f"保存查询链路日志失败：{trace_err}")
                summary = collector.finish(
                    "failed"
                    if collector.error
                    else "handled"
                    if final_intent_response is not None
                    else "success"
                )
            yield f"data: {json.dumps({'trace': summary}, ensure_ascii=False)} \n\n"
            yield f"data: {json.dumps({'done': True}, ensure_ascii=False)} \n\n"
        except asyncio.CancelledError:
            if collector is not None and turn_started and not turn_finished:
                try:
                    await self._fail_conversation_turn(
                        chat_id=chat_id,
                        turn_id=collector.trace_id,
                        user_message_id=user_message_id,
                        status="cancelled",
                        summary="请求已由用户中止。",
                    )
                except Exception as memory_error:
                    logger.error(
                        "标记取消回合失败 error_type=%s",
                        type(memory_error).__name__,
                    )
            raise
        except Exception as e:
            # =============================================================================
            # 异常分支：标记链路失败并尝试持久化失败 trace，再通过 SSE 返回 error
            # =============================================================================
            if collector is not None:
                collector.fail(str(e))
                try:
                    summary = await self._persist_trace(
                        collector, query=query, chat_id=chat_id, sql=final_sql,
                        status="failed", row_count=len(final_result or []), error=str(e),
                    )
                    yield f"data: {json.dumps({'trace': summary}, ensure_ascii=False)} \n\n"
                except Exception as trace_err:
                    logger.error(f"保存查询链路日志失败：{trace_err}")
            if collector is not None and turn_started and not turn_finished:
                try:
                    await self._fail_conversation_turn(
                        chat_id=chat_id,
                        turn_id=collector.trace_id,
                        user_message_id=user_message_id,
                        status="failed",
                        summary=str(e),
                    )
                except Exception as memory_error:
                    logger.error(
                        "标记失败消息状态失败 error_type=%s",
                        type(memory_error).__name__,
                    )
            error_payload = (
                e.to_dict() if isinstance(e, DataSourceContextError) else str(e)
            )
            if isinstance(e, MappingConfirmationRequired):
                yield (
                    "data: "
                    + json.dumps(
                        {"source_mapping": e.proposal},
                        ensure_ascii=False,
                        default=str,
                    )
                    + " \n\n"
                )
            yield f"data: {json.dumps({'error': error_payload}, ensure_ascii=False, default=str)} \n\n"
            yield f"data: {json.dumps({'done': True}, ensure_ascii=False)} \n\n"

    async def _start_conversation_turn(
        self, *, chat_id: str, turn_id: str, query: str
    ) -> None:
        await self.query_persistence.start_turn(
            chat_id=chat_id,
            turn_id=turn_id,
            query=query,
        )

    async def _complete_handled_turn(
        self,
        *,
        chat_id: str,
        turn_id: str,
        user_message_id: int | None,
        response: Mapping[str, object],
    ) -> None:
        """把无 SQL 的预期响应落账为 message-only 成功回合。"""

        if (
            self.memory_service is not None
            and self.settings.memory_enabled
        ):
            await self.memory_service.complete_handled_turn(
                user_message_id,
                chat_id=chat_id,
                trace_id=turn_id,
                response=response,
            )
            return
        await self.query_persistence.complete_handled_turn(
            chat_id=chat_id,
            turn_id=turn_id,
            response=response,
        )

    async def _fail_conversation_turn(
        self,
        *,
        chat_id: str,
        turn_id: str,
        user_message_id: int | None,
        status: str,
        summary: str,
        use_memory_service: bool = True,
    ) -> None:
        if (
            use_memory_service
            and self.memory_service is not None
            and self.settings.memory_enabled
        ):
            await self.memory_service.fail_turn(
                user_message_id,
                chat_id=chat_id,
                trace_id=turn_id,
                status=status,
                summary=summary,
            )
            return
        await self.query_persistence.fail_turn(
            chat_id=chat_id,
            turn_id=turn_id,
            status=status,
            summary=summary,
        )

    async def _persist_trace(self, collector: QueryTraceCollector, *, query: str,
                             chat_id: str, sql: str | None, status: str,
                             row_count: int, error: str | None):
        """
        将本轮查询链路（summary + steps + usages）持久化到 meta MySQL，并返回 summary。

        核心功能：解析当前工作空间对应的版本快照，将 collector 与版本事实交给
                 QueryPersistencePort；具体 Record 映射和独立事务由 Infrastructure 实现。

        业务逻辑：
            1. ExecutionVersionProvider 冻结当前请求的运行版本快照（用于回溯）
            2. QueryPersistencePort 映射并持久化 trace/steps/usages
            3. 返回 summary 供 SSE 推送给前端

        Args:
            collector: 本轮链路采集器，含节点事件/attempt/降级等运行期事实
            query: 用户原始提问
            chat_id: 会话ID
            sql: 本轮最终 SQL（可能为 None）
            status: "success" 或 "failed"
            row_count: 结果行数
            error: 失败时的错误信息，成功时为 None

        Returns:
            dict: 链路 summary，供前端展示耗时/节点/降级等汇总信息
        """
        # 取当前工作空间的治理版本快照，固化本轮链路对应的版本基线
        version_snapshot = self.execution_version_provider.snapshot(
            self.workspace_id or ""
        )
        return await self.query_persistence.persist_trace(
            collector,
            query=query,
            chat_id=chat_id,
            sql=sql,
            status=status,
            row_count=row_count,
            error=error,
            version_snapshot=version_snapshot,
            # 缓存命中标记，用于链路分析命中率
            cache_hit=bool(self.dw_mysql_repo.last_cache_hit),
        )

    async def get_history(self, chat_id: str, limit: int = None):
        """
        获取指定会话的历史记录列表（供 GET /api/chat/{chat_id}/history 使用）。

        Args:
            chat_id: 会话ID
            limit: 取最近几轮，None 则用配置默认值

        Returns:
            list[dict]: 历史记录列表，每项含 query/sql/result/row_count/created_at
        """
        history_limit = max(
            1,
            min(
                int(limit if limit is not None else self.settings.history_limit),
                100,
            ),
        )
        if await self.query_persistence.conversation_session(chat_id) is None:
            return None
        rows = await self.query_persistence.recent_history(
            chat_id,
            history_limit,
        )
        return [
            {
                "query": row.query,
                "sql": row.sql,
                "result": json.loads(row.result) if row.result else None,
                "row_count": row.row_count,
                "created_at": str(row.created_at),
                "standalone_question": row.standalone_question or row.query,
                "follow_up_action": row.follow_up_action or "new_topic",
                "semantic_state": row.semantic_state,
            }
            for row in rows
        ]

    async def get_memory_status(self, chat_id: str) -> dict:
        if await self._conversation_session(chat_id) is None:
            return {"chat_id": chat_id, "status": "not_found"}
        if self.memory_service is None:
            return {"chat_id": chat_id, "status": "disabled"}
        return await self.memory_service.status(chat_id)

    async def reset_memory(
        self, chat_id: str, *, include_long_term: bool = False
    ) -> dict:
        if await self._conversation_session(chat_id) is None:
            return {"chat_id": chat_id, "status": "not_found"}
        if self.memory_service is None:
            return {"chat_id": chat_id, "status": "disabled"}
        return await self.memory_service.reset(
            chat_id, include_long_term=include_long_term
        )

    async def _conversation_session(self, chat_id: str):
        return await self.query_persistence.conversation_session(chat_id)
