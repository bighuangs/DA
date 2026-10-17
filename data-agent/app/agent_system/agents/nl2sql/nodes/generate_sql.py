import logging

import yaml
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent_system.agents.nl2sql.context import DataAgentContext
from app.agent_system.agents.nl2sql.llm import get_task_llm, task_timeout
from app.agent_system.agents.nl2sql.state import DataAgentState
from app.agent_system.agents.nl2sql.semantic_contract import (
    build_query_contract,
    compact_query_contract,
)
from app.agent_system.agents.nl2sql.semantic_hints import build_semantic_contract
from app.agent_system.agents.nl2sql.semantic_ir import compile_semantic_ir
from app.agent_system.agents.nl2sql.llm import (
    LLMInvocationError,
    invoke_llm,
    normalize_sql_output,
    pre_invoke_failure,
    sql_output_validator,
)
from app.agent_system.agents.nl2sql.services.prompt_context import build_model_prompt_payload
from app.agent_system.agents.nl2sql.llm import (
    build_model_clarification,
    clarification_message,
)
from app.agent_system.agents.nl2sql.services.model_sql_policy import (
    deterministic_draft_is_safe,
    deterministic_model_fallback_allowed,
    model_semantic_summary,
    use_model_sql,
)
from app.agent_system.governance.deadline import model_phase_budget, route_deadline_at
from app.agent_system.agents.nl2sql.prompts.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

llm = None


async def generate_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    settings = runtime.context["settings"]
    """
    基于提问与召回的上下文信息，让大模型生成目标 SQL。

    核心功能：将用户提问 query、过滤后的表信息 table_infos、指标信息
             metric_infos、日期信息 date_info、数据库信息 db_info 一起
             塞入 prompt 模板，驱动大模型生成一条 SQL 文本（字符串输出），
             供后续校验与执行节点使用。

    业务逻辑：
        1. 向前端推送当前流水线阶段"生成SQL"
        2. 从 state 取出 query / table_infos / metric_infos / date_info / db_info
        3. 构造 prompt 链（模板 → llm → StrOutputParser）
        4. 将各上下文用 yaml 序列化（保留中文、保持顺序）后送入大模型
        5. 大模型返回纯文本 SQL
        6. 返回 {"sql": sql} 更新到 state

    Args:
        state: 流水线状态，包含 query 及各类上下文信息
        runtime: LangGraph 运行时，携带上下文与流式写入器

    Returns:
        dict: {"sql": str} —— 大模型生成的 SQL 文本

    Raises:
        Exception: 生成过程中任意异常都向上抛出，便于调试定位
    """
    runtime.stream_writer({"stage": "生成SQL"})
    semantic_ir = dict(state.get("semantic_ir") or {})
    # Metric/table filtering has now finished, so rebuild the contract from the
    # exact evidence that will enter the SQL prompt.  This prevents an early,
    # noisy metric candidate from becoming a false formula obligation while
    # preserving the same contract hash for equivalent evidence.
    semantic_ir["query_contract"] = build_query_contract(
        str(state.get("query") or ""),
        state.get("table_infos") or [],
        state.get("metric_infos") or [],
        semantic_ir,
    )
    runtime.stream_writer({
        "semantic_query_contract": compact_query_contract(
            semantic_ir.get("query_contract")
        )
    })
    model_required = use_model_sql(
        semantic_ir,
        query_settings=settings.query,
    )
    deterministic_draft_sql = compile_semantic_ir(semantic_ir)
    compiled_sql = None if model_required else deterministic_draft_sql
    if compiled_sql:
        runtime.stream_writer({"stage": "确定性编译 SQL"})
        updated_ir = dict(semantic_ir)
        updated_ir["compiler"] = {
            **dict(semantic_ir.get("compiler") or {}),
            "emitted": True,
            "sql_dialect": "mysql",
        }
        logger.info("语义 IR 确定性编译 SQL 完成")
        return {"sql": compiled_sql, "semantic_ir": updated_ir, "error": None}
    if model_required:
        runtime.stream_writer({
            "stage": "复杂语义模型生成 SQL",
            "model_sql_route": dict(
                semantic_ir.get("model_sql_policy") or {}
            ),
        })
    timeout = task_timeout(runtime, "sql")
    phase_budget = model_phase_budget(
        state,
        "sql",
        minimum_start_ms=int(
            settings.query.generation_llm_min_start_budget_ms
        ),
        timeout_seconds=timeout,
    )
    if not phase_budget["allowed"]:
        runtime_fallback = runtime.context.get("model_sql_fallback_enabled")
        fallback_allowed = deterministic_model_fallback_allowed(
            semantic_ir,
            runtime_override=(
                bool(runtime_fallback)
                if runtime_fallback is not None else None
            ),
            query_settings=settings.query,
        )
        if (
            model_required
            and fallback_allowed
            and deterministic_draft_is_safe(
                semantic_ir, deterministic_draft_sql
            )
        ):
            updated_ir = dict(semantic_ir)
            updated_ir["compiler"] = {
                **dict(semantic_ir.get("compiler") or {}),
                "emitted": True,
                "sql_dialect": "mysql",
                "emission_reason": "closed_draft_model_budget_fallback",
            }
            updated_ir["model_sql_policy"] = {
                **dict(semantic_ir.get("model_sql_policy") or {}),
                "actual_action": "deterministic_draft",
                "physical_llm_call": False,
                "fallback_reason": phase_budget["reason_code"],
            }
            runtime.stream_writer({
                "stage": "模型预算不足，使用已闭合确定性草案",
                "model_sql_route": updated_ir["model_sql_policy"],
            })
            logger.info(
                "必要模型预算不足，安全使用闭合确定性草案 reason=%s",
                phase_budget["reason_code"],
            )
            return {
                "sql": deterministic_draft_sql,
                "semantic_ir": updated_ir,
                "error": None,
                "model_phase_budget": phase_budget,
            }
        if model_required:
            semantic_ir = dict(semantic_ir)
            semantic_ir["model_sql_policy"] = {
                **dict(semantic_ir.get("model_sql_policy") or {}),
                "actual_action": "clarification",
                "physical_llm_call": False,
                "fallback_reason": phase_budget["reason_code"],
                "deterministic_fallback_allowed": fallback_allowed,
            }
        clarification = build_model_clarification(
            node_id="generate_sql",
            reason_code=phase_budget["reason_code"],
            semantic_ir=semantic_ir,
            retryable=True,
        )
        message = clarification_message(clarification)
        runtime.stream_writer({"clarification": clarification})
        runtime.stream_writer({"error": message})
        return {
            "sql": "",
            "authorized_sql": None,
            "error": message,
            "clarification": clarification,
            "model_phase_budget": phase_budget,
            "semantic_ir": semantic_ir,
        }
    try:
        query = state["query"]
        table_infos = state["table_infos"]
        metric_infos = state["metric_infos"]
        date_info = state["date_info"]
        db_info = state["db_info"]
        # 多轮对话历史：旧→新排列的 [{query, sql}, ...]，无历史时为空列表
        chat_history = state.get("chat_history", [])
        # 查询规划：plan_sql 节点产出的三步规划（表连接/口径/排序），简单问题为空
        plan = state.get("plan", "")
        semantic_contract = build_semantic_contract(
            query,
            table_infos,
            chat_history,
            metric_infos,
            semantic_ir,
        )
        source_context = state.get("data_source_context") or {}
        source_mode = str(source_context.get("source_mode") or "dw_only")
        if source_mode != "dw_only":
            intent = str(
                source_context.get("comparison_intent") or "upload_only"
            )
            mapping = state.get("source_mapping") or {}
            source_rules = [
                "上传表只使用 Schema 中公开的 upload_* 逻辑表名；禁止猜测内部物理表名。",
                "只能引用给定字段，禁止 SELECT *，禁止读取未选择的数据集。",
            ]
            if source_mode == "upload_only":
                source_rules.append("本轮禁止引用任何正式 DW 表。")
            if intent == "independent_aggregate_compare":
                source_rules.append(
                    "仅允许分别聚合各来源后用 UNION ALL 对齐结果；"
                    "禁止上传表与 DW 做行级 JOIN。结果必须带 source_label。"
                )
            if intent in {
                "lookup_enrichment", "union", "upload_dw_row_join",
                "multi_upload",
            }:
                source_rules.append(
                    "只允许使用已确认 mapping 中的字段关联，连接条件不得自行扩展。"
                )
            semantic_contract = (
                semantic_contract
                + "\n数据源合同：\n"
                + "\n".join(f"- {item}" for item in source_rules)
                + (
                    "\n已确认映射："
                    + yaml.dump(
                        mapping.get("mappings") or [],
                        allow_unicode=True,
                        sort_keys=False,
                    )
                    if mapping.get("status") == "confirmed"
                    else ""
                )
            )
        compact, context_budget = build_model_prompt_payload(
            state,
            "generate_sql",
            settings=settings,
        )

        prompt_template = PromptTemplate(
            template=load_prompt("generate_sql"),
            input_variables=["query", "table_infos", "metric_infos", "date_info", "db_info",
                             "chat_history", "memory_context", "training_examples",
                             "plan", "semantic_contract", "semantic_ir_summary"],
        )
        # Keep the raw provider message until the unified invocation boundary
        # has recorded finish_reason, request id and token usage.  SQL text is
        # extracted only after transport/output validation succeeds.
        chain = prompt_template | (llm or get_task_llm(runtime, "sql"))
        # 各上下文用 yaml 序列化：allow_unicode 保留中文，sort_keys 保持字段原始顺序
        payload = {
            "query": query,
            "table_infos": compact["table_infos"],
            "metric_infos": compact["metric_infos"],
            "date_info": yaml.dump(date_info,allow_unicode=True,sort_keys=False),
            "db_info": yaml.dump(db_info,allow_unicode=True,sort_keys=False),
            "chat_history": compact["chat_history"],
            "memory_context": compact["memory_context"],
            "training_examples": compact["training_examples"],
            "plan": plan,
            "semantic_contract": semantic_contract,
            "semantic_ir_summary": yaml.dump(
                model_semantic_summary(
                    semantic_ir,
                    deterministic_draft_sql=(
                        deterministic_draft_sql if model_required else None
                    ),
                    query_settings=settings.query,
                ),
                allow_unicode=True,
                sort_keys=False,
            ),
        }
        recovery_payload = {
            **payload,
            # Recovery is a smaller rendering of the exact same evidence.  It
            # cannot add tables, values, history or examples that were absent
            # from the first attempt.
            "chat_history": "[]\n",
            "memory_context": "{}\n",
            "training_examples": "[]\n",
        }
        invocation = await invoke_llm(
            chain,
            payload,
            node_id="generate_sql",
            route="sql",
            timeout_seconds=timeout,
            max_retries=1,
            core_path=True,
            runtime=runtime,
            request_deadline_at=route_deadline_at(
                state,
                "sql",
                timeout_seconds=timeout,
            ),
            recovery_payload=recovery_payload,
            recovery_min_ms=int(
                settings.query.llm_recovery_min_budget_ms
            ),
            primary_min_ms=int(
                settings.query.generation_llm_primary_min_budget_ms
            ),
            retryable_reason_codes={
                "llm.connection_error",
                "llm.rate_limited",
                "llm.server_error",
                "llm.stream_interrupted",
                "llm.timeout",
                "llm.empty_output",
                "llm.truncated_output",
                "llm.invalid_structured_output",
            },
            output_validator=sql_output_validator,
        )

        logger.info("生成SQL完成")

        return {
            "sql": normalize_sql_output(invocation.value),
            "error": None,
            "semantic_ir": semantic_ir,
            "context_budget": context_budget,
            "llm_attempts": invocation.attempts,
            **({"degradations": invocation.degradations} if invocation.degradations else {}),
        }
    except LLMInvocationError as exc:
        logger.error(
            "生成SQL失败 code=%s error_type=%s",
            exc.reason_code,
            exc.error_type,
        )
        clarification = build_model_clarification(
            node_id="generate_sql",
            reason_code=exc.reason_code,
            semantic_ir=semantic_ir,
            retryable=True,
        )
        message = clarification_message(clarification)
        runtime.stream_writer({"clarification": clarification})
        runtime.stream_writer({"error": message})
        return {
            "sql": "",
            "authorized_sql": None,
            "error": message,
            "clarification": clarification,
            "model_phase_budget": phase_budget,
            "semantic_ir": semantic_ir,
            "llm_attempts": exc.attempts,
            "degradations": exc.degradations,
        }
    except Exception as exc:
        failure = pre_invoke_failure(
            node_id="generate_sql",
            route="sql",
            error=exc,
            core_path=True,
            runtime=runtime,
        )
        logger.error(
            "生成SQL准备失败 code=%s error_type=%s",
            failure.reason_code,
            failure.error_type,
        )
        runtime.stream_writer({"error": failure.user_message})
        return {
            "sql": "",
            "authorized_sql": None,
            "error": failure.user_message,
            "semantic_ir": semantic_ir,
            "llm_attempts": failure.attempts,
            "degradations": failure.degradations,
        }
