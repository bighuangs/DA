import json
import logging
import time
import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent_system.agents.nl2sql.context import DataAgentContext
from app.agent_system.agents.nl2sql.llm import get_task_llm, task_timeout
from app.agent_system.agents.nl2sql.state import DataAgentState
from app.agent_system.agents.nl2sql.semantic_hints import build_semantic_contract
from app.agent_system.agents.nl2sql.semantic_contract import (
    build_query_contract,
    compact_query_contract,
)
from app.agent_system.agents.nl2sql.llm import (
    LLMInvocationError,
    invoke_llm,
    pre_invoke_failure,
)
from app.agent_system.runtime.llm_attempts import normalize_degradation
from app.agent_system.agents.nl2sql.services.prompt_context import build_model_prompt_payload
from app.agent_system.governance.deadline import model_phase_budget, route_deadline_at
from app.agent_system.agents.nl2sql.prompts.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

llm = None


def build_structured_plan(state: DataAgentState) -> str:
    """Build a bounded plan from semantic and live-schema evidence only."""

    semantic_ir = dict(state.get("semantic_ir") or {})
    coverage = dict(semantic_ir.get("semantic_coverage") or {})
    model_routed = bool(
        (semantic_ir.get("model_sql_policy") or {}).get("use_model")
    )
    # Defence in depth for callers that persisted or constructed an IR before
    # routing was finalized: a model route can never inherit a stale closed
    # compiler contract.  Rebuild from the current authorized evidence so the
    # deterministic planning fallback and the later SQL prompt agree on the
    # same guided authority.  Deterministic ablations keep their closed view.
    contract = (
        build_query_contract(
            str(state.get("query") or ""),
            state.get("table_infos") or [],
            state.get("metric_infos") or [],
            semantic_ir,
        )
        if model_routed
        else semantic_ir.get("query_contract")
    )
    query_contract = compact_query_contract(contract)
    authority = (
        "closed"
        if not model_routed and query_contract.get("authority") == "closed"
        else "guided"
    )
    table_names = list(dict.fromkeys(
        str(item.get("name") or "")
        for item in (state.get("table_infos") or [])
        if item.get("name")
    ))
    payload = {
        "schema_version": "structured-plan-v1",
        "source": "deterministic_semantic_contract",
        "authority": authority,
        "semantic_family": semantic_ir.get("semantic_family"),
        "required_tables": (
            list(semantic_ir.get("required_tables") or table_names)
            if authority == "closed" else table_names
        ),
        "detected_slots": (
            list(coverage.get("detected_slots") or [])[:24]
            if authority == "closed" else []
        ),
        "unresolved_slots": (
            list(coverage.get("unresolved_slots") or [])[:12]
            if authority == "closed" else []
        ),
        "query_contract": query_contract,
        "constraints": (
            [
                "只使用授权表与字段",
                "完整保留过滤、否定、集合、窗口、排序和限制槽位",
                "生成后重新执行安全、语义与成本校验",
            ]
            if authority == "closed"
            else [
                "只使用授权表与字段",
                "用户问题是最终语义权威，规划不得删除其任何明确要求",
                "生成后重新执行安全、语义与成本校验",
            ]
        ),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


async def plan_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    settings = runtime.context["settings"]
    """
    基于召回上下文为复杂问题生成 SQL 查询规划，供 generate_sql 落实为 SQL。

    一句话总述：在生成 SQL 前先让大模型输出一份"涉及的表与连接 / 查询逻辑与口径 /
                排序与限制"三步规划（不写任何 SQL），注入 generate_sql prompt 作为
                生成骨架，提升多表连接与复杂聚合场景的 SQL 准确率。

    业务逻辑：
        1. 向前端推送阶段"规划查询"
        2. 从 state 取出 query / table_infos / metric_infos / date_info / db_info /
           chat_history / recall_training 等上下文
        3. 构造 prompt 链（plan_sql 模板 → llm → StrOutputParser），各上下文用 yaml 序列化
        4. 大模型返回纯文本规划（三步）
        5. 返回 {"plan": plan} 更新到 state，供 generate_sql 消费
        6. 异常时返回空 plan 不阻断主流程（generate_sql 仍可独立工作）

    设计说明：
        - 本节点仅在 ``decide_plan_route`` 选择规划时由条件边接入，简单问题直接跳过。
        - 规划只定"表连接、口径、排序"，不写 SQL，避免与 generate_sql 职责重叠。
        - 采用独立 plan 节点与条件边：复杂问题显式规划，简单问题直接跳过，
          使规划证据可单独观测且不增加简单查询开销。

    Args:
        state: 流水线状态，包含 query 与召回合并后的 table_infos/metric_infos 等
        runtime: LangGraph 运行时，携带上下文与流式写入器

    Returns:
        dict: {"plan": str} —— 三步查询规划文本；异常时为空串
    """
    runtime.stream_writer({"stage": "规划查询"})
    node_started = time.perf_counter()
    structured_plan = build_structured_plan(state)

    def emit_timing(
        provider_ms: int,
        status: str,
        *,
        queue_wait_ms: int = 0,
        recovery: bool = False,
    ) -> dict:
        timing = {
            "contract_version": "planning-timing-v2",
            "queue_wait_ms": max(0, int(queue_wait_ms)),
            "provider_ms": max(0, int(provider_ms)),
            "node_ms": max(
                0, int((time.perf_counter() - node_started) * 1000)
            ),
            "time_to_first_token_ms": None,
            "time_to_last_token_ms": None,
            "completion_mode": "non_stream_completion_latency",
            "token_timing_available": False,
            "status": status,
            "recovery": bool(recovery),
        }
        runtime.stream_writer({"planning_timing": timing})
        return timing

    if not bool(settings.query.planning_llm_enabled):
        emit_timing(0, "deterministic", recovery=False)
        return {
            "plan": structured_plan,
            "planning_evidence": {
                "schema_version": "planning-decision-v1",
                "source": "deterministic_semantic_contract",
                "physical_llm_call": False,
                "reason_code": "planning.optional_model_disabled",
            },
        }

    timeout = task_timeout(runtime, "planning")
    phase_budget = model_phase_budget(
        state,
        "planning",
        minimum_start_ms=int(
            settings.query.planning_llm_min_start_budget_ms
        ),
        timeout_seconds=timeout,
    )
    if not phase_budget["allowed"]:
        degradation = normalize_degradation({
            "node_id": "plan_sql",
            "route": "planning",
            "reason_code": "planning.skipped_insufficient_budget",
            "severity": "warning",
            "affects_core_path": False,
            "message": "可选规划模型预算不足，使用结构化确定性规划",
            "duration_ms": 0,
        })
        runtime.stream_writer({"degradation": degradation})
        emit_timing(0, "deterministic_budget_fallback")
        return {
            "plan": structured_plan,
            "model_phase_budget": phase_budget,
            "planning_evidence": {
                "schema_version": "planning-decision-v1",
                "source": "deterministic_semantic_contract",
                "physical_llm_call": False,
                "reason_code": phase_budget["reason_code"],
            },
            "degradations": [degradation],
        }

    try:
        query = state["query"]
        table_infos = state["table_infos"]
        metric_infos = state["metric_infos"]
        date_info = state["date_info"]
        db_info = state["db_info"]
        chat_history = state.get("chat_history", [])
        semantic_contract = build_semantic_contract(
            query,
            table_infos,
            chat_history,
            metric_infos,
            state.get("semantic_ir") or {},
        )
        compact, context_budget = build_model_prompt_payload(
            state,
            "plan_sql",
            settings=settings,
        )

        prompt_template = PromptTemplate(
            template=load_prompt("plan_sql"),
            input_variables=["query", "table_infos", "metric_infos", "date_info",
                             "db_info", "chat_history", "memory_context",
                             "training_examples", "semantic_contract"],
        )
        chain = (
            prompt_template
            | (llm or get_task_llm(runtime, "planning"))
            | StrOutputParser()
        )
        payload = {
            "query": query,
            "table_infos": compact["table_infos"],
            "metric_infos": compact["metric_infos"],
            "date_info": yaml.dump(date_info, allow_unicode=True, sort_keys=False),
            "db_info": yaml.dump(db_info, allow_unicode=True, sort_keys=False),
            "chat_history": compact["chat_history"],
            "memory_context": compact["memory_context"],
            "training_examples": compact["training_examples"],
            "semantic_contract": semantic_contract,
        }
        recovery_payload = {
            **payload,
            "chat_history": "[]\n",
            "memory_context": "{}\n",
            "training_examples": "[]\n",
        }
        invocation = await invoke_llm(
            chain,
            payload,
            node_id="plan_sql",
            route="planning",
            timeout_seconds=timeout,
            max_retries=0,
            core_path=False,
            runtime=runtime,
            request_deadline_at=route_deadline_at(
                state,
                "planning",
                timeout_seconds=timeout,
            ),
            recovery_payload=recovery_payload,
            recovery_enabled=False,
        )
        plan = invocation.value

        final_attempt = invocation.attempts[-1]
        emit_timing(
            int(final_attempt.get("provider_ms") or 0),
            "success",
            queue_wait_ms=int(final_attempt.get("queue_wait_ms") or 0),
            recovery=bool(final_attempt.get("recovery")),
        )
        logger.info(f"生成查询规划完成：{plan}")
        return {
            "plan": plan,
            "context_budget": context_budget,
            "model_phase_budget": phase_budget,
            "planning_evidence": {
                "schema_version": "planning-decision-v1",
                "source": "optional_model",
                "physical_llm_call": True,
                "reason_code": "planning.model_success",
            },
            "llm_attempts": invocation.attempts,
            **({"degradations": invocation.degradations} if invocation.degradations else {}),
        }
    except LLMInvocationError as exc:
        final_attempt = exc.attempts[-1]
        emit_timing(
            int(final_attempt.get("provider_ms") or 0),
            final_attempt["status"],
            queue_wait_ms=int(final_attempt.get("queue_wait_ms") or 0),
            recovery=bool(final_attempt.get("recovery")),
        )
        logger.error(
            "生成查询规划失败 code=%s error_type=%s",
            exc.reason_code,
            exc.error_type,
        )
        return {
            "plan": structured_plan,
            "model_phase_budget": phase_budget,
            "planning_evidence": {
                "schema_version": "planning-decision-v1",
                "source": "deterministic_semantic_contract",
                "physical_llm_call": True,
                "reason_code": exc.reason_code,
            },
            "degradations": exc.degradations,
            "llm_attempts": exc.attempts,
        }
    except Exception as exc:
        failure = pre_invoke_failure(
            node_id="plan_sql",
            route="planning",
            timeout_seconds=timeout,
            error=exc,
            core_path=False,
            runtime=runtime,
        )
        emit_timing(0, "error")
        logger.error(
            "生成查询规划准备失败 code=%s error_type=%s",
            failure.reason_code,
            failure.error_type,
        )
        return {
            "plan": structured_plan,
            "model_phase_budget": phase_budget,
            "planning_evidence": {
                "schema_version": "planning-decision-v1",
                "source": "deterministic_semantic_contract",
                "physical_llm_call": False,
                "reason_code": failure.reason_code,
            },
            "degradations": failure.degradations,
            "llm_attempts": failure.attempts,
        }
