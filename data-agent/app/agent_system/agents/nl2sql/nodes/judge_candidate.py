"""已校验 SQL 的独立语义复核与候选判优。

默认正确性路由会对每条已通过安全、权限、语法与成本校验的主候选
再做一次独立业务语义复核。复核器返回的 SQL 必须重新通过全部确定性校验；
如果复核超时、生成失败或候选无效，保留已校验的 A 并记录降级证据。
关闭 ``sql_semantic_review_enabled`` 后才回到 TODO-45 的置信度自适应候选消融路由。

设计要点：
    - 默认复核不受与主候选同源的 confidence band 阻断，但受一次调用和
      adaptive_candidate_deadline_ms 有界总预算约束。
    - 备选生成全程记录 llm_attempts（成功/超时/取消/空输出）与 degradations，
      保证可观测性与可回放。
    - 只有关闭默认复核的旧消融路由仍可在主备均不可信时阻断。
"""
from __future__ import annotations

import logging
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent_system.agents.nl2sql.context import DataAgentContext
from app.agent_system.agents.nl2sql.llm import get_task_llm, task_timeout
from app.agent_system.agents.nl2sql.semantic_hints import build_semantic_contract
from app.agent_system.agents.nl2sql.state import DataAgentState
from app.agent_system.agents.nl2sql.services.candidate_confidence import (
    CANDIDATE_DECISION_SCHEMA,
    CONFIDENCE_VERSION,
    assess_candidate,
    choose_candidate,
    compact_candidate_decision,
    normalized_sql_equal,
    actionable_weaknesses,
    should_expand_candidate,
    should_review_candidate,
)
from app.agent_system.runtime.performance import elapsed_ms
from app.agent_system.agents.nl2sql.services.prompt_context import build_candidate_prompt_payload
from app.agent_system.runtime.llm_attempts import normalize_degradation
from app.agent_system.agents.nl2sql.llm import (
    LLMInvocationError,
    invoke_llm,
    normalize_sql_output,
    pre_invoke_failure,
    sql_output_validator,
)
from app.agent_system.governance.deadline import route_deadline_at
from app.agent_system.agents.nl2sql.prompts.prompt_loader import load_prompt
from app.agent_system.agents.nl2sql.nodes.validate_sql import validate_candidate_sql
from app.agent_system.agents.nl2sql.services.semantic_repair import semantic_repair_ledger_entry

logger = logging.getLogger(__name__)

# 候选生成任务 LLM（独立路由，可用与主 SQL 不同的模型/超时）
llm = None


def _decision(primary: dict, *, triggered: bool, trigger_reason: str,
              decision_mode: str,
              selected: str = "A", selection_reason: str = "primary_fast_path",
              alternative: dict | None = None, extra_call_used: bool = False,
              generation_ms: int = 0, validation_ms: int = 0,
              candidate_outcome: str | None = None,
              sql_delta: bool | None = None) -> dict:
    """组装紧凑的候选决策证据字典，供可观测性与 state.candidate_decision 使用。

    核心实现思路：把主候选、备选（可选）、触发原因、选择结果、耗时与结果
    打包成 compact_candidate_decision 紧凑格式，便于前端展示与日志审计。
    """
    candidates = [primary, *([alternative] if alternative else [])]
    return compact_candidate_decision({
        "schema": CANDIDATE_DECISION_SCHEMA,
        "version": CONFIDENCE_VERSION,
        "decision_mode": decision_mode,
        "adaptive_triggered": bool(
            triggered and decision_mode == "adaptive_expansion"
        ),
        "semantic_review_triggered": bool(
            triggered and decision_mode == "semantic_review"
        ),
        "semantic_review_eligible": bool(
            decision_mode == "semantic_review" and primary.get("valid")
        ),
        "trigger_reason": trigger_reason,
        "selected": selected,
        "selection_reason": selection_reason,
        "sql_delta": sql_delta,
        "candidate_count": len(candidates),
        "extra_call_used": extra_call_used,
        "generation_ms": generation_ms,
        "validation_ms": validation_ms,
        # 候选结果：not_triggered / unknown / selected_b / same_sql_kept_a /
        # valid_b_kept_a / invalid_b / timeout / generation_error / empty_output
        "candidate_outcome": candidate_outcome or ("not_triggered" if not triggered else "unknown"),
        "candidates": candidates,
    })


def _blocked_result(runtime, decision: dict) -> dict:
    """主备候选均不可信时返回阻断结果，引导用户补充更明确的提问要素。

    核心实现思路：把候选决策与可读错误推送给前端，返回 error 与空 authorized_sql，
    供路由走向 END 终止，避免执行低质量 SQL 误导用户。
    """
    error = "候选置信度不足：主候选与备选均未达到可信执行条件，请补充更明确的指标、维度或筛选条件"
    runtime.stream_writer({"candidate_decision": decision})
    runtime.stream_writer({"error": error})
    return {"candidate_decision": decision, "error": error, "authorized_sql": None}


async def judge_candidate(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    settings = runtime.context["settings"]
    """对已验证主候选做有界语义复核，或运行旧候选消融。

    核心功能：评估主候选 SQL 的置信度，高置信直接放行；低置信触发备选生成，
    校验后做确定性判优，选出最终候选或阻断。

    业务逻辑：
        1. 推送阶段"评估 SQL 置信度"
        2. 评估主候选 A 的置信度（assess_candidate）
        3. should_expand_candidate 判断是否触发备选：不触发则走 fast path 放行 A
        4. 触发则推送阶段"生成低置信备选 SQL"，按总预算计算本次备选超时
        5. 构造 adaptive_candidate prompt，注入主候选弱点与上下文，调 LLM 生成备选
        6. 备选生成异常/空输出：记录 llm_attempt + degradation，按主候选 band 决定
           放行 A 或阻断（_blocked_result）
        7. 备选生成成功：校验备选（validate_candidate_sql），评估备选 B 置信度
        8. choose_candidate 在 A/B 间做确定性判优，组装 decision 并写回 state
        9. 选中 B 则用 B 的 sql/authorized_sql 覆盖 state；选中 A 则保留主链路结果
        10. 主备均不可信（selected=="none"）时返回 _blocked_result

    Returns:
        dict: candidate_decision + （选中 B 时）sql/authorized_sql/error/validation_evidence
    """
    runtime.stream_writer({"stage": "评估 SQL 置信度"})
    # 主候选校验证据：优先用 state 中的 validation_evidence，缺失时按 error 反推
    primary_validation = state.get("validation_evidence") or {"valid": not state.get("error")}
    # 评估主候选 A 的置信度（band/score/weaknesses）
    primary = assess_candidate(
        state.get("sql", ""),
        state,
        validation=primary_validation,
        label="A",
        query_settings=settings.query,
    )
    # 默认正确性路由不把“安全可执行”误当成“业务结果正确”：
    # 任何已通过确定性校验的 A 在预算允许时都独立语义复核一次。
    # 显式关闭语义复核后，才回到 TODO-45 的置信度自适应消融路由。
    semantic_review = bool(
        getattr(settings.query, "sql_semantic_review_enabled", True)
    )
    if semantic_review:
        expand, trigger_reason = should_review_candidate(
            primary,
            state,
            elapsed_ms(state),
            query_settings=settings.query,
        )
    else:
        expand, trigger_reason = should_expand_candidate(
            primary,
            state,
            elapsed_ms(state),
            query_settings=settings.query,
        )
    decision_mode = "semantic_review" if semantic_review else "adaptive_expansion"
    if not expand:
        # A corrected SQL can flow through this node again.  Preserve the first
        # review decision instead of overwriting it with a misleading
        # not-triggered record; this also enforces one review per eligible case.
        if trigger_reason == "semantic_review_already_used":
            decision = compact_candidate_decision(
                state.get("candidate_decision")
            )
            runtime.stream_writer({"candidate_decision": decision})
            return {"candidate_decision": decision}
        # fast path：高置信直接放行主候选，不额外调用 LLM
        decision = _decision(
            primary,
            triggered=False,
            trigger_reason=trigger_reason,
            decision_mode=decision_mode,
        )
        runtime.stream_writer({"candidate_decision": decision})
        if (
            semantic_review
            and primary.get("valid")
            and trigger_reason == "semantic_review_budget_insufficient"
        ):
            degradation = normalize_degradation({
                "node_id": "judge_candidate",
                "route": "candidate",
                "reason_code": "semantic_review_budget_insufficient",
                "severity": "error",
                "affects_core_path": True,
                "message": "已校验主候选剩余预算不足，未完成必需的独立语义复核",
            })
            runtime.stream_writer({"degradation": degradation})
            return {
                "candidate_decision": decision,
                "degradations": [degradation],
            }
        return {"candidate_decision": decision}

    runtime.stream_writer({
        "stage": "独立复核 SQL" if semantic_review else "生成低置信备选 SQL"
    })
    # 备选超时取三者最小：配置上限、任务路由超时、剩余总预算（避免拖垮整体延迟）
    timeout = min(
        float(settings.query.adaptive_candidate_timeout_seconds),
        task_timeout(runtime, "candidate"),
        max(0.1, (int(settings.query.adaptive_candidate_deadline_ms) - elapsed_ms(state)) / 1000),
    )
    prompt = PromptTemplate(
        template=load_prompt("adaptive_candidate"),
        input_variables=[
            "query", "sql", "weaknesses", "table_infos", "metric_infos",
            "semantic_ir", "semantic_contract", "db_info",
        ],
    )
    chain = (
        prompt
        | (llm or get_task_llm(runtime, "candidate"))
        | StrOutputParser()
    )
    # actionable_weaknesses：把主候选弱点转为可操作的修正提示；缺失时回退全部弱点
    concrete_weaknesses = actionable_weaknesses(primary) or list(primary.get("weaknesses") or [])
    compact, context_budget = build_candidate_prompt_payload(
        state,
        concrete_weaknesses,
        settings=settings,
    )
    generation_started = time.perf_counter()
    invocation = None
    candidate_is_required = semantic_review or primary.get("band") == "low"
    try:
        # 调 LLM 生成备选 SQL，注入主候选弱点与上下文
        invocation = await invoke_llm(chain, {
            "query": state.get("query", ""),
            "sql": state.get("sql", ""),
            "weaknesses": compact["weaknesses"],
            "table_infos": compact["table_infos"],
            "metric_infos": compact["metric_infos"],
            "semantic_ir": compact["semantic_ir"],
            "semantic_contract": build_semantic_contract(
                state.get("query", ""),
                state.get("table_infos") or [],
                [],
                state.get("metric_infos") or [],
                state.get("semantic_ir") or {},
            ),
            "db_info": compact["db_info"],
        }, node_id="judge_candidate", route="candidate",
            timeout_seconds=timeout, core_path=candidate_is_required, runtime=runtime,
            request_deadline_at=route_deadline_at(
                state,
                "candidate",
                timeout_seconds=timeout,
            ),
            output_validator=sql_output_validator)
        candidate_sql = normalize_sql_output(invocation.value)
    except LLMInvocationError as exc:
        # 备选生成异常（含超时）：记录 attempt + degradation，按主候选 band 决定放行或阻断
        generation_ms = max(0, int((time.perf_counter() - generation_started) * 1000))
        attempt_status = exc.attempts[-1]["status"]
        # 语义复核是对已验证 A 的有界正确性加固；供应商超时/失败
        # 不能撤销 A 已通过的确定性授权。关闭复核的消融路由仍保留旧行为。
        selected = (
            "A" if semantic_review
            else "none" if primary.get("band") == "low"
            else "A"
        )
        selection_reason = (
            "semantic_review_generation_failed_primary_kept"
            if semantic_review
            else "no_trustworthy_candidate"
            if selected == "none"
            else "alternative_generation_failed"
        )
        decision = _decision(
            primary, triggered=True, trigger_reason=trigger_reason,
            decision_mode=decision_mode,
            selected=selected, selection_reason=selection_reason, extra_call_used=True,
            generation_ms=generation_ms,
            candidate_outcome="timeout" if attempt_status == "timeout" else "generation_error",
        )
        if selected == "none":
            # 阻断：返回 _blocked_result 并附带降级与 attempt 证据
            result = _blocked_result(runtime, decision)
            result["degradations"] = exc.degradations
            result["context_budget"] = context_budget
            result["llm_attempts"] = exc.attempts
            return result
        # 放行 A：保留主链路结果，仅记录降级
        runtime.stream_writer({"candidate_decision": decision})
        return {
            "candidate_decision": decision, "degradations": exc.degradations,
            "context_budget": context_budget, "llm_attempts": exc.attempts,
        }

    candidate_sql = str(candidate_sql or "").strip()
    generation_ms = max(0, int((time.perf_counter() - generation_started) * 1000))
    if not candidate_sql:
        # 备选空输出：记录 attempt + degradation，按主候选 band 决定放行或阻断
        # Defensive only: invoke_llm normally classifies this before returning.
        empty_failure = pre_invoke_failure(
            node_id="judge_candidate",
            route="candidate",
            timeout_seconds=timeout,
            error=ValueError("empty_model_output_after_boundary"),
            core_path=candidate_is_required,
            runtime=runtime,
        )
        selected = (
            "A" if semantic_review
            else "none" if primary.get("band") == "low"
            else "A"
        )
        decision = _decision(
            primary, triggered=True, trigger_reason=trigger_reason,
            decision_mode=decision_mode,
            selected=selected,
            selection_reason=(
                "semantic_review_empty_primary_kept"
                if semantic_review
                else "no_trustworthy_candidate"
                if selected == "none"
                else "alternative_empty"
            ),
            extra_call_used=True,
            generation_ms=generation_ms, candidate_outcome="empty_output",
        )
        degradation = normalize_degradation({
            "node_id": "judge_candidate", "route": "candidate",
            "reason_code": "candidate_empty_output", "severity": "error",
            "affects_core_path": candidate_is_required,
            "message": "候选模型返回空输出",
            "duration_ms": generation_ms,
        })
        runtime.stream_writer({"degradation": degradation})
        if selected == "none":
            result = _blocked_result(runtime, decision)
            result["context_budget"] = context_budget
            result["degradations"] = [degradation]
            result["llm_attempts"] = empty_failure.attempts
            return result
        runtime.stream_writer({"candidate_decision": decision})
        return {
            "candidate_decision": decision, "context_budget": context_budget,
            "degradations": [degradation], "llm_attempts": empty_failure.attempts,
        }

    # 备选生成成功：统一调用边界已记录成功 attempt，继续校验备选 SQL
    validation_started = time.perf_counter()
    # 复用统一校验流水线校验备选 SQL
    validation = await validate_candidate_sql(candidate_sql, state, runtime)
    validation_ms = max(0, int((time.perf_counter() - validation_started) * 1000))
    # 评估备选 B 的置信度
    alternative = assess_candidate(
        candidate_sql,
        state,
        validation=validation,
        label="B",
        query_settings=settings.query,
    )
    primary_sql = str(state.get("sql") or "").strip()
    alternative_sql = str(validation.get("sql") or candidate_sql).strip()
    same_sql = normalized_sql_equal(primary_sql, alternative_sql)
    sql_delta = None if same_sql is None else not same_sql
    # 语义复核 B 除了重新通过全部确定性校验，还必须带来
    # 规范化 AST 上可证明的 SQL 变化。仅格式、大小写或别名写法不同
    # 必须保留 A，避免把同 SQL 重执行的波动误归因为 B 的收益。
    if semantic_review:
        if not alternative.get("valid"):
            selected = "A"
            selection_reason = "semantic_review_rejected_primary_kept"
        elif same_sql is True:
            selected = "A"
            selection_reason = "semantic_review_same_sql_primary_kept"
        else:
            selected = "B"
            selection_reason = "semantic_review_validated"
    else:
        selected, selection_reason = choose_candidate(
            primary,
            alternative,
            primary_sql=primary_sql,
            alternative_sql=alternative_sql,
            query_settings=settings.query,
        )
    decision = _decision(
        primary, triggered=True, trigger_reason=trigger_reason,
        decision_mode=decision_mode,
        selected=selected, selection_reason=selection_reason,
        alternative=alternative, extra_call_used=True,
        generation_ms=generation_ms, validation_ms=validation_ms,
        sql_delta=sql_delta,
        candidate_outcome=(
            "selected_b"
            if selected == "B"
            else "same_sql_kept_a"
            if same_sql is True and alternative.get("valid")
            else "valid_b_kept_a"
            if alternative.get("valid")
            else "invalid_b"
        ),
    )
    runtime.stream_writer({"candidate_decision": decision})
    repair_ledger = [semantic_repair_ledger_entry(
        validation,
        phase="candidate_b_validate",
        committed_at_phase=selected == "B",
    )]
    logger.info(
        "候选判优完成 A=%s B=%s selected=%s reason=%s elapsed_ms=%s",
        primary["score"], alternative["score"], selected, selection_reason, elapsed_ms(state),
    )
    if selected == "none":
        # 主备均不可信：阻断
        result = _blocked_result(runtime, decision)
        result["context_budget"] = context_budget
        result["llm_attempts"] = invocation.attempts
        result["semantic_repair_ledger"] = repair_ledger
        return result
    if selected == "B":
        # 选中备选：用 B 的 sql/authorized_sql 覆盖 state，清空 error
        return {
            "sql": validation["sql"],
            "authorized_sql": validation["authorized_sql"],
            "error": None,
            "validation_evidence": validation,
            "candidate_decision": decision,
            "context_budget": context_budget,
            "llm_attempts": invocation.attempts,
            "semantic_repair_ledger": repair_ledger,
        }
    # 选中 A：保留主链路结果，仅记录候选决策与预算
    return {
        "candidate_decision": decision,
        "context_budget": context_budget,
        "llm_attempts": invocation.attempts,
        "semantic_repair_ledger": repair_ledger,
    }
