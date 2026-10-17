"""在任何检索或 SQL 生成之前判定请求意图。"""

from __future__ import annotations

import logging
from typing import cast

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent_system.agents.nl2sql.context import DataAgentContext
from app.agent_system.agents.nl2sql.contracts.query_intent import (
    QueryIntentCategory,
    QueryIntentModelOutputError,
    build_query_intent_decision,
    model_scope_output_validator,
    parse_model_scope_category,
)
from app.agent_system.agents.nl2sql.llm import (
    LLMInvocationResult,
    get_task_llm,
    invoke_optional_llm,
    task_timeout,
)
from app.agent_system.agents.nl2sql.prompts.prompt_loader import load_prompt
from app.agent_system.agents.nl2sql.state import DataAgentState
from app.agent_system.governance.deadline import route_deadline_at
from app.domain.query.intent_policy import classify_deterministic_query_intent


logger = logging.getLogger(__name__)

# 与其他节点一致，保留可在独立测试中注入的模型槽位。
llm = None
_MODEL_FALLBACK = '{"category":"ambiguous"}'


def _decision_for_deterministic(
    *,
    category: QueryIntentCategory,
    original_query: str,
    standalone_query: str,
    reason_codes: tuple[str, ...],
    rule_ids: tuple[str, ...],
):
    return build_query_intent_decision(
        category=category,
        original_query=original_query,
        standalone_query=standalone_query,
        source="deterministic",
        reason_codes=reason_codes,
        rule_ids=rule_ids,
    )


async def _classify_unknown_with_model(
    state: DataAgentState,
    runtime: Runtime[DataAgentContext],
    *,
    original_query: str,
    standalone_query: str,
) -> tuple[object, LLMInvocationResult | None, list[dict]]:
    """仅判定业务范围；任何失败都返回可澄清的 ambiguous。"""

    manual_degradations: list[dict] = []
    try:
        prompt = PromptTemplate(
            template=load_prompt("query_intent"),
            input_variables=["original_query", "standalone_query"],
        )
        chain = prompt | (llm or get_task_llm(runtime, "keyword")) | StrOutputParser()
        timeout_seconds = task_timeout(runtime, "keyword")
        invocation = await invoke_optional_llm(
            chain,
            {
                "original_query": original_query,
                "standalone_query": standalone_query,
            },
            node_id="classify_query_intent",
            route="keyword",
            fallback=_MODEL_FALLBACK,
            timeout_seconds=timeout_seconds,
            core_path=False,
            runtime=runtime,
            request_deadline_at=route_deadline_at(
                state,
                "keyword",
                timeout_seconds=timeout_seconds,
            ),
            recovery_enabled=False,
            output_validator=model_scope_output_validator,
        )
        if invocation.degradations:
            return "ambiguous", invocation, manual_degradations
        try:
            return (
                parse_model_scope_category(invocation.value),
                invocation,
                manual_degradations,
            )
        except QueryIntentModelOutputError:
            degradation = {
                "node_id": "classify_query_intent",
                "route": "keyword",
                "reason_code": "llm.invalid_structured_output",
                "error_type": "QueryIntentModelOutputError",
                "core_path": False,
                "raw_text_logged": False,
            }
            runtime.stream_writer({"degradation": degradation})
            manual_degradations.append(degradation)
            return "ambiguous", invocation, manual_degradations
    except Exception as exc:
        # 模型装配失败也不能绕过分类门；只记录异常类型，不记录问题或供应商正文。
        degradation = {
            "node_id": "classify_query_intent",
            "route": "keyword",
            "reason_code": "intent.scope_model_unavailable",
            "error_type": type(exc).__name__,
            "core_path": False,
            "raw_text_logged": False,
        }
        runtime.stream_writer({"degradation": degradation})
        manual_degradations.append(degradation)
        return "ambiguous", None, manual_degradations


async def classify_query_intent(
    state: DataAgentState,
    runtime: Runtime[DataAgentContext],
):
    """输出版本化决策，并为终止分支提供固定响应。"""

    runtime.stream_writer({"stage": "识别请求意图"})
    original_query = str(state.get("original_query") or state.get("query") or "")
    standalone_query = str(state.get("query") or original_query)
    deterministic = classify_deterministic_query_intent(
        original_query,
        standalone_query,
    )

    invocation: LLMInvocationResult | None = None
    manual_degradations: list[dict] = []
    if deterministic.category != "unknown":
        decision = _decision_for_deterministic(
            category=cast(QueryIntentCategory, deterministic.category),
            original_query=original_query,
            standalone_query=standalone_query,
            reason_codes=deterministic.reason_codes,
            rule_ids=deterministic.rule_ids,
        )
    else:
        model_category, invocation, manual_degradations = (
            await _classify_unknown_with_model(
                state,
                runtime,
                original_query=original_query,
                standalone_query=standalone_query,
            )
        )
        category = cast(QueryIntentCategory, model_category)
        degraded = bool(
            manual_degradations
            or (invocation is not None and invocation.degradations)
        )
        decision = build_query_intent_decision(
            category=category,
            original_query=original_query,
            standalone_query=standalone_query,
            source="fallback" if degraded else "model",
            reason_codes=(
                "intent.scope_model_failed_closed"
                if degraded
                else f"intent.{category}.model"
            ,),
            rule_ids=(
                "fallback.ambiguous_clarification.v1"
                if degraded
                else "model.scope_classifier.v1"
            ,),
            model_used=invocation is not None,
            model_degraded=degraded,
        )

    payload = decision.privacy_summary()
    runtime.stream_writer({"query_intent": payload})
    output: dict[str, object] = {"query_intent": payload}
    response = decision.intent_response()
    if response:
        output["intent_response"] = response
        runtime.stream_writer({"intent_response": response})
    if invocation is not None and invocation.attempts:
        output["llm_attempts"] = invocation.attempts
    degradations = [
        *(invocation.degradations if invocation is not None else []),
        *manual_degradations,
    ]
    if degradations:
        output["degradations"] = degradations

    logger.info(
        "问数意图识别完成 category=%s source=%s rule_ids=%s query_hash=%s",
        payload["category"],
        payload["source"],
        payload["rule_ids"],
        payload["query_hashes"]["standalone"][:12],
    )
    return output


async def guard_forbidden_query_intent(
    state: DataAgentState,
    runtime: Runtime[DataAgentContext],
):
    """在多轮改写前只拒绝当前轮确定性危险请求。

    非危险请求返回空增量，让省略追问先完成语义闭合，再由完整分类节点判断
    业务范围。此节点不构造也不调用模型。
    """

    original_query = str(state.get("original_query") or state.get("query") or "")
    deterministic = classify_deterministic_query_intent(
        original_query,
        original_query,
    )
    if deterministic.category != "forbidden_operation":
        return {}

    runtime.stream_writer({"stage": "执行只读安全门"})
    decision = _decision_for_deterministic(
        category="forbidden_operation",
        original_query=original_query,
        standalone_query=original_query,
        reason_codes=deterministic.reason_codes,
        rule_ids=deterministic.rule_ids,
    )
    payload = decision.privacy_summary()
    response = decision.intent_response()
    runtime.stream_writer({"query_intent": payload})
    runtime.stream_writer({"intent_response": response})
    logger.info(
        "只读安全门拒绝请求 reason_codes=%s rule_ids=%s query_hash=%s",
        payload["reason_codes"],
        payload["rule_ids"],
        payload["query_hashes"]["original"][:12],
    )
    return {
        "query_intent": payload,
        "intent_response": response,
    }


__all__ = ["classify_query_intent", "guard_forbidden_query_intent"]
