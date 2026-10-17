"""构建 TODO-42 语义 IR（无 LLM 调用）。

本节点把 query、table_infos、metric_infos、chat_history 交给 build_semantic_ir，
做确定性结构化意图解析，产出 semantic_ir 供后续 filter_table / filter_metric /
generate_sql 复用：高置信时可直接跳过筛表 LLM 与 SQL 生成 LLM，由编译器
确定性编译出 SQL，节省尾延迟并提升可重复性。
"""
import logging

from langgraph.runtime import Runtime

from app.agent_system.agents.nl2sql.context import DataAgentContext
from app.agent_system.agents.nl2sql.semantic_contract import (
    build_query_contract,
    compact_query_contract,
)
from app.agent_system.agents.nl2sql.semantic_ir import build_semantic_ir
from app.agent_system.agents.nl2sql.state import DataAgentState
from app.agent_system.agents.nl2sql.source_context import source_mode
from app.agent_system.agents.nl2sql.services.model_sql_policy import (
    decide_model_sql_policy,
    force_model_sql_policy,
)

logger = logging.getLogger(__name__)


async def build_semantic_ir_node(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    settings = runtime.context["settings"]
    """语义 IR 构建节点（确定性，无 LLM 调用）。

    核心功能：把当前 query 与已召回合并的 table_infos/metric_infos/chat_history
    交给 build_semantic_ir，做基于规则的结构化意图解析，产出 semantic_ir
    （含 intent、metric、dimensions、filters、compiler.supported 等）。

    业务逻辑：
        1. 向前端推送阶段"构建语义 IR"
        2. 调用 build_semantic_ir：基于规则解析意图，判定编译器是否支持
        3. 记录解析出的 intent 与 compiler.supported/reason，便于可观测性
        4. 返回 {"semantic_ir": ...} 写入 state，供后续节点复用

    设计说明：
        - 本节点不调用 LLM，纯确定性规则解析，零额外延迟。
        - compiler.supported 为 True 时，filter_table/filter_metric 可跳过 LLM，
          generate_sql 可由 compile_semantic_ir 确定性编译出 SQL。

    Args:
        state: 流水线状态，含 query/table_infos/metric_infos/chat_history
        runtime: LangGraph 运行时，携带流式写入器

    Returns:
        dict: {"semantic_ir": dict} —— 结构化语义意图与编译判定
    """
    runtime.stream_writer({"stage": "构建语义 IR"})
    # build_semantic_ir：基于规则解析意图，判定编译器是否支持当前查询结构
    query = str(state.get("query") or "")
    follow_up = dict(state.get("follow_up") or {})
    already_rewritten = bool(
        follow_up.get("action") not in {None, "new_topic"}
        and query
        == str(follow_up.get("standalone_question") or "").strip()
    )
    semantic_ir = build_semantic_ir(
        query,
        state.get("table_infos", []),
        state.get("metric_infos", []),
        # rewrite_follow_up has already turned an abbreviated utterance into
        # the standalone question. Feeding the same history into the IR builder
        # a second time can apply the previous state again and undo a literal
        # replacement (for example 华东 -> 华南).
        [] if already_rewritten else state.get("chat_history", []),
    )
    mode = source_mode(state)
    if mode != "dw_only":
        semantic_ir = dict(semantic_ir)
        semantic_ir["source_context"] = {
            "mode": mode,
            "comparison_intent": (
                state.get("data_source_context") or {}
            ).get("comparison_intent"),
            "mapping_confirmation_id": (
                state.get("data_source_context") or {}
            ).get("mapping_confirmation_id"),
        }
        semantic_ir["compiler"] = {
            **dict(semantic_ir.get("compiler") or {}),
            "supported": False,
            "reason": "guarded_upload_source_requires_model_sql",
        }
    semantic_ir = dict(semantic_ir)
    model_sql_policy = decide_model_sql_policy(
        semantic_ir,
        query,
        query_settings=settings.query,
    )
    if runtime.context.get("force_model_sql") is True:
        model_sql_policy = force_model_sql_policy(
            model_sql_policy,
            reason=str(
                runtime.context.get("force_model_sql_reason")
                or "trusted_runtime_contract"
            ),
        )
        model_sql_policy["deterministic_fallback_allowed"] = bool(
            runtime.context.get(
                "model_sql_fallback_enabled",
                settings.query.deterministic_sql_model_fallback_enabled,
            )
        )
    semantic_ir["model_sql_policy"] = model_sql_policy
    # The query contract must be built only after the final routing policy is
    # attached.  In correctness-first mode even a compiler-supported IR is a
    # guided model input, not an independently closed authority.  Building the
    # contract before this assignment allowed the optional planning stage to
    # consume a stale ``closed`` view until generate_sql rebuilt it later.
    semantic_ir["query_contract"] = build_query_contract(
        query,
        state.get("table_infos", []),
        state.get("metric_infos", []),
        semantic_ir,
    )
    logger.info(
        "语义 IR 完成 intent=%s supported=%s reason=%s sql_route=%s",
        semantic_ir.get("intent"),
        semantic_ir.get("compiler", {}).get("supported"),
        semantic_ir.get("compiler", {}).get("reason"),
        model_sql_policy.get("decision"),
    )
    coverage = dict(semantic_ir.get("semantic_coverage") or {})
    safe_coverage = {
        "schema_version": coverage.get("schema_version"),
        "registry_version": coverage.get("registry_version"),
        "question_fingerprint": coverage.get("question_fingerprint"),
        "coverage_score": coverage.get("coverage_score"),
        "detected_slots": [
            str(item)[:96] for item in (coverage.get("detected_slots") or [])[:40]
        ],
        "represented_slots": [
            str(item)[:96]
            for item in (coverage.get("represented_slots") or [])[:40]
        ],
        "unresolved_slots": [
            str(item)[:96]
            for item in (coverage.get("unresolved_slots") or [])[:40]
        ],
        "abstain_reasons": [
            str(item)[:96]
            for item in (coverage.get("abstain_reasons") or [])[:12]
        ],
        "compiler_mode": (semantic_ir.get("compiler") or {}).get("mode"),
        "compiler_supported": bool(
            (semantic_ir.get("compiler") or {}).get("supported")
        ),
        "model_sql_policy": model_sql_policy,
        "raw_question_persisted": False,
        "filter_values_persisted": False,
    }
    runtime.stream_writer({"semantic_coverage": safe_coverage})
    runtime.stream_writer({
        "semantic_query_contract": compact_query_contract(
            semantic_ir.get("query_contract")
        )
    })
    runtime.stream_writer({"model_sql_route": model_sql_policy})
    return {"semantic_ir": semantic_ir}
