"""在表/指标筛选之后装包 SQL Prompt 上下文。

本节点不调用 LLM，仅把已过滤的 table_infos / metric_infos / recall_training
按权限指纹与上下文预算做装包裁剪，产出最终注入 generate_sql 的上下文与
结构化预算证据（before/after tokens、裁剪原因、强制保留项）。
"""
import logging

from langgraph.runtime import Runtime

from app.agent_system.agents.nl2sql.context import DataAgentContext
from app.agent_system.agents.nl2sql.state import DataAgentState
from app.agent_system.governance.retrieval_budget import pack_prompt_context, policy_scope_fingerprint

logger = logging.getLogger(__name__)


async def pack_context(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """装包 SQL Prompt 上下文节点。

    核心功能：把 filter_table / filter_metric 之后的 table_infos、metric_infos
    与 recall_training 训练样例，连同 query 与 semantic_ir，交给 pack_prompt_context
    做按权限指纹的装包裁剪，产出最终注入 generate_sql 的上下文与预算证据。

    业务逻辑：
        1. 向前端推送阶段"装包上下文"
        2. 取出行列权限策略，计算权限指纹（用于上下文缓存键与裁剪一致性校验）
        3. 调用 pack_prompt_context：按预算裁剪 table_infos/metric_infos/training，
           返回裁剪后的三组上下文 + 结构化证据（before/after tokens、是否裁剪、缓存命中）
        4. 记录装包前后的 token 数与裁剪情况，便于可观测性
        5. 把 context 证据通过 stream_writer 推送给前端
        6. 返回裁剪后的 table_infos/metric_infos/recall_training 与 context_budget

    Args:
        state: 流水线状态，含 query/table_infos/metric_infos/recall_training/semantic_ir
        runtime: LangGraph 运行时，携带权限策略与流式写入器

    Returns:
        dict: {"table_infos", "metric_infos", "recall_training", "context_budget"}
              —— 裁剪后的上下文与预算证据
    """
    runtime.stream_writer({"stage": "装包上下文"})
    # 权限策略：装包裁剪必须按权限指纹隔离，避免不同权限范围共享裁剪缓存
    policy = runtime.context.get("access_policy")
    # pack_prompt_context：按权限指纹与 token 预算裁剪三组上下文，返回裁剪结果 + 证据
    tables, metrics, training, evidence = pack_prompt_context(
        state.get("query", ""),
        state.get("table_infos", []),
        state.get("metric_infos", []),
        state.get("recall_training", []),
        state.get("semantic_ir", {}),
        policy_scope_fingerprint(policy),
        query_settings=runtime.context["settings"].query,
    )
    logger.info(
        "上下文预算完成 tokens=%s→%s trimmed=%s cache_hit=%s",
        evidence["before_tokens"]["total"],
        evidence["after_tokens"]["total"],
        evidence["trimmed"],
        evidence["cache_hit"],
    )
    # 把上下文预算证据推给前端，供展示装包前后 token 与裁剪原因
    runtime.stream_writer({"retrieval_budget": {"context": evidence}})
    return {
        "table_infos": tables,
        "metric_infos": metrics,
        "recall_training": training,
        "context_budget": evidence,
    }
