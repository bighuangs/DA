import logging

from langgraph.runtime import Runtime

from app.agent_system.agents.nl2sql.context import DataAgentContext, serialized_meta_call
from app.agent_system.agents.nl2sql.state import DataAgentState
from app.agent_system.agents.nl2sql.services.query_signals import (
    build_route_signal_plan,
    query_signal_route_evidence,
)
from app.agent_system.governance.resilience import optional_dependency
from app.agent_system.governance.deadline import route_timeout_seconds
from app.agent_system.governance.retrieval_budget import (
    candidate_cache_get, candidate_cache_key, candidate_cache_set,
    capture_retrieval_cache_generation,
    embed_texts_cached, embedding_runtime_evidence, match_features, policy_scope_fingerprint,
    rerank_and_cut, route_parameters,
)
from app.agent_system.agents.nl2sql.contracts.query_signals import QuerySignalsV1
from app.agent_system.agents.nl2sql.source_context import (
    requires_dw_retrieval,
    skipped_retrieval_evidence,
)
from app.domain.data_sources.context import INTERNAL_TABLE_PREFIXES
from app.domain.query.safety.sql_security import extract_table_names

logger = logging.getLogger(__name__)


def _official_training_sql(value: object) -> bool:
    try:
        return not any(
            str(table).lower().startswith(INTERNAL_TABLE_PREFIXES)
            for table in extract_table_names(str(value or ""))
        )
    except Exception:
        return False


async def recall_training(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """SQL 训练样例召回节点：用精确问题融合 + 动态 K + 安全缓存召回 few-shot 样例。

    整体流程：
        1. 候选缓存命中则直接返回（带 cache_hit 证据）。
        2. 若 training_qdrant_repo 未注入，直接返回空候选并标记 repository_unavailable，
           不阻断主流程（训练样例是可选的 few-shot 增强）。
        3. 本地路径：从 Meta MySQL 拉启用状态的训练样例，做精确/关键词匹配，
           命中即作为本地候选（score=1.0）。
        4. 向量路径：只用 query 自身（不扩词）编码后在 Qdrant 检索，
           按 workspace_id 隔离，取 pool 上限候选。
        5. 融合精排：本地候选 + 向量候选交给 rerank_and_cut 做 RRF 融合与动态 K 裁剪。
        6. 写入候选缓存并对外推送 retrieval_budget / degradation 证据。

    与字段/指标召回的差异：
        - 不做 LLM 关键词扩词（llm_keyword_skipped 恒为 True），只用 query 原文编码，
          因为训练样例匹配的是“问题语义”而非“实体关键词”。
        - 向量检索只用单条 query 向量，而非多关键词并发。
    """
    cache_generation = capture_retrieval_cache_generation()
    runtime.stream_writer({"stage": "召回SQL样例"})
    if not requires_dw_retrieval(state):
        evidence = skipped_retrieval_evidence("training")
        runtime.stream_writer({"retrieval_budget": evidence})
        return {"recall_training": [], "retrieval_metrics": [evidence]}
    query = state["query"]
    context = runtime.context
    settings = context["settings"]
    embedding_client = context["embedding_client"]
    repository = context.get("training_qdrant_repo")
    policy = context.get("access_policy")
    workspace_id = getattr(policy, "workspace_id", "")
    signals = QuerySignalsV1.from_state(state.get("query_signals"))
    route_plan = build_route_signal_plan(
        signals,
        "training",
        route_policy=settings.query_signals.route_policy,
        remote_signal_limit=min(
            settings.query_signals.remote_signal_limit,
            settings.query.retrieval_keyword_limit,
        ),
    )
    keywords = route_plan.rerank_terms
    cache_key = candidate_cache_key(
        "training",
        query,
        keywords,
        workspace_id,
        policy_scope_fingerprint(policy),
        query_signal_profile=signals.provenance.extractor_profile,
        route_policy=route_plan.route_policy,
        ordered_remote_signals=route_plan.remote_terms,
        query_signal_fingerprint=signals.signal_fingerprint,
        cache_namespace=route_plan.cache_namespace,
        generation_token=cache_generation,
        index_version=context["index_versions"].retrieval_version(),
        model_namespace=str(
            (context["llm"].route_manifest().get("sql") or {}).get("deployment_id")
            or "unregistered"
        ),
        query_settings=settings.query,
    )
    cached = candidate_cache_get(cache_key, query_settings=settings.query)
    if cached is not None:
        # 缓存命中：复用 items，证据标记 cache_hit，embedding 统计清零
        evidence = dict(cached["evidence"])
        evidence.update({
            "cache_hit": True,
            "embedding_cache": {"hits": 0, "misses": 0},
            "embedding_runtime": embedding_runtime_evidence(embedding_client),
        })
        runtime.stream_writer({"retrieval_budget": evidence})
        return {"recall_training": cached["items"], "retrieval_metrics": [evidence]}
    # 仓储未注入：返回空候选并标记原因，不阻断主流程
    if repository is None:
        evidence = {"route": "training", "selected_count": 0, "dynamic_k": 0,
                    "cutoff_reason": "repository_unavailable", "cache_hit": False}
        return {"recall_training": [], "retrieval_metrics": [evidence]}

    local_candidates: list[dict] = []
    try:
        # 串行化 Meta 调用，拉取全部训练样例
        for item in await serialized_meta_call(
            context, context["meta_mysql_repo"].list_training_infos
        ):
            # 仅保留启用状态的样例，禁用样例不参与 few-shot
            if not bool(item.enabled):
                continue
            if not _official_training_sql(item.sql):
                continue
            candidate = {
                "id": str(item.id), "question": item.question, "sql": item.sql,
                "score": 1.0, "index_version": "meta-mysql-live",
                "retrieval_sources": ["exact_lexical"],
            }
            features = match_features(query, keywords, candidate, "training")
            # 强匹配或关键词命中才纳入本地候选
            if features["strong_match"] or features["keyword_hits"]:
                local_candidates.append(candidate)
    except Exception as exc:
        # 本地路径失败不阻断主流程，降级为只用向量候选
        logger.warning("训练样例本地匹配失败，继续使用向量路径：%s", exc)

    params = route_parameters("training", settings.query)

    async def retrieve_training():
        """向量召回子流程：只用 query 原文编码后在 Qdrant 检索一条结果。"""
        # Training is normally full-query-only; the route plan remains the
        # authority so K0/K1 routing experiments cannot silently diverge.
        vectors, embedding_stats = await embed_texts_cached(
            context["embedding_client"],
            route_plan.remote_terms,
            mode="query",
            namespace=route_plan.cache_namespace,
            query_settings=settings.query,
            embedding_settings=settings.embedding,
        )
        import asyncio
        results = await asyncio.gather(*(
            repository.search(vector, workspace_id, limit=params["pool"])
            for vector in vectors
        )) if vectors else []
        return results, embedding_stats

    # 向量召回走 optional_dependency：超时降级为 None + 空 embedding 统计
    (search_results, embedding_stats), degradation = await optional_dependency(
        "training_qdrant_recall", retrieve_training, fallback=([], {"hits": 0, "misses": 0}),
        timeout_seconds=route_timeout_seconds(
            state,
            "keyword",
            timeout_seconds=settings.query.hybrid_retrieval_timeout_seconds,
        ),
    )
    vector_candidates = []
    # 汇总向量检索的降级证据；search_result 为空时无降级
    hybrid_degradations = []
    for search_result in search_results:
        hybrid_degradations.extend(search_result.degradations)
        for item in search_result.items:
            row = dict(item)
            if not _official_training_sql(row.get("sql")):
                continue
            vector_candidates.append(row)
    # 融合精排：本地候选 + 向量候选做 RRF 融合与动态 K 裁剪
    selected, evidence = rerank_and_cut(
        "training", query, keywords, [*local_candidates, *vector_candidates],
        min_limit=params["min"], max_limit=params["max"],
        query_settings=settings.query,
    )
    evidence.update({
        "pool_limit": params["pool"], "keyword_count": len(keywords),
        "remote_query_count": len(route_plan.remote_terms), "cache_hit": False,
        "embedding_cache": embedding_stats,
        "embedding_runtime": embedding_runtime_evidence(embedding_client),
        "llm_keyword_skipped": True,
        "embedding_mode": "dense_sparse_query", "fusion_method": "entity_rrf",
        # hybrid_queries 记录向量检索的底层证据；无结果时为空
        "hybrid_queries": [
            dict(search_result.evidence) for search_result in search_results
        ],
        "query_signals": query_signal_route_evidence(
            signals, route_plan, query_encode_mode="query"
        ),
    })
    # 汇总降级证据（向量召回降级 + 各 hybrid 降级）
    degradations = [item for item in (degradation, *hybrid_degradations) if item]
    for item in degradations:
        runtime.stream_writer({"degradation": item})
    runtime.stream_writer({"retrieval_budget": evidence})
    candidate_cache_set(
        cache_key,
        {"items": selected, "evidence": evidence},
        generation_token=cache_generation,
        query_settings=settings.query,
    )
    logger.info(
        "召回SQL样例完成 selected=%s dynamic_k=%s cache=false",
        len(selected),
        evidence["dynamic_k"],
    )
    return {
        "recall_training": selected,
        "degradations": degradations,
        "retrieval_metrics": [evidence],
    }
