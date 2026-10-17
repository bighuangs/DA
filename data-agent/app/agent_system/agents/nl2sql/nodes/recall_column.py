import logging

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent_system.agents.nl2sql.context import DataAgentContext, serialized_meta_call
from app.agent_system.agents.nl2sql.llm import get_task_llm
from app.agent_system.agents.nl2sql.state import DataAgentState
from app.agent_system.agents.nl2sql.llm import invoke_optional_llm
from app.agent_system.governance.deadline import route_deadline_at, route_timeout_seconds
from app.agent_system.agents.nl2sql.services.query_signals import (
    build_route_signal_plan,
    query_signal_route_evidence,
)
from app.agent_system.governance.resilience import optional_dependency
from app.agent_system.governance.retrieval_budget import (
    candidate_cache_get,
    candidate_cache_key,
    candidate_cache_set,
    capture_retrieval_cache_generation,
    embedding_runtime_evidence,
    embed_texts_cached,
    has_strong_match,
    match_features,
    policy_scope_fingerprint,
    rerank_and_cut,
    route_parameters,
)
from app.agent_system.agents.nl2sql.contracts.retrieval import ColumnCandidate
from app.agent_system.agents.nl2sql.contracts.query_signals import QuerySignalsV1
from app.agent_system.agents.nl2sql.prompts.prompt_loader import load_prompt
from app.agent_system.agents.nl2sql.source_context import (
    requires_dw_retrieval,
    skipped_retrieval_evidence,
)
from app.domain.data_sources.context import INTERNAL_TABLE_PREFIXES

logger = logging.getLogger(__name__)

# 关键词扩词用的任务 LLM（共享 keyword 任务路由），用于在向量召回前补全语义关键词
llm = None


def _metadata_candidate(column) -> ColumnCandidate:
    """把 Meta MySQL 中的字段行封装为 score=1.0 的本地候选对象。

    本地候选不走向量检索，而是直接以“精确词法/别名命中”来源参与后续 rerank，
    因此 retrieval_sources 标记为 exact_lexical，index_version 标记为 meta-mysql-live，
    便于在 rerank 证据里区分向量候选与本地候选。
    """
    return ColumnCandidate(
        id=column.id, name=column.name, type=column.type, role=column.role,
        examples=list(column.examples or []), description=column.description or "",
        alias=list(column.alias or []), table_id=column.table_id,
        is_display=bool(column.is_display), score=1.0,
        index_version="meta-mysql-live", retrieval_sources=["exact_lexical"],
    )


async def recall_column(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """字段召回节点：先召回宽候选池，再做可解释的精排与动态裁剪。

    整体流程：
        1. 候选缓存命中则直接返回（带 cache_hit 证据），避免重复向量检索。
        2. 本地路径：从 Meta MySQL 拉全量字段，按行列权限过滤后做精确/别名词法匹配，
           命中即作为本地候选（score=1.0），与向量候选融合。
        3. 关键词扩词：若本地已有强匹配或开关关闭则跳过 LLM 扩词，否则用 LLM 扩词。
        4. 向量路径：对每个关键词编码后在 Qdrant 检索，按 index_version 过滤，
           并按行列权限过滤向量候选。
        5. 融合精排：把本地候选 + 向量候选交给 rerank_and_cut 做 RRF 融合、
           动态 K 与裁剪，产出最终 recall_columns 与结构化证据。
        6. 写入候选缓存并对外推送 retrieval_budget / degradation 证据。
    """
    cache_generation = capture_retrieval_cache_generation()
    runtime.stream_writer({"stage": "召回字段"})
    if not requires_dw_retrieval(state):
        evidence = skipped_retrieval_evidence("column")
        runtime.stream_writer({"retrieval_budget": evidence})
        return {"recall_columns": [], "retrieval_metrics": [evidence]}
    query = state["query"]
    context = runtime.context
    settings = context["settings"]
    embedding_client = context["embedding_client"]
    repository = context["column_qdrant_repo"]
    meta_repo = context["meta_mysql_repo"]
    policy = context.get("access_policy")
    workspace_id = getattr(policy, "workspace_id", "")
    signals = QuerySignalsV1.from_state(state.get("query_signals"))
    remote_limit = min(
        settings.query_signals.remote_signal_limit,
        settings.query.retrieval_keyword_limit,
    )
    route_plan = build_route_signal_plan(
        signals,
        "column",
        route_policy=settings.query_signals.route_policy,
        remote_signal_limit=remote_limit,
    )
    local_terms = route_plan.local_terms
    # 行列权限指纹：缓存键的一部分，保证不同权限范围不会命中彼此的缓存
    scope = policy_scope_fingerprint(policy)
    local_candidates: list[dict] = []
    live_columns = []
    try:
        # serialized_meta_call：Meta 仓储调用串行化，避免并发压垮元数据库连接
        for _, columns in await serialized_meta_call(context, meta_repo.list_table_infos):
            live_columns.extend(columns)
            for column in columns:
                # 行列权限过滤：未被授权的字段不进入本地候选
                if policy and not policy.is_column_allowed(column.table_id, column.name):
                    continue
                candidate = _metadata_candidate(column)
                # match_features：计算精确名/别名命中与关键词命中特征
                features = match_features(query, local_terms, candidate, "column")
                # 强匹配或关键词命中才纳入本地候选，避免无谓的本地候选膨胀
                if features["strong_match"] or features["keyword_hits"]:
                    local_candidates.append(candidate)
    except Exception as exc:
        # 本地路径失败不阻断主流程，降级为只用向量候选
        logger.warning("字段本地 exact/alias 召回失败，继续使用向量路径：%s", exc)

    # 与构建阶段使用相同的全量字段版本；Qdrant 只返回该版本的 Point。
    # 元数据读取失败时，空内容版本不会匹配已有索引，宁可降级为空候选也不混入旧版本。
    index_versions = context["index_versions"]
    index_version = index_versions.column_version(live_columns)

    extended: list[str] = []
    keyword_degradation = None
    keyword_attempts: list[dict] = []
    keyword_llm_degradations: list[dict] = []
    # LLM 扩词跳过条件：开关关闭，或本地已有强匹配（强匹配时扩词收益有限且增加延迟）
    llm_skipped = (
        not settings.query.keyword_recall_llm_enabled
        or has_strong_match(query, local_terms, local_candidates, "column")
    )
    if not llm_skipped:
        # LLM 扩词链：Prompt -> LLM -> JSON 解析，输出扩展关键词列表
        chain = PromptTemplate(
            template=load_prompt("extend_keywords_for_column_recall"), input_variables=["query"]
        ) | (llm or get_task_llm(runtime, "keyword")) | JsonOutputParser()
        # optional_dependency：超时/失败时返回空列表降级，并产出结构化降级证据
        invocation = await invoke_optional_llm(
            chain,
            {"query": query},
            node_id="recall_column",
            route="keyword",
            fallback=[],
            timeout_seconds=settings.query.optional_dependency_timeout_seconds,
            core_path=False,
            allow_empty=True,
            runtime=runtime,
            request_deadline_at=route_deadline_at(
                state,
                "keyword",
                timeout_seconds=settings.query.optional_dependency_timeout_seconds,
            ),
            recovery_enabled=False,
        )
        extended = list(invocation.value or [])
        keyword_attempts = invocation.attempts
        keyword_llm_degradations = invocation.degradations
    route_plan = build_route_signal_plan(
        signals,
        "column",
        route_policy=settings.query_signals.route_policy,
        remote_signal_limit=remote_limit,
        llm_terms=list(extended or []),
    )
    keywords = route_plan.rerank_terms
    cache_key = candidate_cache_key(
        "column",
        query,
        keywords,
        workspace_id,
        scope,
        query_signal_profile=signals.provenance.extractor_profile,
        route_policy=route_plan.route_policy,
        ordered_remote_signals=route_plan.remote_terms,
        query_signal_fingerprint=signals.signal_fingerprint,
        cache_namespace=route_plan.cache_namespace,
        expansion_profile="keyword_llm_v1" if extended else "none",
        generation_token=cache_generation,
        index_version=index_versions.retrieval_version(),
        model_namespace=str(
            (context["llm"].route_manifest().get("sql") or {}).get("deployment_id")
            or "unregistered"
        ),
        query_settings=settings.query,
    )
    cached = candidate_cache_get(cache_key, query_settings=settings.query)
    if cached is not None:
        evidence = dict(cached["evidence"])
        evidence.update({
            "cache_hit": True,
            "embedding_cache": {"hits": 0, "misses": 0},
            "embedding_runtime": embedding_runtime_evidence(embedding_client),
        })
        runtime.stream_writer({"retrieval_budget": evidence})
        return {
            "recall_columns": cached["items"],
            "retrieval_metrics": [evidence],
            "llm_attempts": keyword_attempts,
            **({"degradations": keyword_llm_degradations} if keyword_llm_degradations else {}),
        }
    # 路由参数：pool/min/max/threshold 等都来自配置，按 route 取值
    params = route_parameters("column", settings.query)

    async def retrieve_columns():
        """向量召回子流程：对每个关键词编码后在 Qdrant 并发检索。"""
        # embed_texts_cached：embedding 带缓存，避免同关键词重复编码
        vectors, embedding_stats = await embed_texts_cached(
            embedding_client,
            route_plan.remote_terms,
            mode="query",
            namespace=route_plan.cache_namespace,
            query_settings=settings.query,
            embedding_settings=settings.embedding,
        )
        searches = []
        for vector in vectors:
            # 按 index_version 过滤，保证只取当前索引版本的 Point
            searches.append(repository.search(
                vector, limit=params["pool"], index_version=index_version
            ))
        import asyncio
        # 并发执行所有关键词的向量检索；空批次直接返回空
        return (await asyncio.gather(*searches) if searches else []), embedding_stats

    # 向量召回也走 optional_dependency：超时降级为空候选 + 空 embedding 统计
    (search_results, embedding_stats), recall_degradation = await optional_dependency(
        "column_qdrant_recall", retrieve_columns, fallback=([], {"hits": 0, "misses": 0}),
        timeout_seconds=route_timeout_seconds(
            state,
            "keyword",
            timeout_seconds=settings.query.hybrid_retrieval_timeout_seconds,
        ),
    )
    vector_candidates: list[dict] = []
    hybrid_evidence: list[dict] = []
    hybrid_degradations: list[dict] = []
    # 汇总每个关键词的向量检索结果：证据、降级、候选（带行列权限过滤）
    for result in search_results:
        hybrid_evidence.append(dict(result.evidence))
        hybrid_degradations.extend(result.degradations)
        for item in result.items:
            if str(item.get("table_id") or "").lower().startswith(
                INTERNAL_TABLE_PREFIXES
            ):
                continue
            # 向量候选同样要做行列权限过滤，防止越权字段进入下游
            if policy and not policy.is_column_allowed(str(item.get("table_id") or ""), str(item.get("name") or "")):
                continue
            row = dict(item)
            vector_candidates.append(row)
    # 融合精排：本地候选 + 向量候选一起做 RRF 融合、动态 K 与裁剪
    selected, evidence = rerank_and_cut(
        "column", query, keywords, [*local_candidates, *vector_candidates],
        min_limit=params["min"], max_limit=params["max"],
        query_settings=settings.query,
    )
    # 在精排证据上叠加本节点的路由/缓存/embedding/融合元信息，供可观测性使用
    evidence.update({
        "pool_limit": params["pool"], "keyword_count": len(keywords),
        "llm_keyword_skipped": llm_skipped, "cache_hit": False,
        "embedding_cache": embedding_stats,
        "embedding_runtime": embedding_runtime_evidence(embedding_client),
        "embedding_mode": "dense_sparse_query", "fusion_method": "entity_rrf",
        "hybrid_queries": hybrid_evidence,
        "query_signals": query_signal_route_evidence(
            signals, route_plan, query_encode_mode="query"
        ),
    })
    # 汇总所有降级证据（关键词 LLM、向量召回、各关键词的 hybrid 降级）
    degradations = [
        item for item in (
            keyword_degradation, recall_degradation,
            *keyword_llm_degradations, *hybrid_degradations
        ) if item
    ]
    for item in degradations:
        runtime.stream_writer({"degradation": item})
    runtime.stream_writer({"retrieval_budget": evidence})
    # 写入候选缓存，供后续相同 query+权限范围复用
    candidate_cache_set(
        cache_key,
        {"items": selected, "evidence": evidence},
        generation_token=cache_generation,
        query_settings=settings.query,
    )
    logger.info("召回字段完成 selected=%s raw=%s dynamic_k=%s reason=%s cache=false",
                len(selected), evidence["raw_count"], evidence["dynamic_k"], evidence["cutoff_reason"])
    return {
        "recall_columns": selected,
        "degradations": degradations,
        "retrieval_metrics": [evidence],
        "llm_attempts": keyword_attempts,
    }
