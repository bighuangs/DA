import asyncio
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
    candidate_cache_get, candidate_cache_key, candidate_cache_set,
    capture_retrieval_cache_generation,
    embed_texts_cached, embedding_runtime_evidence, has_strong_match, match_features,
    policy_scope_fingerprint, rerank_and_cut, route_parameters,
)
from app.agent_system.agents.nl2sql.contracts.retrieval import MetricCandidate
from app.agent_system.agents.nl2sql.contracts.query_signals import QuerySignalsV1
from app.agent_system.agents.nl2sql.prompts.prompt_loader import load_prompt
from app.agent_system.agents.nl2sql.source_context import (
    requires_dw_retrieval,
    skipped_retrieval_evidence,
)
from app.domain.data_sources.context import INTERNAL_TABLE_PREFIXES

logger = logging.getLogger(__name__)

# 关键词扩词用的任务 LLM，与字段召回共享 keyword 路由
llm = None


async def recall_metric(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """指标召回节点：融合 Meta MySQL 精确/别名命中与向量候选，并做动态 K 裁剪。

    整体流程与 recall_column 同构，差异点在于：
        - 指标候选需补全 formula_rule（公式规则）与 relevant_columns（相关字段），
          其中种子指标（id==name）从 meta_config 中继承口径。
        - 行列权限按“指标引用的全部字段”判定，任一字段无权则整条指标被过滤。
        - 向量检索按 workspace_id 隔离，不传 index_version（指标索引无版本过滤）。
    """
    cache_generation = capture_retrieval_cache_generation()
    runtime.stream_writer({"stage": "召回指标"})
    if not requires_dw_retrieval(state):
        evidence = skipped_retrieval_evidence("metric")
        runtime.stream_writer({"retrieval_budget": evidence})
        return {"recall_metrics": [], "retrieval_metrics": [evidence]}
    query = state["query"]
    context = runtime.context
    settings = context["settings"]
    embedding_client = context["embedding_client"]
    repository = context["metric_qdrant_repo"]
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
        "metric",
        route_policy=settings.query_signals.route_policy,
        remote_signal_limit=remote_limit,
    )
    local_terms = route_plan.local_terms
    # 权限指纹进入缓存键，避免不同权限范围命中彼此缓存
    scope = policy_scope_fingerprint(policy)
    local_candidates: list[dict] = []
    try:
        # 种子指标表：meta_config 中预置的指标口径，用于为同名指标补全 formula_rule/relevant_columns
        seed_metrics = {item.name: item for item in settings.seed_metrics}
        # 串行化 Meta 调用，避免并发压垮元数据库
        for metric in await serialized_meta_call(context, meta_repo.get_all_metric_infos):
            # 种子继承：仅当指标 id==name（即未自定义 id）时才从 meta_config 继承口径
            seed = seed_metrics.get(metric.name) if metric.id == metric.name else None
            # 公式规则优先用指标自身，其次用种子；保证派生指标也能拿到口径
            formula_rule = metric.formula_rule or (seed.formula_rule if seed else None)
            relevant = list(metric.relevant_columns or [])
            # 相关字段为空时回退到种子的相关字段，避免指标无字段引用导致下游无法生成 SQL
            if not relevant and seed and formula_rule:
                relevant = list(seed.relevant_columns or [])
            candidate = MetricCandidate(
                workspace_id=workspace_id, id=metric.id, name=metric.name,
                description=metric.description or "", relevant_columns=relevant,
                alias=list(metric.alias or []), formula_rule=formula_rule, score=1.0,
                index_version="meta-mysql-live", retrieval_sources=["exact_alias_lexical"],
            )
            # 行列权限：指标引用的全部字段都必须有权限，否则整条指标被过滤
            if policy and not all(
                policy.is_column_allowed(*policy._split_column_id(column_id)) for column_id in relevant
            ):
                continue
            features = match_features(query, local_terms, candidate, "metric")
            # 强匹配或关键词命中才纳入本地候选
            if features["strong_match"] or features["keyword_hits"]:
                local_candidates.append(candidate)
    except Exception as exc:
        # 本地路径失败不阻断主流程，降级为只用向量候选
        logger.warning("指标本地 exact/alias 召回失败，继续使用向量路径：%s", exc)

    extended: list[str] = []
    keyword_degradation = None
    keyword_attempts: list[dict] = []
    keyword_llm_degradations: list[dict] = []
    # LLM 扩词跳过：开关关闭或本地已有强匹配
    llm_skipped = (
        not settings.query.keyword_recall_llm_enabled
        or has_strong_match(query, local_terms, local_candidates, "metric")
    )
    if not llm_skipped:
        chain = PromptTemplate(
            template=load_prompt("extend_keywords_for_metric_recall"), input_variables=["query"]
        ) | (llm or get_task_llm(runtime, "keyword")) | JsonOutputParser()
        invocation = await invoke_optional_llm(
            chain,
            {"query": query},
            node_id="recall_metric",
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
        "metric",
        route_policy=settings.query_signals.route_policy,
        remote_signal_limit=remote_limit,
        llm_terms=list(extended or []),
    )
    keywords = route_plan.rerank_terms
    cache_key = candidate_cache_key(
        "metric",
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
        index_version=context["index_versions"].retrieval_version(),
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
            "recall_metrics": cached["items"],
            "retrieval_metrics": [evidence],
            "llm_attempts": keyword_attempts,
            **({"degradations": keyword_llm_degradations} if keyword_llm_degradations else {}),
        }
    params = route_parameters("metric", settings.query)

    async def retrieve_metrics():
        """向量召回子流程：对每个关键词编码后在 Qdrant 并发检索（按 workspace 隔离）。"""
        vectors, embedding_stats = await embed_texts_cached(
            embedding_client,
            route_plan.remote_terms,
            mode="query",
            namespace=route_plan.cache_namespace,
            query_settings=settings.query,
            embedding_settings=settings.embedding,
        )
        # 指标检索按 workspace_id 隔离，不传 index_version（指标索引无版本过滤）
        results = await asyncio.gather(*(
            repository.search(vector, workspace_id, limit=params["pool"]) for vector in vectors
        )) if vectors else []
        return results, embedding_stats

    # 向量召回走 optional_dependency：超时降级为空候选
    (search_results, embedding_stats), recall_degradation = await optional_dependency(
        "metric_qdrant_recall", retrieve_metrics, fallback=([], {"hits": 0, "misses": 0}),
        timeout_seconds=route_timeout_seconds(
            state,
            "keyword",
            timeout_seconds=settings.query.hybrid_retrieval_timeout_seconds,
        ),
    )
    vector_candidates: list[dict] = []
    hybrid_evidence: list[dict] = []
    hybrid_degradations: list[dict] = []
    for result in search_results:
        hybrid_evidence.append(dict(result.evidence))
        hybrid_degradations.extend(result.degradations)
        for item in result.items:
            # 向量候选同样按“指标引用的全部字段”做行列权限过滤
            relevant = list(item.get("relevant_columns") or [])
            if any(
                str(value).lower().startswith(INTERNAL_TABLE_PREFIXES)
                for value in relevant
            ):
                continue
            if policy and not all(policy.is_column_allowed(*policy._split_column_id(value)) for value in relevant):
                continue
            row = dict(item)
            vector_candidates.append(row)
    # 融合精排：本地候选 + 向量候选做 RRF 融合、动态 K 与裁剪
    selected, evidence = rerank_and_cut(
        "metric", query, keywords, [*local_candidates, *vector_candidates],
        min_limit=params["min"], max_limit=params["max"],
        query_settings=settings.query,
    )
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
    # 汇总降级证据
    degradations = [
        item for item in (
            keyword_degradation, recall_degradation,
            *keyword_llm_degradations, *hybrid_degradations
        ) if item
    ]
    for item in degradations:
        runtime.stream_writer({"degradation": item})
    runtime.stream_writer({"retrieval_budget": evidence})
    candidate_cache_set(
        cache_key,
        {"items": selected, "evidence": evidence},
        generation_token=cache_generation,
        query_settings=settings.query,
    )
    logger.info("召回指标完成 selected=%s raw=%s dynamic_k=%s reason=%s cache=false",
                len(selected), evidence["raw_count"], evidence["dynamic_k"], evidence["cutoff_reason"])
    return {
        "recall_metrics": selected,
        "degradations": degradations,
        "retrieval_metrics": [evidence],
        "llm_attempts": keyword_attempts,
    }
