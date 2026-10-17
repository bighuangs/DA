import asyncio
import json
import logging
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent_system.agents.nl2sql.context import DataAgentContext, serialized_meta_call
from app.agent_system.agents.nl2sql.llm import get_task_llm
from app.agent_system.agents.nl2sql.state import DataAgentState
from app.agent_system.agents.nl2sql.llm import invoke_optional_llm
from app.agent_system.governance.deadline import route_deadline_at
from app.agent_system.agents.nl2sql.services.query_signals import (
    build_route_signal_plan,
    deterministic_value_route_variants,
    query_signal_route_evidence,
)
from app.agent_system.governance.resilience import optional_dependency
from app.agent_system.governance.retrieval_budget import rerank_and_cut, route_parameters
from app.agent_system.agents.nl2sql.services.value_normalization import normalize_value
from app.agent_system.agents.nl2sql.contracts.retrieval import ValueCandidate
from app.agent_system.agents.nl2sql.contracts.query_signals import QuerySignalsV1
from app.agent_system.agents.nl2sql.prompts.prompt_loader import load_prompt
from app.agent_system.agents.nl2sql.source_context import (
    requires_dw_retrieval,
    skipped_retrieval_evidence,
)

logger = logging.getLogger(__name__)

llm = None
"""
0. 对query进行大模型语义化的分词，并与jiaba分词合并
1. 拿各个分词去做召回
2. 去查询qdrant向量库
3. 得到字段值候选列表：list[ValueCandidate]
4. 返回字段值候选列表
"""


def _extract_json_array(text: str) -> list[str]:
    """
    从 LLM 输出中容错提取首个 JSON 字符串数组。

    核心实现思路：
        用两层正则 + json.loads 容错解析，避免 LLM 输出带 Markdown 围栏
        或附加文本时直接抛 OUTPUT_PARSING_FAILURE 导致整节点崩溃。

    业务逻辑：
        1. 去除 Markdown 代码块围栏（```json ... ``` 或 ``` ... ```）
        2. 正则定位首个 [...] 平衡片段
        3. json.loads 解析，仅保留字符串元素；失败则返回空列表
    """
    if not text:
        return []
    # 先尝试匹配带 ```json 围栏的代码块，剥离围栏只取数组部分
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    # 在（可能已剥离围栏的）文本中定位首个 [...] 片段
    match = re.search(r"\[.*?\]", candidate, re.DOTALL)
    if not match:
        return []
    try:
        arr = json.loads(match.group(0))
        # 只保留非空字符串元素，过滤掉数字/null/嵌套结构等异常项
        if isinstance(arr, list):
            return [str(x) for x in arr if x]
        return []
    except Exception:
        # 解析失败安全降级为空列表，绝不抛异常阻断主流程
        return []


def _safe_value_match_hint(
    query: str, keyword: str, values: list[ValueCandidate],
) -> dict | None:
    """Describe value-match shape without exposing any stored value.

    Only an exact substring copied from the current user question is eligible.
    Multiple distinct values sharing that prefix prove a prefix filter; an
    exact indexed value proves equality.  LLM-expanded terms and the matched
    stored values themselves are never returned in the hint.
    """
    normalized_query = normalize_value(query)
    normalized_term = normalize_value(keyword)
    if (
        len(normalized_term) < 2
        or normalized_term not in normalized_query
    ):
        return None
    normalized_values = {
        normalize_value(str(item.get("value") or ""))
        for item in values
        if normalize_value(str(item.get("value") or ""))
    }
    if normalized_term in normalized_values:
        operator = "exact"
        matched_count = 1
    else:
        prefix_values = {
            value for value in normalized_values if value.startswith(normalized_term)
        }
        if len(prefix_values) < 2:
            return None
        operator = "prefix"
        matched_count = len(prefix_values)
    return {
        "query_term": str(keyword).strip(),
        "operator": operator,
        "matched_distinct_lower_bound": matched_count,
        "stored_values_exposed": False,
    }


def _annotate_value_match_hints(
    query: str, keyword: str, values: list[ValueCandidate],
) -> list[ValueCandidate]:
    """Attach query-owned match hints without transferring evidence across columns."""
    annotated = []
    for item in values:
        row = dict(item)
        # Hints are request-scoped evidence and must never be trusted if a
        # repository payload or reused object happens to contain an older one.
        row.pop("value_match_hints", None)
        annotated.append(row)
    indexes_by_column: dict[str, list[int]] = {}
    for index, item in enumerate(annotated):
        column_id = str(item.get("column_id") or "").strip()
        if column_id:
            indexes_by_column.setdefault(column_id, []).append(index)

    normalized_term = normalize_value(keyword)
    for indexes in indexes_by_column.values():
        column_values = [annotated[index] for index in indexes]
        hint = _safe_value_match_hint(query, keyword, column_values)
        if not hint:
            continue
        operator = str(hint["operator"])
        for index in indexes:
            normalized_value = normalize_value(
                str(annotated[index].get("value") or "")
            )
            matches = (
                normalized_value == normalized_term
                if operator == "exact"
                else normalized_value.startswith(normalized_term)
            )
            if matches:
                annotated[index]["value_match_hints"] = [dict(hint)]
    return annotated


async def recall_value(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """
    基于 query 与关键词从 ES（Elasticsearch）召回相关的字段值信息。

    核心功能：先用大模型对 query 做语义化分词，与 jieba 关键词合并成
             扩展关键词集合；再用每个关键词去 ES 字段值索引做关键词检索
             （而非向量检索，字段值多为离散枚举值，关键词匹配更精准）；
             最后按 value_id 去重合并所有命中结果，得到与提问相关的字段值
             信息列表，用于后续将命中值补进字段的 examples。

    业务逻辑：
        1. 向前端推送当前流水线阶段"召回字段值"
        2. 从 state 取出 query 与 jieba 关键词 keywords，从 runtime 取出
           value_es_repo（ES 字段值仓储）
        3. 调用大模型对 query 语义分词，与 jieba 关键词合并去重得到扩展关键词
           （LLM 分词仅为召回增强，解析失败时降级为只用 jieba 关键词，不阻断主流程）
        4. 从 Meta MySQL 加载行列权限范围内的表/字段白名单，作为 ES 检索的过滤范围
           （fail closed：元数据加载失败时按空范围降级，绝不退回全局检索）
        5. 遍历每个关键词：并发查 ES → 收集命中的字段值信息，按 value_id 去重保留高分
        6. 交给 rerank_and_cut 精排与动态 K 裁剪，返回 Agent 自有字段值候选

    Args:
        state: 流水线状态，包含 query 与 keywords
        runtime: LangGraph 运行时，携带 value_es_repo 等上下文

    Returns:
        dict: {"recall_values": list[ValueCandidate]} —— 去重后的字段值信息列表
    """
    runtime.stream_writer({"stage": "召回字段值"})
    if not requires_dw_retrieval(state):
        evidence = skipped_retrieval_evidence("value")
        runtime.stream_writer({"retrieval_budget": evidence})
        return {"recall_values": [], "retrieval_metrics": [evidence]}
    query = state["query"]
    context = runtime.context
    settings = context["settings"]
    value_es_repo = context["value_es_repo"]
    signals = QuerySignalsV1.from_state(state.get("query_signals"))

    # 0.对query进行大模型语义化的分词，并与jiaba分词合并
    # jieba 是非语义分词，这里用大模型补充语义层面的关键词，提升召回覆盖率
    # LLM 语义分词仅为召回增强：解析失败时降级为只用 jieba 关键词，绝不阻断主流程
    llm_keywords: list[str] = []
    keyword_attempts: list[dict] = []
    keyword_degradations: list[dict] = []
    if not settings.query.value_recall_llm_enabled:
        # 开关关闭时复用已有 jieba 关键词，跳过 LLM 扩词以降低延迟
        logger.info("recall_value 复用已有关键词，跳过 LLM 扩词")
    else:
        prompt_template = PromptTemplate(
            template=load_prompt("extend_keywords_for_value_recall"),
            input_variables=["query"],
        )
        # 用 StrOutputParser + 手动容错解析替代 JsonOutputParser，避免 LLM 输出
        # 带 Markdown 围栏或附加文本时抛 OUTPUT_PARSING_FAILURE 导致整节点崩溃
        chain = (
            prompt_template
            | (llm or get_task_llm(runtime, "keyword"))
            | StrOutputParser()
        )
        invocation = await invoke_optional_llm(
            chain,
            {"query": query},
            node_id="recall_value",
            route="keyword",
            fallback="",
            timeout_seconds=settings.query.optional_dependency_timeout_seconds,
            core_path=False,
            runtime=runtime,
            request_deadline_at=route_deadline_at(
                state,
                "keyword",
                timeout_seconds=settings.query.optional_dependency_timeout_seconds,
            ),
            recovery_enabled=False,
        )
        keyword_attempts = invocation.attempts
        keyword_degradations = invocation.degradations
        if not keyword_degradations:
            raw = invocation.value
            llm_keywords = _extract_json_array(raw)
            # 扩词可能包含客户名、手机号等 literal；日志只记录数量。
            logger.info("recall_value LLM 扩词完成 count=%s", len(llm_keywords))
        else:
            logger.warning("recall_value LLM 语义分词失败，降级为确定性关键词")

    # 合并 jieba 关键词与大模型语义关键词，并用 set 去重；
    # prioritized_value_keywords 把实体字面量排到前面，提升 ES 精确匹配命中率
    base_terms = signals.retrieval_terms()
    deterministic_variants = list(
        deterministic_value_route_variants(query, base_terms)
    )
    route_plan = build_route_signal_plan(
        signals,
        "value",
        route_policy=settings.query_signals.route_policy,
        remote_signal_limit=min(
            settings.query_signals.remote_signal_limit,
            settings.query.retrieval_keyword_limit,
        ),
        priority_terms=deterministic_variants,
        llm_terms=llm_keywords,
    )
    keywords = route_plan.rerank_terms
    params = route_parameters("value", settings.query)

    # 字段值检索必须在 ES 查询阶段同时带入 workspace、版本和行列权限范围。
    # 元数据查询失败时保持空范围（fail closed），不能退回无过滤的全局检索。
    table_rows = []
    try:
        # 串行化 Meta 调用，拉取全量表与字段信息用于构建权限白名单
        table_rows = await serialized_meta_call(
            context, context["meta_mysql_repo"].list_table_infos
        )
    except Exception as exc:
        # fail closed：加载失败按空范围降级，宁可召回为空也不越权
        logger.warning("字段值授权范围加载失败，按空范围降级：%s", exc)
    policy = context.get("access_policy")
    allowed_table_ids: list[str] = []
    allowed_column_ids: list[str] = []
    # 遍历全量表/字段，按行列权限构建 ES 检索的白名单
    for table, columns in table_rows:
        table_id = str(getattr(table, "id", "") or getattr(table, "name", ""))
        # 表级权限：未被授权的表整体跳过
        if policy and not policy.is_table_allowed(table_id):
            continue
        allowed_table_ids.append(table_id)
        for column in columns:
            column_name = str(getattr(column, "name", "") or "")
            # 列级权限：未被授权的字段跳过
            if policy and not policy.is_column_allowed(table_id, column_name):
                continue
            # 字段标识优先用 column.id，缺失时回退到 table_id.column_name 复合键
            allowed_column_ids.append(
                str(getattr(column, "id", "") or f"{table_id}.{column_name}")
            )
    # workspace_id 只允许来自当前请求、访问策略或已绑定 Meta 仓储。
    workspace_id = str(
        context.get("workspace_id")
        or getattr(policy, "workspace_id", "")
        or getattr(context["meta_mysql_repo"], "workspace_id", "")
    ).strip()
    if not workspace_id:
        raise ValueError("字段值召回必须绑定 workspace_id")

    # 用来保存所有召回的字段信息对象，需要去重：key: value_id
    value_infos_dict : dict[str, ValueCandidate] = {}
    # 1.# 拿各个分词去做召回
    # 2. ES 关键词检索彼此独立，并发执行，避免慢关键词累加。
    async def retrieve_values():
        """ES 字段值召回子流程：先取 active manifest，再并发检索所有关键词。"""
        # active_manifest：当前 workspace 生效的索引版本与就绪状态，用于版本对齐
        manifest = await value_es_repo.active_manifest(workspace_id)
        # 每个关键词独立并发检索 ES，带入 workspace/版本/表字段白名单
        searches = await asyncio.gather(*(
            value_es_repo.search_with_evidence(
                keyword,
                workspace_id=workspace_id,
                query_mode="full_text",
                index_version=manifest.get("index_version"),
                active_manifest=manifest,
                allowed_table_ids=allowed_table_ids,
                allowed_column_ids=allowed_column_ids,
                page_size=params["pool"],
            )
            for keyword in route_plan.remote_terms
        )) if keywords else []
        # 拆分返回：batches=命中的字段值列表，audits=每次检索的审计证据
        annotated_batches = []
        for keyword, (items, _, _) in zip(route_plan.remote_terms, searches):
            annotated_batches.append(
                _annotate_value_match_hints(query, keyword, items)
            )
        return {
            "manifest": manifest,
            "batches": annotated_batches,
            "audits": [audit for _, _, audit in searches],
        }

    # ES 召回走 optional_dependency：超时降级为空 batches/audits
    retrieval_result, recall_degradation = await optional_dependency(
        "value_es_recall",
        retrieve_values,
        fallback={"manifest": {}, "batches": [], "audits": []},
        timeout_seconds=settings.query.optional_dependency_timeout_seconds,
    )
    batches = retrieval_result["batches"]
    value_audits = retrieval_result["audits"]
    manifest = retrieval_result["manifest"]
    for value_infos in batches:
        # 对查询得到的列表数据进行去重保存：相同 value_id 只保留首次命中
        # 若同一 value_id 被多个关键词命中，保留得分更高的那条
        for value_info in value_infos:
            value_id = value_info["id"]
            previous = value_infos_dict.get(value_id)
            merged_hints = {
                (
                    str(hint.get("query_term") or ""),
                    str(hint.get("operator") or ""),
                ): hint
                for source in (previous, value_info)
                if source
                for hint in (source.get("value_match_hints") or [])
                if isinstance(hint, dict)
            }
            if (
                previous is None
                or float(value_info.get("score") or 0)
                > float(previous.get("score") or 0)
            ):
                value_infos_dict[value_id] = dict(value_info)
            if merged_hints:
                value_infos_dict[value_id]["value_match_hints"] = list(
                    merged_hints.values()
                )

    candidates = []
    for item in value_infos_dict.values():
        row = dict(item)
        # 未标记 retrieval_sources 的候选补一个来源标记，便于 rerank 证据溯源
        if not row.get("retrieval_sources"):
            row["_retrieval_source"] = "lexical_es"
        candidates.append(row)
    # 精排与动态 K 裁剪：字段值不做向量融合，仅按 ES 词法得分排序裁剪
    recall_values, evidence = rerank_and_cut(
        "value", query, keywords, candidates,
        min_limit=params["min"], max_limit=params["max"],
        query_settings=settings.query,
    )
    evidence.update({
        "pool_limit": params["pool"], "keyword_count": len(keywords),
        "remote_query_count": len(route_plan.remote_terms),
        "llm_keyword_skipped": not settings.query.value_recall_llm_enabled,
        # 字段值含敏感实体（客户名/手机号等），不参与候选缓存，避免敏感数据驻留
        "cache_hit": False, "embedding_cache": {"hits": 0, "misses": 0},
        "cache_disabled_reason": "sensitive_entity_values",
        "responsibility": "es_field_value",
        "engine": "elasticsearch",
        # 索引版本与就绪状态来自 active manifest，用于可观测性
        "index_version": manifest.get("index_version"),
        "index_ready": bool(manifest.get("ready")),
        "workspace_scope_applied": True,
        "allowed_table_count": len(allowed_table_ids),
        "allowed_column_count": len(allowed_column_ids),
        # 汇总所有关键词检索的 query_mode（去重排序），便于审计检索路径
        "query_modes": sorted({
            str(item.get("query_mode") or "") for item in value_audits
            if item.get("query_mode")
        }),
        # 只保留前 8 条审计，避免证据过大
        "value_search_audits": value_audits[:8],
        # 字段值只走 ES 词法检索，不推断稀疏字面量
        "literal_source": "elasticsearch_only",
        "sparse_literal_inference": False,
        "query_signals": query_signal_route_evidence(
            signals, route_plan, query_encode_mode="not_applicable_es_lexical"
        ),
    })

    logger.info(f"召回字段值完成 count={len(recall_values)}")
    # 3.# 返回字段信息列表： list[valueInfoQdrant]
    degradations = [
        *keyword_degradations,
        *([recall_degradation] if recall_degradation else []),
    ]
    if recall_degradation:
        runtime.stream_writer({"degradation": recall_degradation})
    runtime.stream_writer({"retrieval_budget": evidence})
    return {
        "recall_values": recall_values,
        "degradations": degradations,
        "retrieval_metrics": [evidence],
        "llm_attempts": keyword_attempts,
    }
