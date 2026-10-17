import json
import logging
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent_system.agents.nl2sql.context import DataAgentContext
from app.agent_system.agents.nl2sql.llm import get_task_llm, task_timeout
from app.agent_system.agents.nl2sql.state import DataAgentState
from app.agent_system.agents.nl2sql.llm import invoke_optional_llm
from app.agent_system.governance.deadline import route_deadline_at
from app.agent_system.runtime.performance import postprocess_budget_exhausted
from app.agent_system.agents.nl2sql.services.prompt_context import build_model_prompt_payload
from app.agent_system.agents.nl2sql.services.recommendation_options import (
    build_contextual_recommendations,
    build_recommendation_context,
    normalize_model_recommendations,
    options_from_queries,
    recommendation_event,
)
from app.agent_system.agents.nl2sql.prompts.prompt_loader import load_prompt
from app.agent_system.governance.retrieval_budget import estimate_tokens

logger = logging.getLogger(__name__)

llm = None
# 推荐问题数量上限：兼顾页面布局、Prompt 成本与用户可选择性。
RECOMMEND_NUMBER = 4


def _bounded_recommendations(
    query: str, candidates: list[str], articles_number: int = RECOMMEND_NUMBER
) -> list[str]:
    normalized_query = re.sub(r"\s+", "", str(query or "")).lower()
    result = []
    for candidate in candidates:
        text = str(candidate or "").strip()
        if not text or re.sub(r"\s+", "", text).lower() == normalized_query:
            continue
        if text not in result:
            result.append(text)
        if len(result) >= max(0, int(articles_number)):
            break
    return result


def deterministic_recommendations(query: str, articles_number: int = RECOMMEND_NUMBER) -> list[str]:
    """不依赖 LLM，按 query 关键词命中返回有界、可执行的推荐问题列表。

    核心实现思路：按销售额/销量/客户/订单四类主题预置候选问题，命中哪类就返回
    对应候选；去重掉与当前 query 完全相同的候选，并截断到 articles_number 条。
    保证 LLM 不可用或预算超限时仍有推荐 chips 引导用户连续问数。

    业务逻辑：
        1. 按 query 关键词命中主题（销售额/销量/客户/订单），选对应候选列表
        2. 未命中任一主题时用通用候选
        3. 去掉与当前 query 完全相同的候选（空白归一化后比较）
        4. 去重并截断到 articles_number 条返回
    """
    text = str(query or "").strip()
    # 按主题命中选候选：销售额类
    if any(token in text.lower() for token in ("销售额", "销售总额", "gmv", "成交额", "消费金额")):
        candidates = ["各地区的销售额", "各品类的销售额", "2025年每月销售额趋势", "销售额最高的客户前十名"]
    # 销量类
    elif any(token in text for token in ("销量", "件数", "购买数量", "售出")):
        candidates = ["各品类的销量总和", "销量前十的商品", "各品牌共售出多少件商品", "2025年每月销量趋势"]
    # 客户/复购类
    elif any(token in text for token in ("客户", "用户", "复购")):
        candidates = ["有过下单行为的客户有多少", "客户按消费金额排名前十", "复购率是多少", "不同地区的客户数"]
    # 订单/客单价类
    elif any(token in text for token in ("订单", "客单价", "aov", "AOV")):
        candidates = ["每个地区的订单总数", "平均订单金额 AOV 是多少", "2025年每月订单数", "订单量最多的地区"]
    # 通用兜底
    else:
        candidates = ["各地区的销售总额", "各品类的销量总和", "2025年每月销售额趋势", "订单量最多的地区"]
    # 空白归一化后比较，去掉与当前 query 完全相同的候选
    return _bounded_recommendations(text, candidates, articles_number)


def upload_schema_recommendations(
    state: DataAgentState, articles_number: int = RECOMMEND_NUMBER
) -> list[str]:
    """仅使用当前授权上传 Schema 生成可执行追问，禁止回落到 DW 固定业务词。"""
    source_context = state.get("data_source_context") or {}
    if str(source_context.get("source_mode") or "dw_only") not in {
        "upload_only", "multi_upload"
    }:
        return []
    profiles = list(state.get("upload_profiles") or source_context.get("profiles") or [])
    if not profiles:
        return []

    columns = [
        column
        for profile in profiles
        for column in (profile.get("columns") or [])
        if isinstance(column, dict)
    ]
    dimensions = [
        str(column.get("display_name") or column.get("name") or "").strip()
        for column in columns
        if column.get("role") == "dimension"
        and column.get("type_family") not in {"time"}
    ]
    measures = [
        str(column.get("display_name") or column.get("name") or "").strip()
        for column in columns
        if column.get("role") == "measure"
        or column.get("type_family") == "number"
    ]
    times = [
        str(column.get("display_name") or column.get("name") or "").strip()
        for column in columns
        if column.get("type_family") == "time"
    ]
    dimensions = list(dict.fromkeys(value for value in dimensions if value))
    descriptive_dimensions = [
        value for value in dimensions
        if not re.search(r"(^|_)(id|code|编号|代码)$", value, re.IGNORECASE)
    ]
    identifier_dimensions = [
        value for value in dimensions if value not in descriptive_dimensions
    ]
    dimensions = descriptive_dimensions + identifier_dimensions
    measures = list(dict.fromkeys(value for value in measures if value))
    times = list(dict.fromkeys(value for value in times if value))

    candidates = []
    if dimensions and measures:
        dimension, measure = dimensions[0], measures[0]
        candidates.extend([
            f"按{dimension}汇总{measure}",
            f"查看{measure}最高的{dimension}",
            f"用柱状图展示各{dimension}的{measure}",
        ])
    if times and measures:
        candidates.append(f"按{times[0]}查看{measures[0]}趋势")
    if dimensions:
        candidates.append(f"统计各{dimensions[0]}的记录数")
    candidates.extend(["统计上传文件总行数", "查看上传文件的前20行"])
    return _bounded_recommendations(
        state.get("query", ""), candidates, articles_number
    )


def _extract_json_array(text: str) -> list[object]:
    """
    从 LLM 输出中容错提取首个 JSON 字符串数组。

    业务逻辑：
        1. 去除 Markdown 代码块围栏（```json ... ``` 或 ``` ... ```）
        2. 正则定位首个 [...] 平衡片段
        3. json.loads 解析，仅保留字符串元素；失败则返回空列表
    """
    if not text:
        return []
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    match = re.search(r"\[.*?\]", candidate, re.DOTALL)
    if not match:
        return []
    try:
        arr = json.loads(match.group(0))
        if isinstance(arr, list):
            return [item for item in arr if isinstance(item, (str, dict)) and item]
        return []
    except Exception:
        return []


async def generate_recommend(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    settings = runtime.context["settings"]
    """生成与本轮 SQL/结果范围闭合的结构化后续分析选项。

    确定性路径负责生成可独立执行且类型多样的候选；可选模型只能在受限上下文
    中补充或重排，非法、重复或丢失范围的候选会被丢弃。SSE 同时保留旧标签
    数组，供滚动发布中的旧客户端继续工作。
    """
    runtime.stream_writer({"stage": "猜你想问"})
    try:
        if postprocess_budget_exhausted(
            state,
            budget_ms=settings.query.postprocess_budget_ms,
        ):
            degradation = {"node_id": "generate_recommend", "reason": "postprocess_budget_exhausted"}
            runtime.stream_writer({"degradation": degradation})
            return {"recommend": [], "degradations": [degradation]}
        query = state["query"]
        upload_fallback = upload_schema_recommendations(state)
        fallback_options = (
            options_from_queries(upload_fallback, RECOMMEND_NUMBER)
            if upload_fallback
            else build_contextual_recommendations(state, RECOMMEND_NUMBER)
        )
        fallback_event = recommendation_event(fallback_options)
        if upload_fallback:
            runtime.stream_writer(fallback_event)
            return fallback_event
        if not settings.query.recommend_llm_enabled:
            runtime.stream_writer(fallback_event)
            return fallback_event

        payload, context_budget = build_model_prompt_payload(
            state,
            "generate_recommend",
            settings=settings,
        )
        recommendation_context = build_recommendation_context(state)
        serialized_recommendation_context = json.dumps(
            recommendation_context,
            ensure_ascii=False,
            allow_nan=False,
        )
        model_prompts = dict(context_budget.get("model_prompts") or {})
        prompt_evidence = dict(model_prompts.get("generate_recommend") or {})
        prompt_evidence["recommendation_context_tokens"] = estimate_tokens(
            recommendation_context
        )
        prompt_evidence["after_tokens"] = int(prompt_evidence.get("after_tokens") or 0) + int(
            prompt_evidence["recommendation_context_tokens"]
        )
        model_prompts["generate_recommend"] = prompt_evidence
        context_budget["model_prompts"] = model_prompts
        if int(prompt_evidence.get("after_tokens") or 0) > int(settings.query.recommend_prompt_token_limit):
            degradation = {
                "node_id": "generate_recommend",
                "reason": "recommend_prompt_budget_exceeded",
                "estimated_tokens": int(prompt_evidence.get("after_tokens") or 0),
                "limit": int(settings.query.recommend_prompt_token_limit),
            }
            runtime.stream_writer({"degradation": degradation})
            runtime.stream_writer(fallback_event)
            return {
                **fallback_event,
                "context_budget": context_budget,
                "degradations": [degradation],
            }

        prompt_template = PromptTemplate(
            template=load_prompt("generate_recommend"),
            input_variables=[
                "query",
                "table_infos",
                "metric_infos",
                "chat_history",
                "recommendation_context",
                "articles_number",
            ],
        )
        chain = (
            prompt_template
            | (llm or get_task_llm(runtime, "recommend"))
            | StrOutputParser()
        )
        invocation = await invoke_optional_llm(
            chain,
            {
                "query": query,
                "table_infos": payload["table_infos"],
                "metric_infos": payload["metric_infos"],
                "chat_history": payload["chat_history"],
                "recommendation_context": serialized_recommendation_context,
                "articles_number": str(RECOMMEND_NUMBER),
            },
            node_id="generate_recommend",
            route="recommend",
            fallback="",
            timeout_seconds=task_timeout(runtime, "recommend"),
            core_path=False,
            runtime=runtime,
            request_deadline_at=route_deadline_at(
                state,
                "recommend",
                timeout_seconds=task_timeout(runtime, "recommend"),
            ),
            recovery_enabled=False,
        )
        raw = invocation.value

        parsed = _extract_json_array(raw)
        recommendations = normalize_model_recommendations(
            parsed,
            state=state,
            fallback=fallback_options,
            limit=RECOMMEND_NUMBER,
        )
        if not parsed:
            logger.warning("推荐问题解析为空，使用确定性上下文推荐")
        else:
            logger.info(
                "生成推荐问题完成：count=%s kinds=%s",
                len(recommendations),
                [item["kind"] for item in recommendations],
            )

        event = recommendation_event(recommendations)
        runtime.stream_writer(event)
        return {
            **event,
            "context_budget": context_budget,
            "llm_attempts": invocation.attempts,
            **({"degradations": invocation.degradations} if invocation.degradations else {}),
        }
    except Exception as e:
        logger.error(f"生成推荐问题失败：{str(e)}")
        degradation = {"node_id": "generate_recommend", "reason": str(e)[:300]}
        runtime.stream_writer({"degradation": degradation})
        upload_fallback = upload_schema_recommendations(state)
        recommendations = (
            options_from_queries(upload_fallback, RECOMMEND_NUMBER)
            if upload_fallback
            else build_contextual_recommendations(state, RECOMMEND_NUMBER)
        )
        event = recommendation_event(recommendations)
        runtime.stream_writer(event)
        return {**event, "degradations": [degradation]}
