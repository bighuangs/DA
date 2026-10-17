import json
import logging
import re

import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent_system.agents.nl2sql.context import DataAgentContext
from app.agent_system.agents.nl2sql.llm import get_task_llm, invoke_optional_llm
from app.agent_system.agents.nl2sql.prompts.prompt_loader import load_prompt
from app.agent_system.agents.nl2sql.state import DataAgentState
from app.agent_system.governance.deadline import route_deadline_at
from app.agent_system.runtime.performance import postprocess_budget_exhausted
from app.domain.query.chart_specs import (
    build_deterministic_chart,
    explicit_chart_type,
    table_chart,
)

logger = logging.getLogger(__name__)

llm = None

def _extract_json(text: str) -> dict | None:
    """
    从 LLM 输出中容错提取首个 JSON 对象。

    业务逻辑：
        1. 去除 Markdown 代码块围栏（```json ... ```）
        2. 正则定位首个 {...} 平衡片段
        3. json.loads 解析，失败则返回 None
    """
    if not text:
        return None
    # 去掉 ```json / ``` 围栏
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    # 取首个 {...} 片段
    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _fallback_table(result_columns: list[str], *, title: str = "查询结果") -> dict:
    """LLM 解析失败时的兜底：生成 table 配置，避免前端无图表可渲染。"""
    return table_chart(result_columns, title=title)


async def generate_chart(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """
    根据查询结果与表结构，让大模型生成图表配置项（type/title/axis 或 columns）。

    核心功能：把用户提问、最终 SQL、结果列名、结果样例、表结构交给 LLM，
             产出一份前端可直接渲染的图表 JSON 配置，并通过 SSE 推送给前端。

    业务逻辑：
        1. 向前端推送阶段"生成图表配置"
        2. 从 state 取 query / sql / result / table_infos
        3. 由 result 派生 result_columns 与 result_sample（前 3 行，长值截断）
        4. 构造 prompt 链并调用 LLM
        5. 容错解析 JSON；失败则兜底为 table 配置
        6. 通过 stream_writer 把 {"chart": config} 推给前端
        7. 返回 {"chart": config} 写入 state

    Args:
        state: 含 query / sql / result / table_infos
        runtime: LangGraph 运行时，携带流式写入器

    Returns:
        dict: {"chart": dict} —— 图表配置
    """
    settings = runtime.context["settings"]
    runtime.stream_writer({"stage": "生成图表配置"})
    try:
        query = state["query"]
        original_query = str(state.get("original_query") or "")
        requested_type = explicit_chart_type(original_query or query)
        sql = state["sql"]
        result = state.get("result") or []
        table_infos = state.get("table_infos", [])

        if not settings.query.chart_llm_enabled or postprocess_budget_exhausted(
            state, budget_ms=settings.query.postprocess_budget_ms
        ):
            chart = build_deterministic_chart(
                query,
                result,
                requested_type=requested_type,
            )
            runtime.stream_writer({"chart": chart, "performance": {"chart_mode": "deterministic"}})
            return {"chart": chart}

        # 由真实结果派生列名与样例，比让 LLM 解析 SQL 别名更可靠
        result_columns = list(result[0].keys()) if result else []
        result_sample = [
            {k: (str(v)[:120] if v is not None else None) for k, v in row.items()}
            for row in result[:3]
        ]

        prompt_template = PromptTemplate(
            template=load_prompt("generate_chart"),
            input_variables=["query", "sql", "result_columns", "result_sample", "table_infos"],
        )
        chain = (
            prompt_template
            | (llm or get_task_llm(runtime, "postprocess"))
            | StrOutputParser()
        )
        invocation = await invoke_optional_llm(
            chain,
            {
                "query": query,
                "sql": sql,
                "result_columns": yaml.dump(result_columns, allow_unicode=True, sort_keys=False),
                "result_sample": yaml.dump(result_sample, allow_unicode=True, sort_keys=False),
                "table_infos": yaml.dump(table_infos, allow_unicode=True, sort_keys=False),
            },
            node_id="generate_chart",
            route="postprocess",
            fallback="",
            timeout_seconds=settings.query.postprocess_timeout_seconds,
            core_path=False,
            runtime=runtime,
            request_deadline_at=route_deadline_at(
                state,
                "postprocess",
                timeout_seconds=settings.query.postprocess_timeout_seconds,
            ),
            recovery_enabled=False,
        )
        raw = invocation.value

        chart = _extract_json(raw)
        if chart is None:
            logger.warning(f"图表配置解析失败，回退 table。原始输出：{raw}")
            chart = _fallback_table(result_columns)
        else:
            logger.info(f"生成图表配置完成：type={chart.get('type')} title={chart.get('title')}")

        # 图表类型属于用户可确定表达的展示契约，不能被模型推荐覆盖。
        if requested_type is not None:
            generated_title = chart.get("title") if isinstance(chart, dict) else None
            chart = build_deterministic_chart(
                query,
                result,
                requested_type=requested_type,
            )
            if isinstance(generated_title, str) and generated_title.strip():
                chart["title"] = generated_title.strip()[:80]

        # 推送给前端渲染
        runtime.stream_writer({"chart": chart})
        return {
            "chart": chart,
            "llm_attempts": invocation.attempts,
            **({"degradations": invocation.degradations} if invocation.degradations else {}),
        }
    except Exception as e:
        logger.error(f"生成图表配置失败：{str(e)}")
        # 失败时仍按用户明确展示要求生成确定性配置，不阻断主流程。
        query = str(state.get("query") or "查询结果")
        original_query = str(state.get("original_query") or "")
        result = state.get("result") or []
        chart = build_deterministic_chart(
            query,
            result,
            requested_type=explicit_chart_type(original_query or query),
        )
        degradation = {"node_id": "generate_chart", "reason": str(e)[:300]}
        runtime.stream_writer({"chart": chart, "degradation": degradation})
        return {"chart": chart, "degradations": [degradation]}
