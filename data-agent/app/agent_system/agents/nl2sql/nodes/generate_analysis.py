import asyncio
import logging
import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent_system.agents.nl2sql.context import DataAgentContext
from app.agent_system.agents.nl2sql.llm import get_task_llm
from app.agent_system.agents.nl2sql.state import DataAgentState
from app.agent_system.agents.nl2sql.llm import invoke_optional_llm
from app.agent_system.governance.deadline import route_deadline_at
from app.agent_system.runtime.performance import postprocess_budget_exhausted
from app.agent_system.agents.nl2sql.prompts.prompt_loader import load_prompt
from app.domain.query.analysis_statistics import (
    compute_analysis_statistics,
    is_numeric_value,
)

logger = logging.getLogger(__name__)

llm = None

_compute_stats = compute_analysis_statistics


_COLUMN_LABELS = {
    "customer_name": "客户",
    "product_name": "商品",
    "region_name": "地区",
    "total_amount": "累计金额",
    "total_sales": "总销量",
    "order_count": "订单数",
    "repurchase_rate": "复购率",
}


def _column_label(column: str) -> str:
    return _COLUMN_LABELS.get(column, column)


def _format_number(value) -> str:
    """把数值格式化为人类可读字符串：>=1000 用千分位，否则保留 4 位小数并去尾零。"""
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return str(value)
    # 大数用千分位并保留 2 位小数，去尾零
    if abs(number) >= 1000:
        return f"{number:,.2f}".rstrip("0").rstrip(".")
    # 小数保留 4 位并去尾零
    return f"{number:.4f}".rstrip("0").rstrip(".")


def build_deterministic_analysis(query: str, rows: list[dict], stats: dict | None = None) -> str:
    """为已校验结果生成无模型、无外推的可用统计摘要。

    核心实现思路：基于 _compute_stats 的聚合统计，用模板拼出“行数 + 数值列
    合计/平均/最值 + 最高记录”摘要，绝不添加结果之外的业务推断，保证确定性
    与安全。供 LLM 不可用/预算耗尽/语义 IR 已编译时复用。

    业务逻辑：
        1. 无结果 -> 提示检查时间范围/筛选条件
        2. 列出返回行数与列标签
        3. 对前 2 个数值列：单行报值，多行报合计/平均/最值
        4. 若存在分类列，附带最高记录的分类标签与数值
        5. 末尾声明“未添加结果之外的业务推断”
    """
    if not rows:
        return "查询已成功执行，但当前条件下没有返回数据。可检查时间范围或筛选条件。"
    stats = stats or _compute_stats(rows)
    columns = list(rows[0].keys())
    # 列标签：用中文友好名替换常见英文列名
    labels = "、".join(_column_label(column) for column in columns)
    lines = [f"本次查询返回 {len(rows)} 行结果，包含{labels}。"]
    category_columns = [column for column in columns if stats.get(column, {}).get("kind") == "category"]
    numeric_columns = [column for column in columns if stats.get(column, {}).get("kind") == "numeric"]
    # 至多描述前 2 个数值列，避免摘要过长
    for column in numeric_columns[:2]:
        item = stats[column]
        label = _column_label(column)
        # 单行结果：直接报值，无需聚合
        if len(rows) == 1:
            lines.append(f"{label}为 {_format_number(rows[0].get(column))}。")
            continue
        # 多行：报合计/平均/最大/最小
        lines.append(
            f"{label}合计 {_format_number(item['sum'])}，平均 {_format_number(item['avg'])}，"
            f"最大 {_format_number(item['max'])}，最小 {_format_number(item['min'])}。"
        )
        # 若有分类列，附带最高记录的分类标签与数值
        if category_columns:
            category = category_columns[0]
            # 只在数值列有值的行中找最高记录
            candidates = [
                row for row in rows if is_numeric_value(row.get(column))
            ]
            if candidates:
                maximum = max(candidates, key=lambda row: float(str(row.get(column))))
                lines.append(
                    f"最高记录为{_column_label(category)}“{maximum.get(category)}”，"
                    f"{label} {_format_number(maximum.get(column))}。"
                )
    # 显式声明不外推，避免用户误以为摘要包含结果之外的业务推断
    lines.append("以上为查询结果的确定性统计摘要，未添加结果之外的业务推断。")
    return "\n".join(lines)


def _fallback_analysis(runtime, state: DataAgentState, *, reason: str, message: str) -> dict:
    result = state.get("result") or []
    analysis = build_deterministic_analysis(state.get("query", ""), result)
    degradation = {
        "node_id": "generate_analysis",
        "reason": reason,
        "message": message,
        "fallback": "deterministic_summary",
    }
    runtime.stream_writer({"degradation": degradation})
    runtime.stream_writer({"analysis": analysis})
    return {"analysis": analysis, "degradations": [degradation]}


async def generate_analysis(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    settings = runtime.context["settings"]
    """
    基于查询结果与表结构，让大模型生成自然语言数据解读。

    核心功能：把用户提问、SQL、图表配置、结果列、聚合统计、样例、表结构交给 LLM，
             产出一段简洁中文解读（概述+关键发现+结论），通过 SSE 推给前端。

    业务逻辑：
        1. 检查请求级 analysis_enabled；关闭时不展示阶段、不调用模型
        2. 开启后向前端推送阶段"数据分析解读"
        3. 从 state 取 query / sql / result / chart / table_infos
        4. 由 result 派生列名、行数、聚合统计、样例（前 10 行，长值截断）
        5. 构造 prompt 链调用 LLM
        6. 通过 stream_writer 把 {"analysis": text} 推给前端并写入 state

    Args:
        state: 含 query / sql / result / chart / table_infos
        runtime: LangGraph 运行时，携带流式写入器

    Returns:
        dict: {"analysis": str} —— 自然语言解读
    """
    try:
        enabled = state.get(
            "analysis_enabled",
            bool(settings.query.analysis_enabled_default),
        )
        if not enabled:
            runtime.stream_writer({
                "optional_feature": {
                    "node_id": "generate_analysis",
                    "status": "disabled",
                }
            })
            logger.info("数据解读未启用，跳过 generate_analysis")
            return {"analysis": ""}
        runtime.stream_writer({"stage": "数据分析解读"})
        if postprocess_budget_exhausted(
            state,
            budget_ms=settings.query.postprocess_budget_ms,
        ):
            return _fallback_analysis(
                runtime,
                state,
                reason="postprocess_budget_exhausted",
                message="完整回答预算已耗尽，已使用确定性统计摘要",
            )
        query = state["query"]
        sql = state["sql"]
        result = state.get("result") or []
        chart = state.get("chart") or {}
        table_infos = state.get("table_infos", [])
        # 多轮对话历史（旧→新，仅含 query/sql/脱敏结构状态，不含结果行）：
        # 解读时可参考上一轮结果做对比/延续，如"相比上一轮总销量，本轮按地区拆分后..."
        chat_history = state.get("chat_history", [])

        result_columns = list(result[0].keys()) if result else []
        row_count = len(result)
        stats = _compute_stats(result)
        result_sample = [
            {k: (str(v)[:120] if v is not None else None) for k, v in row.items()}
            for row in result[:10]
        ]

        # TODO-42 编译器覆盖的聚合/排名/复购率都有明确结构，直接使用本地统计
        # 摘要可省去一次尾延迟较高的 LLM 调用，同时避免超出结果的业务推断。
        if state.get("semantic_ir", {}).get("compiler", {}).get("emitted"):
            analysis = build_deterministic_analysis(query, result, stats)
            logger.info(f"确定性数据解读完成（{len(analysis)}字）")
            runtime.stream_writer({"analysis": analysis})
            return {"analysis": analysis}

        prompt_template = PromptTemplate(
            template=load_prompt("generate_analysis"),
            input_variables=["query", "sql", "chart", "result_columns", "row_count", "stats", "result_sample", "table_infos", "chat_history"],
        )
        chain = (
            prompt_template
            | (llm or get_task_llm(runtime, "postprocess"))
            | StrOutputParser()
        )
        timeout_seconds = max(
            0.1,
            float(getattr(settings.query, "analysis_timeout_seconds", 30.0) or 30.0),
        )
        invocation = await invoke_optional_llm(
            chain,
            {
                "query": query,
                "sql": sql,
                "chart": yaml.dump(chart, allow_unicode=True, sort_keys=False),
                "result_columns": yaml.dump(result_columns, allow_unicode=True, sort_keys=False),
                "row_count": str(row_count),
                "stats": yaml.dump(stats, allow_unicode=True, sort_keys=False),
                "result_sample": yaml.dump(result_sample, allow_unicode=True, sort_keys=False),
                "table_infos": yaml.dump(table_infos, allow_unicode=True, sort_keys=False),
                "chat_history": yaml.dump(chat_history, allow_unicode=True, sort_keys=False),
            },
            node_id="generate_analysis",
            route="postprocess",
            fallback="",
            timeout_seconds=timeout_seconds,
            core_path=False,
            runtime=runtime,
            request_deadline_at=route_deadline_at(
                state,
                "postprocess",
                timeout_seconds=timeout_seconds,
            ),
            recovery_enabled=False,
        )
        analysis = str(invocation.value or "").strip()
        if not analysis:
            fallback_result = _fallback_analysis(
                runtime,
                state,
                reason=(
                    invocation.attempts[-1]["reason_code"]
                    if invocation.attempts else "empty_model_output"
                ),
                message="数据解读模型未返回有效内容，已使用确定性统计摘要",
            )
            fallback_result["llm_attempts"] = invocation.attempts
            fallback_result["degradations"] = [
                *[
                    {**item, "fallback": "deterministic_summary"}
                    for item in invocation.degradations
                ],
                *(fallback_result.get("degradations") or []),
            ]
            return fallback_result

        logger.info(f"生成数据解读完成（{len(analysis)}字）")
        runtime.stream_writer({"analysis": analysis})
        return {"analysis": analysis, "llm_attempts": invocation.attempts}
    except asyncio.TimeoutError:
        timeout_seconds = float(getattr(settings.query, "analysis_timeout_seconds", 30.0) or 30.0)
        message = f"数据解读模型超过 {timeout_seconds:g} 秒未返回"
        logger.warning(f"{message}，使用确定性统计摘要")
        return _fallback_analysis(runtime, state, reason="timeout", message=message)
    except Exception as e:
        message = f"{type(e).__name__}: {str(e) or '未提供错误详情'}"
        logger.error(f"生成数据解读失败：{message}")
        # 解读失败不阻断主流程，前端仍已有 SQL/结果/图表
        return _fallback_analysis(runtime, state, reason="exception", message=message[:300])
