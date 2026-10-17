import logging
import re
import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent_system.agents.nl2sql.context import DataAgentContext
from app.agent_system.agents.nl2sql.llm import get_task_llm, task_timeout
from app.agent_system.agents.nl2sql.state import DataAgentState
from app.agent_system.agents.nl2sql.services.model_sql_policy import use_model_sql
from app.agent_system.agents.nl2sql.llm import invoke_optional_llm
from app.agent_system.runtime.llm_output import parse_json_response
from app.domain.query.safety.query_guard import select_metric_authorities
from app.agent_system.governance.deadline import route_deadline_at
from app.agent_system.agents.nl2sql.prompts.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

llm = None


def stabilize_metric_selection(
    query: str,
    metric_infos: list[dict],
    selection: list[str],
    semantic_ir: dict | None = None,
) -> list[dict]:
    """在 LLM 过滤之后，确定性消解 SUM/AVG 等高置信口径歧义。

    核心实现思路：用 query 中的关键词判定用户真实意图（平均/金额/数量），
    再按指标 formula_rule 的 function 与 column 做确定性保留/剔除，避免模型
    因“销售”字样误选 GMV 当销量、或把销售额误选成 AOV 一类冲突。

    业务逻辑：
        1. 计算 query 是否请求平均/金额/数量等口径
        2. 遍历指标：直接命中（名/别名出现在 query）或被模型选中则默认保留
        3. 数量请求且无金额请求时，剔除 order_amount 类金额指标（互斥口径）
        4. 平均+数量请求且非直接命中时，剔除 order_amount 类金额指标
        5. 平均金额请求时：AVG 保留、SUM 且非直接命中则剔除
        6. 金额请求且非平均时：SUM+order_amount 保留、AVG 且非直接命中则剔除
        7. 用语义 family / population 解析最终公式权威
        8. 返回最终保留的指标列表
    """
    query_text = (query or "").strip().lower()
    # 模型选中的指标名集合
    selected = {str(name) for name in (selection or [])}
    # 平均口径关键词：客单价/单价/AOV 等明确指向 AVG
    average_request = any(token in query_text for token in ("平均", "均值", "客单价", "单价", "aov"))
    # 金额口径关键词：销售额/成交额/GMV 等明确指向 SUM(order_amount)
    amount_request = any(token in query_text for token in ("销售额", "成交额", "销售金额", "订单金额", "总额", "gmv"))
    # 只有明确的“每单/客单”表达才代表注册 AOV。诸如“销售额高于各省
    # 平均销售额”描述的是组间平均，不能因为同时出现“平均+销售额”就
    # 误选 AVG(order_amount)。
    average_amount_request = any(token in query_text for token in (
        "客单价", "aov", "平均订单金额", "订单平均金额", "平均单价",
        "每单金额", "单均金额", "单笔平均金额", "每笔平均金额",
        "平均每单金额", "平均每笔金额",
    ))
    # 数量口径关键词：件数/数量/销量，与金额互斥
    quantity_request = any(token in query_text for token in ("件数", "数量", "销量", "销售量"))
    # ``品牌成交额高于品类品牌平均值`` / ``售出件数高于该品类品牌
    # 平均件数`` asks for an average of already-grouped entity totals.  A
    # recalled raw-row AVG metric is a different business definition even if
    # its display name happens to be a substring of the question.
    grouped_average_comparison = bool(
        re.search(
            r"(?:高于|低于|超过|少于|不低于|不高于)"
            r"[^，。；;]{0,28}(?:平均|均值)",
            query_text,
        )
    )

    result: list[dict] = []
    for metric in metric_infos:
        name = str(metric.get("name") or "")
        aliases = [str(alias) for alias in (metric.get("alias") or [])]
        # 直接命中：指标名或别名出现在 query 中，是比模型选择更强的证据
        direct = any(term and term.lower() in query_text for term in [name, *aliases])
        rule = metric.get("formula_rule") or {}
        rule_type = str(rule.get("type") or "aggregate").lower()
        function = str(rule.get("function") or "").upper()
        column = str(rule.get("column") or "").lower()
        # 默认保留：模型选中 或 直接命中
        keep = name in selected or direct
        if (
            grouped_average_comparison
            and rule_type == "aggregate"
            and function == "AVG"
        ):
            keep = False
        # “销量/件数/数量”与金额指标是互斥口径。模型偶尔会因为“销售”字样
        # 误选 GMV；若问题没有同时明确要求销售额，金额指标必须确定性剔除，
        # 否则后续指标守卫会把本来正确的数量 SQL 强制补成多余金额列。
        if quantity_request and not amount_request and column.endswith("order_amount"):
            keep = False
        elif average_request and quantity_request and column.endswith("order_amount") and not direct:
            keep = False
        elif average_amount_request:
            # 平均金额请求必须同时满足函数和源列口径。只按 AVG 保留会把
            # AVG(order_quantity) 的“平均购买件数”一并带入，继而让指标
            # 守卫否决本来正确的 AVG(order_amount) SQL。
            if (
                rule_type == "aggregate"
                and function == "AVG"
                and column.endswith("order_amount")
            ):
                keep = True
            elif not direct or not column.endswith("order_amount"):
                keep = False
        elif amount_request and not average_request:
            # 纯金额请求：SUM+order_amount 保留，AVG 且非直接命中则剔除
            if (
                rule_type == "aggregate"
                and function == "SUM"
                and column.endswith("order_amount")
            ):
                keep = True
            elif function == "AVG" and not direct:
                keep = False
        if keep:
            result.append(metric)
    return list(select_metric_authorities(
        query,
        metric_infos,
        [str(metric.get("name") or "") for metric in result],
        semantic_ir,
        allow_surface_evidence=False,
    ))


async def filter_metric(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    settings = runtime.context["settings"]
    """
    利用大模型过滤召回的指标，只保留与提问真正相关的指标。

    核心功能：将用户提问与召回的指标信息交给大模型判断，让其输出"需要的
             指标名列表"；再据此对 metric_infos 做原地删减，缩小后续
             生成 SQL 的上下文规模、降低干扰。

    业务逻辑：
        1. 向前端推送当前流水线阶段"过滤指标"
        2. 从 state 取出 query 与 metric_infos
        3. 构造 prompt 链（模板 → llm → JsonOutputParser），将 metric_infos
           用 yaml 序列化后连同 query 一起送入大模型
        4. 大模型返回需要的指标名列表 result
        5. 遍历 metric_infos：不在 result 中的指标直接删除
        6. 返回 {"metric_infos": 过滤后的指标信息列表} 更新到 state

    Args:
        state: 流水线状态，包含 query 与 metric_infos
        runtime: LangGraph 运行时，携带上下文与流式写入器

    Returns:
        dict: {"metric_infos": list[MetricInfoState]} —— 过滤后的指标信息列表

    Raises:
        Exception: 过滤过程中任意异常都向上抛出，便于调试定位
    """
    runtime.stream_writer({"stage": "过滤指标"})
    try:
        query = state["query"]
        metric_infos = state["metric_infos"]

        semantic_ir = state.get("semantic_ir", {})
        if (
            semantic_ir.get("compiler", {}).get("supported")
            and use_model_sql(semantic_ir, query_settings=settings.query)
        ):
            selected = stabilize_metric_selection(
                query, metric_infos, [], semantic_ir
            )
            logger.info(
                "复杂语义模型路径确定性保留相关指标 count=%s", len(selected)
            )
            return {"metric_infos": selected}
        if semantic_ir.get("compiler", {}).get("supported"):
            metric_name = (semantic_ir.get("metric") or {}).get("name")
            selected = [
                metric for metric in metric_infos
                if metric_name and metric.get("name") == metric_name
            ]
            logger.info("高置信语义 IR 已跳过指标过滤 LLM count=%s", len(selected))
            return {"metric_infos": selected}
        if not metric_infos:
            return {"metric_infos": []}

        if not settings.query.selection_llm_enabled:
            selected = stabilize_metric_selection(
                query, metric_infos, [], semantic_ir
            )
            logger.info("确定性指标筛选完成 count=%s", len(selected))
            return {"metric_infos": selected}

        # 1. 调用模型，过滤掉不需要指标，生成需要的指标名的列表
        prompt_template = PromptTemplate(
            template=load_prompt("filter_metric_info"),
            input_variables=["query", "metric_infos"],
        )
        output_parser = StrOutputParser()
        chain = (
            prompt_template
            | (llm or get_task_llm(runtime, "selection"))
            | output_parser
        )
        invocation = await invoke_optional_llm(
            chain,
            {"query": query, "metric_infos": yaml.dump(
                    [{key: metric.get(key) for key in ("name", "alias", "description", "formula_rule")}
                     for metric in metric_infos],
                    allow_unicode=True,
                    sort_keys=False,
                )},
            node_id="filter_metric",
            route="selection",
            fallback="",
            timeout_seconds=task_timeout(runtime, "selection"),
            core_path=True,
            runtime=runtime,
            request_deadline_at=route_deadline_at(
                state,
                "selection",
                timeout_seconds=task_timeout(runtime, "selection"),
            ),
            recovery_enabled=False,
        )
        degradation = None
        if invocation.degradations:
            result = []
        else:
            try:
                result = parse_json_response(invocation.value, list)
            except Exception as exc:
            # 指标筛选失败时只保留问题中直接命中的注册指标；禁止猜测全部口径。
                result = []
                degradation = {
                    "node_id": "filter_metric",
                    "reason": f"structured_output_fallback:{type(exc).__name__}",
                }
                runtime.stream_writer({"degradation": degradation})
        # [指标1， 指标2]

        # 2. 应用模型选择，并确定性消除“销售额被误选成 AOV”一类 SUM/AVG 冲突。
        metric_infos = stabilize_metric_selection(
            query, metric_infos, result, semantic_ir
        )

        logger.info(f"过滤指标完成 count={len(metric_infos)}")

        return {
            "metric_infos": metric_infos,
            "llm_attempts": invocation.attempts,
            **({
                "degradations": [
                    *invocation.degradations,
                    *([degradation] if degradation else []),
                ]
            } if invocation.degradations or degradation else {}),
        }
    except Exception as e:
        logger.error(f"过滤指标信息失败：{str(e)}")
        raise
