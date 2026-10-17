
"""NL2SQL LangGraph 的显式定义入口。

模块导入只声明节点路由和构建函数；Graph 的组装与编译分别由
``build_graph``、``compile_graph`` 在组合根/Runtime 明确触发。
"""

from langgraph.constants import END, START
from app.agent_system.agents.nl2sql.constants import MAX_RETRY


async def _start_postprocess(state, runtime):
    """Fan-out marker: enhancement nodes write distinct state keys in parallel."""
    runtime.stream_writer({"stage": "并行生成图表、分析与推荐"})
    return {}


def _route_after_forbidden_guard(state):
    """硬安全拒绝优先于多轮改写，其他请求继续闭合上下文。"""

    response = state.get("intent_response") or {}
    return END if response.get("decision") == "respond" else "rewrite_follow_up"


def _route_after_follow_up(state):
    return END if state.get("error") else "classify_query_intent"


def _route_after_query_intent(state):
    """只有显式 continue 才能进入数据源、检索与 SQL 主链。"""

    decision = (state.get("query_intent") or {}).get("decision")
    return "resolve_data_source_context" if decision == "continue" else END


def _route_after_plan_decision(state):
    """Use the actual action emitted by the explicit TODO-63 decision node."""

    action = (state.get("planning_route") or {}).get("actual_action")
    return "plan_sql" if action == "plan_then_generate" else "generate_sql"


def _route_after_generate(state):
    """A typed provider failure is terminal; never turn it into an empty-SQL parse error."""
    return END if state.get("error") and not str(state.get("sql") or "").strip() else "validate_sql"


def _route_after_validate(state):
    if state.get("error") is None:
        return "judge_candidate"
    violation = (state.get("validation_evidence") or {}).get("violation") or {}
    # Only an explicitly model-repairable violation may enter correct_sql.
    # deterministic/upstream repairs are handled before this edge; security,
    # cost, dependency and deadline failures are terminal.
    if (
        violation.get("repairability") != "llm"
        or violation.get("safe_to_retry") is not True
    ):
        return END
    if state.get("retry_count", 0) >= MAX_RETRY:
        return END
    return "correct_sql"


# 执行后分流：成功 → 并行后处理；失败且未达上限 → 校正；失败且已达上限 → 终止
def _route_after_execute(state):
    if state.get("error") is None:
        return "start_postprocess"
    violation = (state.get("validation_evidence") or {}).get("violation") or {}
    if (
        violation.get("repairability") != "llm"
        or violation.get("safe_to_retry") is not True
    ):
        return END
    if state.get("retry_count", 0) >= MAX_RETRY:
        return END
    return "correct_sql"


def _route_after_candidate_judge(state):
    return END if state.get("error") else "execute_sql"


def build_graph():
    """组装并返回未编译的 NL2SQL Graph 定义。"""

    # 节点模块可能加载模型路由与具体能力适配器，只能在显式构建阶段导入。
    from langgraph.graph import StateGraph

    from app.agent_system.agents.nl2sql.context import DataAgentContext
    from app.agent_system.agents.nl2sql.state import DataAgentState
    from app.agent_system.agents.nl2sql.nodes.add_extra_context import (
        add_extra_context,
    )
    from app.agent_system.agents.nl2sql.nodes.build_semantic_ir import (
        build_semantic_ir_node,
    )
    from app.agent_system.agents.nl2sql.nodes.correct_sql import correct_sql
    from app.agent_system.agents.nl2sql.nodes.classify_query_intent import (
        classify_query_intent,
        guard_forbidden_query_intent,
    )
    from app.agent_system.agents.nl2sql.nodes.decide_plan_route import decide_plan_route
    from app.agent_system.agents.nl2sql.nodes.execute_sql import execute_sql
    from app.agent_system.agents.nl2sql.nodes.extract_keywords import extract_keywords
    from app.agent_system.agents.nl2sql.nodes.filter_metric import filter_metric
    from app.agent_system.agents.nl2sql.nodes.filter_table import filter_table
    from app.agent_system.agents.nl2sql.nodes.generate_analysis import generate_analysis
    from app.agent_system.agents.nl2sql.nodes.generate_chart import generate_chart
    from app.agent_system.agents.nl2sql.nodes.generate_recommend import generate_recommend
    from app.agent_system.agents.nl2sql.nodes.generate_sql import generate_sql
    from app.agent_system.agents.nl2sql.nodes.judge_candidate import judge_candidate
    from app.agent_system.agents.nl2sql.nodes.merge_retrieved_info import (
        merge_retrieved_info,
    )
    from app.agent_system.agents.nl2sql.nodes.pack_context import pack_context
    from app.agent_system.agents.nl2sql.nodes.plan_sql import plan_sql
    from app.agent_system.agents.nl2sql.nodes.recall_column import recall_column
    from app.agent_system.agents.nl2sql.nodes.recall_metric import recall_metric
    from app.agent_system.agents.nl2sql.nodes.recall_training import recall_training
    from app.agent_system.agents.nl2sql.nodes.recall_value import recall_value
    from app.agent_system.agents.nl2sql.nodes.resolve_data_source_context import (
        profile_upload_schema,
        propose_source_mapping,
        resolve_data_source_context,
    )
    from app.agent_system.agents.nl2sql.nodes.rewrite_follow_up import rewrite_follow_up
    from app.agent_system.agents.nl2sql.nodes.validate_sql import validate_sql
    from app.agent_system.runtime.performance import instrument_node

    graph_builder = StateGraph(
        state_schema=DataAgentState,
        context_schema=DataAgentContext,
    )

    graph_builder.add_node(
        "guard_forbidden_query_intent",
        instrument_node(
            "guard_forbidden_query_intent",
            guard_forbidden_query_intent,
        ),
    )
    graph_builder.add_node(
        "classify_query_intent",
        instrument_node("classify_query_intent", classify_query_intent),
    )

    graph_builder.add_node(
        "resolve_data_source_context",
        instrument_node("resolve_data_source_context", resolve_data_source_context),
    )
    graph_builder.add_node(
        "profile_upload_schema",
        instrument_node("profile_upload_schema", profile_upload_schema),
    )
    graph_builder.add_node(
        "propose_source_mapping",
        instrument_node("propose_source_mapping", propose_source_mapping),
    )
    graph_builder.add_node(
        "rewrite_follow_up", instrument_node("rewrite_follow_up", rewrite_follow_up)
    )
    graph_builder.add_node(
        "extract_keywords", instrument_node("extract_keywords", extract_keywords)
    )
    graph_builder.add_node(
        "recall_column", instrument_node("recall_column", recall_column)
    )
    graph_builder.add_node("recall_value", instrument_node("recall_value", recall_value))
    graph_builder.add_node(
        "recall_metric", instrument_node("recall_metric", recall_metric)
    )
    graph_builder.add_node(
        "recall_training", instrument_node("recall_training", recall_training)
    )
    graph_builder.add_node(
        "merge_retrieved_info",
        instrument_node("merge_retrieved_info", merge_retrieved_info),
    )
    graph_builder.add_node("pack_context", instrument_node("pack_context", pack_context))
    graph_builder.add_node(
        "build_semantic_ir",
        instrument_node("build_semantic_ir", build_semantic_ir_node),
    )
    graph_builder.add_node(
        "filter_metric", instrument_node("filter_metric", filter_metric)
    )
    graph_builder.add_node("filter_table", instrument_node("filter_table", filter_table))
    graph_builder.add_node(
        "add_extra_context", instrument_node("add_extra_context", add_extra_context)
    )
    graph_builder.add_node(
        "decide_plan_route",
        instrument_node("decide_plan_route", decide_plan_route),
    )
    graph_builder.add_node("plan_sql", instrument_node("plan_sql", plan_sql))
    graph_builder.add_node("generate_sql", instrument_node("generate_sql", generate_sql))
    graph_builder.add_node("validate_sql", instrument_node("validate_sql", validate_sql))
    graph_builder.add_node(
        "judge_candidate", instrument_node("judge_candidate", judge_candidate)
    )
    graph_builder.add_node("correct_sql", instrument_node("correct_sql", correct_sql))
    graph_builder.add_node("execute_sql", instrument_node("execute_sql", execute_sql))
    graph_builder.add_node(
        "start_postprocess",
        instrument_node("start_postprocess", _start_postprocess),
    )
    graph_builder.add_node(
        "generate_chart", instrument_node("generate_chart", generate_chart)
    )
    graph_builder.add_node(
        "generate_analysis", instrument_node("generate_analysis", generate_analysis)
    )
    graph_builder.add_node(
        "generate_recommend", instrument_node("generate_recommend", generate_recommend)
    )

    graph_builder.add_edge(START, "guard_forbidden_query_intent")
    graph_builder.add_conditional_edges(
        "guard_forbidden_query_intent",
        _route_after_forbidden_guard,
        {"rewrite_follow_up": "rewrite_follow_up", END: END},
    )
    graph_builder.add_conditional_edges(
        "rewrite_follow_up",
        _route_after_follow_up,
        {"classify_query_intent": "classify_query_intent", END: END},
    )
    graph_builder.add_conditional_edges(
        "classify_query_intent",
        _route_after_query_intent,
        {"resolve_data_source_context": "resolve_data_source_context", END: END},
    )
    graph_builder.add_edge("resolve_data_source_context", "profile_upload_schema")
    graph_builder.add_edge("profile_upload_schema", "propose_source_mapping")
    graph_builder.add_edge("propose_source_mapping", "extract_keywords")
    for recall_node in (
        "recall_column",
        "recall_metric",
        "recall_value",
        "recall_training",
    ):
        graph_builder.add_edge("extract_keywords", recall_node)
        graph_builder.add_edge(recall_node, "merge_retrieved_info")
    graph_builder.add_edge("merge_retrieved_info", "build_semantic_ir")
    graph_builder.add_edge("build_semantic_ir", "filter_metric")
    graph_builder.add_edge("build_semantic_ir", "filter_table")
    graph_builder.add_edge("filter_metric", "pack_context")
    graph_builder.add_edge("filter_table", "pack_context")
    graph_builder.add_edge("pack_context", "add_extra_context")
    graph_builder.add_edge("add_extra_context", "decide_plan_route")
    graph_builder.add_conditional_edges(
        "decide_plan_route",
        _route_after_plan_decision,
        {"plan_sql": "plan_sql", "generate_sql": "generate_sql"},
    )
    graph_builder.add_edge("plan_sql", "generate_sql")
    graph_builder.add_conditional_edges(
        "generate_sql",
        _route_after_generate,
        {"validate_sql": "validate_sql", END: END},
    )
    graph_builder.add_conditional_edges(
        "validate_sql",
        _route_after_validate,
        {
            "judge_candidate": "judge_candidate",
            "correct_sql": "correct_sql",
            END: END,
        },
    )
    graph_builder.add_conditional_edges(
        "judge_candidate",
        _route_after_candidate_judge,
        {"execute_sql": "execute_sql", END: END},
    )
    graph_builder.add_conditional_edges(
        "execute_sql",
        _route_after_execute,
        {
            "start_postprocess": "start_postprocess",
            "correct_sql": "correct_sql",
            END: END,
        },
    )
    # 校正后回到校验；MAX_RETRY 由路由函数统一约束。
    graph_builder.add_edge("correct_sql", "validate_sql")
    # SQL/结果已返回后并行执行增强节点，单个增强失败只降级自身。
    for postprocess_node in (
        "generate_chart",
        "generate_analysis",
        "generate_recommend",
    ):
        graph_builder.add_edge("start_postprocess", postprocess_node)
        graph_builder.add_edge(postprocess_node, END)
    return graph_builder


def compile_graph():
    """显式构建并编译 NL2SQL Graph；调用方负责缓存编译产物。"""

    return build_graph().compile()
