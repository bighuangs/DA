from langgraph.runtime import Runtime

from app.agent_system.agents.nl2sql.context import DataAgentContext
from app.agent_system.agents.nl2sql.follow_up import resolve_follow_up
from app.agent_system.agents.nl2sql.state import DataAgentState
import logging

logger = logging.getLogger(__name__)

_LONG_MEMORY_CUE = (
    "之前", "以前", "上次", "原来", "回到", "接着那个", "最开始",
    "当时", "先前", "还记得", "我说过",
)


async def rewrite_follow_up(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """多轮指代闭合节点（TODO-44）。

    在进入检索/生成 SQL 之前，把用户本轮输入中的省略语、指代词
    （如“那”“换成”“只看”等）按上一轮成功查询的语义状态进行确定性改写，
    产出一个可独立理解的 standalone_question，并对外暴露隐私安全的审计事件。

    设计要点：
    - 改写完全基于规则与上一轮 SQL 的结构化语义状态（dimensions/measures/filters/order/limit），
      不读取历史结果行，避免泄露或误用被截断/脱敏的数据。
    - 当改写无法确定（如“这些客户”引用历史结果、或缺少可继承的成功轮）时，
      不强行猜测，而是要求用户澄清，保证审计可追溯。
    - 通过 runtime.stream_writer 向前端实时推送闭合阶段与审计摘要，
      便于用户感知“多轮指代已闭合”这一中间状态。
    """
    # 取出本轮用户原始输入并去首尾空白；空查询时退化为空串以保证后续处理稳定
    original = str(state.get("query") or "").strip()
    # 当前会话最近成功轮始终优先。只有用户显式指向较早话题，或当前窗口没有
    # 成功轮时，才把经 MySQL 权威回表的长期 episode 作为最后一个候选状态。
    history = list(state.get("chat_history") or [])
    memory_context = dict(state.get("memory_context") or {})
    if not history:
        working = dict(memory_context.get("working_state") or {})
        working_state = dict(working.get("structured_state") or {})
        working_question = str(working.get("last_success_question") or "")
        if working_state and working_question:
            history.append({
                "query": working_question,
                "standalone_question": working_question,
                "sql": "",
                "semantic_state": working_state,
                "memory_source": "working_state",
            })
    if any(cue in original for cue in _LONG_MEMORY_CUE):
        summary_history_added = False
        for summary in list(memory_context.get("selected_summaries") or []):
            supported = [
                fact
                for fact in list(summary.get("structured_facts") or [])
                if fact.get("fact_type") == "successful_query_semantics"
                and fact.get("standalone_question")
                and fact.get("semantic_state")
            ]
            if not supported:
                continue
            fact = supported[-1]
            remembered_question = str(fact.get("standalone_question") or "")
            remembered_state = dict(fact.get("semantic_state") or {})
            history.append({
                "query": remembered_question,
                "standalone_question": remembered_question,
                "sql": "",
                "semantic_state": remembered_state,
                "memory_source": summary.get("summary_id"),
                "memory_source_type": "summary_segment",
            })
            summary_history_added = True
            break
        relevant = list(memory_context.get("relevant_episodes") or [])
        if relevant and not summary_history_added:
            item = relevant[0]
            payload = dict(item.get("structured_payload") or {})
            remembered_question = str(
                payload.get("standalone_question") or item.get("content") or ""
            )
            remembered_state = dict(payload.get("semantic_state") or {})
            if remembered_question and remembered_state:
                history.append({
                    "query": remembered_question,
                    "standalone_question": remembered_question,
                    "sql": "",
                    "semantic_state": remembered_state,
                    "memory_source": item.get("memory_id"),
                })
    # 调用确定性改写器：依据显式选定的历史/记忆状态闭合问题。
    resolution = resolve_follow_up(
        original,
        history,
        result_references=state.get("result_references"),
    )
    action = resolution.get("action")
    # 始终保留原始问题与完整闭合结果，便于后续节点（generate_sql / 校验 / 审计）溯源
    output = {"original_query": original, "follow_up": resolution}
    # 仅在确实发生了多轮闭合（非 new_topic）时向前端推送阶段提示与审计摘要，
    # 避免对独立新问题制造无谓的“闭合”噪声
    if action != "new_topic":
        runtime.stream_writer({"stage": "闭合多轮指代"})
        runtime.stream_writer({"follow_up": {
            "action": action, "summary": resolution.get("summary"),
            "standalone_question": resolution.get("standalone_question"), "deterministic": True,
        }})
    # 改写器判定本轮无法安全闭合（如引用历史结果、缺少可继承轮），
    # 则把澄清提示作为 error 推送给前端并写入日志，节点以 error 返回，
    # 上游节点据此跳过 SQL 生成，等待用户补充完整问题
    if resolution.get("needs_clarification"):
        message = str(resolution.get("clarification") or "请补充完整问题。")
        runtime.stream_writer({"error": message})
        logger.info(f"多轮问题需要澄清 action={action}")
        return {**output, "error": message}
    # 优先使用闭合后的独立问题；若改写器未给出（如纯新话题），回退到原始问题
    standalone = str(resolution.get("standalone_question") or original)
    # 仅在改写确实改变了问题时记录日志，便于排查“闭合是否生效”
    if standalone != original:
        logger.info(f"多轮问题确定性闭合 action={action}")
    # 用闭合后的独立问题覆盖 state.query，供后续 retrieve/generate_sql 节点使用；
    # error 置空表示本轮闭合成功，可继续主流程
    return {**output, "query": standalone, "error": None}
