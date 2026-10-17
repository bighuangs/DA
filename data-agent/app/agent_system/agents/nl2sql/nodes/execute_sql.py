import asyncio
import logging

from langgraph.runtime import Runtime

from app.agent_system.agents.nl2sql.context import DataAgentContext
from app.agent_system.agents.nl2sql.state import DataAgentState, MAX_RETRY
from app.domain.data_sources.context import (
    authorize_sql_assets,
    ensure_context_assets_current,
    redact_physical_upload_tables,
)
from app.domain.query.source_comparison import deterministic_comparison
from app.agent_system.governance.deadline import (
    compact_deadline_evidence,
    execution_timeout_seconds,
)
from app.domain.query.safety.semantic_violation import SemanticViolationV1

logger = logging.getLogger(__name__)


def _execution_violation(error: BaseException) -> SemanticViolationV1:
    """Classify execution failures before graph routing considers correction."""

    from app.domain.query.safety.sql_security import SQLSecurityError

    if isinstance(error, SQLSecurityError):
        return SemanticViolationV1(
            "security.policy_violation",
            "execution_security",
            "reject",
            "do_not_bypass_security_policy",
            False,
        )
    if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        return SemanticViolationV1(
            "execution.deadline_exhausted",
            "execution",
            "reject",
            "reduce_query_scope_or_retry",
            False,
        )
    # Only deterministic SQL-shape failures are useful to the correction model.
    # Dependency/network/transaction failures must never be "repaired" as SQL.
    message = str(error).lower()
    llm_repairable = any(marker in message for marker in (
        "unknown column",
        "ambiguous",
        "syntax",
        "group by",
        "operand",
        "invalid use of group",
        "doesn't exist",
        "not exist",
    ))
    return SemanticViolationV1(
        (
            "execution.sql_shape_error"
            if llm_repairable
            else "execution.dependency_error"
        ),
        "execution",
        "llm" if llm_repairable else "upstream",
        (
            "repair_runtime_sql_shape"
            if llm_repairable
            else "retry_dependency_without_sql_change"
        ),
        llm_repairable,
    )


async def execute_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """
    在数仓库中执行最终 SQL，成功回传结果；失败则把错误写回 state 触发再校正。

    核心功能：从 state 取出 sql 调用数仓库 dw_mysql_repo 真正执行查询。
             成功时通过 stream_writer 把结果回传给调用者，并把 error 置为 None；
             失败时不抛异常，而是把执行错误信息写入 error，由图的条件边分流到
             correct_sql 重新修正（受 MAX_RETRY 限制）。达到重试上限时把错误
             回传给调用者，由路由走向 END 终止。

    业务逻辑：
        1. 向前端推送当前流水线阶段"执行SQL"
        2. 从 state 取出 sql 与当前重试次数 retry_count，从 runtime 取出 dw_mysql_repo
        3. 调用 dw_mysql_repo.execute_sql 真正执行查询
        4. 成功 → stream_writer 回传结果，返回 {"error": None}
        5. 失败 → 捕获异常，返回 {"error": "SQL执行错误：{异常信息}"}
        6. 若 retry_count 已达 MAX_RETRY，则先把错误回传给调用者再返回

    Args:
        state: 流水线状态，包含待执行的 sql 与 retry_count
        runtime: LangGraph 运行时，携带 dw_mysql_repo 与流式写入器

    Returns:
        dict: {"error": None | str} —— 执行成功为 None，失败为错误描述字符串

    Raises:
        Exception: 仅在校正本身失败等意外情况下向上抛出；执行错误已被软捕获
    """
    runtime.stream_writer({"stage": "执行SQL"})

    # 当前已重试次数：未设置时默认为 0
    retry_count = state.get("retry_count", 0)
    sql = state.get("authorized_sql") or state["sql"]
    dw_mysql_repo = runtime.context["dw_mysql_repo"]

    try:
        # 执行前的表名白名单复核：防御深度，防止 correct_sql 改出越权表
        from app.domain.query.safety.sql_security import check_table_whitelist
        allowed_tables = [t.get("name") for t in state.get("table_infos", []) if t.get("name")]
        access_policy = runtime.context.get("access_policy")
        asset_access = runtime.context.get("asset_access_context")
        if asset_access is not None:
            ensure_context_assets_current(asset_access)
            authorize_sql_assets(sql, asset_access)
        if not state.get("authorized_sql") and access_policy:
            sql = access_policy.authorize_sql(sql)
        if asset_access is None and (
            allowed_tables or (access_policy and access_policy.policies)
        ):
            check_table_whitelist(
                sql,
                allowed_tables,
                reject_empty=bool(access_policy and access_policy.policies),
            )

        # 执行sql：在数仓库中真正运行查询语句（内部会先做只读安全校验）
        result = await dw_mysql_repo.execute_sql(
            sql,
            timeout_seconds=execution_timeout_seconds(state),
        )

        # 把最终执行的 SQL 回传给调用者：供前端展示 + Service 层捕获存入对话历史
        display_sql = (
            str(state.get("sql") or "")
            or redact_physical_upload_tables(sql, asset_access)
        )
        runtime.stream_writer({"sql": display_sql})
        # 将查询结果写给调用者：通过流式通道直接回传，而非写入 state
        runtime.stream_writer({"result": result})
        runtime.stream_writer({
            "request_deadline": compact_deadline_evidence(
                state.get("request_deadline")
            )
        })
        if (
            asset_access is not None
            and asset_access.comparison_intent
            == "independent_aggregate_compare"
        ):
            runtime.stream_writer({
                "source_comparison": deterministic_comparison(result)
            })

        logger.info(f"执行SQL完成 rows={len(result)}")
        # 执行成功：清空 error，并把结果写入 state 供 generate_chart 等后处理节点使用
        return {
            "error": None,
            "result": result,
            "authorized_sql": sql,
            "sql": display_sql,
        }
    except Exception as e:
        # 执行失败不抛异常，而是把错误信息写入 state，交给 correct_sql 节点修正
        # 这样运行时错误（如字段不存在、类型不匹配）也能触发再校正，而不只是语法错误
        from app.domain.query.safety.sql_security import SQLSecurityError
        violation = _execution_violation(e)
        if isinstance(e, SQLSecurityError):
            error = f"SQL安全校验失败：{str(e)}"
        elif isinstance(e, (asyncio.TimeoutError, TimeoutError)):
            error = "SQL执行超时：本次查询的执行阶段预算已耗尽"
        else:
            error = f"SQL执行错误：{str(e)}"
        error = redact_physical_upload_tables(
            error, runtime.context.get("asset_access_context")
        )
        logger.error(f"执行SQL失败：{error}")
        # 达到重试上限：把最终错误回传给调用者，路由将走向 END 终止
        if (
            retry_count >= MAX_RETRY
            or violation.repairability != "llm"
            or not violation.safe_to_retry
        ):
            runtime.stream_writer({"error": error})
        runtime.stream_writer({"semantic_violation": violation.to_dict()})
        runtime.stream_writer({
            "request_deadline": compact_deadline_evidence(
                state.get("request_deadline")
            )
        })
        return {
            "error": error,
            "authorized_sql": None,
            "validation_evidence": {
                "error_code": violation.violation_code,
                "violation": violation.to_dict(),
            },
        }
