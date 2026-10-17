"""主候选与自适应候选 SQL 共用的独立确定性校验。

本模块提供 validate_candidate_sql（单条候选校验，不修改 graph state、不向用户报错）
与 validate_sql（图节点，把校验结果写回 state 并按 MAX_RETRY 决定是否对外报错），
供 generate_sql 主链路与 judge_candidate 备选链路复用同一套校验逻辑。

校验流水线（顺序敏感，前序失败即短路）：
    1. apply_semantic_repairs：执行一次有界、版本化修复注册表
    3. access_policy.authorize_sql：行列权限编译，产出 authorized_sql
    4. check_table_whitelist：表名白名单复核（防御深度）
    5. check_semantic_contract：语义契约校验（多轮/口径一致性）
    6. check_semantic_ir：TODO-42 语义 IR 校验
    7. validate_follow_up_sql：TODO-44 多轮继承结构校验
    8. dw_mysql_repo.validate_sql：EXPLAIN 语法/成本校验
    9. check_metric_formulas：指标口径公式校验
"""
from __future__ import annotations

from collections.abc import Mapping
import logging

from langgraph.runtime import Runtime

from app.agent_system.agents.nl2sql.context import DataAgentContext
from app.agent_system.agents.nl2sql.follow_up import validate_follow_up_sql
from app.agent_system.agents.nl2sql.semantic_ir import compile_semantic_ir
from app.agent_system.agents.nl2sql.state import DataAgentState, MAX_RETRY
from app.agent_system.agents.nl2sql.services.model_sql_policy import (
    deterministic_draft_is_safe,
    deterministic_model_fallback_allowed,
    use_model_sql,
)
from app.domain.query.safety.query_guard import MetricFormulaError, QueryCostError, check_metric_formulas
from app.agent_system.agents.nl2sql.services.semantic_guard import check_semantic_contract, check_semantic_ir
from app.agent_system.agents.nl2sql.services.semantic_repair import (
    apply_semantic_repairs,
    semantic_repair_ledger_entry,
)
from app.domain.query.safety.semantic_violation import (
    SemanticContractError,
    SemanticIRError,
    SemanticParseError,
    stable_hash,
    violation_from_exception,
)
from app.domain.query.safety.sql_security import SQLSecurityError, extract_table_names
from app.domain.data_sources.context import (
    authorize_sql_assets,
    translate_logical_upload_tables,
)
from app.domain.query.safety.source_analysis_guard import validate_source_analysis_sql

logger = logging.getLogger(__name__)


def _classified_error(exc: Exception) -> tuple[str, str, dict]:
    """把校验异常归类为 (error_code, 可读错误信息)，便于路由与可观测性分流。

    核心实现思路：按异常类型映射到稳定的 error_code 字符串，并把异常信息
    包装成中文可读描述，供 correct_sql 节点针对性修正与前端展示。
    """
    violation = violation_from_exception(exc).to_dict()
    code = str(violation["violation_code"])
    if isinstance(exc, SemanticParseError):
        return code, f"SQL 解析失败：{exc}", violation
    if isinstance(exc, SemanticIRError):
        return code, f"语义 IR 校验失败：{exc}", violation
    if isinstance(exc, SemanticContractError):
        return code, f"语义契约校验失败：{exc}", violation
    if isinstance(exc, MetricFormulaError):
        return code, f"指标口径校验失败：{exc}", violation
    if isinstance(exc, QueryCostError):
        return code, f"查询成本守卫失败：{exc}", violation
    if isinstance(exc, SQLSecurityError):
        return code, f"SQL安全校验失败：{exc}", violation
    if isinstance(exc, TimeoutError):
        return code, "SQL 校验超时：执行计划阶段预算已耗尽", violation
    # 兜底：未识别的异常统一归为 SQL 语法错误
    return code, f"SQL 校验失败：{exc}", violation


def _allowed_tables(state: Mapping) -> set[str]:
    """计算当前允许出现的表名白名单：当前 table_infos + 历史成功 SQL 的表名。

    核心实现思路：把当前召回过滤后的表名与历史轮次中成功执行过的 SQL 表名合并，
    作为白名单传给 check_table_whitelist，保证多轮继承场景下历史表名不被误拒。
    """
    allowed = {str(item.get("name")) for item in state.get("table_infos", []) if item.get("name")}
    if (state.get("data_source_context") or {}).get("source_mode") != "dw_only":
        return allowed
    # 历史轮次的表名也纳入白名单：多轮改写可能引用上一轮的表，需放行
    for entry in state.get("chat_history", []) or []:
        previous_sql = str(entry.get("sql") or "") if isinstance(entry, Mapping) else ""
        if previous_sql:
            try:
                allowed.update(extract_table_names(previous_sql))
            except Exception:
                # 历史 SQL 解析失败不影响主流程，跳过该轮表名
                pass
    return allowed


def closed_model_draft_fallback(
    state: Mapping,
    rejected_sql: str,
) -> tuple[str, dict] | None:
    """Return a closed compiler draft after a model candidate is rejected.

    This is deliberately narrower than a generic SQL fallback.  It is only
    available for a DW-only request that actually selected the model route,
    and only when the deterministic compiler still owns a fully covered IR.
    The caller must run the returned draft through the complete validation
    pipeline before it can be authorized or executed.
    """

    source_context = dict(state.get("data_source_context") or {})
    if str(source_context.get("source_mode") or "dw_only") != "dw_only":
        return None
    semantic_ir = dict(state.get("semantic_ir") or {})
    if not use_model_sql(semantic_ir):
        return None
    compiler = dict(semantic_ir.get("compiler") or {})
    coverage = dict(semantic_ir.get("semantic_coverage") or {})
    if (
        not compiler.get("supported")
        or list(coverage.get("unresolved_slots") or [])
        or float(coverage.get("coverage_score") or 0.0) < 1.0
    ):
        return None
    try:
        draft_sql = str(compile_semantic_ir(semantic_ir) or "").strip()
    except (KeyError, TypeError, ValueError):
        # Legacy compatibility compilers may not understand a newer IR shape.
        # A fallback is optional and must never turn that mismatch into a
        # request failure.
        return None
    if not deterministic_draft_is_safe(semantic_ir, draft_sql):
        return None
    if not draft_sql or draft_sql == str(rejected_sql or "").strip():
        return None
    policy = dict(semantic_ir.get("model_sql_policy") or {})
    return draft_sql, {
        "schema_version": "model-sql-validation-fallback-v1",
        "trigger": "model_candidate_rejected",
        "compiler_mode": (semantic_ir.get("compiler") or {}).get("mode"),
        "semantic_family": str(
            semantic_ir.get("semantic_family")
            or semantic_ir.get("intent")
            or "unknown"
        ),
        "model_route_selected": True,
        "physical_correction_llm_call": False,
        "original_decision": policy.get("decision"),
        "raw_sql_persisted": False,
    }


async def validate_candidate_sql(
    candidate_sql: str,
    state: Mapping,
    runtime: Runtime[DataAgentContext],
) -> dict:
    """校验单条候选 SQL，不修改 graph state、不向用户报错。

    核心实现思路：把一条候选 SQL 跑完整条校验流水线，成功返回 valid=True 与
    authorized_sql，失败返回 valid=False 与分类后的 error/error_code，供调用方
    （主链路 validate_sql 或备选 judge_candidate）按需写回 state 或参与判优。

    业务逻辑：
        1. 通过版本化注册表运行一次有界修复，得到规范化后的 sql 与审计证据
        2. 行列权限编译得到 authorized_sql
        3. 表名白名单复核（防御深度，防止 correct_sql 改出越权表）
        4. 语义契约 / 语义 IR / 多轮继承结构校验
        5. EXPLAIN 语法与成本校验
        6. 指标口径公式校验
        7. 全部通过返回 valid=True；任一失败捕获异常并归类返回 valid=False
    """
    sql = str(candidate_sql or "").strip()
    semantic_repair: dict = {}
    try:
        previous_repair = (
            (state.get("validation_evidence") or {}).get("semantic_repair") or {}
        )
        attempted_rules = (
            previous_repair.get("attempted_rule_codes") or []
            if previous_repair.get("original_sql_hash") == stable_hash(sql)
            else []
        )
        # 1. 语义修复注册表：固定顺序、单次 pass、全量证据和 P0 回退。
        sql, repair_result = apply_semantic_repairs(
            sql,
            query=state.get("query", ""),
            table_infos=state.get("table_infos", []),
            semantic_ir=state.get("semantic_ir"),
            attempted_rule_codes=attempted_rules,
            settings=runtime.context["settings"],
        )
        semantic_repair = repair_result.to_dict()
        dw_mysql_repo = runtime.context["dw_mysql_repo"]
        access_policy = runtime.context.get("access_policy")
        asset_access = runtime.context.get("asset_access_context")

        # 3. 先在逻辑表名层做 Prompt 白名单复核；上传表的内部物理名从未提供
        # 给模型，也不能由模型直接提交。
        from app.domain.query.safety.sql_security import check_table_whitelist
        allowed_tables = _allowed_tables(state)
        # 存在白名单或存在权限策略时才校验；空白名单 + 无策略时不拦截。
        if allowed_tables or (access_policy and access_policy.policies):
            check_table_whitelist(
                sql,
                sorted(allowed_tables),
                reject_empty=bool(access_policy and access_policy.policies),
            )
        source_guard = validate_source_analysis_sql(
            sql,
            access=asset_access,
            mapping=state.get("source_mapping"),
        )
        # 4. 逻辑别名由服务端 AST 翻译为本轮授权的物理上传版本。
        physical_sql = translate_logical_upload_tables(sql, asset_access)
        # 5. 行列权限编译：正式 DW 表继续使用现有策略；上传表的 owner /
        # workspace/version/status/TTL 则由独立资产上下文控制。
        authorized_sql = (
            access_policy.authorize_sql(physical_sql)
            if access_policy else physical_sql
        )
        authorize_sql_assets(authorized_sql, asset_access)

        # 6. 语义契约校验：多轮口径一致性、维度/度量匹配
        check_semantic_contract(
            sql,
            state.get("query", ""),
            state.get("table_infos", []),
            state.get("chat_history", []),
        )
        # 7. 语义 IR 校验：TODO-42 结构化意图与生成 SQL 的一致性
        check_semantic_ir(sql, state.get("semantic_ir"))
        # 8. 多轮继承结构校验：TODO-44 保证继承结构未被丢失
        validate_follow_up_sql(sql, state.get("follow_up"))
        # 9. EXPLAIN 语法与成本校验只接收已翻译、已授权物理 SQL。
        explain = await dw_mysql_repo.validate_sql(authorized_sql)
        # 10. 指标口径公式校验：逻辑正式表名与物理名相同，上传表无全局指标。
        check_metric_formulas(
            sql,
            state.get("query", ""),
            state.get("metric_infos", []),
            semantic_ir=state.get("semantic_ir"),
        )
        return {
            "valid": True,
            "sql": sql,
            "authorized_sql": authorized_sql,
            "error": None,
            "error_code": None,
            "violation": None,
            "semantic_repair": semantic_repair,
            "source_guard": source_guard,
            "estimated_rows": (explain or {}).get("estimated_rows") if isinstance(explain, Mapping) else None,
        }
    except Exception as exc:
        # 任一校验失败：归类错误码与可读信息，返回 valid=False
        code, error, violation = _classified_error(exc)
        logger.error(
            "SQL 候选校验失败 code=%s error_type=%s sql_hash=%s",
            code,
            type(exc).__name__,
            stable_hash(sql),
        )
        return {
            "valid": False,
            "sql": sql,
            "authorized_sql": None,
            "error": error,
            "error_code": code,
            "violation": violation,
            "semantic_repair": semantic_repair,
            "estimated_rows": None,
        }


async def validate_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """SQL 校验图节点：校验主候选 SQL 并按 MAX_RETRY 决定是否对外报错。

    核心功能：调用 validate_candidate_sql 校验 state["sql"]，成功则把 sql 与
    authorized_sql 写回 state 供 execute_sql 执行；失败则把 error 写回 state
    供 correct_sql 修正，仅在重试次数已达上限时才把错误推送给前端。

    业务逻辑：
        1. 向前端推送阶段"校验SQL"
        2. 调用 validate_candidate_sql 校验当前 sql
        3. 校验成功：清空 error，写回 sql/authorized_sql/validation_evidence
        4. 校验失败：写回 error/sql/validation_evidence（authorized_sql 置空）
        5. 若 retry_count 已达 MAX_RETRY，把错误通过 stream_writer 推给前端
           （未达上限时不推送，由 correct_sql 静默修正）

    Returns:
        dict: {"error", "sql", "authorized_sql", "validation_evidence"}
    """
    runtime.stream_writer({"stage": "校验SQL"})
    # 复用统一校验流水线，避免主链路与备选链路逻辑分叉
    result = await validate_candidate_sql(state["sql"], state, runtime)
    ledger = [semantic_repair_ledger_entry(
        result,
        phase="primary_validate",
        committed_at_phase=result.get("valid") is True,
    )]
    if result.get("semantic_repair"):
        runtime.stream_writer({"semantic_repair": result["semantic_repair"]})
    if result.get("violation"):
        runtime.stream_writer({"semantic_violation": result["violation"]})
    if result["valid"]:
        logger.info("SQL 权限、语义、语法与成本校验成功")
        return {
            "error": None,
            "sql": result["sql"],
            "authorized_sql": result["authorized_sql"],
            "validation_evidence": result,
            "semantic_repair_ledger": ledger,
        }
    runtime_fallback = runtime.context.get("model_sql_fallback_enabled")
    fallback = (
        closed_model_draft_fallback(state, result.get("sql") or state["sql"])
        if deterministic_model_fallback_allowed(
            state.get("semantic_ir"),
            runtime_override=(
                bool(runtime_fallback)
                if runtime_fallback is not None else None
            ),
            query_settings=runtime.context["settings"].query,
        )
        else None
    )
    if fallback is not None:
        draft_sql, fallback_evidence = fallback
        draft_result = await validate_candidate_sql(draft_sql, state, runtime)
        ledger.append(semantic_repair_ledger_entry(
            draft_result,
            phase="deterministic_fallback_validate",
            committed_at_phase=draft_result.get("valid") is True,
        ))
        if draft_result["valid"]:
            fallback_evidence = {
                **fallback_evidence,
                "rejected_error_code": result.get("error_code"),
                "draft_validated": True,
            }
            updated_ir = dict(state.get("semantic_ir") or {})
            updated_ir["model_sql_policy"] = {
                **dict(updated_ir.get("model_sql_policy") or {}),
                "actual_action": "validated_deterministic_draft",
                "fallback_reason": result.get("error_code"),
                "physical_correction_llm_call": False,
            }
            runtime.stream_writer({
                "stage": "模型候选未通过，采用已校验确定性草案",
                "model_sql_fallback": fallback_evidence,
            })
            logger.warning(
                "模型 SQL 候选未通过，安全采用闭合确定性草案 code=%s family=%s",
                result.get("error_code"),
                fallback_evidence["semantic_family"],
            )
            return {
                "error": None,
                "sql": draft_result["sql"],
                "authorized_sql": draft_result["authorized_sql"],
                "semantic_ir": updated_ir,
                "validation_evidence": {
                    **draft_result,
                    "model_sql_fallback": fallback_evidence,
                    "rejected_candidate": {
                        "error_code": result.get("error_code"),
                        "violation": result.get("violation"),
                        "raw_sql_persisted": False,
                    },
                },
                "semantic_repair_ledger": ledger,
            }
    # 校验失败：未达重试上限时不推送给前端，留给 correct_sql 静默修正
    violation = result.get("violation") or {}
    if (
        int(state.get("retry_count") or 0) >= MAX_RETRY
        or violation.get("repairability") == "reject"
        or violation.get("safe_to_retry") is False
    ):
        runtime.stream_writer({"error": result["error"]})
    # No successful deterministic fallback replaced the primary validation.
    # Its failure evidence is therefore the evidence committed to graph state.
    ledger[0]["committed_at_phase"] = True
    ledger[0]["selected"] = True
    return {
        "error": result["error"],
        "sql": result["sql"],
        "authorized_sql": None,
        "validation_evidence": result,
        "semantic_repair_ledger": ledger,
    }
