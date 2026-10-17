"""基于确定性证据完成 SQL 候选置信度评估、自适应扩展与候选选择。"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import sqlglot
from sqlglot import exp

from app.agent_system.agents.nl2sql.semantic_contract import audit_sql_against_query_contract
from app.agent_system.agents.nl2sql.services.semantic_guard import (
    _entity_listing_requested,
    _matching_display_columns,
    analyze_semantic_shape,
    explicit_grouping_entries,
)


# 置信度算法版本会随评估契约变化而更新，用于区分不同规则生成的遥测结果。
CONFIDENCE_VERSION = "candidate-confidence-v4"
CANDIDATE_DECISION_SCHEMA = "candidate-decision-v4"
SEMANTIC_REVIEW_POLICY_VERSION = "semantic-review-v2"
DECISION_MODES = frozenset({"none", "adaptive_expansion", "semantic_review"})

# 只有这些可通过重新生成 SQL 或补全结构契约改善的弱点，才允许触发自适应候选；
# 纯信息性弱点不会额外消耗候选生成预算。
ACTIONABLE_WEAKNESS_PREFIXES = (
    "display_column_missing", "join_shape_incomplete", "anti_join_contract_missing",
    "complex_semantics_without_complete_ir",
    "explain_near_cost_limit", "sql_parse_failed", "validation_failed",
    "query_contract_gap",
)


def _bounded(value: float) -> float:
    """
    将置信度数值限制在合法区间内，并统一保留四位小数。

    核心功能：先把输入值裁剪到 ``[0.0, 1.0]``，再通过统一精度消除浮点数
    尾差，确保后续阈值比较和日志记录具有稳定结果。

    业务逻辑：
        1. 使用下界 ``0.0`` 截断负数
        2. 使用上界 ``1.0`` 截断超过满分的数值
        3. 将合法区间内的结果四舍五入到四位小数

    Args:
        value: 尚未归一化的候选置信度分数。

    Returns:
        float: 位于 ``[0.0, 1.0]`` 且保留四位小数的分数。
    """
    return round(max(0.0, min(1.0, value)), 4)


def _sql_features(sql: str, state: Mapping | None = None) -> dict[str, Any]:
    """
    从候选 SQL 及当前状态中提取可重复计算的结构化特征。

    核心功能：结合语义形状分析结果与 sqlglot AST，统一生成表数量、连接数量、
    投影字段、AST 规模以及连接契约等特征，供置信度评分阶段使用。

    业务逻辑：
        1. 将空状态归一化为空映射
        2. 基于查询、表元数据和语义 IR 分析 SQL 的语义形状
        3. 尝试按 MySQL 方言解析 SQL；若失败则返回带失败标记的保守特征
        4. 从顶层 SELECT 投影表达式中收集实际引用的字段名
        5. 汇总 AST 统计量与语义形状，形成统一特征字典

    Args:
        sql: 待分析的候选 SQL 文本。
        state: 当前数据代理状态，可包含用户查询、表信息和语义 IR。

    Returns:
        dict[str, Any]: SQL 解析状态、结构统计、投影字段及语义契约特征。
    """
    # 空状态统一为只读语义上的空映射，避免后续逐项读取时处理 None。
    state = state or {}

    # 先从业务语义侧分析表连接、反连接等契约；即使 AST 解析失败，
    # 其中不依赖 sqlglot 解析成功的结果仍可作为诊断信息返回。
    shape = analyze_semantic_shape(
        sql,
        str(state.get("query") or ""),
        state.get("table_infos") or [],
        state.get("semantic_ir") or {},
    )

    # 使用固定 MySQL 方言解析，保证同一 SQL 在不同运行环境中的 AST 结构一致。
    try:
        tree = sqlglot.parse_one(str(sql or ""), read="mysql")
    except Exception:
        # 解析失败时返回明确的零值特征，同时保留语义分析产生的其他诊断字段；
        # 排除 shape 自带的 parsed，确保本函数的失败标记拥有最终解释权。
        return {
            "parsed": False, "table_count": 0, "join_count": 0,
            "projection_count": 0, "ast_nodes": 0, "projected_columns": [],
            **{key: value for key, value in shape.items() if key != "parsed"},
        }

    # 遍历顶层 SELECT 的每个投影表达式，收集其中引用的字段并转为小写；
    # 集合去重可避免同一字段在复杂表达式中被重复计数。
    projected = {
        column.name.lower()
        for expression in (tree.expressions if isinstance(tree, exp.Select) else [])
        for column in expression.find_all(exp.Column)
        if column.name
    }

    # AST 统计量描述 SQL 的物理复杂度，shape 则补充业务语义契约；
    # 合并时剔除重复字段，避免覆盖此处基于 AST 得出的解析和表数量结果。
    return {
        "parsed": True,
        "table_count": int(shape["physical_table_count"]),
        "join_count": sum(1 for _ in tree.find_all(exp.Join)),
        "projection_count": len(list(tree.expressions)) if isinstance(tree, exp.Select) else 0,
        "ast_nodes": sum(1 for _ in tree.walk()),
        "projected_columns": sorted(projected),
        **{key: value for key, value in shape.items() if key not in {"parsed", "physical_table_count"}},
    }


def _display_expectation(
    query: str, table_infos: Sequence[Mapping], semantic_ir: Mapping | None = None,
) -> tuple[bool, set[str]]:
    """
    判断用户问题是否要求展示实体字段，并推导可接受的展示列集合。

    核心功能：综合中文查询意图、语义 IR 的维度定义以及表字段的展示属性，
    识别名单、明细和分组类问题是否必须投影可读名称，避免只返回内部 ID。

    业务逻辑：
        1. 归一化查询文本和语义 IR，并提取带字段名的有效维度
        2. 识别名单/名称类展示意图及按实体分组意图
        3. 若不存在展示要求，则立即返回空预期
        4. 将语义维度字段作为首批候选展示列
        5. 对 ``*_id`` 维度补充同表中标记为展示字段的列
        6. 仅在 IR 未给出候选列时，依据查询中的实体词从相关表回退推断

    Args:
        query: 用户的自然语言查询。
        table_infos: 候选表及其字段元数据序列。
        semantic_ir: 可选的结构化语义中间表示。

    Returns:
        tuple[bool, set[str]]: 是否必须展示可读字段，以及小写形式的候选字段集合。
    """
    # 查询文本统一转小写，以兼容其中可能出现的英文实体或字段提示。
    text = str(query or "").lower()
    ir = dict(semantic_ir or {})

    routing = dict(ir.get("model_sql_policy") or {})
    query_contract = dict(ir.get("query_contract") or {})
    guided_model_route = bool(
        routing.get("use_model") and query_contract.get("authority") != "closed"
    )

    # Guided model routes must be reviewed independently.  Only a closed IR may
    # contribute parser-derived dimensions; otherwise use question+metadata
    # evidence below so an erroneous A/IR cannot grade itself as high confidence.
    dimensions = [
        item for item in (ir.get("dimensions") or [])
        if isinstance(item, Mapping) and item.get("column")
    ] if not guided_model_route else []
    independent_grouping = explicit_grouping_entries(text, table_infos)
    listing = _entity_listing_requested(text)
    independently_matched_display = _matching_display_columns(text, table_infos)
    required = (
        listing
        or bool(independent_grouping)
        or bool(independently_matched_display)
        or bool(dimensions)
    )

    # 普通聚合或单值查询没有展示字段契约，无需对投影列施加额外限制。
    if not required:
        return False, set()

    # IR 中明确声明的维度字段拥有最高优先级，可直接作为预期展示列。
    expected = {
        str(item.get("column") or "").lower() for item in dimensions
    } | {column for _, column in independent_grouping} | independently_matched_display
    by_table = {str(table.get("name") or "").lower(): table for table in table_infos or []}

    # 遍历维度，针对内部 ID 字段查找同表内更适合面向用户展示的名称列。
    for dimension in dimensions:
        table_name = str(dimension.get("table") or "").lower()
        column_name = str(dimension.get("column") or "").lower()

        # 仅对可定位到来源表的 ``*_id`` 维度扩展展示列，避免跨表误匹配。
        if column_name.endswith("_id") and table_name in by_table:
            expected.update(
                str(column.get("name") or "").lower()
                for column in (by_table[table_name].get("columns") or [])
                if column.get("is_display")
            )

    # 部分 IR 可能无法识别维度。仅在候选集合为空时，才从查询提及的实体推断
    # 展示字段；不能把整个 schema 的所有展示列都误设为必需字段。
    explicitly_requests_id = any(token in text for token in ("id", "编号", "标识"))
    if explicitly_requests_id:
        expected -= independently_matched_display
    return bool(expected), expected


def actionable_weaknesses(assessment: Mapping) -> list[str]:
    """
    从候选评估结果中筛选可通过额外生成或修正处理的弱点。

    核心功能：按照统一的可操作弱点前缀白名单过滤诊断项，使自适应扩展逻辑
    只响应确有机会通过下一候选改善的问题。

    业务逻辑：
        1. 读取评估结果中的弱点列表，缺失时按空列表处理
        2. 将每个弱点规范化为字符串
        3. 保留以前缀白名单中任一值开头的弱点

    Args:
        assessment: 候选评估结果映射。

    Returns:
        list[str]: 可触发额外候选生成的弱点列表。
    """
    return [
        str(value) for value in (assessment.get("weaknesses") or [])
        if str(value).startswith(ACTIONABLE_WEAKNESS_PREFIXES)
    ]


def assess_candidate(
    sql: str,
    state: Mapping,
    *,
    validation: Mapping | None = None,
    label: str = "A",
    query_settings: Any | None = None,
) -> dict:
    """
    仅依据可验证的确定性证据评估 SQL 候选的置信度。

    核心功能：汇总 SQL 校验、语义 IR 完整度、检索质量、依赖降级、SQL 结构契约、
    展示字段、执行成本和重试历史等信号，计算归一化分数及置信度档位；整个过程
    不采纳模型自报置信度，从而保证结果可解释、可复现。

    业务逻辑：
        1. 初始化校验结果、SQL 特征及基础分
        2. 根据确定性校验、语义 IR、检索质量和依赖状态增减分数
        3. 检查 SQL 解析、连接结构、展示字段及反连接契约
        4. 根据 EXPLAIN 预估行数与配置上限评估成本余量
        5. 对重试历史和未闭合的复杂语义施加惩罚
        6. 将有效候选分数归一化并映射为高、中、低置信度档位
        7. 组装完整证据，并筛出可通过额外候选修复的弱点

    Args:
        sql: 待评分的候选 SQL 文本。
        state: 当前数据代理状态及语义、检索、降级和重试信息。
        validation: 确定性 SQL 校验结果，可包含有效性、错误码和预估行数。
        label: 候选标识，默认主候选为 ``A``。

    Returns:
        dict: 包含分数、档位、正向理由、弱点、SQL 特征和校验证据的评估结果。
    """
    # =============================================================================
    # 阶段 1：初始化校验证据、SQL 特征与基础分
    # =============================================================================

    # 将缺失校验归一化为空字典；只有显式 valid 真值才视为校验通过。
    validation = dict(validation or {})
    valid = bool(validation.get("valid"))
    sql_features = _sql_features(sql, state)

    # reasons 记录加分依据，weaknesses 记录减分或契约缺口，便于后续解释决策。
    reasons: list[str] = []
    weaknesses: list[str] = []
    score = 0.55

    # =============================================================================
    # 阶段 2：评估校验、语义 IR、检索与依赖质量
    # =============================================================================

    # 确定性校验通过是最强基础证据；失败时保留具体错误码供修正流程使用。
    if valid:
        score += 0.25
        reasons.append("deterministic_validation_passed")
    else:
        weaknesses.append(str(validation.get("error_code") or "validation_failed"))

    # 完整 IR 表明查询语义已结构化闭合；部分 IR 仍可使用，但需要小幅降权。
    ir = state.get("semantic_ir") or {}
    if ir.get("status") == "complete":
        score += 0.10
        reasons.append("semantic_ir_complete")
    elif ir:
        score -= 0.03
        weaknesses.append("semantic_ir_partial")

    query_contract_audit = None
    if ir.get("query_contract"):
        query_contract_audit = audit_sql_against_query_contract(
            sql, ir.get("query_contract")
        )
        if query_contract_audit.get("valid"):
            score += 0.05
            reasons.append("shared_query_contract_passed")
        else:
            codes = list(dict.fromkeys(
                str(item.get("code") or "unknown")
                for item in (query_contract_audit.get("violations") or [])
            ))
            score -= min(0.18, 0.06 + 0.03 * len(codes))
            weaknesses.append("query_contract_gap:" + ",".join(codes[:4]))

    # 统计至少命中一次强匹配的检索路由，多路强匹配可有限累积加分。
    retrieval = list(state.get("retrieval_metrics") or [])
    strong_routes = sum(int(item.get("strong_match_count") or 0) > 0 for item in retrieval)
    if strong_routes:
        score += min(0.06, strong_routes * 0.02)
        reasons.append(f"strong_retrieval_routes:{strong_routes}")
    elif retrieval:
        # 已执行检索但没有强匹配，说明上下文相关性不足，记录为非致命弱点。
        weaknesses.append("no_strong_retrieval_match")

    # 依赖降级越多，外部证据越不完整；惩罚设置上限以免单项信号压垮总分。
    degradations = list(state.get("degradations") or [])
    if degradations:
        score -= min(0.12, 0.06 + 0.02 * len(degradations))
        weaknesses.append(f"dependency_degradations:{len(degradations)}")
    else:
        # 无依赖降级代表本次评估基于完整链路，给予少量稳定性加分。
        score += 0.03
        reasons.append("no_dependency_degradation")

    # =============================================================================
    # 阶段 3：检查 SQL 结构、展示字段与反连接契约
    # =============================================================================

    # 仅在 AST 解析成功时评估连接结构；解析失败本身属于高风险问题。
    if sql_features["parsed"]:
        if sql_features["join_contract_satisfied"]:
            score += 0.04

            # 需要连接时记录“连接图已闭合”，无需连接时记录“语义源无需连接”，
            # 区分两类通过原因以便排查评分依据。
            reasons.append(
                "join_shape_connected"
                if sql_features["join_required"]
                else "join_not_required_for_semantic_sources"
            )
        else:
            # 语义来源需要连接但 SQL 未形成完整连接结构，可能产生错误或笛卡尔积。
            score -= 0.18
            weaknesses.append("join_shape_incomplete")
    else:
        # 无法解析的 SQL 不具备可信的结构证据，施加显著惩罚。
        score -= 0.30
        weaknesses.append("sql_parse_failed")

    # 推导名单/分组类问题应展示的字段，并检查 SQL 投影是否命中至少一个候选列。
    display_required, display_columns = _display_expectation(
        state.get("query", ""), state.get("table_infos") or [], ir
    )
    display_hit = bool(display_columns & set(sql_features["projected_columns"]))
    if display_required and not display_hit:
        # 用户要求可读实体信息但 SQL 仅返回 ID 或聚合值时，标记为可修正契约缺口。
        score -= 0.16
        weaknesses.append("display_column_missing")
    elif display_required:
        # 存在展示要求且投影命中可读字段，说明结果可直接面向用户。
        score += 0.04
        reasons.append("display_column_present")
    else:
        # 无展示要求时无需检查具体字段，但给予少量“无额外契约负担”的加分。
        score += 0.02

    # 只有用户语义要求“没有/从未”等反连接时，才检查 SQL 是否满足排除契约。
    if sql_features.get("anti_join_requested"):
        if sql_features.get("anti_join_satisfied"):
            score += 0.06
            reasons.append(
                f"anti_join_contract_satisfied:{sql_features.get('anti_join_strategy')}"
            )
        else:
            # 请求反连接却未使用合规策略，SQL 很可能把“没有”错误解释为普通筛选。
            score -= 0.18
            weaknesses.append("anti_join_contract_missing")

    # =============================================================================
    # 阶段 4：评估 EXPLAIN 成本余量
    # =============================================================================

    # 配置上限至少取 1，防止异常配置导致除零，同时保持成本比例可计算。
    estimated_rows = validation.get("estimated_rows")
    maximum = max(1, int(getattr(query_settings, "max_explain_rows", 5_000_000) or 1))
    if estimated_rows is not None and maximum > 0:
        ratio = float(estimated_rows) / maximum

        # 预估行数超过上限的 80% 时接近成本红线，应降低候选可信度。
        if ratio > 0.8:
            score -= 0.08
            weaknesses.append("explain_near_cost_limit")
        elif ratio <= 0.5:
            # 预估行数不超过上限一半，说明查询具有较充足的执行成本余量。
            score += 0.03
            reasons.append("explain_cost_headroom")

    # =============================================================================
    # 阶段 5：结合修正历史与复杂语义完整度调整分数
    # =============================================================================

    # 多次修正通常意味着生成过程不稳定，按次数扣分并设置惩罚上限。
    retry_count = int(state.get("retry_count") or 0)
    if retry_count:
        score -= min(0.20, retry_count * 0.08)
        weaknesses.append(f"correction_attempts:{retry_count}")

    # 对同比、环比、占比、复购、否定和对比等复杂问题，部分 IR 需要额外契约闭合。
    query = str(state.get("query") or "")
    if ir and ir.get("status") != "complete" and any(
        marker in query for marker in ("同比", "环比", "占比", "复购", "没有", "从未", "对比")
    ):

        # 若复杂语义实际是反连接且 SQL 已满足契约，可用确定性结构证据弥补部分 IR。
        if sql_features.get("anti_join_requested") and sql_features.get("anti_join_satisfied"):
            reasons.append("partial_ir_closed_by_anti_join_contract")
        elif not sql_features.get("anti_join_requested"):
            # 非反连接复杂语义没有其他确定性闭合证据时，保守降低置信度。
            score -= 0.05
            weaknesses.append("complex_semantics_without_complete_ir")

    # =============================================================================
    # 阶段 6：归一化分数、划分档位并组装可解释结果
    # =============================================================================

    # 无效候选无条件归零；有效候选才进行区间裁剪，避免其他加分掩盖校验失败。
    value = _bounded(score) if valid else 0.0
    high = float(getattr(query_settings, "candidate_confidence_high", 0.78))
    low = float(getattr(query_settings, "candidate_confidence_low", 0.58))

    # 依次比较高、低阈值，将连续分数映射为后续决策使用的离散档位。
    band = "high" if value >= high else "medium" if value >= low else "low"
    result = {
        "version": CONFIDENCE_VERSION,
        "label": label,
        "valid": valid,
        "score": value,
        "band": band,
        "reasons": reasons,
        "weaknesses": weaknesses,
        "sql_features": sql_features,
        "query_contract_audit": query_contract_audit,
        "validation": {
            "error_code": validation.get("error_code"),
            "estimated_rows": estimated_rows,
        },
    }

    # 从全部弱点中派生可触发额外候选的子集，避免扩展逻辑重复维护过滤规则。
    result["actionable_weaknesses"] = actionable_weaknesses(result)
    return result


def should_expand_candidate(
    assessment: Mapping,
    state: Mapping,
    elapsed_ms: int,
    *,
    query_settings: Any | None = None,
) -> tuple[bool, str]:
    """
    判断主候选是否应在时间与次数预算内扩展生成备选候选。

    核心功能：按固定优先级检查功能开关、候选有效性、置信度、修正历史、
    扩展预算、时间截止线和可操作弱点，仅在仍有明确改善空间时触发额外生成。

    业务逻辑：
        1. 功能关闭时拒绝扩展
        2. 主候选无效时交由定向修正流程处理
        3. 高置信度候选直接走快速路径
        4. 已使用修正或候选预算时拒绝重复消耗
        5. 达到响应时间截止线时保护整体时延
        6. 中置信度且不存在可操作契约缺口时保留当前候选
        7. 其余情况允许扩展，并返回包含当前档位的原因

    Args:
        assessment: 主候选的确定性评估结果。
        state: 当前数据代理状态及已使用预算信息。
        elapsed_ms: 当前流程已经消耗的毫秒数。

    Returns:
        tuple[bool, str]: 是否触发备选候选生成，以及对应的机器可读原因。
    """
    # 功能开关拥有最高优先级，关闭时不再检查任何候选状态。
    if not bool(getattr(query_settings, "adaptive_candidate_enabled", True)):
        return False, "disabled"

    # 无效主候选应进入已有的定向纠错机制，不能额外消耗自适应候选预算。
    if not assessment.get("valid"):
        return False, "primary_invalid_use_targeted_correction"

    # 高置信度已达到直接采用标准，走快速路径以减少一次模型调用。
    if assessment.get("band") == "high":
        return False, "high_confidence_fast_path"

    # 已发生重试表示纠错预算被使用，禁止在同一请求中继续扩展候选。
    if int(state.get("retry_count") or 0) > 0:
        return False, "correction_budget_already_used"

    # candidate_decision 中的触发标记用于防止流程重复进入扩展分支。
    if state.get("candidate_decision", {}).get("adaptive_triggered"):
        return False, "candidate_budget_already_used"

    # 超过端到端响应截止时间后优先返回已有候选，保护 TTResult 目标。
    if elapsed_ms >= int(getattr(
        query_settings, "adaptive_candidate_deadline_ms", 52_000
    )):
        return False, "ttresult_budget_exhausted"

    # 中置信度但没有可修复契约缺口时，额外候选缺少明确优化目标，不值得扩展。
    if assessment.get("band") == "medium" and not actionable_weaknesses(assessment):
        return False, "no_actionable_contract_gap"

    # 通过全部门禁后触发扩展，并在原因中携带当前置信度档位。
    return True, f"{assessment.get('band')}_confidence_expand"


def infer_candidate_decision_mode(value: Mapping | None) -> str:
    """Return the stable governance lane for current and historical decisions.

    Older artifacts used ``adaptive_triggered`` for both the confidence-driven
    expansion and the newer semantic review.  Semantic-review reason prefixes
    are therefore authoritative before consulting that legacy boolean.
    """
    item = dict(value or {})
    explicit = str(item.get("decision_mode") or "").strip().lower()
    if explicit in DECISION_MODES:
        return explicit

    reason_values = (
        item.get("trigger_reason"),
        item.get("selection_reason"),
        item.get("candidate_outcome"),
    )
    if bool(item.get("semantic_review_triggered")) or any(
        str(reason or "").startswith("semantic_review_")
        for reason in reason_values
    ):
        return "semantic_review"
    if bool(item.get("adaptive_triggered")):
        return "adaptive_expansion"
    if str(item.get("trigger_reason") or "").endswith("_confidence_expand"):
        return "adaptive_expansion"
    return "none"


def should_review_candidate(
    assessment: Mapping,
    state: Mapping,
    elapsed_ms: int,
    *,
    query_settings: Any | None = None,
) -> tuple[bool, str]:
    """Determine whether a validated primary SQL can receive one semantic review.

    The review gate is intentionally independent from the confidence score.  A
    deterministic validator proves that a statement is safe and executable; it
    does not prove that every requested dimension, measure, filter, or output
    grain survived generation.  The review is therefore attempted once for any
    valid primary candidate while a useful bounded model window remains.
    """
    if not bool(getattr(query_settings, "sql_semantic_review_enabled", True)):
        return False, "semantic_review_disabled"
    if not assessment.get("valid"):
        return False, "primary_invalid_use_targeted_correction"
    prior_decision = state.get("candidate_decision") or {}
    if (
        infer_candidate_decision_mode(prior_decision) == "semantic_review"
        and bool(
            prior_decision.get("semantic_review_triggered")
            or prior_decision.get("extra_call_used")
        )
    ):
        return False, "semantic_review_already_used"

    remaining_ms = int(getattr(
        query_settings, "adaptive_candidate_deadline_ms", 52_000
    )) - int(elapsed_ms)
    # Do not start a provider request with a token window that is useful only
    # on paper.  Four seconds is the existing bounded recovery floor; a lower
    # configured route timeout remains authoritative for tests/deployments.
    minimum_start_ms = min(
        max(1, int(float(getattr(
            query_settings, "adaptive_candidate_timeout_seconds", 18.0
        )) * 1000)),
        max(1, int(getattr(query_settings, "llm_recovery_min_budget_ms", 4_000))),
    )
    if remaining_ms < minimum_start_ms:
        return False, "semantic_review_budget_insufficient"
    return True, "semantic_review_required"


def normalized_sql_identity(sql: str) -> str | None:
    """Return a deterministic SQL identity without persisting the SQL itself.

    The identity is intentionally used only in-memory for candidate comparison.
    Valid SQL is canonicalized through the MySQL AST, so harmless formatting,
    keyword case and explicit/implicit aliases do not manufacture a candidate
    delta.  The whitespace fallback keeps the function deterministic for
    malformed candidates while their validation result remains authoritative.
    """
    text = str(sql or "").strip()
    if not text:
        return None
    try:
        return sqlglot.parse_one(text, read="mysql").sql(
            dialect="mysql",
            pretty=False,
            normalize=True,
            comments=False,
        )
    except Exception:  # validation will reject malformed SQL separately
        return " ".join(text.rstrip(";").lower().split())


def normalized_sql_equal(primary_sql: str, alternative_sql: str) -> bool | None:
    """Compare two SQL candidates after canonicalization.

    ``None`` means that at least one side is absent, which is different from a
    proven SQL delta and must not be reported as one in governance evidence.
    """
    primary_identity = normalized_sql_identity(primary_sql)
    alternative_identity = normalized_sql_identity(alternative_sql)
    if primary_identity is None or alternative_identity is None:
        return None
    return primary_identity == alternative_identity


def choose_candidate(
    primary: Mapping,
    alternative: Mapping,
    *,
    primary_sql: str | None = None,
    alternative_sql: str | None = None,
    query_settings: Any | None = None,
) -> tuple[str, str]:
    """
    根据确定性评估结果在主候选 A、备选 B 或不采用之间作出选择。

    核心功能：优先保证候选通过确定性校验，仅当备选候选带来达到配置阈值的
    实质置信度增益时才切换到 B；分数持平或增益不足时保留成本更低的 A。

    业务逻辑：
        1. 若备选无效，则根据主候选档位选择 A 或拒绝全部候选
        2. 若主候选无效而备选有效，则选择唯一有效的 B
        3. 计算备选相对主候选的置信度增益
        4. 若增益达到配置阈值，则选择 B
        5. 若两个候选均为低置信度，则拒绝全部候选
        6. 其余持平或增益不足情况保留 A

    Args:
        primary: 主候选 A 的评估结果。
        alternative: 备选候选 B 的评估结果。

    Returns:
        tuple[str, str]: 选中的候选标签（``A``、``B`` 或 ``none``）及选择原因。
    """
    # 备选未通过确定性校验时绝不选 B；主候选也处于低档则两者都不可信。
    if not alternative.get("valid"):
        if primary.get("band") == "low":
            return "none", "no_trustworthy_candidate"
        return "A", "alternative_rejected_by_deterministic_validation"

    same_sql = normalized_sql_equal(
        primary_sql or "", alternative_sql or "",
    ) if primary_sql is not None and alternative_sql is not None else None
    if same_sql is True:
        # B 没有产生可验证的 SQL 变化时不允许切换标签；否则会把
        # 同一次执行或裁判波动误归因为候选 B 的收益。
        if primary.get("valid"):
            return "A", "alternative_has_no_normalized_sql_delta"
        return "none", "same_sql_validation_conflict"

    # 备选有效而主候选无效时，无需比较分数，直接选择唯一可信候选 B。
    if not primary.get("valid"):
        return "B", "only_alternative_valid"

    # 只比较确定性评分的绝对增量，缺失分数按零处理以保证计算稳定。
    gain = float(alternative.get("score") or 0) - float(primary.get("score") or 0)

    # 只有达到配置的最小实质增益，额外生成 B 的质量收益才足以覆盖切换成本。
    if gain >= float(getattr(
        query_settings, "candidate_min_confidence_gain", 0.03
    )):
        return "B", "alternative_has_material_confidence_gain"

    # 两个候选均落入低置信度档时，即使其中一个略高也不应输出不可信 SQL。
    if primary.get("band") == "low" and alternative.get("band") == "low":
        return "none", "no_trustworthy_candidate"

    # 平分或 B 的增益不足时保留先生成的 A，避免没有实质收益的候选切换。
    return "A", "primary_kept_on_tie_or_insufficient_gain"


def compact_candidate_decision(value: Mapping | None) -> dict:
    """
    将完整候选决策压缩为不包含敏感内容的可观测证据。

    核心功能：仅保留候选标签、有效性、分数、档位、弱点、结构特征、耗时和
    选择原因，主动排除 SQL 文本、提示词、查询结果行及字面量，便于安全记录。

    业务逻辑：
        1. 将空输入归一化为空字典并初始化候选摘要列表
        2. 遍历候选，仅复制允许记录的评估字段
        3. 将弱点列表限制为前十项，控制遥测体积
        4. 从嵌套校验结果中提取非敏感错误码
        5. 汇总决策版本、触发状态、选择结果、调用预算和耗时
        6. 对缺失字段应用稳定默认值并返回隐私安全摘要

    Args:
        value: 完整候选决策映射，允许为空。

    Returns:
        dict: 可用于日志、遥测或状态持久化的候选决策摘要。
    """
    # 空输入转换为空字典，使后续默认值逻辑能够统一处理首次或未触发状态。
    item = dict(value or {})
    decision_mode = infer_candidate_decision_mode(item)
    explicit_mode = str(item.get("decision_mode") or "").strip().lower()
    mode_inferred = explicit_mode not in DECISION_MODES and decision_mode != "none"
    candidates = []

    # 遍历全部候选，只提取不含 SQL 正文、提示词、结果数据或字面量的白名单字段。
    for candidate in item.get("candidates") or []:
        candidates.append({
            "label": candidate.get("label"), "valid": bool(candidate.get("valid")),
            "score": candidate.get("score"), "band": candidate.get("band"),

            # 两类弱点各保留前十项，在维持主要诊断价值的同时限制记录体积。
            "weaknesses": list(candidate.get("weaknesses") or [])[:10],
            "actionable_weaknesses": list(candidate.get("actionable_weaknesses") or [])[:10],

            # 错误码和 SQL 结构特征不包含查询数据，可安全用于聚合分析。
            "error_code": (candidate.get("validation") or {}).get("error_code"),
            "sql_features": dict(candidate.get("sql_features") or {}),
            "query_contract_audit": {
                "schema_version": (
                    candidate.get("query_contract_audit") or {}
                ).get("schema_version"),
                "valid": bool(
                    (candidate.get("query_contract_audit") or {}).get("valid")
                ),
                "violation_codes": [
                    str(row.get("code") or "unknown")
                    for row in (
                        (candidate.get("query_contract_audit") or {}).get("violations")
                        or []
                    )
                ][:8],
                "contract_hash": (
                    candidate.get("query_contract_audit") or {}
                ).get("contract_hash"),
            },
        })

    # 汇总请求级决策信息；所有缺失布尔值、计数和耗时都转换为稳定基础类型。
    legacy_semantic_trigger = bool(
        decision_mode == "semantic_review"
        and (
            item.get("adaptive_triggered")
            or item.get("extra_call_used")
            or str(item.get("trigger_reason") or "") == "semantic_review_required"
        )
    )
    semantic_review_triggered = bool(
        decision_mode == "semantic_review"
        and (
            item.get("semantic_review_triggered")
            if "semantic_review_triggered" in item
            else legacy_semantic_trigger
        )
    )
    adaptive_triggered = bool(
        decision_mode == "adaptive_expansion" and item.get("adaptive_triggered")
    )
    primary = next(
        (candidate for candidate in candidates if candidate.get("label") == "A"),
        {},
    )
    semantic_review_eligible = bool(
        item.get("semantic_review_eligible")
        if "semantic_review_eligible" in item
        else decision_mode == "semantic_review" and primary.get("valid") is True
    )
    any_triggered = adaptive_triggered or semantic_review_triggered
    evidence_complete = bool(
        item
        and candidates
        and item.get("trigger_reason") is not None
        and (
            explicit_mode in DECISION_MODES
            or mode_inferred
            or decision_mode == "none"
        )
    )
    return {
        "schema": item.get("schema") or CANDIDATE_DECISION_SCHEMA,
        "version": item.get("version") or CONFIDENCE_VERSION,
        "decision_mode": decision_mode,
        "decision_mode_inferred": mode_inferred,
        "evidence_complete": evidence_complete,
        # These booleans are intentionally mutually exclusive.  In particular,
        # semantic review must never consume the historical adaptive-expansion
        # trigger budget or its release gate.
        "adaptive_triggered": adaptive_triggered,
        "semantic_review_triggered": semantic_review_triggered,
        "semantic_review_eligible": semantic_review_eligible,
        "trigger_reason": item.get("trigger_reason"),
        "selected": item.get("selected") or "A",
        "selection_reason": item.get("selection_reason"),
        # 只记录规范化后是否存在 SQL 变化，不记录任何规范化 SQL 文本。
        "sql_delta": (
            item.get("sql_delta")
            if isinstance(item.get("sql_delta"), bool)
            else None
        ),
        "candidate_count": int(item.get("candidate_count") or len(candidates) or 1),
        "extra_call_used": bool(item.get("extra_call_used")),
        "generation_ms": int(item.get("generation_ms") or 0),
        "validation_ms": int(item.get("validation_ms") or 0),

        # 未提供结果状态时，根据是否实际触发扩展区分“未触发”和“结果未知”。
        "candidate_outcome": item.get("candidate_outcome") or (
            "not_triggered" if not any_triggered else "unknown"
        ),
        "candidates": candidates,
    }
