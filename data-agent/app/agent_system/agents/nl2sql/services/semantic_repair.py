"""提供版本化语义修复规则注册表、单次有界执行运行时及隐私安全结果契约。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import sqlglot
from sqlglot import exp

from app.agent_system.agents.nl2sql.settings import Nl2SqlSettings
from app.agent_system.agents.nl2sql.services.semantic_guard import (
    repair_invalid_metadata_joins,
    stabilize_rank_limit_ties,
    stabilize_threshold_measure_projection,
)
from app.domain.query.safety.semantic_violation import compact_evidence, stable_hash
from app.domain.query.safety.sql_security import check_sql_read, extract_table_names


# 三个独立版本分别标识输出契约、注册表结构和规则集合，便于灰度、回滚与审计。
REPAIR_RESULT_CONTRACT = "semantic-repair-result-v1"
REPAIR_REGISTRY_VERSION = "semantic-repair-registry-v1"
RULE_SET_VERSION = "semantic-repair-rules-v2"
SEMANTIC_REPAIR_LEDGER_SCHEMA = "semantic-repair-ledger-v1"
SEMANTIC_REPAIR_LEDGER_LIMIT = 12
SEMANTIC_REPAIR_LEDGER_PHASES = {
    "primary_validate",
    "deterministic_fallback_validate",
    "candidate_b_validate",
}

# 每个运行 profile 只启用显式列出的确定性规则；P0 fallback 与线上 P2
# 保持同一组已发布规则，扩展 profile 再额外允许由编译器支持的语义 IR
# 补全唯一 LIMIT/ORDER BY。
PROFILE_RULES = {
    "inline_v1": (
        "join.unique_metadata_edge",
        "projection.threshold_measure",
        "order.rank_limit_tie_breaker",
    ),
    "violation_only_v1": ("join.unique_metadata_edge", "projection.threshold_measure"),
    "registry_same_rules_v1": (
        "join.unique_metadata_edge",
        "projection.threshold_measure",
        "order.rank_limit_tie_breaker",
    ),
    "registry_extended_v1": (
        "join.unique_metadata_edge",
        "projection.threshold_measure",
        "ir.unique_limit_order",
        # IR 先闭合 ORDER BY/LIMIT，随后在同一有界轮次补齐聚合排名并列键。
        # 反向顺序会让后一规则生成前一规则的新适用条件，导致第二次调用再改写。
        "order.rank_limit_tie_breaker",
    ),
    "targeted_correction_v1": (
        "join.unique_metadata_edge", "projection.threshold_measure",
        "order.rank_limit_tie_breaker",
    ),
    "upstream_first_v1": (
        "join.unique_metadata_edge", "projection.threshold_measure",
        "order.rank_limit_tie_breaker",
    ),
    "cascade_v1": (
        "join.unique_metadata_edge",
        "projection.threshold_measure",
        "ir.unique_limit_order",
        "order.rank_limit_tie_breaker",
    ),
}

# 定向纠错 profile 允许把稳定违规码传给模型；其中 upstream-first 与 cascade
# 还要求优先修复上游语义表示，再进行下游 SQL 校正。
TARGETED_CORRECTION_PROFILES = {
    "targeted_correction_v1", "upstream_first_v1", "cascade_v1",
}
UPSTREAM_FIRST_PROFILES = {"upstream_first_v1", "cascade_v1"}

# 进程内覆盖值用于无需重启的快速切换或回滚；None 表示遵循静态配置。
_runtime_profile_override: str | None = None
_DEFAULT_ACTIVE_PROFILE = "registry_same_rules_v1"
_DEFAULT_FALLBACK_PROFILE = "inline_v1"
_DEFAULT_SHADOW_PROFILES = (
    "registry_extended_v1",
    "targeted_correction_v1",
    "cascade_v1",
)
_DEFAULT_MAX_PASSES = 1


def _profile_config(
    settings: Nl2SqlSettings | None = None,
) -> tuple[str, str, tuple[str, ...]]:
    """
    解析当前生效、回退及影子运行 profile，并对未知配置执行安全回退。

    核心功能：合并进程内覆盖值与应用配置，将影子 profile 固化为不可变元组，
    并确保最终 active profile 一定存在于规则注册表中。

    业务逻辑：
        1. 读取语义修复配置
        2. 优先采用运行时覆盖值，否则采用配置的 active profile
        3. 提取 fallback 与 shadow profile 列表
        4. 若 active 未注册，则回退到合法 fallback；fallback 也无效时使用 ``inline_v1``

    Returns:
        tuple[str, str, tuple[str, ...]]: 生效 profile、配置的回退 profile及影子 profile。
    """
    config = settings.semantic_repair if settings is not None else None
    active = str(
        _runtime_profile_override
        or getattr(config, "active_profile", _DEFAULT_ACTIVE_PROFILE)
    )
    fallback = str(getattr(
        config, "fallback_profile", _DEFAULT_FALLBACK_PROFILE
    ))

    # 转换为字符串元组，避免调用方修改配置对象中的原始影子列表。
    shadows = tuple(str(item) for item in (
        getattr(config, "shadow_profiles", _DEFAULT_SHADOW_PROFILES) or ()
    ))

    # 未注册的 active 不得直接执行：优先采用合法回退，否则关闭到最小内联规则集。
    if active not in PROFILE_RULES:
        active = fallback if fallback in PROFILE_RULES else "inline_v1"
    return active, fallback, shadows


def runtime_profile_status(
    settings: Nl2SqlSettings | None = None,
) -> dict[str, Any]:
    """
    生成当前语义修复运行配置的可观测状态。

    核心功能：汇总生效配置、静态配置、回退与影子 profile、版本号、最大执行轮数
    和运行时覆盖状态，为管理接口、纠错策略及遥测提供统一事实来源。

    业务逻辑：
        1. 解析经过安全回退的 profile 配置
        2. 读取原始 active 配置和最大修复轮数
        3. 标记是否存在运行时覆盖
        4. 声明系统支持无需重启的回滚能力

    Returns:
        dict[str, Any]: 当前版本化运行状态。
    """
    active, fallback, shadows = _profile_config(settings)
    config = settings.semantic_repair if settings is not None else None
    return {
        "contract_version": REPAIR_REGISTRY_VERSION,
        "runtime_profile": active,
        "configured_profile": str(getattr(
            config, "active_profile", _DEFAULT_ACTIVE_PROFILE
        )),
        "fallback_profile": fallback,
        "shadow_profiles": list(shadows),
        "rule_set_version": RULE_SET_VERSION,
        "max_passes": int(getattr(config, "max_passes", _DEFAULT_MAX_PASSES)),
        "override_active": _runtime_profile_override is not None,
        "rollback_without_restart": True,
    }


def rollback_to_fallback(
    settings: Nl2SqlSettings | None = None,
) -> dict[str, Any]:
    """
    将进程内语义修复配置立即回滚到安全 fallback。

    核心功能：校验配置中的 fallback 是否已注册，若无效则使用最小 ``inline_v1``，
    并写入运行时覆盖值，使后续请求无需服务重启即可采用回退规则。

    业务逻辑：
        1. 读取配置的 fallback profile
        2. 若 fallback 已注册则设为运行时覆盖
        3. 否则使用 ``inline_v1`` 作为关闭式回退
        4. 返回更新后的运行状态

    Returns:
        dict[str, Any]: 回滚后立即生效的运行配置状态。
    """
    global _runtime_profile_override
    config = settings.semantic_repair if settings is not None else None
    fallback = str(getattr(
        config, "fallback_profile", _DEFAULT_FALLBACK_PROFILE
    ))

    # 未注册 fallback 不可执行，统一退回经过验证的最小内联规则集。
    _runtime_profile_override = fallback if fallback in PROFILE_RULES else "inline_v1"
    return runtime_profile_status(settings)


def set_runtime_profile(
    profile: str | None,
    settings: Nl2SqlSettings | None = None,
) -> dict[str, Any]:
    """
    为测试或管理操作设置、清除进程内运行 profile。

    核心功能：允许显式选择已注册规则集，或通过 ``None`` 恢复静态配置；未知 profile
    立即失败，避免静默执行不可审计的修复组合。

    业务逻辑：
        1. 若 profile 为 ``None``，则清除运行时覆盖
        2. 若字符串 profile 已注册，则将其设为覆盖值
        3. 否则抛出参数错误并保持现有配置
        4. 返回最终运行状态

    Args:
        profile: 已注册 profile 名称；``None`` 表示恢复应用配置。

    Returns:
        dict[str, Any]: 设置后实际生效的运行状态。

    Raises:
        ValueError: profile 非空但未在 ``PROFILE_RULES`` 中注册。
    """
    global _runtime_profile_override

    # 分支 1：清除进程内覆盖，让配置文件重新成为 active 的唯一来源。
    if profile is None:
        _runtime_profile_override = None
    elif str(profile) in PROFILE_RULES:
        # 分支 2：仅允许注册表中已知的 profile 进入运行时。
        _runtime_profile_override = str(profile)
    else:
        # 分支 3：未知 profile 拒绝执行，防止规则集合处于未定义状态。
        raise ValueError(f"未注册的语义修复 profile：{profile}")
    return runtime_profile_status(settings)


@dataclass(frozen=True)
class RepairRuleEvidenceV1:
    """描述单条语义修复规则的执行状态、证明及实际改动节点。"""

    rule_code: str
    rule_version: str
    status: str
    proof: dict[str, Any] = field(default_factory=dict)
    changed_ast_nodes: tuple[str, ...] = ()
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        将规则证据转换为隐私安全、可序列化的字典。

        核心功能：保留规则身份、版本、状态与 AST 改动范围，并通过
        ``compact_evidence`` 压缩证明内容；仅在存在原因时输出 reason。

        业务逻辑：
            1. 复制规则代码、版本和执行状态
            2. 对证明字典执行隐私安全压缩
            3. 将不可变 AST 节点元组转换为列表
            4. 若存在跳过或失败原因，则附加原因字段

        Returns:
            dict[str, Any]: 可用于结果契约和遥测的规则证据。
        """
        # 基础字段始终存在，保证消费者可按稳定结构解析规则执行结果。
        output = {
            "rule_code": self.rule_code,
            "rule_version": self.rule_version,
            "status": self.status,
            "proof": compact_evidence(self.proof),
            "changed_ast_nodes": list(self.changed_ast_nodes),
        }

        # 成功应用通常无需 reason；跳过或失败时才输出，减少无意义空字段。
        if self.reason:
            output["reason"] = self.reason
        return output


@dataclass
class SemanticRepairResultV1:
    """汇总一次有界语义修复的输入输出哈希、规则证据、耗时及不变量。"""

    profile: str
    original_sql_hash: str
    repaired_sql_hash: str
    table_set_before: list[str]
    table_set_after: list[str]
    changed: bool = False
    applied_rules: list[RepairRuleEvidenceV1] = field(default_factory=list)
    skipped_rules: list[RepairRuleEvidenceV1] = field(default_factory=list)
    shadow_rules: list[RepairRuleEvidenceV1] = field(default_factory=list)
    attempted_rule_codes: list[str] = field(default_factory=list)
    repair_ms: float = 0.0
    fallback_used: bool = False
    fallback_reason: str | None = None
    invariants: dict[str, bool] = field(default_factory=dict)
    contract_version: str = REPAIR_RESULT_CONTRACT
    registry_version: str = REPAIR_REGISTRY_VERSION
    rule_set_version: str = RULE_SET_VERSION

    def to_dict(self) -> dict[str, Any]:
        """
        将语义修复结果转换为版本化、隐私安全的契约字典。

        核心功能：序列化规则执行证据、表集合、回退状态与不变量，并基于原 SQL
        哈希和尝试规则生成幂等键；结果明确声明从未记录原始 SQL。

        业务逻辑：
            1. 输出契约、注册表和规则集版本
            2. 序列化已应用、已跳过及影子规则证据
            3. 复制修复前后表集合与尝试规则
            4. 生成不包含 SQL 明文的稳定幂等键
            5. 规范化耗时、回退标记和安全不变量
            6. 写入 ``raw_sql_logged=False`` 隐私声明

        Returns:
            dict[str, Any]: 完整的版本化语义修复结果契约。
        """
        # 所有 SQL 身份均使用稳定哈希表达，避免结果或日志携带原始查询文本。
        return {
            "contract_version": self.contract_version,
            "registry_version": self.registry_version,
            "rule_set_version": self.rule_set_version,
            "profile": self.profile,
            "changed": bool(self.changed),
            "original_sql_hash": self.original_sql_hash,
            "repaired_sql_hash": self.repaired_sql_hash,
            "applied_rules": [item.to_dict() for item in self.applied_rules],
            "skipped_rules": [item.to_dict() for item in self.skipped_rules],
            "shadow_rules": [item.to_dict() for item in self.shadow_rules],
            "table_set_before": list(self.table_set_before),
            "table_set_after": list(self.table_set_after),
            "attempted_rule_codes": list(self.attempted_rule_codes),

            # 同一原始 SQL 与相同规则集合共享幂等键，可用于阻断重复修复。
            "idempotence_key": stable_hash(
                f"{self.original_sql_hash}:{','.join(self.attempted_rule_codes)}"
            ),
            "repair_ms": round(float(self.repair_ms), 4),
            "fallback_used": bool(self.fallback_used),
            "fallback_reason": self.fallback_reason,
            "invariants": dict(self.invariants),
            "raw_sql_logged": False,
        }


@dataclass(frozen=True)
class RepairContext:
    """向每条确定性规则提供只读查询语义、表元数据与语义 IR。"""

    query: str
    table_infos: Sequence[Mapping[str, Any]]
    semantic_ir: Mapping[str, Any]


@dataclass(frozen=True)
class RuleOutcome:
    """表示规则候选输出、改动的 AST 节点、证明和未修改原因。"""

    sql: str
    changed_nodes: tuple[str, ...] = ()
    proof: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None


# 所有注册规则必须符合统一签名：输入 SQL 与只读上下文，输出结构化规则结果。
RuleHandler = Callable[[str, RepairContext], RuleOutcome]


@dataclass(frozen=True)
class RegisteredRepairRule:
    """绑定稳定规则代码、规则版本和具体确定性处理函数。"""

    code: str
    version: str
    handler: RuleHandler


def _tables(sql: str) -> list[str]:
    """
    安全提取 SQL 引用的物理表集合。

    核心功能：复用 SQL 安全模块的表名提取逻辑并进行排序；解析失败时返回空列表，
    使修复不变量检查能够以保守结果继续记录失败。

    业务逻辑：
        1. 尝试提取 SQL 中的全部表名
        2. 对表名去重结果排序，确保比较稳定
        3. 若解析异常，则返回空列表

    Args:
        sql: 待提取表名的 SQL。

    Returns:
        list[str]: 排序后的表名列表；失败时为空。
    """
    try:
        return sorted(extract_table_names(sql))
    except Exception:
        # 表名分析属于证据收集，不应因解析异常在此处中断整个回退流程。
        return []


def _filter_literals(sql: str) -> list[str]:
    """
    提取并哈希 WHERE/HAVING 中的字面量，用于验证过滤语义未被修改。

    核心功能：解析 MySQL AST，仅扫描每个 SELECT 的 WHERE 和 HAVING 子句，
    将其中字面量序列化后稳定哈希；LIMIT 值由对应修复规则负责，不纳入本不变量。

    业务逻辑：
        1. 尝试按 MySQL 方言解析 SQL
        2. 若解析失败，则返回空列表
        3. 遍历所有 SELECT 节点
        4. 分别检查 WHERE 与 HAVING 子句
        5. 哈希每个字面量并排序，形成与出现顺序无关的稳定集合

    Args:
        sql: 待检查过滤字面量的 SQL。

    Returns:
        list[str]: 排序后的字面量哈希列表。
    """
    # 解析失败时无法证明过滤字面量，返回空证据交由上层安全检查共同判定。
    try:
        tree = sqlglot.parse_one(sql, read="mysql")
    except Exception:
        return []
    values: list[str] = []

    # 遍历主查询及子查询中的全部 SELECT，确保嵌套过滤条件也受到保护。
    for select in tree.find_all(exp.Select):

        # WHERE 和 HAVING 都可能承载业务阈值，二者必须采用相同的字面量保护。
        for key in ("where", "having"):
            clause = select.args.get(key)

            # 子句存在时哈希其全部字面量，避免在证据中暴露业务值。
            if clause is not None:
                values.extend(
                    stable_hash(literal.sql(dialect="mysql"))
                    for literal in clause.find_all(exp.Literal)
                )
    return sorted(values)


def _join_rule(sql: str, context: RepairContext) -> RuleOutcome:
    """
    修复由唯一元数据主外键关系能够确定的无效 JOIN 条件。

    核心功能：调用语义守卫依据表元数据重写错误连接；仅当存在唯一可信连接边时
    接受改动，并生成不包含完整元数据的证明哈希。

    业务逻辑：
        1. 基于表元数据尝试修复 JOIN
        2. 若 SQL 未变化，则返回不可唯一证明的跳过原因
        3. 若发生变化，则声明改动节点为 ``join.on``
        4. 对表名、字段名及字段角色构成的元数据证明生成稳定哈希

    Args:
        sql: 待修复的 SQL。
        context: 包含表元数据的只读修复上下文。

    Returns:
        RuleOutcome: 修复后的 SQL、改动节点及元数据证明。
    """
    repaired = repair_invalid_metadata_joins(sql, context.table_infos)

    # 无改动表示不存在唯一可证明的无效连接，规则必须保持原 SQL。
    if repaired == sql:
        return RuleOutcome(sql, reason="no_unique_invalid_join")

    # 只记录候选边数量、关系类型和元数据哈希，不复制完整 schema 到结果中。
    return RuleOutcome(
        repaired,
        ("join.on",),
        {
            "candidate_edge_count": 1,
            "relation_type": "metadata_pk_fk_or_unique_id",
            "metadata_proof_hash": stable_hash(
                [
                    {
                        "name": table.get("name"),
                        "columns": [
                            (column.get("name"), column.get("role"))
                            for column in (table.get("columns") or [])
                        ],
                    }
                    for table in context.table_infos
                ]
            ),
        },
    )


def _threshold_projection_rule(sql: str, context: RepairContext) -> RuleOutcome:
    """
    稳定阈值问题中可唯一识别的度量投影表达式。

    核心功能：依据自然语言问题修正阈值度量投影，并从修复后的 AST 中寻找引用
    ``order_amount`` 的聚合表达式，将其哈希作为确定性证明。

    业务逻辑：
        1. 调用语义守卫尝试稳定度量投影
        2. 若 SQL 未变化，则返回无法唯一证明的跳过原因
        3. 解析修复后的 SQL 并定位目标聚合函数
        4. 对聚合 AST 序列化结果生成稳定哈希；解析失败时使用空值哈希
        5. 返回投影节点改动及唯一计数证明

    Args:
        sql: 待修复的 SQL。
        context: 包含原始自然语言查询的只读修复上下文。

    Returns:
        RuleOutcome: 修复后的 SQL、投影改动节点及聚合证明。
    """
    repaired = stabilize_threshold_measure_projection(sql, context.query)

    # 无唯一度量证据时不得猜测投影表达式，保持输入 SQL 不变。
    if repaired == sql:
        return RuleOutcome(sql, reason="threshold_measure_not_uniquely_proven")

    # 从修复结果重新解析聚合 AST，为“确实修复了目标度量”生成可审计证明。
    try:
        tree = sqlglot.parse_one(repaired, read="mysql")
        aggregate = next(
            (
                item
                for item in tree.find_all(exp.AggFunc)
                if any(
                    column.name.lower() == "order_amount"
                    for column in item.find_all(exp.Column)
                )
            ),
            None,
        )

        # 找到聚合时哈希其规范 SQL；未找到时仍生成稳定空哈希供异常审计。
        aggregate_hash = stable_hash(
            aggregate.sql(dialect="mysql") if aggregate is not None else ""
        )
    except Exception:
        # 证明提取失败不撤销已生成的候选修复，但以空哈希明确证据不足。
        aggregate_hash = stable_hash("")
    return RuleOutcome(
        repaired,
        ("select.projection",),
        {
            "aggregate_ast_hash": aggregate_hash,
            "expected_count": 1,
            "actual_count": 1,
        },
    )


def _rank_limit_tie_breaker_rule(sql: str, context: RepairContext) -> RuleOutcome:
    """Stabilize only already-bounded aggregate rankings at cutoff ties."""
    repaired = stabilize_rank_limit_ties(sql, context.query)
    if repaired == sql:
        return RuleOutcome(sql, reason="rank_limit_tie_breaker_not_proven")
    try:
        tree = sqlglot.parse_one(repaired, read="mysql")
        select = _final_select(tree)
        order = select.args.get("order") if select is not None else None
        ordered = [
            item.sql(dialect="mysql")
            for item in (order.expressions if order is not None else [])
        ]
    except Exception:
        ordered = []
    return RuleOutcome(
        repaired,
        ("select.order",),
        {
            "stable_cutoff": True,
            "order_expression_count": len(ordered),
            "order_ast_hash": stable_hash(ordered),
        },
    )


def _final_select(tree: exp.Expression) -> exp.Select | None:
    """
    获取 SQL AST 中代表最终输出的 SELECT 节点。

    核心功能：若根节点本身为 SELECT 则直接返回，否则查找第一个 SELECT，
    为 UNION 包装、CTE 等非 SELECT 根表达式提供统一访问入口。

    业务逻辑：
        1. 判断根表达式是否为 SELECT
        2. 若是则直接返回根节点
        3. 否则从树中查找 SELECT；不存在时返回 ``None``

    Args:
        tree: 已解析的 sqlglot AST 根表达式。

    Returns:
        exp.Select | None: 最终输出 SELECT，无法定位时为 ``None``。
    """
    return tree if isinstance(tree, exp.Select) else tree.find(exp.Select)


def _unique_ir_limit_order_rule(sql: str, context: RepairContext) -> RuleOutcome:
    """
    根据编译器支持的语义 IR 唯一补全缺失的 ORDER BY 和 LIMIT。

    核心功能：仅在 IR 明确、目标已出现在最终投影且不存在窗口排序歧义时补充排序；
    仅在 LIMIT 可解析并位于安全范围时补充条数限制，任何冲突或歧义都会关闭式跳过。

    业务逻辑：
        1. 校验语义 IR 是否由受支持编译器生成
        2. 解析 SQL 并定位最终 SELECT，拒绝窗口函数带来的排序歧义
        3. 若仅有一个期望排序且 SQL 尚未排序，则验证目标已投影后补全 ORDER BY
        4. 若 SQL 已排序，则只记录实际排序项数量
        5. 若 IR 声明 LIMIT 且 SQL 缺失，则校验整数类型和安全边界后补全
        6. 若 SQL 已有 LIMIT，则拒绝与 IR 冲突的值
        7. 若没有实际改动则返回跳过原因，否则序列化修复后的 MySQL SQL

    Args:
        sql: 待补全排序与限制的 SQL。
        context: 包含编译器语义 IR 的只读修复上下文。

    Returns:
        RuleOutcome: 修复结果、实际改动节点和 IR 证明。

    Raises:
        sqlglot.errors.ParseError: SQL 无法按 MySQL 方言解析时向上抛出，由执行器回退。
    """
    # =============================================================================
    # 阶段 1：验证编译器 IR 与 SQL 结构是否具备确定性修复条件
    # =============================================================================

    ir = dict(context.semantic_ir or {})

    # 仅信任明确标记 compiler.supported 的 IR，避免使用模型自由生成的模糊结构。
    if not (ir.get("compiler") or {}).get("supported"):
        return RuleOutcome(sql, reason="semantic_ir_not_compiler_supported")

    tree = sqlglot.parse_one(sql, read="mysql")
    select = _final_select(tree)

    # 无最终 SELECT 时无法安全定位 ORDER BY/LIMIT 的归属节点。
    if select is None:
        return RuleOutcome(sql, reason="missing_final_select")

    # 窗口函数具有独立排序语义，自动增加顶层排序可能改变结果解释，因此拒绝修复。
    if next(tree.find_all(exp.Window), None) is not None:
        return RuleOutcome(sql, reason="window_order_ambiguous")

    # changed_nodes 记录本规则实际触碰的 AST 路径，proof 收集可审计的唯一性证据。
    changed_nodes: list[str] = []
    proof: dict[str, Any] = {"compiler_supported": True}

    # =============================================================================
    # 阶段 2：根据唯一 IR 排序声明补全或核对 ORDER BY
    # =============================================================================

    # 只保留映射型排序项，过滤残缺或旧版本 IR 中的异常数据。
    expected_order = [
        item for item in (ir.get("order_by") or []) if isinstance(item, Mapping)
    ]
    current_order = select.args.get("order")

    # 多个 IR 排序候选无法证明哪个是 LIMIT 的主排序语义，必须整体拒绝修复。
    if expected_order and len(expected_order) != 1:
        return RuleOutcome(sql, reason="ir_order_not_unique")

    expected_target = (
        str(expected_order[0].get("target") or "").strip()
        if expected_order else ""
    )
    expected_direction = (
        str(expected_order[0].get("direction") or "ASC").upper()
        if expected_order else ""
    )
    if expected_order and expected_direction not in {"ASC", "DESC"}:
        return RuleOutcome(sql, reason="ir_order_direction_invalid")

    # SQL 缺少排序且 IR 只有一个候选时，才具备唯一可修复性。
    if current_order is None and len(expected_order) == 1:
        # 收集最终 SELECT 中显式别名和实际字段，限定排序目标必须已在输出中。
        projection_aliases = {
            str(item.alias_or_name or "").lower()
            for item in select.expressions
            if item.alias_or_name
        }
        projected_columns = {
            str(column.name or "").lower()
            for item in select.expressions
            for column in item.find_all(exp.Column)
        }

        # 排序目标已唯一出现在投影时，使用字段引用构造 ASC/DESC 排序节点。
        if (
            expected_target
            and expected_target.lower() in projection_aliases | projected_columns
        ):
            order_expression = exp.column(expected_target)
            select.set(
                "order",
                exp.Order(
                    expressions=[
                        exp.Ordered(
                            this=order_expression,
                            desc=expected_direction == "DESC",
                        )
                    ]
                ),
            )
            changed_nodes.append("select.order")
            proof["direction"] = expected_direction
            proof["ir_proof_hash"] = stable_hash(expected_order[0])
        elif expected_target:
            # IR 指定了目标但最终输出中不存在该字段时，拒绝引入不可见或歧义排序。
            return RuleOutcome(sql, reason="ir_order_target_not_unique_output")
    elif current_order is not None and expected_order:
        # SQL 已有排序时，必须先证明首要目标和方向与 IR 完全一致；否则补 LIMIT
        # 会把上游错误排序固化成貌似闭合的 Top-N 结果。
        actual_items = list(current_order.expressions or [])
        if not actual_items or not isinstance(actual_items[0], exp.Ordered):
            return RuleOutcome(sql, reason="existing_order_not_canonical")
        primary = actual_items[0]
        primary_expression = primary.this
        actual_target = str(
            getattr(primary_expression, "alias_or_name", "")
            or getattr(primary_expression, "name", "")
            or ""
        ).strip()
        actual_direction = "DESC" if primary.args.get("desc") is True else "ASC"
        if actual_target.lower() != expected_target.lower():
            return RuleOutcome(sql, reason="existing_order_target_conflicts_with_ir")
        if actual_direction != expected_direction:
            return RuleOutcome(sql, reason="existing_order_direction_conflicts_with_ir")
        proof["actual_count"] = len(current_order.expressions)
        proof["direction"] = actual_direction
        proof["ir_proof_hash"] = stable_hash(expected_order[0])

    # =============================================================================
    # 阶段 3：校验并补全 IR 中声明的 LIMIT
    # =============================================================================

    expected_limit = ir.get("limit")
    current_limit = select.args.get("limit")

    # 仅在 IR 有明确值且 SQL 尚无限制时尝试写入 LIMIT。
    if expected_limit is not None and current_limit is None:
        try:
            limit_value = int(expected_limit)
        except (TypeError, ValueError):
            # 非整数 IR 无法形成合法 LIMIT，保持原 SQL 并返回稳定原因。
            return RuleOutcome(sql, reason="ir_limit_not_integer")

        # 限制值必须为正且不超过硬上限，避免语义修复放大查询成本。
        if not 0 < limit_value <= 100000:
            return RuleOutcome(sql, reason="ir_limit_out_of_bounds")
        select.set("limit", exp.Limit(expression=exp.Literal.number(limit_value)))
        changed_nodes.append("select.limit")
        proof["expected_limit"] = limit_value
        proof["actual_limit"] = limit_value
    elif current_limit is not None and expected_limit is not None:
        # 已有 LIMIT 时提取字面量；复杂表达式按 None 处理并与 IR 判定为冲突。
        actual = (
            current_limit.expression.this
            if isinstance(current_limit.expression, exp.Literal)
            else None
        )

        # 现有值与 IR 不一致时不得静默覆盖，以免改变原 SQL 的结果规模。
        if str(actual) != str(expected_limit):
            return RuleOutcome(sql, reason="existing_limit_conflicts_with_ir")

    # =============================================================================
    # 阶段 4：确认实际改动并生成规则结果
    # =============================================================================

    # ORDER BY 与 LIMIT 均已闭合或未声明时，本规则没有可执行改动。
    if not changed_nodes:
        return RuleOutcome(sql, reason="ir_limit_order_already_closed_or_absent")

    # 使用固定 MySQL 方言序列化 AST，并将改动节点固化为不可变元组。
    return RuleOutcome(
        tree.sql(dialect="mysql"),
        tuple(changed_nodes),
        proof,
    )


def _shadow_group_by_rule(sql: str, context: RepairContext) -> RuleOutcome:
    """
    以影子模式评估能否根据唯一语义维度安全补全 GROUP BY。

    核心功能：要求编译器 IR 只声明一个维度、SQL 含聚合且无窗口函数或 DISTINCT，
    同时该维度必须已唯一投影；满足全部条件后构造 GROUP BY，但当前仅用于观测。

    业务逻辑：
        1. 从语义 IR 提取映射型维度
        2. 拒绝非编译器支持或维度数量不唯一的请求
        3. 解析 SQL 并要求最终 SELECT 存在且尚无 GROUP BY
        4. 拒绝无聚合、包含窗口函数或 DISTINCT 的高歧义查询
        5. 验证唯一维度已作为非聚合字段投影
        6. 要求目标字段在非聚合投影中恰好出现一次
        7. 构造 GROUP BY 并返回 IR 哈希与计数证明

    Args:
        sql: 待评估的 SQL。
        context: 包含编译器语义 IR 的只读修复上下文。

    Returns:
        RuleOutcome: 影子修复候选或关闭式跳过原因。

    Raises:
        sqlglot.errors.ParseError: SQL 解析失败时向上抛出，由影子执行器记录失败。
    """
    ir = dict(context.semantic_ir or {})

    # 过滤异常维度项，只使用结构明确的映射型声明。
    dimensions = [
        item for item in (ir.get("dimensions") or []) if isinstance(item, Mapping)
    ]

    # 只有编译器支持且维度恰好唯一时，才允许尝试自动分组。
    if not (ir.get("compiler") or {}).get("supported") or len(dimensions) != 1:
        return RuleOutcome(sql, reason="group_dimension_not_unique")

    tree = sqlglot.parse_one(sql, read="mysql")
    select = _final_select(tree)

    # 缺少最终 SELECT 或查询已有 GROUP BY 时无需也不能追加分组。
    if select is None or select.args.get("group") is not None:
        return RuleOutcome(sql, reason="group_exists_or_select_missing")

    # 自动分组只适用于普通聚合：无聚合没有必要，窗口或 DISTINCT 会引入语义歧义。
    if (
        next(select.find_all(exp.AggFunc), None) is None
        or next(tree.find_all(exp.Window), None) is not None
        or bool(select.args.get("distinct"))
    ):
        return RuleOutcome(sql, reason="aggregate_group_shadow_not_safe")

    # 取唯一 IR 维度字段，并收集最终投影中所有非聚合字段名。
    column_name = str(dimensions[0].get("column") or "").lower()
    projected = {
        str(column.name or "").lower()
        for projection in select.expressions
        if projection.find(exp.AggFunc) is None
        for column in projection.find_all(exp.Column)
    }

    # 维度尚未投影时，增加 GROUP BY 会产生与输出不一致的隐藏语义，必须跳过。
    if column_name not in projected:
        return RuleOutcome(sql, reason="dimension_not_already_projected")

    # 复制所有匹配的字段 AST，避免将原投影节点直接复用到 GROUP BY 后产生父子冲突。
    matches = [
        column.copy()
        for projection in select.expressions
        if projection.find(exp.AggFunc) is None
        for column in projection.find_all(exp.Column)
        if str(column.name or "").lower() == column_name
    ]

    # 同名字段必须唯一，否则无法判断应按哪个表或表达式分组。
    if len(matches) != 1:
        return RuleOutcome(sql, reason="dimension_projection_ambiguous")

    # 条件全部满足后构造影子 GROUP BY；是否真正应用由上层运行策略决定。
    select.set("group", exp.Group(expressions=[matches[0]]))
    return RuleOutcome(
        tree.sql(dialect="mysql"),
        ("select.group",),
        {
            "expected_columns": [column_name],
            "expected_count": 1,
            "actual_count": 1,
            "ir_proof_hash": stable_hash(dimensions[0]),
        },
    )


# 注册表将稳定代码映射到具体版本与处理器；group-by 规则保持 shadow 版本，
# 不进入任何 active profile，仅计算“若应用会怎样”的观测证据。
RULE_REGISTRY = {
    "join.unique_metadata_edge": RegisteredRepairRule(
        "join.unique_metadata_edge", "1", _join_rule
    ),
    "projection.threshold_measure": RegisteredRepairRule(
        "projection.threshold_measure", "1", _threshold_projection_rule
    ),
    "order.rank_limit_tie_breaker": RegisteredRepairRule(
        "order.rank_limit_tie_breaker", "1", _rank_limit_tie_breaker_rule
    ),
    "ir.unique_limit_order": RegisteredRepairRule(
        "ir.unique_limit_order", "1", _unique_ir_limit_order_rule
    ),
    "ir.unique_group_by": RegisteredRepairRule(
        "ir.unique_group_by", "1-shadow", _shadow_group_by_rule
    ),
}


def registry_catalog() -> list[dict[str, Any]]:
    """
    生成语义修复规则注册表的可发现目录。

    核心功能：根据当前 active profile 标记每条规则处于 active、candidate 或 shadow
    模式，并公开稳定代码、版本、可修复性及可能改动的 AST 节点。

    业务逻辑：
        1. 解析当前 active profile
        2. 获取该 profile 启用的规则集合
        3. 遍历全部注册规则
        4. 将 group-by 固定标为 shadow，当前启用规则标为 active，其余标为 candidate
        5. 为每条规则附加确定性修复能力和 AST 改动范围

    Returns:
        list[dict[str, Any]]: 面向管理、调试与文档展示的规则目录。
    """
    active, _, _ = _profile_config()
    active_rules = set(PROFILE_RULES.get(active, ()))

    # 目录只公开规则元数据，不执行处理器，也不接触任何 SQL 内容。
    return [
        {
            "rule_code": rule.code,
            "rule_version": rule.version,
            "mode": (
                "shadow"
                if rule.code == "ir.unique_group_by"
                else "active"
                if rule.code in active_rules
                else "candidate"
            ),
            "repairability": "deterministic",
            "changed_ast_nodes": {
                "join.unique_metadata_edge": ["join.on"],
                "projection.threshold_measure": ["select.projection"],
                "order.rank_limit_tie_breaker": ["select.order"],
                "ir.unique_limit_order": ["select.order", "select.limit"],
                "ir.unique_group_by": ["select.group"],
            }[rule.code],
        }
        for rule in RULE_REGISTRY.values()
    ]


def _execute_profile(
    sql: str,
    context: RepairContext,
    *,
    profile: str,
    attempted_rule_codes: set[str],
) -> tuple[str, list[RepairRuleEvidenceV1], list[RepairRuleEvidenceV1]]:
    """
    按 profile 顺序执行一次确定性修复，并验证每次重写的安全不变量。

    核心功能：依次运行 profile 注册规则，跳过同一 SQL 哈希已尝试过的规则；
    每个产生变化的候选必须保持只读、表集合、过滤字面量及幂等性，全部通过后才应用。

    业务逻辑：
        1. 初始化当前 SQL、已应用证据和已跳过证据
        2. 按 profile 规则顺序遍历处理器
        3. 跳过已在调用方尝试集合中的规则
        4. 执行规则；若 SQL 未变化，则记录跳过原因
        5. 对变化候选校验修复前后只读属性、表集合及过滤字面量
        6. 再次执行同一规则，验证候选具备幂等性
        7. 若任一不变量失败，则拒绝候选；否则更新当前 SQL 并记录应用证据

    Args:
        sql: 本轮修复的原始 SQL。
        context: 共享的只读修复上下文。
        profile: 本轮执行的已注册 profile。
        attempted_rule_codes: 相同 SQL 哈希下已经尝试过的规则代码集合。

    Returns:
        tuple[str, list[RepairRuleEvidenceV1], list[RepairRuleEvidenceV1]]:
        最终 SQL、已应用规则证据和已跳过/失败规则证据。

    Raises:
        Exception: 规则执行、SQL 安全检查或 AST 解析失败时向上抛出，由 profile 回退处理。
    """
    # current 会在规则安全应用后逐步更新，后续规则基于前一规则的合法结果继续执行。
    current = sql
    applied: list[RepairRuleEvidenceV1] = []
    skipped: list[RepairRuleEvidenceV1] = []

    # 按 profile 声明顺序遍历规则；未知 profile 关闭式回退到最小 inline_v1。
    for code in PROFILE_RULES.get(profile, PROFILE_RULES["inline_v1"]):
        rule = RULE_REGISTRY[code]

        # 同一 SQL 哈希下已经尝试过的规则不再执行，防止跨节点重复修复循环。
        if code in attempted_rule_codes:
            skipped.append(RepairRuleEvidenceV1(
                code, rule.version, "skipped", reason="already_attempted_for_sql_hash"
            ))
            continue

        # 保存规则执行前 SQL，既用于变化检测，也用于后续安全不变量对比。
        before = current
        outcome = rule.handler(current, context)

        # 规则未生成变化时记录其证明与稳定原因，然后继续尝试下一条规则。
        if outcome.sql == before:
            skipped.append(RepairRuleEvidenceV1(
                code, rule.version, "skipped",
                proof=outcome.proof,
                reason=outcome.reason or "no_change",
            ))
            continue

        # 每次确定性重写都必须保持只读契约、物理表集合和非目标 WHERE/HAVING
        # 字面量；任一不变量无法证明时都不能接受候选 SQL。
        check_sql_read(before)
        check_sql_read(outcome.sql)
        table_set_unchanged = _tables(before) == _tables(outcome.sql)
        filter_literals_unchanged = _filter_literals(before) == _filter_literals(outcome.sql)

        # 对修复结果再次运行同一规则；第二次不再变化才满足幂等性要求。
        second = rule.handler(outcome.sql, context)
        idempotent = second.sql == outcome.sql

        # 合并规则自身证明和运行时不变量证明，形成可解释的完整证据。
        proof = {
            **outcome.proof,
            "table_set_unchanged": table_set_unchanged,
            "filter_literals_unchanged": filter_literals_unchanged,
            "read_only_preserved": True,
            "idempotent": idempotent,
            "changed_ast_nodes": list(outcome.changed_nodes),
        }

        # 表集合、过滤字面量或幂等性任一失败时拒绝更新 current，并记录失败证据。
        if not (table_set_unchanged and filter_literals_unchanged and idempotent):
            skipped.append(RepairRuleEvidenceV1(
                code, rule.version, "failed_invariant",
                proof=proof,
                reason="repair_invariant_failed",
            ))
            continue

        # 所有不变量通过后才提交本次重写，供 profile 中下一条规则继续处理。
        current = outcome.sql
        applied.append(RepairRuleEvidenceV1(
            code, rule.version, "applied",
            proof=proof,
            changed_ast_nodes=outcome.changed_nodes,
        ))
    return current, applied, skipped


def apply_semantic_repairs(
    sql: str,
    *,
    query: str = "",
    table_infos: Sequence[Mapping[str, Any]] | None = None,
    semantic_ir: Mapping[str, Any] | None = None,
    profile: str | None = None,
    attempted_rule_codes: Sequence[str] | None = None,
    settings: Nl2SqlSettings | None = None,
) -> tuple[str, SemanticRepairResultV1]:
    """
    对 SQL 执行恰好一轮有界、版本化且具备安全回退的语义修复。

    核心功能：选择合法运行 profile，构造只读修复上下文并执行注册规则；active
    profile 失败时回退到兼容的基础 profile，回退仍失败则保留原 SQL；随后独立运行
    group-by 影子规则，并输出包含哈希、规则证据、耗时和安全不变量的结果契约。

    业务逻辑：
        1. 记录起始时间并解析 active、fallback 与调用方指定的 profile
        2. 规范化 SQL、表集合、语义上下文及历史尝试规则
        3. 执行所选 profile；若异常则尝试 fallback profile
        4. 若 fallback 也异常，则保留原 SQL 并生成注册表执行失败证据
        5. 对修复结果运行 group-by 影子规则，仅记录其潜在效果
        6. 计算修复后表集合、SQL 哈希、耗时及全局安全不变量
        7. 返回最终 SQL 与版本化 ``SemanticRepairResultV1``

    Args:
        sql: 待修复的 SQL 文本。
        query: 原始自然语言查询，用于阈值投影等语义规则。
        table_infos: 表和字段元数据序列。
        semantic_ir: 可选的编译器语义中间表示。
        profile: 可选的本次请求 profile；未提供时使用当前 active。
        attempted_rule_codes: 相同 SQL 哈希下已尝试的规则代码，用于阻止重复修复。

    Returns:
        tuple[str, SemanticRepairResultV1]: 最终 SQL 与完整修复结果契约对象。
    """
    # =============================================================================
    # 阶段 1：解析运行配置并规范化修复输入
    # =============================================================================

    # 使用高精度单调计时器统计整个 active、fallback 和 shadow 流程耗时。
    started = time.perf_counter()
    active, fallback, _ = _profile_config(settings)
    selected = str(profile or active)

    # 调用方指定未知 profile 时不执行未注册规则，优先回退到配置 fallback。
    if selected not in PROFILE_RULES:
        selected = fallback if fallback in PROFILE_RULES else "inline_v1"

    # 清理 SQL 首尾空白，使哈希、幂等比较和处理器输入保持稳定。
    original = str(sql or "").strip()
    before_tables = _tables(original)

    # 将可选上下文转换为稳定基础类型，所有规则共享同一个不可变上下文对象。
    context = RepairContext(
        query=str(query or ""),
        table_infos=list(table_infos or []),
        semantic_ir=dict(semantic_ir or {}),
    )

    # 历史尝试代码转为字符串集合，支持 O(1) 重复规则检测。
    attempted = {str(item) for item in (attempted_rule_codes or [])}
    fallback_used = False
    fallback_reason = None
    applied: list[RepairRuleEvidenceV1] = []
    skipped: list[RepairRuleEvidenceV1] = []

    # =============================================================================
    # 阶段 2：执行所选 profile 并在异常时逐级回退
    # =============================================================================

    # 首先尝试调用方指定或当前 active profile；规则异常由本层统一切换到 fallback。
    try:
        repaired, applied, skipped = _execute_profile(
            original, context, profile=selected, attempted_rule_codes=attempted
        )
    except Exception as exc:
        # active 与 fallback 不同时标记发生回退，并只记录异常类型避免泄露 SQL。
        fallback_used = selected != fallback
        fallback_reason = type(exc).__name__
        fallback_profile = fallback if fallback in PROFILE_RULES else "inline_v1"

        # 使用原始 SQL 重新执行 fallback，避免在部分应用的中间结果上继续修复。
        try:
            repaired, applied, skipped = _execute_profile(
                original,
                context,
                profile=fallback_profile,
                attempted_rule_codes=attempted,
            )
        except Exception as fallback_exc:
            # fallback 仍失败时保持原 SQL，清空应用证据并生成注册表级失败记录。
            repaired = original
            applied = []
            skipped = [RepairRuleEvidenceV1(
                "registry.execution",
                "1",
                "failed",
                proof={"sql_hash": stable_hash(original)},
                reason=type(fallback_exc).__name__,
            )]
            fallback_used = True
            fallback_reason = type(fallback_exc).__name__

        # 无论 fallback 成功或失败，结果契约都记录实际落到的 profile。
        selected = fallback_profile

    # =============================================================================
    # 阶段 3：独立执行 GROUP BY 影子规则
    # =============================================================================

    # 影子规则永不修改 repaired，仅用于衡量未来启用自动 GROUP BY 的潜在覆盖率。
    shadow_rows: list[RepairRuleEvidenceV1] = []
    shadow_rule = RULE_REGISTRY["ir.unique_group_by"]
    try:
        shadow = shadow_rule.handler(repaired, context)

        # 输出不同表示规则“若启用将应用”；输出相同则记录规则自身的跳过原因。
        shadow_rows.append(RepairRuleEvidenceV1(
            shadow_rule.code,
            shadow_rule.version,
            "would_apply" if shadow.sql != repaired else "skipped",
            proof=shadow.proof,
            changed_ast_nodes=shadow.changed_nodes,
            reason=shadow.reason,
        ))
    except Exception as exc:
        # 影子规则失败不影响主修复结果，只以异常类型记录独立观测证据。
        shadow_rows.append(RepairRuleEvidenceV1(
            shadow_rule.code,
            shadow_rule.version,
            "failed",
            reason=type(exc).__name__,
        ))

    # =============================================================================
    # 阶段 4：计算全局不变量并组装版本化结果
    # =============================================================================

    # 修复后表集合用于最终全局校验，也会以列表形式写入结果契约。
    after_tables = _tables(repaired)
    result = SemanticRepairResultV1(
        profile=selected,
        changed=repaired != original,
        original_sql_hash=stable_hash(original),
        repaired_sql_hash=stable_hash(repaired),
        table_set_before=before_tables,
        table_set_after=after_tables,
        applied_rules=applied,
        skipped_rules=skipped,
        shadow_rules=shadow_rows,

        # 应用和跳过证据共同代表本轮实际尝试过的 active 规则。
        attempted_rule_codes=[
            item.rule_code for item in [*applied, *skipped]
        ],
        repair_ms=(time.perf_counter() - started) * 1000,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,

        # 全局不变量再次核对完整流水线结果；SQL 只读性已在每个候选提交前验证，
        # 此处记录其契约结论而不重复解析。
        invariants={
            "single_bounded_pass": int(getattr(
                settings.semantic_repair if settings is not None else None,
                "max_passes",
                _DEFAULT_MAX_PASSES,
            )) == 1,
            "table_set_unchanged": before_tables == after_tables,
            "filter_literals_unchanged": (
                _filter_literals(original) == _filter_literals(repaired)
            ),
            "read_only_preserved": True,
        },
    )
    return repaired, result


def compact_semantic_repair(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """
    校验并压缩版本化语义修复结果，只输出隐私安全白名单字段。

    核心功能：首先验证结果契约版本，随后限制字符串长度、规则数量与规则字段，
    对证明内容再次执行安全压缩，并强制声明未记录原始 SQL。

    业务逻辑：
        1. 将空输入归一化为空字典
        2. 拒绝契约版本不匹配的旧数据或任意映射
        3. 定义规则证据允许输出的字段白名单
        4. 逐项过滤并压缩 applied、skipped 和 shadow 规则
        5. 限制规则、尝试代码、版本、profile 和错误原因的长度或数量
        6. 规范化布尔值、数值及不变量映射
        7. 返回明确标记 ``raw_sql_logged=False`` 的安全摘要

    Args:
        value: 待压缩的语义修复结果映射。

    Returns:
        dict[str, Any]: 版本合法时返回隐私安全摘要，否则返回空字典。
    """
    # 空值统一转为字典，避免后续契约字段访问处理 None。
    item = dict(value or {})

    # 只接受当前结果契约，防止未知旧版本字段绕过现有隐私白名单。
    if item.get("contract_version") != REPAIR_RESULT_CONTRACT:
        return {}

    # 单条规则仅允许暴露身份、状态、压缩证明、改动节点和稳定原因。
    rule_keys = {
        "rule_code", "rule_version", "status", "proof",
        "changed_ast_nodes", "reason",
    }

    def rules(key: str) -> list[dict[str, Any]]:
        """
        过滤并压缩指定类别的规则证据。

        核心功能：跳过非映射项，只复制白名单中的非空字段，对 proof 再次执行
        ``compact_evidence``，并将单类规则数量限制为十六条。

        业务逻辑：
            1. 初始化安全输出列表
            2. 遍历输入类别下的规则项
            3. 跳过非映射型异常数据
            4. 复制白名单中的非空字段
            5. 强制以安全压缩后的 proof 覆盖原证明
            6. 返回前十六条规则

        Args:
            key: 规则类别字段名，如 ``applied_rules``。

        Returns:
            list[dict[str, Any]]: 数量受限的安全规则证据。
        """
        output = []

        # 遍历调用方指定的规则类别，缺失或空值按空列表处理。
        for row in item.get(key) or []:

            # 非映射项无法按规则契约读取，直接忽略以保持输出结构稳定。
            if not isinstance(row, Mapping):
                continue

            # 仅复制白名单内的非空字段，避免新增上游字段被自动带入遥测。
            safe = {name: row.get(name) for name in rule_keys if row.get(name) is not None}

            # proof 无论原值为何都重新经过隐私压缩，覆盖可能存在的未清洗内容。
            safe["proof"] = compact_evidence(row.get("proof"))
            output.append(safe)

        # 单类最多保留十六条，防止异常输入导致状态或日志无限膨胀。
        return output[:16]

    # 请求级字段同样执行长度、数量和类型约束，且不包含任何 SQL 明文。
    return {
        "contract_version": REPAIR_RESULT_CONTRACT,
        "registry_version": str(item.get("registry_version") or "")[:80],
        "rule_set_version": str(item.get("rule_set_version") or "")[:80],
        "profile": str(item.get("profile") or "")[:80],
        "changed": bool(item.get("changed")),
        "original_sql_hash": str(item.get("original_sql_hash") or "")[:80],
        "repaired_sql_hash": str(item.get("repaired_sql_hash") or "")[:80],
        "applied_rules": rules("applied_rules"),
        "skipped_rules": rules("skipped_rules"),
        "shadow_rules": rules("shadow_rules"),

        # 尝试规则代码最多保留十六项，每项截断为八十字符。
        "attempted_rule_codes": [
            str(code)[:80] for code in (item.get("attempted_rule_codes") or [])[:16]
        ],
        "repair_ms": float(item.get("repair_ms") or 0),
        "fallback_used": bool(item.get("fallback_used")),

        # 回退原因只保留短异常类型；空原因明确规范化为 None。
        "fallback_reason": (
            str(item.get("fallback_reason"))[:80]
            if item.get("fallback_reason")
            else None
        ),

        # 不变量键截断并将值归一化为布尔类型，避免复杂对象进入安全摘要。
        "invariants": {
            str(key)[:80]: bool(value)
            for key, value in dict(item.get("invariants") or {}).items()
        },
        "raw_sql_logged": False,
    }


def semantic_repair_ledger_entry(
    validation: Mapping[str, Any] | None,
    *,
    phase: str,
    committed_at_phase: bool,
) -> dict[str, Any]:
    """Build one bounded, SQL-free record for an actual repair invocation.

    ``committed_at_phase`` means that this invocation's validation evidence was
    committed to graph state at that phase.  The legacy ``selected`` field is
    retained as an equivalent phase-local alias; the evaluator later derives
    the unique final invocation by final SQL hash and ledger order.
    """

    item = dict(validation or {})
    repair = compact_semantic_repair(item.get("semantic_repair"))
    resolved_phase = str(phase or "")[:64]
    return {
        "schema_version": SEMANTIC_REPAIR_LEDGER_SCHEMA,
        "phase": resolved_phase,
        "committed_at_phase": bool(committed_at_phase),
        "selected": bool(committed_at_phase),
        "validation_valid": item.get("valid") is True,
        "contract_version": str(repair.get("contract_version") or "")[:80],
        "registry_version": str(repair.get("registry_version") or "")[:80],
        "rule_set_version": str(repair.get("rule_set_version") or "")[:80],
        "profile": str(repair.get("profile") or "")[:80],
        "original_sql_hash": str(repair.get("original_sql_hash") or "")[:80],
        "repaired_sql_hash": str(repair.get("repaired_sql_hash") or "")[:80],
        "observed_sql_hash": stable_hash(str(item.get("sql") or "").strip()),
        "attempted_rule_codes": list(
            repair.get("attempted_rule_codes") or []
        )[:16],
        "applied_rules": list(repair.get("applied_rules") or [])[:16],
        "skipped_rules": list(repair.get("skipped_rules") or [])[:16],
        "fallback_used": bool(repair.get("fallback_used")),
        "fallback_reason": repair.get("fallback_reason"),
        "invariants": dict(repair.get("invariants") or {}),
        "raw_sql_logged": False,
    }
