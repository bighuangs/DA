"""定义版本化、隐私安全的语义违规契约，并生成有边界的纠错模型指令。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping


# 违规契约与修复规则分别版本化，使路由消费者能够独立识别数据结构和规则语义。
VIOLATION_CONTRACT_VERSION = "semantic-violation-v1"
REPAIR_RULE_SET_VERSION = "semantic-repair-rules-v1"


def stable_hash(value: Any) -> str:
    """
    将任意值转换为适合证据记录的短稳定哈希，避免暴露原始内容。

    核心功能：以容错 UTF-8 编码规范化输入，计算 SHA-256 后仅保留前二十四位
    十六进制摘要，并加上算法前缀以便下游识别。

    业务逻辑：
        1. 将空值归一化为空字符串，其余值转换为字符串
        2. 使用 UTF-8 编码，异常字符以替代方式处理
        3. 计算 SHA-256 摘要
        4. 截取前二十四位并添加 ``sha256:`` 前缀

    Args:
        value: 需要脱敏标识的任意值。

    Returns:
        str: 固定格式的短 SHA-256 哈希。
    """
    raw = str(value or "").encode("utf-8", errors="replace")
    return "sha256:" + hashlib.sha256(raw).hexdigest()[:24]


def _safe_list(value: Any, *, limit: int = 12) -> list[str]:
    """
    将受支持的序列压缩为数量和长度均有边界的字符串列表。

    核心功能：只接受 list、tuple 或 set，最多保留指定数量的元素，并将每项截断
    为八十个字符，防止任意对象或超长内容进入 Trace 和 SSE。

    业务逻辑：
        1. 检查输入是否为允许的序列类型
        2. 非序列输入返回空列表
        3. 将序列转换为列表并截取前 ``limit`` 项
        4. 把每项转为字符串并限制为八十字符

    Args:
        value: 待压缩的候选序列。
        limit: 最多保留的元素数量，默认十二项。

    Returns:
        list[str]: 长度与单项字符数均受限的安全列表。
    """
    # 映射、字符串和任意可迭代对象均不在白名单内，避免意外展开敏感结构。
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item)[:80] for item in list(value)[:limit]]


def compact_evidence(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """
    按字段白名单压缩用于 SSE、Trace 和报告的结构化违规证据。

    核心功能：将证据字段划分为标量、列表和哈希三类，只复制明确允许的键；
    同时限制类型、列表规模和哈希长度，使 SQL、字面量、问题及无限制元数据无法
    被普通遥测载荷自动带出。

    业务逻辑：
        1. 拒绝非映射输入
        2. 定义可公开的标量、列表及哈希字段白名单
        3. 遍历标量键，仅复制基础类型值
        4. 遍历列表键，使用 ``_safe_list`` 限制数量和长度
        5. 遍历哈希键，仅复制非空值并截断为八十字符
        6. 返回不包含任何白名单外字段的安全证据

    Args:
        value: 原始违规或修复证据映射。

    Returns:
        dict[str, Any]: 可安全写入流式事件、追踪和报告的证据。
    """
    # 非映射内容无法按字段白名单审查，关闭式返回空证据。
    if not isinstance(value, Mapping):
        return {}
    output: dict[str, Any] = {}

    # 标量白名单只包含数量、方向、关系类型、策略与布尔不变量等结构信息。
    scalar_keys = {
        "expected_count", "actual_count", "candidate_edge_count",
        "expected_limit", "actual_limit", "direction", "relation_type",
        "anti_join_strategy", "history_contract", "compiler_supported",
        "table_set_unchanged", "filter_literals_unchanged",
        "read_only_preserved", "idempotent",
    }

    # 列表白名单描述期望/实际字段、表及 AST 改动节点，不允许任意嵌套结构。
    list_keys = {
        "expected_columns", "actual_columns", "missing_columns",
        "expected_tables", "missing_tables", "changed_ast_nodes",
    }

    # 哈希白名单承载元数据、IR、聚合 AST 和 SQL 的不可逆短标识。
    hash_keys = {
        "metadata_proof_hash", "ir_proof_hash", "aggregate_ast_hash",
        "sql_hash", "previous_sql_hash",
    }

    # 遍历标量字段，仅允许字符串、数字、布尔值和 None 等可控基础类型。
    for key in scalar_keys:
        if key in value and isinstance(value[key], (str, int, float, bool, type(None))):
            output[key] = value[key]

    # 遍历列表字段并统一执行元素数量与字符长度约束。
    for key in list_keys:
        if key in value:
            output[key] = _safe_list(value[key])

    # 遍历哈希字段，忽略 None 并限制异常上游值的最大字符数。
    for key in hash_keys:
        if key in value and value[key] is not None:
            output[key] = str(value[key])[:80]
    return output


@dataclass(frozen=True)
class SemanticViolationV1:
    """表示可供路由、聚合和纠错使用的稳定语义违规契约。"""

    violation_code: str
    stage: str
    repairability: str
    repair_hint_code: str
    safe_to_retry: bool
    severity: str = "error"
    evidence: dict[str, Any] = field(default_factory=dict)
    contract_version: str = VIOLATION_CONTRACT_VERSION
    rule_version: str = REPAIR_RULE_SET_VERSION

    def to_dict(self) -> dict[str, Any]:
        """
        将语义违规转换为版本化、隐私安全的字典。

        核心功能：序列化稳定路由字段，并在输出前再次压缩 evidence，确保调用方
        即使传入未清洗证据也不会绕过统一白名单。

        业务逻辑：
            1. 输出契约版本和稳定违规码
            2. 复制阶段、严重程度、可修复性及规则版本
            3. 使用 ``compact_evidence`` 重新清洗证据
            4. 输出修复提示代码
            5. 将重试标记规范化为布尔值

        Returns:
            dict[str, Any]: 可安全传输和持久化的违规契约。
        """
        return {
            "contract_version": self.contract_version,
            "violation_code": self.violation_code,
            "stage": self.stage,
            "severity": self.severity,
            "repairability": self.repairability,
            "rule_version": self.rule_version,
            "evidence": compact_evidence(self.evidence),
            "repair_hint_code": self.repair_hint_code,
            "safe_to_retry": bool(self.safe_to_retry),
        }


class SemanticGuardViolationError(Exception):
    """携带稳定语义违规契约的守卫异常基类。"""

    # 子类通过覆盖这些默认值，将异常映射到稳定的阶段、修复类型和提示代码。
    default_code = "semantic.contract_violation"
    default_stage = "semantic_contract"
    default_repairability = "llm"
    default_hint = "minimal_semantic_correction"

    def __init__(
        self,
        message: str,
        *,
        violation_code: str | None = None,
        stage: str | None = None,
        repairability: str | None = None,
        repair_hint_code: str | None = None,
        safe_to_retry: bool | None = None,
        evidence: Mapping[str, Any] | None = None,
    ):
        """
        创建人类可读异常，同时附加机器可路由的版本化违规对象。

        核心功能：保留异常消息供日志与调试使用，但路由只依赖 ``violation``；
        调用方参数优先于类默认值，重试策略在未显式指定时依据可修复性安全推导。

        业务逻辑：
            1. 使用原始消息初始化 Exception
            2. 为违规码、阶段、可修复性和提示代码应用显式值或类默认值
            3. 若明确传入 safe_to_retry，则采用其布尔值
            4. 否则仅当可修复性不是 ``reject`` 时允许重试
            5. 压缩原始 evidence 并构造不可变违规契约

        Args:
            message: 面向日志和开发者的人类可读异常消息。
            violation_code: 可选的稳定违规路由码。
            stage: 可选的失败阶段标识。
            repairability: 可选的修复方式，如 ``llm``、``upstream`` 或 ``reject``。
            repair_hint_code: 可选的稳定修复提示代码。
            safe_to_retry: 是否允许自动重试；未提供时由 repairability 推导。
            evidence: 可选的结构化证据，将按白名单压缩。
        """
        super().__init__(message)

        # 机器契约与自由文本异常消息解耦，避免调用方依赖不稳定文案进行路由。
        self.violation = SemanticViolationV1(
            violation_code=violation_code or self.default_code,
            stage=stage or self.default_stage,
            repairability=repairability or self.default_repairability,
            repair_hint_code=repair_hint_code or self.default_hint,
            safe_to_retry=(
                bool(safe_to_retry)
                if safe_to_retry is not None
                else (repairability or self.default_repairability) not in {"reject"}
            ),

            # 构造时立即清洗证据，后续 to_dict 仍会二次清洗以形成纵深防护。
            evidence=compact_evidence(evidence),
        )


class SemanticParseError(SemanticGuardViolationError):
    """表示可由模型仅修复 MySQL 语法的解析阶段违规。"""

    default_code = "syntax.parse_error"
    default_stage = "syntax"
    default_repairability = "llm"
    default_hint = "repair_mysql_syntax_only"


class SemanticContractError(SemanticGuardViolationError):
    """表示生成 SQL 违反了由元数据支持的业务语义契约。"""


class SemanticIRError(SemanticContractError):
    """表示 SQL 与语义 IR 不一致、应优先由上游恢复契约的违规。"""

    default_code = "semantic_ir.mismatch"
    default_stage = "semantic_ir"
    default_repairability = "upstream"
    default_hint = "restore_semantic_ir_contract"


def violation_from_exception(exc: Exception) -> SemanticViolationV1:
    """
    将任意 SQL 校验异常归一化为一个版本化语义违规码。

    核心功能：优先复用语义守卫异常自带的契约；对指标公式、成本和安全异常按
    稳定类名映射到专用路由与重试策略，其余数据库或校验错误统一归入模型修复。

    业务逻辑：
        1. 若异常继承语义守卫基类，则直接返回其违规契约
        2. 若为指标公式错误，则标记为上游可修复并允许重试
        3. 若为查询成本错误，则标记为硬拒绝且禁止重试
        4. 若为 SQL 安全错误，则标记为安全硬拒绝且禁止重试
        5. 其余异常统一映射为通用 SQL 校验错误

    Args:
        exc: SQL 解析、语义、安全、成本或数据库校验阶段抛出的异常。

    Returns:
        SemanticViolationV1: 供路由、聚合和纠错决策使用的稳定违规契约。
    """
    # 已结构化的语义守卫异常直接复用，避免丢失具体 evidence 和子类默认策略。
    if isinstance(exc, SemanticGuardViolationError):
        return exc.violation
    name = type(exc).__name__

    # 指标公式不一致应返回上游恢复注册口径，而不是让模型自由选择替代聚合。
    if name == "MetricFormulaError":
        return SemanticViolationV1(
            "metric_formula.mismatch",
            "metric_formula",
            "upstream",
            "restore_registered_metric_formula",
            True,
        )

    # 成本超限属于不可绕过的硬门禁，禁止自动纠错或重试放大查询。
    if name == "QueryCostError":
        return SemanticViolationV1(
            "cost.budget_exceeded",
            "cost_guard",
            "reject",
            "do_not_bypass_cost_guard",
            False,
        )

    # 安全策略违规同样关闭式拒绝，不能向纠错模型提供任何绕过方法。
    if name == "SQLSecurityError":
        return SemanticViolationV1(
            "security.policy_violation",
            "security_policy",
            "reject",
            "do_not_bypass_security_policy",
            False,
        )

    if isinstance(exc, TimeoutError):
        return SemanticViolationV1(
            "sql.validation_deadline_exhausted",
            "sql_validation",
            "reject",
            "reduce_query_scope_or_retry",
            False,
        )

    # 未知异常按通用数据库校验失败处理，只允许模型修复与错误直接相关的部分。
    return SemanticViolationV1(
        "sql.validation_error",
        "sql_validation",
        "llm",
        "repair_database_validation_error",
        True,
    )


# 每个稳定违规码只对应一条最小修改指令；内容明确声明必须保持的业务或安全边界，
# 防止纠错模型借错误修复扩大表、指标、过滤或结果范围。
_DIRECTIVES = {
    "syntax.parse_error": "只修复 MySQL 语法；不得改变表、指标、过滤、分组、排序或限制。",
    "semantic.projection.id_only": "保留实体集合与度量，使用已提供元数据中的唯一展示字段替代仅 ID 展示。",
    "semantic.projection.unrequested_extra": "删除问题未要求的普通展示列；不得删除维度、度量或过滤。",
    "semantic.projection.missing_requested_attribute": (
        "补齐题面明确要求的元数据属性投影；不得添加未要求的展示列、改变过滤或扩大表范围。"
    ),
    "semantic.grouping.missing_dimension": "补齐问题和 Semantic IR 唯一要求的投影与 GROUP BY 维度。",
    "semantic.time.readable_axis_missing": "使用已提供日期维度中的可读时间字段；不得用日期 ID 算术代替。",
    "semantic.time.full_period_filter_required": (
        "把问题中的全部显式年份实现为日期维度 year 的完整年度过滤；"
        "不得以单个 date_id 或 LIMIT 1 日期子查询代替全年。"
    ),
    "semantic.column.wrong_owner": (
        "将字段改为元数据确认的物理所有者别名；若所有者表尚未连接，"
        "仅使用已提供的唯一 PK/FK 关系补齐连接。"
    ),
    "semantic.output.scalar_count_required": (
        "先在内层按被计数实体 GROUP BY 并应用 HAVING 资格条件，"
        "再由无 GROUP BY 的最终 SELECT COUNT(*) 返回一个标量。"
    ),
    "semantic.output.qualified_group_count_structure_required": (
        "构建内层资格集合：按被计数实体 GROUP BY 并应用 HAVING，"
        "然后由最终 SELECT COUNT(*) 计数该集合。"
    ),
    "semantic.output.period_compare_wide_required": (
        "在用户要求的展示粒度上使用条件聚合，每个显式年份生成一个结果列；"
        "不得把 year 添加为最终行粒度。"
    ),
    "semantic.aggregate.entity_total_average_required": (
        "先按 customer_id SUM 金额形成每客户一行，再在外层按用户要求的"
        "展示维度 AVG 该派生累计值。"
    ),
    "semantic.anti_join.missing": "按元数据唯一 PK/FK 使用 LEFT JOIN … IS NULL 或 NOT EXISTS 表达不存在事实。",
    "semantic.anti_join.invalid_key": "只把 anti-join 相关键修正为元数据唯一 PK/FK，不改变保留侧或过滤。",
    "semantic.anti_join.null_unsafe": "使 NOT IN 子查询列具备明确非空约束，或改为等价 NOT EXISTS。",
    "semantic.follow_up.missing_group": "恢复上一轮要求保留的分组结构。",
    "semantic.follow_up.missing_filter": "恢复上一轮要求保留的过滤结构。",
    "semantic.follow_up.missing_order": "恢复上一轮要求保留的排序结构。",
    "semantic.follow_up.missing_limit": "恢复上一轮要求保留的 LIMIT。",
    "semantic_ir.missing_table": "重新依据 Semantic IR 与已召回元数据生成，不得发明表或连接。",
    "semantic_ir.missing_dimension": "恢复 Semantic IR 中唯一的维度投影和分组。",
    "semantic_ir.missing_measure": "恢复 Semantic IR 指定的聚合函数、字段与注册口径。",
    "semantic_ir.order_mismatch": "恢复 Semantic IR 指定的唯一排序目标和方向。",
    "semantic_ir.limit_mismatch": "恢复 Semantic IR 指定的 LIMIT，不得扩大结果范围。",
    "metric_formula.mismatch": "严格使用已注册指标公式；不得自行选择替代聚合。",
}


def correction_directive(violation: Mapping[str, Any] | None) -> dict[str, Any]:
    """
    根据语义违规契约生成范围受控的纠错模型指令。

    核心功能：提取稳定违规码与可修复性，为已知错误选择专用最小修改说明；
    对 ``reject`` 类型只声明禁止纠错和绕过，不向模型暴露任何安全或成本门禁方案。

    业务逻辑：
        1. 将空违规对象归一化为空字典
        2. 提取违规码和可修复性，并应用通用默认值
        3. 若可修复性为 ``reject``，则生成禁止自动纠错的硬门禁指令
        4. 否则按违规码查找专用指令；未知代码使用通用最小修改指令
        5. 组装版本、路由、提示代码、重试标记和指令文本

    Args:
        violation: 版本化语义违规映射；允许为空或仅包含部分字段。

    Returns:
        dict[str, Any]: 可安全注入校正模型提示词的有界纠错指令。
    """
    # 缺失违规信息时仍构建通用 SQL 校验纠错策略，保持下游输入结构稳定。
    item = dict(violation or {})
    code = str(item.get("violation_code") or "sql.validation_error")
    repairability = str(item.get("repairability") or "llm")

    # 安全与成本拒绝类型不提供绕过配方，只允许维持硬门禁。
    if repairability == "reject":
        instruction = "禁止自动纠错；保持安全、权限和成本硬门禁，不得尝试绕过。"
    else:
        # 可修复错误优先使用专用最小指令，未知错误则采用保持全部边界的通用策略。
        instruction = _DIRECTIVES.get(
            code,
            "只进行与该错误直接相关的最小修改，并完整保持业务语义与安全边界。",
        )

    # 输出字段均为稳定路由信息，不包含异常消息、SQL、问题或原始证据。
    return {
        "contract_version": "semantic-correction-directive-v1",
        "violation_code": code,
        "repairability": repairability,
        "repair_hint_code": str(item.get("repair_hint_code") or "minimal_semantic_correction"),
        "safe_to_retry": bool(item.get("safe_to_retry", repairability != "reject")),
        "instruction": instruction,
    }
