"""为 NL2SQL 提示词构建可由历史 SQL 与元数据确定性推导的业务语义约束。"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import sqlglot
from sqlglot import exp

from app.agent_system.agents.nl2sql.semantic_contract import (
    build_query_contract,
    render_query_contract,
)
from app.agent_system.agents.nl2sql.services.semantic_guard import (
    _entity_listing_requested,
    _matching_display_columns,
    explicit_grouping_entries,
)


# 这些语言模式用于识别省略式追问，触发对上一轮分组、过滤、排序与限制的继承。
_FOLLOW_UP_RE = re.compile(r"(?:^那|^那么|呢$|改成|换成|再看|再算|销售额呢|销量呢)")

# 时间粒度按“月 → 季度 → 年”的优先级匹配，每项同时包含中文与英文元数据别名。
_TIME_GRAINS = (
    ("月", ("月", "月份", "month")),
    ("季度", ("季度", "quarter")),
    ("年", ("年份", "year")),
)


def _column_terms(column: Mapping) -> str:
    """
    将字段名称、描述和别名合并为统一的小写检索文本。

    核心功能：从字段元数据中提取所有可用于自然语言匹配的词项，过滤空值后以空格
    拼接并转为小写，供实体展示列和时间粒度识别复用。

    业务逻辑：
        1. 读取字段名和描述
        2. 展开字段别名列表
        3. 过滤空值并将剩余值转换为字符串
        4. 以空格拼接并统一转为小写

    Args:
        column: 单个字段的元数据映射。

    Returns:
        str: 可用于不区分大小写匹配的字段词项文本。
    """
    return " ".join(
        str(value)
        for value in (
            column.get("name", ""),
            column.get("description", ""),
            *(column.get("alias") or []),
        )
        if value
    ).lower()


def _table_terms(table: Mapping) -> str:
    """
    将表名称和描述合并为统一的小写检索文本。

    核心功能：规范化表级元数据，使商品、客户、地区等自然语言实体能够与候选
    维度表进行中英文关键词匹配。

    业务逻辑：
        1. 读取表名和表描述
        2. 过滤空值并转换为字符串
        3. 以空格拼接后统一转为小写

    Args:
        table: 单个表的元数据映射。

    Returns:
        str: 可用于实体表匹配的规范化词项文本。
    """
    return " ".join(
        str(value) for value in (table.get("name", ""), table.get("description", "")) if value
    ).lower()


def _history_contract(query: str, chat_history: Sequence[Mapping]) -> str | None:
    """
    为省略式追问生成必须继承上一轮 SQL 结构的语义约束。

    核心功能：只有当前问题符合追问语言模式且历史中存在 SQL 时才解析上一轮查询，
    提取 GROUP BY、WHERE、ORDER BY 和 LIMIT；解析失败时仍返回保守的整体继承约束。

    业务逻辑：
        1. 检查是否存在历史记录以及当前问题是否为省略式追问
        2. 读取最后一轮 SQL；为空时不生成历史约束
        3. 尝试按 MySQL 方言解析上一轮 SQL
        4. 若解析失败，则要求除明确替换度量外继承全部关键结构
        5. 提取分组、过滤、排序和限制节点
        6. 将实际存在的结构拼接为面向提示词的保留契约

    Args:
        query: 当前轮自然语言查询。
        chat_history: 按时间排序的历史对话记录。

    Returns:
        str | None: 省略式追问约束；不满足触发条件时为 ``None``。
    """
    # 同时满足“有历史”和“语言上像追问”才继承，避免独立问题被上一轮错误污染。
    if not chat_history or not _FOLLOW_UP_RE.search((query or "").strip()):
        return None

    # 仅使用最近一轮 SQL；缺失 SQL 时没有可靠结构可供继承。
    previous_sql = str(chat_history[-1].get("sql") or "").strip()
    if not previous_sql:
        return None

    # 优先解析具体结构；解析失败仍保留一条不依赖 AST 的保守继承要求。
    try:
        tree = sqlglot.parse_one(previous_sql, read="mysql")
    except Exception:
        return "这是承接上一轮的省略追问；除用户明确替换的度量外，保留上一轮维度、过滤、排序和限制。"

    # GROUP BY 可能在主查询或子查询中多次出现，因此收集全部分组节点。
    groups = [node.sql(dialect="mysql") for node in tree.find_all(exp.Group)]
    where = tree.find(exp.Where)
    order = tree.find(exp.Order)
    limit = tree.find(exp.Limit)
    preserved = []

    # 仅把上一轮实际存在的结构写入契约，避免凭空新增约束。
    if groups:
        preserved.append("分组 " + "; ".join(groups))
    if where:
        preserved.append("过滤 " + where.sql(dialect="mysql"))
    if order:
        preserved.append("排序 " + order.sql(dialect="mysql"))
    if limit:
        preserved.append("限制 " + limit.sql(dialect="mysql"))

    # 有具体结构时追加逐项保留清单；否则仍禁止追问退化为全局汇总。
    suffix = "；必须保留：" + "；".join(preserved) if preserved else ""
    return f"这是承接上一轮的省略追问，只替换用户明确提到的度量，不得退化为全局汇总{suffix}。"


def build_semantic_contract(
    query: str,
    table_infos: Sequence[Mapping] | None,
    chat_history: Sequence[Mapping] | None = None,
    metric_infos: Sequence[Mapping] | None = None,
    semantic_ir: Mapping | None = None,
) -> str:
    """
    为单个自然语言问题生成可直接注入提示词的元数据语义契约。

    核心功能：综合省略式追问历史、显式维度、可读实体字段、时间粒度和主外键关系，
    把可确定推导的高价值业务约束写成编号文本；模型仍负责生成 SQL，但不得猜测或
    删除这些已由元数据证明的结构。

    业务逻辑：
        1. 规范化表元数据和问题文本，并尝试生成历史继承约束
        2. 对分组、排名和阈值类问题识别用户明确提及的维度
        3. 对实体类问题推导应优先展示的业务名称字段
        4. 识别月、季度或年粒度，并寻找对应可读时间字段
        5. 根据同名主键/外键推导安全的事实表—维度表连接候选
        6. 若没有额外提示则返回基础安全约束，否则按顺序编号输出全部提示

    Args:
        query: 当前自然语言问题。
        table_infos: 已召回的表及字段元数据。
        chat_history: 可选的历史对话，用于识别省略式追问。

    Returns:
        str: 可直接传入 NL2SQL 提示词的中文语义契约。
    """
    # =============================================================================
    # 阶段 1：规范化输入并继承省略式追问结构
    # =============================================================================

    # 可选表信息转为列表，问题统一小写，所有约束按发现顺序累积到 hints。
    tables = list(table_infos or [])
    query_text = (query or "").lower()
    hints: list[str] = []

    # 仅当语言模式和历史 SQL 均满足条件时，追加上一轮结构继承契约。
    follow_up = _history_contract(query_text, list(chat_history or []))
    if follow_up:
        hints.append(follow_up)

    explicit_years = list(dict.fromkeys(
        int(value)
        for value in re.findall(r"(?:19|20)\d{2}", query_text)
    ))
    if explicit_years:
        year_columns = [
            f"{table.get('name')}.{column.get('name')}"
            for table in tables
            for column in (table.get("columns") or [])
            if str(column.get("role") or "").lower() == "dimension"
            and any(term in _column_terms(column) for term in ("year", "年份"))
        ]
        if year_columns:
            hints.append(
                "显式年份表示完整年度过滤：必须通过 "
                + "、".join(sorted(set(year_columns)))
                + " 约束问题中的全部年份；不能把全年缩成某一个 date_id，"
                "也不能把年份写到不拥有该字段的表别名上。"
            )

    if (
        len(explicit_years) >= 2
        and "分别" in query_text
        and any(token in query_text for token in (
            "销售额", "成交额", "成交金额", "订单金额", "消费金额",
        ))
    ):
        hints.append(
            "同一展示粒度下分别给出多个年份时，使用条件聚合形成宽表，"
            "每个年份一个结果列；年份是过滤/条件轴，不得额外成为最终行粒度。"
        )

    if (
        any(token in query_text for token in ("客户", "用户", "顾客"))
        and any(token in query_text for token in ("累计", "累积", "合计", "总计", "总额"))
        and any(token in query_text for token in ("平均值", "均值", "平均"))
        and any(token in query_text for token in (
            "销售额", "成交额", "成交金额", "订单金额", "消费金额",
        ))
    ):
        hints.append(
            "该指标是实体累计值的平均：先按客户粒度 SUM 金额得到每位客户一行，"
            "再按外层分组维度计算 AVG；不得直接 AVG 事实行或先按外层维度汇总。"
        )

    # =============================================================================
    # 阶段 2：识别分组、排名或阈值问题中的显式维度
    # =============================================================================

    explicit_dimensions = {
        f"{table}.{column}"
        for table, column in explicit_grouping_entries(query_text, tables)
    }
    if explicit_dimensions:
        hints.append(
            "用户明确指定的分组轴必须同时出现在 SELECT 与 GROUP BY："
            + "、".join(sorted(explicit_dimensions))
            + "；物理度量字段也可以是用户明确要求的分桶轴；"
            + "纠错时不得为了通过语法校验而删除这些轴。"
        )

    # =============================================================================
    # 阶段 3：为实体问题推导可读业务展示字段
    # =============================================================================

    # 当问题询问实体且未明确要求 ID 时，依据名称、别名、描述和 is_display 元数据
    # 优先选择可读业务键，避免最终结果只返回内部主键或外键。
    asks_for_entities = _entity_listing_requested(query_text)
    explicitly_requests_id = any(token in query_text for token in ("id", "编号", "标识"))
    if asks_for_entities and not explicitly_requests_id:
        display_names = _matching_display_columns(query_text, tables)
        display_columns = [
            f"{table.get('name')}.{column.get('name')}"
            for table in tables
            for column in (table.get("columns") or [])
            if str(column.get("name") or "").lower() in display_names
        ]

        # 展示字段存在时明确要求同时投影并分组，同时禁止扩展到问题未要求的属性。
        if display_columns:
            hints.append(
                "实体结果必须投影可读业务名称 " + "、".join(sorted(set(display_columns)))
                + "；仅当结果同时包含聚合指标时才把名称放入 GROUP BY，纯目录列表无需分组"
                + "；除非用户明确索要 ID，不得只返回主键/外键 ID；"
                + "不得额外投影问题未提及的品牌、品类等属性。"
            )

    # =============================================================================
    # 阶段 4：匹配时间粒度及可读时间轴字段
    # =============================================================================

    # 按预定义优先级寻找第一个被问题提及的时间粒度。
    for grain_name, terms in _TIME_GRAINS:

        # 当前粒度未在问题中出现时继续检查下一粒度。
        if not any(term in query_text for term in terms):
            continue
        candidates = []

        # 遍历所有表字段，通过字段名、描述和别名寻找对应时间粒度列。
        for table in tables:
            for column in table.get("columns") or []:
                column_terms = _column_terms(column)
                if any(term.lower() in column_terms for term in terms):
                    candidates.append(f"{table.get('name')}.{column.get('name')}")

        # 有元数据候选时要求使用可读时间字段作为投影、分组和排序轴。
        if candidates:
            hints.append(
                f"问题要求{grain_name}粒度，优先使用时间维度中的可读字段 "
                + "、".join(sorted(set(candidates)))
                + " 作为投影、分组和排序轴，不要用日期 ID 算术值代替。"
            )

        # 问题只按优先匹配到的一个粒度生成契约，避免同时追加月/季度/年冲突提示。
        break

    # =============================================================================
    # 阶段 5：根据元数据推导安全的多表连接候选
    # =============================================================================

    # 以规范化字段名索引全部主键，同时收集显式外键及 ``*_id`` 形式的候选外键。
    primary_by_name: dict[str, list[str]] = {}
    foreign_columns: list[tuple[str, str]] = []

    # 遍历表和字段，构建“字段名 → 完全限定主键”及外键候选集合。
    for table in tables:
        for column in table.get("columns") or []:
            name = str(column.get("name") or "")
            role = str(column.get("role") or "").lower()

            # 主键按小写字段名分组，允许同一业务键在多个维度表中分别登记。
            if role == "primary_key":
                primary_by_name.setdefault(name.lower(), []).append(f"{table.get('name')}.{name}")
            elif role == "foreign_key" or name.lower().endswith("_id"):
                # 显式 foreign_key 或命名符合惯例的 ID 字段均作为待匹配外键。
                foreign_columns.append((name.lower(), f"{table.get('name')}.{name}"))
    joins = []

    # 将同名外键与主键组成等值连接，排除字段与自身连接的无效候选。
    for name, foreign in foreign_columns:
        joins.extend(f"{foreign} = {primary}" for primary in primary_by_name.get(name, []) if primary != foreign)

    # 仅在确有连接候选且召回多张表时追加 JOIN 约束。
    if joins and len(tables) > 1:
        hints.append(
            "多表查询必须显式使用关联谓词；元数据可确认的连接包括 "
            + "、".join(sorted(set(joins)))
            + "，禁止逗号连接或 CROSS JOIN。"
        )

    # Retrieval, planning and SQL validation now consume the same structured
    # contract.  Render its high-value parts here so the model cannot infer a
    # different visible grain or denominator merely because the earlier prose
    # hints were underspecified.
    query_contract = build_query_contract(
        query,
        tables,
        metric_infos or [],
        semantic_ir or {},
    )
    if query_contract.get("authority") == "closed":
        contract_heading = (
            "统一结构化语义合同（已由确定性路径闭合；其粒度、口径、关系和"
            "阶段顺序优先于自由推断）"
        )
    else:
        contract_heading = (
            "统一结构化语义线索（guided：编译器已放弃最终语义权威；用户问题中"
            "明确要求的维度、指标、过滤、总体和输出形状优先。下列空缺或冲突"
            "不得作为删除用户要求的依据；注册指标口径与元数据关系仍须遵守）"
        )
    hints.append(contract_heading + "：\n" + render_query_contract(query_contract))

    # =============================================================================
    # 阶段 6：输出基础约束或编号后的完整语义契约
    # =============================================================================

    # 没有可由元数据确定推导的附加提示时，仍返回 Schema、指标和安全基础约束。
    if not hints:
        return "无额外语义约束；仍须严格使用给定 Schema、指标口径和安全规则。"

    # 按发现顺序编号，保持历史、维度、展示、时间和连接约束的稳定优先级。
    return "\n".join(f"{index}. {hint}" for index, hint in enumerate(hints, 1))
