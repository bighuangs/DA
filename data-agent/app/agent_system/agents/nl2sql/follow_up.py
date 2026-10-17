"""提供确定性的多轮追问闭合、隐私安全语义状态提取及继承契约校验。"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import sqlglot
from sqlglot import exp

from app.agent_system.agents.nl2sql.services.semantic_guard import SemanticContractError
from app.domain.query.chart_specs import explicit_chart_type
from app.domain.query.follow_up_semantics import (
    FOLLOW_UP_VERSION,
    semantic_state_from_sql as semantic_state_from_sql,
)

# 从 Domain 重导出版本与 SQL 语义提取器，兼容现有 Agent 调用路径。


class FollowUpContractError(SemanticContractError):
    """表示生成 SQL 丢失了追问必须继承的结构化语义槽位。"""


# 追问提示覆盖承接、替换、追加、删除和省略问句等表达，用于判断是否需要继承历史。
_FOLLOW_UP_CUE = re.compile(
    r"^(?:那|那么|然后|再|其中|改成|换成|改看|只看|仅看|只统计|"
    r"只保留|只留下|限定|筛选|也按|取消|不要|不再|同期|拿|"
    r"仅保留|顺序|排序|横向|另外|倒过来|继续|接着|同样|照旧|"
    r"保持|沿用|在此基础上|基于刚才|除此之外)|(?:呢|如何|怎么样)$|"
    r"^(?:去掉|排除)(?=.{0,20}(?:限制|条件|筛选)$)|"
    r"^同时(?=(?:把|将|也|再|增加|加上|列出|展示))|"
    r"(?:改成|换成|再按|继续按|改按|改为|调整为|顺序反过来|排序方向反过来|倒过来)"
)

# 重置提示显式切断历史语义，后续文本必须能独立构成一个新问题。
_RESET_CUE = re.compile(r"^(?:新问题|换个问题|重新开始|不考虑前面|忽略前面)[：,:，\s]*")

# “这些/那些”可能指向已截断或脱敏的结果行，不能作为确定性历史继承依据。
_UNSAFE_RESULT_REFERENCE = re.compile(r"(?:这些|那些|上述|上面的)(?:客户|商品|产品|地区|订单|记录|数据)")
_SINGULAR_RESULT_REFERENCE = re.compile(
    r"(?:该|这个)(?P<entity>地区|区域|大区|客户|用户|商品|产品|品牌)"
)
_REFERENCE_ENTITY_ALIASES = {
    "地区": ("地区", "区域", "大区"),
    "区域": ("地区", "区域", "大区"),
    "大区": ("地区", "区域", "大区"),
    "客户": ("客户", "用户"),
    "用户": ("客户", "用户"),
    "商品": ("商品", "产品"),
    "产品": ("商品", "产品"),
    "品牌": ("品牌",),
}
_REGION_REFERENCE_COLUMNS = frozenset(
    {"region", "region_name", "area", "area_name", "sales_region", "地区", "区域", "大区"}
)

# 指标词典将常见同义表达归一到稳定业务指标名称，支持追问中的指标替换。
_MEASURE_TERMS = {
    "销售额": (
        "销售总额", "销售金额", "交易金额", "成交金额", "销售额",
        "成交额", "交易额", "金额",
    ),
    "销量": ("销售数量", "售出件数", "卖出件数", "销量", "售出数量", "件数"),
    "订单数": (
        "订单数量", "订单总数", "订单笔数", "交易笔数", "订单数",
        "下单次数",
    ),
    "客户数": (
        "不同的购买顾客", "不同的下单顾客", "不同的顾客",
        "不同购买客户", "不同下单客户", "不同顾客数", "不同客户数",
        "下单顾客数", "购买顾客数", "客户数量", "客户总数",
        "客户数", "顾客数", "用户数",
    ),
    "复购率": ("复购率",),
}

# 维度词典用于识别并规范化分组维度的替换或追加操作。
_DIMENSION_TERMS = {
    "品类": ("商品品类", "产品品类", "品类", "类别"),
    "地区": ("销售地区", "区域", "大区", "地区"),
    "品牌": ("商品品牌", "品牌"),
    "商品": ("商品名称", "产品名称", "商品", "产品"),
    "客户": ("客户名称", "客户", "用户"),
    "性别": ("性别", "男女"),
    "月份": ("月份", "每月", "按月", "月度"),
    "季度": ("季度", "每季", "按季"),
    "年份": ("年份", "每年", "按年", "年度"),
}


def _first_term(text: str, groups: Mapping[str, Sequence[str]]) -> tuple[str, str] | None:
    """
    在文本中寻找词典命中的最长同义词及其规范名称。

    核心功能：展开“规范名称 → 同义词序列”映射，收集所有出现在文本中的候选，
    按命中词长度选择最具体的一项，减少“销售额”被更短的“金额”抢先匹配。

    业务逻辑：
        1. 遍历每个规范名称及其同义词
        2. 收集所有包含于文本的 ``(规范名, 命中词)``
        3. 按命中词长度选择最大项
        4. 没有命中时返回 ``None``

    Args:
        text: 待匹配的自然语言文本。
        groups: 规范名称到同义词序列的映射。

    Returns:
        tuple[str, str] | None: 规范名称及实际命中词；无匹配时为 ``None``。
    """
    hits = [(canonical, term) for canonical, terms in groups.items() for term in terms if term in text]
    return max(hits, key=lambda item: len(item[1]), default=None)


def _is_locally_bound_result_reference(
    text: str,
    reference: re.Match[str],
) -> bool:
    """识别同一句内由分组先行词约束的“该实体”。

    “各大区……该大区”中的后者是当前查询的分配性变量，不应读取上一轮
    单行结果。只有缺少这种本句先行词的“该地区……”才属于跨轮结果指代。
    """

    aliases = _REFERENCE_ENTITY_ALIASES.get(reference.group("entity"), ())
    if not aliases:
        return False
    alias_pattern = "|".join(re.escape(alias) for alias in aliases)
    prefix = text[: reference.start()]
    if re.search(
        rf"(?:各(?:个)?|每(?:个|一)?|所有|逐个)"
        rf"[^。；;！？!?]{{0,16}}(?:{alias_pattern})",
        prefix,
    ):
        return True
    return bool(
        re.search(
            rf"(?:按|以)[^。；;！？!?]{{0,32}}(?:{alias_pattern})",
            prefix,
        )
    )


def _replace_group(text: str, groups: Mapping[str, Sequence[str]], replacement: str) -> tuple[str, str | None]:
    """
    将文本中某一词组的全部同义表达替换为新的规范名称。

    核心功能：先用最长匹配确定旧词组所属的规范类别，再按同义词长度降序替换，
    避免短词先替换破坏更长短语；同时返回旧规范名称用于记录审计操作。

    业务逻辑：
        1. 查找文本中命中的最长同义词
        2. 若未命中，则原样返回文本并标记无旧值
        3. 获取旧规范类别下的全部同义词
        4. 按长度从长到短逐个替换为新规范名称
        5. 返回更新文本及旧规范名称

    Args:
        text: 待改写的历史独立问题。
        groups: 指标或维度同义词映射。
        replacement: 要写入的新规范名称。

    Returns:
        tuple[str, str | None]: 改写后的文本和被替换的旧规范名称。
    """
    old = _first_term(text, groups)

    # 历史问题中没有同类词时不做替换，由上层决定是否追加新语义。
    if old is None:
        return text, None
    updated = text

    # 长词优先可防止“销售总额”先被其中的“销售额”局部替换。
    for term in sorted(groups[old[0]], key=len, reverse=True):
        updated = updated.replace(term, replacement)
    return updated, old[0]


def _history_state(item: Mapping) -> dict:
    """Return only the persisted, current structured semantic state."""
    stored = item.get("semantic_state")
    return dict(stored) if isinstance(stored, Mapping) else {}


def _region_filter_values(where: exp.Expression | None) -> set[str]:
    """提取 WHERE 中以地区白名单列约束的字符串字面量。"""

    if where is None:
        return set()
    values: set[str] = set()
    for node in where.find_all(exp.EQ, exp.Like):
        left = node.this
        right = node.expression
        pairs = ((left, right), (right, left))
        for column, literal in pairs:
            if (
                isinstance(column, exp.Column)
                and column.name.lower() in _REGION_REFERENCE_COLUMNS
                and isinstance(literal, exp.Literal)
                and literal.is_string
            ):
                value = str(literal.this)
                if isinstance(node, exp.Like) and any(token in value for token in ("%", "_")):
                    continue
                values.add(value)
    for node in where.find_all(exp.In):
        column = node.this
        if not (
            isinstance(column, exp.Column)
            and column.name.lower() in _REGION_REFERENCE_COLUMNS
        ):
            continue
        for literal in node.expressions:
            if isinstance(literal, exp.Literal) and literal.is_string:
                values.add(str(literal.this))
    return values


def _audit_summary(action: str, operations: Sequence[Mapping], state: Mapping) -> str:
    """
    生成不包含问题、SQL 或结果值的追问决策审计摘要。

    核心功能：把操作槽位映射为中文标签，记录总体动作和变更类别；若继承历史维度，
    仅记录数量而不暴露具体字段，最终把摘要限制为五百字符。

    业务逻辑：
        1. 定义语义槽位到可读标签的映射
        2. 遍历操作并生成去内容化的变更类别文本
        3. 写入总体 follow_up_action
        4. 若存在变更，则附加变更槽位
        5. 对继承型动作记录上一轮维度数量
        6. 拼接并截断为五百字符

    Args:
        action: 本轮追问的总体动作类型。
        operations: 结构化槽位操作序列。
        state: 上一成功轮的脱敏语义状态。

    Returns:
        str: 可安全写入日志或 Trace 的简短摘要。
    """
    labels = {
        "measure": "指标",
        "dimension": "维度",
        "filter": "过滤",
        "time": "时间",
        "limit": "TopN",
        "order": "排序",
        "presentation": "展示",
        "result_reference": "结果指代",
        "topic": "话题",
    }

    # 只输出槽位标签，不输出 from/to 值，避免审计摘要携带用户问题内容。
    changed = "、".join(labels.get(str(item.get("slot")), str(item.get("slot"))) for item in operations)
    parts = [f"follow_up_action={action}"]

    # 有实际操作时附加受影响的语义类别。
    if changed:
        parts.append(f"变更 {changed}")
    inherited = len(state.get("dimensions") or [])

    # 仅对真正继承历史的动作记录维度数量；重置、新话题和孤立追问不继承。
    if inherited and action not in {"reset", "new_topic", "orphan"}:
        parts.append(f"上一轮维度 {inherited} 项已纳入闭合校验")
    return "；".join(parts)[:500]


def resolve_follow_up(
    query: str,
    chat_history: Sequence[Mapping] | None,
    *,
    result_references: Mapping[str, Mapping[str, object]] | None = None,
) -> dict:
    """
    将多轮省略追问确定性改写为可审计的独立问题，不读取或暴露历史结果行。

    核心功能：通过重置/追问提示、指标与维度词典、过滤和时间模式、TopN 及排序词，
    把当前短问句合并到上一成功问题；同时生成结构化操作、置信度和必须继承的 SQL
    槽位契约，对无法安全闭合的结果指代或缺失历史返回澄清请求。

    业务逻辑：
        1. 初始化新话题基线，并优先处理显式历史重置
        2. 判断是否为追问；检查成功历史和不安全结果指代
        3. 读取上一独立问题与脱敏状态，识别指标和维度替换/追加
        4. 识别过滤、时间、TopN 与排序方向变更
        5. 若未识别具体操作，则清理追问前缀后继承追加剩余文本
        6. 根据操作决定应保留的维度、过滤、排序和 LIMIT
        7. 返回独立问题、操作审计、继承契约及安全摘要

    Args:
        query: 当前轮自然语言问题。
        chat_history: 可选的成功查询历史，不应包含或读取结果行。
        result_references: Application 从紧邻单行结果投影出的白名单指代锚点。

    Returns:
        dict: 版本化追问改写结果、澄清状态、结构操作和继承契约。
    """
    # =============================================================================
    # 阶段 1：初始化基线并处理显式新话题重置
    # =============================================================================

    # 清理当前问题并复制历史列表；默认将输入视为高置信度独立新话题。
    original = str(query or "").strip()
    full_original = original
    history = list(chat_history or [])
    base = {
        "version": FOLLOW_UP_VERSION, "original_question": original,
        "standalone_question": original, "action": "new_topic", "operations": [],
        "deterministic": True, "confidence": 1.0, "needs_clarification": False,
        "clarification": None, "preserve": {}, "summary": "follow_up_action=new_topic",
    }

    # 重置提示拥有最高优先级，明确要求后续问题不继承任何历史语义。
    reset_match = _RESET_CUE.search(original)
    if reset_match:
        standalone = original[reset_match.end():].strip()

        # 用户只说“新问题”却未提供内容时无法形成独立查询，返回澄清请求。
        if not standalone:
            return {**base, "action": "orphan", "needs_clarification": True, "confidence": 0.0,
                    "clarification": "请在“新问题”后补充完整的查询内容。",
                    "summary": "follow_up_action=orphan；缺少独立问题"}

        # 重置后仍有正文时直接作为新独立问题，并记录 topic reset 操作。
        return {**base, "standalone_question": standalone, "action": "reset",
                "operations": [{"op": "reset", "slot": "topic"}],
                "summary": "follow_up_action=reset；清空历史语义"}

    # A topic-switch phrase inside a compound utterance explicitly discards
    # the earlier clause and any persisted history.  Keeping both topics made
    # the planner combine unrelated grains and metrics.
    topic_switch = re.search(
        r"(?:然后|接着|现在)?(?:我想)?(?:换个角度|换个方向|改问|另看)"
        r"[：,:，\s]*(?P<tail>.+)",
        original,
    )
    if topic_switch and topic_switch.group("tail").strip():
        standalone = topic_switch.group("tail").strip()
        return {
            **base,
            "standalone_question": standalone,
            "action": "reset",
            "operations": [{"op": "reset", "slot": "topic"}],
            "summary": "follow_up_action=reset；显式切换分析话题",
        }

    cue_match = _FOLLOW_UP_CUE.search(original)
    # A cue occurring after a substantive prefix belongs to the current
    # self-contained request (for example “先筛选……再按……统计”).  With no
    # persisted history it must remain a new topic instead of becoming an
    # orphan follow-up.
    if not history and cue_match is not None and cue_match.start() > 0:
        return base

    # Some clients repeat the previous question and append the actual change
    # after “现在/然后我想…”.  When history is available, parse only that
    # change clause; the canonical previous question already comes from the
    # persisted successful turn.  This prevents narrated old years/measures
    # from being mistaken for new constraints.
    if history:
        change_clause = re.search(
            r"(?:现在|然后|接下来|这次)(?:我)?想(?:继续)?(?:往下看)?"
            r"[：,:，\s]*(?P<tail>.+)",
            original,
        )
        if change_clause and change_clause.start() > 0:
            original = change_clause.group("tail").strip()

    # “该地区/这个地区”不是普通结构继承，而是对紧邻成功轮单值结果的引用。
    # 只接受 Application 提前投影的白名单锚点；缺失时必须澄清，不能从更早 SQL
    # 或模型上下文猜测旧字面量。绑定成功后也不继承上一轮的 GROUP/ORDER/LIMIT。
    singular_reference = next(
        (
            match
            for match in _SINGULAR_RESULT_REFERENCE.finditer(original)
            if not _is_locally_bound_result_reference(original, match)
        ),
        None,
    )
    if singular_reference:
        referenced_entity = singular_reference.group("entity")
        reference_key = (
            "region" if referenced_entity in {"地区", "区域", "大区"} else None
        )
        raw_reference = (result_references or {}).get(reference_key or "")
        reference = dict(raw_reference) if isinstance(raw_reference, Mapping) else {}
        value = str(reference.get("value") or "").strip()
        if not history or reference.get("entity") != "region" or not value:
            return {
                **base,
                "action": "clarify",
                "needs_clarification": True,
                "confidence": 0.0,
                "clarification": (
                    f"无法从紧邻上一轮确认唯一{referenced_entity}，"
                    f"请直接写明{referenced_entity}名称。"
                ),
                "summary": "follow_up_action=clarify；缺少紧邻单值结果指代",
            }
        region_label = value if value.endswith(("地区", "区域", "大区")) else f"{value}地区"
        standalone = (
            original[: singular_reference.start()]
            + region_label
            + original[singular_reference.end() :]
        )
        operations = [{
            "op": "bind",
            "slot": "result_reference",
            "entity": "region",
        }]
        preserve = {
            "dimensions": [],
            "filters": False,
            "filter_columns": [],
            "order": False,
            "order_directions": [],
            "limit": None,
            "result_reference": {
                "entity": "region",
                "column": str(reference.get("column") or ""),
                "value": value,
            },
        }
        return {
            **base,
            "original_question": full_original,
            "standalone_question": standalone,
            "action": "replace",
            "operations": operations,
            "confidence": 1.0,
            "preserve": preserve,
            "summary": _audit_summary("replace", operations, {}),
        }

    # =============================================================================
    # 阶段 2：识别追问并执行历史可继承性安全门禁
    # =============================================================================

    # “排除华东”这类只有一个操作和值的短句，在已有成功历史时是安全
    # 追问；带有分组/指标正文的“去掉 A 后，各地区交易额”仍是完整新题。
    short_filter_follow_up = bool(
        history
        and re.fullmatch(
            r"(?:排除|去掉|不含|不要)"
            r"[\u4e00-\u9fffA-Za-z0-9_-]{1,16}",
            original,
        )
    )

    # 不命中任何承接、替换或省略提示时保持默认新话题，不读取历史。
    if not _FOLLOW_UP_CUE.search(original) and not short_filter_follow_up:
        return base

    # 追问没有成功历史可继承时必须澄清，不能凭空猜测维度、指标或过滤。
    if not history:
        return {**base, "action": "orphan", "needs_clarification": True, "confidence": 0.0,
                "clarification": "当前会话没有可继承的成功查询，请把维度、指标和条件写完整。",
                "summary": "follow_up_action=orphan；无可继承成功轮"}

    # “这些/那些”指向的可能是截断、分页或脱敏结果值，禁止从历史结果行恢复实体集合。
    if _UNSAFE_RESULT_REFERENCE.search(original):
        return {**base, "action": "clarify", "needs_clarification": True, "confidence": 0.0,
                "clarification": "“这些/那些”可能引用已截断或脱敏的结果，请明确客户、商品、地区或筛选条件。",
                "summary": "follow_up_action=clarify；拒绝继承历史结果值"}

    # 仅继承最近一轮成功查询的独立问题和脱敏语义状态。
    previous = history[-1]
    previous_question = str(previous.get("standalone_question") or previous.get("query") or "").strip()
    previous_state = _history_state(previous)
    if not previous_state:
        return {
            **base,
            "action": "clarify",
            "needs_clarification": True,
            "confidence": 0.0,
            "clarification": "上一轮缺少当前语义状态，请明确本轮指标、维度和筛选条件。",
            "summary": "follow_up_action=clarify；缺少当前 semantic_state 契约",
        }
    standalone = previous_question
    operations: list[dict] = []
    action = "inherit"

    # 展示形式属于结果呈现偏好，不得误判为 SQL 过滤值。保留当前原话，
    # 让后续图表节点能读取明确类型，同时继续继承上一轮 SQL 结构。
    presentation = explicit_chart_type(original)
    if presentation is not None:
        if original not in standalone:
            standalone = f"{standalone}，{original}"
        operations.append({
            "op": "replace",
            "slot": "presentation",
            "to": presentation,
        })
        action = "replace"

    # =============================================================================
    # 阶段 3：解析指标与维度的替换或追加操作
    # =============================================================================

    # 在当前追问中选择最长指标词，并尝试替换上一独立问题中的同类指标。
    new_measure = _first_term(original, _MEASURE_TERMS)
    if new_measure:
        append_measure = bool(
            re.search(
                r"(?:也|同时|一并|另外|再)(?:把|将)?[^，,]{0,8}"
                r"(?:列出|给出|加上|补上|追加|附加|展示|统计)",
                original,
            )
            or re.search(
                r"(?:也列|也加|同时给|一并给|补上|并追加|再追加|并附加)",
                original,
            )
            or re.search(
                r"(?:再|同时|另外|还|除此之外)(?:增加|追加|补充)",
                original,
            )
        )
        if append_measure:
            if new_measure[0] not in standalone:
                standalone = f"{standalone}，同时列出{new_measure[0]}"
            operations.append({
                "op": "append",
                "slot": "measure",
                "to": new_measure[0],
            })
            action = "append"
        else:
            standalone, old_measure = _replace_group(
                standalone, _MEASURE_TERMS, new_measure[0]
            )

            # 新旧规范指标不同时才记录替换；相同指标不产生虚假审计操作。
            if old_measure != new_measure[0]:
                operations.append({
                    "op": "replace",
                    "slot": "measure",
                    "from": old_measure,
                    "to": new_measure[0],
                })
                action = "replace"

    # 同时识别“改成按维度”和“再按维度”两类语义不同的表达。
    dimension_match = re.search(
        r"(?:改成|换成|改为|调整为|改按)(?:按)?"
        r"(品类|类别|地区|区域|大区|品牌|商品|产品|客户|用户|性别|"
        r"月份|月|季度|年份|年)|"
        r"(再按|继续按|也按|并按|按|横向分|横向拆成|再分)"
        r"(品类|类别|地区|区域|大区|品牌|商品|产品|客户|用户|性别|"
        r"男女|月份|月|季度|年份|年)",
        original,
    )
    if dimension_match:
        raw_dimension = dimension_match.group(1) or dimension_match.group(3)

        # 将别名归一为稳定维度名称；词典未覆盖时保留原始匹配词。
        canonical = (_first_term(raw_dimension, _DIMENSION_TERMS) or (raw_dimension, raw_dimension))[0]

        # “再按”表示在原分组上追加维度，而非替换已有分组。
        if dimension_match.group(2) in {
            "再按", "继续按", "也按", "并按", "横向分", "横向拆成", "再分",
        }:
            standalone = f"{standalone}，再按{canonical}拆分"
            operations.append({"op": "append", "slot": "dimension", "to": canonical})
            action = "append"
        else:
            # “改成/按”优先替换历史维度；历史未出现维度词时追加显式改写短语。
            standalone, old_dimension = _replace_group(standalone, _DIMENSION_TERMS, canonical)
            if old_dimension is None:
                standalone = f"{standalone}，改成按{canonical}"
            operations.append({"op": "replace", "slot": "dimension", "from": old_dimension, "to": canonical})
            action = "replace"

    # =============================================================================
    # 阶段 4：解析过滤、时间、TopN 与排序方向变更
    # =============================================================================

    # “只看/筛选”等前缀表示在继承查询上追加过滤条件。
    if re.search(
        r"(?:^|[，,；;])(?:只看|仅看|只对比|仅对比|只统计|只保留|仅保留|只留下|"
        r"限定|筛选|排除|不含|不要|改看)",
        original,
    ):
        standalone = f"{standalone}，{original}"
        operations.append({"op": "append", "slot": "filter"})
        action = "append"
    elif re.search(r"(?:并|同时|且)(?:排除|不含|不要)", original):
        if original not in standalone:
            standalone = f"{standalone}，{original}"
        operations.append({"op": "append", "slot": "filter"})
        action = "append"
    elif re.search(r"(?:去掉|取消|不再|不要).*(?:限制|条件|筛选)|^(?:去掉|取消)", original):
        # “去掉/取消”等表达表示移除历史过滤，后续 preserve 不再要求过滤继承。
        standalone = f"{standalone}，{original}"
        operations.append({"op": "remove", "slot": "filter"})
        action = "remove"
    elif not any(
        item.get("slot") in {"dimension", "presentation"} for item in operations
    ) and re.search(
        r"(?:范围|区域|地区|大区|省份)?(?:改成|换成|改为)"
        r"[\u4e00-\u9fffA-Za-z0-9_-]{1,16}(?:地区|大区|省)?",
        original,
    ):
        # “范围改成华南地区”替换的是过滤值，不是把地区改成分组轴。
        # 保留完整当前短语，让后续字段值解析按最后一次显式值完成替换。
        if original not in standalone:
            standalone = f"{standalone}，{original}"
        operations.append({"op": "replace", "slot": "filter"})
        action = "replace"

    # 显式年份表示覆盖上一轮的年份槽位，而不是把新旧年份同时追加成周期集合。
    # 例如“2025 年各品类销售额”后的“那 2024 年呢”应闭合为
    # “2024 年各品类销售额”。这是一条通用槽位规则，不依赖具体评测问题。
    explicit_years = list(dict.fromkeys(re.findall(r"(?<!\d)(?:19|20)\d{2}年", original)))
    comparison_year_append = bool(
        re.search(r"(?:加上|加入|再加|对比|比较|一并)", original)
        and len(explicit_years) == 1
        and re.search(r"(?<!\d)(?:19|20)\d{2}年", standalone)
    )
    if explicit_years:
        previous_years = re.findall(r"(?<!\d)(?:19|20)\d{2}年", standalone)
        if comparison_year_append:
            comparison_tail = re.sub(
                r"^(?:那|那么|然后|再|其中)",
                "",
                original,
            ).strip(" ，,？?")
            standalone = (
                f"{standalone}，{comparison_tail}"
                if comparison_tail
                else f"{standalone}，加上{explicit_years[0]}作对比"
            )
        elif len(explicit_years) == 1 and previous_years:
            standalone = re.sub(
                r"(?<!\d)(?:19|20)\d{2}年",
                explicit_years[0],
                standalone,
            )
        elif not previous_years:
            standalone = f"{standalone}，限定{'、'.join(explicit_years)}"
        else:
            # 多年份追问通常表达周期比较；移除历史年份后显式附加本轮完整周期，
            # 避免保留未被用户再次指定的旧年份。
            standalone = re.sub(r"(?<!\d)(?:19|20)\d{2}年", "", standalone)
            standalone = f"{'、'.join(explicit_years)}，{standalone.lstrip('，, ')}"
        operations.append({
            "op": "append" if comparison_year_append else "replace",
            "slot": "time",
            "from": previous_years,
            "to": explicit_years,
        })
        action = "append" if comparison_year_append else "replace"

    # 相对时间、时间对比或粒度词会追加到独立问题；包含“改”时标记替换，否则标记追加。
    dimension_time_already_applied = any(
        item.get("slot") == "dimension"
        and item.get("to") in {"月份", "季度", "年份"}
        for item in operations
    )
    if (
        re.search(r"同期|同比|环比|去年|今年|本月|上月|按月|按季|季度|年度", original)
        and not dimension_time_already_applied
        and not comparison_year_append
    ):
        standalone = f"{standalone}，{original}"
        operations.append({"op": "replace" if "改" in original else "append", "slot": "time"})
        action = "replace" if "改" in original else "append"

    # 提取阿拉伯数字或一至十中文数字表达的前/后/TopN 限制。
    limit_match = re.search(r"(?:前|后|top\s*)(\d{1,4}|[一二两三四五六七八九十])", original, re.IGNORECASE)
    if limit_match:
        target = limit_match.group(0)

        # 在历史独立问题中寻找已有 TopN，存在则原位替换，否则追加“改成”短语。
        prior_limit = re.search(r"(?:前|后|top\s*)(?:\d{1,4}|[一二两三四五六七八九十])", standalone, re.IGNORECASE)
        standalone = (standalone[:prior_limit.start()] + target + standalone[prior_limit.end():]
                      if prior_limit else f"{standalone}，改成{target}")
        operations.append({"op": "replace", "slot": "limit"})
        action = "replace"

    # 按固定顺序选择当前追问中首个排序方向词，区分低/升序与高/降序两组。
    reverse_order = bool(
        re.search(
            r"(?:顺序|排序).*(?:反过来|倒过来|反向)|倒序|^倒过来",
            original,
        )
    )
    direction = next((item for item in ("最低", "最少", "升序", "最高", "最多", "降序") if item in original), None)
    if re.search(r"(?:从|由)(?:低|少|小)到(?:高|多|大)|正序", original):
        direction = "升序"
    elif re.search(r"(?:从|由)(?:高|多|大)到(?:低|少|小)", original):
        direction = "降序"
    if reverse_order:
        swapped = standalone
        replacements = (
            ("最高", "__ORDER_LOW__"),
            ("最多", "__ORDER_LEAST__"),
            ("最低", "最高"),
            ("最少", "最多"),
            ("降序", "__ORDER_ASC__"),
            ("升序", "降序"),
        )
        for source, target in replacements:
            swapped = swapped.replace(source, target)
        swapped = (
            swapped
            .replace("__ORDER_LOW__", "最低")
            .replace("__ORDER_LEAST__", "最少")
            .replace("__ORDER_ASC__", "升序")
        )
        standalone = (
            swapped
            if swapped != standalone
            else f"{standalone}，按结果"
            + (
                "升序"
                if any(
                    re.search(r"\bDESC\b", str(item), re.IGNORECASE)
                    for item in (previous_state.get("order") or [])
                )
                else "降序"
            )
        )
        operations.append({"op": "reverse", "slot": "order"})
        action = "replace"
        # “倒过来，从最低项开始”中的“最低”解释反转后的起点，不是
        # Top-1 极值。反转已经生成明确 ASC/DESC，避免下一分支再次追加
        # “按结果最低”并让编译器误加 LIMIT 1。
        if re.search(r"从[^，。；;]{0,24}(?:最低|最少|最短)[^，。；;]{0,12}(?:开始|起)", original):
            direction = None
    if direction:
        opposites = ("最高", "最多", "降序") if direction in {"最低", "最少", "升序"} else ("最低", "最少", "升序")
        updated = standalone

        # 遍历相反方向的全部同义词，将历史问题中的方向统一替换为当前要求。
        for old in opposites:
            updated = updated.replace(old, direction)

        # 历史中存在相反词或已包含目标词时使用改写结果，否则追加明确排序短语。
        standalone = updated if updated != standalone or direction in standalone else f"{standalone}，按结果{direction}"
        operations.append({"op": "replace", "slot": "order"})
        action = "replace"

    # =============================================================================
    # 阶段 5：处理未命中具体槽位的通用继承追问
    # =============================================================================

    # 没有识别指标、维度、过滤、时间、限制或排序变化时，按话题继承处理。
    if not operations:

        # 去除“那/再/其中”等承接前缀，只把剩余且尚未出现的文本追加到独立问题。
        tail = re.sub(r"^(?:那|那么|然后|再|其中)", "", original).strip(" ，,？?")
        if tail and tail not in standalone:
            standalone = f"{standalone}，{tail}"
        operations.append({"op": "inherit", "slot": "topic"})

    # =============================================================================
    # 阶段 6：构建必须继承的结构槽位契约并返回结果
    # =============================================================================

    # 默认继承上一轮全部分组维度；显式替换维度时清空旧维度继承要求。
    preserve_dimensions = list(previous_state.get("dimensions") or [])
    if any(item.get("slot") == "dimension" and item.get("op") == "replace" for item in operations):
        preserve_dimensions = []

    # 除非明确移除过滤，否则保留上一轮过滤存在性及所引用字段。
    preserved_filters = (
        list(previous_state.get("filters") or [])
        if not any(item.get("slot") == "filter" and item.get("op") == "remove" for item in operations)
        else []
    )
    order_items = list(previous_state.get("order") or [])

    # preserve 仅保存脱敏结构：字段名、布尔存在性、排序方向及数值限制。
    preserve = {
        "dimensions": preserve_dimensions,
        "filters": bool(preserved_filters),

        # 汇总上一轮全部过滤字段并规范化去重，供生成后闭合校验。
        "filter_columns": sorted({
            str(column).lower()
            for item in preserved_filters
            for column in (item.get("columns") or [])
        }),

        # 只有本轮未修改排序时才要求完整继承上一轮排序及方向序列。
        "order": bool(order_items) and not any(item.get("slot") == "order" for item in operations),
        "order_directions": [
            "DESC" if re.search(r"\bDESC\b", str(item), re.IGNORECASE) else "ASC"
            for item in order_items
        ],

        # 本轮未修改 TopN 时继承历史 LIMIT，否则交由新问题重新生成。
        "limit": previous_state.get("limit") if not any(item.get("slot") == "limit" for item in operations) else None,
    }

    # 改写成功使用高但非满分置信度，摘要仅包含动作类别和继承数量。
    return {**base, "original_question": full_original,
            "standalone_question": standalone, "action": action, "operations": operations,
            "confidence": 0.99, "preserve": preserve,
            "summary": _audit_summary(action, operations, previous_state)}


def validate_follow_up_sql(sql: str, follow_up: Mapping | None) -> None:
    """
    校验生成 SQL 是否保留了追问契约要求继承的结构槽位。

    核心功能：针对真正继承历史的追问，解析最终 SELECT 并依次核对分组维度、
    过滤存在性与字段、排序存在性与方向、以及 LIMIT；当前请求的表列 RBAC
    由独立访问策略执行，本函数只负责跨轮语义闭合。

    业务逻辑：
        1. 跳过需澄清、新话题、重置、孤立或不安全指代请求
        2. 解析 SQL 并定位最终 SELECT；不存在时交由其他校验器处理
        3. 验证所有待继承分组维度仍位于 GROUP BY
        4. 验证历史过滤未丢失且过滤字段集合未被替换
        5. 验证历史排序未丢失且方向序列完全一致
        6. 验证历史 LIMIT 仍存在且数值一致

    Args:
        sql: 根据独立问题新生成的 SQL。
        follow_up: ``resolve_follow_up`` 生成的追问补丁与 preserve 契约。

    Raises:
        FollowUpContractError: 任一要求继承的维度、过滤、排序或 LIMIT 被删除或改变。
        sqlglot.errors.ParseError: SQL 无法按 MySQL 方言解析。
    """
    # =============================================================================
    # 阶段 1：筛选需要执行继承闭合校验的追问
    # =============================================================================

    # 空补丁按空映射处理，避免调用方为非追问额外构造对象。
    patch = follow_up or {}

    # 澄清请求及不继承历史的动作没有 preserve 契约，直接跳过。
    if patch.get("needs_clarification") or patch.get("action") in {None, "new_topic", "reset", "orphan", "clarify"}:
        return
    preserve = patch.get("preserve") or {}

    # =============================================================================
    # 阶段 2：解析最终 SELECT 并验证分组维度
    # =============================================================================

    tree = sqlglot.parse_one(str(sql or ""), read="mysql")
    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)

    # 无 SELECT 的结构错误由语法/安全校验负责，本函数不重复生成追问错误。
    if select is None:
        return
    result_reference = preserve.get("result_reference") or {}
    if result_reference.get("entity") == "region":
        expected_value = str(result_reference.get("value") or "")
        current_values = _region_filter_values(select.args.get("where"))
        if current_values != {expected_value}:
            raise FollowUpContractError(
                "生成 SQL 未唯一绑定紧邻上一轮的地区结果指代"
            )
    group = select.args.get("group")
    grouped = {column.name.lower() for column in (group.find_all(exp.Column) if group is not None else [])}
    expected = {str(item).split(".")[-1].lower() for item in preserve.get("dimensions") or []}

    # 所有待继承维度都必须仍在 GROUP BY，允许新 SQL 在此基础上追加其他维度。
    if expected and not expected.issubset(grouped):
        raise FollowUpContractError(f"必须继承分组维度 {sorted(expected)}，当前为 {sorted(grouped)}")

    # =============================================================================
    # 阶段 3：验证过滤结构与字段集合
    # =============================================================================

    where = select.args.get("where")

    # preserve.filters 为真表示本轮未移除上一轮过滤，WHERE 必须继续存在。
    if preserve.get("filters"):
        if where is None:
            raise FollowUpContractError("上一轮过滤条件被意外删除")

        # 比较脱敏状态中的过滤字段集合，不读取也不比较历史字面量值。
        current_filter_columns = {column.name.lower() for column in where.find_all(exp.Column)}
        expected_filter_columns = {str(item).lower() for item in preserve.get("filter_columns") or []}

        # 历史过滤字段必须是当前字段集合的子集，允许本轮追加新的过滤条件。
        if expected_filter_columns and not expected_filter_columns.issubset(current_filter_columns):
            raise FollowUpContractError(
                f"上一轮过滤字段被意外替换：应包含 {sorted(expected_filter_columns)}，"
                f"当前为 {sorted(current_filter_columns)}"
            )

    # =============================================================================
    # 阶段 4：验证排序存在性与方向序列
    # =============================================================================

    order = select.args.get("order")

    # 仅当本轮没有显式修改排序且历史存在排序时，要求完整继承。
    if preserve.get("order"):
        if order is None:
            raise FollowUpContractError("上一轮排序被意外删除")
        expected_directions = list(preserve.get("order_directions") or [])

        # 按表达式顺序提取 ASC/DESC，确保多列排序的方向序列没有改变。
        current_directions = ["DESC" if item.args.get("desc") else "ASC" for item in order.expressions]
        # Appending a new dimension may append a deterministic tie-breaker.
        # The inherited ordering contract is preserved when its direction
        # sequence remains the prefix; requiring exact list equality rejected
        # valid “also split by …” follow-ups.
        if (
            expected_directions
            and current_directions[:len(expected_directions)] != expected_directions
        ):
            raise FollowUpContractError(
                f"上一轮排序方向被意外改变：应为 {expected_directions}，当前为 {current_directions}"
            )

    # =============================================================================
    # 阶段 5：验证 TopN LIMIT 数值
    # =============================================================================

    expected_limit = preserve.get("limit")

    # preserve.limit 非空表示本轮未修改 TopN，生成 SQL 必须继续使用相同限制。
    if expected_limit is not None:
        limit = select.args.get("limit")
        if limit is None:
            raise FollowUpContractError("上一轮限制被意外删除")

        # 仅接受可直接转换为整数的 LIMIT 字面量，复杂表达式按不匹配处理。
        try:
            current_limit = int(limit.expression.this)
        except (AttributeError, TypeError, ValueError):
            current_limit = None

        # 当前限制必须与脱敏语义状态中的整数完全一致，不能扩大或缩小结果范围。
        if current_limit != int(expected_limit):
            raise FollowUpContractError(f"上一轮限制被意外改变：应为 {expected_limit}，当前为 {current_limit}")
