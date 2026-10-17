"""问数请求的确定性意图策略。

该模块只处理请求文本本身，不依赖模型、数据库或运行时。安全拒绝优先于
业务范围判断；模型只能接手规则无法确认的 ``unknown`` 请求。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal


DeterministicIntentCategory = Literal[
    "business_query",
    "business_help",
    "out_of_scope",
    "forbidden_operation",
    "ambiguous",
    "unknown",
]


@dataclass(frozen=True, slots=True)
class DeterministicIntentResult:
    """不含原始文本的确定性分类结果。"""

    category: DeterministicIntentCategory
    reason_codes: tuple[str, ...]
    rule_ids: tuple[str, ...]


_SQL_MUTATION_PATTERNS = (
    re.compile(r"\bdelete\s+from\b", re.IGNORECASE),
    re.compile(r"\bupdate\s+[\w`.]+\s+set\b", re.IGNORECASE),
    re.compile(r"\binsert\s+into\b", re.IGNORECASE),
    re.compile(r"\b(?:drop|truncate|alter|create)\s+(?:table|database|schema)\b", re.IGNORECASE),
    re.compile(r"\b(?:grant|revoke)\b", re.IGNORECASE),
)

_PROMPT_INJECTION_PATTERNS = (
    (
        "deny.prompt_injection.override.v1",
        re.compile(
            r"(?:忽略|无视|忘掉|覆盖).{0,16}"
            r"(?:之前|以上|先前|原有|所有|系统|开发者).{0,10}"
            r"(?:指令|要求|规则|提示词?|消息)",
            re.IGNORECASE,
        ),
    ),
    (
        "deny.prompt_injection.bypass.v1",
        re.compile(
            r"(?:绕过|跳过|规避|关闭|禁用|解除).{0,12}"
            r"(?:权限|授权|安全|规则|限制|校验|验证|审计)",
            re.IGNORECASE,
        ),
    ),
    (
        "deny.prompt_injection.disclosure.v1",
        re.compile(
            r"(?:显示|输出|打印|泄露|查看|告诉我).{0,14}"
            r"(?:系统提示词|系统指令|开发者消息|隐藏提示词)",
            re.IGNORECASE,
        ),
    ),
    (
        "deny.prompt_injection.override.v1",
        re.compile(
            r"\b(?:ignore|override|forget)\b.{0,40}"
            r"\b(?:previous|prior|system|developer|all)\b.{0,24}"
            r"\b(?:instructions?|messages?|prompts?|rules?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "deny.prompt_injection.bypass.v1",
        re.compile(
            r"\b(?:bypass|disable|evade)\b.{0,28}"
            r"\b(?:permission|authorization|security|safety|validation)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "deny.prompt_injection.disclosure.v1",
        re.compile(
            r"\b(?:reveal|show|print|leak)\b.{0,24}"
            r"\b(?:system prompt|developer message|hidden prompt)\b",
            re.IGNORECASE,
        ),
    ),
)
_QUOTED_TEXT_PATTERN = re.compile(
    r"‘[^’]{1,200}’|“[^”]{1,200}”|'[^']{1,200}'|\"[^\"]{1,200}\"|`[^`]{1,200}`"
)
_TEXT_ANALYSIS_NEAR_MISS_PATTERN = re.compile(
    r"(?:统计|查询|查看|分析|计算|汇总).{0,16}"
    r"(?:反馈|评论|文本|日志|工单|记录|内容).{0,16}(?:数量|条数|次数|占比)?"
)

_MUTATION_VERB_PATTERN = re.compile(
    r"删除|删掉|删去|移除|清空|修改|更改|更新|改(?=成|为|到|一下)|"
    r"新增|添加|插入|覆盖|写入"
)
_DATA_TARGET_PATTERN = re.compile(
    r"客户|用户|订单|商品|产品|品牌|地区|区域|大区|销售额|销量|营收|"
    r"金额|价格|库存|利润|成本|记录|数据|内容|字段|表(?:格)?|行|列|"
    r"数据库|账户|指标|维度|文件"
)
_NEGATED_MUTATION_PATTERN = re.compile(
    r"(?:不要|别|禁止|无需|不需要|不可|不能|不允许)\s*"
    r"(?:执行|进行)?\s*$"
)
_ANALYTICAL_PREFIX_PATTERN = re.compile(
    r"统计|查询|查看|分析|计算|汇总|展示|显示|对比|比较|多少|哪些|数量"
)
_CONTROL_TARGET_PATTERN = re.compile(
    r"^(?:为|成|到|一个|这个|该|当前|现有)?\s*"
    r"(?:筛选|过滤|排序|查询条件|筛选条件|过滤条件|图表|图形|可视化|"
    r"饼图|柱状图|条形图|折线图|散点图|雷达图|展示|显示|布局|颜色|"
    r"单位|小数位|标题)"
)
_CONTROL_PHRASE_PATTERN = re.compile(
    r"(?:展示|显示)(?:格式|方式)|图表(?:类型|样式)|筛选条件|过滤条件"
)
_AMBIGUOUS_OPERATION_PATTERN = re.compile(
    r"^\s*(?:帮我)?(?:把|将)?(?:它|这个|那个)?\s*"
    r"(?:调整|改动|处理)(?:一下)?\s*"
    r"(?:销售额|销量|数据|内容|记录|指标)?[?？。！!\s]*$"
)

_BUSINESS_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"销售(?:额|量)?|销量|营收|成交额|gmv|收入|利润|成本|客单价|复购|转化率",
        r"订单|客户|商品|产品|品牌|地区|区域|大区|省份|城市|库存|退货|退款",
        r"指标|维度|数据表|字段|数据集|上传(?:文件|数据)|csv|excel",
        r"饼图|柱状图|条形图|折线图|散点图|雷达图|图表|可视化",
        r"同比|环比|趋势|排名|前\s*\d+|后\s*\d+|top\s*\d+",
    )
)
_HELP_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^\s*(?:你好|您好|嗨|hi|hello)[!！。,.，\s]*$",
        r"(?:你|助手|这个系统).{0,8}(?:能做什么|会做什么|支持什么|有哪些功能)",
        r"(?:怎么|如何)(?:使用|提问|问问题)",
        r"(?:怎么|如何).{0,12}(?:上传|查看字段|配置数据集)",
        r"支持哪些(?:图表|文件|数据|功能)",
        r"(?:给我|提供).{0,4}(?:示例问题|提问示例)",
    )
)
_OUT_OF_SCOPE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"天气|天气预报|气温|温度|下雨|降雨|带伞|空气质量",
        r"讲个笑话|笑话|故事|写诗|诗歌|作文|翻译",
        r"菜谱|怎么做菜|导航|路线怎么走|电影推荐|音乐推荐",
        r"星座|运势|体育比分|球赛比分",
        r"(?:写|生成).{0,8}(?:python|java|javascript|代码|程序)",
    )
)


def normalize_query_text(value: object) -> str:
    """统一字符宽度、大小写和空白，不改变请求语义。"""

    text = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    return " ".join(text.split())


def _has_any(patterns: tuple[re.Pattern[str], ...], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _is_control_mutation(text_after_verb: str) -> bool:
    """识别只改变筛选或展示状态、不会写业务数据的动作。"""

    local = text_after_verb[:24].lstrip()
    if _CONTROL_TARGET_PATTERN.search(local):
        return True
    return bool(_CONTROL_PHRASE_PATTERN.search(local[:16]))


def _is_analytical_mutation_reference(
    text: str,
    *,
    start: int,
    end: int,
) -> bool:
    """识别“统计已删除订单”等状态描述，而非执行指令。"""

    prefix = text[max(0, start - 16):start]
    suffix = text[end:end + 3]
    passive = bool(re.search(r"(?:已|被|曾经|已经)\s*$", prefix))
    participle = bool(re.match(r"(?:过|的|了)", suffix))
    analytical = bool(_ANALYTICAL_PREFIX_PATTERN.search(prefix))
    rank_comparison = bool(re.match(r"(?:前|后)", suffix))
    return passive or (analytical and (participle or rank_comparison))


def _prompt_injection_rule(text: str) -> str | None:
    candidate = text
    # 带引号的反馈/日志内容分析是数据查询，不把样本文本当成当前轮指令。
    if _TEXT_ANALYSIS_NEAR_MISS_PATTERN.search(text):
        candidate = _QUOTED_TEXT_PATTERN.sub("[quoted_text]", text)
    for rule_id, pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(candidate):
            return rule_id
    return None


def _mutation_rule(text: str) -> str | None:
    injection_rule = _prompt_injection_rule(text)
    if injection_rule:
        return injection_rule
    for pattern in _SQL_MUTATION_PATTERNS:
        if pattern.search(text):
            return "deny.sql_mutation.v1"

    for match in _MUTATION_VERB_PATTERN.finditer(text):
        prefix = text[max(0, match.start() - 12):match.start()]
        suffix = text[match.end():]
        if _NEGATED_MUTATION_PATTERN.search(prefix):
            continue
        if _is_control_mutation(suffix):
            continue
        if _is_analytical_mutation_reference(
            text,
            start=match.start(),
            end=match.end(),
        ):
            continue

        local_context = text[max(0, match.start() - 24):match.end() + 40]
        if _DATA_TARGET_PATTERN.search(local_context):
            return "deny.business_data_mutation.v1"
    return None


def _scope_rule(text: str) -> tuple[DeterministicIntentCategory, str] | None:
    if _has_any(_HELP_PATTERNS, text):
        return "business_help", "scope.business_help.v1"
    if _has_any(_BUSINESS_PATTERNS, text):
        return "business_query", "scope.business_query.v1"
    if _has_any(_OUT_OF_SCOPE_PATTERNS, text):
        return "out_of_scope", "scope.out_of_scope.v1"
    return None


def classify_deterministic_query_intent(
    original_query: object,
    standalone_query: object,
) -> DeterministicIntentResult:
    """先拒绝写操作，再判定明确业务范围，其余交给模型。

    ``original_query`` 防止多轮改写把本轮危险或越界意图洗成历史业务问题；
    ``standalone_query`` 则让“该地区呢”之类闭合后的业务追问可被正确放行。
    """

    original = normalize_query_text(original_query)
    standalone = normalize_query_text(standalone_query) or original
    if not original and not standalone:
        return DeterministicIntentResult(
            category="ambiguous",
            reason_codes=("intent.empty_input",),
            rule_ids=("clarify.empty_input.v1",),
        )

    for source, text in (("original", original), ("standalone", standalone)):
        rule_id = _mutation_rule(text)
        if rule_id:
            reason = (
                f"intent.prompt_injection.{source}"
                if rule_id.startswith("deny.prompt_injection.")
                else f"intent.forbidden.{source}"
            )
            return DeterministicIntentResult(
                category="forbidden_operation",
                reason_codes=(reason,),
                rule_ids=(rule_id,),
            )

    if _AMBIGUOUS_OPERATION_PATTERN.fullmatch(original):
        return DeterministicIntentResult(
            category="ambiguous",
            reason_codes=("intent.ambiguous_operation.original",),
            rule_ids=("clarify.ambiguous_operation.v1",),
        )

    original_scope = _scope_rule(original) if original else None
    standalone_scope = _scope_rule(standalone) if standalone else None

    original_has_business = _has_any(_BUSINESS_PATTERNS, original)
    original_has_out_of_scope = _has_any(_OUT_OF_SCOPE_PATTERNS, original)
    if original_has_business and original_has_out_of_scope:
        return DeterministicIntentResult(
            category="ambiguous",
            reason_codes=("intent.mixed_scope.original",),
            rule_ids=("clarify.mixed_scope.v1",),
        )

    # 本轮明确越界时不能被上一轮业务上下文覆盖。
    if (
        original_scope
        and original_scope[0] == "out_of_scope"
        and not original_has_business
    ):
        return DeterministicIntentResult(
            category="out_of_scope",
            reason_codes=("intent.out_of_scope.original",),
            rule_ids=(original_scope[1],),
        )

    selected = original_scope or standalone_scope
    if selected:
        category, rule_id = selected
        return DeterministicIntentResult(
            category=category,
            reason_codes=(f"intent.{category}.deterministic",),
            rule_ids=(rule_id,),
        )

    return DeterministicIntentResult(
        category="unknown",
        reason_codes=("intent.scope_unknown",),
        rule_ids=("route.scope_model.v1",),
    )


__all__ = [
    "DeterministicIntentCategory",
    "DeterministicIntentResult",
    "classify_deterministic_query_intent",
    "normalize_query_text",
]
