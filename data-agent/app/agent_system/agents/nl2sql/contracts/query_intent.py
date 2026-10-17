"""版本化、可脱敏持久化的问数意图合同。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from app.domain.query.intent_policy import normalize_query_text


QUERY_INTENT_CONTRACT_VERSION = "query-intent-v1"
INTENT_RESPONSE_SCHEMA_VERSION = "intent-response-v1"

QueryIntentCategory = Literal[
    "business_query",
    "business_help",
    "out_of_scope",
    "forbidden_operation",
    "ambiguous",
]
QueryIntentDecisionType = Literal["continue", "respond"]
QueryIntentResponseKind = Literal["guidance", "refusal", "clarification"]

QUERY_INTENT_CATEGORIES = {
    "business_query",
    "business_help",
    "out_of_scope",
    "forbidden_operation",
    "ambiguous",
}
MODEL_SCOPE_CATEGORIES = {
    "business_query",
    "business_help",
    "out_of_scope",
    "ambiguous",
}

_CATEGORY_CONTRACT: dict[
    str,
    tuple[QueryIntentDecisionType, QueryIntentResponseKind | None, str],
] = {
    "business_query": ("continue", None, ""),
    "business_help": (
        "respond",
        "guidance",
        "我可以帮助你查询和分析销售、订单、客户、商品、地区等业务数据。"
        "例如可以问：“本月各地区销售额趋势”。",
    ),
    "out_of_scope": (
        "respond",
        "guidance",
        "我主要用于销售、订单、客户、商品、地区等业务数据分析。"
        "你可以试试：“本月各地区销售额趋势”。",
    ),
    "forbidden_operation": (
        "respond",
        "refusal",
        "当前助手仅支持只读业务数据查询，不能新增、删除或修改业务数据。"
        "你可以改问：“查询销售额最高的10位客户及其销售额”。",
    ),
    "ambiguous": (
        "respond",
        "clarification",
        "我暂时无法确认这是一个只读业务数据问题。"
        "请明确要查询的指标、维度、时间范围或筛选条件后重试。",
    ),
}

_CATEGORY_SUGGESTIONS: dict[str, tuple[str, ...]] = {
    "business_query": (),
    "business_help": (
        "本月各地区销售额趋势",
        "销量前十的商品",
    ),
    "out_of_scope": ("本月各地区销售额趋势",),
    "forbidden_operation": ("查询销售额最高的10位客户及其销售额",),
    "ambiguous": ("查询本月各地区销售额，并按地区分组",),
}


class QueryIntentModelOutputError(ValueError):
    """模型范围分类输出不符合封闭枚举合同。"""


def query_text_hash(value: object) -> str:
    """生成稳定哈希；合同和日志不保存问题原文。"""

    normalized = normalize_query_text(value)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _bounded_codes(values: Sequence[object], *, limit: int) -> tuple[str, ...]:
    output: list[str] = []
    for value in values[:limit]:
        item = str(value or "").strip()[:96]
        if item and item not in output:
            output.append(item)
    return tuple(output)


def _decode_model_json(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    text = str(value or "").strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise QueryIntentModelOutputError("invalid_json") from exc
    if not isinstance(parsed, Mapping):
        raise QueryIntentModelOutputError("invalid_shape")
    return parsed


def parse_model_scope_category(value: object) -> QueryIntentCategory:
    """只接受单字段 JSON 和封闭枚举，禁止模型产生安全放行指令。"""

    payload = _decode_model_json(value)
    if set(payload) != {"category"}:
        raise QueryIntentModelOutputError("unexpected_fields")
    category = str(payload.get("category") or "").strip()
    if category not in MODEL_SCOPE_CATEGORIES:
        raise QueryIntentModelOutputError("invalid_category")
    return cast(QueryIntentCategory, category)


def model_scope_output_validator(value: object) -> str | None:
    """统一模型边界使用的无原文校验器。"""

    try:
        parse_model_scope_category(value)
    except QueryIntentModelOutputError:
        return "llm.invalid_structured_output"
    return None


@dataclass(frozen=True, slots=True)
class QueryIntentDecision:
    """节点输出的规范化决策；所有用户文案均由类别固定映射产生。"""

    category: QueryIntentCategory
    source: str
    reason_codes: tuple[str, ...]
    rule_ids: tuple[str, ...]
    original_query_hash: str
    standalone_query_hash: str
    model_used: bool = False
    model_degraded: bool = False
    contract_version: str = QUERY_INTENT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.category not in QUERY_INTENT_CATEGORIES:
            raise ValueError(f"无效问数意图类别：{self.category}")
        if not self.original_query_hash or not self.standalone_query_hash:
            raise ValueError("问数意图合同必须包含原始问题与独立问题哈希")

    @property
    def decision(self) -> QueryIntentDecisionType:
        return _CATEGORY_CONTRACT[self.category][0]

    @property
    def response_kind(self) -> QueryIntentResponseKind | None:
        return _CATEGORY_CONTRACT[self.category][1]

    @property
    def message(self) -> str:
        return _CATEGORY_CONTRACT[self.category][2]

    @property
    def reason_code(self) -> str:
        """返回供评测和 UI 使用的封闭原因码。"""

        if self.category == "forbidden_operation" and any(
            "prompt_injection" in item for item in self.reason_codes
        ):
            return "prompt_injection"
        return self.category

    def privacy_summary(self) -> dict[str, Any]:
        """返回 Query State、SSE 和审计共用的唯一安全表示。"""

        return {
            "schema_version": INTENT_RESPONSE_SCHEMA_VERSION,
            "contract_version": self.contract_version,
            "category": self.category,
            "decision": self.decision,
            "response_kind": self.response_kind,
            "reason_code": self.reason_code,
            "message": self.message,
            "suggestions": list(_CATEGORY_SUGGESTIONS[self.category]),
            "source": str(self.source)[:32],
            "reason_codes": list(_bounded_codes(self.reason_codes, limit=8)),
            "rule_ids": list(_bounded_codes(self.rule_ids, limit=8)),
            "query_hashes": {
                "original": self.original_query_hash,
                "standalone": self.standalone_query_hash,
            },
            "model_used": bool(self.model_used),
            "model_degraded": bool(self.model_degraded),
            "raw_text_logged": False,
        }

    def intent_response(self) -> dict[str, Any]:
        """终止分支复用完整规范字段，避免 UI 自行推断类别。"""

        return self.privacy_summary() if self.decision == "respond" else {}


def build_query_intent_decision(
    *,
    category: QueryIntentCategory,
    original_query: object,
    standalone_query: object,
    source: str,
    reason_codes: Sequence[object],
    rule_ids: Sequence[object],
    model_used: bool = False,
    model_degraded: bool = False,
) -> QueryIntentDecision:
    """从请求局部原文构造仅含哈希的版本化决策。"""

    original = normalize_query_text(original_query)
    standalone = normalize_query_text(standalone_query) or original
    return QueryIntentDecision(
        category=category,
        source=source,
        reason_codes=_bounded_codes(reason_codes, limit=8),
        rule_ids=_bounded_codes(rule_ids, limit=8),
        original_query_hash=query_text_hash(original),
        standalone_query_hash=query_text_hash(standalone),
        model_used=model_used,
        model_degraded=model_degraded,
    )


__all__ = [
    "INTENT_RESPONSE_SCHEMA_VERSION",
    "MODEL_SCOPE_CATEGORIES",
    "QUERY_INTENT_CATEGORIES",
    "QUERY_INTENT_CONTRACT_VERSION",
    "QueryIntentCategory",
    "QueryIntentDecision",
    "QueryIntentDecisionType",
    "QueryIntentModelOutputError",
    "QueryIntentResponseKind",
    "build_query_intent_decision",
    "model_scope_output_validator",
    "parse_model_scope_category",
    "query_text_hash",
]
