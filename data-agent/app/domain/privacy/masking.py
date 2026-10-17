"""不依赖配置系统或持久化实现的数据脱敏规则。"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class MaskingPolicy:
    """一次调用使用的不可变脱敏策略快照。"""

    enabled: bool = True
    heuristic_enabled: bool = True
    default_strategy: str = "mask"
    hash_salt: str = ""


@dataclass(frozen=True, slots=True)
class MaskingRule:
    """按字段名匹配的显式脱敏规则。"""

    column_pattern: str
    data_kind: str = "generic"
    strategy: str = "mask"
    enabled: bool = True
    priority: int = 100
    description: str = ""


_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_PHONE_RE = re.compile(r"^(?:\+?86[- ]?)?1[3-9]\d{9}$")
_ID_CARD_RE = re.compile(r"^(?:\d{15}|\d{17}[\dXx])$")
_BANK_RE = re.compile(r"^\d{12,19}$")
_FREE_EMAIL_RE = re.compile(r"(?<![\w.-])([^\s@]+@[^\s@]+\.[^\s@]+)(?![\w.-])")
_FREE_PHONE_RE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?(1[3-9]\d{9})(?!\d)")
_FREE_ID_RE = re.compile(r"(?<!\d)(\d{17}[\dXx]|\d{15})(?!\d)")

_KIND_TOKENS = {
    "phone": ("phone", "mobile", "telephone", "手机号", "手机", "电话"),
    "email": ("email", "邮箱", "电子邮件"),
    "id_card": ("id_card", "idcard", "身份证", "证件号"),
    "bank_card": ("bank_card", "bankcard", "银行卡", "卡号"),
    "name": ("name", "姓名", "客户名", "联系人", "收件人"),
}
_NAME_FIELDS = {"name", "full_name", "customer_name", "contact_name", "姓名", "客户姓名"}
_BUSINESS_IDENTIFIER_TOKENS = (
    "order", "product", "invoice", "serial", "sku", "code", "number", "编号", "序号", "订单"
)


def policy_manifest(policy: MaskingPolicy) -> dict:
    """返回可展示的策略说明，不包含 hash salt。"""

    return {
        "enabled": policy.enabled,
        "heuristic_enabled": policy.heuristic_enabled,
        "default_strategy": policy.default_strategy,
        "explicit_override_strategy": "keep",
        "builtins": [
            {
                "source": "column_name",
                "data_kind": "name",
                "patterns": sorted(_NAME_FIELDS),
                "description": "仅精确匹配高置信姓名字段",
            },
            *[
                {
                    "source": "column_name",
                    "data_kind": kind,
                    "patterns": list(tokens),
                    "description": "按完整字段词元匹配",
                }
                for kind, tokens in _KIND_TOKENS.items()
                if kind != "name"
            ],
            {
                "source": "value_format",
                "data_kind": "email|phone|id_card|bank_card",
                "patterns": [],
                "description": "前 20 个非空样本至少 2 个且命中率不低于 80%",
            },
        ],
    }


def masking_rules_fingerprint(
    policy: MaskingPolicy,
    rules: Sequence[MaskingRule] | None = None,
) -> str:
    """计算影响缓存隔离的策略指纹，不把 salt 写入产物。"""

    payload = {
        "enabled": policy.enabled,
        "heuristic_enabled": policy.heuristic_enabled,
        "default_strategy": policy.default_strategy,
        "rules": [asdict(rule) for rule in (rules or ())],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def compile_rules(items: Iterable[Mapping | MaskingRule | object]) -> tuple[MaskingRule, ...]:
    """把边界记录编译为按优先级排序的不可变规则快照。"""

    loaded: list[MaskingRule] = []
    for item in items:
        if isinstance(item, MaskingRule):
            rule = item
        else:
            def value(key: str, default: Any = None) -> Any:
                if isinstance(item, Mapping):
                    return item.get(key, default)
                return getattr(item, key, default)

            rule = MaskingRule(
                column_pattern=str(value("column_pattern") or "*"),
                data_kind=str(value("data_kind") or "generic"),
                strategy=str(value("strategy") or "mask"),
                enabled=bool(value("enabled", True)),
                priority=int(value("priority", 100)),
                description=str(value("description") or ""),
            )
        if rule.enabled:
            loaded.append(rule)
    return tuple(sorted(loaded, key=lambda item: item.priority))


def _normalized_names(column: str) -> set[str]:
    value = str(column or "").strip().lower().replace("`", "")
    names = {value}
    if "." in value:
        names.add(value.rsplit(".", 1)[-1])
    return names


def _token_matches_name(name: str, token: str) -> bool:
    if re.search(r"[\u4e00-\u9fff]", token):
        return token in name
    return bool(re.search(rf"(?:^|[^a-z0-9]){re.escape(token)}(?:$|[^a-z0-9])", name))


def _kind_from_name(column: str) -> str | None:
    names = _normalized_names(column)
    for kind, tokens in _KIND_TOKENS.items():
        if kind == "name":
            if any(name in _NAME_FIELDS for name in names):
                return kind
            continue
        if any(_token_matches_name(name, token) for name in names for token in tokens):
            return kind
    return None


def _rule_for(column: str, rules: Sequence[MaskingRule] | None = None) -> MaskingRule | None:
    for rule in rules or ():
        pattern = rule.column_pattern.strip().lower().replace("`", "")
        if any(fnmatch.fnmatch(name, pattern) for name in _normalized_names(column)):
            return rule
    return None


def _looks_like_kind(value: Any, kind: str) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if kind == "email":
        return bool(_EMAIL_RE.fullmatch(text))
    if kind == "phone":
        return bool(_PHONE_RE.fullmatch(text.replace(" ", "")))
    if kind == "id_card":
        if not _ID_CARD_RE.fullmatch(text):
            return False
        try:
            datetime.strptime(
                text[6:14] if len(text) == 18 else text[6:12],
                "%Y%m%d" if len(text) == 18 else "%y%m%d",
            )
        except ValueError:
            return False
        if len(text) == 18:
            weights = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
            checks = "10X98765432"
            checksum = sum(int(char) * weight for char, weight in zip(text[:17], weights))
            return checks[checksum % 11] == text[-1].upper()
        return True
    if kind == "bank_card":
        digits = text.replace(" ", "")
        if not _BANK_RE.fullmatch(digits):
            return False
        total = 0
        parity = len(digits) % 2
        for index, char in enumerate(digits):
            number = int(char)
            if index % 2 == parity:
                number *= 2
                if number > 9:
                    number -= 9
            total += number
        return total % 10 == 0
    return False


def _infer_kind(column: str, values: list[Any], policy: MaskingPolicy) -> str | None:
    if not policy.heuristic_enabled:
        return None
    named = _kind_from_name(column)
    if named:
        return named
    samples = [value for value in values[:20] if value is not None and str(value).strip()]
    if len(samples) < 2:
        return None
    names = _normalized_names(column)
    looks_like_business_id = any(
        _token_matches_name(name, token)
        for name in names
        for token in _BUSINESS_IDENTIFIER_TOKENS
    )
    for kind in ("email", "phone", "id_card", "bank_card"):
        if looks_like_business_id and kind in {"id_card", "bank_card"}:
            continue
        matches = sum(_looks_like_kind(value, kind) for value in samples)
        if matches >= 2 and matches / len(samples) >= 0.8:
            return kind
    return None


def _mask_text(value: Any, kind: str, strategy: str, policy: MaskingPolicy) -> Any:
    if value is None:
        return None
    if strategy == "keep":
        return value
    if strategy == "null":
        return None
    if strategy == "hash":
        return hashlib.sha256(f"{policy.hash_salt}{value}".encode("utf-8")).hexdigest()[:12]
    text = str(value)
    if strategy == "keep_last4":
        return "*" * max(0, len(text) - 4) + text[-4:]
    if kind == "phone":
        compact = text.replace(" ", "")
        return compact[:3] + "****" + compact[-4:] if len(compact) >= 7 else "***"
    if kind == "email" and "@" in text:
        local, domain = text.split("@", 1)
        return (local[:1] + "***" if local else "***") + "@" + domain
    if kind == "id_card":
        return text[:6] + "********" + text[-4:] if len(text) >= 10 else "***"
    if kind == "bank_card":
        return "**** **** **** " + text.replace(" ", "")[-4:]
    if kind == "name":
        return text[:1] + "*" * max(1, len(text) - 1)
    return "***"


def mask_value(
    value: Any,
    *,
    column: str,
    policy: MaskingPolicy,
    rule: MaskingRule | None = None,
    inferred_kind: str | None = None,
) -> Any:
    """按显式规则或高置信类型对单个值脱敏。"""

    selected = rule or _rule_for(column)
    kind = selected.data_kind if selected else inferred_kind
    if not selected and not kind and policy.heuristic_enabled:
        kind = _kind_from_name(column)
    if not selected and not kind:
        return value
    strategy = selected.strategy if selected else policy.default_strategy
    return _mask_text(value, kind or "generic", strategy, policy)


def mask_free_text(value: Any, *, policy: MaskingPolicy, max_chars: int = 2000) -> str:
    """仅按高置信格式脱敏自由文本，不猜测人名。"""

    text = str(value or "").strip()[:max(1, int(max_chars))]
    if not policy.enabled or not text:
        return text
    text = _FREE_EMAIL_RE.sub(
        lambda match: str(_mask_text(match.group(1), "email", policy.default_strategy, policy)),
        text,
    )
    text = _FREE_PHONE_RE.sub(
        lambda match: str(_mask_text(match.group(1), "phone", policy.default_strategy, policy)),
        text,
    )
    text = _FREE_ID_RE.sub(
        lambda match: str(_mask_text(match.group(1), "id_card", policy.default_strategy, policy)),
        text,
    )
    return text


def mask_rows(
    rows: list[dict],
    columns: list[str] | None = None,
    rules: Sequence[MaskingRule] | None = None,
    *,
    policy: MaskingPolicy,
) -> list[dict]:
    """返回脱敏后的新列表，不修改调用方提供的行对象。"""

    if not policy.enabled or not rows:
        return rows
    column_names = columns or list(rows[0].keys())
    kinds = {
        column: _infer_kind(column, [row.get(column) for row in rows], policy)
        for column in column_names
    }
    result: list[dict] = []
    for row in rows:
        masked_row: dict = {}
        for key, value in row.items():
            rule = _rule_for(key, rules)
            inferred_kind = kinds.get(key)
            masked_row[key] = (
                mask_value(
                    value,
                    column=key,
                    policy=policy,
                    rule=rule,
                    inferred_kind=inferred_kind,
                )
                if rule or inferred_kind
                else value
            )
        result.append(masked_row)
    return result


def mask_sql_literals(sql: str, *, policy: MaskingPolicy) -> str:
    """脱敏 SQL 字面量中的明显 PII，同时保留 SQL 结构。"""

    if not sql or not policy.enabled:
        return sql

    def replace(match: re.Match) -> str:
        quote, value = match.group(1), match.group(2)
        kind = next(
            (
                candidate
                for candidate in ("email", "phone", "id_card")
                if _looks_like_kind(value, candidate)
            ),
            None,
        )
        if not kind:
            return match.group(0)
        masked = _mask_text(value, kind, policy.default_strategy, policy)
        return quote + str(masked).replace(quote, quote + quote) + quote

    return re.sub(r"(['\"])(.*?)\1", replace, sql, flags=re.DOTALL)


__all__ = [
    "MaskingPolicy",
    "MaskingRule",
    "compile_rules",
    "mask_free_text",
    "mask_rows",
    "mask_sql_literals",
    "mask_value",
    "masking_rules_fingerprint",
    "policy_manifest",
]
