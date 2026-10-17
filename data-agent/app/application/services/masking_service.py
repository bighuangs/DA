"""应用层脱敏门面；具体规则位于纯领域模块。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from app.domain.privacy.masking import (
    MaskingPolicy,
    MaskingRule,
    compile_rules as _compile_rules,
    mask_free_text as _mask_free_text,
    mask_rows as _mask_rows,
    mask_sql_literals as _mask_sql_literals,
    mask_value as _mask_value,
    masking_rules_fingerprint as _masking_rules_fingerprint,
    policy_manifest,
)


_DEFAULT_POLICY = MaskingPolicy()


def current_masking_policy() -> MaskingPolicy:
    """返回无外部配置时使用的安全默认策略。"""

    return _DEFAULT_POLICY


def masking_policy(*, policy: MaskingPolicy = _DEFAULT_POLICY) -> dict:
    """返回管理页面可展示的全局基线策略。"""

    return policy_manifest(policy)


def masking_rules_fingerprint(
    rules: Sequence[MaskingRule] | None = None,
    *,
    policy: MaskingPolicy = _DEFAULT_POLICY,
) -> str:
    """返回与显式策略绑定的缓存作用域指纹。"""

    return _masking_rules_fingerprint(policy, rules)


def compile_rules(
    items: Iterable[Mapping | MaskingRule | object],
) -> tuple[MaskingRule, ...]:
    """把边界记录编译为不可变规则快照。"""

    return _compile_rules(items)


def mask_value(
    value: Any,
    *,
    column: str,
    rule: MaskingRule | None = None,
    inferred_kind: str | None = None,
    policy: MaskingPolicy = _DEFAULT_POLICY,
) -> Any:
    """使用显式策略对单个值脱敏。"""

    return _mask_value(
        value,
        column=column,
        policy=policy,
        rule=rule,
        inferred_kind=inferred_kind,
    )


def mask_free_text(
    value: Any,
    *,
    max_chars: int = 2000,
    policy: MaskingPolicy = _DEFAULT_POLICY,
) -> str:
    """使用显式策略脱敏自由文本。"""

    return _mask_free_text(value, policy=policy, max_chars=max_chars)


def mask_rows(
    rows: list[dict],
    columns: list[str] | None = None,
    rules: Sequence[MaskingRule] | None = None,
    *,
    policy: MaskingPolicy = _DEFAULT_POLICY,
) -> list[dict]:
    """使用显式策略脱敏结果行。"""

    return _mask_rows(
        rows,
        columns=columns,
        rules=rules,
        policy=policy,
    )


def mask_sql_literals(
    sql: str,
    *,
    policy: MaskingPolicy = _DEFAULT_POLICY,
) -> str:
    """使用显式策略脱敏 SQL 字面量。"""

    return _mask_sql_literals(sql, policy=policy)


__all__ = [
    "MaskingPolicy",
    "MaskingRule",
    "compile_rules",
    "current_masking_policy",
    "mask_free_text",
    "mask_rows",
    "mask_sql_literals",
    "mask_value",
    "masking_policy",
    "masking_rules_fingerprint",
]
