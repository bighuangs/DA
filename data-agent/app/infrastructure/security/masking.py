"""把进程配置适配为纯领域脱敏策略。"""

from __future__ import annotations

import os
from collections.abc import Sequence

from app.config.app_config import app_config
from app.domain.privacy.masking import (
    MaskingPolicy,
    MaskingRule,
    mask_free_text as _mask_free_text,
    mask_rows as _mask_rows,
    mask_sql_literals as _mask_sql_literals,
    masking_rules_fingerprint as _masking_rules_fingerprint,
)


def current_masking_policy() -> MaskingPolicy:
    """冻结仓储调用使用的配置快照。"""

    return MaskingPolicy(
        enabled=bool(app_config.masking.enabled),
        heuristic_enabled=bool(app_config.masking.heuristic_enabled),
        default_strategy=str(app_config.masking.default_strategy),
        hash_salt=str(os.getenv("MASKING_HASH_SALT") or app_config.auth.secret_key),
    )


def masking_rules_fingerprint(rules: Sequence[MaskingRule] | None = None) -> str:
    return _masking_rules_fingerprint(current_masking_policy(), rules)


def mask_free_text(value: object, *, max_chars: int = 2000) -> str:
    return _mask_free_text(value, policy=current_masking_policy(), max_chars=max_chars)


def mask_rows(
    rows: list[dict],
    columns: list[str] | None = None,
    rules: Sequence[MaskingRule] | None = None,
) -> list[dict]:
    return _mask_rows(
        rows,
        columns=columns,
        rules=rules,
        policy=current_masking_policy(),
    )


def mask_sql_literals(sql: str) -> str:
    return _mask_sql_literals(sql, policy=current_masking_policy())


__all__ = [
    "MaskingRule",
    "current_masking_policy",
    "mask_free_text",
    "mask_rows",
    "mask_sql_literals",
    "masking_rules_fingerprint",
]
