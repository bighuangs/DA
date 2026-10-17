"""流式问数用例的不可变配置快照。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QueryServiceSettings:
    """由 Bootstrap 从进程配置冻结的 QueryService 策略。"""

    history_limit: int
    analysis_enabled_default: bool
    memory_enabled: bool

    def __post_init__(self) -> None:
        if self.history_limit < 1 or self.history_limit > 100:
            raise ValueError("history_limit 必须位于 1..100")


__all__ = ["QueryServiceSettings"]
