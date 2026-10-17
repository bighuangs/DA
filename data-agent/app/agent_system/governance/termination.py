"""统一终止原因与判定顺序。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping


class TerminationReason(StrEnum):
    CONTINUE = "continue"
    COMPLETED = "completed"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    CANCELLED = "cancelled"
    BUDGET_EXHAUSTED = "budget_exhausted"
    PERMISSION_DENIED = "permission_denied"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class TerminationDecision:
    terminate: bool
    reason: TerminationReason
    retryable: bool = False


@dataclass(frozen=True, slots=True)
class TerminationPolicy:
    """用稳定优先级把运行时信号归并成唯一终止原因。"""

    terminal_statuses: frozenset[str] = frozenset(
        {"succeeded", "failed", "cancelled", "timed_out"}
    )

    def decide(
        self,
        state: Mapping[str, object],
        *,
        cancelled: bool = False,
        deadline_exceeded: bool = False,
        budget_exhausted: bool = False,
        permission_denied: bool = False,
    ) -> TerminationDecision:
        if cancelled:
            return TerminationDecision(True, TerminationReason.CANCELLED)
        if deadline_exceeded:
            return TerminationDecision(True, TerminationReason.DEADLINE_EXCEEDED, True)
        if permission_denied:
            return TerminationDecision(True, TerminationReason.PERMISSION_DENIED)
        if budget_exhausted:
            return TerminationDecision(True, TerminationReason.BUDGET_EXHAUSTED, True)
        status = str(state.get("status") or "")
        if status == "succeeded":
            return TerminationDecision(True, TerminationReason.COMPLETED)
        if status in self.terminal_statuses:
            return TerminationDecision(True, TerminationReason.ERROR)
        if state.get("error"):
            return TerminationDecision(True, TerminationReason.ERROR)
        return TerminationDecision(False, TerminationReason.CONTINUE)
