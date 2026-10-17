"""跨层可用的查询链路采集合同。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Protocol


class QueryTraceCollectorContract(Protocol):
    """持久化映射器所需的最小结构合同。"""

    trace_id: str
    error: str | None
    started_at: datetime
    finished_at: datetime | None
    follow_up_action: str | None
    follow_up_summary: str | None
    steps: Sequence[Any]

    def finish(self, status: str) -> Mapping[str, Any]:
        """冻结并返回可持久化的链路摘要。"""


__all__ = ["QueryTraceCollectorContract"]
