"""可注入时钟，避免业务逻辑直接绑定系统时间。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    """运行时使用的最小时钟接口。"""

    def now(self) -> datetime:
        """返回带时区的当前时间。"""


class SystemClock:
    """使用 UTC 系统时间的生产实现。"""

    def now(self) -> datetime:
        return datetime.now(UTC)
