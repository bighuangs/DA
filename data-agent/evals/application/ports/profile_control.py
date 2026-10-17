"""生产运行 profile 的最小控制端口。"""

from __future__ import annotations

from typing import Protocol


class RuntimeProfileControl(Protocol):
    def status(self) -> dict: ...

    def rollback(self) -> dict: ...
