"""用户反馈用例的持久化端口。"""

from __future__ import annotations

from typing import Protocol


class FeedbackStore(Protocol):
    """反馈用例所需的最小读写合同。"""

    async def create(self, **values: object) -> dict[str, object]: ...

    async def list(
        self,
        *,
        score: int | None,
        limit: int,
    ) -> list[dict[str, object]]: ...


__all__ = ["FeedbackStore"]
