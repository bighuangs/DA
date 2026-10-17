"""用户反馈采集与后台查询用例。"""

from __future__ import annotations

from app.application.ports.feedback import FeedbackStore


class FeedbackService:
    """在 HTTP 之外编排反馈读写，保持 Router 仅负责协议转换。"""

    def __init__(self, store: FeedbackStore) -> None:
        self._store = store

    async def submit(
        self,
        *,
        chat_id: str | None,
        query: str,
        sql: str | None,
        score: int,
        comment: str | None,
    ) -> dict[str, object]:
        return await self._store.create(
            chat_id=chat_id,
            query=query,
            sql=sql,
            score=score,
            comment=comment,
        )

    async def list_feedback(
        self,
        *,
        score: int | None,
        limit: int,
    ) -> list[dict[str, object]]:
        return await self._store.list(score=score, limit=limit)


__all__ = ["FeedbackService"]
