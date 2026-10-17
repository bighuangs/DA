"""反馈 Application 用例回归。"""

from __future__ import annotations

import pytest

from app.application.services.feedback_service import FeedbackService


class _Store:
    def __init__(self) -> None:
        self.created: dict[str, object] | None = None

    async def create(self, **values: object) -> dict[str, object]:
        self.created = dict(values)
        return {"id": 7, **values}

    async def list(self, *, score: int | None, limit: int):
        return [{"id": 7, "score": score, "limit": limit}]


@pytest.mark.asyncio
async def test_feedback_service_owns_store_orchestration() -> None:
    store = _Store()
    service = FeedbackService(store)

    created = await service.submit(
        chat_id="chat-1",
        query="销售额",
        sql="SELECT 1",
        score=-1,
        comment="结果不符",
    )
    listed = await service.list_feedback(score=-1, limit=20)

    assert created["id"] == 7
    assert store.created == {
        "chat_id": "chat-1",
        "query": "销售额",
        "sql": "SELECT 1",
        "score": -1,
        "comment": "结果不符",
    }
    assert listed == [{"id": 7, "score": -1, "limit": 20}]
