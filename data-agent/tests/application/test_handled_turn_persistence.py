from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from app.application.services.conversation_memory_service import (
    ConversationMemoryService,
)
from app.application.settings.conversation import DEFAULT_MEMORY_SETTINGS


class _MemoryRepository:
    def __init__(self) -> None:
        self.status_updates: list[tuple[int, str, tuple[str, ...]]] = []
        self.messages: list[dict[str, object]] = []

    async def set_message_status(
        self,
        message_id: int,
        status: str,
        *,
        expected_statuses: tuple[str, ...],
    ) -> bool:
        self.status_updates.append((message_id, status, expected_statuses))
        return True

    async def append_message(self, **values: object) -> object:
        self.messages.append(dict(values))
        return SimpleNamespace(id=2)


class _HistoryRepository:
    def __init__(self) -> None:
        self.completed: list[dict[str, object]] = []

    async def get_session(self, _chat_id: str, **_options: object) -> object:
        return SimpleNamespace(status="active")

    async def complete_handled_turn(self, **values: object) -> object:
        self.completed.append(dict(values))
        return SimpleNamespace(status="succeeded")


class _UnitOfWork:
    def __init__(self) -> None:
        self.memory = _MemoryRepository()
        self.history = _HistoryRepository()
        self.committed = False

    async def commit(self) -> None:
        self.committed = True


class _Persistence:
    def __init__(self, uow: _UnitOfWork) -> None:
        self.uow = uow

    @asynccontextmanager
    async def transaction(self):
        yield self.uow


@pytest.mark.asyncio
async def test_complete_handled_turn_does_not_create_query_memory() -> None:
    uow = _UnitOfWork()
    service = object.__new__(ConversationMemoryService)
    service.persistence = _Persistence(uow)
    service.settings = DEFAULT_MEMORY_SETTINGS

    evidence = await service.complete_handled_turn(
        1,
        chat_id="chat-1",
        trace_id="trace-1",
        response={"message": "当前助手仅支持只读业务数据查询。"},
    )

    assert uow.committed is True
    assert uow.memory.status_updates == [(1, "success", ("pending",))]
    assert uow.memory.messages[0]["role"] == "assistant"
    assert uow.history.completed[0]["turn_id"] == "trace-1"
    assert evidence == {
        "mysql_committed": True,
        "replay_fidelity": "message_only",
        "memory_items_created": 0,
        "outbox_events_created": 0,
    }
