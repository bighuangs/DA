"""会话数据上下文 scope 门禁回归。"""

from __future__ import annotations

import pytest

from app.application.ports.conversation import ConversationUnavailableError
from app.application.services.conversation_data_context_service import (
    ScopedConversationDataContextService,
)


class _Scope:
    async def exists(self, chat_id: str) -> bool:
        return chat_id == "chat-1"


class _Context:
    async def get_record(self, chat_id: str) -> dict:
        return {"chat_id": chat_id}

    async def set(self, chat_id: str, **values: object) -> dict:
        return {"chat_id": chat_id, **values}

    async def reset(self, chat_id: str, **values: object) -> dict:
        return {"chat_id": chat_id, **values}


@pytest.mark.asyncio
async def test_scoped_context_checks_conversation_before_delegating() -> None:
    service = ScopedConversationDataContextService(
        scope=_Scope(),
        context=_Context(),
    )

    assert await service.get_record("chat-1") == {"chat_id": "chat-1"}
    with pytest.raises(ConversationUnavailableError):
        await service.get_record("missing")
