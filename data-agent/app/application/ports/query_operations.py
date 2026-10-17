"""手动 SQL 与 Trace 查询用例的出站端口。"""

from __future__ import annotations

from typing import Protocol

from app.application.ports.conversation import ConversationScopeReader


class ManualQueryExecutor(Protocol):
    last_cache_hit: bool

    async def validate_sql(self, sql: str) -> object: ...

    async def execute_sql(self, sql: str) -> list[dict[str, object]]: ...


class QueryTraceReader(Protocol):
    async def list_by_chat_id(
        self,
        chat_id: str,
        limit: int,
    ) -> list[dict[str, object]]: ...


__all__ = [
    "ConversationScopeReader",
    "ManualQueryExecutor",
    "QueryTraceReader",
]
