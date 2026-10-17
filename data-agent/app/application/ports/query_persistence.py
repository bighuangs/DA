"""流式问数用例的会话与 Trace 持久化端口。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol

from app.foundation.tracing import QueryTraceCollectorContract


class ChatHistoryView(Protocol):
    """Application 读取历史回合所需的最小视图。"""

    query: str
    sql: str | None
    result: str | None
    row_count: int
    created_at: datetime | None
    standalone_question: str | None
    follow_up_action: str | None
    semantic_state: Mapping[str, object] | None


class QueryPersistencePort(Protocol):
    """封装问数用例的事务边界与具体记录映射。"""

    async def recent_history(
        self,
        chat_id: str,
        limit: int,
    ) -> Sequence[ChatHistoryView]: ...

    async def start_turn(
        self,
        *,
        chat_id: str,
        turn_id: str,
        query: str,
    ) -> None: ...

    async def save_successful_turn(
        self,
        *,
        chat_id: str,
        turn_id: str,
        query: str,
        sql: str,
        result: list[dict],
        standalone_question: str,
        follow_up_action: str,
        semantic_state: Mapping[str, object],
    ) -> None: ...

    async def complete_handled_turn(
        self,
        *,
        chat_id: str,
        turn_id: str,
        response: Mapping[str, object],
    ) -> None: ...

    async def fail_turn(
        self,
        *,
        chat_id: str,
        turn_id: str,
        status: str,
        summary: str,
    ) -> None: ...

    async def conversation_session(self, chat_id: str) -> object | None: ...

    async def persist_trace(
        self,
        collector: QueryTraceCollectorContract,
        *,
        query: str,
        chat_id: str,
        sql: str | None,
        status: str,
        row_count: int,
        error: str | None,
        version_snapshot: Mapping[str, object],
        cache_hit: bool,
    ) -> Mapping[str, object]: ...


class SQLSessionPort(Protocol):
    """数据源上下文统计查询所需的最小 Session 端口。"""

    async def execute(self, statement: object) -> object: ...


class DataWarehouseQueryPort(Protocol):
    """问数用例解析数据源并记录缓存事实所需的 DW 端口。"""

    last_cache_hit: bool
    session: SQLSessionPort

    async def get_schema_snapshot(self) -> Mapping[str, object]: ...


__all__ = [
    "ChatHistoryView",
    "DataWarehouseQueryPort",
    "QueryPersistencePort",
    "SQLSessionPort",
]
