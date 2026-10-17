"""流式问数用例的 MySQL 持久化适配器。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.application.ports.query_persistence import (
    ChatHistoryView,
    QueryPersistencePort,
)
from app.foundation.tracing import QueryTraceCollectorContract
from app.infrastructure.clients.mysql_client_manager import meta_mysql_client_manager
from app.infrastructure.persistence.mysql.repositories.chat_history_repository import (
    ChatHistoryRepository,
)
from app.infrastructure.persistence.mysql.repositories.conversation_history_repository import (
    ConversationHistoryRepository,
)
from app.infrastructure.persistence.mysql.repositories.query_trace_repository import (
    QueryTraceRepository,
    build_trace_records,
)


class MysqlQueryPersistence(QueryPersistencePort):
    """使用请求读 Session 与独立写事务实现 QueryPersistencePort。"""

    def __init__(
        self,
        read_session: AsyncSession,
        *,
        owner_id: str,
        workspace_id: str,
    ) -> None:
        self._read_session = read_session
        self._owner_id = owner_id
        self._workspace_id = workspace_id

    def _history(self, session: AsyncSession) -> ConversationHistoryRepository:
        return ConversationHistoryRepository(
            session,
            owner_id=self._owner_id,
            workspace_id=self._workspace_id,
        )

    def _chat(self, session: AsyncSession) -> ChatHistoryRepository:
        return ChatHistoryRepository(
            session,
            owner_id=self._owner_id,
            workspace_id=self._workspace_id,
        )

    async def recent_history(
        self,
        chat_id: str,
        limit: int,
    ) -> Sequence[ChatHistoryView]:
        authority = self._history(self._read_session)
        return await self._chat(self._read_session).get_recent_history(
            chat_id,
            limit,
            after=await authority.context_reset_at(chat_id),
        )

    async def start_turn(
        self,
        *,
        chat_id: str,
        turn_id: str,
        query: str,
    ) -> None:
        async with meta_mysql_client_manager.session_factory() as session:
            await self._history(session).start_turn(
                chat_id=chat_id,
                turn_id=turn_id,
                user_text=query,
                trace_id=turn_id,
            )
            await session.commit()

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
    ) -> None:
        async with meta_mysql_client_manager.session_factory() as session:
            history = await self._chat(session).save_turn(
                chat_id,
                query,
                sql,
                result,
                standalone_question=standalone_question,
                follow_up_action=follow_up_action,
                semantic_state=dict(semantic_state),
                commit=False,
            )
            await self._history(session).complete_turn(
                chat_id=chat_id,
                turn_id=turn_id,
                source_history_id=int(history.id),
                row_count=len(result),
            )
            await session.commit()

    async def complete_handled_turn(
        self,
        *,
        chat_id: str,
        turn_id: str,
        response: Mapping[str, object],
    ) -> None:
        """将无 SQL 的策略响应诚实记录为已处理回合。"""

        assistant_summary = str(response.get("message") or "").strip()
        if not assistant_summary:
            raise ValueError("已处理回合必须包含用户可见消息")
        async with meta_mysql_client_manager.session_factory() as session:
            await self._history(session).complete_handled_turn(
                chat_id=chat_id,
                turn_id=turn_id,
                assistant_summary=assistant_summary,
            )
            await session.commit()

    async def fail_turn(
        self,
        *,
        chat_id: str,
        turn_id: str,
        status: str,
        summary: str,
    ) -> None:
        async with meta_mysql_client_manager.session_factory() as session:
            await self._history(session).fail_turn(
                chat_id=chat_id,
                turn_id=turn_id,
                status=status,
                assistant_summary=summary,
            )
            await session.commit()

    async def conversation_session(self, chat_id: str) -> object | None:
        return await self._history(self._read_session).get_session(chat_id)

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
    ) -> Mapping[str, object]:
        summary, trace, steps, usages = build_trace_records(
            collector,
            query=query,
            chat_id=chat_id,
            sql=sql,
            status=status,
            row_count=row_count,
            error=error,
            owner_id=self._owner_id,
            workspace_id=self._workspace_id,
            version_snapshot=dict(version_snapshot),
            cache_hit=cache_hit,
        )
        async with meta_mysql_client_manager.session_factory() as session:
            await QueryTraceRepository(
                session,
                owner_id=self._owner_id,
                workspace_id=self._workspace_id,
            ).save_trace(trace, steps, usages)
        return summary


__all__ = ["MysqlQueryPersistence"]
