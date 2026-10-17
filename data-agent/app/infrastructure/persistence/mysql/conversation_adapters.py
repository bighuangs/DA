"""会话 Application ports 的 MySQL/SQLAlchemy 适配器。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.application.settings.conversation import ConversationMemorySettings
from app.infrastructure.persistence.mysql.records.conversation_memory_outbox_mysql import (
    ConversationMemoryOutboxMySQL,
)
from app.infrastructure.persistence.mysql.records.conversation_summary_segment_mysql import (
    ConversationSummarySegmentMySQL,
)
from app.infrastructure.persistence.mysql.repositories.chat_history_repository import (
    ChatHistoryRepository,
)
from app.infrastructure.persistence.mysql.repositories.conversation_history_repository import (
    ConversationHistoryRepository,
)
from app.infrastructure.persistence.mysql.repositories.conversation_memory_repository import (
    ConversationMemoryRepository,
)
from app.infrastructure.persistence.mysql.repositories.conversation_summary_repository import (
    ConversationSummaryRepository,
)
from app.infrastructure.persistence.mysql.repositories.data_context_repository import (
    ConversationDataContextRepository,
)


SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]


class SqlConversationDataContextStore:
    """为每次数据上下文操作提供独立短事务。"""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        owner_id: str,
        workspace_id: str,
    ) -> None:
        self._session_factory = session_factory
        self._owner_id = owner_id
        self._workspace_id = workspace_id

    def _repository(self, session: AsyncSession) -> ConversationDataContextRepository:
        return ConversationDataContextRepository(
            session,
            owner_id=self._owner_id,
            workspace_id=self._workspace_id,
        )

    async def get(self, chat_id: str) -> object | None:
        async with self._session_factory() as session:
            return await self._repository(session).get(chat_id)

    async def set(self, chat_id: str, **values: object) -> object:
        async with self._session_factory() as session:
            return await self._repository(session).set(chat_id, **values)

    async def reset(self, chat_id: str, *, expected_revision: int | None) -> object:
        async with self._session_factory() as session:
            return await self._repository(session).reset(
                chat_id,
                expected_revision=expected_revision,
            )

    async def snapshot_turn(
        self,
        *,
        chat_id: str,
        turn_id: str,
        revision: int,
        snapshot: dict,
    ) -> None:
        async with self._session_factory() as session:
            await self._repository(session).snapshot_turn(
                chat_id=chat_id,
                turn_id=turn_id,
                revision=revision,
                snapshot=snapshot,
            )


class SqlConversationHistoryStore:
    """在调用方持有的 SQLAlchemy 事务中实现目录与 transcript 端口。"""

    def __init__(
        self,
        session: AsyncSession,
        *,
        owner_id: str,
        workspace_id: str,
    ) -> None:
        self._session = session
        self.owner_id = owner_id
        self.workspace_id = workspace_id
        self._history = ConversationHistoryRepository(
            session,
            owner_id=owner_id,
            workspace_id=workspace_id,
        )
        self._contexts = ConversationDataContextRepository(
            session,
            owner_id=owner_id,
            workspace_id=workspace_id,
        )

    async def list_sessions(self, **options: object):
        return await self._history.list_sessions(**options)

    async def get_session(self, chat_id: str, **options: object):
        return await self._history.get_session(chat_id, **options)

    async def list_turns(self, **options: object):
        return await self._history.list_turns(**options)

    async def history_by_ids(self, ids: Iterable[int]):
        return await self._history.history_by_ids(ids)

    async def turn_snapshots(self, turn_ids: Iterable[str]):
        return await self._contexts.turn_snapshots(turn_ids)

    async def update_session(self, chat_id: str, **values: object):
        return await self._history.update_session(chat_id, **values)

    async def soft_delete(self, chat_id: str) -> bool:
        return await self._history.soft_delete(chat_id)

    async def begin_delete(self, chat_id: str) -> bool:
        return await self._history.begin_delete(chat_id)

    async def commit(self) -> None:
        await self._session.commit()

    async def refresh(self, row: object) -> None:
        await self._session.refresh(row)


class SqlConversationMemoryUnitOfWork:
    """在同一事务内暴露会话、记忆、摘要和历史 Repository。"""

    def __init__(
        self,
        session: AsyncSession,
        *,
        owner_id: str,
        workspace_id: str,
        settings: ConversationMemorySettings,
    ) -> None:
        self._session = session
        self._settings = settings
        self.memory = ConversationMemoryRepository(
            session,
            owner_id=owner_id,
            workspace_id=workspace_id,
        )
        self.summary = ConversationSummaryRepository(
            session,
            owner_id=owner_id,
            workspace_id=workspace_id,
        )
        self.history = ConversationHistoryRepository(
            session,
            owner_id=owner_id,
            workspace_id=workspace_id,
        )
        self.chat_history = ChatHistoryRepository(
            session,
            owner_id=owner_id,
            workspace_id=workspace_id,
        )

    async def commit(self) -> None:
        await self._session.commit()

    async def rollback(self) -> None:
        await self._session.rollback()

    async def summary_projection_backfill_status(self) -> dict[str, int | bool]:
        active_filter = (
            ConversationSummarySegmentMySQL.status == "active",
            ConversationSummarySegmentMySQL.deleted_at.is_(None),
        )
        active_ids = select(ConversationSummarySegmentMySQL.summary_id).where(
            *active_filter
        )
        active_count = int(
            await self._session.scalar(
                select(func.count())
                .select_from(ConversationSummarySegmentMySQL)
                .where(*active_filter)
            )
            or 0
        )
        counts: dict[str, int] = {}
        for target, version in (
            ("summary_es", self._settings.summary_es_projection_version),
            ("summary_qdrant", self._settings.summary_qdrant_projection_version),
        ):
            counts[target] = int(
                await self._session.scalar(
                    select(
                        func.count(func.distinct(ConversationMemoryOutboxMySQL.memory_id))
                    ).where(
                        ConversationMemoryOutboxMySQL.memory_id.in_(active_ids),
                        ConversationMemoryOutboxMySQL.target == target,
                        ConversationMemoryOutboxMySQL.operation == "upsert",
                        ConversationMemoryOutboxMySQL.projection_version == version,
                        ConversationMemoryOutboxMySQL.status == "completed",
                    )
                )
                or 0
            )
        return {
            "active_segments": active_count,
            "summary_es_completed": counts["summary_es"],
            "summary_qdrant_completed": counts["summary_qdrant"],
            "ready": (
                counts["summary_es"] == active_count
                and counts["summary_qdrant"] == active_count
            ),
        }


class SqlConversationMemoryPersistence:
    """按会话作用域创建显式事务 Unit of Work。"""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        owner_id: str,
        workspace_id: str,
        settings: ConversationMemorySettings,
    ) -> None:
        self._session_factory = session_factory
        self._owner_id = owner_id
        self._workspace_id = workspace_id
        self._settings = settings

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[SqlConversationMemoryUnitOfWork]:
        async with self._session_factory() as session:
            yield SqlConversationMemoryUnitOfWork(
                session,
                owner_id=self._owner_id,
                workspace_id=self._workspace_id,
                settings=self._settings,
            )


__all__ = [
    "SqlConversationDataContextStore",
    "SqlConversationHistoryStore",
    "SqlConversationMemoryPersistence",
    "SqlConversationMemoryUnitOfWork",
]
