from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.persistence.mysql.records.conversation_data_context_mysql import (
    ConversationDataContextMySQL,
    ConversationTurnDataContextMySQL,
    DatasetResourceLedgerMySQL,
    SourceMappingMySQL,
)


class ConversationDataContextRepository:
    def __init__(self, session: AsyncSession, *, owner_id: str, workspace_id: str):
        self.session = session
        self.owner_id = owner_id
        self.workspace_id = workspace_id

    async def get(self, chat_id: str) -> ConversationDataContextMySQL | None:
        result = await self.session.execute(
            select(ConversationDataContextMySQL)
            .where(ConversationDataContextMySQL.owner_id == self.owner_id)
            .where(ConversationDataContextMySQL.workspace_id == self.workspace_id)
            .where(ConversationDataContextMySQL.chat_id == chat_id)
        )
        return result.scalar_one_or_none()

    async def set(
        self,
        chat_id: str,
        *,
        source_mode: str,
        dataset_ids: list[str],
        comparison_intent: str | None,
        mapping_confirmation_id: str | None,
        expected_revision: int | None = None,
    ) -> ConversationDataContextMySQL:
        item = await self.get(chat_id)
        if item is None:
            if expected_revision not in (None, 0):
                raise ValueError("数据上下文版本已变化，请刷新后重试")
            item = ConversationDataContextMySQL(
                owner_id=self.owner_id,
                workspace_id=self.workspace_id,
                chat_id=chat_id,
                revision=1,
            )
            self.session.add(item)
        else:
            if expected_revision is not None and item.revision != expected_revision:
                raise ValueError("数据上下文版本已变化，请刷新后重试")
            item.revision += 1
        item.source_mode = source_mode
        item.dataset_ids = list(dataset_ids)
        item.comparison_intent = comparison_intent
        item.mapping_confirmation_id = mapping_confirmation_id
        item.invalidation_reason = None
        await self.session.commit()
        await self.session.refresh(item)
        return item

    async def reset(
        self, chat_id: str, *, expected_revision: int | None = None
    ) -> ConversationDataContextMySQL:
        return await self.set(
            chat_id,
            source_mode="dw_only",
            dataset_ids=[],
            comparison_intent=None,
            mapping_confirmation_id=None,
            expected_revision=expected_revision,
        )

    async def snapshot_turn(
        self, *, chat_id: str, turn_id: str, revision: int, snapshot: dict
    ) -> None:
        self.session.add(
            ConversationTurnDataContextMySQL(
                owner_id=self.owner_id,
                workspace_id=self.workspace_id,
                chat_id=chat_id,
                turn_id=turn_id,
                revision=revision,
                context_snapshot=snapshot,
            )
        )
        await self.session.commit()

    async def turn_snapshots(
        self, turn_ids
    ) -> dict[str, dict]:
        ids = [str(value) for value in turn_ids if value]
        if not ids:
            return {}
        result = await self.session.execute(
            select(ConversationTurnDataContextMySQL)
            .where(ConversationTurnDataContextMySQL.owner_id == self.owner_id)
            .where(
                ConversationTurnDataContextMySQL.workspace_id
                == self.workspace_id
            )
            .where(ConversationTurnDataContextMySQL.turn_id.in_(ids))
        )
        return {
            str(item.turn_id): dict(item.context_snapshot or {})
            for item in result.scalars().all()
        }


class SourceMappingRepository:
    def __init__(self, session: AsyncSession, *, owner_id: str, workspace_id: str):
        self.session = session
        self.owner_id = owner_id
        self.workspace_id = workspace_id

    async def get(self, mapping_id: str) -> SourceMappingMySQL | None:
        result = await self.session.execute(
            select(SourceMappingMySQL)
            .where(SourceMappingMySQL.id == mapping_id)
            .where(SourceMappingMySQL.owner_id == self.owner_id)
            .where(SourceMappingMySQL.workspace_id == self.workspace_id)
        )
        return result.scalar_one_or_none()

    async def create(self, item: SourceMappingMySQL) -> SourceMappingMySQL:
        self.session.add(item)
        await self.session.commit()
        await self.session.refresh(item)
        return item

    async def reject(
        self, mapping_id: str, reason: str | None
    ) -> SourceMappingMySQL | None:
        item = await self.get(mapping_id)
        if item is None:
            return None
        item.status = "rejected"
        item.rejection_reason = str(reason or "")[:1000] or None
        await self.session.commit()
        await self.session.refresh(item)
        return item


class DatasetResourceLedgerRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, item: DatasetResourceLedgerMySQL) -> None:
        self.session.add(item)
        await self.session.commit()

    async def for_version(
        self, version_id: str
    ) -> list[DatasetResourceLedgerMySQL]:
        result = await self.session.execute(
            select(DatasetResourceLedgerMySQL).where(
                DatasetResourceLedgerMySQL.dataset_version_id == version_id
            )
        )
        return list(result.scalars().all())

    async def mark_released(self, version_id: str) -> None:
        for item in await self.for_version(version_id):
            item.status = "released"
            item.released_at = datetime.utcnow()
            item.last_error = None
        await self.session.commit()

    async def mark_cleanup_failed(self, version_id: str, error: str) -> None:
        for item in await self.for_version(version_id):
            item.status = "cleanup_failed"
            item.cleanup_attempts += 1
            item.last_error = str(error)[:4000]
        await self.session.commit()
