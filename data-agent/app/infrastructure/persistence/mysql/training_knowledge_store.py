"""MySQL 训练知识导入端口实现。"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.application.ports.training_knowledge import (
    TrainingKnowledgeItem,
    TrainingSeed,
)
from app.infrastructure.persistence.mysql.records.base import Base
from app.infrastructure.persistence.mysql.records.training_info_mysql import (
    TrainingInfoMySQL,
)


class MysqlTrainingKnowledgeStore:
    """在一个 MySQL 事务内替换指定 workspace 的全部样例。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def replace_all(
        self,
        *,
        workspace_id: str,
        seeds: Sequence[TrainingSeed],
    ) -> list[TrainingKnowledgeItem]:
        await self._session.run_sync(
            lambda sync_session: Base.metadata.create_all(sync_session.connection())
        )
        await self._session.execute(
            delete(TrainingInfoMySQL).where(
                TrainingInfoMySQL.workspace_id == workspace_id
            )
        )
        records = [
            TrainingInfoMySQL(
                question=seed.question,
                sql=seed.sql,
                enabled=seed.enabled,
                workspace_id=workspace_id,
            )
            for seed in seeds
        ]
        self._session.add_all(records)
        await self._session.flush()
        items = [
            TrainingKnowledgeItem(
                id=int(record.id),
                question=record.question,
                sql=record.sql,
                enabled=bool(record.enabled),
            )
            for record in records
        ]
        await self._session.commit()
        return items


__all__ = ["MysqlTrainingKnowledgeStore"]
