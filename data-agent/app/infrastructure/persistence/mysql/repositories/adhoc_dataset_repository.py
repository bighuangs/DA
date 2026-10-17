from datetime import datetime

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.persistence.mysql.records.adhoc_dataset_mysql import AdhocDatasetMySQL


class AdhocDatasetRepository:
    """上传资产版本仓库；每个读操作都显式携带 owner/workspace scope。"""

    def __init__(self, session: AsyncSession):
        self.session = session

    @staticmethod
    def _scope(statement, owner_id: str, workspace_id: str):
        return (
            statement
            .where(AdhocDatasetMySQL.owner_id == owner_id)
            .where(AdhocDatasetMySQL.workspace_id == workspace_id)
        )

    async def get_active(
        self, dataset_id: str, owner_id: str, workspace_id: str
    ) -> AdhocDatasetMySQL | None:
        result = await self.session.execute(
            self._scope(select(AdhocDatasetMySQL), owner_id, workspace_id)
            .where(AdhocDatasetMySQL.id == dataset_id)
            .where(AdhocDatasetMySQL.status.in_(("ready", "active")))
        )
        return result.scalar_one_or_none()

    async def get_scoped(
        self, dataset_id: str, owner_id: str, workspace_id: str
    ) -> AdhocDatasetMySQL | None:
        result = await self.session.execute(
            self._scope(select(AdhocDatasetMySQL), owner_id, workspace_id)
            .where(
                or_(
                    AdhocDatasetMySQL.id == dataset_id,
                    AdhocDatasetMySQL.version_id == dataset_id,
                )
            )
        )
        return result.scalar_one_or_none()

    async def get_many_active(
        self, dataset_ids: list[str], owner_id: str, workspace_id: str
    ) -> list[AdhocDatasetMySQL]:
        if not dataset_ids:
            return []
        result = await self.session.execute(
            self._scope(select(AdhocDatasetMySQL), owner_id, workspace_id)
            .where(
                or_(
                    AdhocDatasetMySQL.id.in_(dataset_ids),
                    AdhocDatasetMySQL.version_id.in_(dataset_ids),
                )
            )
            .where(AdhocDatasetMySQL.status.in_(("ready", "active")))
        )
        rows = list(result.scalars().all())
        by_id = {
            key: row
            for row in rows
            for key in (str(row.id), str(row.version_id))
        }
        return [by_id[item] for item in dataset_ids if item in by_id]

    async def find_idempotent(
        self, idempotency_key: str, owner_id: str, workspace_id: str
    ) -> AdhocDatasetMySQL | None:
        if not idempotency_key:
            return None
        result = await self.session.execute(
            self._scope(select(AdhocDatasetMySQL), owner_id, workspace_id)
            .where(AdhocDatasetMySQL.idempotency_key == idempotency_key)
            .where(
                AdhocDatasetMySQL.status.notin_(
                    ("failed", "cleanup_failed", "deleted")
                )
            )
            .order_by(AdhocDatasetMySQL.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def latest_for_asset(
        self, asset_id: str, owner_id: str, workspace_id: str
    ) -> AdhocDatasetMySQL | None:
        result = await self.session.execute(
            self._scope(select(AdhocDatasetMySQL), owner_id, workspace_id)
            .where(AdhocDatasetMySQL.asset_id == asset_id)
            .where(AdhocDatasetMySQL.status != "deleted")
            .order_by(
                AdhocDatasetMySQL.version_no.desc(),
                AdhocDatasetMySQL.created_at.desc(),
            )
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def list_catalog(
        self,
        owner_id: str,
        workspace_id: str,
        *,
        limit: int,
        before_id: str | None = None,
        include_terminal: bool = False,
    ) -> list[AdhocDatasetMySQL]:
        statement = self._scope(
            select(AdhocDatasetMySQL), owner_id, workspace_id
        )
        if not include_terminal:
            statement = statement.where(
                AdhocDatasetMySQL.status.notin_(("deleted", "failed"))
            )
        if before_id:
            anchor = await self.get_scoped(before_id, owner_id, workspace_id)
            if anchor is not None:
                statement = statement.where(
                    or_(
                        AdhocDatasetMySQL.created_at < anchor.created_at,
                        (
                            (AdhocDatasetMySQL.created_at == anchor.created_at)
                            & (AdhocDatasetMySQL.id < anchor.id)
                        ),
                    )
                )
        result = await self.session.execute(
            statement.order_by(
                AdhocDatasetMySQL.created_at.desc(), AdhocDatasetMySQL.id.desc()
            ).limit(limit)
        )
        return list(result.scalars().all())

    async def quota_usage(
        self, owner_id: str, workspace_id: str
    ) -> dict[str, int]:
        result = await self.session.execute(
            self._scope(
                select(
                    func.count(AdhocDatasetMySQL.id),
                    func.coalesce(func.sum(AdhocDatasetMySQL.byte_size), 0),
                    func.coalesce(func.sum(AdhocDatasetMySQL.row_count), 0),
                ),
                owner_id,
                workspace_id,
            ).where(AdhocDatasetMySQL.status.in_(("ready", "active")))
        )
        count, byte_size, rows = result.one()
        return {
            "active_versions": int(count or 0),
            "bytes": int(byte_size or 0),
            "rows": int(rows or 0),
        }

    async def list_expired(self, now: datetime) -> list[AdhocDatasetMySQL]:
        result = await self.session.execute(
            select(AdhocDatasetMySQL)
            .where(AdhocDatasetMySQL.status.in_(("ready", "active", "expiring")))
            .where(AdhocDatasetMySQL.expires_at < now)
        )
        return list(result.scalars().all())

    async def list_cleanup_pending(self) -> list[AdhocDatasetMySQL]:
        result = await self.session.execute(
            select(AdhocDatasetMySQL).where(
                AdhocDatasetMySQL.status.in_(
                    ("delete_pending", "cleanup_pending", "cleanup_failed", "expired")
                )
            )
        )
        return list(result.scalars().all())

    async def create(self, item: AdhocDatasetMySQL) -> AdhocDatasetMySQL:
        self.session.add(item)
        await self.session.commit()
        await self.session.refresh(item)
        return item

    async def save(self, item: AdhocDatasetMySQL) -> AdhocDatasetMySQL:
        item.lifecycle_revision = int(item.lifecycle_revision or 0) + 1
        await self.session.commit()
        await self.session.refresh(item)
        return item

    async def delete(self, item: AdhocDatasetMySQL) -> None:
        item.status = "deleted"
        item.deleted_at = datetime.utcnow()
        await self.session.commit()
