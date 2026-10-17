"""来源映射仓储与 DW 质量探测的 outbound adapters。"""

from __future__ import annotations

import re
from datetime import datetime

from sqlalchemy import text

from app.application.ports.data_sources import (
    MappingQualityProbePort,
    SourceMappingStorePort,
)
from app.domain.data_sources.models import (
    MappingQualityStats,
    SourceMappingRecord,
)
from app.infrastructure.clients.mysql_client_manager import meta_mysql_client_manager
from app.infrastructure.persistence.mysql.records.conversation_data_context_mysql import (
    SourceMappingMySQL,
)
from app.infrastructure.persistence.mysql.repositories.data_context_repository import (
    SourceMappingRepository,
)
from app.infrastructure.persistence.mysql.repositories.dw_mysql_repository import (
    DWMysqlRepository,
)


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _mapping_record(item: SourceMappingMySQL) -> SourceMappingRecord:
    """复制 ORM 映射行，避免会话对象泄漏到 Application。"""

    return SourceMappingRecord(
        id=str(item.id),
        owner_id=str(item.owner_id),
        workspace_id=str(item.workspace_id),
        status=str(item.status),
        intent=str(item.intent),
        dataset_version_ids=tuple(
            str(value) for value in (item.dataset_version_ids or [])
        ),
        mappings=tuple(dict(value) for value in (item.mappings or [])),
        evidence=dict(item.evidence or {}),
        fingerprint=str(item.fingerprint),
        rejection_reason=item.rejection_reason,
        confirmed_at=item.confirmed_at,
        created_at=item.created_at,
    )


class MysqlSourceMappingAdapter(SourceMappingStorePort):
    """把来源映射端口绑定到一个 owner/workspace 作用域。"""

    def __init__(
        self,
        *,
        owner_id: str,
        workspace_id: str,
        manager=meta_mysql_client_manager,
    ) -> None:
        self._owner_id = str(owner_id)
        self._workspace_id = str(workspace_id)
        self._manager = manager

    def _repository(self, session) -> SourceMappingRepository:
        return SourceMappingRepository(
            session,
            owner_id=self._owner_id,
            workspace_id=self._workspace_id,
        )

    async def get(self, mapping_id: str) -> SourceMappingRecord | None:
        async with self._manager.session_factory() as session:
            item = await self._repository(session).get(mapping_id)
        return _mapping_record(item) if item is not None else None

    async def create(self, item: SourceMappingRecord) -> SourceMappingRecord:
        if item.owner_id != self._owner_id or item.workspace_id != self._workspace_id:
            raise ValueError("来源映射作用域与适配器绑定身份不一致")
        record = SourceMappingMySQL(
            id=item.id,
            owner_id=item.owner_id,
            workspace_id=item.workspace_id,
            status=item.status,
            intent=item.intent,
            dataset_version_ids=list(item.dataset_version_ids),
            mappings=[dict(value) for value in item.mappings],
            evidence=dict(item.evidence),
            fingerprint=item.fingerprint,
            rejection_reason=item.rejection_reason,
            confirmed_at=item.confirmed_at,
        )
        async with self._manager.session_factory() as session:
            record = await self._repository(session).create(record)
        return _mapping_record(record)

    async def save_confirmation(
        self,
        mapping_id: str,
        *,
        mappings: tuple[dict, ...],
        evidence: dict,
        status: str,
        confirmed_at: datetime | None,
    ) -> SourceMappingRecord | None:
        async with self._manager.session_factory() as session:
            item = await self._repository(session).get(mapping_id)
            if item is None:
                return None
            item.mappings = [dict(value) for value in mappings]
            item.evidence = dict(evidence)
            item.status = status
            item.confirmed_at = confirmed_at
            await session.commit()
            await session.refresh(item)
            return _mapping_record(item)

    async def reject(
        self, mapping_id: str, reason: str | None
    ) -> SourceMappingRecord | None:
        async with self._manager.session_factory() as session:
            item = await self._repository(session).reject(mapping_id, reason)
        return _mapping_record(item) if item is not None else None


class DWMysqlMappingQualityProbeAdapter(MappingQualityProbePort):
    """通过 DW 会话执行不返回原始值的三组聚合探测。"""

    def __init__(self, repository: DWMysqlRepository) -> None:
        self._repository = repository

    async def probe(
        self,
        *,
        left_table: str,
        left_column: str,
        right_table: str,
        right_column: str,
    ) -> MappingQualityStats:
        identifiers = (left_table, left_column, right_table, right_column)
        if any(not _IDENTIFIER_RE.fullmatch(value or "") for value in identifiers):
            raise ValueError("映射质量探测包含非法资源标识")

        right_stats = (
            (
                await self._repository.session.execute(
                    text(
                        f"SELECT COUNT(`{right_column}`) AS non_null_count, "
                        f"COUNT(DISTINCT `{right_column}`) AS distinct_count "
                        f"FROM `{right_table}`"
                    )
                )
            )
            .mappings()
            .one()
        )
        left_stats = (
            (
                await self._repository.session.execute(
                    text(
                        f"SELECT COUNT(`{left_column}`) AS left_rows, "
                        f"COUNT(DISTINCT `{left_column}`) AS left_distinct "
                        f"FROM `{left_table}`"
                    )
                )
            )
            .mappings()
            .one()
        )
        joined = (
            (
                await self._repository.session.execute(
                    text(
                        "SELECT COUNT(*) AS joined_rows, "
                        f"COUNT(DISTINCT CASE WHEN r.`{right_column}` IS NOT NULL "
                        f"THEN l.`{left_column}` END) AS matched_distinct "
                        f"FROM `{left_table}` l LEFT JOIN `{right_table}` r "
                        f"ON l.`{left_column}` = r.`{right_column}` "
                        f"WHERE l.`{left_column}` IS NOT NULL"
                    )
                )
            )
            .mappings()
            .one()
        )
        return MappingQualityStats(
            right_non_null_count=int(right_stats.get("non_null_count") or 0),
            right_distinct_count=int(right_stats.get("distinct_count") or 0),
            left_row_count=int(left_stats.get("left_rows") or 0),
            left_distinct_count=int(left_stats.get("left_distinct") or 0),
            joined_row_count=int(joined.get("joined_rows") or 0),
            matched_distinct_count=int(joined.get("matched_distinct") or 0),
        )


__all__ = ["DWMysqlMappingQualityProbeAdapter", "MysqlSourceMappingAdapter"]
