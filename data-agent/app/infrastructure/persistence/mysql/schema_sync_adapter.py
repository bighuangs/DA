"""领域 Schema DTO 与 MySQL ORM 记录之间的持久化适配器。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from app.domain.metadata.models import (
    ColumnMetadata,
    SchemaSyncEvent,
    TableMetadata,
)
from app.infrastructure.persistence.mysql.records.column_info_mysql import (
    ColumnInfoMySQL,
)
from app.infrastructure.persistence.mysql.records.schema_sync_log_mysql import (
    SchemaSyncLogMySQL,
)
from app.infrastructure.persistence.mysql.records.table_info_mysql import (
    TableInfoMySQL,
)
from app.infrastructure.persistence.mysql.repositories.meta_mysql_repository import (
    MetaMysqlRepository,
)


class MysqlSchemaSyncAdapter:
    """把 Schema 同步端口映射到现有 Meta MySQL 仓储。"""

    def __init__(self, repository: MetaMysqlRepository) -> None:
        self._repository = repository

    @property
    def workspace_id(self) -> str | None:
        return self._repository.workspace_id

    async def ensure_admin_tables(self) -> None:
        await self._repository.ensure_admin_tables()

    async def list_table_infos(self):
        return await self._repository.list_table_infos()

    async def replace_schema_snapshot(
        self,
        tables: Sequence[TableMetadata],
        columns: Sequence[ColumnMetadata],
    ) -> None:
        table_records = [
            TableInfoMySQL(
                id=item.id,
                name=item.name,
                role=item.role,
                description=item.description,
            )
            for item in tables
        ]
        column_records = [
            ColumnInfoMySQL(
                id=item.id,
                name=item.name,
                type=item.type,
                role=item.role,
                examples=list(item.examples),
                description=item.description,
                alias=list(item.alias),
                table_id=item.table_id,
                sync=bool(item.sync),
                is_display=bool(item.is_display),
            )
            for item in columns
        ]
        await self._repository.replace_schema_infos(table_records, column_records)

    async def record_schema_sync(
        self,
        event: SchemaSyncEvent,
    ) -> SchemaSyncEvent:
        record = SchemaSyncLogMySQL(
            source=event.source,
            status=event.status,
            diff=dict(event.diff),
            snapshot=dict(event.snapshot),
            added_tables=event.added_tables,
            removed_tables=event.removed_tables,
            changed_tables=event.changed_tables,
            added_columns=event.added_columns,
            removed_columns=event.removed_columns,
            changed_columns=event.changed_columns,
            error=event.error,
        )
        stored = await self._repository.save_schema_sync_log(record)
        return replace(event, id=stored.id)


__all__ = ["MysqlSchemaSyncAdapter"]
