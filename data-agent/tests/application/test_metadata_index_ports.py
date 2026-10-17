"""元数据索引端口的发布顺序与 DTO 边界回归。"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.application.ports.metadata_indexing import MetadataIndexSettings
from app.application.services.meta_admin_service import MetaAdminService
from app.application.services.schema_sync_service import SchemaSyncService
from app.domain.metadata.models import ColumnMetadata, SchemaSyncEvent, TableMetadata
from app.infrastructure.persistence.mysql.records.column_info_mysql import (
    ColumnInfoMySQL,
)
from app.infrastructure.persistence.mysql.records.schema_sync_log_mysql import (
    SchemaSyncLogMySQL,
)
from app.infrastructure.persistence.mysql.records.table_info_mysql import (
    TableInfoMySQL,
)
from app.infrastructure.persistence.mysql.schema_sync_adapter import (
    MysqlSchemaSyncAdapter,
)


class _CacheInvalidator:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def invalidate_retrieval(self) -> None:
        self.events.append("invalidate_retrieval")

    def clear_query_results(self) -> None:
        self.events.append("clear_query_results")


class _Projection:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def rebuild_columns(self, rows, *, batch_size=None) -> int:
        self.events.append(f"rebuild_columns:{batch_size}")
        return sum(len(columns) for _, columns in rows)

    async def rebuild_metrics(self, workspace_id, items) -> int:
        self.events.append(f"rebuild_metrics:{workspace_id}")
        return len(items)

    async def rebuild_training(self, workspace_id, items) -> int:
        self.events.append(f"rebuild_training:{workspace_id}")
        return len(items)

    async def overwrite_all_metrics(self) -> int:
        self.events.append("overwrite_all_metrics")
        return 2

    async def overwrite_all_training(self) -> int:
        self.events.append("overwrite_all_training")
        return 3

    async def status(self, **kwargs) -> dict:
        del kwargs
        return {"ready": True}

    def point_strategy(self) -> str:
        return "single_canonical"


@pytest.mark.asyncio
async def test_schema_sync_preserves_cache_and_index_publication_order() -> None:
    events: list[str] = []
    snapshot = {
        "source": "dw",
        "tables": [
            {
                "name": "dim_region",
                "role": "dim",
                "description": "地区",
                "columns": [
                    {
                        "id": "dim_region.region_name",
                        "name": "region_name",
                        "type": "varchar",
                        "role": "dimension",
                        "description": "地区名称",
                    }
                ],
            }
        ],
    }
    source = SimpleNamespace(
        get_schema_snapshot=AsyncMock(return_value=snapshot),
    )

    class _Metadata:
        workspace_id = "workspace-a"

        async def ensure_admin_tables(self) -> None:
            events.append("ensure_admin_tables")

        async def list_table_infos(self):
            return []

        async def replace_schema_snapshot(self, tables, columns) -> None:
            assert isinstance(tables[0], TableMetadata)
            assert isinstance(columns[0], ColumnMetadata)
            events.append("replace_schema")

        async def record_schema_sync(self, event: SchemaSyncEvent):
            events.append(f"record_log:{event.status}")
            return replace(event, id=7)

    service = SchemaSyncService(
        source,
        _Metadata(),
        _Projection(events),
        None,
        _CacheInvalidator(events),
        settings=MetadataIndexSettings(schema_embedding_batch_size=16),
    )

    result = await service.sync()

    assert result["sync_id"] == 7
    assert events == [
        "ensure_admin_tables",
        "replace_schema",
        "clear_query_results",
        "invalidate_retrieval",
        "rebuild_columns:16",
        "record_log:success",
    ]


@pytest.mark.asyncio
async def test_meta_rebuild_invalidates_only_after_projection_publish() -> None:
    events: list[str] = []
    table = SimpleNamespace(id="dim_region", name="dim_region")
    column = SimpleNamespace(id="dim_region.region_name")
    repo = SimpleNamespace(
        workspace_id="workspace-a",
        list_table_infos=AsyncMock(return_value=[(table, [column])]),
        list_metric_infos=AsyncMock(return_value=[]),
        list_training_infos=AsyncMock(return_value=[]),
    )
    service = MetaAdminService(
        meta_repo=repo,
        projection_index=_Projection(events),
        cache_invalidator=_CacheInvalidator(events),
    )

    assert await service.rebuild_columns() == 1
    assert events == ["rebuild_columns:None", "invalidate_retrieval"]


@pytest.mark.asyncio
async def test_mysql_schema_adapter_owns_orm_record_conversion() -> None:
    captured: dict[str, object] = {}

    class _Repository:
        workspace_id = "workspace-a"

        async def replace_schema_infos(self, tables, columns) -> None:
            captured["tables"] = tables
            captured["columns"] = columns

        async def save_schema_sync_log(self, log):
            captured["log"] = log
            log.id = 11
            return log

    adapter = MysqlSchemaSyncAdapter(_Repository())
    await adapter.replace_schema_snapshot(
        [TableMetadata("dim_region", "dim_region", "dim", "地区")],
        [
            ColumnMetadata(
                id="dim_region.region_name",
                name="region_name",
                type="varchar",
                role="dimension",
                table_id="dim_region",
            )
        ],
    )
    event = await adapter.record_schema_sync(
        SchemaSyncEvent(
            source="dw",
            status="success",
            diff={},
            snapshot={},
        )
    )

    assert isinstance(captured["tables"][0], TableInfoMySQL)
    assert isinstance(captured["columns"][0], ColumnInfoMySQL)
    assert isinstance(captured["log"], SchemaSyncLogMySQL)
    assert event.id == 11
