from __future__ import annotations

from collections.abc import Sequence

from app.application.ports.metadata_indexing import (
    MetadataCacheInvalidationPort,
    MetadataIndexSettings,
    MetadataProjectionPort,
    SchemaSourcePort,
    SchemaSyncRepositoryPort,
    ValueIndexPublicationPort,
)
from app.domain.metadata.models import (
    ColumnMetadata,
    SchemaSyncEvent,
    TableMetadata,
)
from app.domain.query.semantic_aliases import merge_business_aliases
from app.domain.retrieval.hybrid_embedding import HybridEmbedding


_INTERNAL_PREFIXES = ("adhoc_", "da_upload_", "eval_upload_", "__da_")


def _official_snapshot_tables(tables: list[dict]) -> list[dict]:
    """服务层第二道隔离：投影器永不接受上传、评测或系统表。"""

    return [
        table
        for table in tables or []
        if table.get("name")
        and not str(table["name"]).lower().startswith(_INTERNAL_PREFIXES)
        and str(table.get("asset_kind") or "official") == "official"
    ]


async def _embed_documents_in_batches(
    embedding_client,
    texts: list[str],
    batch_size: int = 32,
) -> list[HybridEmbedding]:
    """兼容旧调用点的纯批处理助手；生产组合已迁入出站适配器。"""

    vectors: list[HybridEmbedding] = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(
            await embedding_client.aembed_documents(texts[start : start + batch_size])
        )
    return vectors


def _table_map(tables) -> dict[str, dict]:
    result = {}
    for item in tables:
        if isinstance(item, tuple):
            table, columns = item
            result[table.name] = {
                "name": table.name,
                "role": table.role,
                "description": table.description or "",
                "columns": [
                    {
                        "id": column.id,
                        "name": column.name,
                        "type": column.type,
                        "role": column.role,
                        "description": column.description or "",
                        "alias": column.alias or [],
                        "examples": column.examples or [],
                        "sync": bool(column.sync),
                        "is_display": bool(column.is_display),
                    }
                    for column in columns
                ],
            }
        else:
            result[item["name"]] = item
    return result


def infer_display_column(
    table_name: str,
    column_name: str,
    role: str = "",
) -> bool:
    """为首次同步的维表推断默认展示列。"""

    if role in {"primary_key", "foreign_key", "measure"} or not table_name.startswith(
        "dim_"
    ):
        return False
    entity = table_name.removeprefix("dim_")
    return column_name in {f"{entity}_name", "name"}


def diff_schema(snapshot_tables: list[dict], current_tables) -> dict:
    """对比数据源快照和当前 Meta Schema，输出可审计差异。"""

    incoming = _table_map(snapshot_tables)
    current = _table_map(current_tables)
    added_tables = sorted(set(incoming) - set(current))
    removed_tables = sorted(set(current) - set(incoming))
    changed_tables = []
    added_columns = []
    removed_columns = []
    changed_columns = []

    for table_name in sorted(set(incoming) & set(current)):
        new_columns = {
            item["name"]: item for item in incoming[table_name].get("columns", [])
        }
        old_columns = {
            item["name"]: item for item in current[table_name].get("columns", [])
        }
        for name in sorted(set(new_columns) - set(old_columns)):
            added_columns.append(f"{table_name}.{name}")
        for name in sorted(set(old_columns) - set(new_columns)):
            removed_columns.append(f"{table_name}.{name}")
        for name in sorted(set(new_columns) & set(old_columns)):
            new_item, old_item = new_columns[name], old_columns[name]
            fields = [
                field
                for field in ("type", "role", "description")
                if (new_item.get(field) or "") != (old_item.get(field) or "")
            ]
            if fields:
                changed_columns.append(
                    {
                        "id": f"{table_name}.{name}",
                        "fields": fields,
                    }
                )
        table_fields = []
        if (incoming[table_name].get("role") or "") != (
            current[table_name].get("role") or ""
        ):
            table_fields.append("role")
        if (incoming[table_name].get("description") or "") != (
            current[table_name].get("description") or ""
        ):
            table_fields.append("description")
        if table_fields:
            changed_tables.append({"name": table_name, "fields": table_fields})

    return {
        "added_tables": added_tables,
        "removed_tables": removed_tables,
        "changed_tables": changed_tables,
        "added_columns": added_columns,
        "removed_columns": removed_columns,
        "changed_columns": changed_columns,
        "counts": {
            "added_tables": len(added_tables),
            "removed_tables": len(removed_tables),
            "changed_tables": len(changed_tables),
            "added_columns": len(added_columns),
            "removed_columns": len(removed_columns),
            "changed_columns": len(changed_columns),
        },
    }


class SchemaSyncService:
    """读取 DW Schema、合并人工语义并按固定顺序发布外部索引。"""

    def __init__(
        self,
        dw_repo: SchemaSourcePort,
        meta_repo: SchemaSyncRepositoryPort,
        projection_index: MetadataProjectionPort,
        value_index: ValueIndexPublicationPort | None,
        cache_invalidator: MetadataCacheInvalidationPort,
        *,
        settings: MetadataIndexSettings | None = None,
    ) -> None:
        self._schema_source = dw_repo
        self._repository = meta_repo
        self._projection_index = projection_index
        self._value_index = value_index
        self._cache_invalidator = cache_invalidator
        self._settings = settings or MetadataIndexSettings()

    async def preview(self) -> dict:
        await self._repository.ensure_admin_tables()
        snapshot = await self._schema_source.get_schema_snapshot()
        snapshot["tables"] = _official_snapshot_tables(snapshot.get("tables") or [])
        current = await self._repository.list_table_infos()
        diff = diff_schema(snapshot["tables"], current)
        return {
            "source": snapshot.get("source") or "dw",
            "table_count": len(snapshot["tables"]),
            "column_count": sum(len(item["columns"]) for item in snapshot["tables"]),
            "diff": diff,
        }

    async def sync(self, *, rebuild_value_index: bool = False) -> dict:
        await self._repository.ensure_admin_tables()
        snapshot = await self._schema_source.get_schema_snapshot()
        snapshot["tables"] = _official_snapshot_tables(snapshot.get("tables") or [])
        current = await self._repository.list_table_infos()
        diff = diff_schema(snapshot["tables"], current)
        source = snapshot.get("source") or "dw"
        event = SchemaSyncEvent(
            source=source,
            status="failed",
            diff=diff,
            snapshot=snapshot,
            **diff["counts"],
        )
        try:
            tables, columns = await self._merge_schema(snapshot["tables"], current)
            await self._repository.replace_schema_snapshot(tables, columns)
            # 顺序属于现有合同：先清 SQL 结果，再推进检索缓存代次。
            self._cache_invalidator.clear_query_results()
            self._cache_invalidator.invalidate_retrieval()
            await self._rebuild_column_index(tables, columns)
            value_index = None
            if rebuild_value_index:
                value_index = await self._rebuild_value_index(columns)
            event.status = "success"
            event = await self._repository.record_schema_sync(event)
            return {
                "sync_id": event.id,
                "source": source,
                "diff": diff,
                "table_count": len(tables),
                "column_count": len(columns),
                "value_index_rebuilt": rebuild_value_index,
                "value_index": value_index,
            }
        except Exception as exc:
            event.error = str(exc)
            await self._repository.record_schema_sync(event)
            raise

    async def _merge_schema(
        self,
        snapshot_tables: list[dict],
        current_tables,
    ) -> tuple[list[TableMetadata], list[ColumnMetadata]]:
        current = _table_map(current_tables)
        table_infos: list[TableMetadata] = []
        column_infos: list[ColumnMetadata] = []
        for table in snapshot_tables:
            old_table = current.get(table["name"], {})
            table_infos.append(
                TableMetadata(
                    id=table["name"],
                    name=table["name"],
                    role=old_table.get("role") or table.get("role") or "unknown",
                    description=(
                        old_table.get("description") or table.get("description") or ""
                    ),
                )
            )
            old_columns = {item["name"]: item for item in old_table.get("columns", [])}
            for column in table.get("columns", []):
                old_column = old_columns.get(column["name"], {})
                column_infos.append(
                    ColumnMetadata(
                        id=column["id"],
                        name=column["name"],
                        type=column["type"],
                        role=(
                            old_column.get("role") or column.get("role") or "dimension"
                        ),
                        examples=old_column.get("examples") or [],
                        description=(
                            old_column.get("description")
                            or column.get("description")
                            or ""
                        ),
                        alias=merge_business_aliases(
                            table["name"],
                            column["name"],
                            old_column.get("alias"),
                            column.get("alias"),
                        ),
                        table_id=table["name"],
                        sync=bool(old_column.get("sync", False)),
                        is_display=bool(
                            old_column.get(
                                "is_display",
                                infer_display_column(
                                    table["name"],
                                    column["name"],
                                    column.get("role") or "dimension",
                                ),
                            )
                        ),
                    )
                )
        return table_infos, column_infos

    async def _rebuild_column_index(
        self,
        tables: Sequence[TableMetadata],
        columns: Sequence[ColumnMetadata],
    ) -> None:
        grouped: dict[str, list[ColumnMetadata]] = {}
        for column in columns:
            grouped.setdefault(column.table_id, []).append(column)
        await self._projection_index.rebuild_columns(
            [(table, grouped.get(table.id, [])) for table in tables],
            batch_size=self._settings.schema_embedding_batch_size,
        )

    async def _rebuild_value_index(
        self,
        columns: Sequence[ColumnMetadata],
    ) -> dict:
        if self._value_index is None:
            raise RuntimeError("字段值索引发布端口未配置")
        return await self._value_index.rebuild(
            workspace_id=str(self._repository.workspace_id or ""),
            columns=columns,
        )
