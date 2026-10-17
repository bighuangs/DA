"""元数据同步、Point 投影与字段值索引的应用端口。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from app.domain.metadata.models import (
    ColumnMetadata,
    SchemaSyncEvent,
    TableMetadata,
)


@dataclass(frozen=True, slots=True)
class MetadataIndexSettings:
    """索引构建用例使用的有界运行设置。"""

    max_distinct_values_per_column: int = 10_000
    schema_embedding_batch_size: int = 32

    def __post_init__(self) -> None:
        if self.max_distinct_values_per_column <= 0:
            raise ValueError("max_distinct_values_per_column 必须大于 0")
        if self.schema_embedding_batch_size <= 0:
            raise ValueError("schema_embedding_batch_size 必须大于 0")


class MetadataAdminRepositoryPort(Protocol):
    """管理用例读取当前工作空间元数据所需的最小仓储面。"""

    workspace_id: str | None

    async def ensure_admin_tables(self) -> None: ...

    async def list_table_infos(self) -> list[tuple[Any, list[Any]]]: ...

    async def list_metric_infos(self, *args: Any, **kwargs: Any) -> list[Any]: ...

    async def get_metric_info(self, metric_id: str) -> Any | None: ...

    async def create_metric_info(self, **values: Any) -> Any: ...

    async def update_metric_info(self, item: Any, **values: Any) -> Any: ...

    async def delete_metric_info(self, item: Any) -> None: ...

    async def list_training_infos(self, *args: Any, **kwargs: Any) -> list[Any]: ...

    async def get_training_info(self, training_id: int) -> Any | None: ...

    async def create_training_info(self, **values: Any) -> Any: ...

    async def update_training_info(self, item: Any, **values: Any) -> Any: ...

    async def delete_training_info(self, item: Any) -> None: ...

    async def list_masking_rules(self) -> list[Any]: ...

    async def get_masking_rule(self, rule_id: int) -> Any | None: ...

    async def create_masking_rule(self, **values: Any) -> Any: ...

    async def update_masking_rule(self, item: Any, **values: Any) -> Any: ...

    async def delete_masking_rule(self, item: Any) -> None: ...

    async def list_schema_sync_logs(self, limit: int = 10) -> list[Any]: ...


class AccessPolicyAdminPort(Protocol):
    """工作空间数据权限策略的管理持久化端口。"""

    async def list_all(self) -> Sequence[Any]: ...

    async def get(self, policy_id: int) -> Any | None: ...

    async def create(self, **values: Any) -> Any: ...

    async def update(self, item: Any, **values: Any) -> Any: ...

    async def delete(self, item: Any) -> None: ...


class MetadataCacheInvalidationPort(Protocol):
    """Schema 与索引发布完成时使用的缓存失效端口。"""

    def invalidate_retrieval(self) -> None: ...

    def clear_query_results(self) -> None: ...


class SchemaSourcePort(Protocol):
    """数据仓库 Schema 与字段值读取端口。"""

    async def get_schema_snapshot(self) -> dict: ...


class SchemaSyncRepositoryPort(Protocol):
    """Schema 快照发布与同步日志持久化端口。"""

    workspace_id: str | None

    async def ensure_admin_tables(self) -> None: ...

    async def list_table_infos(self) -> list[tuple[Any, list[Any]]]: ...

    async def replace_schema_snapshot(
        self,
        tables: Sequence[TableMetadata],
        columns: Sequence[ColumnMetadata],
    ) -> None: ...

    async def record_schema_sync(
        self,
        event: SchemaSyncEvent,
    ) -> SchemaSyncEvent: ...


class MetadataProjectionPort(Protocol):
    """元数据到 Point 投影及版本发布端口。"""

    async def status(
        self,
        *,
        workspace_id: str,
        table_rows: Sequence[tuple[Any, Sequence[Any]]],
        metric_items: Sequence[Any],
        training_items: Sequence[Any],
    ) -> dict: ...

    async def rebuild_columns(
        self,
        table_rows: Sequence[tuple[Any, Sequence[Any]]],
        *,
        batch_size: int | None = None,
    ) -> int: ...

    async def rebuild_metrics(
        self,
        workspace_id: str,
        items: Sequence[Any],
    ) -> int: ...

    async def rebuild_training(
        self,
        workspace_id: str,
        items: Sequence[Any],
    ) -> int: ...

    async def overwrite_all_metrics(self) -> int: ...

    async def overwrite_all_training(self) -> int: ...

    def point_strategy(self) -> str: ...


class ValueIndexPublicationPort(Protocol):
    """字段值索引计算与原子发布端口。"""

    async def rebuild(
        self,
        *,
        workspace_id: str,
        columns: Sequence[Any],
    ) -> dict: ...


class ValueIndexAdminPort(Protocol):
    """字段值索引状态、检索与回退端口。"""

    async def status(
        self,
        *,
        workspace_id: str,
        columns: Sequence[Any],
    ) -> dict: ...

    async def rollback(self, *, workspace_id: str, physical_index: str) -> dict: ...

    async def search(
        self,
        keyword: str,
        *,
        workspace_id: str,
        query_mode: str,
        allowed_table_ids: Sequence[str],
        allowed_column_ids: Sequence[str],
        table_id: str | None,
        column_id: str | None,
        page: int,
        page_size: int,
    ) -> tuple[list[dict], int, dict]: ...


__all__ = [
    "AccessPolicyAdminPort",
    "MetadataAdminRepositoryPort",
    "MetadataCacheInvalidationPort",
    "MetadataIndexSettings",
    "MetadataProjectionPort",
    "SchemaSourcePort",
    "SchemaSyncRepositoryPort",
    "ValueIndexAdminPort",
    "ValueIndexPublicationPort",
]
