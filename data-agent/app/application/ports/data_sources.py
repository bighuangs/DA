"""上传数据集与数据来源上下文的 Application outbound ports。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from app.domain.data_sources.models import (
    AdhocDatasetRecord,
    MappingQualityStats,
    SourceMappingRecord,
)


@dataclass(frozen=True, slots=True)
class AdhocDatasetSettings:
    """上传校验、配额与生命周期使用的不可变设置。"""

    max_file_bytes: int = 10 * 1024 * 1024
    max_rows: int = 20_000
    max_columns: int = 100
    max_cell_chars: int = 1_000
    ttl_seconds: int = 3_600
    managed_ttl_seconds: int = 30 * 24 * 3_600
    max_extension_seconds: int = 30 * 24 * 3_600
    max_active_datasets: int = 20
    max_total_bytes: int = 100 * 1024 * 1024
    max_total_rows: int = 200_000
    max_selected_datasets: int = 2
    xlsx_expansion_ratio: int = 5

    def __post_init__(self) -> None:
        positive = (
            self.max_file_bytes,
            self.max_rows,
            self.max_columns,
            self.max_cell_chars,
            self.ttl_seconds,
            self.managed_ttl_seconds,
            self.max_extension_seconds,
            self.max_active_datasets,
            self.max_total_bytes,
            self.max_total_rows,
            self.max_selected_datasets,
            self.xlsx_expansion_ratio,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("上传数据集设置必须全部大于 0")


@dataclass(frozen=True, slots=True)
class DataSourceContextSettings:
    """数据来源上下文选择边界。"""

    max_selected_datasets: int = 2

    def __post_init__(self) -> None:
        if self.max_selected_datasets <= 0:
            raise ValueError("max_selected_datasets 必须大于 0")


@dataclass(frozen=True, slots=True)
class DatasetMaterializationSpec:
    """一次上传版本物化所需的完整、不可变输入。"""

    record: AdhocDatasetRecord
    physical_columns: tuple[str, ...]
    column_types: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


class AdhocDatasetStoragePort(Protocol):
    """上传目录、物化事务与生命周期状态机端口。"""

    async def list_catalog(
        self,
        owner_id: str,
        workspace_id: str,
        *,
        limit: int,
        before_id: str | None = None,
        include_terminal: bool = False,
    ) -> list[AdhocDatasetRecord]: ...

    async def get_active(
        self, dataset_id: str, owner_id: str, workspace_id: str
    ) -> AdhocDatasetRecord | None: ...

    async def get_scoped(
        self, dataset_id: str, owner_id: str, workspace_id: str
    ) -> AdhocDatasetRecord | None: ...

    async def get_many_active(
        self, dataset_ids: list[str], owner_id: str, workspace_id: str
    ) -> list[AdhocDatasetRecord]: ...

    async def find_idempotent(
        self, idempotency_key: str, owner_id: str, workspace_id: str
    ) -> AdhocDatasetRecord | None: ...

    async def latest_for_asset(
        self, asset_id: str, owner_id: str, workspace_id: str
    ) -> AdhocDatasetRecord | None: ...

    async def quota_usage(self, owner_id: str, workspace_id: str) -> dict[str, int]: ...

    async def materialize(
        self, spec: DatasetMaterializationSpec
    ) -> AdhocDatasetRecord: ...

    async def delete(
        self, dataset_id: str, owner_id: str, workspace_id: str
    ) -> bool: ...

    async def archive(
        self, dataset_id: str, owner_id: str, workspace_id: str
    ) -> AdhocDatasetRecord | None: ...

    async def extend(
        self,
        dataset_id: str,
        owner_id: str,
        workspace_id: str,
        *,
        seconds: int,
    ) -> AdhocDatasetRecord | None: ...

    async def cleanup_expired(self) -> int: ...

    async def reconcile(self) -> dict[str, int]: ...


class SourceMappingStorePort(Protocol):
    """来源映射证据的 owner/workspace 作用域仓储端口。"""

    async def get(self, mapping_id: str) -> SourceMappingRecord | None: ...

    async def create(self, item: SourceMappingRecord) -> SourceMappingRecord: ...

    async def save_confirmation(
        self,
        mapping_id: str,
        *,
        mappings: tuple[dict, ...],
        evidence: dict,
        status: str,
        confirmed_at: datetime | None,
    ) -> SourceMappingRecord | None: ...

    async def reject(
        self, mapping_id: str, reason: str | None
    ) -> SourceMappingRecord | None: ...


class MappingQualityProbePort(Protocol):
    """在 DW 中执行有界聚合关联质量探测。"""

    async def probe(
        self,
        *,
        left_table: str,
        left_column: str,
        right_table: str,
        right_column: str,
    ) -> MappingQualityStats: ...


__all__ = [
    "AdhocDatasetSettings",
    "AdhocDatasetStoragePort",
    "DataSourceContextSettings",
    "DatasetMaterializationSpec",
    "MappingQualityProbePort",
    "SourceMappingStorePort",
]
