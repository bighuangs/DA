"""上传数据集与来源映射的领域快照。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Protocol


@dataclass(frozen=True, slots=True)
class AdhocDatasetRecord:
    """不携带 ORM 会话的上传数据集版本快照。"""

    id: str
    asset_id: str
    version_id: str
    version_no: int
    asset_kind: str
    retention_mode: str
    owner_id: str
    workspace_id: str
    table_name: str
    filename: str
    file_format: str
    row_count: int
    column_count: int
    columns: tuple[dict, ...] = field(default_factory=tuple)
    preview: tuple[dict, ...] = field(default_factory=tuple)
    status: str = "materializing"
    content_checksum: str | None = None
    schema_fingerprint: str | None = None
    parser_profile: str = "upload-parser-v2"
    byte_size: int = 0
    idempotency_key: str | None = None
    policy_version: str = "upload-policy-v2"
    lifecycle_revision: int = 1
    managed: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None
    expires_at: datetime | None = None
    deleted_at: datetime | None = None
    error: str | None = None
    cleanup_attempts: int = 0
    cleanup_error: str | None = None
    archived_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class SourceMappingRecord:
    """经 owner/workspace 限定的数据来源映射快照。"""

    id: str
    owner_id: str
    workspace_id: str
    status: str
    intent: str
    dataset_version_ids: tuple[str, ...]
    mappings: tuple[dict, ...]
    evidence: dict
    fingerprint: str
    rejection_reason: str | None = None
    confirmed_at: datetime | None = None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class MappingQualityStats:
    """行级关联质量探测的有界聚合结果，不包含原始业务值。"""

    right_non_null_count: int
    right_distinct_count: int
    left_row_count: int
    left_distinct_count: int
    joined_row_count: int
    matched_distinct_count: int


class DatasetLifecycleView(Protocol):
    """清理恢复规则所需的最小数据集视图。"""

    error: str | None
    expires_at: datetime | None


class DatasetCleanupError(RuntimeError):
    """物理资源释放失败；元数据必须保持可重试状态。"""


def reconciled_terminal_status(
    item: DatasetLifecycleView,
    *,
    now: datetime | None = None,
) -> str:
    """清理重试成功后恢复原本对应的终态。"""

    current = now or datetime.utcnow()
    if item.error:
        return "failed"
    if item.expires_at and item.expires_at <= current:
        return "expired"
    return "deleted"


def typed_dataset_value(value: str, sql_type: str) -> object | None:
    """按已推断的受限 SQL 类型转换上传单元格。"""

    value = value.strip()
    if not value:
        return None
    if sql_type == "BIGINT":
        return int(value)
    if sql_type.startswith("DECIMAL"):
        return Decimal(value)
    if sql_type == "BOOLEAN":
        return value.lower() in {"true", "1"}
    if sql_type == "DATE":
        return date.fromisoformat(value)
    if sql_type == "DATETIME":
        return datetime.fromisoformat(value.replace("T", " "))
    return value


__all__ = [
    "AdhocDatasetRecord",
    "DatasetCleanupError",
    "MappingQualityStats",
    "SourceMappingRecord",
    "reconciled_terminal_status",
    "typed_dataset_value",
]
