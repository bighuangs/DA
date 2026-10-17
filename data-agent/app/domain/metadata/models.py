"""与 ORM、搜索引擎和供应商 SDK 无关的元数据 DTO。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NotRequired, TypedDict


@dataclass(slots=True)
class TableMetadata:
    """一次 Schema 发布中的表定义。"""

    id: str
    name: str
    role: str
    description: str


@dataclass(slots=True)
class ColumnMetadata:
    """一次 Schema 发布中的字段定义及人工治理属性。"""

    id: str
    name: str
    type: str
    role: str
    examples: list = field(default_factory=list)
    description: str = ""
    alias: list = field(default_factory=list)
    table_id: str = ""
    sync: bool = False
    is_display: bool = False


@dataclass(slots=True)
class SchemaSyncEvent:
    """Schema 同步的版本化审计事实，不携带数据库会话。"""

    source: str
    status: str
    diff: dict
    snapshot: dict
    added_tables: int = 0
    removed_tables: int = 0
    changed_tables: int = 0
    added_columns: int = 0
    removed_columns: int = 0
    changed_columns: int = 0
    error: str | None = None
    id: int | None = None


class ValueIndexDocument(TypedDict):
    """准备发布到字段值索引的供应商中立文档。"""

    id: str
    workspace_id: str
    value: str
    normalized_value: str
    type: str
    column_id: str
    column_name: str
    table_id: str
    table_name: str
    index_version: str
    indexed_at: str
    is_active: bool
    source_updated_at: NotRequired[str | None]
    responsibility: NotRequired[str]


__all__ = [
    "ColumnMetadata",
    "SchemaSyncEvent",
    "TableMetadata",
    "ValueIndexDocument",
]
