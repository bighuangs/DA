"""TODO-29 文件上传查询的数据集生命周期与文件解析服务。"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import uuid
import zipfile
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath
from xml.etree import ElementTree as ET

from app.application.ports.data_sources import (
    AdhocDatasetSettings,
    AdhocDatasetStoragePort,
    DatasetMaterializationSpec,
)
from app.application.services.masking_service import MaskingRule, mask_rows
from app.domain.data_sources.context import (
    AssetIdentity,
    AssetKind,
    DatasetAccessError,
    DatasetExpiredError,
    READY_DATASET_STATES,
    stable_fingerprint,
)
from app.domain.data_sources.models import (
    AdhocDatasetRecord,
    DatasetCleanupError as DatasetCleanupError,
    reconciled_terminal_status as reconciled_terminal_status,
    typed_dataset_value,
)


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INT_RE = re.compile(r"^-?\d+$")
_DECIMAL_RE = re.compile(r"^-?(?:\d+\.\d+|\d+)$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _column_index(reference: str | None) -> int:
    if not reference:
        return 0
    letters = "".join(ch for ch in reference if ch.isalpha()).upper()
    value = 0
    for char in letters:
        value = value * 26 + ord(char) - ord("A") + 1
    return max(0, value - 1)


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("文件不是可识别的 UTF-8/GB18030 文本")


def _parse_csv(content: bytes) -> list[list[str]]:
    text_content = _decode_text(content)
    try:
        rows = list(csv.reader(io.StringIO(text_content)))
    except csv.Error as exc:
        raise ValueError(f"CSV 解析失败：{exc}") from exc
    return rows


def _parse_xlsx(
    content: bytes,
    *,
    max_expanded_bytes: int = 50 * 1024 * 1024,
) -> list[list[str]]:
    if not zipfile.is_zipfile(io.BytesIO(content)):
        raise ValueError("XLSX 文件结构无效")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        members = {info.filename: info for info in archive.infolist()}
        expanded = sum(max(0, info.file_size) for info in members.values())
        if expanded > max_expanded_bytes:
            raise ValueError("XLSX 解压后体积超过安全上限")

        shared: list[str] = []
        if "xl/sharedStrings.xml" in members:
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.iter():
                if _local_name(item.tag) == "si":
                    shared.append(
                        "".join(
                            node.text or ""
                            for node in item.iter()
                            if _local_name(node.tag) == "t"
                        )
                    )

        sheet_path = "xl/worksheets/sheet1.xml"
        if "xl/workbook.xml" in members and "xl/_rels/workbook.xml.rels" in members:
            workbook = ET.fromstring(archive.read("xl/workbook.xml"))
            rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            rel_map = {
                rel.attrib.get("Id"): rel.attrib.get("Target", "")
                for rel in rels
                if _local_name(rel.tag) == "Relationship"
            }
            first_sheet = next(
                (node for node in workbook.iter() if _local_name(node.tag) == "sheet"),
                None,
            )
            if first_sheet is not None:
                target = rel_map.get(
                    first_sheet.attrib.get(
                        "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
                    ),
                    "",
                )
                target = target.lstrip("/")
                sheet_path = (
                    str(PurePosixPath("xl") / target)
                    if not target.startswith("xl/")
                    else target
                )
        if sheet_path not in members:
            raise ValueError("XLSX 中未找到首个工作表")

        root = ET.fromstring(archive.read(sheet_path))
        rows: list[list[str]] = []
        for row_node in (
            node for node in root.iter() if _local_name(node.tag) == "row"
        ):
            cells: list[str] = []
            for cell in (node for node in row_node if _local_name(node.tag) == "c"):
                index = _column_index(cell.attrib.get("r"))
                while len(cells) <= index:
                    cells.append("")
                cell_type = cell.attrib.get("t")
                value = ""
                if cell_type == "inlineStr":
                    value = "".join(
                        node.text or ""
                        for node in cell.iter()
                        if _local_name(node.tag) == "t"
                    )
                else:
                    value_node = next(
                        (node for node in cell if _local_name(node.tag) == "v"), None
                    )
                    value = value_node.text or "" if value_node is not None else ""
                    if cell_type == "s" and value:
                        try:
                            value = shared[int(value)]
                        except (ValueError, IndexError):
                            raise ValueError("XLSX 共享字符串索引无效")
                    elif cell_type == "b":
                        value = "TRUE" if value == "1" else "FALSE"
                cells[index] = value
            rows.append(cells)
        return rows


def _normalise_rows(rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    while rows and not any(str(value).strip() for value in rows[0]):
        rows.pop(0)
    if not rows:
        raise ValueError("文件没有可用数据")
    width = max(len(row) for row in rows)
    if width == 0:
        raise ValueError("文件没有列")
    headers: list[str] = []
    used: dict[str, int] = {}
    for index, value in enumerate(rows[0], 1):
        header = str(value or "").strip() or f"column_{index}"
        used[header] = used.get(header, 0) + 1
        if used[header] > 1:
            header = f"{header}_{used[header]}"
        headers.append(header[:255])
    while len(headers) < width:
        headers.append(f"column_{len(headers) + 1}")
    data = [(row + [""] * (width - len(row)))[:width] for row in rows[1:]]
    return headers, data


def _infer_type(values: list[str]) -> str:
    non_empty = [
        value.strip() for value in values if value is not None and value.strip()
    ]
    if not non_empty:
        return "TEXT"
    lowered = {value.lower() for value in non_empty}
    if lowered <= {"true", "false", "0", "1"}:
        return "BOOLEAN"
    if all(_INT_RE.fullmatch(value) for value in non_empty):
        try:
            if all(-(2**63) <= int(value) <= 2**63 - 1 for value in non_empty):
                return "BIGINT"
        except ValueError:
            pass
    if all(_DECIMAL_RE.fullmatch(value) for value in non_empty):
        try:
            decimals = [Decimal(value) for value in non_empty]
            scale = max(max(0, -item.as_tuple().exponent) for item in decimals)
            precision = max(len(item.as_tuple().digits) for item in decimals) + scale
            if precision <= 30 and scale <= 6:
                return "DECIMAL(30,6)"
        except InvalidOperation:
            pass
    if all(_DATE_RE.fullmatch(value) for value in non_empty):
        return "DATE"
    if all(_DATETIME_RE.fullmatch(value) for value in non_empty):
        return "DATETIME"
    return "TEXT"


def _typed_value(value: str, sql_type: str):
    """兼容旧测试路径；权威转换规则位于 Domain。"""

    return typed_dataset_value(value, sql_type)


class DatasetQuotaError(ValueError):
    """上传目录配额不足。"""


def _column_profile(
    physical_name: str,
    display_name: str,
    sql_type: str,
    values: list[str],
) -> dict:
    cleaned = [str(value or "").strip() for value in values]
    non_null = [value for value in cleaned if value]
    distinct = list(dict.fromkeys(non_null))
    return {
        "name": physical_name,
        "display_name": display_name,
        "type": sql_type,
        "nullable": len(non_null) != len(cleaned),
        "null_count": len(cleaned) - len(non_null),
        "distinct_count_sample": len(distinct),
        "examples": distinct[:3],
        "profile_sample_size": len(cleaned),
    }


class AdhocDatasetService:
    """上传数据集目录、不可变版本、配额、隔离建表与补偿式清理。"""

    def __init__(
        self,
        *,
        storage: AdhocDatasetStoragePort,
        owner_id: str,
        workspace_id: str,
        settings: AdhocDatasetSettings = AdhocDatasetSettings(),
        masking_rules: tuple[MaskingRule, ...] = (),
    ):
        self.storage = storage
        self.settings = settings
        self.owner_id = str(owner_id).strip()
        self.workspace_id = str(workspace_id).strip()
        if not self.owner_id or not self.workspace_id:
            raise ValueError("上传数据服务必须绑定 owner_id 与 workspace_id")
        self.masking_rules = masking_rules

    @staticmethod
    def public_dict(item: AdhocDatasetRecord) -> dict:
        return {
            "dataset_id": item.id,
            "asset_id": item.asset_id,
            "version_id": item.version_id,
            "version_no": int(item.version_no or 1),
            "asset_kind": item.asset_kind or "managed_upload",
            "retention_mode": item.retention_mode or "temporary",
            "filename": item.filename,
            "format": item.file_format,
            "row_count": item.row_count,
            "column_count": item.column_count,
            "columns": [dict(value) for value in item.columns],
            "preview": [dict(value) for value in item.preview],
            "status": "ready" if item.status == "active" else item.status,
            "schema_fingerprint": item.schema_fingerprint,
            "content_checksum": item.content_checksum,
            "byte_size": int(item.byte_size or 0),
            "parser_profile": item.parser_profile,
            "lifecycle_revision": int(item.lifecycle_revision or 1),
            "managed": bool(item.managed),
            "created_at": str(item.created_at),
            "updated_at": str(item.updated_at or item.created_at),
            "expires_at": str(item.expires_at),
            "cleanup_error": item.cleanup_error,
        }

    async def list_catalog(
        self,
        *,
        limit: int = 20,
        cursor: str | None = None,
        include_terminal: bool = False,
    ) -> dict:
        await self.cleanup_expired()
        safe_limit = max(1, min(int(limit), 50))
        rows = await self.storage.list_catalog(
            self.owner_id,
            self.workspace_id,
            limit=safe_limit + 1,
            before_id=cursor,
            include_terminal=include_terminal,
        )
        has_more = len(rows) > safe_limit
        rows = rows[:safe_limit]
        return {
            "items": [self.public_dict(item) for item in rows],
            "next_cursor": rows[-1].id if has_more and rows else None,
            "has_more": has_more,
        }

    async def get_dataset(self, dataset_id: str) -> AdhocDatasetRecord | None:
        if not re.fullmatch(r"[0-9A-Za-z_-]{8,64}", dataset_id or ""):
            return None
        await self.cleanup_expired()
        return await self.storage.get_active(
            dataset_id, self.owner_id, self.workspace_id
        )

    async def get_scoped_dataset(self, dataset_id: str) -> AdhocDatasetRecord | None:
        if not re.fullmatch(r"[0-9A-Za-z_-]{8,64}", dataset_id or ""):
            return None
        return await self.storage.get_scoped(
            dataset_id, self.owner_id, self.workspace_id
        )

    async def get_datasets(self, dataset_ids: list[str]) -> list[AdhocDatasetRecord]:
        unique_ids = list(dict.fromkeys(str(value) for value in dataset_ids if value))
        if len(unique_ids) > self.settings.max_selected_datasets:
            raise DatasetAccessError(
                f"单轮最多选择 {self.settings.max_selected_datasets} 个上传数据集",
                code="dataset_selection_limit",
            )
        await self.cleanup_expired()
        rows = await self.storage.get_many_active(
            unique_ids, self.owner_id, self.workspace_id
        )
        if len(rows) != len(unique_ids):
            raise DatasetAccessError(
                "至少一个上传数据集不存在、已过期或不属于当前用户/工作空间"
            )
        now = datetime.utcnow()
        for row in rows:
            if row.status not in READY_DATASET_STATES:
                raise DatasetAccessError("上传数据集版本当前不可查询")
            if row.expires_at and row.expires_at <= now:
                raise DatasetExpiredError("上传数据集版本已过期")
        return rows

    async def quota(self) -> dict:
        usage = await self.storage.quota_usage(self.owner_id, self.workspace_id)
        limits = {
            "active_versions": self.settings.max_active_datasets,
            "bytes": self.settings.max_total_bytes,
            "rows": self.settings.max_total_rows,
        }
        return {
            "usage": usage,
            "limits": limits,
            "remaining": {
                key: max(0, int(limits[key]) - int(usage[key])) for key in limits
            },
        }

    async def _enforce_quota(self, *, byte_size: int, rows: int) -> None:
        quota = await self.quota()
        usage = quota["usage"]
        limits = quota["limits"]
        if usage["active_versions"] + 1 > limits["active_versions"]:
            raise DatasetQuotaError("上传数据集数量已达到当前工作空间配额")
        if usage["bytes"] + byte_size > limits["bytes"]:
            raise DatasetQuotaError("上传数据集总容量将超过当前工作空间配额")
        if usage["rows"] + rows > limits["rows"]:
            raise DatasetQuotaError("上传数据集总行数将超过当前工作空间配额")

    async def create_dataset(
        self,
        filename: str,
        content: bytes,
        *,
        idempotency_key: str | None = None,
        retention_mode: str = "temporary",
        asset_id: str | None = None,
    ) -> dict:
        await self.cleanup_expired()
        if not filename:
            raise ValueError("请提供文件名")
        if retention_mode not in {"temporary", "managed"}:
            raise ValueError("retention_mode 仅支持 temporary 或 managed")
        key = str(idempotency_key or "").strip()[:128] or None
        if key:
            existing = await self.storage.find_idempotent(
                key, self.owner_id, self.workspace_id
            )
            if existing is not None:
                return {
                    **self.public_dict(existing),
                    "idempotent_replay": True,
                }
        if len(content) > self.settings.max_file_bytes:
            raise ValueError(
                f"文件大小超过上限 {self.settings.max_file_bytes // 1024 // 1024} MB"
            )
        suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if suffix not in {"csv", "xlsx"}:
            raise ValueError("仅支持 CSV 和 XLSX 文件；暂不支持 XLS/Parquet")
        rows = (
            _parse_csv(content)
            if suffix == "csv"
            else _parse_xlsx(
                content,
                max_expanded_bytes=(
                    self.settings.max_file_bytes * self.settings.xlsx_expansion_ratio
                ),
            )
        )
        headers, data = _normalise_rows(rows)
        if len(headers) > self.settings.max_columns:
            raise ValueError(f"列数超过上限 {self.settings.max_columns}")
        if len(data) > self.settings.max_rows:
            raise ValueError(f"数据行数超过上限 {self.settings.max_rows}")
        if any(
            len(value) > self.settings.max_cell_chars for row in data for value in row
        ):
            raise ValueError(f"单元格长度超过上限 {self.settings.max_cell_chars}")
        await self._enforce_quota(byte_size=len(content), rows=len(data))

        physical_columns = [f"c_{index:03d}" for index in range(1, len(headers) + 1)]
        types = [
            _infer_type([row[index] for row in data]) for index in range(len(headers))
        ]
        columns = [
            _column_profile(
                physical,
                display,
                sql_type,
                [row[index] for row in data],
            )
            for index, (physical, display, sql_type) in enumerate(
                zip(physical_columns, headers, types)
            )
        ]
        dataset_id = str(uuid.uuid4())
        version_id = dataset_id
        requested_asset_id = str(asset_id or "").strip()
        latest = None
        if requested_asset_id:
            if not re.fullmatch(r"[0-9a-fA-F-]{36}", requested_asset_id):
                raise ValueError("asset_id 格式无效")
            candidate = await self.storage.latest_for_asset(
                requested_asset_id, self.owner_id, self.workspace_id
            )
            if candidate is None:
                raise DatasetAccessError("逻辑数据集不存在或无权创建新版本")
            latest = candidate
        logical_asset_id = requested_asset_id or str(uuid.uuid4())
        version_no = int(latest.version_no or 1) + 1 if latest else 1
        table_name = f"da_upload_{dataset_id.replace('-', '')}"
        if not _IDENTIFIER_RE.fullmatch(table_name):
            raise ValueError("上传表内部标识生成失败")
        preview = mask_rows(
            [dict(zip(headers, row)) for row in data[:10]],
            columns=headers,
            rules=self.masking_rules,
        )
        checksum = "sha256:" + hashlib.sha256(content).hexdigest()
        schema_fingerprint = stable_fingerprint(
            [
                {
                    "name": column["name"],
                    "display_name": column["display_name"],
                    "type": column["type"],
                }
                for column in columns
            ]
        )
        ttl = (
            self.settings.managed_ttl_seconds
            if retention_mode == "managed"
            else self.settings.ttl_seconds
        )
        expires_at = datetime.now(UTC).replace(tzinfo=None) + timedelta(seconds=ttl)
        item = AdhocDatasetRecord(
            id=dataset_id,
            asset_id=logical_asset_id,
            version_id=version_id,
            version_no=version_no,
            asset_kind="managed_upload",
            retention_mode=retention_mode,
            owner_id=self.owner_id,
            workspace_id=self.workspace_id,
            table_name=table_name,
            filename=filename[:255],
            file_format=suffix,
            row_count=len(data),
            column_count=len(headers),
            columns=tuple(columns),
            preview=tuple(preview),
            status="materializing",
            content_checksum=checksum,
            schema_fingerprint=schema_fingerprint,
            parser_profile="upload-parser-v2",
            byte_size=len(content),
            idempotency_key=key,
            policy_version="upload-policy-v2",
            managed=retention_mode == "managed",
            expires_at=expires_at,
        )
        materialized = await self.storage.materialize(
            DatasetMaterializationSpec(
                record=item,
                physical_columns=tuple(physical_columns),
                column_types=tuple(types),
                rows=tuple(tuple(row) for row in data),
            )
        )
        return self.public_dict(materialized)

    async def delete_dataset(self, dataset_id: str) -> bool:
        item = await self.get_scoped_dataset(dataset_id)
        if item is None:
            return False
        return await self.storage.delete(dataset_id, self.owner_id, self.workspace_id)

    async def archive_dataset(self, dataset_id: str) -> dict | None:
        item = await self.storage.archive(dataset_id, self.owner_id, self.workspace_id)
        return self.public_dict(item) if item is not None else None

    async def extend_dataset(self, dataset_id: str, seconds: int) -> dict | None:
        safe_seconds = max(60, min(int(seconds), self.settings.max_extension_seconds))
        item = await self.storage.extend(
            dataset_id,
            self.owner_id,
            self.workspace_id,
            seconds=safe_seconds,
        )
        return self.public_dict(item) if item is not None else None

    async def cleanup_expired(self) -> int:
        return await self.storage.cleanup_expired()

    async def reconcile(self) -> dict:
        """重试待清理资源，并报告登记资源与 DW 物理资源的差异。"""
        return await self.storage.reconcile()

    @staticmethod
    def asset_identity(item: AdhocDatasetRecord, logical_name: str) -> AssetIdentity:
        return AssetIdentity(
            asset_id=str(item.asset_id),
            version_id=str(item.version_id),
            kind=AssetKind(str(item.asset_kind or "managed_upload")),
            source="dw-upload",
            catalog="",
            schema="",
            table=str(item.table_name),
            logical_name=logical_name,
            workspace_id=str(item.workspace_id),
            owner_id=str(item.owner_id),
            state="ready" if item.status == "active" else str(item.status),
            schema_fingerprint=str(item.schema_fingerprint or ""),
            policy_version=str(item.policy_version or "upload-policy-v2"),
            expires_at=item.expires_at,
        )
