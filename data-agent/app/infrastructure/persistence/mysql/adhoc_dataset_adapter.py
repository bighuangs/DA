"""上传数据集 Application 端口的 MySQL/DW 适配器。"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta
from typing import Protocol

from sqlalchemy import text

from app.application.ports.data_sources import (
    AdhocDatasetStoragePort,
    DatasetMaterializationSpec,
)
from app.domain.data_sources.models import (
    AdhocDatasetRecord,
    DatasetCleanupError,
    reconciled_terminal_status,
    typed_dataset_value,
)
from app.domain.data_sources.context import READY_DATASET_STATES
from app.infrastructure.cache.result_cache import result_cache
from app.infrastructure.clients.mysql_client_manager import (
    dw_mysql_client_manager,
    meta_mysql_client_manager,
)
from app.infrastructure.observability.logging import logger
from app.infrastructure.persistence.mysql.records.adhoc_dataset_mysql import (
    AdhocDatasetMySQL,
)
from app.infrastructure.persistence.mysql.records.conversation_data_context_mysql import (
    DatasetResourceLedgerMySQL,
)
from app.infrastructure.persistence.mysql.repositories.adhoc_dataset_repository import (
    AdhocDatasetRepository,
)
from app.infrastructure.persistence.mysql.repositories.data_context_repository import (
    DatasetResourceLedgerRepository,
)


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class _CacheInvalidator(Protocol):
    def clear(self) -> None: ...


class _WarningLogger(Protocol):
    def warning(self, message: str, *args: object) -> object: ...


def _record(item: AdhocDatasetMySQL) -> AdhocDatasetRecord:
    """把会话绑定 ORM 行复制为稳定领域快照。"""

    return AdhocDatasetRecord(
        id=str(item.id),
        asset_id=str(item.asset_id),
        version_id=str(item.version_id),
        version_no=int(item.version_no or 1),
        asset_kind=str(item.asset_kind or "managed_upload"),
        retention_mode=str(item.retention_mode or "temporary"),
        owner_id=str(item.owner_id),
        workspace_id=str(item.workspace_id),
        table_name=str(item.table_name),
        filename=str(item.filename),
        file_format=str(item.file_format),
        row_count=int(item.row_count or 0),
        column_count=int(item.column_count or 0),
        columns=tuple(dict(value) for value in (item.columns or [])),
        preview=tuple(dict(value) for value in (item.preview or [])),
        status=str(item.status),
        content_checksum=item.content_checksum,
        schema_fingerprint=item.schema_fingerprint,
        parser_profile=str(item.parser_profile or "upload-parser-v2"),
        byte_size=int(item.byte_size or 0),
        idempotency_key=item.idempotency_key,
        policy_version=str(item.policy_version or "upload-policy-v2"),
        lifecycle_revision=int(item.lifecycle_revision or 1),
        managed=bool(item.managed),
        created_at=item.created_at,
        updated_at=item.updated_at,
        expires_at=item.expires_at,
        deleted_at=item.deleted_at,
        error=item.error,
        cleanup_attempts=int(item.cleanup_attempts or 0),
        cleanup_error=item.cleanup_error,
        archived_at=item.archived_at,
    )


def _new_orm(item: AdhocDatasetRecord) -> AdhocDatasetMySQL:
    """把用例创建命令映射为新的 ORM 行。"""

    return AdhocDatasetMySQL(
        id=item.id,
        asset_id=item.asset_id,
        version_id=item.version_id,
        version_no=item.version_no,
        asset_kind=item.asset_kind,
        retention_mode=item.retention_mode,
        owner_id=item.owner_id,
        workspace_id=item.workspace_id,
        table_name=item.table_name,
        filename=item.filename,
        file_format=item.file_format,
        row_count=item.row_count,
        column_count=item.column_count,
        columns=[dict(value) for value in item.columns],
        preview=[dict(value) for value in item.preview],
        status=item.status,
        content_checksum=item.content_checksum,
        schema_fingerprint=item.schema_fingerprint,
        parser_profile=item.parser_profile,
        byte_size=item.byte_size,
        idempotency_key=item.idempotency_key,
        policy_version=item.policy_version,
        lifecycle_revision=item.lifecycle_revision,
        managed=item.managed,
        expires_at=item.expires_at,
    )


class MysqlAdhocDatasetAdapter(AdhocDatasetStoragePort):
    """封装 Meta 目录、DW 物化、补偿清理与缓存失效。"""

    def __init__(
        self,
        *,
        dw_manager=dw_mysql_client_manager,
        meta_manager=meta_mysql_client_manager,
        cache: _CacheInvalidator = result_cache,
        log: _WarningLogger = logger,
    ) -> None:
        self._dw_manager = dw_manager
        self._meta_manager = meta_manager
        self._cache = cache
        self._log = log

    async def list_catalog(
        self,
        owner_id: str,
        workspace_id: str,
        *,
        limit: int,
        before_id: str | None = None,
        include_terminal: bool = False,
    ) -> list[AdhocDatasetRecord]:
        async with self._meta_manager.session_factory() as session:
            rows = await AdhocDatasetRepository(session).list_catalog(
                owner_id,
                workspace_id,
                limit=limit,
                before_id=before_id,
                include_terminal=include_terminal,
            )
        return [_record(item) for item in rows]

    async def get_active(
        self, dataset_id: str, owner_id: str, workspace_id: str
    ) -> AdhocDatasetRecord | None:
        async with self._meta_manager.session_factory() as session:
            item = await AdhocDatasetRepository(session).get_active(
                dataset_id, owner_id, workspace_id
            )
        return _record(item) if item is not None else None

    async def get_scoped(
        self, dataset_id: str, owner_id: str, workspace_id: str
    ) -> AdhocDatasetRecord | None:
        async with self._meta_manager.session_factory() as session:
            item = await AdhocDatasetRepository(session).get_scoped(
                dataset_id, owner_id, workspace_id
            )
        return _record(item) if item is not None else None

    async def get_many_active(
        self, dataset_ids: list[str], owner_id: str, workspace_id: str
    ) -> list[AdhocDatasetRecord]:
        async with self._meta_manager.session_factory() as session:
            rows = await AdhocDatasetRepository(session).get_many_active(
                dataset_ids, owner_id, workspace_id
            )
        return [_record(item) for item in rows]

    async def find_idempotent(
        self, idempotency_key: str, owner_id: str, workspace_id: str
    ) -> AdhocDatasetRecord | None:
        async with self._meta_manager.session_factory() as session:
            item = await AdhocDatasetRepository(session).find_idempotent(
                idempotency_key, owner_id, workspace_id
            )
        return _record(item) if item is not None else None

    async def latest_for_asset(
        self, asset_id: str, owner_id: str, workspace_id: str
    ) -> AdhocDatasetRecord | None:
        async with self._meta_manager.session_factory() as session:
            item = await AdhocDatasetRepository(session).latest_for_asset(
                asset_id, owner_id, workspace_id
            )
        return _record(item) if item is not None else None

    async def quota_usage(self, owner_id: str, workspace_id: str) -> dict[str, int]:
        async with self._meta_manager.session_factory() as session:
            return await AdhocDatasetRepository(session).quota_usage(
                owner_id, workspace_id
            )

    async def materialize(self, spec: DatasetMaterializationSpec) -> AdhocDatasetRecord:
        item = _new_orm(spec.record)
        self._validate_materialization(spec)
        try:
            async with self._meta_manager.session_factory() as meta_session:
                item = await AdhocDatasetRepository(meta_session).create(item)
                await DatasetResourceLedgerRepository(meta_session).create(
                    DatasetResourceLedgerMySQL(
                        id=str(uuid.uuid4()),
                        dataset_version_id=item.version_id,
                        owner_id=item.owner_id,
                        workspace_id=item.workspace_id,
                        resource_kind="dw_table",
                        resource_ref=item.table_name,
                        status="allocating",
                    )
                )

            create_sql = "CREATE TABLE `{}` ({})".format(
                item.table_name,
                ", ".join(
                    f"`{name}` {sql_type} NULL"
                    for name, sql_type in zip(
                        spec.physical_columns, spec.column_types, strict=True
                    )
                ),
            )
            async with self._dw_manager.session_factory() as dw_session:
                await dw_session.execute(text(create_sql))
                if spec.rows:
                    insert_sql = text(
                        f"INSERT INTO `{item.table_name}` "
                        f"({', '.join(f'`{name}`' for name in spec.physical_columns)}) "
                        f"VALUES ({', '.join(f':{name}' for name in spec.physical_columns)})"
                    )
                    params = [
                        {
                            name: typed_dataset_value(
                                row[index], spec.column_types[index]
                            )
                            for index, name in enumerate(spec.physical_columns)
                        }
                        for row in spec.rows
                    ]
                    for start in range(0, len(params), 500):
                        await dw_session.execute(
                            insert_sql, params[start : start + 500]
                        )
                await dw_session.commit()

            async with self._meta_manager.session_factory() as meta_session:
                repo = AdhocDatasetRepository(meta_session)
                current = await repo.get_scoped(
                    item.id, item.owner_id, item.workspace_id
                )
                if current is None:
                    raise RuntimeError("上传数据版本元数据在物化后丢失")
                current.status = "ready"
                current.error = None
                current = await repo.save(current)
                ledger = await DatasetResourceLedgerRepository(
                    meta_session
                ).for_version(item.version_id)
                for resource in ledger:
                    resource.status = "allocated"
                await meta_session.commit()
            return _record(current)
        except Exception as exc:
            cleanup_error = None
            try:
                await self._drop_table(item.table_name)
            except Exception as drop_exc:
                cleanup_error = str(drop_exc)
            async with self._meta_manager.session_factory() as meta_session:
                repo = AdhocDatasetRepository(meta_session)
                current = await repo.get_scoped(
                    item.id, item.owner_id, item.workspace_id
                )
                if current is not None:
                    current.status = "cleanup_failed" if cleanup_error else "failed"
                    current.error = str(exc)[:4000]
                    current.cleanup_error = (
                        cleanup_error[:4000] if cleanup_error else None
                    )
                    await repo.save(current)
                ledger_repo = DatasetResourceLedgerRepository(meta_session)
                if cleanup_error:
                    await ledger_repo.mark_cleanup_failed(
                        item.version_id, cleanup_error
                    )
                else:
                    await ledger_repo.mark_released(item.version_id)
            raise

    async def delete(self, dataset_id: str, owner_id: str, workspace_id: str) -> bool:
        async with self._meta_manager.session_factory() as session:
            repo = AdhocDatasetRepository(session)
            item = await repo.get_scoped(dataset_id, owner_id, workspace_id)
            if item is None:
                return False
            if item.status == "deleted":
                return True
            table_name = item.table_name
            version_id = item.version_id
            item.status = "delete_pending"
            item.cleanup_error = None
            await repo.save(item)
        try:
            await self._drop_table(table_name)
        except Exception as exc:
            async with self._meta_manager.session_factory() as session:
                repo = AdhocDatasetRepository(session)
                current = await repo.get_scoped(dataset_id, owner_id, workspace_id)
                if current is not None:
                    current.status = "cleanup_failed"
                    current.cleanup_attempts += 1
                    current.cleanup_error = str(exc)[:4000]
                    await repo.save(current)
                await DatasetResourceLedgerRepository(session).mark_cleanup_failed(
                    version_id, str(exc)
                )
            raise DatasetCleanupError(
                "上传数据物理资源清理失败，已进入可重试状态"
            ) from exc
        async with self._meta_manager.session_factory() as session:
            repo = AdhocDatasetRepository(session)
            current = await repo.get_scoped(dataset_id, owner_id, workspace_id)
            if current is not None:
                current.status = "deleted"
                current.deleted_at = datetime.utcnow()
                current.cleanup_error = None
                await repo.save(current)
            await DatasetResourceLedgerRepository(session).mark_released(version_id)
        self._invalidate_cache()
        return True

    async def archive(
        self, dataset_id: str, owner_id: str, workspace_id: str
    ) -> AdhocDatasetRecord | None:
        async with self._meta_manager.session_factory() as session:
            repo = AdhocDatasetRepository(session)
            item = await repo.get_scoped(dataset_id, owner_id, workspace_id)
            if item is None or item.status not in READY_DATASET_STATES:
                return None
            item.status = "archived"
            item.archived_at = datetime.utcnow()
            item = await repo.save(item)
        self._invalidate_cache()
        return _record(item)

    async def extend(
        self,
        dataset_id: str,
        owner_id: str,
        workspace_id: str,
        *,
        seconds: int,
    ) -> AdhocDatasetRecord | None:
        async with self._meta_manager.session_factory() as session:
            repo = AdhocDatasetRepository(session)
            item = await repo.get_scoped(dataset_id, owner_id, workspace_id)
            if item is None or item.status not in READY_DATASET_STATES:
                return None
            now = datetime.utcnow()
            anchor = max(item.expires_at or now, now)
            item.expires_at = anchor + timedelta(seconds=seconds)
            item = await repo.save(item)
        return _record(item)

    async def cleanup_expired(self) -> int:
        now = datetime.utcnow()
        async with self._meta_manager.session_factory() as session:
            expired = await AdhocDatasetRepository(session).list_expired(now)
            for item in expired:
                item.status = "expiring"
            if expired:
                await session.commit()
        cleaned = 0
        for item in expired:
            try:
                await self._drop_table(item.table_name)
                async with self._meta_manager.session_factory() as session:
                    repo = AdhocDatasetRepository(session)
                    current = await repo.get_scoped(
                        item.id, item.owner_id, item.workspace_id
                    )
                    if current is not None:
                        current.status = "expired"
                        current.deleted_at = now
                        current.cleanup_error = None
                        await repo.save(current)
                    await DatasetResourceLedgerRepository(session).mark_released(
                        item.version_id
                    )
                self._invalidate_cache()
                cleaned += 1
            except Exception as exc:
                async with self._meta_manager.session_factory() as session:
                    repo = AdhocDatasetRepository(session)
                    current = await repo.get_scoped(
                        item.id, item.owner_id, item.workspace_id
                    )
                    if current is not None:
                        current.status = "cleanup_failed"
                        current.cleanup_attempts += 1
                        current.cleanup_error = str(exc)[:4000]
                        await repo.save(current)
                    await DatasetResourceLedgerRepository(session).mark_cleanup_failed(
                        item.version_id, str(exc)
                    )
                self._log.warning(
                    "清理过期上传表失败 version={} error_type={}",
                    item.version_id,
                    type(exc).__name__,
                )
        return cleaned

    async def reconcile(self) -> dict[str, int]:
        retried = 0
        recovered = 0
        async with self._meta_manager.session_factory() as session:
            pending = await AdhocDatasetRepository(session).list_cleanup_pending()
        for item in pending:
            retried += 1
            try:
                await self._drop_table(item.table_name)
                async with self._meta_manager.session_factory() as session:
                    repo = AdhocDatasetRepository(session)
                    current = await repo.get_scoped(
                        item.id, item.owner_id, item.workspace_id
                    )
                    if current is not None:
                        # cleanup_failed 有多个来源；恢复后必须回到对应终态。
                        current.status = reconciled_terminal_status(item)
                        current.deleted_at = datetime.utcnow()
                        current.cleanup_error = None
                        await repo.save(current)
                    await DatasetResourceLedgerRepository(session).mark_released(
                        item.version_id
                    )
                recovered += 1
            except Exception:
                continue
        return {
            "retried": retried,
            "recovered": recovered,
            "remaining": retried - recovered,
        }

    @staticmethod
    def _validate_materialization(spec: DatasetMaterializationSpec) -> None:
        record = spec.record
        if not _IDENTIFIER_RE.fullmatch(record.table_name or ""):
            raise ValueError("上传表内部标识无效")
        if len(spec.physical_columns) != len(spec.column_types):
            raise ValueError("上传列名与类型数量不一致")
        if not spec.physical_columns or any(
            not _IDENTIFIER_RE.fullmatch(value) for value in spec.physical_columns
        ):
            raise ValueError("上传物理列标识无效")
        if any(len(row) != len(spec.physical_columns) for row in spec.rows):
            raise ValueError("上传数据行宽与列数不一致")

    def _invalidate_cache(self) -> None:
        # 删除、归档或过期属于低频安全事件，全量清空避免旧版本结果复用。
        self._cache.clear()

    async def _drop_table(self, table_name: str) -> None:
        if not _IDENTIFIER_RE.fullmatch(table_name or ""):
            raise DatasetCleanupError("拒绝清理非法物理资源标识")
        async with self._dw_manager.session_factory() as session:
            await session.execute(text(f"DROP TABLE IF EXISTS `{table_name}`"))
            await session.commit()


__all__ = ["MysqlAdhocDatasetAdapter"]
