"""字段值索引策略、记录构造与 ES 发布的出站适配器。"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from app.application.ports.metadata_indexing import MetadataIndexSettings
from app.domain.metadata.models import ValueIndexDocument
from app.infrastructure.persistence.mysql.repositories.dw_mysql_repository import (
    DWMysqlRepository,
)
from app.infrastructure.search.elasticsearch.repositories.value_es_repository import (
    ValueESRepository,
)
from app.infrastructure.search.elasticsearch.value_retrieval import (
    VALUE_QUERY_MODES,
    VALUE_RETRIEVAL_RESPONSIBILITY,
    normalize_value,
    value_column_decision,
    value_document_id,
    value_index_version,
)


def _get(item: Any, key: str, default: Any = None) -> Any:
    return item.get(key, default) if isinstance(item, dict) else getattr(
        item,
        key,
        default,
    )


class ElasticsearchValueIndexAdapter:
    """实现字段值索引的构建、状态、检索和回退端口。"""

    def __init__(
        self,
        *,
        dw_repo: DWMysqlRepository,
        value_repo: ValueESRepository,
        settings: MetadataIndexSettings,
    ) -> None:
        self._dw_repo = dw_repo
        self._value_repo = value_repo
        self._settings = settings

    async def status(
        self,
        *,
        workspace_id: str,
        columns: Sequence[Any],
    ) -> dict:
        status = await self._value_repo.active_manifest(workspace_id)
        sync_columns = [column for column in columns if bool(_get(column, "sync"))]
        preliminary = [value_column_decision(column) for column in sync_columns]
        return {
            **status,
            "query_modes": list(VALUE_QUERY_MODES),
            "sync_column_count": len(sync_columns),
            "policy_precheck_allowed_count": sum(
                1 for item in preliminary if item.allowed
            ),
            "policy_precheck_blocked": [
                item.to_dict() for item in preliminary if not item.allowed
            ],
            "retained_versions": await self._value_repo.history(workspace_id),
        }

    async def rollback(self, *, workspace_id: str, physical_index: str) -> dict:
        return await self._value_repo.activate_retained(
            workspace_id=workspace_id,
            physical_index=physical_index,
        )

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
    ) -> tuple[list[dict], int, dict]:
        return await self._value_repo.search_with_evidence(
            keyword,
            workspace_id=workspace_id,
            query_mode=query_mode,
            allowed_table_ids=list(allowed_table_ids),
            allowed_column_ids=list(allowed_column_ids),
            table_id=table_id,
            column_id=column_id,
            page=page,
            page_size=page_size,
        )

    async def rebuild(
        self,
        *,
        workspace_id: str,
        columns: Sequence[Any],
    ) -> dict:
        resolved_workspace = str(workspace_id or "").strip()
        if not resolved_workspace:
            raise ValueError("值索引构建必须绑定 workspace_id")
        limit = self._settings.max_distinct_values_per_column
        indexed_at = datetime.now(timezone.utc).isoformat()
        staged: list[tuple[Any, list[tuple[str, str]]]] = []
        admitted: list[dict] = []
        blocked: list[dict] = []
        ignored = 0
        content_fingerprints: list[dict] = []

        for column in columns:
            if not bool(_get(column, "sync", False)):
                ignored += 1
                continue
            preliminary = value_column_decision(column)
            if not preliminary.allowed:
                blocked.append(preliminary.to_dict())
                continue
            raw_values = await self._dw_repo.get_column_values(
                str(_get(column, "table_id") or ""),
                str(_get(column, "name") or ""),
                limit + 1,
            )
            decision = value_column_decision(
                column,
                distinct_count=len(raw_values),
                values=raw_values,
            )
            if not decision.allowed:
                blocked.append(decision.to_dict())
                continue

            deduped: dict[str, str] = {}
            for raw in raw_values:
                if raw is None:
                    continue
                value = str(raw).strip()
                normalized = normalize_value(value)
                if normalized:
                    deduped.setdefault(normalized, value)
            normalized_values = sorted(deduped.items())
            column_id = str(_get(column, "id") or "")
            staged.append((column, normalized_values))
            admitted.append({
                "column_id": column_id,
                "table_id": str(_get(column, "table_id") or ""),
                "document_count": len(normalized_values),
                "reason": decision.reason,
            })
            content_fingerprints.append({
                "column_id": column_id,
                "document_count": len(normalized_values),
                "content_hash": "sha256:" + hashlib.sha256(
                    "\x1e".join(
                        value for value, _ in normalized_values
                    ).encode("utf-8")
                ).hexdigest(),
            })

        version = value_index_version(
            workspace_id=resolved_workspace,
            columns=columns,
            content_fingerprints=content_fingerprints,
        )
        documents: list[ValueIndexDocument] = []
        for column, normalized_values in staged:
            column_id = str(_get(column, "id") or "")
            table_id = str(_get(column, "table_id") or "")
            for normalized, value in normalized_values:
                documents.append({
                    "id": value_document_id(
                        resolved_workspace,
                        column_id,
                        normalized,
                    ),
                    "workspace_id": resolved_workspace,
                    "value": value,
                    "normalized_value": normalized,
                    "type": str(_get(column, "type") or ""),
                    "column_id": column_id,
                    "column_name": str(_get(column, "name") or ""),
                    "table_id": table_id,
                    "table_name": table_id,
                    "index_version": version,
                    "indexed_at": indexed_at,
                    "source_updated_at": None,
                    "is_active": True,
                    "responsibility": VALUE_RETRIEVAL_RESPONSIBILITY,
                })

        publication = await self._value_repo.publish_workspace(
            workspace_id=resolved_workspace,
            index_version=version,
            values=documents,
        )
        return {
            **publication,
            "admitted_columns": admitted,
            "blocked_columns": blocked,
            "admitted_column_count": len(admitted),
            "blocked_column_count": len(blocked),
            "ignored_column_count": ignored,
        }


__all__ = ["ElasticsearchValueIndexAdapter"]
