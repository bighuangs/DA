from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from elasticsearch import AsyncElasticsearch
from elasticsearch import NotFoundError

from app.config.app_config import app_config
from app.infrastructure.observability.logging import logger
from app.infrastructure.search.elasticsearch.value_retrieval import (
    VALUE_QUERY_MODES,
    VALUE_RETRIEVAL_RESPONSIBILITY,
    VALUE_SEARCH_AUDIT_SCHEMA,
    build_value_search_query,
    normalize_value,
    value_index_alias,
    value_physical_index_name,
    value_physical_index_prefix,
    value_query_fingerprint,
)
from app.infrastructure.search.elasticsearch.records.value_info_es import ValueInfoES


def _body(response: Any) -> Any:
    return getattr(response, "body", response)


def _total_value(total: Any) -> int:
    if isinstance(total, dict):
        return int(total.get("value") or 0)
    return int(total or 0)


class ValueESRepository:
    """Workspace-isolated, versioned Elasticsearch field-value index.

    Qdrant owns metadata-entity recall.  This repository is the only retrieval
    surface that may return a real field value for use as a SQL literal.
    """

    index_name = app_config.es.index_name

    def __init__(self, client: AsyncElasticsearch):
        self.client = client

    @staticmethod
    def _mappings(index_version: str | None = None) -> dict:
        analyzer = app_config.es.value_analyzer
        return {
            "dynamic": "strict",
            "_meta": {
                "contract_version": app_config.es.value_contract_version,
                "mapping_version": app_config.es.value_mapping_version,
                "normalization_version": app_config.es.value_normalization_version,
                "responsibility": VALUE_RETRIEVAL_RESPONSIBILITY,
                "index_version": index_version,
            },
            "properties": {
                "id": {"type": "keyword"},
                "workspace_id": {"type": "keyword"},
                "value": {
                    "type": "text",
                    "analyzer": analyzer,
                    "search_analyzer": analyzer,
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 2048}},
                },
                "normalized_value": {"type": "keyword", "ignore_above": 2048},
                "type": {"type": "keyword"},
                "column_id": {"type": "keyword"},
                "column_name": {"type": "keyword"},
                "table_id": {"type": "keyword"},
                "table_name": {"type": "keyword"},
                "index_version": {"type": "keyword"},
                "indexed_at": {"type": "date"},
                "source_updated_at": {"type": "date"},
                "is_active": {"type": "boolean"},
                "responsibility": {"type": "keyword"},
            },
        }

    async def _active_indices(self, workspace_id: str) -> list[str]:
        alias = value_index_alias(workspace_id)
        try:
            response = _body(await self.client.indices.get_alias(name=alias))
        except NotFoundError:
            return []
        return sorted(str(name) for name in (response or {}).keys())

    async def active_manifest(self, workspace_id: str) -> dict:
        alias = value_index_alias(workspace_id)
        active_indices = await self._active_indices(workspace_id)
        if not active_indices:
            return {
                "exists": False,
                "ready": False,
                "workspace_id": workspace_id,
                "alias": alias,
                "physical_index": None,
                "index_version": None,
                "document_count": 0,
                "responsibility": VALUE_RETRIEVAL_RESPONSIBILITY,
                "contract_version": app_config.es.value_contract_version,
                "mapping_version": app_config.es.value_mapping_version,
            }

        physical = active_indices[-1]
        version = None
        document_version = None
        count = 0
        mapping_meta: dict[str, Any] = {}
        try:
            mapping_response = _body(
                await self.client.indices.get_mapping(index=physical)
            )
            mapping_meta = (
                (((mapping_response or {}).get(physical) or {}).get("mappings") or {})
                .get("_meta", {})
            )
            version = mapping_meta.get("index_version")
        except (AttributeError, KeyError, NotFoundError):
            version = None
        try:
            response = _body(await self.client.search(
                index=alias,
                query={
                    "bool": {
                        "filter": [
                            {"term": {"workspace_id": workspace_id}},
                            {"term": {"is_active": True}},
                        ]
                    }
                },
                size=1,
                _source=["index_version"],
            ))
            hits = (response or {}).get("hits") or {}
            count = _total_value(hits.get("total"))
            rows = hits.get("hits") or []
            if rows:
                document_version = (rows[0].get("_source") or {}).get("index_version")
                version = version or document_version
        except NotFoundError:
            active_indices = []

        drift_reasons: list[str] = []
        if len(active_indices) != 1:
            drift_reasons.append("active_alias_target_count")
        if not version:
            drift_reasons.append("index_version_missing")
        if (
            mapping_meta.get("contract_version")
            != app_config.es.value_contract_version
        ):
            drift_reasons.append("contract_version_mismatch")
        if (
            mapping_meta.get("mapping_version")
            != app_config.es.value_mapping_version
        ):
            drift_reasons.append("mapping_version_mismatch")
        if mapping_meta.get("responsibility") != VALUE_RETRIEVAL_RESPONSIBILITY:
            drift_reasons.append("responsibility_mismatch")
        if document_version and version != document_version:
            drift_reasons.append("document_version_mismatch")

        return {
            "exists": bool(active_indices),
            "ready": bool(active_indices) and not drift_reasons,
            "workspace_id": workspace_id,
            "alias": alias,
            "physical_index": physical if active_indices else None,
            "index_version": version,
            "document_count": count,
            "responsibility": VALUE_RETRIEVAL_RESPONSIBILITY,
            "contract_version": app_config.es.value_contract_version,
            "mapping_version": app_config.es.value_mapping_version,
            "drift_reasons": drift_reasons,
        }

    async def publish_workspace(
        self,
        *,
        workspace_id: str,
        index_version: str,
        values: Sequence[ValueInfoES],
    ) -> dict:
        """Build, verify and atomically activate one immutable physical index."""
        alias = value_index_alias(workspace_id)
        physical = value_physical_index_name(workspace_id, index_version)
        old_indices = await self._active_indices(workspace_id)
        created = False
        activation_attempted = False
        history_prune_status = "ok"
        try:
            await self.client.indices.create(
                index=physical,
                mappings=self._mappings(index_version),
                settings={
                    "index": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                    }
                },
            )
            created = True
            batch_size = max(1, int(app_config.es.bulk_batch_documents))
            for start in range(0, len(values), batch_size):
                operations: list[dict] = []
                for value in values[start:start + batch_size]:
                    document = dict(value)
                    document_id = str(document["id"])
                    operations.extend((
                        {"index": {"_index": physical, "_id": document_id}},
                        document,
                    ))
                response = _body(await self.client.bulk(operations=operations))
                if bool((response or {}).get("errors")):
                    failures = [
                        item for item in ((response or {}).get("items") or [])
                        if int(next(iter(item.values())).get("status") or 500) >= 300
                    ]
                    raise RuntimeError(
                        f"字段值索引批量写入失败：{len(failures)} 个文档未写入"
                    )

            await self.client.indices.refresh(index=physical)
            count_response = _body(await self.client.count(
                index=physical,
                query={
                    "bool": {
                        "filter": [
                            {"term": {"workspace_id": workspace_id}},
                            {"term": {"index_version": index_version}},
                            {"term": {"is_active": True}},
                        ]
                    }
                },
            ))
            actual_count = int((count_response or {}).get("count") or 0)
            if actual_count != len(values):
                raise RuntimeError(
                    f"字段值索引校验失败：期望 {len(values)}，实际 {actual_count}"
                )

            actions = [
                {"remove": {"index": index, "alias": alias}}
                for index in old_indices
            ]
            actions.append({"add": {"index": physical, "alias": alias}})
            # update_aliases 是原子操作，但网络超时可能令客户端无法判断服务端是否
            # 已完成切换。自此以后禁止异常清理删除新索引，避免删除不确定状态的 active。
            activation_attempted = True
            await self.client.indices.update_aliases(actions=actions)
            try:
                await self._prune_history(workspace_id, active_index=physical)
            except Exception as exc:
                # 历史清理不是发布正确性的前置条件。保留多余旧版本比误删 active
                # 更安全，并将降级状态交给状态页/运维后续处理。
                history_prune_status = "degraded"
                logger.warning(
                    "字段值索引历史清理失败 workspace={} error_type={}",
                    workspace_id,
                    type(exc).__name__,
                )
            logger.info(
                "字段值索引发布完成 workspace={} version={} documents={}",
                workspace_id,
                index_version,
                actual_count,
            )
            return {
                "exists": True,
                "ready": True,
                "workspace_id": workspace_id,
                "alias": alias,
                "physical_index": physical,
                "index_version": index_version,
                "document_count": actual_count,
                "responsibility": VALUE_RETRIEVAL_RESPONSIBILITY,
                "contract_version": app_config.es.value_contract_version,
                "mapping_version": app_config.es.value_mapping_version,
                "previous_active_indices": old_indices,
                "history_prune_status": history_prune_status,
            }
        except Exception:
            if created and not activation_attempted:
                try:
                    await self.client.indices.delete(index=physical)
                except Exception:
                    pass
            raise

    async def _prune_history(self, workspace_id: str, *, active_index: str) -> None:
        keep_previous = max(0, int(app_config.es.retain_previous_versions))
        prefix = value_physical_index_prefix(workspace_id)
        try:
            response = _body(await self.client.indices.get(index=f"{prefix}*"))
        except NotFoundError:
            return
        names = sorted(
            (str(name) for name in (response or {}).keys()),
            reverse=True,
        )
        keep = set([active_index, *[name for name in names if name != active_index][:keep_previous]])
        for name in names:
            if name not in keep:
                await self.client.indices.delete(index=name)

    async def history(self, workspace_id: str) -> list[dict]:
        """List retained immutable versions without exposing indexed values."""
        prefix = value_physical_index_prefix(workspace_id)
        active = set(await self._active_indices(workspace_id))
        try:
            response = _body(await self.client.indices.get(index=f"{prefix}*"))
        except NotFoundError:
            return []
        rows: list[dict] = []
        for name, metadata in sorted(
            (response or {}).items(),
            key=lambda item: str(item[0]),
            reverse=True,
        ):
            mappings = (metadata or {}).get("mappings") or {}
            meta = mappings.get("_meta") or {}
            count_response = _body(await self.client.count(index=name))
            rows.append({
                "physical_index": str(name),
                "index_version": meta.get("index_version"),
                "document_count": int((count_response or {}).get("count") or 0),
                "active": str(name) in active,
                "contract_version": meta.get("contract_version"),
                "mapping_version": meta.get("mapping_version"),
            })
        return rows

    async def activate_retained(
        self,
        *,
        workspace_id: str,
        physical_index: str,
    ) -> dict:
        """Atomically roll the workspace alias back to a retained verified index."""
        prefix = value_physical_index_prefix(workspace_id)
        target = str(physical_index or "")
        if not target.startswith(prefix):
            raise ValueError("目标索引不属于当前工作空间")
        history = await self.history(workspace_id)
        candidate = next(
            (item for item in history if item["physical_index"] == target),
            None,
        )
        if candidate is None:
            raise LookupError("目标字段值索引版本不存在或已清理")
        current = await self._active_indices(workspace_id)
        alias = value_index_alias(workspace_id)
        actions = [
            {"remove": {"index": index, "alias": alias}}
            for index in current if index != target
        ]
        if target not in current:
            actions.append({"add": {"index": target, "alias": alias}})
        if actions:
            await self.client.indices.update_aliases(actions=actions)
        try:
            status = await self.active_manifest(workspace_id)
            if (
                not status.get("ready")
                or status.get("physical_index") != target
                or status.get("index_version") != candidate.get("index_version")
            ):
                raise RuntimeError("字段值索引回退后校验失败")
        except Exception:
            # 回退校验不通过时尽力恢复原 alias 目标；恢复失败仍保留原异常，
            # 并通过安全日志提示人工检查，不输出任何字段值。
            if target not in current:
                restore_actions = [
                    {"remove": {"index": target, "alias": alias}},
                    *[
                        {"add": {"index": index, "alias": alias}}
                        for index in current
                    ],
                ]
                try:
                    await self.client.indices.update_aliases(
                        actions=restore_actions
                    )
                except Exception as restore_exc:
                    logger.error(
                        "字段值索引回退补偿失败 workspace={} error_type={}",
                        workspace_id,
                        type(restore_exc).__name__,
                    )
            raise
        logger.info(
            "字段值索引回退完成 workspace={} version={} documents={}",
            workspace_id,
            status.get("index_version"),
            status.get("document_count"),
        )
        return status

    @staticmethod
    def _empty_evidence(
        *,
        workspace_id: str,
        keyword: str,
        query_mode: str,
        index_version: str | None,
        allowed_table_ids: Sequence[str] | None,
        allowed_column_ids: Sequence[str] | None,
        reason: str,
    ) -> dict:
        return {
            "schema": VALUE_SEARCH_AUDIT_SCHEMA,
            "responsibility": VALUE_RETRIEVAL_RESPONSIBILITY,
            "engine": "elasticsearch",
            "workspace_id": workspace_id,
            "query_mode": query_mode,
            "query_fingerprint": value_query_fingerprint(keyword),
            "index_version": index_version,
            "allowed_table_count": (
                None if allowed_table_ids is None else len(allowed_table_ids)
            ),
            "allowed_column_count": (
                None if allowed_column_ids is None else len(allowed_column_ids)
            ),
            "hit_count": 0,
            "returned_count": 0,
            "latency_ms": 0.0,
            "raw_value_logged": False,
            "filter_stage": "elasticsearch_query",
            "status": reason,
        }

    async def search_with_evidence(
        self,
        keyword: str,
        *,
        workspace_id: str,
        query_mode: str = "full_text",
        index_version: str | None = None,
        active_manifest: dict | None = None,
        allowed_table_ids: Sequence[str] | None = None,
        allowed_column_ids: Sequence[str] | None = None,
        table_id: str | None = None,
        column_id: str | None = None,
        page: int = 1,
        page_size: int | None = None,
    ) -> tuple[list[ValueInfoES], int, dict]:
        """Search values with tenant, authorization and version filters in ES."""
        mode = str(query_mode or "full_text").strip().lower()
        if mode not in VALUE_QUERY_MODES:
            raise ValueError(f"不支持的字段值查询模式：{query_mode}")
        normalized = normalize_value(keyword)
        size = (
            app_config.query.value_recall_limit
            if page_size is None else max(1, min(int(page_size), 100))
        )
        if not normalized:
            evidence = self._empty_evidence(
                workspace_id=workspace_id,
                keyword=keyword,
                query_mode=mode,
                index_version=index_version,
                allowed_table_ids=allowed_table_ids,
                allowed_column_ids=allowed_column_ids,
                reason="empty_query",
            )
            return [], 0, evidence
        if allowed_table_ids is not None and not allowed_table_ids:
            evidence = self._empty_evidence(
                workspace_id=workspace_id,
                keyword=keyword,
                query_mode=mode,
                index_version=index_version,
                allowed_table_ids=allowed_table_ids,
                allowed_column_ids=allowed_column_ids,
                reason="empty_authorized_scope",
            )
            return [], 0, evidence
        if allowed_column_ids is not None and not allowed_column_ids:
            evidence = self._empty_evidence(
                workspace_id=workspace_id,
                keyword=keyword,
                query_mode=mode,
                index_version=index_version,
                allowed_table_ids=allowed_table_ids,
                allowed_column_ids=allowed_column_ids,
                reason="empty_authorized_scope",
            )
            return [], 0, evidence

        manifest = active_manifest or await self.active_manifest(workspace_id)
        active_version = manifest.get("index_version")
        expected_version = index_version or active_version
        if not manifest.get("ready") or not expected_version:
            evidence = self._empty_evidence(
                workspace_id=workspace_id,
                keyword=keyword,
                query_mode=mode,
                index_version=expected_version,
                allowed_table_ids=allowed_table_ids,
                allowed_column_ids=allowed_column_ids,
                reason="index_not_ready",
            )
            return [], 0, evidence
        if index_version and active_version != index_version:
            evidence = self._empty_evidence(
                workspace_id=workspace_id,
                keyword=keyword,
                query_mode=mode,
                index_version=index_version,
                allowed_table_ids=allowed_table_ids,
                allowed_column_ids=allowed_column_ids,
                reason="version_mismatch",
            )
            evidence["active_index_version"] = active_version
            return [], 0, evidence

        compiled_query = build_value_search_query(
            keyword,
            query_mode=mode,
            workspace_id=workspace_id,
            index_version=expected_version,
            allowed_table_ids=allowed_table_ids,
            allowed_column_ids=allowed_column_ids,
            table_id=table_id,
            column_id=column_id,
        )

        started = time.perf_counter()
        try:
            response = _body(await self.client.search(
                index=manifest["alias"],
                query=compiled_query,
                from_=(max(1, int(page)) - 1) * size,
                size=size,
            ))
        except Exception as exc:
            # Elasticsearch 的原始异常可能回显含 literal 的 request body。
            # 只保留类型和无敏感上下文，供 optional_dependency 安全记录。
            logger.warning(
                "字段值检索失败 workspace={} mode={} version={} error_type={}",
                workspace_id,
                mode,
                expected_version,
                type(exc).__name__,
            )
            raise RuntimeError(
                f"Elasticsearch 字段值检索失败（{type(exc).__name__}）"
            ) from None
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        hits = (response or {}).get("hits") or {}
        total = _total_value(hits.get("total"))
        items: list[ValueInfoES] = []
        document_ids: list[str] = []
        for rank, hit in enumerate(hits.get("hits") or [], start=1):
            source = dict(hit.get("_source") or {})
            document_id = str(hit.get("_id") or source.get("id") or "")
            document_ids.append(document_id)
            source.update({
                "id": str(source.get("id") or document_id),
                "document_id": document_id,
                "point_id": document_id,
                "rank": rank,
                "score": float(hit.get("_score") or 0),
                "retrieval_source": f"es_{mode}",
                "retrieval_sources": [f"es_{mode}"],
                "query_mode": mode,
                "source_index": str(hit.get("_index") or ""),
            })
            items.append(ValueInfoES(**source))

        evidence = {
            "schema": VALUE_SEARCH_AUDIT_SCHEMA,
            "responsibility": VALUE_RETRIEVAL_RESPONSIBILITY,
            "engine": "elasticsearch",
            "workspace_id": workspace_id,
            "query_mode": mode,
            "query_fingerprint": value_query_fingerprint(keyword),
            "index_version": expected_version,
            "allowed_table_count": (
                None if allowed_table_ids is None else len(allowed_table_ids)
            ),
            "allowed_column_count": (
                None if allowed_column_ids is None else len(allowed_column_ids)
            ),
            "explicit_table_filter": bool(table_id),
            "explicit_column_filter": bool(column_id),
            "hit_count": total,
            "returned_count": len(items),
            "latency_ms": elapsed_ms,
            "raw_value_logged": False,
            "filter_stage": "elasticsearch_query",
            "status": "ok",
            "document_ids": document_ids[:20],
        }
        logger.info(
            "字段值检索 evidence={}",
            {
                key: value for key, value in evidence.items()
                if key != "document_ids"
            },
        )
        return items, total, evidence
