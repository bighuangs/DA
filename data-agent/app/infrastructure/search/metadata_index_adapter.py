"""Qdrant Point 投影、Embedding 与索引版本的出站组合适配器。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from app.config.app_config import app_config
from app.infrastructure.clients.hybrid_embedding_client import HybridEmbeddingClient
from app.infrastructure.persistence.mysql.repositories.meta_mysql_repository import (
    MetaMysqlRepository,
)
from app.infrastructure.search.index_version import (
    column_index_version,
    retrieval_index_version,
)
from app.infrastructure.search.projection import (
    SUPPORTED_POINT_STRATEGIES,
    ProjectionRecord,
    alias_manifest_fingerprint,
    build_column_projection_records,
    build_metric_projection_records,
    build_training_projection_records,
    load_alias_manifest,
    projection_manifest,
    validate_point_strategy,
)
from app.infrastructure.search.qdrant.repositories.column_qdrant_repository import (
    ColumnQdrantRepository,
)
from app.infrastructure.search.qdrant.repositories.metric_qdrant_repository import (
    MetricQdrantRepository,
)
from app.infrastructure.search.qdrant.repositories.training_qdrant_repository import (
    TrainingQdrantRepository,
)


class MetadataIndexAdapter:
    """实现元数据投影端口，并集中持有所有具体索引依赖。"""

    def __init__(
        self,
        *,
        meta_repo: MetaMysqlRepository,
        column_repo: ColumnQdrantRepository,
        metric_repo: MetricQdrantRepository,
        training_repo: TrainingQdrantRepository,
        embedding_client: HybridEmbeddingClient,
    ) -> None:
        self._meta_repo = meta_repo
        self._column_repo = column_repo
        self._metric_repo = metric_repo
        self._training_repo = training_repo
        self._embedding_client = embedding_client

    @staticmethod
    def _metric_content(items: Iterable[Any], workspace_id: str) -> list[dict]:
        return sorted(({
            "workspace_id": workspace_id,
            "id": str(item.id),
            "name": item.name,
            "description": item.description or "",
            "relevant_columns": item.relevant_columns or [],
            "alias": item.alias or [],
            "formula_rule": item.formula_rule,
        } for item in items), key=lambda row: row["id"])

    @staticmethod
    def _training_content(items: Iterable[Any], workspace_id: str) -> list[dict]:
        return sorted(({
            "workspace_id": workspace_id,
            "id": str(item.id),
            "question": item.question,
            "sql": item.sql,
        } for item in items), key=lambda row: row["id"])

    @staticmethod
    def _records_payload(
        records: Sequence[ProjectionRecord],
    ) -> tuple[list, list, list, list]:
        return (
            [record.payload for record in records],
            [record.point_id for record in records],
            [record.text for record in records],
            [record.point_role for record in records],
        )

    @staticmethod
    def _with_expected_points(status: dict, expected_points: int) -> dict:
        matching = int(status.get("matching_points_count", 0) or 0)
        return {
            **status,
            "expected_points_count": expected_points,
            "content_compatible": bool(
                status.get("exists")
                and status.get("compatible")
                and matching == expected_points
            ),
        }

    async def _embed(
        self,
        records: Sequence[ProjectionRecord],
        *,
        batch_size: int | None = None,
    ) -> list:
        texts = [record.text for record in records]
        if not texts:
            return []
        if batch_size is None:
            return list(await self._embedding_client.aembed_documents(texts))
        vectors = []
        for start in range(0, len(texts), batch_size):
            vectors.extend(await self._embedding_client.aembed_documents(
                texts[start:start + batch_size]
            ))
        return vectors

    def _projection_context(self) -> tuple[str, dict, str]:
        strategy = validate_point_strategy()
        alias_manifest = load_alias_manifest()
        return (
            strategy,
            alias_manifest,
            alias_manifest_fingerprint(alias_manifest),
        )

    async def status(
        self,
        *,
        workspace_id: str,
        table_rows: Sequence[tuple[Any, Sequence[Any]]],
        metric_items: Sequence[Any],
        training_items: Sequence[Any],
    ) -> dict:
        runtime_error = None
        try:
            runtime = self._embedding_client.validate_runtime()
        except Exception as exc:  # noqa: BLE001 - 状态接口必须返回降级证据
            runtime = self._embedding_client.status()
            runtime_error = str(exc)

        strategy, alias_manifest, manifest_hash = self._projection_context()
        rows = [(table, list(columns)) for table, columns in table_rows]
        columns = [column for _, values in rows for column in values]
        column_version = column_index_version(
            columns,
            point_strategy=strategy,
            alias_manifest_hash=manifest_hash,
        )
        column_records = build_column_projection_records(
            rows,
            index_version=column_version,
            strategy=strategy,
            alias_manifest=alias_manifest,
        )

        metrics = [item for item in metric_items if item.enabled]
        metric_version = retrieval_index_version(
            self._metric_content(metrics, workspace_id),
            point_strategy=strategy,
            alias_manifest_hash=manifest_hash,
        )
        metric_records = build_metric_projection_records(
            metrics,
            workspace_id=workspace_id,
            index_version=metric_version,
            strategy=strategy,
            alias_manifest=alias_manifest,
        )

        training = [item for item in training_items if item.enabled]
        training_version = retrieval_index_version(
            self._training_content(training, workspace_id),
            point_strategy=strategy,
            alias_manifest_hash=manifest_hash,
        )
        training_records = build_training_projection_records(
            training,
            workspace_id=workspace_id,
            index_version=training_version,
            strategy=strategy,
            alias_manifest=alias_manifest,
        )
        collections = {
            "column": self._with_expected_points(
                await self._column_repo.status(column_version),
                len(column_records),
            ),
            "metric": self._with_expected_points(
                await self._metric_repo.status(metric_version, workspace_id),
                len(metric_records),
            ),
            "training": self._with_expected_points(
                await self._training_repo.status(training_version, workspace_id),
                len(training_records),
            ),
        }
        return {
            "active_mode": "bge_m3_dense_sparse",
            "replacement_strategy": "overwrite_active_point_projection",
            "v1_fallback": False,
            "workspace_id": workspace_id,
            "runtime": {**runtime, "error": runtime_error},
            "collections": collections,
            "point_projection": {
                "active_strategy": strategy,
                "supported_strategies": list(SUPPORTED_POINT_STRATEGIES),
                "identity_version": app_config.point_projection.identity_version,
                "canonical_template_version": (
                    app_config.point_projection.canonical_template_version
                ),
                "alias_policy_version": (
                    app_config.point_projection.alias_policy_version
                ),
                "alias_manifest_hash": manifest_hash,
                "approved_alias_count": len(
                    alias_manifest.get("approved_aliases") or []
                ),
                "routes": {
                    "column": projection_manifest(column_records),
                    "metric": projection_manifest(metric_records),
                    "training": projection_manifest(training_records),
                },
            },
            "ready": (
                runtime_error is None
                and all(item.get("compatible") for item in collections.values())
                and all(
                    item.get("content_compatible")
                    for item in collections.values()
                )
            ),
        }

    async def rebuild_columns(
        self,
        table_rows: Sequence[tuple[Any, Sequence[Any]]],
        *,
        batch_size: int | None = None,
    ) -> int:
        strategy, alias_manifest, manifest_hash = self._projection_context()
        rows = [(table, list(columns)) for table, columns in table_rows]
        columns = [column for _, items in rows for column in items]
        version = column_index_version(
            columns,
            point_strategy=strategy,
            alias_manifest_hash=manifest_hash,
        )
        records = build_column_projection_records(
            rows,
            index_version=version,
            strategy=strategy,
            alias_manifest=alias_manifest,
        )
        vectors = await self._embed(records, batch_size=batch_size)
        # Embedding 全部成功后才替换 collection，保持失败时的旧索引可用。
        await self._column_repo.recreate_collection()
        if records:
            payloads, ids, texts, roles = self._records_payload(records)
            await self._column_repo.upsert_column_vectors(
                vectors,
                payloads,
                ids,
                source_texts=texts,
                point_roles=roles,
            )
        return len(columns)

    async def rebuild_metrics(
        self,
        workspace_id: str,
        items: Sequence[Any],
    ) -> int:
        strategy, alias_manifest, manifest_hash = self._projection_context()
        enabled = [item for item in items if item.enabled]
        version = retrieval_index_version(
            self._metric_content(enabled, workspace_id),
            point_strategy=strategy,
            alias_manifest_hash=manifest_hash,
        )
        records = build_metric_projection_records(
            enabled,
            workspace_id=workspace_id,
            index_version=version,
            strategy=strategy,
            alias_manifest=alias_manifest,
        )
        status = await self._metric_repo.status()
        if status["exists"] and not status["compatible"]:
            return await self.overwrite_all_metrics()
        vectors = await self._embed(records)
        payloads, ids, texts, roles = self._records_payload(records)
        await self._metric_repo.replace_workspace_vectors(
            workspace_id,
            vectors,
            payloads,
            ids,
            source_texts=texts,
            point_roles=roles,
        )
        return len(enabled)

    async def rebuild_training(
        self,
        workspace_id: str,
        items: Sequence[Any],
    ) -> int:
        strategy, alias_manifest, manifest_hash = self._projection_context()
        enabled = [item for item in items if item.enabled]
        version = retrieval_index_version(
            self._training_content(enabled, workspace_id),
            point_strategy=strategy,
            alias_manifest_hash=manifest_hash,
        )
        records = build_training_projection_records(
            enabled,
            workspace_id=workspace_id,
            index_version=version,
            strategy=strategy,
            alias_manifest=alias_manifest,
        )
        status = await self._training_repo.status()
        if status["exists"] and not status["compatible"]:
            return await self.overwrite_all_training()
        vectors = await self._embed(records)
        payloads, ids, texts, roles = self._records_payload(records)
        await self._training_repo.replace_workspace_vectors(
            workspace_id,
            vectors,
            payloads,
            ids,
            source_texts=texts,
            point_roles=roles,
        )
        return len(enabled)

    async def overwrite_all_metrics(self) -> int:
        strategy, alias_manifest, manifest_hash = self._projection_context()
        source_repo = MetaMysqlRepository(self._meta_repo.session)
        grouped: dict[str, list[Any]] = defaultdict(list)
        for item in await source_repo.get_all_metric_infos():
            workspace_id = str(item.workspace_id or self._meta_repo.workspace_id or "")
            if item.enabled and workspace_id:
                grouped[workspace_id].append(item)
        records: list[ProjectionRecord] = []
        for workspace_id, items in sorted(grouped.items()):
            version = retrieval_index_version(
                self._metric_content(items, workspace_id),
                point_strategy=strategy,
                alias_manifest_hash=manifest_hash,
            )
            records.extend(build_metric_projection_records(
                items,
                workspace_id=workspace_id,
                index_version=version,
                strategy=strategy,
                alias_manifest=alias_manifest,
            ))
        vectors = await self._embed(records)
        await self._metric_repo.recreate_collection()
        if records:
            payloads, ids, texts, roles = self._records_payload(records)
            await self._metric_repo.upsert_metric_vectors(
                vectors,
                payloads,
                ids,
                source_texts=texts,
                point_roles=roles,
            )
        return sum(len(items) for items in grouped.values())

    async def overwrite_all_training(self) -> int:
        strategy, alias_manifest, manifest_hash = self._projection_context()
        source_repo = MetaMysqlRepository(self._meta_repo.session)
        grouped: dict[str, list[Any]] = defaultdict(list)
        for item in await source_repo.get_all_enabled_training_infos():
            workspace_id = str(item.workspace_id or self._meta_repo.workspace_id or "")
            if item.enabled and workspace_id:
                grouped[workspace_id].append(item)
        records: list[ProjectionRecord] = []
        for workspace_id, items in sorted(grouped.items()):
            version = retrieval_index_version(
                self._training_content(items, workspace_id),
                point_strategy=strategy,
                alias_manifest_hash=manifest_hash,
            )
            records.extend(build_training_projection_records(
                items,
                workspace_id=workspace_id,
                index_version=version,
                strategy=strategy,
                alias_manifest=alias_manifest,
            ))
        vectors = await self._embed(records)
        await self._training_repo.recreate_collection()
        if records:
            payloads, ids, texts, roles = self._records_payload(records)
            await self._training_repo.upsert_training_vectors(
                vectors,
                payloads,
                ids,
                source_texts=texts,
                point_roles=roles,
            )
        return sum(len(items) for items in grouped.values())

    def point_strategy(self) -> str:
        return validate_point_strategy()


__all__ = ["MetadataIndexAdapter"]
