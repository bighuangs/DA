"""One-point-per-summary BGE-M3 Dense + Sparse projection."""
from __future__ import annotations

import time
import uuid

from qdrant_client import AsyncQdrantClient, models

from app.config.app_config import app_config
from app.domain.conversation.fingerprints import memory_query_fingerprint
from app.domain.retrieval.hybrid_embedding import HybridEmbedding
from app.infrastructure.search.qdrant.repositories.hybrid_qdrant import (
    collection_status,
    create_hybrid_collection,
    search_hybrid_entities,
    upsert_hybrid_points,
)


class ConversationSummaryQdrantRepository:
    _POINT_NAMESPACE = uuid.UUID("4154fdd8-ef9a-5cbd-a1d7-1ed514d75102")

    def __init__(
        self,
        client: AsyncQdrantClient,
        *,
        projection_version: str | None = None,
    ):
        if client is None:
            raise RuntimeError("Qdrant client 尚未初始化")
        self.client = client
        self.collection_name = app_config.memory.summary_qdrant_collection_name
        self.projection_version = str(
            projection_version
            or app_config.memory.summary_qdrant_projection_version
        )

    def _point_id(self, summary_id: str) -> str:
        return str(uuid.uuid5(
            self._POINT_NAMESPACE,
            f"{summary_id}|{self.projection_version}",
        ))

    async def ensure_collection(self) -> dict:
        await create_hybrid_collection(self.client, self.collection_name)
        for field in (
            "owner_id", "workspace_id", "chat_id", "status",
            "projection_version", "index_version", "level",
        ):
            schema = (
                models.PayloadSchemaType.INTEGER
                if field == "level"
                else models.PayloadSchemaType.KEYWORD
            )
            try:
                await self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception:
                pass
        return await self.status()

    async def upsert(self, segment, embedding: HybridEmbedding) -> None:
        await self.ensure_collection()
        payload = {
            "id": segment.summary_id,
            "entity_id": segment.summary_id,
            "owner_id": segment.owner_id,
            "workspace_id": segment.workspace_id,
            "chat_id": segment.chat_id,
            "status": segment.status,
            "level": int(segment.level or 0),
            "projection_version": self.projection_version,
            "index_version": self.projection_version,
            "source_content_hash": segment.source_content_hash,
            "point_strategy": "single_summary_segment",
        }
        await upsert_hybrid_points(
            self.client,
            self.collection_name,
            [embedding],
            [payload],
            [self._point_id(segment.summary_id)],
            source_texts=[segment.content_masked],
            point_roles=["canonical"],
            batch_size=1,
        )

    async def delete(self, summary_id: str) -> None:
        await self.ensure_collection()
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=[self._point_id(str(summary_id))]
            ),
        )

    async def search(
        self,
        embedding: HybridEmbedding,
        *,
        query: str,
        owner_id: str,
        workspace_id: str,
        chat_id: str,
        limit: int,
    ) -> tuple[list[str], dict, list[dict]]:
        await self.ensure_collection()
        started = time.perf_counter()
        result = await search_hybrid_entities(
            self.client,
            self.collection_name,
            embedding,
            route="summary_memory",
            query_filter=models.Filter(must=[
                models.FieldCondition(
                    key="owner_id", match=models.MatchValue(value=owner_id)
                ),
                models.FieldCondition(
                    key="workspace_id", match=models.MatchValue(value=workspace_id)
                ),
                models.FieldCondition(
                    key="chat_id", match=models.MatchValue(value=chat_id)
                ),
                models.FieldCondition(
                    key="status", match=models.MatchValue(value="active")
                ),
                models.FieldCondition(
                    key="projection_version",
                    match=models.MatchValue(value=self.projection_version),
                ),
                models.FieldCondition(
                    key="index_version",
                    match=models.MatchValue(value=self.projection_version),
                ),
            ]),
            dense_score_threshold=None,
            sparse_score_threshold=None,
            limit=max(1, min(int(limit), 50)),
            mode="hybrid",
        )
        identifiers = [
            str(item.get("entity_id") or "")
            for item in result.items
            if item.get("entity_id")
        ]
        return identifiers, {
            "engine": "qdrant",
            "route": "summary_qdrant",
            "query_fingerprint": memory_query_fingerprint(query),
            "candidate_count": len(identifiers),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "projection_version": self.projection_version,
            "owner_scope_applied": True,
            "workspace_scope_applied": True,
            "chat_scope_applied": True,
            "raw_text_logged": False,
            "status": "ok",
            "hybrid": result.evidence,
        }, result.degradations

    async def status(self) -> dict:
        return {
            **await collection_status(self.client, self.collection_name),
            "engine": "qdrant",
            "projection_version": self.projection_version,
        }
