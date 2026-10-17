"""Dedicated BGE-M3 dense+sparse projection for long-term conversation memory."""
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


class ConversationMemoryQdrantRepository:
    """Qdrant payload has scope/versions only; authoritative text stays in MySQL."""

    _POINT_NAMESPACE = uuid.UUID("aa4e4bb5-ad2d-5a0c-bbc4-10324fdb62b5")

    def __init__(
        self,
        client: AsyncQdrantClient,
        *,
        projection_version: str | None = None,
    ):
        if client is None:
            raise RuntimeError("Qdrant client 尚未初始化")
        self.client = client
        self.collection_name = app_config.memory.qdrant_collection_name
        self.projection_version = str(
            projection_version or app_config.memory.qdrant_projection_version
        )

    def _point_id(self, memory_id: str) -> str:
        return str(uuid.uuid5(
            self._POINT_NAMESPACE,
            f"{memory_id}|{self.projection_version}",
        ))

    async def ensure_collection(self) -> dict:
        await create_hybrid_collection(self.client, self.collection_name)
        for field in (
            "owner_id", "workspace_id", "status", "verification_status",
            "projection_version", "index_version", "memory_type",
        ):
            try:
                await self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                # Qdrant reports an error when the index already exists on some
                # versions; the collection schema check remains authoritative.
                pass
        return await self.status()

    async def upsert(self, item, embedding: HybridEmbedding) -> None:
        await self.ensure_collection()
        payload = {
            "id": item.memory_id,
            "entity_id": item.memory_id,
            "owner_id": item.owner_id,
            "workspace_id": item.workspace_id,
            "chat_id": item.chat_id,
            "memory_type": item.memory_type,
            "status": item.status,
            "verification_status": item.verification_status,
            "projection_version": self.projection_version,
            # hybrid_payload 的统一检索合同要求每个 Point 带 index_version。
            # 会话记忆没有独立的内容索引版本，因此以投影版本作为权威版本。
            "index_version": self.projection_version,
            "point_strategy": "single_canonical",
        }
        await upsert_hybrid_points(
            self.client,
            self.collection_name,
            [embedding],
            [payload],
            [self._point_id(item.memory_id)],
            source_texts=[item.content_masked],
            point_roles=["canonical"],
            batch_size=1,
        )
        # Remove the pre-versioning point ID after the versioned point exists.
        # This is idempotent and prevents one memory from appearing twice
        # during an in-place rollout.
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=[item.memory_id]),
        )

    async def delete(self, memory_id: str) -> None:
        await self.ensure_collection()
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                # The plain ID removes projections created before point IDs
                # became version-derived; the UUID removes this exact version.
                points=[self._point_id(str(memory_id)), str(memory_id)]
            ),
        )

    async def search(
        self,
        embedding: HybridEmbedding,
        *,
        query: str,
        owner_id: str,
        workspace_id: str,
        limit: int,
    ) -> tuple[list[str], dict, list[dict]]:
        await self.ensure_collection()
        started = time.perf_counter()
        result = await search_hybrid_entities(
            self.client,
            self.collection_name,
            embedding,
            route="memory",
            query_filter=models.Filter(must=[
                models.FieldCondition(
                    key="owner_id", match=models.MatchValue(value=owner_id)
                ),
                models.FieldCondition(
                    key="workspace_id", match=models.MatchValue(value=workspace_id)
                ),
                models.FieldCondition(
                    key="status", match=models.MatchValue(value="active")
                ),
                models.FieldCondition(
                    key="verification_status",
                    match=models.MatchAny(
                        any=["verified", "confirmed", "executed"]
                    ),
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
        evidence = {
            "engine": "qdrant",
            "route": "memory_qdrant",
            "query_fingerprint": memory_query_fingerprint(query),
            "candidate_count": len(identifiers),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "projection_version": self.projection_version,
            "workspace_scope_applied": True,
            "owner_scope_applied": True,
            "raw_text_logged": False,
            "status": "ok",
            "hybrid": result.evidence,
        }
        return identifiers, evidence, result.degradations

    async def status(self) -> dict:
        return {
            **await collection_status(self.client, self.collection_name),
            "engine": "qdrant",
            "projection_version": self.projection_version,
        }
