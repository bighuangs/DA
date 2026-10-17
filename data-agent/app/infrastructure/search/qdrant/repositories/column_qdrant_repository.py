from qdrant_client import AsyncQdrantClient, models

from app.config.app_config import app_config
from app.domain.retrieval.hybrid_embedding import HybridEmbedding
from app.infrastructure.search.qdrant.records.column_info_qdrant import ColumnInfoQdrant
from app.infrastructure.search.qdrant.repositories.hybrid_qdrant import (
    HybridSearchResult,
    collection_status,
    create_hybrid_collection,
    search_hybrid_entities,
    upsert_hybrid_points,
)


class ColumnQdrantRepository:
    """BGE-M3 dense+sparse index for schema columns."""

    collection_name = "data-agent_column_collection"

    def __init__(self, client: AsyncQdrantClient):
        self.client = client

    async def status(self, expected_index_version: str | None = None) -> dict:
        status = await collection_status(self.client, self.collection_name)
        if not expected_index_version or not status["exists"] or not status["compatible"]:
            return status
        result = await self.client.count(
            collection_name=self.collection_name,
            count_filter=models.Filter(must=[
                models.FieldCondition(
                    key="index_version",
                    match=models.MatchValue(value=expected_index_version),
                ),
                models.FieldCondition(
                    key="embedding_mode",
                    match=models.MatchValue(value="dense_sparse"),
                ),
            ]),
            exact=True,
        )
        matching = int(getattr(result, "count", 0) or 0)
        return {
            **status,
            "expected_index_version": expected_index_version,
            "matching_points_count": matching,
            "content_compatible": matching > 0,
        }

    async def _ensure_collection(self):
        await create_hybrid_collection(self.client, self.collection_name)

    async def recreate_collection(self):
        await create_hybrid_collection(self.client, self.collection_name, recreate=True)

    async def upsert_column_vectors(
        self,
        vectors: list[HybridEmbedding],
        payloads: list[ColumnInfoQdrant],
        ids: list,
        *,
        source_texts: list[str] | None = None,
        point_roles: list[str] | None = None,
    ):
        await upsert_hybrid_points(
            self.client, self.collection_name, vectors, payloads, ids,
            source_texts=source_texts, point_roles=point_roles,
        )

    async def search(
        self,
        vector: HybridEmbedding,
        index_version: str,
        score_threshold: float | None = None,
        sparse_score_threshold: float | None = None,
        limit: int | None = None,
        mode: str = "hybrid",
    ) -> HybridSearchResult:
        dense_threshold = (
            app_config.query.column_recall_score_threshold
            if score_threshold is None else score_threshold
        )
        sparse_threshold = (
            app_config.query.column_sparse_score_threshold
            if sparse_score_threshold is None else sparse_score_threshold
        )
        return await search_hybrid_entities(
            self.client,
            self.collection_name,
            vector,
            route="column",
            query_filter=models.Filter(must=[
                models.FieldCondition(
                    key="index_version", match=models.MatchValue(value=index_version)
                ),
                models.FieldCondition(
                    key="embedding_mode", match=models.MatchValue(value="dense_sparse")
                ),
            ]),
            dense_score_threshold=dense_threshold,
            sparse_score_threshold=sparse_threshold,
            limit=limit or app_config.query.column_recall_limit,
            mode=mode,
        )
