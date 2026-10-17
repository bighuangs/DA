import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from qdrant_client import models

from app.config.app_config import app_config
from app.domain.retrieval.hybrid_embedding import HybridEmbedding
from app.infrastructure.search.qdrant.repositories.column_qdrant_repository import ColumnQdrantRepository
from app.infrastructure.search.qdrant.repositories.hybrid_qdrant import (
    HybridIndexSchemaError,
    create_hybrid_collection,
    search_hybrid_entities,
    upsert_hybrid_points,
)


def _embedding():
    return HybridEmbedding(
        dense=(0.6, 0.8),
        sparse_indices=(2, 7),
        sparse_values=(0.4, 0.9),
    )


def _point(point_id, entity_id, role, score):
    return SimpleNamespace(
        id=point_id,
        score=score,
        payload={
            "id": entity_id,
            "entity_id": entity_id,
            "name": entity_id,
            "point_role": role,
            "point_strategy": "single_canonical",
            "index_version": "sha256:test-index",
            "embedding_mode": "dense_sparse",
        },
    )


def _filter():
    return models.Filter(must=[
        models.FieldCondition(
            key="workspace_id", match=models.MatchValue(value="ws-a")
        )
    ])


class HybridQdrantContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_column_status_rejects_schema_ready_but_stale_content_version(self):
        client = AsyncMock()
        client.collection_exists.return_value = True
        client.get_collection.return_value = SimpleNamespace(
            points_count=4,
            status="green",
            config=SimpleNamespace(params=SimpleNamespace(
                vectors={
                    "dense": SimpleNamespace(size=app_config.embedding.dense_size)
                },
                sparse_vectors={"sparse": SimpleNamespace()},
            )),
        )
        client.count.return_value = SimpleNamespace(count=0)
        status = await ColumnQdrantRepository(client).status("expected-version")
        self.assertTrue(status["compatible"])
        self.assertFalse(status["content_compatible"])
        self.assertEqual(status["matching_points_count"], 0)

    async def test_collection_and_point_use_named_dense_sparse_vectors(self):
        client = AsyncMock()
        client.collection_exists.return_value = False
        await create_hybrid_collection(client, "hybrid-test")
        create = client.create_collection.await_args.kwargs
        self.assertIn(app_config.qdrant.dense_vector_name, create["vectors_config"])
        self.assertIn(
            app_config.qdrant.sparse_vector_name, create["sparse_vectors_config"]
        )

        client.reset_mock()
        client.collection_exists.return_value = False
        await upsert_hybrid_points(
            client,
            "hybrid-test",
            [_embedding()],
            [{
                "id": "entity-a",
                "name": "订单金额",
                "point_strategy": "single_canonical",
                "index_version": "sha256:test-index",
            }],
            ["point-a"],
            source_texts=["订单金额"],
            point_roles=["name"],
        )
        point = client.upsert.await_args.kwargs["points"][0]
        self.assertEqual(set(point.vector), {"dense", "sparse"})
        self.assertEqual(point.payload["entity_id"], "entity-a")
        self.assertEqual(point.payload["embedding_mode"], "dense_sparse")
        self.assertEqual(point.payload["point_role"], "name")
        self.assertEqual(point.payload["point_strategy"], "single_canonical")
        self.assertEqual(point.payload["index_version"], "sha256:test-index")

    async def test_unversioned_point_is_rejected(self):
        client = AsyncMock()
        client.collection_exists.return_value = False
        client.query_points.side_effect = [
            SimpleNamespace(points=[SimpleNamespace(
                id="old-point",
                score=.9,
                payload={
                    "id": "entity-a",
                    "entity_id": "entity-a",
                    "name": "entity-a",
                    "point_role": "name",
                    "embedding_mode": "dense_sparse",
                },
            )]),
            SimpleNamespace(points=[]),
        ]

        with self.assertRaises(HybridIndexSchemaError):
            await search_hybrid_entities(
                client,
                "hybrid-test",
                _embedding(),
                route="column",
                query_filter=_filter(),
                dense_score_threshold=0.0,
                sparse_score_threshold=0.0,
                limit=2,
            )

    async def test_entity_rrf_dedupes_alias_points_before_fusion(self):
        client = AsyncMock()
        client.collection_exists.return_value = False
        client.query_points.side_effect = [
            SimpleNamespace(points=[
                _point("a-name", "entity-a", "name", .92),
                _point("a-alias", "entity-a", "alias", .89),
                _point("b-name", "entity-b", "name", .70),
            ]),
            SimpleNamespace(points=[
                _point("b-sparse", "entity-b", "name", 2.2),
                _point("a-sparse", "entity-a", "alias", 1.8),
            ]),
        ]
        result = await search_hybrid_entities(
            client,
            "hybrid-test",
            _embedding(),
            route="metric",
            query_filter=_filter(),
            dense_score_threshold=0.0,
            sparse_score_threshold=0.0,
            limit=4,
        )
        self.assertEqual([item["entity_id"] for item in result.items], [
            "entity-a", "entity-b",
        ])
        self.assertEqual(result.items[0]["dense_rank"], 1)
        self.assertEqual(result.items[0]["sparse_rank"], 2)
        self.assertEqual(
            result.items[0]["matched_point_roles"], ["alias", "name"]
        )
        self.assertEqual(result.evidence["dense"]["raw_points"], 3)
        self.assertEqual(result.evidence["dense"]["unique_entities"], 2)
        self.assertEqual(result.evidence["dense"]["duplicate_occupancy"], .3333)
        self.assertEqual(result.evidence["intersection_entities"], 2)

    async def test_sparse_failure_explicitly_degrades_to_dense(self):
        client = AsyncMock()
        client.collection_exists.return_value = False
        client.query_points.side_effect = [
            SimpleNamespace(points=[_point("a", "entity-a", "name", .8)]),
            RuntimeError("sparse unavailable"),
        ]
        result = await search_hybrid_entities(
            client,
            "hybrid-test",
            _embedding(),
            route="column",
            query_filter=_filter(),
            dense_score_threshold=0.0,
            sparse_score_threshold=0.0,
            limit=2,
        )
        self.assertEqual(result.evidence["effective_mode"], "dense")
        self.assertEqual(result.items[0]["retrieval_sources"], ["dense"])
        self.assertEqual(result.degradations[0]["reason_code"], "sparse_retrieval_failed")
        self.assertEqual(result.degradations[0]["severity"], "warning")

    async def test_dense_failure_does_not_enable_sparse_only_by_default(self):
        client = AsyncMock()
        client.collection_exists.return_value = False
        client.query_points.side_effect = [
            RuntimeError("dense unavailable"),
            SimpleNamespace(points=[_point("a", "entity-a", "name", 2.1)]),
        ]
        result = await search_hybrid_entities(
            client,
            "hybrid-test",
            _embedding(),
            route="training",
            query_filter=_filter(),
            dense_score_threshold=0.0,
            sparse_score_threshold=0.0,
            limit=2,
        )
        self.assertEqual(result.items, [])
        self.assertEqual(result.degradations[0]["reason_code"], "dense_retrieval_failed")
        self.assertEqual(result.degradations[0]["severity"], "critical")

    async def test_both_legs_receive_the_same_workspace_filter(self):
        client = AsyncMock()
        client.collection_exists.return_value = False
        client.query_points.return_value = SimpleNamespace(points=[])
        await search_hybrid_entities(
            client,
            "hybrid-test",
            _embedding(),
            route="metric",
            query_filter=_filter(),
            dense_score_threshold=0.0,
            sparse_score_threshold=0.0,
            limit=2,
        )
        self.assertEqual(client.query_points.await_count, 2)
        for call in client.query_points.await_args_list:
            self.assertIs(call.kwargs["query_filter"], client.query_points.await_args_list[0].kwargs["query_filter"])


if __name__ == "__main__":
    unittest.main()
