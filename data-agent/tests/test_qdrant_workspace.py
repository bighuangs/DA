import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.domain.retrieval.hybrid_embedding import HybridEmbedding
from app.infrastructure.search.qdrant.repositories.metric_qdrant_repository import MetricQdrantRepository
from app.infrastructure.search.qdrant.repositories.training_qdrant_repository import TrainingQdrantRepository


class QdrantWorkspaceIsolationTest(unittest.IsolatedAsyncioTestCase):
    async def test_metric_search_always_filters_workspace(self):
        client = AsyncMock()
        client.collection_exists.return_value = False
        client.query_points.return_value = SimpleNamespace(points=[])
        embedding = HybridEmbedding(
            dense=(0.1, 0.2), sparse_indices=(1,), sparse_values=(0.5,)
        )
        await MetricQdrantRepository(client).search(embedding, "ws-a")
        self.assertEqual(client.query_points.await_count, 2)
        for call in client.query_points.await_args_list:
            query_filter = call.kwargs["query_filter"]
            conditions = {item.key: item.match.value for item in query_filter.must}
            self.assertEqual(conditions["workspace_id"], "ws-a")
            self.assertEqual(conditions["embedding_mode"], "dense_sparse")

    async def test_training_replacement_keeps_new_ids_and_scopes_cleanup(self):
        client = AsyncMock()
        client.collection_exists.return_value = False
        repo = TrainingQdrantRepository(client)
        await repo.replace_workspace_vectors("ws-b", [], [], [])
        selector = client.delete.await_args.kwargs["points_selector"]
        self.assertEqual(selector.filter.must[0].match.value, "ws-b")


if __name__ == "__main__":
    unittest.main()
