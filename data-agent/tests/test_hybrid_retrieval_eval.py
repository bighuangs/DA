import unittest

from evals.legacy.hybrid_retrieval import (
    ISOLATED_COLLECTION_PREFIX,
    _cleanup_isolated_collections,
    _isolated_collection_names,
    _metric_column_candidates,
    _metric,
    _profile_summary,
    load_contract,
    validate_contract,
)


class _FakeQdrant:
    def __init__(self, collections):
        self.collections = set(collections)

    async def collection_exists(self, *, collection_name):
        return collection_name in self.collections

    async def delete_collection(self, *, collection_name):
        self.collections.discard(collection_name)


class HybridRetrievalEvaluationTests(unittest.TestCase):
    def test_contract_contains_base_and_long_tail_cases(self):
        contract = load_contract()
        validation = validate_contract(contract)
        self.assertTrue(validation["valid"], validation["errors"])
        self.assertGreaterEqual(validation["base_cases"], 16)
        self.assertGreaterEqual(validation["extra_cases"], 10)
        self.assertEqual(set(contract["profiles"]), {"R-D1", "R-S1", "R-H1", "R-H3"})

    def test_entity_metrics_use_route_business_identity(self):
        rows = [
            {"_route": "metric", "id": "ws:GMV", "name": "GMV"},
            {"_route": "metric", "id": "ws:AOV", "name": "AOV"},
        ]
        metric = _metric(["GMV"], rows)
        self.assertEqual(metric["recall_at_k"], 1.0)
        self.assertEqual(metric["precision_at_k"], .5)
        self.assertEqual(metric["mrr"], 1.0)

    def test_profile_summary_reports_degradation_and_base_recall(self):
        cases = [{
            "id": "RAG-001",
            "tags": ["identifier"],
            "tenant_leakage": 0,
            "degradations": [{"reason_code": "sparse_retrieval_failed"}],
            "latency_ms": 12,
            "routes": {
                "column": {
                    "expected": ["fact_order.order_amount"],
                    "returned": ["fact_order.order_amount"],
                    "recall_at_k": 1.0,
                    "precision_at_k": 1.0,
                    "mrr": 1.0,
                    "ndcg": 1.0,
                    "missing": [],
                    "evidence": {
                        "dense": {"raw_points": 2},
                        "sparse": {"raw_points": 1},
                        "fused_entities": 1,
                    },
                }
            },
        }]
        summary = _profile_summary(cases, {"RAG-001"})
        self.assertEqual(summary["base_column_recall_at_k"], 1.0)
        self.assertEqual(summary["base_column_precision_at_k"], 1.0)
        self.assertEqual(summary["degradation_count"], 1)
        self.assertEqual(summary["raw_physical_points"], 3)

    def test_exact_metric_closes_onto_declared_columns(self):
        corpora = {
            "column": [
                {"id": "fact_order.order_amount", "name": "order_amount"},
                {"id": "fact_order.order_quantity", "name": "order_quantity"},
            ],
            "metric": [{
                "id": "AOV",
                "name": "AOV",
                "alias": ["平均订单金额"],
                "relevant_columns": [
                    "fact_order.order_amount",
                    "fact_order.order_quantity",
                ],
            }],
            "training": [],
        }
        rows, metrics = _metric_column_candidates(corpora, "2025年平均客单价AOV")
        self.assertEqual(
            {row["id"] for row in rows},
            {"fact_order.order_amount", "fact_order.order_quantity"},
        )
        self.assertEqual(metrics, ["AOV"])
        self.assertTrue(all(
            row["retrieval_sources"] == ["metric_relevant_column"] for row in rows
        ))

    def test_isolated_collection_names_are_scoped_and_deterministic(self):
        names = _isolated_collection_names("workspace-1", run_token="run-token")
        self.assertEqual(set(names), {"column", "metric", "training"})
        self.assertEqual(len(set(names.values())), 3)
        self.assertTrue(all(
            name.startswith(ISOLATED_COLLECTION_PREFIX)
            for name in names.values()
        ))


class HybridRetrievalCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_cleanup_deletes_only_isolated_collections(self):
        isolated = _isolated_collection_names(
            "workspace-1", run_token="cleanup"
        )
        client = _FakeQdrant([
            *isolated.values(),
            "data-agent_column_collection",
        ])

        result = await _cleanup_isolated_collections(
            client, list(isolated.values())
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["deleted"], 3)
        self.assertEqual(result["residual"], 0)
        self.assertEqual(client.collections, {"data-agent_column_collection"})

    async def test_cleanup_rejects_active_collection_name(self):
        client = _FakeQdrant(["data-agent_column_collection"])

        result = await _cleanup_isolated_collections(
            client, ["data-agent_column_collection"]
        )

        self.assertFalse(result["passed"])
        self.assertIn("data-agent_column_collection", client.collections)


if __name__ == "__main__":
    unittest.main()
