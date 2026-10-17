import copy
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.agent_system.agents.nl2sql.nodes.recall_value import (
    _annotate_value_match_hints,
    _extract_json_array,
    _safe_value_match_hint,
)
from app.config.app_config import app_config
from app.agent_system.agents.nl2sql.services.query_signals import deterministic_value_route_variants
from app.agent_system.governance.retrieval_budget import match_features, rerank_and_cut
from app.infrastructure.search.elasticsearch.value_retrieval import normalize_value, value_document_id
from evals.legacy.rag_recall import (
    build_report,
    evaluate_live_run,
    load_contract,
    validate_contract,
)
from app.domain.retrieval.hybrid_embedding import HybridEmbedding
from app.infrastructure.search.qdrant.repositories.column_qdrant_repository import ColumnQdrantRepository
from app.application.services.schema_sync_service import infer_display_column


class RAGRecallEvaluationTests(unittest.TestCase):
    def test_business_semantics_are_strong_column_evidence(self):
        amount = match_features(
            "各地区的销售总额", ["各地区的销售总额"],
            {"id": "fact_order.order_amount", "name": "order_amount", "alias": ["销售额"]},
            "column",
        )
        year = match_features(
            "2025年的GMV是多少", ["2025年的GMV是多少"],
            {"id": "dim_date.year", "name": "year", "alias": ["年份"]},
            "column",
        )
        customer = match_features(
            "复购率是多少", ["复购率"],
            {"id": "fact_order.customer_id", "name": "customer_id", "alias": []},
            "column",
        )
        order = match_features(
            "复购率是多少", ["复购率"],
            {"id": "fact_order.order_id", "name": "order_id", "alias": []},
            "column",
        )
        self.assertTrue(amount["semantic_hit"])
        self.assertTrue(year["semantic_hit"])
        self.assertTrue(customer["semantic_hit"])
        self.assertTrue(order["semantic_hit"])
        dimension_customer = match_features(
            "复购率是多少", ["复购率"],
            {
                "id": "dim_customer.customer_id", "name": "customer_id",
                "table_id": "dim_customer", "role": "primary_key",
                "alias": [],
            },
            "column",
        )
        self.assertFalse(dimension_customer["semantic_hit"])
        self.assertTrue(amount["strong_match"])
        consumption = match_features(
            "哪些客户累计消费超过10000元", ["消费", "客户"],
            {"id": "fact_order.order_amount", "name": "order_amount", "alias": []},
            "column",
        )
        self.assertTrue(consumption["semantic_hit"])

    def test_entity_display_and_anti_join_keys_are_strong_evidence(self):
        product_name = match_features(
            "哪些商品从来没有产生过订单",
            ["哪些商品从来没有产生过订单"],
            {"id": "dim_product.product_name", "name": "product_name", "alias": []},
            "column",
        )
        dimension_key = match_features(
            "哪些商品从来没有产生过订单",
            ["哪些商品从来没有产生过订单"],
            {"id": "dim_product.product_id", "name": "product_id", "alias": []},
            "column",
        )
        fact_key = match_features(
            "哪些商品从来没有产生过订单",
            ["哪些商品从来没有产生过订单"],
            {"id": "fact_order.product_id", "name": "product_id", "alias": []},
            "column",
        )
        self.assertTrue(product_name["strong_match"])
        self.assertTrue(dimension_key["strong_match"])
        self.assertTrue(fact_key["strong_match"])

    def test_entity_value_literal_precedes_broad_query(self):
        keywords = deterministic_value_route_variants(
            "华东地区各品类销售额", ["华东地区各品类销售额", "品类", "销售额"]
        )
        self.assertEqual(keywords[0], "华东")

    def test_value_keyword_json_parser_is_reachable(self):
        self.assertEqual(_extract_json_array('```json\n["华东"]\n```'), ["华东"])

    def test_value_match_hint_proves_prefix_without_exposing_values(self):
        values = [
            {"value": "测试品牌甲"},
            {"value": "测试品牌乙"},
            {"value": "测试品牌丙"},
        ]
        hint = _safe_value_match_hint(
            "统计测试品牌共有多少个品牌", "测试品牌", values,
        )
        self.assertEqual("prefix", hint["operator"])
        self.assertEqual("测试品牌", hint["query_term"])
        self.assertEqual(3, hint["matched_distinct_lower_bound"])
        self.assertFalse(hint["stored_values_exposed"])
        self.assertNotIn("甲", str(hint))

    def test_value_match_hint_prefers_exact_and_rejects_llm_only_term(self):
        values = [{"value": "华东"}, {"value": "华东一区"}]
        exact = _safe_value_match_hint("华东成交额", "华东", values)
        self.assertEqual("exact", exact["operator"])
        self.assertIsNone(_safe_value_match_hint("东部成交额", "华东", values))

    def test_value_match_exact_hint_does_not_pollute_another_column(self):
        annotated = _annotate_value_match_hints(
            "华东成交额",
            "华东",
            [
                {"column_id": "dim_region.region_name", "value": "华东"},
                {
                    "column_id": "dim_product.brand",
                    "value": "华东科技",
                    "value_match_hints": [{"query_term": "旧请求"}],
                },
            ],
        )

        self.assertEqual(
            "exact", annotated[0]["value_match_hints"][0]["operator"],
        )
        self.assertNotIn("value_match_hints", annotated[1])

    def test_value_match_prefix_lower_bound_is_not_combined_across_columns(self):
        annotated = _annotate_value_match_hints(
            "华东成交额",
            "华东",
            [
                {"column_id": "dim_region.region_name", "value": "华东一区"},
                {"column_id": "dim_product.brand", "value": "华东科技"},
            ],
        )

        self.assertTrue(all(
            "value_match_hints" not in item for item in annotated
        ))

    def test_value_match_prefix_hint_uses_only_same_column_and_stays_private(self):
        annotated = _annotate_value_match_hints(
            "测试品牌销售额",
            "测试品牌",
            [
                {"column_id": "dim_product.brand", "value": "测试品牌甲"},
                {"column_id": "dim_product.brand", "value": "测试品牌乙"},
                {"column_id": "dim_region.region_name", "value": "测试品牌园区"},
            ],
        )

        brand_hints = [item["value_match_hints"][0] for item in annotated[:2]]
        self.assertTrue(all(hint["operator"] == "prefix" for hint in brand_hints))
        self.assertTrue(all(
            hint["matched_distinct_lower_bound"] == 2 for hint in brand_hints
        ))
        self.assertNotIn("value_match_hints", annotated[2])
        for hint in brand_hints:
            self.assertEqual("测试品牌", hint["query_term"])
            self.assertFalse(hint["stored_values_exposed"])
            self.assertNotIn("甲", str(hint))
            self.assertNotIn("乙", str(hint))

    def test_semantic_closure_fields_survive_dynamic_cut(self):
        candidates = [
            {"id": f"fact_order.noise_{index}", "name": f"noise_{index}", "score": .99 - index / 100}
            for index in range(10)
        ] + [
            {"id": "fact_order.customer_id", "name": "customer_id", "score": .01},
            {"id": "fact_order.order_id", "name": "order_id", "score": .01},
        ]
        selected, _ = rerank_and_cut(
            "column", "复购率是多少", ["复购率"], candidates, min_limit=3, max_limit=8
        )
        selected_ids = {item["id"] for item in selected}
        self.assertIn("fact_order.customer_id", selected_ids)
        self.assertIn("fact_order.order_id", selected_ids)

    def test_contract_links_sixteen_reviewed_golden_cases(self):
        contract = validate_contract()
        self.assertTrue(contract["valid"], contract["errors"])
        self.assertGreaterEqual(contract["total"], 16)

    def test_contract_rejects_training_qrel_absent_from_live_corpus(self):
        contract = copy.deepcopy(load_contract())
        contract["cases"][0]["relevant_training"] = [
            "不存在于受治理训练语料中的样例",
        ]

        validation = validate_contract(contract)

        self.assertFalse(validation["valid"])
        self.assertTrue(any(
            "relevant_training 不属于 canonical live corpus" in error
            for error in validation["errors"]
        ))

    def test_tuned_profile_passes_all_recall_and_display_gates(self):
        report = build_report(write=False)
        summary = report["summary"]
        self.assertEqual(report["source"], "deterministic_offline_index")
        self.assertFalse(report["live_index_evaluated"])
        self.assertTrue(summary["gate_passed"])
        self.assertGreaterEqual(summary["routes"]["column"]["recall_at_k"], .98)
        self.assertGreaterEqual(summary["routes"]["column"]["precision_at_k"], .40)
        self.assertGreaterEqual(summary["display_hit_rate"], .95)
        self.assertEqual(summary["tenant_leakage"], 0)
        self.assertLess(report["tuned"]["summary"]["candidate_count"], report["fixed_k"]["summary"]["candidate_count"])

    def test_entity_display_inference_is_conservative(self):
        self.assertTrue(infer_display_column("dim_product", "product_name", "dimension"))
        self.assertFalse(infer_display_column("dim_product", "product_id", "primary_key"))
        self.assertFalse(infer_display_column("fact_order", "order_name", "dimension"))

    def test_live_run_uses_persisted_repository_candidates(self):
        workspace_id = "workspace-live-test"
        cases = []
        for expected in load_contract()["cases"]:
            expected_value_ids = []
            for value_label in expected.get("relevant_values") or []:
                column_id, raw_value = str(value_label).split(":", 1)
                expected_value_ids.append(
                    value_document_id(
                        workspace_id,
                        column_id,
                        normalize_value(raw_value),
                    )
                )
            retrieval = {
                "columns": [{"id": value, "rank": index, "score": .9, "index_version": "live-v1"}
                            for index, value in enumerate(expected.get("relevant_columns") or [], 1)],
                "metrics": [{"id": value, "name": value, "rank": index, "score": .9, "index_version": "live-v1"}
                            for index, value in enumerate(expected.get("relevant_metrics") or [], 1)],
                "training": [{"id": value, "question": value, "rank": index, "score": .9, "index_version": "live-v1"}
                             for index, value in enumerate(expected.get("relevant_training") or [], 1)],
                "values": [{"id": value, "rank": index, "score": .9, "index_version": "live-v1"}
                           for index, value in enumerate(expected_value_ids, 1)],
            }
            cases.append({
                "id": expected["id"], "rag_case_id": expected["id"],
                "question": expected["query"],
                "stage_evidence": {"retrieval": retrieval, "filtered_tables": expected["expected_tables"]},
            })
        report = evaluate_live_run({
            "run_id": "full", "workspace_id": workspace_id,
            "cases": cases, "fingerprint": {"run_fingerprint": "fp"},
        })
        self.assertTrue(report["available"])
        self.assertTrue(report["live_index_evaluated"])
        self.assertTrue(report["summary"]["gate_passed"])
        self.assertEqual(report["candidate_index_versions"], ["live-v1"])
        self.assertTrue(all(item["passed"] for item in report["cases"]))

    def test_live_run_reports_exact_missing_field_per_case(self):
        expected = load_contract()["cases"][0]
        actual = {
            "id": expected["id"], "rag_case_id": expected["id"],
            "question": expected["query"],
            "stage_evidence": {
                "retrieval": {"columns": [], "metrics": [], "training": [], "values": []},
                "filtered_tables": expected["expected_tables"],
            },
        }
        report = evaluate_live_run({
            "run_id": "missing",
            "workspace_id": "workspace-live-test",
            "cases": [actual],
        })
        row = next(item for item in report["cases"] if item["id"] == expected["id"])
        self.assertFalse(row["passed"])
        self.assertEqual(row["diagnostics"]["columns"]["missing"], sorted(expected["relevant_columns"]))

    def test_live_run_never_reuses_a_different_contract_question(self):
        expected = load_contract()["cases"][0]
        report = evaluate_live_run({
            "run_id": "wrong-query",
            "cases": [{
                "id": expected["id"], "rag_case_id": expected["id"],
                "question": "与 RAG 契约不同的问题",
                "stage_evidence": {"retrieval": {"columns": []}},
            }],
        })
        self.assertFalse(report["available"])
        self.assertIn(expected["id"], report["missing_case_ids"])


class RAGRepositoryEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_column_search_applies_budget_and_returns_ranked_evidence(self):
        client = AsyncMock()
        client.collection_exists.return_value = False
        point = SimpleNamespace(
            id="point-1", score=.91, payload={
                "id": "dim_product.product_name", "name": "product_name", "type": "varchar",
                "role": "dimension", "examples": [], "description": "商品名称", "alias": ["商品"],
                "table_id": "dim_product", "is_display": True, "index_version": "sha256:test",
                "entity_id": "dim_product.product_name", "point_role": "name",
                "point_strategy": "single_canonical",
            },
        )
        client.query_points.return_value = SimpleNamespace(points=[point])
        embedding = HybridEmbedding(
            dense=(.1, .2), sparse_indices=(1,), sparse_values=(.3,)
        )
        result = await ColumnQdrantRepository(client).search(
            embedding, index_version="sha256:current"
        )
        self.assertEqual(client.query_points.await_count, 2)
        for call in client.query_points.await_args_list:
            kwargs = call.kwargs
            self.assertEqual(
                kwargs["limit"],
                app_config.query.column_recall_limit
                * app_config.query.hybrid_oversample_factor,
            )
            conditions = {item.key: item.match.value for item in kwargs["query_filter"].must}
            self.assertEqual(conditions["index_version"], "sha256:current")
            self.assertEqual(conditions["embedding_mode"], "dense_sparse")
        self.assertEqual(result.items[0]["rank"], 1)
        self.assertEqual(result.items[0]["point_id"], "point-1")
        self.assertEqual(result.items[0]["retrieval_sources"], ["dense", "sparse"])
        self.assertTrue(result.items[0]["is_display"])


if __name__ == "__main__":
    unittest.main()
