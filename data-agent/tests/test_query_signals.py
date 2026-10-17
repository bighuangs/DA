import json
import unittest
from types import SimpleNamespace

from app.agent_system.agents.nl2sql.nodes.extract_keywords import extract_keywords
from app.adapters.outbound.agent.nl2sql_capabilities import build_nl2sql_settings
from app.agent_system.agents.nl2sql.settings import override_settings
from app.agent_system.agents.nl2sql.services.query_signals import (
    build_route_signal_plan,
    extract_query_signals,
    longest_workspace_matches,
)


class QuerySignalExtractionTests(unittest.TestCase):
    def test_k0_keeps_only_complete_question_as_retrieval_term(self):
        query = "2025年销售额同比增长前10的地区"
        signals = extract_query_signals(query, profile="full_query_only")
        self.assertEqual(signals.retrieval_terms(), [query])
        self.assertEqual(signals.lexical_terms, ())
        self.assertEqual(signals.numeric_constraints, ())

    def test_k1_reproduces_full_query_plus_jieba_without_rule_signals(self):
        query = "order_amount与GMV有什么区别"
        signals = extract_query_signals(query, profile="jieba_tfidf_v1")
        self.assertEqual(signals.retrieval_terms()[0], query)
        self.assertIn("GMV", signals.lexical_terms)
        self.assertEqual(signals.identifiers, ())

    def test_k2_preserves_identifier_time_number_operator_and_polarity(self):
        query = "只看2025年order_amount同比增长前10，不要已支付订单"
        signals = extract_query_signals(
            query,
            profile="domain_signals_v1",
            workspace_id="ws-a",
            workspace_terms=["order_amount", "订单金额"],
        )
        self.assertIn("order_amount", signals.identifiers)
        self.assertIn("2025年", signals.time_expressions)
        self.assertIn("前10", signals.numeric_constraints)
        self.assertIn("同比增长", signals.operators)
        self.assertIn("不要", signals.polarity)
        self.assertIn("已支付", signals.polarity)
        self.assertIn("order_amount", signals.domain_terms)

    def test_k2_maps_broad_composition_language_to_typed_local_signals(self):
        signals = extract_query_signals(
            "按组内比重列出首次购买、但未曾回购的顾客分档",
            profile="domain_signals_v1",
        )
        self.assertIn("比重", signals.operators)
        self.assertIn("首次购买", signals.operators)
        self.assertIn("回购", signals.operators)
        self.assertIn("分档", signals.operators)
        self.assertIn("但未", signals.polarity)
        self.assertIn("未曾", signals.polarity)

    def test_workspace_dictionary_uses_longest_non_overlapping_match(self):
        self.assertEqual(
            longest_workspace_matches(
                "比较订单金额和订单量",
                ["订单", "订单金额", "订单量"],
            ),
            ("订单金额", "订单量"),
        )

    def test_workspace_dictionary_is_request_local_and_fingerprinted(self):
        first = extract_query_signals(
            "比较GMV和净销售额",
            profile="domain_signals_v1",
            workspace_id="ws-a",
            workspace_terms=["GMV"],
        )
        second = extract_query_signals(
            "比较GMV和净销售额",
            profile="domain_signals_v1",
            workspace_id="ws-b",
            workspace_terms=["净销售额"],
        )
        self.assertEqual(first.domain_terms, ("GMV",))
        self.assertEqual(second.domain_terms, ("净销售额",))
        self.assertNotEqual(
            first.provenance.dictionary_version,
            second.provenance.dictionary_version,
        )

    def test_explicit_dictionary_version_is_kept_even_for_empty_dictionary(self):
        signals = extract_query_signals(
            "销售额",
            profile="domain_signals_v1",
            dictionary_version="empty-dictionary-v7",
        )
        self.assertEqual(
            signals.provenance.dictionary_version, "empty-dictionary-v7"
        )

    def test_signal_and_profile_fingerprints_are_stable(self):
        kwargs = {
            "profile": "domain_signals_v1",
            "workspace_id": "ws-a",
            "workspace_terms": ["销售额", "地区"],
        }
        first = extract_query_signals("各地区销售额", **kwargs)
        second = extract_query_signals("各地区销售额", **kwargs)
        self.assertEqual(first.signal_fingerprint, second.signal_fingerprint)
        self.assertEqual(
            first.provenance.profile_fingerprint,
            second.provenance.profile_fingerprint,
        )

    def test_privacy_summary_never_contains_signal_text(self):
        signals = extract_query_signals(
            "客户Alice的customer_name",
            profile="domain_signals_v1",
            workspace_id="ws-a",
            workspace_terms=["customer_name"],
        )
        raw = json.dumps(signals.privacy_summary(), ensure_ascii=False)
        self.assertNotIn("Alice", raw)
        self.assertNotIn("customer_name", raw)
        self.assertFalse(signals.privacy_summary()["raw_text_logged"])


class QuerySignalRoutingTests(unittest.TestCase):
    def test_routed_qdrant_uses_only_complete_query(self):
        query = "2025年order_amount同比增长前10"
        signals = extract_query_signals(
            query,
            profile="domain_signals_v1",
            workspace_id="ws-a",
            workspace_terms=["order_amount"],
        )
        for route in ("column", "metric", "training"):
            plan = build_route_signal_plan(
                signals,
                route,
                route_policy="routed_signals_v1",
                remote_signal_limit=4,
            )
            self.assertEqual(plan.remote_terms, [query])
            self.assertIn("order_amount", plan.local_terms)
            if route in {"column", "metric", "training"}:
                self.assertIn("同比增长", plan.local_terms)

    def test_metric_and_training_rerank_consume_operator_and_polarity_only_locally(self):
        query = "按各组贡献率统计，但未包含退款"
        signals = extract_query_signals(
            query,
            profile="domain_signals_v1",
        )
        for route in ("metric", "training"):
            plan = build_route_signal_plan(
                signals,
                route,
                route_policy="routed_signals_v1",
                remote_signal_limit=4,
            )
            self.assertEqual(plan.remote_terms, [query])
            self.assertIn("贡献率", plan.local_terms)
            self.assertIn("但未", plan.local_terms)
            self.assertNotIn("贡献率", plan.remote_terms)

    def test_current_policy_reproduces_multi_query_but_training_stays_single(self):
        query = "统计华东地区各品类销售额"
        signals = extract_query_signals(query, profile="jieba_tfidf_v1")
        column = build_route_signal_plan(
            signals,
            "column",
            route_policy="current_multi_query_v1",
            remote_signal_limit=4,
        )
        training = build_route_signal_plan(
            signals,
            "training",
            route_policy="current_multi_query_v1",
            remote_signal_limit=4,
        )
        self.assertGreater(len(column.remote_terms), 1)
        self.assertEqual(training.remote_terms, [query])

    def test_value_route_prioritizes_entity_without_losing_complete_query(self):
        query = "统计华东地区销售额"
        signals = extract_query_signals(query, profile="jieba_tfidf_v1")
        plan = build_route_signal_plan(
            signals,
            "value",
            route_policy="current_multi_query_v1",
            remote_signal_limit=4,
            priority_terms=["华东"],
        )
        self.assertEqual(plan.remote_terms[0], "华东")
        self.assertIn(query, plan.remote_terms)

    def test_route_namespace_changes_with_policy_and_order(self):
        signals = extract_query_signals(
            "各地区销售额", profile="jieba_tfidf_v1"
        )
        current = build_route_signal_plan(
            signals,
            "column",
            route_policy="current_multi_query_v1",
            remote_signal_limit=4,
        )
        full = build_route_signal_plan(
            signals,
            "column",
            route_policy="full_query_only_v1",
            remote_signal_limit=4,
        )
        self.assertNotEqual(current.cache_namespace, full.cache_namespace)


class QuerySignalNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_domain_dictionary_contains_only_authorized_workspace_meta(self):
        class Repo:
            workspace_id = "ws-a"

            async def list_table_infos(self):
                table = SimpleNamespace(id="fact_order", name="fact_order")
                columns = [
                    SimpleNamespace(
                        name="order_amount",
                        alias=["销售额"],
                        table_id="fact_order",
                    ),
                    SimpleNamespace(
                        name="secret_margin",
                        alias=["机密利润率"],
                        table_id="fact_order",
                    ),
                ]
                return [(table, columns)]

            async def get_all_metric_infos(self):
                return [
                    SimpleNamespace(
                        name="销售总额",
                        alias=["GMV"],
                        relevant_columns=["fact_order.order_amount"],
                    )
                ]

        class Policy:
            workspace_id = "ws-a"
            role = "viewer"

            @staticmethod
            def is_table_allowed(table_id):
                return table_id == "fact_order"

            @staticmethod
            def is_column_allowed(table_id, column):
                return table_id == "fact_order" and column == "order_amount"

            @staticmethod
            def _split_column_id(value):
                return str(value).rsplit(".", 1)

        events = []
        runtime = SimpleNamespace(
            context={
                "meta_mysql_repo": Repo(),
                "access_policy": Policy(),
                "workspace_id": "ws-a",
                "settings": override_settings(
                    build_nl2sql_settings(),
                    "query_signals",
                    profile="domain_signals_v1",
                ),
            },
            stream_writer=events.append,
        )
        result = await extract_keywords(
            {"query": "比较销售额与机密利润率"},
            runtime,
        )
        signals = result["query_signals"]
        self.assertIn("销售额", signals["domain_terms"])
        self.assertNotIn("机密利润率", signals["domain_terms"])
        self.assertEqual(result["degradations"], [])

    async def test_dictionary_failure_falls_back_without_losing_query(self):
        class Repo:
            workspace_id = "ws-a"

            async def list_table_infos(self):
                raise RuntimeError("meta unavailable")

        events = []
        runtime = SimpleNamespace(
            context={
                "meta_mysql_repo": Repo(),
                "workspace_id": "ws-a",
                "settings": override_settings(
                    build_nl2sql_settings(),
                    "query_signals",
                    profile="domain_signals_v1",
                ),
            },
            stream_writer=events.append,
        )
        result = await extract_keywords({"query": "各地区销售额"}, runtime)
        self.assertEqual(
            result["query_signals"]["provenance"]["extractor_profile"],
            "jieba_tfidf_v1",
        )
        self.assertEqual(
            result["degradations"][0]["dependency"],
            "query_signal_dictionary",
        )
        self.assertFalse(result["degradations"][0]["raw_text_logged"])


if __name__ == "__main__":
    unittest.main()
