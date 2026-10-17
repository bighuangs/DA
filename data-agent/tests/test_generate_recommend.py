import unittest
from types import SimpleNamespace

from app.adapters.outbound.agent.nl2sql_capabilities import build_nl2sql_settings
from app.agent_system.agents.nl2sql.nodes.generate_recommend import (
    RECOMMEND_NUMBER,
    deterministic_recommendations,
    generate_recommend,
    upload_schema_recommendations,
)
from app.agent_system.agents.nl2sql.settings import override_settings
from app.agent_system.agents.nl2sql.services.recommendation_options import (
    RECOMMENDATION_CONTRACT_VERSION,
    build_contextual_recommendations,
    build_recommendation_context,
    normalize_model_recommendations,
)


class GenerateRecommendTests(unittest.IsolatedAsyncioTestCase):
    def test_deterministic_recommendations_are_bounded_and_do_not_repeat_query(self):
        result = deterministic_recommendations("各地区的销售额")
        self.assertLessEqual(len(result), RECOMMEND_NUMBER)
        self.assertNotIn("各地区的销售额", result)
        self.assertTrue(all(isinstance(item, str) and item for item in result))

    def test_contextual_recommendations_bind_single_result_and_are_diverse(self):
        state = {
            "query": "销售额最高的地区",
            "sql": (
                "SELECT r.region_name, SUM(f.order_amount) AS total_sales_amount "
                "FROM fact_order f JOIN dim_region r ON f.region_id = r.region_id "
                "GROUP BY r.region_name ORDER BY total_sales_amount DESC LIMIT 1"
            ),
            "result": [{"region_name": "华东", "total_sales_amount": 200000}],
            "semantic_ir": {
                "dimensions": [{"logical": "region_name"}],
                "measures": [{"logical": "order_amount"}],
                "filters": [],
                "limit": 1,
            },
        }

        recommendations = build_contextual_recommendations(state)
        context = build_recommendation_context(state)

        self.assertEqual(len(recommendations), RECOMMEND_NUMBER)
        self.assertEqual(len({item["kind"] for item in recommendations}), RECOMMEND_NUMBER)
        self.assertTrue(all("华东" in item["query"] for item in recommendations))
        self.assertFalse(any("该地区" in item["query"] for item in recommendations))
        self.assertTrue(all(item["label"] != item["query"] for item in recommendations))
        self.assertNotIn("result_sample", context)
        self.assertEqual(context["scope"][0]["value"], "华东")

    def test_model_recommendations_without_result_scope_are_rejected_and_filled(self):
        state = {
            "query": "销售额最高的地区",
            "result": [{"region_name": "华东", "total_sales_amount": 200000}],
            "semantic_ir": {
                "dimensions": [{"logical": "region_name"}],
                "measures": [{"logical": "order_amount"}],
                "filters": [],
                "limit": 1,
            },
        }
        fallback = build_contextual_recommendations(state)

        recommendations = normalize_model_recommendations(
            [
                {"label": "分析品类", "query": "查询各品类销售额", "kind": "breakdown"},
                {"label": "查看趋势", "query": "用折线图展示华东地区每月销售额趋势", "kind": "trend"},
                {"label": "再看趋势", "query": "查看华东地区每日销售额趋势", "kind": "trend"},
            ],
            state=state,
            fallback=fallback,
        )

        self.assertEqual(len(recommendations), RECOMMEND_NUMBER)
        self.assertTrue(all("华东" in item["query"] for item in recommendations))
        self.assertEqual(sum(item["kind"] == "trend" for item in recommendations), 1)
        self.assertNotIn("查询各品类销售额", [item["query"] for item in recommendations])

    async def test_default_path_returns_recommendations_without_model_context(self):
        events = []
        settings = override_settings(
            build_nl2sql_settings(),
            "query",
            recommend_llm_enabled=False,
        )
        result = await generate_recommend(
            {"query": "复购率是多少"},
            SimpleNamespace(
                stream_writer=events.append,
                context={"settings": settings},
            ),
        )
        self.assertTrue(result["recommend"])
        self.assertEqual(
            result["recommendation_contract"]["schema_version"],
            RECOMMENDATION_CONTRACT_VERSION,
        )
        self.assertEqual(
            result["recommendation_contract"]["focus_policy_version"],
            "recommendation-focus-continuity/v1",
        )
        self.assertEqual(
            result["recommend"],
            [item["label"] for item in result["recommendations"]],
        )
        self.assertNotIn("context_budget", result)
        self.assertTrue(any(event.get("recommendations") for event in events))

    async def test_upload_mode_recommendations_only_use_authorized_schema(self):
        events = []
        state = {
            "query": "按地区汇总上传文件中的金额",
            "data_source_context": {
                "source_mode": "upload_only",
                "profiles": [{
                    "columns": [
                        {"name": "c_003", "display_name": "region", "role": "dimension", "type_family": "text"},
                        {"name": "c_004", "display_name": "order_date", "role": "dimension", "type_family": "time"},
                        {"name": "c_005", "display_name": "amount", "role": "measure", "type_family": "number"},
                    ],
                }],
            },
        }
        settings = override_settings(
            build_nl2sql_settings(),
            "query",
            recommend_llm_enabled=True,
        )
        result = await generate_recommend(
            state,
            SimpleNamespace(
                stream_writer=events.append,
                context={"settings": settings},
            ),
        )
        self.assertEqual(result["recommend"], upload_schema_recommendations(state))
        self.assertEqual(
            [item["query"] for item in result["recommendations"]],
            upload_schema_recommendations(state),
        )
        self.assertTrue(result["recommend"])
        self.assertIn("region", result["recommend"][0])
        self.assertFalse(any("品类" in item or "2025" in item for item in result["recommend"]))
        self.assertTrue(all(
            any(name in item for name in ("region", "amount", "order_date", "上传文件"))
            for item in result["recommend"]
        ))


if __name__ == "__main__":
    unittest.main()
