import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

import app.agent_system.agents.nl2sql.nodes.generate_analysis as analysis_module
import app.agent_system.agents.nl2sql.nodes.generate_chart as chart_module
from app.agent_system.agents.nl2sql.llm import LLMInvocationResult
from app.agent_system.agents.nl2sql.nodes.generate_analysis import build_deterministic_analysis
from app.agent_system.agents.nl2sql.nodes.generate_chart import (
    build_deterministic_chart,
    generate_chart,
)
from app.agent_system.agents.nl2sql.settings import override_settings
from app.adapters.outbound.agent.nl2sql_capabilities import build_nl2sql_settings
from app.adapters.inbound.http.schemas.query_schema import QuerySchema
from app.agent_system.runtime.performance import instrument_node
from app.agent_system.agents.nl2sql.services.planning_router import keyword_length_needs_plan
from evals.legacy.run_eval import aggregate


class DeterministicChartTest(unittest.TestCase):
    def test_trend_question_builds_line_chart_without_llm(self):
        chart = build_deterministic_chart(
            "查看每月销售额趋势",
            [{"month": "2026-01", "sales": 12}, {"month": "2026-02", "sales": 18}],
        )
        self.assertEqual(chart["type"], "line")
        self.assertEqual(chart["axis"]["x"]["value"], "month")
        self.assertEqual(chart["axis"]["y"][0]["value"], "sales")

    def test_explicit_line_chart_keeps_numeric_month_as_dimension(self):
        chart = build_deterministic_chart(
            "用折线图展示月度销售额趋势",
            [
                {"month": 1, "total_amount": 632396550.30},
                {"month": 2, "total_amount": 577492829.67},
            ],
        )

        self.assertEqual(chart["type"], "line")
        self.assertEqual(chart["axis"]["x"]["value"], "month")
        self.assertEqual(chart["axis"]["y"][0]["value"], "total_amount")

    def test_non_numeric_rows_fall_back_to_table(self):
        chart = build_deterministic_chart("查看客户", [{"name": "甲"}, {"name": "乙"}])
        self.assertEqual(chart["type"], "table")

    def test_explicit_pie_request_overrides_category_comparison_default(self):
        chart = build_deterministic_chart(
            "销量前十的商品，改为饼图",
            [
                {"product_name": "商品甲", "total_quantity": 16800},
                {"product_name": "商品乙", "total_quantity": 15900},
            ],
        )

        self.assertEqual(chart["type"], "pie")
        self.assertEqual(chart["axis"]["series"]["value"], "product_name")
        self.assertEqual(chart["axis"]["y"]["value"], "total_quantity")

    def test_last_explicit_chart_instruction_wins(self):
        chart = build_deterministic_chart(
            "不要柱状图，改为饼图",
            [{"product_name": "商品甲", "total_quantity": 16800}],
        )

        self.assertEqual(chart["type"], "pie")

    def test_negated_chart_type_is_not_treated_as_requested_type(self):
        chart = build_deterministic_chart(
            "不要饼图",
            [{"product_name": "商品甲", "total_quantity": 16800}],
        )

        self.assertEqual(chart["type"], "column")

    def test_numeric_looking_category_values_remain_pie_dimension(self):
        chart = build_deterministic_chart(
            "销量前十的商品，改为饼图",
            [
                {"product_name": "7000022", "total_quantity": 16800},
                {"product_name": "7000070", "total_quantity": 15900},
            ],
        )

        self.assertEqual(chart["type"], "pie")
        self.assertEqual(chart["axis"]["series"]["value"], "product_name")


class GenerateChartNodeTest(unittest.IsolatedAsyncioTestCase):
    async def test_current_follow_up_utterance_controls_chart_type(self):
        events = []
        runtime = SimpleNamespace(
            stream_writer=events.append,
            context={"settings": build_nl2sql_settings()},
        )
        state = {
            "query": "销量前十的商品，改为饼图",
            "original_query": "改为饼图",
            "sql": "SELECT product_name, total_quantity FROM ranked LIMIT 10",
            "result": [
                {"product_name": "商品甲", "total_quantity": 16800},
                {"product_name": "商品乙", "total_quantity": 15900},
            ],
            "table_infos": [],
        }

        output = await generate_chart(state, runtime)

        self.assertEqual(output["chart"]["type"], "pie")
        self.assertEqual(output["chart"]["axis"]["series"]["value"], "product_name")
        self.assertEqual(output["chart"]["axis"]["y"]["value"], "total_quantity")
        self.assertTrue(any(event.get("chart", {}).get("type") == "pie" for event in events))

    async def test_explicit_pie_overrides_llm_column_recommendation(self):
        events = []
        settings = override_settings(
            build_nl2sql_settings(),
            "query",
            chart_llm_enabled=True,
        )

        class ColumnRecommendationCapability:
            def __init__(self):
                self.calls = 0

            async def invoke_optional(self, _runnable, _payload, **_kwargs):
                self.calls += 1
                return LLMInvocationResult(
                    value=(
                        '{"type":"column","title":"模型推荐柱状图",'
                        '"axis":{"x":{"value":"product_name"},'
                        '"y":[{"value":"total_quantity"}]}}'
                    ),
                    attempts=[],
                    degradations=[],
                )

        capability = ColumnRecommendationCapability()
        runtime = SimpleNamespace(
            stream_writer=events.append,
            context={"settings": settings, "llm": capability},
        )
        state = {
            "query": "销量前十的商品，改为饼图",
            "original_query": "改为饼图",
            "sql": "SELECT product_name, total_quantity FROM ranked LIMIT 10",
            "result": [
                {"product_name": "7000022", "total_quantity": 16800},
                {"product_name": "7000070", "total_quantity": 15900},
            ],
            "table_infos": [],
        }

        with patch.object(
            chart_module,
            "llm",
            RunnableLambda(lambda _payload: "unused"),
        ):
            output = await generate_chart(state, runtime)

        self.assertEqual(capability.calls, 1)
        self.assertEqual(output["chart"]["type"], "pie")
        self.assertEqual(output["chart"]["title"], "模型推荐柱状图")
        self.assertEqual(
            output["chart"]["axis"]["series"]["value"],
            "product_name",
        )
        self.assertTrue(
            any(event.get("chart", {}).get("type") == "pie" for event in events)
        )


class DeterministicAnalysisTest(unittest.IsolatedAsyncioTestCase):
    def test_query_schema_keeps_analysis_optional_and_off_by_server_default(self):
        self.assertIsNone(QuerySchema(query="各地区销售额").enable_analysis)
        self.assertTrue(QuerySchema(query="各地区销售额", enable_analysis=True).enable_analysis)

    def test_summary_uses_only_result_statistics(self):
        analysis = build_deterministic_analysis("大客户有哪些", [
            {"customer_name": "客户甲", "total_amount": 1200},
            {"customer_name": "客户乙", "total_amount": 800},
        ])
        self.assertIn("返回 2 行", analysis)
        self.assertIn("累计金额合计 2,000", analysis)
        self.assertIn("客户甲", analysis)
        self.assertIn("未添加结果之外的业务推断", analysis)

    async def test_compiled_ir_skips_llm_and_streams_analysis(self):
        events = []
        runtime = SimpleNamespace(
            stream_writer=events.append,
            context={"settings": build_nl2sql_settings()},
        )
        state = {
            "query": "大客户有哪些",
            "sql": "SELECT customer_name, total_amount FROM ranked",
            "result": [{"customer_name": "客户甲", "total_amount": 1200}],
            "semantic_ir": {"compiler": {"emitted": True}},
            "analysis_enabled": True,
        }

        async def must_not_run(_):
            raise AssertionError("deterministic IR should not call analysis LLM")

        with patch.object(analysis_module, "llm", RunnableLambda(must_not_run)):
            output = await analysis_module.generate_analysis(state, runtime)
        self.assertTrue(output["analysis"])
        self.assertTrue(any(event.get("analysis") for event in events))

    async def test_llm_timeout_returns_deterministic_fallback(self):
        events = []
        settings = override_settings(
            build_nl2sql_settings(),
            "query",
            analysis_timeout_seconds=0.01,
        )
        runtime = SimpleNamespace(
            stream_writer=events.append,
            context={"settings": settings},
        )
        state = {
            "query": "复杂分析",
            "sql": "SELECT region, amount FROM result",
            "result": [{"region": "华东", "amount": 10}, {"region": "华南", "amount": 8}],
            "semantic_ir": {"compiler": {"emitted": False}},
            "chart": {},
            "table_infos": [],
            "chat_history": [],
            "analysis_enabled": True,
        }

        async def slow_model(_):
            await asyncio.sleep(0.1)
            return "迟到的模型结果"

        class TimeoutLLMCapability:
            def model(self, _task):
                return RunnableLambda(slow_model)

            async def invoke_optional(self, runnable, payload, **_kwargs):
                return await asyncio.wait_for(
                    runnable.ainvoke(payload),
                    timeout=0.01,
                )

        runtime.context["llm"] = TimeoutLLMCapability()
        output = await analysis_module.generate_analysis(state, runtime)
        self.assertTrue(output["analysis"])
        self.assertEqual(output["degradations"][0]["reason"], "timeout")
        self.assertEqual(output["degradations"][0]["fallback"], "deterministic_summary")
        self.assertTrue(any(event.get("analysis") for event in events))

    async def test_disabled_analysis_skips_stage_and_llm(self):
        events = []
        runtime = SimpleNamespace(
            stream_writer=events.append,
            context={"settings": build_nl2sql_settings()},
        )
        state = {
            "query": "复杂分析",
            "result": [{"amount": 10}],
            "analysis_enabled": False,
        }

        async def must_not_run(_):
            raise AssertionError("disabled analysis should not call LLM")

        with patch.object(analysis_module, "llm", RunnableLambda(must_not_run)):
            output = await analysis_module.generate_analysis(state, runtime)

        self.assertEqual(output, {"analysis": ""})
        self.assertFalse(any(event.get("stage") == "数据分析解读" for event in events))
        self.assertTrue(any(event.get("optional_feature", {}).get("status") == "disabled" for event in events))


class PlanningRouteBudgetTest(unittest.TestCase):
    def test_simple_group_aggregation_skips_duplicate_planning_call(self):
        self.assertFalse(keyword_length_needs_plan("各地区的销售总额"))
        self.assertFalse(keyword_length_needs_plan("每个商品的销量"))

    def test_semantically_complex_questions_keep_planning(self):
        self.assertTrue(keyword_length_needs_plan("各地区销售额占比"))
        self.assertTrue(keyword_length_needs_plan("每月销售额趋势"))
        self.assertFalse(keyword_length_needs_plan("累计消费金额超过10000的客户"))


class NodeInstrumentationTest(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_safe_metric_and_events_use_stable_node_id(self):
        events = []
        runtime = SimpleNamespace(stream_writer=events.append)

        async def node(state, runtime):
            await asyncio.sleep(0.001)
            return {"value": 1}

        started = time.perf_counter()
        output = await instrument_node("sample_node", node)(
            {"performance_started_at": started}, runtime
        )
        self.assertEqual(output["value"], 1)
        self.assertEqual(output["node_metrics"][0]["node_id"], "sample_node")
        self.assertEqual(events[0]["trace_event"]["event"], "node_start")
        self.assertEqual(events[-1]["trace_event"]["event"], "node_end")

    async def test_synchronous_graph_node_is_supported(self):
        events = []
        runtime = SimpleNamespace(stream_writer=events.append)

        def node(state, runtime):
            return {"value": 2}

        output = await instrument_node("sync_node", node)(
            {"performance_started_at": time.perf_counter()}, runtime
        )
        self.assertEqual(output["value"], 2)
        self.assertEqual(output["node_metrics"][0]["node_id"], "sync_node")


class PerformanceAggregateTest(unittest.TestCase):
    def test_node_token_cost_and_milestones_are_aggregated(self):
        base = {
            "syntax_valid": True, "exec_success": True, "sql_equivalent": True,
            "structure_equivalent": True, "tables_correct": True,
            "result_equivalent": True, "table_f1": 1.0,
            "chart_generated": True, "analysis_generated": True,
            "recommend_generated": True, "retry_count": 0,
            "failure_type": None, "latency_ms": 100,
        }
        result = aggregate([{**base,
            "tt_sql_ms": 60, "tt_result_ms": 80, "llm_calls": 2,
            "total_tokens": 300, "estimated_cost": 0.004,
            "cost_currency": "USD", "degradations": [],
            "node_metrics": [
                {"node_id": "generate_sql", "duration_ms": 40},
                {"node_id": "execute_sql", "duration_ms": 20},
            ],
        }])
        self.assertEqual(result["tt_result_p95"], 80)
        self.assertEqual(result["llm_calls_mean"], 2)
        self.assertEqual(result["tokens_mean"], 300)
        self.assertEqual(result["estimated_cost_total"], 0.004)
        self.assertEqual(result["node_performance"][0]["node_id"], "generate_sql")


if __name__ == "__main__":
    unittest.main()
