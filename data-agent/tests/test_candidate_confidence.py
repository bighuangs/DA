import time
import asyncio
import unittest
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

from app.adapters.outbound.agent.nl2sql_capabilities import (
    Nl2SqlLLMAdapter,
    build_nl2sql_settings,
)
from app.agent_system.agents.nl2sql.settings import override_settings

from app.agent_system.agents.nl2sql.nodes.judge_candidate import judge_candidate
from app.agent_system.agents.nl2sql.services.candidate_confidence import (
    assess_candidate,
    choose_candidate,
    compact_candidate_decision,
    normalized_sql_equal,
    should_expand_candidate,
    should_review_candidate,
)
from evals.legacy.candidate_confidence_eval import build_report
from app.config.app_config import QueryConfig, app_config


def _near_cost_limit() -> int:
    return int(app_config.query.max_explain_rows * 0.95)


def _table_infos():
    return [
        {"name": "dim_region", "columns": [
            {"name": "region_id", "role": "primary_key", "is_display": False},
            {"name": "region_name", "role": "dimension", "is_display": True},
        ]},
        {"name": "fact_order", "columns": [
            {"name": "region_id", "role": "foreign_key", "is_display": False},
            {"name": "order_amount", "role": "measure", "is_display": False},
        ]},
    ]


def _product_table_infos():
    return [
        {"name": "dim_product", "columns": [
            {"name": "product_id", "role": "primary_key", "is_display": False},
            {"name": "product_name", "role": "dimension", "is_display": True},
        ]},
        {"name": "fact_order", "columns": [
            {"name": "order_id", "role": "primary_key", "is_display": False},
            {"name": "product_id", "role": "foreign_key", "is_display": False},
            {"name": "customer_id", "role": "foreign_key", "is_display": False},
        ]},
    ]


def _state(**overrides):
    value = {
        "query": "对比各地区销售额",
        "table_infos": _table_infos(),
        "metric_infos": [],
        "semantic_ir": {"status": "partial", "compiler": {"supported": False}},
        "retrieval_metrics": [{"route": "column", "strong_match_count": 1}],
        "degradations": [],
        "retry_count": 0,
        "follow_up": {},
        "chat_history": [],
        "db_info": {"dialect": "mysql"},
        "performance_started_at": time.perf_counter(),
        "sql": "SELECT r.region_name, SUM(o.order_amount) total_amount FROM dim_region r JOIN fact_order o ON r.region_id=o.region_id GROUP BY r.region_name",
        "validation_evidence": {"valid": True, "estimated_rows": 100},
    }
    value.update(overrides)
    return value


class CandidateConfidenceTests(unittest.TestCase):
    def test_candidate_fallback_config_matches_engineering_budget(self):
        fallback = QueryConfig(limit_rows=1000)
        self.assertTrue(fallback.sql_semantic_review_enabled)
        self.assertEqual(fallback.request_deadline_ms, 60000)
        self.assertEqual(fallback.adaptive_candidate_deadline_ms, 52000)
        self.assertEqual(fallback.adaptive_candidate_timeout_seconds, 18.0)

    def test_policy_matrix_meets_trigger_repair_and_safety_gates(self):
        summary = build_report(write=False)["summary"]
        self.assertEqual(30, summary["cases"])
        self.assertEqual(.1667, summary["second_candidate_rate"])
        self.assertEqual(.8, summary["targeted_repair_rate"])
        self.assertEqual(0, summary["unsafe_selected"])
        self.assertEqual(3, summary["semantic_contract_cases"])
        self.assertEqual(3, summary["semantic_contract_passed"])
        self.assertEqual(0, summary["semantic_false_positive_count"])
        self.assertTrue(summary["gate_passed"])

    def test_high_confidence_uses_single_candidate_fast_path(self):
        state = _state(semantic_ir={"status": "complete", "compiler": {"supported": True}})
        assessment = assess_candidate(state["sql"], state, validation=state["validation_evidence"])
        self.assertEqual("high", assessment["band"])
        self.assertEqual((False, "high_confidence_fast_path"), should_expand_candidate(assessment, state, 100))

    def test_degradation_and_cost_pressure_trigger_adaptive_candidate(self):
        state = _state(
            degradations=[{"node_id": "recall_column", "reason": "timeout"}],
            validation_evidence={"valid": True, "estimated_rows": _near_cost_limit()},
        )
        assessment = assess_candidate(state["sql"], state, validation=state["validation_evidence"])
        self.assertIn(assessment["band"], {"medium", "low"})
        self.assertTrue(should_expand_candidate(assessment, state, 100)[0])

    def test_existing_correction_and_deadline_block_extra_call(self):
        state = _state(retry_count=1)
        assessment = {"valid": True, "band": "medium"}
        self.assertEqual("correction_budget_already_used", should_expand_candidate(assessment, state, 100)[1])
        state["retry_count"] = 0
        self.assertEqual("ttresult_budget_exhausted", should_expand_candidate(assessment, state, 53000)[1])

    def test_semantic_review_ignores_high_confidence_and_correction_history(self):
        state = _state(
            retry_count=1,
            semantic_ir={"status": "complete", "compiler": {"supported": True}},
        )
        assessment = assess_candidate(
            state["sql"], state, validation=state["validation_evidence"],
        )
        self.assertEqual("high", assessment["band"])
        self.assertEqual(
            (True, "semantic_review_required"),
            should_review_candidate(assessment, state, 100),
        )

    def test_semantic_review_has_explicit_ablation_and_useful_budget_gate(self):
        assessment = {"valid": True, "band": "high"}
        disabled = override_settings(
            build_nl2sql_settings(),
            "query",
            sql_semantic_review_enabled=False,
        )
        self.assertEqual(
            (False, "semantic_review_disabled"),
            should_review_candidate(
                assessment,
                _state(),
                100,
                query_settings=disabled.query,
            ),
        )
        self.assertEqual(
            (False, "semantic_review_budget_insufficient"),
            should_review_candidate(assessment, _state(), 51999),
        )

    def test_grouped_customer_count_does_not_require_customer_display_name(self):
        state = _state(
            query="按会员等级统计下单客户",
            table_infos=[{"name": "dim_customer", "columns": [
                {"name": "customer_id", "role": "primary_key", "is_display": False},
                {"name": "customer_name", "role": "dimension", "is_display": True},
                {"name": "member_level", "role": "dimension", "is_display": False},
            ]}],
            sql="SELECT member_level, COUNT(DISTINCT customer_id) FROM dim_customer GROUP BY member_level",
            semantic_ir={"status": "partial", "dimensions": [{"table": "dim_customer", "column": "member_level"}]},
            degradations=[
                {"node_id": "recall_column", "reason": "dependency_unavailable"},
                {"node_id": "recall_metric", "reason": "dependency_unavailable"},
                {"node_id": "recall_value", "reason": "dependency_unavailable"},
            ],
            validation_evidence={"valid": True},
        )
        assessment = assess_candidate(state["sql"], state, validation=state["validation_evidence"])
        self.assertNotIn("display_column_missing", assessment["weaknesses"])
        self.assertEqual("medium", assessment["band"])
        self.assertEqual((False, "no_actionable_contract_gap"), should_expand_candidate(assessment, state, 100))

    def test_guided_review_ignores_wrong_ir_display_dimension(self):
        state = _state(
            query="每个季度汇总售出件数",
            table_infos=[
                {"name": "fact_order", "columns": [
                    {"name": "date_id", "role": "foreign_key"},
                    {"name": "product_id", "role": "foreign_key"},
                    {"name": "order_quantity", "role": "measure", "alias": ["售出件数"]},
                ]},
                {"name": "dim_date", "columns": [
                    {"name": "date_id", "role": "primary_key"},
                    {"name": "quarter", "role": "dimension", "alias": ["季度"]},
                ]},
                {"name": "dim_product", "columns": [
                    {"name": "product_id", "role": "primary_key"},
                    {"name": "product_name", "role": "dimension", "is_display": True},
                ]},
            ],
            sql=(
                "SELECT d.quarter, SUM(f.order_quantity) AS total_quantity "
                "FROM fact_order f JOIN dim_date d ON f.date_id=d.date_id "
                "GROUP BY d.quarter"
            ),
            semantic_ir={
                "status": "complete",
                "dimensions": [{"table": "dim_product", "column": "product_name"}],
                "model_sql_policy": {"use_model": True},
                "query_contract": {"authority": "guided"},
            },
            validation_evidence={"valid": True},
        )
        assessment = assess_candidate(
            state["sql"], state, validation=state["validation_evidence"],
        )
        self.assertNotIn("display_column_missing", assessment["weaknesses"])

    def test_repeat_rate_cte_is_one_physical_source_and_needs_no_join(self):
        state = _state(
            query="复购率是多少",
            table_infos=[{"name": "fact_order", "columns": [
                {"name": "order_id", "role": "primary_key"},
                {"name": "customer_id", "role": "foreign_key"},
            ]}],
            sql=(
                "WITH customer_orders AS ("
                "SELECT customer_id, COUNT(order_id) AS order_count FROM fact_order GROUP BY customer_id"
                ") SELECT SUM(CASE WHEN order_count >= 2 THEN 1 ELSE 0 END) "
                "/ NULLIF(COUNT(*), 0) AS repurchase_rate FROM customer_orders"
            ),
            semantic_ir={
                "status": "complete", "intent": "repeat_rate",
                "measures": [{"table": "fact_order", "alias": "repurchase_rate"}],
                "compiler": {"supported": True},
            },
            degradations=[
                {"node_id": "recall_column", "reason": "timeout"},
                {"node_id": "recall_metric", "reason": "timeout"},
                {"node_id": "recall_training", "reason": "timeout"},
            ],
            validation_evidence={"valid": True},
        )
        assessment = assess_candidate(state["sql"], state, validation=state["validation_evidence"])
        self.assertEqual(1, assessment["sql_features"]["table_count"])
        self.assertEqual(1, assessment["sql_features"]["cte_count"])
        self.assertFalse(assessment["sql_features"]["join_required"])
        self.assertNotIn("join_shape_incomplete", assessment["weaknesses"])
        self.assertFalse(should_expand_candidate(assessment, state, 100)[0])

    def test_complete_left_anti_join_closes_partial_ir_without_candidate(self):
        state = _state(
            query="没有产生订单的商品",
            table_infos=_product_table_infos(),
            sql=(
                "SELECT p.product_name FROM dim_product p "
                "LEFT JOIN fact_order f ON p.product_id=f.product_id "
                "WHERE f.order_id IS NULL"
            ),
            semantic_ir={"status": "partial", "compiler": {"supported": False}},
            retrieval_metrics=[{"route": "column", "strong_match_count": 0}],
            degradations=[
                {"node_id": "recall_column", "reason": "timeout"},
                {"node_id": "recall_metric", "reason": "timeout"},
                {"node_id": "recall_training", "reason": "timeout"},
            ],
            validation_evidence={"valid": True},
        )
        assessment = assess_candidate(state["sql"], state, validation=state["validation_evidence"])
        self.assertTrue(assessment["sql_features"]["anti_join_satisfied"])
        self.assertEqual("left_join_is_null", assessment["sql_features"]["anti_join_strategy"])
        self.assertNotIn("complex_semantics_without_complete_ir", assessment["weaknesses"])
        self.assertNotIn("anti_join_contract_missing", assessment["weaknesses"])
        self.assertFalse(should_expand_candidate(assessment, state, 100)[0])

    def test_missing_anti_join_is_an_actionable_contract_gap(self):
        state = _state(
            query="没有产生订单的商品",
            table_infos=_product_table_infos(),
            sql="SELECT p.product_name FROM dim_product p",
            semantic_ir={"status": "partial", "compiler": {"supported": False}},
            validation_evidence={"valid": True},
        )
        assessment = assess_candidate(state["sql"], state, validation=state["validation_evidence"])
        self.assertIn("anti_join_contract_missing", assessment["actionable_weaknesses"])
        self.assertTrue(should_expand_candidate(assessment, state, 100)[0])

    def test_invalid_alternative_is_never_selected(self):
        selected, reason = choose_candidate(
            {"valid": True, "score": .60, "band": "medium"}, {"valid": False, "score": 0},
        )
        self.assertEqual("A", selected)
        self.assertIn("rejected", reason)

    def test_two_untrusted_candidates_return_no_selection(self):
        selected, reason = choose_candidate(
            {"valid": True, "score": .52, "band": "low"},
            {"valid": False, "score": 0, "band": "low"},
        )
        self.assertEqual("none", selected)
        self.assertEqual("no_trustworthy_candidate", reason)

    def test_alternative_needs_material_gain_and_ties_keep_primary(self):
        self.assertEqual("B", choose_candidate(
            {"valid": True, "score": .60}, {"valid": True, "score": .80},
        )[0])
        self.assertEqual("A", choose_candidate(
            {"valid": True, "score": .80}, {"valid": True, "score": .81},
        )[0])

    def test_normalized_same_sql_never_selects_alternative(self):
        primary_sql = (
            "SELECT r.region_name, SUM(o.order_amount) total_amount "
            "FROM dim_region r JOIN fact_order o "
            "ON r.region_id=o.region_id GROUP BY r.region_name"
        )
        formatted_sql = (
            "select R.region_name, sum(O.order_amount) AS total_amount "
            "from DIM_REGION AS R join fact_order AS O "
            "on R.region_id = O.region_id group by R.region_name;"
        )
        self.assertTrue(normalized_sql_equal(primary_sql, formatted_sql))
        selected, reason = choose_candidate(
            {"valid": True, "score": .60, "band": "medium"},
            {"valid": True, "score": .99, "band": "high"},
            primary_sql=primary_sql,
            alternative_sql=formatted_sql,
        )
        self.assertEqual("A", selected)
        self.assertEqual("alternative_has_no_normalized_sql_delta", reason)

    def test_compact_decision_never_persists_sql(self):
        result = compact_candidate_decision({
            "selected": "B", "adaptive_triggered": True,
            "candidates": [{"label": "B", "valid": True, "score": .9, "band": "high", "sql": "SELECT secret"}],
        })
        self.assertNotIn("sql", result["candidates"][0])

    def test_compact_decision_persists_only_boolean_sql_delta(self):
        result = compact_candidate_decision({
            "selected": "B",
            "sql_delta": True,
            "candidates": [{"label": "A", "valid": True}],
        })
        self.assertIs(result["sql_delta"], True)
        self.assertNotIn("normalized_sql", result)

    def test_legacy_semantic_review_is_not_charged_as_adaptive_expansion(self):
        result = compact_candidate_decision({
            "adaptive_triggered": True,
            "extra_call_used": True,
            "trigger_reason": "semantic_review_required",
            "selection_reason": "semantic_review_validated",
            "selected": "B",
            "candidates": [
                {"label": "A", "valid": True},
                {"label": "B", "valid": True},
            ],
        })

        self.assertEqual("semantic_review", result["decision_mode"])
        self.assertTrue(result["decision_mode_inferred"])
        self.assertTrue(result["semantic_review_triggered"])
        self.assertTrue(result["semantic_review_eligible"])
        self.assertFalse(result["adaptive_triggered"])

    def test_modern_adaptive_decision_keeps_exclusive_lane(self):
        result = compact_candidate_decision({
            "decision_mode": "adaptive_expansion",
            "adaptive_triggered": True,
            "semantic_review_triggered": True,
            "trigger_reason": "low_confidence_expand",
            "candidates": [{"label": "A", "valid": True}],
        })

        self.assertEqual("adaptive_expansion", result["decision_mode"])
        self.assertTrue(result["adaptive_triggered"])
        self.assertFalse(result["semantic_review_triggered"])


class _DW:
    async def validate_sql(self, sql):
        return {
            "estimated_rows": 100 if "LIMIT 100" in sql else _near_cost_limit(),
            "explain_rows": [],
        }


class _Runtime:
    def __init__(self, settings=None):
        self.context = {
            "dw_mysql_repo": _DW(),
            "access_policy": None,
            "llm": Nl2SqlLLMAdapter(),
            "settings": settings or build_nl2sql_settings(),
        }
        self.events = []

    def stream_writer(self, value):
        self.events.append(value)


class AdaptiveCandidateNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_candidate_timeout_emits_reconciled_core_attempt(self):
        state = _state(
            sql="SELECT r.region_id, SUM(o.order_amount) total_amount FROM dim_region r JOIN fact_order o ON r.region_id=o.region_id GROUP BY r.region_id",
            degradations=[{"node_id": "recall_column", "reason": "timeout"}],
            validation_evidence={"valid": True, "estimated_rows": _near_cost_limit()},
        )

        async def slow(_):
            await asyncio.sleep(.05)
            return "SELECT 1"

        runtime = _Runtime(override_settings(
            build_nl2sql_settings(),
            "query",
            adaptive_candidate_timeout_seconds=.001,
        ))
        with patch(
            "app.agent_system.agents.nl2sql.nodes.judge_candidate.llm",
            RunnableLambda(slow),
        ):
            result = await judge_candidate(state, runtime)
        self.assertEqual(result["llm_attempts"][0]["status"], "timeout")
        self.assertTrue(result["llm_attempts"][0]["affects_core_path"])
        self.assertEqual(result["degradations"][0]["node_id"], "judge_candidate")
        self.assertEqual(result["degradations"][0]["route"], "candidate")
        self.assertTrue(result["degradations"][0]["evidence_complete"])
        self.assertEqual("A", result["candidate_decision"]["selected"])
        self.assertEqual(
            "semantic_review_generation_failed_primary_kept",
            result["candidate_decision"]["selection_reason"],
        )
        self.assertEqual(
            "semantic_review", result["candidate_decision"]["decision_mode"]
        )
        self.assertTrue(result["candidate_decision"]["semantic_review_triggered"])
        self.assertFalse(result["candidate_decision"]["adaptive_triggered"])

    async def test_review_budget_skip_is_an_explicit_core_degradation(self):
        state = _state(validation_evidence={"valid": True, "estimated_rows": 1})
        runtime = _Runtime()

        with patch(
            "app.agent_system.agents.nl2sql.nodes.judge_candidate.elapsed_ms",
            return_value=app_config.query.adaptive_candidate_deadline_ms,
        ):
            result = await judge_candidate(state, runtime)

        decision = result["candidate_decision"]
        self.assertEqual("semantic_review", decision["decision_mode"])
        self.assertTrue(decision["semantic_review_eligible"])
        self.assertFalse(decision["semantic_review_triggered"])
        self.assertEqual(
            "semantic_review_budget_insufficient", decision["trigger_reason"]
        )
        self.assertEqual(
            "semantic_review_budget_insufficient",
            result["degradations"][0]["reason_code"],
        )
        self.assertTrue(result["degradations"][0]["affects_core_path"])

    async def test_two_untrusted_candidates_fail_instead_of_executing_primary(self):
        state = _state(
            sql="SELECT r.region_id, SUM(o.order_amount) total_amount FROM dim_region r JOIN fact_order o ON r.region_id=o.region_id GROUP BY r.region_id",
            degradations=[{"node_id": "recall_column", "reason": "timeout"}],
            validation_evidence={"valid": True, "estimated_rows": _near_cost_limit()},
        )
        runtime = _Runtime(override_settings(
            build_nl2sql_settings(),
            "query",
            sql_semantic_review_enabled=False,
        ))
        with patch(
            "app.agent_system.agents.nl2sql.nodes.judge_candidate.llm",
            RunnableLambda(lambda _: "DELETE FROM fact_order"),
        ):
            result = await judge_candidate(state, runtime)
        self.assertEqual("none", result["candidate_decision"]["selected"])
        self.assertIn("置信度不足", result["error"])
        self.assertIsNone(result["authorized_sql"])

    async def test_medium_confidence_generates_one_validated_alternative_and_selects_b(self):
        state = _state(
            degradations=[{"node_id": "recall_column", "reason": "timeout"}],
            validation_evidence={"valid": True, "estimated_rows": _near_cost_limit()},
        )
        alternative = (
            "SELECT r.region_name, SUM(o.order_amount) total_amount "
            "FROM dim_region r JOIN fact_order o ON r.region_id=o.region_id "
            "GROUP BY r.region_name LIMIT 100"
        )
        runtime = _Runtime()
        with patch("app.agent_system.agents.nl2sql.nodes.judge_candidate.llm", RunnableLambda(lambda _: alternative)):
            result = await judge_candidate(state, runtime)
        self.assertEqual("B", result["candidate_decision"]["selected"])
        self.assertEqual(2, result["candidate_decision"]["candidate_count"])
        self.assertEqual(alternative, result["sql"])
        self.assertTrue(result["authorized_sql"])
        self.assertEqual(
            result["semantic_repair_ledger"][0]["phase"],
            "candidate_b_validate",
        )
        self.assertTrue(result["semantic_repair_ledger"][0]["selected"])
        self.assertTrue(
            result["semantic_repair_ledger"][0]["committed_at_phase"]
        )
        self.assertEqual(
            result["semantic_repair_ledger"][0]["repaired_sql_hash"],
            result["semantic_repair_ledger"][0]["observed_sql_hash"],
        )

    async def test_high_confidence_does_not_call_alternative_model_in_ablation(self):
        state = _state(semantic_ir={"status": "complete", "compiler": {"supported": True}})
        called = False

        def invoke(_):
            nonlocal called
            called = True
            return "SELECT 1"

        settings = override_settings(
            build_nl2sql_settings(),
            "query",
            sql_semantic_review_enabled=False,
        )
        with patch(
            "app.agent_system.agents.nl2sql.nodes.judge_candidate.llm",
            RunnableLambda(invoke),
        ):
            result = await judge_candidate(state, _Runtime(settings))
        self.assertFalse(called)
        self.assertEqual(1, result["candidate_decision"]["candidate_count"])

    async def test_high_confidence_valid_primary_is_semantically_reviewed(self):
        state = _state(
            query="按地区统计销售额和订单数",
            semantic_ir={"status": "complete", "compiler": {"supported": True}},
        )
        reviewed = (
            "SELECT r.region_name, SUM(o.order_amount) total_amount, "
            "COUNT(*) order_count FROM dim_region r JOIN fact_order o "
            "ON r.region_id=o.region_id GROUP BY r.region_name"
        )

        with patch(
            "app.agent_system.agents.nl2sql.nodes.judge_candidate.llm", RunnableLambda(lambda _: reviewed),
        ):
            result = await judge_candidate(state, _Runtime())

        self.assertEqual("semantic_review_required", result["candidate_decision"]["trigger_reason"])
        self.assertEqual("semantic_review_validated", result["candidate_decision"]["selection_reason"])
        self.assertEqual("B", result["candidate_decision"]["selected"])
        self.assertIs(result["candidate_decision"]["sql_delta"], True)
        self.assertEqual(reviewed, result["sql"])

    async def test_semantic_review_same_normalized_sql_keeps_primary(self):
        state = _state(
            semantic_ir={"status": "complete", "compiler": {"supported": True}},
        )
        formatted = (
            "select r.region_name, sum(o.order_amount) AS total_amount "
            "from dim_region AS r join fact_order AS o "
            "on r.region_id = o.region_id group by r.region_name"
        )

        with patch(
            "app.agent_system.agents.nl2sql.nodes.judge_candidate.llm",
            RunnableLambda(lambda _: formatted),
        ):
            result = await judge_candidate(state, _Runtime())

        decision = result["candidate_decision"]
        self.assertEqual("A", decision["selected"])
        self.assertEqual(
            "semantic_review_same_sql_primary_kept",
            decision["selection_reason"],
        )
        self.assertEqual("same_sql_kept_a", decision["candidate_outcome"])
        self.assertIs(decision["sql_delta"], False)
        self.assertNotIn("sql", result)
        self.assertEqual(
            result["semantic_repair_ledger"][0]["phase"],
            "candidate_b_validate",
        )
        self.assertFalse(result["semantic_repair_ledger"][0]["selected"])
        self.assertFalse(
            result["semantic_repair_ledger"][0]["committed_at_phase"]
        )

    async def test_invalid_semantic_review_keeps_any_valid_primary(self):
        state = _state(
            sql=(
                "SELECT r.region_id, SUM(o.order_amount) total_amount "
                "FROM dim_region r JOIN fact_order o ON r.region_id=o.region_id "
                "GROUP BY r.region_id"
            ),
            degradations=[{"node_id": "recall_column", "reason": "timeout"}],
            validation_evidence={"valid": True, "estimated_rows": _near_cost_limit()},
        )
        with patch(
            "app.agent_system.agents.nl2sql.nodes.judge_candidate.llm",
            RunnableLambda(lambda _: "SELECT * FROM missing_table"),
        ):
            result = await judge_candidate(state, _Runtime())

        self.assertEqual("A", result["candidate_decision"]["selected"])
        self.assertEqual(
            "semantic_review_rejected_primary_kept",
            result["candidate_decision"]["selection_reason"],
        )
        self.assertNotIn("error", result)


if __name__ == "__main__":
    unittest.main()
