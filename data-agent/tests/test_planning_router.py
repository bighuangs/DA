import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.adapters.outbound.agent.nl2sql_capabilities import build_nl2sql_settings
from app.agent_system.agents.nl2sql.settings import override_settings
from app.agent_system.agents.nl2sql.services.planning_router import (
    decide_planning_route,
    extract_planning_features,
    keyword_length_needs_plan,
    rollback_to_fallback,
    runtime_profile_status,
    set_runtime_profile_for_test,
)
from app.agent_system.agents.nl2sql.services.query_signals import extract_query_signals
from evals.legacy.planning_counterfactual import (
    label_counterfactual,
    load_planning_contract,
    validate_planning_contract,
)
from evals.legacy.planning_router_ablation import (
    PROFILE_MATRIX,
    build_report,
    case_state,
)
from evals.legacy.evaluator import _planning_route_stage_evidence
from evals.application.services.planning_evaluation_service import PlanningEvaluationService


BASE_SETTINGS = build_nl2sql_settings()


class PlanningRouterTests(unittest.TestCase):
    def tearDown(self):
        set_runtime_profile_for_test(None)

    def test_c1_is_character_compatible_with_original_contract(self):
        cases = {
            "": False,
            "销售总额是多少": False,
            "各地区销售额同比": True,
            "同时告诉我订单数": True,
            "累计消费金额超过10000的客户": False,
            "这是一个只需要统计订单总数但包含了很多礼貌背景信息的很长问题请帮忙看看": True,
        }
        for query, expected in cases.items():
            self.assertEqual(expected, keyword_length_needs_plan(query), query)
            state = case_state({
                "question": query or "空问题",
                "context": {},
                "semantic_ir_supported": False,
            })
            if query:
                decision = decide_planning_route(
                    state,
                    profile="keyword_length_v1",
                    shadow_profiles=(),
                )
                self.assertEqual(
                    "plan_then_generate" if expected else "direct_generate",
                    decision.actual_action,
                )

    def test_semantic_ir_supported_uses_correctness_first_route_by_default(self):
        case = next(
            item
            for item in load_planning_contract()["cases"]
            if item["review_label"] == "ir_bypass"
        )
        payload = decide_planning_route(
            case_state(case),
            profile="always_plan",
        ).privacy_summary()
        self.assertEqual("plan_then_generate", payload["actual_action"])
        self.assertFalse(payload["semantic_ir_supported"])
        self.assertFalse(payload["excluded_from_router_denominator"])
        self.assertEqual(["control_always_plan"], payload["reason_codes"])

    def test_direct_emission_ablation_keeps_audited_ir_bypass(self):
        case = next(
            item
            for item in load_planning_contract()["cases"]
            if item["review_label"] == "ir_bypass"
        )
        settings = override_settings(
            BASE_SETTINGS,
            "query",
            deterministic_sql_direct_emission_enabled=True,
        )
        payload = decide_planning_route(
            case_state(case),
            profile="always_plan",
            settings=settings,
        ).privacy_summary()
        self.assertEqual("deterministic_ir_bypass", payload["actual_action"])
        self.assertTrue(payload["excluded_from_router_denominator"])
        self.assertEqual(["deterministic_ir_supported"], payload["reason_codes"])

    def test_follow_up_boolean_preserve_slots_are_counted_safely(self):
        signals = extract_query_signals(
            "那销售额呢",
            profile="domain_signals_v1",
            workspace_id="planning-test-workspace",
        )
        features = extract_planning_features({
            "query": "那销售额呢",
            "query_signals": signals.to_state(),
            "follow_up": {
                "action": "replace",
                "preserve": {
                    "dimensions": ["dim_product.category"],
                    "filters": True,
                    "filter_columns": ["dim_date.year"],
                    "order": False,
                    "order_directions": [],
                    "limit": 10,
                },
            },
            "table_infos": [],
            "metric_infos": [],
        })
        self.assertTrue(features.is_follow_up)
        self.assertEqual(features.inherited_contract_count, 4)

    def test_c3_c4_and_c5_produce_versioned_privacy_safe_evidence(self):
        case = next(
            item
            for item in load_planning_contract()["cases"]
            if item["id"] == "plan_027"
        )
        for profile in (
            "structured_rules_v1",
            "weighted_scorecard_v1",
            "calibrated_linear_v1",
        ):
            payload = decide_planning_route(
                case_state(case),
                profile=profile,
                shadow_profiles=(),
            ).privacy_summary()
            raw = json.dumps(payload, ensure_ascii=False)
            self.assertEqual("planning-route-v1", payload["contract_version"])
            self.assertEqual("plan_then_generate", payload["actual_action"])
            self.assertFalse(payload["raw_text_logged"])
            self.assertNotIn(case["question"], raw)
            self.assertNotIn("standalone_query", raw)

    def test_scorecard_abstain_uses_c1_fallback_explicitly(self):
        case = {
            "question": (
                "请结合当前已经授权并且能够查看的全部业务数据同时告诉我销售总额"
                "并保持现有指标口径"
            ),
            "context": {},
            "semantic_ir_supported": False,
        }
        payload = decide_planning_route(
            case_state(case),
            profile="weighted_scorecard_v1",
            feature_level="A0",
            shadow_profiles=(),
        ).privacy_summary()
        self.assertEqual("abstain", payload["decision"])
        self.assertTrue(payload["fallback_applied"])
        self.assertEqual("plan_then_generate", payload["actual_action"])

    def test_live_c4_policy_abstain_is_complete_evaluation_evidence(self):
        settings = override_settings(
            BASE_SETTINGS,
            "planning_route",
            active_profile="weighted_scorecard_v1",
        )
        decision = decide_planning_route(
            {
                    "query": "销售额比率",
                    "query_signals": extract_query_signals(
                        "销售额比率",
                        profile="jieba_tfidf_v1",
                        workspace_id="planning-policy-fallback-test",
                    ).to_state(),
                    "table_infos": [],
                    "metric_infos": [],
                },
            shadow_profiles=(),
            settings=settings,
        )
        payload = decision.privacy_summary()
        with patch(
            "evals.legacy.evaluator.planning_runtime_profile_status",
            return_value={
                "configured_profile": "weighted_scorecard_v1",
                "runtime_profile": "weighted_scorecard_v1",
                "runtime_override": None,
                "fallback_profile": "keyword_length_v1",
            },
        ):
            evidence = _planning_route_stage_evidence({
                "planning_route": payload,
            })

        self.assertEqual(payload["decision"], "abstain")
        self.assertTrue(payload["fallback_applied"])
        self.assertFalse(payload["router_degraded"])
        self.assertEqual(evidence["fallback_class"], "policy_abstain")
        self.assertTrue(evidence["policy_fallback_eligible"])
        self.assertTrue(evidence["evidence_complete"], evidence)

    def test_invalid_active_profile_degrades_to_c1(self):
        settings = override_settings(
            BASE_SETTINGS,
            "planning_route",
            active_profile="not_implemented",
        )
        case = next(
            item
            for item in load_planning_contract()["cases"]
            if item["id"] == "plan_031"
        )
        payload = decide_planning_route(
            case_state(case),
            shadow_profiles=(),
            settings=settings,
        ).privacy_summary()
        self.assertTrue(payload["router_degraded"])
        self.assertTrue(payload["fallback_applied"])
        self.assertEqual("keyword_length_v1", payload["profile"])

    def test_runtime_rollback_needs_no_restart(self):
        set_runtime_profile_for_test("weighted_scorecard_v1")
        self.assertEqual(
            "weighted_scorecard_v1",
            runtime_profile_status()["runtime_profile"],
        )
        status = rollback_to_fallback()
        self.assertEqual("keyword_length_v1", status["runtime_profile"])
        self.assertTrue(status["rollback_without_restart"])

    def test_reviewed_pair_labels_and_ablation_contract(self):
        contract = validate_planning_contract()
        report = build_report(write=False)
        self.assertTrue(contract["valid"], contract["errors"])
        self.assertTrue(report["implementation_gates"]["gate_passed"])
        self.assertEqual(set(PROFILE_MATRIX), set(report["profiles"]))
        self.assertEqual(
            {"A0", "A1", "A2", "A3", "A4"},
            set(report["feature_ablations"]),
        )
        self.assertFalse(report["decision"]["release_qualified"])
        self.assertTrue(report["decision"]["active_unchanged"])
        self.assertGreater(
            report["profiles"]["C4"]["plan_recall"],
            report["profiles"]["C1"]["plan_recall"],
        )

    def test_both_failed_and_inconclusive_are_retained(self):
        cases = load_planning_contract()["cases"]
        labels = {
            case["id"]: label_counterfactual(case)["label"]
            for case in cases
        }
        self.assertIn("both_failed", labels.values())
        self.assertIn("inconclusive", labels.values())
        report = build_report(write=False)
        self.assertGreaterEqual(
            report["evaluation_scope"]["excluded"]["both_failed"], 1
        )
        self.assertGreaterEqual(
            report["evaluation_scope"]["excluded"]["inconclusive"], 1
        )

    def test_service_exposes_evidence_boundary_and_runtime(self):
        report = PlanningEvaluationService(
            SimpleNamespace(
                status=runtime_profile_status,
                rollback=rollback_to_fallback,
            )
        ).latest()
        self.assertEqual(
            "reviewed_offline_counterfactual", report["evidence_level"]
        )
        self.assertFalse(report["decision"]["release_qualified"])
        self.assertEqual(
            "keyword_length_v1", report["runtime"]["runtime_profile"]
        )


if __name__ == "__main__":
    unittest.main()
