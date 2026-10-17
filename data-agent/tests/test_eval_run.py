import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.domain.access.policy import PolicySnapshot
from app.domain.data_sources.context import stable_fingerprint
from app.agent_system.runtime.llm_attempts import attempt_accounting, make_llm_attempt
from app.infrastructure.clients.llm.capabilities import candidate_capability
from evals.legacy.compare_runs import run_payload_is_valid
from evals.legacy.evaluator import (
    _sql_contract_metrics,
    run_case,
    semantic_review_case_metrics,
)
from evals.legacy.model_invocation_contract import (
    GENERATE_SQL_PHYSICAL_ATTEMPT_PER_CASE,
    NON_EMPTY_REFERENCE_WITH_PROJECTION,
    evaluate_llm_invocation_contract,
    validate_reference_result_contract,
)
from evals.legacy.run_eval import (
    _todo97_expected_route_identity,
    _table_scope_evidence_complete,
    aggregate,
    build_execution_audit,
    collect_context_fingerprint,
    ensure_experiment_repeat_available,
    provider_failure_audit,
    results_form_valid_run,
    run_cases_fail_fast,
    targeted_report_path,
    targeted_run_path,
)


class EvalRunValidityTests(unittest.TestCase):
    def test_table_scope_safety_is_independent_from_exact_required_tables(self):
        metrics = _sql_contract_metrics(
            "SELECT region_name FROM dim_region",
            "SELECT region_name FROM dim_region JOIN fact_order USING (region_id)",
            {
                "required_tables": ["dim_region", "fact_order"],
                "allowed_tables": ["dim_region", "fact_order", "dim_date"],
            },
        )

        self.assertFalse(metrics["tables_correct"])
        self.assertTrue(metrics["table_scope_safe"])
        self.assertEqual(1.0, metrics["table_precision"])

        unsafe = _sql_contract_metrics(
            "SELECT * FROM secret_table",
            "SELECT * FROM dim_region",
            {
                "required_tables": ["dim_region"],
                "allowed_tables": ["dim_region", "fact_order"],
            },
        )
        self.assertFalse(unsafe["table_scope_safe"])

    def test_live_table_scope_uses_runtime_authority_not_reference_path(self):
        metrics = _sql_contract_metrics(
            (
                "SELECT f.customer_id, SUM(f.order_amount) AS total_amount "
                "FROM fact_order f JOIN dim_customer c "
                "ON f.customer_id = c.customer_id GROUP BY f.customer_id"
            ),
            (
                "SELECT customer_id, SUM(order_amount) AS total_amount "
                "FROM fact_order GROUP BY customer_id"
            ),
            {
                "required_tables": ["fact_order"],
                "allowed_tables": ["fact_order"],
            },
            runtime_state={
                "table_infos": [
                    {"name": "fact_order", "columns": []},
                    {"name": "dim_customer", "columns": []},
                ],
                "data_source_context": {"source_mode": "dw_only"},
            },
            context={"access_policy": PolicySnapshot(tuple(), role="owner")},
        )

        self.assertFalse(metrics["tables_correct"])
        self.assertTrue(metrics["table_scope_safe"])
        evidence = metrics["table_scope_safety_evidence"]
        self.assertEqual(evidence["authorization_source"], "runtime_table_infos")
        self.assertTrue(evidence["request_table_scope_safe"])
        self.assertTrue(evidence["workspace_access_policy_safe"])

    def test_live_table_scope_rejects_runtime_table_and_column_overreach(self):
        state = {
            "table_infos": [{"name": "fact_order", "columns": []}],
            "data_source_context": {"source_mode": "dw_only"},
        }
        unknown_table = _sql_contract_metrics(
            "SELECT id FROM secret_table",
            "SELECT order_id FROM fact_order",
            {
                "required_tables": ["fact_order"],
                "allowed_tables": ["fact_order", "secret_table"],
            },
            runtime_state=state,
            context={"access_policy": PolicySnapshot(tuple(), role="owner")},
        )
        self.assertFalse(unknown_table["table_scope_safe"])
        self.assertEqual(
            unknown_table["table_scope_safety_evidence"]["failure_stage"],
            "request_table_scope",
        )

        column_policy = SimpleNamespace(
            id=1,
            priority=100,
            role="owner",
            user_id=None,
            table_pattern="fact_order",
            allowed_columns=["order_id"],
            row_filter=None,
            access=True,
        )
        denied_column = _sql_contract_metrics(
            "SELECT order_amount FROM fact_order",
            "SELECT order_id FROM fact_order",
            {
                "required_tables": ["fact_order"],
                "allowed_tables": ["fact_order"],
            },
            runtime_state=state,
            context={
                "access_policy": PolicySnapshot((column_policy,), role="owner")
            },
        )
        self.assertFalse(denied_column["table_scope_safe"])
        self.assertEqual(
            denied_column["table_scope_safety_evidence"]["failure_stage"],
            "workspace_access_policy",
        )

    def test_live_table_scope_rejects_disconnected_join(self):
        metrics = _sql_contract_metrics(
            "SELECT f.order_id FROM fact_order f CROSS JOIN dim_customer c",
            "SELECT order_id FROM fact_order",
            {
                "required_tables": ["fact_order"],
                "allowed_tables": ["fact_order"],
            },
            runtime_state={
                "table_infos": [
                    {"name": "fact_order", "columns": []},
                    {"name": "dim_customer", "columns": []},
                ]
            },
            context={"access_policy": PolicySnapshot(tuple(), role="owner")},
        )
        self.assertFalse(metrics["table_scope_safe"])
        self.assertEqual(
            metrics["table_scope_safety_evidence"]["failure_stage"],
            "join_relationships",
        )

    def test_table_scope_evidence_is_bound_to_exact_final_sql(self):
        result = {
            "gen_sql": "SELECT order_id FROM fact_order",
            "final_sql_metrics": {
                "sql": "SELECT order_id FROM fact_order",
                "table_scope_safety_evidence": {
                    "schema": "table-scope-safety-v2",
                    "authorization_source": "runtime_table_infos",
                    "evidence_complete": True,
                    "sql_hash": stable_fingerprint(
                        "SELECT customer_id FROM fact_order"
                    ),
                    "request_scope_fingerprint": "sha256:scope",
                    "authorized_sql_hash": "sha256:authorized",
                },
            },
        }
        self.assertFalse(_table_scope_evidence_complete(result))

        result["final_sql_metrics"]["table_scope_safety_evidence"][
            "sql_hash"
        ] = stable_fingerprint("SELECT order_id FROM fact_order")
        self.assertTrue(_table_scope_evidence_complete(result))

        result["stage_evidence"] = {
            "authorized_sql": "SELECT customer_id FROM fact_order"
        }
        self.assertFalse(_table_scope_evidence_complete(result))
        result["final_sql_metrics"]["table_scope_safety_evidence"][
            "authorized_sql_hash"
        ] = stable_fingerprint("SELECT customer_id FROM fact_order")
        self.assertTrue(_table_scope_evidence_complete(result))

    def test_semantic_review_metrics_separate_same_sql_flow_losses(self):
        result = {
            "llm_attempts": [],
            "llm_attempt_accounting_complete": True,
            "first_sql_metrics": {
                "sql": "SELECT 1 AS value",
                "guard_passed": True,
                "exec_success": True,
                "result_evaluated": True,
                "result_equivalent": True,
            },
            "final_sql_metrics": {
                "sql": "select 1 value;",
                "guard_passed": True,
                "exec_success": False,
                "result_evaluated": True,
                "result_equivalent": False,
            },
            "stage_evidence": {
                "candidate_decision": {
                    "decision_mode": "semantic_review",
                    "semantic_review_triggered": True,
                    "semantic_review_eligible": True,
                    "trigger_reason": "semantic_review_required",
                    "selected": "B",
                    "sql_delta": False,
                    "candidates": [
                        {"label": "A", "valid": True},
                        {"label": "B", "valid": True},
                    ],
                },
                "sql_observations": [
                    {"sql": "SELECT 1 AS value", "source": "generate_sql"},
                ],
            },
        }

        metrics = semantic_review_case_metrics(result)

        self.assertTrue(metrics["first_correct_final_incorrect"])
        self.assertTrue(metrics["same_sql_execution_loss"])
        self.assertTrue(metrics["selected_b_no_sql_delta"])
        self.assertFalse(metrics["modified"])

        result["stage_evidence"]["candidate_decision"].pop("sql_delta")
        result["stage_evidence"]["sql_observations"] = []
        result["final_sql_metrics"].pop("sql")
        missing_delta = semantic_review_case_metrics(result)
        self.assertTrue(missing_delta["selected_b_no_sql_delta"])
        self.assertFalse(missing_delta["sql_delta_evidence_complete"])
        self.assertFalse(missing_delta["evidence_complete"])

    @staticmethod
    def _fatal_provider_result(case_id: str = "eval_001") -> dict:
        attempt = make_llm_attempt(
            node_id="generate_sql",
            status="error",
            latency_ms=10,
            reason_code="llm.payment_required",
            error_type="APIStatusError",
            retryable=False,
            core_path=True,
        )
        return {
            "id": case_id,
            "exec_success": False,
            "sql_equivalent": False,
            "result_equivalent": False,
            "latency_ms": 10,
            "failure_type": "infrastructure",
            "llm_attempts": [attempt],
        }

    def test_core_fatal_provider_failure_invalidates_run_by_structured_reason(self):
        result = self._fatal_provider_result()
        audit = provider_failure_audit([result])

        self.assertEqual(1, audit["fatal_failure_count"])
        self.assertEqual(1, audit["core_fatal_failure_count"])
        self.assertTrue(audit["stop_required"])
        self.assertTrue(audit["invalidates_run"])
        self.assertEqual(
            "llm.payment_required",
            audit["fatal_failures"][0]["reason_code"],
        )
        self.assertNotIn("provider_response", audit["fatal_failures"][0])
        self.assertFalse(results_form_valid_run([result]))
        # Validity also fails closed if an older caller mislabeled the case as
        # a pipeline failure; the typed core attempt remains authoritative.
        mislabeled = {**result, "failure_type": "pipeline"}
        self.assertFalse(results_form_valid_run([mislabeled]))

    def test_execution_audit_keeps_unexecuted_cases_out_of_results(self):
        result = self._fatal_provider_result("medium_026")
        audit = build_execution_audit(
            ["medium_026", "medium_027", "medium_028"],
            [result],
        )

        self.assertTrue(audit["early_terminated"])
        self.assertEqual(1, audit["executed_count"])
        self.assertEqual(2, audit["not_executed_count"])
        self.assertEqual(
            ["medium_027", "medium_028"],
            audit["not_executed_case_ids"],
        )
        self.assertEqual("medium_026", audit["trigger_case_id"])

    def test_todo97_registry_model_id_maps_to_route_model_name(self):
        candidate = candidate_capability("Q1")

        identity = _todo97_expected_route_identity(candidate)

        self.assertEqual(identity["model_name"], "qwen3.7-plus")
        self.assertEqual(identity["candidate_id"], candidate["candidate_id"])

    def test_targeted_latest_artifacts_are_isolated_by_suite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_dir = root / "reports"
            runs_dir = root / "runs"
            report_paths = {
                targeted_report_path(suite, report_dir)
                for suite in (
                    "easy_sql_v1", "medium_sql_v1", "hard_sql_v1",
                )
            }
            run_paths = {
                targeted_run_path(suite, runs_dir)
                for suite in (
                    "easy_sql_v1", "medium_sql_v1", "hard_sql_v1",
                )
            }

            self.assertEqual(len(report_paths), 3)
            self.assertEqual(len(run_paths), 3)
            self.assertNotIn(report_dir / "targeted_latest.md", report_paths)
            self.assertNotIn(runs_dir / "targeted_latest.json", run_paths)
            self.assertEqual(
                {path.name for path in report_paths},
                {
                    "targeted_easy_sql_v1_latest.md",
                    "targeted_medium_sql_v1_latest.md",
                    "targeted_hard_sql_v1_latest.md",
                },
            )
            self.assertEqual(
                {path.name for path in run_paths},
                {
                    "targeted_easy_sql_v1_latest.json",
                    "targeted_medium_sql_v1_latest.json",
                    "targeted_hard_sql_v1_latest.json",
                },
            )

    @staticmethod
    def _hard_result(case_id: str = "hard_001") -> dict:
        attempt = make_llm_attempt(
            node_id="generate_sql",
            route="sql",
            status="success",
            latency_ms=40,
            timeout_budget_ms=1_000,
            physical_attempt_budget_ms=900,
            prompt_tokens=80,
            completion_tokens=20,
            total_tokens=100,
            token_known=True,
        )
        return {
            "id": case_id,
            "llm_attempts": [attempt],
            "stage_evidence": {
                "semantic_ir": {
                    "compiler": {"supported": False, "emitted": False},
                    "model_sql_policy": {
                        "decision": "model_sql",
                        "use_model": True,
                    },
                },
                "model_sql_fallback": {},
            },
        }

    def test_model_invocation_contract_requires_physical_generate_sql_per_case(self):
        results = [
            self._hard_result("hard_001"),
            self._hard_result("hard_002"),
        ]
        audit = evaluate_llm_invocation_contract(
            results,
            GENERATE_SQL_PHYSICAL_ATTEMPT_PER_CASE,
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["policy_model_cases"], 2)
        self.assertEqual(audit["physical_generate_sql_cases"], 2)
        self.assertEqual(audit["physical_generate_sql_attempts"], 2)
        self.assertEqual(audit["fallback_cases"], [])

    def test_model_invocation_contract_rejects_preinvoke_and_fallback_evidence(self):
        result = self._hard_result()
        attempt = result["llm_attempts"][0]
        attempt["pre_invoke_failure"] = True
        result["stage_evidence"]["semantic_ir"]["model_sql_policy"].update({
            "actual_action": "validated_deterministic_draft",
        })
        result["stage_evidence"]["model_sql_fallback"] = {
            "draft_validated": True,
        }
        audit = evaluate_llm_invocation_contract(
            [result],
            GENERATE_SQL_PHYSICAL_ATTEMPT_PER_CASE,
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["missing_physical_attempt_cases"], ["hard_001"])
        self.assertEqual(audit["fallback_cases"], ["hard_001"])

    def test_model_invocation_contract_rejects_attempt_before_provider_boundary(self):
        result = self._hard_result()
        result["llm_attempts"][0]["completion_stage"] = "queue"
        audit = evaluate_llm_invocation_contract(
            [result],
            GENERATE_SQL_PHYSICAL_ATTEMPT_PER_CASE,
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["missing_physical_attempt_cases"], ["hard_001"])

    def test_model_invocation_contract_fails_closed_for_unknown_or_partial_suite(self):
        result = self._hard_result()
        unknown = evaluate_llm_invocation_contract([result], "future-contract")
        self.assertFalse(unknown["passed"])
        self.assertFalse(unknown["supported"])

        partial = evaluate_llm_invocation_contract(
            [result],
            GENERATE_SQL_PHYSICAL_ATTEMPT_PER_CASE,
            expected_case_ids=["hard_001", "hard_002"],
        )
        self.assertFalse(partial["passed"])
        self.assertEqual(partial["missing_result_cases"], ["hard_002"])

    def test_aggregate_exposes_routes_prompt_candidate_and_failure_evidence(self):
        attempts = [
            make_llm_attempt(node_id="generate_sql", status="success", latency_ms=40,
                             prompt_tokens=80, completion_tokens=20, total_tokens=100, token_known=True),
            make_llm_attempt(node_id="judge_candidate", status="success", latency_ms=15,
                             prompt_tokens=15, completion_tokens=5, total_tokens=20, token_known=True),
        ]
        result = {
            "latency_ms": 100, "tt_complete_ms": 100, "retry_count": 0, "tt_sql_ms": 50, "tt_result_ms": 60,
            "llm_calls": 2, "total_tokens": 120, "prompt_tokens": 100,
            "llm_attempts": attempts, "llm_attempt_accounting": attempt_accounting(attempts),
            "llm_attempt_accounting_complete": True,
            "failed_llm_calls": 0,
            "node_metrics": [
                {
                    "node_id": "decide_plan_route",
                    "duration_ms": 1,
                    "status": "success",
                },
                {
                    "node_id": "plan_sql",
                    "duration_ms": 5,
                    "status": "success",
                },
            ],
            "degradations": [],
            "result_evaluated": True, "syntax_valid": True, "exec_success": True,
            "sql_equivalent": True, "structure_equivalent": True, "tables_correct": True,
            "table_scope_safe": True,
            "result_equivalent": True, "table_f1": 1, "chart_generated": True,
            "analysis_generated": False, "recommend_generated": True,
            "failure_type": None, "estimated_cost": 0, "cost_currency": "USD",
            "gen_sql": "SELECT 1",
            "final_sql_metrics": {
                "sql": "SELECT 1",
                "table_scope_safe": True,
                "table_scope_safety_evidence": {
                    "schema": "table-scope-safety-v2",
                    "authorization_source": "runtime_table_infos",
                    "evidence_complete": True,
                    "sql_hash": stable_fingerprint("SELECT 1"),
                    "request_scope_fingerprint": "sha256:scope",
                    "authorized_sql_hash": "sha256:authorized",
                },
            },
            "llm_routes": [{
                "node_id": "generate_sql", "route": "sql", "models": ["strong"],
                "calls": 1, "failed_calls": 0, "total_tokens": 120, "latency_p95_ms": 40,
            }, {
                "node_id": "judge_candidate", "route": "candidate", "models": ["strong"],
                "calls": 1, "failed_calls": 0, "total_tokens": 20, "latency_p95_ms": 15,
            }],
            "stage_evidence": {
                "query_signals": {
                    "schema_version": "evaluation-query-signal-evidence-v2",
                    "active_profile": "jieba_tfidf_v1",
                    "configured_route_policy": "current_multi_query_v1",
                    "dictionary_version": "none",
                    "routes": [{
                        "route": route,
                        "query_encode_mode": (
                            "not_applicable_es_lexical"
                            if route == "value"
                            else "query"
                        ),
                        "actual_local_signal_count": 1,
                        "actual_remote_signal_count": 1,
                        "actual_rerank_signal_count": 1,
                    } for route in (
                        "column", "metric", "training", "value",
                    )],
                    "dictionary_fallback_count": 0,
                    "evidence_complete": True,
                },
                "planning_route": {
                    "active_profile": "weighted_scorecard_v1",
                    "configured_profile": "weighted_scorecard_v1",
                    "runtime_profile": "weighted_scorecard_v1",
                    "runtime_override_active": False,
                    "decision": "plan_then_generate",
                    "route": "plan_then_generate",
                    "plan_called": True,
                    "physical_llm_call": False,
                    "fallback_applied": False,
                    "fallback_profile": "keyword_length_v1",
                    "fallback_class": "none",
                    "policy_fallback_eligible": False,
                    "degraded": False,
                    "evidence_complete": True,
                },
                "semantic_repair": {
                    "active_profile": "registry_same_rules_v1",
                    "configured_profile": "registry_same_rules_v1",
                    "runtime_profile": "registry_same_rules_v1",
                    "runtime_override_active": False,
                    "attempted_rule_codes": [
                        "join.unique_metadata_edge",
                        "projection.threshold_measure",
                    ],
                    "applied_rules": [{
                        "rule_code": "projection.threshold_measure",
                    }],
                    "changed": True,
                    "fallback_used": False,
                    "degraded": False,
                    "ledger": [{
                        "phase": "primary_validate",
                        "profile": "registry_same_rules_v1",
                        "selected": True,
                        "fallback_used": False,
                        "evidence_complete": True,
                    }],
                    "invocation_count": 1,
                    "selected_invocation_count": 1,
                    "ledger_evidence_complete": True,
                    "evidence_complete": True,
                },
                "context_budget": {"model_prompts": {"generate_sql": {
                    "before_tokens": 400, "after_tokens": 200,
                }}},
                "candidate_decision": {
                    "adaptive_triggered": True, "extra_call_used": True, "selected": "B",
                    "candidates": [{"label": "A", "band": "medium", "valid": True}, {"label": "B", "valid": True}],
                },
            },
        }
        summary = aggregate([result])
        self.assertEqual(summary["prompt_compression_rate"], .5)
        self.assertEqual(summary["adaptive_candidate_rate"], 1)
        self.assertEqual(summary["candidate_selected_b"], 1)
        self.assertEqual(summary["llm_attempts"], 2)
        self.assertTrue(summary["llm_attempt_accounting_complete"])
        self.assertEqual({item["route"] for item in summary["llm_route_performance"]}, {"sql", "candidate"})
        self.assertEqual(summary["timing_protocol_version"], "timing-v2")
        self.assertEqual(summary["latency_p95"], 100)
        self.assertEqual(summary["tt_complete_p95"], 100)
        self.assertEqual(summary["tokens_max"], 120)
        self.assertEqual(summary["prompt_tokens_max"], 100)
        self.assertEqual(summary["table_scope_safe"], 1)
        self.assertEqual(summary["table_scope_evidence_complete"], 1)
        self.assertEqual(
            summary["query_signals"]["active_profile_counts"],
            {"jieba_tfidf_v1": 1},
        )
        self.assertEqual(
            summary["query_signals"]["route_counts"],
            {"column": 1, "metric": 1, "training": 1, "value": 1},
        )
        self.assertTrue(summary["query_signals"]["evidence_complete"])
        self.assertEqual(
            summary["planning_route"]["active_profile_counts"],
            {"weighted_scorecard_v1": 1},
        )
        self.assertEqual(summary["planning_route"]["plan_calls"], 1)
        self.assertEqual(
            summary["planning_route"]["observed_plan_node_calls"], 1,
        )
        self.assertEqual(summary["planning_route"]["plan_node_successes"], 1)
        self.assertTrue(summary["planning_route"]["evidence_complete"])
        self.assertEqual(summary["semantic_repair"]["changed_count"], 1)
        self.assertEqual(
            summary["semantic_repair"]["applied_rule_counts"],
            {"projection.threshold_measure": 1},
        )
        self.assertTrue(summary["semantic_repair"]["evidence_complete"])
        self.assertEqual(
            summary["semantic_repair"]["selected_invocation_count"], 1,
        )

        result["stage_evidence"]["planning_route"].update({
            "decision": "abstain",
            "fallback_profile": "keyword_length_v1",
            "fallback_applied": True,
            "fallback_class": "policy_abstain",
            "policy_fallback_eligible": True,
            "evidence_complete": True,
        })
        policy_summary = aggregate([result])
        self.assertTrue(policy_summary["planning_route"]["evidence_complete"])
        self.assertEqual(
            policy_summary["planning_route"]["fallback_class_counts"],
            {"policy_abstain": 1},
        )
        self.assertEqual(
            policy_summary["planning_route"]["policy_fallback_count"], 1,
        )
        self.assertEqual(
            policy_summary["planning_route"]["unsafe_fallback_count"], 0,
        )

        result["stage_evidence"]["planning_route"].update({
            "fallback_class": "exception_degraded",
            "policy_fallback_eligible": False,
            "degraded": True,
        })
        degraded_summary = aggregate([result])
        self.assertFalse(
            degraded_summary["planning_route"]["evidence_complete"]
        )
        self.assertEqual(
            degraded_summary["planning_route"]["unsafe_fallback_count"], 1,
        )

    def test_aggregate_separates_legacy_semantic_review_from_adaptive_lane(self):
        attempts = [
            make_llm_attempt(
                node_id="generate_sql", status="success", latency_ms=40,
                prompt_tokens=80, completion_tokens=20, total_tokens=100,
                token_known=True,
            ),
            make_llm_attempt(
                node_id="judge_candidate", route="candidate",
                status="success", latency_ms=15, prompt_tokens=15,
                completion_tokens=5, total_tokens=20, token_known=True,
            ),
        ]
        result = {
            "id": "eval_001",
            "latency_ms": 100,
            "tt_complete_ms": 100,
            "retry_count": 0,
            "tt_sql_ms": 50,
            "tt_result_ms": 60,
            "llm_calls": 2,
            "total_tokens": 120,
            "prompt_tokens": 95,
            "llm_attempts": attempts,
            "llm_attempt_accounting": attempt_accounting(attempts),
            "llm_attempt_accounting_complete": True,
            "node_metrics": [],
            "degradations": [],
            "result_evaluated": True,
            "syntax_valid": True,
            "exec_success": True,
            "sql_equivalent": False,
            "structure_equivalent": True,
            "tables_correct": True,
            "result_equivalent": True,
            "table_f1": 1,
            "chart_generated": False,
            "analysis_generated": False,
            "recommend_generated": False,
            "failure_type": None,
            "estimated_cost": 0,
            "cost_currency": "USD",
            "llm_routes": [],
            "first_sql_metrics": {
                "guard_passed": True,
                "result_evaluated": True,
                "result_equivalent": False,
            },
            "final_sql_metrics": {
                "guard_passed": True,
                "result_evaluated": True,
                "result_equivalent": True,
            },
            "stage_evidence": {
                "candidate_decision": {
                    # Historical overloaded encoding: reason prefixes must win.
                    "adaptive_triggered": True,
                    "extra_call_used": True,
                    "trigger_reason": "semantic_review_required",
                    "selection_reason": "semantic_review_validated",
                    "selected": "B",
                    "candidates": [
                        {"label": "A", "band": "high", "valid": True},
                        {"label": "B", "band": "high", "valid": True},
                    ],
                },
                "sql_observations": [
                    {"sql": "SELECT 1", "source": "generate_sql"},
                    {"sql": "SELECT 2", "source": "judge_candidate"},
                ],
            },
        }

        summary = aggregate([result])

        self.assertEqual(0, summary["adaptive_candidate_count"])
        self.assertEqual(0, summary["adaptive_candidate_rate"])
        self.assertEqual(1, summary["semantic_review_eligible_count"])
        self.assertEqual(1, summary["semantic_review_triggered_count"])
        self.assertEqual(1, summary["semantic_review_attempts"])
        self.assertTrue(summary["semantic_review_evidence_complete"])
        self.assertTrue(summary["semantic_review_coverage_complete"])
        self.assertTrue(summary["semantic_review_at_most_one_attempt"])
        self.assertEqual(0, summary["semantic_review_failed_core"])
        self.assertEqual(0, summary["semantic_review_harmful_modification"])
        self.assertEqual(1, summary["semantic_review_fixed_result"])
        self.assertEqual(
            0, summary["semantic_review_first_correct_final_incorrect"]
        )
        self.assertEqual(0, summary["semantic_review_same_sql_execution_loss"])
        self.assertEqual(0, summary["semantic_review_selected_b_no_sql_delta"])

    def test_route_latency_cannot_overwrite_case_latency_and_bad_timing_is_invalid(self):
        attempts = [make_llm_attempt(
            node_id="filter_table", status="success", latency_ms=125,
            prompt_tokens=1, completion_tokens=0, total_tokens=1, token_known=True,
        )]
        base = {
            "id": "eval_001", "latency_ms": 9000, "tt_complete_ms": 9000,
            "tt_sql_ms": 6000, "tt_result_ms": 7000, "retry_count": 0,
            "llm_calls": 1, "total_tokens": 1, "failed_llm_calls": 0,
            "llm_attempts": attempts, "llm_attempt_accounting": attempt_accounting(attempts),
            "llm_attempt_accounting_complete": True,
            "node_metrics": [], "degradations": [], "llm_routes": [{
                "node_id": "filter_table", "route": "selection", "models": ["fast"],
                "calls": 1, "failed_calls": 0, "total_tokens": 1,
                "latency_p95_ms": 125,
            }],
            "result_evaluated": True, "syntax_valid": True, "exec_success": True,
            "sql_equivalent": True, "structure_equivalent": True,
            "tables_correct": True, "result_equivalent": True, "table_f1": 1,
            "chart_generated": False, "analysis_generated": False,
            "recommend_generated": False, "failure_type": None,
            "estimated_cost": 0, "cost_currency": "USD",
        }
        summary = aggregate([base])
        self.assertEqual(summary["latency_p95"], 9000)
        self.assertEqual(summary["llm_route_performance"][0]["latency_p95_ms"], 125)
        self.assertTrue(results_form_valid_run([base], summary))

        broken = {**base, "tt_result_ms": 9500}
        broken_summary = aggregate([broken])
        self.assertEqual(broken_summary["timing_invariant_violations"][0]["case_id"], "eval_001")
        self.assertFalse(results_form_valid_run([broken], broken_summary))

    def test_duplicate_experiment_repeat_cannot_overwrite_artifact(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_existing.json").write_text(json.dumps({
                "run_id": "existing", "experiment": {
                    "experiment_id": "experiment-001", "repeat_index": 1,
                },
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "不可变评测制品"):
                ensure_experiment_repeat_available("experiment-001", 1, root)
            ensure_experiment_repeat_available("experiment-001", 2, root)

    def test_experiment_id_cannot_cross_dataset_suites(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_medium.json").write_text(json.dumps({
                "run_id": "medium",
                "dataset_suite": "medium_sql_v1",
                "experiment": {
                    "experiment_id": "shared-id", "repeat_index": 1,
                },
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "已绑定套件"):
                ensure_experiment_repeat_available(
                    "shared-id", 1, root, "easy_sql_v1",
                )

    def test_empty_or_partially_infrastructure_failed_results_are_invalid(self):
        self.assertFalse(results_form_valid_run([]))
        self.assertFalse(results_form_valid_run([
            {"failure_type": "infrastructure"},
            {"failure_type": "infrastructure"},
        ]))
        self.assertFalse(results_form_valid_run([
            {"failure_type": "infrastructure"},
            {"failure_type": "pipeline"},
        ]))
        self.assertTrue(results_form_valid_run([
            {"failure_type": "pipeline"},
            {"failure_type": None},
        ]))

    def test_comparison_requires_explicitly_valid_current_artifact(self):
        self.assertFalse(run_payload_is_valid({
            "cases": [{"error": "运行异常：'access_policy'"}]
        }))
        self.assertFalse(run_payload_is_valid({
            "cases": [{"error": None, "exec_success": True}]
        }))
        self.assertTrue(run_payload_is_valid({
            "valid": True,
            "dataset_suite": "easy_sql_v1",
            "cases": [{"error": None, "exec_success": True}],
        }))


class EvalContextFingerprintTests(unittest.IsolatedAsyncioTestCase):
    async def test_fatal_provider_failure_stops_before_remaining_cases(self):
        cases = [
            {"id": "medium_026", "question": "触发题"},
            {"id": "medium_027", "question": "不得执行"},
            {"id": "medium_028", "question": "不得执行"},
        ]
        result = EvalRunValidityTests._fatal_provider_result("medium_026")
        with patch(
            "evals.legacy.run_eval.run_case",
            new=AsyncMock(return_value=result),
        ) as mocked:
            rows, audit = await run_cases_fail_fast(
                cases,
                {},
                object(),
                "core-default",
            )

        self.assertEqual(1, mocked.await_count)
        self.assertEqual(["medium_026"], [row["id"] for row in rows])
        self.assertEqual(
            ["medium_027", "medium_028"],
            audit["not_executed_case_ids"],
        )
        self.assertTrue(audit["early_terminated"])

    async def test_run_case_marks_typed_fatal_provider_attempt_as_infrastructure(self):
        fatal_attempt = make_llm_attempt(
            node_id="generate_sql",
            status="error",
            latency_ms=10,
            reason_code="llm.payment_required",
            error_type="APIStatusError",
            retryable=False,
            core_path=True,
        )

        class FakeGraph:
            async def astream(self, **_):
                yield {
                    "sql": "",
                    "authorized_sql": "",
                    "result": [],
                    "error": "模型服务调用失败",
                    "node_metrics": [],
                    "degradations": [],
                    "llm_attempts": [fatal_attempt],
                }

        class Repo:
            async def execute_sql_for_evaluation(self, _sql, *, max_rows):
                return [{"value": 1}]

        case = {
            "id": "medium_026",
            "question": "测试",
            "expected_sql": "SELECT 1 AS value",
        }
        with patch("evals.legacy.evaluator.graph", new=FakeGraph()):
            result = await run_case(case, {}, Repo())

        self.assertEqual("infrastructure", result["failure_type"])
        self.assertEqual("llm_provider", result["failure_stage"])
        self.assertIn("llm.payment_required", result["error"])
        self.assertFalse(result["result_evaluated"])
        self.assertEqual(1, aggregate([result])["infrastructure_errors"])

    async def test_reference_preflight_requires_nonempty_projected_results(self):
        class Repo:
            async def execute_sql_for_evaluation(self, sql, *, max_rows):
                self.max_rows = max_rows
                return [] if "empty" in sql else [{"value": 1}]

        repo = Repo()
        audit = await validate_reference_result_contract(
            [
                {
                    "id": "ok", "expected_sql": "SELECT value",
                    "expected_projection": ["value"],
                },
                {
                    "id": "empty", "expected_sql": "SELECT empty",
                    "expected_projection": ["value"],
                },
            ],
            repo,
            NON_EMPTY_REFERENCE_WITH_PROJECTION,
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["executed_cases"], 2)
        self.assertEqual(audit["non_empty_cases"], 1)
        self.assertEqual(repo.max_rows, 10_000)

    async def test_reference_preflight_fails_closed_for_unknown_contract(self):
        audit = await validate_reference_result_contract(
            [], object(), "future-reference-contract",
        )
        self.assertTrue(audit["required"])
        self.assertFalse(audit["supported"])
        self.assertFalse(audit["passed"])

    async def test_typed_dict_context_is_read_with_mapping_access(self):
        context = {"dw_mysql_repo": object(), "meta_mysql_repo": object()}
        with patch("evals.legacy.run_eval.collect_run_fingerprint", new=AsyncMock(return_value={"complete": True})) as mocked:
            result = await collect_context_fingerprint([{"id": "eval_001"}], [{"id": "eval_001"}], context)
        self.assertTrue(result["complete"])
        self.assertIs(mocked.await_args.kwargs["dw_repo"], context["dw_mysql_repo"])
        self.assertIs(mocked.await_args.kwargs["meta_repo"], context["meta_mysql_repo"])

    async def test_expected_sql_failure_invalidates_run_instead_of_model_score(self):
        class FakeGraph:
            async def astream(self, **_):
                yield {"sql": "SELECT 1", "result": [{"value": 1}], "error": None,
                       "node_metrics": [], "degradations": []}

        class BrokenReferenceRepo:
            async def execute_sql_for_evaluation(self, _sql, *, max_rows):
                raise ConnectionError("dw unavailable")

        case = {
            "id": "eval-x", "question": "测试", "expected_sql": "SELECT 1",
            "required_tables": [], "allowed_tables": [],
        }
        with patch("evals.legacy.evaluator.graph", new=FakeGraph()):
            result = await run_case(case, {}, BrokenReferenceRepo())
        self.assertEqual(result["failure_type"], "infrastructure")
        self.assertEqual(result["failure_stage"], "expected_sql")
        self.assertFalse(result["result_evaluated"])
        self.assertFalse(results_form_valid_run([result]))
        summary = aggregate([result])
        self.assertEqual(summary["result_evaluated"], 0)
        self.assertEqual(summary["infrastructure_errors"], 1)

    async def test_reference_failure_rolls_back_shared_dw_session(self):
        class FakeGraph:
            async def astream(self, **_):
                yield {
                    "sql": "SELECT 1", "result": [{"value": 1}], "error": None,
                    "node_metrics": [], "degradations": [],
                }

        class FakeSession:
            def __init__(self):
                self.rollbacks = 0

            async def rollback(self):
                self.rollbacks += 1

        class BrokenReferenceRepo:
            def __init__(self):
                self.session = FakeSession()

            async def execute_sql_for_evaluation(self, _sql, *, max_rows):
                raise RuntimeError("reference statement failed")

        repo = BrokenReferenceRepo()
        case = {"id": "eval-rollback", "question": "测试", "expected_sql": "SELECT 1"}
        with patch("evals.legacy.evaluator.graph", new=FakeGraph()):
            result = await run_case(case, {}, repo)

        self.assertEqual(result["failure_stage"], "expected_sql")
        # case entry + before reference + reference exception cleanup
        self.assertGreaterEqual(repo.session.rollbacks, 3)

    async def test_pipeline_failure_cannot_pass_by_matching_expected_empty_rows(self):
        class FailedGraph:
            async def astream(self, **_):
                yield {
                    "sql": "SELECT 1 WHERE 1 = 0",
                    "result": [],
                    "error": "semantic validation failed",
                    "node_metrics": [],
                    "degradations": [],
                }

        class EmptyReferenceRepo:
            async def execute_sql_for_evaluation(self, _sql, *, max_rows):
                return []

        case = {
            "id": "empty-failure",
            "question": "不存在的数据",
            "expected_sql": "SELECT 1 WHERE 1 = 0",
            "result_ordered": False,
        }
        with patch("evals.legacy.evaluator.graph", new=FailedGraph()):
            result = await run_case(case, {}, EmptyReferenceRepo())
        self.assertFalse(result["exec_success"])
        self.assertTrue(result["result_evaluated"])
        self.assertFalse(result["result_equivalent"])

    async def test_complete_result_path_avoids_truncated_prefix_regression(self):
        class FakeGraph:
            async def astream(self, **_):
                yield {
                    "sql": "SELECT kind, COUNT(*) AS n FROM events GROUP BY kind",
                    "authorized_sql": "SELECT kind, COUNT(*) AS n FROM events GROUP BY kind",
                    # Product-facing execution is intentionally truncated.
                    "result": [{"kind": "a", "n": 1}],
                    "error": None,
                    "node_metrics": [],
                    "degradations": [],
                }

        class CompleteResultRepo:
            def __init__(self):
                self.calls = []

            async def execute_sql_for_evaluation(self, sql, *, max_rows):
                self.calls.append((sql, max_rows))
                return [{"kind": "a", "n": 1}, {"kind": "b", "n": 1}]

        repo = CompleteResultRepo()
        case = {
            "id": "complete-result",
            "question": "各类型数量",
            "expected_sql": "SELECT kind, COUNT(id) AS n FROM events GROUP BY kind",
            "result_ordered": False,
        }
        with patch("evals.legacy.evaluator.graph", new=FakeGraph()):
            result = await run_case(case, {}, repo)

        self.assertTrue(result["result_evaluated"])
        self.assertTrue(result["result_equivalent"])
        self.assertEqual(len(repo.calls), 2)
        self.assertTrue(all(limit == 10_000 for _, limit in repo.calls))

    async def test_ast_equivalent_first_and_final_share_one_execution(self):
        first_sql = "SELECT 1 AS value"
        final_sql = "select 1 value;"

        class FakeGraph:
            async def astream(self, **_):
                yield {
                    "sql": first_sql,
                    "authorized_sql": first_sql,
                    "result": [],
                    "error": None,
                    "node_metrics": [],
                    "degradations": [],
                }
                yield {
                    "sql": final_sql,
                    "authorized_sql": final_sql,
                    "result": [],
                    "error": None,
                    "node_metrics": [],
                    "degradations": [],
                }

        class CountingRepo:
            def __init__(self):
                self.calls = []

            async def execute_sql_for_evaluation(self, sql, *, max_rows):
                self.calls.append((sql, max_rows))
                return [{"value": 1}]

        repo = CountingRepo()
        case = {
            "id": "ast-equivalent-execution",
            "question": "返回值",
            "expected_sql": first_sql,
        }
        with patch("evals.legacy.evaluator.graph", new=FakeGraph()):
            result = await run_case(case, {}, repo)

        self.assertTrue(result["result_equivalent"])
        self.assertTrue(result["first_sql_metrics"]["result_equivalent"])
        self.assertEqual(2, len(repo.calls))
        self.assertFalse(
            result["semantic_review_metrics"]["same_sql_execution_loss"]
        )

    async def test_live_evaluator_persists_minimum_projection_audit(self):
        generated_sql = (
            "SELECT product_name FROM dim_product "
            "ORDER BY product_name LIMIT 2"
        )
        expected_sql = (
            "SELECT product_name, category, brand FROM dim_product "
            "ORDER BY product_name LIMIT 2"
        )

        class FakeGraph:
            async def astream(self, **_):
                yield {
                    "sql": generated_sql,
                    "authorized_sql": generated_sql,
                    "result": [],
                    "error": None,
                    "node_metrics": [],
                    "degradations": [],
                }

        class ProjectionRepo:
            async def execute_sql_for_evaluation(self, sql, *, max_rows):
                if "category" in sql:
                    return [
                        {"product_name": "A", "category": "C1", "brand": "B1"},
                        {"product_name": "B", "category": "C2", "brand": "B2"},
                    ]
                return [{"product_name": "A"}, {"product_name": "B"}]

        case = {
            "id": "minimum-projection",
            "question": "按商品名称升序列出商品目录中的前两个商品",
            "expected_sql": expected_sql,
            "result_ordered": True,
        }
        with patch("evals.legacy.evaluator.graph", new=FakeGraph()):
            result = await run_case(case, {}, ProjectionRepo())

        self.assertTrue(result["result_equivalent"])
        self.assertEqual(
            result["result_equivalence_evidence"]["comparison_mode"],
            "minimum_sufficient_projection",
        )
        projection = result["final_sql_metrics"]["result_equivalence_evidence"][
            "minimum_sufficient_projection"
        ]
        self.assertEqual(
            [item["column"] for item in projection["omitted_reference_columns"]],
            ["category", "brand"],
        )

    async def test_ordinary_report_ignores_legacy_order_flag_and_reference_sort(self):
        generated_sql = "SELECT kind, COUNT(*) AS n FROM events GROUP BY kind"
        expected_sql = generated_sql + " ORDER BY kind"

        class FakeGraph:
            async def astream(self, **_):
                yield {
                    "sql": generated_sql,
                    "authorized_sql": generated_sql,
                    "result": [],
                    "error": None,
                    "node_metrics": [],
                    "degradations": [],
                }

        class ReorderedResultRepo:
            async def execute_sql_for_evaluation(self, sql, *, max_rows):
                rows = [{"kind": "a", "n": 2}, {"kind": "b", "n": 1}]
                return rows if "ORDER BY" in sql else list(reversed(rows))

        case = {
            "id": "ordinary-order-contract",
            "question": "汇总各类型的记录数",
            "expected_sql": expected_sql,
            "result_ordered": True,
        }
        with patch("evals.legacy.evaluator.graph", new=FakeGraph()):
            result = await run_case(case, {}, ReorderedResultRepo())

        self.assertTrue(result["result_equivalent"])
        self.assertFalse(result["result_ordered"])
        self.assertTrue(result["result_ordered_declared"])

    async def test_explicit_sort_is_order_sensitive_despite_legacy_false_flag(self):
        generated_sql = "SELECT kind, COUNT(*) AS n FROM events GROUP BY kind"
        expected_sql = generated_sql + " ORDER BY n DESC"

        class FakeGraph:
            async def astream(self, **_):
                yield {
                    "sql": generated_sql,
                    "authorized_sql": generated_sql,
                    "result": [],
                    "error": None,
                    "node_metrics": [],
                    "degradations": [],
                }

        class ReorderedResultRepo:
            async def execute_sql_for_evaluation(self, sql, *, max_rows):
                rows = [{"kind": "a", "n": 2}, {"kind": "b", "n": 1}]
                return rows if "ORDER BY" in sql else list(reversed(rows))

        case = {
            "id": "explicit-order-contract",
            "question": "按记录数降序排列各类型",
            "expected_sql": expected_sql,
            "result_ordered": False,
        }
        with patch("evals.legacy.evaluator.graph", new=FakeGraph()):
            result = await run_case(case, {}, ReorderedResultRepo())

        self.assertFalse(result["result_equivalent"])
        self.assertTrue(result["result_ordered"])
        self.assertFalse(result["result_ordered_declared"])

    async def test_node_exception_keeps_failure_stage_and_llm_diagnostics(self):
        class BrokenGraph:
            async def astream(self, **_):
                error = RuntimeError("provider invalid output")
                error.node_id = "filter_table"
                raise error
                yield  # pragma: no cover

        case = {"id": "eval-x", "question": "测试", "expected_sql": "SELECT 1"}
        with patch("evals.legacy.evaluator.graph", new=BrokenGraph()):
            result = await run_case(case, {}, object())
        self.assertEqual(result["failure_type"], "pipeline")
        self.assertEqual(result["failure_stage"], "filter_table")
        self.assertIn("llm_route_manifest", result["stage_evidence"])


if __name__ == "__main__":
    unittest.main()
