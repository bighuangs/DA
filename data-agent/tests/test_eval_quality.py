import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from evals.legacy.contracts import CheckResult, QualityReport
from evals.legacy.run_quality import (
    INSTRUMENTATION_SOURCE_PATHS,
    RETIRED_INSTRUMENTATION_PREFIXES,
    _experiment_runs,
    _domain_scores,
    _overall,
    _result_equivalence_target,
    _table_scope_safe_count,
    _thresholds,
    _persist_report,
    _prepare_full_live_run,
    _pytest_junit_evidence,
    _recommendations,
    _render_markdown,
    baseline_checks,
    check_golden_contract,
    check_hybrid_retrieval,
    check_instrumentation,
    check_security_cases,
    evaluate,
    experiment_checks,
    live_run_checks,
    resolve_experiment_live_run,
    resolve_quality_baseline,
)
from evals.legacy.run_scenarios import collect_playwright_specs


class EvalQualityTests(unittest.TestCase):
    def test_pytest_junit_evidence_preserves_exact_scenario_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pytest.xml"
            path.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<testsuites tests="2" failures="1">
  <testsuite name="pytest" tests="2" failures="1">
    <testcase classname="tests.test_example.ExampleTests" name="test_pass" time="0.012" />
    <testcase classname="tests.test_example.ExampleTests" name="test_fail" time="0.003">
      <failure message="assert false" />
    </testcase>
  </testsuite>
</testsuites>
""",
                encoding="utf-8",
            )

            evidence = _pytest_junit_evidence(path)

        self.assertEqual(evidence["tests"], 2)
        self.assertEqual(evidence["passed"], 1)
        self.assertEqual(evidence["failed"], 1)
        self.assertEqual(
            evidence["target_status"][
                "tests.test_example.ExampleTests.test_pass"
            ]["status"],
            "passed",
        )
        self.assertEqual(
            evidence["target_status"][
                "tests.test_example.ExampleTests.test_fail"
            ]["status"],
            "failed",
        )

    def test_full_profile_without_artifact_fails_closed_without_model_execution(self):
        timed_calls = 0

        def fake_timed(check):
            nonlocal timed_calls
            timed_calls += 1
            name = getattr(check, "__name__", "")
            check_id = {
                "check_unit_tests": "unit_tests",
                "check_frontend_e2e": "frontend_e2e",
            }.get(name, f"stub_{timed_calls}")
            return CheckResult(
                check_id,
                "governance",
                check_id,
                "pass",
                100,
                "stub",
                evidence={"target_status": {}},
            )

        with (
            patch("evals.legacy.run_quality._timed", side_effect=fake_timed),
            patch("evals.legacy.run_quality._run") as run_model_eval,
        ):
            report = evaluate(
                "full",
                dataset_suite="easy_sql_v1",
            )

        full_run = next(
            item for item in report.checks if item.id == "live_nl2sql_full_run"
        )
        self.assertEqual(full_run.status, "fail")
        self.assertEqual(
            full_run.evidence["execution_mode"],
            "missing_artifact_fail_closed",
        )
        self.assertFalse(full_run.evidence["model_execution_requested"])
        self.assertFalse(report.gate_passed)
        run_model_eval.assert_not_called()

    def test_full_profile_model_execution_requires_explicit_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "evals.legacy.run_quality._run",
                return_value=(2, "controlled failure"),
            ) as run_model_eval:
                result = _prepare_full_live_run(
                    None,
                    experiment_id=None,
                    dataset_suite="easy_sql_v1",
                    execute_live_run=True,
                    runs_root=root,
                )

        self.assertEqual(result[0], 2)
        self.assertEqual(result[4], "explicit_live_execution")
        run_model_eval.assert_called_once()
        self.assertIn("evals.legacy.run_eval", run_model_eval.call_args.args[0])

    def test_missing_explicit_live_run_path_never_falls_back_to_model(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json"
            with patch("evals.legacy.run_quality._run") as run_model_eval:
                result = _prepare_full_live_run(
                    missing,
                    experiment_id=None,
                    dataset_suite="easy_sql_v1",
                    execute_live_run=True,
                    runs_root=Path(directory),
                )

        self.assertEqual(result[0], 2)
        self.assertEqual(result[4], "invalid_explicit_artifact")
        run_model_eval.assert_not_called()

    def test_explicit_live_run_rejects_stale_local_version_without_model(self):
        profile = {
            "schema": "evaluation-profile-v1",
            "name": "core-default",
        }
        artifact_fingerprint = {
            "code_version": "sha256:artifact-code",
            "dependency_lock_hash": "sha256:shared-lock",
            "config_hash": "sha256:shared-config",
            "dataset_version": "sha256:shared-dataset",
            "evaluation_profile": profile,
        }
        current_fingerprint = {
            **artifact_fingerprint,
            "code_version": "sha256:current-code",
        }
        current_fingerprint.pop("evaluation_profile")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stale.json"
            path.write_text(
                json.dumps({
                    "valid": True,
                    "n": 100,
                    "evaluation_profile": profile,
                    "fingerprint": artifact_fingerprint,
                }),
                encoding="utf-8",
            )
            with (
                patch(
                    "evals.legacy.run_quality.load_suite_cases",
                    return_value=[{"id": "easy_001"}],
                ),
                patch(
                    "evals.legacy.run_quality.local_version_fingerprint",
                    return_value=current_fingerprint,
                ),
                patch("evals.legacy.run_quality._run") as run_model_eval,
            ):
                result = _prepare_full_live_run(
                    path,
                    experiment_id=None,
                    dataset_suite="easy_sql_v1",
                    execute_live_run=True,
                    runs_root=Path(directory),
                )

        self.assertEqual(result[0], 2)
        self.assertEqual(result[3], {})
        self.assertEqual(result[4], "artifact_version_mismatch")
        self.assertIn("code_version", result[1])
        self.assertNotIn("artifact-code", result[1])
        self.assertNotIn("current-code", result[1])
        run_model_eval.assert_not_called()

    def test_explicit_live_run_reuses_matching_local_version(self):
        profile = {
            "schema": "evaluation-profile-v1",
            "name": "core-default",
        }
        local_fingerprint = {
            "code_version": "sha256:code",
            "dependency_lock_hash": "sha256:lock",
            "config_hash": "sha256:config",
            "dataset_version": "sha256:dataset",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "current.json"
            path.write_text(
                json.dumps({
                    "valid": True,
                    "n": 100,
                    "evaluation_profile": profile,
                    "fingerprint": {
                        **local_fingerprint,
                        "evaluation_profile": profile,
                    },
                }),
                encoding="utf-8",
            )
            with (
                patch(
                    "evals.legacy.run_quality.load_suite_cases",
                    return_value=[{"id": "easy_001"}],
                ),
                patch(
                    "evals.legacy.run_quality.local_version_fingerprint",
                    return_value=local_fingerprint,
                ),
                patch("evals.legacy.run_quality._run") as run_model_eval,
            ):
                result = _prepare_full_live_run(
                    path,
                    experiment_id=None,
                    dataset_suite="easy_sql_v1",
                    execute_live_run=False,
                    runs_root=Path(directory),
                )

        self.assertEqual(result[0], 0)
        self.assertEqual(result[4], "explicit_artifact_reuse")
        self.assertEqual(result[3]["n"], 100)
        run_model_eval.assert_not_called()

    def test_cli_wires_live_execution_as_opt_in(self):
        report = QualityReport(
            run_id="cli",
            profile="full",
            dataset_suite="easy_sql_v1",
            generated_at="2026-08-15T00:00:00+08:00",
            environment={},
            gate_passed=True,
            overall_score=100,
            domain_scores={},
            checks=[],
            recommendations=[],
        )
        for extra_args, expected in (([], False), (["--execute-live-run"], True)):
            with self.subTest(extra_args=extra_args):
                with (
                    patch.object(
                        sys,
                        "argv",
                        ["run_quality", "--profile", "full", *extra_args],
                    ),
                    patch(
                        "evals.legacy.run_quality.resolve_quality_baseline",
                        return_value={"available": False},
                    ),
                    patch(
                        "evals.legacy.run_quality.evaluate",
                        return_value=report,
                    ) as evaluate_mock,
                    patch("evals.legacy.run_quality._persist_report"),
                    patch(
                        "evals.legacy.run_quality._render_markdown",
                        return_value="report",
                    ),
                    patch("builtins.print"),
                ):
                    from evals.legacy.run_quality import main

                    self.assertEqual(main(), 0)

                self.assertEqual(
                    evaluate_mock.call_args.kwargs["execute_live_run"],
                    expected,
                )

    def test_instrumentation_gate_tracks_only_current_architecture_paths(self):
        expected_paths = {
            "app/application/services/query_trace_service.py",
            "app/infrastructure/persistence/mysql/repositories/query_trace_repository.py",
            "app/infrastructure/persistence/mysql/repositories/feedback_repository.py",
            "evals/application/services/quality_operations_service.py",
            "evals/adapters/http/router.py",
            "evals/legacy/scenario_matrix.py",
            "evals/legacy/run_scenarios.py",
            "evals/legacy/resilience_suite.py",
            "evals/legacy/run_eval.py",
            "evals/legacy/compare_runs.py",
            "app/agent_system/agents/nl2sql/services/planning_router.py",
            "evals/legacy/planning_router_ablation.py",
            "evals/application/services/planning_evaluation_service.py",
            "app/domain/query/safety/semantic_violation.py",
            "app/agent_system/agents/nl2sql/services/semantic_repair.py",
            "evals/legacy/semantic_repair_ablation.py",
            "evals/application/services/semantic_repair_evaluation_service.py",
            "evals/legacy/adhoc_isolation_suite.py",
            "evals/datasets/legacy/adhoc_isolation_suite.yaml",
        }

        result = check_instrumentation()

        self.assertEqual(result.status, "pass", result.evidence)
        self.assertEqual(result.evidence["missing"], [])
        self.assertEqual(result.evidence["retired"], [])
        self.assertEqual(set(INSTRUMENTATION_SOURCE_PATHS), expected_paths)
        self.assertEqual(len(INSTRUMENTATION_SOURCE_PATHS), len(expected_paths))
        self.assertFalse(
            any(
                path.startswith(RETIRED_INSTRUMENTATION_PREFIXES)
                for path in INSTRUMENTATION_SOURCE_PATHS
            )
        )

    def test_legacy_exact_tables_are_not_a_blocking_scope_proxy(self):
        self.assertEqual(
            _table_scope_safe_count(
                {
                    "aggregate": {
                        "n": 50,
                        "tables_correct": 50,
                        "table_scope_safe": 50,
                    }
                },
                50,
            ),
            0,
        )

    def test_result_correctness_targets_are_suite_specific(self):
        thresholds = _thresholds()
        self.assertEqual(
            _result_equivalence_target(thresholds, "easy_sql_v1"), 0.96,
        )
        self.assertEqual(
            _result_equivalence_target(thresholds, "medium_sql_v1"), 0.86,
        )
        self.assertEqual(
            _result_equivalence_target(thresholds, "hard_sql_v1"), 0.0,
        )

    def test_playwright_json_evidence_is_independent_of_console_reporter_format(self):
        specs = {}
        collect_playwright_specs([{"suites": [], "specs": [{
            "title": "owner can inspect quality trends",
            "tests": [{"results": [{"status": "passed", "duration": 12}]}],
        }]}], specs)
        self.assertEqual(specs["owner can inspect quality trends"]["status"], "passed")
        self.assertEqual(specs["owner can inspect quality trends"]["duration_ms"], 12)

    def test_security_contract_dataset_passes(self):
        result = check_security_cases()
        self.assertEqual(result.status, "pass")
        self.assertEqual(result.score, 100)

    def test_easy_sql_contract_is_a_one_hundred_case_current_gate(self):
        result = check_golden_contract("easy_sql_v1")
        self.assertEqual(result.status, "pass", result.evidence)
        self.assertEqual(result.evidence["suite"], "easy_sql_v1")
        self.assertEqual(result.evidence["cases"], 100)
        self.assertEqual(result.evidence["expected_cases"], 100)
        self.assertGreaterEqual(len(result.evidence["categories"]), 11)
        self.assertEqual(result.evidence["promotion_status"], "approved")

    def test_hybrid_live_check_rejects_stale_query_fingerprint(self):
        from app.agent_system.governance.retrieval_budget import RERANK_VERSION
        from app.infrastructure.search.index_version import retrieval_runtime_fingerprint

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hybrid.json"
            required = {
                "full_contract": True,
                "index_ready": True,
                "dense_evidence_nonempty": True,
                "sparse_evidence_nonempty": True,
                "hybrid_evidence_nonempty": True,
                "base_recall_non_regression": True,
                "column_precision_floor": True,
                "identifier_recall_non_regression": True,
                "identifier_mrr_non_regression": True,
                "tenant_isolation": True,
                "no_active_degradation": True,
            }
            payload = {
                "available": True,
                "summary": {"gate_passed": True, "checks": required},
                "profiles": {"R-H3": {"summary": {
                    "cases": 28,
                    "passed_cases": 28,
                    "routes": {"column": {"recall_at_k": 1.0}},
                    "base_column_recall_at_k": 1.0,
                    "base_column_precision_at_k": .64,
                    "identifier_subset": {"mrr": 1.0},
                }}},
                "query_contract": {
                    "fingerprint": retrieval_runtime_fingerprint(RERANK_VERSION),
                },
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(check_hybrid_retrieval(path).status, "pass")
            payload["query_contract"]["fingerprint"] = "sha256:stale"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(check_hybrid_retrieval(path).status, "fail")

    def test_domain_scores_ignore_skipped_checks(self):
        checks = [
            CheckResult("a", "correctness", "A", "pass", 80, ""),
            CheckResult("b", "correctness", "B", "skip", None, ""),
            CheckResult("c", "safety", "C", "pass", 100, ""),
        ]
        scores = _domain_scores(checks)
        self.assertEqual(scores["correctness"], 80)
        self.assertEqual(scores["safety"], 100)

    def test_overall_renormalizes_available_domains(self):
        scores = {"correctness": 80, "safety": 100, "reliability": None, "efficiency": None, "experience": None, "governance": None}
        self.assertAlmostEqual(_overall(scores), 84.0)

    def test_final_result_has_majority_weight_inside_correctness(self):
        checks = [
            CheckResult("live_nl2sql_results", "correctness", "result", "fail", 50, ""),
            CheckResult("live_nl2sql_syntax", "correctness", "syntax", "pass", 100, ""),
            CheckResult("live_nl2sql_tables", "correctness", "tables", "pass", 100, ""),
        ]

        self.assertEqual(_domain_scores(checks)["correctness"], 65.0)

    def test_recommendations_follow_measured_failures(self):
        checks = [CheckResult("nl2sql_results", "correctness", "result", "fail", 50, "")]
        titles = [item["title"] for item in _recommendations(checks)]
        self.assertTrue(any("召回" in title for title in titles))

    def test_experiment_latency_failure_always_produces_measured_recommendation(self):
        checks = [CheckResult(
            "nl2sql_experiment_latency", "efficiency", "二轮最差值延迟",
            "fail", 0, "TTSQL 15005.0 ms",
        )]
        recommendations = _recommendations(checks)
        self.assertTrue(recommendations)
        self.assertIn("15005.0", recommendations[0]["evidence"])
        self.assertIn("发布策略", recommendations[0]["title"])
        self.assertNotIn("继续压缩", recommendations[0]["title"])

    def test_experiment_binding_reuses_highest_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for repeat in (1, 2, 3):
                (root / f"run_{repeat}.json").write_text(json.dumps({
                    "run_id": f"run-{repeat}", "valid": True,
                    "dataset_suite": "easy_sql_v1",
                    "experiment": {"experiment_id": "exp-123", "repeat_index": repeat},
                }), encoding="utf-8")
            selected = resolve_experiment_live_run(
                "exp-123", "easy_sql_v1", root,
            )
            self.assertEqual(selected.name, "run_3.json")

    def test_experiment_binding_and_aggregation_never_cross_suite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for suite, repeat in (
                ("easy_sql_v1", 1),
                ("easy_sql_v1", 2),
                ("medium_sql_v1", 3),
            ):
                path = root / f"run_{suite}_{repeat}.json"
                path.write_text(json.dumps({
                    "run_id": f"{suite}-{repeat}",
                    "valid": True,
                    "dataset_suite": suite,
                    "experiment": {
                        "experiment_id": "shared-id",
                        "repeat_index": repeat,
                    },
                }), encoding="utf-8")

            selected = resolve_experiment_live_run(
                "shared-id", "easy_sql_v1", root,
            )
            self.assertEqual(selected.name, "run_easy_sql_v1_2.json")
            matches = _experiment_runs(
                "shared-id", "easy_sql_v1", root,
            )
            self.assertEqual(
                [payload["run_id"] for _, payload in matches],
                ["easy_sql_v1-1", "easy_sql_v1-2"],
            )

    def test_live_run_checks_use_current_aggregate(self):
        data = {
            "run_id": "run-1",
            "dataset_suite": "easy_sql_v1",
            "aggregate": {
                "n": 100, "exec_success": 100, "syntax_valid": 100,
                "tables_correct": 99, "result_equivalent": 96,
                "table_scope_safe": 100,
                "table_scope_evidence_complete": 100,
                "security_guard_bypass_count": 0,
                "latency_p95": 25000, "failed_llm_calls": 0,
                "failed_core_llm_calls": 0, "failed_optional_llm_calls": 0,
                "llm_attempt_schema": "llm-attempt-v2", "llm_attempts": 40,
                "llm_attempts_succeeded": 40, "llm_attempts_failed": 0,
                "llm_attempts_timed_out": 0, "llm_attempts_cancelled": 0,
                "llm_outcome_classes": {"success": 40},
                "request_deadline_contract_complete": True,
                "request_deadline_exhausted_cases": 0,
                "llm_attempt_accounting_complete": True,
                "degradation_evidence_complete": True,
                "structured_output_fallbacks": 0,
                "tokens_max": 12000, "tokens_max_case_id": "eval_010",
            },
            "evaluation_profile": {
                "schema": "evaluation-profile-v1",
                "name": "core-default",
                "analysis_enabled": False,
            },
            "by_category": [],
        }
        results = {item.id: item for item in live_run_checks(data)}
        self.assertEqual(results["live_nl2sql_coverage"].status, "pass")
        self.assertEqual(results["live_nl2sql_tables"].status, "observe")
        self.assertEqual(
            results["live_nl2sql_tables"].enforcement, "observational"
        )
        self.assertEqual(
            results["live_nl2sql_table_scope_safety"].status, "pass"
        )
        self.assertEqual(
            results["live_nl2sql_security_guard_integrity"].status, "pass"
        )
        self.assertEqual(results["live_nl2sql_results"].status, "pass")
        self.assertEqual(results["live_nl2sql_latency"].status, "pass")
        self.assertEqual(results["live_nl2sql_model_reliability"].status, "pass")
        self.assertEqual(results["live_nl2sql_token_max"].status, "pass")

    def test_live_table_scope_safety_blocks_without_promoting_exact_match(self):
        data = {
            "run_id": "scope-unsafe",
            "dataset_suite": "easy_sql_v1",
            "aggregate": {
                "n": 50,
                "result_evaluated": 50,
                "exec_success": 50,
                "syntax_valid": 50,
                "tables_correct": 50,
                "table_scope_safe": 49,
                "table_scope_evidence_complete": 50,
                "security_guard_bypass_count": 0,
                "result_equivalent": 50,
                "latency_p95": 1000,
                "failed_core_llm_calls": 0,
                "failed_optional_llm_calls": 0,
                "llm_attempt_schema": "llm-attempt-v2",
                "llm_attempts": 0,
                "llm_outcome_classes": {},
                "llm_attempt_accounting_complete": True,
                "request_deadline_contract_complete": True,
                "request_deadline_exhausted_cases": 0,
                "degradation_evidence_complete": True,
                "structured_output_fallbacks": 0,
                "tokens_max": 0,
            },
            "evaluation_profile": {
                "schema": "evaluation-profile-v1",
                "name": "core-default",
                "analysis_enabled": False,
            },
            "cases": [],
            "by_category": [],
        }

        checks = {item.id: item for item in live_run_checks(data)}

        self.assertEqual(checks["live_nl2sql_tables"].status, "observe")
        self.assertEqual(
            checks["live_nl2sql_table_scope_safety"].status, "fail"
        )
        self.assertEqual(
            checks["live_nl2sql_table_scope_safety"].enforcement, "blocking"
        )

        data["aggregate"]["table_scope_safe"] = 50
        data["aggregate"]["security_guard_bypass_count"] = 1
        guard_checks = {item.id: item for item in live_run_checks(data)}
        self.assertEqual(
            guard_checks["live_nl2sql_security_guard_integrity"].status,
            "fail",
        )
        self.assertEqual(
            guard_checks["live_nl2sql_security_guard_integrity"].enforcement,
            "blocking",
        )

    def test_quality_contract_serializes_blocking_and_observational_accounts(self):
        report = QualityReport(
            run_id="accounting",
            profile="full",
            dataset_suite="easy_sql_v1",
            generated_at="2026-08-08T00:00:00+08:00",
            environment={},
            gate_passed=True,
            overall_score=80,
            domain_scores={"correctness": 80},
            checks=[
                CheckResult(
                    "scope", "safety", "scope", "pass", 100, "safe",
                ),
                CheckResult(
                    "exact", "correctness", "exact", "observe", 60,
                    "diagnostic", enforcement="observational",
                ),
            ],
            recommendations=[],
        )

        accounting = report.to_dict()["check_accounting"]

        self.assertEqual(accounting["blocking"]["ids"], ["scope"])
        self.assertEqual(accounting["observational"]["ids"], ["exact"])
        self.assertEqual(accounting["blocking"]["failed"], 0)

    def test_hard_result_is_observed_at_actual_score_without_blocking(self):
        data = {
            "run_id": "hard-observation",
            "dataset_suite": "hard_sql_v1",
            "aggregate": {
                "n": 50, "result_evaluated": 50,
                "exec_success": 20, "syntax_valid": 45,
                "tables_correct": 25, "result_equivalent": 15,
                "table_scope_safe": 50,
                "table_scope_evidence_complete": 50,
                "security_guard_bypass_count": 0,
                "latency_p95": 8000, "failed_llm_calls": 0,
                "failed_core_llm_calls": 0, "failed_optional_llm_calls": 0,
                "llm_attempt_schema": "llm-attempt-v2", "llm_attempts": 0,
                "llm_attempts_succeeded": 0, "llm_attempts_failed": 0,
                "llm_attempts_timed_out": 0, "llm_attempts_cancelled": 0,
                "llm_outcome_classes": {},
                "request_deadline_contract_complete": True,
                "request_deadline_exhausted_cases": 0,
                "llm_attempt_accounting_complete": True,
                "degradation_evidence_complete": True,
                "structured_output_fallbacks": 0,
                "tokens_max": 0, "tokens_max_case_id": "",
            },
            "evaluation_profile": {
                "schema": "evaluation-profile-v1",
                "name": "core-default",
                "analysis_enabled": False,
            },
            "by_category": [],
            "cases": [],
        }

        result = {
            item.id: item for item in live_run_checks(data)
        }["live_nl2sql_results"]

        assert result.status == "observe"
        assert result.score == 30.0
        assert result.evidence["target"] == 0.0
        assert "不设强制正确率门槛" in result.summary

    def test_hard_live_observation_renders_and_scores_without_blocking(self):
        data = {
            "run_id": "hard-render-observation",
            "dataset_suite": "hard_sql_v1",
            "aggregate": {
                "n": 50,
                "result_evaluated": 50,
                "exec_success": 50,
                "syntax_valid": 50,
                "tables_correct": 50,
                "table_scope_safe": 50,
                "table_scope_evidence_complete": 50,
                "security_guard_bypass_count": 0,
                "result_equivalent": 15,
                "latency_p95": 8000,
                "failed_llm_calls": 0,
                "failed_core_llm_calls": 0,
                "failed_optional_llm_calls": 0,
                "llm_attempt_schema": "llm-attempt-v2",
                "llm_attempts": 0,
                "llm_attempts_succeeded": 0,
                "llm_attempts_failed": 0,
                "llm_attempts_timed_out": 0,
                "llm_attempts_cancelled": 0,
                "llm_outcome_classes": {},
                "request_deadline_contract_complete": True,
                "request_deadline_exhausted_cases": 0,
                "llm_attempt_accounting_complete": True,
                "degradation_evidence_complete": True,
                "structured_output_fallbacks": 0,
                "tokens_max": 0,
                "tokens_max_case_id": "",
            },
            "evaluation_profile": {
                "schema": "evaluation-profile-v1",
                "name": "core-default",
                "analysis_enabled": False,
            },
            "by_category": [],
            "cases": [],
        }
        observed = {
            item.id: item for item in live_run_checks(data)
        }["live_nl2sql_results"]
        supporting = CheckResult(
            "live_nl2sql_syntax",
            "correctness",
            "syntax",
            "pass",
            100,
            "supporting check",
        )
        scores = _domain_scores([observed, supporting])

        self.assertEqual(observed.status, "observe")
        self.assertEqual(observed.score, 30.0)
        self.assertEqual(scores["correctness"], 51.0)
        self.assertFalse(any(
            item.status == "fail" for item in (observed, supporting)
        ))

        report = QualityReport(
            run_id=data["run_id"],
            profile="full",
            dataset_suite="hard_sql_v1",
            generated_at="2026-08-08T00:00:00+08:00",
            environment={},
            gate_passed=True,
            overall_score=51.0,
            domain_scores=scores,
            checks=[observed, supporting],
            recommendations=[],
        )
        markdown = _render_markdown(report)

        self.assertIn("👁️ observe", markdown)
        self.assertIn("30.0%", markdown)
        self.assertIn("质量门禁：**通过**", markdown)

    def test_live_run_token_gate_accepts_deterministic_zero_llm_run(self):
        data = {
            "run_id": "run-deterministic",
            "dataset_suite": "easy_sql_v1",
            "aggregate": {
                "n": 50, "exec_success": 50, "syntax_valid": 50,
                "tables_correct": 50, "result_equivalent": 50,
                "latency_p95": 8000, "failed_llm_calls": 0,
                "failed_core_llm_calls": 0, "failed_optional_llm_calls": 0,
                "llm_attempt_schema": "llm-attempt-v2", "llm_attempts": 0,
                "llm_attempts_succeeded": 0, "llm_attempts_failed": 0,
                "llm_attempts_timed_out": 0, "llm_attempts_cancelled": 0,
                "llm_outcome_classes": {},
                "request_deadline_contract_complete": True,
                "request_deadline_exhausted_cases": 0,
                "llm_attempt_accounting_complete": True,
                "degradation_evidence_complete": True,
                "structured_output_fallbacks": 0,
                "tokens_max": 0, "tokens_max_case_id": "",
            },
            "evaluation_profile": {
                "schema": "evaluation-profile-v1",
                "name": "core-default",
                "analysis_enabled": False,
            },
            "by_category": [],
            "cases": [{"id": "eval_001", "total_tokens": 0}],
        }
        results = {item.id: item for item in live_run_checks(data)}
        token_check = results["live_nl2sql_token_max"]
        self.assertEqual(token_check.status, "pass")
        self.assertIn("确定性路径", token_check.summary)
        self.assertEqual(token_check.evidence["llm_attempts"], 0)

    def test_easy_sql_live_run_coverage_uses_suite_manifest(self):
        data = {
            "run_id": "easy-sql-run",
            "dataset_suite": "easy_sql_v1",
            "aggregate": {
                "n": 100, "exec_success": 100, "syntax_valid": 100,
                "tables_correct": 100, "result_equivalent": 100,
                "latency_p95": 8000, "failed_llm_calls": 0,
                "failed_core_llm_calls": 0, "failed_optional_llm_calls": 0,
                "llm_attempt_schema": "llm-attempt-v2", "llm_attempts": 0,
                "llm_attempts_succeeded": 0, "llm_attempts_failed": 0,
                "llm_attempts_timed_out": 0, "llm_attempts_cancelled": 0,
                "llm_outcome_classes": {},
                "request_deadline_contract_complete": True,
                "request_deadline_exhausted_cases": 0,
                "llm_attempt_accounting_complete": True,
                "degradation_evidence_complete": True,
                "structured_output_fallbacks": 0,
                "tokens_max": 0, "tokens_max_case_id": "",
            },
            "evaluation_profile": {
                "schema": "evaluation-profile-v1",
                "name": "core-default",
                "analysis_enabled": False,
            },
            "by_category": [],
            "cases": [],
        }
        results = {
            item.id: item
            for item in live_run_checks(data)
        }
        coverage = results["live_nl2sql_coverage"]
        self.assertEqual(coverage.status, "pass")
        self.assertEqual(coverage.evidence["expected_cases"], 100)
        self.assertIn("100/100", coverage.summary)

    def test_live_run_rejects_failed_execution_empty_result_false_positive(self):
        data = {
            "run_id": "contract-false-positive",
            "dataset_suite": "easy_sql_v1",
            "aggregate": {
                "n": 2, "result_evaluated": 2,
                "exec_success": 1, "syntax_valid": 2,
                "tables_correct": 2, "result_equivalent": 2,
                "latency_p95": 1000, "failed_core_llm_calls": 0,
                "failed_optional_llm_calls": 0,
                "llm_attempt_schema": "llm-attempt-v2",
                "llm_attempts": 0, "llm_attempts_succeeded": 0,
                "llm_attempts_failed": 0, "llm_attempts_timed_out": 0,
                "llm_attempts_cancelled": 0,
                "llm_outcome_classes": {},
                "request_deadline_contract_complete": True,
                "request_deadline_exhausted_cases": 0,
                "llm_attempt_accounting_complete": True,
                "degradation_evidence_complete": True,
                "structured_output_fallbacks": 0,
                "tokens_max": 0,
            },
            "evaluation_profile": {
                "schema": "evaluation-profile-v1",
                "name": "core-default",
                "analysis_enabled": False,
            },
            "by_category": [],
            "cases": [
                {
                    "id": "easy_ok",
                    "exec_success": True,
                    "result_equivalent": True,
                },
                {
                    "id": "easy_false_positive",
                    "exec_success": False,
                    "result_equivalent": True,
                },
            ],
        }
        results = {item.id: item for item in live_run_checks(data)}
        self.assertEqual(
            results["live_nl2sql_results"].evidence["value"],
            0.5,
        )
        integrity = results["live_nl2sql_result_integrity"]
        self.assertEqual(integrity.status, "fail")
        self.assertEqual(
            integrity.evidence["invalid_case_ids"],
            ["easy_false_positive"],
        )
        self.assertEqual(integrity.evidence["effective_result_equivalent"], 1)

    def test_baseline_checks_use_current_meta_approved_run(self):
        selection = {
            "available": True, "source": "meta_mysql", "status": "approved",
            "schema": "evaluation-v2", "path": "run.json",
            "run_id": "approved-run",
            "payload": {
                "run_id": "approved-run", "workspace_id": "workspace-a",
                "valid": True, "dataset_suite": "easy_sql_v1", "n": 50,
                "evaluation_profile": {
                    "schema": "evaluation-profile-v1",
                    "name": "core-default",
                    "analysis_enabled": False,
                },
                "fingerprint": {"fingerprint_schema": "evaluation-v2"},
                "aggregate": {
                    "n": 50, "result_evaluated": 50,
                    "exec_success": 50, "syntax_valid": 50,
                    "tables_correct": 50, "result_equivalent": 49,
                    "table_scope_safe": 50,
                    "table_scope_evidence_complete": 50,
                    "security_guard_bypass_count": 0,
                    "latency_p95": 41569, "tt_sql_p95": 29389,
                    "tt_result_p95": 29397,
                },
                "cases": [
                    {
                        "id": f"eval_{index:03d}",
                        "exec_success": True,
                        "result_equivalent": index != 27,
                    }
                    for index in range(1, 51)
                ],
            },
        }
        results = {item.id: item for item in baseline_checks(selection)}
        self.assertEqual(results["nl2sql_tables"].status, "observe")
        self.assertEqual(results["nl2sql_tables"].enforcement, "observational")
        self.assertEqual(results["nl2sql_table_scope_safety"].status, "pass")
        self.assertEqual(
            results["nl2sql_security_guard_integrity"].status, "pass"
        )
        self.assertEqual(results["nl2sql_results"].status, "pass")
        self.assertEqual(results["nl2sql_latency"].status, "fail")
        self.assertEqual(results["nl2sql_baseline_provenance"].status, "pass")
        self.assertEqual(results["nl2sql_results"].evidence["run_id"], "approved-run")

    def test_quality_baseline_resolver_passes_workspace_and_suite_to_meta(self):
        approved = {
            "workspace_id": "workspace-a",
            "run_id": "easy-r1",
            "dataset_suite": "easy_sql_v1",
            "artifact": "run_easy-r1.json",
            "status": "approved",
        }
        resolved = {
            "available": True,
            "source": "meta_mysql",
            "run_id": "easy-r1",
        }
        with (
            patch(
                "evals.legacy.run_quality._meta_approved_entry",
                new=AsyncMock(return_value=approved),
            ) as meta_approved,
            patch(
                "evals.legacy.run_quality.resolve_baseline_artifact",
                return_value=resolved,
            ) as resolver,
        ):
            selection = resolve_quality_baseline(
                workspace_id="workspace-a", dataset_suite="easy_sql_v1",
            )

        self.assertEqual(selection, resolved)
        meta_approved.assert_awaited_once_with("workspace-a", "easy_sql_v1")
        resolver.assert_called_once_with(
            workspace_id="workspace-a",
            approved_entry=approved,
            expected_dataset_suite="easy_sql_v1",
        )

    def test_experiment_checks_fail_release_when_protocol_is_complete_but_latency_is_not(self):
        report = {
            "available": True, "experiment_id": "exp", "status": "blocked",
            "complete": True, "repeat_count": 3, "required_repeats": 3,
            "checks": {
                "baseline_non_regression": True, "table_exact_non_regression": True,
                "syntax_floor_gate": True, "execution_floor_gate": True,
                "result_floor_non_regression": True, "ttsql_gate": False,
                "ttresult_gate": False, "ttcomplete_gate": False,
                "failed_llm_calls_zero": True, "structured_output_clean": True,
                "token_max_gate": True, "candidate_trigger_gate": True,
                "candidate_safety_gate": True, "token_non_regression": True,
                "llm_attempt_accounting_complete": True,
                "degradation_evidence_complete": True,
                "optional_postprocess_clean": True,
            },
            "failed_checks": ["ttsql_gate", "ttresult_gate", "ttcomplete_gate"],
            "worst": {"tables_correct": 30, "result_equivalent": 29, "tt_sql_p95": 29000, "tt_result_p95": 30000, "tt_complete_p95": 42000, "tokens_max": 12000, "failed_llm_calls": 0},
            "regressed_case_ids": [], "stability": [], "gate_limits": {},
        }
        results = {item.id: item for item in experiment_checks(report)}
        self.assertEqual(results["nl2sql_experiment_protocol"].status, "pass")
        self.assertEqual(results["nl2sql_experiment_non_regression"].status, "pass")
        self.assertEqual(results["nl2sql_experiment_latency"].status, "fail")
        self.assertEqual(results["nl2sql_experiment_model_reliability"].status, "pass")
        self.assertEqual(
            results["nl2sql_experiment_table_exact_observation"].status,
            "observe",
        )
        self.assertEqual(
            results["nl2sql_experiment_table_exact_observation"].enforcement,
            "observational",
        )

    def test_report_conclusion_does_not_claim_frontend_evidence_is_missing_after_e2e_passes(self):
        report = QualityReport(
            run_id="run-1",
            profile="offline",
            dataset_suite="easy_sql_v1",
            generated_at="2026-07-15T00:00:00+08:00",
            environment={},
            gate_passed=False,
            overall_score=90,
            domain_scores={"experience": 100},
            checks=[
                CheckResult("frontend_e2e", "experience", "e2e", "pass", 100, "20/20"),
                CheckResult("nl2sql_results", "correctness", "results", "fail", 80, "below target"),
            ],
            recommendations=[],
        )
        markdown = _render_markdown(report)
        self.assertIn("前端浏览器自动化证据已通过", markdown)
        self.assertNotIn("浏览器自动化证据不足", markdown)

    def test_report_conclusion_matches_a_passing_gate(self):
        report = QualityReport(
            run_id="run-pass",
            profile="offline",
            dataset_suite="easy_sql_v1",
            generated_at="2026-07-29T00:00:00+08:00",
            environment={},
            gate_passed=True,
            overall_score=100,
            domain_scores={"experience": 100},
            checks=[
                CheckResult(
                    "frontend_e2e",
                    "experience",
                    "e2e",
                    "pass",
                    100,
                    "57/57",
                ),
                CheckResult(
                    "nl2sql_results",
                    "correctness",
                    "results",
                    "pass",
                    100,
                    "target met",
                ),
            ],
            recommendations=[],
        )
        markdown = _render_markdown(report)
        self.assertIn("所有当前已评测门禁通过", markdown)
        self.assertNotIn("仍有未通过项", markdown)

    def test_quality_report_persists_only_suite_scoped_latest_aliases(self):
        report = QualityReport(
            run_id="run-persist",
            profile="offline",
            dataset_suite="medium_sql_v1",
            generated_at="2026-08-06T10:00:00+08:00",
            environment={},
            gate_passed=True,
            overall_score=100,
            domain_scores={},
            checks=[],
            recommendations=[],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _persist_report(report, root)
            expected = {
                "quality_medium_sql_v1_run-persist.json",
                "quality_medium_sql_v1_run-persist.md",
                "quality_medium_sql_v1_latest.json",
                "quality_medium_sql_v1_latest.md",
            }
            self.assertEqual({path.name for path in root.iterdir()}, expected)
            payload = json.loads(
                (root / "quality_medium_sql_v1_latest.json").read_text(
                    encoding="utf-8",
                )
            )
            self.assertEqual(payload["dataset_suite"], "medium_sql_v1")
            markdown = (
                root / "quality_medium_sql_v1_latest.md"
            ).read_text(encoding="utf-8")
            self.assertIn("Suite：`medium_sql_v1`", markdown)

    def test_quality_report_rejects_retired_suite_persistence(self):
        report = QualityReport(
            run_id="legacy",
            profile="offline",
            dataset_suite="common_sql_v1",
            generated_at="2026-08-06T10:00:00+08:00",
            environment={},
            gate_passed=False,
            overall_score=None,
            domain_scores={},
            checks=[],
            recommendations=[],
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "未知数据集套件"):
                _persist_report(report, Path(directory))


if __name__ == "__main__":
    unittest.main()
