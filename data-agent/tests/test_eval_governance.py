import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from app.config.app_config import app_config
from app.agent_system.agents.nl2sql.services.candidate_confidence import SEMANTIC_REVIEW_POLICY_VERSION
from app.agent_system.runtime.llm_attempts import LLM_ATTEMPT_SCHEMA
from app.agent_system.agents.nl2sql.services.planning_router import runtime_profile_status as planning_status
from app.agent_system.agents.nl2sql.services.query_signals import (
    build_route_signal_plan,
    extract_query_signals,
    query_signal_route_evidence,
    stable_signal_hash,
)
from app.infrastructure.search.index_version import retrieval_index_version
from app.agent_system.agents.nl2sql.services.semantic_repair import (
    PROFILE_RULES,
    REPAIR_REGISTRY_VERSION,
    REPAIR_RESULT_CONTRACT,
    RULE_SET_VERSION,
    semantic_repair_ledger_entry,
)
from app.agent_system.agents.nl2sql.services.semantic_repair import (
    runtime_profile_status as repair_status,
)
from app.domain.query.safety.semantic_violation import stable_hash as semantic_sql_hash
from evals.legacy.evaluator import (
    _planning_route_stage_evidence,
    _query_signal_stage_evidence,
    _semantic_repair_stage_evidence,
    _stage_evidence,
)
from evals.legacy.governance import (
    PROMOTION_SUITE_POLICIES,
    REQUIRED_FINGERPRINT_FIELDS,
    _pipeline_ablation_manifest,
    _safe_config_manifest,
    baseline_checklist,
    build_experiment_metadata,
    compare_fingerprints,
    compare_pipeline_ablation_fingerprints,
    experiment_report,
    minimum_passing_results,
    pipeline_evidence_checks,
    resolve_baseline_artifact,
    sample_contract_report,
    stable_hash,
    suite_result_floor_met,
)
from evals.legacy.run_eval import load_cases


class EvaluationGovernanceTests(unittest.TestCase):
    @staticmethod
    def _pipeline_aggregate_evidence(
        total: int,
        query_signal_evidence: dict,
    ) -> dict:
        query_profile = str(app_config.query_signals.profile)
        query_route_policy = str(app_config.query_signals.route_policy)
        dictionary_version = (
            "dictionary-v1"
            if query_profile == "domain_signals_v1"
            and app_config.query_signals.workspace_dictionary_enabled
            else "none"
        )
        planning_profile = str(planning_status()["runtime_profile"])
        repair_profile = str(repair_status()["runtime_profile"])
        rules = list(PROFILE_RULES[repair_profile])
        route_rows = {
            str(row.get("route") or "unknown"): row
            for row in (query_signal_evidence.get("routes") or [])
        }
        extractor_kind_counts = Counter(
            str(signal.get("kind") or "unknown")
            for signal in (query_signal_evidence.get("extractor_signals") or [])
        )
        consumer_kind_counts = Counter(
            f"{route}:{consumer}:{str(signal.get('kind') or 'unknown')}"
            for route, row in route_rows.items()
            for consumer in ("local", "remote", "rerank")
            for signal in (row.get(f"actual_{consumer}_signals") or [])
        )
        return {
            "query_signals": {
                "schema_version": "evaluation-query-signal-aggregate-v2",
                "active_profile_counts": {query_profile: total},
                "route_policy_counts": {query_route_policy: total},
                "dictionary_version_counts": {dictionary_version: total},
                "route_counts": {
                    route: total
                    for route in ("column", "metric", "training", "value")
                },
                "query_encode_mode_counts": {
                    f"{route}:{row.get('query_encode_mode')}": total
                    for route, row in route_rows.items()
                },
                "actual_remote_signal_counts": {
                    route: int(row.get("actual_remote_signal_count") or 0)
                    * total
                    for route, row in route_rows.items()
                },
                "actual_consumer_signal_counts": {
                    f"{route}:{consumer}": int(
                        row.get(f"actual_{consumer}_signal_count") or 0
                    ) * total
                    for route, row in route_rows.items()
                    for consumer in ("local", "remote", "rerank")
                },
                "extractor_kind_counts": {
                    kind: count * total
                    for kind, count in sorted(extractor_kind_counts.items())
                },
                "consumer_kind_counts": {
                    key: count * total
                    for key, count in sorted(consumer_kind_counts.items())
                },
                "evidence_count": total,
                "evidence_complete_count": total,
                "route_evidence_count": total * 4,
                "dictionary_fallback_count": 0,
                "evidence_complete": True,
            },
            "planning_route": {
                "schema_version": "evaluation-planning-route-aggregate-v1",
                "active_profile_counts": {planning_profile: total},
                "configured_profile_counts": {planning_profile: total},
                "fallback_profile_counts": {
                    str(planning_status()["fallback_profile"]): total,
                },
                "fallback_class_counts": {"none": total},
                "route_counts": {"direct_generate": total},
                "plan_calls": 0,
                "decide_node_calls": total,
                "decide_node_successes": total,
                "observed_plan_node_calls": 0,
                "plan_node_successes": 0,
                "physical_llm_calls": 0,
                "physical_llm_attempts": 0,
                "fallback_count": 0,
                "policy_fallback_count": 0,
                "unsafe_fallback_count": 0,
                "degraded_count": 0,
                "evidence_count": total,
                "evidence_complete_count": total,
                "plan_call_accounting_complete": True,
                "decision_accounting_complete": True,
                "physical_llm_call_accounting_complete": True,
                "evidence_complete": True,
            },
            "semantic_repair": {
                "schema_version": "evaluation-semantic-repair-aggregate-v1",
                "active_profile_counts": {repair_profile: total},
                "configured_profile_counts": {repair_profile: total},
                "attempted_rule_counts": {rule: total for rule in rules},
                "applied_rule_counts": {},
                "changed_count": 0,
                "fallback_count": 0,
                "degraded_count": 0,
                "evidence_count": total,
                "evidence_complete_count": total,
                "invocation_count": total,
                "invocation_evidence_complete_count": total,
                "invocation_profile_counts": {repair_profile: total},
                "invocation_phase_counts": {"primary_validate": total},
                "selected_invocation_count": total,
                "invocation_fallback_count": 0,
                "ledger_complete_cases": total,
                "invocation_evidence_complete": True,
                "evidence_complete": True,
            },
        }

    @staticmethod
    def _pipeline_fingerprint(config: dict) -> dict:
        profiles = {
            "query_signals": config["query"]["query_signals"],
            "planning_route": config["planning_route"],
            "semantic_repair": config["semantic_repair"],
        }
        fingerprint = {
            field: f"value-{field}" for field in REQUIRED_FINGERPRINT_FIELDS
        }
        fingerprint.update({
            "complete": True,
            "config_hash": stable_hash(config),
            "pipeline_profiles": json.loads(json.dumps(profiles)),
            "pipeline_ablation": _pipeline_ablation_manifest(config),
        })
        return fingerprint

    @staticmethod
    def _pipeline_stage_state() -> dict:
        query_profile = str(app_config.query_signals.profile)
        dictionary_version = (
            "dictionary-v1"
            if query_profile == "domain_signals_v1"
            and app_config.query_signals.workspace_dictionary_enabled
            else None
        )
        signals = extract_query_signals(
            "测试地区订单量",
            profile=query_profile,
            workspace_id="workspace-a",
            workspace_terms=("地区", "订单量") if dictionary_version else (),
            dictionary_version=dictionary_version,
            lexical_top_k=app_config.query_signals.lexical_top_k,
        )
        retrieval_metrics = []
        for route in ("column", "metric", "training", "value"):
            route_plan = build_route_signal_plan(
                signals,
                route,
                route_policy=app_config.query_signals.route_policy,
                remote_signal_limit=app_config.query_signals.remote_signal_limit,
            )
            retrieval_metrics.append({
                "route": route,
                "query_signals": query_signal_route_evidence(
                    signals,
                    route_plan,
                    query_encode_mode=(
                        "not_applicable_es_lexical"
                        if route == "value"
                        else "query"
                    ),
                ),
            })
        planning = planning_status()
        repair = repair_status()
        repair_profile = str(repair["runtime_profile"])
        rule_codes = list(PROFILE_RULES[repair_profile])
        sql = "SELECT 1"
        sql_hash = semantic_sql_hash(sql)
        semantic_repair = {
            "contract_version": REPAIR_RESULT_CONTRACT,
            "registry_version": REPAIR_REGISTRY_VERSION,
            "rule_set_version": RULE_SET_VERSION,
            "profile": repair_profile,
            "changed": False,
            "original_sql_hash": sql_hash,
            "repaired_sql_hash": sql_hash,
            "applied_rules": [],
            "skipped_rules": [
                {
                    "rule_code": code,
                    "rule_version": "1",
                    "status": "skipped",
                    "reason": "no_change",
                    "proof": {},
                }
                for code in rule_codes
            ],
            "attempted_rule_codes": rule_codes,
            "fallback_used": False,
            "invariants": {
                "single_bounded_pass": True,
                "table_set_unchanged": True,
                "filter_literals_unchanged": True,
                "read_only_preserved": True,
            },
            "raw_sql_logged": False,
        }
        validation = {
            "valid": True,
            "sql": sql,
            "semantic_repair": semantic_repair,
        }
        return {
            "sql": sql,
            "query_signals": signals.to_state(),
            "retrieval_metrics": retrieval_metrics,
            "degradations": [],
            "planning_route": {
                "contract_version": app_config.planning_route.contract_version,
                "profile": planning["runtime_profile"],
                "decision": "direct_generate",
                "actual_action": "direct_generate",
                "fallback_profile": planning["fallback_profile"],
                "fallback_applied": False,
                "router_degraded": False,
                "repair_profile": app_config.planning_route.repair_profile,
                "raw_text_logged": False,
            },
            "validation_evidence": validation,
            "semantic_repair_ledger": [semantic_repair_ledger_entry(
                validation,
                phase="primary_validate",
                committed_at_phase=True,
            )],
        }

    @staticmethod
    def _experiment_run(
        repeat_index: int, *, result: bool = True, prompt_hash: str = "same",
        evaluation_profile: str = "core-default",
        result_count: int | None = None,
        dataset_suite: str = "easy_sql_v1",
        total: int = 100,
    ) -> dict:
        passed_count = int(result_count) if result_count is not None else total if result else 0
        fingerprint = {field: f"value-{field}" for field in REQUIRED_FINGERPRINT_FIELDS}
        fingerprint.update({
            "fingerprint_schema": "evaluation-v2", "complete": True,
            "evaluation_profile": {
                "schema": "evaluation-profile-v1", "name": evaluation_profile,
                "analysis_enabled": evaluation_profile == "enhanced-analysis",
            },
            "prompt_hashes": {"generate_sql.prompt": prompt_hash},
            "run_fingerprint": "sha256:experiment",
            "sample_contract": {"total": total, "complete": True, "all_reviewed": True},
        })
        safe_config = _safe_config_manifest(evaluation_profile)
        fingerprint["pipeline_profiles"] = {
            "query_signals": safe_config["query"]["query_signals"],
            "planning_route": safe_config["planning_route"],
            "semantic_repair": safe_config["semantic_repair"],
        }
        stage_state = EvaluationGovernanceTests._pipeline_stage_state()
        planning_evidence = _planning_route_stage_evidence(stage_state)
        query_signal_evidence = _query_signal_stage_evidence(stage_state)
        repair_evidence = _semantic_repair_stage_evidence(stage_state)
        epoch = {
            "schema": "embedding-runtime-metrics-epoch-v1",
            "id": f"epoch-{repeat_index}",
            "index": repeat_index,
            "started_at": f"2026-08-14T00:00:0{repeat_index}+00:00",
        }
        cases = []
        for case_index in range(1, total + 1):
            case_passed = case_index <= passed_count
            cases.append({
                "id": f"eval_{case_index:03d}",
                "result_equivalent": case_passed,
                "first_sql_metrics": {
                    "result_evaluated": True,
                    "exec_success": True,
                    "result_equivalent": case_passed,
                },
                "stage_evidence": {
                    "retrieval": {}, "filtered_tables": [], "first_sql": "SELECT 1",
                    "correction_sqls": [], "authorized_sql": "SELECT 1",
                    "result_summary": {"masked": True, "values_persisted": False},
                    "semantic_coverage": {
                        "schema_version": "semantic-coverage-v1",
                        "coverage_score": 1.0,
                        "unresolved_slots": [],
                    },
                    "request_deadline": {
                        "contract_version": "request-deadline-v1",
                        "remaining_ms": 1000,
                        "absolute_timestamps_persisted": False,
                    },
                    "query_signals": copy.deepcopy(query_signal_evidence),
                    "retrieval_budget": [{
                        "route": "column",
                        "embedding_runtime": {
                            "metrics_epoch_id": epoch["id"],
                            "metrics_epoch_index": epoch["index"],
                            "metrics_epoch_started_at": epoch["started_at"],
                        },
                    }],
                    "planning_route": copy.deepcopy(planning_evidence),
                    "semantic_repair": copy.deepcopy(repair_evidence),
                },
            })
        return {
            "run_id": f"run-{repeat_index}", "workspace_id": "workspace-a",
            "valid": True, "n": total, "dataset_suite": dataset_suite,
            "experiment": build_experiment_metadata(
                experiment_id="experiment-001", candidate_label="v2-control",
                repeat_index=repeat_index, cache_state="cold" if repeat_index == 1 else "warm",
                provider_parameters={"temperature": 0, "seed": "unavailable"},
                evaluation_profile=evaluation_profile,
            ),
            "fingerprint": fingerprint,
            "embedding_warmup": {
                "schema": "evaluation-embedding-warmup-v1",
                "status": "pass",
                "session_id": "embedding-session-1",
                "session_run_index": repeat_index,
                "resident_reused": repeat_index > 1,
                "profile_fingerprint": "embedding-profile-1",
                "runtime_profile_fingerprint": "runtime-profile-1",
                "dense_size": app_config.embedding.dense_size,
                "sparse_nonzero": 3,
                "model_resident_for_suite": True,
                "resident_loaded_at_suite_start": True,
                "resident_accepting_at_suite_start": True,
                "retrieval_caches_invalidated_after_warmup": True,
                "retrieval_cache_generation_before": 0,
                "retrieval_cache_generation_after": 1,
                "warmup_excluded_from_case_timing": True,
                "suite_started_only_after_pass": True,
                "checks": {
                    "dense_size": True,
                    "sparse_nonzero": True,
                    "model_loaded": True,
                    "single_model_instance": True,
                    "accepting_requests": True,
                    "profile_stable": True,
                    "metrics_epoch_reset": True,
                },
                "metrics_epoch_reset": {
                    "schema": "embedding-runtime-metrics-reset-v1",
                    "status": "pass",
                    "active_epoch": copy.deepcopy(epoch),
                    "queue_state_preserved": True,
                    "model_state_preserved": True,
                    "only_observability_reset": True,
                    "active_inference_before": 0,
                    "active_async_before": 0,
                    "active_observations": {
                        "requests": {
                            "submitted": 0,
                            "completed": 0,
                            "rejected": 0,
                            "cancelled": 0,
                            "timed_out": 0,
                            "failed": 0,
                        },
                        "throughput": {"model_invocations": 0},
                    },
                },
                "suite_metrics_epoch": copy.deepcopy(epoch),
            },
            # 派生字段保留用于证明 governance 不依赖它们。
            "embedding_warmup_checks": {"complete": True},
            "cache_control": {
                "requested_state": "cold" if repeat_index == 1 else "warm",
                "effective_policy": (
                    "cleared_all"
                    if repeat_index == 1
                    else "retained_ttl_best_effort"
                ),
                "result_cache_cleared": repeat_index == 1,
                "retrieval_caches_cleared": repeat_index == 1,
                "retrieval_cache_generation_before": 1 if repeat_index == 1 else 2,
                "retrieval_cache_generation_after": 2,
                "ttl_seconds": {"result": 30, "retrieval": 120},
                "cache_hit_not_guaranteed": repeat_index > 1,
                "latency_attribution_allowed": False,
            },
            "embedding_runtime_final": {
                "schema": "evaluation-embedding-runtime-epoch-v1",
                "status": "pass",
                "checks": {
                    "epoch_stable": True,
                    "queue_idle": True,
                    "native_idle": True,
                    "model_resident": True,
                    "profile_stable": True,
                },
                "epoch": copy.deepcopy(epoch),
                "requests": {
                    "submitted": total,
                    "completed": total,
                    "rejected": 0,
                    "cancelled": 0,
                    "timed_out": 0,
                    "failed": 0,
                },
                "queue": {"depth": 0, "running_requests": 0},
                "profile_fingerprint": "embedding-profile-1",
                "runtime_profile_fingerprint": "runtime-profile-1",
            },
            "embedding_runtime_health": {
                "schema": "evaluation-embedding-runtime-health-v1",
                "eligible": True,
                "expected_epoch": copy.deepcopy(epoch),
                "observed_epoch_count": total,
                "mismatched_epoch_count": 0,
                "mismatched_epochs": [],
                "typed_core_retrieval_failures": [],
                "scheduler_failure_counts": {
                    "rejected": 0,
                    "cancelled": 0,
                    "timed_out": 0,
                    "failed": 0,
                },
            },
            "aggregate": {
                "n": total, "result_evaluated": total,
                "first_sql_evidence_complete": True,
                "first_result_equivalent": passed_count,
                "first_exec_success": total,
                "syntax_valid": total,
                "exec_success": total, "tables_correct": total,
                "table_scope_safe": total,
                "table_scope_evidence_complete": total,
                "security_guard_bypass_count": 0,
                "result_equivalent": passed_count, "infrastructure_errors": 0,
                "tt_sql_p95": 1000, "tt_result_p95": 1200, "latency_p95": 1500,
                "tt_complete_p95": 1500, "timing_protocol_version": "timing-v2",
                "timing_invariant_violations": [],
                "llm_calls_p95": 2, "tokens_p95": 100,
                "tokens_mean": 100, "tokens_max": 100,
                "llm_attempt_schema": LLM_ATTEMPT_SCHEMA, "llm_attempts": 2,
                "llm_attempt_accounting_complete": True,
                "degradation_evidence_complete": True,
                **EvaluationGovernanceTests._pipeline_aggregate_evidence(
                    total,
                    query_signal_evidence,
                ),
                "failed_core_llm_calls": 0, "failed_optional_llm_calls": 0,
                "llm_attempts_timed_out": 0,
                "semantic_coverage_cases": total,
                "semantic_coverage_mean": 1.0,
                "semantic_unresolved_cases": 0,
                "semantic_unsafe_compile_cases": 0,
                "request_deadline_contract_complete": True,
                "request_deadline_exhausted_cases": 0,
                "llm_outcome_classes": {"success": 2},
                "llm_recoveries": 0,
            },
            "cases": cases,
        }

    @staticmethod
    def _enable_semantic_review(runs: list[dict]) -> None:
        for run in runs:
            run["fingerprint"]["semantic_review"] = {
                "enabled": True,
                "policy_version": SEMANTIC_REVIEW_POLICY_VERSION,
            }
            run["aggregate"].update({
                "semantic_review_enabled": True,
                "semantic_review_policy_version": SEMANTIC_REVIEW_POLICY_VERSION,
                "semantic_review_eligible_count": int(run["n"]),
                "semantic_review_evidence_complete": True,
                "semantic_review_coverage_complete": True,
                "semantic_review_at_most_one_attempt": True,
                "semantic_review_unsafe_selected": 0,
                "semantic_review_failed_core": 0,
                "semantic_review_harmful_modification": 0,
                "semantic_review_first_correct_final_incorrect": 0,
                "semantic_review_same_sql_execution_loss": 0,
                "semantic_review_selected_b_no_sql_delta": 0,
                "semantic_review_budget_skipped": 0,
                "semantic_review_result_attribution_complete": True,
                "adaptive_candidate_unsafe_selected": 0,
                "adaptive_candidate_high_confidence_extra_calls": 0,
                "adaptive_candidate_rate": 0,
            })

    def test_all_golden_cases_have_reviewed_contract(self):
        cases = load_cases()
        report = sample_contract_report(cases)
        self.assertEqual(len(cases), 100)
        self.assertEqual(report["total"], len(cases))
        self.assertEqual(report["complete_count"], len(cases))
        self.assertTrue(report["complete"])
        self.assertTrue(report["all_reviewed"])
        self.assertEqual(report["duplicate_ids"], [])
        self.assertEqual(report["unsafe_sql_case_ids"], [])

    def test_fingerprint_comparison_refuses_incomplete_or_changed_environment(self):
        complete = {field: f"value-{field}" for field in REQUIRED_FINGERPRINT_FIELDS}
        complete["complete"] = True
        self.assertTrue(compare_fingerprints(complete, dict(complete))["comparable"])
        changed = dict(complete, prompt_hashes={"generate_sql.prompt": "changed"})
        comparison = compare_fingerprints(complete, changed)
        self.assertFalse(comparison["comparable"])
        self.assertIn("prompt_hashes", comparison["differences"])
        self.assertFalse(compare_fingerprints(complete, {})["comparable"])

    def test_query_tuning_changes_config_but_not_materialized_index_version(self):
        previous_limit = app_config.query.column_recall_limit
        previous_max_length = app_config.embedding.max_length
        config_before = stable_hash(_safe_config_manifest())
        index_before = retrieval_index_version()
        try:
            app_config.query.column_recall_limit = previous_limit + 1
            self.assertNotEqual(config_before, stable_hash(_safe_config_manifest()))
            self.assertEqual(index_before, retrieval_index_version())
            app_config.embedding.max_length = previous_max_length + 1
            self.assertNotEqual(index_before, retrieval_index_version())
        finally:
            app_config.query.column_recall_limit = previous_limit
            app_config.embedding.max_length = previous_max_length

    def test_fingerprint_manifest_versions_semantic_review_policy(self):
        manifest = _safe_config_manifest()
        review = manifest["query"]["semantic_review"]
        invocation = manifest["query"]["model_invocation"]

        self.assertIs(review["enabled"], app_config.query.sql_semantic_review_enabled)
        self.assertEqual(SEMANTIC_REVIEW_POLICY_VERSION, review["policy_version"])
        self.assertEqual(
            invocation["recovery_reserve_ms"],
            app_config.query.llm_recovery_reserve_ms,
        )

    def test_fingerprint_manifest_binds_pipeline_ablation_profiles(self):
        manifest = _safe_config_manifest()

        self.assertEqual(
            manifest["planning_route"]["active_profile"],
            app_config.planning_route.active_profile,
        )
        self.assertEqual(
            manifest["semantic_repair"]["active_profile"],
            app_config.semantic_repair.active_profile,
        )

        original_planning = app_config.planning_route.active_profile
        original_repair = app_config.semantic_repair.active_profile
        baseline_hash = stable_hash(manifest)
        try:
            app_config.planning_route.active_profile = "weighted_scorecard_v1"
            self.assertNotEqual(
                baseline_hash,
                stable_hash(_safe_config_manifest()),
            )
            app_config.planning_route.active_profile = original_planning
            app_config.semantic_repair.active_profile = "registry_extended_v1"
            self.assertNotEqual(
                baseline_hash,
                stable_hash(_safe_config_manifest()),
            )
        finally:
            app_config.planning_route.active_profile = original_planning
            app_config.semantic_repair.active_profile = original_repair

    def test_run_fingerprint_exposes_pipeline_profiles_without_secrets(self):
        manifest = _safe_config_manifest()
        profiles = {
            "query_signals": manifest["query"]["query_signals"],
            "planning_route": manifest["planning_route"],
            "semantic_repair": manifest["semantic_repair"],
        }

        rendered = json.dumps(profiles, ensure_ascii=False)
        self.assertEqual(
            profiles["query_signals"]["profile"],
            app_config.query_signals.profile,
        )
        self.assertEqual(
            profiles["planning_route"]["active_profile"],
            app_config.planning_route.active_profile,
        )
        self.assertEqual(
            profiles["semantic_repair"]["active_profile"],
            app_config.semantic_repair.active_profile,
        )
        self.assertNotIn("api_key", rendered.casefold())
        self.assertNotIn("password", rendered.casefold())

    def test_pipeline_ablation_manifest_proves_one_factor_changes(self):
        baseline = _safe_config_manifest()
        baseline_manifest = _pipeline_ablation_manifest(baseline)

        query_candidate = json.loads(json.dumps(baseline))
        query_candidate["query"]["query_signals"].update({
            "profile": "domain_signals_v1",
            "route_policy": "routed_signals_v1",
        })
        query_manifest = _pipeline_ablation_manifest(query_candidate)
        self.assertEqual(
            baseline_manifest["factors"]["query_signals"]["frozen_config_hash"],
            query_manifest["factors"]["query_signals"]["frozen_config_hash"],
        )
        self.assertNotEqual(
            baseline_manifest["factors"]["planning_route"]["frozen_config_hash"],
            query_manifest["factors"]["planning_route"]["frozen_config_hash"],
        )

        planning_candidate = json.loads(json.dumps(baseline))
        planning_candidate["planning_route"]["active_profile"] = (
            "weighted_scorecard_v1"
        )
        planning_manifest = _pipeline_ablation_manifest(planning_candidate)
        self.assertEqual(
            baseline_manifest["factors"]["planning_route"]["frozen_config_hash"],
            planning_manifest["factors"]["planning_route"]["frozen_config_hash"],
        )

        semantic_candidate = json.loads(json.dumps(baseline))
        semantic_candidate["semantic_repair"]["active_profile"] = (
            "registry_extended_v1"
        )
        semantic_candidate["planning_route"]["repair_profile"] = (
            "registry_extended_v1"
        )
        semantic_manifest = _pipeline_ablation_manifest(semantic_candidate)
        self.assertEqual(
            baseline_manifest["factors"]["semantic_repair"]["frozen_config_hash"],
            semantic_manifest["factors"]["semantic_repair"]["frozen_config_hash"],
        )

    def test_pipeline_comparator_accepts_control_to_c4_as_one_variable(self):
        control = _safe_config_manifest()
        control["planning_route"]["active_profile"] = "keyword_length_v1"
        candidate = json.loads(json.dumps(control))
        candidate["planning_route"]["active_profile"] = (
            "weighted_scorecard_v1"
        )

        comparison = compare_pipeline_ablation_fingerprints(
            self._pipeline_fingerprint(control),
            self._pipeline_fingerprint(candidate),
            factor="planning_route",
        )

        self.assertTrue(comparison["attribution_valid"], comparison)
        self.assertEqual(
            comparison["allowed_differences"],
            ["pipeline_profiles.planning_route.active_profile"],
        )

    def test_pipeline_comparator_supports_query_and_repair_factors(self):
        control = _safe_config_manifest()
        query_candidate = json.loads(json.dumps(control))
        query_candidate["query"]["query_signals"].update({
            "profile": "domain_signals_v1",
            "route_policy": "routed_signals_v1",
        })
        query = compare_pipeline_ablation_fingerprints(
            self._pipeline_fingerprint(control),
            self._pipeline_fingerprint(query_candidate),
            factor="query_signal",
        )
        self.assertTrue(query["attribution_valid"], query)

        repair_candidate = json.loads(json.dumps(control))
        repair_candidate["semantic_repair"]["active_profile"] = (
            "registry_extended_v1"
        )
        repair_candidate["planning_route"]["repair_profile"] = (
            "registry_extended_v1"
        )
        repair = compare_pipeline_ablation_fingerprints(
            self._pipeline_fingerprint(control),
            self._pipeline_fingerprint(repair_candidate),
            factor="semantic_repair",
        )
        self.assertTrue(repair["attribution_valid"], repair)

    def test_pipeline_comparator_rejects_extra_variable_and_bad_evidence(self):
        control = _safe_config_manifest()
        candidate = json.loads(json.dumps(control))
        candidate["planning_route"]["active_profile"] = (
            "weighted_scorecard_v1"
        )
        candidate["query"]["query_signals"]["profile"] = "domain_signals_v1"
        comparison = compare_pipeline_ablation_fingerprints(
            self._pipeline_fingerprint(control),
            self._pipeline_fingerprint(candidate),
            factor="planning_route",
        )
        self.assertFalse(comparison["attribution_valid"])
        self.assertIn("frozen_config_hash", comparison["unexpected_differences"])
        self.assertIn(
            "pipeline_profiles.query_signals.profile",
            comparison["unexpected_differences"],
        )

        bad_fallback = json.loads(json.dumps(candidate))
        bad_fallback["query"]["query_signals"]["profile"] = (
            control["query"]["query_signals"]["profile"]
        )
        bad_fallback["planning_route"]["fallback_profile"] = "unknown"
        invalid = compare_pipeline_ablation_fingerprints(
            self._pipeline_fingerprint(control),
            self._pipeline_fingerprint(bad_fallback),
            factor="planning_route",
        )
        self.assertFalse(invalid["attribution_valid"])
        self.assertIn(
            "right.planning_route.fallback_unknown",
            invalid["unexpected_differences"],
        )

        missing = self._pipeline_fingerprint(candidate)
        missing.pop("pipeline_ablation")
        absent = compare_pipeline_ablation_fingerprints(
            self._pipeline_fingerprint(control),
            missing,
            factor="planning_route",
        )
        self.assertFalse(absent["attribution_valid"])
        self.assertIn(
            "right.pipeline_ablation_manifest_missing",
            absent["unexpected_differences"],
        )

    def test_pipeline_stage_evidence_is_private_and_fail_closed(self):
        state = self._pipeline_stage_state()
        secret = "customer@example.invalid"
        state["query"] = secret
        state["validation_evidence"]["semantic_repair"]["sql"] = secret
        state["validation_evidence"]["semantic_repair"]["skipped_rules"][0][
            "proof"
        ] = {"unsafe_literal": secret}

        query_signals = _query_signal_stage_evidence(state)
        planning = _planning_route_stage_evidence(state)
        repair = _semantic_repair_stage_evidence(state)
        self.assertTrue(query_signals["evidence_complete"], query_signals)
        self.assertTrue(planning["evidence_complete"], planning)
        self.assertTrue(repair["evidence_complete"], repair)
        self.assertEqual(repair["selected_invocation_count"], 1)
        self.assertTrue(repair["ledger"][0]["committed_at_phase"])
        self.assertTrue(repair["ledger"][0]["final_invocation"])
        self.assertNotIn(
            secret,
            json.dumps({
                "query_signals": query_signals,
                "planning": planning,
                "repair": repair,
            }),
        )
        self.assertFalse(query_signals["raw_text_logged"])
        self.assertFalse(planning["raw_text_logged"])
        self.assertFalse(repair["raw_sql_logged"])

        missing_query_route = copy.deepcopy(state)
        missing_query_route["retrieval_metrics"] = (
            missing_query_route["retrieval_metrics"][:-1]
        )
        self.assertFalse(
            _query_signal_stage_evidence(missing_query_route)[
                "evidence_complete"
            ]
        )

        dictionary_fallback = copy.deepcopy(state)
        dictionary_fallback["degradations"] = [{
            "dependency": "query_signal_dictionary",
            "status": "fallback",
            "fallback": "jieba_tfidf_v1",
        }]
        self.assertFalse(
            _query_signal_stage_evidence(dictionary_fallback)[
                "evidence_complete"
            ]
        )

        route_contract_mutations = {
            "missing_actual_remote_signals": lambda row: row.pop(
                "actual_remote_signals"
            ),
            "empty_query_encode_mode": lambda row: row.update({
                "query_encode_mode": "",
            }),
            "negative_remote_count": lambda row: row.update({
                "remote_query_count": -7,
            }),
            "empty_remote_consumption": lambda row: row.update({
                "remote_query_count": 0,
                "actual_remote_signals": [],
            }),
        }
        for name, mutate in route_contract_mutations.items():
            with self.subTest(query_signal_route_contract=name):
                malformed = copy.deepcopy(state)
                mutate(
                    malformed["retrieval_metrics"][0]["query_signals"]
                )
                self.assertFalse(
                    _query_signal_stage_evidence(malformed)[
                        "evidence_complete"
                    ]
                )

        forged_route_state = copy.deepcopy(state)
        raw_route = forged_route_state["retrieval_metrics"][0][
            "query_signals"
        ]
        forged_fingerprint = stable_signal_hash({"forged": "domain-term"})
        raw_route["actual_remote_signals"] = [{
            "signal_id": f"qs-{forged_fingerprint[:12]}",
            "kind": "domain_terms",
            "rank": 1,
            "fingerprint": forged_fingerprint,
            "consumer": "remote",
        }]
        raw_route["remote_query_count"] = 1
        raw_route["consumer_plan_fingerprint"] = stable_signal_hash({
            "contract": raw_route["contract_version"],
            "route_policy_version": raw_route["route_policy_version"],
            "route_policy": raw_route["route_policy"],
            "route": raw_route["route"],
            "profile_fingerprint": raw_route["profile_fingerprint"],
            "signal_fingerprint": raw_route["signal_fingerprint"],
            "query_encode_mode": raw_route["query_encode_mode"],
            "local": raw_route["actual_local_signals"],
            "remote": raw_route["actual_remote_signals"],
            "rerank": raw_route["actual_rerank_signals"],
        })
        self.assertFalse(
            _query_signal_stage_evidence(forged_route_state)[
                "evidence_complete"
            ]
        )

        policy_abstain = copy.deepcopy(state)
        policy_abstain["planning_route"].update({
            "profile": "weighted_scorecard_v1",
            "decision": "abstain",
            "actual_action": "direct_generate",
            "reason_codes": ["score:signal_complexity", "abstain_fallback"],
            "fallback_applied": True,
            "router_degraded": False,
        })
        with patch(
            "evals.legacy.evaluator.planning_runtime_profile_status",
            return_value={
                "configured_profile": "weighted_scorecard_v1",
                "runtime_profile": "weighted_scorecard_v1",
                "runtime_override": None,
                "fallback_profile": "keyword_length_v1",
            },
        ):
            policy_evidence = _planning_route_stage_evidence(policy_abstain)
        self.assertTrue(policy_evidence["evidence_complete"], policy_evidence)
        self.assertEqual(
            policy_evidence["fallback_class"], "policy_abstain"
        )
        self.assertTrue(policy_evidence["policy_fallback_eligible"])

        degraded_fallback = copy.deepcopy(policy_abstain)
        degraded_fallback["planning_route"].update({
            "profile": "keyword_length_v1",
            "decision": "direct_generate",
            "router_degraded": True,
        })
        degraded_evidence = _planning_route_stage_evidence(
            degraded_fallback
        )
        self.assertFalse(degraded_evidence["evidence_complete"])
        self.assertEqual(
            degraded_evidence["fallback_class"], "exception_degraded"
        )

        unknown_fallback = copy.deepcopy(state)
        unknown_fallback["planning_route"]["fallback_applied"] = True
        unknown_evidence = _planning_route_stage_evidence(unknown_fallback)
        self.assertFalse(unknown_evidence["evidence_complete"])
        self.assertEqual(unknown_evidence["fallback_class"], "unknown")

        wrong_profile = json.loads(json.dumps(state))
        wrong_profile["planning_route"]["profile"] = "never_plan"
        self.assertFalse(
            _planning_route_stage_evidence(wrong_profile)["evidence_complete"]
        )
        wrong_fallback = json.loads(json.dumps(state))
        wrong_fallback["planning_route"]["fallback_profile"] = "always_plan"
        self.assertFalse(
            _planning_route_stage_evidence(wrong_fallback)["evidence_complete"]
        )
        missing = _planning_route_stage_evidence({})
        self.assertFalse(missing["evidence_complete"])
        self.assertTrue(missing["evidence_failures"])

        repair_fallback = json.loads(json.dumps(state))
        repair_fallback["validation_evidence"]["semantic_repair"].update({
            "fallback_used": True,
            "fallback_reason": None,
        })
        self.assertFalse(
            _semantic_repair_stage_evidence(repair_fallback)[
                "evidence_complete"
            ]
        )
        self.assertFalse(
            _semantic_repair_stage_evidence({})["evidence_complete"]
        )

        wrong_hash = copy.deepcopy(state)
        wrong_hash["semantic_repair_ledger"][0]["observed_sql_hash"] = (
            "sha256:wrong"
        )
        self.assertFalse(
            _semantic_repair_stage_evidence(wrong_hash)["evidence_complete"]
        )

        terminal_failure = copy.deepcopy(state)
        terminal_failure["validation_evidence"]["valid"] = False
        terminal_failure["semantic_repair_ledger"][0][
            "validation_valid"
        ] = False
        terminal_evidence = _semantic_repair_stage_evidence(terminal_failure)
        self.assertTrue(terminal_evidence["evidence_complete"])
        self.assertTrue(terminal_evidence["ledger"][0]["selected"])
        self.assertTrue(
            terminal_evidence["ledger"][0]["final_invocation"]
        )

        fallback_committed = copy.deepcopy(state)
        fallback_committed["semantic_repair_ledger"][0].update({
            "committed_at_phase": False,
            "selected": False,
        })
        fallback_entry = copy.deepcopy(
            state["semantic_repair_ledger"][0]
        )
        fallback_entry["phase"] = "deterministic_fallback_validate"
        fallback_committed["semantic_repair_ledger"].append(fallback_entry)
        fallback_evidence = _semantic_repair_stage_evidence(
            fallback_committed
        )
        self.assertTrue(fallback_evidence["evidence_complete"])
        self.assertFalse(fallback_evidence["ledger"][0]["selected"])
        self.assertTrue(fallback_evidence["ledger"][1]["selected"])

        unselected_same_sql = copy.deepcopy(state)
        unselected_entry = copy.deepcopy(
            state["semantic_repair_ledger"][0]
        )
        unselected_entry.update({
            "phase": "candidate_b_validate",
            "committed_at_phase": False,
            "selected": False,
        })
        unselected_same_sql["semantic_repair_ledger"].append(
            unselected_entry
        )
        unselected_evidence = _semantic_repair_stage_evidence(
            unselected_same_sql
        )
        self.assertTrue(unselected_evidence["evidence_complete"])
        self.assertTrue(unselected_evidence["ledger"][0]["selected"])
        self.assertFalse(unselected_evidence["ledger"][1]["selected"])

        unselected_fallback = copy.deepcopy(state)
        candidate_entry = copy.deepcopy(
            unselected_fallback["semantic_repair_ledger"][0]
        )
        candidate_entry.update({
            "phase": "candidate_b_validate",
            "committed_at_phase": False,
            "selected": False,
            "fallback_used": True,
            "fallback_reason": "RuntimeError",
        })
        unselected_fallback["semantic_repair_ledger"].append(candidate_entry)
        self.assertFalse(
            _semantic_repair_stage_evidence(unselected_fallback)[
                "evidence_complete"
            ]
        )

        duplicate_selected = copy.deepcopy(state)
        duplicate_entry = copy.deepcopy(
            duplicate_selected["semantic_repair_ledger"][0]
        )
        duplicate_entry["phase"] = "candidate_b_validate"
        duplicate_selected["semantic_repair_ledger"][0][
            "final_invocation"
        ] = True
        duplicate_entry["final_invocation"] = True
        duplicate_selected["semantic_repair_ledger"].append(duplicate_entry)
        duplicate_evidence = _semantic_repair_stage_evidence(
            duplicate_selected
        )
        self.assertEqual(duplicate_evidence["selected_invocation_count"], 1)
        self.assertTrue(duplicate_evidence["evidence_complete"])
        self.assertFalse(duplicate_evidence["ledger"][0]["selected"])
        self.assertFalse(
            duplicate_evidence["ledger"][0]["final_invocation"]
        )
        self.assertTrue(duplicate_evidence["ledger"][1]["selected"])
        self.assertTrue(duplicate_evidence["ledger"][1]["final_invocation"])

        with patch(
            "evals.legacy.evaluator.planning_runtime_profile_status",
            return_value={
                "configured_profile": "keyword_length_v1",
                "runtime_profile": "weighted_scorecard_v1",
                "runtime_override": "weighted_scorecard_v1",
                "fallback_profile": "keyword_length_v1",
            },
        ):
            override_evidence = _planning_route_stage_evidence(state)
            self.assertFalse(override_evidence["evidence_complete"])
            self.assertEqual(
                override_evidence["fallback_class"], "profile_override"
            )
        with patch(
            "evals.legacy.evaluator.semantic_repair_runtime_profile_status",
            return_value={
                "configured_profile": "registry_same_rules_v1",
                "runtime_profile": "registry_extended_v1",
                "fallback_profile": "inline_v1",
                "override_active": True,
            },
        ):
            self.assertFalse(
                _semantic_repair_stage_evidence(state)["evidence_complete"]
            )

    def test_semantic_repair_stage_recovers_execute_error_from_final_ledger(self):
        """执行错误覆盖顶层校验证据后，仍由最终 ledger 证明修复身份。"""

        state = self._pipeline_stage_state()
        final_entry = state["semantic_repair_ledger"][0]
        previous_entry = copy.deepcopy(final_entry)
        previous_hash = semantic_sql_hash("SELECT 2")
        previous_entry.update({
            "original_sql_hash": previous_hash,
            "repaired_sql_hash": previous_hash,
            "observed_sql_hash": previous_hash,
            "validation_valid": False,
            "final_invocation": True,
        })
        historical_final = copy.deepcopy(final_entry)
        historical_final["final_invocation"] = True
        latest_final = copy.deepcopy(final_entry)
        latest_final.update({
            "phase": "deterministic_fallback_validate",
            "final_invocation": False,
        })
        trailing_entry = copy.deepcopy(previous_entry)
        trailing_hash = semantic_sql_hash("SELECT 3")
        trailing_entry.update({
            "phase": "candidate_b_validate",
            "committed_at_phase": False,
            "selected": False,
            "original_sql_hash": trailing_hash,
            "repaired_sql_hash": trailing_hash,
            "observed_sql_hash": trailing_hash,
            "final_invocation": True,
        })
        # easy_073 同型：多次提交同一最终 SQL，且最后还有未提交候选；输入中的
        # final_invocation 标志不可信，必须由 committed+hash+顺序重新推导。
        state["semantic_repair_ledger"] = [
            previous_entry,
            historical_final,
            latest_final,
            trailing_entry,
        ]
        state["error"] = "SQL执行错误：Duplicate column name 'product_id'"
        state["validation_evidence"] = {
            "error_code": "execution.dependency_error",
            "violation": {
                "violation_code": "execution.dependency_error",
                "source": "execution",
                "repairability": "upstream",
                "safe_to_retry": False,
            },
        }

        evidence = _semantic_repair_stage_evidence(state)

        self.assertTrue(evidence["evidence_complete"], evidence)
        self.assertEqual(evidence["summary_source"], "final_ledger")
        self.assertEqual(evidence["invocation_count"], 4)
        self.assertEqual(evidence["selected_invocation_count"], 1)
        self.assertFalse(evidence["ledger"][0]["final_invocation"])
        self.assertFalse(evidence["ledger"][1]["selected"])
        self.assertFalse(evidence["ledger"][1]["final_invocation"])
        self.assertTrue(evidence["ledger"][2]["selected"])
        self.assertTrue(evidence["ledger"][2]["final_invocation"])
        self.assertFalse(evidence["ledger"][3]["selected"])
        self.assertFalse(evidence["ledger"][3]["final_invocation"])
        self.assertEqual(
            evidence["active_profile"],
            repair_status()["runtime_profile"],
        )
        self.assertEqual(
            evidence["repaired_sql_hash"],
            semantic_sql_hash(state["sql"]),
        )

        invalid_states = {}

        missing_ledger = copy.deepcopy(state)
        missing_ledger["semantic_repair_ledger"] = []
        invalid_states["missing_ledger"] = missing_ledger

        missing_match = copy.deepcopy(state)
        missing_match["sql"] = "SELECT 999"
        invalid_states["missing_match"] = missing_match

        wrong_hash = copy.deepcopy(state)
        wrong_hash["semantic_repair_ledger"][2][
            "repaired_sql_hash"
        ] = "sha256:wrong"
        invalid_states["wrong_hash"] = wrong_hash

        wrong_profile = copy.deepcopy(state)
        wrong_profile["semantic_repair_ledger"][2]["profile"] = "inline_v1"
        invalid_states["wrong_profile"] = wrong_profile

        wrong_version = copy.deepcopy(state)
        wrong_version["semantic_repair_ledger"][2]["registry_version"] = (
            "semantic-repair-registry-wrong"
        )
        invalid_states["wrong_version"] = wrong_version

        for name, invalid_state in invalid_states.items():
            with self.subTest(fail_closed=name):
                invalid_evidence = _semantic_repair_stage_evidence(
                    invalid_state
                )
                self.assertFalse(
                    invalid_evidence["evidence_complete"],
                    invalid_evidence,
                )
                self.assertNotEqual(
                    invalid_evidence["summary_source"],
                    "final_ledger",
                )

    def test_pipeline_evidence_binds_counts_nodes_and_fingerprint_profiles(self):
        run = self._experiment_run(1, total=2)
        checks = pipeline_evidence_checks(
            run["fingerprint"], run["aggregate"], run["cases"],
        )
        self.assertTrue(all(checks.values()), checks)

        query_route_mismatch = copy.deepcopy(run)
        query_route_mismatch["cases"][0]["stage_evidence"][
            "query_signals"
        ]["routes"][0]["route_policy"] = "routed_signals_v1"
        self.assertFalse(pipeline_evidence_checks(
            query_route_mismatch["fingerprint"],
            query_route_mismatch["aggregate"],
            query_route_mismatch["cases"],
        )["query_signal_evidence_complete"])

        query_forged_fallback = copy.deepcopy(run)
        query_forged_fallback["cases"][0]["stage_evidence"][
            "query_signals"
        ]["dictionary_fallback_count"] = 1
        self.assertFalse(pipeline_evidence_checks(
            query_forged_fallback["fingerprint"],
            query_forged_fallback["aggregate"],
            query_forged_fallback["cases"],
        )["query_signal_evidence_complete"])

        query_profile_mismatch = copy.deepcopy(run)
        query_profile_mismatch["fingerprint"]["pipeline_profiles"][
            "query_signals"
        ]["profile"] = "domain_signals_v1"
        self.assertFalse(pipeline_evidence_checks(
            query_profile_mismatch["fingerprint"],
            query_profile_mismatch["aggregate"],
            query_profile_mismatch["cases"],
        )["query_signal_evidence_complete"])

        query_execution_mutations = {
            "aggregate_schema": lambda value: value["aggregate"][
                "query_signals"
            ].update({"schema_version": "forged-v999"}),
            "missing_actual_remote_signals": lambda value: value["cases"][0][
                "stage_evidence"
            ]["query_signals"]["routes"][0].pop("actual_remote_signals"),
            "empty_query_encode_mode": lambda value: value["cases"][0][
                "stage_evidence"
            ]["query_signals"]["routes"][0].update({
                "query_encode_mode": "",
            }),
            "negative_remote_count": lambda value: value["cases"][0][
                "stage_evidence"
            ]["query_signals"]["routes"][0].update({
                "remote_query_count": -7,
            }),
            "empty_remote_consumption": lambda value: value["cases"][0][
                "stage_evidence"
            ]["query_signals"]["routes"][0].update({
                "remote_query_count": 0,
                "actual_remote_signal_count": 0,
                "actual_remote_signals": [],
                "remote_signal_fingerprint": stable_hash([]),
            }),
            "aggregate_remote_count": lambda value: value["aggregate"][
                "query_signals"
            ]["actual_remote_signal_counts"].update({"column": 999}),
            "missing_local_consumption": lambda value: value["cases"][0][
                "stage_evidence"
            ]["query_signals"]["routes"][0].pop("actual_local_signals"),
        }
        for name, mutate in query_execution_mutations.items():
            with self.subTest(query_signal_governance=name):
                malformed = copy.deepcopy(run)
                mutate(malformed)
                self.assertFalse(pipeline_evidence_checks(
                    malformed["fingerprint"],
                    malformed["aggregate"],
                    malformed["cases"],
                )["query_signal_evidence_complete"])

        forged_identity = copy.deepcopy(run)
        query_evidence = forged_identity["cases"][0]["stage_evidence"][
            "query_signals"
        ]
        route = query_evidence["routes"][0]
        forged_fingerprint = stable_signal_hash({"forged": "domain-term"})
        route["actual_remote_signals"] = [{
            "signal_id": f"qs-{forged_fingerprint[:12]}",
            "kind": "domain_terms",
            "rank": 1,
            "fingerprint": forged_fingerprint,
            "consumer": "remote",
        }]
        route["remote_signal_fingerprint"] = stable_hash(
            route["actual_remote_signals"]
        )
        route["consumer_plan_fingerprint"] = stable_signal_hash({
            "contract": query_evidence["contract_version"],
            "route_policy_version": route["route_policy_version"],
            "route_policy": route["route_policy"],
            "route": route["route"],
            "profile_fingerprint": route["profile_fingerprint"],
            "signal_fingerprint": route["signal_fingerprint"],
            "query_encode_mode": route["query_encode_mode"],
            "local": route["actual_local_signals"],
            "remote": route["actual_remote_signals"],
            "rerank": route["actual_rerank_signals"],
        })
        self.assertFalse(pipeline_evidence_checks(
            forged_identity["fingerprint"],
            forged_identity["aggregate"],
            forged_identity["cases"],
        )["query_signal_evidence_complete"])

        node_mismatch = copy.deepcopy(run)
        node_mismatch["aggregate"]["planning_route"]["decide_node_calls"] = 1
        self.assertFalse(pipeline_evidence_checks(
            node_mismatch["fingerprint"],
            node_mismatch["aggregate"],
            node_mismatch["cases"],
        )["planning_route_evidence_complete"])

        plan_success_mismatch = copy.deepcopy(run)
        plan_success_mismatch["aggregate"]["planning_route"][
            "plan_node_successes"
        ] = 1
        self.assertFalse(pipeline_evidence_checks(
            plan_success_mismatch["fingerprint"],
            plan_success_mismatch["aggregate"],
            plan_success_mismatch["cases"],
        )["planning_route_evidence_complete"])

        model_call_mismatch = copy.deepcopy(run)
        model_call_mismatch["aggregate"]["planning_route"][
            "physical_llm_attempts"
        ] = 1
        self.assertFalse(pipeline_evidence_checks(
            model_call_mismatch["fingerprint"],
            model_call_mismatch["aggregate"],
            model_call_mismatch["cases"],
        )["planning_route_evidence_complete"])

        route_count_mismatch = copy.deepcopy(run)
        route_count_mismatch["aggregate"]["planning_route"][
            "route_counts"
        ] = {"plan_then_generate": 2}
        route_count_checks = pipeline_evidence_checks(
            route_count_mismatch["fingerprint"],
            route_count_mismatch["aggregate"],
            route_count_mismatch["cases"],
        )
        self.assertFalse(route_count_checks["planning_case_execution_bound"])
        self.assertFalse(route_count_checks["planning_route_evidence_complete"])

        plan_count_mismatch = copy.deepcopy(run)
        plan_count_mismatch["aggregate"]["planning_route"].update({
            "plan_calls": 1,
            "observed_plan_node_calls": 1,
            "plan_node_successes": 1,
        })
        plan_count_checks = pipeline_evidence_checks(
            plan_count_mismatch["fingerprint"],
            plan_count_mismatch["aggregate"],
            plan_count_mismatch["cases"],
        )
        self.assertFalse(plan_count_checks["planning_case_execution_bound"])
        self.assertFalse(plan_count_checks["planning_route_evidence_complete"])

        selected_aggregate_mismatch = copy.deepcopy(run)
        selected_aggregate_mismatch["aggregate"]["semantic_repair"][
            "selected_invocation_count"
        ] = 1
        self.assertFalse(pipeline_evidence_checks(
            selected_aggregate_mismatch["fingerprint"],
            selected_aggregate_mismatch["aggregate"],
            selected_aggregate_mismatch["cases"],
        )["semantic_repair_evidence_complete"])

        selected_case_mismatch = copy.deepcopy(run)
        selected_case_mismatch["cases"][0]["stage_evidence"][
            "semantic_repair"
        ]["selected_invocation_count"] = 2
        self.assertFalse(pipeline_evidence_checks(
            selected_case_mismatch["fingerprint"],
            selected_case_mismatch["aggregate"],
            selected_case_mismatch["cases"],
        )["semantic_repair_evidence_complete"])

        profile_mismatch = copy.deepcopy(run)
        profile_mismatch["fingerprint"]["pipeline_profiles"][
            "planning_route"
        ]["active_profile"] = "weighted_scorecard_v1"
        self.assertFalse(pipeline_evidence_checks(
            profile_mismatch["fingerprint"],
            profile_mismatch["aggregate"],
            profile_mismatch["cases"],
        )["planning_route_evidence_complete"])

        run["baseline_eligible"] = False
        checklist = baseline_checklist(run, 2)
        self.assertFalse(checklist["eligible"])
        self.assertFalse(checklist["checks"]["release_artifact"])

    def test_k2_domain_consumption_cannot_be_coordinately_erased(self):
        previous_profile = app_config.query_signals.profile
        previous_policy = app_config.query_signals.route_policy
        try:
            app_config.query_signals.profile = "domain_signals_v1"
            app_config.query_signals.route_policy = "routed_signals_v1"
            run = self._experiment_run(1, total=2)
        finally:
            app_config.query_signals.profile = previous_profile
            app_config.query_signals.route_policy = previous_policy

        self.assertTrue(all(pipeline_evidence_checks(
            run["fingerprint"], run["aggregate"], run["cases"],
        ).values()))
        self.assertGreater(
            run["aggregate"]["query_signals"]["extractor_kind_counts"].get(
                "domain_terms", 0,
            ),
            0,
        )

        forged = copy.deepcopy(run)
        for case in forged["cases"]:
            evidence = case["stage_evidence"]["query_signals"]
            evidence["extractor_signals"] = [
                signal for signal in evidence["extractor_signals"]
                if signal.get("kind") != "domain_terms"
            ]
            evidence["extractor_signal_identity_fingerprint"] = stable_hash(
                evidence["extractor_signals"]
            )
            evidence["extractor_kind_counts"] = dict(sorted(Counter(
                str(signal.get("kind") or "unknown")
                for signal in evidence["extractor_signals"]
            ).items()))
            for route in evidence["routes"]:
                for consumer, declared_field in (
                    ("local", "local_signal_count"),
                    ("remote", "remote_query_count"),
                    ("rerank", "rerank_signal_count"),
                ):
                    field = f"actual_{consumer}_signals"
                    route[field] = [
                        signal for signal in route[field]
                        if signal.get("kind") != "domain_terms"
                    ]
                    route[declared_field] = len(route[field])
                    route[f"actual_{consumer}_signal_count"] = len(
                        route[field]
                    )
                    route[f"{consumer}_signal_fingerprint"] = stable_hash(
                        route[field]
                    )
                route["consumer_kind_counts"] = {
                    consumer: dict(sorted(Counter(
                        str(signal.get("kind") or "unknown")
                        for signal in route[f"actual_{consumer}_signals"]
                    ).items()))
                    for consumer in ("local", "remote", "rerank")
                }
                route["consumer_plan_fingerprint"] = stable_signal_hash({
                    "contract": evidence["contract_version"],
                    "route_policy_version": route["route_policy_version"],
                    "route_policy": route["route_policy"],
                    "route": route["route"],
                    "profile_fingerprint": route["profile_fingerprint"],
                    "signal_fingerprint": route["signal_fingerprint"],
                    "query_encode_mode": route["query_encode_mode"],
                    "local": route["actual_local_signals"],
                    "remote": route["actual_remote_signals"],
                    "rerank": route["actual_rerank_signals"],
                })

        query_aggregate = forged["aggregate"]["query_signals"]
        query_aggregate["extractor_kind_counts"] = dict(sorted(Counter(
            str(signal.get("kind") or "unknown")
            for case in forged["cases"]
            for signal in case["stage_evidence"]["query_signals"][
                "extractor_signals"
            ]
        ).items()))
        consumer_counts: Counter[str] = Counter()
        consumer_kind_counts: Counter[str] = Counter()
        remote_counts: Counter[str] = Counter()
        for case in forged["cases"]:
            routes = case["stage_evidence"]["query_signals"]["routes"]
            for route in routes:
                route_name = str(route["route"])
                remote_counts[route_name] += int(
                    route["actual_remote_signal_count"]
                )
                for consumer in ("local", "remote", "rerank"):
                    consumer_counts[f"{route_name}:{consumer}"] += int(
                        route[f"actual_{consumer}_signal_count"]
                    )
                    consumer_kind_counts.update(
                        f"{route_name}:{consumer}:"
                        f"{str(signal.get('kind') or 'unknown')}"
                        for signal in route[f"actual_{consumer}_signals"]
                    )
        query_aggregate["actual_remote_signal_counts"] = dict(
            sorted(remote_counts.items())
        )
        query_aggregate["actual_consumer_signal_counts"] = dict(
            sorted(consumer_counts.items())
        )
        query_aggregate["consumer_kind_counts"] = dict(
            sorted(consumer_kind_counts.items())
        )

        checks = pipeline_evidence_checks(
            forged["fingerprint"], forged["aggregate"], forged["cases"],
        )
        self.assertFalse(checks["query_signal_routes_accounted"])
        self.assertFalse(checks["query_signal_evidence_complete"])

    def test_pipeline_evidence_rejects_direct_route_with_plan_call(self):
        run = self._experiment_run(1, total=2)
        route = run["cases"][0]["stage_evidence"]["planning_route"]
        route.update({
            "route": "direct_generate",
            "plan_called": True,
            "plan_source": "model",
            "plan_reason_code": "requested",
        })
        planning = run["aggregate"]["planning_route"]
        planning.update({
            "route_counts": {"direct_generate": 2},
            "plan_calls": 1,
            "observed_plan_node_calls": 1,
            "plan_node_successes": 1,
        })

        checks = pipeline_evidence_checks(
            run["fingerprint"], run["aggregate"], run["cases"],
        )
        self.assertFalse(checks["planning_route_evidence_complete"], checks)

    def test_pipeline_evidence_rejects_plan_route_without_plan_call(self):
        run = self._experiment_run(1, total=2)
        run["cases"][0]["stage_evidence"]["planning_route"]["route"] = (
            "plan_then_generate"
        )
        run["aggregate"]["planning_route"]["route_counts"] = {
            "direct_generate": 1,
            "plan_then_generate": 1,
        }

        checks = pipeline_evidence_checks(
            run["fingerprint"], run["aggregate"], run["cases"],
        )
        self.assertFalse(checks["planning_route_evidence_complete"], checks)

    def test_pipeline_evidence_rejects_plan_without_source_or_reason(self):
        run = self._experiment_run(1, total=2)
        route = run["cases"][0]["stage_evidence"]["planning_route"]
        route.update({
            "route": "plan_then_generate",
            "plan_called": True,
            "plan_source": "",
            "plan_reason_code": "",
        })
        planning = run["aggregate"]["planning_route"]
        planning.update({
            "route_counts": {
                "direct_generate": 1,
                "plan_then_generate": 1,
            },
            "plan_calls": 1,
            "observed_plan_node_calls": 1,
            "plan_node_successes": 1,
        })

        checks = pipeline_evidence_checks(
            run["fingerprint"], run["aggregate"], run["cases"],
        )
        self.assertFalse(checks["planning_route_evidence_complete"], checks)

    def test_pipeline_evidence_rejects_injected_evidence_failures(self):
        run = self._experiment_run(1, total=2)
        run["cases"][0]["stage_evidence"]["planning_route"].update({
            "evidence_complete": True,
            "evidence_failures": ["forged_failure"],
        })

        checks = pipeline_evidence_checks(
            run["fingerprint"], run["aggregate"], run["cases"],
        )
        self.assertFalse(checks["planning_route_evidence_complete"], checks)

    def test_policy_abstain_fallback_is_bound_to_cases_aggregate_and_fingerprint(self):
        run = self._experiment_run(1, total=2)
        planning_profile = "weighted_scorecard_v1"
        fallback_profile = "keyword_length_v1"
        fingerprint_planning = run["fingerprint"]["pipeline_profiles"][
            "planning_route"
        ]
        fingerprint_planning["active_profile"] = planning_profile
        fingerprint_planning["fallback_profile"] = fallback_profile

        for case in run["cases"]:
            case["stage_evidence"]["planning_route"].update({
                "configured_profile": planning_profile,
                "runtime_profile": planning_profile,
                "active_profile": planning_profile,
                "decision": "abstain",
                "route": "direct_generate",
                "fallback_profile": fallback_profile,
                "fallback_applied": True,
                "fallback_class": "policy_abstain",
                "policy_fallback_eligible": True,
                "degraded": False,
                "reason_codes": ["score:signal_complexity", "abstain_fallback"],
                "evidence_complete": True,
            })
        planning = run["aggregate"]["planning_route"]
        planning.update({
            "active_profile_counts": {planning_profile: 2},
            "configured_profile_counts": {planning_profile: 2},
            "fallback_profile_counts": {fallback_profile: 2},
            "fallback_class_counts": {"policy_abstain": 2},
            "fallback_count": 2,
            "policy_fallback_count": 2,
            "unsafe_fallback_count": 0,
            "evidence_complete": True,
        })

        checks = pipeline_evidence_checks(
            run["fingerprint"], run["aggregate"], run["cases"],
        )
        self.assertTrue(checks["planning_route_evidence_complete"], checks)
        self.assertTrue(checks["planning_policy_fallbacks_bound"], checks)

        wrong_fingerprint = copy.deepcopy(run)
        wrong_fingerprint["fingerprint"]["pipeline_profiles"][
            "planning_route"
        ]["fallback_profile"] = "always_plan"
        wrong_checks = pipeline_evidence_checks(
            wrong_fingerprint["fingerprint"],
            wrong_fingerprint["aggregate"],
            wrong_fingerprint["cases"],
        )
        self.assertFalse(wrong_checks["planning_route_evidence_complete"])
        self.assertFalse(wrong_checks["planning_policy_fallbacks_bound"])

        forged_degraded = copy.deepcopy(run)
        forged_degraded["cases"][0]["stage_evidence"][
            "planning_route"
        ].update({
            "fallback_class": "exception_degraded",
            "policy_fallback_eligible": False,
            "degraded": True,
        })
        forged_checks = pipeline_evidence_checks(
            forged_degraded["fingerprint"],
            forged_degraded["aggregate"],
            forged_degraded["cases"],
        )
        self.assertFalse(forged_checks["planning_route_evidence_complete"])

    def test_recovery_tail_policy_changes_the_safe_config_fingerprint(self):
        previous = app_config.query.llm_recovery_reserve_ms
        fingerprint_before = stable_hash(_safe_config_manifest())
        try:
            app_config.query.llm_recovery_reserve_ms = 0
            self.assertNotEqual(
                fingerprint_before,
                stable_hash(_safe_config_manifest()),
            )
        finally:
            app_config.query.llm_recovery_reserve_ms = previous

    def test_stage_evidence_has_reconstructable_chain_without_result_values(self):
        evidence = _stage_evidence({
            "recall_columns": [{"id": "fact_order.order_amount", "table_id": "fact_order", "score": 0.91}],
            "recall_metrics": [{"id": "gmv", "name": "GMV", "score": 0.88}],
            "table_infos": [{"name": "fact_order", "columns": [{"name": "order_amount"}]}],
            "authorized_sql": "SELECT SUM(order_amount) AS gmv FROM fact_order",
            "result": [{"gmv": "123456.78"}],
        }, ["SELECT amount FROM fact_order", "SELECT SUM(order_amount) AS gmv FROM fact_order"])
        self.assertEqual(evidence["first_sql"], "SELECT amount FROM fact_order")
        self.assertEqual(len(evidence["correction_sqls"]), 1)
        self.assertTrue(evidence["result_summary"]["masked"])
        self.assertFalse(evidence["result_summary"]["values_persisted"])
        self.assertEqual(evidence["context_columns"], ["fact_order.order_amount"])
        self.assertNotIn("123456.78", json.dumps(evidence))

    def test_baseline_checklist_rejects_incomplete_current_run(self):
        run = self._experiment_run(1, total=2)
        self.assertTrue(baseline_checklist(run, 2)["eligible"])
        blocked = {**run, "run_id": "run-2", "fingerprint": {}}
        checklist = baseline_checklist(blocked, 2)
        self.assertFalse(checklist["eligible"])
        self.assertFalse(checklist["checks"]["fingerprint_complete"])
        run["cases"][0]["stage_evidence"]["planning_route"][
            "evidence_complete"
        ] = False
        evidence_blocked = baseline_checklist(run, 2)
        self.assertFalse(evidence_blocked["eligible"])
        self.assertFalse(
            evidence_blocked["checks"]["planning_route_evidence_complete"]
        )

    def test_llm_release_evidence_is_recomputed_from_raw_counts(self):
        scenarios = {
            "failed_core_llm_calls": "failed_llm_calls_zero",
            "llm_attempts_timed_out": "llm_attempts_timed_out_zero",
        }
        for raw_field, check_name in scenarios.items():
            with self.subTest(raw_field=raw_field):
                runs = [self._experiment_run(index) for index in (1, 2, 3)]
                runs[0]["baseline_eligible"] = True
                runs[0]["aggregate"][raw_field] = 1

                checklist = baseline_checklist(runs[0], 100)
                self.assertFalse(checklist["eligible"])
                self.assertFalse(checklist["checks"][check_name])

                report = experiment_report(
                    runs, experiment_id="experiment-001",
                )
                self.assertFalse(report["checks"][check_name])
                self.assertFalse(report["complete"])
                self.assertFalse(report["recordable"])
                self.assertFalse(report["qualified"])
                self.assertIn(check_name, report["failed_checks"])

    def test_three_run_experiment_uses_worst_value_and_can_qualify(self):
        runs = [self._experiment_run(index) for index in (1, 2, 3)]
        baseline = {
            "run_id": "approved", "dataset_suite": "easy_sql_v1",
            "aggregate": {"result_equivalent": 1},
            "cases": [{"id": "eval_001", "result_equivalent": True}],
        }
        report = experiment_report(runs, baseline_run=baseline, experiment_id="experiment-001")
        self.assertTrue(report["complete"], report["failed_checks"])
        self.assertTrue(report["qualified"], report["failed_checks"])
        self.assertEqual(report["repeat_count"], 3)
        self.assertEqual(report["worst"]["tt_complete_p95"], 1500)
        self.assertEqual(report["regressed_case_ids"], [])

    def test_fresh_embedding_session_evidence_is_governed_across_repeats(self):
        runs = [self._experiment_run(index) for index in (1, 2, 3)]
        clean = experiment_report(runs, experiment_id="experiment-001")
        self.assertEqual(clean["embedding_protocol_mode"], "fresh")
        self.assertTrue(clean["checks"]["embedding_protocol_declared_complete"])
        self.assertTrue(clean["checks"]["embedding_warmup_evidence_complete"])
        self.assertTrue(clean["checks"]["embedding_runtime_health_complete"])
        self.assertTrue(clean["checks"]["embedding_session_protocol"])
        self.assertTrue(clean["checks"]["embedding_epoch_protocol"])
        self.assertTrue(clean["checks"]["observed_cache_protocol"])
        self.assertTrue(clean["complete"], clean["failed_checks"])

        runs[1]["embedding_warmup"]["session_id"] = "new-process-session"
        blocked = experiment_report(runs, experiment_id="experiment-001")
        self.assertFalse(blocked["checks"]["embedding_session_protocol"])
        self.assertFalse(blocked["complete"])
        self.assertIn("embedding_session_protocol", blocked["failed_checks"])

    def test_embedding_governance_recomputes_raw_evidence(self):
        scenarios = {
            "forged_epoch": lambda runs: runs[1]["embedding_warmup"][
                "metrics_epoch_reset"
            ]["active_epoch"].update({"id": "forged"}),
            "forged_generation": lambda runs: runs[1]["cache_control"].update({
                "retrieval_cache_generation_after": 3,
            }),
            "typed_retrieval_failure": lambda runs: runs[1][
                "embedding_runtime_health"
            ]["typed_core_retrieval_failures"].append({
                "case_id": "eval_001",
                "node_id": "recall_column",
                "reason_code": "timeout",
            }),
            "raw_case_retrieval_failure": lambda runs: runs[1]["cases"][0].setdefault(
                "degradations", []
            ).append({
                "node_id": "recall_column",
                "reason_code": "timeout",
                "affects_core_path": True,
            }),
            "scheduler_failure": lambda runs: runs[1][
                "embedding_runtime_final"
            ]["requests"].update({"cancelled": 1}),
            "final_check_failure": lambda runs: runs[1][
                "embedding_runtime_final"
            ]["checks"].update({"queue_idle": False}),
        }
        for name, mutate in scenarios.items():
            with self.subTest(name=name):
                runs = [self._experiment_run(index) for index in (1, 2, 3)]
                mutate(runs)
                # 即便派生字段仍声称通过，raw 异常也必须阻断。
                runs[1]["embedding_warmup_checks"]["complete"] = True
                runs[1]["embedding_runtime_health"]["eligible"] = True
                report = experiment_report(runs, experiment_id="experiment-001")
                self.assertFalse(report["complete"])
                self.assertTrue(report["failed_checks"])

    def test_embedding_runtime_requires_full_case_coverage_and_closed_requests(self):
        clean_runs = [self._experiment_run(index) for index in (1, 2, 3)]
        clean = experiment_report(clean_runs, experiment_id="experiment-001")
        self.assertTrue(clean["checks"]["embedding_runtime_health_complete"])

        scenarios = {}
        shortened = copy.deepcopy(clean_runs)
        shortened[0]["cases"] = shortened[0]["cases"][:1]
        shortened[0]["embedding_runtime_health"]["observed_epoch_count"] = 1
        scenarios["declared_100_but_only_one_case"] = shortened

        uncovered = copy.deepcopy(clean_runs)
        uncovered[0]["cases"][0]["stage_evidence"]["retrieval_budget"] = []
        uncovered[0]["embedding_runtime_health"]["observed_epoch_count"] -= 1
        scenarios["case_without_epoch_evidence"] = uncovered

        missing_counts = copy.deepcopy(clean_runs)
        requests = missing_counts[0]["embedding_runtime_final"]["requests"]
        requests.pop("submitted")
        requests.pop("completed")
        scenarios["missing_submitted_and_completed"] = missing_counts

        for name, runs in scenarios.items():
            with self.subTest(name=name):
                report = experiment_report(runs, experiment_id="experiment-001")
                self.assertFalse(
                    report["checks"]["embedding_runtime_health_complete"]
                )
                self.assertFalse(report["complete"])

    def test_embedding_schema_must_be_all_fresh_or_all_legacy(self):
        mixed = [self._experiment_run(index) for index in (1, 2, 3)]
        mixed[1]["embedding_warmup"].pop("schema")
        mixed_report = experiment_report(mixed, experiment_id="experiment-001")
        self.assertEqual(
            mixed_report["embedding_protocol_mode"],
            "invalid_mixed_or_unknown",
        )
        self.assertFalse(
            mixed_report["checks"]["embedding_protocol_declared_complete"]
        )
        self.assertFalse(mixed_report["complete"])

        legacy = [self._experiment_run(index) for index in (1, 2, 3)]
        for run in legacy:
            run.pop("embedding_warmup")
            run.pop("embedding_warmup_checks")
            run.pop("embedding_runtime_final")
            run.pop("embedding_runtime_health")
        legacy_report = experiment_report(legacy, experiment_id="experiment-001")
        self.assertEqual(
            legacy_report["embedding_protocol_mode"], "legacy_observational"
        )
        self.assertFalse(
            legacy_report["checks"]["embedding_protocol_declared_complete"]
        )
        self.assertFalse(legacy_report["complete"])

    def test_enabled_semantic_review_has_independent_release_gates(self):
        runs = [self._experiment_run(index) for index in (1, 2, 3)]
        self._enable_semantic_review(runs)

        report = experiment_report(runs, experiment_id="experiment-001")

        self.assertTrue(report["qualified"], report["failed_checks"])
        for key in (
            "semantic_review_policy_fingerprinted",
            "semantic_review_evidence_complete",
            "semantic_review_at_most_one_attempt",
            "semantic_review_safety_gate",
            "semantic_review_failed_core_zero",
            "semantic_review_harmful_modification_zero",
            "semantic_review_first_correct_final_incorrect_zero",
            "semantic_review_same_sql_execution_loss_zero",
            "semantic_review_selected_b_no_sql_delta_zero",
            "semantic_review_result_attribution_complete",
            "semantic_review_coverage_complete",
            "semantic_review_budget_skip_zero",
        ):
            with self.subTest(key=key):
                self.assertTrue(report["checks"][key])

    def test_each_semantic_review_failure_blocks_promotion(self):
        scenarios = {
            "semantic_review_evidence_complete": (
                "semantic_review_evidence_complete", False,
            ),
            "semantic_review_at_most_one_attempt": (
                "semantic_review_at_most_one_attempt", False,
            ),
            "semantic_review_safety_gate": (
                "semantic_review_unsafe_selected", 1,
            ),
            "semantic_review_failed_core_zero": (
                "semantic_review_failed_core", 1,
            ),
            "semantic_review_harmful_modification_zero": (
                "semantic_review_harmful_modification", 1,
            ),
            "semantic_review_first_correct_final_incorrect_zero": (
                "semantic_review_first_correct_final_incorrect", 1,
            ),
            "semantic_review_same_sql_execution_loss_zero": (
                "semantic_review_same_sql_execution_loss", 1,
            ),
            "semantic_review_selected_b_no_sql_delta_zero": (
                "semantic_review_selected_b_no_sql_delta", 1,
            ),
            "semantic_review_coverage_complete": (
                "semantic_review_coverage_complete", False,
            ),
        }
        for expected_gate, (aggregate_key, value) in scenarios.items():
            with self.subTest(gate=expected_gate):
                runs = [
                    self._experiment_run(index) for index in (1, 2, 3)
                ]
                self._enable_semantic_review(runs)
                runs[1]["aggregate"][aggregate_key] = value
                if expected_gate == "semantic_review_coverage_complete":
                    runs[1]["aggregate"]["semantic_review_budget_skipped"] = 1

                report = experiment_report(
                    runs, experiment_id="experiment-001",
                )

                self.assertFalse(report["checks"][expected_gate])
                self.assertFalse(report["qualified"])

    def test_legacy_semantic_reason_is_readable_without_polluting_adaptive_gate(self):
        runs = [self._experiment_run(index) for index in (1, 2, 3)]
        for run in runs:
            run["cases"][0]["stage_evidence"]["candidate_decision"] = {
                "adaptive_triggered": True,
                "extra_call_used": True,
                "trigger_reason": "semantic_review_required",
                "selection_reason": "semantic_review_validated",
                "selected": "B",
                "candidates": [
                    {"label": "A", "valid": True, "band": "high"},
                    {"label": "B", "valid": True, "band": "high"},
                ],
            }

        report = experiment_report(runs, experiment_id="experiment-001")

        self.assertTrue(report["checks"]["candidate_trigger_gate"])
        self.assertTrue(report["checks"]["candidate_safety_gate"])
        self.assertFalse(
            report["checks"]["semantic_review_policy_fingerprinted"]
        )
        self.assertFalse(report["qualified"])

    def test_suite_result_floors_are_strict_and_suite_sized(self):
        scenarios = (
            ("easy_sql_v1", 96, 95),
            ("medium_sql_v1", 86, 85),
        )
        self.assertEqual(PROMOTION_SUITE_POLICIES["easy_sql_v1"]["case_count"], 100)
        self.assertEqual(PROMOTION_SUITE_POLICIES["medium_sql_v1"]["case_count"], 100)
        for suite, required, below in scenarios:
            with self.subTest(suite=suite):
                self.assertEqual(minimum_passing_results(suite, 100), required)
                self.assertTrue(
                    suite_result_floor_met(suite, passed=required, total=100)
                )
                self.assertFalse(
                    suite_result_floor_met(suite, passed=below, total=100)
                )
                self.assertFalse(
                    suite_result_floor_met(suite, passed=required, total=1)
                )
        self.assertIsNone(minimum_passing_results("hard_sql_v1", 50))
        self.assertTrue(
            suite_result_floor_met("hard_sql_v1", passed=0, total=50)
        )

    def test_every_repeat_must_meet_absolute_suite_result_floor(self):
        for suite, required in (("easy_sql_v1", 96), ("medium_sql_v1", 86)):
            with self.subTest(suite=suite, outcome="one_repeat_below"):
                runs = [
                    self._experiment_run(
                        index,
                        dataset_suite=suite,
                        result_count=count,
                    )
                    for index, count in enumerate(
                        (required, required, required - 1), start=1
                    )
                ]
                report = experiment_report(
                    runs, experiment_id="experiment-001",
                )
                self.assertTrue(report["complete"], report["failed_checks"])
                self.assertFalse(report["checks"]["absolute_result_floor"])
                self.assertFalse(report["checks"]["result_floor_non_regression"])
                self.assertFalse(report["qualified"])
                self.assertEqual(
                    [row["met"] for row in report["promotion_policy"]["per_run"]],
                    [True, True, False],
                )

            with self.subTest(suite=suite, outcome="all_repeats_meet"):
                runs = [
                    self._experiment_run(
                        index,
                        dataset_suite=suite,
                        result_count=required,
                    )
                    for index in (1, 2, 3)
                ]
                report = experiment_report(
                    runs, experiment_id="experiment-001",
                )
                self.assertTrue(report["checks"]["absolute_result_floor"])
                self.assertTrue(report["checks"]["result_floor_non_regression"])
                self.assertTrue(report["qualified"], report["failed_checks"])

    def test_same_suite_baseline_non_regression_is_additive_to_absolute_floor(self):
        for suite, required in (("easy_sql_v1", 96), ("medium_sql_v1", 86)):
            with self.subTest(suite=suite):
                runs = [
                    self._experiment_run(
                        index,
                        dataset_suite=suite,
                        result_count=required,
                    )
                    for index in (1, 2, 3)
                ]
                baseline = self._experiment_run(
                    1,
                    dataset_suite=suite,
                    result_count=required + 1,
                )
                baseline["run_id"] = "approved"

                report = experiment_report(
                    runs,
                    baseline_run=baseline,
                    experiment_id="experiment-001",
                )

                self.assertTrue(report["checks"]["absolute_result_floor"])
                self.assertFalse(report["checks"]["baseline_non_regression"])
                self.assertFalse(
                    report["checks"]["baseline_result_non_regression"]
                )
                self.assertFalse(report["checks"]["result_floor_non_regression"])
                self.assertFalse(report["qualified"])

    def test_hard_suite_has_no_result_floor_but_keeps_protocol_gates(self):
        runs = [
            self._experiment_run(
                index,
                dataset_suite="hard_sql_v1",
                result_count=0,
                total=50,
            )
            for index in (1, 2, 3)
        ]
        report = experiment_report(runs, experiment_id="experiment-001")
        self.assertTrue(report["checks"]["absolute_result_floor"])
        self.assertTrue(report["qualified"], report["failed_checks"])

        runs[1]["aggregate"]["infrastructure_errors"] = 1
        blocked = experiment_report(runs, experiment_id="experiment-001")
        self.assertFalse(blocked["checks"]["infrastructure_clean"])
        self.assertFalse(blocked["complete"])
        self.assertFalse(blocked["qualified"])

    def test_experiment_rejects_fingerprint_drift_and_baseline_regression(self):
        runs = [self._experiment_run(1), self._experiment_run(2), self._experiment_run(3, result=False, prompt_hash="changed")]
        baseline = {
            "run_id": "approved", "dataset_suite": "easy_sql_v1",
            "aggregate": {"result_equivalent": 1},
            "cases": [{"id": "eval_001", "result_equivalent": True}],
        }
        report = experiment_report(runs, baseline_run=baseline, experiment_id="experiment-001")
        self.assertFalse(report["complete"])
        self.assertFalse(report["qualified"])
        self.assertFalse(report["checks"]["fingerprints_comparable"])
        self.assertFalse(report["checks"]["baseline_non_regression"])
        self.assertEqual(report["regressed_case_ids"], ["eval_001"])

    def test_unknown_suite_fails_promotion_gate(self):
        runs = [self._experiment_run(index) for index in (1, 2, 3)]
        for run in runs:
            run["dataset_suite"] = "independent-candidate-suite"
            run["cases"][0]["id"] = "candidate_001"

        report = experiment_report(
            runs,
            experiment_id="experiment-001",
        )

        self.assertTrue(report["complete"], report["failed_checks"])
        self.assertFalse(report["qualified"])
        self.assertFalse(report["checks"]["dataset_suite_supported"])
        self.assertIn("dataset_suite_supported", report["failed_checks"])

    def test_experiment_never_mixes_core_and_enhanced_profiles(self):
        runs = [
            self._experiment_run(1), self._experiment_run(2),
            self._experiment_run(3, evaluation_profile="enhanced-analysis"),
        ]
        baseline = {
            "dataset_suite": "easy_sql_v1",
            "aggregate": {"result_equivalent": 1},
            "cases": [{"id": "eval_001", "result_equivalent": True}],
        }
        report = experiment_report(runs, baseline_run=baseline, experiment_id="experiment-001")
        self.assertFalse(report["checks"]["same_evaluation_profile"])
        self.assertFalse(report["complete"])

    def test_route_protocol_rejects_single_case_token_spike(self):
        runs = [self._experiment_run(index) for index in (1, 2, 3)]
        for run in runs:
            run["fingerprint"]["model"] = {"routes": {"sql": {"model_name": "strong"}}}
            run["aggregate"].update({
                "prompt_payloads": 1, "failed_llm_calls": 0,
                "structured_output_fallbacks": 0,
                "adaptive_candidate_rate": 0,
                "candidate_unsafe_selected": 0,
                "candidate_high_confidence_extra_calls": 0,
            })
        runs[0]["aggregate"].update({"tokens_max": 98_000, "tokens_max_case_id": "eval_028"})
        baseline = {
            "run_id": "approved", "dataset_suite": "easy_sql_v1",
            "aggregate": {"result_equivalent": 1},
            "cases": [{"id": "eval_001", "result_equivalent": True}],
        }
        report = experiment_report(runs, baseline_run=baseline, experiment_id="experiment-001")
        self.assertTrue(report["complete"], report["failed_checks"])
        self.assertFalse(report["qualified"])
        self.assertFalse(report["checks"]["token_max_gate"])
        self.assertEqual(report["worst"]["tokens_max"], 98_000)
        self.assertEqual(report["worst"]["tokens_max_case_ids"], ["eval_028"])

    def test_route_protocol_accepts_complete_deterministic_zero_llm_runs(self):
        runs = [self._experiment_run(index) for index in (1, 2, 3)]
        for run in runs:
            run["fingerprint"]["model"] = {"routes": {"sql": {"model_name": "strong"}}}
            run["aggregate"].update({
                "llm_calls_p95": 0,
                "llm_attempts": 0,
                "llm_outcome_classes": {},
                "prompt_payloads": 0,
                "tokens_mean": 0,
                "tokens_p95": 0,
                "tokens_max": 0,
                "tokens_max_case_id": "",
                "failed_llm_calls": 0,
                "structured_output_fallbacks": 0,
                "adaptive_candidate_rate": 0,
                "candidate_unsafe_selected": 0,
                "candidate_high_confidence_extra_calls": 0,
            })
            run["cases"][0]["total_tokens"] = 0
        baseline = {
            "run_id": "approved", "dataset_suite": "easy_sql_v1",
            "aggregate": {"result_equivalent": 1},
            "cases": [{"id": "eval_001", "result_equivalent": True}],
        }
        report = experiment_report(runs, baseline_run=baseline, experiment_id="experiment-001")
        self.assertTrue(report["complete"], report["failed_checks"])
        self.assertTrue(report["qualified"], report["failed_checks"])
        self.assertTrue(report["checks"]["prompt_compression_evidence"])
        self.assertTrue(report["checks"]["token_max_gate"])
        self.assertEqual(report["worst"]["tokens_max_case_ids"], [])

    def test_difficulty_observation_is_recordable_without_passing_release_quality_gates(self):
        runs = [self._experiment_run(index, result=False) for index in (1, 2, 3)]
        for run in runs:
            run["dataset_suite"] = "easy_sql_v1"
            run["aggregate"]["tables_correct"] = 0
            run["aggregate"]["tt_complete_p95"] = 120_000

        report = experiment_report(
            runs, experiment_id="experiment-001",
        )

        self.assertTrue(report["complete"], report["failed_checks"])
        self.assertTrue(report["recordable"], report["failed_checks"])
        self.assertFalse(report["qualified"])
        self.assertEqual(report["status"], "recordable")
        self.assertFalse(report["checks"]["table_exact_non_regression"])
        self.assertFalse(report["checks"]["ttcomplete_gate"])

    def test_table_exact_is_observational_but_scope_safety_is_blocking(self):
        runs = [self._experiment_run(index) for index in (1, 2, 3)]
        for run in runs:
            run["aggregate"]["tables_correct"] = 99
            run["aggregate"]["table_scope_safe"] = 100

        observed = experiment_report(
            runs, experiment_id="experiment-001",
        )

        self.assertFalse(observed["checks"]["table_exact_non_regression"])
        self.assertTrue(observed["checks"]["table_scope_safety"])
        self.assertTrue(observed["qualified"], observed["failed_checks"])
        self.assertIn(
            "table_exact_non_regression", observed["failed_observations"]
        )
        self.assertNotIn(
            "table_exact_non_regression", observed["failed_checks"]
        )
        self.assertIn(
            "table_exact_non_regression",
            observed["check_accounting"]["observational"],
        )

        runs[1]["aggregate"]["table_scope_safe"] = 99
        blocked = experiment_report(
            runs, experiment_id="experiment-001",
        )
        self.assertFalse(blocked["checks"]["table_scope_safety"])
        self.assertFalse(blocked["qualified"])
        self.assertIn("table_scope_safety", blocked["failed_checks"])

    def test_security_guard_integrity_is_independently_blocking(self):
        runs = [self._experiment_run(index) for index in (1, 2, 3)]

        clean = experiment_report(runs, experiment_id="experiment-001")
        self.assertTrue(clean["checks"]["security_guard_integrity"])
        self.assertTrue(clean["qualified"], clean["failed_checks"])

        runs[1]["aggregate"]["security_guard_bypass_count"] = 1
        blocked = experiment_report(runs, experiment_id="experiment-001")
        self.assertFalse(blocked["checks"]["security_guard_integrity"])
        self.assertFalse(blocked["qualified"])
        self.assertIn("security_guard_integrity", blocked["failed_checks"])

    def test_legacy_table_exact_proxy_cannot_satisfy_scope_evidence(self):
        runs = [self._experiment_run(index) for index in (1, 2, 3)]
        for run in runs:
            run["aggregate"].pop("table_scope_evidence_complete")
            run["aggregate"]["tables_correct"] = 100
            run["aggregate"]["table_scope_safe"] = 100

        report = experiment_report(runs, experiment_id="experiment-001")

        self.assertFalse(report["checks"]["table_scope_evidence_complete"])
        self.assertFalse(report["complete"])
        self.assertFalse(report["qualified"])
        self.assertIn("table_scope_evidence_complete", report["failed_checks"])

    def test_cross_suite_baseline_is_never_used_for_case_regression(self):
        runs = [self._experiment_run(index) for index in (1, 2, 3)]
        for run in runs:
            run["dataset_suite"] = "easy_sql_v1"
            run["cases"][0]["result_equivalent"] = False
            run["aggregate"]["result_equivalent"] = 0
        baseline = {
            "run_id": "medium-approved",
            "dataset_suite": "medium_sql_v1",
            "aggregate": {"result_equivalent": 1},
            "cases": [{"id": "eval_001", "result_equivalent": True}],
        }

        report = experiment_report(
            runs, baseline_run=baseline, experiment_id="experiment-001",
        )

        self.assertFalse(report["baseline"]["comparison_applicable"])
        self.assertEqual(report["baseline"]["suite"], "medium_sql_v1")
        self.assertEqual(report["baseline"]["candidate_suite"], "easy_sql_v1")
        self.assertEqual(report["regressed_case_ids"], [])

    def test_baseline_resolution_uses_explicit_or_meta_approved_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            runs.mkdir()
            approved_path = runs / "run_approved.json"
            approved_path.write_text(json.dumps({
                "run_id": "approved", "workspace_id": "workspace-a",
                "valid": True, "dataset_suite": "easy_sql_v1",
                "fingerprint": {"fingerprint_schema": "evaluation-v2"},
                "aggregate": {},
            }), encoding="utf-8")
            explicit_path = runs / "run_explicit.json"
            explicit_path.write_text(json.dumps({
                "run_id": "explicit", "workspace_id": "workspace-a",
                "valid": True, "dataset_suite": "easy_sql_v1",
                "fingerprint": {"fingerprint_schema": "evaluation-v2"},
                "aggregate": {},
            }), encoding="utf-8")
            approved = {"artifact": approved_path.name, "status": "approved"}
            selected = resolve_baseline_artifact(
                explicit=explicit_path, workspace_id="workspace-a", approved_entry=approved,
                runs_dir=runs,
            )
            self.assertEqual(selected["run_id"], "explicit")
            selected = resolve_baseline_artifact(
                workspace_id="workspace-a", approved_entry=approved,
                runs_dir=runs,
            )
            self.assertEqual(selected["run_id"], "approved")
            self.assertEqual(selected["source"], "meta_mysql")
            selected = resolve_baseline_artifact(runs_dir=runs)
            self.assertFalse(selected["available"])
            self.assertEqual(selected["source"], "none")

    def test_baseline_resolution_rejects_registry_and_artifact_suite_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory)
            artifact = runs / "run_medium.json"
            artifact.write_text(json.dumps({
                "run_id": "medium", "workspace_id": "workspace-a",
                "dataset_suite": "medium_sql_v1", "valid": True,
                "fingerprint": {"fingerprint_schema": "evaluation-v2"},
            }), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "artifact 套件不匹配"):
                resolve_baseline_artifact(
                    explicit=artifact,
                    workspace_id="workspace-a",
                    runs_dir=runs,
                    expected_dataset_suite="easy_sql_v1",
                )
            with self.assertRaisesRegex(ValueError, "registry 套件不匹配"):
                resolve_baseline_artifact(
                    workspace_id="workspace-a",
                    approved_entry={
                        "artifact": artifact.name,
                        "status": "approved",
                        "dataset_suite": "medium_sql_v1",
                    },
                    runs_dir=runs,
                    expected_dataset_suite="easy_sql_v1",
                )


if __name__ == "__main__":
    unittest.main()
