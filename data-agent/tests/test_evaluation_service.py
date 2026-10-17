import copy
import json
import tempfile
import unittest
from pathlib import Path

from app.config.app_config import app_config
from app.domain.data_sources.context import stable_fingerprint
from app.domain.query.safety.semantic_violation import stable_hash as semantic_sql_hash
from evals.legacy.governance import (
    REQUIRED_FINGERPRINT_FIELDS,
    build_experiment_metadata,
    resolve_trace_version_snapshot,
)
from app.agent_system.agents.nl2sql.contracts.query_signals import stable_signal_hash
from evals.application.services.evaluation_service import EvaluationService


def pipeline_aggregate_evidence(case_count: int) -> dict:
    return {
        "query_signals": {
            "schema_version": "evaluation-query-signal-aggregate-v2",
            "active_profile_counts": {"jieba_tfidf_v1": case_count},
            "route_policy_counts": {"current_multi_query_v1": case_count},
            "dictionary_version_counts": {"none": case_count},
            "route_counts": {
                route: case_count
                for route in ("column", "metric", "training", "value")
            },
            "query_encode_mode_counts": {
                "column:query": case_count,
                "metric:query": case_count,
                "training:query": case_count,
                "value:not_applicable_es_lexical": case_count,
            },
            "actual_remote_signal_counts": {
                route: case_count
                for route in ("column", "metric", "training", "value")
            },
            "actual_consumer_signal_counts": {
                f"{route}:{consumer}": case_count
                for route in ("column", "metric", "training", "value")
                for consumer in ("local", "remote", "rerank")
            },
            "extractor_kind_counts": {"standalone_query": case_count},
            "consumer_kind_counts": {
                f"{route}:{consumer}:standalone_query": case_count
                for route in ("column", "metric", "training", "value")
                for consumer in ("local", "remote", "rerank")
            },
            "evidence_count": case_count,
            "evidence_complete_count": case_count,
            "route_evidence_count": case_count * 4,
            "dictionary_fallback_count": 0,
            "evidence_complete": True,
        },
        "planning_route": {
            "schema_version": "evaluation-planning-route-aggregate-v1",
            "active_profile_counts": {"keyword_length_v1": case_count},
            "configured_profile_counts": {"keyword_length_v1": case_count},
            "fallback_profile_counts": {"keyword_length_v1": case_count},
            "fallback_class_counts": {"none": case_count},
            "route_counts": {"direct_generate": case_count},
            "plan_calls": 0,
            "decide_node_calls": case_count,
            "decide_node_successes": case_count,
            "observed_plan_node_calls": 0,
            "plan_node_successes": 0,
            "physical_llm_calls": 0,
            "physical_llm_attempts": 0,
            "fallback_count": 0,
            "policy_fallback_count": 0,
            "unsafe_fallback_count": 0,
            "degraded_count": 0,
            "evidence_count": case_count,
            "evidence_complete_count": case_count,
            "decision_accounting_complete": True,
            "plan_call_accounting_complete": True,
            "physical_llm_call_accounting_complete": True,
            "evidence_complete": True,
        },
        "semantic_repair": {
            "schema_version": "evaluation-semantic-repair-aggregate-v1",
            "active_profile_counts": {"registry_same_rules_v1": case_count},
            "configured_profile_counts": {
                "registry_same_rules_v1": case_count,
            },
            "attempted_rule_counts": {
                "join.unique_metadata_edge": case_count,
                "projection.threshold_measure": case_count,
                "order.rank_limit_tie_breaker": case_count,
            },
            "applied_rule_counts": {},
            "changed_count": 0,
            "fallback_count": 0,
            "degraded_count": 0,
            "evidence_count": case_count,
            "evidence_complete_count": case_count,
            "invocation_count": case_count,
            "invocation_evidence_complete_count": case_count,
            "invocation_profile_counts": {
                "registry_same_rules_v1": case_count,
            },
            "invocation_phase_counts": {
                "primary_validate": case_count,
            },
            "selected_invocation_count": case_count,
            "invocation_fallback_count": 0,
            "ledger_complete_cases": case_count,
            "invocation_evidence_complete": True,
            "evidence_complete": True,
        },
    }


def pipeline_stage_evidence() -> dict:
    attempted = [
        "join.unique_metadata_edge",
        "projection.threshold_measure",
        "order.rank_limit_tie_breaker",
    ]
    sql_hash = semantic_sql_hash("SELECT 1")
    invariants = {
        "single_bounded_pass": True,
        "table_set_unchanged": True,
        "filter_literals_unchanged": True,
        "read_only_preserved": True,
    }

    source_fingerprint = stable_signal_hash({"fixture": "standalone-query"})
    extractor_signals = [{
        "signal_id": f"qs-{source_fingerprint[:12]}",
        "kind": "standalone_query",
        "rank": 0,
        "fingerprint": source_fingerprint,
    }]

    def query_route_evidence(route: str) -> dict:
        encode_mode = (
            "not_applicable_es_lexical" if route == "value" else "query"
        )

        def consumer_signals(consumer: str) -> list[dict]:
            return [{**extractor_signals[0], "consumer": consumer}]

        local_signals = consumer_signals("local")
        remote_signals = consumer_signals("remote")
        rerank_signals = consumer_signals("rerank")
        consumer_plan_fingerprint = stable_signal_hash({
            "contract": "query-signals-v1",
            "route_policy_version": "query-signal-routing-v2",
            "route_policy": "current_multi_query_v1",
            "route": route,
            "profile_fingerprint": "profile-fingerprint",
            "signal_fingerprint": "signal-fingerprint",
            "query_encode_mode": encode_mode,
            "local": local_signals,
            "remote": remote_signals,
            "rerank": rerank_signals,
        })
        return {
            "route": route,
            "extractor_profile": "jieba_tfidf_v1",
            "route_policy": "current_multi_query_v1",
            "route_policy_version": "query-signal-routing-v2",
            "dictionary_version": "none",
            "profile_fingerprint": "profile-fingerprint",
            "signal_fingerprint": "signal-fingerprint",
            "cache_namespace_fingerprint": f"cache-{route}",
            "query_encode_mode": encode_mode,
            "local_signal_count": 1,
            "remote_query_count": 1,
            "rerank_signal_count": 1,
            "actual_local_signal_count": 1,
            "actual_remote_signal_count": 1,
            "actual_rerank_signal_count": 1,
            "actual_local_signals": local_signals,
            "actual_remote_signals": remote_signals,
            "actual_rerank_signals": rerank_signals,
            "local_signal_fingerprint": stable_fingerprint(local_signals),
            "remote_signal_fingerprint": stable_fingerprint(remote_signals),
            "rerank_signal_fingerprint": stable_fingerprint(rerank_signals),
            "consumer_plan_fingerprint": consumer_plan_fingerprint,
            "consumer_kind_counts": {
                consumer: {"standalone_query": 1}
                for consumer in ("local", "remote", "rerank")
            },
            "evidence_complete": True,
            "evidence_failures": [],
            "raw_text_logged": False,
        }

    return {
        "query_signals": {
            "schema_version": "evaluation-query-signal-evidence-v2",
            "contract_version": "query-signals-v1",
            "configured_profile": "jieba_tfidf_v1",
            "active_profile": "jieba_tfidf_v1",
            "configured_route_policy": "current_multi_query_v1",
            "dictionary_version": "none",
            "profile_fingerprint": "profile-fingerprint",
            "signal_fingerprint": "signal-fingerprint",
            "extractor_signals": extractor_signals,
            "extractor_signal_identity_fingerprint": stable_fingerprint(
                extractor_signals
            ),
            "extractor_kind_counts": {"standalone_query": 1},
            "route_counts": {
                route: 1
                for route in ("column", "metric", "training", "value")
            },
            "routes": [
                query_route_evidence(route)
                for route in ("column", "metric", "training", "value")
            ],
            "dictionary_fallback_count": 0,
            "evidence_complete": True,
            "evidence_failures": [],
            "raw_text_logged": False,
        },
        "planning_route": {
            "schema_version": "evaluation-planning-route-evidence-v1",
            "contract_version": "planning-route-v1",
            "configured_profile": "keyword_length_v1",
            "runtime_profile": "keyword_length_v1",
            "active_profile": "keyword_length_v1",
            "runtime_override_active": False,
            "decision": "direct_generate",
            "route": "direct_generate",
            "plan_called": False,
            "physical_llm_call": False,
            "fallback_profile": "keyword_length_v1",
            "fallback_applied": False,
            "fallback_class": "none",
            "policy_fallback_eligible": False,
            "degraded": False,
            "repair_profile": "registry_same_rules_v1",
            "reason_codes": [],
            "evidence_complete": True,
            "evidence_failures": [],
            "raw_text_logged": False,
        },
        "semantic_repair": {
            "schema_version": "evaluation-semantic-repair-evidence-v1",
            "contract_version": "semantic-repair-result-v1",
            "registry_version": "semantic-repair-registry-v1",
            "rule_set_version": "semantic-repair-rules-v2",
            "configured_profile": "registry_same_rules_v1",
            "runtime_profile": "registry_same_rules_v1",
            "active_profile": "registry_same_rules_v1",
            "runtime_override_active": False,
            "fallback_profile": "inline_v1",
            "fallback_used": False,
            "degraded": False,
            "changed": False,
            "attempted_rule_codes": attempted,
            "applied_rules": [],
            "skipped_rules": [
                {"rule_code": code, "status": "skipped"}
                for code in attempted
            ],
            "invariants": invariants,
            "original_sql_hash": sql_hash,
            "repaired_sql_hash": sql_hash,
            "final_sql_hash": sql_hash,
            "ledger": [{
                "schema_version": "semantic-repair-ledger-v1",
                "phase": "primary_validate",
                "committed_at_phase": True,
                "selected_at_phase": True,
                "selected": True,
                "validation_valid": True,
                "contract_version": "semantic-repair-result-v1",
                "registry_version": "semantic-repair-registry-v1",
                "rule_set_version": "semantic-repair-rules-v2",
                "profile": "registry_same_rules_v1",
                "original_sql_hash": sql_hash,
                "repaired_sql_hash": sql_hash,
                "observed_sql_hash": sql_hash,
                "attempted_rule_codes": attempted,
                "applied_rules": [],
                "skipped_rules": [],
                "fallback_used": False,
                "fallback_reason": None,
                "invariants": invariants,
                "evidence_complete": True,
                "evidence_failures": [],
                "raw_sql_logged": False,
            }],
            "invocation_count": 1,
            "selected_invocation_count": 1,
            "ledger_evidence_complete": True,
            "ledger_truncated": False,
            "evidence_complete": True,
            "evidence_failures": [],
            "raw_sql_logged": False,
        },
    }


class MetaBaselineStub:
    """Small async stand-in for the Meta MySQL baseline repository contract."""

    def __init__(self, approved_entry: dict | list[dict] | None = None):
        self.entries: dict[tuple[str, str], dict] = {}
        self.approved_calls: list[tuple[str, str]] = []
        approved_entries = (
            approved_entry if isinstance(approved_entry, list)
            else [approved_entry] if approved_entry else []
        )
        for entry in approved_entries:
            key = (
                str(entry["workspace_id"]),
                str(entry["run_id"]),
            )
            self.entries[key] = dict(entry)

    async def get(self, workspace_id: str, run_id: str) -> dict | None:
        entry = self.entries.get((workspace_id, run_id))
        return dict(entry) if entry else None

    async def approved(
        self, workspace_id: str, dataset_suite: str,
    ) -> dict | None:
        self.approved_calls.append((workspace_id, dataset_suite))
        entry = next((
            item for (item_workspace, _), item in self.entries.items()
            if item_workspace == workspace_id
            and item.get("dataset_suite") == dataset_suite
            and item.get("status") == "approved"
        ), None)
        return dict(entry) if entry else None

    async def approved_all(self, workspace_id: str) -> list[dict]:
        return [
            dict(item) for (item_workspace, _), item in self.entries.items()
            if item_workspace == workspace_id and item.get("status") == "approved"
        ]

    async def review(
        self, run: dict, *, decision: str, reviewer: str, note: str = "",
    ) -> dict:
        workspace_id = str(run["workspace_id"])
        dataset_suite = str(run["dataset_suite"])
        if decision == "approve":
            for (item_workspace, _), item in self.entries.items():
                if (
                    item_workspace == workspace_id
                    and item.get("dataset_suite") == dataset_suite
                    and item.get("status") == "approved"
                ):
                    item["status"] = "superseded"
        entry = {
            "workspace_id": workspace_id,
            "run_id": str(run["run_id"]),
            "dataset_suite": dataset_suite,
            "artifact": run.get("_artifact"),
            "status": "approved" if decision == "approve" else "rejected",
            "reviewed_by": reviewer,
            "review_note": note,
        }
        self.entries[(workspace_id, entry["run_id"])] = entry
        return dict(entry)


class EvaluationServiceTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _v2_run(
        repeat_index: int,
        experiment_id: str = "experiment-001",
        *,
        case_count: int = 1,
    ) -> dict:
        sql_route = {
            "candidate_id": "todo97-q1-qwen3.7-plus-standard-nonthinking-v1",
            "model_name": "qwen3.7-plus",
            "model_revision": "snapshot-2026-05-26",
            "provider": "dashscope",
            "model_provider": "openai",
            "access_mode": "standard",
            "endpoint_profile": "dashscope-compatible-cn",
            "endpoint_id": "sha256:test-endpoint",
            "deployment_id": "sha256:test-deployment",
            "adapter_version": "dashscope-openai-chat-v1",
            "capability_registry_version": "llm-capability-registry-v1",
            "capability_lifecycle": "officially-listed",
            "comparison_mode": "strict-nonthinking-sql",
            "promotion_eligible": True,
            "thinking_mode": "disabled",
            "reasoning_effort": None,
            "temperature": 0,
            "timeout_seconds": 22,
            "max_retries": 1,
            "max_completion_tokens": 2048,
            "max_concurrency": 6,
        }
        fingerprint = {field: f"value-{field}" for field in REQUIRED_FINGERPRINT_FIELDS}
        fingerprint.update({
            "fingerprint_schema": "evaluation-v2", "complete": True,
            "evaluation_profile": {"schema": "evaluation-profile-v1", "name": "core-default", "analysis_enabled": False},
            "run_fingerprint": "sha256:v2",
            "sample_contract": {"total": 1, "complete": True, "all_reviewed": True},
            "pipeline_profiles": {
                "query_signals": {
                    "profile": "jieba_tfidf_v1",
                    "route_policy": "current_multi_query_v1",
                    "workspace_dictionary_enabled": True,
                },
                "planning_route": {
                    "active_profile": "keyword_length_v1",
                    "fallback_profile": "keyword_length_v1",
                    "repair_profile": "registry_same_rules_v1",
                },
                "semantic_repair": {
                    "active_profile": "registry_same_rules_v1",
                },
            },
        })
        epoch = {
            "schema": "embedding-runtime-metrics-epoch-v1",
            "id": f"epoch-{repeat_index}",
            "index": repeat_index,
            "started_at": f"2026-08-14T00:00:0{repeat_index}+00:00",
        }
        is_cold = repeat_index == 1
        payload = {
            "run_id": f"v2-run-{repeat_index}", "timestamp": "2026-07-17 10:00:00",
            "workspace_id": "workspace-a", "valid": True, "n": 1,
            "baseline_eligible": True,
            "dataset_suite": "easy_sql_v1",
            "experiment": build_experiment_metadata(
                experiment_id=experiment_id, candidate_label="v2-control",
                repeat_index=repeat_index, cache_state="cold" if repeat_index == 1 else "warm",
                provider_parameters={
                    "temperature": 0,
                    "sql_route": dict(sql_route),
                },
            ),
            "fingerprint": fingerprint,
            "embedding_warmup": {
                "schema": "evaluation-embedding-warmup-v1",
                "status": "pass",
                "session_id": "embedding-session-1",
                "session_run_index": repeat_index,
                "resident_reused": not is_cold,
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
            "cache_control": {
                "requested_state": "cold" if is_cold else "warm",
                "effective_policy": (
                    "cleared_all" if is_cold else "retained_ttl_best_effort"
                ),
                "result_cache_cleared": is_cold,
                "retrieval_caches_cleared": is_cold,
                "retrieval_cache_generation_before": 1 if is_cold else 2,
                "retrieval_cache_generation_after": 2,
                "ttl_seconds": {"result": 30, "retrieval": 120},
                "cache_hit_not_guaranteed": not is_cold,
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
                    "submitted": case_count,
                    "completed": case_count,
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
                "observed_epoch_count": case_count,
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
                "n": 1, "result_evaluated": 1, "syntax_valid": 1, "exec_success": 1,
                "tables_correct": 1, "table_scope_safe": 1,
                "table_scope_evidence_complete": 1,
                "security_guard_bypass_count": 0,
                "result_equivalent": 1, "table_f1_mean": 1,
                "infrastructure_errors": 0, "tt_sql_p95": 1000,
                "tt_result_p95": 1200, "latency_p95": 1500,
                "tt_complete_p95": 1500, "timing_protocol_version": "timing-v2",
                "timing_invariant_violations": [], "llm_attempt_schema": "llm-attempt-v2",
                "llm_attempt_accounting_complete": True, "degradation_evidence_complete": True,
                "failed_core_llm_calls": 0, "failed_optional_llm_calls": 0,
                "llm_attempts_timed_out": 0,
                "llm_attempts": 0, "llm_outcome_classes": {},
                "llm_recoveries": 0,
                "semantic_coverage_cases": 1,
                "semantic_coverage_mean": 1.0,
                "semantic_unresolved_cases": 0,
                "semantic_unsafe_compile_cases": 0,
                "request_deadline_contract_complete": True,
                "request_deadline_exhausted_cases": 0,
                "tokens_mean": 100, "tokens_max": 100,
                "first_sql_evidence_complete": True,
                "first_non_empty": 1,
                "first_syntax_valid": 1,
                "first_guard_passed": 1,
                "first_result_equivalent": 1,
                "first_exec_success": 1,
                "first_tables_correct": 1,
                "first_structure_equivalent": 1,
                "correction_triggered": 0,
                "correction_fixed_execution": 0,
                "correction_fixed_result": 0,
                "correction_harmful_modification": 0,
                "correction_unattributable": 0,
                "correction_llm_attempts": 0,
                "correction_additional_tokens": 0,
                "correction_unknown_token_attempts": 0,
                "correction_additional_latency_ms": 0,
                "estimated_cost_total": 0,
                "cost_estimate_available": False,
                **pipeline_aggregate_evidence(case_count),
            },
            "cases": [{
                "id": "eval_001", "question": "测试", "category": "simple",
                "exec_success": True, "tables_correct": True,
                "table_scope_safe": True, "result_equivalent": True,
                "first_sql_metrics": {
                    "sql": "SELECT 1",
                    "non_empty": True,
                    "syntax_valid": True,
                    "guard_passed": True,
                    "execution_attempted": True,
                    "exec_success": True,
                    "result_evaluated": True,
                    "result_equivalent": True,
                },
                "correction_metrics": {
                    "triggered": False,
                    "attribution_valid": False,
                    "attribution_status": "not_triggered",
                    "additional_tokens": 0,
                    "outcome": "not_triggered",
                },
                "final_sql_metrics": {
                    "sql": "SELECT 1",
                    "non_empty": True,
                    "syntax_valid": True,
                    "table_scope_safe": True,
                    "table_scope_safety_evidence": {
                        "schema": "table-scope-safety-v2",
                        "authorization_source": "runtime_table_infos",
                        "evidence_complete": True,
                        "sql_hash": stable_fingerprint("SELECT 1"),
                        "request_scope_fingerprint": "sha256:test-scope",
                        "authorized_sql_hash": "sha256:test-authorized",
                    },
                    "guard_passed": True,
                    "execution_attempted": True,
                    "exec_success": True,
                    "result_evaluated": True,
                    "result_equivalent": True,
                },
                "stage_evidence": {
                    "retrieval": {}, "filtered_tables": [], "first_sql": "SELECT 1",
                    "correction_sqls": [], "authorized_sql": "SELECT 1",
                    "result_summary": {"masked": True, "values_persisted": False},
                    "llm_route_manifest": {"sql": dict(sql_route)},
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
                    "retrieval_budget": [{
                        "route": "column",
                        "embedding_runtime": {
                            "metrics_epoch_id": epoch["id"],
                            "metrics_epoch_index": epoch["index"],
                            "metrics_epoch_started_at": epoch["started_at"],
                        },
                    }],
                    **pipeline_stage_evidence(),
                },
            }],
        }
        if case_count != 1:
            payload["n"] = case_count
            payload["fingerprint"]["sample_contract"]["total"] = case_count
            for field in (
                "n",
                "result_evaluated",
                "syntax_valid",
                "exec_success",
                "tables_correct",
                "table_scope_safe",
                "table_scope_evidence_complete",
                "result_equivalent",
                "semantic_coverage_cases",
                "first_non_empty",
                "first_syntax_valid",
                "first_guard_passed",
                "first_result_equivalent",
                "first_exec_success",
                "first_tables_correct",
                "first_structure_equivalent",
            ):
                payload["aggregate"][field] = case_count
            template = payload["cases"][0]
            payload["cases"] = []
            for index in range(1, case_count + 1):
                case = copy.deepcopy(template)
                case["id"] = f"eval_{index:03d}"
                payload["cases"].append(case)
        return payload

    async def test_latest_excludes_unjudged_reference_sql_failures_from_result_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                "run_id": "run-result-denominator", "timestamp": "2026-07-16 10:00:00",
                "valid": True, "workspace_id": "workspace-a", "n": 2,
                "dataset_suite": "easy_sql_v1",
                "aggregate": {
                    "syntax_valid": 2, "exec_success": 2, "tables_correct": 2,
                    "result_equivalent": 1, "result_evaluated": 1,
                },
                "cases": [
                    {"id": "ok", "category": "simple", "result_equivalent": True},
                    {"id": "infra", "category": "simple", "result_equivalent": False,
                     "result_evaluated": False, "failure_type": "infrastructure"},
                ],
            }
            (root / "run_20260716_100000.json").write_text(json.dumps(payload), encoding="utf-8")
            report = EvaluationService(root).latest("workspace-a")
            self.assertEqual(report["metrics"]["result_evaluated"], 1)
            self.assertEqual(report["metrics"]["result_rate"], 1.0)

    async def test_online_trace_uses_latest_largest_complete_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = {
                "valid": True, "workspace_id": "workspace-a",
                "fingerprint": {
                    "complete": True, "knowledge_version": "knowledge",
                    "schema_version": "schema", "index_version": "index",
                    "fingerprint_schema": "evaluation-v2", "dataset_version": "dataset",
                    "dw_snapshot": "snapshot", "prompt_hashes": {"p": "h"},
                    "model": {"name": "m"}, "config_hash": "config",
                    "git_commit": "git", "code_version": "code",
                    "dependency_lock_hash": "dependency-lock",
                    "evaluation_profile": {"schema": "evaluation-profile-v1", "name": "core-default", "analysis_enabled": False},
                },
            }
            (root / "run_20260715_100000.json").write_text(json.dumps({
                **base, "run_id": "full-run", "n": 30,
            }), encoding="utf-8")
            (root / "run_20260715_110000.json").write_text(json.dumps({
                **base, "run_id": "targeted-run", "n": 2,
            }), encoding="utf-8")
            snapshot = resolve_trace_version_snapshot("workspace-a", root)
            self.assertEqual(snapshot["version_run_id"], "full-run")
            self.assertTrue(snapshot["version_complete"])
            self.assertTrue(snapshot["run_fingerprint"].startswith("sha256:"))

    async def test_latest_is_workspace_scoped_and_enriches_current_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                "run_id": "20260715_100000", "timestamp": "2026-07-15 10:00:00",
                "valid": True, "workspace_id": "workspace-a", "n": 1,
                "dataset_suite": "easy_sql_v1",
                "evaluation_profile": {
                    "schema": "evaluation-profile-v1",
                    "name": "core-default",
                    "analysis_enabled": False,
                },
                "aggregate": {
                    "syntax_valid": 1, "exec_success": 1, "tables_correct": 1,
                    "result_equivalent": 0, "result_evaluated": 1,
                },
                "cases": [{
                    "id": "eval_013", "question": "各国家的订单金额", "category": "agg_join",
                    "gen_sql": "SELECT r.country, SUM(f.order_amount) FROM fact_order f, dim_region r GROUP BY r.country",
                    "expected_sql": "SELECT r.country, SUM(f.order_amount) FROM fact_order f JOIN dim_region r ON f.region_id=r.region_id GROUP BY r.country",
                    "syntax_valid": True, "exec_success": True, "tables_correct": True,
                    "result_equivalent": False, "error": None,
                }],
            }
            (root / "run_20260715_100000.json").write_text(json.dumps(payload), encoding="utf-8")
            service = EvaluationService(root)
            report = service.latest("workspace-a")
            self.assertTrue(report["available"])
            self.assertEqual(report["run"]["run_id"], "20260715_100000")
            self.assertEqual(report["run"]["dataset_suite"], "easy_sql_v1")
            self.assertEqual(report["summary"]["root_causes"][0]["code"], "join_path")
            self.assertEqual(report["cases"][0]["attribution"]["code"], "join_path")
            self.assertFalse(service.latest("workspace-b")["available"])

    async def test_latest_exposes_safe_model_identity_layer_metrics_and_cost_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self._v2_run(1, experiment_id="todo97-q1")
            sql_route = payload["experiment"]["provider_parameters"]["sql_route"]
            sql_route.update({
                "base_url": "https://user:secret@example.invalid/v1",
                "api_key_env": "DO_NOT_EXPOSE_THIS_ENV_NAME",
            })
            case = payload["cases"][0]
            case["first_sql_metrics"]["sql"] = "SELECT private_first_marker"
            case["final_sql_metrics"]["sql"] = "SELECT private_final_marker"
            case_route = case["stage_evidence"]["llm_route_manifest"]["sql"]
            case_route.update({
                "base_url": "https://case-secret.example.invalid/v1",
                "api_key_env": "DO_NOT_EXPOSE_CASE_ENV_NAME",
            })
            (root / "run_todo97-q1.json").write_text(
                json.dumps(payload), encoding="utf-8",
            )

            report = EvaluationService(root).latest(
                "workspace-a", include_passed=True,
            )

            self.assertEqual(report["run"]["candidate_label"], "v2-control")
            self.assertEqual(report["run"]["model_name"], "qwen3.7-plus")
            self.assertEqual(report["run"]["provider"], "dashscope")
            self.assertEqual(report["run"]["model_provider"], "openai")
            self.assertEqual(
                report["run"]["adapter_version"],
                "dashscope-openai-chat-v1",
            )
            self.assertEqual(
                report["run"]["capability_registry_version"],
                "llm-capability-registry-v1",
            )
            self.assertFalse(report["performance"]["cost_estimate_available"])
            self.assertEqual(report["metrics"]["first_sql"]["result_equivalent"], 1)
            self.assertTrue(report["metrics"]["first_sql"]["evidence_complete"])
            self.assertEqual(report["metrics"]["correction"]["triggered"], 0)
            self.assertEqual(report["metrics"]["final_sql"]["result_equivalent"], 1)

            visible_case = report["cases"][0]
            self.assertEqual(visible_case["model_profile"], report["run"]["model_profile"])
            self.assertEqual(visible_case["candidate_label"], "v2-control")
            self.assertTrue(visible_case["first_sql_metrics"]["result_equivalent"])
            self.assertEqual(
                visible_case["correction_metrics"]["outcome"], "not_triggered",
            )
            self.assertTrue(visible_case["final_sql_metrics"]["result_equivalent"])
            self.assertFalse(visible_case["cost_estimate_available"])
            self.assertNotIn("sql", visible_case["first_sql_metrics"])
            self.assertNotIn("sql", visible_case["final_sql_metrics"])

            encoded = json.dumps(report, ensure_ascii=False)
            for forbidden in (
                "user:secret", "DO_NOT_EXPOSE_THIS_ENV_NAME",
                "case-secret", "DO_NOT_EXPOSE_CASE_ENV_NAME",
                "private_first_marker", "private_final_marker",
            ):
                self.assertNotIn(forbidden, encoded)

    async def test_latest_prefers_newest_formal_experiment_over_older_larger_suite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            older = {
                "run_id": "20260803_190000", "timestamp": "2026-08-03 19:00:00",
                "valid": True, "workspace_id": "workspace-a", "n": 100,
                "dataset_suite": "easy_sql_v1",
                "aggregate": {
                    "syntax_valid": 100, "exec_success": 100,
                    "tables_correct": 100, "result_equivalent": 100,
                    "result_evaluated": 100,
                },
                "cases": [{"id": "eval_older", "result_equivalent": True}],
            }
            current = self._v2_run(3, experiment_id="newer-formal-experiment")
            current.update({
                "run_id": "20260803_204124", "timestamp": "2026-08-03 20:41:24",
                "n": 50,
            })
            current["fingerprint"]["sample_contract"].update({
                "total": 50, "complete": True, "all_reviewed": True,
            })
            current["aggregate"].update({
                "n": 50, "result_evaluated": 50, "syntax_valid": 49,
                "exec_success": 43, "tables_correct": 33, "result_equivalent": 9,
            })
            (root / "run_20260803_190000.json").write_text(
                json.dumps(older, indent=2), encoding="utf-8",
            )
            (root / "run_20260803_204124.json").write_text(
                json.dumps(current, indent=2), encoding="utf-8",
            )

            service = EvaluationService(root, MetaBaselineStub())
            report = service.latest("workspace-a")
            self.assertEqual(report["run"]["run_id"], "20260803_204124")
            self.assertEqual(report["run"]["dataset_suite"], "easy_sql_v1")
            self.assertEqual(report["metrics"]["result_rate"], 0.18)
            runs = await service.list_runs("workspace-a")
            self.assertEqual([item["run_id"] for item in runs], [
                "20260803_204124", "20260803_190000",
            ])
            governance = await service.governance("workspace-a")
            self.assertEqual(governance["dataset_suite"], "easy_sql_v1")
            self.assertEqual(
                governance["runnable_dataset_suite"], "easy_sql_v1",
            )
            self.assertTrue(
                all(
                    "--suite easy_sql_v1" in command
                    for command in governance["experiment_setup"]["commands"]
                )
            )
            suggested_id = governance["experiment_setup"]["suggested_id"]
            self.assertIn("-easy-easy_sql_v1-", suggested_id)
            self.assertRegex(
                suggested_id,
                r"-\d{8}-\d{6}-workspac-[0-9a-f]{6}$",
            )

    async def test_unfiltered_catalog_excludes_retired_dataset_suites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run_id, suite in (
                ("current", "easy_sql_v1"),
                ("retired", "common_sql_v1"),
            ):
                payload = self._v2_run(
                    1, experiment_id=f"experiment-{run_id}",
                )
                payload.update({"run_id": run_id, "dataset_suite": suite})
                (root / f"run_{run_id}.json").write_text(
                    json.dumps(payload), encoding="utf-8",
                )

            service = EvaluationService(root, MetaBaselineStub())
            self.assertEqual(
                [row["run_id"] for row in await service.list_runs("workspace-a")],
                ["current"],
            )
            self.assertEqual(
                [
                    row["experiment_id"]
                    for row in await service.list_experiments("workspace-a")
                ],
                ["experiment-current"],
            )
            self.assertEqual(
                service.latest("workspace-a")["run"]["run_id"], "current",
            )

    async def test_experiment_catalog_groups_same_id_independently_by_suite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for suite in ("easy_sql_v1", "medium_sql_v1"):
                for repeat in (1, 2, 3):
                    payload = self._v2_run(
                        repeat, experiment_id="shared-experiment-id",
                    )
                    run_id = f"{suite}-{repeat}"
                    payload.update({
                        "run_id": run_id,
                        "dataset_suite": suite,
                    })
                    (root / f"run_{run_id}.json").write_text(
                        json.dumps(payload), encoding="utf-8",
                    )

            rows = await EvaluationService(
                root, MetaBaselineStub(),
            ).list_experiments("workspace-a")
            shared = [
                row for row in rows
                if row["experiment_id"] == "shared-experiment-id"
            ]
            self.assertEqual(len(shared), 2)
            by_suite = {row["dataset_suite"]: row for row in shared}
            self.assertEqual(set(by_suite), {
                "easy_sql_v1", "medium_sql_v1",
            })
            self.assertEqual(by_suite["easy_sql_v1"]["difficulty"], "easy")
            self.assertEqual(
                by_suite["medium_sql_v1"]["difficulty"], "medium",
            )
            for suite, report in by_suite.items():
                self.assertEqual(
                    {run["run_id"].rsplit("-", 1)[0] for run in report["runs"]},
                    {suite},
                )

    async def test_explicit_nonrelease_artifact_is_hidden_even_if_misplaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = {
                "run_id": "release", "workspace_id": "workspace-a",
                "valid": True, "n": 1,
                "dataset_suite": "easy_sql_v1",
                "baseline_eligible": True,
                "artifact_scope": "release_candidate",
                "aggregate": {"result_equivalent": 1, "result_evaluated": 1},
                "cases": [{"id": "eval_001", "result_equivalent": True}],
            }
            boundary = {
                **release,
                "run_id": "boundary",
                "baseline_eligible": False,
                "artifact_scope": "non_promotable_observational",
            }
            (root / "run_20260805_100000.json").write_text(
                json.dumps(release), encoding="utf-8",
            )
            (root / "run_20260805_110000.json").write_text(
                json.dumps(boundary), encoding="utf-8",
            )
            service = EvaluationService(root, MetaBaselineStub())
            report = service.latest("workspace-a")
            self.assertEqual(report["run"]["run_id"], "release")
            runs = await service.list_runs("workspace-a")
            self.assertEqual([item["run_id"] for item in runs], ["release"])

    async def test_v2_baseline_requires_qualified_three_run_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self._v2_run(
                3,
                experiment_id="approved-control",
                case_count=100,
            )
            baseline.update({
                "run_id": "approved-baseline",
                "timestamp": "2026-07-16 10:00:00",
            })
            baseline_artifact = "run_approved-baseline.json"
            (root / baseline_artifact).write_text(
                json.dumps(baseline), encoding="utf-8",
            )
            registry = MetaBaselineStub({
                "run_id": "approved-baseline",
                "workspace_id": "workspace-a",
                "dataset_suite": "easy_sql_v1",
                "artifact": baseline_artifact,
                "status": "approved",
                "eligible": True,
                "checklist": {},
            })
            first = self._v2_run(1, case_count=100)
            (root / "run_v2-run-1.json").write_text(json.dumps(first), encoding="utf-8")
            service = EvaluationService(root, registry)
            with self.assertRaisesRegex(ValueError, "三次完整证据准入"):
                await service.review_baseline(
                    "workspace-a", "v2-run-1", decision="approve", reviewer="alice",
                )
            for index in (2, 3):
                payload = self._v2_run(index, case_count=100)
                (root / f"run_v2-run-{index}.json").write_text(json.dumps(payload), encoding="utf-8")
            approved = await service.review_baseline(
                "workspace-a", "v2-run-3", decision="approve", reviewer="alice",
            )
            self.assertEqual(approved["status"], "approved")
            experiments = await service.list_experiments("workspace-a")
            experiment = next(
                item for item in experiments
                if item["experiment_id"] == "experiment-001"
            )
            self.assertTrue(experiment["qualified"], experiment["failed_checks"])

    async def test_difficulty_filters_list_latest_and_governance_without_cross_suite_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures = (
                ("easy", "easy_sql_v1", "2026-08-05 09:00:00"),
                ("medium", "medium_sql_v1", "2026-08-05 10:00:00"),
                ("hard", "hard_sql_v1", "2026-08-05 11:00:00"),
            )
            for run_id, suite, timestamp in fixtures:
                payload = self._v2_run(1, experiment_id=f"experiment-{run_id}")
                payload.update({
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "dataset_suite": suite,
                })
                (root / f"run_{run_id}.json").write_text(
                    json.dumps(payload), encoding="utf-8",
                )

            service = EvaluationService(root, MetaBaselineStub())
            easy_rows = await service.list_runs("workspace-a", difficulty="easy")
            self.assertEqual([row["run_id"] for row in easy_rows], ["easy"])
            self.assertEqual(easy_rows[0]["dataset_suite"], "easy_sql_v1")
            self.assertEqual(service.latest("workspace-a", difficulty="medium")["run"]["run_id"], "medium")

            governance = await service.governance(
                "workspace-a", run_id="medium", difficulty="easy",
            )
            self.assertFalse(governance["available"])
            self.assertIn("未找到 run medium", governance["message"])

    async def test_list_baselines_returns_ordered_three_difficulty_cards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for difficulty, suite, passed in (
                ("easy", "easy_sql_v1", 48),
                ("hard", "hard_sql_v1", 17),
            ):
                artifact = f"run_{difficulty}-baseline.json"
                payload = {
                    "run_id": f"{difficulty}-baseline",
                    "workspace_id": "workspace-a",
                    "dataset_suite": suite,
                    "valid": True,
                    "n": 50,
                    "fingerprint": {"fingerprint_schema": "evaluation-v2"},
                    "aggregate": {
                        "n": 50,
                        "result_evaluated": 50,
                        "result_equivalent": passed,
                        "tables_correct": passed + 1,
                        "exec_success": 50,
                        "tt_complete_p95": 2500,
                        "llm_attempts": 50,
                        "tokens_mean": 321,
                        "estimated_cost_total": 1.25,
                        "cost_estimate_available": True,
                    },
                    "cases": [],
                }
                (root / artifact).write_text(json.dumps(payload), encoding="utf-8")
                entries.append({
                    "workspace_id": "workspace-a",
                    "run_id": payload["run_id"],
                    "dataset_suite": suite,
                    "artifact": artifact,
                    "status": "approved",
                    "reviewed_by": "alice",
                    "reviewed_at": "2026-08-05T12:00:00+08:00",
                })

            registry = MetaBaselineStub(entries)
            cards = (await EvaluationService(root, registry).list_baselines(
                "workspace-a",
            ))["baselines"]

            self.assertEqual(
                [card["difficulty"] for card in cards],
                ["easy", "medium", "hard"],
            )
            self.assertEqual(
                [card["dataset_suite"] for card in cards],
                ["easy_sql_v1", "medium_sql_v1", "hard_sql_v1"],
            )
            self.assertTrue(cards[0]["available"])
            self.assertEqual(cards[0]["metrics"]["total"], 50)
            self.assertEqual(cards[0]["metrics"]["result_rate"], 0.96)
            self.assertTrue(cards[0]["metrics"]["cost_estimate_available"])
            self.assertEqual(cards[0]["resolution_status"], "resolved")
            self.assertIsNone(cards[0]["resolution_error"])
            self.assertFalse(cards[1]["available"])
            self.assertEqual(cards[1]["status"], "missing")
            self.assertEqual(cards[1]["resolution_status"], "not_approved")
            self.assertIsNone(cards[1]["resolution_error"])
            self.assertTrue(cards[2]["available"])

    async def test_list_baselines_keeps_approved_row_visible_when_artifact_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = MetaBaselineStub([{
                "workspace_id": "workspace-a",
                "run_id": "easy-missing-artifact",
                "dataset_suite": "easy_sql_v1",
                "artifact": "run_easy-missing-artifact.json",
                "status": "approved",
                "reviewed_by": "alice",
                "reviewed_at": "2026-08-05T12:00:00+08:00",
            }])

            cards = (await EvaluationService(root, registry).list_baselines(
                "workspace-a",
            ))["baselines"]
            easy = cards[0]

            self.assertFalse(easy["available"])
            self.assertEqual(easy["status"], "approved")
            self.assertEqual(easy["run_id"], "easy-missing-artifact")
            self.assertEqual(easy["resolution_status"], "artifact_missing")
            self.assertEqual(
                easy["resolution_error"],
                "已审批，但基线制品不存在或无法访问。",
            )

    async def test_list_baselines_isolates_an_unparseable_approved_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_hard-invalid.json").write_text("{", encoding="utf-8")
            registry = MetaBaselineStub([{
                "workspace_id": "workspace-a",
                "run_id": "hard-invalid-artifact",
                "dataset_suite": "hard_sql_v1",
                "artifact": "run_hard-invalid.json",
                "status": "approved",
            }])

            cards = (await EvaluationService(root, registry).list_baselines(
                "workspace-a",
            ))["baselines"]
            hard = cards[2]

            self.assertFalse(hard["available"])
            self.assertEqual(hard["status"], "approved")
            self.assertEqual(hard["resolution_status"], "artifact_invalid")
            self.assertEqual(
                hard["resolution_error"],
                "已审批，但基线制品内容无法解析。",
            )

    async def test_approved_selection_resolves_only_the_requested_suite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for run_id, suite in (
                ("medium-approved", "medium_sql_v1"),
                ("easy-approved", "easy_sql_v1"),
            ):
                artifact = f"run_{run_id}.json"
                (root / artifact).write_text(json.dumps({
                    "run_id": run_id,
                    "workspace_id": "workspace-a",
                    "dataset_suite": suite,
                    "valid": True,
                    "fingerprint": {"fingerprint_schema": "evaluation-v2"},
                    "aggregate": {},
                }), encoding="utf-8")
                entries.append({
                    "workspace_id": "workspace-a",
                    "run_id": run_id,
                    "dataset_suite": suite,
                    "artifact": artifact,
                    "status": "approved",
                })

            registry = MetaBaselineStub(entries)
            approved, selection = await EvaluationService(
                root, registry,
            )._approved_selection("workspace-a", "easy_sql_v1")

            self.assertEqual(approved["run_id"], "easy-approved")
            self.assertEqual(selection["run_id"], "easy-approved")
            self.assertEqual(
                selection["payload"]["dataset_suite"], "easy_sql_v1",
            )
            self.assertEqual(
                registry.approved_calls, [("workspace-a", "easy_sql_v1")],
            )

    async def test_observational_baseline_approval_records_r1_cold_even_when_r3_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in (1, 2, 3):
                payload = self._v2_run(
                    index,
                    experiment_id="easy-observation",
                    case_count=100,
                )
                payload["dataset_suite"] = "easy_sql_v1"
                payload["aggregate"]["result_equivalent"] = 0
                payload["aggregate"]["first_result_equivalent"] = 0
                for case in payload["cases"]:
                    case["result_equivalent"] = False
                    case["first_sql_metrics"]["result_equivalent"] = False
                    case["final_sql_metrics"]["result_equivalent"] = False
                (root / f"run_v2-run-{index}.json").write_text(
                    json.dumps(payload), encoding="utf-8",
                )

            registry = MetaBaselineStub()
            approved = await EvaluationService(root, registry).review_baseline(
                "workspace-a", "v2-run-3",
                decision="approve", reviewer="alice", note="首轮难度观测",
            )

            self.assertEqual(approved["status"], "approved")
            self.assertEqual(approved["run_id"], "v2-run-1")
            self.assertEqual(approved["dataset_suite"], "easy_sql_v1")
            self.assertIn("代表 run=R1 cold", approved["review_note"])
            self.assertNotIn(("workspace-a", "v2-run-3"), registry.entries)


if __name__ == "__main__":
    unittest.main()
