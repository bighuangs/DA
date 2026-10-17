"""Deterministic evaluation fingerprints and release-baseline governance.

Artifacts deliberately contain hashes, counts and non-secret configuration only.
They never persist credentials, vectors, or result row values.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from sqlalchemy import text

from app.agent_system.agents.nl2sql.follow_up import FOLLOW_UP_VERSION
from app.adapters.outbound.agent.llm_provider import llm_route_manifest
from app.config.app_config import app_config
from app.agent_system.agents.nl2sql.services.candidate_confidence import (
    CONFIDENCE_VERSION,
    SEMANTIC_REVIEW_POLICY_VERSION,
    compact_candidate_decision,
)
from app.agent_system.runtime.llm_attempts import LLM_ATTEMPT_SCHEMA
from app.agent_system.agents.nl2sql.services.planning_router import (
    IMPLEMENTED_PROFILES,
    POLICY_FALLBACK_PROFILES,
)
from app.agent_system.agents.nl2sql.services.prompt_context import PROMPT_PAYLOAD_VERSION
from app.agent_system.agents.nl2sql.services.query_signals import (
    EXTRACTOR_VERSION,
    PROFILE_VERSIONS,
    ROUTE_POLICIES,
    ROUTE_POLICY_VERSION,
)
from app.infrastructure.search.index_version import retrieval_parameters
from app.agent_system.agents.nl2sql.services.semantic_repair import PROFILE_RULES
from app.domain.query.safety.sql_security import check_sql_read
from evals.legacy.profiles import evaluation_profile_name, resolve_evaluation_profile
from app.agent_system.agents.nl2sql.contracts.query_signals import (
    QUERY_SIGNAL_CONTRACT_VERSION,
    SIGNAL_KINDS,
    stable_signal_hash,
)
from evals.paths import PROJECT_ROOT, RUNS_ROOT

RUNS_DIR = RUNS_ROOT
EXPERIMENT_SCHEMA = "evaluation-experiment-v2"
EVALUATION_EMBEDDING_WARMUP_SCHEMA = "evaluation-embedding-warmup-v1"
MODEL_ABLATION_SCHEMA = "model-ablation-v1"
PIPELINE_ABLATION_SCHEMA = "pipeline-ablation-v1"
EXPERIMENT_REQUIRED_REPEATS = 3
CORE_RETRIEVAL_NODES = frozenset({
    "recall_column", "recall_metric", "recall_training", "recall_value",
})
QUERY_SIGNAL_REQUIRED_ROUTES = frozenset({
    "column", "metric", "training", "value",
})
QUERY_SIGNAL_ROUTE_ENCODE_MODES = {
    "column": "query",
    "metric": "query",
    "training": "query",
    "value": "not_applicable_es_lexical",
}
QUERY_SIGNAL_CONSUMER_KINDS = frozenset({
    "standalone_query",
    *SIGNAL_KINDS,
    "llm_expansion",
    "deterministic_route_variant",
})
CORE_RETRIEVAL_FAILURE_REASONS = frozenset({
    "timeout", "dependency_unavailable",
})
EXPERIMENT_GATE_LIMITS = {
    "tt_sql_p95_ms": 15_000.0,
    "tt_result_p95_ms": 20_000.0,
    "tt_complete_p95_ms": 30_000.0,
    "tokens_max": 30_000.0,
}
# Exact table matching remains useful diagnostic evidence, but it conflates a
# missing optional/alternative path with an unsafe table expansion.  Promotion
# is therefore blocked by the independent scope boundary below, while exact
# matching is accounted as an observation.
EXPERIMENT_OBSERVATIONAL_CHECKS = frozenset({"table_exact_non_regression"})
MODEL_ABLATION_SQL_ROUTE_FIELDS = frozenset({
    "provider", "model_provider", "model_name", "model_revision",
    "candidate_id", "endpoint_profile", "endpoint_id", "deployment_id",
    "access_mode", "adapter_version", "thinking_mode", "reasoning_effort",
    "service_fingerprint", "temperature", "comparison_mode", "capability_lifecycle",
    "network_profile",
})
REQUIRED_CASE_FIELDS = (
    "id", "tier", "question", "expected_sql", "category", "result_ordered", "tags",
    "owner", "source", "review_status", "difficulty", "required_tables",
    "allowed_tables", "expected_projection", "contains_sensitive_data",
)
REQUIRED_FINGERPRINT_FIELDS = (
    "fingerprint_schema", "dataset_version", "dw_snapshot", "schema_version", "knowledge_version",
    "index_version", "prompt_hashes", "model", "config_hash", "git_commit", "code_version",
    "dependency_lock_hash", "evaluation_profile",
)
LOCAL_VERSION_FINGERPRINT_FIELDS = (
    "code_version",
    "dependency_lock_hash",
    "config_hash",
    "dataset_version",
)


def stable_hash(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


def _model_ablation_manifest(safe_config: dict[str, Any]) -> dict[str, Any]:
    """Freeze all config except pre-registered SQL-route capability fields."""
    body = json.loads(json.dumps(safe_config, ensure_ascii=False, default=str))
    routes = ((body.get("llm") or {}).get("routes") or {})
    sql_route = dict(routes.get("sql") or {})
    for route_manifest in (
        routes,
        body.get("llm_route_manifest") or {},
    ):
        route_sql = dict(route_manifest.get("sql") or {})
        route_manifest["sql"] = {
            key: (
                "<authorized-model-ablation>"
                if key in MODEL_ABLATION_SQL_ROUTE_FIELDS else value
            )
            for key, value in route_sql.items()
        }
    return {
        "schema": MODEL_ABLATION_SCHEMA,
        "authorized_route": "sql",
        "allowed_route_fields": sorted(MODEL_ABLATION_SQL_ROUTE_FIELDS),
        "frozen_config_hash": stable_hash(body),
        "sql_route": sql_route,
        "correction_route": dict(routes.get("correction") or {}),
    }


def _pipeline_ablation_manifest(safe_config: dict[str, Any]) -> dict[str, Any]:
    """Freeze every setting except one pre-registered pipeline factor.

    Each factor receives its own normalized hash.  This keeps ordinary
    baseline governance strict while allowing an offline comparator to prove
    that a paid A/B run changed only the declared ablation variable.
    """

    source = json.loads(json.dumps(safe_config, ensure_ascii=False, default=str))

    query_signal = json.loads(json.dumps(source, ensure_ascii=False))
    query_manifest = ((query_signal.get("query") or {}).get("query_signals") or {})
    query_selected = {
        "profile": query_manifest.get("profile"),
        "route_policy": query_manifest.get("route_policy"),
    }
    query_manifest["profile"] = "<authorized-query-signal-ablation>"
    query_manifest["route_policy"] = "<authorized-query-signal-ablation>"

    planning = json.loads(json.dumps(source, ensure_ascii=False))
    planning_manifest = planning.get("planning_route") or {}
    planning_selected = {
        "active_profile": planning_manifest.get("active_profile"),
    }
    planning_manifest["active_profile"] = "<authorized-planning-route-ablation>"

    semantic = json.loads(json.dumps(source, ensure_ascii=False))
    semantic_manifest = semantic.get("semantic_repair") or {}
    semantic_planning = semantic.get("planning_route") or {}
    semantic_selected = {
        "active_profile": semantic_manifest.get("active_profile"),
        "planning_repair_profile": semantic_planning.get("repair_profile"),
    }
    semantic_manifest["active_profile"] = "<authorized-semantic-repair-ablation>"
    semantic_planning["repair_profile"] = (
        "<authorized-semantic-repair-ablation>"
    )

    return {
        "schema": PIPELINE_ABLATION_SCHEMA,
        "factors": {
            "query_signals": {
                "selected": query_selected,
                "frozen_config_hash": stable_hash(query_signal),
            },
            "planning_route": {
                "selected": planning_selected,
                "frozen_config_hash": stable_hash(planning),
            },
            "semantic_repair": {
                "selected": semantic_selected,
                "frozen_config_hash": stable_hash(semantic),
            },
        },
    }


def compare_model_ablation_fingerprints(
    left: dict | None, right: dict | None,
) -> dict[str, Any]:
    """Compare two runs while allowing only registered SQL-route differences."""
    left, right = left or {}, right or {}
    left_manifest = left.get("model_ablation") or {}
    right_manifest = right.get("model_ablation") or {}
    unexpected: list[str] = []
    if (
        left.get("complete") is not True
        or right.get("complete") is not True
        or any(not left.get(field) or not right.get(field) for field in REQUIRED_FINGERPRINT_FIELDS)
    ):
        unexpected.append("fingerprint_incomplete")
    if (
        left_manifest.get("schema") != MODEL_ABLATION_SCHEMA
        or right_manifest.get("schema") != MODEL_ABLATION_SCHEMA
    ):
        unexpected.append("model_ablation_manifest_missing")
    frozen_fields = [
        field for field in REQUIRED_FINGERPRINT_FIELDS
        if field not in {"model", "config_hash"}
    ]
    unexpected.extend(
        field for field in frozen_fields
        if left.get(field) != right.get(field)
    )
    if left_manifest.get("frozen_config_hash") != right_manifest.get("frozen_config_hash"):
        unexpected.append("frozen_config_hash")
    left_sql = dict(left_manifest.get("sql_route") or {})
    right_sql = dict(right_manifest.get("sql_route") or {})
    route_keys = sorted(set(left_sql) | set(right_sql))
    allowed = [
        f"model.routes.sql.{key}"
        for key in route_keys
        if left_sql.get(key) != right_sql.get(key)
        and key in MODEL_ABLATION_SQL_ROUTE_FIELDS
    ]
    unexpected.extend(
        f"model.routes.sql.{key}"
        for key in route_keys
        if left_sql.get(key) != right_sql.get(key)
        and key not in MODEL_ABLATION_SQL_ROUTE_FIELDS
    )
    unexpected = sorted(set(unexpected))
    return {
        "schema": MODEL_ABLATION_SCHEMA,
        "allowed_differences": allowed,
        "unexpected_differences": unexpected,
        "attribution_valid": not unexpected,
    }


def _flatten_pipeline_profiles(
    value: dict[str, Any], prefix: str = "pipeline_profiles",
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, item in sorted(value.items()):
        path = f"{prefix}.{key}"
        if isinstance(item, dict):
            output.update(_flatten_pipeline_profiles(item, path))
        else:
            output[path] = item
    return output


def _pipeline_manifest_errors(
    fingerprint: dict[str, Any], *, side: str,
) -> list[str]:
    """Validate registry identity and manifest/profile consistency."""

    errors: list[str] = []
    manifest = fingerprint.get("pipeline_ablation") or {}
    manifest = manifest if isinstance(manifest, dict) else {}
    profiles = fingerprint.get("pipeline_profiles") or {}
    profiles = profiles if isinstance(profiles, dict) else {}
    factors = manifest.get("factors") or {}
    factors = factors if isinstance(factors, dict) else {}
    expected_factors = {"query_signals", "planning_route", "semantic_repair"}
    if manifest.get("schema") != PIPELINE_ABLATION_SCHEMA:
        errors.append(f"{side}.pipeline_ablation_manifest_missing")
    if set(factors) != expected_factors:
        errors.append(f"{side}.pipeline_ablation_factors_incomplete")
    if not isinstance(profiles, dict) or set(profiles) != expected_factors:
        errors.append(f"{side}.pipeline_profiles_incomplete")
        return errors

    query = profiles.get("query_signals") or {}
    planning = profiles.get("planning_route") or {}
    repair = profiles.get("semantic_repair") or {}
    expected_selected = {
        "query_signals": {
            "profile": query.get("profile"),
            "route_policy": query.get("route_policy"),
        },
        "planning_route": {
            "active_profile": planning.get("active_profile"),
        },
        "semantic_repair": {
            "active_profile": repair.get("active_profile"),
            "planning_repair_profile": planning.get("repair_profile"),
        },
    }
    for factor in sorted(expected_factors):
        factor_manifest = factors.get(factor) or {}
        if factor_manifest.get("selected") != expected_selected[factor]:
            errors.append(f"{side}.{factor}.selected_mismatch")
        if not factor_manifest.get("frozen_config_hash"):
            errors.append(f"{side}.{factor}.frozen_config_hash_missing")

    if query.get("profile") not in PROFILE_VERSIONS:
        errors.append(f"{side}.query_signals.profile_unknown")
    if query.get("route_policy") not in ROUTE_POLICIES:
        errors.append(f"{side}.query_signals.route_policy_unknown")
    if planning.get("active_profile") not in IMPLEMENTED_PROFILES:
        errors.append(f"{side}.planning_route.profile_unknown")
    if planning.get("fallback_profile") not in IMPLEMENTED_PROFILES:
        errors.append(f"{side}.planning_route.fallback_unknown")
    if repair.get("active_profile") not in PROFILE_RULES:
        errors.append(f"{side}.semantic_repair.profile_unknown")
    if repair.get("fallback_profile") not in PROFILE_RULES:
        errors.append(f"{side}.semantic_repair.fallback_unknown")
    if planning.get("repair_profile") not in PROFILE_RULES:
        errors.append(f"{side}.planning_route.repair_profile_unknown")
    if planning.get("repair_profile") != repair.get("active_profile"):
        errors.append(f"{side}.semantic_repair.profile_alignment_mismatch")
    return errors


def compare_pipeline_ablation_fingerprints(
    left: dict | None,
    right: dict | None,
    *,
    factor: str,
) -> dict[str, Any]:
    """Prove that exactly one registered pipeline factor changed.

    The factor-specific frozen hash masks only the selected setting(s), so any
    retrieval, model, prompt, fallback, threshold or other config drift fails
    closed.  Per-run manifests are also checked against their allow-listed
    profile summaries to prevent a hand-edited artifact from claiming another
    profile than the one bound into its fingerprint.
    """

    aliases = {
        "query_signal": "query_signals",
        "query_signals": "query_signals",
        "planning_route": "planning_route",
        "semantic_repair": "semantic_repair",
    }
    canonical = aliases.get(str(factor or "").strip())
    allowed_paths = {
        "query_signals": {
            "pipeline_profiles.query_signals.profile",
            "pipeline_profiles.query_signals.route_policy",
        },
        "planning_route": {
            "pipeline_profiles.planning_route.active_profile",
        },
        "semantic_repair": {
            "pipeline_profiles.semantic_repair.active_profile",
            "pipeline_profiles.planning_route.repair_profile",
        },
    }
    left, right = left or {}, right or {}
    unexpected: list[str] = []
    allowed: list[str] = []
    if canonical is None:
        unexpected.append("unsupported_factor")
    if (
        left.get("complete") is not True
        or right.get("complete") is not True
        or any(
            not left.get(field) or not right.get(field)
            for field in REQUIRED_FINGERPRINT_FIELDS
        )
    ):
        unexpected.append("fingerprint_incomplete")

    unexpected.extend(_pipeline_manifest_errors(left, side="left"))
    unexpected.extend(_pipeline_manifest_errors(right, side="right"))
    frozen_fields = [
        field for field in REQUIRED_FINGERPRINT_FIELDS if field != "config_hash"
    ]
    unexpected.extend(
        field for field in frozen_fields if left.get(field) != right.get(field)
    )

    left_profiles = left.get("pipeline_profiles") or {}
    left_profiles = left_profiles if isinstance(left_profiles, dict) else {}
    right_profiles = right.get("pipeline_profiles") or {}
    right_profiles = right_profiles if isinstance(right_profiles, dict) else {}
    left_flat = _flatten_pipeline_profiles(left_profiles)
    right_flat = _flatten_pipeline_profiles(right_profiles)
    changed_paths = {
        path
        for path in set(left_flat) | set(right_flat)
        if left_flat.get(path) != right_flat.get(path)
    }
    if canonical is not None:
        allowed = sorted(changed_paths & allowed_paths[canonical])
        unexpected.extend(sorted(changed_paths - allowed_paths[canonical]))
        left_manifest = left.get("pipeline_ablation") or {}
        left_manifest = left_manifest if isinstance(left_manifest, dict) else {}
        right_manifest = right.get("pipeline_ablation") or {}
        right_manifest = right_manifest if isinstance(right_manifest, dict) else {}
        left_factors = left_manifest.get("factors") or {}
        left_factors = left_factors if isinstance(left_factors, dict) else {}
        right_factors = right_manifest.get("factors") or {}
        right_factors = right_factors if isinstance(right_factors, dict) else {}
        left_factor = left_factors.get(canonical) or {}
        right_factor = right_factors.get(canonical) or {}
        if (
            not left_factor.get("frozen_config_hash")
            or left_factor.get("frozen_config_hash")
            != right_factor.get("frozen_config_hash")
        ):
            unexpected.append("frozen_config_hash")
        if left_factor.get("selected") == right_factor.get("selected"):
            unexpected.append("declared_factor_unchanged")
        for other in sorted(set(allowed_paths) - {canonical}):
            if (left_factors.get(other) or {}).get("selected") != (
                right_factors.get(other) or {}
            ).get("selected"):
                unexpected.append(f"factors.{other}.selected")
        if left.get("config_hash") == right.get("config_hash"):
            unexpected.append("config_hash_unchanged")

    unexpected = sorted(set(unexpected))
    return {
        "schema": PIPELINE_ABLATION_SCHEMA,
        "factor": canonical or str(factor or ""),
        "allowed_differences": allowed,
        "unexpected_differences": unexpected,
        "attribution_valid": not unexpected,
    }


def sample_contract_report(cases: list[dict]) -> dict:
    ids = [str(case.get("id") or "") for case in cases]
    missing: dict[str, list[str]] = {}
    unsafe: list[str] = []
    invalid_tables: list[str] = []
    for case in cases:
        case_id = str(case.get("id") or "<unknown>")
        absent = [field for field in REQUIRED_CASE_FIELDS if field not in case or case.get(field) is None]
        if absent:
            missing[case_id] = absent
        try:
            check_sql_read(str(case.get("expected_sql") or ""))
        except Exception:  # noqa: BLE001 - contract treats every parser failure as unsafe
            unsafe.append(case_id)
        required = {str(item) for item in (case.get("required_tables") or [])}
        allowed = {str(item) for item in (case.get("allowed_tables") or [])}
        if not required or not required.issubset(allowed):
            invalid_tables.append(case_id)
    duplicates = sorted({case_id for case_id in ids if case_id and ids.count(case_id) > 1})
    reviewed = sum(str(case.get("review_status") or "").lower() == "approved" for case in cases)
    complete = not missing and not duplicates and not unsafe and not invalid_tables
    return {
        "total": len(cases),
        "complete": complete,
        "complete_count": len(cases) - len(missing),
        "reviewed_count": reviewed,
        "all_reviewed": reviewed == len(cases) and bool(cases),
        "duplicate_ids": duplicates,
        "unsafe_sql_case_ids": unsafe,
        "invalid_table_contract_case_ids": invalid_tables,
        "missing_fields": missing,
        "dataset_version": stable_hash(cases),
    }


def _safe_config_manifest(evaluation_profile: str = "core-default") -> dict:
    """Allow-list non-secret settings; never serialize config objects wholesale."""
    profile = resolve_evaluation_profile(evaluation_profile)
    return {
        "evaluation_profile": profile,
        "llm": {
            "model_name": app_config.llm.model_name,
            "temperature": 0,
            "timeout_seconds": app_config.llm.timeout_seconds,
            "max_retries": app_config.llm.max_retries,
            "max_completion_tokens": app_config.llm.max_completion_tokens,
            "routes": llm_route_manifest(),
        },
        "embedding": {
            "model": app_config.embedding.model,
            "revision": app_config.embedding.revision,
            "runtime_mode": app_config.embedding.runtime_mode,
            "scheduler": {
                "queue_capacity": app_config.embedding.scheduler.queue_capacity,
                "batch_window_ms": app_config.embedding.scheduler.batch_window_ms,
                "max_batch_texts": app_config.embedding.scheduler.max_batch_texts,
                "max_batch_tokens": app_config.embedding.scheduler.max_batch_tokens,
                "request_deadline_ms": (
                    app_config.embedding.scheduler.request_deadline_ms
                ),
                "offline_corpus_quota": (
                    app_config.embedding.scheduler.offline_corpus_quota
                ),
            },
        },
        "query": {
            "limit_rows": app_config.query.limit_rows,
            "cost_guard_enabled": app_config.query.cost_guard_enabled,
            "max_explain_rows": app_config.query.max_explain_rows,
            "chart_llm_enabled": app_config.query.chart_llm_enabled,
            "value_recall_llm_enabled": app_config.query.value_recall_llm_enabled,
            "postprocess_timeout_seconds": app_config.query.postprocess_timeout_seconds,
            "evaluation_sql_timeout_seconds": (
                app_config.query.evaluation_sql_timeout_seconds
            ),
            "mysql_server_cancellation": {
                "control_timeout_seconds": (
                    app_config.query.mysql_cancel_control_timeout_seconds
                ),
                "drain_timeout_seconds": (
                    app_config.query.mysql_cancel_drain_timeout_seconds
                ),
                "max_cleanup_grace_seconds": (
                    app_config.query.mysql_cancel_control_timeout_seconds
                    + app_config.query.mysql_cancel_drain_timeout_seconds
                ),
                "strategy": (
                    "independent_asyncmy_kill_then_transport_close_fallback_v2"
                ),
                "transport_close_fallback": True,
                "session_invalidate_only_after_query_task_drained": True,
            },
            "analysis_enabled_default": app_config.query.analysis_enabled_default,
            "analysis_enabled_for_evaluation": profile["analysis_enabled"],
            "analysis_timeout_seconds": app_config.query.analysis_timeout_seconds,
            "postprocess_budget_ms": app_config.query.postprocess_budget_ms,
            "model_invocation": {
                "planning_llm_enabled": (
                    app_config.query.planning_llm_enabled
                ),
                "request_deadline_ms": (
                    app_config.query.request_deadline_ms
                ),
                "phase_budgets_ms": dict(
                    app_config.query.request_phase_budgets_ms or {}
                ),
                "execution_reserve_ms": (
                    app_config.query.request_execution_reserve_ms
                ),
                "recovery_reserve_ms": (
                    app_config.query.llm_recovery_reserve_ms
                ),
                "recovery_min_budget_ms": (
                    app_config.query.llm_recovery_min_budget_ms
                ),
                "planning_min_start_budget_ms": (
                    app_config.query.planning_llm_min_start_budget_ms
                ),
                "generation_min_start_budget_ms": (
                    app_config.query.generation_llm_min_start_budget_ms
                ),
                "correction_min_start_budget_ms": (
                    app_config.query.correction_llm_min_start_budget_ms
                ),
                "generation_primary_min_budget_ms": (
                    app_config.query.generation_llm_primary_min_budget_ms
                ),
                "correction_primary_min_budget_ms": (
                    app_config.query.correction_llm_primary_min_budget_ms
                ),
                "timeout_recovery_enabled": True,
            },
            "retrieval": retrieval_parameters(),
            "candidate_confidence": {
                "version": CONFIDENCE_VERSION,
                "enabled": app_config.query.adaptive_candidate_enabled,
                "high": app_config.query.candidate_confidence_high,
                "low": app_config.query.candidate_confidence_low,
                "min_gain": app_config.query.candidate_min_confidence_gain,
                "deadline_ms": app_config.query.adaptive_candidate_deadline_ms,
                "timeout_seconds": app_config.query.adaptive_candidate_timeout_seconds,
            },
            "semantic_review": {
                "enabled": bool(
                    getattr(
                        app_config.query,
                        "sql_semantic_review_enabled",
                        True,
                    )
                ),
                "policy_version": SEMANTIC_REVIEW_POLICY_VERSION,
            },
            "prompt_payload": {
                "version": PROMPT_PAYLOAD_VERSION,
                "history_limit": app_config.query.prompt_history_limit,
                "training_limit": app_config.query.prompt_training_limit,
                "column_example_limit": app_config.query.prompt_column_example_limit,
                "description_max_chars": app_config.query.prompt_description_max_chars,
            },
            "query_signals": {
                "contract_version": QUERY_SIGNAL_CONTRACT_VERSION,
                "extractor_version": EXTRACTOR_VERSION,
                "route_policy_version": ROUTE_POLICY_VERSION,
                "profile": app_config.query_signals.profile,
                "route_policy": app_config.query_signals.route_policy,
                "lexical_top_k": app_config.query_signals.lexical_top_k,
                "remote_signal_limit": app_config.query_signals.remote_signal_limit,
                "workspace_dictionary_enabled": (
                    app_config.query_signals.workspace_dictionary_enabled
                ),
                "custom_idf_enabled": (
                    app_config.query_signals.custom_idf_enabled
                ),
                "cache_namespace_version": (
                    app_config.query_signals.cache_namespace_version
                ),
                "log_plaintext": app_config.query_signals.log_plaintext,
            },
        },
        # Pipeline ablations must be distinguishable in immutable run
        # fingerprints.  These values are deliberately allow-listed and
        # contain no credentials or raw user text.  Without them, two runs
        # using different planning/repair profiles could share config_hash
        # and appear comparable even though they exercised different code
        # paths.
        "planning_route": {
            "active_profile": app_config.planning_route.active_profile,
            "fallback_profile": app_config.planning_route.fallback_profile,
            "shadow_profiles": list(
                app_config.planning_route.shadow_profiles or []
            ),
            "contract_version": app_config.planning_route.contract_version,
            "feature_schema": app_config.planning_route.feature_schema,
            "repair_profile": app_config.planning_route.repair_profile,
            "cache_namespace_version": (
                app_config.planning_route.cache_namespace_version
            ),
            "direct_threshold": app_config.planning_route.direct_threshold,
            "plan_threshold": app_config.planning_route.plan_threshold,
            "linear_direct_threshold": (
                app_config.planning_route.linear_direct_threshold
            ),
            "linear_plan_threshold": (
                app_config.planning_route.linear_plan_threshold
            ),
            "scorecard_weights": dict(
                app_config.planning_route.scorecard_weights or {}
            ),
            "linear_coefficients": dict(
                app_config.planning_route.linear_coefficients or {}
            ),
            "log_plaintext": app_config.planning_route.log_plaintext,
        },
        "semantic_repair": {
            "active_profile": app_config.semantic_repair.active_profile,
            "fallback_profile": app_config.semantic_repair.fallback_profile,
            "shadow_profiles": list(
                app_config.semantic_repair.shadow_profiles or []
            ),
            "contract_version": app_config.semantic_repair.contract_version,
            "registry_version": app_config.semantic_repair.registry_version,
            "rule_set_version": app_config.semantic_repair.rule_set_version,
            "max_passes": app_config.semantic_repair.max_passes,
            "log_plaintext": app_config.semantic_repair.log_plaintext,
        },
        "chat": {
            "history_limit": app_config.chat.history_limit,
            "follow_up_version": FOLLOW_UP_VERSION,
        },
        "memory": {
            "enabled": app_config.memory.enabled,
            "long_term_enabled": app_config.memory.long_term_enabled,
            "context_schema_version": app_config.memory.context_schema_version,
            "extractor_version": app_config.memory.extractor_version,
            "es_projection_version": app_config.memory.es_projection_version,
            "qdrant_projection_version": app_config.memory.qdrant_projection_version,
            "long_term_top_k": app_config.memory.long_term_top_k,
            "fusion_rrf_k": app_config.memory.fusion_rrf_k,
            "context_token_budget": app_config.memory.context_token_budget,
        },
        "masking": {
            "enabled": app_config.masking.enabled,
            "heuristic_enabled": app_config.masking.heuristic_enabled,
            "default_strategy": app_config.masking.default_strategy,
        },
    }


def _prompt_hashes() -> dict[str, str]:
    prompt_dir = PROJECT_ROOT / "prompts"
    return {
        path.name: "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(prompt_dir.glob("*.prompt"))
    }


def resolve_trace_version_snapshot(workspace_id: str, runs_dir: Path | None = None) -> dict:
    """Bind an online trace to the newest complete full-dataset evaluation fingerprint."""
    root = runs_dir or RUNS_ROOT
    selected: dict = {}
    candidates: list[dict] = []
    if root.exists():
        for path in sorted(root.glob("run_*.json"), reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            fingerprint = payload.get("fingerprint") or {}
            if (
                str(payload.get("workspace_id") or "") == workspace_id
                and payload.get("valid") is True
                and fingerprint.get("complete") is True
                and all(fingerprint.get(field) for field in REQUIRED_FINGERPRINT_FIELDS)
            ):
                candidates.append(payload)
    if candidates:
        largest_sample = max(int(item.get("n") or len(item.get("cases") or [])) for item in candidates)
        selected = next(
            item for item in candidates
            if int(item.get("n") or len(item.get("cases") or [])) == largest_sample
        )
    fingerprint = selected.get("fingerprint") or {}
    prompt_version = stable_hash(_prompt_hashes())
    components = {
        "prompt_version": prompt_version,
        "knowledge_version": fingerprint.get("knowledge_version") or "",
        "schema_version": fingerprint.get("schema_version") or "",
        "index_version": fingerprint.get("index_version") or "",
        "model_version": app_config.llm.model_name,
        "version_run_id": selected.get("run_id"),
    }
    complete = all(components.get(key) for key in (
        "prompt_version", "knowledge_version", "schema_version", "index_version", "model_version", "version_run_id",
    ))
    return {
        **components,
        "run_fingerprint": stable_hash(components) if complete else "",
        "version_complete": complete,
    }


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=5, check=True,
        )
        return result.stdout.strip() or "unavailable"
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _code_version() -> str:
    """Hash source content as a safeguard when the project is outside Git tracking."""
    digest = hashlib.sha256()
    paths = sorted((PROJECT_ROOT / "app").rglob("*.py"))
    paths += sorted((PROJECT_ROOT / "prompts").glob("*.prompt"))
    for path in paths:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest() if paths else ""


def _dependency_lock_hash() -> str:
    digest = hashlib.sha256()
    paths = [PROJECT_ROOT / "pyproject.toml", PROJECT_ROOT / "uv.lock"]
    existing = [path for path in paths if path.exists()]
    for path in existing:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest() if len(existing) == len(paths) else ""


def local_version_fingerprint(
    *,
    cases: list[dict[str, Any]],
    evaluation_profile: str,
) -> dict[str, str]:
    """只用本地冻结输入生成可复用制品的最低版本绑定。"""
    return {
        "code_version": _code_version(),
        "dependency_lock_hash": _dependency_lock_hash(),
        "config_hash": stable_hash(_safe_config_manifest(evaluation_profile)),
        "dataset_version": sample_contract_report(cases)["dataset_version"],
    }


def compare_local_version_fingerprints(
    artifact: dict[str, Any] | None,
    current: dict[str, Any] | None,
) -> dict[str, Any]:
    """比较可公开报告的字段名，不返回任一侧的实际指纹值。"""
    artifact, current = artifact or {}, current or {}
    differences = [
        field
        for field in LOCAL_VERSION_FINGERPRINT_FIELDS
        if not artifact.get(field)
        or not current.get(field)
        or artifact.get(field) != current.get(field)
    ]
    return {
        "comparable": not differences,
        "differences": differences,
    }


async def _dw_manifest(dw_repo) -> tuple[dict, dict]:
    snapshot = await dw_repo.get_schema_snapshot()
    schema = {
        "source": snapshot.get("source"),
        "tables": [
            {
                "name": table.get("name"),
                "role": table.get("role"),
                "columns": [
                    {"name": col.get("name"), "type": col.get("type"), "role": col.get("role")}
                    for col in table.get("columns") or []
                ],
            }
            for table in snapshot.get("tables") or []
        ],
    }
    row_counts: dict[str, int] = {}
    for table in schema["tables"]:
        name = str(table.get("name") or "")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"非法快照表名：{name}")
        row_counts[name] = int((await dw_repo.session.execute(text(f"SELECT COUNT(*) FROM `{name}`"))).scalar() or 0)
    return schema, {"schema_version": stable_hash(schema), "row_counts": row_counts}


async def _knowledge_manifest(meta_repo) -> dict:
    tables = []
    for table, columns in await meta_repo.list_table_infos():
        tables.append({
            "id": table.id, "name": table.name, "role": table.role,
            "columns": [
                {
                    "id": col.id, "name": col.name, "type": col.type, "role": col.role,
                    "description": col.description, "alias": col.alias, "sync": col.sync,
                    "is_display": bool(getattr(col, "is_display", False)),
                }
                for col in columns
            ],
        })
    metrics = [
        {
            "id": item.id, "name": item.name, "description": item.description,
            "relevant_columns": item.relevant_columns, "alias": item.alias,
            "formula_rule": item.formula_rule,
        }
        for item in await meta_repo.get_all_metric_infos()
    ]
    trainings = [
        {"id": item.id, "question_hash": stable_hash(item.question), "sql_hash": stable_hash(item.sql)}
        for item in await meta_repo.get_all_enabled_training_infos()
    ]
    return {"tables": tables, "metrics": metrics, "trainings": trainings}


async def collect_run_fingerprint(
    *, selected_cases: list[dict], all_cases: list[dict], dw_repo, meta_repo,
    value_repo=None,
    evaluation_profile: str = "core-default",
) -> dict:
    contract = sample_contract_report(all_cases)
    profile = resolve_evaluation_profile(evaluation_profile)
    safe_config = _safe_config_manifest(profile["name"])
    errors: list[str] = []
    schema: dict = {}
    dw: dict = {}
    knowledge: dict = {}
    value_manifest: dict = {}
    try:
        schema, dw = await _dw_manifest(dw_repo)
    except Exception as exc:  # noqa: BLE001 - fingerprint must degrade, not abort
        errors.append(f"dw_snapshot:{type(exc).__name__}")
    try:
        knowledge = await _knowledge_manifest(meta_repo)
    except Exception as exc:  # noqa: BLE001 - fingerprint must degrade, not abort
        errors.append(f"knowledge:{type(exc).__name__}")
    if value_repo is not None:
        try:
            value_manifest = await value_repo.active_manifest(
                str(getattr(meta_repo, "workspace_id", "") or "unknown")
            )
        except Exception as exc:  # noqa: BLE001 - optional index evidence
            errors.append(f"value_index:{type(exc).__name__}")
    schema_version = stable_hash(schema) if schema else ""
    knowledge_version = stable_hash(knowledge) if knowledge else ""
    rag_index_version = stable_hash({
        "knowledge_version": knowledge_version,
        "retrieval": retrieval_parameters(),
        "embedding_model": app_config.embedding.model,
        "embedding_revision": app_config.embedding.revision,
        "point_strategy": app_config.point_projection.strategy,
    }) if knowledge else ""
    value_active_version = str(value_manifest.get("index_version") or "")
    index_manifest = {
        "fingerprint_schema": "evaluation-v2",
        "schema_version": schema_version,
        "knowledge_version": knowledge_version,
        "rag_index_version": rag_index_version,
        "value_index_version": value_active_version,
        "retrieval": retrieval_parameters(),
        "value_index_alias": value_manifest.get("alias"),
        "value_index_contract": app_config.es.value_contract_version,
        "conversation_memory": {
            "enabled": app_config.memory.enabled,
            "context_schema_version": app_config.memory.context_schema_version,
            "extractor_version": app_config.memory.extractor_version,
            "es_index": app_config.memory.es_index_name,
            "es_projection_version": app_config.memory.es_projection_version,
            "qdrant_collection": app_config.memory.qdrant_collection_name,
            "qdrant_projection_version": app_config.memory.qdrant_projection_version,
        },
    }
    fingerprint = {
        "fingerprint_schema": "evaluation-v2",
        "dataset_version": contract["dataset_version"],
        "selection_version": stable_hash([case.get("id") for case in selected_cases]),
        "dw_snapshot": stable_hash(dw) if dw else "",
        "schema_version": schema_version,
        "knowledge_version": knowledge_version,
        "rag_index_version": rag_index_version,
        "value_index_version": value_active_version,
        "index_manifest": index_manifest,
        "index_version": stable_hash(index_manifest) if schema and knowledge else "",
        "prompt_hashes": _prompt_hashes(),
        "model": safe_config["llm"] | {
            "embedding_model": app_config.embedding.model,
            "embedding_runtime_profile": (
                "microbatch_inprocess_v1"
                if app_config.embedding.runtime_mode == "microbatch"
                else "serial_inprocess_v1"
            ),
        },
        "evaluation_profile": profile,
        "semantic_review": safe_config["query"]["semantic_review"],
        "pipeline_profiles": {
            "query_signals": safe_config["query"]["query_signals"],
            "planning_route": safe_config["planning_route"],
            "semantic_repair": safe_config["semantic_repair"],
        },
        "config_hash": stable_hash(safe_config),
        "model_ablation": _model_ablation_manifest(safe_config),
        "pipeline_ablation": _pipeline_ablation_manifest(safe_config),
        "git_commit": _git_revision(),
        "code_version": _code_version(),
        "dependency_lock_hash": _dependency_lock_hash(),
        "sample_contract": contract,
        "snapshot_summary": {
            "table_count": len(schema.get("tables") or []),
            "row_count_total": sum((dw.get("row_counts") or {}).values()),
            "knowledge_tables": len(knowledge.get("tables") or []),
            "knowledge_metrics": len(knowledge.get("metrics") or []),
            "knowledge_trainings": len(knowledge.get("trainings") or []),
        },
        "errors": errors,
    }
    fingerprint["complete"] = (
        not errors
        and contract["complete"]
        and contract["all_reviewed"]
        and fingerprint["git_commit"] != "unavailable"
        and all(bool(fingerprint.get(field)) for field in REQUIRED_FINGERPRINT_FIELDS)
    )
    fingerprint["run_fingerprint"] = stable_hash({
        field: fingerprint.get(field) for field in REQUIRED_FINGERPRINT_FIELDS
    }) if fingerprint["complete"] else ""
    return fingerprint


def compare_fingerprints(left: dict | None, right: dict | None) -> dict:
    left, right = left or {}, right or {}
    if (not left.get("complete") or not right.get("complete")
            or not all(left.get(field) for field in REQUIRED_FINGERPRINT_FIELDS)
            or not all(right.get(field) for field in REQUIRED_FINGERPRINT_FIELDS)):
        return {"comparable": False, "differences": ["fingerprint_incomplete"]}
    differences = [
        field for field in REQUIRED_FINGERPRINT_FIELDS
        if left.get(field) != right.get(field)
    ]
    return {"comparable": not differences, "differences": differences}


def build_experiment_metadata(
    *, experiment_id: str, candidate_label: str, repeat_index: int,
    cache_state: str, provider_parameters: dict | None = None,
    evaluation_profile: str = "core-default",
) -> dict:
    """Create validated, non-secret protocol metadata for one full evaluation run."""
    experiment_id = str(experiment_id or "").strip()
    candidate_label = str(candidate_label or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,79}", experiment_id):
        raise ValueError("experiment_id 需为 3～80 位字母、数字、点、下划线或连字符")
    if not candidate_label or len(candidate_label) > 120:
        raise ValueError("candidate_label 不能为空且最多 120 字符")
    if repeat_index not in range(1, EXPERIMENT_REQUIRED_REPEATS + 1):
        raise ValueError(f"repeat_index 只允许 1～{EXPERIMENT_REQUIRED_REPEATS}")
    if cache_state not in {"cold", "warm"}:
        raise ValueError("cache_state 只允许 cold 或 warm")
    profile = resolve_evaluation_profile(evaluation_profile)
    return {
        "schema": EXPERIMENT_SCHEMA,
        "experiment_id": experiment_id,
        "candidate_label": candidate_label,
        "repeat_index": int(repeat_index),
        "cache_state": cache_state,
        "provider_parameters": dict(provider_parameters or {}),
        "evaluation_profile": profile,
        "recorded_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def _experiment_metadata_complete(value: dict) -> bool:
    return bool(
        value.get("schema") == EXPERIMENT_SCHEMA
        and value.get("experiment_id")
        and value.get("candidate_label")
        and value.get("repeat_index") in range(1, EXPERIMENT_REQUIRED_REPEATS + 1)
        and value.get("cache_state") in {"cold", "warm"}
        and isinstance(value.get("provider_parameters"), dict)
        and evaluation_profile_name(value.get("evaluation_profile"))
        in {"core-default", "enhanced-analysis"}
    )


def _case_result_map(run: dict) -> dict[str, bool]:
    return {
        str(case.get("id")): bool(case.get("result_equivalent"))
        for case in (run.get("cases") or []) if case.get("id")
    }


def _case_first_result_map(run: dict) -> dict[str, bool]:
    return {
        str(case.get("id")): bool(
            (case.get("first_sql_metrics") or {}).get("result_equivalent")
        )
        for case in (run.get("cases") or []) if case.get("id")
    }


PROMOTION_SUITE_POLICIES: dict[str, dict[str, int | float | None]] = {
    # The rates are exclusive: Easy must be >95% (96/100), while Medium
    # must be >85% (86/100).  Hard remains an observation suite and has no
    # mandatory result rate, but it is still subject to the common protocol,
    # validity, infrastructure and evidence checks in ``experiment_report``.
    "easy_sql_v1": {"case_count": 100, "result_rate_exclusive": 0.95},
    "medium_sql_v1": {"case_count": 100, "result_rate_exclusive": 0.85},
    "hard_sql_v1": {"case_count": 50, "result_rate_exclusive": None},
}
_SUPPORTED_PROMOTION_SUITES = frozenset(PROMOTION_SUITE_POLICIES)


def minimum_passing_results(dataset_suite: str, total: int) -> int | None:
    """Return the strict absolute result minimum for a promotion suite.

    ``None`` means that the supported suite intentionally has no absolute
    result-rate gate.  Unsupported suites also return ``None`` and are rejected
    separately by ``dataset_suite_supported``.
    """
    policy = PROMOTION_SUITE_POLICIES.get(str(dataset_suite or "").strip())
    rate = (policy or {}).get("result_rate_exclusive")
    if rate is None:
        return None
    return math.floor(max(0, int(total)) * float(rate)) + 1


def suite_result_floor_met(
    dataset_suite: str, *, passed: int, total: int,
) -> bool:
    """Check the suite-sized, strict final-result promotion floor."""
    policy = PROMOTION_SUITE_POLICIES.get(str(dataset_suite or "").strip())
    if not policy or int(total) != int(policy["case_count"] or 0):
        return False
    minimum = minimum_passing_results(dataset_suite, total)
    return minimum is None or int(passed) >= minimum


def _dataset_suite_id(run: dict | None) -> str:
    """Return a canonical suite identity without reviving retired suites."""
    payload = run or {}
    return str(payload.get("dataset_suite") or "").strip()


def _case_semantic_family(case: dict) -> str:
    evidence = case.get("stage_evidence") or {}
    semantic_ir = evidence.get("semantic_ir") or {}
    return str(
        semantic_ir.get("semantic_family")
        or semantic_ir.get("intent")
        or case.get("category")
        or "unclassified"
    )


def _candidate_governance_summary(run: dict) -> dict[str, Any]:
    """Normalize split candidate lanes, including historical artifacts.

    The pre-v3 artifact schema overloaded ``adaptive_triggered`` for semantic
    review.  Re-reading each case through ``compact_candidate_decision`` lets
    semantic reason prefixes recover the correct lane without mutating the
    frozen artifact.
    """
    aggregate = run.get("aggregate") or {}
    cases = list(run.get("cases") or [])
    fingerprint_contract = (run.get("fingerprint") or {}).get(
        "semantic_review"
    )
    explicit_contract = isinstance(fingerprint_contract, dict) and isinstance(
        fingerprint_contract.get("enabled"), bool
    )

    normalized_decisions = [
        compact_candidate_decision(
            (case.get("stage_evidence") or {}).get("candidate_decision")
        )
        for case in cases
    ]
    inferred_review = any(
        item.get("decision_mode") == "semantic_review"
        for item in normalized_decisions
    )
    if explicit_contract:
        review_enabled = bool(fingerprint_contract.get("enabled"))
        policy_version = str(
            fingerprint_contract.get("policy_version") or ""
        )
    elif "semantic_review_enabled" in aggregate:
        review_enabled = bool(aggregate.get("semantic_review_enabled"))
        policy_version = str(
            aggregate.get("semantic_review_policy_version") or "legacy"
        )
    else:
        review_enabled = inferred_review
        policy_version = "legacy-inferred" if inferred_review else "disabled-legacy"

    if "adaptive_candidate_unsafe_selected" in aggregate:
        adaptive = {
            "rate": float(aggregate.get("adaptive_candidate_rate") or 0),
            "unsafe_selected": int(
                aggregate.get("adaptive_candidate_unsafe_selected") or 0
            ),
            "high_confidence_extra_calls": int(
                aggregate.get(
                    "adaptive_candidate_high_confidence_extra_calls"
                ) or 0
            ),
        }
    else:
        adaptive_rows = [
            decision for decision in normalized_decisions
            if decision.get("decision_mode") == "adaptive_expansion"
        ]
        if normalized_decisions:
            adaptive = {
                "rate": round(
                    sum(bool(row.get("adaptive_triggered")) for row in adaptive_rows)
                    / max(1, len(cases)),
                    4,
                ),
                "unsafe_selected": sum(
                    any(
                        candidate.get("label") == row.get("selected")
                        and candidate.get("valid") is False
                        for candidate in row.get("candidates") or []
                    )
                    for row in adaptive_rows
                ),
                "high_confidence_extra_calls": sum(
                    bool(row.get("extra_call_used"))
                    and any(
                        candidate.get("label") == "A"
                        and candidate.get("band") == "high"
                        for candidate in row.get("candidates") or []
                    )
                    for row in adaptive_rows
                ),
            }
        else:
            adaptive = {
                "rate": float(aggregate.get("adaptive_candidate_rate") or 0),
                "unsafe_selected": int(
                    aggregate.get("candidate_unsafe_selected") or 0
                ),
                "high_confidence_extra_calls": int(
                    aggregate.get("candidate_high_confidence_extra_calls") or 0
                ),
            }

    review_keys = {
        "semantic_review_eligible_count",
        "semantic_review_evidence_complete",
        "semantic_review_coverage_complete",
        "semantic_review_at_most_one_attempt",
        "semantic_review_unsafe_selected",
        "semantic_review_failed_core",
        "semantic_review_harmful_modification",
        "semantic_review_first_correct_final_incorrect",
        "semantic_review_same_sql_execution_loss",
        "semantic_review_selected_b_no_sql_delta",
        "semantic_review_result_attribution_complete",
    }
    if review_keys.issubset(aggregate):
        review = {
            "eligible": int(aggregate.get("semantic_review_eligible_count") or 0),
            "evidence_complete": bool(
                aggregate.get("semantic_review_evidence_complete")
            ),
            "coverage_complete": bool(
                aggregate.get("semantic_review_coverage_complete")
            ),
            "at_most_one_attempt": bool(
                aggregate.get("semantic_review_at_most_one_attempt")
            ),
            "unsafe_selected": int(
                aggregate.get("semantic_review_unsafe_selected") or 0
            ),
            "failed_core": int(
                aggregate.get("semantic_review_failed_core") or 0
            ),
            "harmful_modification": int(
                aggregate.get("semantic_review_harmful_modification") or 0
            ),
            "first_correct_final_incorrect": int(
                aggregate.get(
                    "semantic_review_first_correct_final_incorrect"
                ) or 0
            ),
            "same_sql_execution_loss": int(
                aggregate.get("semantic_review_same_sql_execution_loss") or 0
            ),
            "selected_b_no_sql_delta": int(
                aggregate.get("semantic_review_selected_b_no_sql_delta") or 0
            ),
            "budget_skipped": int(
                aggregate.get("semantic_review_budget_skipped") or 0
            ),
            "result_attribution_complete": bool(
                aggregate.get(
                    "semantic_review_result_attribution_complete",
                    True,
                )
            ),
        }
    else:
        # Local import avoids pulling the graph/evaluator into lightweight
        # fingerprint-only callers unless a historical artifact needs repair.
        from evals.legacy.evaluator import semantic_review_case_metrics

        metrics = [semantic_review_case_metrics(case) for case in cases]
        eligible_rows = [item for item in metrics if item.get("eligible")]
        review = {
            "eligible": len(eligible_rows),
            "evidence_complete": all(
                bool(item.get("evidence_complete")) for item in eligible_rows
            ),
            "coverage_complete": all(
                bool(item.get("coverage_complete")) for item in eligible_rows
            ),
            "at_most_one_attempt": all(
                bool(item.get("at_most_one_attempt")) for item in eligible_rows
            ),
            "unsafe_selected": sum(
                bool(item.get("unsafe_selected")) for item in eligible_rows
            ),
            "failed_core": sum(
                int(item.get("failed_core") or 0) for item in eligible_rows
            ),
            "harmful_modification": sum(
                bool(item.get("harmful_modification")) for item in eligible_rows
            ),
            "first_correct_final_incorrect": sum(
                bool(item.get("first_correct_final_incorrect"))
                for item in eligible_rows
            ),
            "same_sql_execution_loss": sum(
                bool(item.get("same_sql_execution_loss"))
                for item in eligible_rows
            ),
            "selected_b_no_sql_delta": sum(
                bool(item.get("selected_b_no_sql_delta"))
                for item in eligible_rows
            ),
            "budget_skipped": sum(
                bool(item.get("budget_skipped")) for item in eligible_rows
            ),
            "result_attribution_complete": all(
                bool(item.get("result_attribution_complete"))
                for item in eligible_rows
            ),
        }
    return {
        "review_enabled": review_enabled,
        "review_contract_explicit": explicit_contract,
        "review_policy_version": policy_version,
        "adaptive": adaptive,
        "review": review,
    }


def _strict_int(value: Any) -> int | None:
    """仅接受制品中的原生整数，避免宽松转换掩盖损坏证据。"""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def llm_release_evidence_checks(
    aggregate: dict | None,
) -> dict[str, bool]:
    """从原始聚合计数重新计算 LLM 正式准入门禁。"""

    aggregate = aggregate or {}
    failed_core = _strict_int(aggregate.get("failed_core_llm_calls"))
    timed_out = _strict_int(aggregate.get("llm_attempts_timed_out"))
    return {
        # 保留既有公开检查名，实际语义始终是核心链路失败为零。
        "failed_llm_calls_zero": failed_core == 0,
        "llm_attempts_timed_out_zero": timed_out == 0,
    }


def _embedding_epoch_identity(value: Any) -> tuple[str, int, str] | None:
    """读取一个完整且带 schema 的 metrics epoch 身份。"""
    if not isinstance(value, dict):
        return None
    epoch_id = str(value.get("id") or "")
    index = _strict_int(value.get("index"))
    started_at = str(value.get("started_at") or "")
    if (
        value.get("schema") != "embedding-runtime-metrics-epoch-v1"
        or not epoch_id
        or index is None
        or not started_at
    ):
        return None
    return epoch_id, index, started_at


def _schema_of(value: Any) -> str:
    """安全读取嵌套制品 schema。"""
    return str(value.get("schema") or "") if isinstance(value, dict) else ""


def _required_zero_counts(value: Any, keys: tuple[str, ...]) -> bool:
    """验证必需计数存在且均为原生零值。"""
    return bool(
        isinstance(value, dict)
        and set(keys).issubset(value)
        and all(_strict_int(value.get(key)) == 0 for key in keys)
    )


def _required_true_checks(value: Any, keys: tuple[str, ...]) -> bool:
    """验证合同要求的检查项完整，且不存在额外失败项。"""
    return bool(
        isinstance(value, dict)
        and set(keys).issubset(value)
        and all(item is True for item in value.values())
    )


def _completed_request_counts_valid(value: Any) -> bool:
    """验证 physical request 计数闭合，但不假定等于 route 证据数。"""
    if not isinstance(value, dict):
        return False
    submitted = _strict_int(value.get("submitted"))
    completed = _strict_int(value.get("completed"))
    return bool(
        submitted is not None
        and completed is not None
        and submitted >= 0
        and completed >= 0
        and submitted == completed
        and _required_zero_counts(
            value, ("rejected", "cancelled", "timed_out", "failed")
        )
    )


def _retrieval_generation_boundary(value: dict, *, cold: bool) -> bool:
    """验证一次 cache-control 的 retrieval generation 边界。"""
    before = _strict_int(value.get("retrieval_cache_generation_before"))
    after = _strict_int(value.get("retrieval_cache_generation_after"))
    return bool(
        before is not None
        and after is not None
        and (after > before if cold else after == before)
    )


def _case_embedding_epoch_check(
    run: dict, expected: tuple[str, int, str],
) -> tuple[int, int, bool]:
    """直接从逐题 retrieval evidence 重算 epoch 绑定。"""
    observed = 0
    covered_cases = 0
    matches = True
    for case in run.get("cases") or []:
        case_observed = 0
        retrieval = (case.get("stage_evidence") or {}).get(
            "retrieval_budget"
        ) or []
        for route in retrieval:
            runtime = (route or {}).get("embedding_runtime") or {}
            fields = (
                runtime.get("metrics_epoch_id"),
                runtime.get("metrics_epoch_index"),
                runtime.get("metrics_epoch_started_at"),
            )
            if not any(value is not None for value in fields):
                continue
            observed += 1
            case_observed += 1
            actual = (
                str(fields[0] or ""),
                _strict_int(fields[1]),
                str(fields[2] or ""),
            )
            matches = matches and actual == expected
        covered_cases += int(case_observed > 0)
    return observed, covered_cases, matches


def _raw_typed_core_retrieval_clean(run: dict) -> bool:
    """从逐题 degradation 重算 core retrieval 健康状态。"""
    return not any(
        str(item.get("node_id") or "") in CORE_RETRIEVAL_NODES
        and str(item.get("reason_code") or "")
        in CORE_RETRIEVAL_FAILURE_REASONS
        and item.get("affects_core_path") is True
        for case in run.get("cases") or []
        for item in case.get("degradations") or []
        if isinstance(item, dict)
    )


def _raw_embedding_protocol_checks(
    runs: list[dict],
) -> tuple[dict[str, bool], str]:
    """从 raw 制品重算三轮 embedding 准入，拒绝派生布尔伪造。"""
    warmups = [dict(run.get("embedding_warmup") or {}) for run in runs]
    resets = [dict(item.get("metrics_epoch_reset") or {}) for item in warmups]
    finals = [dict(run.get("embedding_runtime_final") or {}) for run in runs]
    health = [dict(run.get("embedding_runtime_health") or {}) for run in runs]
    schema_declarations = [
        any((
            _schema_of(warmup),
            _schema_of(reset),
            _schema_of(warmup.get("suite_metrics_epoch")),
            _schema_of(final),
            _schema_of(final.get("epoch")),
            _schema_of(item),
            _schema_of(item.get("expected_epoch")),
        ))
        for warmup, reset, final, item in zip(
            warmups, resets, finals, health,
        )
    ]
    all_legacy = bool(runs) and not any(schema_declarations)
    all_fresh = (
        len(runs) == EXPERIMENT_REQUIRED_REPEATS
        and all(
            item.get("schema") == EVALUATION_EMBEDDING_WARMUP_SCHEMA
            for item in warmups
        )
    )
    empty = {
        "embedding_protocol_declared_complete": False,
        "embedding_warmup_evidence_complete": False,
        "embedding_session_protocol": False,
        "embedding_epoch_protocol": False,
        "embedding_runtime_health_complete": False,
        "observed_cache_protocol": False,
    }
    if not all_fresh:
        mode = "legacy_observational" if all_legacy else "invalid_mixed_or_unknown"
        return empty, mode

    caches = [dict(run.get("cache_control") or {}) for run in runs]
    suite_epochs = [
        _embedding_epoch_identity(item.get("suite_metrics_epoch"))
        for item in warmups
    ]
    reset_epochs = [
        _embedding_epoch_identity(item.get("active_epoch")) for item in resets
    ]
    final_epochs = [
        _embedding_epoch_identity(item.get("epoch")) for item in finals
    ]
    health_epochs = [
        _embedding_epoch_identity(item.get("expected_epoch")) for item in health
    ]

    warmup_complete = all(
        item.get("status") == "pass"
        and _required_true_checks(
            item.get("checks"),
            (
                "dense_size", "sparse_nonzero", "model_loaded",
                "single_model_instance", "accepting_requests",
                "profile_stable", "metrics_epoch_reset",
            ),
        )
        and _strict_int(item.get("dense_size"))
        == int(app_config.embedding.dense_size)
        and (_strict_int(item.get("sparse_nonzero")) or 0) > 0
        and bool(item.get("profile_fingerprint"))
        and bool(item.get("runtime_profile_fingerprint"))
        and item.get("model_resident_for_suite") is True
        and item.get("resident_loaded_at_suite_start") is True
        and item.get("resident_accepting_at_suite_start") is True
        and item.get("retrieval_caches_invalidated_after_warmup") is True
        and item.get("warmup_excluded_from_case_timing") is True
        and item.get("suite_started_only_after_pass") is True
        and _retrieval_generation_boundary(item, cold=True)
        and reset.get("schema") == "embedding-runtime-metrics-reset-v1"
        and reset.get("status") == "pass"
        and reset.get("queue_state_preserved") is True
        and reset.get("model_state_preserved") is True
        and reset.get("only_observability_reset") is True
        and _strict_int(reset.get("active_inference_before")) == 0
        and _strict_int(reset.get("active_async_before")) == 0
        and isinstance(reset.get("active_observations"), dict)
        and _required_zero_counts(
            (reset.get("active_observations") or {}).get("requests"),
            (
                "submitted", "completed", "rejected", "cancelled",
                "timed_out", "failed",
            ),
        )
        and _strict_int(
            ((reset.get("active_observations") or {}).get("throughput") or {}).get(
                "model_invocations"
            )
        ) == 0
        for item, reset in zip(warmups, resets)
    )
    sessions = [str(item.get("session_id") or "") for item in warmups]
    session_protocol = (
        len(set(sessions)) == 1
        and bool(sessions[0])
        and [_strict_int(item.get("session_run_index")) for item in warmups]
        == [1, 2, 3]
        and [item.get("resident_reused") for item in warmups]
        == [False, True, True]
    )
    epoch_protocol = bool(
        all(suite_epochs)
        and suite_epochs == reset_epochs == final_epochs == health_epochs
        and [item[1] for item in suite_epochs if item] == [1, 2, 3]
        and len({item[0] for item in suite_epochs if item}) == 3
        and len({item[2] for item in suite_epochs if item}) == 3
    )

    runtime_complete = epoch_protocol
    if runtime_complete:
        for run, item, final, expected in zip(runs, health, finals, suite_epochs):
            assert expected is not None
            observed, covered_cases, case_matches = (
                _case_embedding_epoch_check(run, expected)
            )
            case_count = len(run.get("cases") or [])
            scheduler_counts = item.get("scheduler_failure_counts")
            final_requests = final.get("requests")
            final_checks = final.get("checks")
            runtime_complete = bool(
                item.get("schema") == "evaluation-embedding-runtime-health-v1"
                and isinstance(item.get("typed_core_retrieval_failures"), list)
                and not item["typed_core_retrieval_failures"]
                and _raw_typed_core_retrieval_clean(run)
                and _required_zero_counts(
                    scheduler_counts,
                    ("rejected", "cancelled", "timed_out", "failed"),
                )
                and all(
                    _strict_int(value) == 0
                    for value in scheduler_counts.values()
                )
                and _strict_int(item.get("mismatched_epoch_count")) == 0
                and item.get("mismatched_epochs") == []
                and _strict_int(item.get("observed_epoch_count")) == observed
                and _strict_int(run.get("n")) == case_count
                and case_count > 0
                and covered_cases == case_count
                and observed >= case_count
                and case_matches
                and final.get("schema")
                == "evaluation-embedding-runtime-epoch-v1"
                and final.get("status") == "pass"
                and _required_true_checks(
                    final_checks,
                    (
                        "epoch_stable", "queue_idle", "native_idle",
                        "model_resident", "profile_stable",
                    ),
                )
                and _completed_request_counts_valid(final_requests)
                and _required_zero_counts(
                    final.get("queue"), ("depth", "running_requests")
                )
                and final.get("profile_fingerprint")
                == warmups[expected[1] - 1].get("profile_fingerprint")
                and final.get("runtime_profile_fingerprint")
                == warmups[expected[1] - 1].get(
                    "runtime_profile_fingerprint"
                )
            )
            if not runtime_complete:
                break

    expected_states = ["cold", "warm", "warm"]
    cache_protocol = all(
        item.get("requested_state") == expected_states[index]
        and item.get("result_cache_cleared") == (index == 0)
        and isinstance(item.get("result_cache_cleared"), bool)
        and item.get("retrieval_caches_cleared") == (index == 0)
        and isinstance(item.get("retrieval_caches_cleared"), bool)
        and item.get("effective_policy")
        == ("cleared_all" if index == 0 else "retained_ttl_best_effort")
        and item.get("cache_hit_not_guaranteed") == (index > 0)
        and isinstance(item.get("cache_hit_not_guaranteed"), bool)
        and item.get("latency_attribution_allowed") is False
        and isinstance(item.get("ttl_seconds"), dict)
        and _strict_int((item.get("ttl_seconds") or {}).get("result")) is not None
        and _strict_int((item.get("ttl_seconds") or {}).get("result")) > 0
        and _strict_int((item.get("ttl_seconds") or {}).get("retrieval")) is not None
        and _strict_int((item.get("ttl_seconds") or {}).get("retrieval")) > 0
        and _retrieval_generation_boundary(item, cold=index == 0)
        for index, item in enumerate(caches)
    )
    return {
        "embedding_protocol_declared_complete": True,
        "embedding_warmup_evidence_complete": warmup_complete,
        "embedding_session_protocol": session_protocol,
        "embedding_epoch_protocol": epoch_protocol,
        "embedding_runtime_health_complete": runtime_complete,
        "observed_cache_protocol": cache_protocol,
    }, "fresh"


def experiment_report(
    runs: list[dict], *, baseline_run: dict | None = None,
    experiment_id: str | None = None,
) -> dict:
    """Aggregate a three-run experiment and apply worst-run release gates.

    Protocol completeness and release qualification are intentionally separate:
    an unchanged control version may form a valid experiment while still missing
    latency/correctness targets.  Such an experiment is useful evidence but cannot
    replace the approved baseline.
    """
    selected = [
        run for run in runs
        if (run.get("experiment") or {}).get("experiment_id")
        and (experiment_id is None or (run.get("experiment") or {}).get("experiment_id") == experiment_id)
    ]
    if not selected:
        return {
            "available": False, "experiment_id": experiment_id,
            "status": "not_started", "complete": False, "qualified": False,
            "message": "尚无该 experiment 的评测 run。",
        }
    selected.sort(key=lambda run: int((run.get("experiment") or {}).get("repeat_index") or 0))
    metadata = [dict(run.get("experiment") or {}) for run in selected]
    resolved_id = str(metadata[0].get("experiment_id") or experiment_id or "")
    workspace_ids = {str(run.get("workspace_id") or "") for run in selected}
    labels = {str(item.get("candidate_label") or "") for item in metadata}
    profiles = {
        evaluation_profile_name(item.get("evaluation_profile"))
        for item in metadata
    }
    repeat_indexes = [int(item.get("repeat_index") or 0) for item in metadata]
    cache_states = [str(item.get("cache_state") or "") for item in metadata]
    embedding_protocol_checks, embedding_protocol_mode = (
        _raw_embedding_protocol_checks(selected)
    )
    provider_manifests = [stable_hash(item.get("provider_parameters") or {}) for item in metadata]
    fingerprints = [run.get("fingerprint") or {} for run in selected]
    fingerprint_comparisons = [
        compare_fingerprints(fingerprints[0], item) for item in fingerprints[1:]
    ] if fingerprints else []
    case_totals = [int(run.get("n") or len(run.get("cases") or [])) for run in selected]
    contract_totals = [
        int(((run.get("fingerprint") or {}).get("sample_contract") or {}).get("total") or 0)
        for run in selected
    ]
    aggregates = [run.get("aggregate") or {} for run in selected]
    llm_release_checks = [
        llm_release_evidence_checks(aggregate) for aggregate in aggregates
    ]
    candidate_governance = [
        _candidate_governance_summary(run) for run in selected
    ]
    review_enabled_values = {
        bool(item.get("review_enabled")) for item in candidate_governance
    }
    review_enabled = True in review_enabled_values
    token_peaks = []
    for run, agg in zip(selected, aggregates):
        peak_case = max(
            run.get("cases") or [],
            key=lambda case: float(case.get("total_tokens") or 0),
            default={},
        )
        token_max = float(agg.get("tokens_max") or peak_case.get("total_tokens") or 0)
        token_peaks.append({
            "tokens_max": token_max,
            # 全确定性路径没有 Token 峰值，也就不存在“峰值用例”。
            "case_id": (
                str(agg.get("tokens_max_case_id") or peak_case.get("id") or "")
                if token_max > 0 else ""
            ),
        })
    route_protocol = any(
        isinstance(fp.get("model"), dict) and bool((fp.get("model") or {}).get("routes"))
        for fp in fingerprints
    )
    timing_protocol = route_protocol or any(
        agg.get("timing_protocol_version") for agg in aggregates
    )
    invocation_contract = any(
        "llm_attempt_coverage_missing_cases" in agg
        or "llm_pre_invoke_failures" in agg
        for agg in aggregates
    )

    selected_suites = {_dataset_suite_id(run) for run in selected}
    candidate_suite = next(iter(selected_suites), "")
    baseline_suite = _dataset_suite_id(baseline_run) if baseline_run else ""
    baseline_comparison_applicable = bool(
        baseline_run
        and len(selected_suites) == 1
        and baseline_suite == candidate_suite
    )
    promotion_policy = dict(PROMOTION_SUITE_POLICIES.get(candidate_suite) or {})
    result_floor_rows = [
        {
            "repeat_index": int((run.get("experiment") or {}).get("repeat_index") or 0),
            "passed": int(agg.get("result_equivalent") or 0),
            "total": total,
            "minimum": minimum_passing_results(candidate_suite, total),
            "met": suite_result_floor_met(
                candidate_suite,
                passed=int(agg.get("result_equivalent") or 0),
                total=total,
            ),
        }
        for run, agg, total in zip(selected, aggregates, case_totals)
    ]
    absolute_result_floor = bool(result_floor_rows) and all(
        row["met"] for row in result_floor_rows
    )
    baseline_cases = (
        _case_result_map(baseline_run or {})
        if baseline_comparison_applicable
        else {}
    )
    baseline_passed = sorted(case_id for case_id, passed in baseline_cases.items() if passed)
    run_case_maps = [_case_result_map(run) for run in selected]
    first_case_maps = [_case_first_result_map(run) for run in selected]
    regressed_case_ids = sorted({
        case_id for case_id in baseline_passed
        if any(not case_map.get(case_id, False) for case_map in run_case_maps)
    })
    baseline_result_count = int(
        ((baseline_run or {}).get("aggregate") or {}).get("result_equivalent")
        or 0
    )
    baseline_result_non_regression = (
        not baseline_comparison_applicable
        or all(
            int(agg.get("result_equivalent") or 0) >= baseline_result_count
            for agg in aggregates
        )
    )
    stability = []
    all_case_ids = sorted({case_id for case_map in run_case_maps for case_id in case_map})
    case_catalog: dict[str, dict] = {}
    for run in selected:
        for case in run.get("cases") or []:
            if case.get("id"):
                case_catalog.setdefault(str(case["id"]), case)
    for case_id in all_case_ids:
        passed = sum(bool(case_map.get(case_id)) for case_map in run_case_maps)
        first_passed = sum(
            bool(case_map.get(case_id)) for case_map in first_case_maps
        )
        stability.append({
            "case_id": case_id, "passed_repeats": passed,
            "first_passed_repeats": first_passed,
            "total_repeats": len(selected),
            "semantic_family": _case_semantic_family(case_catalog.get(case_id, {})),
            "first_status": (
                "stable_pass" if first_passed == len(selected)
                else "stable_fail" if first_passed == 0 else "flaky"
            ),
            "status": "stable_pass" if passed == len(selected) else "stable_fail" if passed == 0 else "flaky",
        })
    stability_summary = {
        "total": len(stability),
        "stable_pass": sum(item["status"] == "stable_pass" for item in stability),
        "stable_fail": sum(item["status"] == "stable_fail" for item in stability),
        "variable": sum(item["status"] == "flaky" for item in stability),
        "first_stable_pass": sum(
            item["first_status"] == "stable_pass" for item in stability
        ),
        "first_stable_fail": sum(
            item["first_status"] == "stable_fail" for item in stability
        ),
        "first_variable": sum(
            item["first_status"] == "flaky" for item in stability
        ),
    }
    semantic_family_stability = []
    for family in sorted({item["semantic_family"] for item in stability}):
        family_rows = [item for item in stability if item["semantic_family"] == family]
        family_ids = {item["case_id"] for item in family_rows}
        repeat_rows = []
        for run, case_map in zip(selected, run_case_maps):
            passed = sum(bool(case_map.get(case_id)) for case_id in family_ids)
            total = len(family_ids)
            repeat_rows.append({
                "repeat_index": int((run.get("experiment") or {}).get("repeat_index") or 0),
                "passed": passed,
                "total": total,
                "pass_rate": round(passed / total, 4) if total else 0.0,
            })
        semantic_family_stability.append({
            "semantic_family": family,
            "total": len(family_rows),
            "stable_pass": sum(item["status"] == "stable_pass" for item in family_rows),
            "stable_fail": sum(item["status"] == "stable_fail" for item in family_rows),
            "variable": sum(item["status"] == "flaky" for item in family_rows),
            "worst_pass_rate": min(
                (item["pass_rate"] for item in repeat_rows),
                default=0.0,
            ),
            "repeats": repeat_rows,
        })

    expected_indexes = list(range(1, EXPERIMENT_REQUIRED_REPEATS + 1))
    full_dataset = bool(selected) and all(
        total > 0 and total == contract_total
        for total, contract_total in zip(case_totals, contract_totals)
    )
    per_run_eligible = [
        baseline_checklist(run, contract_total)["eligible"]
        for run, contract_total in zip(selected, contract_totals)
    ]
    checks = {
        "metadata_complete": all(_experiment_metadata_complete(item) for item in metadata),
        "three_repeats": len(selected) == EXPERIMENT_REQUIRED_REPEATS,
        "repeat_indexes": repeat_indexes == expected_indexes,
        "cache_protocol": cache_states == ["cold", "warm", "warm"],
        **embedding_protocol_checks,
        "same_workspace": len(workspace_ids) == 1 and "" not in workspace_ids,
        "same_candidate": len(labels) == 1 and "" not in labels and len(set(provider_manifests)) == 1,
        "same_evaluation_profile": len(profiles) == 1 and "unknown" not in profiles,
        "dataset_suite_supported": (
            len(selected_suites) == 1
            and candidate_suite in _SUPPORTED_PROMOTION_SUITES
        ),
        "suite_case_count": bool(promotion_policy) and all(
            total == int(promotion_policy.get("case_count") or 0)
            for total in case_totals
        ),
        "full_dataset": full_dataset,
        "valid_runs": all(run.get("valid") is True for run in selected),
        "infrastructure_clean": all(int(agg.get("infrastructure_errors") or 0) == 0 for agg in aggregates),
        "fingerprint_v2": all(fp.get("fingerprint_schema") == "evaluation-v2" for fp in fingerprints),
        "fingerprints_complete": all(
            fp.get("complete") is True and all(fp.get(field) for field in REQUIRED_FINGERPRINT_FIELDS)
            for fp in fingerprints
        ),
        "fingerprints_comparable": bool(fingerprints) and all(
            item.get("comparable") for item in fingerprint_comparisons
        ),
        "result_denominator_complete": all(
            int(agg.get("result_evaluated") or 0) == total
            for agg, total in zip(aggregates, case_totals)
        ),
        "first_final_evidence_complete": all(
            agg.get("first_sql_evidence_complete") is True
            and int(agg.get("result_evaluated") or 0) == total
            for agg, total in zip(aggregates, case_totals)
        ),
        "run_checklists_eligible": all(per_run_eligible),
        # A candidate suite must never be compared case-by-case with another
        # suite merely because both belong to the same workspace.  Cross-suite
        # protection is supplied by absolute suite floors plus a separate run
        # of the approved suite.
        "baseline_non_regression": (
            not baseline_comparison_applicable or not regressed_case_ids
        ),
        "baseline_result_non_regression": baseline_result_non_regression,
        "table_exact_non_regression": all(
            int(agg.get("tables_correct") or 0) == total
            for agg, total in zip(aggregates, case_totals)
        ),
        "table_scope_safety": all(
            "table_scope_safe" in agg
            and int(agg.get("table_scope_safe") or 0) == total
            for agg, total in zip(aggregates, case_totals)
        ),
        "table_scope_evidence_complete": all(
            int(agg.get("table_scope_evidence_complete") or 0) == total
            for agg, total in zip(aggregates, case_totals)
        ),
        "security_guard_integrity": all(
            int(agg.get("security_guard_bypass_count") or 0) == 0
            for agg in aggregates
        ),
        "syntax_floor_gate": all(
            minimum_passing_results(candidate_suite, total) is None
            or int(agg.get("syntax_valid") or 0)
            >= int(minimum_passing_results(candidate_suite, total) or 0)
            for agg, total in zip(aggregates, case_totals)
        ),
        "execution_floor_gate": all(
            minimum_passing_results(candidate_suite, total) is None
            or int(agg.get("exec_success") or 0)
            >= int(minimum_passing_results(candidate_suite, total) or 0)
            for agg, total in zip(aggregates, case_totals)
        ),
        "absolute_result_floor": absolute_result_floor,
        # Kept as the aggregate correctness key consumed by run-quality.  It
        # cannot be satisfied by an empty floor map: every repeat must meet the
        # absolute suite floor and, when applicable, the same-suite baseline.
        "result_floor_non_regression": (
            absolute_result_floor and baseline_result_non_regression
        ),
        "ttsql_gate": all(
            0 < float(agg.get("tt_sql_p95") or 0) <= EXPERIMENT_GATE_LIMITS["tt_sql_p95_ms"]
            for agg in aggregates
        ),
        "ttresult_gate": all(
            0 < float(agg.get("tt_result_p95") or 0) <= EXPERIMENT_GATE_LIMITS["tt_result_p95_ms"]
            for agg in aggregates
        ),
        "ttcomplete_gate": all(
            0 < float(agg.get("tt_complete_p95") or agg.get("latency_p95") or 0)
            <= EXPERIMENT_GATE_LIMITS["tt_complete_p95_ms"]
            for agg in aggregates
        ),
        "timing_protocol_complete": (not timing_protocol) or all(
            agg.get("timing_protocol_version") == "timing-v2"
            and not agg.get("timing_invariant_violations")
            for agg in aggregates
        ),
        # TODO-48 checks are activated by the TODO-46 route manifest. Historical
        # artifacts remain readable but cannot pretend to contain new diagnostics.
        "routing_evidence_complete": (not route_protocol) or all(
            isinstance(fp.get("model"), dict) and bool((fp.get("model") or {}).get("routes"))
            for fp in fingerprints
        ),
        "prompt_compression_evidence": (not route_protocol) or all(
            int(agg.get("prompt_payloads") or 0) > 0
            or int(agg.get("llm_attempts") or 0) == 0
            for agg in aggregates
        ),
        "llm_attempt_accounting_complete": all(
            agg.get("llm_attempt_schema") == LLM_ATTEMPT_SCHEMA
            and bool(agg.get("llm_attempt_accounting_complete"))
            for agg in aggregates
        ),
        "llm_attempt_coverage_complete": (not invocation_contract) or all(
            int(agg.get("llm_attempt_coverage_missing_cases") or 0) == 0
            for agg in aggregates
        ),
        "semantic_coverage_evidence_complete": all(
            int(agg.get("semantic_coverage_cases") or 0) == total
            and int(agg.get("semantic_unsafe_compile_cases") or 0) == 0
            for agg, total in zip(aggregates, case_totals)
        ),
        "request_deadline_evidence_complete": all(
            agg.get("request_deadline_contract_complete") is True
            and int(agg.get("request_deadline_exhausted_cases") or 0) == 0
            for agg in aggregates
        ),
        "llm_outcome_taxonomy_complete": all(
            sum(
                int(value or 0)
                for value in (agg.get("llm_outcome_classes") or {}).values()
            ) == int(agg.get("llm_attempts") or 0)
            for agg in aggregates
        ),
        "llm_recovery_bounded": all(
            0 <= int(agg.get("llm_recoveries") or 0)
            <= int(agg.get("llm_attempts") or 0)
            for agg in aggregates
        ),
        "degradation_evidence_complete": all(
            agg.get("degradation_evidence_complete") is True for agg in aggregates
        ),
        "failed_llm_calls_zero": all(
            item["failed_llm_calls_zero"] for item in llm_release_checks
        ),
        "llm_attempts_timed_out_zero": all(
            item["llm_attempts_timed_out_zero"]
            for item in llm_release_checks
        ),
        "optional_postprocess_clean": (
            "enhanced-analysis" not in profiles
            or all(int(agg.get("failed_optional_llm_calls") or 0) == 0 for agg in aggregates)
        ),
        "structured_output_clean": all(int(agg.get("structured_output_fallbacks") or 0) == 0 for agg in aggregates),
        # TODO-45 adaptive expansion keeps its original bounded trigger and
        # safety gates.  Default semantic review is accounted independently.
        "candidate_trigger_gate": all(
            float(item["adaptive"].get("rate") or 0) <= .20
            for item in candidate_governance
        ),
        "candidate_safety_gate": all(
            int(item["adaptive"].get("unsafe_selected") or 0) == 0
            and int(
                item["adaptive"].get("high_confidence_extra_calls") or 0
            ) == 0
            for item in candidate_governance
        ),
        "semantic_review_policy_consistent": len(review_enabled_values) == 1,
        "semantic_review_policy_fingerprinted": (
            not review_enabled
            or all(
                item.get("review_contract_explicit") is True
                and item.get("review_policy_version")
                == SEMANTIC_REVIEW_POLICY_VERSION
                for item in candidate_governance
            )
        ),
        "semantic_review_evidence_complete": (
            not review_enabled
            or all(
                item["review"].get("evidence_complete") is True
                for item in candidate_governance
            )
        ),
        "semantic_review_at_most_one_attempt": (
            not review_enabled
            or all(
                item["review"].get("at_most_one_attempt") is True
                for item in candidate_governance
            )
        ),
        "semantic_review_safety_gate": (
            not review_enabled
            or all(
                int(item["review"].get("unsafe_selected") or 0) == 0
                for item in candidate_governance
            )
        ),
        "semantic_review_failed_core_zero": (
            not review_enabled
            or all(
                int(item["review"].get("failed_core") or 0) == 0
                for item in candidate_governance
            )
        ),
        "semantic_review_harmful_modification_zero": (
            not review_enabled
            or all(
                int(item["review"].get("harmful_modification") or 0) == 0
                for item in candidate_governance
            )
        ),
        "semantic_review_first_correct_final_incorrect_zero": (
            not review_enabled
            or all(
                int(
                    item["review"].get("first_correct_final_incorrect") or 0
                ) == 0
                for item in candidate_governance
            )
        ),
        "semantic_review_same_sql_execution_loss_zero": (
            not review_enabled
            or all(
                int(item["review"].get("same_sql_execution_loss") or 0) == 0
                for item in candidate_governance
            )
        ),
        "semantic_review_selected_b_no_sql_delta_zero": (
            not review_enabled
            or all(
                int(item["review"].get("selected_b_no_sql_delta") or 0) == 0
                for item in candidate_governance
            )
        ),
        "semantic_review_result_attribution_complete": (
            not review_enabled
            or all(
                item["review"].get("result_attribution_complete") is True
                for item in candidate_governance
            )
        ),
        "semantic_review_coverage_complete": (
            not review_enabled
            or all(
                item["review"].get("coverage_complete") is True
                for item in candidate_governance
            )
        ),
        "semantic_review_budget_skip_zero": (
            not review_enabled
            or all(
                int(item["review"].get("budget_skipped") or 0) == 0
                for item in candidate_governance
            )
        ),
        "token_non_regression": (
            not baseline_run
            or not float((baseline_run.get("aggregate") or {}).get("tokens_mean") or 0)
            or all(
                float(agg.get("tokens_mean") or 0)
                <= float((baseline_run.get("aggregate") or {}).get("tokens_mean") or 0) * 1.10
                for agg in aggregates
            )
        ),
        "token_max_gate": (not route_protocol) or all(
            (
                int(agg.get("llm_attempts") or 0) == 0
                and float(peak.get("tokens_max") or 0) == 0
            )
            or (
                int(agg.get("llm_attempts") or 0) > 0
                and 0 < float(peak.get("tokens_max") or 0)
                <= EXPERIMENT_GATE_LIMITS["tokens_max"]
            )
            for agg, peak in zip(aggregates, token_peaks)
        ),
    }
    completeness_keys = (
        "metadata_complete", "three_repeats", "repeat_indexes", "cache_protocol",
        "embedding_protocol_declared_complete",
        "embedding_warmup_evidence_complete", "embedding_session_protocol",
        "embedding_epoch_protocol", "embedding_runtime_health_complete",
        "observed_cache_protocol",
        "same_workspace", "same_candidate", "same_evaluation_profile", "full_dataset", "valid_runs",
        "infrastructure_clean", "fingerprint_v2", "fingerprints_complete",
        "fingerprints_comparable", "result_denominator_complete",
        "first_final_evidence_complete", "run_checklists_eligible",
        "table_scope_evidence_complete",
        "routing_evidence_complete", "prompt_compression_evidence",
        "timing_protocol_complete",
        "llm_attempt_accounting_complete", "llm_attempt_coverage_complete",
        "semantic_coverage_evidence_complete",
        "request_deadline_evidence_complete",
        "llm_outcome_taxonomy_complete", "llm_recovery_bounded",
        "degradation_evidence_complete", "failed_llm_calls_zero",
        "llm_attempts_timed_out_zero",
        "semantic_review_policy_consistent",
        "semantic_review_policy_fingerprinted",
        "semantic_review_evidence_complete",
        "semantic_review_result_attribution_complete",
    )
    complete = all(checks[key] for key in completeness_keys)
    # Difficulty baselines are measurements of the current system, including
    # stable failures.  ``recordable`` therefore proves protocol/evidence
    # integrity without pretending that release-quality gates passed.  The
    # stricter ``qualified`` flag remains the release/non-regression decision.
    recordable = bool(complete and checks["dataset_suite_supported"])
    blocking_checks = {
        key: passed
        for key, passed in checks.items()
        if key not in EXPERIMENT_OBSERVATIONAL_CHECKS
    }
    observational_checks = {
        key: passed
        for key, passed in checks.items()
        if key in EXPERIMENT_OBSERVATIONAL_CHECKS
    }
    qualified = complete and all(blocking_checks.values())
    run_rows = []
    for run, item, agg, token_peak, candidate_summary in zip(
        selected, metadata, aggregates, token_peaks, candidate_governance,
    ):
        total = int(run.get("n") or len(run.get("cases") or []))
        run_rows.append({
            "run_id": run.get("run_id"), "repeat_index": item.get("repeat_index"),
            "cache_state": item.get("cache_state"),
            "evaluation_profile": evaluation_profile_name(item.get("evaluation_profile")),
            "total": total,
            "first_result_equivalent": int(
                agg.get("first_result_equivalent") or 0
            ),
            "first_exec_success": int(agg.get("first_exec_success") or 0),
            "correction_triggered": int(agg.get("correction_triggered") or 0),
            "correction_fixed_result": int(
                agg.get("correction_fixed_result") or 0
            ),
            "correction_harmful_modification": int(
                agg.get("correction_harmful_modification") or 0
            ),
            "tables_correct": int(agg.get("tables_correct") or 0),
            "table_scope_safe": int(
                agg.get("table_scope_safe")
                if "table_scope_safe" in agg
                else 0
            ),
            "table_scope_evidence_complete": int(
                agg.get("table_scope_evidence_complete") or 0
            ),
            "result_equivalent": int(agg.get("result_equivalent") or 0),
            "tt_sql_p95": float(agg.get("tt_sql_p95") or 0),
            "tt_result_p95": float(agg.get("tt_result_p95") or 0),
            "tt_complete_p95": float(agg.get("tt_complete_p95") or agg.get("latency_p95") or 0),
            "llm_calls_p95": float(agg.get("llm_calls_p95") or 0),
            "tokens_p95": float(agg.get("tokens_p95") or 0),
            "tokens_mean": float(agg.get("tokens_mean") or 0),
            "tokens_max": float(token_peak.get("tokens_max") or 0),
            "tokens_max_case_id": str(token_peak.get("case_id") or ""),
            "failed_llm_calls": int(agg.get("failed_llm_calls") or 0),
            "failed_core_llm_calls": int(agg.get("failed_core_llm_calls") or 0),
            "failed_optional_llm_calls": int(agg.get("failed_optional_llm_calls") or 0),
            "llm_attempts": int(agg.get("llm_attempts") or 0),
            "llm_callback_missing": int(agg.get("llm_callback_missing") or 0),
            "llm_pre_invoke_failures": int(agg.get("llm_pre_invoke_failures") or 0),
            "llm_recoveries": int(agg.get("llm_recoveries") or 0),
            "semantic_coverage_mean": float(agg.get("semantic_coverage_mean") or 0),
            "semantic_unresolved_cases": int(agg.get("semantic_unresolved_cases") or 0),
            "semantic_unsafe_compile_cases": int(
                agg.get("semantic_unsafe_compile_cases") or 0
            ),
            "request_deadline_exhausted_cases": int(
                agg.get("request_deadline_exhausted_cases") or 0
            ),
            "llm_attempt_coverage_missing_cases": int(
                agg.get("llm_attempt_coverage_missing_cases") or 0
            ),
            "prompt_compression_rate": float(agg.get("prompt_compression_rate") or 0),
            "adaptive_candidate_rate": float(
                candidate_summary["adaptive"].get("rate") or 0
            ),
            "semantic_review_enabled": bool(
                candidate_summary.get("review_enabled")
            ),
            "semantic_review_eligible": int(
                candidate_summary["review"].get("eligible") or 0
            ),
            "semantic_review_failed_core": int(
                candidate_summary["review"].get("failed_core") or 0
            ),
            "semantic_review_unsafe_selected": int(
                candidate_summary["review"].get("unsafe_selected") or 0
            ),
            "semantic_review_harmful_modification": int(
                candidate_summary["review"].get("harmful_modification") or 0
            ),
            "semantic_review_first_correct_final_incorrect": int(
                candidate_summary["review"].get(
                    "first_correct_final_incorrect"
                ) or 0
            ),
            "semantic_review_same_sql_execution_loss": int(
                candidate_summary["review"].get("same_sql_execution_loss") or 0
            ),
            "semantic_review_selected_b_no_sql_delta": int(
                candidate_summary["review"].get("selected_b_no_sql_delta") or 0
            ),
            "semantic_review_budget_skipped": int(
                candidate_summary["review"].get("budget_skipped") or 0
            ),
            "candidate_selected_b": int(agg.get("candidate_selected_b") or 0),
            "candidate_valid_b": int(agg.get("candidate_valid_b") or 0),
            "candidate_failed": int(agg.get("candidate_failed") or 0),
            "candidate_wait_total_ms": float(agg.get("candidate_wait_total_ms") or 0),
            "infrastructure_errors": int(agg.get("infrastructure_errors") or 0),
            "security_guard_bypass_count": int(
                agg.get("security_guard_bypass_count") or 0
            ),
        })
    worst = {
        "first_result_equivalent": min(
            (row["first_result_equivalent"] for row in run_rows), default=0
        ),
        "first_exec_success": min(
            (row["first_exec_success"] for row in run_rows), default=0
        ),
        "tables_correct": min((row["tables_correct"] for row in run_rows), default=0),
        "table_scope_safe": min(
            (row["table_scope_safe"] for row in run_rows), default=0
        ),
        "table_scope_evidence_complete": min(
            (row["table_scope_evidence_complete"] for row in run_rows),
            default=0,
        ),
        "result_equivalent": min((row["result_equivalent"] for row in run_rows), default=0),
        "tt_sql_p95": max((row["tt_sql_p95"] for row in run_rows), default=0),
        "tt_result_p95": max((row["tt_result_p95"] for row in run_rows), default=0),
        "tt_complete_p95": max((row["tt_complete_p95"] for row in run_rows), default=0),
        "llm_calls_p95": max((row["llm_calls_p95"] for row in run_rows), default=0),
        "tokens_p95": max((row["tokens_p95"] for row in run_rows), default=0),
        "tokens_mean": max((row["tokens_mean"] for row in run_rows), default=0),
        "tokens_max": max((row["tokens_max"] for row in run_rows), default=0),
        "tokens_max_case_ids": sorted({
            row["tokens_max_case_id"] for row in run_rows
            if row["tokens_max"] == max((item["tokens_max"] for item in run_rows), default=0)
            and row["tokens_max_case_id"]
        }),
        "failed_llm_calls": max((row["failed_llm_calls"] for row in run_rows), default=0),
        "failed_core_llm_calls": max((row["failed_core_llm_calls"] for row in run_rows), default=0),
        "failed_optional_llm_calls": max((row["failed_optional_llm_calls"] for row in run_rows), default=0),
        "llm_callback_missing": max((row["llm_callback_missing"] for row in run_rows), default=0),
        "llm_pre_invoke_failures": max((row["llm_pre_invoke_failures"] for row in run_rows), default=0),
        "llm_attempt_coverage_missing_cases": max(
            (row["llm_attempt_coverage_missing_cases"] for row in run_rows),
            default=0,
        ),
        "candidate_valid_b": min((row["candidate_valid_b"] for row in run_rows), default=0),
        "candidate_failed": max((row["candidate_failed"] for row in run_rows), default=0),
        "candidate_wait_total_ms": max((row["candidate_wait_total_ms"] for row in run_rows), default=0),
        "prompt_compression_rate": min((row["prompt_compression_rate"] for row in run_rows), default=0),
        "adaptive_candidate_rate": max((row["adaptive_candidate_rate"] for row in run_rows), default=0),
        "semantic_review_eligible": min(
            (row["semantic_review_eligible"] for row in run_rows), default=0
        ),
        "semantic_review_failed_core": max(
            (row["semantic_review_failed_core"] for row in run_rows), default=0
        ),
        "semantic_review_unsafe_selected": max(
            (row["semantic_review_unsafe_selected"] for row in run_rows),
            default=0,
        ),
        "semantic_review_harmful_modification": max(
            (
                row["semantic_review_harmful_modification"]
                for row in run_rows
            ),
            default=0,
        ),
        "semantic_review_first_correct_final_incorrect": max(
            (
                row["semantic_review_first_correct_final_incorrect"]
                for row in run_rows
            ),
            default=0,
        ),
        "semantic_review_same_sql_execution_loss": max(
            (
                row["semantic_review_same_sql_execution_loss"]
                for row in run_rows
            ),
            default=0,
        ),
        "semantic_review_selected_b_no_sql_delta": max(
            (
                row["semantic_review_selected_b_no_sql_delta"]
                for row in run_rows
            ),
            default=0,
        ),
        "semantic_review_budget_skipped": max(
            (row["semantic_review_budget_skipped"] for row in run_rows),
            default=0,
        ),
        "candidate_selected_b": max((row["candidate_selected_b"] for row in run_rows), default=0),
        "security_guard_bypass_count": max(
            (row["security_guard_bypass_count"] for row in run_rows), default=0
        ),
    }
    return {
        "available": True,
        "schema": EXPERIMENT_SCHEMA,
        "experiment_id": resolved_id,
        "candidate_label": next(iter(labels), ""),
        "evaluation_profile": next(iter(profiles), "unknown"),
        "workspace_id": next(iter(workspace_ids), ""),
        "status": (
            "qualified" if qualified else "recordable" if recordable
            else "incomplete"
        ),
        "complete": complete,
        "recordable": recordable,
        "qualified": qualified,
        "baseline_mode": (
            "non_regression" if baseline_comparison_applicable else "bootstrap"
        ),
        "required_repeats": EXPERIMENT_REQUIRED_REPEATS,
        "repeat_count": len(selected),
        "embedding_protocol_mode": embedding_protocol_mode,
        "checks": checks,
        "blocking_checks": blocking_checks,
        "observational_checks": observational_checks,
        "check_accounting": {
            "blocking": blocking_checks,
            "observational": observational_checks,
        },
        "failed_checks": [
            key for key, passed in blocking_checks.items() if not passed
        ],
        "failed_observations": [
            key for key, passed in observational_checks.items() if not passed
        ],
        "fingerprint_differences": sorted({
            difference for item in fingerprint_comparisons for difference in item.get("differences") or []
        }),
        "baseline": {
            "run_id": (baseline_run or {}).get("run_id"),
            "suite": baseline_suite,
            "candidate_suite": candidate_suite,
            "comparison_applicable": baseline_comparison_applicable,
            "fingerprint_schema": ((baseline_run or {}).get("fingerprint") or {}).get("fingerprint_schema") or "unknown",
            "business_comparison_only": bool(baseline_run) and not compare_fingerprints(
                (baseline_run or {}).get("fingerprint"), fingerprints[0] if fingerprints else None,
            ).get("comparable"),
            "passed_case_count": len(baseline_passed),
        },
        "promotion_policy": {
            "dataset_suite": candidate_suite,
            **promotion_policy,
            "per_run": result_floor_rows,
        },
        "regressed_case_ids": regressed_case_ids,
        "runs": run_rows,
        "worst": worst,
        "stability": stability,
        "stability_summary": stability_summary,
        "semantic_family_stability": semantic_family_stability,
        "gate_limits": dict(EXPERIMENT_GATE_LIMITS),
        "engineering_targets": {
            "tt_sql_p95_ms": 13_500.0,
            "ttsql_headroom_met": max((row["tt_sql_p95"] for row in run_rows), default=0) <= 13_500.0,
        },
    }


def resolve_baseline_artifact(
    *, explicit: str | Path | None = None, workspace_id: str | None = None,
    approved_entry: dict | None = None, runs_dir: Path | None = None,
    expected_dataset_suite: str | None = None,
) -> dict:
    """Resolve an explicit artifact or the workspace's MySQL-approved run."""
    root = runs_dir or RUNS_DIR

    def load(path: Path, source: str, status: str) -> dict:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload_workspace = str(payload.get("workspace_id") or "")
        if workspace_id and payload_workspace and payload_workspace != workspace_id:
            raise ValueError("baseline artifact 不属于指定工作空间")
        payload_suite = str(payload.get("dataset_suite") or "")
        if expected_dataset_suite and payload_suite != expected_dataset_suite:
            raise ValueError(
                "baseline artifact 套件不匹配："
                f"期望 {expected_dataset_suite}，实际 {payload_suite or 'missing'}"
            )
        fingerprint = payload.get("fingerprint") or {}
        return {
            "available": True, "source": source, "status": status,
            "schema": fingerprint.get("fingerprint_schema") or "unknown",
            "path": str(path), "run_id": payload.get("run_id"), "payload": payload,
        }

    if explicit:
        candidate = Path(explicit)
        if not candidate.exists():
            run_id = str(explicit).removeprefix("run_").removesuffix(".json")
            candidate = root / f"run_{run_id}.json"
        if not candidate.exists():
            raise ValueError(f"显式 baseline 不存在：{explicit}")
        return load(candidate.resolve(), "explicit", "explicit")
    if approved_entry and approved_entry.get("artifact"):
        entry_suite = str(approved_entry.get("dataset_suite") or "")
        if expected_dataset_suite and entry_suite != expected_dataset_suite:
            raise ValueError(
                "baseline registry 套件不匹配："
                f"期望 {expected_dataset_suite}，实际 {entry_suite or 'missing'}"
            )
        candidate = root / str(approved_entry["artifact"])
        if candidate.exists():
            return load(candidate, "meta_mysql", str(approved_entry.get("status") or "approved"))
    return {
        "available": False, "source": "none", "status": "missing",
        "schema": "unknown", "path": None, "run_id": None, "payload": None,
    }


def is_explicit_nonrelease(run: dict) -> bool:
    """Whether an artifact explicitly declares that it cannot be a baseline."""

    artifact_scope = str(run.get("artifact_scope") or "")
    return bool(
        run.get("baseline_eligible") is False
        or (artifact_scope and artifact_scope != "release_candidate")
    )


def pipeline_evidence_checks(
    fingerprint: dict | None,
    aggregate: dict | None,
    cases: list[dict] | None,
) -> dict[str, bool]:
    """Bind observed pipeline execution to the immutable selected profiles."""

    fingerprint, aggregate, cases = fingerprint or {}, aggregate or {}, cases or []
    profiles = fingerprint.get("pipeline_profiles") or {}
    query_profile = str(
        ((profiles.get("query_signals") or {}).get("profile")) or ""
    )
    query_route_policy = str(
        ((profiles.get("query_signals") or {}).get("route_policy")) or ""
    )
    query_dictionary_enabled = (
        (profiles.get("query_signals") or {}).get(
            "workspace_dictionary_enabled"
        )
        is True
    )
    planning_profile = str(
        ((profiles.get("planning_route") or {}).get("active_profile")) or ""
    )
    planning_fallback_profile = str(
        ((profiles.get("planning_route") or {}).get("fallback_profile")) or ""
    )
    repair_profile = str(
        ((profiles.get("semantic_repair") or {}).get("active_profile")) or ""
    )
    planning_repair_profile = str(
        ((profiles.get("planning_route") or {}).get("repair_profile")) or ""
    )
    total = len(cases)
    query_signals = aggregate.get("query_signals") or {}
    planning = aggregate.get("planning_route") or {}
    repair = aggregate.get("semantic_repair") or {}
    query_cases = [
        (case.get("stage_evidence") or {}).get("query_signals") or {}
        for case in cases
    ]
    planning_cases = [
        (case.get("stage_evidence") or {}).get("planning_route") or {}
        for case in cases
    ]
    repair_cases = [
        (case.get("stage_evidence") or {}).get("semantic_repair") or {}
        for case in cases
    ]

    expected_query_route_counts = {
        route: 1 for route in sorted(QUERY_SIGNAL_REQUIRED_ROUTES)
    }

    def query_case_valid(item: dict[str, Any]) -> bool:
        dictionary_version = str(item.get("dictionary_version") or "")
        routes = [
            row for row in (item.get("routes") or [])
            if isinstance(row, dict)
        ]
        route_counts = Counter(
            str(row.get("route") or "unknown") for row in routes
        )
        dictionary_ready = bool(
            not (
                query_profile == "domain_signals_v1"
                and query_dictionary_enabled
            )
            or dictionary_version not in {"", "none"}
        )

        def non_negative_int(value: Any) -> bool:
            return bool(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
            )

        extractor_signals = item.get("extractor_signals")
        if not isinstance(extractor_signals, list):
            return False
        extractor_identities: set[tuple[str, str, str]] = set()
        for signal in extractor_signals:
            if not isinstance(signal, dict):
                return False
            signal_id = str(signal.get("signal_id") or "")
            kind = str(signal.get("kind") or "")
            fingerprint = str(signal.get("fingerprint") or "")
            identity = (signal_id, kind, fingerprint)
            if (
                len(fingerprint) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in fingerprint
                )
                or signal_id != f"qs-{fingerprint[:12]}"
                or kind not in {"standalone_query", *SIGNAL_KINDS}
                or not non_negative_int(signal.get("rank"))
                or "text" in signal
                or identity in extractor_identities
            ):
                return False
            extractor_identities.add(identity)
        if (
            not extractor_signals
            or item.get("extractor_signal_identity_fingerprint")
            != stable_hash(extractor_signals)
        ):
            return False
        extractor_kind_counts = Counter(
            str(signal.get("kind") or "unknown")
            for signal in extractor_signals
        )
        if item.get("extractor_kind_counts") != dict(
            sorted(extractor_kind_counts.items())
        ):
            return False
        domain_identities = {
            identity
            for identity in extractor_identities
            if identity[1] == "domain_terms"
        }

        def route_execution_valid(row: dict[str, Any]) -> bool:
            route = str(row.get("route") or "")
            consumer_rows: dict[str, list[dict[str, Any]]] = {}
            for consumer in ("local", "remote", "rerank"):
                signals = row.get(f"actual_{consumer}_signals")
                if not isinstance(signals, list) or not signals:
                    return False
                signal_keys: set[tuple[str, str]] = set()
                for signal in signals:
                    if not isinstance(signal, dict):
                        return False
                    signal_id = str(signal.get("signal_id") or "")
                    kind = str(signal.get("kind") or "")
                    fingerprint = str(signal.get("fingerprint") or "")
                    identity = (signal_id, kind, fingerprint)
                    key = (kind, fingerprint)
                    if (
                        len(fingerprint) != 64
                        or any(
                            character not in "0123456789abcdef"
                            for character in fingerprint
                        )
                        or signal_id != f"qs-{fingerprint[:12]}"
                        or kind not in QUERY_SIGNAL_CONSUMER_KINDS
                        or signal.get("consumer") != consumer
                        or not non_negative_int(signal.get("rank"))
                        or "text" in signal
                        or key in signal_keys
                        or (
                            kind in {"standalone_query", *SIGNAL_KINDS}
                            and identity not in extractor_identities
                        )
                    ):
                        return False
                    signal_keys.add(key)
                if row.get(f"{consumer}_signal_fingerprint") != stable_hash(
                    signals
                ):
                    return False
                consumer_rows[consumer] = signals
            consumer_kind_counts = {
                consumer: dict(sorted(Counter(
                    str(signal.get("kind") or "unknown")
                    for signal in signals
                ).items()))
                for consumer, signals in consumer_rows.items()
            }
            if row.get("consumer_kind_counts") != consumer_kind_counts:
                return False
            declared_counts = {
                "local": row.get("local_signal_count"),
                "remote": row.get("remote_query_count"),
                "rerank": row.get("rerank_signal_count"),
            }
            actual_counts = {
                consumer: row.get(f"actual_{consumer}_signal_count")
                for consumer in ("local", "remote", "rerank")
            }
            for consumer in ("local", "remote", "rerank"):
                if (
                    not non_negative_int(declared_counts[consumer])
                    or not non_negative_int(actual_counts[consumer])
                    or declared_counts[consumer] <= 0
                    or actual_counts[consumer] <= 0
                    or declared_counts[consumer]
                    != actual_counts[consumer]
                    or actual_counts[consumer]
                    != len(consumer_rows[consumer])
                ):
                    return False
            if query_profile == "domain_signals_v1":
                for consumer in ("local", "rerank"):
                    observed = {
                        (
                            str(signal.get("signal_id") or ""),
                            str(signal.get("kind") or ""),
                            str(signal.get("fingerprint") or ""),
                        )
                        for signal in consumer_rows[consumer]
                    }
                    observed_domain = {
                        identity for identity in observed
                        if identity[1] == "domain_terms"
                    }
                    if observed_domain != domain_identities:
                        return False
            expected_plan_fingerprint = stable_signal_hash({
                "contract": QUERY_SIGNAL_CONTRACT_VERSION,
                "route_policy_version": ROUTE_POLICY_VERSION,
                "route_policy": query_route_policy,
                "route": route,
                "profile_fingerprint": item.get("profile_fingerprint"),
                "signal_fingerprint": item.get("signal_fingerprint"),
                "query_encode_mode": row.get("query_encode_mode"),
                "local": consumer_rows["local"],
                "remote": consumer_rows["remote"],
                "rerank": consumer_rows["rerank"],
            })
            return bool(
                row.get("query_encode_mode")
                == QUERY_SIGNAL_ROUTE_ENCODE_MODES.get(route)
                and row.get("consumer_plan_fingerprint")
                == expected_plan_fingerprint
            )

        return bool(
            item.get("schema_version")
            == "evaluation-query-signal-evidence-v2"
            and item.get("contract_version") == QUERY_SIGNAL_CONTRACT_VERSION
            and item.get("configured_profile") == query_profile
            and item.get("active_profile") == query_profile
            and item.get("configured_route_policy") == query_route_policy
            and bool(str(item.get("profile_fingerprint") or ""))
            and bool(str(item.get("signal_fingerprint") or ""))
            and dictionary_ready
            and item.get("route_counts") == expected_query_route_counts
            and route_counts == Counter(expected_query_route_counts)
            and int(item.get("dictionary_fallback_count") or 0) == 0
            and item.get("evidence_complete") is True
            and item.get("evidence_failures") == []
            and item.get("raw_text_logged") is False
            and all(
                row.get("route") in QUERY_SIGNAL_REQUIRED_ROUTES
                and row.get("extractor_profile") == query_profile
                and row.get("route_policy") == query_route_policy
                and row.get("route_policy_version") == ROUTE_POLICY_VERSION
                and row.get("dictionary_version") == dictionary_version
                and row.get("profile_fingerprint")
                == item.get("profile_fingerprint")
                and row.get("signal_fingerprint")
                == item.get("signal_fingerprint")
                and bool(
                    str(row.get("cache_namespace_fingerprint") or "")
                )
                and row.get("evidence_complete") is True
                and row.get("evidence_failures") == []
                and row.get("raw_text_logged") is False
                and route_execution_valid(row)
                for row in routes
            )
        )

    observed_query_profiles = Counter(
        str(item.get("active_profile") or "unknown")
        for item in query_cases
    )
    observed_query_policies = Counter(
        str(item.get("configured_route_policy") or "unknown")
        for item in query_cases
    )
    observed_dictionary_versions = Counter(
        str(item.get("dictionary_version") or "unknown")
        for item in query_cases
    )
    observed_query_routes = Counter(
        str(row.get("route") or "unknown")
        for item in query_cases
        for row in (item.get("routes") or [])
        if isinstance(row, dict)
    )
    observed_query_encode_modes: Counter[str] = Counter()
    observed_remote_signal_counts: Counter[str] = Counter()
    observed_consumer_signal_counts: Counter[str] = Counter()
    observed_extractor_kind_counts: Counter[str] = Counter()
    observed_consumer_kind_counts: Counter[str] = Counter()
    for item in query_cases:
        for signal in (item.get("extractor_signals") or []):
            if isinstance(signal, dict):
                observed_extractor_kind_counts.update([
                    str(signal.get("kind") or "unknown")
                ])
        for row in (item.get("routes") or []):
            if not isinstance(row, dict):
                continue
            route = str(row.get("route") or "unknown")
            encode_mode = str(row.get("query_encode_mode") or "unknown")
            observed_query_encode_modes[f"{route}:{encode_mode}"] += 1
            remote_count = row.get("actual_remote_signal_count")
            if (
                isinstance(remote_count, int)
                and not isinstance(remote_count, bool)
                and remote_count >= 0
            ):
                observed_remote_signal_counts[route] += remote_count
            for consumer in ("local", "remote", "rerank"):
                consumer_count = row.get(
                    f"actual_{consumer}_signal_count"
                )
                if (
                    isinstance(consumer_count, int)
                    and not isinstance(consumer_count, bool)
                    and consumer_count >= 0
                ):
                    observed_consumer_signal_counts[
                        f"{route}:{consumer}"
                    ] += consumer_count
                for signal in (
                    row.get(f"actual_{consumer}_signals") or []
                ):
                    if isinstance(signal, dict):
                        observed_consumer_kind_counts.update([
                            f"{route}:{consumer}:"
                            f"{str(signal.get('kind') or 'unknown')}"
                        ])
    query_profiles_bound = bool(
        total
        and query_profile in PROFILE_VERSIONS
        and query_route_policy in ROUTE_POLICIES
        and query_signals.get("active_profile_counts")
        == dict(sorted(observed_query_profiles.items()))
        == {query_profile: total}
        and query_signals.get("route_policy_counts")
        == dict(sorted(observed_query_policies.items()))
        == {query_route_policy: total}
        and query_signals.get("dictionary_version_counts")
        == dict(sorted(observed_dictionary_versions.items()))
    )
    query_routes_accounted = bool(
        total
        and query_signals.get("schema_version")
        == "evaluation-query-signal-aggregate-v2"
        and query_signals.get("route_counts")
        == dict(sorted(observed_query_routes.items()))
        == {
            route: total for route in sorted(QUERY_SIGNAL_REQUIRED_ROUTES)
        }
        and int(query_signals.get("route_evidence_count") or 0)
        == total * len(QUERY_SIGNAL_REQUIRED_ROUTES)
        and int(query_signals.get("evidence_count") or 0) == total
        and int(query_signals.get("evidence_complete_count") or 0) == total
        and query_signals.get("query_encode_mode_counts")
        == dict(sorted(observed_query_encode_modes.items()))
        and query_signals.get("actual_remote_signal_counts")
        == dict(sorted(observed_remote_signal_counts.items()))
        and query_signals.get("actual_consumer_signal_counts")
        == dict(sorted(observed_consumer_signal_counts.items()))
        and query_signals.get("extractor_kind_counts")
        == dict(sorted(observed_extractor_kind_counts.items()))
        and query_signals.get("consumer_kind_counts")
        == dict(sorted(observed_consumer_kind_counts.items()))
        and (
            query_profile != "domain_signals_v1"
            or observed_extractor_kind_counts.get("domain_terms", 0) > 0
        )
    )
    query_execution_clean = bool(
        total
        and int(query_signals.get("dictionary_fallback_count") or 0) == 0
        and all(query_case_valid(item) for item in query_cases)
    )
    planning_profiles_bound = bool(planning_profile and total) and (
        planning.get("active_profile_counts") == {planning_profile: total}
        and planning.get("configured_profile_counts")
        == {planning_profile: total}
        and all(
            item.get("configured_profile") == planning_profile
            and item.get("runtime_profile") == planning_profile
            and item.get("active_profile") == planning_profile
            and item.get("runtime_override_active") is False
            for item in planning_cases
        )
    )
    observed_fallback_profiles = Counter(
        str(item.get("fallback_profile") or "unknown")
        for item in planning_cases
    )
    observed_fallback_classes = Counter(
        str(item.get("fallback_class") or "unknown")
        for item in planning_cases
    )
    observed_routes = Counter(
        str(item.get("route") or "unknown") for item in planning_cases
    )
    observed_plan_calls = sum(
        item.get("plan_called") is True for item in planning_cases
    )

    def case_execution_valid(item: dict[str, Any]) -> bool:
        route = str(item.get("route") or "")
        plan_called = item.get("plan_called")
        expected_plan_call = route == "plan_then_generate"
        return bool(
            isinstance(plan_called, bool)
            and plan_called is expected_plan_call
            and item.get("evidence_failures") == []
            and (
                not expected_plan_call
                or (
                    bool(str(item.get("plan_source") or "").strip())
                    and bool(
                        str(item.get("plan_reason_code") or "").strip()
                    )
                )
            )
        )

    planning_case_execution_bound = bool(
        total
        and planning.get("route_counts")
        == dict(sorted(observed_routes.items()))
        and int(planning.get("plan_calls") or 0) == observed_plan_calls
        and all(case_execution_valid(item) for item in planning_cases)
    )

    def policy_fallback_valid(item: dict[str, Any]) -> bool:
        return bool(
            item.get("fallback_class") == "policy_abstain"
            and item.get("fallback_applied") is True
            and item.get("policy_fallback_eligible") is True
            and item.get("decision") == "abstain"
            and item.get("route")
            in {"direct_generate", "plan_then_generate"}
            and item.get("configured_profile") == planning_profile
            and item.get("runtime_profile") == planning_profile
            and item.get("active_profile") == planning_profile
            and item.get("runtime_override_active") is False
            and item.get("fallback_profile") == planning_fallback_profile
            and planning_fallback_profile in POLICY_FALLBACK_PROFILES
            and planning_profile != planning_fallback_profile
            and item.get("degraded") is False
            and "abstain_fallback" in (item.get("reason_codes") or [])
        )

    def no_fallback_valid(item: dict[str, Any]) -> bool:
        return bool(
            item.get("fallback_class") == "none"
            and item.get("fallback_applied") is False
            and item.get("policy_fallback_eligible") is False
            and item.get("decision") != "abstain"
            and item.get("runtime_override_active") is False
            and item.get("fallback_profile") == planning_fallback_profile
            and item.get("degraded") is False
        )

    policy_fallback_count = sum(
        policy_fallback_valid(item) for item in planning_cases
    )
    unsafe_fallback_count = sum(
        not (policy_fallback_valid(item) or no_fallback_valid(item))
        for item in planning_cases
    )
    planning_policy_fallbacks_bound = bool(
        total
        and planning_fallback_profile in POLICY_FALLBACK_PROFILES
        and planning.get("fallback_profile_counts")
        == dict(sorted(observed_fallback_profiles.items()))
        == {planning_fallback_profile: total}
        and planning.get("fallback_class_counts")
        == dict(sorted(observed_fallback_classes.items()))
        and int(planning.get("policy_fallback_count") or 0)
        == policy_fallback_count
        and int(planning.get("unsafe_fallback_count") or 0)
        == unsafe_fallback_count
    )
    planning_accounted = bool(total) and (
        planning_case_execution_bound
        and int(planning.get("evidence_count") or 0) == total
        and int(planning.get("evidence_complete_count") or 0) == total
        and int(planning.get("decide_node_calls") or 0) == total
        and int(planning.get("decide_node_successes") or 0) == total
        and int(planning.get("observed_plan_node_calls") or 0)
        == int(planning.get("plan_calls") or 0)
        and int(planning.get("plan_node_successes") or 0)
        == int(planning.get("plan_calls") or 0)
        and int(planning.get("physical_llm_attempts") or 0)
        == int(planning.get("physical_llm_calls") or 0)
        and planning.get("decision_accounting_complete") is True
        and planning.get("plan_call_accounting_complete") is True
        and planning.get("physical_llm_call_accounting_complete") is True
    )
    planning_clean = bool(total) and (
        int(planning.get("fallback_count") or 0) == policy_fallback_count
        and unsafe_fallback_count == 0
        and int(planning.get("degraded_count") or 0) == 0
        and all(
            item.get("evidence_complete") is True
            and item.get("degraded") is False
            and (policy_fallback_valid(item) or no_fallback_valid(item))
            for item in planning_cases
        )
    )

    invocation_count = int(repair.get("invocation_count") or 0)
    repair_profiles_bound = bool(repair_profile and total) and (
        planning_repair_profile == repair_profile
        and repair.get("active_profile_counts") == {repair_profile: total}
        and repair.get("configured_profile_counts") == {repair_profile: total}
        and repair.get("invocation_profile_counts")
        == {repair_profile: invocation_count}
        and all(
            item.get("configured_profile") == repair_profile
            and item.get("runtime_profile") == repair_profile
            and item.get("active_profile") == repair_profile
            and item.get("runtime_override_active") is False
            for item in repair_cases
        )
    )
    repair_accounted = bool(total and invocation_count >= total) and (
        int(repair.get("evidence_count") or 0) == total
        and int(repair.get("evidence_complete_count") or 0) == total
        and int(repair.get("ledger_complete_cases") or 0) == total
        and int(repair.get("invocation_evidence_complete_count") or 0)
        == invocation_count
        and repair.get("invocation_evidence_complete") is True
        and int(repair.get("selected_invocation_count") or 0) == total
        and all(
            int(item.get("invocation_count") or 0) >= 1
            and int(item.get("selected_invocation_count") or 0) == 1
            and item.get("ledger_evidence_complete") is True
            for item in repair_cases
        )
    )
    repair_clean = bool(total) and (
        int(repair.get("fallback_count") or 0) == 0
        and int(repair.get("degraded_count") or 0) == 0
        and int(repair.get("invocation_fallback_count") or 0) == 0
        and all(
            item.get("evidence_complete") is True
            and item.get("fallback_used") is False
            and item.get("degraded") is False
            for item in repair_cases
        )
    )
    return {
        "query_signal_profile_bound": query_profiles_bound,
        "query_signal_routes_accounted": query_routes_accounted,
        "query_signal_execution_clean": query_execution_clean,
        "query_signal_evidence_complete": bool(
            query_signals.get("evidence_complete") is True
            and query_profiles_bound
            and query_routes_accounted
            and query_execution_clean
        ),
        "planning_profile_bound": planning_profiles_bound,
        "planning_case_execution_bound": planning_case_execution_bound,
        "planning_execution_accounted": planning_accounted,
        "planning_policy_fallbacks_bound": planning_policy_fallbacks_bound,
        "planning_execution_clean": planning_clean,
        "planning_route_evidence_complete": bool(
            planning.get("evidence_complete") is True
            and planning_profiles_bound
            and planning_policy_fallbacks_bound
            and planning_accounted
            and planning_clean
        ),
        "semantic_repair_profile_bound": repair_profiles_bound,
        "semantic_repair_invocations_accounted": repair_accounted,
        "semantic_repair_execution_clean": repair_clean,
        "semantic_repair_evidence_complete": bool(
            repair.get("evidence_complete") is True
            and repair_profiles_bound
            and repair_accounted
            and repair_clean
        ),
    }


def baseline_checklist(run: dict, full_case_count: int) -> dict:
    fingerprint = run.get("fingerprint") or {}
    aggregate = run.get("aggregate") or {}
    cases = run.get("cases") or []
    candidate_summary = _candidate_governance_summary(run)
    review = candidate_summary["review"]
    pipeline_checks = pipeline_evidence_checks(fingerprint, aggregate, cases)
    checks = {
        "release_artifact": not is_explicit_nonrelease(run),
        "valid_run": bool(run.get("valid")),
        "full_dataset": int(run.get("n") or len(cases)) == full_case_count,
        "fingerprint_complete": bool(fingerprint.get("complete")) and all(
            fingerprint.get(field) for field in REQUIRED_FINGERPRINT_FIELDS
        ),
        "sample_contract_complete": bool((fingerprint.get("sample_contract") or {}).get("complete")),
        "samples_reviewed": bool((fingerprint.get("sample_contract") or {}).get("all_reviewed")),
        "infrastructure_clean": int((run.get("aggregate") or {}).get("infrastructure_errors") or 0) == 0,
        "evidence_chain_complete": bool(cases) and all(
            all(key in (case.get("stage_evidence") or {}) for key in (
                "retrieval", "filtered_tables", "first_sql", "correction_sqls",
                "authorized_sql", "result_summary", "semantic_coverage",
                "request_deadline", "planning_route", "semantic_repair",
                "query_signals",
            )) for case in cases
        ),
        **pipeline_checks,
        "no_raw_result_rows": all("result" not in case for case in cases),
        "evaluation_profile_complete": evaluation_profile_name(
            fingerprint.get("evaluation_profile")
        ) in {"core-default", "enhanced-analysis"},
        "llm_attempt_accounting_complete": (
            aggregate.get("llm_attempt_schema") == LLM_ATTEMPT_SCHEMA
            and aggregate.get("llm_attempt_accounting_complete") is True
        ),
        **llm_release_evidence_checks(aggregate),
        "degradation_evidence_complete": aggregate.get("degradation_evidence_complete") is True,
        "semantic_review_evidence_complete": (
            not candidate_summary.get("review_enabled")
            or (
                candidate_summary.get("review_contract_explicit") is True
                and candidate_summary.get("review_policy_version")
                == SEMANTIC_REVIEW_POLICY_VERSION
                and review.get("evidence_complete") is True
                and review.get("result_attribution_complete") is True
            )
        ),
    }
    return {"checks": checks, "eligible": all(checks.values())}
