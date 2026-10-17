"""
NL2SQL 评测执行器（TODO-23）。

一句话总述：对黄金集每条用例驱动 LangGraph 跑完整问数流水线，采集最终 state，
            调 metrics 计算多维指标，返回逐用例结果供 run_eval 聚合为报告。
"""
import time
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from app.agent_system.agents.nl2sql.context import DataAgentContext
from app.agent_system.agents.nl2sql.graph import compile_graph
from app.agent_system.agents.nl2sql.services.candidate_confidence import (
    compact_candidate_decision,
    normalized_sql_equal,
)
from app.agent_system.agents.nl2sql.state import DataAgentState
from app.infrastructure.observability.logging import logger
from evals.legacy.metrics import (
    question_requires_result_order,
    result_equivalence_evidence,
    sql_equivalent,
    sql_structure_equivalent,
    syntax_valid,
    table_prf,
    table_contract_prf,
    tables_match_contract,
    tables_correct,
)
from evals.legacy.attribution import attribute_case
from app.application.services.query_trace_service import QueryTraceCollector
from app.adapters.outbound.agent.llm_provider import llm_route_manifest
from app.config.app_config import app_config
from app.agent_system.runtime.llm_attempts import attempt_coverage, normalize_degradation
from app.infrastructure.clients.llm.invocation import FATAL_PROVIDER_REASON_CODES
from app.agent_system.agents.nl2sql.services.planning_router import (
    IMPLEMENTED_PROFILES,
    POLICY_FALLBACK_PROFILES,
    runtime_profile_status as planning_runtime_profile_status,
)
from app.agent_system.agents.nl2sql.services.query_signals import (
    PROFILE_VERSIONS as QUERY_SIGNAL_PROFILES,
    ROUTE_POLICIES as QUERY_SIGNAL_ROUTE_POLICIES,
    ROUTE_POLICY_VERSION as QUERY_SIGNAL_ROUTE_POLICY_VERSION,
)
from app.agent_system.agents.nl2sql.services.semantic_repair import (
    PROFILE_RULES,
    REPAIR_REGISTRY_VERSION,
    REPAIR_RESULT_CONTRACT,
    RULE_SET_VERSION,
    SEMANTIC_REPAIR_LEDGER_LIMIT,
    SEMANTIC_REPAIR_LEDGER_PHASES,
    SEMANTIC_REPAIR_LEDGER_SCHEMA,
    compact_semantic_repair,
    runtime_profile_status as semantic_repair_runtime_profile_status,
)
from app.domain.query.safety.semantic_violation import stable_hash as semantic_sql_hash
from evals.legacy.profiles import resolve_evaluation_profile
from app.agent_system.agents.nl2sql.contracts.planning_route import (
    ACTUAL_ACTIONS,
    PLANNING_ROUTE_CONTRACT_VERSION,
    compact_planning_route,
)
from app.agent_system.agents.nl2sql.contracts.query_signals import (
    QUERY_SIGNAL_CONTRACT_VERSION,
    SIGNAL_KINDS,
    QuerySignalsV1,
    stable_signal_hash,
)
from app.agent_system.governance.deadline import (
    compact_deadline_evidence,
    initialize_request_deadline,
)
from app.domain.data_sources.context import (
    authorize_sql_assets,
    ensure_context_assets_current,
    stable_fingerprint,
    translate_logical_upload_tables,
)
from app.domain.query.safety.source_analysis_guard import validate_source_analysis_sql
from app.domain.query.safety.sql_security import (
    check_join_relationships,
    check_sql_read,
    check_table_whitelist,
    extract_table_names,
)


@lru_cache(maxsize=1)
def _compiled_evaluation_graph():
    """仅在真实评测执行时编译并复用 Graph，模块导入阶段不产生副作用。"""

    return compile_graph()


class _LazyEvaluationGraph:
    """保留旧测试注入点，同时把 Graph 编译延迟到首次评测调用。"""

    async def astream(self, **kwargs):
        async for state in _compiled_evaluation_graph().astream(**kwargs):
            yield state


graph = _LazyEvaluationGraph()


EVALUATION_COMPLETE_RESULT_LIMIT = 10_000
QUERY_SIGNAL_EVIDENCE_SCHEMA = "evaluation-query-signal-evidence-v2"
PLANNING_ROUTE_EVIDENCE_SCHEMA = "evaluation-planning-route-evidence-v1"
SEMANTIC_REPAIR_EVIDENCE_SCHEMA = "evaluation-semantic-repair-evidence-v1"
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
PLANNING_FALLBACK_CLASSES = frozenset({
    "none",
    "policy_abstain",
    "exception_degraded",
    "profile_override",
    "unknown",
})


def _query_signal_stage_evidence(
    state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """把实际 extractor 与四路消费证据绑定到当前静态配置。"""

    payload = dict(state or {})
    configured_profile = str(app_config.query_signals.profile or "")
    configured_route_policy = str(app_config.query_signals.route_policy or "")
    failures: list[str] = []
    try:
        signals = QuerySignalsV1.from_state(payload.get("query_signals"))
        summary = signals.privacy_summary()
    except (TypeError, ValueError):
        summary = {}
        failures.append("query_signal_contract_missing")

    active_profile = str(summary.get("extractor_profile") or "")
    dictionary_version = str(summary.get("dictionary_version") or "")
    profile_fingerprint = str(summary.get("profile_fingerprint") or "")
    signal_fingerprint = str(summary.get("signal_fingerprint") or "")
    if configured_profile not in QUERY_SIGNAL_PROFILES:
        failures.append("configured_profile_unknown")
    if configured_route_policy not in QUERY_SIGNAL_ROUTE_POLICIES:
        failures.append("configured_route_policy_unknown")
    if summary.get("contract_version") != QUERY_SIGNAL_CONTRACT_VERSION:
        failures.append("contract_version_mismatch")
    if active_profile != configured_profile:
        failures.append("active_profile_mismatch")
    if not profile_fingerprint:
        failures.append("profile_fingerprint_missing")
    if not signal_fingerprint:
        failures.append("signal_fingerprint_missing")
    if summary.get("raw_text_logged") is not False:
        failures.append("privacy_declaration_missing")
    if (
        configured_profile == "domain_signals_v1"
        and app_config.query_signals.workspace_dictionary_enabled
        and dictionary_version in {"", "none"}
    ):
        failures.append("workspace_dictionary_missing")

    extractor_signals: list[dict[str, Any]] = []
    extractor_signal_keys: set[tuple[str, str, str]] = set()
    raw_extractor_signals = summary.get("signals")
    if not isinstance(raw_extractor_signals, list):
        raw_extractor_signals = []
        failures.append("extractor_signal_identities_missing")
    for signal_index, signal_value in enumerate(raw_extractor_signals):
        signal = (
            dict(signal_value)
            if isinstance(signal_value, Mapping)
            else {}
        )
        signal_id = str(signal.get("signal_id") or "")
        kind = str(signal.get("kind") or "")
        fingerprint = str(signal.get("fingerprint") or "")
        rank = signal.get("rank")
        prefix = f"extractor_signals[{signal_index}]"
        if (
            len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            failures.append(f"{prefix}.fingerprint_invalid")
        if signal_id != f"qs-{fingerprint[:12]}":
            failures.append(f"{prefix}.signal_id_mismatch")
        if kind not in {"standalone_query", *SIGNAL_KINDS}:
            failures.append(f"{prefix}.kind_invalid")
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank < 0
        ):
            failures.append(f"{prefix}.rank_invalid")
        identity = (signal_id, kind, fingerprint)
        if identity in extractor_signal_keys:
            failures.append(f"{prefix}.duplicate")
        extractor_signal_keys.add(identity)
        extractor_signals.append({
            "signal_id": signal_id,
            "kind": kind,
            "rank": rank,
            "fingerprint": fingerprint,
        })
    if not extractor_signals:
        failures.append("extractor_signal_identities_empty")
    extractor_signal_identity_fingerprint = stable_fingerprint(
        extractor_signals
    )
    extractor_kind_counts: dict[str, int] = {}
    for row in extractor_signals:
        kind = str(row.get("kind") or "unknown")
        extractor_kind_counts[kind] = extractor_kind_counts.get(kind, 0) + 1
    domain_signal_keys = {
        (row["signal_id"], row["kind"], row["fingerprint"])
        for row in extractor_signals
        if row.get("kind") == "domain_terms"
    }

    routes: list[dict[str, Any]] = []
    retrieval_rows = payload.get("retrieval_metrics")
    retrieval_rows = retrieval_rows if isinstance(retrieval_rows, list) else []
    for index, value in enumerate(retrieval_rows):
        row = dict(value) if isinstance(value, Mapping) else {}
        raw = row.get("query_signals")
        if not isinstance(raw, Mapping):
            continue
        route = str(raw.get("route") or row.get("route") or "")
        row_failures: list[str] = []
        if route not in QUERY_SIGNAL_REQUIRED_ROUTES:
            row_failures.append("route_invalid")
        if raw.get("contract_version") != QUERY_SIGNAL_CONTRACT_VERSION:
            row_failures.append("contract_version_mismatch")
        if str(raw.get("extractor_profile") or "") != active_profile:
            row_failures.append("active_profile_mismatch")
        if str(raw.get("route_policy") or "") != configured_route_policy:
            row_failures.append("route_policy_mismatch")
        if raw.get("route_policy_version") != QUERY_SIGNAL_ROUTE_POLICY_VERSION:
            row_failures.append("route_policy_version_mismatch")
        if str(raw.get("dictionary_version") or "") != dictionary_version:
            row_failures.append("dictionary_version_mismatch")
        if str(raw.get("profile_fingerprint") or "") != profile_fingerprint:
            row_failures.append("profile_fingerprint_mismatch")
        if str(raw.get("signal_fingerprint") or "") != signal_fingerprint:
            row_failures.append("signal_fingerprint_mismatch")
        if not str(raw.get("cache_namespace_fingerprint") or ""):
            row_failures.append("cache_namespace_fingerprint_missing")
        if raw.get("raw_text_logged") is not False:
            row_failures.append("privacy_declaration_missing")

        def non_negative_count(field: str) -> int | None:
            value = raw.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                row_failures.append(f"{field}_invalid")
                return None
            return value

        local_signal_count = non_negative_count("local_signal_count")
        remote_query_count = non_negative_count("remote_query_count")
        rerank_signal_count = non_negative_count("rerank_signal_count")
        query_encode_mode = str(raw.get("query_encode_mode") or "")
        expected_encode_mode = QUERY_SIGNAL_ROUTE_ENCODE_MODES.get(route)
        if query_encode_mode != expected_encode_mode:
            row_failures.append("query_encode_mode_mismatch")

        def compact_consumer_signals(
            field: str,
            consumer: str,
        ) -> list[dict[str, Any]]:
            raw_signals = raw.get(field)
            if not isinstance(raw_signals, list):
                raw_signals = []
                row_failures.append(f"{field}_missing")
            compacted: list[dict[str, Any]] = []
            consumer_keys: set[tuple[str, str]] = set()
            for signal_index, signal_value in enumerate(raw_signals):
                signal = (
                    dict(signal_value)
                    if isinstance(signal_value, Mapping)
                    else {}
                )
                signal_id = str(signal.get("signal_id") or "")
                kind = str(signal.get("kind") or "")
                fingerprint = str(signal.get("fingerprint") or "")
                actual_consumer = str(signal.get("consumer") or "")
                rank = signal.get("rank")
                prefix = f"{field}[{signal_index}]"
                if (
                    len(fingerprint) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in fingerprint
                    )
                ):
                    row_failures.append(f"{prefix}.fingerprint_invalid")
                if signal_id != f"qs-{fingerprint[:12]}":
                    row_failures.append(f"{prefix}.signal_id_mismatch")
                if kind not in QUERY_SIGNAL_CONSUMER_KINDS:
                    row_failures.append(f"{prefix}.kind_invalid")
                if actual_consumer != consumer:
                    row_failures.append(f"{prefix}.consumer_invalid")
                if (
                    not isinstance(rank, int)
                    or isinstance(rank, bool)
                    or rank < 0
                ):
                    row_failures.append(f"{prefix}.rank_invalid")
                if "text" in signal:
                    row_failures.append(f"{prefix}.raw_text_present")
                identity = (signal_id, kind, fingerprint)
                if (
                    kind in {"standalone_query", *SIGNAL_KINDS}
                    and identity not in extractor_signal_keys
                ):
                    row_failures.append(
                        f"{prefix}.extractor_identity_mismatch"
                    )
                signal_key = (kind, fingerprint)
                if signal_key in consumer_keys:
                    row_failures.append(f"{prefix}.duplicate")
                consumer_keys.add(signal_key)
                compacted.append({
                    "signal_id": signal_id,
                    "kind": kind,
                    "rank": rank,
                    "fingerprint": fingerprint,
                    "consumer": actual_consumer,
                })
            return compacted

        local_signals = compact_consumer_signals(
            "actual_local_signals", "local"
        )
        remote_signals = compact_consumer_signals(
            "actual_remote_signals", "remote"
        )
        rerank_signals = compact_consumer_signals(
            "actual_rerank_signals", "rerank"
        )
        consumer_counts = {
            "local": (local_signal_count, len(local_signals)),
            "remote": (remote_query_count, len(remote_signals)),
            "rerank": (rerank_signal_count, len(rerank_signals)),
        }
        for consumer, (declared_count, observed_count) in consumer_counts.items():
            if declared_count != observed_count:
                row_failures.append(f"{consumer}_signal_count_mismatch")
            if observed_count == 0:
                row_failures.append(f"actual_{consumer}_signals_empty")
        local_keys = {
            (item["signal_id"], item["kind"], item["fingerprint"])
            for item in local_signals
        }
        rerank_keys = {
            (item["signal_id"], item["kind"], item["fingerprint"])
            for item in rerank_signals
        }
        if active_profile == "domain_signals_v1":
            local_domain_keys = {
                identity for identity in local_keys
                if identity[1] == "domain_terms"
            }
            rerank_domain_keys = {
                identity for identity in rerank_keys
                if identity[1] == "domain_terms"
            }
            if local_domain_keys != domain_signal_keys:
                row_failures.append("domain_signals_missing_from_local")
            if rerank_domain_keys != domain_signal_keys:
                row_failures.append("domain_signals_missing_from_rerank")
        expected_consumer_plan_fingerprint = stable_signal_hash({
            "contract": QUERY_SIGNAL_CONTRACT_VERSION,
            "route_policy_version": QUERY_SIGNAL_ROUTE_POLICY_VERSION,
            "route_policy": configured_route_policy,
            "route": route,
            "profile_fingerprint": profile_fingerprint,
            "signal_fingerprint": signal_fingerprint,
            "query_encode_mode": query_encode_mode,
            "local": local_signals,
            "remote": remote_signals,
            "rerank": rerank_signals,
        })
        consumer_plan_fingerprint = str(
            raw.get("consumer_plan_fingerprint") or ""
        )
        if consumer_plan_fingerprint != expected_consumer_plan_fingerprint:
            row_failures.append("consumer_plan_fingerprint_mismatch")
        consumer_kind_counts: dict[str, dict[str, int]] = {}
        for consumer, signals in (
            ("local", local_signals),
            ("remote", remote_signals),
            ("rerank", rerank_signals),
        ):
            counts: dict[str, int] = {}
            for signal in signals:
                kind = str(signal.get("kind") or "unknown")
                counts[kind] = counts.get(kind, 0) + 1
            consumer_kind_counts[consumer] = dict(sorted(counts.items()))
        routes.append({
            "route": route,
            "extractor_profile": str(raw.get("extractor_profile") or ""),
            "route_policy": str(raw.get("route_policy") or ""),
            "route_policy_version": str(
                raw.get("route_policy_version") or ""
            ),
            "dictionary_version": str(raw.get("dictionary_version") or ""),
            "profile_fingerprint": str(
                raw.get("profile_fingerprint") or ""
            ),
            "signal_fingerprint": str(raw.get("signal_fingerprint") or ""),
            "cache_namespace_fingerprint": str(
                raw.get("cache_namespace_fingerprint") or ""
            ),
            "query_encode_mode": query_encode_mode,
            "local_signal_count": local_signal_count,
            "remote_query_count": remote_query_count,
            "rerank_signal_count": rerank_signal_count,
            "actual_local_signal_count": len(local_signals),
            "actual_remote_signal_count": len(remote_signals),
            "actual_rerank_signal_count": len(rerank_signals),
            "actual_local_signals": local_signals,
            "actual_remote_signals": remote_signals,
            "actual_rerank_signals": rerank_signals,
            "local_signal_fingerprint": stable_fingerprint(local_signals),
            "remote_signal_fingerprint": stable_fingerprint(remote_signals),
            "rerank_signal_fingerprint": stable_fingerprint(rerank_signals),
            "consumer_plan_fingerprint": consumer_plan_fingerprint,
            "consumer_kind_counts": consumer_kind_counts,
            "evidence_complete": not row_failures,
            "evidence_failures": row_failures,
            "raw_text_logged": False,
        })
        failures.extend(
            f"route[{index}]:{reason}" for reason in row_failures
        )

    route_counts: dict[str, int] = {}
    for row in routes:
        route = str(row.get("route") or "unknown")
        route_counts[route] = route_counts.get(route, 0) + 1
    expected_route_counts = {
        route: 1 for route in sorted(QUERY_SIGNAL_REQUIRED_ROUTES)
    }
    if route_counts != expected_route_counts:
        failures.append("route_coverage_mismatch")

    dictionary_fallback_count = sum(
        isinstance(value, Mapping)
        and value.get("dependency") == "query_signal_dictionary"
        and value.get("status") == "fallback"
        for value in (payload.get("degradations") or [])
    )
    if dictionary_fallback_count:
        failures.append("workspace_dictionary_fallback")

    return {
        "schema_version": QUERY_SIGNAL_EVIDENCE_SCHEMA,
        "contract_version": str(summary.get("contract_version") or ""),
        "configured_profile": configured_profile,
        "active_profile": active_profile,
        "configured_route_policy": configured_route_policy,
        "dictionary_version": dictionary_version,
        "profile_fingerprint": profile_fingerprint,
        "signal_fingerprint": signal_fingerprint,
        "extractor_signals": extractor_signals,
        "extractor_signal_identity_fingerprint": (
            extractor_signal_identity_fingerprint
        ),
        "extractor_kind_counts": dict(sorted(extractor_kind_counts.items())),
        "route_counts": dict(sorted(route_counts.items())),
        "routes": routes,
        "dictionary_fallback_count": dictionary_fallback_count,
        "evidence_complete": not failures,
        "evidence_failures": failures,
        "raw_text_logged": False,
    }


def _planning_fallback_class(
    *,
    decision: str,
    fallback_applied: bool,
    degraded: bool,
    runtime_override: Any,
) -> str:
    """仅用有界控制字段分类 fallback，不持久化问题或 SQL。"""

    if runtime_override is not None:
        return "profile_override"
    if degraded:
        return "exception_degraded"
    if fallback_applied and decision == "abstain":
        return "policy_abstain"
    if fallback_applied:
        return "unknown"
    return "none"


def _planning_route_stage_evidence(state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Build fail-closed, privacy-safe evidence for the route actually used."""

    payload = dict(state or {})
    raw = payload.get("planning_route")
    raw = dict(raw) if isinstance(raw, Mapping) else {}
    compact = compact_planning_route(raw)
    runtime = planning_runtime_profile_status()
    configured_profile = str(runtime.get("configured_profile") or "")
    runtime_profile = str(runtime.get("runtime_profile") or "")
    runtime_override = runtime.get("runtime_override")
    configured_fallback = str(runtime.get("fallback_profile") or "")
    configured_repair = str(app_config.planning_route.repair_profile or "")
    active_profile = str(compact.get("profile") or "")
    decision = str(compact.get("decision") or "")
    route = str(compact.get("actual_action") or "")
    fallback_applied = bool(compact.get("fallback_applied"))
    degraded = bool(compact.get("router_degraded"))
    plan_evidence = payload.get("planning_evidence")
    plan_evidence = (
        dict(plan_evidence) if isinstance(plan_evidence, Mapping) else {}
    )
    plan_called = bool(plan_evidence)
    physical_llm_call = bool(plan_evidence.get("physical_llm_call"))
    fallback_class = _planning_fallback_class(
        decision=decision,
        fallback_applied=fallback_applied,
        degraded=degraded,
        runtime_override=runtime_override,
    )
    reason_codes = [
        str(value)[:80] for value in (compact.get("reason_codes") or [])[:16]
    ]
    policy_fallback_eligible = bool(
        fallback_class == "policy_abstain"
        and configured_fallback in POLICY_FALLBACK_PROFILES
        and active_profile == configured_profile == runtime_profile
        and active_profile != configured_fallback
        and route in {"direct_generate", "plan_then_generate"}
        and "abstain_fallback" in reason_codes
    )

    failures: list[str] = []
    required_fields = {
        "contract_version", "profile", "decision", "actual_action",
        "fallback_profile", "fallback_applied", "router_degraded",
        "repair_profile", "raw_text_logged",
    }
    missing = sorted(required_fields - set(raw))
    if missing:
        failures.extend(f"missing:{field}" for field in missing)
    if compact.get("contract_version") != PLANNING_ROUTE_CONTRACT_VERSION:
        failures.append("contract_version_mismatch")
    if configured_profile not in IMPLEMENTED_PROFILES:
        failures.append("configured_profile_unknown")
    if runtime_profile not in IMPLEMENTED_PROFILES:
        failures.append("runtime_profile_unknown")
    if configured_profile != runtime_profile:
        failures.append("configured_runtime_profile_mismatch")
    if runtime_override is not None:
        failures.append("runtime_override_active")
    if configured_fallback not in IMPLEMENTED_PROFILES:
        failures.append("configured_fallback_unknown")
    if str(compact.get("fallback_profile") or "") != configured_fallback:
        failures.append("fallback_profile_mismatch")
    if active_profile != configured_profile:
        failures.append("active_profile_mismatch")
    if fallback_class == "policy_abstain":
        if not policy_fallback_eligible:
            failures.append("policy_fallback_invalid")
    elif fallback_class == "exception_degraded":
        failures.append("router_degraded")
    elif fallback_class == "profile_override":
        failures.append("profile_override_fallback")
    elif fallback_class == "unknown":
        failures.append("fallback_unclassified")
    elif fallback_class != "none":
        failures.append("fallback_class_unknown")
    if decision == "abstain" and fallback_class != "policy_abstain":
        failures.append("abstain_without_policy_fallback")
    if route not in ACTUAL_ACTIONS:
        failures.append("route_invalid")
    if str(compact.get("repair_profile") or "") != configured_repair:
        failures.append("repair_profile_mismatch")
    if raw.get("raw_text_logged") is not False:
        failures.append("privacy_declaration_missing")
    expected_plan_call = route == "plan_then_generate"
    if plan_called != expected_plan_call:
        failures.append("plan_call_mismatch")
    if plan_called:
        if plan_evidence.get("schema_version") != "planning-decision-v1":
            failures.append("plan_evidence_contract_mismatch")
        if not str(plan_evidence.get("source") or ""):
            failures.append("plan_source_missing")
        if not str(plan_evidence.get("reason_code") or ""):
            failures.append("plan_reason_missing")

    return {
        "schema_version": PLANNING_ROUTE_EVIDENCE_SCHEMA,
        "contract_version": str(compact.get("contract_version") or ""),
        "configured_profile": configured_profile,
        "runtime_profile": runtime_profile,
        "runtime_override_active": runtime_override is not None,
        "active_profile": active_profile,
        "decision": decision,
        "route": route,
        "plan_called": plan_called,
        "physical_llm_call": physical_llm_call,
        "plan_source": str(plan_evidence.get("source") or "")[:64],
        "plan_reason_code": str(plan_evidence.get("reason_code") or "")[:96],
        "fallback_profile": str(compact.get("fallback_profile") or ""),
        "fallback_applied": fallback_applied,
        "fallback_class": fallback_class,
        "policy_fallback_eligible": policy_fallback_eligible,
        "degraded": degraded,
        "repair_profile": str(compact.get("repair_profile") or ""),
        "reason_codes": reason_codes,
        "evidence_complete": not failures,
        "evidence_failures": failures,
        "raw_text_logged": False,
    }


def _semantic_repair_stage_evidence(
    state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Bind final repair and every candidate invocation to SQL-free hashes."""

    payload = dict(state or {})
    validation = payload.get("validation_evidence")
    validation = dict(validation) if isinstance(validation, Mapping) else {}
    raw = validation.get("semantic_repair")
    raw = dict(raw) if isinstance(raw, Mapping) else {}
    runtime = semantic_repair_runtime_profile_status()
    configured_profile = str(runtime.get("configured_profile") or "")
    runtime_profile = str(runtime.get("runtime_profile") or "")
    runtime_override_active = bool(runtime.get("override_active"))
    configured_fallback = str(runtime.get("fallback_profile") or "")
    planning_repair_profile = str(
        app_config.planning_route.repair_profile or ""
    )
    final_sql = str(payload.get("sql") or "").strip()
    final_sql_hash = semantic_sql_hash(final_sql)

    raw_ledger = payload.get("semantic_repair_ledger")
    ledger_source = raw_ledger if isinstance(raw_ledger, list) else []
    ledger_failures: list[str] = []
    if not isinstance(raw_ledger, list) or not ledger_source:
        ledger_failures.append("ledger_missing")
    if len(ledger_source) > SEMANTIC_REPAIR_LEDGER_LIMIT:
        ledger_failures.append("ledger_limit_exceeded")
    ledger: list[dict[str, Any]] = []
    for index, value in enumerate(ledger_source[:SEMANTIC_REPAIR_LEDGER_LIMIT]):
        item = dict(value) if isinstance(value, Mapping) else {}
        repair = compact_semantic_repair(item)
        row_failures: list[str] = []
        row_required = {
            "schema_version", "phase", "committed_at_phase", "selected",
            "validation_valid",
            "contract_version", "registry_version", "rule_set_version",
            "profile", "original_sql_hash", "repaired_sql_hash",
            "observed_sql_hash", "attempted_rule_codes", "applied_rules",
            "skipped_rules", "fallback_used", "invariants",
            "raw_sql_logged",
        }
        row_missing = sorted(row_required - set(item))
        row_failures.extend(f"missing:{field}" for field in row_missing)
        phase = str(item.get("phase") or "")[:64]
        if item.get("schema_version") != SEMANTIC_REPAIR_LEDGER_SCHEMA:
            row_failures.append("schema_version_mismatch")
        if phase not in SEMANTIC_REPAIR_LEDGER_PHASES:
            row_failures.append("phase_invalid")
        if not isinstance(item.get("committed_at_phase"), bool):
            row_failures.append("committed_at_phase_invalid")
        if not isinstance(item.get("selected"), bool):
            row_failures.append("selected_invalid")
        if item.get("selected") is not item.get("committed_at_phase"):
            row_failures.append("selected_commit_alias_mismatch")
        if repair.get("contract_version") != REPAIR_RESULT_CONTRACT:
            row_failures.append("contract_version_mismatch")
        if repair.get("registry_version") != REPAIR_REGISTRY_VERSION:
            row_failures.append("registry_version_mismatch")
        if repair.get("rule_set_version") != RULE_SET_VERSION:
            row_failures.append("rule_set_version_mismatch")
        row_profile = str(repair.get("profile") or "")
        if row_profile != configured_profile:
            row_failures.append("profile_mismatch")
        original_hash = str(repair.get("original_sql_hash") or "")
        repaired_hash = str(repair.get("repaired_sql_hash") or "")
        observed_hash = str(item.get("observed_sql_hash") or "")[:80]
        committed_at_phase = item.get("committed_at_phase") is True
        if not original_hash or not repaired_hash or not observed_hash:
            row_failures.append("sql_hash_missing")
        if repaired_hash != observed_hash:
            row_failures.append("observed_sql_hash_mismatch")
        if repair.get("fallback_used"):
            row_failures.append("fallback_used")
        if item.get("raw_sql_logged") is not False:
            row_failures.append("privacy_declaration_missing")
        row_attempted = [
            str(code) for code in repair.get("attempted_rule_codes") or []
        ]
        row_observed_rules = {
            str(row.get("rule_code") or "")
            for key in ("applied_rules", "skipped_rules")
            for row in repair.get(key) or []
            if isinstance(row, Mapping) and row.get("rule_code")
        }
        if set(row_attempted) != set(PROFILE_RULES.get(row_profile) or ()):
            row_failures.append("profile_rule_set_mismatch")
        if set(row_attempted) != row_observed_rules:
            row_failures.append("rule_evidence_incomplete")
        row_invariants = dict(repair.get("invariants") or {})
        if not row_invariants or not all(row_invariants.values()):
            row_failures.append("repair_invariants_incomplete")
        row = {
            "schema_version": SEMANTIC_REPAIR_LEDGER_SCHEMA,
            "phase": phase,
            "committed_at_phase": committed_at_phase,
            "selected_at_phase": committed_at_phase,
            "selected": False,
            "final_invocation": False,
            "validation_valid": item.get("validation_valid") is True,
            "contract_version": str(repair.get("contract_version") or ""),
            "registry_version": str(repair.get("registry_version") or ""),
            "rule_set_version": str(repair.get("rule_set_version") or ""),
            "profile": row_profile,
            "original_sql_hash": original_hash,
            "repaired_sql_hash": repaired_hash,
            "observed_sql_hash": observed_hash,
            "attempted_rule_codes": row_attempted,
            "applied_rules": list(repair.get("applied_rules") or []),
            "skipped_rules": list(repair.get("skipped_rules") or []),
            "fallback_used": bool(repair.get("fallback_used")),
            "fallback_reason": repair.get("fallback_reason"),
            "invariants": row_invariants,
            "evidence_complete": not row_failures,
            "evidence_failures": row_failures,
            "raw_sql_logged": False,
        }
        ledger.append(row)
        ledger_failures.extend(
            f"entry[{index}]:{reason}" for reason in row_failures
        )
    final_candidates = [
        index
        for index, row in enumerate(ledger)
        if row.get("committed_at_phase") is True
        and row.get("observed_sql_hash") == final_sql_hash
    ]
    final_index: int | None = None
    if final_candidates:
        # ledger 是跨重试 append-only 的历史；同一 SQL 可在多个阶段正常收敛并
        # 重复提交。派生 final 身份不信任输入标志，固定由最后一个已提交且与
        # 最终 SQL 哈希一致的调用确定，再把输出归一为唯一 selected/final。
        final_index = final_candidates[-1]
        ledger[final_index]["selected"] = True
        ledger[final_index]["final_invocation"] = True
    else:
        ledger_failures.append("final_committed_invocation_missing")
    selected_invocation_count = int(final_index is not None)

    summary_source = "validation_evidence"
    execution_error_code = str(validation.get("error_code") or "")
    if (
        not raw
        and execution_error_code.startswith("execution.")
        and final_index is not None
        and ledger[final_index].get("evidence_complete") is True
    ):
        # execute_sql 会以执行错误覆盖 validation_evidence。最终 ledger 已在
        # validate_sql 阶段固化完整且脱敏的修复结果，可在严格身份与哈希校验后
        # 恢复请求级摘要；执行正确性仍由独立结果门判定，不在此处放宽。
        raw = dict(ledger_source[final_index])
        summary_source = "final_ledger"

    compact = compact_semantic_repair(raw)
    actual_profile = str(compact.get("profile") or "")
    fallback_used = bool(compact.get("fallback_used"))
    failures: list[str] = []
    required_fields = {
        "contract_version", "registry_version", "rule_set_version", "profile",
        "original_sql_hash", "repaired_sql_hash", "applied_rules",
        "skipped_rules", "attempted_rule_codes", "fallback_used",
        "invariants", "raw_sql_logged",
    }
    missing = sorted(required_fields - set(raw))
    if missing:
        failures.extend(f"missing:{field}" for field in missing)
    if compact.get("contract_version") != REPAIR_RESULT_CONTRACT:
        failures.append("contract_version_mismatch")
    if compact.get("registry_version") != REPAIR_REGISTRY_VERSION:
        failures.append("registry_version_mismatch")
    if compact.get("rule_set_version") != RULE_SET_VERSION:
        failures.append("rule_set_version_mismatch")
    if configured_profile not in PROFILE_RULES:
        failures.append("configured_profile_unknown")
    if runtime_profile not in PROFILE_RULES:
        failures.append("runtime_profile_unknown")
    if configured_profile != runtime_profile:
        failures.append("configured_runtime_profile_mismatch")
    if runtime_override_active:
        failures.append("runtime_override_active")
    if configured_fallback not in PROFILE_RULES:
        failures.append("configured_fallback_unknown")
    if planning_repair_profile != configured_profile:
        failures.append("planning_repair_profile_mismatch")
    if actual_profile != configured_profile:
        failures.append("active_profile_mismatch")
    if fallback_used:
        failures.append("fallback_used")
    if raw.get("raw_sql_logged") is not False:
        failures.append("privacy_declaration_missing")
    if not compact.get("original_sql_hash") or not compact.get(
        "repaired_sql_hash"
    ):
        failures.append("sql_hash_missing")
    if not final_sql:
        failures.append("final_sql_missing")
    if compact.get("repaired_sql_hash") != final_sql_hash:
        failures.append("final_sql_hash_mismatch")
    attempted = [
        str(value) for value in (compact.get("attempted_rule_codes") or [])
    ]
    observed = {
        str(row.get("rule_code") or "")
        for key in ("applied_rules", "skipped_rules")
        for row in (compact.get(key) or [])
        if isinstance(row, Mapping) and row.get("rule_code")
    }
    if not attempted:
        failures.append("attempted_rules_missing")
    expected_rules = set(PROFILE_RULES.get(actual_profile) or ())
    if set(attempted) != expected_rules:
        failures.append("profile_rule_set_mismatch")
    if set(attempted) != observed:
        failures.append("rule_evidence_incomplete")
    invariants = dict(compact.get("invariants") or {})
    if not invariants or not all(invariants.values()):
        failures.append("repair_invariants_incomplete")
    failures.extend(ledger_failures)

    return {
        "schema_version": SEMANTIC_REPAIR_EVIDENCE_SCHEMA,
        "contract_version": str(compact.get("contract_version") or ""),
        "registry_version": str(compact.get("registry_version") or ""),
        "rule_set_version": str(compact.get("rule_set_version") or ""),
        "configured_profile": configured_profile,
        "runtime_profile": runtime_profile,
        "runtime_override_active": runtime_override_active,
        "active_profile": actual_profile,
        "summary_source": summary_source,
        "fallback_profile": configured_fallback,
        "fallback_used": fallback_used,
        "fallback_reason": compact.get("fallback_reason"),
        "degraded": fallback_used,
        "changed": bool(compact.get("changed")),
        "original_sql_hash": str(compact.get("original_sql_hash") or ""),
        "repaired_sql_hash": str(compact.get("repaired_sql_hash") or ""),
        "final_sql_hash": final_sql_hash,
        "attempted_rule_codes": attempted,
        "applied_rules": list(compact.get("applied_rules") or []),
        "skipped_rules": list(compact.get("skipped_rules") or []),
        "invariants": invariants,
        "ledger": ledger,
        "invocation_count": len(ledger_source),
        "selected_invocation_count": selected_invocation_count,
        "ledger_evidence_complete": not ledger_failures,
        "ledger_truncated": len(ledger_source) > SEMANTIC_REPAIR_LEDGER_LIMIT,
        "evidence_complete": not failures,
        "evidence_failures": failures,
        "raw_sql_logged": False,
    }


async def _execute_complete_result(dw_repo, sql: str) -> list[dict]:
    """Execute the governed complete-result path used by current evaluations."""
    return await dw_repo.execute_sql_for_evaluation(
        sql,
        max_rows=EVALUATION_COMPLETE_RESULT_LIMIT,
    )


def _observe_sql_state(observations: list[dict[str, Any]], state: dict | None) -> None:
    """Track generated SQL revisions and whether each one crossed all guards."""
    payload = state or {}
    sql = str(payload.get("sql") or "").strip()
    if not sql:
        return
    retry_count = int(payload.get("retry_count") or 0)
    if not observations or observations[-1]["sql"] != sql:
        if not observations:
            source = "generate_sql"
        elif retry_count > max(
            int(item.get("retry_count") or 0) for item in observations
        ):
            source = "correct_sql"
        elif (payload.get("candidate_decision") or {}).get("selected") == "B":
            source = "judge_candidate"
        elif (payload.get("validation_evidence") or {}).get("semantic_repair"):
            source = "semantic_repair"
        else:
            source = "post_generation_unknown"
        observations.append({
            "sql": sql,
            "source": source,
            "retry_count": retry_count,
            "authorized_sql": "",
            "guard_passed": False,
            "error_code": None,
        })
    current = observations[-1]
    current["retry_count"] = max(int(current.get("retry_count") or 0), retry_count)
    authorized = str(payload.get("authorized_sql") or "").strip()
    if authorized:
        current["authorized_sql"] = authorized
        current["guard_passed"] = True
    validation = payload.get("validation_evidence") or {}
    if validation.get("error_code"):
        current["error_code"] = str(validation["error_code"])


def _runtime_allowed_tables(state: dict[str, Any] | None) -> set[str]:
    """Return the request-scoped table whitelist used by production validation."""
    payload = state or {}
    allowed = {
        str(item.get("name") or "")
        for item in [
            *(payload.get("table_infos") or []),
            *(payload.get("source_table_infos") or []),
        ]
        if isinstance(item, dict) and item.get("name")
    }
    if (payload.get("data_source_context") or {}).get("source_mode") != "dw_only":
        return allowed
    for entry in payload.get("chat_history") or []:
        previous_sql = str(entry.get("sql") or "") if isinstance(entry, dict) else ""
        if not previous_sql:
            continue
        try:
            allowed.update(extract_table_names(previous_sql))
        except Exception:  # noqa: BLE001 - malformed history is not authorization
            continue
    return allowed


def _table_scope_safety_evidence(
    sql: str,
    *,
    runtime_state: dict[str, Any] | None = None,
    context: DataAgentContext | dict[str, Any] | None = None,
    fallback_allowed_tables: list[str] | None = None,
) -> dict[str, Any]:
    """Audit SQL scope without reusing Reference-table exactness as authority.

    Live evaluation follows the same non-semantic security boundary as the
    production validator: request-scoped table whitelist, read-only/cartesian
    guards, workspace row/column policy, and upload-asset authorization.  A
    dataset ``allowed_tables`` fallback exists only for isolated static unit
    checks that have no runtime state; live artifacts always identify their
    evidence source as ``runtime_table_infos``.
    """
    candidate = str(sql or "").strip()
    state_supplied = runtime_state is not None
    allowed_tables = (
        _runtime_allowed_tables(runtime_state)
        if state_supplied
        else {
            str(item) for item in (fallback_allowed_tables or []) if str(item)
        }
    )
    evidence: dict[str, Any] = {
        "schema": "table-scope-safety-v2",
        "safe": False,
        "table_scope_safe": False,
        "sql_boundary_safe": False,
        "non_empty": bool(candidate),
        "sql_hash": stable_fingerprint(candidate),
        "authorization_source": (
            "runtime_table_infos"
            if state_supplied
            else "dataset_allowed_tables_fallback"
        ),
        "allowed_table_count": len(allowed_tables),
        "request_scope_fingerprint": stable_fingerprint(
            sorted(name.lower() for name in allowed_tables)
        ),
        "read_only_safe": False,
        "join_relationships_safe": False,
        "join_relationship_status": "not_applicable",
        "request_table_scope_safe": False,
        "workspace_access_policy_safe": False,
        "source_analysis_safe": False,
        "asset_scope_safe": False,
        "asset_context_fingerprint": None,
        "authorized_sql_hash": None,
        "evidence_complete": False,
        "failure_stage": None,
    }
    if not candidate:
        evidence["failure_stage"] = "empty_sql"
        return evidence

    try:
        check_sql_read(candidate)
        evidence["read_only_safe"] = True
    except Exception:  # noqa: BLE001 - persist stable stage, not raw exception
        evidence["failure_stage"] = "read_only"
        return evidence
    try:
        check_join_relationships(candidate)
        evidence["join_relationships_safe"] = True
        evidence["join_relationship_status"] = (
            "connectivity_passed_metadata_unproved"
            if len(extract_table_names(candidate)) > 1
            else "not_applicable"
        )
    except Exception:  # noqa: BLE001
        evidence["failure_stage"] = "join_relationships"
        return evidence
    try:
        check_table_whitelist(candidate, sorted(allowed_tables), reject_empty=True)
        evidence["request_table_scope_safe"] = True
    except Exception:  # noqa: BLE001
        evidence["failure_stage"] = "request_table_scope"
        return evidence

    runtime_context = context or {}
    state = runtime_state or {}
    asset_access = runtime_context.get("asset_access_context")
    if asset_access is not None:
        evidence["asset_context_fingerprint"] = asset_access.public_dict().get(
            "context_fingerprint"
        )
    try:
        if asset_access is not None:
            ensure_context_assets_current(asset_access)
        validate_source_analysis_sql(
            candidate,
            access=asset_access,
            mapping=state.get("source_mapping"),
        )
        evidence["source_analysis_safe"] = True
    except Exception:  # noqa: BLE001
        evidence["failure_stage"] = "source_analysis"
        return evidence
    try:
        physical_sql = translate_logical_upload_tables(candidate, asset_access)
        access_policy = runtime_context.get("access_policy")
        authorized_sql = (
            access_policy.authorize_sql(physical_sql)
            if access_policy is not None else physical_sql
        )
        evidence["authorized_sql_hash"] = stable_fingerprint(authorized_sql)
        evidence["workspace_access_policy_safe"] = True
    except Exception:  # noqa: BLE001
        evidence["failure_stage"] = "workspace_access_policy"
        return evidence
    try:
        authorize_sql_assets(authorized_sql, asset_access)
        evidence["asset_scope_safe"] = True
    except Exception:  # noqa: BLE001
        evidence["failure_stage"] = "asset_scope"
        return evidence

    evidence["table_scope_safe"] = True
    evidence["sql_boundary_safe"] = True
    evidence["evidence_complete"] = bool(
        state_supplied
        and context is not None
        and "access_policy" in runtime_context
        and evidence["sql_hash"]
        and evidence["authorized_sql_hash"]
        and evidence["request_scope_fingerprint"]
    )
    evidence["safe"] = bool(evidence["sql_boundary_safe"])
    return evidence


def _sql_contract_metrics(
    sql: str,
    expected_sql: str,
    case: dict,
    *,
    runtime_state: dict[str, Any] | None = None,
    context: DataAgentContext | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute non-executing SQL quality evidence shared by first/final layers."""
    candidate = str(sql or "").strip()
    read_only_safe = False
    if candidate:
        try:
            check_sql_read(candidate)
            check_join_relationships(candidate)
            read_only_safe = True
        except Exception:  # noqa: BLE001 - exact guard taxonomy is captured upstream
            read_only_safe = False
    use_contract = bool(case.get("required_tables") and case.get("allowed_tables"))
    table_match = (
        tables_match_contract(
            candidate, case.get("required_tables"), case.get("allowed_tables")
        )
        if candidate and use_contract
        else tables_correct(candidate, expected_sql) if candidate else False
    )
    table_metrics = (
        table_contract_prf(
            candidate, case.get("required_tables"), case.get("allowed_tables")
        )
        if candidate and use_contract
        else table_prf(candidate, expected_sql) if candidate else {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
        }
    )
    scope_evidence = _table_scope_safety_evidence(
        candidate,
        runtime_state=runtime_state,
        context=context,
        fallback_allowed_tables=case.get("allowed_tables"),
    )
    return {
        "sql": candidate,
        "non_empty": bool(candidate),
        "syntax_valid": syntax_valid(candidate),
        "read_only_safe": read_only_safe,
        "sql_equivalent": sql_equivalent(candidate, expected_sql) if candidate else False,
        "structure_equivalent": (
            sql_structure_equivalent(candidate, expected_sql) if candidate else False
        ),
        "tables_correct": table_match,
        # Scope safety is independent from exact/required-table recall.  The
        # runtime whitelist and access policy above are the security authority;
        # Reference tables remain diagnostic quality evidence only.
        "table_scope_safe": bool(scope_evidence["safe"]),
        "table_scope_safety_evidence": scope_evidence,
        "table_precision": table_metrics["precision"],
        "table_recall": table_metrics["recall"],
        "table_f1": table_metrics["f1"],
        "guard_passed": False,
        "execution_attempted": False,
        "exec_success": False,
        "result_evaluated": False,
        "result_equivalent": False,
        "result_equivalence_evidence": {},
        "execution_error_type": None,
    }


def semantic_review_case_metrics(
    result: dict,
    sql_observations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build review-only outcome evidence without charging adaptive expansion.

    The first/final result comparison is the source of truth for harmful or
    beneficial changes.  If a later correction superseded review candidate B,
    attribution deliberately stays incomplete instead of guessing.
    """
    stage = result.get("stage_evidence") or {}
    decision = compact_candidate_decision(stage.get("candidate_decision"))
    mode = str(decision.get("decision_mode") or "none")
    first = result.get("first_sql_metrics") or {}
    final = result.get("final_sql_metrics") or {}
    observations = list(
        sql_observations
        if sql_observations is not None
        else stage.get("sql_observations") or []
    )
    review_attempts = [
        item for item in (result.get("llm_attempts") or [])
        if str(item.get("node_id") or "") == "judge_candidate"
    ]

    # A guarded primary is review-eligible even if a malformed/missing decision
    # record hid that fact.  This fail-closed fallback is what makes incomplete
    # coverage observable in governance.
    eligible = bool(
        decision.get("semantic_review_eligible")
        or first.get("guard_passed") is True
    )
    triggered = bool(
        mode == "semantic_review" and decision.get("semantic_review_triggered")
    )
    selected_label = str(decision.get("selected") or "A")
    selected = next(
        (
            item for item in decision.get("candidates") or []
            if str(item.get("label") or "") == selected_label
        ),
        {},
    )
    selected_b = mode == "semantic_review" and selected_label == "B"
    review_observations = [
        item for item in observations if item.get("source") == "judge_candidate"
    ]
    first_sql = str(
        first.get("sql")
        or (observations[0] if observations else {}).get("sql")
        or ""
    )
    reviewed_sql = str((review_observations[-1] if review_observations else {}).get("sql") or "")
    final_sql = str(
        final.get("sql")
        or (observations[-1] if observations else {}).get("sql")
        or ""
    )
    explicit_delta = decision.get("sql_delta")
    if isinstance(explicit_delta, bool):
        sql_delta = explicit_delta
    else:
        comparison_sql = reviewed_sql or final_sql
        same_sql = normalized_sql_equal(first_sql, comparison_sql)
        sql_delta = None if same_sql is None else not same_sql
    modified = bool(selected_b and sql_delta is True)
    final_observation = observations[-1] if observations else {}
    direct_attribution = bool(
        not selected_b
        or not modified
        or final_observation.get("source") == "judge_candidate"
    )
    result_attribution_complete = bool(
        not selected_b
        or (
            direct_attribution
            and first.get("result_evaluated") is True
            and final.get("result_evaluated") is True
        )
    )
    harmful = bool(
        selected_b
        and modified
        and result_attribution_complete
        and first.get("result_equivalent") is True
        and final.get("result_equivalent") is not True
    )
    fixed_result = bool(
        selected_b
        and modified
        and result_attribution_complete
        and first.get("result_equivalent") is not True
        and final.get("result_equivalent") is True
    )
    result_comparison_complete = bool(
        first.get("result_evaluated") is True
        and final.get("result_evaluated") is True
    )
    first_correct_final_incorrect = bool(
        result_comparison_complete
        and first.get("result_equivalent") is True
        and final.get("result_equivalent") is not True
    )
    same_final_sql = normalized_sql_equal(first_sql, final_sql)
    same_sql_execution_loss = bool(
        result_comparison_complete
        and same_final_sql is True
        and first.get("exec_success") is True
        and final.get("exec_success") is not True
    )
    sql_delta_evidence_complete = bool(not selected_b or sql_delta is not None)
    # A selected B must prove a normalized AST delta. Missing evidence is
    # governed fail-closed alongside an explicitly unchanged candidate.
    selected_b_no_sql_delta = bool(
        selected_b and sql_delta is not True
    )
    failed_core = sum(
        str(item.get("status") or "") != "success"
        and bool(item.get("affects_core_path"))
        for item in review_attempts
    )
    budget_skipped = bool(
        mode == "semantic_review"
        and eligible
        and not triggered
        and str(decision.get("trigger_reason") or "")
        == "semantic_review_budget_insufficient"
    )
    attempt_count = len(review_attempts)
    attempt_evidence_complete = bool(
        "llm_attempts" in result
        and result.get("llm_attempt_accounting_complete") is True
    )
    decision_evidence_complete = bool(
        mode == "semantic_review" and decision.get("evidence_complete")
    )
    return {
        "policy_mode": mode,
        "eligible": eligible,
        "triggered": triggered,
        "decision_evidence_complete": decision_evidence_complete,
        "attempt_evidence_complete": attempt_evidence_complete,
        "evidence_complete": bool(
            not eligible
            or (
                decision_evidence_complete
                and attempt_evidence_complete
                and result_attribution_complete
                and sql_delta_evidence_complete
            )
        ),
        "coverage_complete": bool(
            not eligible or (triggered and attempt_count == 1)
        ),
        "attempt_count": attempt_count,
        "at_most_one_attempt": attempt_count <= 1,
        "failed_core": failed_core,
        "budget_skipped": budget_skipped,
        "selected": selected_label,
        "selected_b": selected_b,
        "sql_delta": sql_delta,
        "sql_delta_evidence_complete": sql_delta_evidence_complete,
        "selected_b_no_sql_delta": selected_b_no_sql_delta,
        "unsafe_selected": bool(selected and selected.get("valid") is False),
        "modified": modified,
        "result_attribution_complete": result_attribution_complete,
        "harmful_modification": harmful,
        "fixed_result": fixed_result,
        "first_correct_final_incorrect": first_correct_final_incorrect,
        "same_sql_execution_loss": same_sql_execution_loss,
        "outcome": (
            "not_eligible" if not eligible
            else "budget_skipped" if budget_skipped
            else "not_covered" if not triggered
            else "harmful" if harmful
            else "fixed_result" if fixed_result
            else "reviewed"
        ),
    }


def _llm_coverage(final_state: dict | None, node_metrics: list[dict],
                  attempts: list[dict], failure_stage: str | None = None) -> dict:
    """Derive provider-observation expectations from privacy-safe state."""
    state = final_state or {}
    failed_nodes = {
        str(item.get("node_id") or "")
        for item in node_metrics
        if str(item.get("status") or "").lower() in {"error", "failed", "timeout", "cancelled"}
    }
    if failure_stage:
        failed_nodes.add(str(failure_stage))
    output_nodes: set[str] = set()
    compiler_emitted = bool(
        ((state.get("semantic_ir") or {}).get("compiler") or {}).get("emitted")
    )
    if str(state.get("sql") or "").strip() and not compiler_emitted:
        output_nodes.add("generate_sql")
    planning_evidence = dict(state.get("planning_evidence") or {})
    if (
        str(state.get("plan") or "").strip()
        and planning_evidence.get("physical_llm_call") is True
    ):
        output_nodes.add("plan_sql")
    if (state.get("candidate_decision") or {}).get("extra_call_used"):
        output_nodes.add("judge_candidate")
    if state.get("repair_attempts"):
        output_nodes.add("correct_sql")
    return attempt_coverage(
        attempts,
        failed_nodes=failed_nodes,
        output_nodes=output_nodes,
    )


def _stage_evidence(
    final_state: dict | None,
    sql_attempts: list[str],
    sql_observations: list[dict[str, Any]] | None = None,
) -> dict:
    """Persist compact stage evidence without serializing full vectors or result rows."""
    state = final_state or {}
    recalled_tables = sorted({
        str(item.get("table_id"))
        for item in (state.get("recall_columns") or [])
        if isinstance(item, dict) and item.get("table_id")
    })
    filtered_tables = [
        str(item.get("name"))
        for item in (state.get("table_infos") or [])
        if isinstance(item, dict) and item.get("name")
    ]
    context_columns = sorted({
        f"{table.get('name')}.{column.get('name')}"
        for table in (state.get("table_infos") or []) if isinstance(table, dict) and table.get("name")
        for column in (table.get("columns") or []) if isinstance(column, dict) and column.get("name")
    })
    def compact(items: list, *, kind: str) -> list[dict]:
        output = []
        for item in items:
            if not isinstance(item, dict):
                continue
            identifier = item.get("id") or item.get("table_id") or item.get("name")
            if not identifier:
                continue
            row = {"kind": kind, "id": str(identifier)}
            if item.get("score") is not None:
                row["score"] = round(float(item["score"]), 6)
            for key in ("rank", "index_version", "name", "question", "table_id", "column_id",
                        "rerank_score", "retrieval_sources", "match_features",
                        "entity_id", "dense_rank", "dense_score", "sparse_rank",
                        "sparse_score", "fusion_rank", "fusion_score", "fusion_method",
                        "matched_point_roles"):
                if item.get(key) is not None:
                    row[key] = item.get(key)
            output.append(row)
        return output[:50]

    result_rows = state.get("result") or []
    columns = sorted({str(key) for row in result_rows if isinstance(row, dict) for key in row})
    semantic_ir = state.get("semantic_ir") or {}
    raw_coverage = semantic_ir.get("semantic_coverage") or {}
    semantic_coverage = {
        key: raw_coverage.get(key)
        for key in (
            "schema_version", "registry_version", "question_fingerprint",
            "coverage_score",
        )
        if raw_coverage.get(key) is not None
    }
    for key, limit in (
        ("detected_slots", 40),
        ("represented_slots", 40),
        ("unresolved_slots", 40),
        ("abstain_reasons", 12),
    ):
        semantic_coverage[key] = [
            str(value)[:96]
            for value in (raw_coverage.get(key) or [])[:limit]
        ]
    compact_ir = {
        "schema_version": semantic_ir.get("schema_version"),
        "status": semantic_ir.get("status"),
        "intent": semantic_ir.get("intent"),
        "semantic_family": semantic_ir.get("semantic_family"),
        "dimensions": [
            {key: item.get(key) for key in ("table", "column", "source") if item.get(key) is not None}
            for item in (semantic_ir.get("dimensions") or [])
        ],
        "measures": [
            {key: item.get(key) for key in ("table", "column", "function", "alias", "source") if item.get(key) is not None}
            for item in (semantic_ir.get("measures") or [])
        ],
        "joins": list(semantic_ir.get("joins") or []),
        "order_by": list(semantic_ir.get("order_by") or []),
        "limit": semantic_ir.get("limit"),
        "metric": semantic_ir.get("metric"),
        "confidence": semantic_ir.get("confidence"),
        "compiler": semantic_ir.get("compiler") or {},
        "model_sql_policy": {
            key: value
            for key, value in (
                semantic_ir.get("model_sql_policy") or {}
            ).items()
            if key in {
                "schema_version", "enabled", "decision", "use_model",
                "semantic_family", "compiler_supported",
                "trusted_simple_compiler", "slot_count", "operation_count",
                "dimension_count", "measure_count", "filter_count",
                "reason_codes", "actual_action", "physical_llm_call",
                "fallback_reason", "physical_correction_llm_call",
                "forced_by_runtime_context", "force_reason",
                "deterministic_fallback_allowed",
            }
        },
        "required_tables": list(semantic_ir.get("required_tables") or []),
        "required_column_count": sum(
            len(columns)
            for columns in (semantic_ir.get("required_columns") or {}).values()
        ),
        # Persist only structural slot names and non-sensitive scalar controls.
        # Runtime filter/entity values remain excluded below.
        "semantic_slots": {
            key: value
            for key, value in (semantic_ir.get("semantic_slots") or {}).items()
            if key in {
                "calendar_grain", "comparison_periods", "comparison_mode",
                "window_function", "partition_keys", "order_keys", "rank_type",
                "rank_direction", "top_n", "tie_breaker", "event_selector",
                "cohort_key", "observation_year", "repeat_threshold",
                "aggregate_grain", "subquery_grain", "outer_filter",
                "zero_denominator", "empty_result", "boundary_policy",
                "anti_join_strategy",
            }
        },
        # Filter values and context references are deliberately not persisted.
        "filter_count": len(semantic_ir.get("filters") or []),
        "context_ref_count": len(semantic_ir.get("context_refs") or []),
    } if semantic_ir else {}
    observations = [
        {key: value for key, value in item.items() if key != "authorized_sql"}
        for item in (sql_observations or [])
    ]
    correction_sqls = [
        str(item.get("sql") or "")
        for item in observations[1:]
        if int(item.get("retry_count") or 0) > 0
    ] if observations else (sql_attempts[1:] if len(sql_attempts) > 1 else [])
    return {
        "retrieval": {
            "columns": compact(state.get("recall_columns") or [], kind="column"),
            "metrics": compact(state.get("recall_metrics") or [], kind="metric"),
            "training": compact(state.get("recall_training") or [], kind="training"),
            "values": compact(state.get("recall_values") or [], kind="value"),
        },
        "recalled_tables": recalled_tables,
        "filtered_tables": filtered_tables,
        "context_columns": context_columns,
        "recalled_metrics": [
            str(item.get("name")) for item in (state.get("recall_metrics") or [])
            if isinstance(item, dict) and item.get("name")
        ],
        "filtered_metrics": [
            str(item.get("name")) for item in (state.get("metric_infos") or [])
            if isinstance(item, dict) and item.get("name")
        ],
        "training_examples": [
            str(item.get("question")) for item in (state.get("recall_training") or [])
            if isinstance(item, dict) and item.get("question")
        ],
        "retrieval_budget": list(state.get("retrieval_metrics") or []),
        "context_budget": dict(state.get("context_budget") or {}),
        "candidate_decision": compact_candidate_decision(state.get("candidate_decision")),
        "query_signals": _query_signal_stage_evidence(state),
        "planning_route": _planning_route_stage_evidence(state),
        "semantic_repair": _semantic_repair_stage_evidence(state),
        "plan": str(state.get("plan") or ""),
        "semantic_ir": compact_ir,
        "semantic_coverage": semantic_coverage,
        "request_deadline": compact_deadline_evidence(
            state.get("request_deadline")
        ),
        "sql_attempts": sql_attempts,
        "sql_observations": observations,
        "first_sql": sql_attempts[0] if sql_attempts else "",
        "post_generation_sqls": sql_attempts[1:] if len(sql_attempts) > 1 else [],
        "correction_sqls": correction_sqls,
        "authorized_sql": str(state.get("authorized_sql") or ""),
        "model_sql_fallback": dict(
            (state.get("validation_evidence") or {}).get(
                "model_sql_fallback"
            ) or {}
        ),
        "result_summary": {
            "row_count": len(result_rows),
            "columns": columns,
            "masked": True,
            "values_persisted": False,
        },
        "llm_route_manifest": llm_route_manifest(),
    }


async def run_case(
    case: dict, context: DataAgentContext, dw_repo,
    evaluation_profile: str = "core-default",
) -> dict:
    """
    跑单条用例并计算指标。

    业务逻辑：
        1. 构造 state（query），graph.astream(stream_mode=values) 取最终 state
        2. 计时得到 latency_ms
        3. 从最终 state 取 sql / result / error / retry_count
        4. 计算：syntax_valid / exec_success / sql_equivalent / tables_correct
        5. 用 dw_repo 跑期望 SQL 得 expected_rows，与生成结果比对 result_equivalent
        6. 单用例异常不中断整体，记 error 字段

    Returns:
        dict: 用例 ID + 各指标 + 生成 SQL + 错误信息 + 延迟
    """
    expected_sql = case.get("expected_sql", "")
    question = str(case.get("question") or "")
    result_ordered = question_requires_result_order(question)
    profile = resolve_evaluation_profile(evaluation_profile)
    result: dict[str, Any] = {
        "id": case.get("id"),
        "question": case.get("question"),
        "category": case.get("category", "unknown"),
        "tags": case.get("tags", []),
        "owner": case.get("owner"),
        "source": case.get("source"),
        "review_status": case.get("review_status"),
        "difficulty": case.get("difficulty"),
        "required_tables": case.get("required_tables", []),
        "allowed_tables": case.get("allowed_tables", []),
        "expected_projection": case.get("expected_projection", []),
        "contains_sensitive_data": bool(case.get("contains_sensitive_data")),
        # Effective order semantics come from the user request.  Keep the
        # historical fixture flag separately for audit; it is not authoritative.
        "result_ordered": result_ordered,
        "result_ordered_declared": bool(case.get("result_ordered")),
        "expected_sql": expected_sql,
        "gen_sql": "",
        "error": None,
        "failure_type": None,
        "failure_stage": None,
        "result_evaluated": False,
        "latency_ms": 0,
        "retry_count": 0,
        "syntax_valid": False,
        "exec_success": False,
        "sql_equivalent": False,
        "structure_equivalent": False,
        "tables_correct": False,
        "table_scope_safe": False,
        "table_precision": 0.0,
        "table_recall": 0.0,
        "table_f1": 0.0,
        "result_equivalent": False,
        "result_equivalence_evidence": {},
        "first_sql_metrics": {},
        "correction_metrics": {},
        "semantic_review_metrics": {},
        "final_sql_metrics": {},
        "chart_generated": False,
        "analysis_generated": False,
        "recommend_generated": False,
        "stage_evidence": {},
        "attribution": {},
        "node_metrics": [],
        "degradations": [],
        "tt_sql_ms": 0,
        "tt_result_ms": 0,
        "tt_complete_ms": 0,
        "llm_calls": 0,
        "failed_llm_calls": 0,
        "llm_routes": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost": 0.0,
        "cost_currency": "USD",
        "evaluation_profile": profile["name"],
        "analysis_enabled": bool(profile["analysis_enabled"]),
        "llm_attempts": [],
        "llm_attempt_accounting": {},
        "llm_attempt_coverage": {},
        "llm_attempt_accounting_complete": True,
        "failed_core_llm_calls": 0,
        "failed_optional_llm_calls": 0,
        "degradation_evidence_complete": True,
    }

    start = time.perf_counter()
    collector = QueryTraceCollector(chat_id=f"eval-{case.get('id')}")
    final_state = None
    sql_attempts: list[str] = []
    sql_observations: list[dict[str, Any]] = []
    try:
        # The release runner intentionally reuses one read-only DW session for
        # the whole suite.  A DBAPI failure leaves SQLAlchemy in a pending
        # rollback state, so every case starts from a clean transaction.
        await _rollback_dw_session(dw_repo)
        state = DataAgentState(
            query=case["question"],
            chat_history=case.get("history") or [],
            analysis_enabled=bool(profile["analysis_enabled"]),
            performance_started_at=start,
            request_deadline=initialize_request_deadline(start),
            node_metrics=[],
            degradations=[],
            retrieval_metrics=[],
            llm_attempts=[],
            semantic_repair_ledger=[],
        )
        async for s in graph.astream(
            input=state,
            context=context,
            stream_mode="values",
            config={"callbacks": [collector.callback]},
        ):
            final_state = s
            candidate = str((s or {}).get("sql") or "").strip()
            if candidate and (not sql_attempts or candidate != sql_attempts[-1]):
                sql_attempts.append(candidate)
            _observe_sql_state(sql_observations, s)
        end = time.perf_counter()
        result["latency_ms"] = round((end - start) * 1000, 1)

        gen_sql = (final_state or {}).get("sql") or ""
        err = (final_state or {}).get("error")
        retry_count = (final_state or {}).get("retry_count", 0) or 0
        result["stage_evidence"] = _stage_evidence(
            final_state, sql_attempts, sql_observations,
        )
        result["follow_up"] = dict((final_state or {}).get("follow_up") or {})
        node_metrics = list((final_state or {}).get("node_metrics") or [])
        execute_metrics = [item for item in node_metrics if item.get("node_id") == "execute_sql"]
        collector.record_llm_attempts((final_state or {}).get("llm_attempts") or [])
        for degradation in (final_state or {}).get("degradations") or []:
            normalized = normalize_degradation(degradation)
            if normalized not in collector.degradations:
                collector.degradations.append(normalized)
        usage_summary = collector.finish("success" if not (final_state or {}).get("error") else "failed")
        coverage = _llm_coverage(
            final_state,
            node_metrics,
            usage_summary["llm_attempts"],
        )
        accounting = dict(usage_summary["llm_attempt_accounting"])
        accounting["coverage"] = coverage
        accounting["evidence_complete"] = bool(
            accounting.get("evidence_complete") and coverage["complete"]
        )
        result.update({
            "node_metrics": node_metrics,
            "degradations": usage_summary["degradations"],
            "tt_sql_ms": max((int(item.get("start_ms") or 0) for item in execute_metrics), default=0),
            "tt_result_ms": max((int(item.get("end_ms") or 0) for item in execute_metrics), default=0),
            "tt_complete_ms": round((end - start) * 1000, 1),
            "llm_calls": usage_summary["llm_calls"],
            "failed_llm_calls": usage_summary["failed_llm_calls"],
            "failed_core_llm_calls": usage_summary["failed_core_llm_calls"],
            "failed_optional_llm_calls": usage_summary["failed_optional_llm_calls"],
            "llm_attempts": usage_summary["llm_attempts"],
            "llm_attempt_accounting": accounting,
            "llm_attempt_coverage": coverage,
            "llm_attempt_accounting_complete": bool(
                accounting.get("reconciled") and accounting.get("evidence_complete")
            ),
            "degradation_evidence_complete": usage_summary["degradation_evidence_complete"],
            "llm_routes": usage_summary["usage_by_node"],
            "prompt_tokens": usage_summary["prompt_tokens"],
            "completion_tokens": usage_summary["completion_tokens"],
            "total_tokens": usage_summary["total_tokens"],
            "estimated_cost": usage_summary["estimated_cost"],
            "cost_currency": usage_summary["cost_currency"],
        })
        result["stage_evidence"]["llm_routing"] = usage_summary["usage_by_node"]
        result["stage_evidence"]["llm_attempt_accounting"] = accounting
        result["stage_evidence"]["llm_attempt_coverage"] = coverage
        result["stage_evidence"]["evaluation_profile"] = profile

        result["gen_sql"] = gen_sql
        result["error"] = err
        if err:
            result["failure_type"] = "pipeline"
        result["retry_count"] = retry_count
        final_metrics = _sql_contract_metrics(
            gen_sql,
            expected_sql,
            case,
            runtime_state=final_state,
            context=context,
        )
        final_authorized_sql = str(
            (final_state or {}).get("authorized_sql") or ""
        ).strip()
        final_metrics["guard_passed"] = bool(final_authorized_sql)
        final_metrics["exec_success"] = (
            (err is None) and bool(gen_sql) and bool(final_authorized_sql)
        )
        result.update({
            "syntax_valid": final_metrics["syntax_valid"],
            "exec_success": final_metrics["exec_success"],
            "sql_equivalent": final_metrics["sql_equivalent"],
            "structure_equivalent": final_metrics["structure_equivalent"],
            "tables_correct": final_metrics["tables_correct"],
            "table_scope_safe": final_metrics["table_scope_safe"],
            "table_precision": final_metrics["table_precision"],
            "table_recall": final_metrics["table_recall"],
            "table_f1": final_metrics["table_f1"],
        })
        first_observation = sql_observations[0] if sql_observations else {}
        first_sql = str(
            first_observation.get("sql")
            or (sql_attempts[0] if sql_attempts else "")
        )
        first_metrics = _sql_contract_metrics(
            first_sql,
            expected_sql,
            case,
            runtime_state=final_state,
            context=context,
        )
        first_metrics["guard_passed"] = bool(first_observation.get("guard_passed"))
        first_metrics["guard_error_code"] = first_observation.get("error_code")
        result["chart_generated"] = bool((final_state or {}).get("chart"))
        result["analysis_generated"] = bool((final_state or {}).get("analysis"))
        result["recommend_generated"] = bool((final_state or {}).get("recommend"))

        # 结果一致：重新执行生成 SQL 与期望 SQL，比较完整结果而非浏览器截断前缀。
        # 两侧均受只读、关联连通与评测硬上限约束；超限会使本轮评测无效，
        # 不会把不完整的前 1000 行误记为模型通过或回归。
        try:
            await _rollback_dw_session(dw_repo)
            expected_rows = await _execute_complete_result(dw_repo, expected_sql)
        except Exception as e:
            await _rollback_dw_session(dw_repo)
            logger.warning(f"用例 {case.get('id')} 期望 SQL 执行失败，本轮评测无效：{e}")
            result["result_equivalent"] = False
            result["result_evaluated"] = False
            result["failure_type"] = "infrastructure"
            result["failure_stage"] = "expected_sql"
            result["error"] = f"参考 SQL 执行失败：{type(e).__name__}: {e}"
        else:
            # Generated-query failures are model/pipeline evidence, not a broken
            # reference denominator.  Keep the two transaction boundaries apart.
            result["result_evaluated"] = True
            final_metrics["result_evaluated"] = True
            if result["exec_success"]:
                final_metrics["execution_attempted"] = True
                try:
                    await _rollback_dw_session(dw_repo)
                    generated_rows = await _execute_complete_result(
                        dw_repo, final_authorized_sql,
                    )
                    comparison = result_equivalence_evidence(
                        generated_rows, expected_rows, expected_sql,
                        question=question,
                        generated_sql=final_authorized_sql,
                    )
                    final_metrics["result_equivalence_evidence"] = comparison
                    final_metrics["result_equivalent"] = bool(
                        comparison["equivalent"]
                    )
                except Exception as exc:  # generated SQL is a scored failure
                    await _rollback_dw_session(dw_repo)
                    final_metrics["exec_success"] = False
                    final_metrics["execution_error_type"] = type(exc).__name__
                    result["exec_success"] = False
                    result["failure_type"] = result.get("failure_type") or "pipeline"
                    result["failure_stage"] = result.get("failure_stage") or "final_sql_execution"
                else:
                    await _rollback_dw_session(dw_repo)
            result["result_equivalent"] = bool(final_metrics["result_equivalent"])

            # First SQL may execute only after the graph's normal validation,
            # authorization and cost guards marked that exact revision safe.
            first_metrics["result_evaluated"] = True
            first_authorized_sql = str(
                first_observation.get("authorized_sql") or ""
            ).strip()
            if first_metrics["guard_passed"] and first_authorized_sql:
                first_metrics["execution_attempted"] = True
                if (
                    normalized_sql_equal(first_sql, gen_sql)
                    and normalized_sql_equal(
                        first_authorized_sql, final_authorized_sql,
                    )
                    and final_metrics["execution_attempted"]
                ):
                    first_metrics["exec_success"] = final_metrics["exec_success"]
                    first_metrics["result_equivalent"] = final_metrics[
                        "result_equivalent"
                    ]
                    first_metrics["result_equivalence_evidence"] = final_metrics[
                        "result_equivalence_evidence"
                    ]
                    first_metrics["execution_error_type"] = final_metrics[
                        "execution_error_type"
                    ]
                else:
                    try:
                        await _rollback_dw_session(dw_repo)
                        first_rows = await _execute_complete_result(
                            dw_repo, first_authorized_sql,
                        )
                        first_metrics["exec_success"] = True
                        comparison = result_equivalence_evidence(
                            first_rows, expected_rows, expected_sql,
                            question=question,
                            generated_sql=first_authorized_sql,
                        )
                        first_metrics["result_equivalence_evidence"] = comparison
                        first_metrics["result_equivalent"] = bool(
                            comparison["equivalent"]
                        )
                    except Exception as exc:  # valid first-SQL failure, not infra
                        first_metrics["execution_error_type"] = type(exc).__name__
                    finally:
                        await _rollback_dw_session(dw_repo)

        final_metrics["exec_success"] = bool(result["exec_success"])
        final_metrics["result_evaluated"] = bool(result["result_evaluated"])
        final_metrics["result_equivalent"] = bool(result["result_equivalent"])
        result["result_equivalence_evidence"] = dict(
            final_metrics.get("result_equivalence_evidence") or {}
        )
        result["first_sql_metrics"] = first_metrics
        result["final_sql_metrics"] = final_metrics
        correction_attempts = [
            item for item in (result.get("llm_attempts") or [])
            if str(item.get("node_id") or "") == "correct_sql"
        ]
        correction_triggered = bool(retry_count or correction_attempts)
        corrected_observations = [
            item for item in sql_observations
            if item.get("source") == "correct_sql"
        ]
        final_observation = sql_observations[-1] if sql_observations else {}
        correction_attribution_valid = bool(
            correction_triggered
            and corrected_observations
            and final_observation.get("source") == "correct_sql"
            and final_observation.get("sql") == corrected_observations[-1].get("sql")
        )
        fixed_execution = bool(
            correction_attribution_valid
            and not first_metrics["exec_success"]
            and final_metrics["exec_success"]
        )
        fixed_result = bool(
            correction_attribution_valid
            and not first_metrics["result_equivalent"]
            and final_metrics["result_equivalent"]
        )
        harmful = bool(
            correction_attribution_valid
            and first_metrics["result_equivalent"]
            and not final_metrics["result_equivalent"]
        )
        result["correction_metrics"] = {
            "triggered": correction_triggered,
            "attribution_valid": correction_attribution_valid,
            "attribution_status": (
                "not_triggered" if not correction_triggered
                else "direct" if correction_attribution_valid
                else "downstream_sql_revision_present"
            ),
            "trigger_reason_codes": sorted({
                str(item.get("error_code")) for item in sql_observations
                if item.get("error_code")
            }),
            "retry_count": int(retry_count),
            "correction_output_count": len(corrected_observations),
            "sql_revision_count": len(
                result["stage_evidence"].get("correction_sqls") or []
            ),
            "llm_attempts": len(correction_attempts),
            "failed_llm_attempts": sum(
                str(item.get("status") or "") != "success"
                for item in correction_attempts
            ),
            "additional_tokens": sum(
                int(item.get("total_tokens") or 0) for item in correction_attempts
            ),
            "additional_tokens_are_lower_bound": any(
                not bool(item.get("token_known")) for item in correction_attempts
            ),
            "unknown_token_attempts": sum(
                not bool(item.get("token_known")) for item in correction_attempts
            ),
            "additional_latency_ms": sum(
                int(item.get("latency_ms") or 0) for item in correction_attempts
            ),
            "fixed_execution": fixed_execution,
            "fixed_result": fixed_result,
            "harmful_modification": harmful,
            "outcome": (
                "not_triggered" if not correction_triggered
                else "not_attributable" if not correction_attribution_valid
                else "harmful" if harmful
                else "fixed_result" if fixed_result
                else "fixed_execution" if fixed_execution
                else "no_strict_gain"
            ),
        }
    except Exception as e:
        await _rollback_dw_session(dw_repo)
        end = time.perf_counter()
        collector.fail(f"{type(e).__name__}: {e}")
        collector.record_llm_attempts((final_state or {}).get("llm_attempts") or [])
        collector.record_llm_attempts(getattr(e, "llm_attempts", None) or [])
        for degradation in (final_state or {}).get("degradations") or []:
            normalized = normalize_degradation(degradation)
            if normalized not in collector.degradations:
                collector.degradations.append(normalized)
        for degradation in getattr(e, "degradations", None) or []:
            normalized = normalize_degradation(degradation)
            if normalized not in collector.degradations:
                collector.degradations.append(normalized)
        usage_summary = collector.finish("failed")
        failure_stage = str(getattr(e, "node_id", "") or "evaluation")
        failed_node_metrics = [{
            "node_id": step.name,
            "duration_ms": step.duration_ms,
            "status": step.status,
        } for step in collector.steps]
        coverage = _llm_coverage(
            final_state,
            failed_node_metrics,
            usage_summary["llm_attempts"],
            failure_stage,
        )
        accounting = dict(usage_summary["llm_attempt_accounting"])
        accounting["coverage"] = coverage
        accounting["evidence_complete"] = bool(
            accounting.get("evidence_complete") and coverage["complete"]
        )
        result["latency_ms"] = round((end - start) * 1000, 1)
        error_text = str(e).strip() or type(e).__name__ or "UnknownError"
        result["error"] = f"运行异常：{error_text}"
        result["failure_type"] = "pipeline" if failure_stage != "evaluation" else "infrastructure"
        result["failure_stage"] = failure_stage
        result["stage_evidence"] = _stage_evidence(
            final_state, sql_attempts, sql_observations,
        )
        result.update({
            "node_metrics": failed_node_metrics,
            "tt_complete_ms": round((end - start) * 1000, 1),
            "llm_calls": usage_summary["llm_calls"],
            "failed_llm_calls": usage_summary["failed_llm_calls"],
            "failed_core_llm_calls": usage_summary["failed_core_llm_calls"],
            "failed_optional_llm_calls": usage_summary["failed_optional_llm_calls"],
            "llm_attempts": usage_summary["llm_attempts"],
            "llm_attempt_accounting": accounting,
            "llm_attempt_coverage": coverage,
            "llm_attempt_accounting_complete": bool(
                accounting.get("reconciled") and accounting.get("evidence_complete")
            ),
            "degradations": usage_summary["degradations"],
            "degradation_evidence_complete": usage_summary["degradation_evidence_complete"],
            "llm_routes": usage_summary["usage_by_node"],
            "prompt_tokens": usage_summary["prompt_tokens"],
            "completion_tokens": usage_summary["completion_tokens"],
            "total_tokens": usage_summary["total_tokens"],
            "estimated_cost": usage_summary["estimated_cost"],
            "cost_currency": usage_summary["cost_currency"],
        })
        result["stage_evidence"]["llm_routing"] = usage_summary["usage_by_node"]
        result["stage_evidence"]["llm_attempt_accounting"] = accounting
        result["stage_evidence"]["llm_attempt_coverage"] = coverage
        result["stage_evidence"]["evaluation_profile"] = profile
        first_observation = sql_observations[0] if sql_observations else {}
        first_sql = str(first_observation.get("sql") or "")
        final_sql = str(
            (sql_observations[-1] if sql_observations else {}).get("sql") or ""
        )
        result["gen_sql"] = final_sql
        first_metrics = _sql_contract_metrics(
            first_sql,
            expected_sql,
            case,
            runtime_state=final_state,
            context=context,
        )
        first_metrics["guard_passed"] = bool(first_observation.get("guard_passed"))
        final_metrics = _sql_contract_metrics(
            final_sql,
            expected_sql,
            case,
            runtime_state=final_state,
            context=context,
        )
        final_metrics["guard_passed"] = bool(
            (sql_observations[-1] if sql_observations else {}).get("guard_passed")
        )
        executor = getattr(dw_repo, "execute_sql_for_evaluation", None)
        if callable(executor):
            try:
                await _rollback_dw_session(dw_repo)
                expected_rows = await _execute_complete_result(dw_repo, expected_sql)
            except Exception as reference_exc:
                await _rollback_dw_session(dw_repo)
                result["failure_type"] = "infrastructure"
                result["failure_stage"] = "expected_sql"
                result["error"] = (
                    "参考 SQL 执行失败："
                    f"{type(reference_exc).__name__}: {reference_exc}"
                )
            else:
                result["result_evaluated"] = True
                first_metrics["result_evaluated"] = True
                final_metrics["result_evaluated"] = True
                first_authorized = str(
                    first_observation.get("authorized_sql") or ""
                ).strip()
                if first_metrics["guard_passed"] and first_authorized:
                    first_metrics["execution_attempted"] = True
                    try:
                        first_rows = await _execute_complete_result(
                            dw_repo, first_authorized,
                        )
                        first_metrics["exec_success"] = True
                        comparison = result_equivalence_evidence(
                            first_rows, expected_rows, expected_sql,
                            question=question,
                            generated_sql=first_authorized,
                        )
                        first_metrics["result_equivalence_evidence"] = comparison
                        first_metrics["result_equivalent"] = bool(
                            comparison["equivalent"]
                        )
                    except Exception as first_exc:
                        first_metrics["execution_error_type"] = type(first_exc).__name__
                    finally:
                        await _rollback_dw_session(dw_repo)
        result["first_sql_metrics"] = first_metrics
        result["final_sql_metrics"] = final_metrics
        correction_attempts = [
            item for item in (result.get("llm_attempts") or [])
            if str(item.get("node_id") or "") == "correct_sql"
        ]
        correction_triggered = bool(correction_attempts) or any(
            int(item.get("retry_count") or 0) > 0 for item in sql_observations
        )
        result["correction_metrics"] = {
            "triggered": correction_triggered,
            "attribution_valid": False,
            "attribution_status": (
                "pipeline_failed_before_final_attribution"
                if correction_triggered else "not_triggered"
            ),
            "trigger_reason_codes": sorted({
                str(item.get("error_code")) for item in sql_observations
                if item.get("error_code")
            }),
            "retry_count": max(
                (int(item.get("retry_count") or 0) for item in sql_observations),
                default=0,
            ),
            "correction_output_count": sum(
                item.get("source") == "correct_sql" for item in sql_observations
            ),
            "sql_revision_count": len(
                result["stage_evidence"].get("correction_sqls") or []
            ),
            "llm_attempts": len(correction_attempts),
            "failed_llm_attempts": sum(
                str(item.get("status") or "") != "success"
                for item in correction_attempts
            ),
            "additional_tokens": sum(
                int(item.get("total_tokens") or 0) for item in correction_attempts
            ),
            "additional_tokens_are_lower_bound": any(
                not bool(item.get("token_known")) for item in correction_attempts
            ),
            "unknown_token_attempts": sum(
                not bool(item.get("token_known")) for item in correction_attempts
            ),
            "additional_latency_ms": sum(
                int(item.get("latency_ms") or 0) for item in correction_attempts
            ),
            "fixed_execution": False,
            "fixed_result": False,
            "harmful_modification": False,
            "outcome": "pipeline_failed" if correction_triggered else "not_triggered",
        }
        logger.error(f"用例 {case.get('id')} 运行异常：{e}")
    fatal_provider_attempt = next((
        attempt
        for attempt in (result.get("llm_attempts") or [])
        if str(attempt.get("status") or "") != "success"
        and str(attempt.get("reason_code") or "")
        in FATAL_PROVIDER_REASON_CODES
    ), None)
    if fatal_provider_attempt:
        reason_code = str(fatal_provider_attempt.get("reason_code") or "")
        # A provider account/auth outage is evaluation infrastructure evidence,
        # never a model-capability miss.  Keep the message typed and body-free.
        result["failure_type"] = "infrastructure"
        result["failure_stage"] = "llm_provider"
        result["error"] = f"模型服务基础设施失败（{reason_code}）"
        if not result.get("exec_success"):
            result["result_evaluated"] = False
            for key in ("first_sql_metrics", "final_sql_metrics"):
                metrics = result.get(key) or {}
                metrics["result_evaluated"] = False
    result["semantic_review_metrics"] = semantic_review_case_metrics(
        result, sql_observations,
    )
    result["attribution"] = attribute_case(result, case, result.get("stage_evidence"))
    return result


async def _rollback_dw_session(dw_repo) -> None:
    """Best-effort transaction reset for the shared read-only evaluation session."""

    session = getattr(dw_repo, "session", None)
    rollback = getattr(session, "rollback", None)
    if not callable(rollback):
        return
    try:
        await rollback()
    except Exception as exc:  # pragma: no cover - cleanup must not hide root cause
        logger.warning(f"评测 DW 会话回滚失败：{type(exc).__name__}: {exc}")
