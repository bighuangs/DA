"""Trace 与 SSE 共用的隐私安全审计载荷规范化函数。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


CANDIDATE_DECISION_SCHEMA = "candidate-decision-v4"
CONFIDENCE_VERSION = "candidate-confidence-v4"
DECISION_MODES = frozenset({"none", "adaptive_expansion", "semantic_review"})
PLANNING_ROUTE_CONTRACT_VERSION = "planning-route-v1"
REPAIR_RESULT_CONTRACT = "semantic-repair-result-v1"
PLANNING_FEATURE_FIELDS = frozenset({
    "ambiguous_join_group_count", "clause_count", "column_count",
    "conjunction_count", "dimension_table_count", "entity_signal_count",
    "fact_table_count", "has_aggregation", "has_anti_join",
    "has_comparison", "has_distinct", "has_having", "has_multi_condition",
    "has_multi_measure", "has_ranking", "has_ratio", "has_trend",
    "has_window", "identifier_signal_count", "inherited_contract_count",
    "ir_dimension_count", "ir_intent", "ir_join_count", "ir_measure_count",
    "ir_reason_code", "ir_status", "is_follow_up", "join_key_group_count",
    "legacy_keyword_count", "metric_count", "numeric_signal_count",
    "operator_signal_count", "planning_timeout_bucket", "polarity_signal_count",
    "query_length_bucket", "query_length_capped", "table_count",
    "time_signal_count", "feature_schema",
})


def _items(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def compact_semantic_evidence(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """按白名单压缩语义违规或修复证明。"""

    if not isinstance(value, Mapping):
        return {}
    output: dict[str, Any] = {}
    scalar_keys = {
        "expected_count", "actual_count", "candidate_edge_count",
        "expected_limit", "actual_limit", "direction", "relation_type",
        "anti_join_strategy", "history_contract", "compiler_supported",
        "table_set_unchanged", "filter_literals_unchanged",
        "read_only_preserved", "idempotent",
    }
    list_keys = {
        "expected_columns", "actual_columns", "missing_columns",
        "expected_tables", "missing_tables", "changed_ast_nodes",
    }
    hash_keys = {
        "metadata_proof_hash", "ir_proof_hash", "aggregate_ast_hash",
        "sql_hash", "previous_sql_hash",
    }
    for key in scalar_keys:
        if key in value and isinstance(
            value[key], (str, int, float, bool, type(None))
        ):
            output[key] = value[key]
    for key in list_keys:
        if key in value:
            raw = value[key]
            output[key] = (
                [str(item)[:80] for item in list(raw)[:12]]
                if isinstance(raw, (list, tuple, set))
                else []
            )
    for key in hash_keys:
        if key in value and value[key] is not None:
            output[key] = str(value[key])[:80]
    return output


def _candidate_decision_mode(value: Mapping | None) -> str:
    item = dict(value or {})
    explicit = str(item.get("decision_mode") or "").strip().lower()
    if explicit in DECISION_MODES:
        return explicit
    reasons = (
        item.get("trigger_reason"),
        item.get("selection_reason"),
        item.get("candidate_outcome"),
    )
    if bool(item.get("semantic_review_triggered")) or any(
        str(reason or "").startswith("semantic_review_") for reason in reasons
    ):
        return "semantic_review"
    if bool(item.get("adaptive_triggered")):
        return "adaptive_expansion"
    if str(item.get("trigger_reason") or "").endswith("_confidence_expand"):
        return "adaptive_expansion"
    return "none"


def compact_candidate_decision(value: Mapping | None) -> dict:
    """移除候选 SQL 与结果，只保留有界的决策证据。"""

    item = dict(value or {})
    decision_mode = _candidate_decision_mode(item)
    explicit_mode = str(item.get("decision_mode") or "").strip().lower()
    mode_inferred = explicit_mode not in DECISION_MODES and decision_mode != "none"
    candidates = []
    for candidate in item.get("candidates") or []:
        candidates.append({
            "label": candidate.get("label"),
            "valid": bool(candidate.get("valid")),
            "score": candidate.get("score"),
            "band": candidate.get("band"),
            "weaknesses": list(candidate.get("weaknesses") or [])[:10],
            "actionable_weaknesses": list(
                candidate.get("actionable_weaknesses") or []
            )[:10],
            "error_code": (candidate.get("validation") or {}).get("error_code"),
            "sql_features": dict(candidate.get("sql_features") or {}),
            "query_contract_audit": {
                "schema_version": (
                    candidate.get("query_contract_audit") or {}
                ).get("schema_version"),
                "valid": bool(
                    (candidate.get("query_contract_audit") or {}).get("valid")
                ),
                "violation_codes": [
                    str(row.get("code") or "unknown")
                    for row in (
                        (candidate.get("query_contract_audit") or {}).get("violations")
                        or []
                    )
                ][:8],
                "contract_hash": (
                    candidate.get("query_contract_audit") or {}
                ).get("contract_hash"),
            },
        })
    legacy_semantic_trigger = bool(
        decision_mode == "semantic_review"
        and (
            item.get("adaptive_triggered")
            or item.get("extra_call_used")
            or str(item.get("trigger_reason") or "")
            == "semantic_review_required"
        )
    )
    semantic_review_triggered = bool(
        decision_mode == "semantic_review"
        and (
            item.get("semantic_review_triggered")
            if "semantic_review_triggered" in item
            else legacy_semantic_trigger
        )
    )
    adaptive_triggered = bool(
        decision_mode == "adaptive_expansion" and item.get("adaptive_triggered")
    )
    primary = next(
        (candidate for candidate in candidates if candidate.get("label") == "A"),
        {},
    )
    semantic_review_eligible = bool(
        item.get("semantic_review_eligible")
        if "semantic_review_eligible" in item
        else decision_mode == "semantic_review" and primary.get("valid") is True
    )
    any_triggered = adaptive_triggered or semantic_review_triggered
    evidence_complete = bool(
        item
        and candidates
        and item.get("trigger_reason") is not None
        and (
            explicit_mode in DECISION_MODES
            or mode_inferred
            or decision_mode == "none"
        )
    )
    return {
        "schema": item.get("schema") or CANDIDATE_DECISION_SCHEMA,
        "version": item.get("version") or CONFIDENCE_VERSION,
        "decision_mode": decision_mode,
        "decision_mode_inferred": mode_inferred,
        "evidence_complete": evidence_complete,
        "adaptive_triggered": adaptive_triggered,
        "semantic_review_triggered": semantic_review_triggered,
        "semantic_review_eligible": semantic_review_eligible,
        "trigger_reason": item.get("trigger_reason"),
        "selected": item.get("selected") or "A",
        "selection_reason": item.get("selection_reason"),
        "sql_delta": (
            item.get("sql_delta")
            if isinstance(item.get("sql_delta"), bool)
            else None
        ),
        "candidate_count": int(
            item.get("candidate_count") or len(candidates) or 1
        ),
        "extra_call_used": bool(item.get("extra_call_used")),
        "generation_ms": int(item.get("generation_ms") or 0),
        "validation_ms": int(item.get("validation_ms") or 0),
        "candidate_outcome": item.get("candidate_outcome") or (
            "not_triggered" if not any_triggered else "unknown"
        ),
        "candidates": candidates,
    }


def compact_planning_route(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """压缩规划路由决策，拒绝原始问题与文本特征。"""

    if not isinstance(value, Mapping):
        return {}
    feature = value.get("feature_summary")
    safe_feature = {}
    if isinstance(feature, Mapping):
        safe_feature = {
            str(key): item
            for key, item in feature.items()
            if key in PLANNING_FEATURE_FIELDS
            and (isinstance(item, (str, int, float, bool)) or item is None)
        }
    shadows = []
    for row in (value.get("shadow_decisions") or [])[:8]:
        if not isinstance(row, Mapping):
            continue
        shadows.append({
            "profile": str(row.get("profile") or "")[:64],
            "decision": str(row.get("decision") or "")[:32],
            "score": round(float(row.get("score") or 0), 4),
            "confidence": round(float(row.get("confidence") or 0), 4),
            "reason_codes": [
                str(item)[:80] for item in (row.get("reason_codes") or [])[:12]
            ],
        })
    return {
        "contract_version": str(
            value.get("contract_version") or PLANNING_ROUTE_CONTRACT_VERSION
        )[:48],
        "profile": str(value.get("profile") or "")[:64],
        "decision": str(value.get("decision") or "")[:32],
        "actual_action": str(value.get("actual_action") or "")[:32],
        "score": round(float(value.get("score") or 0), 4),
        "confidence": round(float(value.get("confidence") or 0), 4),
        "reason_codes": [
            str(item)[:80] for item in (value.get("reason_codes") or [])[:16]
        ],
        "feature_schema": str(value.get("feature_schema") or "")[:64],
        "feature_fingerprint": str(value.get("feature_fingerprint") or "")[:80],
        "feature_summary": safe_feature,
        "semantic_ir_supported": bool(value.get("semantic_ir_supported")),
        "excluded_from_router_denominator": bool(
            value.get("excluded_from_router_denominator")
        ),
        "budget_snapshot": {
            key: (value.get("budget_snapshot") or {}).get(key)
            for key in (
                "remaining_deadline_ms",
                "planning_timeout_ms",
                "planning_timeout_bucket",
            )
        },
        "decision_ms": round(float(value.get("decision_ms") or 0), 4),
        "fallback_profile": str(value.get("fallback_profile") or "")[:64],
        "fallback_applied": bool(value.get("fallback_applied")),
        "router_degraded": bool(value.get("router_degraded")),
        "cache_namespace": str(value.get("cache_namespace") or "")[:80],
        "repair_profile": str(value.get("repair_profile") or "")[:80],
        "shadow_decisions": shadows,
        "raw_text_logged": False,
    }


def compact_semantic_repair(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """压缩版本化语义修复证据，并拒绝未知合同。"""

    item = dict(value or {})
    if item.get("contract_version") != REPAIR_RESULT_CONTRACT:
        return {}
    rule_keys = {
        "rule_code", "rule_version", "status", "proof",
        "changed_ast_nodes", "reason",
    }

    def rules(key: str) -> list[dict[str, Any]]:
        output = []
        for row in item.get(key) or []:
            if not isinstance(row, Mapping):
                continue
            safe = {
                name: row.get(name)
                for name in rule_keys
                if row.get(name) is not None
            }
            safe["proof"] = compact_semantic_evidence(row.get("proof"))
            output.append(safe)
        return output[:16]

    return {
        "contract_version": REPAIR_RESULT_CONTRACT,
        "registry_version": str(item.get("registry_version") or "")[:80],
        "rule_set_version": str(item.get("rule_set_version") or "")[:80],
        "profile": str(item.get("profile") or "")[:80],
        "changed": bool(item.get("changed")),
        "original_sql_hash": str(item.get("original_sql_hash") or "")[:80],
        "repaired_sql_hash": str(item.get("repaired_sql_hash") or "")[:80],
        "applied_rules": rules("applied_rules"),
        "skipped_rules": rules("skipped_rules"),
        "shadow_rules": rules("shadow_rules"),
        "attempted_rule_codes": [
            str(code)[:80]
            for code in (item.get("attempted_rule_codes") or [])[:16]
        ],
        "repair_ms": float(item.get("repair_ms") or 0),
        "fallback_used": bool(item.get("fallback_used")),
        "fallback_reason": (
            str(item.get("fallback_reason"))[:80]
            if item.get("fallback_reason")
            else None
        ),
        "invariants": {
            str(key)[:80]: bool(raw)
            for key, raw in dict(item.get("invariants") or {}).items()
        },
        "raw_sql_logged": False,
    }


def compact_relation_closure(
    closure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """压缩有界的元数据关系闭包证明。"""

    value = dict(closure or {})

    def names(items: Any, *, limit: int = 16) -> list[str]:
        return [str(item)[:128] for item in list(items or [])[:limit]]

    required_columns = {
        str(table)[:128]: names(columns, limit=24)
        for table, columns in list(
            dict(value.get("required_columns") or {}).items()
        )[:16]
    }
    added_columns = {
        str(table)[:128]: names(columns, limit=16)
        for table, columns in list(
            dict(value.get("added_columns") or {}).items()
        )[:16]
    }
    joins = []
    for raw in list(value.get("joins") or [])[:16]:
        if not isinstance(raw, Mapping):
            continue
        joins.append({
            key: raw.get(key)
            for key in (
                "left_table", "left_column", "right_table", "right_column",
                "relationship", "evidence", "confidence",
            )
            if raw.get(key) is not None
        })
    ambiguous = []
    for raw in list(value.get("ambiguous_paths") or [])[:8]:
        if not isinstance(raw, Mapping):
            continue
        ambiguous.append({
            key: raw.get(key)
            for key in ("target_table", "candidate_path_count", "hop_count")
            if raw.get(key) is not None
        })
    return {
        "schema_version": value.get("schema_version"),
        "status": value.get("status"),
        "required_columns": required_columns,
        "terminal_tables": names(value.get("terminal_tables")),
        "added_tables": names(value.get("added_tables")),
        "added_columns": added_columns,
        "joins": joins,
        "unresolved_tables": names(value.get("unresolved_tables")),
        "ambiguous_paths": ambiguous,
        "edge_count": int(value.get("edge_count") or 0),
        "raw_question_persisted": False,
        "filter_values_persisted": False,
    }


def compact_query_contract(
    contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """返回语义查询合同中适合 Trace 的白名单子集。"""

    value = dict(contract or {})
    relation = dict(value.get("relation_plan") or {})
    return {
        "schema_version": value.get("schema_version"),
        "semantic_family": value.get("semantic_family"),
        "authority": value.get("authority"),
        "confidence": value.get("confidence"),
        "contract_hash": value.get("contract_hash"),
        "output_grain": _items(value.get("output_grain"))[:12],
        "metric_contracts": _items(value.get("metric_contracts"))[:12],
        "filter_contract": dict(value.get("filter_contract") or {}),
        "operator_contract": dict(value.get("operator_contract") or {}),
        "operations": [
            str(item) for item in _items(value.get("operations"))[:16]
        ],
        "stage_plan": _items(value.get("stage_plan"))[:10],
        "relation_plan": compact_relation_closure(relation),
        "unresolved_slots": [
            str(item)[:120]
            for item in _items(value.get("unresolved_slots"))[:16]
        ],
        "raw_question_persisted": False,
        "filter_values_persisted": False,
    }


__all__ = [
    "compact_candidate_decision",
    "compact_planning_route",
    "compact_query_contract",
    "compact_relation_closure",
    "compact_semantic_evidence",
    "compact_semantic_repair",
]
