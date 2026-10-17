"""Versioned, privacy-safe SQL planning-route contracts.

Request-local feature extraction may inspect the standalone question, but the
serialised contract deliberately contains only bounded counts, booleans,
enumerations and fingerprints.  Neither the question nor entity literals are
written to Trace, logs or evaluation reports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from app.agent_system.agents.nl2sql.contracts.query_signals import stable_signal_hash


PLANNING_ROUTE_CONTRACT_VERSION = "planning-route-v1"
PLANNING_FEATURE_SCHEMA = "planning-features-v1"
PLANNING_ACTIONS = {
    "direct_generate",
    "plan_then_generate",
    "abstain",
}
ACTUAL_ACTIONS = {
    "direct_generate",
    "plan_then_generate",
    "deterministic_ir_bypass",
}


@dataclass(frozen=True)
class PlanningRouteFeaturesV1:
    """Bounded planning features grouped for nested A0→A4 ablations."""

    query_length_bucket: str
    query_length_capped: int
    clause_count: int
    legacy_keyword_count: int
    conjunction_count: int

    time_signal_count: int = 0
    numeric_signal_count: int = 0
    operator_signal_count: int = 0
    polarity_signal_count: int = 0
    identifier_signal_count: int = 0
    entity_signal_count: int = 0

    ir_status: str = "unsupported"
    ir_intent: str = "unknown"
    ir_reason_code: str = "unknown"
    ir_dimension_count: int = 0
    ir_measure_count: int = 0
    ir_join_count: int = 0

    table_count: int = 0
    fact_table_count: int = 0
    dimension_table_count: int = 0
    column_count: int = 0
    metric_count: int = 0
    join_key_group_count: int = 0
    ambiguous_join_group_count: int = 0

    has_aggregation: bool = False
    has_ratio: bool = False
    has_comparison: bool = False
    has_trend: bool = False
    has_having: bool = False
    has_anti_join: bool = False
    has_distinct: bool = False
    has_ranking: bool = False
    has_window: bool = False
    has_multi_measure: bool = False
    has_multi_condition: bool = False

    is_follow_up: bool = False
    inherited_contract_count: int = 0
    planning_timeout_bucket: str = "under_10s"

    def safe_payload(self) -> dict[str, Any]:
        return {
            "feature_schema": PLANNING_FEATURE_SCHEMA,
            **asdict(self),
        }

    @property
    def fingerprint(self) -> str:
        return stable_signal_hash(self.safe_payload())


@dataclass(frozen=True)
class PlanningRouteDecisionV1:
    """One actual route plus bounded shadow decisions and rollback evidence."""

    profile: str
    decision: str
    actual_action: str
    score: float
    confidence: float
    reason_codes: tuple[str, ...]
    features: PlanningRouteFeaturesV1
    semantic_ir_supported: bool
    budget_snapshot: Mapping[str, Any]
    decision_ms: float
    fallback_profile: str
    fallback_applied: bool = False
    router_degraded: bool = False
    cache_namespace: str = ""
    repair_profile: str = "semantic-repair-inline-v1"
    shadow_decisions: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    contract_version: str = PLANNING_ROUTE_CONTRACT_VERSION
    feature_schema: str = PLANNING_FEATURE_SCHEMA

    def __post_init__(self) -> None:
        if self.decision not in PLANNING_ACTIONS:
            raise ValueError(f"无效规划决策：{self.decision}")
        if self.actual_action not in ACTUAL_ACTIONS:
            raise ValueError(f"无效实际规划动作：{self.actual_action}")

    def privacy_summary(self) -> dict[str, Any]:
        """Return the only representation allowed in Trace/SSE/persistence."""

        return {
            "contract_version": self.contract_version,
            "profile": self.profile,
            "decision": self.decision,
            "actual_action": self.actual_action,
            "score": round(max(0.0, min(1.0, float(self.score))), 4),
            "confidence": round(
                max(0.0, min(1.0, float(self.confidence))), 4
            ),
            "reason_codes": list(self.reason_codes)[:16],
            "feature_schema": self.feature_schema,
            "feature_fingerprint": self.features.fingerprint,
            "feature_summary": self.features.safe_payload(),
            "semantic_ir_supported": bool(self.semantic_ir_supported),
            "excluded_from_router_denominator": bool(
                self.semantic_ir_supported
            ),
            "budget_snapshot": {
                key: self.budget_snapshot.get(key)
                for key in (
                    "remaining_deadline_ms",
                    "planning_timeout_ms",
                    "planning_timeout_bucket",
                )
            },
            "decision_ms": round(max(0.0, float(self.decision_ms)), 4),
            "fallback_profile": self.fallback_profile,
            "fallback_applied": bool(self.fallback_applied),
            "router_degraded": bool(self.router_degraded),
            "cache_namespace": self.cache_namespace,
            "repair_profile": self.repair_profile,
            "shadow_decisions": [
                {
                    key: row.get(key)
                    for key in (
                        "profile",
                        "decision",
                        "score",
                        "confidence",
                        "reason_codes",
                    )
                }
                for row in self.shadow_decisions[:8]
            ],
            "raw_text_logged": False,
        }


def compact_planning_route(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Whitelist an already serialised route before persistence or API output."""

    if not isinstance(value, Mapping):
        return {}
    feature = value.get("feature_summary")
    safe_feature = {}
    feature_fields = {
        *PlanningRouteFeaturesV1.__dataclass_fields__,
        "feature_schema",
    }
    if isinstance(feature, Mapping):
        safe_feature = {
            str(key): item
            for key, item in feature.items()
            if key in feature_fields
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
                str(item)[:80]
                for item in (row.get("reason_codes") or [])[:12]
            ],
        })
    return {
        "contract_version": str(
            value.get("contract_version")
            or PLANNING_ROUTE_CONTRACT_VERSION
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
        "feature_fingerprint": str(
            value.get("feature_fingerprint") or ""
        )[:80],
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
