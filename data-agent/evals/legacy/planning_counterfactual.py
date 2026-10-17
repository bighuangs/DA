"""Reviewed Direct/Plan counterfactual contract for TODO-63.

The reviewed YAML is an evaluation asset and may contain synthetic business
questions.  Returned reports expose only case IDs, families, strata, labels and
aggregate metrics; raw questions never enter runtime Trace or generated reports.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

import yaml

from app.agent_system.agents.nl2sql.contracts.query_signals import stable_signal_hash
from evals.paths import LEGACY_DATASETS_ROOT


CONTRACT_PATH = LEGACY_DATASETS_ROOT / "planning_cases.yaml"
COMPARABLE_LABELS = {
    "plan_helpful",
    "plan_neutral",
    "plan_harmful",
}
ALL_LABELS = {
    *COMPARABLE_LABELS,
    "both_failed",
    "inconclusive",
    "ir_bypass",
}


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            isinstance(value, Mapping)
            and isinstance(result.get(key), Mapping)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_planning_contract() -> dict[str, Any]:
    payload = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8")) or {}
    defaults = dict(payload.get("defaults") or {})
    cases = []
    for raw in payload.get("cases") or []:
        case = dict(raw)
        case["direct"] = _deep_merge(
            defaults.get("direct") or {}, case.get("direct") or {}
        )
        case["plan"] = _deep_merge(
            defaults.get("plan") or {}, case.get("plan") or {}
        )
        case["context"] = _deep_merge(
            defaults.get("context") or {}, case.get("context") or {}
        )
        cases.append(case)
    return {
        **payload,
        "cases": cases,
    }


def _quality_tier(outcome: Mapping[str, Any]) -> tuple[int, ...]:
    return (
        int(bool(outcome.get("safety_pass"))),
        int(bool(outcome.get("parse_pass"))),
        int(bool(outcome.get("execution_pass"))),
        int(bool(outcome.get("result_equivalent"))),
        int(bool(outcome.get("semantic_pass"))),
        -int(outcome.get("correction_retries") or 0),
    )


def label_counterfactual(case: Mapping[str, Any]) -> dict[str, Any]:
    """Apply quality-first ordering; cost only breaks true quality ties."""

    if bool(case.get("semantic_ir_supported")):
        return {
            "label": "ir_bypass",
            "comparable": False,
            "reason_code": "deterministic_ir_supported",
        }
    direct = dict(case.get("direct") or {})
    plan = dict(case.get("plan") or {})
    if direct.get("infrastructure_error") or plan.get("infrastructure_error"):
        return {
            "label": "inconclusive",
            "comparable": False,
            "reason_code": "infrastructure_or_provider_error",
        }
    direct_tier = _quality_tier(direct)
    plan_tier = _quality_tier(plan)
    if (
        not direct.get("execution_pass")
        and not plan.get("execution_pass")
    ):
        return {
            "label": "both_failed",
            "comparable": False,
            "reason_code": "both_paths_failed",
        }
    if plan_tier > direct_tier:
        return {
            "label": "plan_helpful",
            "comparable": True,
            "reason_code": "plan_improved_higher_priority_quality",
        }
    if plan_tier < direct_tier:
        return {
            "label": "plan_harmful",
            "comparable": True,
            "reason_code": "plan_regressed_higher_priority_quality",
        }

    direct_result = int(direct.get("tt_result_ms") or 0)
    plan_result = int(plan.get("tt_result_ms") or 0)
    direct_tokens = int(direct.get("total_tokens") or 0)
    plan_tokens = int(plan.get("total_tokens") or 0)
    # A cost-only helpful label is deliberately conservative: both latency and
    # token must improve by at least 10%, which a normal extra planning call
    # will not satisfy.
    if (
        direct_result > 0
        and plan_result <= direct_result * .9
        and plan_tokens <= direct_tokens * .9
    ):
        return {
            "label": "plan_helpful",
            "comparable": True,
            "reason_code": "quality_tie_with_preregistered_net_utility_gain",
        }
    return {
        "label": "plan_neutral",
        "comparable": True,
        "reason_code": "quality_tie_without_net_utility_gain",
    }


def validate_planning_contract() -> dict[str, Any]:
    payload = load_planning_contract()
    cases = list(payload.get("cases") or [])
    ids = [str(case.get("id") or "") for case in cases]
    families = [str(case.get("family_id") or "") for case in cases]
    splits = {str(case.get("split") or "") for case in cases}
    categories = {str(case.get("category") or "") for case in cases}
    errors: list[str] = []
    counts = {label: 0 for label in ALL_LABELS}
    for case in cases:
        case_id = str(case.get("id") or "<unknown>")
        required = (
            "id",
            "family_id",
            "split",
            "category",
            "question",
            "review_status",
            "review_label",
            "context",
            "direct",
            "plan",
        )
        missing = [name for name in required if case.get(name) in (None, "")]
        if missing:
            errors.append(f"{case_id}:missing:{','.join(missing)}")
            continue
        derived = label_counterfactual(case)
        expected = str(case.get("review_label") or "")
        if expected not in ALL_LABELS:
            errors.append(f"{case_id}:invalid_label:{expected}")
        elif derived["label"] != expected:
            errors.append(
                f"{case_id}:label_mismatch:{expected}!={derived['label']}"
            )
        counts[derived["label"]] += 1
        if case.get("review_status") != "reviewed":
            errors.append(f"{case_id}:not_reviewed")
    checks = {
        "versioned": bool(payload.get("version")),
        "reviewed": payload.get("review_status") == "reviewed",
        "minimum_cases": len(cases) >= 36,
        "unique_ids": len(ids) == len(set(ids)) and all(ids),
        "family_ids_present": all(families),
        "split_contract": {"train", "dev", "test"}.issubset(splits),
        "category_coverage": len(categories) >= 12,
        "positive_and_negative_labels": (
            counts["plan_helpful"] >= 8
            and counts["plan_neutral"] >= 8
            and counts["plan_harmful"] >= 2
        ),
        "excluded_labels_retained": (
            counts["both_failed"] >= 1
            and counts["inconclusive"] >= 1
            and counts["ir_bypass"] >= 2
        ),
        "derived_labels_match_review": not errors,
    }
    safe_fingerprint_payload = {
        "version": payload.get("version"),
        "ids": ids,
        "families": families,
        "splits": [case.get("split") for case in cases],
        "labels": [case.get("review_label") for case in cases],
        "contexts": [case.get("context") for case in cases],
    }
    return {
        "valid": all(checks.values()),
        "version": payload.get("version"),
        "fingerprint": stable_signal_hash(safe_fingerprint_payload),
        "total": len(cases),
        "categories": len(categories),
        "splits": sorted(splits),
        "label_counts": counts,
        "checks": checks,
        "errors": errors,
        "raw_questions_in_report": False,
    }
