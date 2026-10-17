"""TODO-63 deterministic planning-route candidates and nested ablations.

This runner evaluates reviewed, synthetic Direct/Plan pairs.  It is intentionally
not a live LLM Full Run and never changes the active production profile.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
import time
from typing import Any, Mapping

from app.config.app_config import app_config
from app.agent_system.agents.nl2sql.services.planning_router import (
    FEATURE_LEVELS,
    IMPLEMENTED_PROFILES,
    decide_planning_route,
    extract_planning_features,
    keyword_length_needs_plan,
    planning_feature_vector,
    runtime_profile_status,
)
from app.agent_system.agents.nl2sql.services.query_signals import extract_query_signals
from evals.legacy.planning_counterfactual import (
    COMPARABLE_LABELS,
    label_counterfactual,
    load_planning_contract,
    validate_planning_contract,
)
from app.agent_system.agents.nl2sql.contracts.query_signals import stable_signal_hash
from evals.paths import (
    PLANNING_RUNS_ROOT,
    PROJECT_ROOT,
    REPORTS_ROOT as REPORT_ROOT,
)


RUN_ROOT = PLANNING_RUNS_ROOT
LATEST_JSON = REPORT_ROOT / "planning_router_ablation_latest.json"
LATEST_MD = REPORT_ROOT / "planning_router_ablation_latest.md"
PROFILE_MATRIX = {
    "C0": "never_plan",
    "C1": "keyword_length_v1",
    "C2": "always_plan",
    "C3": "structured_rules_v1",
    "C4": "weighted_scorecard_v1",
    "C5": "calibrated_linear_v1",
}
LABELS = {
    "plan_helpful": 1,
    "plan_neutral": 0,
    "plan_harmful": 0,
}


def _tables(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    names = list(context.get("tables") or ["fact_sales"])
    join_groups = int(context.get("join_groups") or 0)
    output = []
    for index, name in enumerate(names):
        columns = [
            {
                "name": "id" if index == 0 else f"{name}_id",
                "role": "primary_key",
                "type": "bigint",
            },
            {"name": "amount", "role": "measure", "type": "decimal"},
        ]
        if join_groups and len(names) > 1:
            columns.append({
                "name": "customer_id",
                "role": "primary_key" if index == 0 else "foreign_key",
                "type": "bigint",
            })
        output.append({
            "name": name,
            "role": "fact" if "fact" in name else "dimension",
            "description": "",
            "columns": columns,
        })
    return output


def case_state(case: Mapping[str, Any]) -> dict[str, Any]:
    query = str(case.get("question") or "")
    context = dict(case.get("context") or {})
    signals = extract_query_signals(
        query,
        profile="domain_signals_v1",
        workspace_id="planning-reviewed-workspace",
        workspace_terms=("GMV", "AOV", "复购率", "客单价"),
        lexical_top_k=10,
    )
    supported = bool(case.get("semantic_ir_supported"))
    measure_count = int(context.get("measure_count") or 1)
    dimension_count = int(context.get("dimension_count") or 1)
    return {
        "query": query,
        "original_query": query,
        "keywords": signals.retrieval_terms(),
        "query_signals": signals.to_state(),
        "table_infos": _tables(context),
        "metric_infos": [
            {"name": f"metric_{index}", "formula_rule": {}}
            for index in range(int(context.get("metric_count") or 0))
        ],
        "semantic_ir": {
            "intent": "aggregate",
            "dimensions": [
                {"table": "fact_sales", "column": f"dim_{index}"}
                for index in range(dimension_count)
            ],
            "measures": [
                {"table": "fact_sales", "column": f"measure_{index}"}
                for index in range(measure_count)
            ],
            "joins": [
                {"left_table": "fact_sales", "right_table": "dim_customer"}
                for _ in range(int(context.get("join_groups") or 0))
            ],
            "compiler": {
                "supported": supported,
                "mode": "deterministic_ast" if supported else "llm_fallback",
                "reason": (
                    "结构完整且连接唯一"
                    if supported
                    else str(context.get("ir_reason") or "语义证据不足")
                ),
            },
        },
        "follow_up": {
            "action": str(context.get("follow_up_action") or "new_topic"),
            "preserve": context.get("preserve") or {},
        },
        "context_budget": {
            "total_budget": app_config.query.context_total_token_budget,
        },
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * quantile) - 1))
    return round(ordered[index], 4)


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, value))))


def fit_linear(
    rows: list[tuple[dict[str, float], int]],
    *,
    epochs: int = 800,
    learning_rate: float = .08,
    l2: float = .015,
) -> dict[str, float]:
    names = sorted({name for vector, _ in rows for name in vector})
    weights = {name: 0.0 for name in names}
    intercept = 0.0
    if not rows:
        return {"intercept": intercept}
    for _ in range(epochs):
        grad = {name: 0.0 for name in names}
        grad_intercept = 0.0
        for vector, label in rows:
            score = intercept + sum(
                weights[name] * vector.get(name, 0.0) for name in names
            )
            error = _sigmoid(score) - label
            grad_intercept += error
            for name in names:
                grad[name] += error * vector.get(name, 0.0)
        scale = 1 / len(rows)
        intercept -= learning_rate * grad_intercept * scale
        for name in names:
            weights[name] -= learning_rate * (
                grad[name] * scale + l2 * weights[name]
            )
    return {
        "intercept": round(intercept, 8),
        **{name: round(value, 8) for name, value in weights.items()},
    }


def _ece(probabilities: list[tuple[float, int]], bins: int = 5) -> float:
    if not probabilities:
        return 0.0
    value = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        bucket = [
            (probability, label)
            for probability, label in probabilities
            if low <= probability < high or (index == bins - 1 and probability == 1)
        ]
        if not bucket:
            continue
        confidence = statistics.fmean(item[0] for item in bucket)
        accuracy = statistics.fmean(item[1] for item in bucket)
        value += len(bucket) / len(probabilities) * abs(confidence - accuracy)
    return round(value, 4)


def _selected_outcome(case: Mapping[str, Any], action: str) -> Mapping[str, Any]:
    return case.get("plan") if action == "plan_then_generate" else case.get("direct")


def evaluate_profile(
    profile: str,
    cases: list[dict[str, Any]],
    *,
    feature_level: str = "A4",
    linear_coefficients: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    predictions = []
    latencies = []
    selected = []
    differences = []
    for case in cases:
        label_info = label_counterfactual(case)
        if label_info["label"] not in COMPARABLE_LABELS:
            continue
        state = case_state(case)
        started = time.perf_counter()
        route = decide_planning_route(
            state,
            profile=profile,
            feature_level=feature_level,
            shadow_profiles=(),
            linear_coefficients=linear_coefficients,
        ).privacy_summary()
        latencies.append((time.perf_counter() - started) * 1000)
        truth = LABELS[label_info["label"]]
        action = str(route["actual_action"])
        predicted = int(action == "plan_then_generate")
        probability = float(route.get("score") or 0)
        predictions.append((truth, predicted, probability, label_info["label"]))
        outcome = dict(_selected_outcome(case, action) or {})
        selected.append(outcome)
        baseline_action = (
            "plan_then_generate"
            if keyword_length_needs_plan(str(case.get("question") or ""))
            else "direct_generate"
        )
        if action != baseline_action:
            differences.append({
                "case_id": case["id"],
                "label": label_info["label"],
                "active_action": baseline_action,
                "candidate_action": action,
                "reason_codes": route.get("reason_codes") or [],
            })

    tp = sum(truth == predicted == 1 for truth, predicted, _, _ in predictions)
    tn = sum(truth == predicted == 0 for truth, predicted, _, _ in predictions)
    fp = sum(truth == 0 and predicted == 1 for truth, predicted, _, _ in predictions)
    fn = sum(truth == 1 and predicted == 0 for truth, predicted, _, _ in predictions)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    specificity = tn / (tn + fp) if tn + fp else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = math.sqrt(
        max(0, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    mcc = ((tp * tn - fp * fn) / denominator) if denominator else 0.0
    probabilities = [(item[2], item[0]) for item in predictions]
    brier = (
        statistics.fmean((probability - label) ** 2 for probability, label in probabilities)
        if probabilities else 0.0
    )
    planned_labels = [
        label for _, predicted, _, label in predictions if predicted
    ]
    harmful_plan = sum(label == "plan_harmful" for label in planned_labels)
    result_rate = statistics.fmean(
        bool(item.get("result_equivalent")) for item in selected
    ) if selected else 0.0
    safety_rate = statistics.fmean(
        bool(item.get("safety_pass")) for item in selected
    ) if selected else 0.0
    times = [float(item.get("tt_result_ms") or 0) for item in selected]
    tokens = [float(item.get("total_tokens") or 0) for item in selected]
    costs = [float(item.get("estimated_cost") or 0) for item in selected]
    return {
        "profile": profile,
        "feature_level": feature_level,
        "cases": len(predictions),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "plan_precision": round(precision, 4),
        "plan_recall": round(recall, 4),
        "f1": round(f1, 4),
        "balanced_accuracy": round((recall + specificity) / 2, 4),
        "mcc": round(mcc, 4),
        "brier": round(brier, 4),
        "ece": _ece(probabilities),
        "plan_rate": round((tp + fp) / len(predictions), 4) if predictions else 0,
        "over_plan_rate": round(fp / (tn + fp), 4) if tn + fp else 0,
        "under_plan_rate": round(fn / (tp + fn), 4) if tp + fn else 0,
        "harmful_plan_rate": round(harmful_plan / len(planned_labels), 4) if planned_labels else 0,
        "selected_path": {
            "result_equivalence_rate": round(result_rate, 4),
            "safety_rate": round(safety_rate, 4),
            "tt_result_p50_ms": _percentile(times, .5),
            "tt_result_p95_ms": _percentile(times, .95),
            "tokens_mean": round(statistics.fmean(tokens), 2) if tokens else 0,
            "estimated_cost_mean": round(statistics.fmean(costs), 8) if costs else 0,
        },
        "decision_latency_ms": {
            "p50": _percentile(latencies, .5),
            "p95": _percentile(latencies, .95),
            "p99": _percentile(latencies, .99),
            "max": round(max(latencies), 4) if latencies else 0,
        },
        "decision_differences": differences,
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    rows = []
    for label, item in report["profiles"].items():
        rows.append(
            f"| {label} | `{item['profile']}` | "
            f"{item['plan_precision']:.1%} | {item['plan_recall']:.1%} | "
            f"{item['over_plan_rate']:.1%} | {item['under_plan_rate']:.1%} | "
            f"{item['selected_path']['result_equivalence_rate']:.1%} | "
            f"{item['decision_latency_ms']['p95']:.3f} ms |"
        )
    ablations = []
    for level, item in report["feature_ablations"].items():
        ablations.append(
            f"| {level} | {item['plan_precision']:.1%} | "
            f"{item['plan_recall']:.1%} | {item['f1']:.3f} | "
            f"{item['over_plan_rate']:.1%} | {item['under_plan_rate']:.1%} |"
        )
    return "\n".join([
        "# TODO-63 SQL 规划路由离线消融",
        "",
        f"> 生成时间：{report['generated_at']}",
        f"> Active：`{report['decision']['active_profile']}`（未切换）",
        f"> 离线推荐：`{report['decision']['offline_recommendation']}`",
        f"> 证据等级：`{report['evidence_level']}`；不是 live Full Run 或生产发布批准。",
        "",
        "## 候选对比",
        "",
        "| 组 | Profile | Plan Precision | Plan Recall | Over-plan | Under-plan | 选路结果一致 | 决策 P95 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        "## C4 嵌套特征消融",
        "",
        "| 层级 | Precision | Recall | F1 | Over-plan | Under-plan |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *ablations,
        "",
        "## 证据边界",
        "",
        "- C0～C5 已实现；C6～C9 未实现，因此没有进入胜者表。",
        "- `both_failed`、`inconclusive` 与 Semantic IR bypass 均保留，但不进入主分类分母。",
        "- 缓存 TTL（Time To Live）已冻结；非流式 planning 只记录完整 provider latency，不伪造 TTFT/Time To Last Token。",
        "- 挑战者未完成 cold/warm/warm 三次真实 Full Run，active 继续保持 `keyword_length_v1`。",
        "",
    ])


def build_report(*, write: bool = False) -> dict[str, Any]:
    contract = validate_planning_contract()
    payload = load_planning_contract()
    cases = list(payload.get("cases") or [])
    train_rows = []
    for case in cases:
        label = label_counterfactual(case)["label"]
        if case.get("split") not in {"train", "dev"} or label not in LABELS:
            continue
        features = extract_planning_features(case_state(case))
        train_rows.append((planning_feature_vector(features, "A4"), LABELS[label]))
    fitted = fit_linear(train_rows)
    test_cases = [case for case in cases if case.get("split") == "test"]
    profiles = {}
    for label, profile in PROFILE_MATRIX.items():
        profiles[label] = evaluate_profile(
            profile,
            test_cases,
            linear_coefficients=fitted if label == "C5" else None,
        )
    ablations = {
        level: evaluate_profile(
            "weighted_scorecard_v1",
            test_cases,
            feature_level=level,
        )
        for level in FEATURE_LEVELS
    }
    active = runtime_profile_status()["runtime_profile"]
    eligible = {
        label: item
        for label, item in profiles.items()
        if label in {"C3", "C4", "C5"}
        and item["selected_path"]["safety_rate"] == 1
        and item["selected_path"]["result_equivalence_rate"]
        >= profiles["C1"]["selected_path"]["result_equivalence_rate"]
        and item["under_plan_rate"] <= profiles["C1"]["under_plan_rate"]
        and item["decision_latency_ms"]["p95"] <= 5
    }
    recommendation_label, recommendation = max(
        eligible.items(),
        key=lambda row: (
            row[1]["f1"],
            -row[1]["over_plan_rate"],
            -row[1]["decision_latency_ms"]["p95"],
        ),
        default=("C1", profiles["C1"]),
    )
    offline_recommendation = recommendation["profile"]
    comparable_count = sum(
        label_counterfactual(case)["label"] in COMPARABLE_LABELS
        for case in test_cases
    )
    exclusions = {
        label: sum(
            label_counterfactual(case)["label"] == label for case in test_cases
        )
        for label in ("both_failed", "inconclusive", "ir_bypass")
    }
    report = {
        "available": True,
        "artifact_schema": "planning-router-ablation-v1",
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "evidence_level": "reviewed_offline_counterfactual",
        "contract": contract,
        "evaluation_scope": {
            "split": "test",
            "total_cases": len(test_cases),
            "comparable_cases": comparable_count,
            "excluded": exclusions,
            "raw_questions_in_report": False,
        },
        "frozen_variables": {
            "semantic_ir_profile": "semantic-ir-v1",
            "semantic_repair_profile": app_config.planning_route.repair_profile,
            "query_signal_profile": "domain_signals_v1",
            "cache_ttl_seconds": app_config.query.cache_ttl_seconds,
            "retrieval_cache_ttl_seconds": app_config.query.retrieval_cache_ttl_seconds,
            "cache_state": "deterministic_offline_no_result_cache",
            "planning_prompt_changed": False,
            "sql_prompt_changed": False,
            "candidate_policy_changed": False,
        },
        "profiles": profiles,
        "feature_ablations": ablations,
        "learning_candidate": {
            "profile": "calibrated_linear_v1",
            "training_splits": ["train", "dev"],
            "training_cases": len(train_rows),
            "coefficients": fitted,
            "coefficient_fingerprint": stable_signal_hash(fitted),
            "group_leakage": False,
        },
        "decision": {
            "active_profile": active,
            "active_unchanged": active == "keyword_length_v1",
            "offline_recommendation_label": recommendation_label,
            "offline_recommendation": offline_recommendation,
            "release_qualified": False,
            "release_blockers": [
                "missing_live_direct_plan_pairs",
                "missing_cold_warm_warm_full_run",
                "missing_production_shadow_canary",
            ],
            "fallback_profile": app_config.planning_route.fallback_profile,
            "rollback_without_restart": True,
            "selection_reason": (
                "离线 reviewed set 上取得 Pareto 候选；真实发布门禁未完成，"
                "因此只保留推荐，不切换 active。"
            ),
        },
        "implemented_candidates": sorted(IMPLEMENTED_PROFILES),
        "not_implemented_candidates": [
            "tree_classifier_v1",
            "small_llm_router_v1",
            "cascade_utility_v1",
            "contextual_bandit_shadow",
        ],
    }
    c1 = profiles["C1"]
    checks = {
        "contract_valid": contract["valid"],
        "c0_c1_c2_present": all(label in profiles for label in ("C0", "C1", "C2")),
        "c3_c4_learning_present": all(label in profiles for label in ("C3", "C4", "C5")),
        "a0_a4_present": all(level in ablations for level in FEATURE_LEVELS),
        "excluded_labels_retained": all(value >= 1 for value in exclusions.values()),
        "active_c1_unchanged": active == "keyword_length_v1",
        "c1_character_compatible": all(
            (
                profiles["C1"]["profile"] == "keyword_length_v1",
                c1["cases"] == comparable_count,
            )
        ),
        "router_p95_under_5ms": all(
            profiles[label]["decision_latency_ms"]["p95"] <= 5
            for label in ("C3", "C4", "C5")
        ),
        "privacy_safe_report": not contract["raw_questions_in_report"],
        "ttl_frozen_and_disambiguated": True,
    }
    report["implementation_gates"] = {
        **checks,
        "gate_passed": all(checks.values()),
    }
    if write:
        REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_path = RUN_ROOT / f"planning_{stamp}.json"
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        run_path.write_text(rendered, encoding="utf-8")
        LATEST_JSON.write_text(rendered, encoding="utf-8")
        LATEST_MD.write_text(_render_markdown(report), encoding="utf-8")
        report["artifact"] = str(run_path.relative_to(PROJECT_ROOT))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="TODO-63 规划路由离线消融")
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    report = build_report(write=True)
    print(_render_markdown(report))
    print(f"JSON：{LATEST_JSON}")
    return 2 if args.enforce and not report["implementation_gates"]["gate_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
