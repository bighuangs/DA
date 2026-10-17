"""TODO-64 deterministic semantic-repair registry ablation and release gates."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import time
from typing import Any, Mapping

import yaml

from app.agent_system.agents.nl2sql.services.semantic_repair import (
    PROFILE_RULES,
    apply_semantic_repairs,
    registry_catalog,
    runtime_profile_status,
)
from app.domain.query.safety.semantic_violation import stable_hash
from app.domain.query.safety.sql_security import SQLSecurityError, check_sql_read
from evals.paths import (
    LEGACY_DATASETS_ROOT,
    PROJECT_ROOT,
    REPORTS_ROOT as REPORT_ROOT,
    SEMANTIC_REPAIR_RUNS_ROOT,
)


CASE_PATH = LEGACY_DATASETS_ROOT / "semantic_repair_cases.yaml"
RUN_ROOT = SEMANTIC_REPAIR_RUNS_ROOT
LATEST_JSON = REPORT_ROOT / "semantic_repair_ablation_latest.json"
LATEST_MD = REPORT_ROOT / "semantic_repair_ablation_latest.md"
PROFILE_MATRIX = {
    "P0": "inline_v1",
    "P1": "violation_only_v1",
    "P2": "registry_same_rules_v1",
    "P3": "registry_extended_v1",
    "P4": "targeted_correction_v1",
    "P5": "upstream_first_v1",
    "P6": "cascade_v1",
}


def _metadata(name: str) -> list[dict[str, Any]]:
    orders = {
        "name": "orders",
        "columns": [
            {"name": "order_id", "role": "primary_key"},
            {"name": "customer_id", "role": "foreign_key"},
            {"name": "account_id", "role": "foreign_key"},
            {"name": "order_amount", "role": "measure"},
            {"name": "order_date", "role": "dimension"},
            {"name": "status", "role": "dimension"},
        ],
    }
    customer_columns = [
        {"name": "customer_id", "role": "primary_key"},
        {"name": "customer_name", "role": "dimension"},
    ]
    if name == "ambiguous_customer":
        customer_columns.append({"name": "account_id", "role": "primary_key"})
    customers = {"name": "customers", "columns": customer_columns}
    return [orders] if name == "orders_only" else [customers, orders]


def load_contract() -> dict[str, Any]:
    payload = yaml.safe_load(CASE_PATH.read_text(encoding="utf-8")) or {}
    cases = [dict(item) for item in payload.get("cases") or []]
    ids = [str(item.get("id") or "") for item in cases]
    splits = {str(item.get("split") or "") for item in cases}
    families = {str(item.get("family") or "") for item in cases}
    checks = {
        "schema_version": payload.get("schema_version") == "semantic-repair-reviewed-set-v1",
        "minimum_30_cases": len(cases) >= 30,
        "unique_ids": len(ids) == len(set(ids)) and all(ids),
        "train_dev_test": {"train", "dev", "test"}.issubset(splits),
        "positive_near_miss_safety": {"join", "projection", "ir", "near_miss", "safety"}.issubset(families),
        "test_majority": sum(item.get("split") == "test" for item in cases) >= 20,
    }
    return {
        "schema_version": payload.get("schema_version"),
        "cases": cases,
        "total": len(cases),
        "splits": {split: sum(item.get("split") == split for item in cases) for split in sorted(splits)},
        "families": {family: sum(item.get("family") == family for item in cases) for family in sorted(families)},
        "raw_questions_in_report": False,
        "checks": checks,
        "valid": all(checks.values()),
    }


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(len(ordered) * ratio) - 1)], 4)


def _read_only(sql: str) -> bool:
    try:
        check_sql_read(sql)
    except (SQLSecurityError, Exception):
        return False
    return True


def evaluate_profile(profile: str, cases: list[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    positives = correct_repairs = total_changes = false_triggers = harmful = 0
    safe_cases = safe_unchanged = idempotent = 0
    shadow_would_apply = 0
    for case in cases:
        original = str(case.get("sql") or "")
        expected_rule = str(case.get("expected_rule") or "")
        expected_change = bool(expected_rule)
        started = time.perf_counter()
        repaired, result = apply_semantic_repairs(
            original,
            query=str(case.get("query") or ""),
            table_infos=_metadata(str(case.get("metadata") or "orders_only")),
            semantic_ir=dict(case.get("semantic_ir") or {}),
            profile=profile,
        )
        measured_ms = (time.perf_counter() - started) * 1000
        latencies.append(measured_ms)
        evidence = result.to_dict()
        applied = [row["rule_code"] for row in evidence.get("applied_rules") or []]
        changed = repaired != original
        if expected_change:
            positives += 1
            correct_repairs += expected_rule in applied
        if changed:
            total_changes += 1
        if not expected_change and changed:
            false_triggers += 1
        second, _ = apply_semantic_repairs(
            repaired,
            query=str(case.get("query") or ""),
            table_infos=_metadata(str(case.get("metadata") or "orders_only")),
            semantic_ir=dict(case.get("semantic_ir") or {}),
            profile=profile,
        )
        is_idempotent = second == repaired
        idempotent += is_idempotent
        unsafe_input = not _read_only(original)
        if unsafe_input:
            safe_cases += 1
            safe_unchanged += not changed
        invariants = evidence.get("invariants") or {}
        invariant_ok = all(bool(value) for value in invariants.values())
        is_harmful = changed and (unsafe_input or not invariant_ok)
        harmful += is_harmful
        shadow_would_apply += any(
            row.get("status") == "would_apply"
            for row in evidence.get("shadow_rules") or []
        )
        rows.append({
            "case_id": case.get("id"),
            "split": case.get("split"),
            "family": case.get("family"),
            "question_hash": stable_hash(case.get("query")),
            "sql_hash": stable_hash(original),
            "expected_rule": expected_rule or None,
            "changed": changed,
            "applied_rules": applied,
            "correct": (expected_rule in applied) if expected_change else not changed,
            "idempotent": is_idempotent,
            "invariants": invariant_ok,
            "unsafe_input": unsafe_input,
            "raw_sql_logged": False,
        })
    negatives = max(0, len(cases) - positives)
    return {
        "profile": profile,
        "rules": list(PROFILE_RULES[profile]),
        "cases": len(cases),
        "positive_cases": positives,
        "negative_cases": negatives,
        "correct_repairs": correct_repairs,
        "changed_cases": total_changes,
        "repair_precision": round(correct_repairs / total_changes, 4) if total_changes else 1.0,
        "targeted_rescue_rate": round(correct_repairs / positives, 4) if positives else 1.0,
        "false_trigger_count": false_triggers,
        "near_miss_false_trigger_rate": round(false_triggers / negatives, 4) if negatives else 0.0,
        "harmful_repair_count": harmful,
        "idempotence_rate": round(idempotent / len(cases), 4) if cases else 1.0,
        "unsafe_unchanged_rate": round(safe_unchanged / safe_cases, 4) if safe_cases else 1.0,
        "repair_latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 4) if latencies else 0.0,
            "p95": _percentile(latencies, .95),
            "max": round(max(latencies), 4) if latencies else 0.0,
        },
        "shadow_would_apply": shadow_would_apply,
        "rows": rows,
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# TODO-64 语义修复注册表消融",
        "",
        f"> 生成时间：{report['generated_at']} ｜ 状态：`{'pass' if report['implementation_gates']['gate_passed'] else 'blocked'}`",
        "",
        "| 层级 | Profile | 规则数 | Precision | Rescue | 误触发 | 幂等 | 安全 | P95 | 资格 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for label, item in report["profiles"].items():
        lines.append(
            f"| {label} | `{item['profile']}` | {len(item['rules'])} | "
            f"{item['repair_precision']:.1%} | {item['targeted_rescue_rate']:.1%} | "
            f"{item['near_miss_false_trigger_rate']:.1%} | {item['idempotence_rate']:.1%} | "
            f"{item['unsafe_unchanged_rate']:.1%} | {item['repair_latency_ms']['p95']:.3f} ms | "
            f"{'active' if label == 'P2' else 'offline'} |"
        )
    lines.extend([
        "",
        "## 发布边界",
        "",
        f"- 线上 active：`{report['decision']['active_profile']}`（P2，与 P0 同规则）",
        f"- P0 fallback：`{report['decision']['fallback_profile']}`，进程内即时回退",
        "- P3–P6：只保留离线/影子证据，不因本报告自动发布",
        "- 报告只保存 case/hash/规则码，不保存原始 SQL 或原始问题",
        "",
        "## 门禁",
        "",
    ])
    lines.extend(
        f"- {'✅' if passed else '❌'} `{name}`"
        for name, passed in report["implementation_gates"].items()
        if name != "gate_passed"
    )
    return "\n".join(lines) + "\n"


def build_report(*, write: bool = True) -> dict[str, Any]:
    contract = load_contract()
    profiles = {
        label: evaluate_profile(profile, contract["cases"])
        for label, profile in PROFILE_MATRIX.items()
    }
    p0, p2, p3 = profiles["P0"], profiles["P2"], profiles["P3"]
    compatibility = all(
        left["changed"] == right["changed"]
        and left["applied_rules"] == right["applied_rules"]
        for left, right in zip(p0["rows"], p2["rows"])
    )
    runtime = runtime_profile_status()
    active_checks = {
        "contract_valid": contract["valid"],
        "p0_p6_present": set(profiles) == set(PROFILE_MATRIX),
        "active_p2_same_rules": runtime["runtime_profile"] == "registry_same_rules_v1",
        "p0_p2_character_compatible": compatibility,
        "active_precision_100": p2["repair_precision"] == 1,
        "active_false_trigger_zero": p2["false_trigger_count"] == 0,
        "active_harmful_repair_zero": p2["harmful_repair_count"] == 0,
        "active_idempotence_100": p2["idempotence_rate"] == 1,
        "active_unsafe_unchanged_100": p2["unsafe_unchanged_rate"] == 1,
        "active_p95_under_20ms": p2["repair_latency_ms"]["p95"] <= 20,
        "privacy_safe_artifact": all(
            not row["raw_sql_logged"]
            for item in profiles.values()
            for row in item["rows"]
        ),
        "rollback_without_restart": bool(runtime["rollback_without_restart"]),
    }
    candidate_checks = {
        "candidate_p3_precision_100": p3["repair_precision"] == 1,
        "candidate_p3_rescue_100": p3["targeted_rescue_rate"] == 1,
        "candidate_p3_false_trigger_zero": p3["false_trigger_count"] == 0,
        "candidate_p3_harmful_repair_zero": p3["harmful_repair_count"] == 0,
        "candidate_p3_idempotence_100": p3["idempotence_rate"] == 1,
        "candidate_p3_unsafe_unchanged_100": p3["unsafe_unchanged_rate"] == 1,
        "candidate_p3_p95_under_20ms": p3["repair_latency_ms"]["p95"] <= 20,
    }
    checks = {**active_checks, **candidate_checks}
    report = {
        "available": True,
        "artifact_schema": "semantic-repair-ablation-v1",
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "evidence_level": "reviewed_deterministic_offline",
        "contract": {key: value for key, value in contract.items() if key != "cases"},
        "runtime": runtime,
        "profiles": profiles,
        "registry": registry_catalog(),
        "taxonomy": [
            {"layer": "syntax", "prefix": "sql.syntax", "default": "llm"},
            {"layer": "semantic", "prefix": "semantic", "default": "deterministic/llm"},
            {"layer": "semantic_ir", "prefix": "semantic_ir", "default": "deterministic/llm"},
            {"layer": "metric", "prefix": "metric", "default": "llm"},
            {"layer": "security", "prefix": "security", "default": "reject"},
            {"layer": "cost", "prefix": "cost", "default": "reject"},
        ],
        "governance_layers": [
            {"level": "L0", "name": "拒绝", "scope": "安全、权限、成本；禁止自动扩大边界"},
            {"level": "L1", "name": "确定性修复", "scope": "唯一元数据边、唯一投影、唯一 IR 约束"},
            {"level": "L2", "name": "定向模型纠错", "scope": "稳定 violation code + 有界指令 + 单次重试"},
            {"level": "L3", "name": "上游再生成", "scope": "仅候选；需 Full Run 与影子证据后发布"},
        ],
        "decision": {
            "active_label": "P2",
            "active_profile": runtime["runtime_profile"],
            "fallback_label": "P0",
            "fallback_profile": runtime["fallback_profile"],
            "active_unchanged_semantics": compatibility,
            "offline_candidates": ["P3", "P4", "P5", "P6"],
            "offline_candidate_label": "P3",
            "offline_candidate_profile": p3["profile"],
            "offline_candidate_qualified": all(candidate_checks.values()),
            "offline_candidate_blockers": [
                name for name, passed in candidate_checks.items() if not passed
            ],
            "release_qualified": False,
            "release_blockers": [
                "missing_live_llm_targeted_correction_pairs",
                "missing_cold_warm_warm_full_run",
                "missing_shadow_canary_and_negative_feedback_review",
            ],
            "selection_reason": (
                "P2 把已发布的三条修复规则纳入版本化注册表并保持 P0 字符兼容；"
                "扩展规则与模型纠错只做离线/影子评测，避免错误自动修复扩大语义边界。"
            ),
        },
        "implementation_gates": {**checks, "gate_passed": all(checks.values())},
    }
    if write:
        REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RUN_ROOT / f"semantic_repair_{stamp}.json"
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        path.write_text(rendered, encoding="utf-8")
        LATEST_JSON.write_text(rendered, encoding="utf-8")
        LATEST_MD.write_text(_render_markdown(report), encoding="utf-8")
        report["artifact"] = str(path.relative_to(PROJECT_ROOT))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="TODO-64 语义修复注册表离线消融")
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    report = build_report(write=True)
    print(_render_markdown(report))
    print(f"JSON：{LATEST_JSON}")
    return 2 if args.enforce and not report["implementation_gates"]["gate_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
