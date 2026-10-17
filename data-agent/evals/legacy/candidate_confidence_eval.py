"""Deterministic TODO-45 policy matrix; it does not call the LLM or DW."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from app.agent_system.agents.nl2sql.services.candidate_confidence import (
    CONFIDENCE_VERSION,
    assess_candidate,
    choose_candidate,
    should_expand_candidate,
)
from evals.paths import REPORTS_ROOT


REPORT_PATH = REPORTS_ROOT / "candidate_confidence_latest.json"


def build_report(*, write: bool = True) -> dict:
    cases = []
    for index in range(25):
        primary = {"label": "A", "valid": True, "score": .90, "band": "high"}
        expand, reason = should_expand_candidate(primary, {"retry_count": 0}, 100)
        cases.append({
            "id": f"CONF-HIGH-{index + 1:02d}", "primary_correct": True,
            "adaptive_triggered": expand, "trigger_reason": reason,
            "selected": "A", "adaptive_correct": True, "unsafe_selected": False,
        })
    for index in range(5):
        primary = {"label": "A", "valid": True, "score": .52, "band": "low"}
        alternative = {
            "label": "B", "valid": index < 4,
            "score": .86 if index < 4 else 0.0,
            "band": "high" if index < 4 else "low",
        }
        expand, reason = should_expand_candidate(primary, {"retry_count": 0}, 100)
        selected, selection_reason = choose_candidate(primary, alternative)
        cases.append({
            "id": f"CONF-LOW-{index + 1:02d}", "primary_correct": False,
            "adaptive_triggered": expand, "trigger_reason": reason,
            "selected": selected, "selection_reason": selection_reason,
            "adaptive_correct": selected == "B" and alternative["valid"],
            "unsafe_selected": selected == "B" and not alternative["valid"],
        })

    total = len(cases)
    triggered = sum(item["adaptive_triggered"] for item in cases)
    repaired = sum(item["adaptive_correct"] and not item["primary_correct"] for item in cases)
    baseline_correct = sum(item["primary_correct"] for item in cases)
    adaptive_correct = sum(item["adaptive_correct"] for item in cases)
    unsafe = sum(item["unsafe_selected"] for item in cases)
    high_extra = sum(item["adaptive_triggered"] for item in cases if item["id"].startswith("CONF-HIGH"))

    product_tables = [
        {"name": "dim_product", "columns": [
            {"name": "product_id", "role": "primary_key", "is_display": False},
            {"name": "product_name", "role": "dimension", "is_display": True},
        ]},
        {"name": "fact_order", "columns": [
            {"name": "order_id", "role": "primary_key", "is_display": False},
            {"name": "product_id", "role": "foreign_key", "is_display": False},
            {"name": "customer_id", "role": "foreign_key", "is_display": False},
        ]},
    ]
    repeat_sql = (
        "WITH customer_orders AS (SELECT customer_id, COUNT(order_id) AS order_count "
        "FROM fact_order GROUP BY customer_id) "
        "SELECT SUM(CASE WHEN order_count >= 2 THEN 1 ELSE 0 END) "
        "/ NULLIF(COUNT(*), 0) AS repurchase_rate FROM customer_orders"
    )
    semantic_specs = [
        {
            "id": "SEM-REPEAT-CTE", "query": "复购率是多少", "sql": repeat_sql,
            "table_infos": [product_tables[1]],
            "semantic_ir": {
                "status": "complete", "intent": "repeat_rate",
                "measures": [{"table": "fact_order", "alias": "repurchase_rate"}],
                "compiler": {"supported": True},
            },
            "expect_expand": False, "expected_feature": "join_contract_satisfied",
        },
        {
            "id": "SEM-ANTI-LEFT", "query": "没有产生订单的商品",
            "sql": (
                "SELECT p.product_name FROM dim_product p LEFT JOIN fact_order f "
                "ON p.product_id=f.product_id WHERE f.order_id IS NULL"
            ),
            "table_infos": product_tables,
            "semantic_ir": {"status": "partial", "compiler": {"supported": False}},
            "expect_expand": False, "expected_feature": "anti_join_satisfied",
        },
        {
            "id": "SEM-ANTI-MISSING", "query": "没有产生订单的商品",
            "sql": "SELECT p.product_name FROM dim_product p",
            "table_infos": product_tables,
            "semantic_ir": {"status": "partial", "compiler": {"supported": False}},
            "expect_expand": True, "expected_weakness": "anti_join_contract_missing",
        },
    ]
    semantic_cases = []
    for spec in semantic_specs:
        state = {
            "query": spec["query"], "table_infos": spec["table_infos"],
            "semantic_ir": spec["semantic_ir"], "retrieval_metrics": [],
            "degradations": [], "retry_count": 0,
        }
        assessment = assess_candidate(spec["sql"], state, validation={"valid": True})
        expand, reason = should_expand_candidate(assessment, state, 100)
        feature_ok = bool(assessment["sql_features"].get(spec.get("expected_feature"))) if spec.get("expected_feature") else True
        weakness_ok = spec.get("expected_weakness") in assessment["actionable_weaknesses"] if spec.get("expected_weakness") else True
        semantic_cases.append({
            "id": spec["id"], "expected_expand": spec["expect_expand"],
            "actual_expand": expand, "trigger_reason": reason,
            "feature_ok": feature_ok, "weakness_ok": weakness_ok,
            "passed": expand == spec["expect_expand"] and feature_ok and weakness_ok,
        })
    semantic_passed = sum(item["passed"] for item in semantic_cases)
    summary = {
        "cases": total,
        "baseline_correct": baseline_correct,
        "baseline_success_rate": round(baseline_correct / total, 4),
        "adaptive_correct": adaptive_correct,
        "adaptive_success_rate": round(adaptive_correct / total, 4),
        "second_candidate_count": triggered,
        "second_candidate_rate": round(triggered / total, 4),
        "targeted_repair_count": repaired,
        "targeted_repair_rate": round(repaired / max(1, triggered), 4),
        "unsafe_selected": unsafe,
        "high_confidence_extra_calls": high_extra,
        "semantic_contract_cases": len(semantic_cases),
        "semantic_contract_passed": semantic_passed,
        "semantic_false_positive_count": sum(
            item["actual_expand"] and not item["expected_expand"] for item in semantic_cases
        ),
    }
    summary["gate_passed"] = bool(
        summary["targeted_repair_rate"] >= .8
        and summary["second_candidate_rate"] <= .2
        and unsafe == 0
        and high_extra == 0
        and semantic_passed == len(semantic_cases)
    )
    report = {
        "available": True,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "source": "deterministic_candidate_policy_matrix",
        "live_model_evaluated": False,
        "version": CONFIDENCE_VERSION,
        "summary": summary,
        "cases": cases,
        "semantic_contract_cases": semantic_cases,
        "evidence_boundary": "验证触发、预算和判优规则；不代表真实模型候选质量或 TTResult。",
    }
    if write:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    report = build_report(write=True)
    summary = report["summary"]
    print(
        f"候选判优矩阵：{summary['cases']} 条，单候选 {summary['baseline_success_rate']:.1%} → "
        f"自适应 {summary['adaptive_success_rate']:.1%}，备选触发 {summary['second_candidate_rate']:.1%}，"
        f"定向修复 {summary['targeted_repair_rate']:.1%}，门禁={'通过' if summary['gate_passed'] else '失败'}"
    )
    print(f"报告：{REPORT_PATH}")
    return 0 if summary["gate_passed"] or not args.enforce else 2


if __name__ == "__main__":
    raise SystemExit(main())
