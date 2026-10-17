from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from app.agent_system.agents.nl2sql.follow_up import resolve_follow_up, semantic_state_from_sql
from evals.paths import LEGACY_DATASETS_ROOT


DEFAULT_SET = LEGACY_DATASETS_ROOT / "multi_turn_rewrite_set.yaml"
DEFAULT_PRIOR_QUESTION = "各品类的销量总和"
DEFAULT_PRIOR_SQL = (
    "SELECT p.category, SUM(f.order_quantity) AS total_sales FROM fact_order f "
    "JOIN dim_product p ON f.product_id=p.product_id GROUP BY p.category ORDER BY total_sales DESC"
)


def load_cases(path: Path = DEFAULT_SET) -> list[dict]:
    return list(yaml.safe_load(path.read_text(encoding="utf-8")) or [])


def evaluate_cases(cases: list[dict]) -> dict:
    results = []
    latencies = []
    for case in cases:
        prior_question = str(case.get("prior_question") or DEFAULT_PRIOR_QUESTION)
        prior_sql = str(case.get("prior_sql") or DEFAULT_PRIOR_SQL)
        history = [] if case.get("no_history") else [{
            "query": prior_question,
            "standalone_question": prior_question,
            "sql": prior_sql,
            "semantic_state": semantic_state_from_sql(prior_question, prior_sql),
        }]
        started = time.perf_counter()
        output = resolve_follow_up(str(case["question"]), history)
        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)
        standalone = str(output.get("standalone_question") or "")
        checks = [output.get("action") == case.get("expected_action")]
        checks.extend(str(item) in standalone for item in case.get("contains") or [])
        checks.extend(str(item) not in standalone for item in case.get("excludes") or [])
        if "needs_clarification" in case:
            checks.append(bool(output.get("needs_clarification")) == bool(case["needs_clarification"]))
        preserve = output.get("preserve") or {}
        if case.get("preserve_filters"):
            checks.append(bool(preserve.get("filters")))
        if case.get("preserve_order"):
            checks.append(bool(preserve.get("order")))
        if case.get("preserve_limit") is not None:
            checks.append(preserve.get("limit") == case.get("preserve_limit"))
        results.append({
            "id": case.get("id"), "passed": all(checks), "latency_ms": round(latency_ms, 3),
            "action": output.get("action"), "standalone_question": standalone,
            "summary": output.get("summary"),
        })
    ordered = sorted(latencies)
    p95 = ordered[min(len(ordered) - 1, max(0, int(len(ordered) * .95)))] if ordered else 0
    passed = sum(bool(item["passed"]) for item in results)
    return {
        "cases": len(results), "passed": passed, "accuracy": passed / len(results) if results else 0,
        "p95_ms": round(p95, 3), "deterministic_llm_calls": 0, "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="TODO-44 deterministic multi-turn rewrite evaluation")
    parser.add_argument("--set", type=Path, default=DEFAULT_SET)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_cases(load_cases(args.set))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] == report["cases"] and report["p95_ms"] <= 3000 else 2


if __name__ == "__main__":
    raise SystemExit(main())
