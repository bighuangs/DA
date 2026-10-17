import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "evals"
    / "legacy"
    / "summarize_todo97_matrix.py"
)
SPEC = importlib.util.spec_from_file_location("todo97_matrix_summary", SCRIPT)
assert SPEC and SPEC.loader
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def _case(
    case_id: str, *, first: bool, final: bool,
    correction_triggered: bool = False,
    correction_fixed: bool = False,
    timeout: bool = False,
) -> dict:
    return {
        "id": case_id,
        "category": "fixture",
        "question": f"question {case_id}",
        "exec_success": True,
        "result_evaluated": True,
        "result_equivalent": final,
        "first_sql_metrics": {
            "sql": "SELECT must-not-leak",
            "exec_success": True,
            "result_evaluated": True,
            "result_equivalent": first,
        },
        "final_sql_metrics": {
            "sql": "SELECT must-not-leak",
            "exec_success": True,
            "result_evaluated": True,
            "result_equivalent": final,
        },
        "correction_metrics": {
            "triggered": correction_triggered,
            "fixed_execution": correction_fixed,
            "fixed_result": correction_fixed,
            "harmful_modification": False,
            "attribution_status": (
                "direct" if correction_triggered else "not_triggered"
            ),
            "outcome": (
                "fixed_result" if correction_fixed else "not_triggered"
            ),
        },
        "llm_attempts": ([{
            "outcome_class": "timeout",
            "status": "failed",
            "reason_code": "llm.timeout",
        }] if timeout else [{
            "outcome_class": "success",
            "status": "success",
            "reason_code": "llm.success",
        }]),
        "failed_core_llm_calls": int(timeout),
    }


def _run(label: str, repeat: int, *, candidate_failure: bool) -> dict:
    timeout = candidate_failure and repeat == 2
    cases = [
        _case("case-a", first=True, final=True),
        _case(
            "case-b",
            first=not candidate_failure,
            final=not candidate_failure,
            correction_triggered=candidate_failure,
            timeout=timeout,
        ),
    ]
    return {
        "run_id": f"{label.lower()}-{repeat}",
        "_artifact": f"run_{label.lower()}_{repeat}.json",
        "n": 2,
        "valid": True,
        "dataset_suite": "easy_sql_v1",
        "experiment": {
            "experiment_id": f"{label.lower()}-easy",
            "candidate_label": label,
            "repeat_index": repeat,
            "cache_state": "cold" if repeat == 1 else "warm",
            "provider_parameters": {"sql_route": {
                "candidate_id": f"candidate-{label.lower()}",
                "model_name": "deepseek-v4-pro" if label == "D0" else "qwen",
                "provider": "deepseek" if label == "D0" else "dashscope",
            }},
        },
        "aggregate": {
            "first_exec_success": 2,
            "first_result_equivalent": 1 if candidate_failure else 2,
            "exec_success": 2,
            "result_equivalent": 1 if candidate_failure else 2,
            "correction_triggered": int(candidate_failure),
            "correction_fixed_execution": 0,
            "correction_fixed_result": 0,
            "correction_harmful_modification": 0,
            "correction_unattributable": 0,
            "correction_llm_attempts": int(candidate_failure),
            "correction_additional_tokens": 0,
            "correction_unknown_token_attempts": 0,
            "correction_additional_latency_ms": 0,
            "llm_attempts_timed_out": int(timeout),
            "llm_outcome_classes": {"timeout": int(timeout)},
            "request_deadline_exhausted_cases": 0,
            "failed_core_llm_calls": int(timeout),
            "infrastructure_errors": 0,
        },
        "cases": cases,
    }


def _protocol(runs, experiment_id=None):
    case_ids = [case["id"] for case in runs[0]["cases"]] if runs else []
    stability = []
    for case_id in case_ids:
        rows = [
            next(case for case in run["cases"] if case["id"] == case_id)
            for run in runs
        ]
        first_passed = sum(
            (case.get("first_sql_metrics") or {}).get("result_equivalent")
            is True
            for case in rows
        )
        final_passed = sum(
            case["final_sql_metrics"]["result_equivalent"] for case in rows
        )
        stability.append({
            "case_id": case_id,
            "first_status": (
                "stable_pass" if first_passed == len(rows)
                else "stable_fail" if first_passed == 0 else "flaky"
            ),
            "status": (
                "stable_pass" if final_passed == len(rows)
                else "stable_fail" if final_passed == 0 else "flaky"
            ),
        })
    return {
        "status": "recordable",
        "complete": True,
        "recordable": True,
        "qualified": False,
        "repeat_count": len(runs),
        "failed_checks": ["fixture_quality_gate"],
        "stability": stability,
        "stability_summary": {"total": len(stability)},
    }


def test_parse_experiment_specs_rejects_duplicate_cells():
    with pytest.raises(ValueError, match="重复"):
        summary.parse_experiment_specs([
            "easy_sql_v1:D0=one",
            "easy_sql_v1:D0=two",
        ])


def test_matrix_reports_each_repeat_timeout_failures_and_control_intersection(
    monkeypatch,
):
    runs = {
        ("d0-easy", "easy_sql_v1"): [
            _run("D0", repeat, candidate_failure=False)
            for repeat in (1, 2, 3)
        ],
        ("q1-easy", "easy_sql_v1"): [
            _run("Q1", repeat, candidate_failure=True)
            for repeat in (1, 2, 3)
        ],
    }
    monkeypatch.setattr(summary, "experiment_report", _protocol)
    report = summary.build_matrix(
        {
            ("easy_sql_v1", "D0"): "d0-easy",
            ("easy_sql_v1", "Q1"): "q1-easy",
        },
        suites=["easy_sql_v1"],
        candidate_labels=["D0", "Q1"],
        loader=lambda experiment_id, suite: runs[(experiment_id, suite)],
    )

    assert report["matrix_complete"]
    assert report["evidence_ready"]
    q1 = next(
        item for item in report["groups"]
        if item["candidate_label"] == "Q1"
    )
    assert [run["first"]["result_equivalent"] for run in q1["runs"]] == [1, 1, 1]
    assert [run["final"]["result_equivalent"] for run in q1["runs"]] == [1, 1, 1]
    assert [run["correction"]["triggered"] for run in q1["runs"]] == [1, 1, 1]
    assert [run["timeout"]["attempts"] for run in q1["runs"]] == [0, 1, 0]
    assert all(
        len(run["final_failure_cases"]) == 1 for run in q1["runs"]
    )
    assert q1["stable_intersections"]["final_stable_pass_case_ids"] == [
        "case-a"
    ]
    assert q1["stable_intersections"]["final_stable_fail_case_ids"] == [
        "case-b"
    ]
    comparison = report["control_comparisons"][0]["candidates"][0]
    assert comparison["final_stable_pass_intersection_case_ids"] == ["case-a"]
    assert comparison["final_control_only_stable_pass_case_ids"] == ["case-b"]
    assert "SELECT must-not-leak" not in json.dumps(report, ensure_ascii=False)


def test_missing_first_evidence_and_timeout_remain_unknown(monkeypatch):
    run = _run("D0", 1, candidate_failure=False)
    run["aggregate"].pop("llm_attempts_timed_out")
    run["aggregate"]["llm_outcome_classes"].pop("timeout")
    run["aggregate"].pop("first_result_equivalent")
    run["cases"][0].pop("first_sql_metrics")
    monkeypatch.setattr(summary, "experiment_report", _protocol)

    result = summary.summarize_experiment(
        "easy_sql_v1", "D0", "d0-easy", [run]
    )

    first = result["runs"][0]["first"]
    timeout = result["runs"][0]["timeout"]
    assert first["result_equivalent"] is None
    assert not first["case_evidence"]["complete"]
    assert timeout["attempts"] is None
    assert timeout["derived_attempts"] == 0


def test_three_suite_default_shape_detects_missing_cells(monkeypatch):
    monkeypatch.setattr(summary, "experiment_report", _protocol)
    report = summary.build_matrix(
        {("easy_sql_v1", "D0"): "d0-easy"},
        suites=list(summary.DEFAULT_SUITES),
        candidate_labels=["D0"],
        loader=lambda experiment_id, suite: [
            _run("D0", repeat, candidate_failure=False)
            for repeat in (1, 2, 3)
        ],
    )
    assert not report["matrix_complete"]
    assert report["missing_cells"] == [
        "hard_sql_v1:D0", "medium_sql_v1:D0",
    ]
