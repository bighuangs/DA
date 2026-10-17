"""Offline TODO-97 evidence matrix for the three SQL difficulty suites.

This command only reads existing ``var/evals/legacy/runs/run_*.json`` artifacts.  It does
not construct an LLM client, execute SQL, mutate a baseline, or write a report.
It deliberately reuses :func:`evals.legacy.governance.experiment_report` for the
three-repeat protocol and stability contract, then adds the evidence views that
the generic experiment CLI does not expose:

* per-repeat First / Final / Correction / timeout counts;
* stable case intersections across cold/warm/warm repeats;
* repeat-aligned deltas against the DeepSeek (D0) control;
* every Final failure, identified without copying generated or expected SQL.

Example (one ``--experiment`` per matrix cell)::

    uv run python -m evals.legacy.summarize_todo97_matrix \
      --candidate-labels D0,Q1,Q3 \
      --experiment easy_sql_v1:D0=todo97-d0-easy-v1 \
      --experiment easy_sql_v1:Q1=todo97-q1-easy-v1 \
      --experiment easy_sql_v1:Q3=todo97-q3-easy-v1 \
      --experiment medium_sql_v1:D0=todo97-d0-medium-v1 \
      --experiment medium_sql_v1:Q1=todo97-q1-medium-v1 \
      --experiment medium_sql_v1:Q3=todo97-q3-medium-v1 \
      --experiment hard_sql_v1:D0=todo97-d0-hard-v1 \
      --experiment hard_sql_v1:Q1=todo97-q1-hard-v1 \
      --experiment hard_sql_v1:Q3=todo97-q3-hard-v1

Use ``--format json`` when the output will feed another auditable report.
Incomplete or non-recordable matrices exit with status 2 unless
``--allow-incomplete`` is explicitly selected.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from typing import Any, Iterable

from evals.paths import PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.legacy.compare_runs import load_experiment_runs  # noqa: E402
from evals.legacy.governance import experiment_report  # noqa: E402


DEFAULT_SUITES = ("easy_sql_v1", "medium_sql_v1", "hard_sql_v1")
SUMMARY_SCHEMA = "todo97-offline-evidence-matrix-v1"


def parse_csv(value: str) -> list[str]:
    """Return non-empty, de-duplicated CSV values while preserving order."""
    result: list[str] = []
    for raw in str(value or "").split(","):
        item = raw.strip()
        if item and item not in result:
            result.append(item)
    return result


def parse_experiment_specs(values: Iterable[str]) -> dict[tuple[str, str], str]:
    """Parse ``SUITE:LABEL=experiment_id`` cells and reject ambiguity."""
    cells: dict[tuple[str, str], str] = {}
    for raw in values:
        left, separator, experiment_id = str(raw or "").partition("=")
        suite, label_separator, label = left.rpartition(":")
        suite = suite.strip()
        label = label.strip().upper()
        experiment_id = experiment_id.strip()
        if not separator or not label_separator or not all(
            (suite, label, experiment_id)
        ):
            raise ValueError(
                "--experiment 必须使用 SUITE:LABEL=experiment_id"
            )
        key = (suite, label)
        if key in cells:
            raise ValueError(f"重复的实验矩阵单元：{suite}:{label}")
        cells[key] = experiment_id
    return cells


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool_count(cases: list[dict], path: tuple[str, ...]) -> tuple[int, int]:
    """Return ``(true_count, evidence_count)`` for a nested boolean field."""
    true_count = 0
    evidence_count = 0
    for case in cases:
        current: Any = case
        found = True
        for key in path:
            if not isinstance(current, dict) or key not in current:
                found = False
                break
            current = current[key]
        if found and isinstance(current, bool):
            evidence_count += 1
            true_count += int(current)
    return true_count, evidence_count


def _is_timeout_attempt(attempt: dict) -> bool:
    explicit_values = {
        str(attempt.get("outcome_class") or "").strip().lower(),
        str(attempt.get("status") or "").strip().lower(),
        str(attempt.get("error_type") or "").strip().lower(),
    }
    if explicit_values & {"timeout", "timed_out", "time_out"}:
        return True
    reason = str(attempt.get("reason_code") or "").strip().lower()
    return reason.startswith("llm.timeout") or reason.startswith("deadline.timeout")


def _timeout_evidence(cases: list[dict], aggregate: dict) -> dict[str, Any]:
    derived_attempts = 0
    case_ids: list[str] = []
    for case in cases:
        count = sum(
            _is_timeout_attempt(attempt)
            for attempt in (case.get("llm_attempts") or [])
            if isinstance(attempt, dict)
        )
        if count:
            derived_attempts += count
            case_ids.append(str(case.get("id") or ""))
    aggregate_value = _int_or_none(aggregate.get("llm_attempts_timed_out"))
    if aggregate_value is None:
        aggregate_value = _int_or_none(
            (aggregate.get("llm_outcome_classes") or {}).get("timeout")
        )
    return {
        "attempts": aggregate_value,
        "derived_attempts": derived_attempts,
        "case_count": len(case_ids),
        "case_ids": sorted(case_id for case_id in case_ids if case_id),
        "request_deadline_exhausted_cases": _int_or_none(
            aggregate.get("request_deadline_exhausted_cases")
        ),
        "aggregate_matches_case_evidence": (
            None if aggregate_value is None else aggregate_value == derived_attempts
        ),
    }


def _metric_block(
    cases: list[dict], aggregate: dict, *, prefix: str,
) -> dict[str, Any]:
    if prefix == "first":
        exec_path = ("first_sql_metrics", "exec_success")
        evaluated_path = ("first_sql_metrics", "result_evaluated")
        equivalent_path = ("first_sql_metrics", "result_equivalent")
        aggregate_exec = "first_exec_success"
        aggregate_equivalent = "first_result_equivalent"
    else:
        exec_path = ("final_sql_metrics", "exec_success")
        evaluated_path = ("final_sql_metrics", "result_evaluated")
        equivalent_path = ("final_sql_metrics", "result_equivalent")
        aggregate_exec = "exec_success"
        aggregate_equivalent = "result_equivalent"
    exec_count, exec_evidence = _bool_count(cases, exec_path)
    evaluated_count, evaluated_evidence = _bool_count(cases, evaluated_path)
    equivalent_count, equivalent_evidence = _bool_count(cases, equivalent_path)
    aggregate_exec_value = _int_or_none(aggregate.get(aggregate_exec))
    aggregate_equivalent_value = _int_or_none(
        aggregate.get(aggregate_equivalent)
    )
    complete = all(
        coverage == len(cases)
        for coverage in (exec_evidence, evaluated_evidence, equivalent_evidence)
    )
    return {
        "exec_success": aggregate_exec_value,
        "result_evaluated": (
            evaluated_count if evaluated_evidence == len(cases) else None
        ),
        "result_equivalent": aggregate_equivalent_value,
        "case_evidence": {
            "exec_success": exec_count,
            "result_evaluated": evaluated_count,
            "result_equivalent": equivalent_count,
            "coverage": min(exec_evidence, evaluated_evidence, equivalent_evidence),
            "complete": complete,
        },
        "aggregate_matches_case_evidence": {
            "exec_success": (
                None if aggregate_exec_value is None or not complete
                else aggregate_exec_value == exec_count
            ),
            "result_equivalent": (
                None if aggregate_equivalent_value is None or not complete
                else aggregate_equivalent_value == equivalent_count
            ),
        },
    }


def _correction_block(cases: list[dict], aggregate: dict) -> dict[str, Any]:
    fields = {
        "triggered": "correction_triggered",
        "fixed_execution": "correction_fixed_execution",
        "fixed_result": "correction_fixed_result",
        "harmful_modification": "correction_harmful_modification",
    }
    block: dict[str, Any] = {}
    coverage: dict[str, int] = {}
    consistency: dict[str, bool | None] = {}
    for field, aggregate_field in fields.items():
        derived, evidence = _bool_count(cases, ("correction_metrics", field))
        aggregate_value = _int_or_none(aggregate.get(aggregate_field))
        block[field] = aggregate_value
        coverage[field] = evidence
        consistency[field] = (
            None if aggregate_value is None or evidence != len(cases)
            else aggregate_value == derived
        )
    block.update({
        "unattributable": _int_or_none(
            aggregate.get("correction_unattributable")
        ),
        "llm_attempts": _int_or_none(aggregate.get("correction_llm_attempts")),
        "additional_tokens": _int_or_none(
            aggregate.get("correction_additional_tokens")
        ),
        "unknown_token_attempts": _int_or_none(
            aggregate.get("correction_unknown_token_attempts")
        ),
        "additional_latency_ms": _int_or_none(
            aggregate.get("correction_additional_latency_ms")
        ),
        "case_evidence_coverage": coverage,
        "aggregate_matches_case_evidence": consistency,
    })
    return block


def _failure_status(case: dict) -> str:
    timeout_count = sum(
        _is_timeout_attempt(attempt)
        for attempt in (case.get("llm_attempts") or [])
        if isinstance(attempt, dict)
    )
    if timeout_count:
        return "timeout_observed"
    final = case.get("final_sql_metrics") or {}
    if final.get("guard_passed") is False:
        return "guard_rejected"
    if final.get("exec_success") is False or case.get("exec_success") is False:
        return "execution_failed"
    if final.get("result_evaluated") is not True:
        return "result_not_evaluated"
    return "result_mismatch"


def _short_text(value: Any, limit: int = 280) -> str | None:
    if value in (None, ""):
        return None
    normalized = " ".join(str(value).split())
    return normalized if len(normalized) <= limit else normalized[: limit - 1] + "…"


def _failure_cases(cases: list[dict]) -> list[dict[str, Any]]:
    failed: list[dict[str, Any]] = []
    for case in cases:
        final = case.get("final_sql_metrics") or {}
        final_passed = (
            final.get("result_equivalent") is True
            if final else case.get("result_equivalent") is True
        )
        if final_passed:
            continue
        first = case.get("first_sql_metrics") or {}
        correction = case.get("correction_metrics") or {}
        attempts = [
            attempt for attempt in (case.get("llm_attempts") or [])
            if isinstance(attempt, dict)
        ]
        failed.append({
            "case_id": str(case.get("id") or ""),
            "category": str(case.get("category") or "unclassified"),
            "question": _short_text(case.get("question")),
            "status": _failure_status(case),
            "first_exec_success": first.get("exec_success"),
            "first_result_equivalent": first.get("result_equivalent"),
            "final_exec_success": (
                final.get("exec_success")
                if final else case.get("exec_success")
            ),
            "final_result_evaluated": (
                final.get("result_evaluated")
                if final else case.get("result_evaluated")
            ),
            "final_result_equivalent": (
                final.get("result_equivalent")
                if final else case.get("result_equivalent")
            ),
            "correction_triggered": correction.get("triggered"),
            "correction_attribution_status": correction.get(
                "attribution_status"
            ),
            "correction_outcome": correction.get("outcome"),
            "failure_stage": _short_text(case.get("failure_stage"), 100),
            "failure_type": _short_text(case.get("failure_type"), 100),
            "error": _short_text(case.get("error")),
            "timeout_attempts": sum(_is_timeout_attempt(item) for item in attempts),
            "failed_core_llm_calls": _int_or_none(
                case.get("failed_core_llm_calls")
            ),
        })
    return failed


def _run_summary(run: dict) -> dict[str, Any]:
    metadata = run.get("experiment") or {}
    aggregate = run.get("aggregate") or {}
    cases = list(run.get("cases") or [])
    return {
        "run_id": str(run.get("run_id") or ""),
        "artifact": str(run.get("_artifact") or ""),
        "repeat_index": _int_or_none(metadata.get("repeat_index")),
        "cache_state": str(metadata.get("cache_state") or ""),
        "valid": run.get("valid") is True,
        "total": _int_or_none(run.get("n")) or len(cases),
        "first": _metric_block(cases, aggregate, prefix="first"),
        "final": _metric_block(cases, aggregate, prefix="final"),
        "correction": _correction_block(cases, aggregate),
        "timeout": _timeout_evidence(cases, aggregate),
        "failed_core_llm_calls": _int_or_none(
            aggregate.get("failed_core_llm_calls")
        ),
        "infrastructure_errors": _int_or_none(
            aggregate.get("infrastructure_errors")
        ),
        "final_failure_cases": _failure_cases(cases),
    }


def _case_map(runs: list[dict]) -> dict[str, list[dict]]:
    mapped: dict[str, list[dict]] = {}
    for run in runs:
        for case in run.get("cases") or []:
            case_id = str(case.get("id") or "")
            if case_id:
                mapped.setdefault(case_id, []).append(case)
    return mapped


def _stable_intersections(runs: list[dict], protocol: dict) -> dict[str, Any]:
    stability = list(protocol.get("stability") or [])
    first_stable_pass = sorted(
        str(item.get("case_id")) for item in stability
        if item.get("first_status") == "stable_pass"
    )
    final_stable_pass = sorted(
        str(item.get("case_id")) for item in stability
        if item.get("status") == "stable_pass"
    )
    final_stable_fail = sorted(
        str(item.get("case_id")) for item in stability
        if item.get("status") == "stable_fail"
    )
    mapped = _case_map(runs)
    repeat_count = len(runs)

    def correction_count(case_id: str, field: str) -> int:
        return sum(
            (case.get("correction_metrics") or {}).get(field) is True
            for case in mapped.get(case_id, [])
        )

    correction_triggered_all = sorted(
        case_id for case_id in mapped
        if repeat_count and correction_count(case_id, "triggered") == repeat_count
    )
    correction_fixed_all = sorted(
        case_id for case_id in mapped
        if repeat_count and correction_count(case_id, "fixed_result") == repeat_count
    )
    correction_harmful_any = sorted(
        case_id for case_id in mapped
        if correction_count(case_id, "harmful_modification") > 0
    )
    timeout_any = sorted({
        str(case.get("id"))
        for run in runs
        for case in (run.get("cases") or [])
        if any(
            _is_timeout_attempt(attempt)
            for attempt in (case.get("llm_attempts") or [])
            if isinstance(attempt, dict)
        )
    })
    return {
        "first_stable_pass_case_ids": first_stable_pass,
        "final_stable_pass_case_ids": final_stable_pass,
        "first_and_final_stable_pass_case_ids": sorted(
            set(first_stable_pass) & set(final_stable_pass)
        ),
        "final_stable_fail_case_ids": final_stable_fail,
        "final_flaky_case_ids": sorted(
            str(item.get("case_id")) for item in stability
            if item.get("status") == "flaky"
        ),
        "correction_triggered_all_repeats_case_ids": correction_triggered_all,
        "correction_fixed_result_all_repeats_case_ids": correction_fixed_all,
        "correction_harmful_any_repeat_case_ids": correction_harmful_any,
        "timeout_any_repeat_case_ids": timeout_any,
    }


def _route_identity(runs: list[dict]) -> dict[str, Any]:
    if not runs:
        return {}
    route = (
        ((runs[0].get("experiment") or {}).get("provider_parameters") or {})
        .get("sql_route") or {}
    )
    allowed = (
        "candidate_id", "model_name", "model_revision", "provider",
        "model_provider", "access_mode", "thinking_mode", "reasoning_effort",
        "temperature", "promotion_eligible", "capability_lifecycle",
    )
    return {key: route.get(key) for key in allowed}


def summarize_experiment(
    suite: str, label: str, experiment_id: str, runs: list[dict],
) -> dict[str, Any]:
    protocol = experiment_report(runs, experiment_id=experiment_id)
    actual_labels = sorted({
        str((run.get("experiment") or {}).get("candidate_label") or "")
        for run in runs
    })
    identity_matches = actual_labels == [label]
    sorted_runs = sorted(
        runs,
        key=lambda run: int(
            (run.get("experiment") or {}).get("repeat_index") or 0
        ),
    )
    return {
        "suite": suite,
        "candidate_label": label,
        "experiment_id": experiment_id,
        "identity": _route_identity(sorted_runs),
        "experiment_candidate_labels": actual_labels,
        "candidate_label_matches": identity_matches,
        "protocol": {
            "status": protocol.get("status"),
            "complete": protocol.get("complete") is True,
            "recordable": protocol.get("recordable") is True,
            "qualified": protocol.get("qualified") is True,
            "repeat_count": _int_or_none(protocol.get("repeat_count")),
            "failed_checks": list(protocol.get("failed_checks") or []),
            "stability_summary": dict(
                protocol.get("stability_summary") or {}
            ),
        },
        "runs": [_run_summary(run) for run in sorted_runs],
        "stable_intersections": _stable_intersections(sorted_runs, protocol),
    }


def _repeat_deltas(control: dict, candidate: dict) -> list[dict[str, Any]]:
    control_runs = {
        item.get("repeat_index"): item for item in control.get("runs") or []
    }
    candidate_runs = {
        item.get("repeat_index"): item for item in candidate.get("runs") or []
    }
    deltas: list[dict[str, Any]] = []

    def delta(left: Any, right: Any) -> int | None:
        if left is None or right is None:
            return None
        return int(right) - int(left)

    for repeat in sorted(set(control_runs) & set(candidate_runs)):
        base = control_runs[repeat]
        trial = candidate_runs[repeat]
        deltas.append({
            "repeat_index": repeat,
            "cache_state_matches": base.get("cache_state") == trial.get("cache_state"),
            "first_result_equivalent_delta": delta(
                (base.get("first") or {}).get("result_equivalent"),
                (trial.get("first") or {}).get("result_equivalent"),
            ),
            "final_result_equivalent_delta": delta(
                (base.get("final") or {}).get("result_equivalent"),
                (trial.get("final") or {}).get("result_equivalent"),
            ),
            "correction_fixed_result_delta": delta(
                (base.get("correction") or {}).get("fixed_result"),
                (trial.get("correction") or {}).get("fixed_result"),
            ),
            "timeout_attempts_delta": delta(
                (base.get("timeout") or {}).get("attempts"),
                (trial.get("timeout") or {}).get("attempts"),
            ),
        })
    return deltas


def _control_comparison(control: dict, candidate: dict) -> dict[str, Any]:
    base = control.get("stable_intersections") or {}
    trial = candidate.get("stable_intersections") or {}
    first_base = set(base.get("first_stable_pass_case_ids") or [])
    first_trial = set(trial.get("first_stable_pass_case_ids") or [])
    final_base = set(base.get("final_stable_pass_case_ids") or [])
    final_trial = set(trial.get("final_stable_pass_case_ids") or [])
    return {
        "candidate_label": candidate.get("candidate_label"),
        "repeat_deltas_candidate_minus_control": _repeat_deltas(
            control, candidate
        ),
        "first_stable_pass_intersection_case_ids": sorted(
            first_base & first_trial
        ),
        "first_control_only_stable_pass_case_ids": sorted(
            first_base - first_trial
        ),
        "first_candidate_only_stable_pass_case_ids": sorted(
            first_trial - first_base
        ),
        "final_stable_pass_intersection_case_ids": sorted(
            final_base & final_trial
        ),
        "final_control_only_stable_pass_case_ids": sorted(
            final_base - final_trial
        ),
        "final_candidate_only_stable_pass_case_ids": sorted(
            final_trial - final_base
        ),
    }


def build_matrix(
    specs: dict[tuple[str, str], str], *, suites: list[str],
    candidate_labels: list[str], baseline_label: str = "D0",
    loader=load_experiment_runs,
) -> dict[str, Any]:
    """Build a factual evidence matrix; no model/promotion conclusion is made."""
    baseline_label = baseline_label.strip().upper()
    expected_cells = {
        (suite, label) for suite in suites for label in candidate_labels
    }
    supplied_cells = set(specs)
    missing = sorted(f"{suite}:{label}" for suite, label in expected_cells - supplied_cells)
    unexpected = sorted(
        f"{suite}:{label}" for suite, label in supplied_cells - expected_cells
    )
    groups: list[dict[str, Any]] = []
    for suite in suites:
        for label in candidate_labels:
            experiment_id = specs.get((suite, label))
            if not experiment_id:
                continue
            runs = loader(experiment_id, suite)
            groups.append(summarize_experiment(
                suite, label, experiment_id, runs,
            ))
    comparisons: list[dict[str, Any]] = []
    for suite in suites:
        by_label = {
            item["candidate_label"]: item
            for item in groups if item["suite"] == suite
        }
        control = by_label.get(baseline_label)
        if not control:
            continue
        comparisons.append({
            "suite": suite,
            "control_label": baseline_label,
            "candidates": [
                _control_comparison(control, by_label[label])
                for label in candidate_labels
                if label != baseline_label and label in by_label
            ],
        })
    group_evidence_ready = all(
        item.get("candidate_label_matches") is True
        and (item.get("protocol") or {}).get("recordable") is True
        for item in groups
    )
    matrix_complete = not missing and not unexpected and len(groups) == len(
        expected_cells
    )
    return {
        "schema": SUMMARY_SCHEMA,
        "generated_at": dt.datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "source": "var/evals/legacy/runs/run_*.json (offline, read-only)",
        "suites": suites,
        "candidate_labels": candidate_labels,
        "baseline_label": baseline_label,
        "matrix_complete": matrix_complete,
        "evidence_ready": matrix_complete and group_evidence_ready,
        "missing_cells": missing,
        "unexpected_cells": unexpected,
        "groups": groups,
        "control_comparisons": comparisons,
        "interpretation_contract": {
            "first_final_source": "experiment_report + case evidence",
            "timeout_source": "llm-attempt-v2 outcome/status with aggregate reconciliation",
            "stable_intersection": "case passes all available repeats; protocol completeness is reported separately",
            "correction_attribution": "only fixed_result from persisted correction_metrics; unattributable remains separate",
            "promotion_conclusion": "out_of_scope",
        },
    }


def _cell(value: Any) -> str:
    return "未知" if value is None else str(value)


def _md_text(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# TODO-97 三档离线评测证据矩阵",
        "",
        f"> 合同：`{report['schema']}` ｜ 矩阵完整：{report['matrix_complete']} "
        f"｜ 证据可记录：{report['evidence_ready']}",
        "",
    ]
    if report.get("missing_cells"):
        lines.append(
            "- 缺失矩阵单元：" + ", ".join(report["missing_cells"])
        )
    if report.get("unexpected_cells"):
        lines.append(
            "- 额外矩阵单元：" + ", ".join(report["unexpected_cells"])
        )
    for suite in report.get("suites") or []:
        lines.extend([f"## {suite}", ""])
        suite_groups = [
            item for item in report.get("groups") or []
            if item.get("suite") == suite
        ]
        for group in suite_groups:
            protocol = group.get("protocol") or {}
            identity = group.get("identity") or {}
            lines.extend([
                f"### {group['candidate_label']} / "
                f"{_md_text(identity.get('model_name') or '未知模型')}",
                "",
                f"- Experiment：`{_md_text(group['experiment_id'])}`；"
                f"状态：`{_md_text(protocol.get('status'))}`；"
                f"recordable：{protocol.get('recordable')}。",
                "",
                "| Repeat | Cache | First result | Final result | Correction trigger/fix/harm | Timeout attempts/cases | Final failures |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ])
            for run in group.get("runs") or []:
                first = run.get("first") or {}
                final = run.get("final") or {}
                correction = run.get("correction") or {}
                timeout = run.get("timeout") or {}
                lines.append(
                    f"| {_cell(run.get('repeat_index'))} | "
                    f"{_md_text(run.get('cache_state'))} | "
                    f"{_cell(first.get('result_equivalent'))}/{_cell(run.get('total'))} | "
                    f"{_cell(final.get('result_equivalent'))}/{_cell(run.get('total'))} | "
                    f"{_cell(correction.get('triggered'))}/"
                    f"{_cell(correction.get('fixed_result'))}/"
                    f"{_cell(correction.get('harmful_modification'))} | "
                    f"{_cell(timeout.get('attempts'))}/"
                    f"{_cell(timeout.get('case_count'))} | "
                    f"{len(run.get('final_failure_cases') or [])} |"
                )
            stable = group.get("stable_intersections") or {}
            lines.extend([
                "",
                "- First 三轮稳定通过：" + (
                    ", ".join(stable.get("first_stable_pass_case_ids") or [])
                    or "无"
                ),
                "- Final 三轮稳定通过：" + (
                    ", ".join(stable.get("final_stable_pass_case_ids") or [])
                    or "无"
                ),
                "- Final 三轮稳定失败：" + (
                    ", ".join(stable.get("final_stable_fail_case_ids") or [])
                    or "无"
                ),
                "- Correction 三轮均修复严格结果：" + (
                    ", ".join(
                        stable.get(
                            "correction_fixed_result_all_repeats_case_ids"
                        ) or []
                    ) or "无"
                ),
                "",
            ])
            for run in group.get("runs") or []:
                failures = run.get("final_failure_cases") or []
                if not failures:
                    continue
                lines.extend([
                    f"#### Repeat {run.get('repeat_index')} Final 失败样例（全量）",
                    "",
                    "| Case | 类别 | 状态 | First | Final exec | Correction | Timeout | 问题 |",
                    "| --- | --- | --- | --- | --- | --- | ---: | --- |",
                ])
                for case in failures:
                    lines.append(
                        f"| {_md_text(case.get('case_id'))} | "
                        f"{_md_text(case.get('category'))} | "
                        f"{_md_text(case.get('status'))} | "
                        f"{_cell(case.get('first_result_equivalent'))} | "
                        f"{_cell(case.get('final_exec_success'))} | "
                        f"{_md_text(case.get('correction_outcome')) or '未知'} | "
                        f"{_cell(case.get('timeout_attempts'))} | "
                        f"{_md_text(case.get('question'))} |"
                    )
                lines.append("")
        suite_comparison = next((
            item for item in report.get("control_comparisons") or []
            if item.get("suite") == suite
        ), None)
        if suite_comparison:
            lines.extend(["### 与 DeepSeek 控制组的稳定交集", ""])
            for item in suite_comparison.get("candidates") or []:
                lines.extend([
                    f"- {item.get('candidate_label')} Final 共同稳定通过："
                    + (", ".join(
                        item.get("final_stable_pass_intersection_case_ids") or []
                    ) or "无"),
                    f"- {item.get('candidate_label')} 相对 D0 丢失的稳定通过："
                    + (", ".join(
                        item.get("final_control_only_stable_pass_case_ids") or []
                    ) or "无"),
                    f"- {item.get('candidate_label')} 相对 D0 新增的稳定通过："
                    + (", ".join(
                        item.get("final_candidate_only_stable_pass_case_ids") or []
                    ) or "无"),
                ])
            lines.append("")
    lines.extend([
        "## 解释边界",
        "",
        "- 本报告只汇总持久化 run；不调用模型、不执行 SQL、不修改 baseline。",
        "- timeout 同时给出 aggregate 与逐 attempt 重算的一致性；未知值不按 0 处理。",
        "- Correction 仅按已持久化的直接归因字段计数，不把不可归因路径算作修复。",
        "- 本报告不作模型晋级或能力边界结论；结论须另行结合 Phase0、指纹和门禁。",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="TODO-97 三档多候选离线证据汇总（只读）"
    )
    parser.add_argument(
        "--experiment", action="append", required=True,
        help="可重复：SUITE:LABEL=experiment_id",
    )
    parser.add_argument(
        "--suites", default=",".join(DEFAULT_SUITES),
        help="默认 easy_sql_v1,medium_sql_v1,hard_sql_v1",
    )
    parser.add_argument(
        "--candidate-labels", default=None,
        help="显式候选矩阵，例如 D0,Q1,Q3；省略时从参数推导",
    )
    parser.add_argument("--baseline-label", default="D0")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="仅用于中途核对；允许不完整/不可记录矩阵返回 0",
    )
    args = parser.parse_args(argv)
    try:
        specs = parse_experiment_specs(args.experiment)
    except ValueError as exc:
        parser.error(str(exc))
    suites = parse_csv(args.suites)
    if not suites:
        parser.error("--suites 不能为空")
    candidate_labels = (
        [item.upper() for item in parse_csv(args.candidate_labels)]
        if args.candidate_labels
        else sorted({label for _, label in specs})
    )
    if not candidate_labels:
        parser.error("候选列表不能为空")
    baseline_label = str(args.baseline_label or "").strip().upper()
    if baseline_label not in candidate_labels:
        parser.error("baseline-label 必须出现在 candidate-labels 中")
    report = build_matrix(
        specs,
        suites=suites,
        candidate_labels=candidate_labels,
        baseline_label=baseline_label,
    )
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(report))
    return 0 if args.allow_incomplete or report.get("evidence_ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
