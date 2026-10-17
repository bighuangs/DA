"""TODO-97 controlled SQL-route ablation and Pareto reporting."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from evals.legacy.compare_runs import load_experiment_runs
from app.infrastructure.clients.llm.capabilities import candidate_capability
from evals.legacy.governance import (
    MODEL_ABLATION_SCHEMA,
    compare_model_ablation_fingerprints,
    experiment_report,
)
from evals.paths import PROJECT_ROOT, REPORTS_ROOT


REPORT_DIR = REPORTS_ROOT / "model_ablation"
TODO97_EXECUTION_SCOPE = ("Q1", "Q3", "Q4")


def _passed_case_ids(runs: list[dict]) -> set[str]:
    return {
        str(case.get("id"))
        for run in runs
        for case in (run.get("cases") or [])
        if case.get("id") and case.get("result_equivalent") is True
    }


def _candidate_regressions(
    control_passed: set[str], candidate_runs: list[dict],
) -> list[str]:
    maps = [
        {
            str(case.get("id")): bool(case.get("result_equivalent"))
            for case in (run.get("cases") or []) if case.get("id")
        }
        for run in candidate_runs
    ]
    return sorted(
        case_id for case_id in control_passed
        if any(not case_map.get(case_id, False) for case_map in maps)
    )


def _route_profile(runs: list[dict]) -> dict[str, Any]:
    if not runs:
        return {}
    parameters = (runs[0].get("experiment") or {}).get("provider_parameters") or {}
    return dict(parameters.get("sql_route") or {})


def _validate_phase0_payload(label: str, payload: dict[str, Any]) -> None:
    expected = candidate_capability(label)
    candidate = payload.get("candidate") or {}
    profile = payload.get("profile") or {}
    gates = payload.get("gates") or {}
    preflight_names = {
        "registered_model", "provider_matches", "adapter_matches",
        "access_mode_matches", "thinking_mode_supported",
        "reasoning_effort_supported", "temperature_supported",
        "candidate_id_matches", "phase1_thinking_recipe_matches",
        "phase1_temperature_recipe_matches",
    }
    checks = {
        "schema": payload.get("schema_version") == "todo97-phase0-probe-v2",
        "registry": payload.get("capability_registry_version")
        == expected["registry_version"],
        "status": payload.get("status") in {"pass", "fail"},
        "repeat": int(payload.get("repeat") or 0) >= 3,
        "prompt_not_persisted": payload.get("prompt_persisted") is False,
        "response_not_persisted": payload.get("response_body_persisted") is False,
        "label": candidate.get("candidate_label") == label,
        "candidate_id": profile.get("candidate_id") == expected["candidate_id"],
        "model": profile.get("model_name") == expected["model_id"],
        "provider": profile.get("provider") == expected["provider"],
        "adapter": profile.get("model_provider") == expected["model_provider"],
        "access": profile.get("access_mode") == expected["access_mode"],
        "preflight": preflight_names.issubset(gates)
        and all(bool(gates.get(name)) for name in preflight_names),
        "attempt_evidence": float(
            (payload.get("summary") or {}).get("evidence_complete_rate") or 0
        ) == 1.0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            f"Phase0 artifact {label} 身份/证据不完整：{','.join(failed)}"
        )


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    quality_left = (
        int(left["worst_final_result"]), int(left["worst_first_result"]),
    )
    quality_right = (
        int(right["worst_final_result"]), int(right["worst_first_result"]),
    )
    costs_left = (
        float(left["ttcomplete_p95_ms"]), float(left["tokens_mean"]),
        int(left["failed_core_llm_calls"]),
    )
    costs_right = (
        float(right["ttcomplete_p95_ms"]), float(right["tokens_mean"]),
        int(right["failed_core_llm_calls"]),
    )
    no_worse = all(a >= b for a, b in zip(quality_left, quality_right)) and all(
        a <= b for a, b in zip(costs_left, costs_right)
    )
    strictly_better = any(a > b for a, b in zip(quality_left, quality_right)) or any(
        a < b for a, b in zip(costs_left, costs_right)
    )
    return no_worse and strictly_better


def model_ablation_report(
    control_runs: list[dict],
    candidate_groups: dict[str, list[dict]],
    *,
    phase0: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply TODO-97 shortlist gates without silently relaxing a failed check."""
    phase0 = phase0 or {}
    control = experiment_report(control_runs)
    control_route = _route_profile(control_runs)
    d0 = candidate_capability("D0")
    control_identity_matches = bool(control_route) and all((
        control_route.get("candidate_id") == d0["candidate_id"],
        control_route.get("model_name") == d0["model_id"],
        control_route.get("provider") == d0["provider"],
        control_route.get("model_provider") == d0["model_provider"],
        control_route.get("access_mode") == d0["access_mode"],
    ))
    control_phase0_passed = (phase0.get("D0") or {}).get("status") == "pass"
    control_recordable = bool(
        control.get("recordable")
        and control_identity_matches
        and control_phase0_passed
    )
    control_passed = _passed_case_ids(control_runs)
    control_fingerprint = (control_runs[0].get("fingerprint") or {}) if control_runs else {}
    control_worst = control.get("worst") or {}
    candidates: list[dict[str, Any]] = []
    for label in TODO97_EXECUTION_SCOPE:
        runs = candidate_groups.get(label) or []
        report = experiment_report(runs)
        comparisons = [
            compare_model_ablation_fingerprints(
                control_fingerprint, run.get("fingerprint") or {},
            )
            for run in runs
        ]
        route = _route_profile(runs)
        try:
            expected_identity = candidate_capability(label)
            identity_matches = bool(route) and all((
                route.get("candidate_id") == expected_identity["candidate_id"],
                route.get("model_name") == expected_identity["model_id"],
                route.get("provider") == expected_identity["provider"],
                route.get("model_provider") == expected_identity["model_provider"],
                route.get("access_mode") == expected_identity["access_mode"],
            ))
        except ValueError:
            identity_matches = False
        regressions = _candidate_regressions(control_passed, runs)
        worst = report.get("worst") or {}
        probe = phase0.get(label) or {}
        phase0_passed = probe.get("status") == "pass"
        checks = {
            "phase0_passed": phase0_passed,
            "candidate_identity_matches": identity_matches,
            "control_recordable": control_recordable,
            "candidate_recordable": report.get("recordable") is True,
            "controlled_attribution": bool(comparisons) and all(
                item.get("attribution_valid") is True for item in comparisons
            ),
            "final_non_regression": int(worst.get("result_equivalent") or 0)
            >= int(control_worst.get("result_equivalent") or 0),
            "d0_passed_cases_no_regression": not regressions,
            "first_final_evidence_complete": bool(
                (report.get("checks") or {}).get("first_final_evidence_complete")
            ),
            "security_guard_bypass_zero": int(
                worst.get("security_guard_bypass_count") or 0
            ) == 0,
            "failed_core_llm_calls_non_regression": int(
                worst.get("failed_core_llm_calls") or 0
            ) <= int(control_worst.get("failed_core_llm_calls") or 0),
            "ttsql_gate": bool((report.get("checks") or {}).get("ttsql_gate")),
            "ttresult_gate": bool((report.get("checks") or {}).get("ttresult_gate")),
            "ttcomplete_gate": bool((report.get("checks") or {}).get("ttcomplete_gate")),
            "token_max_gate": bool((report.get("checks") or {}).get("token_max_gate")),
            "token_mean_non_regression": (
                not float(control_worst.get("tokens_mean") or 0)
                or float(worst.get("tokens_mean") or 0)
                <= float(control_worst.get("tokens_mean") or 0) * 1.10
            ),
            "formal_promotion_identity": bool(route.get("promotion_eligible")),
        }
        within_candidate_recordable = report.get("recordable") is True
        ablation_recordable = bool(
            within_candidate_recordable
            and control_recordable
            and phase0_passed
            and identity_matches
            and checks["controlled_attribution"]
        )
        candidates.append({
            "candidate_label": label,
            "experiment_id": report.get("experiment_id"),
            "model_name": route.get("model_name"),
            "candidate_id": route.get("candidate_id"),
            "lifecycle": route.get("capability_lifecycle"),
            "within_candidate_recordable": within_candidate_recordable,
            "ablation_recordable": ablation_recordable,
            "recordable": ablation_recordable,
            "shortlisted": all(checks.values()),
            "checks": checks,
            "failed_checks": [key for key, value in checks.items() if not value],
            "regressed_case_ids": regressions,
            "allowed_differences": sorted({
                value for item in comparisons
                for value in item.get("allowed_differences") or []
            }),
            "unexpected_differences": sorted({
                value for item in comparisons
                for value in item.get("unexpected_differences") or []
            }),
            "worst_first_result": int(worst.get("first_result_equivalent") or 0),
            "worst_final_result": int(worst.get("result_equivalent") or 0),
            "ttcomplete_p95_ms": float(worst.get("tt_complete_p95") or 0),
            "tokens_mean": float(worst.get("tokens_mean") or 0),
            "tokens_max": float(worst.get("tokens_max") or 0),
            "failed_core_llm_calls": int(worst.get("failed_core_llm_calls") or 0),
            "stability_summary": report.get("stability_summary") or {},
        })
    eligible = [item for item in candidates if item["shortlisted"]]
    pareto = [
        item for item in eligible
        if not any(
            other is not item and _dominates(other, item) for other in eligible
        )
    ]
    pareto.sort(key=lambda item: (
        -item["worst_final_result"], -item["worst_first_result"],
        item["ttcomplete_p95_ms"], item["tokens_mean"],
    ))
    execution_complete = control_recordable and all(
        (
            (phase0.get(item["candidate_label"]) or {}).get("status") == "fail"
            and not candidate_groups.get(item["candidate_label"])
        )
        or (
            (phase0.get(item["candidate_label"]) or {}).get("status") == "pass"
            and item["ablation_recordable"]
        )
        for item in candidates
    )
    return {
        "schema": MODEL_ABLATION_SCHEMA,
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "control": {
            "experiment_id": control.get("experiment_id"),
            "within_candidate_recordable": control.get("recordable") is True,
            "identity_matches": control_identity_matches,
            "phase0_passed": control_phase0_passed,
            "recordable": control_recordable,
            "worst_first_result": int(control_worst.get("first_result_equivalent") or 0),
            "worst_final_result": int(control_worst.get("result_equivalent") or 0),
            "passed_case_ids": sorted(control_passed),
        },
        "candidates": candidates,
        "execution_scope": list(TODO97_EXECUTION_SCOPE),
        "explicitly_excluded_candidates": ["Q2", "Q4P", "X1"],
        "pareto_top2": [item["candidate_label"] for item in pareto[:2]],
        "phase2_candidates": ["D0", *[item["candidate_label"] for item in pareto[:2]]],
        "execution_complete": execution_complete,
        "cost_interpretation": "unknown-unless-provider-pricing-configured",
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# TODO-97 SQL 模型受控消融",
        "",
        f"> 合同：`{report['schema']}` ｜ D0：`{report['control'].get('experiment_id')}`",
        "",
        "| 候选 | First最差 | Final最差 | TTComplete P95 | Token均值/最大 | 核心失败 | 可记录 | Shortlist |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for item in report.get("candidates") or []:
        lines.append(
            f"| {item['candidate_label']} / {item.get('model_name')} | "
            f"{item['worst_first_result']} | {item['worst_final_result']} | "
            f"{item['ttcomplete_p95_ms']:.1f} ms | {item['tokens_mean']:.1f} / "
            f"{item['tokens_max']:.0f} | {item['failed_core_llm_calls']} | "
            f"{'✅' if item['recordable'] else '❌'} | "
            f"{'✅' if item['shortlisted'] else '❌'} |"
        )
    lines.extend(["", "## 门禁与归因", ""])
    for item in report.get("candidates") or []:
        lines.extend([
            f"### {item['candidate_label']}",
            "",
            f"- 未通过：{', '.join(item['failed_checks']) or '无'}",
            f"- D0 已通过题回退：{', '.join(item['regressed_case_ids']) or '无'}",
            f"- 允许差异：{', '.join(item['allowed_differences']) or '无'}",
            f"- 意外差异：{', '.join(item['unexpected_differences']) or '无'}",
            "",
        ])
    lines.extend([
        "## Pareto 与后续",
        "",
        f"- Pareto Top 2：{', '.join(report.get('pareto_top2') or []) or '无'}",
        f"- Phase 2 候选：{', '.join(report.get('phase2_candidates') or [])}",
        "- 费用：价格未配置时为未知，不解释为 0 元。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="TODO-97 model-ablation-v1 比较")
    parser.add_argument("--suite", default="hard_sql_v1")
    parser.add_argument("--control-experiment", required=True)
    parser.add_argument(
        "--candidate-experiments", required=True,
        help="逗号分隔 LABEL=experiment_id，例如 Q1=id1,Q3=id2",
    )
    parser.add_argument(
        "--phase0-artifacts", required=True,
        help="逗号分隔 LABEL=probe.json；Phase0 缺证一律失败",
    )
    args = parser.parse_args()
    groups: dict[str, list[dict]] = {}
    for entry in str(args.candidate_experiments).split(","):
        label, separator, experiment_id = entry.partition("=")
        if not separator or not label.strip() or not experiment_id.strip():
            parser.error("candidate-experiments 必须使用 LABEL=experiment_id")
        groups[label.strip().upper()] = load_experiment_runs(
            experiment_id.strip(), args.suite,
        )
    if set(groups) != set(TODO97_EXECUTION_SCOPE):
        parser.error("candidate-experiments 必须显式包含且仅包含 Q1、Q3、Q4")
    phase0: dict[str, dict[str, Any]] = {}
    for entry in str(args.phase0_artifacts).split(","):
        label, separator, artifact = entry.partition("=")
        if not separator or not label.strip() or not artifact.strip():
            parser.error("phase0-artifacts 必须使用 LABEL=probe.json")
        path = Path(artifact.strip())
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        payload = json.loads(path.read_text(encoding="utf-8"))
        actual_label = str((payload.get("candidate") or {}).get("candidate_label") or "")
        if actual_label != label.strip().upper():
            parser.error(f"Phase0 artifact 候选不匹配：{label} != {actual_label}")
        normalized_label = label.strip().upper()
        try:
            _validate_phase0_payload(normalized_label, payload)
        except ValueError as exc:
            parser.error(str(exc))
        phase0[normalized_label] = payload
    if set(phase0) != {"D0", *TODO97_EXECUTION_SCOPE}:
        parser.error("phase0-artifacts 必须显式包含 D0、Q1、Q3、Q4")
    report = model_ablation_report(
        load_experiment_runs(args.control_experiment, args.suite), groups,
        phase0=phase0,
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = REPORT_DIR / f"todo97_ablation_{timestamp}.json"
    md_path = REPORT_DIR / f"todo97_ablation_{timestamp}.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    (REPORT_DIR / "todo97_ablation_latest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (REPORT_DIR / "todo97_ablation_latest.md").write_text(
        render_markdown(report), encoding="utf-8",
    )
    print(render_markdown(report))
    return 0 if report.get("execution_complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
