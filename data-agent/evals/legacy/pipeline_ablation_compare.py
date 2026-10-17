"""为 pipeline 单因子消融生成不可变、脱敏的 Control×Candidate 对比制品。"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

from evals.legacy.dataset_suite import SUITE_PATHS
from evals.legacy.governance import (
    compare_pipeline_ablation_fingerprints,
    experiment_report,
    stable_hash,
)
from evals.legacy.run_eval import _percentile
from evals.paths import REPORTS_ROOT, RUNS_ROOT


RUNS_DIR = RUNS_ROOT
OUTPUT_DIR = REPORTS_ROOT / "pipeline_ablation"
ARTIFACT_SCHEMA = "pipeline-factor-comparison-v1"
SUPPORTED_FACTORS = (
    "query_signals",
    "planning_route",
    "semantic_repair",
)
EXPECTED_REPEAT_CACHE = ((1, "cold"), (2, "warm"), (3, "warm"))
PAIR_METRIC_FIELDS = (
    "first_result_equivalent",
    "first_exec_success",
    "result_equivalent",
    "exec_success",
    "tables_correct",
    "table_scope_safe",
    "llm_attempts",
    "failed_core_llm_calls",
    "tt_complete_p95",
    "tokens_mean",
    "tokens_max",
    "estimated_cost_total",
)
CASE_COUNT_FIELDS = (
    "result_equivalent",
    "exec_success",
    "tables_correct",
    "table_scope_safe",
)


class ComparisonRejected(ValueError):
    """表示输入制品不能形成可审计的严格单因子归因。"""


def load_experiment_runs(
    runs_dir: Path,
    *,
    experiment_id: str,
    dataset_suite: str,
) -> list[dict[str, Any]]:
    """只加载精确 experiment 与 suite，不读取 ``latest`` 或基线注册表。"""

    output: list[dict[str, Any]] = []
    if not runs_dir.exists():
        return output
    for path in sorted(runs_dir.glob("run_*.json")):
        try:
            source_bytes = path.read_bytes()
            payload = json.loads(source_bytes.decode("utf-8"))
        except (OSError, UnicodeError, ValueError):
            continue
        experiment = payload.get("experiment") or {}
        if (
            experiment.get("experiment_id") == experiment_id
            and str(payload.get("dataset_suite") or "") == dataset_suite
        ):
            payload["_artifact"] = path.name
            payload["_artifact_sha256"] = (
                "sha256:" + hashlib.sha256(source_bytes).hexdigest()
            )
            output.append(payload)
    return output


def _group_index(
    runs: Iterable[dict[str, Any]],
    *,
    experiment_id: str,
    side: str,
) -> dict[int, dict[str, Any]]:
    """建立 repeat 索引，并在进入 fingerprint 比较前拒绝重复或错序。"""

    indexed: dict[int, dict[str, Any]] = {}
    for run in runs:
        metadata = run.get("experiment") or {}
        if metadata.get("experiment_id") != experiment_id:
            raise ComparisonRejected(f"{side}.experiment_id_mismatch")
        try:
            repeat_index = int(metadata.get("repeat_index"))
        except (TypeError, ValueError) as exc:
            raise ComparisonRejected(f"{side}.repeat_index_invalid") from exc
        if repeat_index in indexed:
            raise ComparisonRejected(f"{side}.repeat_index_duplicate:{repeat_index}")
        indexed[repeat_index] = run

    observed = tuple(
        (
            repeat_index,
            str((indexed[repeat_index].get("experiment") or {}).get("cache_state") or ""),
        )
        for repeat_index in sorted(indexed)
    )
    if observed != EXPECTED_REPEAT_CACHE:
        raise ComparisonRejected(
            f"{side}.repeat_cache_protocol_mismatch:{observed!r}"
        )
    return indexed


def _validate_group_report(report: dict[str, Any], *, side: str) -> None:
    """只接受完整且可记录的三轮组；qualified 不是形成事实记录的前提。"""

    if report.get("available") is not True:
        raise ComparisonRejected(f"{side}.experiment_unavailable")
    if report.get("complete") is not True:
        raise ComparisonRejected(f"{side}.experiment_incomplete")
    if report.get("recordable") is not True:
        raise ComparisonRejected(f"{side}.experiment_not_recordable")
    if int(report.get("repeat_count") or 0) != len(EXPECTED_REPEAT_CACHE):
        raise ComparisonRejected(f"{side}.repeat_count_invalid")


def _cross_group_identity_checks(
    control: dict[int, dict[str, Any]],
    candidate: dict[int, dict[str, Any]],
    *,
    dataset_suite: str,
) -> None:
    """校验不属于 fingerprint 因子的实验身份，防止错组或跨 workspace 配对。"""

    fields = ("dataset_suite", "workspace_id")
    for repeat_index, _ in EXPECTED_REPEAT_CACHE:
        left = control[repeat_index]
        right = candidate[repeat_index]
        if (
            left.get("dataset_suite") != dataset_suite
            or right.get("dataset_suite") != dataset_suite
        ):
            raise ComparisonRejected(
                f"pair[{repeat_index}].dataset_suite_unexpected"
            )
        for field in fields:
            if left.get(field) != right.get(field):
                raise ComparisonRejected(
                    f"pair[{repeat_index}].{field}_mismatch"
                )
        left_metadata = left.get("experiment") or {}
        right_metadata = right.get("experiment") or {}
        for field in ("candidate_label", "cache_state"):
            if left_metadata.get(field) != right_metadata.get(field):
                raise ComparisonRejected(
                    f"pair[{repeat_index}].{field}_mismatch"
                )


def _safe_group_summary(report: dict[str, Any]) -> dict[str, Any]:
    """保留发布决策所需汇总，不复制逐题问题、SQL 或 Trace。"""

    return {
        key: report.get(key)
        for key in (
            "experiment_id",
            "candidate_label",
            "evaluation_profile",
            "workspace_id",
            "status",
            "complete",
            "recordable",
            "qualified",
            "repeat_count",
            "embedding_protocol_mode",
            "failed_checks",
            "failed_observations",
            "fingerprint_differences",
            "worst",
            "stability_summary",
        )
    }


def _run_metrics(run: dict[str, Any]) -> dict[str, Any]:
    """从 case 重算核心指标，并拒绝与 aggregate 不一致的输入。"""

    def finite_float(value: Any, *, field: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ComparisonRejected(f"run.metric_invalid:{field}") from exc
        if not math.isfinite(number):
            raise ComparisonRejected(f"run.metric_invalid:{field}")
        return number

    aggregate = run.get("aggregate") or {}
    cases = run.get("cases") or []
    if not isinstance(cases, list) or not cases:
        raise ComparisonRejected("run.cases_missing")
    declared_n = int(run.get("n") or 0)
    if declared_n != len(cases):
        raise ComparisonRejected(
            f"run.case_count_mismatch:{declared_n}!={len(cases)}"
        )
    case_ids = [str(case.get("id") or "") for case in cases]
    if "" in case_ids or len(case_ids) != len(set(case_ids)):
        raise ComparisonRejected("run.case_ids_invalid")

    recomputed_counts = {
        field: sum(bool(case.get(field)) for case in cases)
        for field in CASE_COUNT_FIELDS
    }
    for field in CASE_COUNT_FIELDS:
        if any(field not in case for case in cases):
            raise ComparisonRejected(f"run.case_metric_missing:{field}")
    first_sql_metrics = [case.get("first_sql_metrics") for case in cases]
    if any(not isinstance(item, dict) for item in first_sql_metrics):
        raise ComparisonRejected("run.first_sql_metrics_missing")
    for field in ("result_equivalent", "exec_success"):
        if any(field not in item for item in first_sql_metrics):
            raise ComparisonRejected(f"run.first_sql_metric_missing:{field}")

    def llm_attempt_count(case: dict[str, Any]) -> int:
        value = case.get("llm_attempts")
        if isinstance(value, list):
            return len(value)
        try:
            count = int(value or 0)
        except (TypeError, ValueError) as exc:
            raise ComparisonRejected("run.case_metric_invalid:llm_attempts") from exc
        if count < 0:
            raise ComparisonRejected("run.case_metric_invalid:llm_attempts")
        return count

    recomputed_counts.update({
        "first_result_equivalent": sum(
            bool(item.get("result_equivalent")) for item in first_sql_metrics
        ),
        "first_exec_success": sum(
            bool(item.get("exec_success")) for item in first_sql_metrics
        ),
        "llm_attempts": sum(llm_attempt_count(case) for case in cases),
        "failed_core_llm_calls": sum(
            int(case.get("failed_core_llm_calls") or 0) for case in cases
        ),
    })
    for field, expected in recomputed_counts.items():
        if field not in aggregate:
            raise ComparisonRejected(f"run.aggregate_metric_missing:{field}")
        try:
            actual = int(aggregate[field])
        except (TypeError, ValueError) as exc:
            raise ComparisonRejected(
                f"run.aggregate_metric_invalid:{field}"
            ) from exc
        if actual != expected:
            raise ComparisonRejected(
                f"run.aggregate_case_mismatch:{field}:{actual}!={expected}"
            )

    tt_complete = sorted(
        finite_float(
            case.get("tt_complete_ms") or case.get("latency_ms") or 0,
            field="tt_complete_ms",
        )
        for case in cases
    )
    if any(value <= 0 for value in tt_complete):
        raise ComparisonRejected("run.case_metric_invalid:tt_complete_ms")
    tt_complete_p95 = _percentile(tt_complete, 0.95)
    for field in ("tt_complete_p95", "latency_p95"):
        try:
            raw_value = aggregate[field]
        except KeyError as exc:
            raise ComparisonRejected(
                f"run.aggregate_metric_invalid:{field}"
            ) from exc
        actual = finite_float(raw_value, field=f"aggregate.{field}")
        if actual != tt_complete_p95:
            raise ComparisonRejected(
                f"run.aggregate_case_mismatch:{field}:{actual}!={tt_complete_p95}"
            )

    token_values = [int(case.get("total_tokens") or 0) for case in cases]
    tokens_total = sum(token_values)
    tokens_mean = tokens_total / len(cases)
    tokens_max = max(token_values, default=0)
    for field, expected in (
        ("tokens_mean", round(tokens_mean, 1)),
        ("tokens_max", tokens_max),
    ):
        try:
            raw_value = aggregate[field]
        except KeyError as exc:
            raise ComparisonRejected(
                f"run.aggregate_metric_invalid:{field}"
            ) from exc
        actual = finite_float(raw_value, field=f"aggregate.{field}")
        if abs(actual - expected) > 1e-6:
            raise ComparisonRejected(
                f"run.aggregate_case_mismatch:{field}:{actual}!={expected}"
            )

    case_costs: list[float] = []
    case_currencies: list[str] = []
    all_attempts: list[dict[str, Any]] = []
    priced_attempt_observed = False
    for case in cases:
        try:
            raw_case_cost = case["estimated_cost"]
        except KeyError as exc:
            raise ComparisonRejected(
                "run.case_metric_invalid:estimated_cost"
            ) from exc
        case_cost = finite_float(raw_case_cost, field="case.estimated_cost")
        if case_cost < 0:
            raise ComparisonRejected("run.case_metric_invalid:estimated_cost")
        currency = str(case.get("cost_currency") or "")
        if not currency:
            raise ComparisonRejected("run.case_metric_invalid:cost_currency")
        attempts = case.get("llm_attempts")
        if not isinstance(attempts, list):
            raise ComparisonRejected("run.case_metric_invalid:llm_attempts")
        attempt_cost = 0.0
        for attempt in attempts:
            if not isinstance(attempt, dict):
                raise ComparisonRejected("run.case_metric_invalid:llm_attempts")
            if "token_known" not in attempt or "estimated_cost" not in attempt:
                raise ComparisonRejected("run.case_cost_evidence_incomplete")
            value = finite_float(
                attempt["estimated_cost"],
                field="attempt.estimated_cost",
            )
            if value < 0:
                raise ComparisonRejected(
                    "run.case_metric_invalid:attempt_estimated_cost"
                )
            attempt_cost += value
            priced_attempt_observed = priced_attempt_observed or value > 0
            all_attempts.append(attempt)
        expected_case_cost = round(attempt_cost, 8)
        if round(case_cost, 8) != expected_case_cost:
            raise ComparisonRejected(
                "run.attempt_case_cost_mismatch:"
                f"{round(case_cost, 8)}!={expected_case_cost}"
            )
        case_costs.append(case_cost)
        case_currencies.append(currency)

    if not all_attempts:
        raise ComparisonRejected("run.cost_availability_evidence_missing")
    currencies = set(case_currencies)
    if len(currencies) != 1:
        raise ComparisonRejected("run.case_cost_currency_mixed")
    cost_currency = case_currencies[0]
    estimated_cost_total = round(sum(case_costs), 8)
    unknown_tokens = sum(not bool(item.get("token_known")) for item in all_attempts)
    cost_estimate_available = priced_attempt_observed and unknown_tokens == 0
    expected_cost_fields: dict[str, Any] = {
        "estimated_cost_total": estimated_cost_total,
        "cost_currency": cost_currency,
        "cost_estimate_available": cost_estimate_available,
        "llm_attempt_unknown_tokens": unknown_tokens,
    }
    for field, expected in expected_cost_fields.items():
        if field not in aggregate:
            raise ComparisonRejected(f"run.aggregate_metric_missing:{field}")
        actual = aggregate[field]
        if field == "estimated_cost_total":
            actual = round(
                finite_float(actual, field=f"aggregate.{field}"),
                8,
            )
        if actual != expected:
            raise ComparisonRejected(
                f"run.aggregate_case_mismatch:{field}:{actual}!={expected}"
            )

    metrics = {
        field: aggregate.get(field)
        for field in PAIR_METRIC_FIELDS
    }
    metrics.update({
        "n": declared_n,
        "tokens_total": tokens_total,
        "tt_complete_p95": tt_complete_p95,
        "estimated_cost_total": estimated_cost_total,
        "metrics_verified_from_cases": True,
        "cost_estimate_available": cost_estimate_available,
        "cost_currency": cost_currency,
    })
    return metrics


def _source_binding(run: dict[str, Any]) -> dict[str, Any]:
    """绑定完整输入 payload；hash 会覆盖问题/SQL，但不会在报告中泄露原文。"""

    payload = {
        key: value for key, value in run.items()
        if not str(key).startswith("_")
    }
    artifact = str(run.get("_artifact") or "")
    artifact_sha256 = str(run.get("_artifact_sha256") or "")
    run_id = str(run.get("run_id") or "")
    if not artifact or not run_id:
        raise ComparisonRejected("run.source_identity_missing")
    digest = artifact_sha256.removeprefix("sha256:")
    if (
        not artifact_sha256.startswith("sha256:")
        or len(digest) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in digest)
    ):
        raise ComparisonRejected("run.source_artifact_sha256_missing")
    return {
        "run_id": run_id,
        "artifact": artifact,
        "artifact_sha256": artifact_sha256.lower(),
        "payload_fingerprint": stable_hash(payload),
    }


def _numeric_delta(
    control: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, int | float]:
    output: dict[str, int | float] = {}
    for key in sorted(set(control) & set(candidate)):
        left = control.get(key)
        right = candidate.get(key)
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
        ):
            output[key] = right - left
    return output


def build_factor_comparison(
    control_runs: list[dict[str, Any]],
    candidate_runs: list[dict[str, Any]],
    *,
    factor: str,
    dataset_suite: str,
    control_experiment_id: str,
    candidate_experiment_id: str,
    generated_at: dt.datetime | None = None,
) -> dict[str, Any]:
    """生成严格 Control×Candidate 配对；任一治理门失败即拒绝返回报告。"""

    if factor not in SUPPORTED_FACTORS:
        raise ComparisonRejected(f"unsupported_factor:{factor}")
    if not control_experiment_id or not candidate_experiment_id:
        raise ComparisonRejected("experiment_id_missing")
    if control_experiment_id == candidate_experiment_id:
        raise ComparisonRejected("control_candidate_experiment_same")

    control_report = experiment_report(
        control_runs,
        experiment_id=control_experiment_id,
    )
    candidate_report = experiment_report(
        candidate_runs,
        experiment_id=candidate_experiment_id,
    )
    _validate_group_report(control_report, side="control")
    _validate_group_report(candidate_report, side="candidate")

    control = _group_index(
        control_runs,
        experiment_id=control_experiment_id,
        side="control",
    )
    candidate = _group_index(
        candidate_runs,
        experiment_id=candidate_experiment_id,
        side="candidate",
    )
    _cross_group_identity_checks(
        control,
        candidate,
        dataset_suite=dataset_suite,
    )

    pairs: list[dict[str, Any]] = []
    for repeat_index, cache_state in EXPECTED_REPEAT_CACHE:
        left = control[repeat_index]
        right = candidate[repeat_index]
        attribution = compare_pipeline_ablation_fingerprints(
            left.get("fingerprint"),
            right.get("fingerprint"),
            factor=factor,
        )
        if attribution.get("attribution_valid") is not True:
            reasons = ",".join(
                str(item)
                for item in attribution.get("unexpected_differences") or []
            ) or "unknown"
            raise ComparisonRejected(
                f"pair[{repeat_index}].attribution_invalid:{reasons}"
            )
        control_metrics = _run_metrics(left)
        candidate_metrics = _run_metrics(right)
        pairs.append({
            "repeat_index": repeat_index,
            "cache_state": cache_state,
            "control_source": _source_binding(left),
            "candidate_source": _source_binding(right),
            "attribution": attribution,
            "control_metrics": control_metrics,
            "candidate_metrics": candidate_metrics,
            "candidate_minus_control": _numeric_delta(
                control_metrics,
                candidate_metrics,
            ),
        })

    first_control = control[1].get("fingerprint") or {}
    first_candidate = candidate[1].get("fingerprint") or {}
    created = generated_at or dt.datetime.now(dt.UTC).astimezone()
    report: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA,
        "generated_at": created.isoformat(timespec="microseconds"),
        "dataset_suite": dataset_suite,
        "factor": factor,
        "control_experiment_id": control_experiment_id,
        "candidate_experiment_id": candidate_experiment_id,
        "attribution_valid": True,
        "control": _safe_group_summary(control_report),
        "candidate": _safe_group_summary(candidate_report),
        "control_pipeline_profiles": first_control.get("pipeline_profiles") or {},
        "candidate_pipeline_profiles": first_candidate.get("pipeline_profiles") or {},
        "pairs": pairs,
        "privacy": {
            "raw_questions": False,
            "raw_sql": False,
            "raw_vectors": False,
            "secrets": False,
        },
        "limitations": [
            "latency remains observational when run artifacts disallow attribution",
            "cost amount remains unknown when provider price mapping is unavailable",
            (
                "cost availability is reconstructed from observed positive attempt cost; "
                "zero-attempt or configured-price-but-zero-cost artifacts fail closed"
            ),
            "Full Live attribution does not itself approve production rollout",
        ],
    }
    report["report_fingerprint"] = stable_hash(report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    """渲染与 JSON 同源的紧凑审计报告。"""

    control = report.get("control") or {}
    candidate = report.get("candidate") or {}
    lines = [
        f"# Pipeline 单因子对比：{report.get('factor')}",
        "",
        (
            f"> 合同：`{report.get('schema_version')}` ｜ "
            f"归因有效：`{report.get('attribution_valid')}` ｜ "
            f"报告指纹：`{report.get('report_fingerprint')}`"
        ),
        "",
        "## 实验组",
        "",
        "| 角色 | Experiment | 状态 | Complete | Recordable | Qualified |",
        "|---|---|---|---:|---:|---:|",
        (
            f"| Control | `{report.get('control_experiment_id')}` | "
            f"{control.get('status')} | {control.get('complete')} | "
            f"{control.get('recordable')} | {control.get('qualified')} |"
        ),
        (
            f"| Candidate | `{report.get('candidate_experiment_id')}` | "
            f"{candidate.get('status')} | {candidate.get('complete')} | "
            f"{candidate.get('recordable')} | {candidate.get('qualified')} |"
        ),
        "",
        "## Repeat 配对",
        "",
        "| Repeat | Cache | Control Final | Candidate Final | Delta | Token Delta | 允许差异 |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for pair in report.get("pairs") or []:
        left = pair.get("control_metrics") or {}
        right = pair.get("candidate_metrics") or {}
        delta = pair.get("candidate_minus_control") or {}
        allowed = ", ".join(
            (pair.get("attribution") or {}).get("allowed_differences") or []
        )
        lines.append(
            f"| {pair.get('repeat_index')} | {pair.get('cache_state')} | "
            f"{left.get('result_equivalent')}/{left.get('n')} | "
            f"{right.get('result_equivalent')}/{right.get('n')} | "
            f"{delta.get('result_equivalent')} | {delta.get('tokens_total')} | "
            f"{allowed or '无'} |"
        )
    lines.extend([
        "",
        "## 已知限制",
        "",
    ])
    lines.extend(
        f"- {item}" for item in report.get("limitations") or []
    )
    lines.extend([
        "",
        "## 证据边界",
        "",
        "- JSON 与本文来自同一汇总对象；没有复制问题、SQL、向量或 Secret。",
        "- `recordable` 证明协议与证据完整，`qualified` 才是发布质量门；二者不可互换。",
        "- 本报告只证明已注册 pipeline 因子的配对归因，不批准生产切换。",
        "",
    ])
    return "\n".join(lines)


def write_artifacts(
    report: dict[str, Any],
    *,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[Path, Path]:
    """只写时间戳文件，不创建或覆盖 ``latest`` 指针。"""

    generated_at = dt.datetime.fromisoformat(str(report["generated_at"]))
    stamp = generated_at.strftime("%Y%m%d_%H%M%S_%f")
    factor = str(report.get("factor") or "unknown")
    stem = f"pipeline_ablation_{factor}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError(f"不可覆盖已有不可变制品：{stem}")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="生成严格 pipeline 单因子 Control×Candidate 不可变对比",
    )
    parser.add_argument("--factor", choices=SUPPORTED_FACTORS, required=True)
    parser.add_argument("--suite", choices=tuple(SUITE_PATHS), required=True)
    parser.add_argument("--control-experiment", required=True)
    parser.add_argument("--candidate-experiment", required=True)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    control_runs = load_experiment_runs(
        args.runs_dir,
        experiment_id=args.control_experiment,
        dataset_suite=args.suite,
    )
    candidate_runs = load_experiment_runs(
        args.runs_dir,
        experiment_id=args.candidate_experiment,
        dataset_suite=args.suite,
    )
    try:
        report = build_factor_comparison(
            control_runs,
            candidate_runs,
            factor=args.factor,
            dataset_suite=args.suite,
            control_experiment_id=args.control_experiment,
            candidate_experiment_id=args.candidate_experiment,
        )
        json_path, markdown_path = write_artifacts(
            report,
            output_dir=args.output_dir,
        )
    except (ComparisonRejected, FileExistsError) as exc:
        print(f"pipeline factor comparison rejected: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({
        "schema_version": report["schema_version"],
        "factor": report["factor"],
        "attribution_valid": report["attribution_valid"],
        "report_fingerprint": report["report_fingerprint"],
        "json": str(json_path),
        "markdown": str(markdown_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
