"""Read-only service exposing workspace-scoped NL2SQL evaluation artifacts."""
from __future__ import annotations

import datetime as dt
import inspect
import json
import secrets
import threading
from pathlib import Path
from typing import Any, ClassVar

from evals.legacy.attribution import enrich_results, summarize_attributions
from evals.legacy.compare_runs import run_payload_is_valid
from evals.legacy.dataset_suite import SUITE_SPECS, load_suite_cases
from evals.legacy.governance import (
    baseline_checklist,
    compare_fingerprints,
    experiment_report,
    is_explicit_nonrelease,
    resolve_baseline_artifact,
    sample_contract_report,
)
from evals.legacy.profiles import evaluation_profile_name
from evals.application.ports.baseline_registry import EvaluationBaselineRegistry
from evals.paths import RUNS_ROOT

DIFFICULTY_ORDER = ("easy", "medium", "hard")

# Evaluation artifacts are durable and may outlive the code that produced them.
# Never forward a stored route dictionary wholesale: older or malformed artifacts
# could contain a raw endpoint or credential variable.  The quality API exposes
# only the non-secret, versioned identity required to compare model candidates.
_SAFE_MODEL_PROFILE_FIELDS = (
    "candidate_id",
    "model_name",
    "model_revision",
    "provider",
    "model_provider",
    "access_mode",
    "endpoint_profile",
    "endpoint_id",
    "deployment_id",
    "adapter_version",
    "capability_registry_version",
    "capability_lifecycle",
    "comparison_mode",
    "promotion_eligible",
    "thinking_mode",
    "reasoning_effort",
    "temperature",
    "timeout_seconds",
    "max_retries",
    "max_completion_tokens",
    "max_concurrency",
)

_SAFE_SQL_METRIC_FIELDS = (
    "non_empty",
    "syntax_valid",
    "read_only_safe",
    "sql_equivalent",
    "structure_equivalent",
    "tables_correct",
    "table_precision",
    "table_recall",
    "table_f1",
    "guard_passed",
    "guard_error_code",
    "execution_attempted",
    "exec_success",
    "result_evaluated",
    "result_equivalent",
    "execution_error_type",
)

_SAFE_CORRECTION_METRIC_FIELDS = (
    "triggered",
    "attribution_valid",
    "attribution_status",
    "trigger_reason_codes",
    "retry_count",
    "correction_output_count",
    "sql_revision_count",
    "llm_attempts",
    "failed_llm_attempts",
    "additional_tokens",
    "additional_tokens_are_lower_bound",
    "unknown_token_attempts",
    "additional_latency_ms",
    "fixed_execution",
    "fixed_result",
    "harmful_modification",
    "outcome",
)


def _suite_difficulty(suite: str) -> str:
    return str((SUITE_SPECS.get(suite) or {}).get("difficulty") or "unknown")


def _suite_label(suite: str) -> str:
    return str((SUITE_SPECS.get(suite) or {}).get("display_name") or suite)


def _baseline_suite_for_difficulty(difficulty: str | None) -> str:
    for name, spec in SUITE_SPECS.items():
        if (
            spec.get("difficulty") == difficulty
            and spec.get("baseline_policy") == "observational_three_repeat"
        ):
            return name
    return "easy_sql_v1"


def _allowlisted(value: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    return {field: source[field] for field in fields if field in source}


def _safe_model_profile(value: Any) -> dict[str, Any]:
    return _allowlisted(value, _SAFE_MODEL_PROFILE_FIELDS)


def _resolved_model_profile(payload: dict, route: str = "sql") -> dict[str, Any]:
    """Resolve a route identity from new and historical artifact layouts."""
    experiment = payload.get("experiment") or {}
    provider_parameters = experiment.get("provider_parameters") or {}
    fingerprint_model = (payload.get("fingerprint") or {}).get("model") or {}
    stage_manifest = (
        (payload.get("stage_evidence") or {}).get("llm_route_manifest") or {}
    )
    candidates = (
        provider_parameters.get(f"{route}_route"),
        (fingerprint_model.get("routes") or {}).get(route)
        if isinstance(fingerprint_model, dict) else None,
        stage_manifest.get(route) if isinstance(stage_manifest, dict) else None,
    )
    for candidate in candidates:
        profile = _safe_model_profile(candidate)
        if profile:
            return profile

    # Older experiment artifacts stored a few SQL fields at the provider-
    # parameter root.  They remain useful when passed through the same allowlist.
    return _safe_model_profile(provider_parameters)


def _safe_stage_evidence(value: Any) -> dict[str, Any]:
    evidence = dict(value) if isinstance(value, dict) else {}
    manifest = evidence.get("llm_route_manifest")
    if isinstance(manifest, dict):
        evidence["llm_route_manifest"] = {
            str(route): _safe_model_profile(profile)
            for route, profile in manifest.items()
            if isinstance(profile, dict)
        }
    return evidence


class EvaluationService:
    _summary_cache: ClassVar[
        dict[str, tuple[int, int, dict[str, Any]]]
    ] = {}
    _summary_cache_lock = threading.Lock()

    def __init__(
        self,
        runs_dir: Path | None = None,
        registry: EvaluationBaselineRegistry | None = None,
    ):
        self.runs_dir = runs_dir or RUNS_ROOT
        self.registry = registry

    async def _registry_call(self, name: str, *args, **kwargs):
        if self.registry is None:
            raise RuntimeError("baseline 治理需要 Meta MySQL 会话")
        value = getattr(self.registry, name)(*args, **kwargs)
        return await value if inspect.isawaitable(value) else value

    async def approved_entries(self, workspace_id: str) -> list[dict]:
        """通过 baseline 端口返回工作空间已批准条目。"""
        return await self._registry_call("approved_all", workspace_id)

    def _runs(
        self, workspace_id: str, *, include_invalid: bool = False,
        difficulty: str | None = None,
    ) -> list[dict[str, Any]]:
        return [
            payload
            for summary in self._run_summaries(
                workspace_id, include_invalid=include_invalid,
                difficulty=difficulty,
            )
            if (payload := self._load_run(summary)) is not None
        ]

    @classmethod
    def _read_run_summary(cls, path: Path) -> dict[str, Any] | None:
        """Read the top-level run metadata without materializing multi-MB case evidence.

        ``run_eval`` writes ``cases`` after fingerprints and aggregates. The admin
        catalog only needs the preceding fields, so closing the JSON object at that
        boundary avoids repeatedly parsing every case on each page load.
        """
        try:
            stat = path.stat()
        except OSError:
            return None
        cache_key = str(path.resolve())
        with cls._summary_cache_lock:
            cached = cls._summary_cache.get(cache_key)
            if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
                return dict(cached[2])

        lines: list[str] = []
        found_cases = False
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith('  "cases"'):
                        found_cases = True
                        break
                    lines.append(line)
            if found_cases:
                prefix = "".join(lines).rstrip()
                prefix = prefix.removesuffix(",")
                payload = json.loads(f"{prefix}\n}}")
                if "valid" not in payload:
                    return None
            else:
                payload = json.loads("".join(lines))
        except (OSError, ValueError, TypeError):
            return None

        payload["_artifact"] = path.name
        with cls._summary_cache_lock:
            cls._summary_cache[cache_key] = (stat.st_mtime_ns, stat.st_size, dict(payload))
        return payload

    def _run_summaries(
        self, workspace_id: str, *, include_invalid: bool = False,
        difficulty: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self.runs_dir.exists():
            return []
        rows: list[dict[str, Any]] = []
        for path in sorted(self.runs_dir.glob("run_*.json"), reverse=True):
            payload = self._read_run_summary(path)
            if payload is None or str(payload.get("workspace_id") or "") != workspace_id:
                continue
            if is_explicit_nonrelease(payload):
                continue
            dataset_suite = str(payload.get("dataset_suite") or "")
            if dataset_suite not in SUITE_SPECS:
                continue
            if difficulty and _suite_difficulty(dataset_suite) != difficulty:
                continue
            if not include_invalid and not run_payload_is_valid(payload):
                continue
            rows.append(payload)
        return rows

    def _load_run(self, summary: dict[str, Any]) -> dict[str, Any] | None:
        artifact = str(summary.get("_artifact") or "")
        if not artifact or Path(artifact).name != artifact:
            return None
        path = self.runs_dir / artifact
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        payload["_artifact"] = artifact
        return payload

    @staticmethod
    def _is_formal_experiment(summary: dict[str, Any]) -> bool:
        experiment = summary.get("experiment") or {}
        contract = (summary.get("fingerprint") or {}).get("sample_contract") or {}
        total = int(summary.get("n") or 0)
        return bool(
            experiment.get("experiment_id")
            and experiment.get("repeat_index") in range(1, 4)
            and experiment.get("cache_state") in {"cold", "warm"}
            and total > 0
            and total == int(contract.get("total") or 0)
            and contract.get("complete") is True
            and contract.get("all_reviewed") is True
        )

    @classmethod
    def _default_run_summary(cls, runs: list[dict[str, Any]]) -> dict[str, Any]:
        # Files are already newest-first. Prefer a reviewed full-suite experiment.
        formal = next((item for item in runs if cls._is_formal_experiment(item)), None)
        if formal is not None:
            return formal
        return runs[0]

    async def list_runs(
        self, workspace_id: str, limit: int = 20,
        difficulty: str | None = None,
    ) -> list[dict]:
        rows = []
        for item in self._run_summaries(
            workspace_id, difficulty=difficulty,
        )[:limit]:
            header = self._run_header(item)
            entry = await self._registry_call("get", workspace_id, str(item.get("run_id")))
            header["baseline_status"] = (entry or {}).get("status") or "unreviewed"
            header["fingerprint_complete"] = bool((item.get("fingerprint") or {}).get("complete"))
            rows.append(header)
        return rows

    async def _approved_selection(
        self, workspace_id: str, dataset_suite: str,
    ) -> tuple[dict | None, dict]:
        approved_entry = await self._registry_call(
            "approved", workspace_id, dataset_suite,
        )
        selection = resolve_baseline_artifact(
            workspace_id=workspace_id, approved_entry=approved_entry, runs_dir=self.runs_dir,
            expected_dataset_suite=dataset_suite,
        )
        return approved_entry, selection

    async def list_experiments(self, workspace_id: str) -> list[dict]:
        runs = self._runs(workspace_id, include_invalid=True)
        experiment_keys = sorted({
            (
                str(run.get("dataset_suite") or ""),
                str((run.get("experiment") or {}).get("experiment_id")),
            )
            for run in runs
            if (run.get("experiment") or {}).get("experiment_id")
        }, reverse=True)
        reports: list[dict] = []
        for dataset_suite, experiment_id in experiment_keys:
            selected = [
                run for run in runs
                if (run.get("experiment") or {}).get("experiment_id")
                == experiment_id
                and str(run.get("dataset_suite") or "") == dataset_suite
            ]
            baseline_run = None
            if dataset_suite in SUITE_SPECS:
                _, selection = await self._approved_selection(
                    workspace_id, dataset_suite,
                )
                baseline_run = selection.get("payload")
            report = experiment_report(
                selected, baseline_run=baseline_run,
                experiment_id=experiment_id,
            )
            report.update({
                "dataset_suite": dataset_suite,
                "suite_label": _suite_label(dataset_suite),
                "difficulty": _suite_difficulty(dataset_suite),
                "model_profile": _resolved_model_profile(selected[0])
                if selected else {},
            })
            reports.append(report)
        return reports

    async def list_baselines(self, workspace_id: str) -> dict:
        """Return one independently governed baseline card per difficulty."""
        entries = await self._registry_call("approved_all", workspace_id)
        by_suite = {
            str(entry.get("dataset_suite") or ""): entry for entry in entries
        }
        rows: list[dict[str, Any]] = []
        for difficulty in DIFFICULTY_ORDER:
            suites = [
                name for name, spec in SUITE_SPECS.items()
                if spec.get("difficulty") == difficulty
                and spec.get("baseline_policy") == "observational_three_repeat"
            ]
            dataset_suite = suites[0] if suites else ""
            entry = by_suite.get(dataset_suite)
            row: dict[str, Any] = {
                "difficulty": difficulty,
                "difficulty_label": {
                    "easy": "简单", "medium": "中等", "hard": "困难",
                }[difficulty],
                "dataset_suite": dataset_suite,
                "suite_label": _suite_label(dataset_suite),
                "available": False,
                "status": "missing",
                "run_id": None,
                "candidate_label": None,
                "model_profile": {},
                "reviewed_at": None,
                "metrics": {},
                "resolution_status": "not_approved",
                "resolution_error": None,
            }
            if entry:
                try:
                    selection = resolve_baseline_artifact(
                        workspace_id=workspace_id,
                        approved_entry=entry,
                        runs_dir=self.runs_dir,
                        expected_dataset_suite=dataset_suite,
                    )
                except (OSError, TypeError, ValueError):
                    # A broken approved artifact must not make the complete
                    # three-difficulty catalog fail. Surface a stable, safe
                    # state to the admin UI while preserving the approval row.
                    selection = {"available": False, "payload": None}
                    resolution_status = "artifact_invalid"
                    resolution_error = "已审批，但基线制品内容无法解析。"
                else:
                    if selection.get("available"):
                        resolution_status = "resolved"
                        resolution_error = None
                    elif entry.get("artifact"):
                        resolution_status = "artifact_missing"
                        resolution_error = "已审批，但基线制品不存在或无法访问。"
                    else:
                        resolution_status = "artifact_unbound"
                        resolution_error = "已审批，但基线记录未绑定制品。"
                payload = selection.get("payload") or {}
                aggregate = payload.get("aggregate") or {}
                total = int(payload.get("n") or len(payload.get("cases") or []))
                denominator = int(aggregate.get("result_evaluated") or total)
                run_header = self._run_header(payload) if payload else {}
                row.update({
                    "available": bool(selection.get("available")),
                    "status": entry.get("status") or "approved",
                    "run_id": entry.get("run_id"),
                    "candidate_label": run_header.get("candidate_label"),
                    "model_profile": run_header.get("model_profile") or {},
                    "reviewed_at": entry.get("reviewed_at"),
                    "reviewed_by": entry.get("reviewed_by"),
                    "resolution_status": resolution_status,
                    "resolution_error": resolution_error,
                    "metrics": {
                        "total": total,
                        "result_rate": self._rate(
                            aggregate.get("result_equivalent"), denominator,
                        ),
                        "first_result_rate": self._rate(
                            aggregate.get("first_result_equivalent"), total,
                        ),
                        "first_sql_evidence_complete": (
                            aggregate.get("first_sql_evidence_complete") is True
                        ),
                        "table_exact_rate": self._rate(
                            aggregate.get("tables_correct"), total,
                        ),
                        "execution_rate": self._rate(
                            aggregate.get("exec_success"), total,
                        ),
                        "latency_p95": float(
                            aggregate.get("tt_complete_p95")
                            or aggregate.get("latency_p95") or 0
                        ),
                        "llm_attempts": int(aggregate.get("llm_attempts") or 0),
                        "tokens_mean": float(aggregate.get("tokens_mean") or 0),
                        "estimated_cost_total": float(
                            aggregate.get("estimated_cost_total") or 0
                        ),
                        "cost_estimate_available": (
                            aggregate.get("cost_estimate_available") is True
                        ),
                        "cost_currency": aggregate.get("cost_currency") or "USD",
                    },
                })
            rows.append(row)
        return {"baselines": rows}

    @staticmethod
    def _golden_cases(suite: str = "easy_sql_v1") -> list[dict]:
        if suite not in SUITE_SPECS:
            raise ValueError(f"当前版本不支持评测套件：{suite}")
        return load_suite_cases(suite)

    async def governance(
        self, workspace_id: str, run_id: str | None = None,
        difficulty: str | None = None,
    ) -> dict:
        summaries = self._run_summaries(workspace_id, difficulty=difficulty)
        if not summaries:
            empty_suite = _baseline_suite_for_difficulty(difficulty)
            contract = sample_contract_report(self._golden_cases(empty_suite))
            return {"available": False, "sample_contract": contract, "message": "当前工作空间还没有有效评测 run。"}
        selected_summary = (
            next((item for item in summaries if str(item.get("run_id")) == run_id), None)
            if run_id else self._default_run_summary(summaries)
        )
        selected = self._load_run(selected_summary) if selected_summary is not None else None
        if selected is None:
            missing_suite = _baseline_suite_for_difficulty(difficulty)
            contract = sample_contract_report(self._golden_cases(missing_suite))
            return {"available": False, "sample_contract": contract, "message": f"未找到 run {run_id}。"}
        artifact_suite = str(selected.get("dataset_suite") or "")
        if artifact_suite not in SUITE_SPECS:
            return {
                "available": False,
                "message": f"run {selected.get('run_id')} 不符合当前评测套件合同。",
            }
        runnable_suite = artifact_suite
        cases = self._golden_cases(runnable_suite)
        contract = sample_contract_report(cases)
        checklist = baseline_checklist(selected, len(cases))
        entry = await self._registry_call("get", workspace_id, str(selected.get("run_id"))) or {
            "status": "unreviewed", "eligible": checklist["eligible"], "checklist": checklist["checks"],
        }
        approved_entry, baseline_selection = await self._approved_selection(
            workspace_id, artifact_suite,
        )
        approved_run = baseline_selection.get("payload")
        selected_experiment_id = (selected.get("experiment") or {}).get("experiment_id")
        experiment_summaries = self._run_summaries(
            workspace_id, include_invalid=True,
            difficulty=_suite_difficulty(artifact_suite),
        )
        experiment_runs = [
            payload
            for item in experiment_summaries
            if (item.get("experiment") or {}).get("experiment_id") == selected_experiment_id
            and (payload := self._load_run(item)) is not None
        ]
        experiment = experiment_report(
            experiment_runs, baseline_run=approved_run, experiment_id=selected_experiment_id,
        ) if selected_experiment_id else None
        run_eligible = checklist["eligible"]
        baseline_policy = str(
            SUITE_SPECS[artifact_suite].get("baseline_policy") or "release"
        )
        experiment_eligible = bool(
            experiment
            and (
                experiment.get("recordable")
                if baseline_policy == "observational_three_repeat"
                else experiment.get("qualified")
            )
        )
        approval_eligible = bool(run_eligible and experiment_eligible)
        now = dt.datetime.now(dt.UTC).astimezone().strftime("%Y%m%d-%H%M%S")
        nonce = secrets.token_hex(3)
        difficulty_name = _suite_difficulty(runnable_suite)
        suggested_id = (
            f"nl2sql-v2-{difficulty_name}-{runnable_suite}-{now}-"
            f"{workspace_id[:8]}-{nonce}"
        )
        command_prefix = (
            f"python -m evals.legacy.run_eval --workspace-id {workspace_id} "
            f"--suite {runnable_suite} "
            f"--experiment-id {suggested_id} --candidate-label v2-control "
            f"--evaluation-profile core-default"
        )
        return {
            "available": True,
            "sample_contract": contract,
            "dataset_suite": artifact_suite,
            "suite_label": _suite_label(artifact_suite),
            "difficulty": _suite_difficulty(artifact_suite),
            "baseline_policy": baseline_policy,
            "runnable_dataset_suite": runnable_suite,
            "selected": {
                "run": self._run_header(selected),
                "status": entry.get("status") or "unreviewed",
                "eligible": approval_eligible,
                "run_eligible": run_eligible,
                "approval_eligible": approval_eligible,
                "checklist": checklist["checks"],
                "fingerprint": selected.get("fingerprint") or {},
                "experiment": selected.get("experiment"),
                "evaluation_profile": (
                    selected.get("evaluation_profile")
                    or (selected.get("fingerprint") or {}).get("evaluation_profile")
                    or {"name": "unknown"}
                ),
                "reviewed_by": entry.get("reviewed_by"),
                "reviewed_at": entry.get("reviewed_at"),
                "review_note": entry.get("review_note"),
            },
            "approved": approved_entry,
            "baseline_resolution": {
                key: value for key, value in baseline_selection.items() if key != "payload"
            },
            "experiment": experiment,
            "experiment_setup": {
                "suggested_id": suggested_id,
                "suite": runnable_suite,
                "commands": [
                    f"{command_prefix} --repeat-index 1 --cache-state cold",
                    f"{command_prefix} --repeat-index 2 --cache-state warm",
                    f"{command_prefix} --repeat-index 3 --cache-state warm",
                ],
            },
            "comparison": compare_fingerprints(
                selected.get("fingerprint"),
                approved_run.get("fingerprint") if approved_run else None,
            ) if approved_entry else None,
        }

    async def review_baseline(self, workspace_id: str, run_id: str, *, decision: str,
                        reviewer: str, note: str = "") -> dict:
        summary = next(
            (item for item in self._run_summaries(workspace_id) if str(item.get("run_id")) == run_id),
            None,
        )
        run = self._load_run(summary) if summary is not None else None
        if run is None:
            raise ValueError("当前工作空间不存在该有效评测 run")
        dataset_suite = str(run.get("dataset_suite") or "")
        if dataset_suite not in SUITE_SPECS:
            raise ValueError("评测 run 的 dataset_suite 不受当前版本支持")
        baseline_policy = str(
            SUITE_SPECS[dataset_suite].get("baseline_policy") or "release"
        )
        if decision == "approve" and (run.get("fingerprint") or {}).get("fingerprint_schema") == "evaluation-v2":
            experiment_id = (run.get("experiment") or {}).get("experiment_id")
            if not experiment_id:
                raise ValueError("evaluation-v2 run 必须属于三次可比 experiment，单次 run 不得批准")
            summaries = self._run_summaries(workspace_id, include_invalid=True)
            runs = [
                payload
                for item in summaries
                if (item.get("experiment") or {}).get("experiment_id") == experiment_id
                and str(item.get("dataset_suite") or "") == dataset_suite
                and (payload := self._load_run(item)) is not None
            ]
            _, baseline_selection = await self._approved_selection(
                workspace_id, dataset_suite,
            )
            experiment = experiment_report(
                runs, baseline_run=baseline_selection.get("payload"),
                experiment_id=str(experiment_id),
            )
            eligible = (
                experiment.get("recordable")
                if baseline_policy == "observational_three_repeat"
                else experiment.get("qualified")
            )
            if not eligible:
                failed = "、".join(experiment.get("failed_checks") or ["experiment_not_qualified"])
                if baseline_policy == "observational_three_repeat":
                    raise ValueError(
                        f"该 experiment 未满足三次完整证据准入：{failed}"
                    )
                raise ValueError(
                    f"该 experiment 未满足三次最差值准入：{failed}"
                )
            if baseline_policy == "observational_three_repeat":
                representative = next(
                    (
                        item for item in runs
                        if (item.get("experiment") or {}).get("repeat_index") == 1
                        and (item.get("experiment") or {}).get("cache_state") == "cold"
                    ),
                    None,
                )
                if representative is None:
                    raise ValueError("难度基线必须以 R1 cold 作为代表 run")
                run = representative
                suffix = f"三轮观测基线 {experiment_id}；代表 run=R1 cold"
                note = f"{note}；{suffix}" if note else suffix
        return await self._registry_call(
            "review", run, decision=decision, reviewer=reviewer, note=note,
        )

    def latest(
        self, workspace_id: str, include_passed: bool = False,
        run_id: str | None = None, difficulty: str | None = None,
    ) -> dict:
        summaries = self._run_summaries(
            workspace_id, difficulty=difficulty,
        )
        if not summaries:
            suite = _baseline_suite_for_difficulty(difficulty)
            return {
                "available": False,
                "message": "当前工作空间还没有有效评测结果，请先运行 NL2SQL 评测。",
                "workspace_id": workspace_id,
                "run_command": (
                    "python -m evals.legacy.run_eval "
                    f"--suite {suite} --workspace-id {workspace_id} "
                    "--evaluation-profile core-default"
                ),
            }
        if run_id:
            selected_summary = next(
                (item for item in summaries if str(item.get("run_id")) == run_id), None,
            )
            if selected_summary is None:
                return {
                    "available": False,
                    "message": f"当前工作空间不存在有效评测批次 {run_id}。",
                    "workspace_id": workspace_id,
                    "run_command": f"python -m evals.legacy.run_eval --workspace-id {workspace_id} --evaluation-profile core-default",
                }
        else:
            selected_summary = self._default_run_summary(summaries)
        payload = self._load_run(selected_summary)
        if payload is None:
            return {
                "available": False,
                "message": "评测产物读取失败，请检查对应 JSON 文件是否完整。",
                "workspace_id": workspace_id,
                "run_command": f"python -m evals.legacy.run_eval --workspace-id {workspace_id} --evaluation-profile core-default",
            }
        cases = enrich_results(payload.get("cases") or [])
        summary = summarize_attributions(cases, workspace_id)
        aggregate = payload.get("aggregate") or {}
        n = int(payload.get("n") or len(cases))
        result_evaluated = int(aggregate.get("result_evaluated") or 0)
        visible_cases = cases if include_passed else [case for case in cases if not case.get("result_equivalent")]
        run_header = self._run_header(payload)
        model_profile = run_header.get("model_profile") or {}
        cost_estimate_available = aggregate.get("cost_estimate_available") is True
        return {
            "available": True,
            "run": run_header,
            "metrics": {
                "total": n,
                "result_evaluated": result_evaluated,
                "syntax_rate": self._rate(aggregate.get("syntax_valid"), n),
                "execution_rate": self._rate(aggregate.get("exec_success"), n),
                "table_exact_rate": self._rate(aggregate.get("tables_correct"), n),
                "result_rate": self._rate(aggregate.get("result_equivalent"), result_evaluated),
                "table_f1_mean": float(aggregate.get("table_f1_mean") or 0),
                "latency_p95": float(aggregate.get("latency_p95") or 0),
                "first_sql": {
                    "evidence_complete": (
                        aggregate.get("first_sql_evidence_complete") is True
                    ),
                    "non_empty": int(aggregate.get("first_non_empty") or 0),
                    "syntax_valid": int(
                        aggregate.get("first_syntax_valid") or 0
                    ),
                    "guard_passed": int(
                        aggregate.get("first_guard_passed") or 0
                    ),
                    "exec_success": int(
                        aggregate.get("first_exec_success") or 0
                    ),
                    "tables_correct": int(
                        aggregate.get("first_tables_correct") or 0
                    ),
                    "structure_equivalent": int(
                        aggregate.get("first_structure_equivalent") or 0
                    ),
                    "result_equivalent": int(
                        aggregate.get("first_result_equivalent") or 0
                    ),
                    "denominator": n,
                },
                "correction": {
                    "triggered": int(
                        aggregate.get("correction_triggered") or 0
                    ),
                    "fixed_execution": int(
                        aggregate.get("correction_fixed_execution") or 0
                    ),
                    "fixed_result": int(
                        aggregate.get("correction_fixed_result") or 0
                    ),
                    "harmful_modification": int(
                        aggregate.get("correction_harmful_modification") or 0
                    ),
                    "unattributable": int(
                        aggregate.get("correction_unattributable") or 0
                    ),
                    "llm_attempts": int(
                        aggregate.get("correction_llm_attempts") or 0
                    ),
                    "additional_tokens": int(
                        aggregate.get("correction_additional_tokens") or 0
                    ),
                    "unknown_token_attempts": int(
                        aggregate.get("correction_unknown_token_attempts") or 0
                    ),
                    "additional_latency_ms": int(
                        aggregate.get("correction_additional_latency_ms") or 0
                    ),
                    "denominator": n,
                },
                "final_sql": {
                    "syntax_valid": int(aggregate.get("syntax_valid") or 0),
                    "exec_success": int(aggregate.get("exec_success") or 0),
                    "tables_correct": int(
                        aggregate.get("tables_correct") or 0
                    ),
                    "structure_equivalent": int(
                        aggregate.get("structure_equivalent") or 0
                    ),
                    "result_equivalent": int(
                        aggregate.get("result_equivalent") or 0
                    ),
                    "result_evaluated": result_evaluated,
                    "denominator": n,
                },
            },
            "performance": {
                "timing_protocol_version": aggregate.get("timing_protocol_version") or "unknown",
                "timing_invariant_violations": list(aggregate.get("timing_invariant_violations") or []),
                "tt_complete_mean": float(aggregate.get("tt_complete_mean") or 0),
                "tt_complete_p95": float(aggregate.get("tt_complete_p95") or 0),
                "tt_sql_mean": float(aggregate.get("tt_sql_mean") or 0),
                "tt_sql_p95": float(aggregate.get("tt_sql_p95") or 0),
                "tt_result_mean": float(aggregate.get("tt_result_mean") or 0),
                "tt_result_p95": float(aggregate.get("tt_result_p95") or 0),
                "llm_calls_mean": float(aggregate.get("llm_calls_mean") or 0),
                "llm_calls_p95": float(aggregate.get("llm_calls_p95") or 0),
                "tokens_mean": float(aggregate.get("tokens_mean") or 0),
                "tokens_p95": float(aggregate.get("tokens_p95") or 0),
                "estimated_cost_total": float(aggregate.get("estimated_cost_total") or 0),
                "cost_estimate_available": cost_estimate_available,
                "cost_currency": aggregate.get("cost_currency") or "USD",
                "degraded_cases": int(aggregate.get("degraded_cases") or 0),
                "llm_attempt_schema": aggregate.get("llm_attempt_schema") or "unknown",
                "llm_attempts": aggregate.get("llm_attempts"),
                "llm_attempts_succeeded": aggregate.get("llm_attempts_succeeded"),
                "llm_attempts_failed": aggregate.get("llm_attempts_failed"),
                "llm_attempts_timed_out": aggregate.get("llm_attempts_timed_out"),
                "llm_attempts_cancelled": aggregate.get("llm_attempts_cancelled"),
                "llm_attempts_empty_output": aggregate.get("llm_attempts_empty_output"),
                "llm_attempt_unknown_tokens": aggregate.get("llm_attempt_unknown_tokens"),
                "llm_callback_missing": aggregate.get("llm_callback_missing"),
                "llm_pre_invoke_failures": aggregate.get("llm_pre_invoke_failures"),
                "llm_recoveries": int(aggregate.get("llm_recoveries") or 0),
                "llm_provider_request_id_unavailable": int(
                    aggregate.get("llm_provider_request_id_unavailable") or 0
                ),
                "llm_token_timing_unavailable": int(
                    aggregate.get("llm_token_timing_unavailable") or 0
                ),
                "llm_queue_wait_p95_ms": float(
                    aggregate.get("llm_queue_wait_p95_ms") or 0
                ),
                "llm_provider_p95_ms": float(
                    aggregate.get("llm_provider_p95_ms") or 0
                ),
                "llm_outcome_classes": dict(
                    aggregate.get("llm_outcome_classes") or {}
                ),
                "llm_attempt_coverage_missing_cases": aggregate.get(
                    "llm_attempt_coverage_missing_cases"
                ),
                "llm_attempt_accounting_complete": aggregate.get("llm_attempt_accounting_complete"),
                "failed_llm_calls": aggregate.get("failed_llm_calls"),
                "failed_core_llm_calls": aggregate.get("failed_core_llm_calls"),
                "failed_optional_llm_calls": aggregate.get("failed_optional_llm_calls"),
                "core_degradation_count": aggregate.get("core_degradation_count"),
                "optional_degradation_count": aggregate.get("optional_degradation_count"),
                "degradation_evidence_complete": aggregate.get("degradation_evidence_complete"),
                "structured_output_fallbacks": int(aggregate.get("structured_output_fallbacks") or 0),
                "semantic_coverage_cases": int(
                    aggregate.get("semantic_coverage_cases") or 0
                ),
                "semantic_coverage_mean": float(
                    aggregate.get("semantic_coverage_mean") or 0
                ),
                "semantic_unresolved_cases": int(
                    aggregate.get("semantic_unresolved_cases") or 0
                ),
                "semantic_unsafe_compile_cases": int(
                    aggregate.get("semantic_unsafe_compile_cases") or 0
                ),
                "request_deadline_contract_complete": aggregate.get(
                    "request_deadline_contract_complete"
                ),
                "request_deadline_exhausted_cases": int(
                    aggregate.get("request_deadline_exhausted_cases") or 0
                ),
                "prompt_payloads": int(aggregate.get("prompt_payloads") or 0),
                "prompt_tokens_before": int(aggregate.get("prompt_tokens_before") or 0),
                "prompt_tokens_after": int(aggregate.get("prompt_tokens_after") or 0),
                "prompt_compression_rate": float(aggregate.get("prompt_compression_rate") or 0),
                "adaptive_candidate_rate": float(aggregate.get("adaptive_candidate_rate") or 0),
                "candidate_selected_b": int(aggregate.get("candidate_selected_b") or 0),
                "candidate_valid_b": int(aggregate.get("candidate_valid_b") or 0),
                "candidate_repaired": int(aggregate.get("candidate_repaired") or 0),
                "candidate_failed": int(aggregate.get("candidate_failed") or 0),
                "candidate_wait_total_ms": float(aggregate.get("candidate_wait_total_ms") or 0),
                "candidate_wait_p95_ms": float(aggregate.get("candidate_wait_p95_ms") or 0),
                "node_performance": list(aggregate.get("node_performance") or []),
                "llm_route_performance": list(aggregate.get("llm_route_performance") or []),
            },
            "summary": summary,
            "cases": [
                self._case_view(
                    case,
                    candidate_label=run_header.get("candidate_label"),
                    model_profile=model_profile,
                    cost_estimate_available=cost_estimate_available,
                )
                for case in visible_cases
            ],
            "run_command": (
                "python -m evals.legacy.run_eval "
                f"--suite {payload.get('dataset_suite') or 'easy_sql_v1'} "
                f"--workspace-id {workspace_id} --evaluation-profile core-default"
            ),
        }

    @staticmethod
    def _rate(value: Any, total: int) -> float:
        return round(int(value or 0) / total, 4) if total else 0.0

    @staticmethod
    def _run_header(payload: dict) -> dict:
        aggregate = payload.get("aggregate") or {}
        n = int(payload.get("n") or len(payload.get("cases") or []))
        experiment = payload.get("experiment") or {}
        model_profile = _resolved_model_profile(payload)
        return {
            "run_id": payload.get("run_id"),
            "timestamp": payload.get("timestamp"),
            "workspace_id": payload.get("workspace_id"),
            "dataset_suite": payload.get("dataset_suite") or "unknown",
            "suite_label": _suite_label(
                str(payload.get("dataset_suite") or "unknown")
            ),
            "difficulty": _suite_difficulty(
                str(payload.get("dataset_suite") or "")
            ),
            "artifact": payload.get("_artifact"),
            "experiment_id": experiment.get("experiment_id"),
            "candidate_label": experiment.get("candidate_label"),
            "repeat_index": experiment.get("repeat_index"),
            "cache_state": experiment.get("cache_state"),
            "model_profile": model_profile,
            "candidate_id": model_profile.get("candidate_id"),
            "model_name": model_profile.get("model_name"),
            "provider": model_profile.get("provider"),
            "model_provider": model_profile.get("model_provider"),
            "adapter_version": model_profile.get("adapter_version"),
            "capability_registry_version": model_profile.get(
                "capability_registry_version"
            ),
            "evaluation_profile": (
                evaluation_profile_name(payload.get("evaluation_profile"), default="")
                or evaluation_profile_name(
                    (payload.get("fingerprint") or {}).get("evaluation_profile")
                )
            ),
            "total": n,
            "first_result_passed": int(
                aggregate.get("first_result_equivalent") or 0
            ),
            "first_sql_evidence_complete": (
                aggregate.get("first_sql_evidence_complete") is True
            ),
            "result_passed": int(aggregate.get("result_equivalent") or 0),
            "table_exact": int(aggregate.get("tables_correct") or 0),
            "latency_p95": float(aggregate.get("latency_p95") or 0),
            "cost_estimate_available": (
                aggregate.get("cost_estimate_available") is True
            ),
        }

    @staticmethod
    def _case_view(
        case: dict,
        *,
        candidate_label: str | None = None,
        model_profile: dict[str, Any] | None = None,
        cost_estimate_available: bool = False,
    ) -> dict:
        resolved_profile = _safe_model_profile(
            model_profile or _resolved_model_profile(case)
        )
        return {
            "id": case.get("id"),
            "question": case.get("question"),
            "category": case.get("category"),
            "result_equivalent": bool(case.get("result_equivalent")),
            "tables_correct": bool(case.get("tables_correct")),
            "exec_success": bool(case.get("exec_success")),
            "latency_ms": case.get("latency_ms"),
            "tt_sql_ms": case.get("tt_sql_ms"),
            "tt_result_ms": case.get("tt_result_ms"),
            "llm_calls": case.get("llm_calls"),
            "total_tokens": case.get("total_tokens"),
            "estimated_cost": case.get("estimated_cost"),
            "cost_estimate_available": cost_estimate_available is True,
            "candidate_label": candidate_label,
            "model_profile": resolved_profile,
            "candidate_id": resolved_profile.get("candidate_id"),
            "model_name": resolved_profile.get("model_name"),
            "provider": resolved_profile.get("provider"),
            "model_provider": resolved_profile.get("model_provider"),
            "adapter_version": resolved_profile.get("adapter_version"),
            "capability_registry_version": resolved_profile.get(
                "capability_registry_version"
            ),
            "first_sql_metrics": _allowlisted(
                case.get("first_sql_metrics"), _SAFE_SQL_METRIC_FIELDS
            ),
            "correction_metrics": _allowlisted(
                case.get("correction_metrics"),
                _SAFE_CORRECTION_METRIC_FIELDS,
            ),
            "final_sql_metrics": _allowlisted(
                case.get("final_sql_metrics"), _SAFE_SQL_METRIC_FIELDS
            ),
            "degradations": case.get("degradations") or [],
            "llm_attempts": case.get("llm_attempts") or [],
            "llm_attempt_accounting": case.get("llm_attempt_accounting") or {},
            "llm_attempt_coverage": case.get("llm_attempt_coverage") or {},
            "evaluation_profile": case.get("evaluation_profile") or "unknown",
            "retry_count": case.get("retry_count"),
            "gen_sql": case.get("gen_sql") or "",
            "expected_sql": case.get("expected_sql") or "",
            "error": case.get("error"),
            "attribution": case.get("attribution") or {},
            "stage_evidence": _safe_stage_evidence(case.get("stage_evidence")),
        }
