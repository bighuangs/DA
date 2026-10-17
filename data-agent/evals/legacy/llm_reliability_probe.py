"""Controlled live probe for TODO-95 structured SQL model reliability.

The probe deliberately uses an artificial schema and stores only hashes and
transport metadata.  It does not load any sealed evaluation question, SQL, or
production data and it never writes the provider response body to disk.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from langchain_core.messages import HumanMessage
import sqlglot

from app.adapters.outbound.agent.llm_provider import (
    get_task_llm,
    public_task_profile,
    task_profile,
)
from app.infrastructure.clients.llm.capabilities import (
    CAPABILITY_REGISTRY_VERSION,
    TODO97_CANDIDATE_SET,
    candidate_capability,
    capability_for_model,
    validate_profile_capability,
)
from app.agent_system.runtime.llm_attempts import attempt_accounting
from app.infrastructure.clients.llm.invocation import (
    LLMInvocationError,
    invoke_llm,
    model_output_text,
    normalize_sql_output,
    sql_output_validator,
)
from evals.paths import (
    LEGACY_DATASETS_ROOT,
    PROJECT_ROOT,
    REPORTS_ROOT as LEGACY_REPORTS_ROOT,
)


ROOT = PROJECT_ROOT
REPORT_ROOT = LEGACY_REPORTS_ROOT / "llm_reliability"
CONTRACT_PATH = LEGACY_DATASETS_ROOT / "llm_reliability_dev_v2.yaml"
PROBE_VERSION = "todo97-phase0-probe-v2"
ARTIFICIAL_PROMPT = """你是 MySQL 8 SQL 生成器。只返回一条 SELECT/CTE SQL，不要解释或 Markdown。
人工 Schema：
fact_event(event_id, user_id, item_id, zone_id, date_id, amount, quantity)
dim_date(date_id, year, month)
dim_item(item_id, category, brand)
dim_user(user_id, tier)
dim_zone(zone_id, area)
问题：2033年按区域和月份给出成交额、年内累计成交额、较上月差额与环比增长率。
要求：正确连接维表，按区域分区并按月份排序，除数为零时返回 NULL。"""


def _sha256(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _response_metadata(value: Any) -> dict[str, Any]:
    metadata = getattr(value, "response_metadata", None)
    if not isinstance(metadata, dict) and isinstance(value, dict):
        metadata = value.get("response_metadata")
    return dict(metadata or {}) if isinstance(metadata, dict) else {}


def _finish_reason(value: Any) -> str:
    metadata = _response_metadata(value)
    return str(
        metadata.get("finish_reason")
        or metadata.get("stop_reason")
        or (value.get("finish_reason") if isinstance(value, dict) else "")
        or "unavailable"
    ).lower()


def _safe_response_summary(value: Any) -> dict[str, Any]:
    text = model_output_text(value)
    normalized = normalize_sql_output(value)
    parse_error: str | None = None
    statement_type: str | None = None
    try:
        parsed = sqlglot.parse_one(normalized, read="mysql")
        statement_type = parsed.key.upper()
    except Exception as exc:  # pragma: no cover - exact parser messages vary
        parse_error = type(exc).__name__
    metadata = _response_metadata(value)
    token_usage = metadata.get("token_usage")
    if not isinstance(token_usage, dict):
        token_usage = getattr(value, "usage_metadata", None)
    token_usage = token_usage if isinstance(token_usage, dict) else {}
    return {
        "finish_reason": _finish_reason(value),
        "content_chars": len(text),
        "sql_sha256": _sha256(normalized) if normalized else None,
        "statement_type": statement_type,
        "parse_error_type": parse_error,
        "input_tokens": token_usage.get("input_tokens", token_usage.get("prompt_tokens")),
        "output_tokens": token_usage.get("output_tokens", token_usage.get("completion_tokens")),
        "total_tokens": token_usage.get("total_tokens"),
        "structured_output_clean": bool(
            normalized
            and parse_error is None
            and normalized.lower().startswith(("select ", "select\n", "with "))
        ),
    }


def _profile_contract_gates(profile: dict[str, Any]) -> dict[str, bool]:
    """Validate the active SQL route against its provider/model capabilities."""
    capability_gates = validate_profile_capability(profile)
    return {
        **capability_gates,
        "thinking_contract_valid": capability_gates["thinking_mode_supported"],
        "reasoning_effort_contract_valid": capability_gates[
            "reasoning_effort_supported"
        ],
        "temperature_contract_valid": capability_gates["temperature_supported"],
        "visible_output_budget_at_least_2048": int(
            profile.get("max_completion_tokens") or 0
        ) >= 2048,
        "sql_route_timeout_positive": float(profile.get("timeout_seconds") or 0) > 0,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    gates = report["gates"]
    lines = [
        "# TODO-97 Phase 0 模型兼容性探针",
        "",
        f"> Probe：`{report['probe_id']}` ｜ 状态：`{report['status']}` ｜ "
        f"候选：`{report['candidate']['candidate_label']}` ｜ "
        f"模型：`{report['profile']['model_name']}`",
        "",
        "本报告使用人工 Schema/问题，仅保存 prompt/SQL 哈希及调用证据，不保存模型正文。",
        "",
        "## 生效配置",
        "",
        "| 路由 | Thinking | 输出预算 | 调用 TTL | 重复次数 |",
        "| --- | --- | ---: | ---: | ---: |",
        f"| sql | {report['profile']['thinking_mode']} | "
        f"{report['profile']['max_completion_tokens']} | "
        f"{report['profile']['timeout_seconds']} s | {report['repeat']} |",
        "",
        "## 结果",
        "",
        "| Repeat | 状态 | Finish | 耗时 | 正文字符 | SQL AST | 失败尝试 | 证据完整 |",
        "| ---: | --- | --- | ---: | ---: | --- | ---: | --- |",
    ]
    for row in report["runs"]:
        lines.append(
            f"| {row['repeat_index']} | {row['status']} | {row['finish_reason']} | "
            f"{row['elapsed_ms']:.1f} ms | {row['content_chars']} | "
            f"{row.get('statement_type') or '-'} | {row['failed_attempts']} | "
            f"{'✅' if row['evidence_complete'] else '❌'} |"
        )
    lines.extend([
        "",
        "## 发布门禁",
        "",
        "| 检查 | 结果 |",
        "| --- | --- |",
    ])
    lines.extend(
        f"| {name} | {'✅' if passed else '❌'} |"
        for name, passed in gates.items()
    )
    lines.extend([
        "",
        f"- 必要调用成功率：{report['summary']['necessary_call_success_rate']:.1%}",
        f"- 结构化输出洁净率：{report['summary']['structured_output_clean_rate']:.1%}",
        f"- 核心失败尝试：{report['summary']['failed_core_llm_calls']}",
        f"- P95：{report['summary']['p95_ms']:.1f} ms",
        "",
    ])
    return "\n".join(lines)


async def run_probe(
    *,
    repeat: int = 3,
    candidate_label: str | None = None,
    persist: bool = True,
    report_root: Path = REPORT_ROOT,
) -> dict[str, Any]:
    repeat = max(1, int(repeat))
    profile = task_profile("sql")
    resolved_capability = capability_for_model(str(profile.get("model_name") or ""))
    if candidate_label:
        expected = candidate_capability(candidate_label)
        if not resolved_capability or (
            resolved_capability["candidate_label"] != expected["candidate_label"]
        ):
            raise ValueError(
                f"Phase 0 候选 {expected['candidate_label']} 与生效 SQL route "
                f"{profile.get('model_name')} 不一致"
            )
        candidate = expected
    elif resolved_capability:
        candidate = resolved_capability
    else:
        raise ValueError(
            f"生效 SQL 模型 {profile.get('model_name')} 未进入版本化能力注册表"
        )
    recipe = candidate.get("phase1") or {}
    preflight_gates = {
        **_profile_contract_gates(profile),
        "candidate_id_matches": str(profile.get("candidate_id") or "")
        == str(candidate.get("candidate_id") or ""),
        "phase1_thinking_recipe_matches": str(
            profile.get("thinking_mode") or "auto"
        ) == str(recipe.get("thinking_mode") or "auto"),
        "phase1_temperature_recipe_matches": float(
            profile.get("temperature") or 0
        ) == float(recipe.get("temperature") or 0),
    }
    failed_preflight = [
        name for name, passed in preflight_gates.items() if not passed
    ]
    if failed_preflight:
        raise ValueError(
            "Phase 0 在读取凭据/创建客户端前拒绝不匹配路由："
            + ",".join(failed_preflight)
        )
    runnable = get_task_llm("sql")
    prompt_hash = _sha256(ARTIFICIAL_PROMPT)
    runs: list[dict[str, Any]] = []
    all_attempts: list[dict[str, Any]] = []

    for index in range(1, repeat + 1):
        started = time.perf_counter()
        try:
            result = await invoke_llm(
                runnable,
                [HumanMessage(content=ARTIFICIAL_PROMPT)],
                node_id="generate_sql",
                route="sql",
                timeout_seconds=float(profile["timeout_seconds"]),
                max_retries=0,
                recovery_enabled=False,
                output_validator=sql_output_validator,
            )
            attempts = list(result.attempts)
            summary = _safe_response_summary(result.value)
            status = "pass" if summary["structured_output_clean"] else "fail"
            reason_code = None
        except LLMInvocationError as exc:
            attempts = list(exc.attempts)
            summary = {
                "finish_reason": "unavailable",
                "content_chars": 0,
                "sql_sha256": None,
                "statement_type": None,
                "parse_error_type": None,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "structured_output_clean": False,
            }
            status = "fail"
            reason_code = exc.reason_code
        elapsed_ms = (time.perf_counter() - started) * 1000
        accounting = attempt_accounting(attempts)
        all_attempts.extend(attempts)
        runs.append({
            "repeat_index": index,
            "status": status,
            "reason_code": reason_code,
            "elapsed_ms": round(elapsed_ms, 3),
            **summary,
            "attempts": accounting["attempts"],
            "failed_attempts": accounting["failed"] + accounting["timed_out"],
            "evidence_complete": bool(accounting["evidence_complete"]),
        })

    successes = sum(row["status"] == "pass" for row in runs)
    clean = sum(bool(row["structured_output_clean"]) for row in runs)
    accounting = attempt_accounting(all_attempts)
    latencies = sorted(float(row["elapsed_ms"]) for row in runs)
    p95_index = max(0, min(len(latencies) - 1, int((len(latencies) - 1) * 0.95 + 0.999)))
    summary = {
        "necessary_call_success_rate": successes / repeat,
        "structured_output_clean_rate": clean / repeat,
        "failed_core_llm_calls": int(accounting["core_failures"]),
        "evidence_complete_rate": sum(row["evidence_complete"] for row in runs) / repeat,
        "p95_ms": latencies[p95_index],
        "attempt_accounting": accounting,
    }
    gates = {
        "minimum_three_repeats": repeat >= 3,
        **preflight_gates,
        "necessary_call_success_rate_100pct": summary["necessary_call_success_rate"] == 1.0,
        "failed_core_llm_calls_zero": summary["failed_core_llm_calls"] == 0,
        "structured_output_clean_100pct": summary["structured_output_clean_rate"] == 1.0,
        "attempt_evidence_complete_100pct": summary["evidence_complete_rate"] == 1.0,
        "finish_reason_not_truncated": all(
            row["finish_reason"] not in {"length", "max_tokens", "max_output_tokens"}
            for row in runs
        ),
    }
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    probe_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    report = {
        "schema_version": PROBE_VERSION,
        "probe_id": probe_id,
        "generated_at": generated_at,
        "status": "pass" if all(gates.values()) else "fail",
        "contract_path": str(CONTRACT_PATH.relative_to(ROOT)).replace("\\", "/"),
        "prompt_sha256": prompt_hash,
        "prompt_persisted": False,
        "response_body_persisted": False,
        "capability_registry_version": CAPABILITY_REGISTRY_VERSION,
        "candidate": candidate,
        "repeat": repeat,
        "profile": public_task_profile(profile),
        "runs": runs,
        "summary": summary,
        "gates": gates,
    }
    if persist:
        report_root.mkdir(parents=True, exist_ok=True)
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        label_slug = str(candidate["candidate_label"]).lower()
        (report_root / f"probe_{label_slug}_{probe_id}.json").write_text(rendered, encoding="utf-8")
        (report_root / f"probe_{label_slug}_{probe_id}.md").write_text(
            _render_markdown(report), encoding="utf-8"
        )
        (report_root / f"probe_{label_slug}_latest.json").write_text(rendered, encoding="utf-8")
        (report_root / f"probe_{label_slug}_latest.md").write_text(
            _render_markdown(report), encoding="utf-8"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TODO-97 人工 Schema Provider/模型兼容性探针"
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--candidate", choices=TODO97_CANDIDATE_SET, default=None)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    report = asyncio.run(run_probe(repeat=args.repeat, candidate_label=args.candidate))
    print(_render_markdown(report))
    print(f"报告目录：{REPORT_ROOT}")
    return 1 if args.enforce and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
