"""Deterministic TODO-39 resilience probes and report generation."""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import time
from collections import Counter
from typing import Any

from omegaconf import OmegaConf
from pydantic import ValidationError

from app.adapters.inbound.http.schemas.query_schema import QuerySchema
from app.config.app_config import app_config
from app.foundation.request_context import get_req_id, set_req_id
from app.agent_system.governance.resilience import (
    AsyncCapacityGovernor,
    AsyncSingleFlight,
    CapacityExceededError,
    DependencyUnavailableError,
    optional_dependency,
    required_dependency,
)
from app.infrastructure.cache.result_cache import ResultCache
from app.domain.query.safety.sql_security import enforce_limit
from app.application.services.export_service import (
    ExportLimitError,
    ExportLimits,
    validate_export_shape,
)
from evals.paths import LEGACY_DATASETS_ROOT, REPORTS_ROOT as REPORT_ROOT

SUITE_PATH = LEGACY_DATASETS_ROOT / "resilience_suite.yaml"
LATEST_PATH = REPORT_ROOT / "resilience_latest.json"
REQUIRED_CATEGORIES = {"dependency", "concurrency", "cancellation", "resource", "cache", "soak"}


def load_suite() -> dict[str, Any]:
    raw = OmegaConf.to_container(OmegaConf.load(SUITE_PATH), resolve=True) or {}
    return dict(raw)


def validate_suite(min_cases: int = 20) -> dict[str, Any]:
    suite = load_suite()
    cases = list(suite.get("cases") or [])
    errors: list[str] = []
    ids: set[str] = set()
    categories: Counter[str] = Counter()
    for index, case in enumerate(cases, 1):
        missing = [key for key in ("id", "category", "title", "risk", "probe") if not case.get(key)]
        if missing:
            errors.append(f"第 {index} 条缺少 {','.join(missing)}")
        case_id = str(case.get("id") or "")
        if case_id in ids:
            errors.append(f"重复 ID：{case_id}")
        ids.add(case_id)
        categories[str(case.get("category") or "")] += 1
    missing_categories = sorted(REQUIRED_CATEGORIES - set(categories))
    if missing_categories:
        errors.append(f"缺少类别：{','.join(missing_categories)}")
    if len(cases) < min_cases:
        errors.append(f"用例数 {len(cases)} 少于门槛 {min_cases}")
    fingerprint = hashlib.sha256(SUITE_PATH.read_bytes()).hexdigest()
    return {
        "valid": not errors,
        "errors": errors,
        "total": len(cases),
        "categories": dict(sorted(categories.items())),
        "fingerprint": f"sha256:{fingerprint}",
        "version": suite.get("version"),
    }


async def _probe_optional_failure(case: dict) -> tuple[bool, dict]:
    async def fail():
        raise ConnectionError("injected connection failure")
    value, degradation = await optional_dependency(
        str(case["dependency"]), fail, fallback=[], timeout_seconds=0.05,
    )
    passed = value == [] and degradation and degradation["reason"] == "dependency_unavailable"
    return bool(passed), {"fallback": value, "degradation": degradation}


async def _probe_optional_timeout(case: dict) -> tuple[bool, dict]:
    async def slow():
        await asyncio.sleep(0.05)
        return [1]
    value, degradation = await optional_dependency(
        str(case["dependency"]), slow, fallback=[], timeout_seconds=0.005,
    )
    passed = value == [] and degradation and degradation["reason"] == "timeout"
    return bool(passed), {"fallback": value, "degradation": degradation}


async def _probe_required_failure(case: dict) -> tuple[bool, dict]:
    async def fail():
        raise ConnectionError("injected hard dependency failure")
    try:
        await required_dependency(str(case["dependency"]), fail, timeout_seconds=0.05)
    except DependencyUnavailableError as exc:
        return exc.dependency == case["dependency"], {
            "dependency": exc.dependency, "reason": exc.reason, "message": str(exc),
        }
    return False, {"error": "required dependency failure was swallowed"}


async def _probe_concurrency(case: dict) -> tuple[bool, dict]:
    level = int(case["level"])
    governor = AsyncCapacityGovernor(20, 5, 0.5)
    durations: list[float] = []
    errors: list[str] = []

    async def worker(index: int):
        started = time.perf_counter()
        try:
            lease = await governor.acquire("workspace-a")
            async with lease:
                await asyncio.sleep(0.006 + (index % 3) * 0.001)
        except Exception as exc:
            errors.append(str(exc))
        durations.append((time.perf_counter() - started) * 1000)

    started = time.perf_counter()
    await asyncio.gather(*(worker(index) for index in range(level)))
    elapsed = max(0.000001, time.perf_counter() - started)
    snapshot = await governor.snapshot()
    p95 = sorted(durations)[max(0, int(len(durations) * 0.95) - 1)] if durations else 0
    error_rate = len(errors) / level
    passed = error_rate < 0.01 and snapshot.active == 0 and snapshot.completed == level and snapshot.peak <= 5
    return passed, {
        "concurrency": level, "completed": snapshot.completed, "peak_active": snapshot.peak,
        "error_rate": error_rate, "throughput_rps": round(level / elapsed, 1),
        "p95_ms": round(p95, 1), "errors": errors[:3],
    }


async def _probe_context_isolation(_: dict) -> tuple[bool, dict]:
    async def worker(index: int):
        expected = f"req-{index}"
        set_req_id(expected)
        await asyncio.sleep((index % 4) * 0.001)
        return expected, get_req_id()
    pairs = await asyncio.gather(*(worker(index) for index in range(20)))
    mismatches = [pair for pair in pairs if pair[0] != pair[1]]
    return not mismatches, {"requests": len(pairs), "mismatches": mismatches}


async def _probe_saturation(_: dict) -> tuple[bool, dict]:
    governor = AsyncCapacityGovernor(1, 1, 0.01)
    lease = await governor.acquire("workspace-a")
    rejected = False
    try:
        await governor.acquire("workspace-a")
    except CapacityExceededError:
        rejected = True
    await lease.release()
    snapshot = await governor.snapshot()
    return rejected and snapshot.active == 0 and snapshot.rejected == 1, {
        "rejected": snapshot.rejected, "active_after_release": snapshot.active,
    }


async def _probe_cancellation_release(_: dict) -> tuple[bool, dict]:
    governor = AsyncCapacityGovernor(1, 1, 0.1)
    entered = asyncio.Event()

    async def worker():
        lease = await governor.acquire("workspace-a")
        async with lease:
            entered.set()
            await asyncio.sleep(10)

    task = asyncio.create_task(worker())
    await entered.wait()
    started = time.perf_counter()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    release_ms = (time.perf_counter() - started) * 1000
    snapshot = await governor.snapshot()
    return snapshot.active == 0 and release_ms < 5000, {
        "release_ms": round(release_ms, 2), "active_after_cancel": snapshot.active,
    }


async def _probe_cancellation_propagates(_: dict) -> tuple[bool, dict]:
    started = asyncio.Event()

    async def slow():
        started.set()
        await asyncio.sleep(10)

    task = asyncio.create_task(optional_dependency("optional", slow, fallback=[], timeout_seconds=20))
    await started.wait()
    task.cancel()
    propagated = False
    try:
        await task
    except asyncio.CancelledError:
        propagated = True
    return propagated, {"cancelled_error_propagated": propagated}


async def _probe_question_limit(_: dict) -> tuple[bool, dict]:
    rejected = False
    try:
        QuerySchema(query="问" * 2001)
    except ValidationError:
        rejected = True
    return rejected, {"max_chars": 2000, "attempted_chars": 2001, "rejected": rejected}


async def _probe_sql_limit(_: dict) -> tuple[bool, dict]:
    sql = enforce_limit("SELECT id FROM orders", app_config.query.limit_rows)
    passed = f"LIMIT {app_config.query.limit_rows}" in sql.upper()
    return passed, {"limit_rows": app_config.query.limit_rows, "guarded_sql": sql}


async def _probe_export(case: dict) -> tuple[bool, dict]:
    probe = case["probe"]
    attr = {
        "export_rows": "export_max_rows",
        "export_cells": "export_max_cells",
        "export_text": "export_max_cell_chars",
    }[probe]
    configured = getattr(app_config.query, attr)
    limits = ExportLimits(
        max_rows=1 if probe == "export_rows" else app_config.query.export_max_rows,
        max_columns=app_config.query.export_max_columns,
        max_cells=1 if probe == "export_cells" else app_config.query.export_max_cells,
        max_cell_chars=1 if probe == "export_text" else app_config.query.export_max_cell_chars,
    )
    rejected = False
    try:
        if probe == "export_rows":
            columns, rows = ["a"], [{"a": 1}, {"a": 2}]
        elif probe == "export_cells":
            columns, rows = ["a", "b"], [{"a": 1, "b": 2}]
        else:
            columns, rows = ["a"], [{"a": "xx"}]
        validate_export_shape(columns, rows, limits=limits)
    except ExportLimitError:
        rejected = True
    return rejected, {"boundary": attr, "configured": configured, "rejected": rejected}


async def _probe_upload_contract(_: dict) -> tuple[bool, dict]:
    limits = {
        "max_file_bytes": app_config.adhoc.max_file_bytes,
        "max_rows": app_config.adhoc.max_rows,
        "max_columns": app_config.adhoc.max_columns,
        "max_cell_chars": app_config.adhoc.max_cell_chars,
    }
    passed = all(isinstance(value, int) and value > 0 for value in limits.values())
    return passed, limits


async def _probe_singleflight(_: dict) -> tuple[bool, dict]:
    flight = AsyncSingleFlight()
    calls = 0

    async def loader():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return [{"value": 1}]

    results = await asyncio.gather(*(flight.run("same-query", loader) for _ in range(20)))
    results[0][0]["value"] = 9
    isolated = all(item[0]["value"] == 1 for item in results[1:])
    return calls == 1 and isolated and await flight.inflight() == 0, {
        "callers": 20, "loader_calls": calls, "followers": flight.followers,
        "mutable_results_isolated": isolated, "inflight_after": await flight.inflight(),
    }


async def _probe_cache_ttl(_: dict) -> tuple[bool, dict]:
    cache = ResultCache(max_entries=2)
    cache.set("key", {"value": 1}, ttl_seconds=0.01)
    await asyncio.sleep(0.02)
    value = cache.get("key", ttl_seconds=1)
    return value is None, {"expired_value": value, **cache.stats()}


async def _probe_cache_lru(_: dict) -> tuple[bool, dict]:
    cache = ResultCache(max_entries=8)
    for index in range(100):
        cache.set(str(index), index, ttl_seconds=30)
    stats = cache.stats()
    return stats["entries"] == 8, {"writes": 100, "max_entries": 8, **stats}


async def _probe_soak_cycles(_: dict) -> tuple[bool, dict]:
    governor = AsyncCapacityGovernor(5, 2, 0.2)
    cache = ResultCache(max_entries=16)
    for index in range(250):
        lease = await governor.acquire(f"workspace-{index % 3}")
        async with lease:
            cache.set(str(index), {"index": index}, ttl_seconds=30)
    snapshot = await governor.snapshot()
    stats = cache.stats()
    passed = snapshot.active == 0 and snapshot.completed == 250 and stats["entries"] <= 16
    return passed, {"cycles": 250, "active_after": snapshot.active, "completed": snapshot.completed, **stats}


def certification_status(seconds: int, required: int = 1800) -> str:
    """Name the evidence level without implying production-equivalent coverage."""
    if seconds <= 0:
        return "not_run"
    return "in_process_soak_certified" if seconds >= required else "in_process_smoke_only"


PROBES = {
    "optional_failure": _probe_optional_failure,
    "optional_timeout": _probe_optional_timeout,
    "required_failure": _probe_required_failure,
    "concurrency": _probe_concurrency,
    "context_isolation": _probe_context_isolation,
    "saturation": _probe_saturation,
    "cancellation_release": _probe_cancellation_release,
    "cancellation_propagates": _probe_cancellation_propagates,
    "question_limit": _probe_question_limit,
    "sql_limit": _probe_sql_limit,
    "export_rows": _probe_export,
    "export_cells": _probe_export,
    "export_text": _probe_export,
    "upload_contract": _probe_upload_contract,
    "singleflight": _probe_singleflight,
    "cache_ttl": _probe_cache_ttl,
    "cache_lru": _probe_cache_lru,
    "soak_cycles": _probe_soak_cycles,
}


async def _certification_soak(seconds: int) -> dict[str, Any]:
    required = 1800
    if seconds <= 0:
        return {
            "status": "not_run", "duration_seconds": 0, "required_seconds": required, "cycles": 0,
            "evidence_level": "in_process", "live_dependencies_evaluated": False,
            "resource_metrics_collected": False,
        }
    deadline = time.monotonic() + seconds
    cycles = 0
    while time.monotonic() < deadline:
        passed, _ = await _probe_soak_cycles({})
        if not passed:
            return {
                "status": "failed", "duration_seconds": seconds, "required_seconds": required, "cycles": cycles,
                "evidence_level": "in_process", "live_dependencies_evaluated": False,
                "resource_metrics_collected": False,
            }
        cycles += 250
        await asyncio.sleep(0)
    return {
        "status": certification_status(seconds, required),
        "duration_seconds": seconds, "required_seconds": required, "cycles": cycles,
        "evidence_level": "in_process", "live_dependencies_evaluated": False,
        "resource_metrics_collected": False,
    }


async def run_suite(*, soak_seconds: int = 0, write: bool = True) -> dict[str, Any]:
    suite = load_suite()
    contract = validate_suite()
    results: list[dict[str, Any]] = []
    for case in suite.get("cases") or []:
        started = time.perf_counter()
        error = None
        metrics: dict[str, Any] = {}
        try:
            probe = PROBES[str(case["probe"])]
            passed, metrics = await probe(dict(case))
        except Exception as exc:
            passed, error = False, f"{type(exc).__name__}: {exc}"
        results.append({
            **dict(case), "status": "passed" if passed else "failed",
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "metrics": metrics, "error": error,
        })
    certification = await _certification_soak(soak_seconds)
    counts = Counter(item["status"] for item in results)
    categories = []
    for category in sorted(REQUIRED_CATEGORIES):
        selected = [item for item in results if item["category"] == category]
        categories.append({
            "category": category, "total": len(selected),
            "passed": sum(item["status"] == "passed" for item in selected),
            "failed": sum(item["status"] == "failed" for item in selected),
        })
    now = dt.datetime.now().astimezone()
    report = {
        "available": True,
        "generated_at": now.isoformat(timespec="seconds"),
        "source": "deterministic_fault_injection",
        "contract": contract,
        "summary": {
            "total": len(results), "passed": counts["passed"], "failed": counts["failed"],
            "pass_rate": counts["passed"] / len(results) if results else 0,
            "gate_passed": contract["valid"] and counts["failed"] == 0,
        },
        "certification": certification,
        "categories": categories,
        "resource_limits": {
            "query_chars": 2000,
            "result_rows": app_config.query.limit_rows,
            "upload_bytes": app_config.adhoc.max_file_bytes,
            "upload_rows": app_config.adhoc.max_rows,
            "export_cells": app_config.query.export_max_cells,
            "global_concurrency": app_config.query.max_concurrent_queries,
            "workspace_concurrency": app_config.query.max_concurrent_queries_per_workspace,
        },
        "cases": results,
    }
    if write:
        REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(report, ensure_ascii=False, indent=2)
        LATEST_PATH.write_text(payload, encoding="utf-8")
        stamped = REPORT_ROOT / f"resilience_{now.strftime('%Y%m%d_%H%M%S')}.json"
        stamped.write_text(payload, encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="执行依赖故障、并发与资源边界评测")
    parser.add_argument("--soak-seconds", type=int, default=0, help="额外长稳认证时长；正式认证使用 1800")
    parser.add_argument("--enforce", action="store_true", help="门禁失败时返回非零")
    args = parser.parse_args()
    report = asyncio.run(run_suite(soak_seconds=max(0, args.soak_seconds)))
    summary = report["summary"]
    print(
        f"稳定性评测：{summary['passed']}/{summary['total']} 通过，"
        f"失败 {summary['failed']}；30 分钟认证 {report['certification']['status']}"
    )
    for item in report["categories"]:
        print(f"- {item['category']}: {item['passed']}/{item['total']}")
    return 2 if args.enforce and not summary["gate_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
