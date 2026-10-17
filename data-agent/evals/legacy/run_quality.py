"""Project-wide quality evaluation CLI.

The offline profile is deterministic and needs no LLM or infrastructure. The full profile
adds a real governed-suite NL2SQL run. Reports are written to
``var/evals/legacy/reports`` as JSON and Markdown.
"""
from __future__ import annotations

import argparse
import asyncio
import ast
import datetime as dt
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from xml.etree import ElementTree
from pathlib import Path
from typing import Any, Callable

from omegaconf import OmegaConf

from app.domain.query.safety.sql_security import SQLSecurityError, check_sql_read
from evals.legacy.contracts import CheckResult, QualityReport
from evals.legacy.profiles import evaluation_profile_name
from evals.legacy.governance import (
    EXPERIMENT_GATE_LIMITS,
    compare_local_version_fingerprints,
    experiment_report,
    local_version_fingerprint,
    resolve_baseline_artifact,
    sample_contract_report,
)
from evals.legacy.dataset_suite import (
    SUITE_SPECS,
    dataset_suite_report,
    load_suite_cases,
)
from evals.legacy.scenario_matrix import load_matrix, report_from_target_status, validate_matrix
from evals.legacy.resilience_suite import run_suite as run_resilience_suite, validate_suite as validate_resilience_suite
from evals.legacy.rag_recall import build_report as build_rag_report, validate_contract as validate_rag_contract
from evals.legacy.run_scenarios import collect_playwright_specs
from app.agent_system.governance.retrieval_budget import RERANK_VERSION
from app.infrastructure.search.index_version import retrieval_runtime_fingerprint
from app.config.app_config import app_config
from app.agent_system.runtime.llm_attempts import LLM_ATTEMPT_SCHEMA
from app.infrastructure.search.projection import alias_manifest_fingerprint
from evals.legacy.point_projection import point_experiment_query_contract
from evals.legacy.retrieval_responsibility import (
    build_report as build_retrieval_responsibility_report,
)
from evals.legacy.conversation_memory_eval import (
    build_report as build_conversation_memory_report,
)
from evals.legacy.embedding_capacity import (
    build_report as build_embedding_capacity_report,
)
from evals.legacy.query_signal_ablation import (
    build_report as build_query_signal_report,
)
from evals.legacy.planning_router_ablation import (
    build_report as build_planning_route_report,
)
from evals.legacy.semantic_repair_ablation import (
    build_report as build_semantic_repair_report,
)
from evals.legacy.adhoc_isolation_suite import (
    run_suite as run_adhoc_isolation_suite,
    validate_suite as validate_adhoc_isolation_suite,
)
from evals.paths import (
    CONFIGS_ROOT,
    LEGACY_DATASETS_ROOT,
    PROJECT_ROOT,
    REPORTS_ROOT as REPORT_ROOT,
    RUNS_ROOT,
)

WORKSPACE_ROOT = PROJECT_ROOT.parent
GATES_PATH = CONFIGS_ROOT / "quality_gates.yaml"
SECURITY_PATH = LEGACY_DATASETS_ROOT / "security_cases.yaml"
FRONTEND_ROOT = WORKSPACE_ROOT / "date-agent-frontend"
HYBRID_REPORT_PATH = REPORT_ROOT / "hybrid_retrieval_latest.json"
POINT_PROJECTION_REPORT_PATH = REPORT_ROOT / "point_projection_latest.json"

INSTRUMENTATION_SOURCE_PATHS = (
    "app/application/services/query_trace_service.py",
    "app/infrastructure/persistence/mysql/repositories/query_trace_repository.py",
    "app/infrastructure/persistence/mysql/repositories/feedback_repository.py",
    "evals/application/services/quality_operations_service.py",
    "evals/adapters/http/router.py",
    "evals/legacy/scenario_matrix.py",
    "evals/legacy/run_scenarios.py",
    "evals/legacy/resilience_suite.py",
    "evals/legacy/run_eval.py",
    "evals/legacy/compare_runs.py",
    "app/agent_system/agents/nl2sql/services/planning_router.py",
    "evals/legacy/planning_router_ablation.py",
    "evals/application/services/planning_evaluation_service.py",
    "app/domain/query/safety/semantic_violation.py",
    "app/agent_system/agents/nl2sql/services/semantic_repair.py",
    "evals/legacy/semantic_repair_ablation.py",
    "evals/application/services/semantic_repair_evaluation_service.py",
    "evals/legacy/adhoc_isolation_suite.py",
    "evals/datasets/legacy/adhoc_isolation_suite.yaml",
)
RETIRED_INSTRUMENTATION_PREFIXES = (
    "app/api/",
    "app/core/",
    "app/repositories/",
    "app/services/",
)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _timed(check: Callable[[], CheckResult]) -> CheckResult:
    started = time.perf_counter()
    result = check()
    result.duration_ms = round((time.perf_counter() - started) * 1000, 1)
    return result


def _run(
    command: list[str], cwd: Path, timeout: int = 180,
    env_overrides: dict[str, str] | None = None,
) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env={**os.environ, "PYTHONUTF8": "1", **(env_overrides or {})},
            check=False,
        )
        return completed.returncode, ANSI_RE.sub("", completed.stdout + completed.stderr)
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        return 124, f"{output}\n命令在 {timeout}s 后超时"
    except OSError as exc:
        return 127, str(exc)


def _thresholds() -> dict[str, Any]:
    raw = OmegaConf.to_container(OmegaConf.load(GATES_PATH), resolve=True)
    return dict((raw or {}).get("thresholds") or {})


def _result_equivalence_target(
    thresholds: dict[str, Any], dataset_suite: str | None,
) -> float:
    """Return the governed result-correctness floor for one difficulty suite."""

    by_suite = dict(
        thresholds.get("nl2sql_result_equivalence_rate_by_suite") or {}
    )
    suite = str(dataset_suite or "").strip()
    return float(
        by_suite.get(
            suite,
            thresholds["nl2sql_result_equivalence_rate"],
        )
    )


def check_compile() -> CheckResult:
    code, output = _run([sys.executable, "-m", "compileall", "-q", "app", "main.py"], PROJECT_ROOT)
    return CheckResult(
        id="python_compile",
        domain="reliability",
        name="Python 编译",
        status="pass" if code == 0 else "fail",
        score=100 if code == 0 else 0,
        summary="应用代码可完成字节码编译" if code == 0 else "应用代码存在编译错误",
        evidence={"exit_code": code, "tail": output[-1200:]},
        remediation=None if code == 0 else "先修复语法和导入错误，再运行其他评测。",
    )


def _test_inventory() -> dict[str, int]:
    groups = {"safety": 0, "correctness": 0, "reliability": 0, "other": 0}
    safety = {"sql_security", "query_guard", "access_policy", "masking_service", "auth_service"}
    correctness = {"schema_sync", "result_cache_export", "model_metadata", "adhoc_dataset", "eval_metrics"}
    reliability = {"mysql_client_manager", "qdrant_workspace", "query_trace", "eval_quality"}
    for path in (PROJECT_ROOT / "tests").glob("test_*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        count = sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_") for node in ast.walk(tree))
        stem = path.stem.removeprefix("test_")
        group = "safety" if stem in safety else "correctness" if stem in correctness else "reliability" if stem in reliability else "other"
        groups[group] += count
    return groups


def _pytest_junit_evidence(path: Path) -> dict[str, Any]:
    """把 pytest JUnit 转成场景矩阵可复用的精确 Test Target。"""

    root = ElementTree.parse(path).getroot()
    target_status: dict[str, dict[str, object]] = {}
    status_counts = {"passed": 0, "failed": 0, "skipped": 0}
    for case in root.iter("testcase"):
        classname = str(case.attrib.get("classname") or "").strip()
        name = str(case.attrib.get("name") or "").strip()
        target = ".".join(part for part in (classname, name) if part)
        failure = case.find("failure")
        if failure is None:
            failure = case.find("error")
        skipped = case.find("skipped")
        if failure is not None:
            status = "failed"
            error = str(
                failure.attrib.get("message")
                or failure.text
                or failure.tag
            )[:500]
        elif skipped is not None:
            status = "skipped"
            error = str(skipped.attrib.get("message") or "skipped")[:500]
        else:
            status = "passed"
            error = None
        status_counts[status] += 1
        try:
            duration_ms = round(float(case.attrib.get("time") or 0) * 1000, 3)
        except (TypeError, ValueError):
            duration_ms = 0.0
        if target:
            target_status[target] = {
                "status": status,
                "duration_ms": duration_ms,
                "error": error,
            }
    return {
        "tests": sum(status_counts.values()),
        "passed": status_counts["passed"],
        "failed": status_counts["failed"],
        "skipped": status_counts["skipped"],
        "target_status": target_status,
    }


def check_unit_tests() -> CheckResult:
    with tempfile.TemporaryDirectory(prefix="data-agent-quality-pytest-") as directory:
        junit_path = Path(directory) / "pytest-junit.xml"
        code, output = _run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--junitxml",
                str(junit_path),
                "tests",
            ],
            PROJECT_ROOT,
            1200,
            env_overrides={
                "PYTHONDONTWRITEBYTECODE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
        )
        try:
            junit = _pytest_junit_evidence(junit_path)
        except (OSError, ElementTree.ParseError) as exc:
            junit = {
                "tests": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "target_status": {},
                "junit_error": f"{type(exc).__name__}: {exc}",
            }
    count = int(junit["tests"])
    inventory = _test_inventory()
    passed = code == 0 and count > 0
    return CheckResult(
        id="unit_tests",
        domain="reliability",
        name="确定性单元回归",
        status="pass" if passed else "fail",
        score=100 if passed else 0,
        summary=f"{count} 项测试全部通过" if passed else "单元测试失败或未发现测试",
        evidence={
            **junit,
            "inventory": inventory,
            "exit_code": code,
            "runner": "pytest-junit",
            "tail": output[-1800:],
        },
        remediation=None if passed else "修复失败测试；质量门禁要求单元测试通过率为 100%。",
    )


def check_security_cases() -> CheckResult:
    cases = OmegaConf.to_container(OmegaConf.load(SECURITY_PATH), resolve=True) or []
    failures: list[dict[str, str]] = []
    for case in cases:
        rejected = False
        try:
            check_sql_read(str(case["sql"]))
        except SQLSecurityError:
            rejected = True
        expected_rejected = case["decision"] == "deny"
        if rejected != expected_rejected:
            failures.append({"id": case["id"], "expected": case["decision"]})
    rate = (len(cases) - len(failures)) / len(cases) if cases else 0
    minimum = int(_thresholds()["golden_min_security_cases"])
    passed = not failures and len(cases) >= minimum
    return CheckResult(
        id="security_contracts",
        domain="safety",
        name="SQL 安全攻击集",
        status="pass" if passed else "fail",
        score=round(rate * 100, 1),
        summary=f"{len(cases) - len(failures)}/{len(cases)} 个只读/攻击契约符合预期",
        evidence={"cases": len(cases), "minimum": minimum, "failures": failures},
        remediation=None if passed else "扩充或修复安全用例，任何攻击样本误放行都必须阻断发布。",
    )


def check_golden_contract(suite: str = "easy_sql_v1") -> CheckResult:
    cases = load_suite_cases(suite)
    suite_report = dataset_suite_report(suite)
    governance = sample_contract_report(cases)
    ids: set[str] = set()
    categories: set[str] = set()
    tags: set[str] = set()
    errors: list[str] = []
    smoke_count = 0
    for index, case in enumerate(cases, 1):
        missing = [key for key in ("id", "question", "expected_sql", "category", "tags", "tier") if not case.get(key)]
        if missing:
            errors.append(f"第 {index} 条缺少 {','.join(missing)}")
            continue
        if case["id"] in ids:
            errors.append(f"重复 ID：{case['id']}")
        ids.add(case["id"])
        categories.add(case["category"])
        tags.update(case.get("tags") or [])
        smoke_count += case["tier"] == "smoke"
        try:
            check_sql_read(case["expected_sql"])
        except SQLSecurityError as exc:
            errors.append(f"{case['id']} 期望 SQL 不安全：{exc}")
    thresholds = _thresholds()
    expected_cases = int(SUITE_SPECS[suite]["expected_cases"])
    passed = (
        not errors
        and governance["complete"]
        and governance["all_reviewed"]
        and suite_report["structural_ready"]
        and len(cases) == expected_cases
        and len(categories) >= int(thresholds["golden_min_categories"])
        and smoke_count > 0
    )
    coverage_score = min(
        100.0,
        len(cases) / expected_cases * 50
        + len(categories) / int(thresholds["golden_min_categories"]) * 50,
    )
    return CheckResult(
        id="golden_contract",
        domain="governance",
        name="NL2SQL 分档评测集契约",
        status="pass" if passed else "fail",
        score=round(coverage_score, 1) if not errors else 0,
        summary=f"{len(cases)} 个用例、{len(categories)} 个类别、{len(tags)} 个能力标签、{smoke_count} 个冒烟用例",
        evidence={
            "suite": suite, "expected_cases": expected_cases,
            "promotion_status": suite_report["promotion_status"],
            "independence": suite_report["independence"],
            "cases": len(cases), "categories": sorted(categories), "tags": sorted(tags),
            "smoke_cases": smoke_count, "errors": errors, "sample_contract": governance,
        },
        remediation=None if passed else "补齐用例字段、类别覆盖或安全期望 SQL，达到质量门槛后再用于回归。",
    )
def check_documentation() -> CheckResult:
    docs_root = PROJECT_ROOT / "docs"
    missing: list[str] = []
    checked = 0
    pattern = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    for source in docs_root.rglob("*.md"):
        text = source.read_text(encoding="utf-8")
        for raw in pattern.findall(text):
            target = raw.strip().split("#", 1)[0].strip().strip("<>")
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            checked += 1
            resolved = (source.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{source.relative_to(PROJECT_ROOT)} -> {target}")
    threshold = int(_thresholds()["documentation_missing_links"])
    passed = len(missing) <= threshold
    return CheckResult(
        id="documentation_links",
        domain="governance",
        name="文档链接完整性",
        status="pass" if passed else "fail",
        score=100 if passed else max(0, round((checked - len(missing)) / checked * 100, 1)) if checked else 0,
        summary=f"检查 {checked} 个本地链接，缺失 {len(missing)} 个",
        evidence={"checked": checked, "missing": missing[:30]},
        remediation=None if passed else "修复失效链接，避免路线状态和实现文档脱节。",
    )


def check_frontend_build() -> CheckResult:
    npm = shutil.which("npm")
    if not FRONTEND_ROOT.exists() or not npm:
        return CheckResult("frontend_build", "experience", "前端生产构建", "skip", None, "未找到前端或 npm", {"frontend": str(FRONTEND_ROOT)})
    code, output = _run([npm, "run", "build"], FRONTEND_ROOT, 240)
    match = re.search(r"assets/index-[^\s]+\.js\s+([\d.]+) kB\s+[^\n]*gzip:\s+([\d.]+) kB", output)
    initial_kb = float(match.group(1)) if match else None
    gzip_kb = float(match.group(2)) if match else None
    limit = float(_thresholds()["frontend_initial_gzip_kb"])
    passed = code == 0 and gzip_kb is not None and gzip_kb <= limit
    status = "pass" if passed else "warn" if code == 0 else "fail"
    score = 100 if passed else round(min(100, limit / gzip_kb * 100), 1) if gzip_kb else 0
    return CheckResult(
        id="frontend_build",
        domain="experience",
        name="前端生产构建与首包预算",
        status=status,
        score=score,
        summary=f"构建通过，初始 JS {initial_kb} kB / gzip {gzip_kb} kB" if code == 0 else "前端生产构建失败",
        evidence={"exit_code": code, "initial_kb": initial_kb, "gzip_kb": gzip_kb, "budget_gzip_kb": limit, "tail": output[-1600:]},
        remediation=None if passed else "保持生产构建可用，并继续把初始 gzip 控制在预算内。",
    )


def check_frontend_e2e() -> CheckResult:
    patterns = ("*.spec.ts", "*.spec.js", "*.test.ts", "*.test.js")
    search_roots = [FRONTEND_ROOT / name for name in ("src", "tests", "e2e", "e2e-real")]
    files = {
        path
        for root in search_roots if root.exists()
        for pattern in patterns
        for path in root.rglob(pattern)
    }
    count = 0
    for path in files:
        try:
            # Do not count calls such as `/pattern/.test(value)` as Playwright
            # declarations. They previously inflated the denominator whenever
            # an E2E assertion used RegExp.test().
            count += len(
                re.findall(
                    r"(?<![\w.])(?:it|test)\s*\(",
                    path.read_text(encoding="utf-8"),
                )
            )
        except OSError:
            pass
    minimum = int(_thresholds()["frontend_e2e_min_cases"])
    npm = shutil.which("npm")
    if not npm or not FRONTEND_ROOT.exists():
        return CheckResult(
            id="frontend_e2e",
            domain="experience",
            name="前端关键路径自动化",
            status="fail",
            score=0,
            summary="无法执行前端 E2E：未找到前端目录或 npm",
            evidence={"frontend": str(FRONTEND_ROOT), "npm": npm, "cases": count, "minimum": minimum},
            remediation="安装 Node.js/npm 并恢复前端目录后重新执行 Playwright 回归。",
        )
    specs: dict[str, dict] = {}
    outputs = []
    codes = []
    for script in ("test:e2e:mock", "test:e2e:real"):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "playwright.json"
            code, output = _run(
                [npm, "run", script, "--", "--reporter=json"], FRONTEND_ROOT, 420,
                {"PLAYWRIGHT_JSON_OUTPUT_FILE": str(output_path), "NO_COLOR": "1"},
            )
            codes.append(code)
            outputs.append(output)
            if output_path.exists():
                try:
                    payload = json.loads(output_path.read_text(encoding="utf-8"))
                    collect_playwright_specs(payload.get("suites") or [], specs)
                except (OSError, ValueError, TypeError) as exc:
                    outputs.append(f"Playwright JSON 解析失败：{exc}")
    code = max(codes, default=127)
    output = "\n".join(outputs)
    executed = sum(item.get("status") == "passed" for item in specs.values())
    matrix_targets = {
        str(case["evidence"]["target"])
        for case in load_matrix()
        if case["evidence"]["type"] == "e2e"
    }
    target_status = {}
    for target in matrix_targets:
        target_status[target] = specs.get(target, {
            "status": "not_run", "duration_ms": 0,
            "error": "Playwright JSON 未返回该目标",
        })
    passed = count >= minimum and code == 0 and executed >= minimum
    return CheckResult(
        id="frontend_e2e",
        domain="experience",
        name="前端关键路径自动化",
        status="pass" if passed else "fail",
        score=100 if passed else min(100, round(executed / minimum * 100, 1)) if minimum else 0,
        summary=f"Playwright {executed}/{count} 项通过（受控 API + 真实 HTTP/SQLite），门槛 {minimum}" if code == 0 else f"Playwright 执行失败：{executed}/{count} 项通过",
        evidence={
            "files": [str(path.relative_to(FRONTEND_ROOT)) for path in sorted(files)],
            "cases": count,
            "executed_passed": executed,
            "minimum": minimum,
            "exit_code": code,
            "target_status": target_status,
            "tail": output[-2400:],
        },
        remediation=None if passed else "修复 Playwright 失败项，并保持登录、问数 SSE、RBAC、管理后台、临时数据、仪表盘、键盘和三档视口覆盖。",
    )


def check_instrumentation() -> CheckResult:
    required = list(INSTRUMENTATION_SOURCE_PATHS)
    missing = [path for path in required if not (PROJECT_ROOT / path).exists()]
    retired = [
        path
        for path in required
        if path.startswith(RETIRED_INSTRUMENTATION_PREFIXES)
    ]
    passed = not missing and not retired
    invalid = set(missing) | set(retired)
    return CheckResult(
        id="quality_instrumentation",
        domain="governance",
        name="运行态质量证据链",
        status="pass" if passed else "fail",
        score=round(
            (len(required) - len(invalid)) / len(required) * 100,
            1,
        ),
        summary=(
            "trace、Token、反馈、版本运营与离线评测入口完整"
            if passed
            else "质量证据链存在缺口或仍引用退役源码路径"
        ),
        evidence={"required": required, "missing": missing, "retired": retired},
        remediation=(
            None
            if passed
            else "补齐当前 trace、反馈或评测入口，并移除退役路径后再做版本质量归因。"
        ),
    )


def check_scenario_matrix(*, unit_status: dict[str, dict], e2e_status: dict[str, dict]) -> CheckResult:
    thresholds = _thresholds()
    contract = validate_matrix(
        min_cases=int(thresholds["scenario_matrix_min_cases"]),
        min_per_capability=int(thresholds["scenario_matrix_min_per_capability"]),
    )
    report = report_from_target_status(unit_status=unit_status, e2e_status=e2e_status)
    summary = report["summary"]
    passed = contract["valid"] and summary["gate_passed"]
    return CheckResult(
        id="business_scenario_matrix", domain="governance", name="核心业务场景矩阵",
        status="pass" if passed else "fail",
        score=round(summary["passed"] / summary["total"] * 100, 1) if summary["total"] else 0,
        summary=(
            f"{summary['passed']}/{summary['total']} 场景通过；"
            f"{len(contract['capabilities'])} 个能力、{len(contract['roles'])} 种身份、"
            f"{len(contract['failure_layers'])} 个失败层"
        ),
        evidence={"contract": contract, "capabilities": report["capabilities"], "source": report["source"]},
        remediation=None if passed else "修复矩阵契约、证据引用或对应单测/E2E；不允许用静态清单替代执行证据。",
    )


def check_adhoc_isolation() -> CheckResult:
    thresholds = _thresholds()
    contract = validate_adhoc_isolation_suite(
        minimum_cases=int(thresholds["adhoc_isolation_min_cases"]),
        minimum_per_p0_category=int(
            thresholds["adhoc_isolation_min_per_p0_category"]
        ),
    )
    report = asyncio.run(
        run_adhoc_isolation_suite(
            evidence_level="offline_contract",
            write=False,
        )
    )
    summary = report["summary"]
    metrics = report["metrics"]
    passed = bool(
        contract["valid"]
        and summary["gate_passed"]
        and metrics["real_upload_content_occurrences"] == 0
        and metrics["shared_resource_delta"] == 0
        and metrics["sandbox_residual_resources"] == 0
        and metrics["unauthorized_db_execution_count"] == 0
        and metrics["schema_index_pollution_count"] == 0
        and metrics["cleanup_success_rate"] == 1.0
        and metrics["compensation_recovery_rate"] == 1.0
        and metrics["denominator_consistency"]
    )
    return CheckResult(
        id="adhoc_isolation_contract",
        domain="safety",
        name="上传数据资产隔离、隐私与清理门禁",
        status="pass" if passed else "fail",
        score=round(summary["pass_rate"] * 100, 1),
        summary=(
            f"{summary['passed']}/{summary['total']} 场景通过；"
            f"越权执行 {metrics['unauthorized_db_execution_count']}；"
            f"污染 {metrics['schema_index_pollution_count']}；"
            f"残留 {metrics['sandbox_residual_resources']}"
        ),
        evidence={
            "evidence_level": report["evidence_level"],
            "production_equivalent": report["production_equivalent"],
            "contract": contract,
            "denominator": report["denominator"],
            "metrics": metrics,
            "fixture_manifest": {
                "fixture_count": report["fixture_manifest"]["fixture_count"],
                "origin_counts": report["fixture_manifest"]["origin_counts"],
                "provenance_completeness": report["fixture_manifest"][
                    "provenance_completeness"
                ],
                "real_user_fixture_count": report["fixture_manifest"][
                    "real_user_fixture_count"
                ],
            },
        },
        remediation=None if passed else (
            "修复上传资产越权、Schema/索引污染、补偿清理、夹具来源或"
            "独立分母失败项；真实依赖发布前另跑 isolated_dependencies。"
        ),
    )


def check_resilience_suite() -> CheckResult:
    thresholds = _thresholds()
    contract = validate_resilience_suite(min_cases=int(thresholds["resilience_min_cases"]))
    report = asyncio.run(run_resilience_suite(write=False))
    summary = report["summary"]
    concurrency = next(item for item in report["cases"] if item["id"] == "CON-5")
    cancellation = next(item for item in report["cases"] if item["id"] == "CANCEL-LEASE")
    metrics_ok = (
        float(concurrency["metrics"]["error_rate"]) < float(thresholds["resilience_concurrency_error_rate_max"])
        and float(cancellation["metrics"]["release_ms"]) <= float(thresholds["resilience_cancel_release_ms"])
    )
    passed = contract["valid"] and summary["gate_passed"] and metrics_ok
    return CheckResult(
        id="resilience_boundary_suite", domain="reliability", name="依赖故障、并发与资源边界",
        status="pass" if passed else "fail",
        score=round(summary["passed"] / summary["total"] * 100, 1) if summary["total"] else 0,
        summary=(
            f"{summary['passed']}/{summary['total']} 探针通过；"
            f"5 并发错误率 {concurrency['metrics']['error_rate']:.1%}；"
            f"取消释放 {cancellation['metrics']['release_ms']:.2f} ms"
        ),
        evidence={
            "contract": contract, "categories": report["categories"],
            "certification": report["certification"], "resource_limits": report["resource_limits"],
        },
        remediation=None if passed else "修复故障降级、并发槽位、取消释放、缓存合并或资源边界失败项。",
    )


def check_rag_recall() -> CheckResult:
    thresholds = _thresholds()
    contract = validate_rag_contract()
    report = build_rag_report(write=False)
    summary = report["summary"]
    passed = (
        contract["valid"]
        and contract["total"] >= int(thresholds["rag_recall_min_cases"])
        and summary["gate_passed"]
    )
    return CheckResult(
        id="rag_offline_contract", domain="governance", name="RAG 离线标注与配置契约",
        status="pass" if passed else "fail",
        score=round((
            summary["routes"]["column"]["recall_at_k"]
            + summary["table_filter_recall"] + summary["display_hit_rate"]
        ) / 3 * 100, 1),
        summary=(
            f"离线确定性索引 {summary['cases']} 条；字段 Recall@K {summary['routes']['column']['recall_at_k']:.1%}；"
            f"筛表 {summary['table_filter_recall']:.1%}；展示字段 {summary['display_hit_rate']:.1%}"
        ),
        evidence={
            "contract": contract, "source": report["source"], "live_index_evaluated": False,
            "index": report["index"],
            "routes": summary["routes"], "comparison": report["comparison"],
        },
        remediation=None if passed else "检查 RAG 离线标注契约、召回配置、实体展示字段或问题级失败证据。",
    )


def check_retrieval_responsibility() -> CheckResult:
    report = build_retrieval_responsibility_report(write=False)
    passed = bool(report.get("gate_passed"))
    total = int(report.get("total_checks") or 0)
    passed_count = int(report.get("passed_checks") or 0)
    return CheckResult(
        id="retrieval_responsibility",
        domain="governance",
        name="Sparse RAG 与 ES 字段值检索职责",
        status="pass" if passed else "fail",
        score=round(passed_count / total * 100, 1) if total else 0,
        summary=(
            f"{passed_count}/{total} 项通过；Qdrant 仅元知识，"
            "ES 负责受版本与权限约束的真实字段值"
        ),
        evidence={
            "contract": report.get("contract"),
            "responsibilities": report.get("responsibilities"),
            "checks": report.get("checks"),
        },
        remediation=None if passed else (
            "修复字段值 mapping、ES 查询阶段权限过滤、敏感/高基数准入、"
            "稳定文档 ID 或 Qdrant payload 越界。"
        ),
    )


def check_conversation_memory() -> CheckResult:
    report = build_conversation_memory_report()
    summary = report["summary"]
    passed = bool(
        report["contract"]["valid"] and summary["gate_passed"]
    )
    return CheckResult(
        id="conversation_memory_contract",
        domain="governance",
        name="分层会话记忆、隔离与长期检索合同",
        status="pass" if passed else "fail",
        score=round(float(summary["pass_rate"]) * 100, 1),
        summary=(
            f"{summary['passed']}/{summary['total']} 场景通过；"
            "Redis 可重建短期层、MySQL 权威层与 ES/Qdrant 双投影"
        ),
        evidence={
            "contract": report["contract"],
            "profile": report["profile"],
            "failed_cases": [
                item for item in report["cases"]
                if item["status"] != "passed"
            ],
        },
        remediation=None if passed else (
            "修复 owner/workspace 隔离、TTL/删除、偏好替代、Prompt 注入、"
            "Redis 回退或双投影版本合同后重跑 conversation_memory_eval。"
        ),
    )


def check_embedding_capacity() -> CheckResult:
    """Run the deterministic TODO-60 scheduling gate without loading BGE-M3."""
    report = asyncio.run(build_embedding_capacity_report(
        live=False,
        concurrency=20,
        idle_repeats=5,
        write=False,
    ))
    summary = report["summary"]
    comparison = report["comparison"]
    minimum_gain = float(_thresholds()["embedding_capacity_gain_min"])
    passed = bool(
        summary["gate_passed"]
        and comparison["burst_texts_per_second_ratio"] >= minimum_gain
    )
    return CheckResult(
        id="embedding_capacity_contract",
        domain="governance",
        name="Embedding 调度合同与确定性容量",
        status="pass" if passed else "fail",
        score=(
            100.0
            if passed
            else round(
                min(
                    1.0,
                    comparison["burst_texts_per_second_ratio"]
                    / max(0.001, minimum_gain),
                )
                * 100,
                1,
            )
        ),
        summary=(
            f"serial → microbatch 吞吐 "
            f"{comparison['burst_texts_per_second_ratio']:.2f}×；"
            f"向量同协议 {'通过' if report['correctness']['query']['passed'] else '失败'}；"
            f"active {summary['selected_runtime_profile']}"
        ),
        evidence={
            "source": report["source"],
            "live_model_evaluated": report["live_model_evaluated"],
            "comparison": comparison,
            "gates": report["gates"],
            "remote_service_triggered": summary["remote_service_triggered"],
        },
        remediation=None if passed else (
            "修复有界队列、批次拆分、向量一致性或容量收益后，"
            "运行 python -m evals.legacy.embedding_capacity --enforce。"
        ),
    )


def check_query_signals() -> CheckResult:
    """Run deterministic TODO-61 signal, routing, privacy and cache contracts."""

    report = build_query_signal_report(write=False)
    summary = report["summary"]
    k1 = report["profiles"]["K1"]
    k2 = report["profiles"]["K2"]
    minimum_cases = int(_thresholds()["query_signal_min_cases"])
    passed = bool(
        summary["gate_passed"]
        and int(summary["cases"]) >= minimum_cases
        and report["decision"]["selected_profile"]
        == app_config.query_signals.profile
        and report["decision"]["selected_route_policy"]
        == app_config.query_signals.route_policy
    )
    return CheckResult(
        id="query_signal_ablation",
        domain="governance",
        name="查询信号消融、路由与隐私合同",
        status="pass" if passed else "fail",
        score=100 if passed else 0,
        summary=(
            f"K1 辅助 Recall {k1['auxiliary_recall']:.1%}；"
            f"K2 {k2['auxiliary_recall']:.1%}，Column 远端查询 "
            f"{k1['routing']['column']['remote_queries_mean']:.2f}→"
            f"{k2['routing']['column']['remote_queries_mean']:.2f}；"
            f"active 保留 {report['decision']['selected_profile']}"
        ),
        evidence={
            "source": report["source"],
            "cases": summary["cases"],
            "gates": report["gates"],
            "phase0": report["phase0"],
            "privacy": report["privacy"],
            "decision": report["decision"],
            "profiles": {
                label: {
                    "profile": item["profile"],
                    "route_policy": item["route_policy"],
                    "auxiliary_recall": item["auxiliary_recall"],
                    "typed_modifier_recall": item["typed_modifier_recall"],
                    "standalone_modifier_retention": (
                        item["standalone_modifier_retention"]
                    ),
                    "stability_rate": item["stability_rate"],
                    "routing": item["routing"],
                }
                for label, item in report["profiles"].items()
            },
        },
        remediation=None if passed else (
            "修复 query/document 编码、profile/cache/Trace 隔离、"
            "modifier 或租户隐私门禁后重跑 query_signal_ablation。"
        ),
    )


def check_planning_router() -> CheckResult:
    """Validate TODO-63 contracts without claiming live release approval."""

    report = build_planning_route_report(write=False)
    gates = report["implementation_gates"]
    c1 = report["profiles"]["C1"]
    recommendation = report["decision"]["offline_recommendation"]
    passed = bool(gates.get("gate_passed"))
    return CheckResult(
        id="planning_route_ablation",
        domain="governance",
        name="SQL 规划路由合同与离线反事实消融",
        status="pass" if passed else "fail",
        score=100 if passed else 0,
        summary=(
            f"C0～C5 / A0～A4 已评测；C1 Recall "
            f"{c1['plan_recall']:.1%}；离线推荐 {recommendation}；"
            "active 保持 keyword_length_v1"
        ),
        evidence={
            "evidence_level": report["evidence_level"],
            "contract": report["contract"],
            "implementation_gates": gates,
            "evaluation_scope": report["evaluation_scope"],
            "decision": report["decision"],
            "profiles": {
                label: {
                    key: item[key]
                    for key in (
                        "profile",
                        "cases",
                        "plan_precision",
                        "plan_recall",
                        "over_plan_rate",
                        "under_plan_rate",
                        "harmful_plan_rate",
                        "decision_latency_ms",
                    )
                }
                for label, item in report["profiles"].items()
            },
        },
        remediation=None if passed else (
            "修复 planning reviewed set、C1 兼容、C3/C4/C5、A0～A4、"
            "隐私或耗时合同后重跑 planning_router_ablation --enforce。"
        ),
    )


def check_semantic_repair() -> CheckResult:
    """Validate TODO-64 registry, safety and compatibility contracts."""

    report = build_semantic_repair_report(write=False)
    gates = report["implementation_gates"]
    active = report["profiles"]["P2"]
    passed = bool(gates.get("gate_passed"))
    return CheckResult(
        id="semantic_repair_ablation",
        domain="correctness",
        name="语义守卫修复注册表与分层纠错",
        status="pass" if passed else "fail",
        score=100 if passed else 0,
        summary=(
            f"P0～P6 已评测；P2 Precision {active['repair_precision']:.1%}；"
            f"近似样本误触发 {active['false_trigger_count']}；"
            f"幂等 {active['idempotence_rate']:.1%}；active 保持同规则注册表"
        ),
        evidence={
            "evidence_level": report["evidence_level"],
            "contract": report["contract"],
            "implementation_gates": gates,
            "registry": report["registry"],
            "decision": report["decision"],
            "profiles": {
                label: {
                    key: item[key]
                    for key in (
                        "profile", "rules", "cases", "repair_precision",
                        "targeted_rescue_rate", "false_trigger_count",
                        "harmful_repair_count", "idempotence_rate",
                        "unsafe_unchanged_rate", "repair_latency_ms",
                    )
                }
                for label, item in report["profiles"].items()
            },
        },
        remediation=None if passed else (
            "修复 P0/P2 兼容、误触发、幂等、安全不变量、隐私或延迟门禁后，"
            "重跑 semantic_repair_ablation --enforce。"
        ),
    )


def check_hybrid_retrieval(
    report_path: Path = HYBRID_REPORT_PATH,
) -> CheckResult:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return CheckResult(
            id="hybrid_retrieval_live",
            domain="correctness",
            name="BGE-M3 Dense + Sparse 真实混合检索",
            status="fail",
            score=0,
            summary="没有可用的 28 题真实混合检索制品",
            evidence={"artifact": str(report_path), "error": str(exc)},
            remediation=(
                "运行 python -m evals.legacy.hybrid_retrieval "
                "--workspace-id <workspace-id> --isolated-indexes --enforce。"
            ),
        )

    summary = report.get("summary") or {}
    checks = summary.get("checks") or {}
    active = ((report.get("profiles") or {}).get("R-H3") or {}).get("summary") or {}
    column = (active.get("routes") or {}).get("column") or {}
    query_contract = report.get("query_contract") or {}
    current_fingerprint = retrieval_runtime_fingerprint(RERANK_VERSION)
    required_checks = (
        "full_contract",
        "index_ready",
        "dense_evidence_nonempty",
        "sparse_evidence_nonempty",
        "hybrid_evidence_nonempty",
        "base_recall_non_regression",
        "column_precision_floor",
        "identifier_recall_non_regression",
        "identifier_mrr_non_regression",
        "tenant_isolation",
        "no_active_degradation",
    )
    passed = (
        report.get("available") is True
        and summary.get("gate_passed") is True
        and all(checks.get(key) is True for key in required_checks)
        and query_contract.get("fingerprint") == current_fingerprint
        and int(active.get("cases") or 0) >= 28
    )
    return CheckResult(
        id="hybrid_retrieval_live",
        domain="correctness",
        name="BGE-M3 Dense + Sparse 真实混合检索",
        status="pass" if passed else "fail",
        score=100 if passed else 0,
        summary=(
            f"R-H3 {active.get('passed_cases', 0)}/{active.get('cases', 0)}；"
            f"首轮字段 Recall {float(active.get('base_column_recall_at_k') or 0):.1%}；"
            f"Precision {float(active.get('base_column_precision_at_k') or 0):.1%}；"
            f"长尾 MRR {float((active.get('identifier_subset') or {}).get('mrr') or 0):.3f}"
        ),
        evidence={
            "artifact": str(report_path),
            "workspace_id": report.get("workspace_id"),
            "generated_at": report.get("generated_at"),
            "checks": checks,
            "active_profile": {
                "cases": active.get("cases"),
                "passed_cases": active.get("passed_cases"),
                "base_column_recall_at_k": active.get("base_column_recall_at_k"),
                "base_column_precision_at_k": active.get("base_column_precision_at_k"),
                "column": column,
                "identifier_subset": active.get("identifier_subset"),
                "raw_physical_points": active.get("raw_physical_points"),
            },
            "query_fingerprint": query_contract.get("fingerprint"),
            "current_query_fingerprint": current_fingerprint,
        },
        remediation=None if passed else (
            "报告缺失、已过期或真实 Dense/Sparse/Hybrid 门禁失败；"
            "重建不兼容索引后重新运行 hybrid_retrieval --enforce。"
        ),
    )


def check_point_projection(
    report_path: Path = POINT_PROJECTION_REPORT_PATH,
) -> CheckResult:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return CheckResult(
            id="point_projection_ablation",
            domain="correctness",
            name="Point 投影隔离消融与 active 选择",
            status="fail",
            score=0,
            summary="没有可用的 28 题 Point 投影消融制品",
            evidence={"artifact": str(report_path), "error": str(exc)},
            remediation=(
                "运行 python -m evals.legacy.point_projection "
                "--workspace-id <workspace-id> --enforce。"
            ),
        )

    profiles = report.get("profiles") or {}
    decision = report.get("decision") or {}
    selected = str(decision.get("selected_strategy") or "")
    selected_profile = profiles.get(selected) or {}
    selected_manifest = selected_profile.get("manifest") or {}
    baseline_manifest = (profiles.get("multi_all") or {}).get("manifest") or {}
    query_contract = report.get("query_contract") or {}
    current_query_contract = point_experiment_query_contract()
    manifest_contract = report.get("alias_manifest") or {}
    current_manifest_hash = alias_manifest_fingerprint()
    route_manifests = selected_manifest.get("routes") or {}
    identity_current = all(
        item.get("identity_version")
        == app_config.point_projection.identity_version
        for item in route_manifests.values()
    )
    checks = {
        "artifact_available": report.get("available") is True,
        "full_contract": bool((report.get("summary") or {}).get("full_contract")),
        "all_profiles_present": all(
            name in profiles
            for name in ("single_canonical", "multi_all", "mixed_selective")
        ),
        "selected_profile_eligible": selected_profile.get("eligible") is True,
        "decision_gate_passed": decision.get("gate_passed") is True,
        "active_strategy_matches_selection": (
            selected == app_config.point_projection.strategy
        ),
        "query_contract_current": (
            query_contract.get("fingerprint")
            == current_query_contract.get("fingerprint")
        ),
        "alias_manifest_current": (
            manifest_contract.get("fingerprint") == current_manifest_hash
        ),
        "identity_contract_current": identity_current,
        "stable_point_ids": bool(selected_manifest.get("stable_ids_valid")),
        "vector_integrity": bool(
            (selected_manifest.get("vector_integrity") or {}).get("valid")
        ),
    }
    passed = all(checks.values())
    selected_points = int(selected_manifest.get("point_count") or 0)
    baseline_points = int(baseline_manifest.get("point_count") or 0)
    reduction = (
        (baseline_points - selected_points) / baseline_points
        if baseline_points else 0.0
    )
    return CheckResult(
        id="point_projection_ablation",
        domain="correctness",
        name="Point 投影隔离消融与 active 选择",
        status="pass" if passed else "fail",
        score=100 if passed else 0,
        summary=(
            f"选择 {selected or '—'}；Point {selected_points}/{baseline_points} "
            f"（减少 {reduction:.1%}）；active "
            f"{app_config.point_projection.strategy}"
        ),
        evidence={
            "artifact": str(report_path),
            "workspace_id": report.get("workspace_id"),
            "generated_at": report.get("generated_at"),
            "checks": checks,
            "decision": decision,
            "selected_manifest": selected_manifest,
            "comparison": report.get("comparison"),
            "query_fingerprint": query_contract.get("fingerprint"),
            "current_query_fingerprint": current_query_contract.get("fingerprint"),
        },
        remediation=None if passed else (
            "报告缺失/过期、消融门禁失败，或 active 策略尚未切换并覆盖重建；"
            "按 decision 修复后重跑 point_projection 与 active 混合检索门禁。"
        ),
    )


async def _meta_approved_entry(
    workspace_id: str, dataset_suite: str,
) -> dict | None:
    from app.infrastructure.clients.mysql_client_manager import meta_mysql_client_manager
    from evals.infrastructure.persistence.mysql.evaluation_baseline_repository import EvaluationBaselineRepository
    meta_mysql_client_manager.init()
    try:
        async with meta_mysql_client_manager.session_factory() as session:
            return await EvaluationBaselineRepository(session).approved(
                workspace_id, dataset_suite,
            )
    finally:
        await meta_mysql_client_manager.close()


def resolve_quality_baseline(
    *, explicit: Path | str | None = None, workspace_id: str | None = None,
    dataset_suite: str | None = None,
) -> dict:
    """Resolve an explicit artifact or a workspace's Meta MySQL approval."""
    if explicit:
        return resolve_baseline_artifact(
            explicit=explicit, workspace_id=workspace_id,
            expected_dataset_suite=dataset_suite,
        )
    if not workspace_id:
        return resolve_baseline_artifact()
    try:
        if not dataset_suite:
            raise ValueError("从 Meta MySQL 解析 baseline 时必须指定 suite")
        approved = asyncio.run(_meta_approved_entry(workspace_id, dataset_suite))
    except Exception as exc:
        selection = resolve_baseline_artifact(workspace_id=workspace_id)
        selection["error"] = f"Meta MySQL baseline resolution failed: {exc}"
        return selection
    return resolve_baseline_artifact(
        workspace_id=workspace_id,
        approved_entry=approved,
        expected_dataset_suite=dataset_suite,
    )


def _baseline_values(selection: dict) -> dict:
    data = selection.get("payload") or {}
    agg = data.get("aggregate") or {}
    n = int(agg.get("n") or data.get("n") or len(data.get("cases") or []))
    result_denominator = int(agg.get("result_evaluated") or n)
    table_scope_safe = _table_scope_safe_count(data, n)
    return {
        "run_id": data.get("run_id"), "case_count": n,
        "dataset_suite": data.get("dataset_suite"),
        "execution_success_rate": int(agg.get("exec_success") or 0) / n if n else 0,
        "syntax_valid_rate": int(agg.get("syntax_valid") or 0) / n if n else 0,
        "table_accuracy": int(agg.get("tables_correct") or 0) / n if n else 0,
        "table_scope_safety_rate": table_scope_safe / n if n else 0,
        "security_guard_integrity": (
            1.0
            if "security_guard_bypass_count" in agg
            and int(agg.get("security_guard_bypass_count") or 0) == 0
            else 0.0
        ),
        "result_equivalence_rate": int(agg.get("result_equivalent") or 0) / result_denominator if result_denominator else 0,
        "latency_p95_ms": float(agg.get("latency_p95") or 0),
        "tt_sql_p95_ms": float(agg.get("tt_sql_p95") or 0),
        "tt_result_p95_ms": float(agg.get("tt_result_p95") or 0),
        "failed_case_ids": [
            str(case.get("id")) for case in (data.get("cases") or [])
            if case.get("result_equivalent") is not True
        ],
        "limitations": [],
    }


def _table_scope_safe_count(data: dict, total: int) -> int:
    """Read only complete runtime-bound scope evidence; legacy proxies are ineligible."""
    aggregate = data.get("aggregate") or {}
    if (
        int(aggregate.get("table_scope_evidence_complete") or 0) == total
        and "table_scope_safe" in aggregate
    ):
        return int(aggregate.get("table_scope_safe") or 0)
    cases = list(data.get("cases") or [])
    if cases and len(cases) == total and all(
        ((case.get("final_sql_metrics") or {}).get(
            "table_scope_safety_evidence"
        ) or {}).get("evidence_complete") is True
        for case in cases
    ):
        return sum(bool(
            (case.get("final_sql_metrics") or {}).get(
                "table_scope_safe", case.get("table_scope_safe")
            )
        ) for case in cases)
    return 0


def baseline_checks(selection: dict | Path | None = None) -> list[CheckResult]:
    if isinstance(selection, Path):
        selection = resolve_baseline_artifact(explicit=selection)
    selection = selection or resolve_quality_baseline()
    if not selection.get("available"):
        return [CheckResult("nl2sql_baseline", "correctness", "NL2SQL 基线", "skip", None, "没有可用基线", {"path": selection.get("path")})]
    data = _baseline_values(selection)
    thresholds = _thresholds()
    result_target = _result_equivalence_target(
        thresholds, data.get("dataset_suite"),
    )
    specs = [
        ("nl2sql_execution", "reliability", "执行成功率", data["execution_success_rate"], thresholds["nl2sql_execution_success_rate"], True),
        ("nl2sql_syntax", "correctness", "语法正确率", data["syntax_valid_rate"], thresholds["nl2sql_syntax_valid_rate"], True),
        ("nl2sql_tables", "correctness", "表选择准确率", data["table_accuracy"], thresholds["nl2sql_table_accuracy"], True),
        ("nl2sql_table_scope_safety", "safety", "表范围安全率", data["table_scope_safety_rate"], 1.0, True),
        ("nl2sql_security_guard_integrity", "safety", "安全守卫完整性", data["security_guard_integrity"], 1.0, True),
        ("nl2sql_results", "correctness", "结果一致率", data["result_equivalence_rate"], result_target, True),
        ("nl2sql_latency", "efficiency", "P95 延迟", data["latency_p95_ms"], thresholds["nl2sql_p95_latency_ms"], False),
    ]
    results = []
    for check_id, domain, name, value, target, higher_better in specs:
        observational_result = bool(
            check_id == "nl2sql_tables"
            or (
                check_id == "nl2sql_results"
                and higher_better
                and target <= 0
            )
        )
        passed = value >= target if higher_better else value <= target
        if observational_result:
            score = max(0.0, min(100.0, value * 100.0))
            display = (
                f"{value:.1%}，精确表匹配观测项（安全由表范围门禁承担）"
                if check_id == "nl2sql_tables"
                else f"{value:.1%}，观测项（不设强制正确率门槛）"
            )
        elif higher_better:
            score = min(100, value / target * 100) if target else 100
            display = f"{value:.1%}，目标 ≥ {target:.1%}"
        else:
            score = min(100, target / value * 100) if value else 100
            display = f"{value:.1f} ms，目标 ≤ {target:.0f} ms"
        results.append(CheckResult(
            id=check_id,
            domain=domain,
            name=f"当前基线 NL2SQL {name}",
            status=(
                "observe" if observational_result
                else "pass" if passed else "fail"
            ),
            score=round(score, 1),
            summary=display,
            evidence={
                "run_id": data["run_id"], "case_count": data["case_count"], "value": value,
                "target": target, "failed_case_ids": data["failed_case_ids"],
                "baseline_source": selection.get("source"), "baseline_status": selection.get("status"),
                "fingerprint_schema": selection.get("schema"),
                "tt_sql_p95_ms": data["tt_sql_p95_ms"], "tt_result_p95_ms": data["tt_result_p95_ms"],
            },
            remediation=(
                None if observational_result or passed
                else ("按类别定位召回、口径、SQL 结构和结果差异。" if higher_better else "拆分节点耗时并减少不必要模型调用、召回和纠错。")
            ),
            enforcement=(
                "observational" if observational_result else "blocking"
            ),
        ))
    coverage_passed = data["case_count"] >= int(thresholds["golden_min_cases"])
    results.append(CheckResult(
        id="nl2sql_baseline_coverage",
        domain="governance",
        name="当前基线覆盖度",
        status="pass" if coverage_passed else "warn",
        score=round(min(100, data["case_count"] / thresholds["golden_min_cases"] * 100), 1),
        summary=f"基线已执行 {data['case_count']} 个用例，门槛 {int(thresholds['golden_min_cases'])}",
        evidence={
            "limitations": data["limitations"], "baseline_source": selection.get("source"),
            "baseline_status": selection.get("status"), "fingerprint_schema": selection.get("schema"),
        },
        remediation=None if coverage_passed else "在完整依赖可用时运行完整分档评测集，形成可比较的新基线。",
    ))
    provenance_passed = selection.get("source") in {"meta_mysql", "explicit"}
    results.append(CheckResult(
        id="nl2sql_baseline_provenance", domain="governance", name="当前基线来源",
        status="pass" if provenance_passed else "warn",
        score=100 if provenance_passed else 70,
        summary=(
            f"{selection.get('source')} · {selection.get('status')} · {selection.get('schema')} · run {selection.get('run_id')}"
        ),
        evidence={key: selection.get(key) for key in ("source", "status", "schema", "path", "run_id")},
        remediation=None if provenance_passed else "传入 --workspace-id 从 Meta MySQL 解析 approved baseline，或使用 --baseline-run 显式选择。",
    ))
    return results


def experiment_checks(report: dict) -> list[CheckResult]:
    if not report.get("available"):
        return [CheckResult(
            "nl2sql_experiment_protocol", "governance", "二轮实验协议", "fail", 0,
            str(report.get("message") or "experiment 不存在"), report,
            "先完成 cold、warm、warm 三次完整评测。",
        )]
    worst = report.get("worst") or {}
    report_checks = report.get("checks") or {}
    protocol_passed = bool(report.get("complete"))
    correctness_passed = all(report_checks.get(key) for key in (
        "baseline_non_regression", "syntax_floor_gate",
        "execution_floor_gate", "result_floor_non_regression",
    )) and bool(report_checks.get(
        "table_scope_safety",
        report_checks.get("table_exact_non_regression"),
    )) and bool(report_checks.get("security_guard_integrity", True))
    latency_passed = all(report_checks.get(key) for key in (
        "ttsql_gate", "ttresult_gate", "ttcomplete_gate",
    ))
    model_reliability_keys = (
        "failed_llm_calls_zero", "structured_output_clean", "token_max_gate",
        "candidate_trigger_gate", "candidate_safety_gate", "token_non_regression",
        "llm_attempt_accounting_complete", "degradation_evidence_complete",
        "optional_postprocess_clean",
    ) + tuple(
        key for key in (
            "semantic_review_first_correct_final_incorrect_zero",
            "semantic_review_same_sql_execution_loss_zero",
            "semantic_review_selected_b_no_sql_delta_zero",
        )
        if key in report_checks
    )
    model_reliability_passed = all(
        report_checks.get(key) for key in model_reliability_keys
    )
    table_exact_passed = bool(report_checks.get("table_exact_non_regression"))
    return [
        CheckResult(
            "nl2sql_experiment_protocol", "governance", "二轮三次可比实验",
            "pass" if protocol_passed else "fail", 100 if protocol_passed else 0,
            f"{report.get('repeat_count', 0)}/{report.get('required_repeats', 3)} 次；状态 {report.get('status')}",
            {
                "experiment_id": report.get("experiment_id"),
                "checks": report_checks,
                "check_accounting": report.get("check_accounting"),
                "failed_checks": report.get("failed_checks"),
                "failed_observations": report.get("failed_observations"),
            },
            None if protocol_passed else "修复重复序号、冷/热顺序、指纹差异、基础设施错误或证据链缺口。",
        ),
        CheckResult(
            "nl2sql_experiment_non_regression", "correctness", "二轮正确性准入",
            "pass" if correctness_passed else "fail", 100 if correctness_passed else 0,
            f"最差表 {worst.get('tables_correct', 0)}；最差结果 {worst.get('result_equivalent', 0)}；回归 {len(report.get('regressed_case_ids') or [])} 题",
            {
                "regressed_case_ids": report.get("regressed_case_ids"),
                "stability": report.get("stability"),
                "baseline": report.get("baseline"),
                "table_scope_safe": worst.get("table_scope_safe"),
            },
            (
                None
                if correctness_passed
                else "同套件恢复逐题非回退；跨套件修复语法、执行、表选择或结果绝对下限。"
            ),
        ),
        CheckResult(
            "nl2sql_experiment_latency", "efficiency", "二轮最差值延迟",
            "pass" if latency_passed else "fail", 100 if latency_passed else 0,
            f"TTSQL {worst.get('tt_sql_p95', 0):.1f} ms；TTResult {worst.get('tt_result_p95', 0):.1f} ms；TTComplete {worst.get('tt_complete_p95', 0):.1f} ms",
            {"worst": worst, "gate_limits": report.get("gate_limits")},
            None if latency_passed else "按最差一次节点 P95 优化，不能用最好一次或平均值准入。",
        ),
        CheckResult(
            "nl2sql_experiment_model_reliability", "reliability", "二轮模型可靠性与 Token 边界",
            "pass" if model_reliability_passed else "fail", 100 if model_reliability_passed else 0,
            (
                f"模型失败最大 {worst.get('failed_llm_calls', 0)}；"
                f"单题 Token 最大 {worst.get('tokens_max', 0):.0f}；"
                f"未通过 {', '.join(key for key in model_reliability_keys if not (report.get('checks') or {}).get(key)) or '无'}"
            ),
            {"worst": worst, "checks": {key: (report.get("checks") or {}).get(key) for key in model_reliability_keys}},
            None if model_reliability_passed else "修复模型调用失败、结构化回退或单题 Token 尖峰后重跑三次实验。",
        ),
        CheckResult(
            "nl2sql_experiment_table_exact_observation",
            "correctness",
            "二轮精确表匹配观测",
            "observe",
            100 if table_exact_passed else 0,
            (
                f"最差精确表匹配 {worst.get('tables_correct', 0)}；"
                f"范围安全 {worst.get('table_scope_safe', 0)}"
            ),
            {
                "table_exact_non_regression": table_exact_passed,
                "table_scope_safety": report_checks.get("table_scope_safety"),
            },
            None,
            enforcement="observational",
        ),
    ]


def live_run_checks(
    data: dict[str, Any],
    *,
    dataset_suite: str | None = None,
) -> list[CheckResult]:
    """Turn a freshly executed NL2SQL run into current quality gates."""
    agg = data["aggregate"]
    n = int(agg.get("n") or data.get("n") or 0)
    cases = list(data.get("cases") or [])
    invalid_result_case_ids = [
        str(case.get("id") or "")
        for case in cases
        if case.get("result_equivalent") is True
        and case.get("exec_success") is not True
    ]
    # A failed pipeline cannot be result-equivalent merely because both the
    # generated side and the reference side happen to contain zero rows.  New
    # runs enforce this in evaluator.py; this recomputation also prevents an
    # older immutable artifact from inflating a current quality report.
    effective_result_equivalent = (
        sum(
            case.get("exec_success") is True
            and case.get("result_equivalent") is True
            for case in cases
        )
        if cases
        else min(
            int(agg.get("result_equivalent") or 0),
            int(agg.get("exec_success") or 0),
        )
    )
    result_denominator = int(agg.get("result_evaluated") or n)
    thresholds = _thresholds()
    suite = str(dataset_suite or data.get("dataset_suite") or "")
    expected_cases = int(SUITE_SPECS[suite]["expected_cases"])
    result_target = _result_equivalence_target(thresholds, suite)
    table_scope_safe = _table_scope_safe_count(data, n)
    specs = [
        ("live_nl2sql_execution", "reliability", "执行成功率", agg.get("exec_success", 0) / n if n else 0, thresholds["nl2sql_execution_success_rate"], True),
        ("live_nl2sql_syntax", "correctness", "语法正确率", agg.get("syntax_valid", 0) / n if n else 0, thresholds["nl2sql_syntax_valid_rate"], True),
        ("live_nl2sql_tables", "correctness", "表选择准确率", agg.get("tables_correct", 0) / n if n else 0, thresholds["nl2sql_table_accuracy"], True),
        ("live_nl2sql_table_scope_safety", "safety", "表范围安全率", table_scope_safe / n if n else 0, 1.0, True),
        ("live_nl2sql_security_guard_integrity", "safety", "安全守卫完整性", 1.0 if "security_guard_bypass_count" in agg and int(agg.get("security_guard_bypass_count") or 0) == 0 else 0.0, 1.0, True),
        ("live_nl2sql_results", "correctness", "结果一致率", effective_result_equivalent / result_denominator if result_denominator else 0, result_target, True),
        ("live_nl2sql_latency", "efficiency", "P95 延迟", float(agg.get("latency_p95") or 0), thresholds["nl2sql_p95_latency_ms"], False),
    ]
    results: list[CheckResult] = []
    for check_id, domain, name, value, target, higher_better in specs:
        observational_result = bool(
            check_id == "live_nl2sql_tables"
            or (
                check_id == "live_nl2sql_results"
                and higher_better
                and target <= 0
            )
        )
        passed = value >= target if higher_better else 0 < value <= target
        if observational_result:
            score = max(0.0, min(100.0, value * 100.0))
            display = (
                f"{value:.1%}，精确表匹配观测项（安全由表范围门禁承担）"
                if check_id == "live_nl2sql_tables"
                else f"{value:.1%}，观测项（不设强制正确率门槛）"
            )
        elif higher_better:
            score = min(100, value / target * 100) if target else 100
            display = f"{value:.1%}，目标 ≥ {target:.1%}"
        else:
            score = min(100, target / value * 100) if value else 0
            display = f"{value:.1f} ms，目标 ≤ {target:.0f} ms"
        results.append(CheckResult(
            id=check_id,
            domain=domain,
            name=f"当前全量 NL2SQL {name}",
            status=(
                "observe" if observational_result
                else "pass" if passed else "fail"
            ),
            score=round(score, 1),
            summary=display,
            evidence={"run_id": data.get("run_id"), "case_count": n, "value": value, "target": target},
            remediation=(
                None if observational_result or passed
                else ("按当前逐题结果和类别聚合定位正确性失败。" if higher_better else "使用步骤 trace 定位高延迟节点并控制模型调用。")
            ),
            enforcement=(
                "observational" if observational_result else "blocking"
            ),
        ))
    result_integrity_ok = not invalid_result_case_ids
    results.append(CheckResult(
        id="live_nl2sql_result_integrity",
        domain="governance",
        name="当前全量结果裁判一致性",
        status="pass" if result_integrity_ok else "fail",
        score=100 if result_integrity_ok else 0,
        summary=(
            "执行失败不得计为结果一致"
            if result_integrity_ok
            else f"发现 {len(invalid_result_case_ids)} 条未执行却标记结果一致"
        ),
        evidence={
            "run_id": data.get("run_id"),
            "raw_result_equivalent": int(agg.get("result_equivalent") or 0),
            "effective_result_equivalent": effective_result_equivalent,
            "invalid_case_ids": invalid_result_case_ids,
            "case_evidence_available": bool(cases),
        },
        remediation=None if result_integrity_ok else (
            "使用修正后的 evaluator 重新运行；历史不可变制品保留，但不得作为批准依据。"
        ),
    ))
    coverage_passed = n == expected_cases
    results.append(CheckResult(
        id="live_nl2sql_coverage",
        domain="governance",
        name="当前全量评测套件覆盖度",
        status="pass" if coverage_passed else "fail",
        score=min(100, round(n / expected_cases * 100, 1)),
        summary=f"{suite} 真实运行 {n}/{expected_cases} 个用例",
        evidence={
            "run_id": data.get("run_id"),
            "dataset_suite": suite,
            "expected_cases": expected_cases,
            "by_category": data.get("by_category", []),
        },
        remediation=None if coverage_passed else "运行完整 50 题分档评测集，不能只用 smoke 结果替代基线。",
    ))
    failed_calls = int(agg.get("failed_core_llm_calls") or 0)
    optional_failed_calls = int(agg.get("failed_optional_llm_calls") or 0)
    structured_fallbacks = int(agg.get("structured_output_fallbacks") or 0)
    attempt_complete = (
        agg.get("llm_attempt_schema") == LLM_ATTEMPT_SCHEMA
        and agg.get("llm_attempt_accounting_complete") is True
    )
    outcome_taxonomy_complete = sum(
        int(value or 0)
        for value in (agg.get("llm_outcome_classes") or {}).values()
    ) == int(agg.get("llm_attempts") or 0)
    deadline_complete = (
        agg.get("request_deadline_contract_complete") is True
        and int(agg.get("request_deadline_exhausted_cases") or 0) == 0
    )
    degradation_complete = agg.get("degradation_evidence_complete") is True
    evaluation_profile = (
        evaluation_profile_name(data.get("evaluation_profile"), default="")
        or evaluation_profile_name((data.get("fingerprint") or {}).get("evaluation_profile"))
    )
    optional_clean = evaluation_profile != "enhanced-analysis" or optional_failed_calls == 0
    model_clean = (
        failed_calls == 0 and structured_fallbacks == 0
        and attempt_complete and outcome_taxonomy_complete
        and deadline_complete and degradation_complete and optional_clean
    )
    results.append(CheckResult(
        id="live_nl2sql_model_reliability", domain="reliability", name="当前全量模型调用可靠性",
        status="pass" if model_clean else "fail", score=100 if model_clean else 0,
        summary=(
            f"核心失败 {failed_calls}；可选失败 {optional_failed_calls}；"
            f"结构化输出回退 {structured_fallbacks}；attempt 对账 {'完整' if attempt_complete else '缺失'}；"
            f"截止时间证据 {'完整' if deadline_complete else '缺失/耗尽'}"
        ),
        evidence={"run_id": data.get("run_id"), "failed_llm_calls": failed_calls,
                  "failed_optional_llm_calls": optional_failed_calls,
                  "structured_output_fallbacks": structured_fallbacks,
                  "attempt_accounting": {
                      key: agg.get(key) for key in (
                          "llm_attempts", "llm_attempts_succeeded", "llm_attempts_failed",
                          "llm_attempts_timed_out", "llm_attempts_cancelled",
                      )
                  },
                   "evaluation_profile": evaluation_profile,
                   "llm_outcome_classes": agg.get("llm_outcome_classes") or {},
                   "llm_recoveries": int(agg.get("llm_recoveries") or 0),
                   "llm_queue_wait_p95_ms": float(agg.get("llm_queue_wait_p95_ms") or 0),
                   "llm_provider_p95_ms": float(agg.get("llm_provider_p95_ms") or 0),
                   "request_deadline_contract_complete": deadline_complete,
                   "degradation_evidence_complete": degradation_complete},
        remediation=None if model_clean else "按节点 route 定位超时或非法结构化输出，并用确定性降级移出非关键模型依赖。",
    ))
    review_integrity = {
        "first_correct_final_incorrect": int(
            agg.get("semantic_review_first_correct_final_incorrect") or 0
        ),
        "same_sql_execution_loss": int(
            agg.get("semantic_review_same_sql_execution_loss") or 0
        ),
        "selected_b_no_sql_delta": int(
            agg.get("semantic_review_selected_b_no_sql_delta") or 0
        ),
    }
    review_integrity_ok = all(value == 0 for value in review_integrity.values())
    results.append(CheckResult(
        id="live_nl2sql_review_integrity",
        domain="correctness",
        name="当前全量候选复核结果完整性",
        status="pass" if review_integrity_ok else "fail",
        score=100 if review_integrity_ok else 0,
        summary=(
            "首次正确终态错误 / 同 SQL 执行损失 / 选 B 无 SQL 变化："
            f"{review_integrity['first_correct_final_incorrect']} / "
            f"{review_integrity['same_sql_execution_loss']} / "
            f"{review_integrity['selected_b_no_sql_delta']}"
        ),
        evidence={"run_id": data.get("run_id"), **review_integrity},
        remediation=(
            None if review_integrity_ok
            else "修复候选选择或执行归因流程后重跑，不得将流程损失计为模型错题。"
        ),
    ))
    semantic_coverage_complete = (
        int(agg.get("semantic_coverage_cases") or 0) == n
        and int(agg.get("semantic_unsafe_compile_cases") or 0) == 0
    )
    results.append(CheckResult(
        id="live_nl2sql_semantic_coverage",
        domain="correctness",
        name="当前全量语义槽位覆盖与保守编译",
        status="pass" if semantic_coverage_complete else "fail",
        score=100 if semantic_coverage_complete else 0,
        summary=(
            f"覆盖证据 {int(agg.get('semantic_coverage_cases') or 0)}/{n}；"
            f"平均覆盖 {float(agg.get('semantic_coverage_mean') or 0):.1%}；"
            f"未闭合 {int(agg.get('semantic_unresolved_cases') or 0)}；"
            f"带缺口误编译 {int(agg.get('semantic_unsafe_compile_cases') or 0)}"
        ),
        evidence={
            "run_id": data.get("run_id"),
            "coverage_cases": int(agg.get("semantic_coverage_cases") or 0),
            "coverage_mean": float(agg.get("semantic_coverage_mean") or 0),
            "unresolved_cases": int(agg.get("semantic_unresolved_cases") or 0),
            "unsafe_compile_cases": int(
                agg.get("semantic_unsafe_compile_cases") or 0
            ),
        },
        remediation=(
            None if semantic_coverage_complete
            else "补齐 detected/represented/unresolved 槽位证据；存在缺口时必须 abstain，禁止确定性编译。"
        ),
    ))
    peak_case = max(
        data.get("cases") or [],
        key=lambda case: float(case.get("total_tokens") or 0),
        default={},
    )
    token_max = float(agg.get("tokens_max") or peak_case.get("total_tokens") or 0)
    llm_attempts = int(agg.get("llm_attempts") or 0)
    token_max_case_id = (
        str(agg.get("tokens_max_case_id") or peak_case.get("id") or "")
        if token_max > 0 else ""
    )
    token_limit = float(EXPERIMENT_GATE_LIMITS["tokens_max"])
    token_ok = (
        (llm_attempts == 0 and token_max == 0)
        or (llm_attempts > 0 and 0 < token_max <= token_limit)
    )
    results.append(CheckResult(
        id="live_nl2sql_token_max", domain="efficiency", name="当前全量单题 Token 边界",
        status="pass" if token_ok else "fail", score=100 if token_ok else 0,
        summary=(
            (
                "本轮全部命中确定性路径，0 次模型调用、0 tokens"
                if llm_attempts == 0 and token_max == 0
                else (
                    f"最大 {token_max:.0f} tokens（{token_max_case_id or 'N/A'}），"
                    f"目标 0 < max ≤ {token_limit:.0f}"
                )
            )
        ),
        evidence={"run_id": data.get("run_id"), "tokens_max": token_max,
                  "tokens_max_case_id": token_max_case_id, "target": token_limit,
                  "llm_attempts": llm_attempts},
        remediation=None if token_ok else "检查单题模型 route 与 Prompt 装包，禁止大上下文进入可选后处理节点。",
    ))
    return results


def _domain_scores(checks: list[CheckResult]) -> dict[str, float | None]:
    domains = ["correctness", "safety", "reliability", "efficiency", "experience", "governance"]
    scores: dict[str, float | None] = {}
    for domain in domains:
        rows = [
            item for item in checks
            if item.domain == domain
            and item.score is not None
            and item.status != "skip"
        ]
        if domain == "correctness" and rows:
            result_rows = [
                item for item in rows
                if item.id in {"nl2sql_results", "live_nl2sql_results"}
            ]
            supporting_rows = [item for item in rows if item not in result_rows]
            if result_rows and supporting_rows:
                raw = OmegaConf.to_container(
                    OmegaConf.load(GATES_PATH), resolve=True,
                ) or {}
                contract = dict(raw.get("correctness_scoring") or {})
                result_weight = float(
                    contract.get("result_equivalence_weight", 0.70)
                )
                supporting_weight = float(
                    contract.get("supporting_checks_weight", 0.30)
                )
                total_weight = max(1e-9, result_weight + supporting_weight)
                result_score = sum(
                    float(item.score) for item in result_rows
                ) / len(result_rows)
                supporting_score = sum(
                    float(item.score) for item in supporting_rows
                ) / len(supporting_rows)
                scores[domain] = round(
                    (
                        result_score * result_weight
                        + supporting_score * supporting_weight
                    ) / total_weight,
                    1,
                )
                continue
        values = [float(item.score) for item in rows]
        scores[domain] = round(sum(values) / len(values), 1) if values else None
    return scores


def _overall(scores: dict[str, float | None]) -> float | None:
    raw = OmegaConf.to_container(OmegaConf.load(GATES_PATH), resolve=True) or {}
    weights = raw.get("weights") or {}
    active = [(scores[key], float(weight)) for key, weight in weights.items() if scores.get(key) is not None]
    total_weight = sum(weight for _, weight in active)
    return round(sum(score * weight for score, weight in active) / total_weight, 1) if total_weight else None


def _recommendations(checks: list[CheckResult]) -> list[dict[str, Any]]:
    failed = {item.id for item in checks if item.status in {"fail", "warn"}}
    recommendations = []
    if {"nl2sql_results", "nl2sql_tables", "live_nl2sql_results", "live_nl2sql_tables"} & failed:
        result_check = next((item for item in checks if item.id in {"live_nl2sql_results", "nl2sql_results"}), None)
        table_check = next((item for item in checks if item.id in {"live_nl2sql_tables", "nl2sql_tables"}), None)
        evidence = "；".join(item.summary for item in (result_check, table_check) if item)
        recommendations.append({"priority": "P1", "title": "按失败类别优化召回、口径与 SQL 生成", "evidence": evidence or "正确性门禁未通过。"})
    if {"nl2sql_latency", "live_nl2sql_latency", "nl2sql_experiment_latency"} & failed:
        latency_check = next((item for item in checks if item.id in {
            "live_nl2sql_latency", "nl2sql_latency", "nl2sql_experiment_latency",
        }), None)
        recommendations.append({
            "priority": "P1",
            "title": "按既定发布策略处理延迟门禁并补跑可比基线",
            "evidence": (
                f"{latency_check.summary if latency_check else '延迟门禁未通过。'}；"
                "当前决策不再新增 TTL/延迟优化任务，因此保留实测与门禁，"
                "由发布策略决定继续阻断、更新同指纹基线或显式调整准入合同。"
            ),
        })
    if {"live_nl2sql_model_reliability", "live_nl2sql_token_max", "nl2sql_experiment_model_reliability"} & failed:
        recommendations.append({"priority": "P1", "title": "修复模型失败、结构化回退与单题 Token 尖峰", "evidence": "平均值和 P95 不能替代单题资源边界与零失败门禁。"})
    if "frontend_e2e" in failed:
        recommendations.append({"priority": "P1", "title": "补齐前端关键路径 E2E 与可访问性回归", "evidence": "生产构建已通过，但关键用户路径没有浏览器自动化证据。"})
    if {"nl2sql_baseline_coverage", "live_nl2sql_coverage"} & failed:
        recommendations.append({"priority": "P1", "title": "重跑指定难度的 50 题 SQL 评测集并治理样本", "evidence": "当前运行未覆盖完整分档基线，不能代表该难度下的聚合、过滤、连接、分组与排序能力。"})
    if "quality_instrumentation" in failed:
        recommendations.append({"priority": "P2", "title": "补齐 trace、反馈和离线评测的质量运营链路", "evidence": "运行态质量证据链存在缺口。"})
    if "business_scenario_matrix" in failed:
        recommendations.append({"priority": "P2", "title": "修复核心业务组合场景", "evidence": "角色、入口、资源状态组合矩阵存在失败或证据缺口。"})
    if "adhoc_isolation_contract" in failed:
        recommendations.append({
            "priority": "P0",
            "title": "修复上传资产隔离、隐私或最终清理门禁",
            "evidence": (
                "上传安全使用独立 adhoc_isolation 分母；不得用普通问数成功"
                "替代越权零执行、共享面零污染和资源零残留证据。"
            ),
        })
    if "resilience_boundary_suite" in failed:
        recommendations.append({"priority": "P2", "title": "修复依赖故障、并发与资源边界", "evidence": "稳定性探针存在失败；不得以顺序问数结果替代故障和并发证据。"})
    if (
        "rag_offline_contract" in failed
        or "rag_live_recall" in failed
        or "hybrid_retrieval_live" in failed
    ):
        recommendations.append({"priority": "P2", "title": "修复 RAG 漏召回、展示字段或索引参数", "evidence": "四路召回或实体展示字段门禁未通过。"})
    if "point_projection_ablation" in failed:
        recommendations.append({
            "priority": "P2",
            "title": "完成 Point 投影消融选择与 active 覆盖重建",
            "evidence": "三种隔离策略的质量门禁、制品指纹或 active 策略对齐尚未通过。",
        })
    covered_groups = (
        {"nl2sql_results", "nl2sql_tables", "live_nl2sql_results", "live_nl2sql_tables"},
        {"nl2sql_latency", "live_nl2sql_latency", "nl2sql_experiment_latency"},
        {"live_nl2sql_model_reliability", "live_nl2sql_token_max", "nl2sql_experiment_model_reliability"},
        {"frontend_e2e"}, {"nl2sql_baseline_coverage", "live_nl2sql_coverage"},
        {"quality_instrumentation"}, {"business_scenario_matrix"},
        {"adhoc_isolation_contract"},
        {"resilience_boundary_suite"},
        {"rag_offline_contract", "rag_live_recall", "hybrid_retrieval_live"},
        {"point_projection_ablation"},
    )
    covered = set().union(*(group for group in covered_groups if group & failed))
    for item in checks:
        if item.status not in {"fail", "warn"} or item.id in covered:
            continue
        recommendations.append({
            "priority": "P1" if item.domain in {"correctness", "reliability", "efficiency"} else "P2",
            "title": f"处理门禁：{item.name}",
            "evidence": f"{item.summary}；{item.remediation or '根据该检查的 evidence 修复后重跑。'}",
            "check_id": item.id,
        })
    return recommendations


def _render_markdown(report: QualityReport) -> str:
    status = "通过" if report.gate_passed else "未通过"
    by_id = {item.id: item for item in report.checks}
    frontend_ready = by_id.get("frontend_e2e") and by_id["frontend_e2e"].status == "pass"
    remaining = []
    if any(by_id.get(check_id) and by_id[check_id].status in {"fail", "warn"} for check_id in ("nl2sql_results", "nl2sql_tables", "live_nl2sql_results", "live_nl2sql_tables")):
        remaining.append("NL2SQL 结果正确性")
    if any(by_id.get(check_id) and by_id[check_id].status in {"fail", "warn"} for check_id in ("nl2sql_latency", "live_nl2sql_latency")):
        remaining.append("端到端延迟门禁")
    remaining_text = "、".join(remaining) if remaining else "其他当前已评测门禁"
    if frontend_ready and report.gate_passed:
        conclusion = "安全、确定性回归和前端浏览器自动化证据已通过；所有当前已评测门禁通过。"
    elif frontend_ready:
        conclusion = (
            f"安全、确定性回归和前端浏览器自动化证据已通过；"
            f"{remaining_text}仍有未通过项。"
        )
    else:
        conclusion = (
            f"安全与确定性回归基础较稳，但{remaining_text}及"
            "前端浏览器自动化证据仍有缺口。"
        )
    conclusion += "综合分只汇总已有证据；未测能力不会被默认视为通过。"
    lines = [
        "# Data Agent 质量评估报告",
        "",
        f"> Run：`{report.run_id}` ｜ Suite：`{report.dataset_suite}` ｜ Profile：`{report.profile}` ｜ 生成时间：{report.generated_at}",
        f"> 质量门禁：**{status}** ｜ 综合分：**{report.overall_score if report.overall_score is not None else 'N/A'} / 100**",
        "", "## 证据来源", "",
    ]
    if report.evidence_sources:
        for name, source in report.evidence_sources.items():
            if not isinstance(source, dict):
                continue
            lines.append(
                f"- **{name}**：run `{source.get('run_id') or 'N/A'}` ｜ "
                f"profile `{source.get('evaluation_profile') or 'unknown'}` ｜ "
                f"用途 {source.get('purpose') or '未标注'} ｜ 制品 `{source.get('artifact') or 'N/A'}`"
            )
    else:
        lines.append("- 当前报告未绑定运行态制品（offline profile）。")
    lines.extend([
        "",
        "## 结论",
        "",
        conclusion,
        "",
        "## 分域得分",
        "",
        "| 领域 | 得分 |",
        "| --- | ---: |",
    ])
    labels = {"correctness": "正确性", "safety": "安全与权限", "reliability": "稳定性", "efficiency": "性能与成本", "experience": "前端体验", "governance": "评测治理"}
    for key, value in report.domain_scores.items():
        lines.append(f"| {labels.get(key, key)} | {value if value is not None else '未测'} |")
    blocking = [
        item for item in report.checks if item.enforcement == "blocking"
    ]
    observational = [
        item for item in report.checks if item.enforcement == "observational"
    ]
    lines.extend([
        "",
        "## 质量门禁明细",
        "",
        f"> Blocking {len(blocking)} 项（失败 {sum(item.status == 'fail' for item in blocking)}）｜"
        f"Observational {len(observational)} 项。",
        "",
        "| 检查 | 领域 | 分账 | 状态 | 得分 | 结论 |",
        "| --- | --- | --- | --- | ---: | --- |",
    ])
    marks = {
        "pass": "✅",
        "warn": "⚠️",
        "fail": "❌",
        "skip": "⏭️",
        "observe": "👁️",
    }
    for item in report.checks:
        lines.append(
            f"| {item.name} | {labels.get(item.domain, item.domain)} | "
            f"{item.enforcement} | {marks[item.status]} {item.status} | "
            f"{item.score if item.score is not None else '—'} | {item.summary} |"
        )
    lines.extend(["", "## 评估后建议", ""])
    for item in report.recommendations:
        lines.append(f"### {item['priority']} · {item['title']}")
        lines.append("")
        lines.append(item["evidence"])
        lines.append("")
    lines.extend([
        "## 运行方式", "", "```powershell",
        "# 不依赖外部服务，生成报告",
        "python -m evals.legacy.run_quality", "",
        "# CI 门禁：有 fail 时返回非零",
        "python -m evals.legacy.run_quality --enforce", "",
        "# 题数由各套件 manifest 决定；选择一档运行真实门禁，需要 MySQL/Qdrant/ES/Embedding/LLM",
        "python -m evals.legacy.run_eval --suite easy_sql_v1 --workspace-id <workspace-id>",
        "python -m evals.legacy.run_quality --profile full --suite easy_sql_v1 --live-run <run.json>",
        "python -m evals.legacy.run_eval --suite medium_sql_v1 --workspace-id <workspace-id>",
        "python -m evals.legacy.run_quality --profile full --suite medium_sql_v1 --live-run <run.json>",
        "python -m evals.legacy.run_eval --suite hard_sql_v1 --workspace-id <workspace-id>",
        "python -m evals.legacy.run_quality --profile full --suite hard_sql_v1 --live-run <run.json>", "",
        "# 逐条执行 RAG 集自身的 case-id/问题合同",
        "python -m evals.legacy.run_rag_live --workspace-id <workspace-id>", "",
        "# BGE-M3 Dense/Sparse/Hybrid 28 题真实门禁",
        "python -m evals.legacy.hybrid_retrieval --workspace-id <workspace-id> --isolated-indexes --enforce", "",
        "# 三种 Point 投影策略的隔离 collection 消融与选择",
        "python -m evals.legacy.point_projection --workspace-id <workspace-id> --enforce", "",
        "# 上传资产离线隔离合同；不访问真实用户上传",
        "python -m evals.legacy.adhoc_isolation_suite --evidence-level offline_contract --enforce", "",
        "# 本机 Docker 隔离依赖资源创建/二次清理证据",
        "python -m evals.legacy.adhoc_isolation_suite --evidence-level isolated_dependencies --enforce", "",
        "# 复用已完成的 NL2SQL 与精确 RAG 制品生成同一份 full 报告",
        "python -m evals.legacy.run_quality --profile full --live-run <run.json> --rag-live-run <rag_run.json> --workspace-id <workspace-id> --experiment-id <experiment-id>",
        "```", "",
    ])
    return "\n".join(lines)


def _experiment_runs(
    experiment_id: str,
    dataset_suite: str,
    runs_dir: Path | None = None,
) -> list[tuple[Path, dict[str, Any]]]:
    """Load one experiment without ever crossing its governed dataset suite."""
    if dataset_suite not in SUITE_SPECS:
        raise ValueError(f"未知数据集套件：{dataset_suite}")
    matches: list[tuple[Path, dict[str, Any]]] = []
    root = runs_dir or RUNS_ROOT
    for path in sorted(root.glob("run_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        experiment = payload.get("experiment") or {}
        if (
            experiment.get("experiment_id") != experiment_id
            or str(payload.get("dataset_suite") or "") != dataset_suite
        ):
            continue
        matches.append((path, payload))
    return matches


def resolve_experiment_live_run(
    experiment_id: str,
    dataset_suite: str,
    runs_dir: Path | None = None,
) -> Path | None:
    """Deterministically reuse the highest valid repeat within one suite."""
    candidates = [(
            int(experiment.get("repeat_index") or 0), str(payload.get("run_id") or ""), path,
        ) for path, payload in _experiment_runs(
            experiment_id, dataset_suite, runs_dir,
        ) if payload.get("valid") is True
        for experiment in [payload.get("experiment") or {}]
    ]
    return max(candidates, default=(0, "", None), key=lambda item: (item[0], item[1]))[2]


def _run_source(path: Path, data: dict, purpose: str) -> dict[str, Any]:
    return {
        "artifact": str(path),
        "run_id": data.get("run_id"),
        "evaluation_profile": (
            evaluation_profile_name(data.get("evaluation_profile"), default="")
            or evaluation_profile_name((data.get("fingerprint") or {}).get("evaluation_profile"))
        ),
        "fingerprint": (data.get("fingerprint") or {}).get("run_fingerprint") or "",
        "purpose": purpose,
    }


def _prepare_full_live_run(
    live_run: Path | None,
    *,
    experiment_id: str | None,
    dataset_suite: str,
    execute_live_run: bool,
    runs_root: Path = RUNS_ROOT,
) -> tuple[int, str, Path | None, dict[str, Any], str, set[Path]]:
    """解析显式制品；只有明确授权时才启动付费真实评测。"""
    if live_run is None and experiment_id:
        live_run = resolve_experiment_live_run(
            experiment_id,
            dataset_suite,
            runs_root,
        )
    before = set(runs_root.glob("run_*.json")) if runs_root.exists() else set()
    if live_run is not None:
        try:
            payload = json.loads(live_run.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return (
                2,
                f"指定 run 不可读取：{type(exc).__name__}",
                live_run,
                {},
                "invalid_explicit_artifact",
                before,
            )
        if not isinstance(payload, dict):
            return (
                2,
                "指定 run 必须是 JSON object",
                live_run,
                {},
                "invalid_explicit_artifact",
                before,
            )
        valid = payload.get("valid") is True and int(payload.get("n") or 0) > 0
        if valid:
            fingerprint = payload.get("fingerprint") or {}
            profile_name = evaluation_profile_name(
                payload.get("evaluation_profile"),
                default=evaluation_profile_name(
                    fingerprint.get("evaluation_profile"),
                    default="",
                ),
            )
            if not profile_name:
                differences = ["config_hash"]
            else:
                try:
                    current_fingerprint = local_version_fingerprint(
                        cases=load_suite_cases(dataset_suite),
                        evaluation_profile=profile_name,
                    )
                except (OSError, TypeError, ValueError):
                    current_fingerprint = {}
                differences = compare_local_version_fingerprints(
                    fingerprint,
                    current_fingerprint,
                )["differences"]
            if differences:
                return (
                    2,
                    "指定 run 与当前本地版本不匹配：" + ", ".join(differences),
                    live_run,
                    {},
                    "artifact_version_mismatch",
                    before,
                )
        return (
            0 if valid else 2,
            f"复用已完成的有效真实评测：{live_run}" if valid else "指定 run 无效",
            live_run,
            payload,
            "experiment_repeat_reuse" if experiment_id else "explicit_artifact_reuse",
            before,
        )
    if experiment_id:
        return (
            2,
            f"experiment {experiment_id} 没有可复用的有效 run；已拒绝启动匿名第四次评测",
            None,
            {},
            "missing_experiment_artifact",
            before,
        )
    if not execute_live_run:
        return (
            2,
            "full profile 缺少显式 --live-run；已拒绝隐式启动模型评测",
            None,
            {},
            "missing_artifact_fail_closed",
            before,
        )
    code, output = _run(
        [
            sys.executable,
            "-m",
            "evals.legacy.run_eval",
            "--suite",
            dataset_suite,
        ],
        PROJECT_ROOT,
        4000,
    )
    return code, output, None, {}, "explicit_live_execution", before


def evaluate(
    profile: str = "offline",
    live_run: Path | None = None,
    *,
    rag_live_run: Path | None = None,
    baseline_selection: dict | None = None,
    experiment_id: str | None = None,
    dataset_suite: str = "easy_sql_v1",
    execute_live_run: bool = False,
) -> QualityReport:
    if dataset_suite not in SUITE_SPECS:
        raise ValueError(f"未知数据集套件：{dataset_suite}")
    evidence_sources: dict[str, Any] = {}
    checks = [
        _timed(check_compile),
        _timed(check_unit_tests),
        _timed(check_security_cases),
        _timed(lambda: check_golden_contract(dataset_suite)),
        _timed(check_documentation),
        _timed(check_frontend_build),
        _timed(check_frontend_e2e),
        _timed(check_instrumentation),
        _timed(check_retrieval_responsibility),
        _timed(check_conversation_memory),
        _timed(check_embedding_capacity),
        _timed(check_query_signals),
        _timed(check_planning_router),
        _timed(check_semantic_repair),
        _timed(check_adhoc_isolation),
    ]
    unit_check = next(item for item in checks if item.id == "unit_tests")
    e2e_check = next(item for item in checks if item.id == "frontend_e2e")
    checks.append(_timed(lambda: check_scenario_matrix(
        unit_status=dict(unit_check.evidence.get("target_status") or {}),
        e2e_status=dict(e2e_check.evidence.get("target_status") or {}),
    )))
    checks.append(_timed(check_resilience_suite))
    checks.append(_timed(check_rag_recall))
    if profile == "full":
        hybrid_check = _timed(check_hybrid_retrieval)
        checks.append(hybrid_check)
        evidence_sources["hybrid_rag"] = {
            "artifact": str(HYBRID_REPORT_PATH),
            "workspace_id": hybrid_check.evidence.get("workspace_id"),
            "query_fingerprint": hybrid_check.evidence.get("query_fingerprint"),
            "purpose": "bge_m3_dense_sparse_live_gate",
        }
        point_check = _timed(check_point_projection)
        checks.append(point_check)
        evidence_sources["point_projection"] = {
            "artifact": str(POINT_PROJECTION_REPORT_PATH),
            "workspace_id": point_check.evidence.get("workspace_id"),
            "query_fingerprint": point_check.evidence.get("query_fingerprint"),
            "purpose": "isolated_point_projection_selection_gate",
        }
        (
            code,
            output,
            live_run,
            live_data,
            execution_mode,
            before,
        ) = _prepare_full_live_run(
            live_run,
            experiment_id=experiment_id,
            dataset_suite=dataset_suite,
            execute_live_run=execute_live_run,
        )
        if code == 0 and live_run is not None:
            evidence_sources["current_nl2sql"] = _run_source(
                live_run,
                live_data,
                execution_mode,
            )
        checks.append(CheckResult(
            id="live_nl2sql_full_run", domain="correctness", name="真实 NL2SQL 全量运行",
            status="pass" if code == 0 else "fail", score=100 if code == 0 else 0,
            summary=(
                f"{dataset_suite} 的 {SUITE_SPECS[dataset_suite]['expected_cases']} 题运行完成"
                if code == 0 else output
            ),
            evidence={
                "dataset_suite": dataset_suite,
                "expected_cases": SUITE_SPECS[dataset_suite]["expected_cases"],
                "exit_code": code,
                "execution_mode": execution_mode,
                "model_execution_requested": execute_live_run,
                "live_run_artifact": str(live_run) if live_run is not None else None,
                "tail": output[-3000:],
            },
            remediation=(
                None
                if code == 0
                else (
                    "先独立运行 run_eval 并通过 --live-run 传入制品；"
                    "如确需由本命令调用模型，显式增加 --execute-live-run。"
                    if execution_mode == "missing_artifact_fail_closed"
                    else "检查显式制品、外部依赖、模型额度和失败用例后重跑。"
                )
            ),
        ))
        after = set(RUNS_ROOT.glob("run_*.json")) if RUNS_ROOT.exists() else set()
        candidates = sorted(after - before, key=lambda path: path.name)
        if code == 0 and (live_run is not None or candidates):
            if live_run is None:
                live_data = json.loads(candidates[-1].read_text(encoding="utf-8"))
                evidence_sources["current_nl2sql"] = _run_source(
                    candidates[-1], live_data, "new_full_run",
                )
            artifact_suite = str(live_data.get("dataset_suite") or "")
            if artifact_suite != dataset_suite:
                checks.append(CheckResult(
                    id="live_nl2sql_suite_match",
                    domain="governance",
                    name="真实评测套件一致性",
                    status="fail",
                    score=0,
                    summary=(
                        f"请求 {dataset_suite}，制品实际为 {artifact_suite}"
                    ),
                    evidence={
                        "requested_suite": dataset_suite,
                        "artifact_suite": artifact_suite,
                        "run_id": live_data.get("run_id"),
                    },
                    remediation="选择与 --suite 一致的 live run，禁止跨套件混算。",
                ))
            checks.extend(live_run_checks(
                live_data,
                dataset_suite=dataset_suite,
            ))
            from evals.legacy.rag_recall import evaluate_live_run
            rag_data = (
                json.loads(rag_live_run.read_text(encoding="utf-8"))
                if rag_live_run is not None else live_data
            )
            evidence_sources["rag"] = _run_source(
                rag_live_run or live_run or candidates[-1], rag_data,
                "explicit_rag_run" if rag_live_run is not None else "derived_from_nl2sql_run",
            )
            live_rag = evaluate_live_run(rag_data)
            live_summary = live_rag.get("summary") or {}
            live_available = bool(live_rag.get("available"))
            live_passed = live_available and bool(live_summary.get("gate_passed"))
            checks.append(CheckResult(
                id="rag_live_recall", domain="correctness", name="RAG 真实索引召回",
                status="pass" if live_passed else "fail",
                score=round(float(live_summary.get("score") or 0), 1) if live_available else 0,
                summary=(
                    f"真实 run {live_summary.get('evaluated_cases', 0)}/{live_summary.get('required_cases', 0)} 条；"
                    f"字段 Recall@K {float(live_summary.get('column_recall_at_k') or 0):.1%}；"
                    f"筛表 {float(live_summary.get('table_filter_recall') or 0):.1%}"
                ) if live_available else str(live_rag.get("message") or "真实召回证据不可用"),
                evidence=live_rag,
                remediation=None if live_passed else "使用指纹完整的全量 run，核对真实 Qdrant/ES 候选、筛表和索引版本。",
            ))
        elif code == 0:
            checks.append(CheckResult(
                "live_nl2sql_artifact", "governance", "真实评测产物", "fail", 0,
                "全量命令成功但没有生成新的结构化 run", {"runs_dir": str(RUNS_ROOT)},
                "修复 run 持久化后重跑，不能只依赖控制台输出。",
            ))
    else:
        checks.extend(baseline_checks(baseline_selection))
    if experiment_id:
        runs = [
            payload for _, payload in _experiment_runs(
                experiment_id, dataset_suite,
            )
        ]
        evidence_sources["experiment"] = {
            "artifact": ", ".join(str(run.get("run_id") or "") for run in runs),
            "run_id": experiment_id,
            "evaluation_profile": (
                evaluation_profile_name(
                    (runs[0].get("experiment") or {}).get("evaluation_profile")
                ) if runs else "unknown"
            ),
            "purpose": "cold_warm_warm_worst_gate",
        }
        baseline_payload = (baseline_selection or {}).get("payload")
        checks.extend(experiment_checks(experiment_report(
            runs, baseline_run=baseline_payload, experiment_id=experiment_id,
        )))
    scores = _domain_scores(checks)
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return QualityReport(
        run_id=run_id,
        generated_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        profile=profile,
        dataset_suite=dataset_suite,
        environment={"python": platform.python_version(), "platform": platform.platform(), "project_root": str(PROJECT_ROOT)},
        checks=checks,
        domain_scores=scores,
        overall_score=_overall(scores),
        gate_passed=not any(
            item.status == "fail" and item.enforcement == "blocking"
            for item in checks
        ),
        recommendations=_recommendations(checks),
        evidence_sources=evidence_sources,
    )


def _persist_report(report: QualityReport, report_root: Path = REPORT_ROOT) -> None:
    """Persist immutable and suite-scoped latest reports."""
    if report.dataset_suite not in SUITE_SPECS:
        raise ValueError(f"未知数据集套件：{report.dataset_suite}")
    report_root.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    md_text = _render_markdown(report)
    report_stem = f"quality_{report.dataset_suite}_{report.run_id}"
    latest_stem = f"quality_{report.dataset_suite}_latest"
    for suffix, content in (("json", json_text), ("md", md_text)):
        (report_root / f"{report_stem}.{suffix}").write_text(
            content, encoding="utf-8",
        )
        (report_root / f"{latest_stem}.{suffix}").write_text(
            content, encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Data Agent 分层质量评估")
    parser.add_argument("--profile", choices=("offline", "full"), default="offline")
    parser.add_argument(
        "--suite",
        choices=tuple(SUITE_SPECS),
        default=None,
        help="不指定时从 --live-run 推断；无制品时默认 easy_sql_v1",
    )
    parser.add_argument("--live-run", type=Path, default=None, help="复用已完成的有效真实 run，不再次调用 LLM")
    parser.add_argument("--rag-live-run", type=Path, default=None, help="复用逐条执行 RAG 契约问题的真实召回制品")
    parser.add_argument("--workspace-id", default=None, help="从 Meta MySQL 解析该工作空间 approved baseline")
    parser.add_argument("--baseline-run", type=Path, default=None, help="显式 baseline run；优先级高于 Meta MySQL")
    parser.add_argument("--experiment-id", default=None, help="对三次 experiment 使用最差值执行质量门禁")
    parser.add_argument(
        "--execute-live-run",
        action="store_true",
        help="显式授权 full profile 在无 --live-run 时调用模型执行 run_eval",
    )
    parser.add_argument("--enforce", action="store_true", help="质量门禁失败时返回退出码 1")
    args = parser.parse_args()
    workspace_id = args.workspace_id or os.getenv("EVAL_WORKSPACE_ID") or None
    live_run = args.live_run.resolve() if args.live_run else None
    dataset_suite = args.suite
    if live_run is None and args.experiment_id:
        dataset_suite = dataset_suite or "easy_sql_v1"
        live_run = resolve_experiment_live_run(
            args.experiment_id, dataset_suite,
        )
    if dataset_suite is None and live_run is not None:
        try:
            live_payload = json.loads(live_run.read_text(encoding="utf-8"))
            dataset_suite = str(live_payload.get("dataset_suite") or "")
        except (OSError, ValueError):
            dataset_suite = "easy_sql_v1"
    dataset_suite = dataset_suite or "easy_sql_v1"
    baseline_selection = resolve_quality_baseline(
        explicit=args.baseline_run.resolve() if args.baseline_run else None,
        workspace_id=workspace_id,
        dataset_suite=dataset_suite,
    )
    report = evaluate(
        args.profile, live_run,
        rag_live_run=args.rag_live_run.resolve() if args.rag_live_run else None,
        baseline_selection=baseline_selection, experiment_id=args.experiment_id,
        dataset_suite=dataset_suite,
        execute_live_run=args.execute_live_run,
    )
    _persist_report(report)
    md_text = _render_markdown(report)
    print(md_text)
    print(f"\n报告目录：{REPORT_ROOT}")
    return 1 if args.enforce and not report.gate_passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
