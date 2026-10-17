"""Execute the TODO-38 matrix against exact unittest and Playwright evidence targets."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from evals.legacy.scenario_matrix import FRONTEND_ROOT, REPORT_PATH, build_report, load_matrix, validate_matrix
from evals.paths import PROJECT_ROOT


def _parse_unittest_statuses(output: str) -> dict[str, str]:
    """Parse verbose unittest output, including IsolatedAsyncio debug lines."""
    starts = list(re.finditer(r"^(test_\w+) \(([^)]+)\) \.\.\. ", output, re.MULTILINE))
    statuses: dict[str, str] = {}
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(output)
        chunk = output[match.end():end]
        status = re.search(r"(?:^|[ \t])(ok|FAIL|ERROR|skipped[^\r\n]*)\r?$", chunk, re.MULTILINE)
        if status:
            method, owner = match.group(1), match.group(2)
            target = owner if owner.endswith(f".{method}") else f"{owner}.{method}"
            statuses[target] = status.group(1)
    return statuses


def _run_unit(targets: list[str]) -> dict[tuple[str, str], dict]:
    started = time.perf_counter()
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [
        str(PROJECT_ROOT / "tests"), os.environ.get("PYTHONPATH", ""),
    ]))}
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", *targets, "-v"], cwd=PROJECT_ROOT,
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300, env=env,
    )
    duration = (time.perf_counter() - started) * 1000
    output = completed.stdout + completed.stderr
    parsed = _parse_unittest_statuses(output)
    results = {}
    for target in targets:
        passed = parsed.get(target) == "ok"
        results[("unit", target)] = {
            "status": "passed" if passed else "failed", "duration_ms": duration / max(1, len(targets)),
            "error": None if passed else f"unittest 失败或未返回目标：{target}",
        }
    return results


def collect_playwright_specs(suites: list[dict], output: dict[str, dict]) -> None:
    for suite in suites:
        for spec in suite.get("specs") or []:
            statuses = [result.get("status") for test in spec.get("tests") or [] for result in test.get("results") or []]
            output[str(spec.get("title"))] = {
                "status": "passed" if statuses and all(item == "passed" for item in statuses) else "failed",
                "duration_ms": sum(float(result.get("duration") or 0) for test in spec.get("tests") or [] for result in test.get("results") or []),
                "error": None if statuses and all(item == "passed" for item in statuses) else "Playwright 用例失败",
            }
        collect_playwright_specs(suite.get("suites") or [], output)


def _run_e2e(targets: list[str]) -> dict[tuple[str, str], dict]:
    with tempfile.TemporaryDirectory() as directory:
        output_path = Path(directory) / "playwright.json"
        env = {**os.environ, "PLAYWRIGHT_JSON_OUTPUT_FILE": str(output_path), "NO_COLOR": "1"}
        npm = shutil.which("npm")
        if not npm:
            return {("e2e", target): {"status": "failed", "duration_ms": 0, "error": "未找到 npm"} for target in targets}
        # The scenario matrix targets the deterministic contract suite.  Keep the
        # real-backend smoke suite as an independent quality-gate layer so npm
        # argument forwarding cannot overwrite this JSON report.
        completed = subprocess.run(
            [npm, "run", "test:e2e:mock", "--", "--reporter=json"], cwd=FRONTEND_ROOT,
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=420, env=env,
        )
        specs: dict[str, dict] = {}
        if output_path.exists():
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            collect_playwright_specs(payload.get("suites") or [], specs)
        return {
            ("e2e", target): specs.get(target, {
                "status": "failed", "duration_ms": 0,
                "error": f"Playwright 未返回目标；进程退出码 {completed.returncode}",
            }) for target in targets
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="执行核心业务场景矩阵")
    parser.add_argument("--skip-e2e", action="store_true", help="只执行确定性单测证据；E2E 场景标记 not_run")
    args = parser.parse_args()
    cases = load_matrix()
    contract = validate_matrix(cases)
    statuses: dict[tuple[str, str], dict] = {}
    if contract["valid"]:
        unit_targets = sorted({case["evidence"]["target"] for case in cases if case["evidence"]["type"] == "unit"})
        e2e_targets = sorted({case["evidence"]["target"] for case in cases if case["evidence"]["type"] == "e2e"})
        statuses.update(_run_unit(unit_targets))
        if not args.skip_e2e:
            statuses.update(_run_e2e(e2e_targets))
    report = build_report(cases, statuses, source="standalone_runner", contract=contract)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = report["summary"]
    print(f"场景矩阵：{summary['passed']}/{summary['total']} 通过，失败 {summary['failed']}，未执行 {summary['not_run']}")
    for item in report["capabilities"]:
        print(f"- {item['capability']}: {item['passed']}/{item['total']}")
    if contract["errors"]:
        print("契约错误：")
        for error in contract["errors"]:
            print(f"- {error}")
    print(f"报告：{REPORT_PATH}")
    return 0 if summary["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
