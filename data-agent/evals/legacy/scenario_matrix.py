"""Executable business-scenario matrix contract and report helpers (TODO-38)."""
from __future__ import annotations

import hashlib
import importlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from evals.paths import LEGACY_DATASETS_ROOT, PROJECT_ROOT, REPORTS_ROOT


FRONTEND_ROOT = PROJECT_ROOT.parent / "date-agent-frontend"
MATRIX_PATH = LEGACY_DATASETS_ROOT / "scenario_matrix.yaml"
REPORT_PATH = REPORTS_ROOT / "scenario_matrix_latest.json"
CAPABILITIES = {
    "conversation", "chart", "knowledge", "adhoc", "masking", "rbac", "access_policy", "dashboard",
}
FAILURE_LAYERS = {"none", "identity", "resource", "policy", "sql", "dependency", "presentation"}
REQUIRED_TAGS = {
    "cross_workspace", "removed_member", "sensitive_output", "sql_entry",
}
REQUIRED_ROLES = {"owner", "admin", "analyst", "viewer", "removed"}
REQUIRED_FIELDS = {
    "id", "capability", "title", "risk", "principal", "entrypoint",
    "resource_state", "expected", "evidence", "tags",
}

def load_matrix(path: Path | None = None) -> list[dict[str, Any]]:
    source = path or MATRIX_PATH
    return [dict(item) for item in (OmegaConf.to_container(OmegaConf.load(source), resolve=True) or [])]


def _e2e_titles() -> set[str]:
    titles: set[str] = set()
    pattern = re.compile(r"\btest\(\s*[\"'](.+?)[\"']\s*,")
    for path in (FRONTEND_ROOT / "e2e").glob("*.spec.js"):
        titles.update(pattern.findall(path.read_text(encoding="utf-8")))
    return titles


def _unit_target_exists(target: str) -> bool:
    try:
        module_name, class_name, method_name = target.rsplit(".", 2)
        test_class = getattr(importlib.import_module(module_name), class_name)
        return callable(getattr(test_class, method_name, None)) and method_name.startswith("test_")
    except (ValueError, ImportError, AttributeError):
        return False


def validate_matrix(cases: list[dict] | None = None, *, min_cases: int = 50,
                    min_per_capability: int = 5) -> dict:
    cases = cases if cases is not None else load_matrix()
    errors: list[str] = []
    ids: set[str] = set()
    capabilities: Counter = Counter()
    roles: set[str] = set()
    tags: set[str] = set()
    layers: set[str] = set()
    e2e_titles = _e2e_titles()
    evidence_types: Counter = Counter()
    for index, case in enumerate(cases, 1):
        missing = sorted(REQUIRED_FIELDS - set(case))
        if missing:
            errors.append(f"第 {index} 条缺少字段：{','.join(missing)}")
            continue
        case_id = str(case.get("id") or "")
        if case_id in ids:
            errors.append(f"重复场景 ID：{case_id}")
        ids.add(case_id)
        capability = str(case.get("capability") or "")
        if capability not in CAPABILITIES:
            errors.append(f"{case_id} capability 无效：{capability}")
        capabilities[capability] += 1
        roles.add(str(case.get("principal") or ""))
        tags.update(str(item) for item in (case.get("tags") or []))
        expected = case.get("expected") or {}
        layer = str(expected.get("failure_layer") or "")
        layers.add(layer)
        if layer not in FAILURE_LAYERS:
            errors.append(f"{case_id} failure_layer 无效：{layer}")
        if expected.get("decision") not in {"allow", "deny", "pass"}:
            errors.append(f"{case_id} expected.decision 无效")
        evidence = case.get("evidence") or {}
        evidence_type = str(evidence.get("type") or "")
        target = str(evidence.get("target") or "")
        evidence_types[evidence_type] += 1
        if evidence_type == "unit" and not _unit_target_exists(target):
            errors.append(f"{case_id} 单测证据不存在：{target}")
        elif evidence_type == "e2e" and target not in e2e_titles:
            errors.append(f"{case_id} E2E 证据不存在：{target}")
        elif evidence_type not in {"unit", "e2e"}:
            errors.append(f"{case_id} evidence.type 只允许 unit/e2e")
    if len(cases) < min_cases:
        errors.append(f"场景总数 {len(cases)}，低于门槛 {min_cases}")
    for capability in sorted(CAPABILITIES):
        if capabilities[capability] < min_per_capability:
            errors.append(f"{capability} 仅 {capabilities[capability]} 条，低于门槛 {min_per_capability}")
    missing_roles = sorted(REQUIRED_ROLES - roles)
    missing_tags = sorted(REQUIRED_TAGS - tags)
    missing_layers = sorted(FAILURE_LAYERS - layers)
    if missing_roles:
        errors.append("缺少角色：" + ",".join(missing_roles))
    if missing_tags:
        errors.append("缺少强制场景标签：" + ",".join(missing_tags))
    if missing_layers:
        errors.append("缺少失败层：" + ",".join(missing_layers))
    body = json.dumps(cases, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "valid": not errors, "errors": errors, "total": len(cases),
        "capabilities": dict(sorted(capabilities.items())),
        "roles": sorted(roles), "tags": sorted(tags), "failure_layers": sorted(layers),
        "evidence_types": dict(evidence_types),
        "matrix_version": "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest(),
    }


def build_report(cases: list[dict], evidence_status: dict[tuple[str, str], dict],
                 *, source: str, contract: dict | None = None) -> dict:
    contract = contract or validate_matrix(cases)
    rows = []
    for case in cases:
        evidence = case.get("evidence") or {}
        result = evidence_status.get((str(evidence.get("type")), str(evidence.get("target"))), {})
        status = str(result.get("status") or "not_run")
        rows.append({
            **case, "status": status,
            "duration_ms": round(float(result.get("duration_ms") or 0), 1),
            "error": str(result.get("error") or "")[:500] or None,
        })
    status_counts = Counter(row["status"] for row in rows)
    by_capability: dict[str, dict] = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0, "not_run": 0})
    for row in rows:
        bucket = by_capability[row["capability"]]
        bucket["total"] += 1
        key = "passed" if row["status"] == "passed" else "failed" if row["status"] == "failed" else "not_run"
        bucket[key] += 1
    passed = int(status_counts.get("passed", 0))
    executed = passed + int(status_counts.get("failed", 0))
    return {
        "available": True,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source, "contract": contract,
        "summary": {
            "total": len(rows), "executed": executed, "passed": passed,
            "failed": int(status_counts.get("failed", 0)),
            "not_run": len(rows) - executed,
            "pass_rate": round(passed / executed, 4) if executed else 0.0,
            "gate_passed": contract.get("valid") is True and executed == len(rows) and passed == len(rows),
        },
        "capabilities": [{"capability": key, **value} for key, value in sorted(by_capability.items())],
        "cases": rows,
    }


def report_from_suite_status(*, unit_passed: bool, e2e_passed: bool) -> dict:
    cases = load_matrix()
    contract = validate_matrix(cases)
    status = {}
    for case in cases:
        evidence = case["evidence"]
        passed = unit_passed if evidence["type"] == "unit" else e2e_passed
        status[(evidence["type"], evidence["target"])] = {
            "status": "passed" if passed else "failed",
            "error": None if passed else f"{evidence['type']} 回归套件未通过",
        }
    return build_report(cases, status, source="quality_gate_suite_reuse", contract=contract)


def report_from_target_status(*, unit_status: dict[str, dict],
                              e2e_status: dict[str, dict]) -> dict:
    """Build the gate from exact targets already executed by the parent suites."""
    cases = load_matrix()
    contract = validate_matrix(cases)
    status: dict[tuple[str, str], dict] = {}
    for case in cases:
        evidence = case["evidence"]
        target = str(evidence["target"])
        source = unit_status if evidence["type"] == "unit" else e2e_status
        result = source.get(target)
        status[(evidence["type"], target)] = result or {
            "status": "not_run", "duration_ms": 0,
            "error": f"对应 {evidence['type']} target 未返回执行结果",
        }
    return build_report(cases, status, source="quality_gate_exact_targets", contract=contract)
