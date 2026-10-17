"""Read-only TODO-64 evidence and process-local P0 rollback controls."""
from __future__ import annotations

import json

from evals.application.ports.profile_control import RuntimeProfileControl
from evals.legacy.semantic_repair_ablation import LATEST_JSON, build_report


class SemanticRepairEvaluationService:
    def __init__(self, profile_control: RuntimeProfileControl):
        self._profile_control = profile_control

    def latest(self) -> dict:
        report = None
        if LATEST_JSON.exists():
            try:
                report = json.loads(LATEST_JSON.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                report = None
        if report is None:
            report = build_report(write=False)
        report["runtime"] = self._profile_control.status()
        report["decision"]["active_profile"] = report["runtime"]["runtime_profile"]
        report["decision"]["active_unchanged_semantics"] = (
            report["runtime"]["runtime_profile"] == "registry_same_rules_v1"
        )
        return report

    def rollback(self) -> dict:
        return {
            "status": "rolled_back",
            "runtime": self._profile_control.rollback(),
            "message": (
                "语义修复已在当前后端进程回切 P0 inline_v1；"
                "无需数据库迁移、索引重建或重启。"
            ),
        }
