"""Read-only TODO-63 planning evaluation artifacts and safe runtime rollback."""

from __future__ import annotations

import json

from evals.application.ports.profile_control import RuntimeProfileControl
from evals.legacy.planning_router_ablation import LATEST_JSON, build_report


class PlanningEvaluationService:
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
        report["decision"]["active_profile"] = report["runtime"][
            "runtime_profile"
        ]
        report["decision"]["active_unchanged"] = (
            report["runtime"]["runtime_profile"] == "keyword_length_v1"
        )
        return report

    def rollback(self) -> dict:
        return {
            "status": "rolled_back",
            "runtime": self._profile_control.rollback(),
            "message": (
                "规划路由已在当前进程立即回切到预注册 fallback；"
                "无需数据库迁移或索引重建。"
            ),
        }
