"""Read-only access to the latest TODO-39 resilience artifact."""
from __future__ import annotations

import json

from evals.legacy.resilience_suite import LATEST_PATH, validate_suite


class ResilienceEvaluationService:
    def latest(self) -> dict:
        if LATEST_PATH.exists():
            try:
                return json.loads(LATEST_PATH.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        contract = validate_suite()
        return {
            "available": False,
            "generated_at": None,
            "source": "contract_only",
            "contract": contract,
            "summary": {"total": contract["total"], "passed": 0, "failed": 0, "not_run": contract["total"], "pass_rate": 0, "gate_passed": False},
            "certification": {
                "status": "not_run", "duration_seconds": 0, "required_seconds": 1800, "cycles": 0,
                "evidence_level": "in_process", "live_dependencies_evaluated": False,
                "resource_metrics_collected": False,
            },
            "categories": [], "resource_limits": {}, "cases": [],
        }
