"""Read-only owner/admin view of the latest executable scenario matrix report."""
from __future__ import annotations

import json

from evals.legacy.scenario_matrix import REPORT_PATH, build_report, load_matrix, validate_matrix


class ScenarioMatrixService:
    def latest(self) -> dict:
        if REPORT_PATH.exists():
            try:
                payload = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
                if isinstance(payload.get("cases"), list):
                    return payload
            except (OSError, ValueError, AttributeError):
                pass
        cases = load_matrix()
        return build_report(cases, {}, source="contract_only", contract=validate_matrix(cases))
