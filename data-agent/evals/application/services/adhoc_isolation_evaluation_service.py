"""Read-only owner/admin access to the latest upload-isolation evidence."""
from __future__ import annotations

import json

from evals.legacy.adhoc_isolation_suite import (
    LATEST_PATH,
    FACET,
    validate_suite,
)


class AdhocIsolationEvaluationService:
    def latest(self) -> dict:
        if LATEST_PATH.exists():
            try:
                payload = json.loads(LATEST_PATH.read_text(encoding="utf-8"))
                if (
                    payload.get("facet") == FACET
                    and isinstance(payload.get("cases"), list)
                ):
                    return payload
            except (OSError, ValueError, AttributeError):
                pass
        contract = validate_suite()
        return {
            "available": False,
            "source": "contract_only",
            "run_id": None,
            "generated_at": None,
            "facet": FACET,
            "suite_version": contract["version"],
            "evidence_level": "not_run",
            "production_equivalent": False,
            "contract": {
                key: value
                for key, value in contract.items()
                if key != "fixture_contracts"
            },
            "fixture_manifest": {
                "fixture_count": contract["fixtures"],
                "origin_counts": {},
                "provenance_completeness": 0,
                "real_user_fixture_count": 0,
            },
            "sandbox": {
                "manifest": {},
                "preflight": {"valid": False, "errors": ["尚未执行"]},
                "dependency_evidence": {"status": "not_run"},
            },
            "denominator": {
                "declared": contract["total"],
                "eligible": contract["total"] if contract["valid"] else 0,
                "selected": 0,
                "attempted": 0,
                "passed": 0,
                "failed": 0,
                "skipped": contract["total"],
                "infrastructure_failed": 0,
                "cleanup_failed": 0,
                "attempt_records": 0,
                "trace_records": 0,
            },
            "metrics": {
                "real_upload_content_occurrences": 0,
                "shared_resource_delta": 0,
                "sandbox_residual_resources": 0,
                "unauthorized_db_execution_count": 0,
                "schema_index_pollution_count": 0,
                "cleanup_success_rate": 0,
                "compensation_recovery_rate": 0,
                "denominator_consistency": False,
                "cross_tenant_leak_count": 0,
                "fixture_provenance_completeness": 0,
            },
            "summary": {
                "total": contract["total"],
                "passed": 0,
                "failed": 0,
                "pass_rate": 0,
                "gate_passed": False,
            },
            "cases": [],
            "run_command": (
                "python -m evals.legacy.adhoc_isolation_suite "
                "--evidence-level offline_contract --enforce"
            ),
        }
