"""Evaluation result contracts shared by quality checks and report renderers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Status = Literal["pass", "warn", "fail", "skip", "observe"]
Enforcement = Literal["blocking", "observational"]


@dataclass(slots=True)
class CheckResult:
    id: str
    domain: str
    name: str
    status: Status
    score: float | None
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)
    remediation: str | None = None
    duration_ms: float = 0
    enforcement: Enforcement = "blocking"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["blocking"] = self.enforcement == "blocking"
        return payload


@dataclass(slots=True)
class QualityReport:
    run_id: str
    generated_at: str
    profile: str
    dataset_suite: str
    environment: dict[str, Any]
    checks: list[CheckResult]
    domain_scores: dict[str, float | None]
    overall_score: float | None
    gate_passed: bool
    recommendations: list[dict[str, Any]]
    evidence_sources: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checks"] = [item.to_dict() for item in self.checks]
        blocking = [
            item for item in self.checks if item.enforcement == "blocking"
        ]
        observational = [
            item for item in self.checks if item.enforcement == "observational"
        ]
        payload["check_accounting"] = {
            "blocking": {
                "total": len(blocking),
                "failed": sum(item.status == "fail" for item in blocking),
                "ids": [item.id for item in blocking],
                "failed_ids": [
                    item.id for item in blocking if item.status == "fail"
                ],
            },
            "observational": {
                "total": len(observational),
                "observed": sum(item.status == "observe" for item in observational),
                "ids": [item.id for item in observational],
            },
        }
        return payload
