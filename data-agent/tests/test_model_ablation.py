from unittest.mock import patch

from evals.legacy.governance import (
    REQUIRED_FINGERPRINT_FIELDS,
    _model_ablation_manifest,
    compare_model_ablation_fingerprints,
)
from evals.legacy.model_ablation import TODO97_EXECUTION_SCOPE, model_ablation_report
from app.infrastructure.clients.llm.capabilities import candidate_capability


def _fingerprint(sql_model: str, *, timeout: int = 22) -> dict:
    route = {
        "model_name": sql_model,
        "provider": "dashscope" if sql_model != "d0" else "deepseek",
        "timeout_seconds": timeout,
        "max_retries": 1,
        "max_completion_tokens": 2048,
        "candidate_id": sql_model,
    }
    safe = {
        "llm": {"routes": {
            "sql": dict(route),
            "correction": {"model_name": "d0", "timeout_seconds": 16},
        }},
        # Historical manifests may duplicate the resolved route.  Both copies
        # must be normalized by model-ablation-v1.
        "llm_route_manifest": {
            "sql": dict(route),
            "correction": {"model_name": "d0", "timeout_seconds": 16},
        },
        "query": {"limit_rows": 1000},
    }
    payload = {field: f"same-{field}" for field in REQUIRED_FINGERPRINT_FIELDS}
    payload.update({
        "complete": True,
        "model": {"routes": safe["llm"]["routes"]},
        "config_hash": f"config-{sql_model}-{timeout}",
        "model_ablation": _model_ablation_manifest(safe),
    })
    return payload


def test_model_ablation_allows_only_registered_sql_identity_fields():
    comparison = compare_model_ablation_fingerprints(
        _fingerprint("d0"), _fingerprint("q1"),
    )
    assert comparison["attribution_valid"]
    assert "model.routes.sql.model_name" in comparison["allowed_differences"]
    assert comparison["unexpected_differences"] == []


def test_model_ablation_rejects_timeout_drift():
    comparison = compare_model_ablation_fingerprints(
        _fingerprint("d0"), _fingerprint("q1", timeout=23),
    )
    assert not comparison["attribution_valid"]
    assert "model.routes.sql.timeout_seconds" in comparison["unexpected_differences"]


def test_model_ablation_rejects_dependency_lock_drift():
    control = _fingerprint("d0")
    candidate = _fingerprint("q1")
    candidate["dependency_lock_hash"] = "sha256:different-lock"

    comparison = compare_model_ablation_fingerprints(control, candidate)

    assert not comparison["attribution_valid"]
    assert "dependency_lock_hash" in comparison["unexpected_differences"]


def test_execution_scope_keeps_d0_as_control_only():
    assert TODO97_EXECUTION_SCOPE == ("Q1", "Q3", "Q4")
    assert "D0" not in TODO97_EXECUTION_SCOPE


def _experiment_summary(experiment_id: str) -> dict:
    return {
        "experiment_id": experiment_id,
        "recordable": True,
        "checks": {
            "first_final_evidence_complete": True,
            "ttsql_gate": True,
            "ttresult_gate": True,
            "ttcomplete_gate": True,
            "token_max_gate": True,
        },
        "worst": {
            "first_result_equivalent": 8,
            "result_equivalent": 9,
            "security_guard_bypass_count": 0,
            "failed_core_llm_calls": 0,
            "tt_complete_p95": 9000,
            "tokens_mean": 1000,
            "tokens_max": 2000,
        },
        "stability_summary": {"stable_pass": 8},
    }


def _run(experiment_id: str, label: str, promotion: bool) -> dict:
    capability = candidate_capability(label)
    return {
        "fingerprint": _fingerprint(label.lower()),
        "experiment": {
            "experiment_id": experiment_id,
            "provider_parameters": {"sql_route": {
                "model_name": capability["model_id"],
                "candidate_id": capability["candidate_id"],
                "provider": capability["provider"],
                "model_provider": capability["model_provider"],
                "access_mode": capability["access_mode"],
                "promotion_eligible": promotion,
                "capability_lifecycle": "officially-listed",
            }},
        },
        "cases": [{"id": "case-1", "result_equivalent": True}],
    }


def test_phase0_is_fail_closed_and_passing_probe_enables_shortlist():
    control_run = _run("d0-exp", "D0", True)
    candidate_run = _run("q1-exp", "Q1", True)

    def fake_experiment_report(runs):
        if not runs:
            return {"recordable": False, "worst": {}, "checks": {}}
        return _experiment_summary(
            (runs[0].get("experiment") or {}).get("experiment_id")
        )

    with patch(
        "evals.legacy.model_ablation.experiment_report",
        side_effect=fake_experiment_report,
    ):
        missing = model_ablation_report(
            [control_run], {"Q1": [candidate_run]},
            phase0={"D0": {"status": "pass"}},
        )
        passed = model_ablation_report(
            [control_run], {"Q1": [candidate_run], "Q3": [], "Q4": []},
            phase0={
                "D0": {"status": "pass"},
                "Q1": {"status": "pass"},
                "Q3": {"status": "fail"},
                "Q4": {"status": "fail"},
            },
        )

    assert not missing["candidates"][0]["checks"]["phase0_passed"]
    assert not missing["candidates"][0]["shortlisted"]
    assert passed["candidates"][0]["shortlisted"]
    assert passed["pareto_top2"] == ["Q1"]
    assert passed["execution_scope"] == ["Q1", "Q3", "Q4"]
    assert passed["control"]["experiment_id"] == "d0-exp"
    assert passed["phase2_candidates"][0] == "D0"
    assert passed["execution_complete"]


def test_failed_phase0_is_complete_without_phase1_and_not_ranked():
    control_run = _run("d0-exp", "D0", True)
    with patch(
        "evals.legacy.model_ablation.experiment_report",
        side_effect=lambda runs: (
            _experiment_summary("d0-exp") if runs else {
                "recordable": False, "worst": {}, "checks": {},
            }
        ),
    ):
        report = model_ablation_report(
            [control_run], {"Q1": [], "Q3": [], "Q4": []},
            phase0={
                "D0": {"status": "pass"},
                "Q1": {"status": "fail"},
                "Q3": {"status": "fail"},
                "Q4": {"status": "fail"},
            },
        )
    assert report["execution_complete"]
    assert report["pareto_top2"] == []
