from app.agent_system.governance.deadline import (
    REQUEST_DEADLINE_VERSION,
    compact_deadline_evidence,
    execution_timeout_seconds,
    initialize_request_deadline,
    model_phase_budget,
    route_deadline_at,
    route_timeout_seconds,
)


def test_route_deadline_reserves_execution_tail_from_shared_budget():
    contract = initialize_request_deadline(100.0, budget_ms=5000)
    contract["execution_reserve_ms"] = 1500
    contract["phase_budgets_ms"]["generation"] = 10000
    deadline = route_deadline_at(
        {"request_deadline": contract},
        "sql",
        timeout_seconds=10,
        now=100.0,
    )
    assert deadline == 103.5


def test_execution_timeout_uses_smaller_of_phase_and_remaining_budget():
    contract = initialize_request_deadline(100.0, budget_ms=5000)
    contract["phase_budgets_ms"]["execution"] = 2500
    assert execution_timeout_seconds(
        {"request_deadline": contract},
        now=104.0,
    ) == 1.0


def test_dependency_timeout_cannot_exceed_route_phase_budget():
    contract = initialize_request_deadline(100.0, budget_ms=30000)
    contract["phase_budgets_ms"]["retrieval"] = 7000
    assert route_timeout_seconds(
        {"request_deadline": contract},
        "keyword",
        timeout_seconds=45,
        now=100.0,
    ) == 7.0


def test_dependency_timeout_shrinks_with_consumed_request_budget():
    contract = initialize_request_deadline(100.0, budget_ms=10000)
    contract["execution_reserve_ms"] = 4000
    contract["phase_budgets_ms"]["retrieval"] = 7000
    assert route_timeout_seconds(
        {"request_deadline": contract},
        "keyword",
        timeout_seconds=45,
        now=105.0,
    ) == 1.0


def test_default_contract_reserves_realistic_dw_execution_budget():
    contract = initialize_request_deadline(100.0)
    assert contract["phase_budgets_ms"]["execution"] == 6000
    assert contract["execution_reserve_ms"] == 4000


def test_persisted_deadline_evidence_contains_no_absolute_timestamp():
    contract = initialize_request_deadline(100.0, budget_ms=5000)
    evidence = compact_deadline_evidence(contract, now=100.25)
    assert evidence["contract_version"] == REQUEST_DEADLINE_VERSION
    assert evidence["remaining_ms"] == 4750
    assert evidence["absolute_timestamps_persisted"] is False
    assert "started_at" not in evidence
    assert "deadline_at" not in evidence


def test_model_phase_budget_blocks_calls_without_a_completion_window():
    contract = initialize_request_deadline(100.0, budget_ms=5000)
    contract["execution_reserve_ms"] = 1500
    decision = model_phase_budget(
        {"request_deadline": contract},
        "sql",
        minimum_start_ms=4000,
        timeout_seconds=10,
        now=100.0,
    )
    assert decision["allowed"] is False
    assert decision["available_ms"] == 3500
    assert decision["reason_code"] == "model_phase.insufficient_budget"
    assert decision["absolute_timestamps_persisted"] is False
