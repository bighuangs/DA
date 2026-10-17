import pytest

from app.agent_system.contracts import InvocationContext
from app.agent_system.governance import (
    BudgetExceeded,
    BudgetPolicy,
    BudgetUsage,
    PermissionPolicy,
    TerminationPolicy,
    TerminationReason,
)


def test_budget_usage_fails_closed_at_configured_limit() -> None:
    policy = BudgetPolicy(max_model_calls=1)
    usage = BudgetUsage().consume(policy, model_calls=1)

    with pytest.raises(BudgetExceeded) as exc_info:
        usage.consume(policy, model_calls=1)
    assert exc_info.value.dimension == "model_calls"


def test_permission_policy_checks_operation_and_data_source() -> None:
    context = InvocationContext(
        request_id="req-1",
        actor_id="user-1",
        workspace_id="workspace-1",
        roles=("analyst",),
        allowed_data_sources=("sales",),
    )
    policy = PermissionPolicy(
        allowed_roles=frozenset({"analyst"}),
        allowed_operations=frozenset({"execute_sql"}),
    )

    assert policy.evaluate(
        context,
        operation="execute_sql",
        data_source="sales",
    ).allowed
    assert policy.evaluate(
        context,
        operation="execute_sql",
        data_source="finance",
    ).reason_code == "data_source.denied"


def test_termination_policy_prioritizes_cancellation() -> None:
    decision = TerminationPolicy().decide(
        {"status": "failed", "error": {"code": "x"}},
        cancelled=True,
        deadline_exceeded=True,
    )

    assert decision.terminate is True
    assert decision.reason is TerminationReason.CANCELLED
