import json
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from app.agent_system.contracts import (
    Error,
    ExecutionSnapshot,
    ExecutionStatus,
    InvocationContext,
    OutputEvent,
    OutputEventType,
    QueryInput,
    deserialize_state,
    new_state,
    serialize_state,
)


def _query_input() -> QueryInput:
    return QueryInput(
        request_id="req-1",
        query="本月销售额是多少？",
        actor_id="user-1",
        workspace_id="workspace-1",
        conversation_id="conversation-1",
        trace_id="trace-1",
        analysis_enabled=True,
        attributes={"source": "api", "attempt": 1},
    )


def test_versioned_contracts_json_roundtrip() -> None:
    query_input = _query_input()
    assert QueryInput.from_json(query_input.to_json()) == query_input

    error = Error(
        code="sql.validation_failed",
        message="SQL 校验失败",
        retryable=True,
        stage="validate_sql",
        details={"violations": ["unknown_column"]},
    )
    event = OutputEvent(
        request_id=query_input.request_id,
        sequence=2,
        type=OutputEventType.ERROR,
        emitted_at="2026-08-14T12:00:00+00:00",
        payload={"node": "validate_sql"},
        error=error,
    )
    assert OutputEvent.from_json(event.to_json()) == event

    snapshot = ExecutionSnapshot(
        request_id=query_input.request_id,
        agent_id="data-agent.nl2sql",
        status=ExecutionStatus.FAILED,
        started_at="2026-08-14T12:00:00+00:00",
        finished_at="2026-08-14T12:00:01+00:00",
        last_sequence=2,
        state={"status": "failed"},
        error=error,
        metrics={"elapsed_ms": 1000},
    )
    assert ExecutionSnapshot.from_json(snapshot.to_json()) == snapshot


def test_state_v2_roundtrip_contains_only_json_values() -> None:
    state = new_state(_query_input())
    restored = deserialize_state(serialize_state(state))

    assert restored == state
    assert json.loads(json.dumps(restored, ensure_ascii=False)) == restored


def test_contract_rejects_unknown_version() -> None:
    value = _query_input().to_dict()
    value["version"] = "query-input/v99"
    with pytest.raises(ValueError, match="不支持"):
        QueryInput.from_dict(value)


def test_invocation_context_is_immutable_and_contains_no_runtime_resources() -> None:
    context = InvocationContext(
        request_id="req-1",
        actor_id="user-1",
        workspace_id="workspace-1",
        roles=("analyst",),
        allowed_data_sources=("sales",),
        feature_flags=frozenset({"semantic-ir-v2"}),
    )

    forbidden = ("client", "session", "lock", "secret", "password", "credential", "api_key")
    field_names = {item.name.lower() for item in fields(context)}
    assert not {
        name for name in field_names if any(fragment in name for fragment in forbidden)
    }
    assert json.loads(context.to_json())["allowed_data_sources"] == ["sales"]
    with pytest.raises(FrozenInstanceError):
        context.actor_id = "another-user"  # type: ignore[misc]


def test_nl2sql_context_and_state_do_not_import_persistence_types() -> None:
    package_root = (
        Path(__file__).resolve().parents[2]
        / "app"
        / "agent_system"
        / "agents"
        / "nl2sql"
    )
    sources = "\n".join(
        (package_root / name).read_text(encoding="utf-8")
        for name in ("context.py", "state.py")
    )

    assert "app.infrastructure" not in sources
    assert "ColumnInfoQdrant" not in sources
    assert "MetricInfoQdrant" not in sources
    assert "ValueInfoES" not in sources
