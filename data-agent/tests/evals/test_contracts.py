from datetime import UTC, datetime, timedelta

import pytest

from evals.contracts import (
    ContractError,
    EvaluationContract,
    EventRecord,
    EventStatus,
    FailureMode,
    TraceKind,
    TraceRecord,
)


def _event(**overrides: object) -> EventRecord:
    now = datetime.now(UTC)
    values: dict[str, object] = {
        "evaluation_id": "eval-1",
        "case_id": "case-1",
        "trace_id": "trace-1",
        "event_id": "event-1",
        "event_type": "retrieval.search",
        "status": EventStatus.SUCCEEDED,
        "started_at": now,
        "ended_at": now + timedelta(milliseconds=5),
    }
    values.update(overrides)
    return EventRecord(**values)  # type: ignore[arg-type]


def test_failure_mode_contract_is_exactly_rag_f01_to_f08() -> None:
    assert [mode.value for mode in FailureMode] == [
        "RAG-F01",
        "RAG-F02",
        "RAG-F03",
        "RAG-F04",
        "RAG-F05",
        "RAG-F06",
        "RAG-F07",
        "RAG-F08",
    ]


def test_contract_rejects_naive_timestamp() -> None:
    with pytest.raises(ContractError, match="时区"):
        EvaluationContract(
            evaluation_id="eval-1",
            dataset_id="dataset-1",
            dataset_version="v1",
            system_version="commit-1",
            created_at=datetime.now(),
        )


def test_trace_rejects_cross_case_event() -> None:
    with pytest.raises(ContractError, match="关联不一致"):
        TraceRecord(
            evaluation_id="eval-1",
            case_id="case-2",
            trace_id="trace-1",
            kind=TraceKind.SYSTEM,
            events=(_event(),),
        )


def test_trace_rejects_unknown_parent_event() -> None:
    with pytest.raises(ContractError, match="parent_event_id"):
        TraceRecord(
            evaluation_id="eval-1",
            case_id="case-1",
            trace_id="trace-1",
            kind=TraceKind.SYSTEM,
            events=(_event(parent_event_id="event-missing"),),
        )


def test_failed_event_requires_error_type() -> None:
    with pytest.raises(ContractError, match="error_type"):
        _event(status=EventStatus.FAILED)
