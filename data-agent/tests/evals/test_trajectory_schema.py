from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from evals.contracts import EventRecord, EventStatus, TraceKind, TraceRecord
from evals.contracts.serialization import trace_record_to_dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = PROJECT_ROOT / "evals" / "contracts" / "trajectory-record.schema.json"


def test_trajectory_schema_is_versioned_and_matches_trace_serializer() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    required = set(schema["required"])
    now = datetime.now(UTC)
    trace = TraceRecord(
        evaluation_id="eval-1",
        case_id="case-1",
        trace_id="trace-1",
        kind=TraceKind.SYSTEM,
        events=(
            EventRecord(
                evaluation_id="eval-1",
                case_id="case-1",
                trace_id="trace-1",
                event_id="event-1",
                event_type="target.evaluate",
                status=EventStatus.SUCCEEDED,
                started_at=now,
                ended_at=now,
                attributes={"tokens": 3},
            ),
        ),
        versions={"system": "test-v1"},
    )

    payload = trace_record_to_dict(trace)

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert required == set(payload)
    assert payload["events"][0]["status"] == "succeeded"
    assert payload["events"][0]["attributes"] == {"tokens": 3}
    json.dumps(payload)
