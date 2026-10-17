"""评测轨迹合同的稳定 JSON 序列化。"""

from __future__ import annotations

from typing import Any

from evals.contracts.models import ArtifactRef, EventRecord, TraceRecord


def _artifact_ref_to_dict(value: ArtifactRef) -> dict[str, Any]:
    return {
        "evaluation_id": value.evaluation_id,
        "relative_path": value.relative_path,
        "uri": value.uri,
        "sha256": value.sha256,
        "size_bytes": value.size_bytes,
        "media_type": value.media_type,
    }


def _event_to_dict(value: EventRecord) -> dict[str, Any]:
    return {
        "evaluation_id": value.evaluation_id,
        "case_id": value.case_id,
        "trace_id": value.trace_id,
        "event_id": value.event_id,
        "event_type": value.event_type,
        "status": value.status.value,
        "started_at": value.started_at.isoformat(),
        "ended_at": value.ended_at.isoformat() if value.ended_at else None,
        "parent_event_id": value.parent_event_id,
        "input_artifacts": [
            _artifact_ref_to_dict(item) for item in value.input_artifacts
        ],
        "output_artifacts": [
            _artifact_ref_to_dict(item) for item in value.output_artifacts
        ],
        "error_type": value.error_type,
        "error_message": value.error_message,
        "retry_count": value.retry_count,
        "attributes": dict(value.attributes),
    }


def trace_record_to_dict(value: TraceRecord) -> dict[str, Any]:
    """按 ``trajectory-record.schema.json`` 生成可持久字典。"""

    return {
        "evaluation_id": value.evaluation_id,
        "case_id": value.case_id,
        "trace_id": value.trace_id,
        "kind": value.kind.value,
        "events": [_event_to_dict(item) for item in value.events],
        "versions": dict(value.versions),
    }


__all__ = ["trace_record_to_dict"]
