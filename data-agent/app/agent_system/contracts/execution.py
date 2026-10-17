"""跨 API、Runtime 与评测系统共享的版本化执行合同。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeAlias


QUERY_INPUT_VERSION = "query-input/v2"
OUTPUT_EVENT_VERSION = "output-event/v2"
ERROR_CONTRACT_VERSION = "execution-error/v2"
EXECUTION_SNAPSHOT_VERSION = "execution-snapshot/v2"

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _json_copy(value: Any, *, field_name: str) -> JsonValue:
    """验证并复制 JSON 值，避免合同持有调用方的可变对象。"""

    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是有效 JSON 值") from exc


def _require_text(value: str, *, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} 不能为空")
    return normalized


def _require_version(actual: object, expected: str, *, contract_name: str) -> None:
    if actual != expected:
        raise ValueError(
            f"不支持的 {contract_name} 版本：{actual!r}，当前仅支持 {expected!r}"
        )


class OutputEventType(StrEnum):
    """Runtime 对外输出的稳定事件类型。"""

    STARTED = "started"
    STATE = "state"
    MESSAGE = "message"
    RESULT = "result"
    ERROR = "error"
    SNAPSHOT = "snapshot"
    COMPLETED = "completed"


class ExecutionStatus(StrEnum):
    """一次 Agent 执行的终态或当前状态。"""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True, slots=True)
class QueryInput:
    """进入 Agent 系统的最小、可审计查询输入。"""

    request_id: str
    query: str
    actor_id: str
    workspace_id: str
    conversation_id: str | None = None
    trace_id: str | None = None
    analysis_enabled: bool = False
    attributes: dict[str, JsonValue] = field(default_factory=dict)
    version: str = QUERY_INPUT_VERSION

    def __post_init__(self) -> None:
        _require_version(self.version, QUERY_INPUT_VERSION, contract_name="QueryInput")
        for name in ("request_id", "query", "actor_id", "workspace_id"):
            object.__setattr__(
                self,
                name,
                _require_text(getattr(self, name), field_name=name),
            )
        for name in ("conversation_id", "trace_id"):
            value = getattr(self, name)
            object.__setattr__(self, name, str(value).strip() or None if value else None)
        copied = _json_copy(self.attributes, field_name="attributes")
        if not isinstance(copied, dict):
            raise ValueError("attributes 必须是 JSON 对象")
        object.__setattr__(self, "attributes", copied)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "version": self.version,
            "request_id": self.request_id,
            "query": self.query,
            "actor_id": self.actor_id,
            "workspace_id": self.workspace_id,
            "conversation_id": self.conversation_id,
            "trace_id": self.trace_id,
            "analysis_enabled": self.analysis_enabled,
            "attributes": _json_copy(self.attributes, field_name="attributes"),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> QueryInput:
        _require_version(value.get("version"), QUERY_INPUT_VERSION, contract_name="QueryInput")
        return cls(
            version=str(value["version"]),
            request_id=str(value.get("request_id") or ""),
            query=str(value.get("query") or ""),
            actor_id=str(value.get("actor_id") or ""),
            workspace_id=str(value.get("workspace_id") or ""),
            conversation_id=value.get("conversation_id"),
            trace_id=value.get("trace_id"),
            analysis_enabled=bool(value.get("analysis_enabled", False)),
            attributes=dict(value.get("attributes") or {}),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_json(cls, value: str) -> QueryInput:
        decoded = json.loads(value)
        if not isinstance(decoded, dict):
            raise ValueError("QueryInput JSON 必须是对象")
        return cls.from_dict(decoded)


@dataclass(frozen=True, slots=True)
class Error:
    """可安全跨进程传输的结构化执行错误。"""

    code: str
    message: str
    retryable: bool = False
    stage: str | None = None
    details: dict[str, JsonValue] = field(default_factory=dict)
    version: str = ERROR_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_version(self.version, ERROR_CONTRACT_VERSION, contract_name="Error")
        object.__setattr__(self, "code", _require_text(self.code, field_name="code"))
        object.__setattr__(self, "message", _require_text(self.message, field_name="message"))
        if self.stage is not None:
            object.__setattr__(self, "stage", str(self.stage).strip() or None)
        copied = _json_copy(self.details, field_name="details")
        if not isinstance(copied, dict):
            raise ValueError("details 必须是 JSON 对象")
        object.__setattr__(self, "details", copied)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "version": self.version,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "stage": self.stage,
            "details": _json_copy(self.details, field_name="details"),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Error:
        _require_version(value.get("version"), ERROR_CONTRACT_VERSION, contract_name="Error")
        return cls(
            version=str(value["version"]),
            code=str(value.get("code") or ""),
            message=str(value.get("message") or ""),
            retryable=bool(value.get("retryable", False)),
            stage=value.get("stage"),
            details=dict(value.get("details") or {}),
        )


# 迁移期允许调用方使用语义更明确的名称，序列化合同保持一致。
ExecutionError = Error


@dataclass(frozen=True, slots=True)
class OutputEvent:
    """流式执行的单个有序输出事件。"""

    request_id: str
    sequence: int
    type: OutputEventType
    emitted_at: str
    payload: dict[str, JsonValue] = field(default_factory=dict)
    error: Error | None = None
    version: str = OUTPUT_EVENT_VERSION

    def __post_init__(self) -> None:
        _require_version(self.version, OUTPUT_EVENT_VERSION, contract_name="OutputEvent")
        object.__setattr__(
            self,
            "request_id",
            _require_text(self.request_id, field_name="request_id"),
        )
        if self.sequence < 0:
            raise ValueError("sequence 不能小于 0")
        object.__setattr__(self, "type", OutputEventType(self.type))
        object.__setattr__(
            self,
            "emitted_at",
            _require_text(self.emitted_at, field_name="emitted_at"),
        )
        copied = _json_copy(self.payload, field_name="payload")
        if not isinstance(copied, dict):
            raise ValueError("payload 必须是 JSON 对象")
        object.__setattr__(self, "payload", copied)
        if self.type is OutputEventType.ERROR and self.error is None:
            raise ValueError("error 事件必须携带 Error 合同")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "version": self.version,
            "request_id": self.request_id,
            "sequence": self.sequence,
            "type": self.type.value,
            "emitted_at": self.emitted_at,
            "payload": _json_copy(self.payload, field_name="payload"),
            "error": self.error.to_dict() if self.error else None,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> OutputEvent:
        _require_version(value.get("version"), OUTPUT_EVENT_VERSION, contract_name="OutputEvent")
        raw_error = value.get("error")
        return cls(
            version=str(value["version"]),
            request_id=str(value.get("request_id") or ""),
            sequence=int(value.get("sequence", -1)),
            type=OutputEventType(str(value.get("type") or "")),
            emitted_at=str(value.get("emitted_at") or ""),
            payload=dict(value.get("payload") or {}),
            error=Error.from_dict(raw_error) if isinstance(raw_error, dict) else None,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_json(cls, value: str) -> OutputEvent:
        decoded = json.loads(value)
        if not isinstance(decoded, dict):
            raise ValueError("OutputEvent JSON 必须是对象")
        return cls.from_dict(decoded)


@dataclass(frozen=True, slots=True)
class ExecutionSnapshot:
    """一次执行在终止点的可持久化快照。"""

    request_id: str
    agent_id: str
    status: ExecutionStatus
    started_at: str
    finished_at: str | None
    last_sequence: int
    state: dict[str, JsonValue]
    error: Error | None = None
    metrics: dict[str, JsonValue] = field(default_factory=dict)
    version: str = EXECUTION_SNAPSHOT_VERSION

    def __post_init__(self) -> None:
        _require_version(
            self.version,
            EXECUTION_SNAPSHOT_VERSION,
            contract_name="ExecutionSnapshot",
        )
        object.__setattr__(
            self,
            "request_id",
            _require_text(self.request_id, field_name="request_id"),
        )
        object.__setattr__(self, "agent_id", _require_text(self.agent_id, field_name="agent_id"))
        object.__setattr__(self, "status", ExecutionStatus(self.status))
        object.__setattr__(
            self,
            "started_at",
            _require_text(self.started_at, field_name="started_at"),
        )
        if self.last_sequence < 0:
            raise ValueError("last_sequence 不能小于 0")
        for name in ("state", "metrics"):
            copied = _json_copy(getattr(self, name), field_name=name)
            if not isinstance(copied, dict):
                raise ValueError(f"{name} 必须是 JSON 对象")
            object.__setattr__(self, name, copied)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "version": self.version,
            "request_id": self.request_id,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "last_sequence": self.last_sequence,
            "state": _json_copy(self.state, field_name="state"),
            "error": self.error.to_dict() if self.error else None,
            "metrics": _json_copy(self.metrics, field_name="metrics"),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExecutionSnapshot:
        _require_version(
            value.get("version"),
            EXECUTION_SNAPSHOT_VERSION,
            contract_name="ExecutionSnapshot",
        )
        raw_error = value.get("error")
        return cls(
            version=str(value["version"]),
            request_id=str(value.get("request_id") or ""),
            agent_id=str(value.get("agent_id") or ""),
            status=ExecutionStatus(str(value.get("status") or "")),
            started_at=str(value.get("started_at") or ""),
            finished_at=value.get("finished_at"),
            last_sequence=int(value.get("last_sequence", -1)),
            state=dict(value.get("state") or {}),
            error=Error.from_dict(raw_error) if isinstance(raw_error, dict) else None,
            metrics=dict(value.get("metrics") or {}),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_json(cls, value: str) -> ExecutionSnapshot:
        decoded = json.loads(value)
        if not isinstance(decoded, dict):
            raise ValueError("ExecutionSnapshot JSON 必须是对象")
        return cls.from_dict(decoded)
