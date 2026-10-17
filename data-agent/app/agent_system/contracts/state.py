"""不依赖持久化 DTO 的 NL2SQL 可序列化状态合同。"""

from __future__ import annotations

import json
from typing import Any, Literal, NotRequired, Required, TypedDict, cast

from app.agent_system.contracts.execution import QueryInput


STATE_V2_VERSION = "nl2sql-state/v2"

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
StateStatus = Literal["pending", "running", "succeeded", "failed", "cancelled", "timed_out"]


class StateV2(TypedDict, total=False):
    """Graph 节点共享的纯 JSON 状态，不包含 Client、Session 或 ORM 对象。"""

    schema_version: Required[str]
    request_id: Required[str]
    original_query: Required[str]
    query: Required[str]
    status: Required[StateStatus]
    phase: Required[str]
    sql: NotRequired[str]
    authorized_sql: NotRequired[str]
    result_rows: NotRequired[list[dict[str, JsonValue]]]
    chart: NotRequired[dict[str, JsonValue]]
    analysis: NotRequired[str]
    recommendations: NotRequired[list[str | dict[str, JsonValue]]]
    result_references: NotRequired[dict[str, JsonValue]]
    query_intent: NotRequired[dict[str, JsonValue]]
    intent_response: NotRequired[dict[str, JsonValue]]
    retrieval_evidence: NotRequired[list[dict[str, JsonValue]]]
    planning_evidence: NotRequired[dict[str, JsonValue]]
    validation_evidence: NotRequired[dict[str, JsonValue]]
    node_attempts: NotRequired[list[dict[str, JsonValue]]]
    degradations: NotRequired[list[dict[str, JsonValue]]]
    error: NotRequired[dict[str, JsonValue] | None]


def _json_object(value: object, *, field_name: str) -> dict[str, Any]:
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须只包含 JSON 值") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{field_name} 必须是 JSON 对象")
    return decoded


def validate_state(value: object) -> StateV2:
    """校验版本和最小不变量，并返回与调用方隔离的状态副本。"""

    decoded = _json_object(value, field_name="StateV2")
    if decoded.get("schema_version") != STATE_V2_VERSION:
        raise ValueError(f"不支持的 StateV2 版本：{decoded.get('schema_version')!r}")
    for name in ("request_id", "original_query", "query", "status", "phase"):
        if not str(decoded.get(name) or "").strip():
            raise ValueError(f"StateV2.{name} 不能为空")
    if decoded["status"] not in {
        "pending",
        "running",
        "succeeded",
        "failed",
        "cancelled",
        "timed_out",
    }:
        raise ValueError(f"未知的 StateV2.status：{decoded['status']!r}")
    return cast(StateV2, decoded)


def new_state(query_input: QueryInput) -> StateV2:
    """从版本化输入创建不含基础设施对象的初始状态。"""

    return StateV2(
        schema_version=STATE_V2_VERSION,
        request_id=query_input.request_id,
        original_query=query_input.query,
        query=query_input.query,
        status="pending",
        phase="initialized",
        result_rows=[],
        retrieval_evidence=[],
        node_attempts=[],
        degradations=[],
        error=None,
    )


def serialize_state(state: StateV2) -> str:
    return json.dumps(validate_state(state), ensure_ascii=False, allow_nan=False)


def deserialize_state(value: str) -> StateV2:
    decoded = json.loads(value)
    return validate_state(decoded)
