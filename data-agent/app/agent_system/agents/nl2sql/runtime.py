"""NL2SQL Graph 与通用受治理 Runtime 的装配适配器。"""

from __future__ import annotations

import json
from typing import cast

from app.agent_system.agents.nl2sql.graph import build_graph
from app.agent_system.agents.nl2sql.manifest import DEFAULT_NL2SQL_MANIFEST
from app.agent_system.contracts import Error, OutputEvent, OutputEventType
from app.agent_system.contracts.execution import JsonValue
from app.agent_system.runtime import GraphFactory, GraphRunner


def _json_chunk(raw: object) -> JsonValue:
    """把旧 stream_writer 输出收敛为稳定 JSON 值。"""

    encoded = json.dumps(raw, ensure_ascii=False, allow_nan=False, default=str)
    return cast(JsonValue, json.loads(encoded))


def map_nl2sql_stream_event(
    raw: object,
    request_id: str,
    sequence: int,
    emitted_at: str,
) -> OutputEvent:
    """将 NL2SQL custom stream chunk 包装为版本化 Runtime 事件。"""

    chunk = _json_chunk(raw)
    payload = {"chunk": chunk}
    if isinstance(chunk, dict) and chunk.get("error") is not None:
        return OutputEvent(
            request_id=request_id,
            sequence=sequence,
            type=OutputEventType.ERROR,
            emitted_at=emitted_at,
            payload=payload,
            error=Error(
                code="agent.reported_failure",
                message=str(chunk["error"]),
                stage="graph",
            ),
        )
    return OutputEvent(
        request_id=request_id,
        sequence=sequence,
        type=OutputEventType.MESSAGE,
        emitted_at=emitted_at,
        payload=payload,
    )


def create_nl2sql_graph_runner(
    *,
    factory: GraphFactory | None = None,
) -> GraphRunner:
    """创建进程作用域 Runner；Graph 在首次请求时才构建并只编译一次。"""

    graph_factory = factory or GraphFactory(
        builder=lambda _capabilities: build_graph(),
        # 迁移中的节点仍从每次 Invocation 的 DataAgentContext 获取能力；
        # 后续节点端口化后可在此注入 AgentCapabilities。
        capabilities=None,
    )
    return GraphRunner(
        factory=graph_factory,
        agent_id=DEFAULT_NL2SQL_MANIFEST.agent_id,
        event_mapper=map_nl2sql_stream_event,
        stream_mode="custom",
    )


__all__ = ["create_nl2sql_graph_runner", "map_nl2sql_stream_event"]
