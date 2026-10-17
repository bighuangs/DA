"""QueryService 对预期意图响应的应用层终止合同。"""

from __future__ import annotations

import json

import pytest

from app.application.ports.agent_runtime import (
    AgentEventKind,
    AgentExecutionPlan,
    AgentStreamEvent,
)
from app.application.ports.llm_routing import MissingLLMRouteCatalog
from app.application.ports.query_settings import QueryServiceSettings
from app.application.ports.query_tracing import QueryTraceLLMSettings
from app.application.services.data_source_context_service import (
    ResolvedDataSourceContext,
)
from app.application.services.query_service import QueryService
from app.domain.data_sources.context import AssetAccessContext, SourceMode


INTENT_RESPONSE = {
    "schema_version": "intent-response-v1",
    "contract_version": "query-intent-v1",
    "category": "forbidden_operation",
    "decision": "respond",
    "response_kind": "refusal",
    "reason_code": "forbidden_operation",
    "message": "当前助手仅支持只读业务数据查询，不能新增、删除或修改业务数据。",
    "suggestions": ["查询销售额最高的10位客户及其销售额"],
    "source": "deterministic",
    "reason_codes": ["intent.forbidden.original"],
    "rule_ids": ["deny.business_data_mutation.v1"],
    "query_hashes": {"original": "sha256:test", "standalone": "sha256:test"},
    "model_used": False,
    "model_degraded": False,
    "raw_text_logged": False,
}


class _DataWarehouse:
    last_cache_hit = False


class _Persistence:
    def __init__(self) -> None:
        self.started: list[dict] = []
        self.handled: list[dict] = []
        self.successful: list[dict] = []
        self.failed: list[dict] = []
        self.trace_statuses: list[str] = []
        self.trace_payloads: list[dict] = []

    async def recent_history(self, chat_id: str, limit: int):
        del chat_id, limit
        return []

    async def start_turn(self, **values) -> None:
        self.started.append(dict(values))

    async def complete_handled_turn(self, **values) -> None:
        self.handled.append(dict(values))

    async def save_successful_turn(self, **values) -> None:
        self.successful.append(dict(values))

    async def fail_turn(self, **values) -> None:
        self.failed.append(dict(values))

    async def persist_trace(self, collector, **values):
        status = str(values["status"])
        self.trace_statuses.append(status)
        self.trace_payloads.append(dict(values))
        return collector.finish(status)


class _SourceContext:
    def __init__(self) -> None:
        access = AssetAccessContext(
            workspace_id="workspace-1",
            owner_id="user-1",
            source_mode=SourceMode.DW_ONLY,
            official_tables=frozenset(),
        )
        self.resolved = ResolvedDataSourceContext(
            access=access,
            profiles=(),
            table_infos=(),
            mapping={},
        )

    async def resolve(self, **_):
        return self.resolved


class _ConversationContext:
    def __init__(self, resolved: ResolvedDataSourceContext) -> None:
        self.resolved = resolved
        self.snapshots: list[dict] = []

    async def resolve_default(self, chat_id: str):
        del chat_id
        return self.resolved

    async def snapshot_turn(self, **values) -> None:
        self.snapshots.append(dict(values))


class _Runtime:
    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self.initial_state: dict | None = None

    def prepare(self, *, performance_started_at: float, initial_state):
        del performance_started_at
        self.initial_state = dict(initial_state)
        return AgentExecutionPlan(
            initial_state=initial_state,
            deadline_evidence={"schema_version": "deadline-v1"},
            timeout_seconds=1.0,
        )

    async def stream(self, request):
        del request
        for chunk in self.chunks:
            yield AgentStreamEvent(
                kind=AgentEventKind.RESULT,
                payload={"chunk": chunk},
            )


class _MemoryService:
    def __init__(self) -> None:
        self.completed: list[dict] = []

    async def start_turn(self, **_) -> int:
        return 17

    async def load_context(self, **_):
        return {
            "schema_version": "memory-context-v2",
            "confirmed_preferences": [],
            "relevant_episodes": [],
            "confirmed_profile_facts": [],
            "long_term_goals": [],
            "selected_summaries": [],
            "evidence": {"degradations": []},
        }

    async def complete_handled_turn(self, user_message_id: int | None, **values):
        self.completed.append(
            {"user_message_id": user_message_id, **dict(values)}
        )


def _service(
    *,
    persistence: _Persistence,
    memory_service: _MemoryService | None = None,
    chunks: list[dict] | None = None,
) -> QueryService:
    source = _SourceContext()
    runtime = _Runtime(
        chunks
        or [
            {"stage": "识别问题意图"},
            {"intent_response": INTENT_RESPONSE},
            # Runtime 即使违反终止合同，也不能越过 Application 边界。
            {"sql": "SELECT * FROM dim_customer"},
            {"result": [{"customer": "不应出现"}]},
            {"error": "终态后的错误不应覆盖已处理响应"},
        ]
    )
    return QueryService(
        dw_mysql_repo=_DataWarehouse(),
        query_persistence=persistence,
        data_source_context_service=source,
        conversation_context_service=_ConversationContext(source.resolved),
        graph_runner=runtime,
        settings=QueryServiceSettings(
            history_limit=3,
            analysis_enabled_default=False,
            memory_enabled=memory_service is not None,
        ),
        llm_route_catalog=MissingLLMRouteCatalog(),
        trace_llm_settings=QueryTraceLLMSettings(),
        owner_id="user-1",
        workspace_id="workspace-1",
        memory_service=memory_service,
    )


async def _search_events(service: QueryService) -> list[dict]:
    chunks = [
        chunk async for chunk in service.search("删除销售额最高的大客户", "chat-1")
    ]
    return [json.loads(chunk.removeprefix("data:").strip()) for chunk in chunks]


@pytest.mark.asyncio
async def test_handled_intent_is_not_failed_and_suppresses_query_artifacts():
    persistence = _Persistence()
    service = _service(persistence=persistence)

    events = await _search_events(service)

    assert {"intent_response": INTENT_RESPONSE} in events
    assert not any("sql" in event for event in events)
    assert not any("result" in event for event in events)
    assert not any("error" in event for event in events)
    assert events[-2]["trace"]["status"] == "handled"
    assert events[-1] == {"done": True}
    assert persistence.handled == [
        {
            "chat_id": "chat-1",
            "turn_id": persistence.started[0]["turn_id"],
            "response": INTENT_RESPONSE,
        }
    ]
    assert persistence.failed == []
    assert persistence.successful == []
    assert persistence.trace_statuses == ["handled"]
    assert service.agent_runtime.initial_state["original_query"] == (
        "删除销售额最高的大客户"
    )


@pytest.mark.asyncio
async def test_memory_enabled_handled_intent_completes_message_only_turn():
    persistence = _Persistence()
    memory = _MemoryService()

    events = await _search_events(
        _service(persistence=persistence, memory_service=memory)
    )

    assert events[-2]["trace"]["status"] == "handled"
    assert memory.completed == [
        {
            "user_message_id": 17,
            "chat_id": "chat-1",
            "trace_id": events[0]["trace_id"],
            "response": INTENT_RESPONSE,
        }
    ]
    assert persistence.started == []
    assert persistence.handled == []
    assert persistence.failed == []


@pytest.mark.asyncio
async def test_handled_intent_discards_query_artifacts_emitted_before_terminal():
    persistence = _Persistence()
    service = _service(
        persistence=persistence,
        chunks=[
            {"sql": "SELECT * FROM dim_customer"},
            {"result": [{"customer": "不应持久化"}]},
            {"intent_response": INTENT_RESPONSE},
        ],
    )

    events = await _search_events(service)

    assert {"intent_response": INTENT_RESPONSE} in events
    assert persistence.successful == []
    assert persistence.handled[0]["response"] == INTENT_RESPONSE
    assert events[-2]["trace"]["status"] == "handled"
    assert events[-2]["trace"].get("sql") is None
    assert events[-2]["trace"].get("row_count", 0) == 0
    assert persistence.trace_payloads[-1]["sql"] is None
    assert persistence.trace_payloads[-1]["row_count"] == 0


@pytest.mark.asyncio
async def test_error_before_intent_response_keeps_the_first_failed_terminal():
    persistence = _Persistence()
    service = _service(
        persistence=persistence,
        chunks=[
            {"error": "Runtime 已公开失败"},
            {"intent_response": INTENT_RESPONSE},
        ],
    )

    events = await _search_events(service)

    assert {"error": "Runtime 已公开失败"} in events
    assert not any("intent_response" in event for event in events)
    assert events[-2]["trace"]["status"] == "failed"
    assert events[-1] == {"done": True}
    assert persistence.handled == []
    assert persistence.failed[0]["status"] == "failed"
    assert persistence.trace_payloads[-1]["status"] == "failed"
