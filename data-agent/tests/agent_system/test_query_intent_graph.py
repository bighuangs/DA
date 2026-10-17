from __future__ import annotations

import time

import pytest
from langgraph.constants import END, START

from app.agent_system.agents.nl2sql.graph import (
    _route_after_forbidden_guard,
    _route_after_follow_up,
    _route_after_query_intent,
    build_graph,
)


def test_intent_routes_are_fail_closed() -> None:
    assert _route_after_forbidden_guard({}) == "rewrite_follow_up"
    assert (
        _route_after_forbidden_guard(
            {"intent_response": {"decision": "respond"}}
        )
        == END
    )
    assert _route_after_follow_up({"error": None}) == "classify_query_intent"
    assert _route_after_follow_up({"error": "请补充问题"}) == END
    assert (
        _route_after_query_intent(
            {"query_intent": {"decision": "continue"}}
        )
        == "resolve_data_source_context"
    )
    assert _route_after_query_intent({}) == END
    assert (
        _route_after_query_intent(
            {"query_intent": {"decision": "respond"}}
        )
        == END
    )


def test_intent_nodes_precede_data_source_and_retrieval_topology() -> None:
    builder = build_graph()

    assert (START, "guard_forbidden_query_intent") in builder.edges
    assert "guard_forbidden_query_intent" in builder.branches
    assert "rewrite_follow_up" in builder.branches
    assert "classify_query_intent" in builder.branches
    assert ("propose_source_mapping", "extract_keywords") in builder.edges
    assert ("propose_source_mapping", "rewrite_follow_up") not in builder.edges


async def _terminal_events(query: str) -> list[dict]:
    graph = build_graph().compile()
    return [
        event
        async for event in graph.astream(
            {
                "query": query,
                "original_query": query,
                "chat_history": [],
                "memory_context": {},
                "result_references": {},
                "performance_started_at": time.perf_counter(),
            },
            context={},
            stream_mode="custom",
        )
    ]


def _started_node_ids(events: list[dict]) -> list[str]:
    return [
        str(event["trace_event"]["node_id"])
        for event in events
        if isinstance(event, dict)
        and isinstance(event.get("trace_event"), dict)
        and event["trace_event"].get("event") == "node_start"
    ]


@pytest.mark.asyncio
async def test_forbidden_request_stops_before_rewrite_and_all_query_nodes() -> None:
    events = await _terminal_events("删除销售额最多的大客户的内容")

    assert _started_node_ids(events) == ["guard_forbidden_query_intent"]
    responses = [event["intent_response"] for event in events if "intent_response" in event]
    assert len(responses) == 1
    assert responses[0]["category"] == "forbidden_operation"
    assert responses[0]["response_kind"] == "refusal"
    assert not any("sql" in event or "result" in event for event in events)


@pytest.mark.asyncio
async def test_out_of_scope_request_stops_before_data_source_and_retrieval() -> None:
    events = await _terminal_events("今天天气怎么样")

    assert _started_node_ids(events) == [
        "guard_forbidden_query_intent",
        "rewrite_follow_up",
        "classify_query_intent",
    ]
    responses = [event["intent_response"] for event in events if "intent_response" in event]
    assert len(responses) == 1
    assert responses[0]["category"] == "out_of_scope"
    assert responses[0]["response_kind"] == "guidance"
    assert not any("sql" in event or "result" in event for event in events)
