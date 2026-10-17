from types import SimpleNamespace

import pytest
from langchain_core.runnables import RunnableLambda

from app.agent_system.runtime.llm_output import StructuredOutputError, parse_json_response
from app.adapters.outbound.agent.nl2sql_capabilities import (
    Nl2SqlLLMAdapter,
    build_nl2sql_settings,
)
from app.agent_system.agents.nl2sql.settings import override_settings
from app.agent_system.agents.nl2sql.nodes import filter_metric as filter_metric_module
from app.agent_system.agents.nl2sql.nodes import filter_table as filter_table_module


def test_parse_json_response_accepts_fence_and_short_prefix():
    assert parse_json_response("结果如下：```json\n{\"fact_order\":[\"order_id\"]}\n```", dict) == {
        "fact_order": ["order_id"]
    }
    assert parse_json_response("选择： [\"销售额\"]", list) == ["销售额"]


def test_parse_json_response_rejects_empty_and_wrong_shape_without_raw_echo():
    with pytest.raises(StructuredOutputError, match="empty_structured_output"):
        parse_json_response("", dict)
    with pytest.raises(StructuredOutputError, match="unexpected_structured_output_shape"):
        parse_json_response("[]", dict)


@pytest.mark.asyncio
async def test_filter_table_invalid_json_degrades_to_recalled_schema(monkeypatch):
    monkeypatch.setattr(filter_table_module, "llm", RunnableLambda(lambda _: "not-json"))
    settings = override_settings(
        build_nl2sql_settings(),
        "query",
        selection_llm_enabled=True,
    )
    tables = [{
        "name": "fact_order",
        "columns": [
            {"name": "order_id", "role": "primary_key"},
            {"name": "order_amount", "role": "measure", "alias": ["销售额"]},
        ],
    }]
    events = []
    output = await filter_table_module.filter_table(
        {"query": "销售额", "table_infos": tables, "chat_history": [], "semantic_ir": {}},
        SimpleNamespace(
            stream_writer=events.append,
            context={"llm": Nl2SqlLLMAdapter(), "settings": settings},
        ),
    )
    assert output["table_infos"][0]["columns"]
    assert output["degradations"][0]["node_id"] == "filter_table"
    assert any("degradation" in event for event in events)


@pytest.mark.asyncio
async def test_filter_metric_invalid_json_keeps_only_direct_registered_metric(monkeypatch):
    monkeypatch.setattr(filter_metric_module, "llm", RunnableLambda(lambda _: ""))
    settings = override_settings(
        build_nl2sql_settings(),
        "query",
        selection_llm_enabled=True,
    )
    metrics = [
        {"name": "销售总额", "alias": ["销售额"], "formula_rule": {"function": "SUM", "column": "fact_order.order_amount"}},
        {"name": "平均订单金额", "alias": ["客单价"], "formula_rule": {"function": "AVG", "column": "fact_order.order_amount"}},
    ]
    output = await filter_metric_module.filter_metric(
        {"query": "销售额是多少", "metric_infos": metrics, "semantic_ir": {}},
        SimpleNamespace(
            stream_writer=lambda event: None,
            context={"llm": Nl2SqlLLMAdapter(), "settings": settings},
        ),
    )
    assert [item["name"] for item in output["metric_infos"]] == ["销售总额"]
    assert output["degradations"][0]["node_id"] == "filter_metric"
