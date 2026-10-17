"""生产等价 NL2SQL EvaluationTarget 的离线合同测试。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from app.adapters.outbound.agent.nl2sql_capabilities import (
    Nl2SqlIndexVersionAdapter,
    Nl2SqlLLMAdapter,
)
from app.adapters.outbound.agent.nl2sql_runtime import Nl2SqlApplicationRuntime
from app.agent_system.agents.nl2sql.nodes.extract_keywords import extract_keywords
from app.agent_system.agents.nl2sql.settings import Nl2SqlSettings
from app.application.ports.agent_runtime import (
    AgentEventKind,
    AgentExecutionPlan,
    AgentExecutionRequest,
    AgentStreamEvent,
)
from app.domain.access.policy import PolicySnapshot
from evals.adapters.outbound import (
    Nl2SqlEvaluationTarget,
    build_nl2sql_evaluation_context,
    build_nl2sql_evaluation_state,
    create_nl2sql_evaluation_target,
)
from evals.contracts import EvaluationContract
from evals.datasets import (
    Answerability,
    DatasetBundle,
    DatasetCase,
    RelevanceJudgment,
    ReviewStatus,
)
from evals.harness import CaseRunStatus, EvaluationHarness
from evals.tracing import InMemoryTraceRecorder


class _FakeApplicationRuntime:
    def __init__(self) -> None:
        self.request: AgentExecutionRequest | None = None

    def prepare(
        self,
        *,
        performance_started_at: float,
        initial_state: dict[str, object],
    ) -> AgentExecutionPlan:
        return AgentExecutionPlan(
            initial_state=initial_state,
            deadline_evidence={"configured": True},
            timeout_seconds=1.0,
        )

    async def stream(
        self,
        request: AgentExecutionRequest,
    ) -> AsyncIterator[AgentStreamEvent]:
        self.request = request
        final_state = {
            "table_infos": [{"name": "fact_order"}],
            "recall_columns": [
                {
                    "id": "fact_order.order_amount",
                    "name": "order_amount",
                    "table_id": "fact_order",
                    "rerank_score": 0.9,
                }
            ],
            "recall_metrics": [{"id": "GMV", "name": "GMV", "score": 0.8}],
            "recall_values": [
                {
                    "column_id": "dim_region.region_name",
                    "value": "华东",
                    "score": 0.7,
                }
            ],
            "recall_training": [{"question": "按月统计销售额", "score": 0.6}],
            "sql": "SELECT 1",
            "result": [{"value": 1}],
            "analysis": "离线结果",
            "llm_attempts": [
                {
                    "attempt_id": "attempt-1",
                    "node_id": "generate_sql",
                    "route": "sql",
                    "retry_index": 0,
                    "token_known": True,
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "estimated_cost": 0.001,
                }
            ],
        }
        yield AgentStreamEvent(kind=AgentEventKind.STARTED, payload={})
        yield AgentStreamEvent(
            kind=AgentEventKind.MESSAGE,
            payload={
                "chunk": {
                    "trace_event": {
                        "event": "node_start",
                        "node_id": "extract_keywords",
                        "start_ms": 1,
                    }
                }
            },
        )
        yield AgentStreamEvent(
            kind=AgentEventKind.MESSAGE,
            payload={
                "chunk": {
                    "trace_event": {
                        "event": "node_end",
                        "node_id": "extract_keywords",
                        "start_ms": 1,
                        "end_ms": 4,
                        "duration_ms": 3,
                        "status": "success",
                        "error": None,
                    }
                }
            },
        )
        yield AgentStreamEvent(
            kind=AgentEventKind.MESSAGE,
            payload={"chunk": {"sql": "SELECT 1"}},
        )
        yield AgentStreamEvent(
            kind=AgentEventKind.MESSAGE,
            payload={"chunk": {"result": [{"value": 1}]}},
        )
        yield AgentStreamEvent(
            kind=AgentEventKind.SNAPSHOT,
            payload={"snapshot": {"status": "succeeded", "state": final_state}},
        )
        yield AgentStreamEvent(
            kind=AgentEventKind.COMPLETED,
            payload={"status": "succeeded"},
        )


class _FakeMetaRepository:
    workspace_id = "workspace-1"

    async def list_table_infos(self):
        return []

    async def get_all_metric_infos(self):
        return []


def _policy() -> PolicySnapshot:
    return PolicySnapshot(
        policies=(),
        user_id="evaluation-user",
        username="evaluation",
        workspace_id="workspace-1",
        role="owner",
    )


def _case() -> DatasetCase:
    return DatasetCase(
        case_id="case-1",
        dataset_id="dataset-1",
        dataset_version="v1",
        split="test",
        query="统计销售额",
        answerability=Answerability.ANSWERABLE,
        attributes={"evaluation_scope": "retrieval_only"},
    )


def test_composer_uses_production_runtime_and_keeps_graph_lazy() -> None:
    dependency = object()

    target = create_nl2sql_evaluation_target(
        workspace_id="workspace-1",
        access_policy=_policy(),
        dw_mysql_repo=dependency,
        meta_mysql_repo=dependency,
        value_es_repo=dependency,
        column_qdrant_repo=dependency,
        metric_qdrant_repo=dependency,
        training_qdrant_repo=dependency,
        embedding_client=dependency,
    )

    assert isinstance(target.runtime, Nl2SqlApplicationRuntime)
    assert target.runner is not None
    assert target.runner.factory.is_compiled is False
    assert target.runner.factory.compile_count == 0
    assert isinstance(target.fixed_capabilities["llm"], Nl2SqlLLMAdapter)
    assert isinstance(
        target.fixed_capabilities["index_versions"],
        Nl2SqlIndexVersionAdapter,
    )
    assert isinstance(target.fixed_capabilities["settings"], Nl2SqlSettings)


@pytest.mark.asyncio
async def test_target_maps_runtime_state_usage_and_typed_node_trace() -> None:
    runtime = _FakeApplicationRuntime()
    policy = _policy()
    target = Nl2SqlEvaluationTarget(
        runtime,
        workspace_id="workspace-1",
        access_policy=policy,
    )
    recorder = InMemoryTraceRecorder(
        evaluation_id="evaluation-1",
        case_id="case-1",
        trace_id="trace-1",
    )

    response = await target.evaluate(_case(), recorder)
    trace = recorder.snapshot()

    assert response.answer == "离线结果"
    assert response.prompt_tokens == 7
    assert response.completion_tokens == 3
    assert response.cost == pytest.approx(0.001)
    assert [item.document_id for item in response.retrieved] == [
        "table:fact_order",
        "column:fact_order.order_amount",
        "metric:GMV",
        "value:dim_region.region_name:华东",
        "training:按月统计销售额",
    ]
    assert response.metadata["result_row_count"] == 1
    assert runtime.request is not None
    assert runtime.request.capabilities["access_policy"] is policy
    assert len(trace.events) == 1
    assert trace.events[0].event_type == "agent.node.extract_keywords"
    assert trace.events[0].duration_ms == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_real_first_settings_consumer_uses_composed_context_offline() -> None:
    dependency = object()
    context = build_nl2sql_evaluation_context(
        dw_mysql_repo=dependency,
        meta_mysql_repo=_FakeMetaRepository(),
        value_es_repo=dependency,
        column_qdrant_repo=dependency,
        metric_qdrant_repo=dependency,
        training_qdrant_repo=dependency,
        embedding_client=dependency,
        access_policy=_policy(),
        workspace_id="workspace-1",
    )
    chunks: list[dict[str, object]] = []
    runtime = SimpleNamespace(context=context, stream_writer=chunks.append)
    state = build_nl2sql_evaluation_state("统计销售额")

    output = await extract_keywords(state, runtime)

    assert "query_signals" in output
    assert isinstance(context["settings"], Nl2SqlSettings)
    assert isinstance(context["llm"], Nl2SqlLLMAdapter)
    assert isinstance(context["index_versions"], Nl2SqlIndexVersionAdapter)
    assert chunks == [{"stage": "提取关键字"}]


@pytest.mark.asyncio
async def test_harness_runs_concrete_target_without_external_services() -> None:
    case = _case()
    dataset = DatasetBundle(
        cases=(case,),
        qrels=(
            RelevanceJudgment(
                case_id=case.case_id,
                document_id="table:fact_order",
                chunk_id=None,
                relevance_grade=3,
                review_status=ReviewStatus.ADJUDICATED,
            ),
        ),
        claims=(),
    )
    contract = EvaluationContract(
        evaluation_id="evaluation-1",
        dataset_id=case.dataset_id,
        dataset_version=case.dataset_version,
        system_version="test-system-v1",
    )
    target = Nl2SqlEvaluationTarget(
        _FakeApplicationRuntime(),
        workspace_id="workspace-1",
        access_policy=_policy(),
    )

    run = await EvaluationHarness(graders=()).run(
        contract=contract,
        dataset=dataset,
        target=target,
    )

    result = run.cases[0]
    assert result.status is CaseRunStatus.SUCCEEDED
    assert result.response is not None
    assert result.response.answer == "离线结果"
    assert [event.event_type for event in result.trace.events] == [
        "agent.node.extract_keywords",
        "target.evaluate",
    ]
