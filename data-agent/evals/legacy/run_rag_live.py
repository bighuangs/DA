"""Execute the exact RAG contract questions against live retrieval dependencies."""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import time
import uuid
from builtins import BaseExceptionGroup, ExceptionGroup
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from pathlib import Path

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from app.agent_system.agents.nl2sql.context import DataAgentContext
from app.agent_system.agents.nl2sql.state import DataAgentState
from app.agent_system.agents.nl2sql.nodes.build_semantic_ir import build_semantic_ir_node
from app.agent_system.agents.nl2sql.nodes.extract_keywords import extract_keywords
from app.agent_system.agents.nl2sql.nodes.filter_metric import filter_metric
from app.agent_system.agents.nl2sql.nodes.filter_table import filter_table
from app.agent_system.agents.nl2sql.nodes.merge_retrieved_info import merge_retrieved_info
from app.agent_system.agents.nl2sql.nodes.recall_column import recall_column
from app.agent_system.agents.nl2sql.nodes.recall_metric import recall_metric
from app.agent_system.agents.nl2sql.nodes.recall_training import recall_training
from app.agent_system.agents.nl2sql.nodes.recall_value import recall_value
from app.agent_system.agents.nl2sql.nodes.rewrite_follow_up import rewrite_follow_up
from app.infrastructure.clients.embedding_client_manager import embedding_client_manager
from app.infrastructure.clients.es_client_manager import es_client_manager
from app.infrastructure.clients.mysql_client_manager import dw_mysql_client_manager, meta_mysql_client_manager
from app.infrastructure.clients.qdrant_client_manager import qdrant_client_manager
from app.domain.access.policy import PolicySnapshot
from app.agent_system.contracts import (
    ExecutionSnapshot,
    InvocationContext,
    OutputEventType,
    QueryInput,
)
from app.agent_system.runtime import GraphFactory, GraphRunner
from app.agent_system.runtime.performance import instrument_node
from app.infrastructure.search.index_version import retrieval_index_version
from evals.legacy.evaluator import _stage_evidence
from evals.legacy.rag_recall import evaluate_live_run, load_contract, validate_contract
from evals.legacy.run_eval import resolve_workspace_id
from evals.adapters.outbound.nl2sql import (
    build_nl2sql_evaluation_context,
    build_nl2sql_evaluation_state,
)
from app.infrastructure.search.elasticsearch.repositories.value_es_repository import ValueESRepository
from app.infrastructure.persistence.mysql.repositories.dw_mysql_repository import DWMysqlRepository
from app.infrastructure.persistence.mysql.repositories.meta_mysql_repository import MetaMysqlRepository
from app.infrastructure.search.qdrant.repositories.column_qdrant_repository import ColumnQdrantRepository
from app.infrastructure.search.qdrant.repositories.metric_qdrant_repository import MetricQdrantRepository
from app.infrastructure.search.qdrant.repositories.training_qdrant_repository import TrainingQdrantRepository
from evals.paths import RAG_RUNS_ROOT


RUNS_DIR = RAG_RUNS_ROOT


def _live_dependencies() -> tuple[tuple[str, object], ...]:
    """返回独立评测进程拥有的资源；解析发生在显式运行阶段。"""

    return (
        ("embedding", embedding_client_manager),
        ("elasticsearch", es_client_manager),
        ("dw_mysql", dw_mysql_client_manager),
        ("meta_mysql", meta_mysql_client_manager),
        ("qdrant", qdrant_client_manager),
    )


async def _close_live_dependencies() -> list[tuple[str, BaseException]]:
    """并行关闭全部评测资源，并返回具名错误而不是提前中断。"""

    dependencies = _live_dependencies()
    results = await asyncio.gather(
        *(manager.close() for _, manager in dependencies),
        return_exceptions=True,
    )
    return [
        (name, result)
        for (name, _), result in zip(dependencies, results, strict=True)
        if isinstance(result, BaseException)
    ]


def _raise_close_failures(
    failures: list[tuple[str, BaseException]],
) -> None:
    errors = [error for _, error in failures]
    if all(isinstance(error, Exception) for error in errors):
        raise ExceptionGroup(
            "RAG live 资源关闭失败",
            [error for error in errors if isinstance(error, Exception)],
        )
    raise BaseExceptionGroup("RAG live 资源关闭失败", errors)


@asynccontextmanager
async def _live_dependency_scope() -> AsyncIterator[None]:
    """初始化并可靠关闭 live 依赖，cleanup 不覆盖运行主异常。"""

    primary_error: BaseException | None = None
    try:
        for _, manager in _live_dependencies():
            manager.init()
        yield
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        failures = await _close_live_dependencies()
        if failures and primary_error is not None:
            for name, error in failures:
                primary_error.add_note(
                    "RAG live 资源关闭失败 "
                    f"resource={name} error_type={type(error).__name__}"
                )
        elif failures:
            _raise_close_failures(failures)


def build_parser() -> argparse.ArgumentParser:
    """创建 CLI Parser；解析参数不会初始化任何运行时资源。"""

    parser = argparse.ArgumentParser(description="逐条执行真实 RAG 契约问题")
    parser.add_argument("--workspace-id", default=None)
    parser.add_argument(
        "--only",
        "--case-id",
        dest="only",
        default=None,
        help="仅执行指定 case_id；多个 ID 用逗号分隔",
    )
    parser.add_argument("--enforce", action="store_true")
    return parser


def select_contract_cases(
    cases: list[dict],
    only: str | None = None,
) -> list[dict]:
    """按原合同顺序选择显式 Case；未知 ID 必须失败关闭。"""

    if not str(only or "").strip():
        return [dict(case) for case in cases]
    requested = {
        value.strip()
        for value in str(only).split(",")
        if value.strip()
    }
    if not requested:
        raise ValueError("--only/--case-id 至少需要一个非空 case_id")
    available = {str(case.get("id") or "") for case in cases}
    missing = sorted(requested - available)
    if missing:
        raise ValueError("未知 RAG case_id：" + ",".join(missing))
    return [dict(case) for case in cases if str(case.get("id") or "") in requested]


def _build_retrieval_graph():
    builder = StateGraph(state_schema=DataAgentState, context_schema=DataAgentContext)
    nodes = {
        "rewrite_follow_up": rewrite_follow_up, "extract_keywords": extract_keywords,
        "recall_column": recall_column, "recall_metric": recall_metric,
        "recall_value": recall_value, "recall_training": recall_training,
        "merge_retrieved_info": merge_retrieved_info,
        "build_semantic_ir": build_semantic_ir_node,
        "filter_metric": filter_metric, "filter_table": filter_table,
    }
    for name, node in nodes.items():
        builder.add_node(name, instrument_node(name, node))
    builder.add_edge(START, "rewrite_follow_up")
    builder.add_conditional_edges(
        "rewrite_follow_up", lambda state: END if state.get("error") else "extract_keywords",
        {"extract_keywords": "extract_keywords", END: END},
    )
    for route in ("recall_column", "recall_metric", "recall_value", "recall_training"):
        builder.add_edge("extract_keywords", route)
        builder.add_edge(route, "merge_retrieved_info")
    builder.add_edge("merge_retrieved_info", "build_semantic_ir")
    builder.add_edge("build_semantic_ir", "filter_metric")
    builder.add_edge("build_semantic_ir", "filter_table")
    builder.add_edge("filter_metric", END)
    builder.add_edge("filter_table", END)
    return builder


def create_retrieval_graph_runner(
    *, factory: GraphFactory | None = None,
) -> GraphRunner:
    """创建显式惰性检索 Runner；构造和模块导入阶段均不编译 Graph。"""

    resolved_factory = factory or GraphFactory(
        builder=lambda _capabilities: _build_retrieval_graph(),
    )
    return GraphRunner(
        factory=resolved_factory,
        agent_id="data-agent.nl2sql.retrieval-evaluation",
        stream_mode="values",
    )


async def _run_case(
    case: dict,
    context: DataAgentContext,
    runner: GraphRunner,
) -> dict:
    started = time.perf_counter()
    final_state = None
    sql_attempts: list[str] = []
    error = None
    try:
        state = build_nl2sql_evaluation_state(
            str(case["query"]),
            performance_started_at=started,
            analysis_enabled=False,
        )
        request_id = f"rag-{case['id']}-{uuid.uuid4().hex[:12]}"
        query_input = QueryInput(
            request_id=request_id,
            query=str(case["query"]),
            actor_id="rag-eval-user",
            workspace_id=str(context.get("workspace_id") or ""),
            analysis_enabled=False,
        )
        invocation_context = InvocationContext(
            request_id=request_id,
            actor_id=query_input.actor_id,
            workspace_id=query_input.workspace_id,
            roles=("owner",),
            analysis_enabled=False,
        )
        async for event in runner.stream(
            query_input,
            invocation_context,
            initial_state=state,
            graph_context=context,
        ):
            if event.type is OutputEventType.STATE:
                snapshot = event.payload.get("state")
                if isinstance(snapshot, dict):
                    final_state = snapshot
            elif event.type is OutputEventType.SNAPSHOT:
                raw_snapshot = event.payload.get("snapshot")
                if isinstance(raw_snapshot, dict):
                    final_state = ExecutionSnapshot.from_dict(raw_snapshot).state
            elif event.type is OutputEventType.ERROR and event.error is not None:
                error = event.error.message
            sql = str((final_state or {}).get("sql") or "").strip()
            if sql and (not sql_attempts or sql != sql_attempts[-1]):
                sql_attempts.append(sql)
    except Exception as exc:  # keep the remaining contract cases observable
        error = f"{type(exc).__name__}: {exc}"
    return {
        "id": str(case["id"]), "rag_case_id": str(case["id"]),
        "question": str(case["query"]),
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        "error": error or (final_state or {}).get("error"),
        "stage_evidence": _stage_evidence(final_state, sql_attempts),
    }


async def main(
    workspace_id: str | None = None,
    only: str | None = None,
) -> tuple[Path, dict]:
    validation = validate_contract()
    if not validation["valid"]:
        raise RuntimeError("RAG 契约无效：" + "；".join(validation["errors"]))
    selected_cases = select_contract_cases(
        list(load_contract().get("cases") or []),
        only,
    )
    async with _live_dependency_scope():
        # 生产 lifespan 会在接收请求前预热 Embedding；独立评测入口必须保持
        # 同一生命周期语义，避免把冷启动模型加载误计入单 Case deadline。
        warmup = await embedding_client_manager.client.warmup()
        warmup_evidence = {
            "loaded": bool(warmup.get("loaded")),
            "dense_size": int(warmup.get("dense_size") or 0),
            "sparse_nonzero": int(warmup.get("sparse_nonzero") or 0),
            "runtime_profile": str(
                warmup.get("runtime_profile") or "unknown"
            ),
        }
        runner = create_retrieval_graph_runner()
        async with (dw_mysql_client_manager.session_factory() as dw_session,
                    meta_mysql_client_manager.session_factory() as meta_session):
            selected_workspace = await resolve_workspace_id(meta_session, workspace_id)
            context = build_nl2sql_evaluation_context(
                dw_mysql_repo=DWMysqlRepository(dw_session),
                meta_mysql_repo=MetaMysqlRepository(meta_session, selected_workspace),
                value_es_repo=ValueESRepository(es_client_manager.client),
                column_qdrant_repo=ColumnQdrantRepository(qdrant_client_manager.client),
                metric_qdrant_repo=MetricQdrantRepository(qdrant_client_manager.client),
                training_qdrant_repo=TrainingQdrantRepository(qdrant_client_manager.client),
                embedding_client=embedding_client_manager.client,
                access_policy=PolicySnapshot(
                    policies=tuple(), user_id="rag-eval-user", username="rag-eval",
                    workspace_id=selected_workspace, role="owner",
                ),
                workspace_id=selected_workspace,
            )
            rows = []
            for case in selected_cases:
                # Query text may contain entity literals; progress output keeps
                # only the reviewed case identifier.
                print(f"RAG live {case['id']}", flush=True)
                rows.append(await _run_case(dict(case), context, runner))
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            payload = {
                "artifact_schema": "rag-live-v1", "run_id": timestamp,
                "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "workspace_id": selected_workspace,
                "contract_fingerprint": validation["fingerprint"],
                "retrieval_index_version": retrieval_index_version(),
                "embedding_warmup": warmup_evidence,
                "targeted": bool(str(only or "").strip()),
                "requested_case_ids": [str(case["id"]) for case in selected_cases],
                "valid": all(not row.get("error") for row in rows),
                "n": len(rows), "cases": rows,
            }
            report = evaluate_live_run(payload)
            RUNS_DIR.mkdir(parents=True, exist_ok=True)
            path = RUNS_DIR / f"rag_run_{timestamp}.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return path, report


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    artifact, report = asyncio.run(main(args.workspace_id, args.only))
    summary = report.get("summary") or {}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"RAG live 制品：{artifact}")
    raise SystemExit(1 if args.enforce and not summary.get("gate_passed") else 0)
