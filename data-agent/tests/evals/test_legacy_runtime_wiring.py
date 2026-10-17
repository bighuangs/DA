"""Legacy live 入口的惰性 Graph 与 capability 接线门禁。"""

from __future__ import annotations

import asyncio
import importlib
import sys
from types import SimpleNamespace

import pytest
from langgraph.graph import StateGraph


class _FakeManager:
    def __init__(
        self,
        name: str,
        *,
        warmup_error: BaseException | None = None,
        close_error: BaseException | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.name = name
        self.warmup_error = warmup_error
        self.close_error = close_error
        self.events = events if events is not None else []
        self.client = SimpleNamespace(warmup=self.warmup)

    def init(self) -> None:
        self.events.append(f"{self.name}:init")

    async def warmup(self) -> dict[str, object]:
        self.events.append(f"{self.name}:warmup")
        if self.warmup_error is not None:
            raise self.warmup_error
        return {
            "loaded": True,
            "dense_size": 1024,
            "sparse_nonzero": 3,
            "runtime_profile": "test",
        }

    async def close(self) -> None:
        self.events.append(f"{self.name}:close")
        if self.close_error is not None:
            raise self.close_error


def _install_fake_managers(monkeypatch, module, managers) -> None:
    for attribute, manager in zip(
        (
            "embedding_client_manager",
            "es_client_manager",
            "dw_mysql_client_manager",
            "meta_mysql_client_manager",
            "qdrant_client_manager",
        ),
        managers,
        strict=True,
    ):
        monkeypatch.setattr(module, attribute, manager)


def _stub_contract(monkeypatch, module) -> None:
    monkeypatch.setattr(
        module,
        "validate_contract",
        lambda: {"valid": True, "errors": [], "fingerprint": "contract-v1"},
    )
    monkeypatch.setattr(
        module,
        "load_contract",
        lambda: {"cases": [{"id": "RAG-001", "query": "test"}]},
    )


def test_run_rag_live_import_does_not_compile_graph(monkeypatch) -> None:
    compile_calls = 0
    original_compile = StateGraph.compile

    def record_compile(self, *args, **kwargs):
        nonlocal compile_calls
        compile_calls += 1
        return original_compile(self, *args, **kwargs)

    monkeypatch.setattr(StateGraph, "compile", record_compile)
    sys.modules.pop("evals.legacy.run_rag_live", None)

    module = importlib.import_module("evals.legacy.run_rag_live")
    runner = module.create_retrieval_graph_runner()

    assert compile_calls == 0
    assert runner.factory.is_compiled is False
    assert runner.factory.compile_count == 0


def test_run_rag_live_cli_selects_one_case_without_initializing_runtime() -> None:
    module = importlib.import_module("evals.legacy.run_rag_live")
    parser = module.build_parser()

    args = parser.parse_args(["--case-id", "RAG-002", "--enforce"])
    selected = module.select_contract_cases(
        [{"id": "RAG-001"}, {"id": "RAG-002"}],
        args.only,
    )

    assert args.only == "RAG-002"
    assert args.enforce is True
    assert selected == [{"id": "RAG-002"}]


def test_run_rag_live_cli_keeps_full_default_and_rejects_unknown_case() -> None:
    module = importlib.import_module("evals.legacy.run_rag_live")
    cases = [{"id": "RAG-001"}, {"id": "RAG-002"}]

    assert module.select_contract_cases(cases) == cases
    with pytest.raises(ValueError, match="RAG-404"):
        module.select_contract_cases(cases, "RAG-404")
    with pytest.raises(ValueError, match="至少需要一个"):
        module.select_contract_cases(cases, ",,")


@pytest.mark.asyncio
async def test_warmup_failure_closes_all_resources_and_keeps_primary_error(
    monkeypatch,
) -> None:
    module = importlib.import_module("evals.legacy.run_rag_live")
    events: list[str] = []
    primary = RuntimeError("warmup failed")
    managers = (
        _FakeManager("embedding", warmup_error=primary, events=events),
        _FakeManager(
            "elasticsearch",
            close_error=RuntimeError("es close failed"),
            events=events,
        ),
        _FakeManager("dw_mysql", events=events),
        _FakeManager(
            "meta_mysql",
            close_error=RuntimeError("meta close failed"),
            events=events,
        ),
        _FakeManager("qdrant", events=events),
    )
    _install_fake_managers(monkeypatch, module, managers)
    _stub_contract(monkeypatch, module)

    with pytest.raises(RuntimeError, match="warmup failed") as captured:
        await module.main()

    assert captured.value is primary
    assert [event for event in events if event.endswith(":close")] == [
        "embedding:close",
        "elasticsearch:close",
        "dw_mysql:close",
        "meta_mysql:close",
        "qdrant:close",
    ]
    assert any(
        "resource=elasticsearch error_type=RuntimeError" in note
        for note in captured.value.__notes__
    )
    assert any(
        "resource=meta_mysql error_type=RuntimeError" in note
        for note in captured.value.__notes__
    )


@pytest.mark.asyncio
async def test_close_live_dependencies_starts_all_closes_concurrently(
    monkeypatch,
) -> None:
    module = importlib.import_module("evals.legacy.run_rag_live")
    started: list[str] = []
    all_started = asyncio.Event()
    release = asyncio.Event()

    class _ConcurrentManager:
        def __init__(self, name: str) -> None:
            self.name = name

        async def close(self) -> None:
            started.append(self.name)
            if len(started) == 3:
                all_started.set()
            await release.wait()

    dependencies = tuple(
        (name, _ConcurrentManager(name))
        for name in ("one", "two", "three")
    )
    monkeypatch.setattr(module, "_live_dependencies", lambda: dependencies)

    close_task = asyncio.create_task(module._close_live_dependencies())
    await asyncio.wait_for(all_started.wait(), timeout=0.5)
    release.set()

    assert await close_task == []
    assert set(started) == {"one", "two", "three"}


@pytest.mark.asyncio
async def test_live_runner_is_created_after_successful_warmup(monkeypatch) -> None:
    module = importlib.import_module("evals.legacy.run_rag_live")
    events: list[str] = []
    managers = tuple(
        _FakeManager(name, events=events)
        for name in (
            "embedding",
            "elasticsearch",
            "dw_mysql",
            "meta_mysql",
            "qdrant",
        )
    )
    _install_fake_managers(monkeypatch, module, managers)
    _stub_contract(monkeypatch, module)

    class _StopAfterWarmup(RuntimeError):
        pass

    def stop_after_warmup():
        assert "embedding:warmup" in events
        raise _StopAfterWarmup("runner boundary reached")

    monkeypatch.setattr(
        module,
        "create_retrieval_graph_runner",
        stop_after_warmup,
    )

    with pytest.raises(_StopAfterWarmup, match="runner boundary reached"):
        await module.main()

    assert events.index("embedding:warmup") < events.index("embedding:close")
