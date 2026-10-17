"""组合根与生命周期行为测试。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI

from app.bootstrap.api import create_app
from app.bootstrap.container import BootstrapContainer
from app.bootstrap.lifecycle import LifecycleCoordinator
from app.bootstrap.settings import BootstrapSettings
from app.bootstrap.worker import run_worker


class _RecordingResource:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def start(self) -> None:
        self._events.append("resource:start")

    async def close(self) -> None:
        self._events.append("resource:close")


class _RecordingWorker:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def start(self) -> None:
        self._events.append("worker:start")

    async def stop(self) -> None:
        self._events.append("worker:stop")


@pytest.mark.asyncio
async def test_http_factory_and_resources_are_deferred_until_explicit_phases() -> None:
    events: list[str] = []

    def application_factory() -> FastAPI:
        events.append("application:create")

        @asynccontextmanager
        async def legacy_lifespan(_: FastAPI):
            events.append("legacy:start")
            try:
                yield
            finally:
                events.append("legacy:close")

        return FastAPI(lifespan=legacy_lifespan)

    coordinator = LifecycleCoordinator((lambda: _RecordingResource(events),))
    container = BootstrapContainer(
        settings=BootstrapSettings(),
        lifecycle=coordinator,
        http_application_factory=application_factory,
    )

    assert events == []
    application = create_app(container=container)
    assert events == ["application:create"]
    assert application.state.bootstrap_container is container

    async with application.router.lifespan_context(application):
        assert events == ["application:create", "legacy:start", "resource:start"]
        assert coordinator.active

    assert events == [
        "application:create",
        "legacy:start",
        "resource:start",
        "resource:close",
        "legacy:close",
    ]
    assert not coordinator.active


def test_default_http_factory_is_selected_without_a_legacy_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = FastAPI()
    monkeypatch.setattr(
        "app.bootstrap.container._default_http_application_factory",
        lambda _settings: expected,
    )
    container = BootstrapContainer(
        settings=BootstrapSettings(),
        lifecycle=LifecycleCoordinator(),
    )

    application = container.build_http_application()

    assert application is expected
    assert application.state.bootstrap_container is container


@pytest.mark.asyncio
async def test_worker_is_created_only_after_lifecycle_start() -> None:
    events: list[str] = []
    stop_event = asyncio.Event()
    stop_event.set()

    def worker_factory() -> _RecordingWorker:
        events.append("worker:create")
        return _RecordingWorker(events)

    await run_worker(
        worker_factory=worker_factory,
        stop_event=stop_event,
        lifecycle=LifecycleCoordinator((lambda: _RecordingResource(events),)),
    )

    assert events == [
        "resource:start",
        "worker:create",
        "worker:start",
        "worker:stop",
        "resource:close",
    ]


@pytest.mark.asyncio
async def test_worker_cancellation_stops_worker_before_resources_close() -> None:
    events: list[str] = []
    stop_event = asyncio.Event()
    worker_started = asyncio.Event()

    class _BlockingWorker(_RecordingWorker):
        def start(self) -> None:
            super().start()
            worker_started.set()

    task = asyncio.create_task(
        run_worker(
            worker_factory=lambda: _BlockingWorker(events),
            stop_event=stop_event,
            lifecycle=LifecycleCoordinator(
                (lambda: _RecordingResource(events),)
            ),
        )
    )
    await worker_started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert events == [
        "resource:start",
        "worker:start",
        "worker:stop",
        "resource:close",
    ]


@pytest.mark.asyncio
async def test_worker_cancellation_remains_primary_when_stop_fails() -> None:
    events: list[str] = []
    worker_started = asyncio.Event()

    class _FailingStopWorker(_RecordingWorker):
        def start(self) -> None:
            super().start()
            worker_started.set()

        async def stop(self) -> None:
            await super().stop()
            raise RuntimeError("stop failed")

    task = asyncio.create_task(
        run_worker(
            worker_factory=lambda: _FailingStopWorker(events),
            stop_event=asyncio.Event(),
            lifecycle=LifecycleCoordinator(
                (lambda: _RecordingResource(events),)
            ),
        )
    )
    await worker_started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError) as captured:
        await task

    assert any("Worker 停止失败" in note for note in captured.value.__notes__)
    assert events == [
        "resource:start",
        "worker:start",
        "worker:stop",
        "resource:close",
    ]
