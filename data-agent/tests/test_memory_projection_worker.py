from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.adapters.inbound.worker.memory_projection_worker import (
    MemoryProjectionWorker,
)
from app.application.ports.memory_projection import ProjectionBatchResult
from app.bootstrap import worker as worker_bootstrap


@pytest.mark.asyncio
async def test_notify_during_batch_triggers_immediate_follow_up() -> None:
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_started = asyncio.Event()

    class _Processor:
        calls = 0

        async def process_once(self, **_: object) -> ProjectionBatchResult:
            self.calls += 1
            if self.calls == 1:
                first_started.set()
                await release_first.wait()
            elif self.calls == 2:
                second_started.set()
            return ProjectionBatchResult()

    processor = _Processor()
    worker = MemoryProjectionWorker(processor=processor, poll_seconds=30)
    worker.start()
    await first_started.wait()

    worker.notify()
    release_first.set()
    await asyncio.wait_for(second_started.wait(), timeout=0.5)
    await worker.stop()

    assert processor.calls >= 2
    assert worker.status()["running"] is False


@pytest.mark.asyncio
async def test_stop_cancels_inflight_processor() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class _Processor:
        async def process_once(self, **_: object) -> ProjectionBatchResult:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

    worker = MemoryProjectionWorker(processor=_Processor())
    worker.start()
    await started.wait()

    await worker.stop()

    assert cancelled.is_set()
    assert worker.status()["running"] is False


@pytest.mark.asyncio
async def test_projection_maintenance_runs_before_outbox_batch() -> None:
    order: list[str] = []

    class _Processor:
        async def process_once(self, **_: object) -> ProjectionBatchResult:
            order.append("processor")
            return ProjectionBatchResult()

    async def maintenance() -> dict[str, object]:
        order.append("maintenance")
        return {"status": "queued", "requested_events": 2}

    worker = MemoryProjectionWorker(
        processor=_Processor(),
        maintenance=maintenance,
    )

    assert await worker.process_once() == 0
    assert order == ["maintenance", "processor"]
    assert worker.status()["maintenance"] == {
        "status": "queued",
        "requested_events": 2,
    }


@pytest.mark.asyncio
async def test_projection_maintenance_failure_does_not_block_outbox() -> None:
    processor = AsyncMock(return_value=ProjectionBatchResult())

    async def maintenance() -> dict[str, object]:
        raise RuntimeError("maintenance failed")

    worker = MemoryProjectionWorker(
        processor=SimpleNamespace(process_once=processor),
        maintenance=maintenance,
    )

    assert await worker.process_once() == 0
    processor.assert_awaited_once()
    assert worker.status()["maintenance"] == {
        "status": "failed",
        "error_type": "RuntimeError",
    }


@pytest.mark.asyncio
async def test_bootstrap_factory_resolves_session_factory_after_construction() -> None:
    @asynccontextmanager
    async def session_factory():
        yield object()

    with (
        patch.object(worker_bootstrap.app_config.memory, "enabled", True),
        patch.object(
            worker_bootstrap.app_config.memory,
            "summary_projection_enabled",
            False,
        ),
    ):
        worker = worker_bootstrap.create_memory_projection_worker()

    with (
        patch.object(
            worker_bootstrap.meta_mysql_client_manager,
            "session_factory",
            session_factory,
        ),
        patch.object(
            worker_bootstrap.ConversationMemoryRepository,
            "claim_pending_outbox",
            AsyncMock(return_value=[]),
        ),
    ):
        assert await worker.process_once() == 0


def test_console_entry_is_disabled_before_async_runtime_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(worker_bootstrap.WORKER_ENABLE_ENV, raising=False)
    async_run = AsyncMock()

    with (
        patch.object(worker_bootstrap.asyncio, "run", async_run),
        pytest.raises(SystemExit, match="默认关闭"),
    ):
        worker_bootstrap.main()

    async_run.assert_not_called()
