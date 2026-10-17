from __future__ import annotations

import asyncio

import pytest

from app.application.ports.memory_projection import ProjectionEvent
from app.application.services.memory_projection_service import (
    MemoryProjectionService,
    MemoryProjectionSettings,
)
from app.infrastructure.observability.logging import logger


EVENT = ProjectionEvent(
    event_id="event-1",
    memory_id="memory-1",
    owner_id="owner-1",
    workspace_id="workspace-1",
    target="es",
    operation="upsert",
    projection_version="memory-v1",
    attempt_count=1,
)


class _Store:
    def __init__(self, *, authoritative_active: bool = True) -> None:
        self.authoritative_active = authoritative_active
        self.completed: list[tuple[str, str]] = []
        self.failed: list[tuple[str, str, str]] = []

    async def claim_pending(self, **_: object):
        return (EVENT,)

    async def complete(self, event_id: str, *, lease_owner: str) -> bool:
        self.completed.append((event_id, lease_owner))
        return True

    async def fail(
        self,
        event: ProjectionEvent,
        *,
        error_type: str,
        lease_owner: str,
    ) -> bool:
        self.failed.append((event.event_id, error_type, lease_owner))
        return True

    async def renew(self, *_: object, **__: object) -> bool:
        return True

    async def authoritative_projection_active(
        self,
        _: ProjectionEvent,
    ) -> bool:
        return self.authoritative_active

    async def summary_projection_backfill_status(self):
        return {"ready": False}


class _Delivery:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.delivered: list[ProjectionEvent] = []
        self.deleted: list[ProjectionEvent] = []

    async def deliver(self, event: ProjectionEvent) -> bool:
        self.delivered.append(event)
        if self.error is not None:
            raise self.error
        return True

    async def delete(self, event: ProjectionEvent) -> None:
        self.deleted.append(event)

    async def promote_summary_alias(self) -> None:
        return None


@pytest.mark.asyncio
async def test_projection_service_compensates_stale_upsert_before_completion() -> None:
    store = _Store(authoritative_active=False)
    delivery = _Delivery()
    service = MemoryProjectionService(
        store=store,
        delivery=delivery,
        settings=MemoryProjectionSettings(summary_projection_enabled=False),
    )

    result = await service.process_once(lease_owner="lease-1")

    assert result.claimed == 1
    assert result.completed == 1
    assert result.failed == 0
    assert delivery.deleted == [EVENT]
    assert store.completed == [(EVENT.event_id, "lease-1")]


@pytest.mark.asyncio
async def test_projection_service_records_delivery_failure_without_completion(
) -> None:
    store = _Store()
    sensitive_detail = "invalid target token=do-not-log"
    service = MemoryProjectionService(
        store=store,
        delivery=_Delivery(error=ValueError(sensitive_detail)),
        settings=MemoryProjectionSettings(summary_projection_enabled=False),
    )

    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(message.record["message"]),
        format="{message}",
    )
    try:
        result = await service.process_once(lease_owner="lease-2")
    finally:
        logger.remove(sink_id)

    assert result.completed == 0
    assert result.failed == 1
    assert store.completed == []
    assert store.failed == [(EVENT.event_id, "ValueError", "lease-2")]
    rendered = "\n".join(messages)
    assert sensitive_detail not in rendered
    assert "reason=redacted" in rendered


@pytest.mark.asyncio
async def test_projection_service_propagates_cancellation_without_marking_failure() -> None:
    started = asyncio.Event()
    blocker = asyncio.Event()
    store = _Store()

    class _BlockingDelivery(_Delivery):
        async def deliver(self, event: ProjectionEvent) -> bool:
            self.delivered.append(event)
            started.set()
            await blocker.wait()
            return True

    service = MemoryProjectionService(
        store=store,
        delivery=_BlockingDelivery(),
        settings=MemoryProjectionSettings(summary_projection_enabled=False),
    )
    task = asyncio.create_task(service.process_once(lease_owner="lease-3"))
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert store.completed == []
    assert store.failed == []
