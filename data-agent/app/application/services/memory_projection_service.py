"""会话记忆 Outbox 的短事务投影用例。"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass

from loguru import logger

from app.application.ports.memory_projection import (
    MemoryProjectionDelivery,
    MemoryProjectionStore,
    ProjectionBatchResult,
    ProjectionEvent,
)
from app.foundation.request_context import get_req_id



def _projection_error_reason(exc: BaseException) -> str:
    """仅暴露稳定错误码，避免把外部异常中的内容或请求参数写入日志。"""

    detail = str(exc)
    if detail.startswith("Qdrant hybrid payload 缺少当前投影契约字段"):
        return "qdrant_payload_contract_missing_fields"
    if detail.startswith("embedding/payload/id 数量不一致"):
        return "embedding_payload_cardinality_mismatch"
    return "redacted"


@dataclass(frozen=True, slots=True)
class MemoryProjectionSettings:
    """一次 Worker 进程使用的不可变批处理设置。"""

    enabled: bool = True
    batch_size: int = 50
    max_attempts: int = 8
    lease_seconds: int = 90
    summary_projection_enabled: bool = False

    def __post_init__(self) -> None:
        for name in ("batch_size", "max_attempts", "lease_seconds"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} 必须大于 0")


class MemoryProjectionService:
    """以 at-least-once 语义投递一批投影事件。"""

    def __init__(
        self,
        *,
        store: MemoryProjectionStore,
        delivery: MemoryProjectionDelivery,
        settings: MemoryProjectionSettings,
    ) -> None:
        self._store = store
        self._delivery = delivery
        self._settings = settings

    async def process_once(
        self,
        *,
        lease_owner: str,
        owner_allowlist: Sequence[str] = (),
    ) -> ProjectionBatchResult:
        """Claim、投递并确认一批事件；取消不会被转成普通失败。"""

        if not self._settings.enabled:
            return ProjectionBatchResult()

        events = tuple(
            await self._store.claim_pending(
                limit=self._settings.batch_size,
                max_attempts=self._settings.max_attempts,
                lease_owner=lease_owner,
                lease_seconds=self._settings.lease_seconds,
                owner_allowlist=owner_allowlist,
                skip_summary_upserts=(
                    not self._settings.summary_projection_enabled
                ),
            )
        )
        completed_count = 0
        failed_count = 0
        heartbeats = {
            event.event_id: asyncio.create_task(
                self._heartbeat(event.event_id, lease_owner),
                name=f"memory-outbox-heartbeat:{event.event_id}",
            )
            for event in events
        }
        try:
            for event in events:
                try:
                    projected_active = await self._delivery.deliver(event)
                    await self.compensate_stale_upsert(
                        event,
                        projected_active=projected_active,
                    )
                    if await self._store.complete(
                        event.event_id,
                        lease_owner=lease_owner,
                    ):
                        completed_count += 1
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if await self._store.fail(
                        event,
                        error_type=type(exc).__name__,
                        lease_owner=lease_owner,
                    ):
                        failed_count += 1
                    logger.bind(request_id=get_req_id()).warning(
                        "conversation memory projection 失败 "
                        "target={} event={} error_type={} attempt={}/{} reason={}",
                        event.target,
                        event.event_id,
                        type(exc).__name__,
                        event.attempt_count,
                        self._settings.max_attempts,
                        _projection_error_reason(exc),
                    )
                finally:
                    heartbeat = heartbeats[event.event_id]
                    heartbeat.cancel()
                    with suppress(asyncio.CancelledError):
                        await heartbeat
        finally:
            for heartbeat in heartbeats.values():
                heartbeat.cancel()
            for heartbeat in heartbeats.values():
                with suppress(asyncio.CancelledError):
                    await heartbeat

        summary_backfill: dict[str, object] = {"status": "disabled"}
        if self._settings.summary_projection_enabled:
            try:
                summary_backfill = dict(
                    await self._store.summary_projection_backfill_status()
                )
                if bool(summary_backfill.get("ready")):
                    await self._delivery.promote_summary_alias()
                    summary_backfill["alias_promoted"] = True
            except Exception as exc:
                summary_backfill = {
                    "status": "degraded",
                    "error_type": type(exc).__name__,
                }

        return ProjectionBatchResult(
            claimed=len(events),
            completed=completed_count,
            failed=failed_count,
            summary_backfill=summary_backfill,
        )

    async def compensate_stale_upsert(
        self,
        event: ProjectionEvent,
        *,
        projected_active: bool,
    ) -> bool:
        """外部写完成后重读权威状态，撤销竞态产生的陈旧投影。"""

        if not projected_active:
            return False
        if await self._store.authoritative_projection_active(event):
            return False
        await self._delivery.delete(event)
        return True

    async def _heartbeat(self, event_id: str, lease_owner: str) -> None:
        lease_seconds = max(30, int(self._settings.lease_seconds))
        interval = max(5.0, lease_seconds / 3)
        while True:
            await asyncio.sleep(interval)
            owned = await self._store.renew(
                event_id,
                lease_owner=lease_owner,
                lease_seconds=lease_seconds,
            )
            if not owned:
                return


__all__ = ["MemoryProjectionService", "MemoryProjectionSettings"]
