"""基础设施无关的会话记忆投影 Worker 进程适配器。"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress

from loguru import logger

from app.application.ports.memory_projection import (
    MemoryProjectionProcessor,
)



ProjectionMaintenance = Callable[[], Awaitable[Mapping[str, object]]]


class MemoryProjectionWorker:
    """只负责批处理唤醒、轮询、取消和进程级指标。"""

    def __init__(
        self,
        *,
        processor: MemoryProjectionProcessor,
        poll_seconds: float = 1.0,
        owner_allowlist: Sequence[str] = (),
        maintenance: ProjectionMaintenance | None = None,
    ) -> None:
        if float(poll_seconds) <= 0:
            raise ValueError("poll_seconds 必须大于 0")
        self._processor = processor
        self._maintenance = maintenance
        self._poll_seconds = max(0.5, float(poll_seconds))
        self._task: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()
        self._stopping = False
        self.processed = 0
        self.failed = 0
        self.worker_id = f"memory-projection-{uuid.uuid4()}"
        self.owner_allowlist = tuple(
            dict.fromkeys(
                str(value)
                for value in owner_allowlist
                if str(value or "").strip()
            )
        )
        self.summary_backfill: dict[str, object] = {"status": "disabled"}
        self.maintenance_status: dict[str, object] = {"status": "disabled"}

    def _new_lease_owner(self) -> str:
        """为每批 claim 生成唯一且满足持久化长度上限的租约 Owner。"""

        return f"{self.worker_id[:36]}:{uuid.uuid4().hex}"[:80]

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stopping = False
            self._task = asyncio.create_task(
                self._run(),
                name="conversation-memory-projection-worker",
            )

    def notify(self) -> None:
        self._wake.set()

    async def stop(self) -> None:
        self._stopping = True
        self._wake.set()
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def _run(self) -> None:
        while not self._stopping:
            # 先消费旧唤醒标记；process_once 期间的新 notify 必须保留，
            # 这样当前批次完成后会立即再跑一批而不是等待轮询超时。
            self._wake.clear()
            try:
                await self.process_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.bind(request_id="").warning(
                    "conversation memory outbox 批处理失败 error_type={}",
                    type(exc).__name__,
                )
            try:
                await asyncio.wait_for(
                    self._wake.wait(),
                    timeout=self._poll_seconds,
                )
            except asyncio.TimeoutError:
                pass

    async def process_once(self) -> int:
        if self._maintenance is not None:
            try:
                self.maintenance_status = dict(await self._maintenance())
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # 投影版本补齐是可恢复维护任务，失败时不能阻断已有 Outbox 消费。
                self.maintenance_status = {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                }
                logger.bind(request_id="").warning(
                    "conversation memory projection maintenance 失败 "
                    "error_type={}",
                    type(exc).__name__,
                )
        result = await self._processor.process_once(
            lease_owner=self._new_lease_owner(),
            owner_allowlist=self.owner_allowlist,
        )
        self.processed += int(result.completed)
        self.failed += int(result.failed)
        self.summary_backfill = dict(result.summary_backfill)
        return int(result.claimed)

    def status(self) -> dict[str, object]:
        return {
            "running": bool(self._task and not self._task.done()),
            "processed": self.processed,
            "failed": self.failed,
            "summary_backfill": dict(self.summary_backfill),
            "maintenance": dict(self.maintenance_status),
        }


__all__ = ["MemoryProjectionWorker"]
