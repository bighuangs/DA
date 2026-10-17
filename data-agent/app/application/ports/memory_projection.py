"""会话记忆投影 Worker 的应用边界合同。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ProjectionEvent:
    """从持久化 Outbox 读取的最小、基础设施无关事件视图。"""

    event_id: str
    memory_id: str
    owner_id: str
    workspace_id: str
    target: str
    operation: str
    projection_version: str
    attempt_count: int


@dataclass(frozen=True, slots=True)
class ProjectionBatchResult:
    """一次批处理的可观测结果。"""

    claimed: int = 0
    completed: int = 0
    failed: int = 0
    summary_backfill: Mapping[str, object] = field(
        default_factory=lambda: {"status": "disabled"}
    )


class MemoryProjectionStore(Protocol):
    """Outbox claim、租约和权威状态读取端口。"""

    async def claim_pending(
        self,
        *,
        limit: int,
        max_attempts: int,
        lease_owner: str,
        lease_seconds: int,
        owner_allowlist: Sequence[str],
        skip_summary_upserts: bool,
    ) -> Sequence[ProjectionEvent]: ...

    async def complete(self, event_id: str, *, lease_owner: str) -> bool: ...

    async def fail(
        self,
        event: ProjectionEvent,
        *,
        error_type: str,
        lease_owner: str,
    ) -> bool: ...

    async def renew(
        self,
        event_id: str,
        *,
        lease_owner: str,
        lease_seconds: int,
    ) -> bool: ...

    async def authoritative_projection_active(
        self, event: ProjectionEvent
    ) -> bool: ...

    async def summary_projection_backfill_status(
        self,
    ) -> Mapping[str, int | bool]: ...


class MemoryProjectionDelivery(Protocol):
    """将一个 Outbox 事件投递到具体搜索后端的端口。"""

    async def deliver(self, event: ProjectionEvent) -> bool:
        """投递事件，并返回是否刚写入了 active 文档。"""

        ...

    async def delete(self, event: ProjectionEvent) -> None: ...

    async def promote_summary_alias(self) -> None: ...


class MemoryProjectionProcessor(Protocol):
    """Inbound Worker 每次唤醒时调用的应用用例。"""

    async def process_once(
        self,
        *,
        lease_owner: str,
        owner_allowlist: Sequence[str] = (),
    ) -> ProjectionBatchResult: ...


__all__ = [
    "MemoryProjectionDelivery",
    "MemoryProjectionProcessor",
    "MemoryProjectionStore",
    "ProjectionBatchResult",
    "ProjectionEvent",
]
