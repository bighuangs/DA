"""会话记忆投影 Worker 的唯一生产组合根与进程入口。"""

from __future__ import annotations

import asyncio
import inspect
import os
import signal
from builtins import ExceptionGroup
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Protocol

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.inbound.worker.memory_projection_worker import (
    MemoryProjectionWorker,
)
from app.application.ports.memory_projection import (
    MemoryProjectionDelivery,
    MemoryProjectionProcessor,
    MemoryProjectionStore,
    ProjectionEvent,
)
from app.application.services.memory_projection_service import (
    MemoryProjectionService,
    MemoryProjectionSettings,
)
from app.bootstrap.lifecycle import LifecycleCoordinator
from app.config.app_config import app_config
from app.infrastructure.clients.embedding_client_manager import (
    embedding_client_manager,
)
from app.infrastructure.clients.es_client_manager import es_client_manager
from app.infrastructure.clients.mysql_client_manager import (
    meta_mysql_client_manager,
)
from app.infrastructure.clients.qdrant_client_manager import (
    qdrant_client_manager,
)
from app.infrastructure.persistence.mysql.records.conversation_memory_outbox_mysql import (
    ConversationMemoryOutboxMySQL,
)
from app.infrastructure.persistence.mysql.records.conversation_memory_item_mysql import (
    ConversationMemoryItemMySQL,
)
from app.infrastructure.persistence.mysql.records.conversation_summary_segment_mysql import (
    ConversationSummarySegmentMySQL,
)
from app.infrastructure.persistence.mysql.repositories.conversation_memory_repository import (
    ConversationMemoryRepository,
)
from app.infrastructure.persistence.mysql.repositories.conversation_summary_repository import (
    ConversationSummaryRepository,
)
from app.infrastructure.observability.logging import logger
from app.infrastructure.search.elasticsearch.repositories.conversation_memory_es_repository import (
    ConversationMemoryESRepository,
)
from app.infrastructure.search.elasticsearch.repositories.conversation_summary_es_repository import (
    ConversationSummaryESRepository,
)
from app.infrastructure.search.qdrant.repositories.conversation_memory_qdrant_repository import (
    ConversationMemoryQdrantRepository,
)
from app.infrastructure.search.qdrant.repositories.conversation_summary_qdrant_repository import (
    ConversationSummaryQdrantRepository,
)


WORKER_ENABLE_ENV = "MEMORY_PROJECTION_WORKER_ENABLED"
WORKER_OWNER_ALLOWLIST_ENV = "MEMORY_PROJECTION_OWNER_ALLOWLIST"


class ManagedWorker(Protocol):
    """可由组合根启动和停止的后台 Worker。"""

    def start(self) -> None | Awaitable[None]: ...

    async def stop(self) -> None: ...


class _ManagedClient(Protocol):
    """Worker 进程拥有的最小 Client Manager 生命周期。"""

    def init(self) -> None: ...

    async def close(self) -> None: ...


WorkerFactory = Callable[[], ManagedWorker]


def _projection_event(row: ConversationMemoryOutboxMySQL) -> ProjectionEvent:
    return ProjectionEvent(
        event_id=str(row.event_id),
        memory_id=str(row.memory_id),
        owner_id=str(row.owner_id),
        workspace_id=str(row.workspace_id),
        target=str(row.target),
        operation=str(row.operation),
        projection_version=str(row.projection_version),
        attempt_count=int(row.attempt_count or 0),
    )


async def enqueue_summary_projection_backfill(
    session: AsyncSession,
    *,
    batch_size: int = 500,
) -> dict[str, int]:
    """在 Bootstrap 的具体持久化边界补齐当前摘要投影事件。"""

    cursor = ""
    segment_count = 0
    requested_events = 0
    size = max(1, min(int(batch_size), 2000))
    while True:
        rows = list(
            (
                await session.execute(
                    Select(ConversationSummarySegmentMySQL)
                    .where(
                        ConversationSummarySegmentMySQL.status == "active",
                        ConversationSummarySegmentMySQL.deleted_at.is_(None),
                        ConversationSummarySegmentMySQL.summary_id > cursor,
                    )
                    .order_by(ConversationSummarySegmentMySQL.summary_id)
                    .limit(size)
                )
            )
            .scalars()
            .all()
        )
        if not rows:
            break
        for row in rows:
            repository = ConversationMemoryRepository(
                session,
                owner_id=row.owner_id,
                workspace_id=row.workspace_id,
            )
            for target, version in (
                (
                    "summary_es",
                    app_config.memory.summary_es_projection_version,
                ),
                (
                    "summary_qdrant",
                    app_config.memory.summary_qdrant_projection_version,
                ),
            ):
                await repository.enqueue_projection(
                    memory_id=row.summary_id,
                    target=target,
                    operation="upsert",
                    projection_version=version,
                    requeue_existing=True,
                )
                requested_events += 1
            segment_count += 1
        cursor = str(rows[-1].summary_id)
        if len(rows) < size:
            break
    return {
        "active_segments": segment_count,
        "requested_events": requested_events,
    }


async def enqueue_memory_qdrant_projection_backfill_batch(
    session: AsyncSession,
    *,
    projection_version: str | None = None,
    batch_size: int = 500,
) -> dict[str, int | bool | str]:
    """为当前显式投影版本补齐一批缺失的 active memory 事件。

    投影载荷合同变更通过 ``projection_version`` 升级表达，不再把异常类型当作
    迁移哨兵。每次只处理一个短事务批次；唯一键保证多实例并发补齐仍然幂等。
    返回的 ``complete`` 仅表示 Outbox enqueue 扫描完成，不代表队列已排空或
    Qdrant 物理投影已经完成。
    """

    version = str(
        projection_version or app_config.memory.qdrant_projection_version
    )
    size = max(1, min(int(batch_size), 2000))
    current_event_exists = (
        select(ConversationMemoryOutboxMySQL.event_id)
        .where(
            ConversationMemoryOutboxMySQL.memory_id
            == ConversationMemoryItemMySQL.memory_id,
            ConversationMemoryOutboxMySQL.owner_id
            == ConversationMemoryItemMySQL.owner_id,
            ConversationMemoryOutboxMySQL.workspace_id
            == ConversationMemoryItemMySQL.workspace_id,
            ConversationMemoryOutboxMySQL.target == "qdrant",
            ConversationMemoryOutboxMySQL.operation == "upsert",
            ConversationMemoryOutboxMySQL.projection_version == version,
        )
        .correlate(ConversationMemoryItemMySQL)
        .exists()
    )
    active_rows = list(
        (
            await session.execute(
                Select(ConversationMemoryItemMySQL)
                .where(
                    ConversationMemoryItemMySQL.status == "active",
                    ConversationMemoryItemMySQL.deleted_at.is_(None),
                    ~current_event_exists,
                )
                .order_by(ConversationMemoryItemMySQL.memory_id)
                .limit(size)
            )
        )
        .scalars()
        .all()
    )
    for item in active_rows:
        await ConversationMemoryRepository(
            session,
            owner_id=str(item.owner_id),
            workspace_id=str(item.workspace_id),
        ).enqueue_projection(
            memory_id=str(item.memory_id),
            target="qdrant",
            operation="upsert",
            projection_version=version,
        )
    await session.flush()
    return {
        "status": "queued" if active_rows else "complete",
        "projection_version": version,
        # 表示本轮发出的幂等 enqueue 请求数，不宣称均由本实例新建。
        "requested_events": len(active_rows),
        # 恰好填满一批时再跑一轮确认，避免额外 count 扫描。
        "complete": len(active_rows) < size,
    }


class _MemoryQdrantProjectionBackfill:
    """由 Worker 逐轮执行的短事务投影版本补齐任务。"""

    def __init__(self, *, batch_size: int = 500) -> None:
        self._batch_size = max(1, min(int(batch_size), 2000))
        self._complete = False
        self._status: dict[str, object] = {"status": "pending"}

    async def __call__(self) -> Mapping[str, object]:
        if self._complete:
            return dict(self._status)
        if not (
            app_config.memory.enabled and app_config.memory.long_term_enabled
        ):
            self._complete = True
            self._status = {"status": "disabled", "complete": True}
            return dict(self._status)
        async with _meta_session() as session:
            result = await enqueue_memory_qdrant_projection_backfill_batch(
                session,
                batch_size=self._batch_size,
            )
            await session.commit()
        self._status = dict(result)
        self._complete = bool(result.get("complete"))
        if int(result.get("requested_events") or 0) > 0:
            logger.info(
                "conversation memory Qdrant 幂等补齐请求已提交 "
                "projection_version={} enqueue_requests={} "
                "enqueue_scan_complete={}",
                result.get("projection_version"),
                result.get("requested_events"),
                self._complete,
            )
        return dict(self._status)


async def summary_projection_backfill_status(
    session: AsyncSession,
) -> dict[str, int | bool]:
    """读取当前投影版本的完整 backfill 状态。"""

    active_filter = (
        ConversationSummarySegmentMySQL.status == "active",
        ConversationSummarySegmentMySQL.deleted_at.is_(None),
    )
    active_ids = select(ConversationSummarySegmentMySQL.summary_id).where(
        *active_filter
    )
    active_count = int(
        await session.scalar(
            select(func.count())
            .select_from(ConversationSummarySegmentMySQL)
            .where(*active_filter)
        )
        or 0
    )
    counts: dict[str, int] = {}
    for target, version in (
        ("summary_es", app_config.memory.summary_es_projection_version),
        (
            "summary_qdrant",
            app_config.memory.summary_qdrant_projection_version,
        ),
    ):
        counts[target] = int(
            await session.scalar(
                select(
                    func.count(
                        func.distinct(
                            ConversationMemoryOutboxMySQL.memory_id
                        )
                    )
                ).where(
                    ConversationMemoryOutboxMySQL.memory_id.in_(active_ids),
                    ConversationMemoryOutboxMySQL.target == target,
                    ConversationMemoryOutboxMySQL.operation == "upsert",
                    ConversationMemoryOutboxMySQL.projection_version == version,
                    ConversationMemoryOutboxMySQL.status == "completed",
                )
            )
            or 0
        )
    return {
        "active_segments": active_count,
        "summary_es_completed": counts["summary_es"],
        "summary_qdrant_completed": counts["summary_qdrant"],
        "ready": (
            counts["summary_es"] == active_count
            and counts["summary_qdrant"] == active_count
        ),
    }


class _SqlMemoryProjectionStore(MemoryProjectionStore):
    """通过短 SQLAlchemy Session 实现 Outbox 与权威状态端口。"""

    def __init__(self, session_factory: Callable[[], Any]) -> None:
        self._session_factory = session_factory

    async def claim_pending(
        self,
        *,
        limit: int,
        max_attempts: int,
        lease_owner: str,
        lease_seconds: int,
        owner_allowlist: Sequence[str],
        skip_summary_upserts: bool,
    ) -> Sequence[ProjectionEvent]:
        async with self._session_factory() as session:
            rows = await ConversationMemoryRepository.claim_pending_outbox(
                session,
                limit=limit,
                max_attempts=max_attempts,
                worker_id=lease_owner,
                lease_seconds=lease_seconds,
                owner_allowlist=owner_allowlist,
                skip_summary_upserts=skip_summary_upserts,
            )
        return tuple(_projection_event(row) for row in rows)

    async def complete(self, event_id: str, *, lease_owner: str) -> bool:
        async with self._session_factory() as session:
            return await ConversationMemoryRepository.complete_outbox(
                session,
                event_id,
                worker_id=lease_owner,
            )

    async def fail(
        self,
        event: ProjectionEvent,
        *,
        error_type: str,
        lease_owner: str,
    ) -> bool:
        async with self._session_factory() as session:
            return await ConversationMemoryRepository.fail_outbox(
                session,
                event.event_id,
                error_type,
                event.attempt_count,
                worker_id=lease_owner,
            )

    async def renew(
        self,
        event_id: str,
        *,
        lease_owner: str,
        lease_seconds: int,
    ) -> bool:
        async with self._session_factory() as session:
            return await ConversationMemoryRepository.renew_outbox_lease(
                session,
                event_id,
                worker_id=lease_owner,
                lease_seconds=lease_seconds,
            )

    async def authoritative_projection_active(
        self,
        event: ProjectionEvent,
    ) -> bool:
        async with self._session_factory() as session:
            if event.target.startswith("summary_"):
                item = await ConversationSummaryRepository.outbox_summary(
                    session,
                    event.memory_id,
                    event.owner_id,
                    event.workspace_id,
                )
            else:
                item = await ConversationMemoryRepository.outbox_memory(
                    session,
                    event,
                )
        return bool(
            item is not None
            and item.status == "active"
            and item.deleted_at is None
        )

    async def summary_projection_backfill_status(
        self,
    ) -> Mapping[str, int | bool]:
        async with self._session_factory() as session:
            return await summary_projection_backfill_status(session)


class _SearchMemoryProjectionDelivery(MemoryProjectionDelivery):
    """把投影事件映射到 ES/Qdrant 具体 Repository。"""

    def __init__(self, session_factory: Callable[[], Any]) -> None:
        self._session_factory = session_factory

    def _target(self, event: ProjectionEvent) -> Any:
        if event.target == "es":
            return ConversationMemoryESRepository(
                es_client_manager.client,
                projection_version=event.projection_version,
                manage_alias=False,
            )
        if event.target == "qdrant":
            return ConversationMemoryQdrantRepository(
                qdrant_client_manager.client,
                projection_version=event.projection_version,
            )
        if event.target == "summary_es":
            return ConversationSummaryESRepository(
                es_client_manager.client,
                projection_version=event.projection_version,
                manage_alias=False,
            )
        if event.target == "summary_qdrant":
            return ConversationSummaryQdrantRepository(
                qdrant_client_manager.client,
                projection_version=event.projection_version,
            )
        raise ValueError(f"未知 memory projection target：{event.target}")

    async def _load(self, event: ProjectionEvent) -> tuple[object | None, list]:
        async with self._session_factory() as session:
            if event.target.startswith("summary_"):
                item = await ConversationSummaryRepository.outbox_summary(
                    session,
                    event.memory_id,
                    event.owner_id,
                    event.workspace_id,
                )
                signals = (
                    await ConversationSummaryRepository(
                        session,
                        owner_id=event.owner_id,
                        workspace_id=event.workspace_id,
                    ).list_signals_for_summaries([event.memory_id])
                ).get(event.memory_id, [])
                return item, list(signals)
            item = await ConversationMemoryRepository.outbox_memory(
                session,
                event,
            )
            return item, []

    async def deliver(self, event: ProjectionEvent) -> bool:
        target = self._target(event)
        item, signals = await self._load(event)
        if (
            event.operation == "delete"
            or item is None
            or getattr(item, "status", None) != "active"
        ):
            await target.delete(event.memory_id)
            return False
        if event.target == "es":
            await target.upsert(item)
        elif event.target == "summary_es":
            await target.upsert(item, signals)
        else:
            embedding = (
                await embedding_client_manager.client.aembed_documents(
                    [str(getattr(item, "content_masked"))]
                )
            )[0]
            await target.upsert(item, embedding)
        return True

    async def delete(self, event: ProjectionEvent) -> None:
        await self._target(event).delete(event.memory_id)

    async def promote_summary_alias(self) -> None:
        await ConversationSummaryESRepository(
            es_client_manager.client,
            projection_version=(
                app_config.memory.summary_es_projection_version
            ),
        ).promote_alias()


class _MemoryProjectionInfrastructure:
    """独立 Worker 进程拥有的外部 Client 生命周期。"""

    def __init__(self, managers: Sequence[_ManagedClient]) -> None:
        self._managers = tuple(managers)
        self._started: list[_ManagedClient] = []

    async def start(self) -> None:
        for manager in self._managers:
            manager.init()
            self._started.append(manager)

    async def close(self) -> None:
        managers, self._started = self._started, []
        errors: list[Exception] = []
        for manager in reversed(managers):
            try:
                await manager.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup("投影 Worker 资源关闭失败", errors)


def _worker_settings() -> MemoryProjectionSettings:
    return MemoryProjectionSettings(
        enabled=bool(app_config.memory.enabled),
        batch_size=int(app_config.memory.projection_batch_size),
        max_attempts=int(app_config.memory.projection_max_attempts),
        lease_seconds=int(app_config.memory.projection_lease_seconds),
        summary_projection_enabled=bool(
            app_config.memory.summary_projection_enabled
        ),
    )


def _meta_session() -> Any:
    """在实际批处理时解析 Session Factory，避免导入阶段捕获未初始化值。"""

    session_factory = meta_mysql_client_manager.session_factory
    if session_factory is None:
        raise RuntimeError("Meta MySQL 尚未初始化，无法运行投影 Worker")
    return session_factory()


def create_memory_projection_processor(
    *,
    store: MemoryProjectionStore | None = None,
    delivery: MemoryProjectionDelivery | None = None,
) -> MemoryProjectionService:
    """装配一次可测试的投影用例，不在构造阶段解析任何 Client。"""

    return MemoryProjectionService(
        store=store or _SqlMemoryProjectionStore(_meta_session),
        delivery=delivery or _SearchMemoryProjectionDelivery(_meta_session),
        settings=_worker_settings(),
    )


def create_memory_projection_worker(
    *,
    processor: MemoryProjectionProcessor | None = None,
    store: MemoryProjectionStore | None = None,
    delivery: MemoryProjectionDelivery | None = None,
    owner_allowlist: Sequence[str] = (),
    maintenance: Callable[[], Awaitable[Mapping[str, object]]] | None = None,
) -> MemoryProjectionWorker:
    """显式装配 Worker；构造阶段不初始化或连接任何外部 Client。"""

    if processor is not None and (store is not None or delivery is not None):
        raise ValueError("传入 processor 时不能再传入 store 或 delivery")
    resolved_processor = processor or create_memory_projection_processor(
        store=store,
        delivery=delivery,
    )
    return MemoryProjectionWorker(
        processor=resolved_processor,
        poll_seconds=float(app_config.memory.projection_poll_seconds),
        owner_allowlist=owner_allowlist,
        maintenance=maintenance,
    )


def create_memory_projection_lifecycle() -> LifecycleCoordinator:
    """创建独立进程资源生命周期；调用本函数仍不会初始化 Client。"""

    managers: tuple[_ManagedClient, ...] = (
        embedding_client_manager,
        es_client_manager,
        meta_mysql_client_manager,
        qdrant_client_manager,
    )
    return LifecycleCoordinator(
        (lambda: _MemoryProjectionInfrastructure(managers),)
    )


async def run_worker(
    *,
    worker_factory: WorkerFactory,
    stop_event: asyncio.Event,
    lifecycle: LifecycleCoordinator | None = None,
) -> None:
    """在基础资源可用后创建 Worker，并在退出或取消时保证停止。"""

    coordinator = lifecycle or LifecycleCoordinator()
    async with coordinator.scope():
        worker = worker_factory()
        try:
            started = worker.start()
            if inspect.isawaitable(started):
                await started
            await stop_event.wait()
        except BaseException as primary_error:
            try:
                await worker.stop()
            except BaseException as stop_error:
                primary_error.add_note(
                    "Worker 停止失败："
                    f"{type(stop_error).__name__}: {stop_error}"
                )
            raise
        else:
            await worker.stop()


def _enabled(environ: Mapping[str, str]) -> bool:
    return str(environ.get(WORKER_ENABLE_ENV, "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _owner_allowlist(environ: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            item.strip()
            for item in str(
                environ.get(WORKER_OWNER_ALLOWLIST_ENV, "")
            ).split(",")
            if item.strip()
        )
    )


def _install_stop_signal_handlers(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop_event.set)
        except (NotImplementedError, RuntimeError):
            signal.signal(
                signum,
                lambda _signum, _frame: loop.call_soon_threadsafe(
                    stop_event.set
                ),
            )


async def run_production_memory_projection_worker(
    *,
    stop_event: asyncio.Event | None = None,
    environ: Mapping[str, str] | None = None,
) -> None:
    """运行可部署 Worker；只有显式调用后才初始化外部资源。"""

    resolved_environ = os.environ if environ is None else environ
    resolved_stop_event = stop_event or asyncio.Event()
    if stop_event is None:
        _install_stop_signal_handlers(resolved_stop_event)
    await run_worker(
        worker_factory=lambda: create_memory_projection_worker(
            owner_allowlist=_owner_allowlist(resolved_environ),
            maintenance=_MemoryQdrantProjectionBackfill(),
        ),
        stop_event=resolved_stop_event,
        lifecycle=create_memory_projection_lifecycle(),
    )


def main() -> None:
    """Console 入口；默认拒绝误启动，避免意外处理持久化 Outbox。"""

    if not _enabled(os.environ):
        raise SystemExit(
            f"独立投影 Worker 默认关闭；请显式设置 {WORKER_ENABLE_ENV}=true"
        )
    asyncio.run(run_production_memory_projection_worker())


# HTTP lifespan 与请求内 notifier 共享同一进程级 Worker。这里只完成对象
# 装配，不调用任何 Client Manager.init()，也不读取 manager.client。
memory_projection_worker = create_memory_projection_worker(
    maintenance=_MemoryQdrantProjectionBackfill()
)


__all__ = [
    "ManagedWorker",
    "WORKER_ENABLE_ENV",
    "WORKER_OWNER_ALLOWLIST_ENV",
    "WorkerFactory",
    "create_memory_projection_lifecycle",
    "create_memory_projection_processor",
    "create_memory_projection_worker",
    "enqueue_memory_qdrant_projection_backfill_batch",
    "enqueue_summary_projection_backfill",
    "main",
    "memory_projection_worker",
    "run_production_memory_projection_worker",
    "run_worker",
    "summary_projection_backfill_status",
]
