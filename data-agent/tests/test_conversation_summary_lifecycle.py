"""Failure, lifecycle and rollout contracts for TODO-96."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import Select, func, select, update
from sqlalchemy.dialects import mysql
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.config.app_config import app_config
from app.infrastructure.clients.mysql_client_manager import meta_mysql_client_manager
from app.infrastructure.config.conversation import load_conversation_settings
from app.infrastructure.persistence.mysql.conversation_adapters import (
    SqlConversationMemoryPersistence,
    SqlConversationMemoryUnitOfWork,
)
from app.application.services.conversation_memory_policy import build_extractive_summary_segment
from app.infrastructure.persistence.mysql.records.base import Base
from app.infrastructure.persistence.mysql.records.conversation_memory_item_mysql import (
    ConversationMemoryItemMySQL,
)
from app.infrastructure.persistence.mysql.records.conversation_memory_outbox_mysql import (
    ConversationMemoryOutboxMySQL,
)
from app.infrastructure.persistence.mysql.records.conversation_session_mysql import ConversationSessionMySQL
from app.infrastructure.persistence.mysql.records.conversation_summary_segment_mysql import (
    ConversationSummarySegmentMySQL,
)
from app.infrastructure.search.elasticsearch.repositories.conversation_summary_es_repository import (
    ConversationSummaryESRepository,
)
from app.infrastructure.search.elasticsearch.repositories.conversation_memory_es_repository import (
    ConversationMemoryESRepository,
)
from app.infrastructure.persistence.mysql.repositories.conversation_memory_repository import (
    ConversationMemoryRepository,
    _projection_insert_statement,
    _projection_lookup_statement,
)
from app.infrastructure.persistence.mysql.repositories.conversation_summary_repository import (
    ConversationSummaryRepository,
)
from app.infrastructure.search.qdrant.repositories.conversation_summary_qdrant_repository import (
    ConversationSummaryQdrantRepository,
)
from app.infrastructure.search.qdrant.repositories.conversation_memory_qdrant_repository import (
    ConversationMemoryQdrantRepository,
)
from app.application.services.conversation_summary_service import ConversationSummaryService
from app.application.services.conversation_memory_service import ConversationMemoryService
from app.application.ports.memory_projection import ProjectionEvent
from app.adapters.inbound.worker.memory_projection_worker import (
    MemoryProjectionWorker,
)
from app.bootstrap.worker import (
    create_memory_projection_processor,
    enqueue_memory_qdrant_projection_backfill_batch,
    enqueue_summary_projection_backfill,
    summary_projection_backfill_status,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _segment(index: int) -> ConversationSummarySegmentMySQL:
    summary_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"summary-{index}"))
    return ConversationSummarySegmentMySQL(
        summary_id=summary_id,
        owner_id="owner-a",
        workspace_id="workspace-a",
        chat_id="chat-a",
        level=0,
        source_start_message_id=index * 2 + 1,
        source_end_message_id=index * 2 + 2,
        source_message_ids=[index * 2 + 1, index * 2 + 2],
        child_summary_ids=[],
        source_content_hash=f"sha256:{index:064x}",
        content_masked=f"summary {index}",
        structured_facts=[],
        open_items=[],
        source_token_count=2,
        summary_token_count=2,
        summary_schema_version="summary-segment-v1",
        extractor_version="summary-extractive-v1",
        projection_version="es:v2|qdrant:v2",
        status="active",
    )


def _memory(index: int) -> ConversationMemoryItemMySQL:
    return ConversationMemoryItemMySQL(
        memory_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"memory-{index}")),
        owner_id="owner-a",
        workspace_id="workspace-a",
        chat_id="chat-a",
        memory_type="query_episode",
        content_masked=f"memory {index}",
        structured_payload={},
        source_message_ids=[index],
        data_source_id="dw-main",
        confidence=1.0,
        verification_status="verified",
        sensitivity="masked",
        extractor_version="test-v1",
        status="active",
    )


@pytest_asyncio.fixture
async def session_factory():
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        yield factory
    finally:
        await engine.dispose()


def test_summary_rollout_defaults_are_all_off():
    assert app_config.memory.summary_enabled is False
    assert app_config.memory.summary_build_enabled is False
    assert app_config.memory.summary_projection_enabled is False
    assert app_config.memory.stable_fact_enabled is False


def test_summary_identity_changes_when_supported_semantic_fact_changes():
    messages = [
        {"id": 1, "role": "user", "status": "success", "content_masked": "各地区销售额"},
        {"id": 2, "role": "assistant", "status": "success", "content_masked": "查询已完成"},
    ]
    first, _ = build_extractive_summary_segment(
        owner_id="owner-a",
        workspace_id="workspace-a",
        chat_id="chat-a",
        messages=messages,
        structured_facts=[{
            "fact_type": "successful_query_semantics",
            "content": "各地区销售额",
            "source_message_ids": [1],
            "status": "supported",
        }],
    )
    second, _ = build_extractive_summary_segment(
        owner_id="owner-a",
        workspace_id="workspace-a",
        chat_id="chat-a",
        messages=messages,
        structured_facts=[{
            "fact_type": "successful_query_semantics",
            "content": "各地区订单数",
            "source_message_ids": [1],
            "status": "supported",
        }],
    )

    assert first["source_content_hash"] != second["source_content_hash"]
    assert first["summary_id"] != second["summary_id"]


@pytest.mark.asyncio
async def test_pending_lower_message_blocks_watermark_until_success(session_factory):
    with (
        patch.object(app_config.memory, "summary_enabled", True),
        patch.object(app_config.memory, "summary_build_enabled", True),
        patch.object(app_config.memory, "summary_projection_enabled", False),
        patch.object(app_config.memory, "summary_turn_high_watermark", 2),
        patch.object(app_config.memory, "summary_turn_low_watermark", 1),
        patch.object(app_config.memory, "summary_recent_protected_turns", 1),
    ):
        settings = load_conversation_settings()[1]
        async with session_factory() as session:
            session.add(ConversationSessionMySQL(
                chat_id="chat-a",
                owner_id="owner-a",
                workspace_id="workspace-a",
                title="测试会话",
                preview="测试会话",
                status="active",
                last_turn_status="pending",
                turn_count=2,
                version=1,
            ))
            repo = ConversationMemoryRepository(
                session, owner_id="owner-a", workspace_id="workspace-a"
            )
            pending = await repo.append_message(
                chat_id="chat-a", role="user", content_masked="第一问",
                request_id="pending", trace_id="pending", status="pending",
            )
            for index in range(4):
                await repo.append_message(
                    chat_id="chat-a", role="assistant" if index % 2 == 0 else "user",
                    content_masked=f"later-{index}", request_id=f"later-{index}",
                    trace_id=f"later-{index}", status="success",
                )
            await session.commit()

            service = ConversationSummaryService(
                owner_id="owner-a",
                workspace_id="workspace-a",
                settings=settings,
            )
            blocked = await service.build_if_needed(
                SqlConversationMemoryUnitOfWork(
                    session,
                    owner_id="owner-a",
                    workspace_id="workspace-a",
                    settings=settings,
                ),
                chat_id="chat-a",
                expected_session_version=1,
            )
            state_repo = ConversationSummaryRepository(
                session, owner_id="owner-a", workspace_id="workspace-a"
            )
            state = await state_repo.get_state("chat-a")
            assert blocked["segments_created"] == 0
            assert state.covered_through_message_id is None

            await repo.set_message_status(int(pending.id), "success")
            await session.commit()
            resumed = await service.build_if_needed(
                SqlConversationMemoryUnitOfWork(
                    session,
                    owner_id="owner-a",
                    workspace_id="workspace-a",
                    settings=settings,
                ),
                chat_id="chat-a",
                expected_session_version=1,
            )
            assert resumed["segments_created"] == 1
            assert resumed["covered_through_message_id"] >= int(pending.id)
            assert resumed["outbox_events_created"] == 0


@pytest.mark.asyncio
async def test_outbox_lease_reclaims_crash_and_disabled_projection_keeps_deletes(
    session_factory,
):
    async with session_factory() as session:
        repo = ConversationMemoryRepository(
            session, owner_id="owner-a", workspace_id="workspace-a"
        )
        await repo.enqueue_projection(
            memory_id=str(uuid.uuid4()), target="summary_es", operation="upsert",
            projection_version="summary-v2",
        )
        deletion = await repo.enqueue_projection(
            memory_id=str(uuid.uuid4()), target="summary_es", operation="delete",
            projection_version="summary-v1",
        )
        await session.commit()

    async with session_factory() as session:
        claimed = await ConversationMemoryRepository.claim_pending_outbox(
            session, limit=10, max_attempts=8, worker_id="worker-a",
            lease_seconds=30, skip_summary_upserts=True,
        )
        assert [row.event_id for row in claimed] == [deletion.event_id]
        assert claimed[0].worker_id == "worker-a"
        await session.execute(
            update(ConversationMemoryOutboxMySQL)
            .where(ConversationMemoryOutboxMySQL.event_id == deletion.event_id)
            .values(lease_until=_utcnow() - timedelta(seconds=1))
        )
        await session.commit()

    async with session_factory() as session:
        reclaimed = await ConversationMemoryRepository.claim_pending_outbox(
            session, limit=10, max_attempts=8, worker_id="worker-b",
            lease_seconds=30, skip_summary_upserts=True,
        )
        assert [row.event_id for row in reclaimed] == [deletion.event_id]
        assert reclaimed[0].worker_id == "worker-b"
        assert reclaimed[0].attempt_count == 2


@pytest.mark.asyncio
async def test_outbox_claim_owner_allowlist_is_scoped_and_default_is_unfiltered(
    session_factory,
):
    event_ids: dict[str, str] = {}
    async with session_factory() as session:
        for owner in ("owner-a", "owner-b"):
            row = await ConversationMemoryRepository(
                session,
                owner_id=owner,
                workspace_id="workspace-a",
            ).enqueue_projection(
                memory_id=str(uuid.uuid4()),
                target="summary_es",
                operation="delete",
                projection_version="summary-v1",
            )
            event_ids[owner] = row.event_id
        await session.commit()

    async with session_factory() as session:
        scoped = await ConversationMemoryRepository.claim_pending_outbox(
            session,
            limit=10,
            max_attempts=8,
            worker_id="scoped-worker",
            owner_allowlist=("owner-a", "owner-a", ""),
        )
    assert [row.event_id for row in scoped] == [event_ids["owner-a"]]

    async with session_factory() as session:
        unfiltered = await ConversationMemoryRepository.claim_pending_outbox(
            session,
            limit=10,
            max_attempts=8,
            worker_id="default-worker",
        )
    assert [row.event_id for row in unfiltered] == [event_ids["owner-b"]]


@pytest.mark.asyncio
async def test_mysql_claim_owner_allowlist_is_parameterized():
    empty_rows = SimpleNamespace(
        scalars=lambda: SimpleNamespace(all=lambda: [])
    )
    session = SimpleNamespace(
        execute=AsyncMock(side_effect=[
            SimpleNamespace(rowcount=0),
            empty_rows,
        ]),
        commit=AsyncMock(),
    )

    rows = await ConversationMemoryRepository._claim_pending_outbox_mysql_once(
        session,
        limit=4,
        max_attempts=8,
        worker_id="mysql-worker",
        lease_seconds=30,
        excluded_targets=(),
        owner_allowlist=("owner-a",),
        skip_summary_upserts=False,
    )

    statement, parameters = session.execute.await_args_list[0].args
    assert rows == []
    assert "owner_id IN (:allowed_owner_0)" in str(statement)
    assert parameters["allowed_owner_0"] == "owner-a"
    session.commit.assert_awaited_once()


def test_projection_worker_uses_unique_bounded_batch_lease_owners():
    worker = MemoryProjectionWorker(
        processor=SimpleNamespace(process_once=AsyncMock()),
        owner_allowlist=("owner-a", "owner-a", "")
    )
    first = worker._new_lease_owner()
    second = worker._new_lease_owner()

    assert worker.owner_allowlist == ("owner-a",)
    assert first != second
    assert len(first) <= 80
    assert len(second) <= 80


@pytest.mark.asyncio
async def test_revoke_has_no_old_200_or_500_row_cap(session_factory):
    async with session_factory() as session:
        session.add_all([_memory(index) for index in range(205)])
        session.add_all([_segment(index) for index in range(505)])
        await session.commit()

        memory_repo = ConversationMemoryRepository(
            session, owner_id="owner-a", workspace_id="workspace-a"
        )
        summary_repo = ConversationSummaryRepository(
            session, owner_id="owner-a", workspace_id="workspace-a"
        )
        memory_ids = await memory_repo.revoke_chat_memories("chat-a")
        summary_ids = await summary_repo.revoke_chat("chat-a")
        await session.commit()

        assert len(memory_ids) == 205
        assert len(summary_ids) == 505
        active_memories = await session.scalar(select(func.count()).select_from(
            ConversationMemoryItemMySQL
        ).where(ConversationMemoryItemMySQL.status == "active"))
        active_summaries = await session.scalar(select(func.count()).select_from(
            ConversationSummarySegmentMySQL
        ).where(ConversationSummarySegmentMySQL.status == "active"))
        assert active_memories == 0
        assert active_summaries == 0


def test_projection_versions_isolate_physical_documents_and_vector_points():
    es_v1 = ConversationSummaryESRepository(
        object(), projection_version="summary-lexical-v1", manage_alias=False
    )
    es_v2 = ConversationSummaryESRepository(
        object(), projection_version="summary-lexical-v2", manage_alias=False
    )
    qdrant_v1 = ConversationSummaryQdrantRepository(
        object(), projection_version="summary-vector-v1"
    )
    qdrant_v2 = ConversationSummaryQdrantRepository(
        object(), projection_version="summary-vector-v2"
    )

    assert es_v1.physical_index != es_v2.physical_index
    assert qdrant_v1._point_id("summary-a") != qdrant_v2._point_id("summary-a")

    memory_es_v1 = ConversationMemoryESRepository(
        object(), projection_version="memory-lexical-v1", manage_alias=False
    )
    memory_es_v2 = ConversationMemoryESRepository(
        object(), projection_version="memory-lexical-v2", manage_alias=False
    )
    memory_qdrant_v1 = ConversationMemoryQdrantRepository(
        object(), projection_version="memory-vector-v1"
    )
    memory_qdrant_v2 = ConversationMemoryQdrantRepository(
        object(), projection_version="memory-vector-v2"
    )
    assert memory_es_v1.physical_index != memory_es_v2.physical_index
    assert (
        memory_qdrant_v1._point_id("memory-a")
        != memory_qdrant_v2._point_id("memory-a")
    )


@pytest.mark.asyncio
async def test_delete_requeues_all_projection_versions_and_ignores_attempt_ceiling(
    session_factory,
):
    memory_id = str(uuid.uuid4())
    async with session_factory() as session:
        repo = ConversationMemoryRepository(
            session, owner_id="owner-a", workspace_id="workspace-a"
        )
        for version in ("summary-v1", "summary-v2"):
            await repo.enqueue_projection(
                memory_id=memory_id,
                target="summary_es",
                operation="upsert",
                projection_version=version,
            )
        deletes = await repo.enqueue_projection_deletes(
            memory_id=memory_id,
            target="summary_es",
            current_projection_version="summary-v3",
        )
        assert {row.projection_version for row in deletes} == {
            "summary-v1", "summary-v2", "summary-v3"
        }
        blocked = deletes[0]
        blocked.status = "failed"
        blocked.attempt_count = 8
        blocked.next_retry_at = None
        await session.commit()

    async with session_factory() as session:
        claimed = await ConversationMemoryRepository.claim_pending_outbox(
            session,
            limit=10,
            max_attempts=8,
            worker_id="delete-worker",
        )
        claimed_by_id = {row.event_id: row for row in claimed}
        assert blocked.event_id in claimed_by_id
        assert claimed_by_id[blocked.event_id].attempt_count == 9


@pytest.mark.asyncio
async def test_outbox_completion_requires_current_lease_owner(session_factory):
    async with session_factory() as session:
        repo = ConversationMemoryRepository(
            session, owner_id="owner-a", workspace_id="workspace-a"
        )
        event = await repo.enqueue_projection(
            memory_id=str(uuid.uuid4()),
            target="summary_es",
            operation="delete",
            projection_version="summary-v1",
        )
        await session.commit()

    async with session_factory() as session:
        claimed = await ConversationMemoryRepository.claim_pending_outbox(
            session, limit=1, max_attempts=8, worker_id="worker-a"
        )
        assert claimed[0].event_id == event.event_id

    async with session_factory() as session:
        assert not await ConversationMemoryRepository.complete_outbox(
            session, event.event_id, worker_id="worker-b"
        )
        current = await session.get(
            ConversationMemoryOutboxMySQL, event.event_id
        )
        assert current.status == "processing"
        assert current.worker_id == "worker-a"
        assert await ConversationMemoryRepository.complete_outbox(
            session, event.event_id, worker_id="worker-a"
        )


@pytest.mark.asyncio
async def test_reset_message_status_blocks_late_turn_completion(session_factory):
    async with session_factory() as session:
        repo = ConversationMemoryRepository(
            session, owner_id="owner-a", workspace_id="workspace-a"
        )
        message = await repo.append_message(
            chat_id="chat-a",
            role="user",
            content_masked="在途请求",
            request_id="trace-a",
            trace_id="trace-a",
            status="pending",
        )
        await repo.reset_working_transcript("chat-a")
        claimed = await repo.set_message_status(
            int(message.id), "success", expected_statuses=("pending",)
        )
        assert claimed is False


@pytest.mark.parametrize(
    ("next_status", "reset_at"),
    (
        ("active", _utcnow()),
        ("deleting", None),
    ),
)
@pytest.mark.asyncio
async def test_old_summary_builder_cannot_cross_session_generation(
    session_factory,
    next_status,
    reset_at,
):
    with (
        patch.object(app_config.memory, "summary_enabled", True),
        patch.object(app_config.memory, "summary_build_enabled", True),
        patch.object(app_config.memory, "summary_projection_enabled", True),
        patch.object(app_config.memory, "summary_turn_high_watermark", 2),
        patch.object(app_config.memory, "summary_turn_low_watermark", 1),
        patch.object(app_config.memory, "summary_recent_protected_turns", 1),
    ):
        settings = load_conversation_settings()[1]
        async with session_factory() as session:
            session.add(ConversationSessionMySQL(
                chat_id="chat-generation",
                owner_id="owner-a",
                workspace_id="workspace-a",
                title="generation fence",
                preview="generation fence",
                status="active",
                last_turn_status="succeeded",
                turn_count=3,
                version=1,
            ))
            repo = ConversationMemoryRepository(
                session, owner_id="owner-a", workspace_id="workspace-a"
            )
            for index in range(3):
                await repo.append_message(
                    chat_id="chat-generation",
                    role="user",
                    content_masked=f"问题 {index}",
                    request_id=f"user-{index}",
                    trace_id=f"trace-{index}",
                    status="success",
                )
                await repo.append_message(
                    chat_id="chat-generation",
                    role="assistant",
                    content_masked="查询成功",
                    request_id=f"assistant-{index}",
                    trace_id=f"trace-{index}",
                    status="success",
                )
            await session.commit()

        # Preserve successful source rows to model the stale RR snapshot held
        # by an old builder; only the authoritative session generation moves.
        async with session_factory() as session:
            await session.execute(
                update(ConversationSessionMySQL)
                .where(ConversationSessionMySQL.chat_id == "chat-generation")
                .values(
                    status=next_status,
                    context_reset_at=reset_at,
                    version=2,
                )
            )
            await session.commit()

        async with session_factory() as session:
            result = await ConversationSummaryService(
                owner_id="owner-a",
                workspace_id="workspace-a",
                settings=settings,
            ).build_if_needed(
                SqlConversationMemoryUnitOfWork(
                    session,
                    owner_id="owner-a",
                    workspace_id="workspace-a",
                    settings=settings,
                ),
                chat_id="chat-generation",
                expected_session_version=1,
            )
            await session.commit()
            segment_count = await session.scalar(
                select(func.count()).select_from(ConversationSummarySegmentMySQL)
            )
            outbox_count = await session.scalar(
                select(func.count()).select_from(ConversationMemoryOutboxMySQL)
            )

        assert result["status"] == "stale_generation"
        assert result["segments_created"] == 0
        assert segment_count == 0
        assert outbox_count == 0


@pytest.mark.asyncio
async def test_sqlite_projection_enqueue_is_idempotent_across_transactions(
    session_factory,
):
    memory_id = str(uuid.uuid4())
    async with session_factory() as session:
        first = await ConversationMemoryRepository(
            session, owner_id="owner-a", workspace_id="workspace-a"
        ).enqueue_projection(
            memory_id=memory_id,
            target="summary_es",
            operation="upsert",
            projection_version="summary-v2",
        )
        first_event_id = first.event_id
        await session.commit()

    async with session_factory() as session:
        duplicate = await ConversationMemoryRepository(
            session, owner_id="owner-a", workspace_id="workspace-a"
        ).enqueue_projection(
            memory_id=memory_id,
            target="summary_es",
            operation="upsert",
            projection_version="summary-v2",
        )
        duplicate.status = "completed"
        duplicate.completed_at = _utcnow()
        await session.commit()
        count = await session.scalar(
            select(func.count()).select_from(ConversationMemoryOutboxMySQL)
        )

    assert duplicate.event_id == first_event_id
    assert count == 1

    async with session_factory() as session:
        requeued = await ConversationMemoryRepository(
            session, owner_id="owner-a", workspace_id="workspace-a"
        ).enqueue_projection(
            memory_id=memory_id,
            target="summary_es",
            operation="upsert",
            projection_version="summary-v2",
            requeue_existing=True,
        )
        await session.commit()

    assert requeued.event_id == first_event_id
    assert requeued.status == "pending"
    assert requeued.completed_at is None


def test_mysql_projection_enqueue_uses_atomic_upsert_and_current_read():
    values = {
        "event_id": str(uuid.uuid4()),
        "memory_id": str(uuid.uuid4()),
        "owner_id": "owner-a",
        "workspace_id": "workspace-a",
        "target": "summary_es",
        "operation": "upsert",
        "projection_version": "summary-v2",
        "status": "pending",
    }
    insert_sql = str(
        _projection_insert_statement("mysql", values).compile(
            dialect=mysql.dialect()
        )
    ).upper()
    lookup_sql = str(
        _projection_lookup_statement(
            memory_id=values["memory_id"],
            target=values["target"],
            operation=values["operation"],
            projection_version=values["projection_version"],
        ).compile(dialect=mysql.dialect())
    ).upper()

    assert "ON DUPLICATE KEY UPDATE" in insert_sql
    assert "EVENT_ID = CONVERSATION_MEMORY_OUTBOX.EVENT_ID" in insert_sql
    assert "FOR UPDATE" in lookup_sql


@pytest.mark.asyncio
async def test_memory_qdrant_version_backfill_is_explicit_and_idempotent(
    session_factory,
):
    legacy_version = "conversation-memory-vector-v1-test"
    current_version = "conversation-memory-vector-v2-test"
    active_with_legacy = _memory(101)
    active_without_event = _memory(102)
    active_with_current = _memory(103)
    revoked = _memory(104)
    revoked.status = "revoked"
    active_with_legacy_processing = _memory(105)

    async with session_factory() as session:
        session.add_all([
            active_with_legacy,
            active_without_event,
            active_with_current,
            revoked,
            active_with_legacy_processing,
        ])
        completed_legacy = await ConversationMemoryRepository(
            session,
            owner_id=active_with_legacy.owner_id,
            workspace_id=active_with_legacy.workspace_id,
        ).enqueue_projection(
            memory_id=active_with_legacy.memory_id,
            target="qdrant",
            operation="upsert",
            projection_version=legacy_version,
        )
        completed_legacy.status = "completed"
        completed_legacy.completed_at = _utcnow()
        completed_current = await ConversationMemoryRepository(
            session,
            owner_id=active_with_current.owner_id,
            workspace_id=active_with_current.workspace_id,
        ).enqueue_projection(
            memory_id=active_with_current.memory_id,
            target="qdrant",
            operation="upsert",
            projection_version=current_version,
        )
        completed_current.status = "completed"
        completed_current.completed_at = _utcnow()
        unrelated_failure = await ConversationMemoryRepository(
            session,
            owner_id=revoked.owner_id,
            workspace_id=revoked.workspace_id,
        ).enqueue_projection(
            memory_id=revoked.memory_id,
            target="qdrant",
            operation="upsert",
            projection_version=legacy_version,
        )
        unrelated_failure.status = "failed"
        unrelated_failure.attempt_count = 8
        unrelated_failure.last_error = "ValueError"
        legacy_processing = await ConversationMemoryRepository(
            session,
            owner_id=active_with_legacy_processing.owner_id,
            workspace_id=active_with_legacy_processing.workspace_id,
        ).enqueue_projection(
            memory_id=active_with_legacy_processing.memory_id,
            target="qdrant",
            operation="upsert",
            projection_version=legacy_version,
        )
        legacy_processing.status = "processing"
        legacy_processing.worker_id = "another-live-worker"
        await session.commit()

    async with session_factory() as session:
        first = await enqueue_memory_qdrant_projection_backfill_batch(
            session,
            projection_version=current_version,
            batch_size=2,
        )
        await session.commit()

    async with session_factory() as session:
        second = await enqueue_memory_qdrant_projection_backfill_batch(
            session,
            projection_version=current_version,
            batch_size=2,
        )
        await session.commit()

    async with session_factory() as session:
        current_rows = list(
            (
                await session.execute(
                    Select(ConversationMemoryOutboxMySQL).where(
                        ConversationMemoryOutboxMySQL.target == "qdrant",
                        ConversationMemoryOutboxMySQL.operation == "upsert",
                        ConversationMemoryOutboxMySQL.projection_version
                        == current_version,
                    )
                )
            )
            .scalars()
            .all()
        )
        legacy_rows = list(
            (
                await session.execute(
                    Select(ConversationMemoryOutboxMySQL).where(
                        ConversationMemoryOutboxMySQL.projection_version
                        == legacy_version
                    )
                )
            )
            .scalars()
            .all()
        )

    assert first == {
        "status": "queued",
        "projection_version": current_version,
        "requested_events": 2,
        "complete": False,
    }
    assert second == {
        "status": "queued",
        "projection_version": current_version,
        "requested_events": 1,
        "complete": True,
    }
    assert len(current_rows) == 4
    assert sum(row.status == "pending" for row in current_rows) == 3
    assert sum(row.status == "completed" for row in current_rows) == 1
    assert {row.memory_id for row in current_rows} == {
        active_with_legacy.memory_id,
        active_without_event.memory_id,
        active_with_current.memory_id,
        active_with_legacy_processing.memory_id,
    }
    assert {row.status for row in legacy_rows} == {
        "completed",
        "failed",
        "processing",
    }

    async with session_factory() as session:
        third = await enqueue_memory_qdrant_projection_backfill_batch(
            session,
            projection_version=current_version,
            batch_size=2,
        )
    assert third == {
        "status": "complete",
        "projection_version": current_version,
        "requested_events": 0,
        "complete": True,
    }


@pytest.mark.asyncio
async def test_summary_projection_backfill_is_complete_and_idempotent(
    session_factory,
):
    async with session_factory() as session:
        active = _segment(1)
        revoked = _segment(2)
        revoked.status = "revoked"
        session.add_all([active, revoked])
        await session.commit()

        first = await enqueue_summary_projection_backfill(
            session, batch_size=1
        )
        second = await enqueue_summary_projection_backfill(
            session, batch_size=1
        )
        await session.commit()
        events = list((await session.execute(
            Select(ConversationMemoryOutboxMySQL).where(
                ConversationMemoryOutboxMySQL.memory_id == active.summary_id,
                ConversationMemoryOutboxMySQL.operation == "upsert",
            )
        )).scalars().all())
        assert first == {"active_segments": 1, "requested_events": 2}
        assert second == first
        assert len(events) == 2
        assert {row.target for row in events} == {
            "summary_es", "summary_qdrant"
        }
        for row in events:
            row.status = "completed"
        await session.commit()
        status = await summary_projection_backfill_status(session)
        assert status["active_segments"] == 1
        assert status["ready"] is True

        restarted = await enqueue_summary_projection_backfill(
            session, batch_size=1
        )
        await session.commit()
        await session.refresh(events[0])
        status = await summary_projection_backfill_status(session)
        assert restarted == first
        assert events[0].status == "pending"
        assert status["ready"] is False


@pytest.mark.asyncio
async def test_summary_reads_wait_for_complete_backfill(session_factory):
    async with session_factory() as session:
        session.add(_segment(1))
        await session.commit()

    summary_es = AsyncMock()
    summary_qdrant = AsyncMock()
    with (
        patch.object(app_config.memory, "summary_enabled", True),
        patch.object(app_config.memory, "summary_projection_enabled", True),
        patch.object(app_config.memory, "summary_retrieval_always", True),
    ):
        settings = load_conversation_settings()[1]
        service = ConversationMemoryService(
            owner_id="owner-a",
            workspace_id="workspace-a",
            persistence=SqlConversationMemoryPersistence(
                session_factory,
                owner_id="owner-a",
                workspace_id="workspace-a",
                settings=settings,
            ),
            embedding_client=AsyncMock(),
            es_repository=AsyncMock(),
            qdrant_repository=AsyncMock(),
            summary_es_repository=summary_es,
            summary_qdrant_repository=summary_qdrant,
            settings=settings,
        )
        _, _, evidence, _ = await service._load_summary_context(
            chat_id="chat-a",
            query="之前的销售问题",
            recent_message_ids=set(),
            omitted_recent=5,
        )

    summary_es.search.assert_not_awaited()
    summary_qdrant.search.assert_not_awaited()
    assert evidence["es"]["status"] == "backfill_pending"
    assert evidence["qdrant"]["status"] == "backfill_pending"
    assert evidence["projection_backfill"]["ready"] is False


@pytest.mark.asyncio
async def test_projection_upsert_is_compensated_after_authority_revocation(
    session_factory,
):
    segment = _segment(1)
    async with session_factory() as session:
        session.add(segment)
        event = await ConversationMemoryRepository(
            session, owner_id="owner-a", workspace_id="workspace-a"
        ).enqueue_projection(
            memory_id=segment.summary_id,
            target="summary_es",
            operation="upsert",
            projection_version="summary-v2",
        )
        await session.commit()

    async with session_factory() as session:
        await ConversationSummaryRepository(
            session, owner_id="owner-a", workspace_id="workspace-a"
        ).revoke_chat("chat-a")
        await session.commit()

    projection_event = ProjectionEvent(
        event_id=str(event.event_id),
        memory_id=str(event.memory_id),
        owner_id=str(event.owner_id),
        workspace_id=str(event.workspace_id),
        target=str(event.target),
        operation=str(event.operation),
        projection_version=str(event.projection_version),
        attempt_count=int(event.attempt_count or 0),
    )
    delivery = AsyncMock()
    with patch.object(
        meta_mysql_client_manager, "session_factory", session_factory
    ):
        processor = create_memory_projection_processor(delivery=delivery)
        compensated = await processor.compensate_stale_upsert(
            projection_event,
            projected_active=True,
        )

    assert compensated is True
    delivery.delete.assert_awaited_once_with(projection_event)


@pytest.mark.asyncio
async def test_summary_es_search_does_not_create_or_read_unpromoted_alias():
    indices = SimpleNamespace(
        exists=AsyncMock(return_value=True),
        exists_alias=AsyncMock(return_value=False),
        get_alias=AsyncMock(),
    )
    client = SimpleNamespace(
        indices=indices,
        count=AsyncMock(return_value={"count": 0}),
        search=AsyncMock(),
    )
    repository = ConversationSummaryESRepository(
        client,
        projection_version="summary-v2",
        manage_alias=False,
    )

    identifiers, evidence = await repository.search(
        "此前销售额",
        owner_id="owner-a",
        workspace_id="workspace-a",
        chat_id="chat-a",
        limit=5,
    )

    assert identifiers == []
    assert evidence["status"] == "backfill_pending"
    client.search.assert_not_awaited()
