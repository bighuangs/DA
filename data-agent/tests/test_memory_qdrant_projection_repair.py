"""Qdrant conversation-memory 投影迁移与恢复合同。"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from qdrant_client import models
from sqlalchemy import Select, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.application.ports.memory_projection import ProjectionEvent
from app.bootstrap import worker as worker_bootstrap
from app.domain.retrieval.hybrid_embedding import HybridEmbedding
from app.infrastructure.persistence.mysql.records.base import Base
from app.infrastructure.persistence.mysql.records.conversation_memory_item_mysql import (
    ConversationMemoryItemMySQL,
)
from app.infrastructure.persistence.mysql.records.conversation_memory_outbox_mysql import (
    ConversationMemoryOutboxMySQL,
)
from app.infrastructure.persistence.mysql.repositories.conversation_memory_repository import (
    ConversationMemoryRepository,
)
from app.infrastructure.search.qdrant.repositories.conversation_memory_qdrant_repository import (
    ConversationMemoryQdrantRepository,
)
from app.infrastructure.search.qdrant.repositories.conversation_summary_qdrant_repository import (
    ConversationSummaryQdrantRepository,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _memory(index: int, *, status: str = "active") -> ConversationMemoryItemMySQL:
    return ConversationMemoryItemMySQL(
        memory_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"qdrant-repair-{index}")),
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
        status=status,
    )


def _event(memory_id: str) -> ProjectionEvent:
    return ProjectionEvent(
        event_id=str(uuid.uuid4()),
        memory_id=memory_id,
        owner_id="owner-a",
        workspace_id="workspace-a",
        target="qdrant",
        operation="upsert",
        projection_version="conversation-memory-vector-v2-test",
        attempt_count=1,
    )


@pytest_asyncio.fixture
async def memory_session_factory():
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


@pytest.mark.asyncio
async def test_memory_collection_indexes_both_projection_version_fields() -> None:
    client = SimpleNamespace(create_payload_index=AsyncMock())
    repository = ConversationMemoryQdrantRepository(client)

    with (
        patch(
            "app.infrastructure.search.qdrant.repositories."
            "conversation_memory_qdrant_repository.create_hybrid_collection",
            new=AsyncMock(),
        ),
        patch.object(repository, "status", new=AsyncMock(return_value={})),
    ):
        await repository.ensure_collection()

    indexes = {
        call.kwargs["field_name"]: call.kwargs["field_schema"]
        for call in client.create_payload_index.await_args_list
    }
    assert indexes["projection_version"] == models.PayloadSchemaType.KEYWORD
    assert indexes["index_version"] == models.PayloadSchemaType.KEYWORD


@pytest.mark.asyncio
async def test_summary_collection_indexes_both_projection_version_fields() -> None:
    client = SimpleNamespace(create_payload_index=AsyncMock())
    repository = ConversationSummaryQdrantRepository(client)

    with (
        patch(
            "app.infrastructure.search.qdrant.repositories."
            "conversation_summary_qdrant_repository.create_hybrid_collection",
            new=AsyncMock(),
        ),
        patch.object(repository, "status", new=AsyncMock(return_value={})),
    ):
        await repository.ensure_collection()

    indexes = {
        call.kwargs["field_name"]: call.kwargs["field_schema"]
        for call in client.create_payload_index.await_args_list
    }
    assert indexes["projection_version"] == models.PayloadSchemaType.KEYWORD
    assert indexes["index_version"] == models.PayloadSchemaType.KEYWORD


@pytest.mark.asyncio
async def test_memory_search_requires_projection_and_index_version() -> None:
    version = "conversation-memory-vector-v2-test"
    repository = ConversationMemoryQdrantRepository(
        object(), projection_version=version
    )
    repository.ensure_collection = AsyncMock(return_value={})
    result = SimpleNamespace(items=[], evidence={}, degradations=[])
    embedding = HybridEmbedding(
        dense=(1.0,), sparse_indices=(1,), sparse_values=(1.0,)
    )

    with patch(
        "app.infrastructure.search.qdrant.repositories."
        "conversation_memory_qdrant_repository.search_hybrid_entities",
        new=AsyncMock(return_value=result),
    ) as search:
        await repository.search(
            embedding,
            query="地区销售额",
            owner_id="owner-a",
            workspace_id="workspace-a",
            limit=5,
        )

    conditions = {
        condition.key: condition.match
        for condition in search.await_args.kwargs["query_filter"].must
    }
    assert conditions["projection_version"].value == version
    assert conditions["index_version"].value == version


@pytest.mark.asyncio
async def test_summary_search_requires_projection_and_index_version() -> None:
    version = "conversation-summary-vector-v2-test"
    repository = ConversationSummaryQdrantRepository(
        object(), projection_version=version
    )
    repository.ensure_collection = AsyncMock(return_value={})
    result = SimpleNamespace(items=[], evidence={}, degradations=[])
    embedding = HybridEmbedding(
        dense=(1.0,), sparse_indices=(1,), sparse_values=(1.0,)
    )

    with patch(
        "app.infrastructure.search.qdrant.repositories."
        "conversation_summary_qdrant_repository.search_hybrid_entities",
        new=AsyncMock(return_value=result),
    ) as search:
        await repository.search(
            embedding,
            query="月度摘要",
            owner_id="owner-a",
            workspace_id="workspace-a",
            chat_id="chat-a",
            limit=5,
        )

    conditions = {
        condition.key: condition.match
        for condition in search.await_args.kwargs["query_filter"].must
    }
    assert conditions["projection_version"].value == version
    assert conditions["index_version"].value == version


@pytest.mark.asyncio
async def test_unrelated_value_error_is_not_a_version_migration_signal(
    memory_session_factory,
) -> None:
    version = "conversation-memory-vector-v2-test"
    memory = _memory(1)
    async with memory_session_factory() as session:
        session.add(memory)
        await session.flush()
        event = await ConversationMemoryRepository(
            session,
            owner_id=memory.owner_id,
            workspace_id=memory.workspace_id,
        ).enqueue_projection(
            memory_id=memory.memory_id,
            target="qdrant",
            operation="upsert",
            projection_version=version,
        )
        event.status = "failed"
        event.attempt_count = 8
        event.last_error = "ValueError"
        original_event_id = event.event_id
        await session.commit()

    async with memory_session_factory() as session:
        result = (
            await worker_bootstrap.enqueue_memory_qdrant_projection_backfill_batch(
                session,
                projection_version=version,
                batch_size=1,
            )
        )
        await session.commit()
        rows = list(
            (
                await session.execute(
                    Select(ConversationMemoryOutboxMySQL).where(
                        ConversationMemoryOutboxMySQL.memory_id
                        == memory.memory_id,
                        ConversationMemoryOutboxMySQL.target == "qdrant",
                        ConversationMemoryOutboxMySQL.projection_version == version,
                    )
                )
            )
            .scalars()
            .all()
        )

    assert result["requested_events"] == 0
    assert result["complete"] is True
    assert len(rows) == 1
    assert rows[0].event_id == original_event_id
    assert rows[0].status == "failed"
    assert rows[0].attempt_count == 8
    assert rows[0].last_error == "ValueError"


@pytest.mark.asyncio
async def test_two_workers_are_disjoint_and_expired_processing_is_fenced(
    memory_session_factory,
) -> None:
    version = "conversation-memory-vector-v2-test"
    async with memory_session_factory() as session:
        event_ids = []
        for index in (10, 11):
            row = await ConversationMemoryRepository(
                session,
                owner_id="owner-a",
                workspace_id="workspace-a",
            ).enqueue_projection(
                memory_id=_memory(index).memory_id,
                target="qdrant",
                operation="upsert",
                projection_version=version,
            )
            event_ids.append(row.event_id)
        await session.commit()

    async with memory_session_factory() as session:
        worker_a = await ConversationMemoryRepository.claim_pending_outbox(
            session,
            limit=1,
            max_attempts=8,
            worker_id="worker-a",
            lease_seconds=30,
        )
    async with memory_session_factory() as session:
        worker_b = await ConversationMemoryRepository.claim_pending_outbox(
            session,
            limit=10,
            max_attempts=8,
            worker_id="worker-b",
            lease_seconds=30,
        )

    assert len(worker_a) == 1
    assert len(worker_b) == 1
    assert {worker_a[0].event_id, worker_b[0].event_id} == set(event_ids)
    assert worker_a[0].event_id != worker_b[0].event_id

    async with memory_session_factory() as session:
        assert await ConversationMemoryRepository.complete_outbox(
            session,
            worker_b[0].event_id,
            worker_id="worker-b",
        )
        await session.execute(
            update(ConversationMemoryOutboxMySQL)
            .where(
                ConversationMemoryOutboxMySQL.event_id == worker_a[0].event_id
            )
            .values(lease_until=_utcnow() - timedelta(seconds=1))
        )
        await session.commit()

    async with memory_session_factory() as session:
        worker_c = await ConversationMemoryRepository.claim_pending_outbox(
            session,
            limit=10,
            max_attempts=8,
            worker_id="worker-c",
            lease_seconds=30,
        )
    assert [row.event_id for row in worker_c] == [worker_a[0].event_id]
    assert worker_c[0].attempt_count == 2

    async with memory_session_factory() as session:
        assert not await ConversationMemoryRepository.renew_outbox_lease(
            session,
            worker_a[0].event_id,
            worker_id="worker-a",
            lease_seconds=30,
        )
        assert not await ConversationMemoryRepository.complete_outbox(
            session,
            worker_a[0].event_id,
            worker_id="worker-a",
        )
        assert not await ConversationMemoryRepository.fail_outbox(
            session,
            worker_a[0].event_id,
            "RuntimeError",
            worker_a[0].attempt_count,
            worker_id="worker-a",
        )
        assert await ConversationMemoryRepository.complete_outbox(
            session,
            worker_a[0].event_id,
            worker_id="worker-c",
        )
        states = list(
            (
                await session.execute(
                    select(ConversationMemoryOutboxMySQL.status).where(
                        ConversationMemoryOutboxMySQL.event_id.in_(event_ids)
                    )
                )
            )
            .scalars()
            .all()
        )
    assert states == ["completed", "completed"]


@pytest.mark.asyncio
async def test_delivery_routes_active_revoked_and_missing_authority_without_qdrant(
    memory_session_factory,
) -> None:
    active = _memory(20)
    revoked = _memory(21, status="revoked")
    async with memory_session_factory() as session:
        session.add_all([active, revoked])
        await session.commit()

    delivery = worker_bootstrap._SearchMemoryProjectionDelivery(
        memory_session_factory
    )
    target = SimpleNamespace(upsert=AsyncMock(), delete=AsyncMock())
    embedding = HybridEmbedding(
        dense=(1.0,), sparse_indices=(1,), sparse_values=(1.0,)
    )
    embedder = SimpleNamespace(
        aembed_documents=AsyncMock(return_value=[embedding])
    )
    missing_id = str(uuid.uuid4())

    with (
        patch.object(delivery, "_target", return_value=target),
        patch.object(worker_bootstrap.embedding_client_manager, "client", embedder),
    ):
        active_result = await delivery.deliver(_event(active.memory_id))
        revoked_result = await delivery.deliver(_event(revoked.memory_id))
        missing_result = await delivery.deliver(_event(missing_id))

    assert active_result is True
    assert revoked_result is False
    assert missing_result is False
    embedder.aembed_documents.assert_awaited_once_with([active.content_masked])
    projected_item, projected_embedding = target.upsert.await_args.args
    assert projected_item.memory_id == active.memory_id
    assert projected_embedding == embedding
    assert [call.args[0] for call in target.delete.await_args_list] == [
        revoked.memory_id,
        missing_id,
    ]
