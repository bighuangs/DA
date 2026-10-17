"""MySQL authority repository for transcript, checkpoints, memories and outbox."""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

from sqlalchemy import (
    Select,
    and_,
    delete,
    desc,
    func,
    insert,
    not_,
    or_,
    select,
    text,
    update,
)
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.persistence.mysql.records.conversation_memory_item_mysql import ConversationMemoryItemMySQL
from app.infrastructure.persistence.mysql.records.conversation_memory_outbox_mysql import ConversationMemoryOutboxMySQL
from app.infrastructure.persistence.mysql.records.conversation_message_mysql import ConversationMessageMySQL
from app.infrastructure.persistence.mysql.records.conversation_working_memory_checkpoint_mysql import (
    ConversationWorkingMemoryCheckpointMySQL,
)


def _required(value: str | None, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"conversation memory repository 必须绑定 {name}")
    return text


def _utcnow() -> datetime:
    """Return naive UTC for the existing MySQL DATETIME column contract."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _active_memory_predicates(now: datetime) -> tuple:
    return (
        ConversationMemoryItemMySQL.status == "active",
        ConversationMemoryItemMySQL.verification_status.in_(
            ("verified", "confirmed", "executed")
        ),
        or_(
            ConversationMemoryItemMySQL.valid_to.is_(None),
            ConversationMemoryItemMySQL.valid_to > now,
        ),
        or_(
            ConversationMemoryItemMySQL.expires_at.is_(None),
            ConversationMemoryItemMySQL.expires_at > now,
        ),
        ConversationMemoryItemMySQL.deleted_at.is_(None),
    )


def _projection_insert_statement(dialect_name: str, values: dict):
    """Build an atomic idempotency insert for the active SQL dialect."""
    if dialect_name == "mysql":
        statement = mysql_insert(ConversationMemoryOutboxMySQL).values(**values)
        return statement.on_duplicate_key_update(
            event_id=ConversationMemoryOutboxMySQL.event_id
        )
    if dialect_name == "sqlite":
        statement = sqlite_insert(ConversationMemoryOutboxMySQL).values(**values)
        return statement.on_conflict_do_nothing(
            index_elements=(
                "memory_id",
                "target",
                "operation",
                "projection_version",
            )
        )
    return insert(ConversationMemoryOutboxMySQL).values(**values)


def _projection_lookup_statement(
    *, memory_id: str, target: str, operation: str, projection_version: str
):
    """Use a locking current read after an atomic insert/upsert."""
    return Select(ConversationMemoryOutboxMySQL).where(
        ConversationMemoryOutboxMySQL.memory_id == memory_id,
        ConversationMemoryOutboxMySQL.target == target,
        ConversationMemoryOutboxMySQL.operation == operation,
        ConversationMemoryOutboxMySQL.projection_version == projection_version,
    ).with_for_update().execution_options(populate_existing=True)


class ConversationMemoryRepository:
    """Every public memory query is hard-bound to owner + workspace."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        owner_id: str,
        workspace_id: str,
    ):
        self.session = session
        self.owner_id = _required(owner_id, "owner_id")
        self.workspace_id = _required(workspace_id, "workspace_id")

    @property
    def scope_predicates(self) -> tuple:
        return (
            ConversationMemoryItemMySQL.owner_id == self.owner_id,
            ConversationMemoryItemMySQL.workspace_id == self.workspace_id,
        )

    async def append_message(
        self,
        *,
        chat_id: str,
        role: str,
        content_masked: str,
        request_id: str | None,
        trace_id: str | None,
        status: str,
    ) -> ConversationMessageMySQL:
        if request_id:
            existing = await self.session.execute(
                Select(ConversationMessageMySQL).where(
                    ConversationMessageMySQL.owner_id == self.owner_id,
                    ConversationMessageMySQL.workspace_id == self.workspace_id,
                    ConversationMessageMySQL.request_id == request_id,
                    ConversationMessageMySQL.role == role,
                )
            )
            row = existing.scalar_one_or_none()
            if row is not None:
                return row
        row = ConversationMessageMySQL(
            owner_id=self.owner_id,
            workspace_id=self.workspace_id,
            chat_id=str(chat_id)[:64],
            role=str(role)[:16],
            content_masked=content_masked,
            request_id=(str(request_id)[:64] if request_id else None),
            trace_id=(str(trace_id)[:64] if trace_id else None),
            status=str(status)[:24],
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def set_message_status(
        self,
        message_id: int,
        status: str,
        *,
        expected_statuses: Sequence[str] | None = None,
    ) -> bool:
        statement = update(ConversationMessageMySQL).where(
                ConversationMessageMySQL.id == int(message_id),
                ConversationMessageMySQL.owner_id == self.owner_id,
                ConversationMessageMySQL.workspace_id == self.workspace_id,
            )
        if expected_statuses:
            statement = statement.where(
                ConversationMessageMySQL.status.in_(tuple(expected_statuses))
            )
        result = await self.session.execute(
            statement.values(status=str(status)[:24])
        )
        return bool(result.rowcount)

    async def recent_messages(
        self, chat_id: str, limit: int
    ) -> list[ConversationMessageMySQL]:
        result = await self.session.execute(
            Select(ConversationMessageMySQL).where(
                ConversationMessageMySQL.owner_id == self.owner_id,
                ConversationMessageMySQL.workspace_id == self.workspace_id,
                ConversationMessageMySQL.chat_id == chat_id,
                ConversationMessageMySQL.deleted_at.is_(None),
                ConversationMessageMySQL.status.in_(("success", "pending")),
            ).order_by(
                desc(ConversationMessageMySQL.created_at),
                desc(ConversationMessageMySQL.id),
            ).limit(max(1, min(int(limit), 100)))
        )
        return list(reversed(result.scalars().all()))

    async def get_checkpoint(
        self,
        chat_id: str,
        *,
        for_update: bool = False,
        include_expired: bool = False,
    ) -> ConversationWorkingMemoryCheckpointMySQL | None:
        statement = Select(ConversationWorkingMemoryCheckpointMySQL).where(
                ConversationWorkingMemoryCheckpointMySQL.owner_id == self.owner_id,
                ConversationWorkingMemoryCheckpointMySQL.workspace_id == self.workspace_id,
                ConversationWorkingMemoryCheckpointMySQL.chat_id == chat_id,
            )
        if not include_expired:
            statement = statement.where(or_(
                ConversationWorkingMemoryCheckpointMySQL.expires_at.is_(None),
                ConversationWorkingMemoryCheckpointMySQL.expires_at > _utcnow(),
            ))
        if for_update:
            statement = statement.with_for_update()
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def upsert_checkpoint(
        self,
        *,
        chat_id: str,
        version: int,
        structured_state: dict,
        rolling_summary: str,
        recent_message_ids: Sequence[int],
        schema_version: str,
        ttl_seconds: int,
    ) -> ConversationWorkingMemoryCheckpointMySQL:
        row = await self.get_checkpoint(chat_id, include_expired=True)
        if row is None:
            row = ConversationWorkingMemoryCheckpointMySQL(
                owner_id=self.owner_id,
                workspace_id=self.workspace_id,
                chat_id=chat_id,
                version=max(0, int(version)),
                structured_state=dict(structured_state or {}),
                rolling_summary=str(rolling_summary or ""),
                recent_message_ids=[int(value) for value in recent_message_ids],
                schema_version=str(schema_version),
            )
            self.session.add(row)
        else:
            row.version = max(int(row.version or 0) + 1, int(version))
            row.structured_state = dict(structured_state or {})
            row.rolling_summary = str(rolling_summary or "")
            row.recent_message_ids = [int(value) for value in recent_message_ids]
            row.schema_version = str(schema_version)
        row.expires_at = _utcnow() + timedelta(seconds=max(60, int(ttl_seconds)))
        await self.session.flush()
        return row

    async def delete_checkpoint(self, chat_id: str) -> int:
        result = await self.session.execute(
            delete(ConversationWorkingMemoryCheckpointMySQL).where(
                ConversationWorkingMemoryCheckpointMySQL.owner_id == self.owner_id,
                ConversationWorkingMemoryCheckpointMySQL.workspace_id == self.workspace_id,
                ConversationWorkingMemoryCheckpointMySQL.chat_id == chat_id,
            )
        )
        return int(result.rowcount or 0)

    async def reset_working_transcript(self, chat_id: str) -> dict[str, int]:
        """Clear Agent inheritance while retaining authoritative fact sources."""
        messages = await self.session.execute(
            update(ConversationMessageMySQL).where(
                ConversationMessageMySQL.owner_id == self.owner_id,
                ConversationMessageMySQL.workspace_id == self.workspace_id,
                ConversationMessageMySQL.chat_id == chat_id,
                ConversationMessageMySQL.deleted_at.is_(None),
            ).values(status="reset")
        )
        return {
            "messages": int(messages.rowcount or 0),
            # chat_history 是可回放快照；上下文重置由 conversation_session
            # context_reset_at 截断，不再混同“删除会话”。
            "history": 0,
        }

    async def upsert_memory_item(self, values: dict) -> tuple[ConversationMemoryItemMySQL, bool]:
        memory_id = _required(values.get("memory_id"), "memory_id")
        row = await self.session.get(ConversationMemoryItemMySQL, memory_id)
        if row is not None:
            if row.owner_id != self.owner_id or row.workspace_id != self.workspace_id:
                raise PermissionError("memory_id 已属于其他 owner/workspace")
            return row, False
        row = ConversationMemoryItemMySQL(
            **{
                **values,
                "memory_id": memory_id,
                "owner_id": self.owner_id,
                "workspace_id": self.workspace_id,
            }
        )
        self.session.add(row)
        await self.session.flush()
        return row, True

    async def supersede_active_preferences(
        self,
        *,
        preference_slot: str,
        superseded_by_id: str,
    ) -> list[str]:
        """Close older preferences in the same known slot under MySQL authority."""
        slot = str(preference_slot or "").strip()
        if not slot or slot == "independent":
            return []
        current = await self.session.get(
            ConversationMemoryItemMySQL, superseded_by_id
        )
        if current is None:
            raise ValueError("用于 supersede 的新 preference 不存在")
        if current.owner_id != self.owner_id or current.workspace_id != self.workspace_id:
            raise PermissionError("用于 supersede 的 preference scope 不一致")
        candidates = await self.list_active_candidates(
            memory_types=("preference",), limit=200
        )
        old_rows = [
            row
            for row in candidates
            if row.memory_id != superseded_by_id
            and str((row.structured_payload or {}).get("preference_slot") or "")
            == slot
        ]
        identifiers = [row.memory_id for row in old_rows]
        if not identifiers:
            return []
        current.supersedes_id = identifiers[0]
        await self.session.execute(
            update(ConversationMemoryItemMySQL).where(
                ConversationMemoryItemMySQL.memory_id.in_(identifiers),
                *self.scope_predicates,
            ).values(status="superseded", valid_to=_utcnow())
        )
        await self.session.flush()
        return identifiers

    async def supersede_active_fact_slot(
        self,
        *,
        memory_type: str,
        fact_slot: str,
        superseded_by_id: str,
    ) -> list[str]:
        """Replace older explicit profile/goal facts in one governed slot."""
        kind = str(memory_type or "").strip()
        slot = str(fact_slot or "").strip()
        if kind not in {"profile_fact", "long_term_goal"} or not slot:
            return []
        current = await self.session.get(
            ConversationMemoryItemMySQL, str(superseded_by_id)
        )
        if current is None:
            raise ValueError("用于 supersede 的新 stable fact 不存在")
        if current.owner_id != self.owner_id or current.workspace_id != self.workspace_id:
            raise PermissionError("用于 supersede 的 stable fact scope 不一致")
        candidates = await self.list_active_candidates(
            memory_types=(kind,), limit=200
        )
        old_rows = [
            row for row in candidates
            if row.memory_id != superseded_by_id
            and str((row.structured_payload or {}).get("fact_slot") or "") == slot
        ]
        identifiers = [row.memory_id for row in old_rows]
        if not identifiers:
            return []
        current.supersedes_id = identifiers[0]
        await self.session.execute(
            update(ConversationMemoryItemMySQL).where(
                ConversationMemoryItemMySQL.memory_id.in_(identifiers),
                *self.scope_predicates,
            ).values(status="superseded", valid_to=_utcnow())
        )
        await self.session.flush()
        return identifiers

    async def enqueue_projection(
        self,
        *,
        memory_id: str,
        target: str,
        operation: str,
        projection_version: str,
        requeue_existing: bool = False,
    ) -> ConversationMemoryOutboxMySQL:
        values = {
            "event_id": str(uuid.uuid4()),
            "memory_id": memory_id,
            "owner_id": self.owner_id,
            "workspace_id": self.workspace_id,
            "target": target,
            "operation": operation,
            "projection_version": projection_version,
            "status": "pending",
        }
        dialect_name = self.session.get_bind().dialect.name
        statement = _projection_insert_statement(dialect_name, values)
        if dialect_name in {"mysql", "sqlite"}:
            await self.session.execute(statement)
        else:
            # Other development/test dialects keep a savepoint fallback.  The
            # post-write lookup is a locking current read, so a duplicate that
            # committed after an earlier snapshot remains visible.
            try:
                async with self.session.begin_nested():
                    await self.session.execute(statement)
            except IntegrityError:
                pass
        current = await self.session.execute(
            _projection_lookup_statement(
                memory_id=memory_id,
                target=target,
                operation=operation,
                projection_version=projection_version,
            )
        )
        row = current.scalar_one_or_none()
        if row is None:
            raise RuntimeError("projection event 原子写入后不可见")
        if row.owner_id != self.owner_id or row.workspace_id != self.workspace_id:
            raise PermissionError("projection event 已属于其他 owner/workspace")
        # Delete delivery is security-sensitive and is always requeued.  A
        # repair/backfill may explicitly requeue an existing upsert after its
        # physical projection was lost.  Never revoke a live worker's lease.
        should_requeue = operation == "delete" or (
            operation == "upsert" and bool(requeue_existing)
        )
        if should_requeue and row.status != "processing":
            row.status = "pending"
            row.attempt_count = 0
            row.next_retry_at = None
            row.claimed_at = None
            row.lease_until = None
            row.worker_id = None
            row.last_error = None
            row.completed_at = None
        return row

    async def enqueue_projection_deletes(
        self,
        *,
        memory_id: str,
        target: str,
        current_projection_version: str,
    ) -> list[ConversationMemoryOutboxMySQL]:
        """Delete every version ever projected for one authoritative item."""
        result = await self.session.execute(
            select(ConversationMemoryOutboxMySQL.projection_version).where(
                ConversationMemoryOutboxMySQL.memory_id == str(memory_id),
                ConversationMemoryOutboxMySQL.target == str(target),
                ConversationMemoryOutboxMySQL.owner_id == self.owner_id,
                ConversationMemoryOutboxMySQL.workspace_id == self.workspace_id,
            ).distinct()
        )
        versions = {
            str(value) for value in result.scalars().all() if str(value or "")
        }
        versions.add(str(current_projection_version))
        rows: list[ConversationMemoryOutboxMySQL] = []
        for version in sorted(versions):
            rows.append(await self.enqueue_projection(
                memory_id=memory_id,
                target=target,
                operation="delete",
                projection_version=version,
            ))
        return rows

    async def list_active_candidates(
        self,
        *,
        memory_types: Sequence[str] | None = None,
        limit: int = 50,
    ) -> list[ConversationMemoryItemMySQL]:
        statement = Select(ConversationMemoryItemMySQL).where(
            *self.scope_predicates,
            *_active_memory_predicates(_utcnow()),
        )
        if memory_types:
            statement = statement.where(
                ConversationMemoryItemMySQL.memory_type.in_(tuple(memory_types))
            )
        result = await self.session.execute(
            statement.order_by(
                desc(ConversationMemoryItemMySQL.updated_at),
                desc(ConversationMemoryItemMySQL.memory_id),
            ).limit(max(1, min(int(limit), 200)))
        )
        return list(result.scalars().all())

    async def list_authoritative_by_ids(
        self, memory_ids: Iterable[str]
    ) -> list[ConversationMemoryItemMySQL]:
        ids = list(dict.fromkeys(str(value) for value in memory_ids if value))[:100]
        if not ids:
            return []
        result = await self.session.execute(
            Select(ConversationMemoryItemMySQL).where(
                ConversationMemoryItemMySQL.memory_id.in_(ids),
                *self.scope_predicates,
                *_active_memory_predicates(_utcnow()),
            )
        )
        by_id = {row.memory_id: row for row in result.scalars().all()}
        return [by_id[value] for value in ids if value in by_id]

    async def list_chat_memories(
        self, chat_id: str, *, limit: int = 100
    ) -> list[ConversationMemoryItemMySQL]:
        result = await self.session.execute(
            Select(ConversationMemoryItemMySQL).where(
                *self.scope_predicates,
                ConversationMemoryItemMySQL.chat_id == chat_id,
                ConversationMemoryItemMySQL.deleted_at.is_(None),
            ).order_by(
                desc(ConversationMemoryItemMySQL.updated_at),
                desc(ConversationMemoryItemMySQL.memory_id),
            ).limit(max(1, min(int(limit), 200)))
        )
        return list(result.scalars().all())

    async def revoke_chat_memories(self, chat_id: str) -> list[str]:
        result = await self.session.execute(
            select(ConversationMemoryItemMySQL.memory_id).where(
                *self.scope_predicates,
                ConversationMemoryItemMySQL.chat_id == str(chat_id),
                ConversationMemoryItemMySQL.status == "active",
                ConversationMemoryItemMySQL.deleted_at.is_(None),
            )
        )
        identifiers = [str(value) for value in result.scalars().all()]
        if identifiers:
            await self.session.execute(
                update(ConversationMemoryItemMySQL).where(
                    ConversationMemoryItemMySQL.memory_id.in_(identifiers),
                    *self.scope_predicates,
                ).values(
                    status="revoked",
                    valid_to=_utcnow(),
                )
            )
        return identifiers

    @staticmethod
    async def _claim_pending_outbox_once(
        session: AsyncSession,
        *,
        limit: int,
        max_attempts: int,
        worker_id: str = "memory-projection-worker",
        lease_seconds: int = 90,
        excluded_targets: Sequence[str] = (),
        owner_allowlist: Sequence[str] = (),
        skip_summary_upserts: bool = False,
    ) -> list[ConversationMemoryOutboxMySQL]:
        now = _utcnow()
        predicates = [
            or_(
                ConversationMemoryOutboxMySQL.status.in_(("pending", "failed")),
                (
                    (ConversationMemoryOutboxMySQL.status == "processing")
                    & (
                        ConversationMemoryOutboxMySQL.lease_until.is_(None)
                        | (ConversationMemoryOutboxMySQL.lease_until <= now)
                    )
                ),
            ),
            or_(
                ConversationMemoryOutboxMySQL.operation == "delete",
                ConversationMemoryOutboxMySQL.attempt_count
                < max(1, int(max_attempts)),
            ),
            or_(
                ConversationMemoryOutboxMySQL.next_retry_at.is_(None),
                ConversationMemoryOutboxMySQL.next_retry_at <= now,
            ),
        ]
        excluded = tuple(
            dict.fromkeys(str(value) for value in excluded_targets if value)
        )
        if excluded:
            predicates.append(
                ConversationMemoryOutboxMySQL.target.not_in(excluded)
            )
        owners = tuple(dict.fromkeys(
            str(value) for value in owner_allowlist if str(value or "").strip()
        ))
        if owners:
            predicates.append(
                ConversationMemoryOutboxMySQL.owner_id.in_(owners)
            )
        if skip_summary_upserts:
            predicates.append(not_(and_(
                ConversationMemoryOutboxMySQL.target.in_((
                    "summary_es", "summary_qdrant",
                )),
                ConversationMemoryOutboxMySQL.operation == "upsert",
            )))
        result = await session.execute(
            Select(ConversationMemoryOutboxMySQL).where(*predicates).order_by(
                ConversationMemoryOutboxMySQL.created_at,
                ConversationMemoryOutboxMySQL.event_id,
            ).limit(max(1, min(int(limit), 100))).with_for_update(
                skip_locked=True
            )
        )
        rows = list(result.scalars().all())
        for row in rows:
            row.status = "processing"
            row.attempt_count = int(row.attempt_count or 0) + 1
            row.claimed_at = now
            row.lease_until = now + timedelta(
                seconds=max(30, int(lease_seconds))
            )
            row.worker_id = str(worker_id or "memory-projection-worker")[:80]
        await session.commit()
        return rows

    @staticmethod
    async def _claim_pending_outbox_mysql_once(
        session: AsyncSession,
        *,
        limit: int,
        max_attempts: int,
        worker_id: str,
        lease_seconds: int,
        excluded_targets: Sequence[str],
        owner_allowlist: Sequence[str],
        skip_summary_upserts: bool,
    ) -> list[ConversationMemoryOutboxMySQL]:
        """Atomically claim a MySQL batch without a locking range read.

        A ``SELECT ... FOR UPDATE SKIP LOCKED`` with the queue's retry/status
        predicates needs a filesort on MySQL.  InnoDB can lock more qualifying
        index records than the final LIMIT, causing a second worker to skip the
        entire backlog.  A single ordered UPDATE serializes only the short
        claim mutation; after the first commit, the next worker re-evaluates
        the predicate and takes the next batch.  Projection delivery remains
        concurrent and lease-owner completion is unchanged.
        """
        size = max(1, min(int(limit), 100))
        maximum_attempts = max(1, int(max_attempts))
        owner = str(worker_id or "memory-projection-worker")[:80]
        lease = max(30, int(lease_seconds))
        parameters: dict[str, object] = {
            "worker_id": owner,
            "max_attempts": maximum_attempts,
            "lease_seconds": lease,
            "batch_limit": size,
        }
        predicates = [
            "("
            "status IN ('pending','failed') "
            "OR (status='processing' AND "
            "(lease_until IS NULL OR lease_until <= UTC_TIMESTAMP()))"
            ")",
            "(operation='delete' OR attempt_count < :max_attempts)",
            "(next_retry_at IS NULL OR next_retry_at <= UTC_TIMESTAMP())",
        ]
        excluded = tuple(dict.fromkeys(
            str(value) for value in excluded_targets if value
        ))
        if excluded:
            placeholders = []
            for index, target in enumerate(excluded):
                name = f"excluded_target_{index}"
                placeholders.append(f":{name}")
                parameters[name] = target
            predicates.append(
                "target NOT IN (" + ",".join(placeholders) + ")"
            )
        owners = tuple(dict.fromkeys(
            str(value) for value in owner_allowlist if str(value or "").strip()
        ))
        if owners:
            placeholders = []
            for index, owner_id in enumerate(owners):
                name = f"allowed_owner_{index}"
                placeholders.append(f":{name}")
                parameters[name] = owner_id
            predicates.append(
                "owner_id IN (" + ",".join(placeholders) + ")"
            )
        if skip_summary_upserts:
            predicates.append(
                "NOT (target IN ('summary_es','summary_qdrant') "
                "AND operation='upsert')"
            )
        statement = text(
            "UPDATE conversation_memory_outbox SET "
            "status='processing', "
            "attempt_count=attempt_count + 1, "
            "claimed_at=UTC_TIMESTAMP(), "
            "lease_until=DATE_ADD(UTC_TIMESTAMP(), "
            "INTERVAL :lease_seconds SECOND), "
            "worker_id=:worker_id "
            "WHERE " + " AND ".join(predicates) + " "
            "ORDER BY created_at, event_id LIMIT :batch_limit"
        )
        result = await session.execute(statement, parameters)
        claimed_count = int(result.rowcount or 0)
        claimed_predicates = [
            ConversationMemoryOutboxMySQL.status == "processing",
            ConversationMemoryOutboxMySQL.worker_id == owner,
        ]
        if owners:
            claimed_predicates.append(
                ConversationMemoryOutboxMySQL.owner_id.in_(owners)
            )
        rows = list((await session.execute(
            Select(ConversationMemoryOutboxMySQL).where(
                *claimed_predicates,
            ).order_by(
                ConversationMemoryOutboxMySQL.created_at,
                ConversationMemoryOutboxMySQL.event_id,
            )
        )).scalars().all())
        if len(rows) != claimed_count:
            raise RuntimeError(
                "同一 projection worker_id 不允许并发 claim"
            )
        await session.commit()
        return rows

    @staticmethod
    async def claim_pending_outbox(
        session: AsyncSession,
        *,
        limit: int,
        max_attempts: int,
        worker_id: str = "memory-projection-worker",
        lease_seconds: int = 90,
        excluded_targets: Sequence[str] = (),
        owner_allowlist: Sequence[str] = (),
        skip_summary_upserts: bool = False,
    ) -> list[ConversationMemoryOutboxMySQL]:
        """Claim a disjoint batch and retry transient InnoDB lock victims.

        MySQL uses one ordered atomic UPDATE because its locking range read can
        cover rows beyond LIMIT.  Other dialects retain the locking-select
        implementation.  Changing ``status`` also rewrites the pending
        secondary index, so MySQL can still choose one transaction as a
        deadlock victim.  Its transaction is fully rolled back and a short
        bounded retry is safe: no lease from that failed attempt was durable.
        """
        dialect_name = session.get_bind().dialect.name
        if dialect_name == "mysql":
            if session.in_transaction():
                raise RuntimeError(
                    "MySQL projection claim 必须使用未开启事务的新 session"
                )
        saw_transient_lock_error = False
        for retry_index in range(3):
            try:
                claim_once = (
                    ConversationMemoryRepository._claim_pending_outbox_mysql_once
                    if dialect_name == "mysql"
                    else ConversationMemoryRepository._claim_pending_outbox_once
                )
                rows = await claim_once(
                    session,
                    limit=limit,
                    max_attempts=max_attempts,
                    worker_id=worker_id,
                    lease_seconds=lease_seconds,
                    excluded_targets=excluded_targets,
                    owner_allowlist=owner_allowlist,
                    skip_summary_upserts=skip_summary_upserts,
                )
                # Immediately after this transaction lost a deadlock, the
                # winner may still hold all candidate index ranges.  A
                # SKIP-LOCKED retry can validly return no rows even though the
                # winner only claimed part of the backlog.  Give that commit a
                # bounded window, then look once more before yielding idle.
                if (
                    rows
                    or not saw_transient_lock_error
                    or retry_index >= 2
                ):
                    return rows
                await asyncio.sleep(0.05 * (2 ** retry_index))
            except OperationalError as exc:
                await session.rollback()
                error_args = tuple(getattr(exc.orig, "args", ()) or ())
                error_code = error_args[0] if error_args else None
                transient = (
                    dialect_name == "mysql"
                    and error_code in {1205, 1213}
                )
                if not transient or retry_index >= 2:
                    raise
                saw_transient_lock_error = True
                await asyncio.sleep(0.05 * (2 ** retry_index))
        raise RuntimeError("projection claim retry loop unexpectedly exhausted")

    async def outbox_status(self) -> dict:
        result = await self.session.execute(
            select(
                ConversationMemoryOutboxMySQL.target,
                ConversationMemoryOutboxMySQL.status,
                func.count(ConversationMemoryOutboxMySQL.event_id),
            ).where(
                ConversationMemoryOutboxMySQL.owner_id == self.owner_id,
                ConversationMemoryOutboxMySQL.workspace_id == self.workspace_id,
            ).group_by(
                ConversationMemoryOutboxMySQL.target,
                ConversationMemoryOutboxMySQL.status,
            )
        )
        counts: dict[str, dict[str, int]] = {}
        for target, status, count in result.all():
            counts.setdefault(str(target), {})[str(status)] = int(count)
        backlog = sum(
            states.get("pending", 0)
            + states.get("processing", 0)
            + states.get("failed", 0)
            for states in counts.values()
        )
        return {"by_target": counts, "backlog": backlog}

    @staticmethod
    async def outbox_memory(
        session: AsyncSession, event: ConversationMemoryOutboxMySQL
    ) -> ConversationMemoryItemMySQL | None:
        row = await session.get(ConversationMemoryItemMySQL, event.memory_id)
        if row is None:
            return None
        if row.owner_id != event.owner_id or row.workspace_id != event.workspace_id:
            raise PermissionError("outbox 与 memory scope 不一致")
        return row

    @staticmethod
    async def renew_outbox_lease(
        session: AsyncSession,
        event_id: str,
        *,
        worker_id: str,
        lease_seconds: int,
    ) -> bool:
        now = _utcnow()
        result = await session.execute(
            update(ConversationMemoryOutboxMySQL).where(
                ConversationMemoryOutboxMySQL.event_id == str(event_id),
                ConversationMemoryOutboxMySQL.status == "processing",
                ConversationMemoryOutboxMySQL.worker_id
                == str(worker_id)[:80],
            ).values(
                lease_until=now + timedelta(
                    seconds=max(30, int(lease_seconds))
                )
            )
        )
        await session.commit()
        return bool(result.rowcount)

    @staticmethod
    async def complete_outbox(
        session: AsyncSession, event_id: str, *, worker_id: str | None = None
    ) -> bool:
        statement = update(ConversationMemoryOutboxMySQL).where(
            ConversationMemoryOutboxMySQL.event_id == event_id,
            ConversationMemoryOutboxMySQL.status == "processing",
        )
        if worker_id is not None:
            statement = statement.where(
                ConversationMemoryOutboxMySQL.worker_id == str(worker_id)[:80]
            )
        result = await session.execute(
            statement.values(
                status="completed",
                last_error=None,
                next_retry_at=None,
                claimed_at=None,
                lease_until=None,
                worker_id=None,
                completed_at=_utcnow(),
            )
        )
        await session.commit()
        return bool(result.rowcount)

    @staticmethod
    async def fail_outbox(
        session: AsyncSession,
        event_id: str,
        error_type: str,
        attempt_count: int,
        *,
        worker_id: str | None = None,
    ) -> bool:
        delay = min(300, 2 ** min(max(1, int(attempt_count)), 8))
        statement = update(ConversationMemoryOutboxMySQL).where(
            ConversationMemoryOutboxMySQL.event_id == event_id,
            ConversationMemoryOutboxMySQL.status == "processing",
        )
        if worker_id is not None:
            statement = statement.where(
                ConversationMemoryOutboxMySQL.worker_id == str(worker_id)[:80]
            )
        result = await session.execute(
            statement.values(
                status="failed",
                last_error=str(error_type)[:200],
                next_retry_at=_utcnow() + timedelta(seconds=delay),
                claimed_at=None,
                lease_until=None,
                worker_id=None,
            )
        )
        await session.commit()
        return bool(result.rowcount)
