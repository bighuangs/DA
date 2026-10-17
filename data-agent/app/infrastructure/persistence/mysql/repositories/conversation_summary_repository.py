"""MySQL authority for TODO-96 summary segments, signals and coverage."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime, timezone

from sqlalchemy import Select, desc, func, select, update
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.persistence.mysql.records.conversation_message_mysql import ConversationMessageMySQL
from app.infrastructure.persistence.mysql.records.conversation_turn_mysql import ConversationTurnMySQL
from app.infrastructure.persistence.mysql.records.chat_history_mysql import ChatHistoryMySQL
from app.infrastructure.persistence.mysql.records.conversation_summary_segment_mysql import (
    ConversationSummarySegmentMySQL,
)
from app.infrastructure.persistence.mysql.records.conversation_summary_signal_mysql import (
    ConversationSummarySignalMySQL,
)
from app.infrastructure.persistence.mysql.records.conversation_summary_state_mysql import (
    ConversationSummaryStateMySQL,
)


def _required(value: str | None, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"conversation summary repository 必须绑定 {name}")
    return text


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ConversationSummaryRepository:
    """Every query is bound to owner + workspace; summaries also bind chat."""

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
    def segment_scope(self) -> tuple:
        return (
            ConversationSummarySegmentMySQL.owner_id == self.owner_id,
            ConversationSummarySegmentMySQL.workspace_id == self.workspace_id,
        )

    async def get_state(
        self, chat_id: str, *, for_update: bool = False
    ) -> ConversationSummaryStateMySQL | None:
        statement = Select(ConversationSummaryStateMySQL).where(
            ConversationSummaryStateMySQL.owner_id == self.owner_id,
            ConversationSummaryStateMySQL.workspace_id == self.workspace_id,
            ConversationSummaryStateMySQL.chat_id == str(chat_id),
        )
        if for_update:
            statement = statement.with_for_update()
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def ensure_state(
        self, chat_id: str, *, schema_version: str, for_update: bool = False
    ) -> ConversationSummaryStateMySQL:
        chat = str(chat_id)[:64]
        dialect_name = self.session.get_bind().dialect.name
        values = {
            "owner_id": self.owner_id,
            "workspace_id": self.workspace_id,
            "chat_id": chat,
            "version": 0,
            "active_segment_count": 0,
            "schema_version": str(schema_version),
        }

        # A missing-row SELECT ... FOR UPDATE takes a gap lock under InnoDB
        # REPEATABLE READ.  If two first builders both take that gap lock and
        # then INSERT inside a savepoint, MySQL can deadlock and roll back the
        # whole transaction; SQLAlchemy will then fail again because the
        # savepoint no longer exists.  Let the unique key serialize an atomic
        # insert/no-op update first, then use a locking current read.  This also
        # makes a row committed after an earlier RR snapshot visible.
        if dialect_name == "mysql":
            statement = mysql_insert(ConversationSummaryStateMySQL).values(
                **values
            )
            await self.session.execute(statement.on_duplicate_key_update(
                id=ConversationSummaryStateMySQL.id
            ))
            row = await self.get_state(chat, for_update=for_update)
            if row is None:
                raise RuntimeError("summary state 原子写入后不可见")
            return row

        if dialect_name == "sqlite":
            statement = sqlite_insert(ConversationSummaryStateMySQL).values(
                **values
            ).on_conflict_do_nothing(
                index_elements=["owner_id", "workspace_id", "chat_id"]
            )
            await self.session.execute(statement)
            row = await self.get_state(chat, for_update=for_update)
            if row is None:
                raise RuntimeError("summary state 原子写入后不可见")
            return row

        row = await self.get_state(chat, for_update=for_update)
        if row is None:
            candidate = ConversationSummaryStateMySQL(
                **values,
            )
            try:
                # The unique scope key is the final arbiter when two first
                # successful turns create summary state concurrently.  A
                # savepoint contains the duplicate-key failure so the caller's
                # successful turn transaction is not poisoned.
                async with self.session.begin_nested():
                    self.session.add(candidate)
                    await self.session.flush()
                row = candidate
            except IntegrityError:
                row = await self.get_state(chat, for_update=for_update)
                if row is None:
                    raise
        return row

    async def list_uncovered_summary_messages(
        self,
        chat_id: str,
        *,
        after_message_id: int | None,
        limit: int = 500,
    ) -> list[ConversationMessageMySQL]:
        statement = Select(ConversationMessageMySQL).where(
            ConversationMessageMySQL.owner_id == self.owner_id,
            ConversationMessageMySQL.workspace_id == self.workspace_id,
            ConversationMessageMySQL.chat_id == str(chat_id),
            ConversationMessageMySQL.deleted_at.is_(None),
            # A lower-id pending message is a coverage barrier.  Returning it
            # lets the builder stop before the gap instead of advancing the
            # watermark past a turn that may succeed later.
            ConversationMessageMySQL.status.in_(("success", "pending")),
        )
        if after_message_id is not None:
            statement = statement.where(
                ConversationMessageMySQL.id > int(after_message_id)
            )
        result = await self.session.execute(
            statement.order_by(ConversationMessageMySQL.id).limit(
                max(1, min(int(limit), 2000))
            )
        )
        return list(result.scalars().all())

    async def semantic_facts_for_messages(
        self, message_ids: Sequence[int]
    ) -> list[dict]:
        identifiers = list(dict.fromkeys(int(value) for value in message_ids))[:500]
        if not identifiers:
            return []
        result = await self.session.execute(
            select(ConversationTurnMySQL, ChatHistoryMySQL)
            .join(
                ChatHistoryMySQL,
                ChatHistoryMySQL.id == ConversationTurnMySQL.source_history_id,
            )
            .where(
                ConversationTurnMySQL.owner_id == self.owner_id,
                ConversationTurnMySQL.workspace_id == self.workspace_id,
                ConversationTurnMySQL.source_message_id.in_(identifiers),
                ConversationTurnMySQL.status == "succeeded",
            )
            .order_by(ConversationTurnMySQL.id)
        )
        return [
            {
                "fact_type": "successful_query_semantics",
                "content": str(history.standalone_question or history.query or ""),
                "standalone_question": str(
                    history.standalone_question or history.query or ""
                ),
                "semantic_state": dict(history.semantic_state or {}),
                "source_message_ids": [int(turn.source_message_id)],
                "source_history_id": int(history.id),
                "status": "supported",
            }
            for turn, history in result.all()
        ]

    async def insert_segment_with_signals(
        self, values: dict, signals: Sequence[dict]
    ) -> tuple[ConversationSummarySegmentMySQL, bool]:
        summary_id = _required(values.get("summary_id"), "summary_id")
        existing = await self.session.get(ConversationSummarySegmentMySQL, summary_id)
        if existing is not None:
            if (
                existing.owner_id != self.owner_id
                or existing.workspace_id != self.workspace_id
            ):
                raise PermissionError("summary_id 已属于其他 owner/workspace")
            return existing, False
        row = ConversationSummarySegmentMySQL(
            **{
                **values,
                "summary_id": summary_id,
                "owner_id": self.owner_id,
                "workspace_id": self.workspace_id,
            }
        )
        self.session.add(row)
        for signal in signals:
            self.session.add(ConversationSummarySignalMySQL(**{
                **signal,
                "summary_id": summary_id,
                "owner_id": self.owner_id,
                "workspace_id": self.workspace_id,
                "chat_id": row.chat_id,
            }))
        await self.session.flush()
        return row, True

    async def advance_state(
        self,
        state: ConversationSummaryStateMySQL,
        *,
        covered_through_message_id: int,
        source_hash: str,
        schema_version: str,
    ) -> None:
        next_id = int(covered_through_message_id)
        current_id = int(state.covered_through_message_id or 0)
        if next_id < current_id:
            raise ValueError("summary coverage watermark 不允许回退")
        state.covered_through_message_id = next_id
        state.version = int(state.version or 0) + 1
        state.active_segment_count = await self.active_segment_count(state.chat_id)
        state.last_source_hash = str(source_hash)
        state.schema_version = str(schema_version)
        await self.session.flush()

    async def active_segment_count(self, chat_id: str) -> int:
        result = await self.session.execute(
            select(func.count(ConversationSummarySegmentMySQL.summary_id)).where(
                *self.segment_scope,
                ConversationSummarySegmentMySQL.chat_id == str(chat_id),
                ConversationSummarySegmentMySQL.status == "active",
                ConversationSummarySegmentMySQL.deleted_at.is_(None),
            )
        )
        return int(result.scalar() or 0)

    async def list_active_segments(
        self,
        chat_id: str,
        *,
        limit: int = 100,
    ) -> list[ConversationSummarySegmentMySQL]:
        result = await self.session.execute(
            Select(ConversationSummarySegmentMySQL).where(
                *self.segment_scope,
                ConversationSummarySegmentMySQL.chat_id == str(chat_id),
                ConversationSummarySegmentMySQL.status == "active",
                ConversationSummarySegmentMySQL.deleted_at.is_(None),
            ).order_by(
                desc(ConversationSummarySegmentMySQL.level),
                desc(ConversationSummarySegmentMySQL.source_end_message_id),
                ConversationSummarySegmentMySQL.summary_id,
            ).limit(max(1, min(int(limit), 500)))
        )
        return list(result.scalars().all())

    async def list_authoritative_by_ids(
        self, chat_id: str, summary_ids: Iterable[str]
    ) -> list[ConversationSummarySegmentMySQL]:
        identifiers = list(dict.fromkeys(
            str(value) for value in summary_ids if value
        ))[:100]
        if not identifiers:
            return []
        result = await self.session.execute(
            Select(ConversationSummarySegmentMySQL).where(
                ConversationSummarySegmentMySQL.summary_id.in_(identifiers),
                *self.segment_scope,
                ConversationSummarySegmentMySQL.chat_id == str(chat_id),
                ConversationSummarySegmentMySQL.status == "active",
                ConversationSummarySegmentMySQL.deleted_at.is_(None),
            )
        )
        by_id = {row.summary_id: row for row in result.scalars().all()}
        return [by_id[value] for value in identifiers if value in by_id]

    async def list_signals_for_summaries(
        self, summary_ids: Iterable[str]
    ) -> dict[str, list[ConversationSummarySignalMySQL]]:
        identifiers = list(dict.fromkeys(
            str(value) for value in summary_ids if value
        ))[:100]
        if not identifiers:
            return {}
        result = await self.session.execute(
            Select(ConversationSummarySignalMySQL).where(
                ConversationSummarySignalMySQL.summary_id.in_(identifiers),
                ConversationSummarySignalMySQL.owner_id == self.owner_id,
                ConversationSummarySignalMySQL.workspace_id == self.workspace_id,
                ConversationSummarySignalMySQL.status == "active",
                ConversationSummarySignalMySQL.deleted_at.is_(None),
            ).order_by(
                ConversationSummarySignalMySQL.summary_id,
                ConversationSummarySignalMySQL.source_order,
                ConversationSummarySignalMySQL.signal_id,
            )
        )
        grouped: dict[str, list[ConversationSummarySignalMySQL]] = {}
        for row in result.scalars().all():
            grouped.setdefault(row.summary_id, []).append(row)
        return grouped

    async def search_signal_ids(
        self,
        chat_id: str,
        query_signals: Sequence[str],
        *,
        limit: int,
    ) -> tuple[list[str], dict]:
        normalized = [
            " ".join(str(value or "").strip().lower().split())
            for value in query_signals
            if str(value or "").strip()
        ][:24]
        if not normalized:
            return [], {"status": "no_signals", "candidate_count": 0}
        result = await self.session.execute(
            Select(ConversationSummarySignalMySQL).where(
                ConversationSummarySignalMySQL.owner_id == self.owner_id,
                ConversationSummarySignalMySQL.workspace_id == self.workspace_id,
                ConversationSummarySignalMySQL.chat_id == str(chat_id),
                ConversationSummarySignalMySQL.status == "active",
                ConversationSummarySignalMySQL.deleted_at.is_(None),
            ).limit(1000)
        )
        scores: dict[str, float] = {}
        for row in result.scalars().all():
            value = str(row.normalized_value or "").lower()
            best = 0.0
            for signal in normalized:
                if signal == value:
                    best = max(best, 2.0)
                elif signal in value or value in signal:
                    best = max(best, 1.0)
            if best:
                scores[row.summary_id] = scores.get(row.summary_id, 0.0) + (
                    best * float(row.weight or 1.0)
                )
        ordered = sorted(scores, key=lambda value: (-scores[value], value))[
            :max(1, int(limit))
        ]
        return ordered, {
            "status": "ok",
            "query_signal_count": len(normalized),
            "candidate_count": len(ordered),
            "raw_text_logged": False,
            "owner_scope_applied": True,
            "workspace_scope_applied": True,
            "chat_scope_applied": True,
        }

    async def revoke_chat(self, chat_id: str) -> list[str]:
        result = await self.session.execute(
            select(ConversationSummarySegmentMySQL.summary_id).where(
                *self.segment_scope,
                ConversationSummarySegmentMySQL.chat_id == str(chat_id),
                ConversationSummarySegmentMySQL.status == "active",
                ConversationSummarySegmentMySQL.deleted_at.is_(None),
            )
        )
        identifiers = [str(value) for value in result.scalars().all()]
        if not identifiers:
            return []
        now = _utcnow()
        await self.session.execute(
            update(ConversationSummarySegmentMySQL).where(
                ConversationSummarySegmentMySQL.summary_id.in_(identifiers),
                *self.segment_scope,
            ).values(status="revoked", valid_to=now)
        )
        await self.session.execute(
            update(ConversationSummarySignalMySQL).where(
                ConversationSummarySignalMySQL.summary_id.in_(identifiers),
                ConversationSummarySignalMySQL.owner_id == self.owner_id,
                ConversationSummarySignalMySQL.workspace_id == self.workspace_id,
            ).values(status="revoked", deleted_at=now)
        )
        state = await self.get_state(chat_id, for_update=True)
        if state is not None:
            state.version = int(state.version or 0) + 1
            state.active_segment_count = 0
        await self.session.flush()
        return identifiers

    async def status(self, chat_id: str) -> dict:
        state = await self.get_state(chat_id)
        segments = await self.list_active_segments(chat_id, limit=500)
        by_level: dict[str, int] = {}
        source_tokens = summary_tokens = 0
        for row in segments:
            key = f"L{int(row.level or 0)}"
            by_level[key] = by_level.get(key, 0) + 1
            source_tokens += int(row.source_token_count or 0)
            summary_tokens += int(row.summary_token_count or 0)
        signal_result = await self.session.execute(
            select(func.count(ConversationSummarySignalMySQL.signal_id)).where(
                ConversationSummarySignalMySQL.owner_id == self.owner_id,
                ConversationSummarySignalMySQL.workspace_id == self.workspace_id,
                ConversationSummarySignalMySQL.chat_id == str(chat_id),
                ConversationSummarySignalMySQL.status == "active",
                ConversationSummarySignalMySQL.deleted_at.is_(None),
            )
        )
        return {
            "enabled": True,
            "active_segment_count": len(segments),
            "by_level": by_level,
            "signal_count": int(signal_result.scalar() or 0),
            "covered_through_message_id": (
                int(state.covered_through_message_id)
                if state and state.covered_through_message_id is not None
                else None
            ),
            "watermark_version": int(getattr(state, "version", 0) or 0),
            "source_tokens": source_tokens,
            "summary_tokens": summary_tokens,
            "compression_ratio": round(summary_tokens / max(1, source_tokens), 4),
            "extractor_version": (
                segments[0].extractor_version if segments else None
            ),
            "items": [
                {
                    "summary_id": row.summary_id,
                    "level": int(row.level or 0),
                    "content": row.content_masked,
                    "source_start_message_id": int(row.source_start_message_id),
                    "source_end_message_id": int(row.source_end_message_id),
                    "source_message_count": len(row.source_message_ids or []),
                    "summary_tokens": int(row.summary_token_count or 0),
                    "created_at": str(row.created_at),
                }
                for row in segments[:20]
            ],
        }

    @staticmethod
    async def outbox_summary(
        session: AsyncSession, summary_id: str, owner_id: str, workspace_id: str
    ) -> ConversationSummarySegmentMySQL | None:
        row = await session.get(ConversationSummarySegmentMySQL, str(summary_id))
        if row is None:
            return None
        if row.owner_id != owner_id or row.workspace_id != workspace_id:
            raise PermissionError("outbox 与 summary scope 不一致")
        return row
