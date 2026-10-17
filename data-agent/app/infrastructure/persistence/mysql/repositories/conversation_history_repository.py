"""MySQL authority for conversation discovery, transcript paging and lifecycle."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from sqlalchemy import Select, and_, desc, or_, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.conversation.errors import ConversationUnavailableError
from app.infrastructure.persistence.mysql.records.chat_history_mysql import ChatHistoryMySQL
from app.infrastructure.persistence.mysql.records.conversation_session_mysql import ConversationSessionMySQL
from app.infrastructure.persistence.mysql.records.conversation_turn_mysql import ConversationTurnMySQL
from app.infrastructure.security.masking import mask_free_text


def _required(value: str | None, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"conversation history repository 必须绑定 {name}")
    return text


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _as_datetime(value: object | None) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    text = str(value or "").strip()
    if text:
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(
                tzinfo=None
            )
        except ValueError:
            pass
    return _utcnow()


def _title(value: str) -> str:
    text = mask_free_text(value, max_chars=160).strip()
    return text or "未命名会话"


def _preview(value: str) -> str:
    return mask_free_text(value, max_chars=320).strip()


class ConversationHistoryRepository:
    """Every public operation is hard-bound to owner and workspace."""

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
    def session_scope(self) -> tuple:
        return (
            ConversationSessionMySQL.owner_id == self.owner_id,
            ConversationSessionMySQL.workspace_id == self.workspace_id,
        )

    @property
    def turn_scope(self) -> tuple:
        return (
            ConversationTurnMySQL.owner_id == self.owner_id,
            ConversationTurnMySQL.workspace_id == self.workspace_id,
        )

    async def get_session(
        self,
        chat_id: str,
        *,
        include_archived: bool = True,
        include_deleted: bool = False,
        for_update: bool = False,
    ) -> ConversationSessionMySQL | None:
        statement = Select(ConversationSessionMySQL).where(
            *self.session_scope,
            ConversationSessionMySQL.chat_id == str(chat_id),
        )
        if not include_deleted:
            statement = statement.where(
                ConversationSessionMySQL.status != "deleted",
                ConversationSessionMySQL.deleted_at.is_(None),
            )
        if not include_archived:
            statement = statement.where(
                ConversationSessionMySQL.status == "active"
            )
        if for_update:
            statement = statement.with_for_update()
        return (await self.session.execute(statement)).scalar_one_or_none()

    async def context_reset_at(self, chat_id: str) -> datetime | None:
        row = await self.get_session(chat_id)
        return row.context_reset_at if row is not None else None

    async def start_turn(
        self,
        *,
        chat_id: str,
        turn_id: str,
        user_text: str,
        trace_id: str | None = None,
        source_message_id: int | None = None,
    ) -> ConversationTurnMySQL:
        """Idempotently expose a submitted turn before Agent execution."""
        chat = str(chat_id)[:64]
        turn_key = str(turn_id)[:80]
        now = _utcnow()
        conversation = await self.get_session(
            chat,
            include_archived=True,
            include_deleted=True,
            for_update=True,
        )
        if conversation is not None and (
            conversation.status in {"deleted", "deleting"}
            or conversation.deleted_at is not None
        ):
            raise ConversationUnavailableError("会话不存在或无权访问")
        if conversation is not None and conversation.status == "archived":
            raise ConversationUnavailableError("会话已归档，请先恢复后再继续")

        existing = (
            await self.session.execute(
                Select(ConversationTurnMySQL).where(
                    *self.turn_scope,
                    ConversationTurnMySQL.turn_id == turn_key,
                )
            )
        ).scalar_one_or_none()
        safe_text = mask_free_text(user_text, max_chars=2000)
        if existing is None:
            existing = ConversationTurnMySQL(
                turn_id=turn_key,
                chat_id=chat,
                owner_id=self.owner_id,
                workspace_id=self.workspace_id,
                trace_id=(str(trace_id)[:64] if trace_id else None),
                source_message_id=source_message_id,
                user_text=safe_text,
                status="pending",
                replay_fidelity="message_only",
                created_at=now,
            )
            self.session.add(existing)
            if conversation is None:
                conversation = ConversationSessionMySQL(
                    chat_id=chat,
                    owner_id=self.owner_id,
                    workspace_id=self.workspace_id,
                    title=_title(safe_text),
                    preview=_preview(safe_text),
                    status="active",
                    last_turn_status="pending",
                    turn_count=1,
                    last_trace_id=(str(trace_id)[:64] if trace_id else None),
                    created_at=now,
                    last_message_at=now,
                    version=1,
                )
                self.session.add(conversation)
            else:
                conversation.turn_count = int(conversation.turn_count or 0) + 1
        if conversation is None:
            raise RuntimeError("会话目录创建失败")
        conversation.preview = _preview(safe_text)
        conversation.last_turn_status = "pending"
        conversation.last_trace_id = (
            str(trace_id)[:64] if trace_id else conversation.last_trace_id
        )
        conversation.last_message_at = now
        conversation.version = int(conversation.version or 0) + 1
        await self.session.flush()
        return existing

    async def complete_turn(
        self,
        *,
        chat_id: str,
        turn_id: str,
        source_history_id: int,
        row_count: int,
        assistant_summary: str | None = None,
    ) -> ConversationTurnMySQL:
        conversation = await self.get_session(
            str(chat_id),
            include_archived=True,
            include_deleted=True,
            for_update=True,
        )
        if conversation is None or conversation.status != "active":
            raise ConversationUnavailableError("会话已重置、删除或不可继续")
        turn = (
            await self.session.execute(
                Select(ConversationTurnMySQL).where(
                    *self.turn_scope,
                    ConversationTurnMySQL.chat_id == str(chat_id),
                    ConversationTurnMySQL.turn_id == str(turn_id)[:80],
                )
            )
        ).scalar_one_or_none()
        if turn is None:
            raise ConversationUnavailableError("待完成的会话回合不存在")
        now = _utcnow()
        turn.source_history_id = int(source_history_id)
        turn.status = "succeeded"
        turn.assistant_summary = mask_free_text(
            assistant_summary or f"查询成功，返回 {max(0, int(row_count))} 行。",
            max_chars=500,
        )
        turn.replay_fidelity = "result_snapshot"
        turn.completed_at = now
        conversation.last_turn_status = "succeeded"
        conversation.last_message_at = now
        conversation.version = int(conversation.version or 0) + 1
        await self.session.flush()
        return turn

    async def complete_handled_turn(
        self,
        *,
        chat_id: str,
        turn_id: str,
        assistant_summary: str,
    ) -> ConversationTurnMySQL:
        """完成无 SQL 的业务引导、拒绝或澄清回合。"""

        conversation = await self.get_session(
            str(chat_id),
            include_archived=True,
            include_deleted=True,
            for_update=True,
        )
        if conversation is None or conversation.status != "active":
            raise ConversationUnavailableError("会话已重置、删除或不可继续")
        turn = (
            await self.session.execute(
                Select(ConversationTurnMySQL).where(
                    *self.turn_scope,
                    ConversationTurnMySQL.chat_id == str(chat_id),
                    ConversationTurnMySQL.turn_id == str(turn_id)[:80],
                )
            )
        ).scalar_one_or_none()
        if turn is None:
            raise ConversationUnavailableError("待完成的会话回合不存在")
        if turn.status == "succeeded" and turn.replay_fidelity == "message_only":
            return turn
        if turn.status != "pending":
            raise ConversationUnavailableError("会话回合已结束，不能改写为已处理状态")

        now = _utcnow()
        turn.source_history_id = None
        turn.status = "succeeded"
        turn.assistant_summary = mask_free_text(
            assistant_summary,
            max_chars=500,
        )
        turn.replay_fidelity = "message_only"
        turn.completed_at = now
        conversation.last_turn_status = "succeeded"
        conversation.last_message_at = now
        conversation.version = int(conversation.version or 0) + 1
        await self.session.flush()
        return turn

    async def fail_turn(
        self,
        *,
        chat_id: str,
        turn_id: str,
        status: str = "failed",
        assistant_summary: str | None = None,
    ) -> ConversationTurnMySQL | None:
        normalized = status if status in {"failed", "cancelled"} else "failed"
        turn = (
            await self.session.execute(
                Select(ConversationTurnMySQL).where(
                    *self.turn_scope,
                    ConversationTurnMySQL.chat_id == str(chat_id),
                    ConversationTurnMySQL.turn_id == str(turn_id)[:80],
                )
            )
        ).scalar_one_or_none()
        if turn is None:
            return None
        if turn.status == "succeeded":
            return turn
        now = _utcnow()
        turn.status = normalized
        turn.assistant_summary = mask_free_text(
            assistant_summary
            or ("请求已取消。" if normalized == "cancelled" else "本轮未完成，可重新提问。"),
            max_chars=500,
        )
        turn.completed_at = now
        conversation = await self.get_session(str(chat_id))
        if conversation is not None:
            conversation.last_turn_status = normalized
            conversation.last_message_at = now
            conversation.version = int(conversation.version or 0) + 1
        await self.session.flush()
        return turn

    async def list_sessions(
        self,
        *,
        status: str,
        limit: int,
        before_at: datetime | None = None,
        before_id: int | None = None,
    ) -> list[ConversationSessionMySQL]:
        statement = Select(ConversationSessionMySQL).where(
            *self.session_scope,
            ConversationSessionMySQL.status == status,
            ConversationSessionMySQL.deleted_at.is_(None),
        )
        if before_at is not None and before_id is not None:
            statement = statement.where(
                or_(
                    ConversationSessionMySQL.last_message_at < before_at,
                    and_(
                        ConversationSessionMySQL.last_message_at == before_at,
                        ConversationSessionMySQL.id < int(before_id),
                    ),
                )
            )
        result = await self.session.execute(
            statement.order_by(
                desc(ConversationSessionMySQL.last_message_at),
                desc(ConversationSessionMySQL.id),
            ).limit(max(1, min(int(limit), 50)) + 1)
        )
        return list(result.scalars().all())

    async def list_turns(
        self,
        *,
        chat_id: str,
        limit: int,
        before_at: datetime | None = None,
        before_id: int | None = None,
    ) -> list[ConversationTurnMySQL]:
        statement = Select(ConversationTurnMySQL).where(
            *self.turn_scope,
            ConversationTurnMySQL.chat_id == str(chat_id),
            ConversationTurnMySQL.deleted_at.is_(None),
        )
        if before_at is not None and before_id is not None:
            statement = statement.where(
                or_(
                    ConversationTurnMySQL.created_at < before_at,
                    and_(
                        ConversationTurnMySQL.created_at == before_at,
                        ConversationTurnMySQL.id < int(before_id),
                    ),
                )
            )
        result = await self.session.execute(
            statement.order_by(
                desc(ConversationTurnMySQL.created_at),
                desc(ConversationTurnMySQL.id),
            ).limit(max(1, min(int(limit), 100)) + 1)
        )
        return list(result.scalars().all())

    async def history_by_ids(
        self, source_history_ids: Iterable[int]
    ) -> dict[int, ChatHistoryMySQL]:
        identifiers = list(
            dict.fromkeys(
                int(value) for value in source_history_ids if value is not None
            )
        )[:100]
        if not identifiers:
            return {}
        result = await self.session.execute(
            Select(ChatHistoryMySQL).where(
                ChatHistoryMySQL.id.in_(identifiers),
                ChatHistoryMySQL.owner_id == self.owner_id,
                ChatHistoryMySQL.workspace_id == self.workspace_id,
            )
        )
        return {int(row.id): row for row in result.scalars().all()}

    async def update_session(
        self,
        chat_id: str,
        *,
        title: str | None = None,
        status: str | None = None,
    ) -> ConversationSessionMySQL | None:
        conversation = await self.get_session(chat_id)
        if conversation is None:
            return None
        if conversation.status == "deleting":
            raise ConversationUnavailableError("会话正在删除，不能修改")
        now = _utcnow()
        if title is not None:
            conversation.title = _title(title)
        if status is not None:
            if status not in {"active", "archived"}:
                raise ValueError("会话状态只支持 active 或 archived")
            conversation.status = status
            conversation.archived_at = now if status == "archived" else None
        conversation.version = int(conversation.version or 0) + 1
        await self.session.flush()
        return conversation

    async def soft_delete(self, chat_id: str) -> bool:
        conversation = await self.get_session(chat_id)
        if conversation is None:
            return False
        now = _utcnow()
        conversation.status = "deleted"
        conversation.deleted_at = now
        conversation.version = int(conversation.version or 0) + 1
        await self.session.execute(
            update(ConversationTurnMySQL)
            .where(
                *self.turn_scope,
                ConversationTurnMySQL.chat_id == str(chat_id),
                ConversationTurnMySQL.deleted_at.is_(None),
            )
            .values(deleted_at=now)
        )
        await self.session.flush()
        return True

    async def begin_delete(self, chat_id: str) -> bool:
        """Install a durable fence before cross-store cleanup begins."""
        conversation = await self.get_session(
            chat_id,
            include_archived=True,
            include_deleted=True,
            for_update=True,
        )
        if conversation is None or conversation.status == "deleted":
            return False
        if conversation.status != "deleting":
            conversation.status = "deleting"
            conversation.version = int(conversation.version or 0) + 1
            await self.session.flush()
        return True

    async def reset_context(self, chat_id: str) -> bool:
        conversation = await self.get_session(
            chat_id, include_archived=True, for_update=True
        )
        if conversation is None:
            return False
        conversation.context_reset_at = _utcnow()
        conversation.version = int(conversation.version or 0) + 1
        await self.session.flush()
        return True

    async def rebuild_session(self, chat_id: str) -> ConversationSessionMySQL | None:
        result = await self.session.execute(
            Select(ConversationTurnMySQL)
            .where(
                *self.turn_scope,
                ConversationTurnMySQL.chat_id == str(chat_id),
                ConversationTurnMySQL.deleted_at.is_(None),
            )
            .order_by(
                ConversationTurnMySQL.created_at,
                ConversationTurnMySQL.id,
            )
        )
        turns = list(result.scalars().all())
        if not turns:
            return None
        conversation = await self.get_session(
            chat_id, include_archived=True, include_deleted=True
        )
        first, last = turns[0], turns[-1]
        if conversation is None:
            conversation = ConversationSessionMySQL(
                chat_id=str(chat_id)[:64],
                owner_id=self.owner_id,
                workspace_id=self.workspace_id,
                title=_title(first.user_text),
                preview=_preview(last.user_text),
                status="active",
                last_turn_status=last.status,
                turn_count=len(turns),
                last_trace_id=last.trace_id,
                created_at=_as_datetime(first.created_at),
                last_message_at=_as_datetime(last.created_at),
                version=1,
            )
            self.session.add(conversation)
        elif conversation.status != "deleted":
            conversation.preview = _preview(last.user_text)
            conversation.last_turn_status = last.status
            conversation.turn_count = len(turns)
            conversation.last_trace_id = last.trace_id
            conversation.last_message_at = _as_datetime(last.created_at)
        await self.session.flush()
        return conversation
