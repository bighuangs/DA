"""会话默认数据上下文与不可变回合快照。"""
from __future__ import annotations

from app.application.ports.conversation import (
    ConversationDataContextStore,
    ConversationScopeReader,
    ConversationUnavailableError,
)
from app.application.services.data_source_context_service import (
    DataSourceContextService,
    ResolvedDataSourceContext,
)


def _record_dict(item) -> dict:
    return {
        "chat_id": item.chat_id,
        "revision": int(item.revision or 1),
        "source_mode": item.source_mode,
        "dataset_ids": list(item.dataset_ids or []),
        "comparison_intent": item.comparison_intent,
        "mapping_confirmation_id": item.mapping_confirmation_id,
        "invalidation_reason": item.invalidation_reason,
        "updated_at": item.updated_at.isoformat() if item.updated_at else None,
    }


class ConversationDataContextService:
    def __init__(
        self,
        *,
        owner_id: str,
        workspace_id: str,
        resolver: DataSourceContextService,
        store: ConversationDataContextStore,
    ):
        self.owner_id = owner_id
        self.workspace_id = workspace_id
        self.resolver = resolver
        self.store = store

    async def get_record(self, chat_id: str) -> dict:
        item = await self.store.get(chat_id)
        if item is None:
            return {
                "chat_id": chat_id,
                "revision": 0,
                "source_mode": "dw_only",
                "dataset_ids": [],
                "comparison_intent": None,
                "mapping_confirmation_id": None,
                "invalidation_reason": None,
                "updated_at": None,
            }
        return _record_dict(item)

    async def resolve_default(
        self, chat_id: str
    ) -> ResolvedDataSourceContext:
        record = await self.get_record(chat_id)
        return await self.resolver.resolve(
            dataset_ids=record["dataset_ids"],
            source_mode=record["source_mode"],
            comparison_intent=record["comparison_intent"],
            mapping_confirmation_id=record["mapping_confirmation_id"],
            revision=max(1, int(record["revision"] or 1)),
        )

    async def set(
        self,
        chat_id: str,
        *,
        source_mode: str,
        dataset_ids: list[str],
        comparison_intent: str | None,
        mapping_confirmation_id: str | None,
        expected_revision: int | None,
    ) -> dict:
        # 先做 owner/workspace/version/state/TTL 与映射确认校验，数据库中只保存
        # 已能冻结为权威上下文的引用。
        resolved = await self.resolver.resolve(
            dataset_ids=dataset_ids,
            source_mode=source_mode,
            comparison_intent=comparison_intent,
            mapping_confirmation_id=mapping_confirmation_id,
            revision=max(1, int(expected_revision or 0) + 1),
        )
        item = await self.store.set(
            chat_id,
            source_mode=source_mode,
            dataset_ids=dataset_ids,
            comparison_intent=comparison_intent,
            mapping_confirmation_id=mapping_confirmation_id,
            expected_revision=expected_revision,
        )
        return {
            **_record_dict(item),
            "resolved_context": resolved.access.public_dict(),
        }

    async def reset(
        self, chat_id: str, *, expected_revision: int | None
    ) -> dict:
        item = await self.store.reset(
            chat_id,
            expected_revision=expected_revision,
        )
        return _record_dict(item)

    async def snapshot_turn(
        self,
        *,
        chat_id: str,
        turn_id: str,
        resolved: ResolvedDataSourceContext,
    ) -> None:
        snapshot = resolved.state_dict()["data_source_context"]
        await self.store.snapshot_turn(
            chat_id=chat_id,
            turn_id=turn_id,
            revision=resolved.access.revision,
            snapshot=snapshot,
        )


class ScopedConversationDataContextService:
    """在会话归属校验后委托数据上下文用例。"""

    def __init__(
        self,
        *,
        scope: ConversationScopeReader,
        context: ConversationDataContextService,
    ) -> None:
        self._scope = scope
        self._context = context

    async def _ensure_scope(self, chat_id: str) -> None:
        if not await self._scope.exists(chat_id):
            raise ConversationUnavailableError("会话不存在或无权访问")

    async def get_record(self, chat_id: str) -> dict:
        await self._ensure_scope(chat_id)
        return await self._context.get_record(chat_id)

    async def set(
        self,
        chat_id: str,
        *,
        source_mode: str,
        dataset_ids: list[str],
        comparison_intent: str | None,
        mapping_confirmation_id: str | None,
        expected_revision: int | None,
    ) -> dict:
        await self._ensure_scope(chat_id)
        return await self._context.set(
            chat_id,
            source_mode=source_mode,
            dataset_ids=dataset_ids,
            comparison_intent=comparison_intent,
            mapping_confirmation_id=mapping_confirmation_id,
            expected_revision=expected_revision,
        )

    async def reset(
        self,
        chat_id: str,
        *,
        expected_revision: int | None,
    ) -> dict:
        await self._ensure_scope(chat_id)
        return await self._context.reset(
            chat_id,
            expected_revision=expected_revision,
        )


__all__ = [
    "ConversationDataContextService",
    "ScopedConversationDataContextService",
]
