"""Authenticated conversation directory and transcript recovery endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Query

from app.adapters.inbound.http.dependencies import (
    get_conversation_history_service,
    get_conversation_memory_service,
    get_scoped_conversation_data_context_service,
)
from app.adapters.inbound.http.schemas.conversation_schema import ConversationUpdateSchema
from app.adapters.inbound.http.schemas.upload_schema import (
    ConversationDataContextSchema,
    DataContextResetSchema,
)
from app.domain.data_sources.context import (
    DataSourceContextError,
    MappingConfirmationRequired,
)
from app.application.ports.conversation import ConversationUnavailableError
from app.application.services.conversation_history_service import (
    ConversationHistoryService,
    InvalidConversationCursor,
)
from app.application.services.conversation_data_context_service import (
    ScopedConversationDataContextService,
)
from app.application.services.conversation_memory_service import ConversationMemoryService


conversation_router = APIRouter(tags=["conversations"])


@conversation_router.get("/api/chats")
async def list_conversations(
    status: str = Query(default="active", pattern=r"^(active|archived)$"),
    limit: int | None = Query(default=None, ge=1, le=50),
    cursor: str | None = Query(default=None, max_length=512),
    service: ConversationHistoryService = Depends(
        get_conversation_history_service
    ),
):
    try:
        return await service.list_sessions(
            status=status, limit=limit, cursor=cursor
        )
    except InvalidConversationCursor as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@conversation_router.get("/api/chats/{chat_id}/turns")
async def get_conversation_turns(
    chat_id: str,
    limit: int | None = Query(default=None, ge=1, le=100),
    before: str | None = Query(default=None, max_length=512),
    service: ConversationHistoryService = Depends(
        get_conversation_history_service
    ),
):
    try:
        result = await service.get_turns(chat_id, limit=limit, before=before)
    except InvalidConversationCursor as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="会话不存在或无权访问")
    return result


@conversation_router.patch("/api/chats/{chat_id}")
async def update_conversation(
    chat_id: str,
    body: ConversationUpdateSchema,
    service: ConversationHistoryService = Depends(
        get_conversation_history_service
    ),
):
    if not body.model_fields_set:
        raise HTTPException(status_code=400, detail="至少需要修改标题或状态")
    try:
        result = await service.update_session(
            chat_id,
            title=body.title if "title" in body.model_fields_set else None,
            status=body.status if "status" in body.model_fields_set else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="会话不存在或无权访问")
    return result


@conversation_router.delete("/api/chats/{chat_id}", status_code=204)
async def delete_conversation(
    chat_id: str,
    service: ConversationHistoryService = Depends(
        get_conversation_history_service
    ),
    memory_service: ConversationMemoryService = Depends(
        get_conversation_memory_service
    ),
):
    if not await service.begin_delete(chat_id):
        raise HTTPException(status_code=404, detail="会话不存在或无权访问")
    # The committed `deleting` fence prevents new or in-flight turns from
    # recreating memory after the scoped cleanup snapshot.
    try:
        await memory_service.reset(
            chat_id, include_long_term=True, strict_redis=True
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail="会话记忆清理尚未完成，请重试删除",
        ) from exc
    deleted = await service.delete_session(chat_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="会话不存在或无权访问")


@conversation_router.get("/api/chats/{chat_id}/data-context")
async def get_conversation_data_context(
    chat_id: str,
    service: ScopedConversationDataContextService = Depends(
        get_scoped_conversation_data_context_service
    ),
):
    try:
        return await service.get_record(chat_id)
    except ConversationUnavailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@conversation_router.put("/api/chats/{chat_id}/data-context")
async def set_conversation_data_context(
    chat_id: str,
    body: ConversationDataContextSchema,
    service: ScopedConversationDataContextService = Depends(
        get_scoped_conversation_data_context_service
    ),
):
    try:
        return await service.set(
            chat_id,
            source_mode=body.source_mode,
            dataset_ids=body.dataset_ids,
            comparison_intent=body.comparison_intent,
            mapping_confirmation_id=body.mapping_confirmation_id,
            expected_revision=body.expected_revision,
        )
    except ConversationUnavailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MappingConfirmationRequired as exc:
        raise HTTPException(status_code=409, detail=exc.to_dict()) from exc
    except (DataSourceContextError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@conversation_router.post("/api/chats/{chat_id}/data-context/reset")
async def reset_conversation_data_context(
    chat_id: str,
    body: DataContextResetSchema,
    service: ScopedConversationDataContextService = Depends(
        get_scoped_conversation_data_context_service
    ),
):
    try:
        return await service.reset(
            chat_id, expected_revision=body.expected_revision
        )
    except ConversationUnavailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
