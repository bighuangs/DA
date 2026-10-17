"""Isolated HTTP/SQLite backend used by the real Playwright integration smoke.

It runs the production auth/query/evaluation routers and real SQLAlchemy models.
Only the costly NL2SQL provider is replaced at the dependency boundary, keeping
HTTP, Pydantic, bearer auth, RBAC, SSE and database transactions real.
"""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.adapters.inbound.http.dependencies import (
    get_auth_use_case,
    get_conversation_data_context_service,
    get_conversation_history_service,
    get_current_principal,
    get_meta_session,
    get_query_capacity,
    get_query_service,
    get_scoped_conversation_data_context_service,
)
from app.adapters.inbound.http.routers.auth_router import auth_router
from app.adapters.inbound.http.routers.query_router import query_router
from app.adapters.inbound.http.routers.conversation_router import conversation_router
from app.config.app_config import app_config
from app.infrastructure.persistence.mysql.records.base import Base
from app.infrastructure.persistence.mysql.records.chat_history_mysql import ChatHistoryMySQL
from app.infrastructure.persistence.mysql.records.user_mysql import UserMySQL  # noqa: F401
from app.infrastructure.persistence.mysql.records.workspace_mysql import WorkspaceMySQL  # noqa: F401
from app.infrastructure.persistence.mysql.records.workspace_member_mysql import WorkspaceMemberMySQL  # noqa: F401
from app.infrastructure.persistence.mysql.repositories.conversation_history_repository import (
    ConversationHistoryRepository,
)
from app.infrastructure.persistence.mysql.conversation_adapters import (
    SqlConversationHistoryStore,
)
from app.infrastructure.config.conversation import load_conversation_settings
from app.application.services.auth_service import Principal
from app.application.services.conversation_data_context_service import (
    ScopedConversationDataContextService,
)
from app.application.services.conversation_history_service import (
    ConversationHistoryService,
)
from app.domain.access.policy import PolicySnapshot
from app.bootstrap.http_dependencies import (
    provide_auth_use_case,
    provide_current_principal,
    provide_meta_session as provide_bootstrap_meta_session,
    provide_query_capacity,
)
from evals.bootstrap.http import mount_evaluation_http


engine = create_async_engine(
    "sqlite+aiosqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
)
session_factory = async_sessionmaker(engine, expire_on_commit=False)
smoke_data_contexts: dict[tuple[str, str, str], dict] = {}
_conversation_history_settings, _ = load_conversation_settings()


class SmokeQueryService:
    def __init__(self, principal: Principal):
        self.principal = principal

    async def search(
        self,
        query: str,
        chat_id: str | None = None,
        dataset_id: str | None = None,
        dataset_ids: list[str] | None = None,
        compare_dw: bool = False,
        source_mode: str | None = None,
        comparison_intent: str | None = None,
        mapping_confirmation_id: str | None = None,
        data_context_revision: int | None = None,
        enable_analysis: bool | None = None,
    ):
        # Keep the fake provider boundary signature aligned with the production
        # QueryService. HTTP/Pydantic/router wiring remains real even though
        # this smoke intentionally does not execute an external NL2SQL model.
        del (
            dataset_id,
            dataset_ids,
            compare_dw,
            source_mode,
            comparison_intent,
            mapping_confirmation_id,
            data_context_revision,
            enable_analysis,
        )
        resolved_chat_id = chat_id or "real-smoke-chat"
        async with session_factory() as session:
            history_repo = ConversationHistoryRepository(
                session,
                owner_id=self.principal.user_id,
                workspace_id=self.principal.workspace_id,
            )
            await history_repo.start_turn(
                chat_id=resolved_chat_id,
                turn_id="real-smoke-trace",
                user_text=query,
                trace_id="real-smoke-trace",
            )
            history = ChatHistoryMySQL(
                chat_id=resolved_chat_id,
                owner_id=self.principal.user_id,
                workspace_id=self.principal.workspace_id,
                query=query,
                sql="SELECT 1 AS smoke_value",
                result='[{"smoke_value": 1}]',
                row_count=1,
            )
            session.add(history)
            await session.flush()
            await history_repo.complete_turn(
                chat_id=resolved_chat_id,
                turn_id="real-smoke-trace",
                source_history_id=int(history.id),
                row_count=1,
            )
            await session.commit()
        events = [
            {"chat_id": resolved_chat_id, "trace_id": "real-smoke-trace"},
            {"stage": "生成SQL"},
            {"sql": "SELECT 1 AS smoke_value"},
            {"result": [{"smoke_value": 1}]},
            {"trace": {"trace_id": "real-smoke-trace", "status": "success", "total_tokens": 0}},
            {"done": True},
        ]
        for event in events:
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            # Preserve real streaming chunk boundaries through Vite's proxy;
            # a zero-latency synthetic generator may otherwise be coalesced.
            await asyncio.sleep(0.01)

    async def get_history(self, chat_id: str, limit: int | None = None):
        return []


class SmokeConversationDataContextService:
    """SQLite smoke double for the production data-context dependency boundary."""

    def __init__(self, principal: Principal):
        self.principal = principal

    def _key(self, chat_id: str) -> tuple[str, str, str]:
        return (
            self.principal.user_id,
            self.principal.workspace_id,
            chat_id,
        )

    async def get_record(self, chat_id: str) -> dict:
        return dict(
            smoke_data_contexts.get(
                self._key(chat_id),
                {
                    "chat_id": chat_id,
                    "revision": 0,
                    "source_mode": "dw_only",
                    "dataset_ids": [],
                    "comparison_intent": None,
                    "mapping_confirmation_id": None,
                    "invalidation_reason": None,
                    "updated_at": None,
                },
            )
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
        current = await self.get_record(chat_id)
        if (
            expected_revision is not None
            and int(expected_revision) != int(current["revision"])
        ):
            raise ValueError("数据上下文版本已变化")
        record = {
            **current,
            "revision": int(current["revision"]) + 1,
            "source_mode": source_mode,
            "dataset_ids": list(dataset_ids),
            "comparison_intent": comparison_intent,
            "mapping_confirmation_id": mapping_confirmation_id,
            "invalidation_reason": None,
        }
        smoke_data_contexts[self._key(chat_id)] = record
        return {**record, "resolved_context": {"source_mode": source_mode}}

    async def reset(
        self, chat_id: str, *, expected_revision: int | None
    ) -> dict:
        return await self.set(
            chat_id,
            source_mode="dw_only",
            dataset_ids=[],
            comparison_intent=None,
            mapping_confirmation_id=None,
            expected_revision=expected_revision,
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    app_config.auth.secret_key = "real-backend-smoke-secret-key-at-least-32-characters"
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


app = FastAPI(lifespan=lifespan)
app.include_router(auth_router)
app.include_router(query_router)
app.include_router(conversation_router)
mount_evaluation_http(app)


async def smoke_meta_session():
    async with session_factory() as session:
        yield session


app.dependency_overrides[get_meta_session] = smoke_meta_session
app.dependency_overrides[provide_bootstrap_meta_session] = smoke_meta_session
app.dependency_overrides[get_auth_use_case] = provide_auth_use_case
app.dependency_overrides[get_current_principal] = provide_current_principal
app.dependency_overrides[get_query_capacity] = provide_query_capacity


def smoke_query_service(
    principal: Principal = Depends(get_current_principal),
):
    return SmokeQueryService(principal)


app.dependency_overrides[get_query_service] = smoke_query_service


def smoke_conversation_history_service(
    session=Depends(get_meta_session),
    principal: Principal = Depends(get_current_principal),
):
    return ConversationHistoryService(
        SqlConversationHistoryStore(
            session,
            owner_id=principal.user_id,
            workspace_id=principal.workspace_id,
        ),
        access_policy=PolicySnapshot(
            policies=(),
            user_id=principal.user_id,
            username=principal.username,
            workspace_id=principal.workspace_id,
            role=principal.role,
        ),
        masking_rules=(),
        settings=_conversation_history_settings,
    )


app.dependency_overrides[get_conversation_history_service] = (
    smoke_conversation_history_service
)


def smoke_conversation_data_context_service(
    principal: Principal = Depends(get_current_principal),
):
    return SmokeConversationDataContextService(principal)


app.dependency_overrides[get_conversation_data_context_service] = (
    smoke_conversation_data_context_service
)


class _SmokeConversationScope:
    """使用同一 SQLite 权威会话表执行 owner/workspace scope 校验。"""

    def __init__(self, principal: Principal) -> None:
        self._principal = principal

    async def exists(self, chat_id: str) -> bool:
        async with session_factory() as session:
            repository = ConversationHistoryRepository(
                session,
                owner_id=self._principal.user_id,
                workspace_id=self._principal.workspace_id,
            )
            return await repository.get_session(chat_id) is not None


def smoke_scoped_conversation_data_context_service(
    principal: Principal = Depends(get_current_principal),
):
    return ScopedConversationDataContextService(
        scope=_SmokeConversationScope(principal),
        context=SmokeConversationDataContextService(principal),
    )


app.dependency_overrides[get_scoped_conversation_data_context_service] = (
    smoke_scoped_conversation_data_context_service
)


@app.get("/health")
async def health():
    return {"status": "ok", "database": "sqlite", "provider": "deterministic-smoke"}


def main() -> None:
    port = int(os.getenv("REAL_BACKEND_SMOKE_PORT", "8001"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
