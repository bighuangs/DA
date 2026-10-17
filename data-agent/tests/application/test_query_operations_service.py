"""手动 SQL 与 Trace Application 用例回归。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.application.services.query_operations_service import (
    QueryConversationNotFoundError,
    QueryOperationsService,
)
from app.domain.data_sources.context import AssetAccessContext, SourceMode


class _Scope:
    async def exists(self, chat_id: str) -> bool:
        return chat_id == "chat-1"


class _Traces:
    async def list_by_chat_id(self, chat_id: str, limit: int):
        return [{"chat_id": chat_id, "limit": limit}]


class _Query:
    last_cache_hit = True

    def __init__(self) -> None:
        self.validated: list[str] = []

    async def validate_sql(self, sql: str) -> None:
        self.validated.append(sql)

    async def execute_sql(self, sql: str):
        assert sql == "SELECT amount FROM fact_order"
        return [{"amount": 12}]


class _SourceContext:
    async def resolve(self, **options):
        self.options = options
        return SimpleNamespace(
            access=AssetAccessContext(
                workspace_id="workspace-1",
                owner_id="user-1",
                source_mode=SourceMode.DW_ONLY,
                official_tables=frozenset({"fact_order"}),
            )
        )


class _Policy:
    def authorize_sql(self, sql: str) -> str:
        return sql


def _service() -> QueryOperationsService:
    return QueryOperationsService(
        scope=_Scope(),
        traces=_Traces(),
        query=_Query(),
        source_context=_SourceContext(),
        access_policy=_Policy(),
    )


@pytest.mark.asyncio
async def test_query_operations_enforces_scope_before_reading_trace() -> None:
    service = _service()

    assert await service.trace("chat-1", limit=5) == [
        {"chat_id": "chat-1", "limit": 5}
    ]
    with pytest.raises(QueryConversationNotFoundError):
        await service.trace("missing", limit=5)


@pytest.mark.asyncio
async def test_query_operations_owns_manual_sql_safety_chain() -> None:
    service = _service()

    result = await service.execute_sql(
        "SELECT amount FROM fact_order",
        dataset_id=None,
        dataset_ids=None,
        source_mode="dw_only",
        compare_dw=False,
        comparison_intent=None,
        mapping_confirmation_id=None,
    )

    assert result.columns == ("amount",)
    assert result.rows == ({"amount": 12},)
    assert result.cache_hit is True
