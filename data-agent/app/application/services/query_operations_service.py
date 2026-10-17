"""Trace 查询与受治理手动 SQL 执行用例。"""

from __future__ import annotations

from dataclasses import dataclass

from app.application.ports.query_operations import (
    ConversationScopeReader,
    ManualQueryExecutor,
    QueryTraceReader,
)
from app.application.services.data_source_context_service import (
    DataSourceContextService,
)
from app.domain.access.policy import PolicySnapshot
from app.domain.data_sources.context import (
    authorize_sql_assets,
    translate_logical_upload_tables,
)


class QueryConversationNotFoundError(LookupError):
    """Trace 对应会话不存在或不属于当前主体。"""


@dataclass(frozen=True, slots=True)
class ManualQueryResult:
    columns: tuple[str, ...]
    rows: tuple[dict[str, object], ...]
    cache_hit: bool


class QueryOperationsService:
    """把 Trace scope 与手动 SQL 安全链从 HTTP Adapter 中移出。"""

    def __init__(
        self,
        *,
        scope: ConversationScopeReader,
        traces: QueryTraceReader,
        query: ManualQueryExecutor,
        source_context: DataSourceContextService,
        access_policy: PolicySnapshot,
    ) -> None:
        self._scope = scope
        self._traces = traces
        self._query = query
        self._source_context = source_context
        self._access_policy = access_policy

    async def trace(self, chat_id: str, *, limit: int) -> list[dict[str, object]]:
        if not await self._scope.exists(chat_id):
            raise QueryConversationNotFoundError("会话不存在或无权访问")
        return await self._traces.list_by_chat_id(chat_id, limit)

    async def execute_sql(
        self,
        sql: str,
        *,
        dataset_id: str | None,
        dataset_ids: list[str] | None,
        source_mode: str | None,
        compare_dw: bool,
        comparison_intent: str | None,
        mapping_confirmation_id: str | None,
    ) -> ManualQueryResult:
        resolved = await self._source_context.resolve(
            dataset_id=dataset_id,
            dataset_ids=dataset_ids,
            source_mode=source_mode,
            compare_dw=compare_dw,
            comparison_intent=comparison_intent,
            mapping_confirmation_id=mapping_confirmation_id,
        )
        physical_sql = translate_logical_upload_tables(sql, resolved.access)
        authorized_sql = self._access_policy.authorize_sql(physical_sql)
        authorize_sql_assets(authorized_sql, resolved.access)
        await self._query.validate_sql(authorized_sql)
        rows = await self._query.execute_sql(authorized_sql)
        columns = tuple(str(key) for key in rows[0]) if rows else ()
        return ManualQueryResult(
            columns=columns,
            rows=tuple(dict(row) for row in rows),
            cache_hit=bool(self._query.last_cache_hit),
        )


__all__ = [
    "ManualQueryResult",
    "QueryConversationNotFoundError",
    "QueryOperationsService",
]
