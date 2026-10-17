from fastapi import APIRouter, Depends, HTTPException, Query
from starlette.responses import Response, StreamingResponse

from app.adapters.inbound.http.dependencies import (
    get_current_principal,
    get_export_limits,
    get_masking_policy,
    get_masking_rules,
    get_query_capacity,
    get_query_operations_service,
    get_query_service,
)
from app.adapters.inbound.http.schemas.query_schema import ExecSqlSchema, ExportSchema, QuerySchema
from app.application.ports.http_runtime import QueryCapacityExceeded, QueryCapacityPort
from app.application.services.export_service import (
    ExportLimitError,
    ExportLimits,
    build_xlsx,
    safe_filename,
    validate_export_shape,
)
from app.application.services.query_service import QueryService
from app.application.services.query_operations_service import (
    QueryConversationNotFoundError,
    QueryOperationsService,
)
from app.application.services.masking_service import MaskingRule, mask_rows
from app.application.services.auth_service import Principal
from app.domain.privacy.masking import MaskingPolicy
from app.domain.data_sources.context import DataSourceContextError

# 创建查询的路由器
query_router = APIRouter()


async def _guarded_search(lease, stream):
    """Keep the capacity lease for the complete SSE lifetime, including cancellation."""
    async with lease:
        async for chunk in stream:
            yield chunk


# 注册搜索的路由
@query_router.post("/api/query")
async def search(
    query_schema: QuerySchema,
    service: QueryService = Depends(get_query_service),
    principal: Principal = Depends(get_current_principal),
    capacity: QueryCapacityPort = Depends(get_query_capacity),
):
    """
    问数接口（SSE 流式）。

    业务逻辑：
        1. 接收 {query, chat_id?}，chat_id 不传则由 Service 新建会话
        2. 以 text/event-stream 流式返回阶段进度 / SQL / 结果 / 错误
        3. 首个 chunk 为 {"chat_id": "..."}（新建会话时），供前端后续轮次复用
    """
    try:
        lease = await capacity.acquire(principal.workspace_id)
    except QueryCapacityExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "2"}) from exc
    return StreamingResponse(
        _guarded_search(lease, service.search(
            query=query_schema.query,
            chat_id=query_schema.chat_id,
            dataset_id=query_schema.dataset_id,
            dataset_ids=query_schema.dataset_ids,
            compare_dw=query_schema.compare_dw,
            source_mode=query_schema.source_mode,
            comparison_intent=query_schema.comparison_intent,
            mapping_confirmation_id=query_schema.mapping_confirmation_id,
            data_context_revision=query_schema.data_context_revision,
            enable_analysis=query_schema.enable_analysis,
        )),
        media_type="text/event-stream"
    )


# 获取指定会话的对话历史
@query_router.get("/api/chat/{chat_id}/history")
async def get_history(
    chat_id: str,
    limit: int | None = Query(default=None, ge=1, le=100),
    service: QueryService = Depends(get_query_service),
):
    """
    获取指定会话的对话历史（最近 limit 轮，默认取配置 chat.history_limit）。

    Returns:
        list[dict]: 历史记录列表，每项含 query/sql/result/row_count/created_at
    """
    result = await service.get_history(chat_id, limit)
    if result is None:
        raise HTTPException(status_code=404, detail="会话不存在或无权访问")
    return result


@query_router.get("/api/chat/{chat_id}/memory")
async def get_chat_memory(
    chat_id: str,
    service: QueryService = Depends(get_query_service),
):
    """Return only the authenticated owner's scoped, already-masked memory state."""
    result = await service.get_memory_status(chat_id)
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="会话不存在或无权访问")
    return result


@query_router.delete("/api/chat/{chat_id}/memory")
async def reset_chat_memory(
    chat_id: str,
    include_long_term: bool = Query(default=False),
    service: QueryService = Depends(get_query_service),
):
    """Reset current transcript/working state; long-term items require explicit opt-in."""
    result = await service.reset_memory(
        chat_id, include_long_term=include_long_term
    )
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="会话不存在或无权访问")
    return result


@query_router.get("/api/chat/{chat_id}/trace")
async def get_query_trace(
    chat_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    service: QueryOperationsService = Depends(get_query_operations_service),
):
    """查询指定会话的步骤时间线和 Token 统计。"""
    try:
        return await service.trace(chat_id, limit=limit)
    except QueryConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# 手动编辑 SQL 重跑（TODO-22）：复用 dw_mysql_repo 的只读校验 + enforce_limit 安全
@query_router.post("/api/query/exec")
async def exec_sql(
    body: ExecSqlSchema,
    response: Response,
    service: QueryOperationsService = Depends(get_query_operations_service),
):
    """
    手动编辑 SQL 重跑接口。

    业务逻辑：
        1. 接收 {sql}（前端 SQL 卡片编辑后的 SQL）
        2. 先执行只读校验、EXPLAIN 语法校验和查询成本守卫
        3. 调 dw_repo.execute_sql：内部再次 check_sql_read，并 enforce_limit 强制行数上限
        4. 返回 {columns, rows}；SQL 不合法/越权/成本过高时返回 400 + 错误信息
    """
    try:
        result = await service.execute_sql(
            body.sql,
            dataset_id=body.dataset_id,
            dataset_ids=body.dataset_ids,
            source_mode=body.source_mode,
            compare_dw=body.compare_dw,
            comparison_intent=body.comparison_intent,
            mapping_confirmation_id=body.mapping_confirmation_id,
        )
        response.headers["X-Result-Cache"] = "HIT" if result.cache_hit else "MISS"
        return {"columns": list(result.columns), "rows": list(result.rows)}
    except DataSourceContextError as e:
        raise HTTPException(status_code=409, detail=e.to_dict()) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@query_router.post("/api/query/export")
async def export_query_result(
    body: ExportSchema,
    principal: Principal = Depends(get_current_principal),
    policy: MaskingPolicy = Depends(get_masking_policy),
    rules: tuple[MaskingRule, ...] = Depends(get_masking_rules),
    limits: ExportLimits = Depends(get_export_limits),
):
    """导出当前结果快照为 XLSX，不重新执行 SQL。"""
    columns = body.columns or (list(body.rows[0].keys()) if body.rows else [])
    try:
        validate_export_shape(columns, body.rows, limits=limits)
    except ExportLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    content = build_xlsx(
        columns,
        mask_rows(body.rows, columns=columns, rules=rules, policy=policy),
    )
    filename = f"{safe_filename(body.filename)}.xlsx"
    return Response(
        content=content,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
