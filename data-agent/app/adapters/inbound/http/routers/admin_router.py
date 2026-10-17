"""元数据管理 HTTP 入站适配器。"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.adapters.inbound.http.dependencies import (
    get_meta_admin_service,
    get_schema_sync_service,
)
from app.adapters.inbound.http.schemas.meta_schema import (
    AccessPolicyPatchSchema,
    AccessPolicySchema,
    MaskingRulePatchSchema,
    MaskingRuleSchema,
    MetricPatchSchema,
    MetricSchema,
    RebuildSchema,
    SchemaSyncSchema,
    TrainingPatchSchema,
    TrainingSchema,
    ValueIndexRollbackSchema,
)
from app.application.services.meta_admin_service import (
    AdminConflictError,
    AdminForbiddenError,
    AdminNotFoundError,
    AdminUseCaseError,
    AdminValidationError,
    MetaAdminService,
)
from app.application.services.schema_sync_service import SchemaSyncService


admin_router = APIRouter(prefix="/api/admin/meta", tags=["meta-admin"])


def _status_for(error: AdminUseCaseError) -> int:
    if isinstance(error, AdminConflictError):
        return 409
    if isinstance(error, AdminNotFoundError):
        return 404
    if isinstance(error, AdminValidationError):
        return 422
    if isinstance(error, AdminForbiddenError):
        return 403
    return 500


async def _call(operation: Awaitable[Any]) -> Any:
    """把 Application typed error 映射为稳定 HTTP 状态。"""

    try:
        return await operation
    except AdminUseCaseError as exc:
        raise HTTPException(status_code=_status_for(exc), detail=str(exc)) from exc


@admin_router.get("/metrics")
async def list_metrics(
    kind: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None),
    service: MetaAdminService = Depends(get_meta_admin_service),
):
    return await service.list_metrics(kind=kind, search=search)


@admin_router.get("/overview")
async def overview(service: MetaAdminService = Depends(get_meta_admin_service)):
    return await service.overview()


@admin_router.post("/metrics", status_code=201)
async def create_metric(
    body: MetricSchema,
    service: MetaAdminService = Depends(get_meta_admin_service),
):
    return await _call(service.create_metric(**body.model_dump()))


@admin_router.patch("/metrics/{metric_id}")
async def update_metric(
    metric_id: str,
    body: MetricPatchSchema,
    service: MetaAdminService = Depends(get_meta_admin_service),
):
    return await _call(service.update_metric(metric_id, **body.model_dump()))


@admin_router.delete("/metrics/{metric_id}", status_code=204)
async def delete_metric(
    metric_id: str,
    service: MetaAdminService = Depends(get_meta_admin_service),
):
    await _call(service.delete_metric(metric_id))


@admin_router.get("/training")
async def list_training(
    search: Optional[str] = Query(default=None),
    service: MetaAdminService = Depends(get_meta_admin_service),
):
    return await service.list_training(search=search)


@admin_router.post("/training", status_code=201)
async def create_training(
    body: TrainingSchema,
    service: MetaAdminService = Depends(get_meta_admin_service),
):
    return await service.create_training(**body.model_dump())


@admin_router.patch("/training/{training_id}")
async def update_training(
    training_id: int,
    body: TrainingPatchSchema,
    service: MetaAdminService = Depends(get_meta_admin_service),
):
    return await _call(service.update_training(training_id, **body.model_dump()))


@admin_router.delete("/training/{training_id}", status_code=204)
async def delete_training(
    training_id: int,
    service: MetaAdminService = Depends(get_meta_admin_service),
):
    await _call(service.delete_training(training_id))


@admin_router.get("/tables")
async def list_tables(service: MetaAdminService = Depends(get_meta_admin_service)):
    return await service.list_tables()


@admin_router.get("/masking/rules")
async def list_masking_rules(
    service: MetaAdminService = Depends(get_meta_admin_service),
):
    return await service.list_masking_rules()


@admin_router.get("/masking/policy")
async def get_masking_policy(
    service: MetaAdminService = Depends(get_meta_admin_service),
):
    return service.get_masking_policy()


@admin_router.post("/masking/rules", status_code=201)
async def create_masking_rule(
    body: MaskingRuleSchema,
    service: MetaAdminService = Depends(get_meta_admin_service),
):
    return await service.create_masking_rule(**body.model_dump())


@admin_router.patch("/masking/rules/{rule_id}")
async def update_masking_rule(
    rule_id: int,
    body: MaskingRulePatchSchema,
    service: MetaAdminService = Depends(get_meta_admin_service),
):
    return await _call(service.update_masking_rule(rule_id, **body.model_dump()))


@admin_router.delete("/masking/rules/{rule_id}", status_code=204)
async def delete_masking_rule(
    rule_id: int,
    service: MetaAdminService = Depends(get_meta_admin_service),
):
    await _call(service.delete_masking_rule(rule_id))


@admin_router.get("/access-policies")
async def list_access_policies(
    service: MetaAdminService = Depends(get_meta_admin_service),
):
    return await service.list_access_policies()


@admin_router.post("/access-policies", status_code=201)
async def create_access_policy(
    body: AccessPolicySchema,
    service: MetaAdminService = Depends(get_meta_admin_service),
):
    return await _call(service.create_access_policy(**body.model_dump()))


@admin_router.patch("/access-policies/{policy_id}")
async def update_access_policy(
    policy_id: int,
    body: AccessPolicyPatchSchema,
    service: MetaAdminService = Depends(get_meta_admin_service),
):
    return await _call(service.update_access_policy(policy_id, **body.model_dump()))


@admin_router.delete("/access-policies/{policy_id}", status_code=204)
async def delete_access_policy(
    policy_id: int,
    service: MetaAdminService = Depends(get_meta_admin_service),
):
    await _call(service.delete_access_policy(policy_id))


@admin_router.post("/rebuild")
async def rebuild(
    body: RebuildSchema,
    service: MetaAdminService = Depends(get_meta_admin_service),
):
    return await service.rebuild(body.target)


@admin_router.get("/hybrid/status")
async def hybrid_index_status(
    service: MetaAdminService = Depends(get_meta_admin_service),
):
    """返回模型与三类 collection 的公开合同。"""

    return await service.hybrid_status()


@admin_router.get("/value-index/status")
async def value_index_status(
    service: MetaAdminService = Depends(get_meta_admin_service),
):
    """返回当前工作空间字段值索引的 active 版本与治理状态。"""

    return await service.value_index_status()


@admin_router.get("/value-index/search")
async def search_value_index(
    keyword: str = Query(min_length=1, max_length=255),
    query_mode: str = Query(default="full_text", pattern="^(exact|prefix|full_text)$"),
    table_id: str | None = Query(default=None, max_length=128),
    column_id: str | None = Query(default=None, max_length=255),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    service: MetaAdminService = Depends(get_meta_admin_service),
):
    """按受控模式检索 ES 字段值；权限过滤由 Application 执行。"""

    return await _call(
        service.search_values(
            keyword,
            query_mode=query_mode,
            table_id=table_id,
            column_id=column_id,
            page=page,
            page_size=page_size,
        )
    )


@admin_router.post("/value-index/rollback")
async def rollback_value_index(
    body: ValueIndexRollbackSchema,
    service: MetaAdminService = Depends(get_meta_admin_service),
):
    """独立回退 ES value active alias，不改动 Qdrant RAG profile。"""

    return await _call(service.rollback_value_index(body.physical_index))


@admin_router.get("/schema/preview")
async def schema_preview(
    sync_service: SchemaSyncService = Depends(get_schema_sync_service),
):
    """读取 DW 当前 Schema 并返回差异，不修改 meta 或索引。"""

    return await sync_service.preview()


@admin_router.post("/schema/sync")
async def schema_sync(
    body: SchemaSyncSchema,
    sync_service: SchemaSyncService = Depends(get_schema_sync_service),
):
    """发布 DW Schema 快照，并重建字段检索索引。"""

    return await sync_service.sync(rebuild_value_index=body.rebuild_value_index)


@admin_router.get("/schema/history")
async def schema_sync_history(
    limit: int = Query(default=10, ge=1, le=50),
    service: MetaAdminService = Depends(get_meta_admin_service),
):
    return await service.schema_sync_history(limit)
