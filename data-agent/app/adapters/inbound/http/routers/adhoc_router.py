from fastapi import (
    APIRouter,
    Depends,
    File,
    Header,
    HTTPException,
    Query,
    UploadFile,
)
from starlette.responses import Response

from app.adapters.inbound.http.dependencies import (
    get_adhoc_dataset_service,
    get_data_source_context_service,
    get_upload_limits,
)
from app.adapters.inbound.http.schemas.upload_schema import DatasetExtendSchema, MappingDecisionSchema
from app.domain.data_sources.context import DataSourceContextError
from app.application.services.adhoc_dataset_service import (
    AdhocDatasetService,
    DatasetCleanupError,
    DatasetQuotaError,
)
from app.application.services.data_source_context_service import DataSourceContextService
from app.application.ports.http_runtime import UploadLimits


adhoc_router = APIRouter()


@adhoc_router.get("/api/uploaded-datasets")
async def list_uploaded_datasets(
    limit: int = Query(default=20, ge=1, le=50),
    cursor: str | None = Query(default=None, max_length=64),
    include_terminal: bool = Query(default=False),
    service: AdhocDatasetService = Depends(get_adhoc_dataset_service),
):
    return await service.list_catalog(
        limit=limit, cursor=cursor, include_terminal=include_terminal
    )


@adhoc_router.get("/api/uploaded-datasets/quota")
async def get_uploaded_dataset_quota(
    service: AdhocDatasetService = Depends(get_adhoc_dataset_service),
):
    return await service.quota()


@adhoc_router.post("/api/uploaded-datasets")
async def create_uploaded_dataset(
    file: UploadFile = File(...),
    retention_mode: str = Query(default="managed", pattern=r"^(temporary|managed)$"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    service: AdhocDatasetService = Depends(get_adhoc_dataset_service),
    limits: UploadLimits = Depends(get_upload_limits),
):
    content = await file.read(limits.max_file_bytes + 1)
    if len(content) > limits.max_file_bytes:
        raise HTTPException(status_code=413, detail="文件大小超过配置上限")
    try:
        return await service.create_dataset(
            file.filename or "upload",
            content,
            idempotency_key=idempotency_key,
            retention_mode=retention_mode,
        )
    except DatasetQuotaError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, DataSourceContextError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@adhoc_router.post("/api/uploaded-datasets/{asset_id}/versions")
async def create_uploaded_dataset_version(
    asset_id: str,
    file: UploadFile = File(...),
    retention_mode: str = Query(default="managed", pattern=r"^(temporary|managed)$"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    service: AdhocDatasetService = Depends(get_adhoc_dataset_service),
    limits: UploadLimits = Depends(get_upload_limits),
):
    content = await file.read(limits.max_file_bytes + 1)
    if len(content) > limits.max_file_bytes:
        raise HTTPException(status_code=413, detail="文件大小超过配置上限")
    try:
        return await service.create_dataset(
            file.filename or "upload",
            content,
            idempotency_key=idempotency_key,
            retention_mode=retention_mode,
            asset_id=asset_id,
        )
    except DatasetQuotaError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, DataSourceContextError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@adhoc_router.get("/api/uploaded-datasets/{dataset_id}")
async def get_uploaded_dataset(
    dataset_id: str,
    service: AdhocDatasetService = Depends(get_adhoc_dataset_service),
):
    item = await service.get_scoped_dataset(dataset_id)
    if item is None:
        raise HTTPException(status_code=404, detail="上传数据集不存在或无权访问")
    return service.public_dict(item)


@adhoc_router.get("/api/uploaded-datasets/{dataset_id}/preview")
async def preview_uploaded_dataset(
    dataset_id: str,
    service: AdhocDatasetService = Depends(get_adhoc_dataset_service),
):
    item = await service.get_dataset(dataset_id)
    if item is None:
        raise HTTPException(status_code=404, detail="上传数据集不可查询或已过期")
    payload = service.public_dict(item)
    return {
        "dataset_id": payload["dataset_id"],
        "version_id": payload["version_id"],
        "columns": payload["columns"],
        "rows": payload["preview"],
        "bounded": True,
    }


@adhoc_router.post("/api/uploaded-datasets/{dataset_id}/extend")
async def extend_uploaded_dataset(
    dataset_id: str,
    body: DatasetExtendSchema,
    service: AdhocDatasetService = Depends(get_adhoc_dataset_service),
):
    result = await service.extend_dataset(dataset_id, body.seconds)
    if result is None:
        raise HTTPException(status_code=404, detail="上传数据集不可续期")
    return result


@adhoc_router.post("/api/uploaded-datasets/{dataset_id}/archive")
async def archive_uploaded_dataset(
    dataset_id: str,
    service: AdhocDatasetService = Depends(get_adhoc_dataset_service),
):
    result = await service.archive_dataset(dataset_id)
    if result is None:
        raise HTTPException(status_code=404, detail="上传数据集不可归档")
    return result


@adhoc_router.delete("/api/uploaded-datasets/{dataset_id}", status_code=204)
async def delete_uploaded_dataset(
    dataset_id: str,
    service: AdhocDatasetService = Depends(get_adhoc_dataset_service),
):
    try:
        if not await service.delete_dataset(dataset_id):
            raise HTTPException(status_code=404, detail="上传数据集不存在")
    except DatasetCleanupError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return Response(status_code=204)


@adhoc_router.post("/api/uploaded-datasets/reconcile")
async def reconcile_uploaded_datasets(
    service: AdhocDatasetService = Depends(get_adhoc_dataset_service),
):
    return await service.reconcile()


@adhoc_router.post("/api/source-mappings/{mapping_id}/confirm")
async def confirm_source_mapping(
    mapping_id: str,
    service: DataSourceContextService = Depends(get_data_source_context_service),
):
    try:
        result = await service.confirm_mapping(mapping_id)
    except DataSourceContextError as exc:
        raise HTTPException(status_code=409, detail=exc.to_dict()) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="关联映射不存在或无权访问")
    return result


@adhoc_router.post("/api/source-mappings/{mapping_id}/reject")
async def reject_source_mapping(
    mapping_id: str,
    body: MappingDecisionSchema,
    service: DataSourceContextService = Depends(get_data_source_context_service),
):
    result = await service.reject_mapping(mapping_id, body.reason)
    if result is None:
        raise HTTPException(status_code=404, detail="关联映射不存在或无权访问")
    return result


@adhoc_router.get("/api/source-mappings/{mapping_id}/evidence")
async def get_source_mapping_evidence(
    mapping_id: str,
    service: DataSourceContextService = Depends(get_data_source_context_service),
):
    result = await service.mapping_evidence(mapping_id)
    if result is None:
        raise HTTPException(status_code=404, detail="关联映射不存在或无权访问")
    return result
