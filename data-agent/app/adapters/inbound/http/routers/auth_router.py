"""认证与工作空间 HTTP 协议映射。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.adapters.inbound.http.dependencies import (
    get_auth_use_case,
    get_current_principal,
    require_roles,
)
from app.adapters.inbound.http.schemas.auth_schema import (
    LoginSchema,
    MemberRoleSchema,
    MemberSchema,
    RegisterSchema,
    SwitchWorkspaceSchema,
    WorkspaceSchema,
)
from app.application.services.auth_service import Principal
from app.application.services.auth_use_case import (
    AuthAuthenticationError,
    AuthAuthorizationError,
    AuthConflictError,
    AuthNotFoundError,
    AuthSessionDto,
    AuthUseCase,
    AuthUseCaseError,
    AuthValidationError,
    MemberMutationDto,
    WorkspaceContextDto,
    WorkspaceListItemDto,
    WorkspaceMemberDto,
)


auth_router = APIRouter(tags=["auth"])


def _http_error(error: AuthUseCaseError) -> HTTPException:
    if isinstance(error, AuthConflictError):
        status_code = 409
    elif isinstance(error, AuthValidationError):
        status_code = 400
    elif isinstance(error, AuthAuthenticationError):
        status_code = 401
    elif isinstance(error, AuthAuthorizationError):
        status_code = 403
    elif isinstance(error, AuthNotFoundError):
        status_code = 404
    else:  # pragma: no cover - 新增错误类型必须先声明协议映射
        status_code = 500
    return HTTPException(status_code=status_code, detail=str(error))


def _workspace_context_response(item: WorkspaceContextDto) -> dict[str, object]:
    return {"id": item.id, "name": item.name, "role": item.role}


def _workspace_response(item: WorkspaceListItemDto) -> dict[str, object]:
    return {
        "id": item.id,
        "name": item.name,
        "owner_id": item.owner_id,
        "role": item.role,
        "created_at": item.created_at,
    }


def _member_response(item: WorkspaceMemberDto) -> dict[str, object]:
    return {
        "user_id": item.user_id,
        "username": item.username,
        "display_name": item.display_name,
        "role": item.role,
        "created_at": item.created_at,
    }


def _member_mutation_response(item: MemberMutationDto) -> dict[str, object]:
    response: dict[str, object] = {"success": item.success}
    if item.user_id is not None:
        response["user_id"] = item.user_id
    return response


def _auth_response(result: AuthSessionDto) -> dict[str, object]:
    return {
        "access_token": result.access_token,
        "token_type": result.token_type,
        "user": {
            "id": result.user.id,
            "username": result.user.username,
            "display_name": result.user.display_name,
            "workspace": _workspace_context_response(result.user.workspace),
        },
        "workspaces": [_workspace_response(item) for item in result.workspaces],
    }


@auth_router.post("/api/auth/register", status_code=201)
async def register(
    body: RegisterSchema,
    use_case: AuthUseCase = Depends(get_auth_use_case),
):
    try:
        result = await use_case.register(
            username=body.username,
            password=body.password,
            display_name=body.display_name,
            workspace_name=body.workspace_name,
        )
    except AuthUseCaseError as exc:
        raise _http_error(exc) from exc
    return _auth_response(result)


@auth_router.post("/api/auth/login")
async def login(
    body: LoginSchema,
    use_case: AuthUseCase = Depends(get_auth_use_case),
):
    try:
        result = await use_case.login(
            username=body.username,
            password=body.password,
        )
    except AuthUseCaseError as exc:
        raise _http_error(exc) from exc
    return _auth_response(result)


@auth_router.get("/api/auth/me")
async def me(principal: Principal = Depends(get_current_principal)):
    return {
        "user": {
            "id": principal.user_id,
            "username": principal.username,
            "display_name": principal.display_name,
        },
        "workspace": {
            "id": principal.workspace_id,
            "name": principal.workspace_name,
            "role": principal.role,
        },
    }


@auth_router.post("/api/auth/switch-workspace")
async def switch_workspace(
    body: SwitchWorkspaceSchema,
    principal: Principal = Depends(get_current_principal),
    use_case: AuthUseCase = Depends(get_auth_use_case),
):
    try:
        result = await use_case.switch_workspace(
            principal=principal,
            workspace_id=body.workspace_id,
        )
    except AuthUseCaseError as exc:
        raise _http_error(exc) from exc
    return _auth_response(result)


@auth_router.get("/api/workspaces")
async def list_workspaces(
    principal: Principal = Depends(get_current_principal),
    use_case: AuthUseCase = Depends(get_auth_use_case),
):
    result = await use_case.list_workspaces(principal=principal)
    return [_workspace_response(item) for item in result]


@auth_router.post("/api/workspaces")
async def create_workspace(
    body: WorkspaceSchema,
    principal: Principal = Depends(get_current_principal),
    use_case: AuthUseCase = Depends(get_auth_use_case),
):
    result = await use_case.create_workspace(principal=principal, name=body.name)
    return _workspace_context_response(result)


@auth_router.get("/api/workspaces/current/members")
async def list_members(
    principal: Principal = Depends(get_current_principal),
    use_case: AuthUseCase = Depends(get_auth_use_case),
):
    result = await use_case.list_members(principal=principal)
    return [_member_response(item) for item in result]


@auth_router.post("/api/workspaces/current/members")
async def add_member(
    body: MemberSchema,
    principal: Principal = Depends(require_roles("owner", "admin")),
    use_case: AuthUseCase = Depends(get_auth_use_case),
):
    try:
        result = await use_case.add_member(
            principal=principal,
            username=body.username,
            role=body.role,
        )
    except AuthUseCaseError as exc:
        raise _http_error(exc) from exc
    return _member_mutation_response(result)


@auth_router.patch("/api/workspaces/current/members/{user_id}")
async def update_member(
    user_id: str,
    body: MemberRoleSchema,
    principal: Principal = Depends(require_roles("owner", "admin")),
    use_case: AuthUseCase = Depends(get_auth_use_case),
):
    try:
        result = await use_case.update_member(
            principal=principal,
            user_id=user_id,
            role=body.role,
        )
    except AuthUseCaseError as exc:
        raise _http_error(exc) from exc
    return _member_mutation_response(result)


@auth_router.delete("/api/workspaces/current/members/{user_id}", status_code=204)
async def remove_member(
    user_id: str,
    principal: Principal = Depends(require_roles("owner", "admin")),
    use_case: AuthUseCase = Depends(get_auth_use_case),
) -> None:
    try:
        await use_case.remove_member(principal=principal, user_id=user_id)
    except AuthUseCaseError as exc:
        raise _http_error(exc) from exc
