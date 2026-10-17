"""认证 HTTP Adapter 的响应、错误与角色守卫合同。"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.adapters.inbound.http.dependencies import (
    get_auth_use_case,
    get_current_principal,
)
from app.adapters.inbound.http.routers.auth_router import auth_router
from app.application.services.auth_service import Principal
from app.application.services.auth_use_case import (
    AuthAuthenticationError,
    AuthAuthorizationError,
    AuthConflictError,
    AuthNotFoundError,
    AuthSessionDto,
    AuthValidationError,
    AuthenticatedUserDto,
    MemberMutationDto,
    WorkspaceContextDto,
    WorkspaceListItemDto,
)


def _session() -> AuthSessionDto:
    workspace = WorkspaceContextDto(
        id="workspace-1",
        name="主空间",
        role="owner",
    )
    return AuthSessionDto(
        access_token="signed-token",
        token_type="bearer",
        user=AuthenticatedUserDto(
            id="user-1",
            username="alice",
            display_name="Alice",
            workspace=workspace,
        ),
        workspaces=(
            WorkspaceListItemDto(
                id=workspace.id,
                name=workspace.name,
                owner_id="user-1",
                role=workspace.role,
                created_at=None,
            ),
        ),
    )


class _UseCase:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.add_member_called = False

    async def register(self, **values: object) -> AuthSessionDto:
        if self.error is not None:
            raise self.error
        return _session()

    async def add_member(self, **values: object) -> MemberMutationDto:
        self.add_member_called = True
        return MemberMutationDto(user_id="user-2")


def _principal(*, role: str = "owner") -> Principal:
    return Principal(
        user_id="user-1",
        username="alice",
        display_name="Alice",
        workspace_id="workspace-1",
        workspace_name="主空间",
        role=role,
    )


def _test_app(use_case: _UseCase, *, role: str = "owner") -> FastAPI:
    app = FastAPI()
    app.include_router(auth_router)
    app.dependency_overrides[get_auth_use_case] = lambda: use_case
    app.dependency_overrides[get_current_principal] = lambda: _principal(role=role)
    return app


def test_register_maps_application_dto_to_existing_http_shape() -> None:
    with TestClient(_test_app(_UseCase())) as client:
        response = client.post(
            "/api/auth/register",
            json={
                "username": "alice",
                "password": "password-123",
                "display_name": "Alice",
                "workspace_name": "主空间",
            },
        )

    assert response.status_code == 201
    assert response.json() == {
        "access_token": "signed-token",
        "token_type": "bearer",
        "user": {
            "id": "user-1",
            "username": "alice",
            "display_name": "Alice",
            "workspace": {
                "id": "workspace-1",
                "name": "主空间",
                "role": "owner",
            },
        },
        "workspaces": [
            {
                "id": "workspace-1",
                "name": "主空间",
                "owner_id": "user-1",
                "role": "owner",
                "created_at": None,
            }
        ],
    }


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (AuthValidationError("校验失败"), 400),
        (AuthAuthenticationError("认证失败"), 401),
        (AuthAuthorizationError("没有权限"), 403),
        (AuthNotFoundError("对象不存在"), 404),
        (AuthConflictError("资源冲突"), 409),
    ],
)
def test_router_maps_typed_application_errors(
    error: Exception,
    status_code: int,
) -> None:
    with TestClient(_test_app(_UseCase(error))) as client:
        response = client.post(
            "/api/auth/register",
            json={
                "username": "alice",
                "password": "password-123",
                "workspace_name": "主空间",
            },
        )

    assert response.status_code == status_code
    assert response.json() == {"detail": str(error)}


def test_member_role_dependency_rejects_viewer_before_use_case() -> None:
    use_case = _UseCase()
    with TestClient(_test_app(use_case, role="viewer")) as client:
        response = client.post(
            "/api/workspaces/current/members",
            json={"username": "bob", "role": "analyst"},
        )

    assert response.status_code == 403
    assert response.json() == {"detail": "需要角色：owner / admin"}
    assert use_case.add_member_called is False
