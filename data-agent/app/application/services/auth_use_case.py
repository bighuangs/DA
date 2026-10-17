"""认证、工作空间与成员管理的应用用例。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from app.application.ports.http_runtime import (
    AuthRepositoryPort,
    AuthUserView,
)
from app.application.services.auth_service import (
    AuthTokenService,
    PasswordPolicyError,
    Principal,
    hash_password,
    validate_password,
    verify_password,
)


class AuthUseCaseError(Exception):
    """认证用例可由入站 Adapter 显式映射的预期错误。"""


class AuthConflictError(AuthUseCaseError):
    """认证资源与现有状态冲突。"""


class AuthValidationError(AuthUseCaseError):
    """认证操作违反业务校验规则。"""


class AuthAuthenticationError(AuthUseCaseError):
    """用户凭据无法通过认证。"""


class AuthAuthorizationError(AuthUseCaseError):
    """当前主体无权执行目标认证操作。"""


class AuthNotFoundError(AuthUseCaseError):
    """认证操作所需的用户或成员不存在。"""


@dataclass(frozen=True, slots=True)
class WorkspaceContextDto:
    """当前认证会话中的工作空间上下文。"""

    id: str
    name: str
    role: str


@dataclass(frozen=True, slots=True)
class AuthenticatedUserDto:
    """认证成功后可返回给入站 Adapter 的用户视图。"""

    id: str
    username: str
    display_name: str
    workspace: WorkspaceContextDto


@dataclass(frozen=True, slots=True)
class WorkspaceListItemDto:
    """用户可访问工作空间的稳定输出合同。"""

    id: str
    name: str
    owner_id: str
    role: str
    created_at: str | None


@dataclass(frozen=True, slots=True)
class WorkspaceMemberDto:
    """工作空间成员列表的稳定输出合同。"""

    user_id: str
    username: str
    display_name: str
    role: str
    created_at: str | None


@dataclass(frozen=True, slots=True)
class AuthSessionDto:
    """注册、登录或切换工作空间后的认证结果。"""

    access_token: str
    token_type: str
    user: AuthenticatedUserDto
    workspaces: tuple[WorkspaceListItemDto, ...]


@dataclass(frozen=True, slots=True)
class MemberMutationDto:
    """成员写操作的确认结果。"""

    success: bool = True
    user_id: str | None = None


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


def _workspace_item(item: Mapping[str, object]) -> WorkspaceListItemDto:
    return WorkspaceListItemDto(
        id=str(item["id"]),
        name=str(item["name"]),
        owner_id=str(item["owner_id"]),
        role=str(item["role"]),
        created_at=_optional_text(item.get("created_at")),
    )


def _member_item(item: Mapping[str, object]) -> WorkspaceMemberDto:
    return WorkspaceMemberDto(
        user_id=str(item["user_id"]),
        username=str(item["username"]),
        display_name=str(item["display_name"]),
        role=str(item["role"]),
        created_at=_optional_text(item.get("created_at")),
    )


class AuthUseCase:
    """在 HTTP 之外编排认证事务、令牌签发与成员规则。"""

    _MANAGER_ROLES = frozenset({"owner", "admin"})

    def __init__(
        self,
        *,
        repository: AuthRepositoryPort,
        tokens: AuthTokenService,
    ) -> None:
        self._repository = repository
        self._tokens = tokens

    @staticmethod
    def _require_manager(principal: Principal) -> None:
        if principal.role not in AuthUseCase._MANAGER_ROLES:
            raise AuthAuthorizationError("需要角色：owner / admin")

    async def _auth_session(
        self,
        *,
        user: AuthUserView,
        workspace_id: str,
    ) -> AuthSessionDto:
        workspace = await self._repository.get_workspace(workspace_id)
        member = await self._repository.get_member(workspace_id, user.id)
        if workspace is None or member is None:
            raise AuthAuthorizationError("工作空间无效")

        # 最近工作空间属于认证状态，必须与令牌签发处于同一用例边界。
        user.last_workspace_id = workspace.id
        await self._repository.commit()
        access_token = self._tokens.create(
            user_id=user.id,
            workspace_id=workspace.id,
        )
        workspaces = tuple(
            _workspace_item(item)
            for item in await self._repository.list_workspaces(user.id)
        )
        return AuthSessionDto(
            access_token=access_token,
            token_type="bearer",
            user=AuthenticatedUserDto(
                id=user.id,
                username=user.username,
                display_name=user.display_name,
                workspace=WorkspaceContextDto(
                    id=workspace.id,
                    name=workspace.name,
                    role=member.role,
                ),
            ),
            workspaces=workspaces,
        )

    async def register(
        self,
        *,
        username: str,
        password: str,
        display_name: str,
        workspace_name: str,
    ) -> AuthSessionDto:
        if await self._repository.get_user_by_username(username) is not None:
            raise AuthConflictError("用户名已存在")
        try:
            validate_password(password)
        except PasswordPolicyError as exc:
            raise AuthValidationError(str(exc)) from exc

        user = await self._repository.create_user(
            username=username,
            display_name=display_name or username,
            password_hash=hash_password(password),
        )
        workspace = await self._repository.create_workspace(
            name=workspace_name,
            owner_id=user.id,
        )
        await self._repository.commit()
        return await self._auth_session(user=user, workspace_id=workspace.id)

    async def login(self, *, username: str, password: str) -> AuthSessionDto:
        user = await self._repository.get_user_by_username(username)
        if (
            user is None
            or not user.is_active
            or not verify_password(password, user.password_hash)
        ):
            raise AuthAuthenticationError("用户名或密码错误")

        workspaces = await self._repository.list_workspaces(user.id)
        if not workspaces:
            raise AuthAuthorizationError("用户尚未加入工作空间")
        workspace_id = await self._repository.preferred_workspace_id(user)
        if workspace_id is None:
            raise AuthAuthorizationError("工作空间无效")
        return await self._auth_session(user=user, workspace_id=workspace_id)

    async def switch_workspace(
        self,
        *,
        principal: Principal,
        workspace_id: str,
    ) -> AuthSessionDto:
        member = await self._repository.get_member(workspace_id, principal.user_id)
        if member is None:
            raise AuthAuthorizationError("你不是该工作空间成员")
        user = await self._repository.get_user(principal.user_id)
        if user is None:
            raise AuthAuthorizationError("工作空间无效")
        return await self._auth_session(user=user, workspace_id=workspace_id)

    async def list_workspaces(
        self,
        *,
        principal: Principal,
    ) -> tuple[WorkspaceListItemDto, ...]:
        return tuple(
            _workspace_item(item)
            for item in await self._repository.list_workspaces(principal.user_id)
        )

    async def create_workspace(
        self,
        *,
        principal: Principal,
        name: str,
    ) -> WorkspaceContextDto:
        workspace = await self._repository.create_workspace(
            name=name,
            owner_id=principal.user_id,
        )
        await self._repository.commit()
        return WorkspaceContextDto(id=workspace.id, name=workspace.name, role="owner")

    async def list_members(
        self,
        *,
        principal: Principal,
    ) -> tuple[WorkspaceMemberDto, ...]:
        return tuple(
            _member_item(item)
            for item in await self._repository.list_members(principal.workspace_id)
        )

    async def add_member(
        self,
        *,
        principal: Principal,
        username: str,
        role: str,
    ) -> MemberMutationDto:
        self._require_manager(principal)
        user = await self._repository.get_user_by_username(username)
        if user is None:
            raise AuthNotFoundError("用户不存在，请先注册该用户")
        await self._repository.add_member(
            workspace_id=principal.workspace_id,
            user_id=user.id,
            role=role,
        )
        return MemberMutationDto(user_id=user.id)

    async def update_member(
        self,
        *,
        principal: Principal,
        user_id: str,
        role: str,
    ) -> MemberMutationDto:
        self._require_manager(principal)
        member = await self._repository.get_member(principal.workspace_id, user_id)
        if member is None:
            raise AuthNotFoundError("成员不存在")
        if member.role == "owner":
            raise AuthValidationError("不能修改工作空间所有者")
        if principal.role == "admin" and role == "admin":
            raise AuthAuthorizationError("管理员不能提升其他成员为管理员")
        await self._repository.update_member_role(member, role)
        return MemberMutationDto()

    async def remove_member(
        self,
        *,
        principal: Principal,
        user_id: str,
    ) -> None:
        self._require_manager(principal)
        member = await self._repository.get_member(principal.workspace_id, user_id)
        if member is None:
            raise AuthNotFoundError("成员不存在")
        if member.role == "owner" or member.user_id == principal.user_id:
            raise AuthValidationError("不能移除工作空间所有者或自己")
        await self._repository.delete_member(member)


__all__ = [
    "AuthAuthenticationError",
    "AuthAuthorizationError",
    "AuthConflictError",
    "AuthNotFoundError",
    "AuthSessionDto",
    "AuthUseCase",
    "AuthUseCaseError",
    "AuthValidationError",
    "AuthenticatedUserDto",
    "MemberMutationDto",
    "WorkspaceContextDto",
    "WorkspaceListItemDto",
    "WorkspaceMemberDto",
]
