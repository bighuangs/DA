"""认证 Application 用例的事务、令牌与角色规则回归。"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.application.services.auth_service import (
    AuthTokenPolicy,
    AuthTokenService,
    Principal,
    hash_password,
    verify_password,
)
from app.application.services.auth_use_case import (
    AuthAuthenticationError,
    AuthAuthorizationError,
    AuthConflictError,
    AuthNotFoundError,
    AuthUseCase,
    AuthValidationError,
)


@dataclass
class _User:
    id: str
    username: str
    display_name: str
    password_hash: str
    is_active: bool = True
    last_workspace_id: str | None = None


@dataclass
class _Workspace:
    id: str
    name: str
    owner_id: str


@dataclass
class _Member:
    user_id: str
    role: str


class _Repository:
    def __init__(self) -> None:
        self.users: dict[str, _User] = {}
        self.workspaces: dict[str, _Workspace] = {}
        self.members: dict[tuple[str, str], _Member] = {}
        self.commit_count = 0

    async def get_user(self, user_id: str) -> _User | None:
        return self.users.get(user_id)

    async def get_user_by_username(self, username: str) -> _User | None:
        return next(
            (item for item in self.users.values() if item.username == username),
            None,
        )

    async def create_user(
        self,
        *,
        username: str,
        display_name: str,
        password_hash: str,
    ) -> _User:
        user = _User(
            id=f"user-{len(self.users) + 1}",
            username=username,
            display_name=display_name,
            password_hash=password_hash,
        )
        self.users[user.id] = user
        return user

    async def get_workspace(self, workspace_id: str) -> _Workspace | None:
        return self.workspaces.get(workspace_id)

    async def create_workspace(self, *, name: str, owner_id: str) -> _Workspace:
        workspace = _Workspace(
            id=f"workspace-{len(self.workspaces) + 1}",
            name=name,
            owner_id=owner_id,
        )
        self.workspaces[workspace.id] = workspace
        self.members[(workspace.id, owner_id)] = _Member(
            user_id=owner_id,
            role="owner",
        )
        return workspace

    async def get_member(self, workspace_id: str, user_id: str) -> _Member | None:
        return self.members.get((workspace_id, user_id))

    async def list_workspaces(self, user_id: str) -> list[dict[str, object]]:
        return [
            {
                "id": workspace.id,
                "name": workspace.name,
                "owner_id": workspace.owner_id,
                "role": member.role,
                "created_at": None,
            }
            for workspace in self.workspaces.values()
            if (member := self.members.get((workspace.id, user_id))) is not None
        ]

    async def preferred_workspace_id(self, user: _User) -> str | None:
        available = [
            item["id"]
            for item in await self.list_workspaces(user.id)
        ]
        if user.last_workspace_id in available:
            return user.last_workspace_id
        return str(available[0]) if available else None

    async def list_members(self, workspace_id: str) -> list[dict[str, object]]:
        return [
            {
                "user_id": user_id,
                "username": self.users[user_id].username,
                "display_name": self.users[user_id].display_name,
                "role": member.role,
                "created_at": None,
            }
            for (member_workspace_id, user_id), member in self.members.items()
            if member_workspace_id == workspace_id
        ]

    async def add_member(
        self,
        *,
        workspace_id: str,
        user_id: str,
        role: str,
    ) -> _Member:
        member = self.members.get((workspace_id, user_id))
        if member is None:
            member = _Member(user_id=user_id, role=role)
            self.members[(workspace_id, user_id)] = member
        else:
            member.role = role
        return member

    async def update_member_role(self, member: _Member, role: str) -> None:
        member.role = role

    async def delete_member(self, member: _Member) -> None:
        key = next(key for key, item in self.members.items() if item is member)
        del self.members[key]

    async def commit(self) -> None:
        self.commit_count += 1


def _service(repository: _Repository) -> AuthUseCase:
    return AuthUseCase(
        repository=repository,
        tokens=AuthTokenService(
            AuthTokenPolicy(
                secret_key="test-secret",
                access_token_ttl_minutes=30,
            )
        ),
    )


def _principal(*, role: str = "owner") -> Principal:
    return Principal(
        user_id="user-1",
        username="alice",
        display_name="Alice",
        workspace_id="workspace-1",
        workspace_name="主空间",
        role=role,
    )


@pytest.mark.asyncio
async def test_register_and_login_preserve_transaction_and_last_workspace() -> None:
    repository = _Repository()
    service = _service(repository)

    registered = await service.register(
        username="alice",
        password="password-123",
        display_name="",
        workspace_name="主空间",
    )

    user = repository.users["user-1"]
    assert registered.user.display_name == "alice"
    assert registered.user.workspace.role == "owner"
    assert registered.token_type == "bearer"
    assert verify_password("password-123", user.password_hash)
    assert user.last_workspace_id == "workspace-1"
    assert repository.commit_count == 2

    second = await service.create_workspace(
        principal=_principal(),
        name="第二空间",
    )
    user.last_workspace_id = second.id
    logged_in = await service.login(username="alice", password="password-123")

    assert logged_in.user.workspace.id == "workspace-2"
    assert [item.id for item in logged_in.workspaces] == [
        "workspace-1",
        "workspace-2",
    ]
    assert repository.commit_count == 4


@pytest.mark.asyncio
async def test_authentication_failures_are_typed_without_mutating_state() -> None:
    repository = _Repository()
    service = _service(repository)
    repository.users["user-1"] = _User(
        id="user-1",
        username="alice",
        display_name="Alice",
        password_hash=hash_password("password-123"),
    )

    with pytest.raises(AuthConflictError, match="用户名已存在"):
        await service.register(
            username="alice",
            password="password-123",
            display_name="Alice",
            workspace_name="主空间",
        )
    with pytest.raises(AuthValidationError, match="密码至少需要 8 个字符"):
        await service.register(
            username="bob",
            password="short",
            display_name="Bob",
            workspace_name="主空间",
        )
    with pytest.raises(AuthAuthenticationError, match="用户名或密码错误"):
        await service.login(username="alice", password="wrong-password")
    with pytest.raises(AuthAuthorizationError, match="用户尚未加入工作空间"):
        await service.login(username="alice", password="password-123")

    assert repository.commit_count == 0


@pytest.mark.asyncio
async def test_workspace_switch_and_member_crud_preserve_role_rules() -> None:
    repository = _Repository()
    service = _service(repository)
    alice = await repository.create_user(
        username="alice",
        display_name="Alice",
        password_hash=hash_password("password-123"),
    )
    bob = await repository.create_user(
        username="bob",
        display_name="Bob",
        password_hash=hash_password("password-123"),
    )
    workspace = await repository.create_workspace(name="主空间", owner_id=alice.id)
    principal = _principal()

    added = await service.add_member(
        principal=principal,
        username="bob",
        role="viewer",
    )
    members = await service.list_members(principal=principal)
    updated = await service.update_member(
        principal=principal,
        user_id=bob.id,
        role="analyst",
    )

    assert added.user_id == bob.id
    assert updated.success is True
    assert [item.role for item in members] == ["owner", "viewer"]
    assert repository.members[(workspace.id, bob.id)].role == "analyst"

    admin = _principal(role="admin")
    with pytest.raises(AuthAuthorizationError, match="不能提升"):
        await service.update_member(
            principal=admin,
            user_id=bob.id,
            role="admin",
        )
    with pytest.raises(AuthValidationError, match="不能修改工作空间所有者"):
        await service.update_member(
            principal=principal,
            user_id=alice.id,
            role="viewer",
        )
    with pytest.raises(AuthValidationError, match="不能移除.*自己"):
        await service.remove_member(principal=principal, user_id=alice.id)
    with pytest.raises(AuthNotFoundError, match="用户不存在"):
        await service.add_member(
            principal=principal,
            username="missing",
            role="viewer",
        )

    switched = await service.switch_workspace(
        principal=Principal(
            user_id=bob.id,
            username=bob.username,
            display_name=bob.display_name,
            workspace_id=workspace.id,
            workspace_name=workspace.name,
            role="analyst",
        ),
        workspace_id=workspace.id,
    )
    assert switched.user.workspace.role == "analyst"
    assert bob.last_workspace_id == workspace.id

    await service.remove_member(principal=principal, user_id=bob.id)
    assert (workspace.id, bob.id) not in repository.members
    with pytest.raises(AuthAuthorizationError, match="你不是该工作空间成员"):
        await service.switch_workspace(
            principal=Principal(
                user_id=bob.id,
                username=bob.username,
                display_name=bob.display_name,
                workspace_id="other",
                workspace_name="其他空间",
                role="viewer",
            ),
            workspace_id=workspace.id,
        )
