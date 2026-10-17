"""HTTP Adapter 可依赖的应用端口与轻量配置值。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

class AuthUserView(Protocol):
    id: str
    username: str
    display_name: str
    password_hash: str
    is_active: bool
    last_workspace_id: str | None


class WorkspaceView(Protocol):
    id: str
    name: str


class WorkspaceMemberView(Protocol):
    user_id: str
    role: str


class AuthRepositoryPort(Protocol):
    async def get_user(self, user_id: str) -> AuthUserView | None: ...

    async def get_user_by_username(self, username: str) -> AuthUserView | None: ...

    async def create_user(
        self, *, username: str, display_name: str, password_hash: str
    ) -> AuthUserView: ...

    async def get_workspace(self, workspace_id: str) -> WorkspaceView | None: ...

    async def create_workspace(self, *, name: str, owner_id: str) -> WorkspaceView: ...

    async def get_member(
        self, workspace_id: str, user_id: str
    ) -> WorkspaceMemberView | None: ...

    async def list_workspaces(self, user_id: str) -> list[dict[str, object]]: ...

    async def preferred_workspace_id(self, user: AuthUserView) -> str: ...

    async def list_members(self, workspace_id: str) -> list[dict[str, object]]: ...

    async def add_member(
        self, *, workspace_id: str, user_id: str, role: str
    ) -> object: ...

    async def update_member_role(
        self, member: WorkspaceMemberView, role: str
    ) -> None: ...

    async def delete_member(self, member: WorkspaceMemberView) -> None: ...

    async def commit(self) -> None: ...


class QueryCapacityLease(Protocol):
    async def __aenter__(self) -> object: ...

    async def __aexit__(self, exc_type, exc, traceback) -> None: ...


class QueryCapacityPort(Protocol):
    async def acquire(self, workspace_id: str) -> QueryCapacityLease: ...


class QueryCapacityExceeded(RuntimeError):
    """HTTP 查询进入执行槽位前超过应用并发预算。"""


@dataclass(frozen=True, slots=True)
class UploadLimits:
    """上传 Adapter 读取文件体前需要的有界配置。"""

    max_file_bytes: int

    def __post_init__(self) -> None:
        if self.max_file_bytes <= 0:
            raise ValueError("max_file_bytes 必须大于 0")


__all__ = [
    "AuthRepositoryPort",
    "QueryCapacityLease",
    "QueryCapacityExceeded",
    "QueryCapacityPort",
    "UploadLimits",
]
