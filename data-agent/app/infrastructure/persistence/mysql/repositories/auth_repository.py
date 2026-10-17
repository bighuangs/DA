from typing import Optional
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.persistence.mysql.records.user_mysql import UserMySQL
from app.infrastructure.persistence.mysql.records.workspace_member_mysql import WorkspaceMemberMySQL
from app.infrastructure.persistence.mysql.records.workspace_mysql import WorkspaceMySQL


class AuthRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_user_by_username(self, username: str) -> Optional[UserMySQL]:
        return (await self.session.execute(
            select(UserMySQL).where(UserMySQL.username == username)
        )).scalar_one_or_none()

    async def get_user(self, user_id: str) -> Optional[UserMySQL]:
        return await self.session.get(UserMySQL, user_id)

    async def create_user(self, *, username: str, display_name: str, password_hash: str) -> UserMySQL:
        item = UserMySQL(id=str(uuid.uuid4()), username=username, display_name=display_name,
                         password_hash=password_hash, is_active=True)
        self.session.add(item)
        await self.session.flush()
        return item

    async def create_workspace(self, *, name: str, owner_id: str) -> WorkspaceMySQL:
        item = WorkspaceMySQL(id=str(uuid.uuid4()), name=name, owner_id=owner_id)
        self.session.add(item)
        await self.session.flush()
        self.session.add(WorkspaceMemberMySQL(workspace_id=item.id, user_id=owner_id, role="owner"))
        return item

    async def list_workspaces(self, user_id: str) -> list[dict]:
        result = await self.session.execute(
            select(WorkspaceMySQL, WorkspaceMemberMySQL.role)
            .join(WorkspaceMemberMySQL, WorkspaceMemberMySQL.workspace_id == WorkspaceMySQL.id)
            .where(WorkspaceMemberMySQL.user_id == user_id)
            .order_by(WorkspaceMySQL.created_at)
        )
        return [{"id": workspace.id, "name": workspace.name, "owner_id": workspace.owner_id,
                 "role": role, "created_at": str(workspace.created_at) if workspace.created_at else None}
                for workspace, role in result.all()]

    async def preferred_workspace_id(self, user: UserMySQL) -> str | None:
        """Return the last still-authorized workspace, else the stable first one."""
        workspaces = await self.list_workspaces(user.id)
        if not workspaces:
            return None
        available = {str(item["id"]) for item in workspaces}
        preferred = str(user.last_workspace_id or "")
        return preferred if preferred in available else str(workspaces[0]["id"])

    async def get_workspace(self, workspace_id: str) -> Optional[WorkspaceMySQL]:
        return await self.session.get(WorkspaceMySQL, workspace_id)

    async def get_member(self, workspace_id: str, user_id: str) -> Optional[WorkspaceMemberMySQL]:
        return (await self.session.execute(
            select(WorkspaceMemberMySQL)
            .where(WorkspaceMemberMySQL.workspace_id == workspace_id)
            .where(WorkspaceMemberMySQL.user_id == user_id)
        )).scalar_one_or_none()

    async def list_members(self, workspace_id: str) -> list[dict]:
        result = await self.session.execute(
            select(WorkspaceMemberMySQL, UserMySQL)
            .join(UserMySQL, UserMySQL.id == WorkspaceMemberMySQL.user_id)
            .where(WorkspaceMemberMySQL.workspace_id == workspace_id)
            .order_by(UserMySQL.username)
        )
        return [{"user_id": member.user_id, "username": user.username,
                 "display_name": user.display_name, "role": member.role,
                 "created_at": str(member.created_at) if member.created_at else None}
                for member, user in result.all()]

    async def add_member(self, *, workspace_id: str, user_id: str, role: str) -> WorkspaceMemberMySQL:
        item = await self.get_member(workspace_id, user_id)
        if item:
            item.role = role
        else:
            item = WorkspaceMemberMySQL(workspace_id=workspace_id, user_id=user_id, role=role)
            self.session.add(item)
        await self.session.commit()
        await self.session.refresh(item)
        return item

    async def update_member_role(self, item: WorkspaceMemberMySQL, role: str) -> WorkspaceMemberMySQL:
        item.role = role
        await self.session.commit()
        await self.session.refresh(item)
        return item

    async def delete_member(self, item: WorkspaceMemberMySQL) -> None:
        await self.session.delete(item)
        await self.session.commit()
