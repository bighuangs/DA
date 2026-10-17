from fnmatch import fnmatchcase

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.access.policy import ROLE_VALUES, validate_row_filter
from app.infrastructure.persistence.mysql.records.access_policy_mysql import AccessPolicyMySQL
from app.infrastructure.persistence.mysql.records.table_info_mysql import TableInfoMySQL
from app.infrastructure.persistence.mysql.records.column_info_mysql import ColumnInfoMySQL
from app.infrastructure.persistence.mysql.records.workspace_member_mysql import WorkspaceMemberMySQL


class AccessPolicyRepository:
    def __init__(self, session: AsyncSession, workspace_id: str):
        self.session = session
        self.workspace_id = workspace_id

    async def list_all(self) -> list[AccessPolicyMySQL]:
        result = await self.session.execute(
            select(AccessPolicyMySQL)
            .where(AccessPolicyMySQL.workspace_id == self.workspace_id)
            .order_by(AccessPolicyMySQL.priority, AccessPolicyMySQL.id)
        )
        return result.scalars().all()

    async def list_enabled(self) -> list[AccessPolicyMySQL]:
        result = await self.session.execute(
            select(AccessPolicyMySQL)
            .where(
                AccessPolicyMySQL.workspace_id == self.workspace_id,
                AccessPolicyMySQL.enabled.is_(True),
            )
            .order_by(AccessPolicyMySQL.priority, AccessPolicyMySQL.id)
        )
        return result.scalars().all()

    async def get(self, policy_id: int) -> AccessPolicyMySQL | None:
        result = await self.session.execute(
            select(AccessPolicyMySQL).where(
                AccessPolicyMySQL.id == policy_id,
                AccessPolicyMySQL.workspace_id == self.workspace_id,
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    def validate_payload(*, role: str, table_pattern: str, allowed_columns: list[str] | None,
                         row_filter: str | None) -> None:
        if role not in ROLE_VALUES:
            raise ValueError("无效的策略角色")
        if not str(table_pattern or "").strip():
            raise ValueError("表匹配规则不能为空")
        if len(str(table_pattern)) > 128:
            raise ValueError("表匹配规则不能超过 128 个字符")
        if allowed_columns is not None and len(allowed_columns) > 200:
            raise ValueError("允许列数量不能超过 200")
        validate_row_filter(row_filter)

    async def validate_references(self, *, user_id: str | None, table_pattern: str,
                                  allowed_columns: list[str] | None) -> None:
        if user_id:
            member = await self.session.execute(select(WorkspaceMemberMySQL.id).where(
                WorkspaceMemberMySQL.workspace_id == self.workspace_id,
                WorkspaceMemberMySQL.user_id == user_id,
            ))
            if member.scalar_one_or_none() is None:
                raise ValueError("策略用户不是当前工作空间成员")
        tables = (await self.session.execute(select(TableInfoMySQL.name))).scalars().all()
        matched = [name for name in tables if fnmatchcase(name.lower(), table_pattern.lower())]
        if not matched:
            raise ValueError(f"表匹配规则未命中当前 Schema：{table_pattern}")
        if allowed_columns:
            rows = await self.session.execute(select(
                ColumnInfoMySQL.table_id, ColumnInfoMySQL.name
            ).where(ColumnInfoMySQL.table_id.in_(matched)))
            available = {str(row.name).lower() for row in rows}
            unknown = sorted({str(name).split(".")[-1] for name in allowed_columns
                              if str(name).split(".")[-1].lower() not in available})
            if unknown:
                raise ValueError(f"允许列不存在于匹配表：{unknown}")

    async def create(self, *, role: str, user_id: str | None, table_pattern: str,
                     allowed_columns: list[str] | None, row_filter: str | None,
                     access: bool, enabled: bool, priority: int,
                     description: str) -> AccessPolicyMySQL:
        self.validate_payload(role=role, table_pattern=table_pattern,
                              allowed_columns=allowed_columns, row_filter=row_filter)
        await self.validate_references(
            user_id=user_id, table_pattern=table_pattern, allowed_columns=allowed_columns
        )
        item = AccessPolicyMySQL(
            workspace_id=self.workspace_id,
            role=role,
            user_id=user_id or None,
            table_pattern=table_pattern.strip(),
            allowed_columns=allowed_columns or None,
            row_filter=validate_row_filter(row_filter),
            access=access,
            enabled=enabled,
            priority=priority,
            description=description or "",
        )
        self.session.add(item)
        await self.session.commit()
        await self.session.refresh(item)
        return item

    async def update(self, item: AccessPolicyMySQL, *, role: str, user_id: str | None,
                     table_pattern: str, allowed_columns: list[str] | None,
                     row_filter: str | None, access: bool, enabled: bool,
                     priority: int, description: str) -> AccessPolicyMySQL:
        self.validate_payload(role=role, table_pattern=table_pattern,
                              allowed_columns=allowed_columns, row_filter=row_filter)
        await self.validate_references(
            user_id=user_id, table_pattern=table_pattern, allowed_columns=allowed_columns
        )
        item.role = role
        item.user_id = user_id or None
        item.table_pattern = table_pattern.strip()
        item.allowed_columns = allowed_columns or None
        item.row_filter = validate_row_filter(row_filter)
        item.access = access
        item.enabled = enabled
        item.priority = priority
        item.description = description or ""
        await self.session.commit()
        await self.session.refresh(item)
        return item

    async def delete(self, item: AccessPolicyMySQL) -> None:
        await self.session.execute(delete(AccessPolicyMySQL).where(AccessPolicyMySQL.id == item.id))
        await self.session.commit()
