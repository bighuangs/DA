"""受治理元数据后台的 Application 用例。"""

from __future__ import annotations

from typing import Any

from app.application.ports.metadata_indexing import (
    AccessPolicyAdminPort,
    MetadataAdminRepositoryPort,
    MetadataCacheInvalidationPort,
    MetadataProjectionPort,
    ValueIndexAdminPort,
)
from app.application.services.masking_service import masking_policy as policy_manifest
from app.domain.access.policy import PolicySnapshot
from app.domain.privacy.masking import MaskingPolicy


class AdminUseCaseError(RuntimeError):
    """管理用例可稳定映射到传输层的错误基类。"""


class AdminNotFoundError(AdminUseCaseError):
    """管理对象在当前工作空间不存在。"""


class AdminConflictError(AdminUseCaseError):
    """管理对象与已有唯一身份冲突。"""


class AdminValidationError(AdminUseCaseError):
    """输入通过传输校验后仍违反管理业务规则。"""


class AdminForbiddenError(AdminUseCaseError):
    """当前管理身份无权访问指定资源。"""


class MetaAdminService:
    """编排元数据 CRUD、治理策略与索引发布用例。"""

    def __init__(
        self,
        *,
        meta_repo: MetadataAdminRepositoryPort,
        projection_index: MetadataProjectionPort,
        cache_invalidator: MetadataCacheInvalidationPort,
        value_index: ValueIndexAdminPort | None = None,
        access_policy: PolicySnapshot | None = None,
        access_policy_admin: AccessPolicyAdminPort | None = None,
        masking_policy: MaskingPolicy | None = None,
    ) -> None:
        self._repository = meta_repo
        self._projection_index = projection_index
        self._cache_invalidator = cache_invalidator
        self._value_index = value_index
        self._access_policy = access_policy
        self._access_policy_admin = access_policy_admin
        self._masking_policy = masking_policy

    async def _ensure_admin_tables(self) -> None:
        await self._repository.ensure_admin_tables()

    def _workspace_id(self) -> str:
        workspace_id = str(self._repository.workspace_id or "").strip()
        if not workspace_id:
            raise ValueError("Meta 管理服务必须绑定 workspace_id")
        return workspace_id

    def _require_value_index(self) -> ValueIndexAdminPort:
        if self._value_index is None:
            raise RuntimeError("字段值索引仓储未配置")
        return self._value_index

    def _require_access_policy_admin(self) -> AccessPolicyAdminPort:
        if self._access_policy_admin is None:
            raise RuntimeError("数据权限策略管理端口未配置")
        return self._access_policy_admin

    def get_masking_policy(self) -> dict:
        if self._masking_policy is None:
            raise RuntimeError("脱敏策略未配置")
        return policy_manifest(policy=self._masking_policy)

    @staticmethod
    def metric_dict(item: Any) -> dict:
        return {
            "id": item.id,
            "name": item.name,
            "description": item.description or "",
            "relevant_columns": item.relevant_columns or [],
            "alias": item.alias or [],
            "formula_rule": item.formula_rule,
            "enabled": bool(item.enabled),
            "kind": "terminology" if not item.relevant_columns else "metric",
        }

    @staticmethod
    def training_dict(item: Any) -> dict:
        return {
            "id": item.id,
            "question": item.question,
            "sql": item.sql,
            "enabled": bool(item.enabled),
            "created_at": str(item.created_at) if item.created_at else None,
        }

    @staticmethod
    def _table_dict(table: Any, columns: list[Any]) -> dict:
        return {
            "id": table.id,
            "name": table.name,
            "role": table.role,
            "description": table.description or "",
            "columns": [
                {
                    "id": column.id,
                    "name": column.name,
                    "type": column.type,
                    "role": column.role,
                    "description": column.description or "",
                    "alias": column.alias or [],
                    "sync": bool(column.sync),
                    "is_display": bool(column.is_display),
                }
                for column in columns
            ],
        }

    @staticmethod
    def _masking_dict(item: Any) -> dict:
        return {
            "id": item.id,
            "column_pattern": item.column_pattern,
            "data_kind": item.data_kind,
            "strategy": item.strategy,
            "enabled": item.enabled,
            "priority": item.priority,
            "description": item.description or "",
            "created_at": str(item.created_at) if item.created_at else None,
        }

    @staticmethod
    def _access_policy_dict(item: Any) -> dict:
        return {
            "id": item.id,
            "workspace_id": item.workspace_id,
            "role": item.role,
            "user_id": item.user_id,
            "table_pattern": item.table_pattern,
            "allowed_columns": item.allowed_columns or [],
            "row_filter": item.row_filter or "",
            "access": bool(item.access),
            "enabled": bool(item.enabled),
            "priority": item.priority,
            "description": item.description or "",
            "created_at": str(item.created_at) if item.created_at else None,
        }

    @staticmethod
    def _schema_log_dict(item: Any) -> dict:
        return {
            "id": item.id,
            "source": item.source,
            "status": item.status,
            "diff": item.diff or {},
            "error": item.error,
            "created_at": str(item.created_at) if item.created_at else None,
        }

    async def list_metrics(
        self,
        *,
        kind: str | None = None,
        search: str | None = None,
    ) -> list[dict]:
        await self._ensure_admin_tables()
        items = await self._repository.list_metric_infos(kind=kind, search=search)
        return [self.metric_dict(item) for item in items]

    async def overview(self) -> dict:
        await self._ensure_admin_tables()
        metrics = await self._repository.list_metric_infos()
        training = await self._repository.list_training_infos()
        tables = await self._repository.list_table_infos()
        return {
            "metrics": len([item for item in metrics if item.relevant_columns]),
            "terminology": len([item for item in metrics if not item.relevant_columns]),
            "training": len(training),
            "tables": len(tables),
            "enabled_metrics": len([item for item in metrics if item.enabled]),
            "enabled_training": len([item for item in training if item.enabled]),
        }

    async def create_metric(
        self,
        *,
        name: str,
        description: str,
        relevant_columns: list[str],
        alias: list[str],
        enabled: bool,
        formula_rule: dict | None,
    ) -> dict:
        await self._ensure_admin_tables()
        workspace_id = str(self._repository.workspace_id or "")
        metric_id = f"{workspace_id}:{name}" if workspace_id else name
        if await self._repository.get_metric_info(metric_id) is not None:
            raise AdminConflictError("指标名称已存在")
        item = await self._repository.create_metric_info(
            metric_id=metric_id,
            name=name,
            description=description,
            relevant_columns=relevant_columns,
            alias=alias,
            enabled=enabled,
            formula_rule=formula_rule,
        )
        self._cache_invalidator.invalidate_retrieval()
        return self.metric_dict(item)

    async def update_metric(
        self,
        metric_id: str,
        *,
        description: str,
        relevant_columns: list[str],
        alias: list[str],
        enabled: bool,
        formula_rule: dict | None,
    ) -> dict:
        await self._ensure_admin_tables()
        item = await self._repository.get_metric_info(metric_id)
        if item is None:
            raise AdminNotFoundError("指标不存在")
        item = await self._repository.update_metric_info(
            item,
            name=item.name,
            description=description,
            relevant_columns=relevant_columns,
            alias=alias,
            enabled=enabled,
            formula_rule=formula_rule,
        )
        self._cache_invalidator.invalidate_retrieval()
        return self.metric_dict(item)

    async def delete_metric(self, metric_id: str) -> None:
        await self._ensure_admin_tables()
        item = await self._repository.get_metric_info(metric_id)
        if item is None:
            raise AdminNotFoundError("指标不存在")
        await self._repository.delete_metric_info(item)
        self._cache_invalidator.invalidate_retrieval()

    async def list_training(self, *, search: str | None = None) -> list[dict]:
        await self._ensure_admin_tables()
        items = await self._repository.list_training_infos(search=search)
        return [self.training_dict(item) for item in items]

    async def create_training(
        self,
        *,
        question: str,
        sql: str,
        enabled: bool,
    ) -> dict:
        await self._ensure_admin_tables()
        item = await self._repository.create_training_info(
            question=question,
            sql=sql,
            enabled=enabled,
        )
        self._cache_invalidator.invalidate_retrieval()
        return self.training_dict(item)

    async def update_training(
        self,
        training_id: int,
        *,
        question: str,
        sql: str,
        enabled: bool,
    ) -> dict:
        await self._ensure_admin_tables()
        item = await self._repository.get_training_info(training_id)
        if item is None:
            raise AdminNotFoundError("训练样例不存在")
        item = await self._repository.update_training_info(
            item,
            question=question,
            sql=sql,
            enabled=enabled,
        )
        self._cache_invalidator.invalidate_retrieval()
        return self.training_dict(item)

    async def delete_training(self, training_id: int) -> None:
        await self._ensure_admin_tables()
        item = await self._repository.get_training_info(training_id)
        if item is None:
            raise AdminNotFoundError("训练样例不存在")
        await self._repository.delete_training_info(item)
        self._cache_invalidator.invalidate_retrieval()

    async def list_tables(self) -> list[dict]:
        await self._ensure_admin_tables()
        rows = await self._repository.list_table_infos()
        return [self._table_dict(table, columns) for table, columns in rows]

    async def list_masking_rules(self) -> list[dict]:
        await self._ensure_admin_tables()
        return [
            self._masking_dict(item)
            for item in await self._repository.list_masking_rules()
        ]

    async def create_masking_rule(self, **values: object) -> dict:
        await self._ensure_admin_tables()
        item = await self._repository.create_masking_rule(**values)
        self._cache_invalidator.clear_query_results()
        return self._masking_dict(item)

    async def update_masking_rule(
        self,
        rule_id: int,
        **values: object,
    ) -> dict:
        await self._ensure_admin_tables()
        item = await self._repository.get_masking_rule(rule_id)
        if item is None:
            raise AdminNotFoundError("脱敏规则不存在")
        item = await self._repository.update_masking_rule(item, **values)
        self._cache_invalidator.clear_query_results()
        return self._masking_dict(item)

    async def delete_masking_rule(self, rule_id: int) -> None:
        await self._ensure_admin_tables()
        item = await self._repository.get_masking_rule(rule_id)
        if item is None:
            raise AdminNotFoundError("脱敏规则不存在")
        await self._repository.delete_masking_rule(item)
        self._cache_invalidator.clear_query_results()

    async def list_access_policies(self) -> list[dict]:
        await self._ensure_admin_tables()
        return [
            self._access_policy_dict(item)
            for item in await self._require_access_policy_admin().list_all()
        ]

    async def create_access_policy(self, **values: object) -> dict:
        await self._ensure_admin_tables()
        try:
            item = await self._require_access_policy_admin().create(**values)
        except ValueError as exc:
            raise AdminValidationError(str(exc)) from exc
        self._cache_invalidator.invalidate_retrieval()
        return self._access_policy_dict(item)

    async def update_access_policy(
        self,
        policy_id: int,
        **values: object,
    ) -> dict:
        await self._ensure_admin_tables()
        repository = self._require_access_policy_admin()
        item = await repository.get(policy_id)
        if item is None:
            raise AdminNotFoundError("数据权限策略不存在")
        try:
            item = await repository.update(item, **values)
        except ValueError as exc:
            raise AdminValidationError(str(exc)) from exc
        self._cache_invalidator.invalidate_retrieval()
        return self._access_policy_dict(item)

    async def delete_access_policy(self, policy_id: int) -> None:
        await self._ensure_admin_tables()
        repository = self._require_access_policy_admin()
        item = await repository.get(policy_id)
        if item is None:
            raise AdminNotFoundError("数据权限策略不存在")
        await repository.delete(item)
        self._cache_invalidator.invalidate_retrieval()

    async def schema_sync_history(self, limit: int) -> list[dict]:
        await self._ensure_admin_tables()
        return [
            self._schema_log_dict(item)
            for item in await self._repository.list_schema_sync_logs(limit)
        ]

    async def value_index_status(self) -> dict:
        await self._ensure_admin_tables()
        table_rows = await self._repository.list_table_infos()
        columns = [column for _, values in table_rows for column in values]
        return await self._require_value_index().status(
            workspace_id=self._workspace_id(),
            columns=columns,
        )

    async def rollback_value_index(self, physical_index: str) -> dict:
        await self._ensure_admin_tables()
        try:
            return await self._require_value_index().rollback(
                workspace_id=self._workspace_id(),
                physical_index=physical_index,
            )
        except ValueError as exc:
            raise AdminForbiddenError(str(exc)) from exc
        except LookupError as exc:
            raise AdminNotFoundError(str(exc)) from exc

    async def search_values(
        self,
        keyword: str,
        *,
        query_mode: str,
        table_id: str | None,
        column_id: str | None,
        page: int,
        page_size: int,
    ) -> dict:
        await self._ensure_admin_tables()
        workspace_id = self._workspace_id()
        table_rows = await self._repository.list_table_infos()
        allowed_tables: list[str] = []
        allowed_columns: list[str] = []
        known_columns: set[str] = set()
        for table, columns in table_rows:
            current_table = str(table.id or table.name)
            for column in columns:
                known_columns.add(str(column.id))
            if self._access_policy and not self._access_policy.is_table_allowed(
                current_table
            ):
                continue
            allowed_tables.append(current_table)
            for column in columns:
                if self._access_policy and not self._access_policy.is_column_allowed(
                    current_table,
                    column.name,
                ):
                    continue
                allowed_columns.append(str(column.id))
        if table_id and table_id not in allowed_tables:
            raise AdminForbiddenError("无权检索该表的字段值")
        if column_id and column_id not in known_columns:
            raise AdminNotFoundError("字段不存在")
        if column_id and column_id not in allowed_columns:
            raise AdminForbiddenError("无权检索该字段的值")

        items, total, evidence = await self._require_value_index().search(
            keyword,
            workspace_id=workspace_id,
            query_mode=query_mode,
            allowed_table_ids=allowed_tables,
            allowed_column_ids=allowed_columns,
            table_id=table_id,
            column_id=column_id,
            page=page,
            page_size=page_size,
        )
        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "evidence": evidence,
        }

    async def _hybrid_status(self) -> dict:
        workspace_id = str(self._repository.workspace_id or "")
        table_rows = await self._repository.list_table_infos()
        metrics = await self._repository.list_metric_infos()
        training = await self._repository.list_training_infos()
        return await self._projection_index.status(
            workspace_id=workspace_id,
            table_rows=table_rows,
            metric_items=metrics,
            training_items=training,
        )

    async def hybrid_status(self) -> dict:
        await self._ensure_admin_tables()
        return await self._hybrid_status()

    async def rebuild_columns(self) -> int:
        count = await self._projection_index.rebuild_columns(
            await self._repository.list_table_infos()
        )
        self._cache_invalidator.invalidate_retrieval()
        return count

    async def rebuild_metrics(self) -> int:
        workspace_id = self._workspace_id()
        count = await self._projection_index.rebuild_metrics(
            workspace_id,
            await self._repository.list_metric_infos(),
        )
        self._cache_invalidator.invalidate_retrieval()
        return count

    async def rebuild_training(self) -> int:
        workspace_id = self._workspace_id()
        count = await self._projection_index.rebuild_training(
            workspace_id,
            await self._repository.list_training_infos(),
        )
        self._cache_invalidator.invalidate_retrieval()
        return count

    async def _overwrite_all_metrics(self) -> int:
        count = await self._projection_index.overwrite_all_metrics()
        self._cache_invalidator.invalidate_retrieval()
        return count

    async def _overwrite_all_training(self) -> int:
        count = await self._projection_index.overwrite_all_training()
        self._cache_invalidator.invalidate_retrieval()
        return count

    async def rebuild_all(self) -> dict:
        """按原有顺序覆盖三类 Point 投影并逐类失效检索缓存。"""

        return {
            "columns": await self.rebuild_columns(),
            "metrics": await self._overwrite_all_metrics(),
            "training": await self._overwrite_all_training(),
            "point_strategy": self._projection_index.point_strategy(),
        }

    async def rebuild(self, target: str) -> dict:
        """执行后台索引重建并返回原管理 API 合同。"""

        await self._ensure_admin_tables()
        counts = await self.rebuild_all() if target == "all" else {}
        if target == "column":
            counts["columns"] = await self.rebuild_columns()
        if target == "metric":
            counts["metrics"] = await self.rebuild_metrics()
        if target == "training":
            counts["training"] = await self.rebuild_training()
        return {
            "target": target,
            "embedding_mode": "dense_sparse",
            "replacement_strategy": "overwrite_existing_collections",
            "v1_fallback": False,
            "indexed": counts,
            "status": await self._hybrid_status(),
        }


__all__ = [
    "AdminConflictError",
    "AdminForbiddenError",
    "AdminNotFoundError",
    "AdminUseCaseError",
    "AdminValidationError",
    "MetaAdminService",
]
