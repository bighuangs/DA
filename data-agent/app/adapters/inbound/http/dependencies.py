"""HTTP 依赖声明。

本模块只定义 Adapter 所需的 Application 端口。生产实现由 Bootstrap 通过
``dependency_overrides`` 显式安装，避免 HTTP 层成为 Infrastructure 组合根。
"""

from __future__ import annotations

from typing import Never

from fastapi import Depends, HTTPException

from app.application.ports.http_runtime import (
    QueryCapacityPort,
    UploadLimits,
)
from app.application.services.adhoc_dataset_service import AdhocDatasetService
from app.application.services.analysis_service import AnalysisService
from app.application.services.export_service import ExportLimits
from app.application.services.feedback_service import FeedbackService
from app.application.services.auth_service import Principal
from app.application.services.auth_use_case import AuthUseCase
from app.application.services.conversation_data_context_service import (
    ConversationDataContextService,
    ScopedConversationDataContextService,
)
from app.application.services.conversation_history_service import (
    ConversationHistoryService,
)
from app.application.services.conversation_memory_service import ConversationMemoryService
from app.application.services.dashboard_service import DashboardService
from app.application.services.data_source_context_service import DataSourceContextService
from app.application.services.masking_service import MaskingRule
from app.application.services.meta_admin_service import MetaAdminService
from app.application.services.query_service import QueryService
from app.application.services.query_operations_service import QueryOperationsService
from app.application.services.schema_sync_service import SchemaSyncService
from app.domain.access.policy import PolicySnapshot
from app.domain.privacy.masking import MaskingPolicy


def _missing(name: str) -> Never:
    raise RuntimeError(f"HTTP 依赖 {name} 尚未由 Bootstrap 装配")


def get_meta_session() -> object:
    """为评测 HTTP Adapter 暴露抽象会话依赖，具体会话由 Bootstrap 提供。"""
    return _missing("get_meta_session")


def get_current_principal() -> Principal:
    return _missing("get_current_principal")


def require_roles(*roles: str):
    """构造只依赖 Principal 端口的角色守卫。"""

    async def checker(
        principal: Principal = Depends(get_current_principal),
    ) -> Principal:
        if principal.role not in roles:
            raise HTTPException(
                status_code=403,
                detail=f"需要角色：{' / '.join(roles)}",
            )
        return principal

    return checker


def get_auth_use_case() -> AuthUseCase:
    return _missing("get_auth_use_case")


def get_dashboard_service() -> DashboardService:
    return _missing("get_dashboard_service")


def get_feedback_service() -> FeedbackService:
    return _missing("get_feedback_service")


def get_query_capacity() -> QueryCapacityPort:
    return _missing("get_query_capacity")


def get_upload_limits() -> UploadLimits:
    return _missing("get_upload_limits")


def get_export_limits() -> ExportLimits:
    return _missing("get_export_limits")


def get_access_policy() -> PolicySnapshot:
    return _missing("get_access_policy")


def get_masking_rules() -> tuple[MaskingRule, ...]:
    return _missing("get_masking_rules")


def get_masking_policy() -> MaskingPolicy:
    return _missing("get_masking_policy")


def get_adhoc_dataset_service() -> AdhocDatasetService:
    return _missing("get_adhoc_dataset_service")


def get_analysis_service() -> AnalysisService:
    return _missing("get_analysis_service")


def get_data_source_context_service() -> DataSourceContextService:
    return _missing("get_data_source_context_service")


def get_conversation_data_context_service() -> ConversationDataContextService:
    return _missing("get_conversation_data_context_service")


def get_scoped_conversation_data_context_service(
) -> ScopedConversationDataContextService:
    return _missing("get_scoped_conversation_data_context_service")


def get_conversation_history_service() -> ConversationHistoryService:
    return _missing("get_conversation_history_service")


def get_conversation_memory_service() -> ConversationMemoryService:
    return _missing("get_conversation_memory_service")


def get_meta_admin_service() -> MetaAdminService:
    return _missing("get_meta_admin_service")


def get_schema_sync_service() -> SchemaSyncService:
    return _missing("get_schema_sync_service")


def get_query_service() -> QueryService:
    return _missing("get_query_service")


def get_query_operations_service() -> QueryOperationsService:
    return _missing("get_query_operations_service")


__all__ = [name for name in globals() if name.startswith("get_")] + [
    "require_roles"
]
