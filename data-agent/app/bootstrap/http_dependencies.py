"""生产 HTTP 依赖的唯一具体组合根。"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.inbound.http import dependencies as http_ports
from app.bootstrap.worker import (
    memory_projection_worker,
)
from app.adapters.outbound.agent import Nl2SqlApplicationRuntime
from app.adapters.outbound.agent.nl2sql_capabilities import (
    Nl2SqlIndexVersionAdapter,
    Nl2SqlLLMAdapter,
    build_nl2sql_settings,
)
from app.adapters.outbound.llm import LLMAnalysisModel, Nl2SqlRouteCatalog
from app.agent_system.governance.resilience import (
    AsyncCapacityGovernor,
    CapacityExceededError,
)
from app.agent_system.governance.retrieval_budget import invalidate_retrieval_caches
from app.agent_system.runtime import GraphRunner
from app.application.ports.http_runtime import (
    AuthRepositoryPort,
    QueryCapacityExceeded,
    QueryCapacityPort,
    UploadLimits,
)
from app.application.ports.conversation import ConversationScopeReader
from app.application.ports.metadata_indexing import MetadataIndexSettings
from app.application.ports.query_settings import QueryServiceSettings
from app.application.ports.query_tracing import QueryTraceLLMSettings
from app.application.services.adhoc_dataset_service import AdhocDatasetService
from app.application.services.analysis_service import AnalysisService
from app.application.services.auth_service import (
    AuthTokenPolicy,
    AuthTokenService,
    InvalidAccessTokenError,
    Principal,
)
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
from app.application.services.export_service import ExportLimits
from app.application.services.feedback_service import FeedbackService
from app.application.services.masking_service import MaskingRule, compile_rules
from app.application.services.meta_admin_service import MetaAdminService
from app.application.services.query_service import QueryService
from app.application.services.query_operations_service import QueryOperationsService
from app.application.services.schema_sync_service import SchemaSyncService
from app.config.app_config import app_config
from app.domain.access.policy import PolicySnapshot
from app.domain.privacy.masking import MaskingPolicy
from app.infrastructure.cache.result_cache import result_cache
from app.infrastructure.clients.embedding_client_manager import embedding_client_manager
from app.infrastructure.clients.es_client_manager import es_client_manager
from app.infrastructure.clients.hybrid_embedding_client import HybridEmbeddingClient
from app.infrastructure.clients.mysql_client_manager import (
    dw_mysql_client_manager,
    meta_mysql_client_manager,
)
from app.infrastructure.clients.qdrant_client_manager import qdrant_client_manager
from app.infrastructure.clients.redis_client_manager import redis_client_manager
from app.infrastructure.config.conversation import load_conversation_settings
from app.infrastructure.config.data_sources import (
    adhoc_dataset_settings,
    data_source_context_settings,
)
from app.infrastructure.observability.execution_versions import (
    RuntimeExecutionVersionProvider,
)
from app.infrastructure.persistence.mysql.conversation_adapters import (
    SqlConversationDataContextStore,
    SqlConversationHistoryStore,
    SqlConversationMemoryPersistence,
)
from app.infrastructure.persistence.mysql.adhoc_dataset_adapter import (
    MysqlAdhocDatasetAdapter,
)
from app.infrastructure.persistence.mysql.data_source_context_adapters import (
    DWMysqlMappingQualityProbeAdapter,
    MysqlSourceMappingAdapter,
)
from app.infrastructure.persistence.mysql.repositories.access_policy_repository import (
    AccessPolicyRepository,
)
from app.infrastructure.persistence.mysql.repositories.auth_repository import AuthRepository
from app.infrastructure.persistence.mysql.repositories.chat_history_repository import (
    ChatHistoryRepository,
)
from app.infrastructure.persistence.mysql.repositories.conversation_history_repository import (
    ConversationHistoryRepository,
)
from app.infrastructure.persistence.mysql.repositories.dashboard_repository import (
    DashboardRepository,
)
from app.infrastructure.persistence.mysql.repositories.dw_mysql_repository import (
    DWMysqlRepository,
)
from app.infrastructure.persistence.mysql.repositories.feedback_repository import (
    FeedbackRepository,
)
from app.infrastructure.persistence.mysql.repositories.meta_mysql_repository import (
    MetaMysqlRepository,
)
from app.infrastructure.persistence.mysql.repositories.query_trace_repository import (
    QueryTraceRepository,
)
from app.infrastructure.persistence.mysql.query_persistence import MysqlQueryPersistence
from app.infrastructure.persistence.mysql.schema_sync_adapter import (
    MysqlSchemaSyncAdapter,
)
from app.infrastructure.search.elasticsearch.value_index_adapter import (
    ElasticsearchValueIndexAdapter,
)
from app.infrastructure.search.elasticsearch.repositories.conversation_memory_es_repository import (
    ConversationMemoryESRepository,
)
from app.infrastructure.search.elasticsearch.repositories.conversation_summary_es_repository import (
    ConversationSummaryESRepository,
)
from app.infrastructure.search.elasticsearch.repositories.value_es_repository import (
    ValueESRepository,
)
from app.infrastructure.search.qdrant.repositories.column_qdrant_repository import (
    ColumnQdrantRepository,
)
from app.infrastructure.search.qdrant.repositories.conversation_memory_qdrant_repository import (
    ConversationMemoryQdrantRepository,
)
from app.infrastructure.search.qdrant.repositories.conversation_summary_qdrant_repository import (
    ConversationSummaryQdrantRepository,
)
from app.infrastructure.search.qdrant.repositories.metric_qdrant_repository import (
    MetricQdrantRepository,
)
from app.infrastructure.search.qdrant.repositories.training_qdrant_repository import (
    TrainingQdrantRepository,
)
from app.infrastructure.search.metadata_index_adapter import MetadataIndexAdapter
from app.infrastructure.persistence.redis.repositories.conversation_working_memory_repository import (
    ConversationWorkingMemoryRepository,
)


_bearer = HTTPBearer(auto_error=False)
_require_admin = http_ports.require_roles("owner", "admin")
(
    _conversation_history_settings,
    _conversation_memory_settings,
) = load_conversation_settings()
_masking_policy: MaskingPolicy = _conversation_history_settings.masking_policy


class _AuthRepositoryAdapter:
    """为 AuthRepository 补充显式事务端口，避免 Router 接触 Session。"""

    def __init__(self, repository: AuthRepository, session: AsyncSession) -> None:
        self._repository = repository
        self._session = session

    async def get_user(self, user_id: str):
        return await self._repository.get_user(user_id)

    async def get_user_by_username(self, username: str):
        return await self._repository.get_user_by_username(username)

    async def create_user(self, **values: object):
        return await self._repository.create_user(**values)

    async def get_workspace(self, workspace_id: str):
        return await self._repository.get_workspace(workspace_id)

    async def create_workspace(self, **values: object):
        return await self._repository.create_workspace(**values)

    async def get_member(self, workspace_id: str, user_id: str):
        return await self._repository.get_member(workspace_id, user_id)

    async def list_workspaces(self, user_id: str):
        return await self._repository.list_workspaces(user_id)

    async def preferred_workspace_id(self, user):
        return await self._repository.preferred_workspace_id(user)

    async def list_members(self, workspace_id: str):
        return await self._repository.list_members(workspace_id)

    async def add_member(self, **values: object):
        return await self._repository.add_member(**values)

    async def update_member_role(self, member, role: str) -> None:
        await self._repository.update_member_role(member, role)

    async def delete_member(self, member) -> None:
        await self._repository.delete_member(member)

    async def commit(self) -> None:
        await self._session.commit()


class _ConversationScopeAdapter:
    def __init__(self, repository: ConversationHistoryRepository) -> None:
        self._repository = repository

    async def exists(self, chat_id: str) -> bool:
        return await self._repository.get_session(chat_id) is not None


class _CacheInvalidationAdapter:
    def invalidate_retrieval(self) -> None:
        invalidate_retrieval_caches()

    def clear_query_results(self) -> None:
        result_cache.clear()


class _QueryCapacityAdapter:
    def __init__(self, governor: AsyncCapacityGovernor) -> None:
        self._governor = governor

    async def acquire(self, workspace_id: str):
        try:
            return await self._governor.acquire(workspace_id)
        except CapacityExceededError as exc:
            raise QueryCapacityExceeded(str(exc)) from exc


_cache_invalidator = _CacheInvalidationAdapter()
_query_capacity = _QueryCapacityAdapter(
    AsyncCapacityGovernor(
        app_config.query.max_concurrent_queries,
        app_config.query.max_concurrent_queries_per_workspace,
        app_config.query.concurrency_queue_timeout_seconds,
    )
)
_upload_limits = UploadLimits(max_file_bytes=app_config.adhoc.max_file_bytes)
_execution_versions = RuntimeExecutionVersionProvider()
_auth_tokens = AuthTokenService(
    AuthTokenPolicy(
        secret_key=app_config.auth.secret_key,
        access_token_ttl_minutes=app_config.auth.access_token_ttl_minutes,
    )
)
_export_limits = ExportLimits(
    max_rows=app_config.query.export_max_rows,
    max_columns=app_config.query.export_max_columns,
    max_cells=app_config.query.export_max_cells,
    max_cell_chars=app_config.query.export_max_cell_chars,
)
_analysis_service = AnalysisService(LLMAnalysisModel())
_llm_routes = Nl2SqlRouteCatalog()
_trace_llm_settings = QueryTraceLLMSettings(
    default_model=app_config.llm.model_name,
    input_cost_per_million=app_config.llm.input_cost_per_million,
    output_cost_per_million=app_config.llm.output_cost_per_million,
    cost_currency=app_config.llm.cost_currency,
)
_metadata_index_settings = MetadataIndexSettings(
    max_distinct_values_per_column=app_config.es.max_distinct_values_per_column,
    schema_embedding_batch_size=32,
)
_nl2sql_settings = build_nl2sql_settings()
_nl2sql_llm = Nl2SqlLLMAdapter()
_nl2sql_index_versions = Nl2SqlIndexVersionAdapter()


async def provide_dw_session():
    async with dw_mysql_client_manager.session_factory() as session:
        yield session


async def provide_meta_session():
    async with meta_mysql_client_manager.session_factory() as session:
        yield session


async def provide_current_principal(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    session: AsyncSession = Depends(provide_meta_session),
) -> Principal:
    if not app_config.auth.enabled:
        raise HTTPException(status_code=503, detail="认证服务未启用")
    if not credentials:
        raise HTTPException(status_code=401, detail="请先登录")
    try:
        payload = _auth_tokens.decode(credentials.credentials)
    except InvalidAccessTokenError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    repo = AuthRepository(session)
    user = await repo.get_user(str(payload.get("sub")))
    workspace_id = str(payload.get("workspace_id") or "")
    workspace = await repo.get_workspace(workspace_id)
    member = (
        await repo.get_member(workspace_id, user.id)
        if user is not None and workspace is not None
        else None
    )
    if not user or not user.is_active or not workspace or not member:
        raise HTTPException(
            status_code=401,
            detail="用户或工作空间已失效，请重新登录",
        )
    return Principal(
        user_id=user.id,
        username=user.username,
        display_name=user.display_name,
        workspace_id=workspace.id,
        workspace_name=workspace.name,
        role=member.role,
    )


def provide_auth_repository(
    session: AsyncSession = Depends(provide_meta_session),
) -> AuthRepositoryPort:
    return _AuthRepositoryAdapter(AuthRepository(session), session)


def provide_auth_use_case(
    repository: AuthRepositoryPort = Depends(provide_auth_repository),
) -> AuthUseCase:
    return AuthUseCase(repository=repository, tokens=_auth_tokens)


def provide_meta_mysql_repo(
    session: AsyncSession = Depends(provide_meta_session),
    principal: Principal = Depends(provide_current_principal),
) -> MetaMysqlRepository:
    return MetaMysqlRepository(session, workspace_id=principal.workspace_id)


def provide_chat_history_repo(
    session: AsyncSession = Depends(provide_meta_session),
    principal: Principal = Depends(provide_current_principal),
) -> ChatHistoryRepository:
    return ChatHistoryRepository(
        session,
        owner_id=principal.user_id,
        workspace_id=principal.workspace_id,
    )


def provide_dashboard_repo(
    session: AsyncSession = Depends(provide_meta_session),
    principal: Principal = Depends(provide_current_principal),
) -> DashboardRepository:
    return DashboardRepository(
        session,
        owner_id=principal.user_id,
        workspace_id=principal.workspace_id,
    )


def provide_feedback_service(
    session: AsyncSession = Depends(provide_meta_session),
    principal: Principal = Depends(provide_current_principal),
) -> FeedbackService:
    return FeedbackService(
        FeedbackRepository(
            session,
            owner_id=principal.user_id,
            workspace_id=principal.workspace_id,
        )
    )


def provide_value_es_repo() -> ValueESRepository:
    return ValueESRepository(es_client_manager.client)


def provide_column_qdrant_repo() -> ColumnQdrantRepository:
    return ColumnQdrantRepository(qdrant_client_manager.client)


def provide_metric_qdrant_repo() -> MetricQdrantRepository:
    return MetricQdrantRepository(qdrant_client_manager.client)


def provide_training_qdrant_repo() -> TrainingQdrantRepository:
    return TrainingQdrantRepository(qdrant_client_manager.client)


def provide_embedding_client() -> HybridEmbeddingClient:
    return embedding_client_manager.client


def provide_metadata_projection_index(
    meta_repo: MetaMysqlRepository = Depends(provide_meta_mysql_repo),
    column_repo: ColumnQdrantRepository = Depends(provide_column_qdrant_repo),
    metric_repo: MetricQdrantRepository = Depends(provide_metric_qdrant_repo),
    training_repo: TrainingQdrantRepository = Depends(provide_training_qdrant_repo),
    embedding_client: HybridEmbeddingClient = Depends(provide_embedding_client),
) -> MetadataIndexAdapter:
    return MetadataIndexAdapter(
        meta_repo=meta_repo,
        column_repo=column_repo,
        metric_repo=metric_repo,
        training_repo=training_repo,
        embedding_client=embedding_client,
    )


async def provide_access_policy(
    session: AsyncSession = Depends(provide_meta_session),
    principal: Principal = Depends(provide_current_principal),
) -> PolicySnapshot:
    policies = await AccessPolicyRepository(
        session, principal.workspace_id
    ).list_enabled()
    return PolicySnapshot(
        policies=tuple(policies),
        user_id=principal.user_id,
        username=principal.username,
        workspace_id=principal.workspace_id,
        role=principal.role,
    )


async def provide_masking_rules(
    session: AsyncSession = Depends(provide_meta_session),
    principal: Principal = Depends(provide_current_principal),
) -> tuple[MaskingRule, ...]:
    items = await MetaMysqlRepository(
        session, principal.workspace_id
    ).list_masking_rules()
    return compile_rules(items)


def provide_dw_mysql_repo(
    session: AsyncSession = Depends(provide_dw_session),
    rules: tuple[MaskingRule, ...] = Depends(provide_masking_rules),
    principal: Principal = Depends(provide_current_principal),
) -> DWMysqlRepository:
    return DWMysqlRepository(
        session,
        masking_rules=rules,
        cache_scope=principal.workspace_id,
        capability_versions=_execution_versions,
    )


def provide_dashboard_service(
    repository: DashboardRepository = Depends(provide_dashboard_repo),
    dw_repo: DWMysqlRepository = Depends(provide_dw_mysql_repo),
    access_policy: PolicySnapshot = Depends(provide_access_policy),
) -> DashboardService:
    return DashboardService(
        store=repository,
        query=dw_repo,
        access_policy=access_policy,
    )


def provide_value_index(
    dw_repo: DWMysqlRepository = Depends(provide_dw_mysql_repo),
    value_repo: ValueESRepository = Depends(provide_value_es_repo),
) -> ElasticsearchValueIndexAdapter:
    return ElasticsearchValueIndexAdapter(
        dw_repo=dw_repo,
        value_repo=value_repo,
        settings=_metadata_index_settings,
    )


def provide_adhoc_dataset_service(
    principal: Principal = Depends(provide_current_principal),
    rules: tuple[MaskingRule, ...] = Depends(provide_masking_rules),
) -> AdhocDatasetService:
    return AdhocDatasetService(
        storage=MysqlAdhocDatasetAdapter(),
        owner_id=principal.user_id,
        workspace_id=principal.workspace_id,
        settings=adhoc_dataset_settings(),
        masking_rules=rules,
    )


def provide_data_source_context_service(
    adhoc_service: AdhocDatasetService = Depends(provide_adhoc_dataset_service),
    dw_repo: DWMysqlRepository = Depends(provide_dw_mysql_repo),
    access_policy: PolicySnapshot = Depends(provide_access_policy),
    principal: Principal = Depends(provide_current_principal),
) -> DataSourceContextService:
    return DataSourceContextService(
        adhoc_service=adhoc_service,
        dw_repo=dw_repo,
        mapping_store=MysqlSourceMappingAdapter(
            owner_id=principal.user_id,
            workspace_id=principal.workspace_id,
        ),
        mapping_probe=DWMysqlMappingQualityProbeAdapter(dw_repo),
        settings=data_source_context_settings(),
        access_policy=access_policy,
        owner_id=principal.user_id,
        workspace_id=principal.workspace_id,
    )


def provide_conversation_data_context_service(
    resolver: DataSourceContextService = Depends(provide_data_source_context_service),
    principal: Principal = Depends(provide_current_principal),
) -> ConversationDataContextService:
    return ConversationDataContextService(
        owner_id=principal.user_id,
        workspace_id=principal.workspace_id,
        resolver=resolver,
        store=SqlConversationDataContextStore(
            meta_mysql_client_manager.session_factory,
            owner_id=principal.user_id,
            workspace_id=principal.workspace_id,
        ),
    )


def provide_conversation_memory_service(
    embedding_client: HybridEmbeddingClient = Depends(provide_embedding_client),
    principal: Principal = Depends(provide_current_principal),
) -> ConversationMemoryService:
    working_memory = (
        ConversationWorkingMemoryRepository(
            redis_client_manager.client,
            owner_id=principal.user_id,
            workspace_id=principal.workspace_id,
            ttl_seconds=_conversation_memory_settings.working_ttl_seconds,
        )
        if redis_client_manager.client is not None
        else None
    )
    return ConversationMemoryService(
        owner_id=principal.user_id,
        workspace_id=principal.workspace_id,
        persistence=SqlConversationMemoryPersistence(
            meta_mysql_client_manager.session_factory,
            owner_id=principal.user_id,
            workspace_id=principal.workspace_id,
            settings=_conversation_memory_settings,
        ),
        embedding_client=embedding_client,
        es_repository=ConversationMemoryESRepository(
            es_client_manager.client,
            projection_version=_conversation_memory_settings.es_projection_version,
        ),
        qdrant_repository=ConversationMemoryQdrantRepository(
            qdrant_client_manager.client,
            projection_version=(
                _conversation_memory_settings.qdrant_projection_version
            ),
        ),
        summary_es_repository=ConversationSummaryESRepository(
            es_client_manager.client,
            projection_version=(
                _conversation_memory_settings.summary_es_projection_version
            ),
            manage_alias=False,
        ),
        summary_qdrant_repository=ConversationSummaryQdrantRepository(
            qdrant_client_manager.client,
            projection_version=(
                _conversation_memory_settings.summary_qdrant_projection_version
            ),
        ),
        working_memory=working_memory,
        notifier=memory_projection_worker,
        settings=_conversation_memory_settings,
    )


def provide_conversation_history_service(
    session: AsyncSession = Depends(provide_meta_session),
    principal: Principal = Depends(provide_current_principal),
    access_policy: PolicySnapshot = Depends(provide_access_policy),
    rules: tuple[MaskingRule, ...] = Depends(provide_masking_rules),
) -> ConversationHistoryService:
    return ConversationHistoryService(
        SqlConversationHistoryStore(
            session,
            owner_id=principal.user_id,
            workspace_id=principal.workspace_id,
        ),
        access_policy=access_policy,
        masking_rules=rules,
        settings=_conversation_history_settings,
    )


def provide_conversation_scope(
    session: AsyncSession = Depends(provide_meta_session),
    principal: Principal = Depends(provide_current_principal),
) -> ConversationScopeReader:
    return _ConversationScopeAdapter(
        ConversationHistoryRepository(
            session,
            owner_id=principal.user_id,
            workspace_id=principal.workspace_id,
        )
    )


def provide_scoped_conversation_data_context_service(
    scope: ConversationScopeReader = Depends(provide_conversation_scope),
    context: ConversationDataContextService = Depends(
        provide_conversation_data_context_service
    ),
) -> ScopedConversationDataContextService:
    return ScopedConversationDataContextService(scope=scope, context=context)


def provide_query_trace_reader(
    session: AsyncSession = Depends(provide_meta_session),
    principal: Principal = Depends(provide_current_principal),
) -> QueryTraceRepository:
    return QueryTraceRepository(
        session,
        owner_id=principal.user_id,
        workspace_id=principal.workspace_id,
    )


def provide_query_operations_service(
    scope: ConversationScopeReader = Depends(provide_conversation_scope),
    traces: QueryTraceRepository = Depends(provide_query_trace_reader),
    dw_repo: DWMysqlRepository = Depends(provide_dw_mysql_repo),
    source_context: DataSourceContextService = Depends(
        provide_data_source_context_service
    ),
    access_policy: PolicySnapshot = Depends(provide_access_policy),
) -> QueryOperationsService:
    return QueryOperationsService(
        scope=scope,
        traces=traces,
        query=dw_repo,
        source_context=source_context,
        access_policy=access_policy,
    )


def provide_access_policy_admin_repository(
    session: AsyncSession = Depends(provide_meta_session),
    principal: Principal = Depends(provide_current_principal),
) -> AccessPolicyRepository:
    return AccessPolicyRepository(session, principal.workspace_id)


def provide_query_capacity() -> QueryCapacityPort:
    return _query_capacity


def provide_nl2sql_graph_runner() -> GraphRunner:
    """声明 Bootstrap 私有的 GraphRunner 注入点。"""

    raise RuntimeError("NL2SQL GraphRunner 尚未由 HTTP 组合根装配")


def provide_upload_limits() -> UploadLimits:
    return _upload_limits


def provide_meta_admin_service(
    meta_repo: MetaMysqlRepository = Depends(provide_meta_mysql_repo),
    projection_index: MetadataIndexAdapter = Depends(
        provide_metadata_projection_index
    ),
    value_index: ElasticsearchValueIndexAdapter = Depends(provide_value_index),
    access_policy: PolicySnapshot = Depends(provide_access_policy),
    access_policy_admin: AccessPolicyRepository = Depends(
        provide_access_policy_admin_repository
    ),
    _principal: Principal = Depends(_require_admin),
) -> MetaAdminService:
    return MetaAdminService(
        meta_repo=meta_repo,
        projection_index=projection_index,
        value_index=value_index,
        cache_invalidator=_cache_invalidator,
        access_policy=access_policy,
        access_policy_admin=access_policy_admin,
        masking_policy=_masking_policy,
    )


def provide_query_service(
    dw_repo: DWMysqlRepository = Depends(provide_dw_mysql_repo),
    meta_repo: MetaMysqlRepository = Depends(provide_meta_mysql_repo),
    chat_repo: ChatHistoryRepository = Depends(provide_chat_history_repo),
    value_repo: ValueESRepository = Depends(provide_value_es_repo),
    column_repo: ColumnQdrantRepository = Depends(provide_column_qdrant_repo),
    metric_repo: MetricQdrantRepository = Depends(provide_metric_qdrant_repo),
    training_repo: TrainingQdrantRepository = Depends(provide_training_qdrant_repo),
    embedding_client: HybridEmbeddingClient = Depends(provide_embedding_client),
    source_context_service: DataSourceContextService = Depends(
        provide_data_source_context_service
    ),
    conversation_context_service: ConversationDataContextService = Depends(
        provide_conversation_data_context_service
    ),
    access_policy: PolicySnapshot = Depends(provide_access_policy),
    principal: Principal = Depends(provide_current_principal),
    memory_service: ConversationMemoryService = Depends(
        provide_conversation_memory_service
    ),
    graph_runner: GraphRunner = Depends(provide_nl2sql_graph_runner),
) -> QueryService:
    return QueryService(
        dw_mysql_repo=dw_repo,
        query_persistence=MysqlQueryPersistence(
            chat_repo.session,
            owner_id=principal.user_id,
            workspace_id=principal.workspace_id,
        ),
        data_source_context_service=source_context_service,
        conversation_context_service=conversation_context_service,
        graph_runner=Nl2SqlApplicationRuntime(
            graph_runner,
            capabilities={
                "dw_mysql_repo": dw_repo,
                "meta_mysql_repo": meta_repo,
                "value_es_repo": value_repo,
                "column_qdrant_repo": column_repo,
                "metric_qdrant_repo": metric_repo,
                "training_qdrant_repo": training_repo,
                "embedding_client": embedding_client,
                "llm": _nl2sql_llm,
                "index_versions": _nl2sql_index_versions,
                "settings": _nl2sql_settings,
            },
        ),
        settings=QueryServiceSettings(
            history_limit=app_config.chat.history_limit,
            analysis_enabled_default=app_config.query.analysis_enabled_default,
            memory_enabled=app_config.memory.enabled,
        ),
        llm_route_catalog=_llm_routes,
        trace_llm_settings=_trace_llm_settings,
        owner_id=principal.user_id,
        workspace_id=principal.workspace_id,
        access_policy=access_policy,
        memory_service=memory_service,
        execution_version_provider=_execution_versions,
    )


def provide_schema_sync_service(
    dw_repo: DWMysqlRepository = Depends(provide_dw_mysql_repo),
    meta_repo: MetaMysqlRepository = Depends(provide_meta_mysql_repo),
    projection_index: MetadataIndexAdapter = Depends(
        provide_metadata_projection_index
    ),
    value_index: ElasticsearchValueIndexAdapter = Depends(provide_value_index),
    _admin: Principal = Depends(_require_admin),
) -> SchemaSyncService:
    return SchemaSyncService(
        dw_repo=dw_repo,
        meta_repo=MysqlSchemaSyncAdapter(meta_repo),
        projection_index=projection_index,
        value_index=value_index,
        cache_invalidator=_cache_invalidator,
        settings=_metadata_index_settings,
    )


def build_http_dependency_overrides(
    graph_runner: GraphRunner,
) -> Mapping[Callable[..., object], Callable[..., object]]:
    """返回生产应用的完整依赖映射，不创建 Session 或外部连接。"""

    return {
        http_ports.get_meta_session: provide_meta_session,
        http_ports.get_current_principal: provide_current_principal,
        http_ports.get_auth_use_case: provide_auth_use_case,
        http_ports.get_dashboard_service: provide_dashboard_service,
        http_ports.get_feedback_service: provide_feedback_service,
        http_ports.get_query_capacity: provide_query_capacity,
        http_ports.get_upload_limits: provide_upload_limits,
        http_ports.get_export_limits: lambda: _export_limits,
        http_ports.get_access_policy: provide_access_policy,
        http_ports.get_masking_rules: provide_masking_rules,
        http_ports.get_masking_policy: lambda: _masking_policy,
        http_ports.get_adhoc_dataset_service: provide_adhoc_dataset_service,
        http_ports.get_analysis_service: lambda: _analysis_service,
        http_ports.get_data_source_context_service: (
            provide_data_source_context_service
        ),
        http_ports.get_conversation_data_context_service: (
            provide_conversation_data_context_service
        ),
        http_ports.get_scoped_conversation_data_context_service: (
            provide_scoped_conversation_data_context_service
        ),
        http_ports.get_conversation_history_service: (
            provide_conversation_history_service
        ),
        http_ports.get_conversation_memory_service: (
            provide_conversation_memory_service
        ),
        http_ports.get_meta_admin_service: provide_meta_admin_service,
        http_ports.get_schema_sync_service: provide_schema_sync_service,
        provide_nl2sql_graph_runner: lambda: graph_runner,
        http_ports.get_query_service: provide_query_service,
        http_ports.get_query_operations_service: provide_query_operations_service,
    }


__all__ = ["build_http_dependency_overrides"]
