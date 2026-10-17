"""NL2SQL 节点的基础设施无关能力上下文。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import NotRequired, Protocol, TypeVar, TypedDict

from app.agent_system.agents.nl2sql.contracts.retrieval import ValueCandidate
from app.agent_system.agents.nl2sql.llm import Nl2SqlLLMCapability
from app.agent_system.agents.nl2sql.settings import Nl2SqlSettings
from app.domain.access.policy import PolicySnapshot
from app.domain.data_sources.context import AssetAccessContext
from app.domain.retrieval.hybrid_embedding import HybridEmbedding


class ColumnMetadata(Protocol):
    """Agent 读取字段元数据时依赖的最小只读视图。"""

    id: str
    name: str
    type: str
    role: str
    description: str
    alias: Sequence[str] | None
    examples: Sequence[object] | None
    table_id: str
    is_display: bool


class TableMetadata(Protocol):
    """Agent 读取表元数据时依赖的最小只读视图。"""

    id: str
    name: str
    role: str
    description: str


class MetricMetadata(Protocol):
    """Agent 读取指标元数据时依赖的最小只读视图。"""

    id: str
    name: str
    description: str
    relevant_columns: Sequence[str] | None
    alias: Sequence[str] | None
    formula_rule: Mapping[str, object] | None


class TrainingMetadata(Protocol):
    """Agent 读取训练样例时依赖的最小只读视图。"""

    id: int
    question: str
    sql: str
    enabled: bool


class HybridEmbeddingCapability(Protocol):
    """查询侧混合向量编码能力。"""

    profile_fingerprint: str

    async def aembed_queries(
        self, texts: list[str]
    ) -> Sequence[HybridEmbedding]: ...

    async def aembed_documents(
        self, texts: list[str]
    ) -> Sequence[HybridEmbedding]: ...

    def status(self) -> Mapping[str, object]: ...


class HybridSearchResultView(Protocol):
    """混合检索结果的 Agent 只读视图。"""

    items: Sequence[Mapping[str, object]]
    evidence: Mapping[str, object]


class ColumnRetrievalCapability(Protocol):
    async def search(
        self,
        vector: HybridEmbedding,
        index_version: str,
        score_threshold: float | None = None,
        sparse_score_threshold: float | None = None,
        limit: int | None = None,
        mode: str = "hybrid",
    ) -> HybridSearchResultView: ...


class WorkspaceRetrievalCapability(Protocol):
    async def search(
        self,
        vector: HybridEmbedding,
        workspace_id: str,
        score_threshold: float | None = None,
        sparse_score_threshold: float | None = None,
        limit: int | None = None,
        mode: str = "hybrid",
    ) -> HybridSearchResultView: ...


class ValueRetrievalCapability(Protocol):
    async def active_manifest(self, workspace_id: str) -> dict[str, object]: ...

    async def search_with_evidence(
        self,
        keyword: str,
        *,
        workspace_id: str,
        query_mode: str = "full_text",
        index_version: str | None = None,
        active_manifest: Mapping[str, object] | None = None,
        allowed_table_ids: Sequence[str] | None = None,
        allowed_column_ids: Sequence[str] | None = None,
        table_id: str | None = None,
        column_id: str | None = None,
        page: int = 1,
        page_size: int | None = None,
    ) -> tuple[list[ValueCandidate], int, dict[str, object]]: ...


class MetadataCapability(Protocol):
    """NL2SQL 节点需要的元数据查询端口。"""

    workspace_id: str | None

    async def get_colun_info_by_id(
        self, column_id: str
    ) -> ColumnMetadata | None: ...

    async def get_column_infos_by_table_id(
        self, table_id: str
    ) -> Sequence[ColumnMetadata]: ...

    async def get_table_info_by_table_id(
        self, table_id: str
    ) -> TableMetadata | None: ...

    async def get_all_metric_infos(self) -> Sequence[MetricMetadata]: ...

    async def list_training_infos(
        self, search: str | None = None
    ) -> Sequence[TrainingMetadata]: ...

    async def list_table_infos(
        self,
    ) -> Sequence[tuple[TableMetadata, Sequence[ColumnMetadata]]]: ...


class SQLExecutionCapability(Protocol):
    """只读 DW 信息、校验与执行端口。"""

    async def get_db_infos(self) -> Mapping[str, str]: ...

    async def validate_sql(self, sql: str) -> object: ...

    async def execute_sql(
        self,
        sql: str,
        *,
        timeout_seconds: float | None = None,
    ) -> list[dict[str, object]]: ...


T = TypeVar("T")


class MetadataCallGate(Protocol):
    """限制一次 Execution 内并行元数据调用的运行时能力。"""

    async def run(self, operation: Callable[[], Awaitable[T]]) -> T: ...


class IndexVersionCapability(Protocol):
    """Agent 读取物化索引身份时依赖的只读能力。"""

    def column_version(self, columns: Sequence[ColumnMetadata]) -> str: ...

    def retrieval_version(self) -> str: ...


async def serialized_meta_call(
    context: "DataAgentContext",
    operation: Callable[[], Awaitable[T]],
) -> T:
    """通过注入的请求级 gate 执行元数据调用；独立单测可直接调用。"""

    gate = context.get("metadata_call_gate")
    return await (gate.run(operation) if gate is not None else operation())


class DataAgentContext(TypedDict):
    """每次 Invocation 注入的能力集合与不可变授权事实。"""

    embedding_client: HybridEmbeddingCapability
    column_qdrant_repo: ColumnRetrievalCapability
    metric_qdrant_repo: WorkspaceRetrievalCapability
    training_qdrant_repo: WorkspaceRetrievalCapability
    value_es_repo: ValueRetrievalCapability
    meta_mysql_repo: MetadataCapability
    dw_mysql_repo: SQLExecutionCapability
    access_policy: PolicySnapshot
    llm: Nl2SqlLLMCapability
    index_versions: IndexVersionCapability
    settings: Nl2SqlSettings
    workspace_id: NotRequired[str]
    metadata_call_gate: NotRequired[MetadataCallGate]
    asset_access_context: NotRequired[AssetAccessContext]
    # 以下覆盖只允许受信 Runtime/评测装配设置，API 输入不能直接赋值。
    force_model_sql: NotRequired[bool]
    force_model_sql_reason: NotRequired[str]
    model_sql_fallback_enabled: NotRequired[bool]


__all__ = [
    "ColumnMetadata",
    "ColumnRetrievalCapability",
    "DataAgentContext",
    "HybridEmbeddingCapability",
    "HybridSearchResultView",
    "IndexVersionCapability",
    "MetadataCapability",
    "MetricMetadata",
    "MetadataCallGate",
    "SQLExecutionCapability",
    "TableMetadata",
    "TrainingMetadata",
    "ValueRetrievalCapability",
    "WorkspaceRetrievalCapability",
    "serialized_meta_call",
]
