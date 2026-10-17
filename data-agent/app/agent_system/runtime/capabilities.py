"""Agent 节点可调用能力的抽象端口。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True, slots=True)
class EmbeddingVector:
    """同时承载稠密与稀疏向量的基础设施无关结果。"""

    dense: tuple[float, ...]
    sparse: tuple[tuple[int, float], ...] = ()


@runtime_checkable
class EmbeddingCapability(Protocol):
    async def embed(
        self,
        texts: Sequence[str],
        *,
        timeout_seconds: float | None = None,
    ) -> Sequence[EmbeddingVector]: ...


@runtime_checkable
class RetrievalCapability(Protocol):
    async def search(
        self,
        *,
        corpus: str,
        query: str,
        workspace_id: str,
        limit: int,
        timeout_seconds: float | None = None,
    ) -> Sequence[Mapping[str, object]]: ...


@runtime_checkable
class MetadataCapability(Protocol):
    async def describe(
        self,
        *,
        workspace_id: str,
        object_names: Sequence[str],
        timeout_seconds: float | None = None,
    ) -> Mapping[str, object]: ...


@runtime_checkable
class ModelCapability(Protocol):
    async def generate(
        self,
        *,
        operation: str,
        prompt: str,
        timeout_seconds: float | None = None,
    ) -> Mapping[str, object]: ...


@runtime_checkable
class SQLExecutionCapability(Protocol):
    async def execute(
        self,
        *,
        workspace_id: str,
        sql: str,
        row_limit: int,
        timeout_seconds: float | None = None,
    ) -> Sequence[Mapping[str, object]]: ...


@runtime_checkable
class AuditCapability(Protocol):
    async def record(self, event: Mapping[str, object]) -> None: ...


@dataclass(frozen=True, slots=True)
class AgentCapabilities:
    """由组合根显式注入、由 Graph 节点闭包使用的能力集合。"""

    embedding: EmbeddingCapability
    retrieval: RetrievalCapability
    metadata: MetadataCapability
    model: ModelCapability
    sql_execution: SQLExecutionCapability
    audit: AuditCapability | None = None
