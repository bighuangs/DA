"""SQL 训练知识导入与检索投影用例。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.application.ports.metadata_indexing import MetadataProjectionPort
from app.application.ports.training_knowledge import TrainingKnowledgeStore, TrainingSeed


@dataclass(frozen=True, slots=True)
class TrainingKnowledgeImportResult:
    """一次导入的可审计结果摘要。"""

    workspace_id: str
    persisted_count: int
    indexed_count: int


class TrainingKnowledgeService:
    """原子替换 workspace 样例，随后发布其检索投影。"""

    def __init__(
        self,
        *,
        store: TrainingKnowledgeStore,
        projection: MetadataProjectionPort,
    ) -> None:
        self._store = store
        self._projection = projection

    async def replace_and_rebuild(
        self,
        *,
        workspace_id: str,
        seeds: Sequence[TrainingSeed],
    ) -> TrainingKnowledgeImportResult:
        normalized_workspace_id = str(workspace_id or "").strip()
        if not normalized_workspace_id:
            raise ValueError("导入训练知识必须显式指定 workspace_id")

        items = await self._store.replace_all(
            workspace_id=normalized_workspace_id,
            seeds=tuple(seeds),
        )
        indexed_count = await self._projection.rebuild_training(
            normalized_workspace_id,
            items,
        )
        return TrainingKnowledgeImportResult(
            workspace_id=normalized_workspace_id,
            persisted_count=len(items),
            indexed_count=indexed_count,
        )


__all__ = ["TrainingKnowledgeImportResult", "TrainingKnowledgeService"]
