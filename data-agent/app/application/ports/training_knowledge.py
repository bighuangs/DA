"""SQL 训练知识导入用例的稳定端口。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class TrainingSeed:
    """待导入的问题与 SQL 样例。"""

    question: str
    sql: str
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class TrainingKnowledgeItem:
    """已持久化且可用于构建投影的训练样例。"""

    id: int
    question: str
    sql: str
    enabled: bool


class TrainingKnowledgeStore(Protocol):
    """以 workspace 为边界替换训练知识的持久化端口。"""

    async def replace_all(
        self,
        *,
        workspace_id: str,
        seeds: Sequence[TrainingSeed],
    ) -> list[TrainingKnowledgeItem]: ...


__all__ = ["TrainingKnowledgeItem", "TrainingKnowledgeStore", "TrainingSeed"]
