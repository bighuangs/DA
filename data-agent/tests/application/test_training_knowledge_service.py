from __future__ import annotations

import pytest

from app.application.ports.training_knowledge import TrainingKnowledgeItem, TrainingSeed
from app.application.services.training_knowledge_service import TrainingKnowledgeService


class _Store:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[TrainingSeed, ...]]] = []

    async def replace_all(self, *, workspace_id, seeds):
        normalized = tuple(seeds)
        self.calls.append((workspace_id, normalized))
        return [
            TrainingKnowledgeItem(
                id=index,
                question=seed.question,
                sql=seed.sql,
                enabled=seed.enabled,
            )
            for index, seed in enumerate(normalized, start=1)
        ]


class _Projection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[TrainingKnowledgeItem]]] = []

    async def rebuild_training(self, workspace_id, items):
        normalized = list(items)
        self.calls.append((workspace_id, normalized))
        return len([item for item in normalized if item.enabled])


@pytest.mark.asyncio
async def test_replace_and_rebuild_uses_workspace_scoped_ports() -> None:
    store = _Store()
    projection = _Projection()
    service = TrainingKnowledgeService(store=store, projection=projection)
    seeds = (
        TrainingSeed(question="q1", sql="select 1"),
        TrainingSeed(question="q2", sql="select 2", enabled=False),
    )

    result = await service.replace_and_rebuild(
        workspace_id=" workspace-a ",
        seeds=seeds,
    )

    assert result.workspace_id == "workspace-a"
    assert result.persisted_count == 2
    assert result.indexed_count == 1
    assert store.calls == [("workspace-a", seeds)]
    assert projection.calls[0][0] == "workspace-a"


@pytest.mark.asyncio
async def test_replace_and_rebuild_rejects_missing_workspace() -> None:
    service = TrainingKnowledgeService(store=_Store(), projection=_Projection())

    with pytest.raises(ValueError, match="workspace_id"):
        await service.replace_and_rebuild(workspace_id=" ", seeds=())
