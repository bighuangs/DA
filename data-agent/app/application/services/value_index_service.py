from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.application.ports.metadata_indexing import ValueIndexPublicationPort


class ValueIndexService:
    """为一个工作空间编排字段值索引的计算与原子发布。"""

    def __init__(
        self,
        publisher: ValueIndexPublicationPort,
        *,
        workspace_id: str,
    ) -> None:
        self.publisher = publisher
        self.workspace_id = str(workspace_id or "").strip()
        if not self.workspace_id:
            raise ValueError("值索引构建必须绑定 workspace_id")

    async def rebuild(self, columns: Sequence[Any]) -> dict:
        return await self.publisher.rebuild(
            workspace_id=self.workspace_id,
            columns=columns,
        )
