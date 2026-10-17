"""具体元数据投影适配器的版本与发布顺序回归。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.infrastructure.search.index_version import column_index_version
from app.infrastructure.search.metadata_index_adapter import MetadataIndexAdapter
from app.infrastructure.search.projection import (
    alias_manifest_fingerprint,
    load_alias_manifest,
    validate_point_strategy,
)


@pytest.mark.asyncio
async def test_column_adapter_embeds_before_replace_and_keeps_index_version() -> None:
    events: list[str] = []

    class _Embedding:
        async def aembed_documents(self, texts):
            events.append(f"embed:{len(texts)}")
            return [[float(index)] for index, _ in enumerate(texts)]

    class _ColumnRepository:
        async def recreate_collection(self) -> None:
            events.append("recreate")

        async def upsert_column_vectors(
            self,
            vectors,
            payloads,
            ids,
            **kwargs,
        ) -> None:
            del vectors, ids, kwargs
            events.append("upsert")
            self.payloads = payloads

    table = SimpleNamespace(id="fact_order", name="fact_order")
    columns = [SimpleNamespace(
        id="fact_order.order_amount",
        name="order_amount",
        type="decimal",
        role="measure",
        examples=[],
        description="订单金额",
        alias=["销售额"],
        table_id="fact_order",
        is_display=False,
    )]
    column_repo = _ColumnRepository()
    adapter = MetadataIndexAdapter(
        meta_repo=SimpleNamespace(session=None, workspace_id="workspace-a"),
        column_repo=column_repo,
        metric_repo=SimpleNamespace(status=AsyncMock()),
        training_repo=SimpleNamespace(status=AsyncMock()),
        embedding_client=_Embedding(),
    )

    count = await adapter.rebuild_columns([(table, columns)], batch_size=1)

    strategy = validate_point_strategy()
    manifest_hash = alias_manifest_fingerprint(load_alias_manifest())
    expected_version = column_index_version(
        columns,
        point_strategy=strategy,
        alias_manifest_hash=manifest_hash,
    )
    assert count == 1
    assert events == ["embed:1", "recreate", "upsert"]
    assert column_repo.payloads[0]["index_version"] == expected_version
