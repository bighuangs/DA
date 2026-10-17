"""数据来源具体适配器的 SQL 与配置投影回归。"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app.domain.data_sources.models import SourceMappingRecord
from app.infrastructure.config.data_sources import (
    adhoc_dataset_settings,
    data_source_context_settings,
)
from app.infrastructure.persistence.mysql import data_source_context_adapters as module
from app.infrastructure.persistence.mysql.data_source_context_adapters import (
    DWMysqlMappingQualityProbeAdapter,
    MysqlSourceMappingAdapter,
)


class _MappingResult:
    def __init__(self, value: dict) -> None:
        self.value = value

    def mappings(self):
        return self

    def one(self) -> dict:
        return self.value


class _ProbeSession:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.results = iter(
            (
                {"non_null_count": 100, "distinct_count": 98},
                {"left_rows": 80, "left_distinct": 75},
                {"joined_rows": 82, "matched_distinct": 70},
            )
        )

    async def execute(self, statement):
        self.queries.append(str(statement))
        return _MappingResult(next(self.results))


class _SessionContext:
    def __init__(self, session) -> None:
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback


class _MappingSession:
    def __init__(self) -> None:
        self.commit_count = 0
        self.refresh_count = 0

    async def commit(self) -> None:
        self.commit_count += 1

    async def refresh(self, item) -> None:
        del item


class _MappingManager:
    def __init__(self, session) -> None:
        self.session = session

    def session_factory(self):
        return _SessionContext(self.session)


@pytest.mark.asyncio
async def test_dw_mapping_probe_returns_only_bounded_aggregates() -> None:
    session = _ProbeSession()
    adapter = DWMysqlMappingQualityProbeAdapter(SimpleNamespace(session=session))

    stats = await adapter.probe(
        left_table="da_upload_a",
        left_column="c_001",
        right_table="dim_region",
        right_column="region_code",
    )

    assert stats.right_non_null_count == 100
    assert stats.right_distinct_count == 98
    assert stats.left_row_count == 80
    assert stats.left_distinct_count == 75
    assert stats.joined_row_count == 82
    assert stats.matched_distinct_count == 70
    assert len(session.queries) == 3
    assert all("SELECT" in value for value in session.queries)


@pytest.mark.asyncio
async def test_dw_mapping_probe_rejects_untrusted_identifiers_before_sql() -> None:
    session = _ProbeSession()
    adapter = DWMysqlMappingQualityProbeAdapter(SimpleNamespace(session=session))

    with pytest.raises(ValueError, match="非法资源标识"):
        await adapter.probe(
            left_table="da_upload_a; DROP TABLE x",
            left_column="c_001",
            right_table="dim_region",
            right_column="region_code",
        )
    assert session.queries == []


def test_runtime_config_is_copied_into_frozen_application_settings() -> None:
    adhoc = adhoc_dataset_settings()
    context = data_source_context_settings()

    assert adhoc.max_selected_datasets == context.max_selected_datasets
    assert adhoc.max_file_bytes > 0


@pytest.mark.asyncio
async def test_source_mapping_adapter_owns_orm_transactions(monkeypatch) -> None:
    rows: dict[str, object] = {}

    class _Repository:
        def __init__(self, session, *, owner_id, workspace_id) -> None:
            self.session = session
            self.owner_id = owner_id
            self.workspace_id = workspace_id

        async def get(self, mapping_id):
            item = rows.get(mapping_id)
            if item is None:
                return None
            if item.owner_id != self.owner_id or item.workspace_id != self.workspace_id:
                return None
            return item

        async def create(self, item):
            item.created_at = datetime.now(UTC).replace(tzinfo=None)
            rows[item.id] = item
            return item

        async def reject(self, mapping_id, reason):
            item = await self.get(mapping_id)
            if item is None:
                return None
            item.status = "rejected"
            item.rejection_reason = reason
            await self.session.commit()
            return item

    monkeypatch.setattr(module, "SourceMappingRepository", _Repository)
    session = _MappingSession()
    adapter = MysqlSourceMappingAdapter(
        owner_id="owner-a",
        workspace_id="workspace-a",
        manager=_MappingManager(session),
    )
    item = SourceMappingRecord(
        id="mapping-a",
        owner_id="owner-a",
        workspace_id="workspace-a",
        status="proposed",
        intent="upload_dw_row_join",
        dataset_version_ids=("version-a",),
        mappings=({"left": {}, "right": {}},),
        evidence={"auto_join_allowed": False},
        fingerprint="sha256:mapping",
    )

    created = await adapter.create(item)
    confirmed = await adapter.save_confirmation(
        "mapping-a",
        mappings=created.mappings,
        evidence={"auto_join_allowed": True},
        status="confirmed",
        confirmed_at=datetime.now(UTC).replace(tzinfo=None),
    )
    rejected = await adapter.reject("mapping-a", "用户拒绝")

    assert created.dataset_version_ids == ("version-a",)
    assert confirmed is not None and confirmed.status == "confirmed"
    assert rejected is not None and rejected.status == "rejected"
    assert session.commit_count == 2
