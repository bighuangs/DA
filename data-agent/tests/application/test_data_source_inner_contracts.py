"""上传数据与来源上下文的 Application 边界回归。"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.application.ports.data_sources import (
    AdhocDatasetSettings,
    DataSourceContextSettings,
)
from app.application.services.adhoc_dataset_service import (
    AdhocDatasetService,
    DatasetCleanupError as LegacyDatasetCleanupError,
)
from app.application.services.data_source_context_service import (
    DataSourceContextService,
)
from app.domain.data_sources.models import (
    AdhocDatasetRecord,
    DatasetCleanupError,
    MappingQualityStats,
)


class _AdhocStorage:
    def __init__(self) -> None:
        self.materialization = None
        self.cleanup_calls = 0

    async def cleanup_expired(self) -> int:
        self.cleanup_calls += 1
        return 0

    async def find_idempotent(self, *args):
        del args
        return None

    async def quota_usage(self, *args):
        del args
        return {"active_versions": 0, "bytes": 0, "rows": 0}

    async def latest_for_asset(self, *args):
        del args
        return None

    async def materialize(self, spec):
        self.materialization = spec
        now = datetime.now(UTC).replace(tzinfo=None)
        return replace(
            spec.record,
            status="ready",
            created_at=now,
            updated_at=now,
        )


class _MappingStore:
    async def get(self, mapping_id):
        del mapping_id
        return None


class _QualityProbe:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def probe(self, **kwargs):
        self.calls.append(kwargs)
        return MappingQualityStats(
            right_non_null_count=100,
            right_distinct_count=100,
            left_row_count=80,
            left_distinct_count=80,
            joined_row_count=80,
            matched_distinct_count=72,
        )


@pytest.mark.asyncio
async def test_upload_validation_builds_an_immutable_materialization_command() -> None:
    storage = _AdhocStorage()
    settings = AdhocDatasetSettings(max_file_bytes=1_024)
    service = AdhocDatasetService(
        storage=storage,
        owner_id="owner-a",
        workspace_id="workspace-a",
        settings=settings,
    )

    result = await service.create_dataset(
        "orders.csv",
        b"name,amount\nalice,12\nbob,20\n",
    )

    spec = storage.materialization
    assert spec is not None
    assert spec.record.status == "materializing"
    assert spec.record.owner_id == "owner-a"
    assert spec.physical_columns == ("c_001", "c_002")
    assert spec.column_types == ("TEXT", "BIGINT")
    assert spec.rows == (("alice", "12"), ("bob", "20"))
    assert result["status"] == "ready"
    assert "table_name" not in result
    assert storage.cleanup_calls == 1


@pytest.mark.asyncio
async def test_mapping_quality_uses_probe_port_and_preserves_gate_thresholds() -> None:
    probe = _QualityProbe()
    service = DataSourceContextService(
        adhoc_service=SimpleNamespace(),
        dw_repo=SimpleNamespace(),
        mapping_store=_MappingStore(),
        mapping_probe=probe,
        access_policy=SimpleNamespace(),
        owner_id="owner-a",
        workspace_id="workspace-a",
        settings=DataSourceContextSettings(max_selected_datasets=2),
    )
    mapping = {
        "left": {"table": "upload_1_a", "column": "c_001"},
        "right": {
            "table": "dim_region",
            "column": "region_code",
            "source_kind": "official",
        },
        "quality": {"type_compatible": True},
    }

    evaluated = await service._evaluate_mapping(
        mapping,
        logical_to_physical={"upload_1_a": "da_upload_a"},
        official_columns={("dim_region", "region_code")},
        intent="upload_dw_row_join",
    )

    assert probe.calls == [
        {
            "left_table": "da_upload_a",
            "left_column": "c_001",
            "right_table": "dim_region",
            "right_column": "region_code",
        }
    ]
    assert evaluated["quality"]["coverage"] == 0.9
    assert evaluated["quality"]["fanout"] == 1.0
    assert evaluated["quality"]["right_uniqueness"] == 1.0
    assert evaluated["quality"]["gate_passed"] is True


def test_settings_are_frozen_and_services_have_no_outer_imports() -> None:
    settings = AdhocDatasetSettings()
    with pytest.raises(FrozenInstanceError):
        settings.max_rows = 1

    root = Path(__file__).resolve().parents[2]
    forbidden = ("sqlalchemy", "app.config", "app.infrastructure")
    for relative in (
        "app/application/services/adhoc_dataset_service.py",
        "app/application/services/data_source_context_service.py",
    ):
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
            elif isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
        assert not [
            name
            for name in imported
            if any(
                name == prefix or name.startswith(prefix + ".") for prefix in forbidden
            )
        ], relative


def test_dataset_record_is_an_orm_free_snapshot() -> None:
    item = AdhocDatasetRecord(
        id="dataset-a",
        asset_id="asset-a",
        version_id="version-a",
        version_no=1,
        asset_kind="managed_upload",
        retention_mode="temporary",
        owner_id="owner-a",
        workspace_id="workspace-a",
        table_name="da_upload_a",
        filename="orders.csv",
        file_format="csv",
        row_count=0,
        column_count=1,
    )
    assert item.__class__.__module__ == "app.domain.data_sources.models"
    assert LegacyDatasetCleanupError is DatasetCleanupError
