"""Non-destructive safety tests for the opt-in TODO-96 runtime harness."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

import evals.legacy.todo96_runtime_acceptance as acceptance_module
from evals.legacy.todo96_runtime_acceptance import (
    AcceptanceFailure,
    Todo96RuntimeAcceptance,
    _persist_report,
)


def _acceptance() -> Todo96RuntimeAcceptance:
    return Todo96RuntimeAcceptance(
        run_id="unit-safety",
        capacity_segments=1001,
        canary_segments=8,
        lock_hold_ms=500,
        allow_nonlocal=False,
    )


@pytest.mark.asyncio
async def test_external_cleanup_attempts_qdrant_after_elasticsearch_failure():
    indices = SimpleNamespace(
        exists=AsyncMock(side_effect=RuntimeError("es unavailable")),
    )
    es_client = SimpleNamespace(indices=indices)
    qdrant_client = SimpleNamespace(
        collection_exists=AsyncMock(side_effect=[True, False]),
        delete_collection=AsyncMock(),
    )
    acceptance = _acceptance()

    with (
        patch.object(acceptance_module.es_client_manager, "client", es_client),
        patch.object(
            acceptance_module.qdrant_client_manager,
            "client",
            qdrant_client,
        ),
    ):
        result = await acceptance._cleanup_external()

    assert result["errors"]["elasticsearch"]["error_type"] == "RuntimeError"
    assert result["qdrant_collection_deleted"] is True
    assert result["qdrant_collection_absent"] is True
    qdrant_client.delete_collection.assert_awaited_once_with(
        collection_name=acceptance.qdrant_collection
    )


@pytest.mark.asyncio
async def test_run_cleanup_and_client_close_are_independent_best_effort():
    acceptance = _acceptance()
    acceptance._external_initialized = True
    acceptance.preflight = AsyncMock(side_effect=RuntimeError("preflight"))
    acceptance._cleanup_owners = AsyncMock(side_effect=RuntimeError("database"))
    acceptance._cleanup_external = AsyncMock(return_value={
        "errors": {
            "elasticsearch": {
                "error_type": "RuntimeError",
                "message": "external",
            }
        }
    })
    embedding_close = AsyncMock(side_effect=RuntimeError("embedding"))
    qdrant_close = AsyncMock()
    es_close = AsyncMock()
    mysql_close = AsyncMock()

    with (
        patch.object(acceptance_module.meta_mysql_client_manager, "init", Mock()),
        patch.object(
            acceptance_module.embedding_client_manager,
            "close",
            embedding_close,
        ),
        patch.object(
            acceptance_module.qdrant_client_manager,
            "close",
            qdrant_close,
        ),
        patch.object(acceptance_module.es_client_manager, "close", es_close),
        patch.object(
            acceptance_module.meta_mysql_client_manager,
            "close",
            mysql_close,
        ),
    ):
        with pytest.raises(AcceptanceFailure):
            await acceptance.run()

    acceptance._cleanup_owners.assert_awaited_once()
    acceptance._cleanup_external.assert_awaited_once()
    embedding_close.assert_awaited_once()
    qdrant_close.assert_awaited_once()
    es_close.assert_awaited_once()
    mysql_close.assert_awaited_once()
    assert set(acceptance.report["cleanup_errors"]) == {
        "database",
        "elasticsearch",
        "embedding_close",
    }


def test_complete_raw_report_is_persisted_atomically(tmp_path):
    report = {
        "run_id": "unit-report",
        "status": "passed",
        "checks": [{"name": "gate", "passed": True}],
        "execution": {"script_sha256": "abc"},
    }

    target = _persist_report(report, str(tmp_path))

    assert target.exists()
    assert not target.with_suffix(".json.tmp").exists()
    payload = target.read_text(encoding="utf-8")
    assert '"checks"' in payload
    assert '"script_sha256": "abc"' in payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("capacity", "canary", "allow_large", "message"),
    (
        (5001, 8, False, "require --allow-large"),
        (50_001, 8, True, "must be <= 50000"),
        (1001, 204, True, "must be <= 200"),
    ),
)
async def test_cli_rejects_unsafe_acceptance_sizes_before_initialization(
    tmp_path,
    capacity,
    canary,
    allow_large,
    message,
):
    arguments = SimpleNamespace(
        execute=True,
        capacity_segments=capacity,
        canary_segments=canary,
        lock_hold_ms=500,
        allow_nonlocal=False,
        allow_large=allow_large,
        report_dir=str(tmp_path),
    )

    with patch.object(
        acceptance_module,
        "_arguments",
        return_value=arguments,
    ):
        with pytest.raises(SystemExit, match=message):
            await acceptance_module._main()
