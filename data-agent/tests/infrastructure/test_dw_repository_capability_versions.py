"""DW 仓储能力版本注入测试。"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.config.app_config import app_config
from app.infrastructure.cache.result_cache import result_cache
from app.infrastructure.persistence.mysql.repositories.dw_mysql_repository import (
    DWMysqlRepository,
)
from app.infrastructure.observability.execution_versions import (
    RuntimeExecutionVersionProvider,
)


@dataclass(frozen=True)
class _VersionProvider:
    version: str

    def version_for(self, capability: str) -> str:
        assert capability == "llm.sql"
        return self.version


class _Result:
    def mappings(self) -> "_Result":
        return self

    def all(self) -> list[dict]:
        return [{"value": 1}]


class _Session:
    def __init__(self) -> None:
        self.calls = 0

    def get_bind(self) -> SimpleNamespace:
        return SimpleNamespace(url=SimpleNamespace(database="warehouse"))

    async def connection(self) -> SimpleNamespace:
        async def get_raw_connection() -> SimpleNamespace:
            return SimpleNamespace(
                driver_connection=SimpleNamespace(thread_id=lambda: 4321)
            )

        return SimpleNamespace(get_raw_connection=get_raw_connection)

    async def execute(self, statement: object) -> _Result:
        del statement
        self.calls += 1
        return _Result()


@pytest.mark.asyncio
async def test_model_capability_version_isolates_result_cache() -> None:
    session = _Session()
    result_cache.clear()
    with (
        patch(
            "app.infrastructure.persistence.mysql.repositories.dw_mysql_repository.app_config.query.cache_enabled",
            True,
        ),
        patch(
            "app.infrastructure.persistence.mysql.repositories.dw_mysql_repository.app_config.query.cache_ttl_seconds",
            60,
        ),
    ):
        first = DWMysqlRepository(
            session,
            capability_versions=_VersionProvider("model-a"),
        )
        second = DWMysqlRepository(
            session,
            capability_versions=_VersionProvider("model-b"),
        )
        assert await first.execute_sql("SELECT 1 AS value") == [{"value": 1}]
        assert await first.execute_sql("SELECT 1 AS value") == [{"value": 1}]
        assert first.last_cache_hit is True
        assert await second.execute_sql("SELECT 1 AS value") == [{"value": 1}]

    assert session.calls == 2
    assert second.last_cache_hit is False
    result_cache.clear()


def test_runtime_model_version_changes_with_global_model_fallback() -> None:
    provider = RuntimeExecutionVersionProvider()
    with patch.object(app_config.llm, "model_name", "model-a"):
        first = provider.version_for("llm.sql")
    with patch.object(app_config.llm, "model_name", "model-b"):
        second = provider.version_for("llm.sql")

    assert first.startswith("sha256:")
    assert second.startswith("sha256:")
    assert first != second
