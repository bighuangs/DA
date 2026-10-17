"""Query Application 用例与 Agent/持久层组合根的回归门禁。"""

from __future__ import annotations

from pathlib import Path

from app.adapters.outbound.agent import Nl2SqlApplicationRuntime
from app.application.services.auth_service import Principal
from app.bootstrap.http_dependencies import provide_query_service
from app.infrastructure.persistence.mysql.query_persistence import MysqlQueryPersistence
from tests.architecture._imports import imports_from, matches_prefix


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUERY_SERVICE = PROJECT_ROOT / "app" / "application" / "services" / "query_service.py"


def test_query_service_only_depends_on_inner_application_boundaries() -> None:
    forbidden = (
        "app.adapters",
        "app.agent_system",
        "app.bootstrap",
        "app.config",
        "app.infrastructure",
        "fastapi",
        "langgraph",
        "sqlalchemy",
    )
    violations = [
        f"{reference.line}:{reference.module}"
        for reference in imports_from(PROJECT_ROOT, QUERY_SERVICE)
        if matches_prefix(reference.module, forbidden)
    ]
    assert violations == []


def test_bootstrap_composes_query_runtime_and_persistence_adapters() -> None:
    class _ChatRepository:
        session = object()

    class _DWRepository:
        last_cache_hit = False

    principal = Principal(
        user_id="user-1",
        username="tester",
        display_name="测试用户",
        workspace_id="workspace-1",
        workspace_name="测试空间",
        role="admin",
    )
    service = provide_query_service(
        dw_repo=_DWRepository(),
        meta_repo=object(),
        chat_repo=_ChatRepository(),
        value_repo=object(),
        column_repo=object(),
        metric_repo=object(),
        training_repo=object(),
        embedding_client=object(),
        source_context_service=object(),
        conversation_context_service=object(),
        access_policy=None,
        principal=principal,
        memory_service=None,
        graph_runner=object(),
    )

    assert isinstance(service.agent_runtime, Nl2SqlApplicationRuntime)
    assert isinstance(service.query_persistence, MysqlQueryPersistence)
    assert service.data_source_context_service is not None
    assert service.conversation_context_service is not None
    assert service.settings.history_limit >= 1
