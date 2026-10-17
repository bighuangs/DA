"""生产 FastAPI 应用装配入口。"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agent_system.agents.nl2sql.runtime import create_nl2sql_graph_runner
from app.agent_system.runtime import GraphRunner
from app.adapters.inbound.http.routers.adhoc_router import adhoc_router
from app.adapters.inbound.http.routers.admin_router import admin_router
from app.adapters.inbound.http.routers.analysis_router import analysis_router
from app.adapters.inbound.http.routers.auth_router import auth_router
from app.adapters.inbound.http.routers.conversation_router import conversation_router
from app.adapters.inbound.http.routers.dashboard_router import dashboard_router
from app.adapters.inbound.http.routers.feedback_router import feedback_router
from app.adapters.inbound.http.routers.query_router import query_router
from app.adapters.inbound.http.middleware import register_http_middleware
from app.bootstrap.runtime_lifecycle import lifespan
from app.bootstrap.http_dependencies import build_http_dependency_overrides
from app.bootstrap.settings import BootstrapSettings


_CORE_ROUTERS = (
    query_router,
    conversation_router,
    adhoc_router,
    auth_router,
    analysis_router,
    dashboard_router,
    feedback_router,
    admin_router,
)


def create_http_application(
    *,
    graph_runner: GraphRunner | None = None,
    settings: BootstrapSettings | None = None,
) -> FastAPI:
    """创建并装配一个生产 FastAPI 实例，不启动 ASGI 生命周期。"""
    resolved_settings = settings or BootstrapSettings.from_environment()
    resolved_graph_runner = graph_runner or create_nl2sql_graph_runner()
    app = FastAPI(lifespan=lifespan)
    app.state.evaluation_http_enabled = resolved_settings.evaluation_http_enabled
    app.dependency_overrides.update(
        build_http_dependency_overrides(resolved_graph_runner)
    )
    allowed_origins = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ALLOWED_ORIGINS",
            "http://127.0.0.1:5173,http://localhost:5173",
        ).split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_http_middleware(app)
    for router in _CORE_ROUTERS:
        app.include_router(router)
    if resolved_settings.evaluation_http_enabled:
        # Eval HTTP 只在显式启用后加载，避免生产进程默认依赖评测包。
        from evals.bootstrap.http import mount_evaluation_http

        mount_evaluation_http(app)
    return app
