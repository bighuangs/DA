"""ASGI 进程的公开应用工厂。"""

from __future__ import annotations

from collections.abc import Iterable

from fastapi import FastAPI

from app.bootstrap.container import (
    BootstrapContainer,
    HttpApplicationFactory,
    create_container,
)
from app.bootstrap.lifecycle import LifecycleResourceFactory
from app.bootstrap.settings import BootstrapSettings


def create_app(
    *,
    container: BootstrapContainer | None = None,
    settings: BootstrapSettings | None = None,
    http_application_factory: HttpApplicationFactory | None = None,
    lifecycle_resource_factories: Iterable[LifecycleResourceFactory] = (),
) -> FastAPI:
    """显式构建 FastAPI；调用前不会加载旧 API 或创建外部客户端。"""
    if container is not None and (
        settings is not None
        or http_application_factory is not None
        or tuple(lifecycle_resource_factories)
    ):
        raise ValueError("传入 container 时不能再传入其他组合参数")

    resolved_container = container or create_container(
        settings=settings,
        http_application_factory=http_application_factory,
        lifecycle_resource_factories=lifecycle_resource_factories,
    )
    return resolved_container.build_http_application()
