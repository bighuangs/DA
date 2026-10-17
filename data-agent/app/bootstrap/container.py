"""组合根定义：只保存工厂，不在导入阶段创建运行时对象。"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from fastapi import FastAPI

from app.bootstrap.lifecycle import (
    LifecycleCoordinator,
    LifecycleResourceFactory,
    attach_application_lifecycle,
)
from app.bootstrap.settings import BootstrapSettings


HttpApplicationFactory = Callable[[], FastAPI]


def _default_http_application_factory(settings: BootstrapSettings) -> FastAPI:
    """显式构建时才加载 Router、Client Manager 与生产生命周期。"""

    from app.bootstrap.http_application import create_http_application

    return create_http_application(settings=settings)


@dataclass(slots=True)
class BootstrapContainer:
    """保存进程级依赖工厂，并显式构建 HTTP 应用。"""

    settings: BootstrapSettings
    lifecycle: LifecycleCoordinator
    http_application_factory: HttpApplicationFactory | None = None

    def build_http_application(self) -> FastAPI:
        application = (
            self.http_application_factory()
            if self.http_application_factory is not None
            else _default_http_application_factory(self.settings)
        )
        application.state.bootstrap_container = self
        application.state.bootstrap_settings = self.settings
        attach_application_lifecycle(application, self.lifecycle)
        return application


def create_container(
    *,
    settings: BootstrapSettings | None = None,
    http_application_factory: HttpApplicationFactory | None = None,
    lifecycle_resource_factories: Iterable[LifecycleResourceFactory] = (),
) -> BootstrapContainer:
    """创建轻量组合定义；所有资源工厂都延迟到生命周期启动。"""
    resolved_settings = settings or BootstrapSettings.from_environment()
    return BootstrapContainer(
        settings=resolved_settings,
        lifecycle=LifecycleCoordinator(lifecycle_resource_factories),
        http_application_factory=http_application_factory,
    )
