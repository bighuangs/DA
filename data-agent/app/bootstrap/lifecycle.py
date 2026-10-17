"""进程资源生命周期的统一所有权边界。"""

from __future__ import annotations

from builtins import ExceptionGroup
from collections.abc import Callable, Iterable
from contextlib import asynccontextmanager
from typing import AsyncIterator, Protocol

from fastapi import FastAPI


class LifecycleResource(Protocol):
    """由组合根创建并按逆序释放的异步资源。"""

    async def start(self) -> None: ...

    async def close(self) -> None: ...


LifecycleResourceFactory = Callable[[], LifecycleResource]


class LifecycleCoordinator:
    """在进入生命周期时才实例化资源，避免导入阶段产生外部连接。"""

    def __init__(
        self,
        resource_factories: Iterable[LifecycleResourceFactory] = (),
    ) -> None:
        self._resource_factories = tuple(resource_factories)
        self._resources: list[LifecycleResource] = []
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    async def start(self) -> None:
        if self._active:
            raise RuntimeError("应用生命周期不能重复启动")
        started: list[LifecycleResource] = []
        try:
            for factory in self._resource_factories:
                resource = factory()
                started.append(resource)
                await resource.start()
        except BaseException as startup_error:
            cleanup_errors = await self._close_resources(started)
            if cleanup_errors:
                startup_error.add_note(
                    "启动回滚时有资源关闭失败："
                    + ", ".join(type(error).__name__ for error in cleanup_errors)
                )
            raise
        self._resources = started
        self._active = True

    async def close(self) -> None:
        resources, self._resources = self._resources, []
        self._active = False
        errors = await self._close_resources(resources)
        if errors:
            raise ExceptionGroup("应用生命周期资源关闭失败", errors)

    @staticmethod
    async def _close_resources(
        resources: Iterable[LifecycleResource],
    ) -> list[Exception]:
        errors: list[Exception] = []
        for resource in reversed(tuple(resources)):
            try:
                await resource.close()
            except Exception as error:
                errors.append(error)
        return errors

    @asynccontextmanager
    async def scope(self) -> AsyncIterator[None]:
        await self.start()
        try:
            yield
        finally:
            await self.close()


def attach_application_lifecycle(
    application: FastAPI,
    coordinator: LifecycleCoordinator,
) -> None:
    """在旧生命周期启动后接管新增资源，并在旧资源前完成释放。"""
    inherited_lifespan = application.router.lifespan_context

    @asynccontextmanager
    async def composed_lifespan(app: FastAPI) -> AsyncIterator[object]:
        async with inherited_lifespan(app) as inherited_state:
            async with coordinator.scope():
                yield inherited_state

    application.router.lifespan_context = composed_lifespan
