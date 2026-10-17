"""不在模块导入时编译的 Graph 工厂。"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, TypeAlias, cast

from app.agent_system.runtime.capabilities import AgentCapabilities


class RunnableGraph(Protocol):
    """GraphRunner 所需的最小编译产物接口。"""

    def astream(
        self,
        state: Mapping[str, object],
        *,
        context: object,
        stream_mode: str,
        config: Mapping[str, object] | None = None,
    ) -> AsyncIterator[object]: ...


class CompilableGraph(Protocol):
    """LangGraph StateGraph 等构建器的最小编译接口。"""

    def compile(self) -> RunnableGraph | Awaitable[RunnableGraph]: ...


GraphBuilder: TypeAlias = Callable[
    [AgentCapabilities | None], CompilableGraph | Awaitable[CompilableGraph]
]


@dataclass(frozen=True, slots=True)
class ExplicitGraphAdapter:
    """显式构造旧节点并组装 Graph，避免导入旧的全局已编译实例。"""

    node_factory: Callable[[AgentCapabilities], Mapping[str, object]]
    graph_factory: Callable[[Mapping[str, object]], CompilableGraph]

    def __call__(self, capabilities: AgentCapabilities | None) -> CompilableGraph:
        if capabilities is None:
            raise ValueError("ExplicitGraphAdapter 必须注入 AgentCapabilities")
        nodes = dict(self.node_factory(capabilities))
        if not nodes:
            raise ValueError("显式 Graph Adapter 至少需要一个节点")
        return self.graph_factory(nodes)


class GraphFactory:
    """首次执行时才构建并编译 Graph，实例作用域由组合根决定。"""

    def __init__(
        self,
        *,
        builder: GraphBuilder,
        capabilities: AgentCapabilities | None = None,
    ) -> None:
        self._builder = builder
        self._capabilities = capabilities
        self._compiled: RunnableGraph | None = None
        self._compile_lock: asyncio.Lock | None = None
        self._compile_count = 0

    @property
    def is_compiled(self) -> bool:
        return self._compiled is not None

    @property
    def compile_count(self) -> int:
        return self._compile_count

    def _lock(self) -> asyncio.Lock:
        if self._compile_lock is None:
            self._compile_lock = asyncio.Lock()
        return self._compile_lock

    async def get_graph(self) -> RunnableGraph:
        if self._compiled is not None:
            return self._compiled
        async with self._lock():
            if self._compiled is not None:
                return self._compiled
            definition = self._builder(self._capabilities)
            if inspect.isawaitable(definition):
                definition = await definition
            compile_method = getattr(definition, "compile", None)
            if not callable(compile_method):
                raise TypeError("Graph builder 必须返回带 compile() 的显式定义")
            compiled = compile_method()
            if inspect.isawaitable(compiled):
                compiled = await compiled
            if not callable(getattr(compiled, "astream", None)):
                raise TypeError("Graph compile() 结果必须实现 astream()")
            self._compiled = cast(RunnableGraph, compiled)
            self._compile_count += 1
            return self._compiled
