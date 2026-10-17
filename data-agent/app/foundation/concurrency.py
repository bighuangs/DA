"""不依赖具体运行时的并发协调原语。"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Awaitable, Callable
from typing import Any


class SingleFlightLeaderCancelledError(TimeoutError):
    """单飞 leader 被调用方取消；follower 按截止时间耗尽处理。"""


class AsyncSingleFlight:
    """按 key 合并并发异步调用，并隔离 follower 获得的可变结果。"""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._flights: dict[str, asyncio.Future[Any]] = {}
        self.leaders = 0
        self.followers = 0

    async def run(
        self,
        key: str,
        factory: Callable[[], Awaitable[Any]],
        *,
        follower_timeout_seconds: float | None = None,
    ) -> Any:
        """仅让首个调用者执行工厂，follower 可独立超时且不取消共享工作。"""

        loop = asyncio.get_running_loop()
        async with self._lock:
            future = self._flights.get(key)
            leader = future is None
            if leader:
                future = loop.create_future()
                self._flights[key] = future
                self.leaders += 1
            else:
                self.followers += 1

        if not leader:
            if follower_timeout_seconds is not None:
                timeout = float(follower_timeout_seconds)
                if timeout <= 0:
                    raise ValueError("singleflight follower 超时必须大于 0")
                return copy.deepcopy(
                    await asyncio.wait_for(
                        asyncio.shield(future),
                        timeout=timeout,
                    )
                )
            return copy.deepcopy(await asyncio.shield(future))

        try:
            value = await factory()
            if not future.done():
                future.set_result(copy.deepcopy(value))
            return value
        except asyncio.CancelledError:
            if not future.done():
                # leader 保留原始取消；follower 获得普通 Exception，避免其请求
                # 被错误地当成调用方主动取消并绕过结构化错误终态。
                future.set_exception(SingleFlightLeaderCancelledError(
                    "singleflight leader 被取消"
                ))
                future.exception()
            raise
        except BaseException as exc:
            if not future.done():
                future.set_exception(exc)
                # 主动消费异常，避免没有 follower 时 Future 回收产生告警。
                future.exception()
            raise
        finally:
            async with self._lock:
                if self._flights.get(key) is future:
                    self._flights.pop(key, None)

    async def inflight(self) -> int:
        """返回当前正在执行的 key 数量。"""

        async with self._lock:
            return len(self._flights)


singleflight = AsyncSingleFlight()

__all__ = [
    "AsyncSingleFlight",
    "SingleFlightLeaderCancelledError",
    "singleflight",
]
