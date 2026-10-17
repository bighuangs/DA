"""运行期弹性（resilience）原语集合，供生产链路与 TODO-39 探针共用。

本模块提供三类弹性能力：
1. 容量治理（AsyncCapacityGovernor）：全局 + 工作空间双层并发限流，超时即拒绝
2. 单飞合并（AsyncSingleFlight）：相同 key 的并发请求合并为一次实际执行
3. 依赖调用包装（optional_dependency / required_dependency）：统一超时与降级证据收集
"""
from __future__ import annotations

import asyncio
import copy
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.agent_system.runtime.llm_attempts import normalize_degradation
from app.agent_system.runtime.performance import current_node_id
from app.foundation.concurrency import (
    AsyncSingleFlight as AsyncSingleFlight,
    singleflight as singleflight,
)


class CapacityExceededError(RuntimeError):
    """在配置的排队超时时间内未获得任何可用查询槽位时抛出。"""


class DependencyUnavailableError(RuntimeError):
    """
    硬依赖不可用异常，携带结构化的依赖名、原因码与详情。

    核心功能：将依赖调用中的超时/异常统一包装为带上下文的 RuntimeError，
             供上层在日志、降级证据中按 dependency / reason 维度归因。

    Args:
        dependency: 依赖名称（如 "es" / "qdrant"），用于归因
        reason: 原因码，目前为 "timeout" 或 "dependency_unavailable"
        detail: 原始异常信息（已截断），便于排查
    """

    def __init__(self, dependency: str, reason: str, detail: str):
        super().__init__(f"{dependency} 不可用（{reason}）：{detail}")
        self.dependency = dependency
        self.reason = reason
        self.detail = detail


@dataclass
class CapacitySnapshot:
    """容量治理快照，反映 governor 在某一时刻的并发统计状态。"""

    active: int      # 当前活跃查询数
    peak: int        # 历史峰值活跃数
    rejected: int    # 因超时被拒绝的累计请求数
    completed: int   # 已完成（已释放槽位）的累计请求数


class _CapacityLease:
    """
    容量槽位租约，作为 AsyncCapacityGovernor.acquire 的返回值用于 RAII 式释放。

    核心功能：以异步上下文管理器形式持有一个全局 + 一个工作空间槽位，
             在 __aexit__ / release() 时将两个槽位归还给对应信号量，
             并在 governor 的统计计数器中减活跃、加完成数。

    业务逻辑：
        1. 构造时记录所属 governor 与 workspace_id，标记 released=False
        2. __aenter__ 直接返回自身，实际获取动作在 acquire() 中已完成
        3. __aexit__ 委托给 release()
        4. release() 幂等：若已释放则直接返回；否则在锁内更新计数并释放两个信号量

    Args:
        governor: 持有该租约的 AsyncCapacityGovernor 实例
        workspace_id: 申请槽位的工作空间标识
    """

    def __init__(self, governor: "AsyncCapacityGovernor", workspace_id: str):
        self.governor = governor
        self.workspace_id = workspace_id
        # 防止重复释放的幂等标志
        self.released = False

    async def __aenter__(self):
        # 槽位已在 acquire() 中获取，这里仅返回自身供 as 使用
        return self

    async def __aexit__(self, exc_type, exc, tb):
        del exc_type, tb
        await self.release()

    async def release(self) -> None:
        # 幂等保护：多次调用 release / 重复退出上下文时只生效一次
        if self.released:
            return
        self.released = True
        # 在状态锁内更新计数，避免与 acquire 并发时的计数错乱
        async with self.governor._state_lock:
            self.governor._active -= 1
            self.governor._workspace_active[self.workspace_id] -= 1
            self.governor._completed += 1
        # 先释放工作空间信号量，再释放全局信号量（与获取顺序相反）
        self.governor._workspace_semaphores[self.workspace_id].release()
        self.governor._global_semaphore.release()


class AsyncCapacityGovernor:
    """
    异步容量治理器，对查询并发做全局 + 工作空间双层限流，并暴露可观测的释放状态。

    核心功能：通过两个 asyncio.Semaphore（全局 + 每工作空间）实现双层并发上限，
             配合排队超时机制在拿不到槽位时快速拒绝；同时维护活跃/峰值/拒绝/完成
             四类计数器，供监控与诊断使用。所有共享状态变更均在 _state_lock 内进行。

    业务逻辑：
        1. 构造时规范化三个上限参数，并创建全局信号量、工作空间信号量字典、状态锁与计数器
        2. acquire：先抢全局信号量，再抢工作空间信号量；超时则回滚已抢到的全局槽位并拒绝
        3. acquire 成功后在锁内累加 _active / _workspace_active，并更新峰值 _peak
        4. snapshot：在锁内一次性返回四项统计的快照
        5. workspace_active：在锁内返回指定工作空间的当前活跃数
    """

    def __init__(self, global_limit: int, workspace_limit: int, queue_timeout_seconds: float):
        # 全局并发上限，至少为 1
        self.global_limit = max(1, int(global_limit))
        # 工作空间并发上限，钳制在 [1, global_limit] 之间，避免单工作空间超过全局上限
        self.workspace_limit = max(1, min(int(workspace_limit), self.global_limit))
        # 排队等待超时，最小 0.01s，避免传入 0 导致永远立即超时
        self.queue_timeout_seconds = max(0.01, float(queue_timeout_seconds))
        # 全局信号量：限制整个进程同时进行的查询数
        self._global_semaphore = asyncio.Semaphore(self.global_limit)
        # 工作空间信号量字典：按需创建，每个工作空间独立限流
        self._workspace_semaphores: defaultdict[str, asyncio.Semaphore] = defaultdict(
            lambda: asyncio.Semaphore(self.workspace_limit)
        )
        # 工作空间活跃计数（与信号量计数同步维护，用于可观测查询）
        self._workspace_active: defaultdict[str, int] = defaultdict(int)
        # 状态锁：保护 _active / _peak / _rejected / _completed / _workspace_active 的并发更新
        self._state_lock = asyncio.Lock()
        # 以下为运行期统计计数器
        self._active = 0      # 当前活跃查询数
        self._peak = 0        # 历史峰值活跃数
        self._rejected = 0    # 因超时被拒绝的累计请求数
        self._completed = 0   # 已完成的累计请求数

    async def acquire(self, workspace_id: str) -> _CapacityLease:
        """
        为指定工作空间申请一个查询槽位，超时则抛 CapacityExceededError。

        业务逻辑：
            1. 校验非空 workspace_id，保证容量隔离键来自当前认证上下文
            2. 在 queue_timeout_seconds 内先抢全局信号量，再抢工作空间信号量
            3. 超时：若已抢到全局信号量则归还，并在锁内累加 _rejected，抛 CapacityExceededError
            4. 取消（CancelledError）：若已抢到全局信号量则归还，原样向上抛出
            5. 成功：在锁内累加 _active / _workspace_active，更新峰值，返回 _CapacityLease

        Args:
            workspace_id: 当前工作空间标识

        Returns:
            _CapacityLease: 持有全局 + 工作空间槽位的租约，需在结束时 release

        Raises:
            CapacityExceededError: 排队超时未获得槽位
        """
        workspace_id = str(workspace_id or "").strip()
        if not workspace_id:
            raise ValueError("容量治理必须绑定 workspace_id")
        # 标记是否已抢到全局信号量，用于异常路径下决定是否需要归还
        global_acquired = False
        try:
            # 在整体超时窗口内完成"全局 + 工作空间"两步获取
            async with asyncio.timeout(self.queue_timeout_seconds):
                await self._global_semaphore.acquire()
                global_acquired = True
                await self._workspace_semaphores[workspace_id].acquire()
        except TimeoutError as exc:
            # 超时回滚：若已抢到全局槽位则归还，避免泄漏
            if global_acquired:
                self._global_semaphore.release()
            async with self._state_lock:
                self._rejected += 1
            raise CapacityExceededError("当前工作空间查询并发已满，请稍后重试") from exc
        except asyncio.CancelledError:
            # 任务被取消时同样需要回滚已获取的全局槽位，然后原样抛出取消异常
            if global_acquired:
                self._global_semaphore.release()
            raise
        # 成功路径：在锁内更新计数与峰值
        async with self._state_lock:
            self._active += 1
            self._workspace_active[workspace_id] += 1
            self._peak = max(self._peak, self._active)
        return _CapacityLease(self, workspace_id)

    async def snapshot(self) -> CapacitySnapshot:
        """在锁内一次性返回当前活跃/峰值/拒绝/完成四项统计的快照。"""
        async with self._state_lock:
            return CapacitySnapshot(self._active, self._peak, self._rejected, self._completed)

    async def workspace_active(self, workspace_id: str) -> int:
        """返回指定工作空间的当前活跃查询数（在锁内读取，保证一致性）。"""
        async with self._state_lock:
            return self._workspace_active[workspace_id]


async def optional_dependency(
    dependency: str,
    operation: Callable[[], Awaitable[Any]],
    *,
    fallback: Any,
    timeout_seconds: float,
) -> tuple[Any, dict[str, Any] | None]:
    """
    执行一个"可选"依赖调用：超时或失败时返回降级值，并产出结构化降级证据。

    核心功能：在限时窗口内执行 operation，成功则原样返回且证据为 None；
             任何异常（除取消外）都被吞掉，改为返回 deepcopy 的 fallback 值，
             并通过 normalize_degradation 生成一份带 node_id / reason_code /
             duration_ms 等字段的降级证据，供上层记录"哪些依赖在何时降级了"。

    业务逻辑：
        1. 记录起始时间戳，用于计算耗时
        2. 在 max(0.001, timeout_seconds) 限时内执行 operation
        3. 成功：返回 (结果, None)
        4. CancelledError：原样向上抛出，不视为降级
        5. 其他异常：根据是否为 TimeoutError 选择 reason_code，组装降级证据并返回 (fallback, 证据)

    Args:
        dependency: 依赖名称，用于降级证据中的归因
        operation: 实际要执行的异步依赖调用
        fallback: 依赖不可用时的降级返回值（会被 deepcopy 隔离）
        timeout_seconds: 超时阈值，过小值会被钳制为 0.001s

    Returns:
        (结果, 降级证据) 二元组；成功时证据为 None，降级时结果为 fallback 的副本
    """
    started = time.perf_counter()
    try:
        async with asyncio.timeout(max(0.001, timeout_seconds)):
            return await operation(), None
    except asyncio.CancelledError:
        # 取消异常不视为降级，原样抛出以保留任务取消语义
        raise
    except Exception as exc:
        # 降级路径：返回 fallback 副本，并附结构化降级证据
        return copy.deepcopy(fallback), normalize_degradation({
            "node_id": current_node_id.get() or dependency,
            "route": "dependency",
            "reason_code": "timeout" if isinstance(exc, TimeoutError) else "dependency_unavailable",
            "detail": str(exc)[:300],
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "severity": "warning",
            "affects_core_path": True,
        })


async def required_dependency(
    dependency: str,
    operation: Callable[[], Awaitable[Any]],
    *,
    timeout_seconds: float,
) -> Any:
    """
    执行一个"硬"依赖调用：超时或失败统一抛 DependencyUnavailableError，但保留取消语义。

    核心功能：在限时窗口内执行 operation，成功则原样返回；任何异常（除取消外）
             都被归一化为 DependencyUnavailableError，并携带 dependency 名称、
             reason_code（timeout / dependency_unavailable）与截断后的异常详情，
             便于上层在统一位置记录与处理硬依赖故障。

    业务逻辑：
        1. 在 max(0.001, timeout_seconds) 限时内执行 operation
        2. 成功：原样返回结果
        3. CancelledError：原样向上抛出，保留任务取消语义
        4. 其他异常：根据是否为 TimeoutError 选择 reason_code，包装为 DependencyUnavailableError 抛出

    Args:
        dependency: 依赖名称，用于异常归因
        operation: 实际要执行的异步依赖调用
        timeout_seconds: 超时阈值，过小值会被钳制为 0.001s

    Returns:
        operation 的原始返回值

    Raises:
        DependencyUnavailableError: 依赖超时或抛出异常时统一抛出
        asyncio.CancelledError: 任务被取消时原样向上传播
    """
    try:
        async with asyncio.timeout(max(0.001, timeout_seconds)):
            return await operation()
    except asyncio.CancelledError:
        # 取消异常不包装，原样抛出
        raise
    except Exception as exc:
        reason = "timeout" if isinstance(exc, TimeoutError) else "dependency_unavailable"
        raise DependencyUnavailableError(dependency, reason, str(exc)[:300]) from exc
