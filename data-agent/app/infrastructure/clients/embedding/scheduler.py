"""TODO-60: bounded, observable single-model Embedding micro-batch scheduler.

The scheduler deliberately does not run model calls concurrently.  It merges
compatible requests before invoking one asynchronous dispatcher, while the
client's single-worker executor remains the final model-serialization boundary.
No raw input text is retained in metrics or status snapshots.
"""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


class EmbeddingCapacityError(RuntimeError):
    """The bounded scheduler cannot admit another request."""


class EmbeddingDeadlineExceeded(TimeoutError):
    """An embedding request exceeded its configured queue/execution deadline."""


class EmbeddingSchedulerClosed(RuntimeError):
    """The scheduler is draining or already closed."""


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * quantile) - 1))
    return round(ordered[index], 3)


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": round(max(values), 3) if values else 0.0,
    }


class EmbeddingRuntimeMetrics:
    """Small in-process rolling metrics window without input content."""

    def __init__(self, runtime_profile: str, window_size: int = 512):
        self.runtime_profile = runtime_profile
        self._window_size = max(16, int(window_size))
        self._queue_wait_ms: deque[float] = deque(maxlen=self._window_size)
        self._inference_ms: deque[float] = deque(maxlen=self._window_size)
        self._total_ms: deque[float] = deque(maxlen=self._window_size)
        self._batch_sizes: deque[int] = deque(maxlen=self._window_size)
        self._requests_per_batch: deque[int] = deque(maxlen=self._window_size)
        self.epoch_index = 0
        self.epoch_id = uuid.uuid4().hex
        self.epoch_reason = "runtime_initial"
        self.started_at = datetime.now(UTC).isoformat()
        self._started_monotonic = time.perf_counter()
        self._clear_observations()

    def _clear_observations(self) -> None:
        self._queue_wait_ms.clear()
        self._inference_ms.clear()
        self._total_ms.clear()
        self._batch_sizes.clear()
        self._requests_per_batch.clear()
        self.submitted = 0
        self.completed = 0
        self.rejected = 0
        self.cancelled = 0
        self.timed_out = 0
        self.failed = 0
        self.texts_completed = 0
        self.model_invocations = 0
        self.max_queue_depth = 0
        self.cancel_stages = {
            "before_queue": 0,
            "queued": 0,
            "dispatched": 0,
            "running": 0,
            "after_result": 0,
        }

    def reset_epoch(self, *, reason: str) -> dict[str, Any]:
        """建立新的纯观测 epoch，不改变调度队列、策略或模型状态。"""
        previous = self.snapshot()
        previous_epoch = dict(previous["epoch"])
        self.epoch_index += 1
        self.epoch_id = uuid.uuid4().hex
        self.epoch_reason = str(reason or "unspecified")
        self.started_at = datetime.now(UTC).isoformat()
        self._started_monotonic = time.perf_counter()
        self._clear_observations()
        active = self.snapshot()
        return {
            "schema": "embedding-runtime-metrics-reset-v1",
            "status": "pass",
            "reason": self.epoch_reason,
            "previous_epoch": previous_epoch,
            "active_epoch": dict(active["epoch"]),
            "previous_observations": {
                "requests": previous["requests"],
                "throughput": previous["throughput"],
                "latency_ms": previous["latency_ms"],
                "batching": previous["batching"],
            },
            "active_observations": {
                "requests": active["requests"],
                "throughput": active["throughput"],
                "latency_ms": active["latency_ms"],
                "batching": active["batching"],
            },
            "raw_text_persisted": False,
        }

    def submitted_request(self, queue_depth: int) -> None:
        self.submitted += 1
        self.max_queue_depth = max(self.max_queue_depth, max(0, int(queue_depth)))

    def rejected_request(self) -> None:
        self.rejected += 1

    def cancelled_request(self, stage: str, *, timed_out: bool = False) -> None:
        normalized = stage if stage in self.cancel_stages else "queued"
        self.cancel_stages[normalized] += 1
        if timed_out:
            self.timed_out += 1
        else:
            self.cancelled += 1

    def failed_request(self, count: int = 1) -> None:
        self.failed += max(1, int(count))

    def record_batch(
        self,
        *,
        request_count: int,
        text_count: int,
        queue_wait_ms: Sequence[float],
        inference_ms: float,
        total_ms: Sequence[float],
    ) -> None:
        self.model_invocations += 1
        self.completed += max(0, int(request_count))
        self.texts_completed += max(0, int(text_count))
        self._batch_sizes.append(max(0, int(text_count)))
        self._requests_per_batch.append(max(0, int(request_count)))
        self._queue_wait_ms.extend(max(0.0, float(value)) for value in queue_wait_ms)
        self._inference_ms.append(max(0.0, float(inference_ms)))
        self._total_ms.extend(max(0.0, float(value)) for value in total_ms)

    def snapshot(self) -> dict[str, Any]:
        elapsed = max(0.001, time.perf_counter() - self._started_monotonic)
        batch_histogram = {"1": 0, "2-4": 0, "5-8": 0, "9-16": 0, "17+": 0}
        for size in self._batch_sizes:
            if size <= 1:
                batch_histogram["1"] += 1
            elif size <= 4:
                batch_histogram["2-4"] += 1
            elif size <= 8:
                batch_histogram["5-8"] += 1
            elif size <= 16:
                batch_histogram["9-16"] += 1
            else:
                batch_histogram["17+"] += 1
        return {
            "runtime_profile": self.runtime_profile,
            "started_at": self.started_at,
            "epoch": {
                "schema": "embedding-runtime-metrics-epoch-v1",
                "id": self.epoch_id,
                "index": self.epoch_index,
                "reason": self.epoch_reason,
                "started_at": self.started_at,
            },
            "window_size": self._window_size,
            "requests": {
                "submitted": self.submitted,
                "completed": self.completed,
                "rejected": self.rejected,
                "cancelled": self.cancelled,
                "timed_out": self.timed_out,
                "failed": self.failed,
            },
            "throughput": {
                "requests_per_second": round(self.completed / elapsed, 3),
                "texts_per_second": round(self.texts_completed / elapsed, 3),
                "texts_completed": self.texts_completed,
                "model_invocations": self.model_invocations,
            },
            "latency_ms": {
                "queue_wait": _latency_summary(self._queue_wait_ms),
                "inference": _latency_summary(self._inference_ms),
                "total": _latency_summary(self._total_ms),
            },
            "batching": {
                "last_batch_texts": self._batch_sizes[-1] if self._batch_sizes else 0,
                "max_batch_texts_seen": max(self._batch_sizes, default=0),
                "max_requests_per_batch": max(self._requests_per_batch, default=0),
                "merged_batch_count": sum(
                    1 for value in self._requests_per_batch if value > 1
                ),
                "histogram": batch_histogram,
            },
            "max_queue_depth": self.max_queue_depth,
            "cancel_stages": dict(self.cancel_stages),
        }


@dataclass(slots=True)
class _EmbeddingRequest:
    request_id: str
    mode: str
    texts: tuple[str, ...]
    token_estimate: int
    future: asyncio.Future
    enqueued_at: float
    deadline_at: float | None
    state: str = "queued"
    cancelled: bool = False
    cancellation_recorded: bool = False


class MicroBatchEmbeddingScheduler:
    """Merge compatible requests into bounded query/document model batches."""

    def __init__(
        self,
        *,
        config: Any,
        dispatcher: Callable[[list[str], str], Awaitable[list[Any]]],
        token_estimator: Callable[[str], int],
    ):
        self.config = config
        self._dispatcher = dispatcher
        self._token_estimator = token_estimator
        self._query_queue: deque[_EmbeddingRequest] = deque()
        self._document_queue: deque[_EmbeddingRequest] = deque()
        self._condition = asyncio.Condition()
        self._worker: asyncio.Task | None = None
        self._closing = False
        self._closed = False
        self._running_requests = 0
        self._online_batches_since_document = 0
        self.metrics = EmbeddingRuntimeMetrics(
            "microbatch_inprocess_v1",
            int(getattr(config, "metrics_window_size", 512)),
        )

    @property
    def queue_capacity(self) -> int:
        return max(1, int(getattr(self.config, "queue_capacity", 128)))

    @property
    def max_batch_texts(self) -> int:
        return max(1, int(getattr(self.config, "max_batch_texts", 32)))

    @property
    def max_batch_tokens(self) -> int:
        return max(1, int(getattr(self.config, "max_batch_tokens", 8192)))

    def _queue_depth(self) -> int:
        return len(self._query_queue) + len(self._document_queue)

    def _ensure_worker(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(
                self._run(), name="data-agent-bge-m3-microbatch"
            )

    def _request_deadline(self, now: float) -> float | None:
        deadline_ms = int(getattr(self.config, "request_deadline_ms", 45000))
        return now + deadline_ms / 1000 if deadline_ms > 0 else None

    async def submit(self, texts: Sequence[str], mode: str) -> list[Any]:
        if mode not in {"query", "document"}:
            raise ValueError(f"未知 Embedding 编码模式：{mode}")
        now = time.perf_counter()
        loop = asyncio.get_running_loop()
        request = _EmbeddingRequest(
            request_id=str(uuid.uuid4()),
            mode=mode,
            texts=tuple(texts),
            token_estimate=sum(self._token_estimator(text) for text in texts),
            future=loop.create_future(),
            enqueued_at=now,
            deadline_at=self._request_deadline(now),
        )
        async with self._condition:
            self._discard_tombstones()
            if self._closing or self._closed:
                self.metrics.rejected_request()
                raise EmbeddingSchedulerClosed("BGE-M3 调度器正在关闭")
            if self._queue_depth() >= self.queue_capacity:
                self.metrics.rejected_request()
                raise EmbeddingCapacityError(
                    f"BGE-M3 调度队列已满（capacity={self.queue_capacity}）"
                )
            target = (
                self._query_queue if mode == "query" else self._document_queue
            )
            target.append(request)
            self.metrics.submitted_request(self._queue_depth())
            self._ensure_worker()
            self._condition.notify_all()

        try:
            if request.deadline_at is None:
                return await asyncio.shield(request.future)
            remaining = max(0.0, request.deadline_at - time.perf_counter())
            return await asyncio.wait_for(
                asyncio.shield(request.future), timeout=remaining
            )
        except TimeoutError as exc:
            await self._cancel_request(request, timed_out=True)
            raise EmbeddingDeadlineExceeded(
                "BGE-M3 请求超过调度 deadline"
            ) from exc
        except asyncio.CancelledError:
            await self._cancel_request(request, timed_out=False)
            raise

    async def _cancel_request(
        self, request: _EmbeddingRequest, *, timed_out: bool
    ) -> None:
        async with self._condition:
            if request.future.done() and request.state == "succeeded":
                stage = "after_result"
            else:
                stage = request.state
            request.cancelled = True
            if not request.future.done():
                request.future.cancel()
            if not request.cancellation_recorded:
                self.metrics.cancelled_request(stage, timed_out=timed_out)
                request.cancellation_recorded = True
            self._condition.notify_all()

    def _discard_tombstones_from(
        self, queue: deque[_EmbeddingRequest]
    ) -> None:
        kept: deque[_EmbeddingRequest] = deque()
        now = time.perf_counter()
        while queue:
            item = queue.popleft()
            expired = item.deadline_at is not None and item.deadline_at <= now
            if item.cancelled or item.future.cancelled():
                continue
            if expired:
                item.cancelled = True
                item.state = "queued"
                if not item.future.done():
                    item.future.set_exception(
                        EmbeddingDeadlineExceeded(
                            "BGE-M3 请求在队列中超过 deadline"
                        )
                    )
                if not item.cancellation_recorded:
                    self.metrics.cancelled_request("queued", timed_out=True)
                    item.cancellation_recorded = True
                continue
            kept.append(item)
        queue.extend(kept)

    def _discard_tombstones(self) -> None:
        self._discard_tombstones_from(self._query_queue)
        self._discard_tombstones_from(self._document_queue)

    def _choose_mode(self) -> str | None:
        has_query = bool(self._query_queue)
        has_document = bool(self._document_queue)
        if not has_query and not has_document:
            return None
        quota = max(1, int(getattr(self.config, "offline_corpus_quota", 4)))
        if has_query and (
            not has_document or self._online_batches_since_document < quota
        ):
            return "query"
        return "document"

    @staticmethod
    def _fits(
        request: _EmbeddingRequest,
        *,
        text_count: int,
        token_count: int,
        max_texts: int,
        max_tokens: int,
    ) -> bool:
        if text_count == 0:
            return True
        return (
            text_count + len(request.texts) <= max_texts
            and token_count + request.token_estimate <= max_tokens
        )

    async def _next_batch(self) -> list[_EmbeddingRequest] | None:
        async with self._condition:
            while True:
                self._discard_tombstones()
                if self._closing and self._queue_depth() == 0:
                    return None
                mode = self._choose_mode()
                if mode is not None:
                    break
                await self._condition.wait()

            queue = (
                self._query_queue if mode == "query" else self._document_queue
            )
            batch: list[_EmbeddingRequest] = []
            text_count = 0
            token_count = 0
            window_seconds = max(
                0.0, float(getattr(self.config, "batch_window_ms", 6.0)) / 1000
            )
            window_deadline = time.perf_counter() + window_seconds

            while True:
                self._discard_tombstones_from(queue)
                while queue:
                    candidate = queue[0]
                    if not self._fits(
                        candidate,
                        text_count=text_count,
                        token_count=token_count,
                        max_texts=self.max_batch_texts,
                        max_tokens=self.max_batch_tokens,
                    ):
                        break
                    candidate = queue.popleft()
                    candidate.state = "dispatched"
                    batch.append(candidate)
                    text_count += len(candidate.texts)
                    token_count += candidate.token_estimate
                    if (
                        text_count >= self.max_batch_texts
                        or token_count >= self.max_batch_tokens
                    ):
                        break

                if (
                    text_count >= self.max_batch_texts
                    or token_count >= self.max_batch_tokens
                    or window_seconds <= 0
                ):
                    break
                remaining = window_deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                except TimeoutError:
                    break

            if mode == "query":
                self._online_batches_since_document += 1
            else:
                self._online_batches_since_document = 0
            return batch

    async def _execute(self, batch: list[_EmbeddingRequest]) -> None:
        active = [
            request
            for request in batch
            if not request.cancelled and not request.future.cancelled()
        ]
        if not active:
            return
        started = time.perf_counter()
        for request in active:
            request.state = "running"
        self._running_requests = len(active)
        flattened = [text for request in active for text in request.texts]
        mode = active[0].mode
        queue_waits = [
            max(0.0, (started - request.enqueued_at) * 1000)
            for request in active
        ]
        try:
            outputs = await self._dispatcher(flattened, mode)
            if len(outputs) != len(flattened):
                raise RuntimeError(
                    "BGE-M3 调度拆批数量异常："
                    f"输入 {len(flattened)}，输出 {len(outputs)}"
                )
            finished = time.perf_counter()
            offset = 0
            completed = 0
            completed_texts = 0
            totals: list[float] = []
            for request in active:
                size = len(request.texts)
                result = outputs[offset: offset + size]
                offset += size
                if request.cancelled or request.future.cancelled():
                    continue
                request.state = "succeeded"
                request.future.set_result(result)
                completed += 1
                completed_texts += size
                totals.append((finished - request.enqueued_at) * 1000)
            self.metrics.record_batch(
                request_count=completed,
                text_count=completed_texts,
                queue_wait_ms=queue_waits,
                inference_ms=(finished - started) * 1000,
                total_ms=totals,
            )
        except BaseException as exc:
            self.metrics.failed_request(len(active))
            for request in active:
                request.state = "failed"
                if not request.future.done():
                    request.future.set_exception(exc)
            if isinstance(exc, asyncio.CancelledError):
                raise
        finally:
            self._running_requests = 0

    async def _run(self) -> None:
        while True:
            batch = await self._next_batch()
            if batch is None:
                break
            await self._execute(batch)
        self._closed = True

    def status(self) -> dict[str, Any]:
        snapshot = self.metrics.snapshot()
        return {
            "state": (
                "closed"
                if self._closed
                else "draining"
                if self._closing
                else "running"
                if self._worker is not None
                else "idle"
            ),
            "queue": {
                "capacity": self.queue_capacity,
                "depth": self._queue_depth(),
                "query_depth": len(self._query_queue),
                "document_depth": len(self._document_queue),
                "running_requests": self._running_requests,
                "max_depth_seen": snapshot["max_queue_depth"],
            },
            "policy": {
                "batch_window_ms": float(
                    getattr(self.config, "batch_window_ms", 6.0)
                ),
                "max_batch_texts": self.max_batch_texts,
                "max_batch_tokens": self.max_batch_tokens,
                "request_deadline_ms": int(
                    getattr(self.config, "request_deadline_ms", 45000)
                ),
                "online_query_priority": int(
                    getattr(self.config, "online_query_priority", 0)
                ),
                "offline_corpus_priority": int(
                    getattr(self.config, "offline_corpus_priority", 10)
                ),
                "offline_corpus_quota": int(
                    getattr(self.config, "offline_corpus_quota", 4)
                ),
            },
            "metrics": snapshot,
        }

    def reset_metrics_epoch(self, *, reason: str) -> dict[str, Any]:
        """仅在 scheduler 完全空闲时重置观测窗口。"""
        if self._closing or self._closed:
            raise EmbeddingSchedulerClosed("BGE-M3 调度器关闭期间不能重置 metrics")
        if self._queue_depth() or self._running_requests:
            raise EmbeddingCapacityError(
                "BGE-M3 metrics epoch 只能在 queue/running 均为 0 时重置"
            )
        return self.metrics.reset_epoch(reason=reason)

    async def close(self) -> dict[str, Any]:
        if self._closed:
            return self.status()
        async with self._condition:
            self._closing = True
            for queue in (self._query_queue, self._document_queue):
                while queue:
                    request = queue.popleft()
                    request.cancelled = True
                    request.state = "queued"
                    if not request.future.done():
                        request.future.cancel()
                    if not request.cancellation_recorded:
                        self.metrics.cancelled_request("queued")
                        request.cancellation_recorded = True
            self._condition.notify_all()

        worker = self._worker
        if worker is not None:
            if worker.done():
                # asyncio.run() cancels and joins outstanding tasks before it
                # closes its loop.  Such a cancelled worker is terminal and
                # reclaimable: the HybridEmbeddingClient separately drains the
                # loop-independent native inference future.  A real worker
                # exception, however, must remain fail-closed.
                if not worker.cancelled():
                    error = worker.exception()
                    if error is not None:
                        raise error
            else:
                current_loop = asyncio.get_running_loop()
                if worker.get_loop() is not current_loop:
                    raise EmbeddingSchedulerClosed(
                        "BGE-M3 调度 worker 仍在其他 event loop 运行，"
                        "无法安全排空"
                    )
                timeout = max(
                    0.001,
                    int(
                        getattr(
                            self.config,
                            "shutdown_drain_timeout_ms",
                            30000,
                        )
                    )
                    / 1000,
                )
                try:
                    await asyncio.wait_for(
                        asyncio.shield(worker), timeout=timeout
                    )
                except TimeoutError:
                    worker.cancel()
                    try:
                        await worker
                    except asyncio.CancelledError:
                        pass
            self._worker = None
        self._closed = True
        return self.status()
