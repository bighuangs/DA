"""Data Agent 自有的 BGE-M3 本地 Embedding 适配器。

本模块负责把 BGE-M3（FlagEmbedding）封装成 Data Agent 可直接调用的异步客户端，
输出“稠密向量 + 学习型稀疏词权重”的混合向量（HybridEmbedding），供 Qdrant 检索使用。

设计要点：
- 不复用 STT/SQLBot 等外部项目的实现或运行时导入；FlagEmbedding 从 Data Agent
  自己的依赖集中惰性加载，模型目录是外部可配置的只读资产。
- BGE-M3 是同步且可能阻塞事件循环的本地推理，因此所有模型调用都通过单线程
  ThreadPoolExecutor 串行化执行，并用 asyncio.shield 保护，避免被取消时
  在底层触发并发模型调用。
- 通过 profile_fingerprint 把“模型 + 配置 + 元数据指纹”聚合成一个稳定哈希，
  作为索引/缓存的可观测契约，便于发现配置漂移。
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import math
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from app.config.app_config import EmbeddingConfig, EmbeddingSchedulerConfig
from app.infrastructure.clients.embedding.scheduler import (
    EmbeddingRuntimeMetrics,
    MicroBatchEmbeddingScheduler,
)
from app.domain.retrieval.hybrid_embedding import HybridEmbedding


class HybridEmbeddingRuntimeError(RuntimeError):
    """配置的 BGE-M3 运行时或输出契约不可用（依赖缺失、模型目录不完整、输出异常等）。"""


def _small_file_fingerprint(model_path: Path) -> str:
    """对模型目录中的小体积元数据文件做指纹，用于检测模型目录是否被替换或漂移。

    只读 config.json / tokenizer_config.json / special_tokens_map.json 这几个
    小文件，避免对大权重文件做哈希（既慢又无必要）。
    """
    digest = hashlib.sha256()
    found = False
    # 选取模型目录中体积小但能反映模型身份/分词器配置的元数据文件
    for name in ("config.json", "tokenizer_config.json", "special_tokens_map.json"):
        path = model_path / name
        if not path.is_file():
            continue
        found = True
        # 把文件名与文件内容一起喂给哈希，避免不同文件内容相同导致指纹碰撞
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    # 至少命中一个元数据文件才返回 sha256 指纹，否则标记为“缺失模型元数据”
    return f"sha256:{digest.hexdigest()}" if found else "missing-model-metadata"


class HybridEmbeddingClient:
    """可回退串行基线 + 有界动态微批，且不阻塞 asyncio 事件循环。

    两种运行模式共享单线程线程池 + 两把异步锁（init/encode）：
    - 模型懒加载，首次编码时才真正加载权重；
    - ``serial`` 保留原有逐调用串行路径；
    - ``microbatch`` 将同模式请求有界合批，但模型调用仍严格串行。
    """

    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        model_factory: Callable[[], Any] | None = None,
    ):
        self.config = config
        # 可注入的模型工厂，主要用于测试时替换为 mock 模型，避免真正加载权重
        self._model_factory = model_factory
        # 真正加载好的 BGE-M3 模型实例，懒加载；None 表示尚未初始化
        self._model: Any | None = None
        # 初始化锁：保证并发首次编码时模型只被加载一次（double-check 模式）
        self._init_lock = asyncio.Lock()
        # 编码锁：把并发编码请求串行化，单线程池是最终串行边界，此锁避免任务堆积
        self._encode_lock = asyncio.Lock()
        # 单线程线程池：BGE-M3 同步推理的唯一执行场所，max_workers=1 保证串行
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="data-agent-bge-m3"
        )
        # Track the executor-native futures rather than only their event-loop
        # wrappers.  An asyncio waiter may be cancelled (or its loop may be
        # closed) while the synchronous model call is still running; the
        # concurrent future remains safe to observe and drain from a later
        # lifecycle loop.
        self._lifecycle_lock = threading.Lock()
        self._active_inference_futures: set[ConcurrentFuture[Any]] = set()
        self._active_async_operations: set[ConcurrentFuture[None]] = set()
        self._closing = False
        self._close_in_progress = False
        self._close_failed = False
        self._last_close_error: str | None = None
        self._executor_shutdown = False
        # 关闭标志：close() 后不再接受新的编码请求
        self._closed = False
        # 实际生效的设备字符串（cpu / cuda:0），加载模型时确定
        self._device: str | None = None
        # 模型元数据指纹缓存，避免重复读盘计算
        self._metadata_fingerprint: str | None = None
        configured_mode = str(
            getattr(self.config, "runtime_mode", "serial") or "serial"
        ).strip().lower()
        self.runtime_mode = configured_mode
        scheduler_config = getattr(self.config, "scheduler", None)
        self.scheduler_config = scheduler_config or EmbeddingSchedulerConfig()
        self._serial_waiters = 0
        self._serial_metrics = EmbeddingRuntimeMetrics(
            "serial_inprocess_v1",
            int(getattr(self.scheduler_config, "metrics_window_size", 512)),
        )
        self._scheduler = (
            MicroBatchEmbeddingScheduler(
                config=self.scheduler_config,
                dispatcher=self._dispatch_microbatch,
                token_estimator=self._estimate_tokens,
            )
            if self.runtime_mode == "microbatch"
            else None
        )

    @property
    def model_path(self) -> Path:
        """模型目录路径，支持环境变量与 ~ 展开，返回 Path 对象。"""
        return Path(os.path.expandvars(str(self.config.model_path))).expanduser()

    @property
    def profile_fingerprint(self) -> str:
        """聚合“契约 + 模型 + revision + 元数据指纹 + 关键编码参数”的稳定指纹。

        用于索引/缓存的可观测性：任何一项发生漂移（如换模型、改 dense_size、
        改 normalize_dense），指纹都会变化，从而提示需要重建索引。
        """
        if self._metadata_fingerprint is None:
            # 首次访问时计算并缓存模型元数据指纹
            self._metadata_fingerprint = _small_file_fingerprint(self.model_path)
        payload = {
            "contract": "data-agent-bge-m3-dense-sparse-v1",
            "model": self.config.model,
            "revision": self.config.revision,
            "model_metadata": self._metadata_fingerprint,
            "dense_size": self.config.dense_size,
            "max_length": self.config.max_length,
            "normalize_dense": self.config.normalize_dense,
        }
        # 用排序键 + 紧凑分隔符序列化，保证不同字段顺序下指纹一致
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return f"sha256:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"

    @property
    def runtime_profile(self) -> str:
        """Return the executable runtime profile, independent of vector semantics."""
        return (
            "microbatch_inprocess_v1"
            if self.runtime_mode == "microbatch"
            else "serial_inprocess_v1"
        )

    @property
    def runtime_profile_fingerprint(self) -> str:
        """Fingerprint scheduling/deployment settings without invalidating vectors."""
        payload = {
            "contract": "data-agent-embedding-runtime-v1",
            "embedding_profile": self.profile_fingerprint,
            "runtime_profile": self.runtime_profile,
            "scheduler": {
                name: getattr(self.scheduler_config, name, None)
                for name in (
                    "queue_capacity",
                    "batch_window_ms",
                    "max_batch_texts",
                    "max_batch_tokens",
                    "request_deadline_ms",
                    "online_query_priority",
                    "offline_corpus_priority",
                    "offline_corpus_quota",
                    "shutdown_drain_timeout_ms",
                )
            },
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return f"sha256:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"

    def _resolve_device(self) -> str:
        """解析实际运行设备：显式配置优先，auto 时按 torch.cuda 可用性自动选择。"""
        configured = str(self.config.device or "auto").strip().lower()
        if configured != "auto":
            # 用户显式指定了设备（cpu / cuda:0 等），直接采用
            return configured
        try:
            import torch
        except ImportError:
            # 未安装 torch 时只能用 CPU
            return "cpu"
        # 有 CUDA 才用 cuda:0，否则回退 CPU
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def validate_runtime(self) -> dict:
        """启动期/加载前运行时校验：依赖、模型目录与关键文件是否齐备。

        校验通过返回当前 status()，校验失败抛 HybridEmbeddingRuntimeError，
        让上层在真正编码前就暴露配置问题，而非在首次编码时才报错。
        """
        if self.runtime_mode not in {"serial", "microbatch"}:
            raise HybridEmbeddingRuntimeError(
                "BGE-M3 runtime_mode 仅支持 serial / microbatch；"
                "remote 需要独立容量证据和部署评审后再启用"
            )
        if self.runtime_mode == "microbatch":
            positive_fields = (
                "queue_capacity",
                "max_batch_texts",
                "max_batch_tokens",
                "request_deadline_ms",
                "shutdown_drain_timeout_ms",
            )
            invalid = [
                name
                for name in positive_fields
                if int(getattr(self.scheduler_config, name, 0) or 0) <= 0
            ]
            if invalid:
                raise HybridEmbeddingRuntimeError(
                    "BGE-M3 调度参数必须为正数：" + "、".join(invalid)
                )
            if float(getattr(self.scheduler_config, "batch_window_ms", -1)) < 0:
                raise HybridEmbeddingRuntimeError(
                    "BGE-M3 batch_window_ms 不能为负数"
                )
        path = self.model_path
        # 1) 校验 Data Agent 自己的依赖 FlagEmbedding 是否安装
        if importlib.util.find_spec("FlagEmbedding") is None:
            raise HybridEmbeddingRuntimeError(
                "缺少 Data Agent 依赖 FlagEmbedding；请执行 uv sync"
            )
        # 2) 校验模型目录存在
        if not path.is_dir():
            raise HybridEmbeddingRuntimeError(f"BGE-M3 模型目录不存在：{path}")
        # 3) 校验模型目录至少包含 config.json，避免指向空目录
        if not (path / "config.json").is_file():
            raise HybridEmbeddingRuntimeError(f"BGE-M3 模型目录不完整（缺 config.json）：{path}")
        return self.status()

    def _build_model(self):
        """真正构造 BGE-M3 模型实例。

        优先使用注入的 model_factory（测试场景）；否则惰性导入 FlagEmbedding
        并按配置初始化 BGEM3FlagModel，固定输出 dense+sparse，关闭 colbert。
        """
        if self._model_factory is not None:
            # 测试场景：交给注入的工厂构造 mock 模型
            return self._model_factory()
        from FlagEmbedding import BGEM3FlagModel

        # 在构造模型前确定设备，供 use_fp16 判定与状态上报
        self._device = self._resolve_device()
        return BGEM3FlagModel(
            str(self.model_path),
            # fp16 仅在 CUDA 上启用，CPU 上强制关闭以避免不支持
            use_fp16=bool(self.config.use_fp16 and self._device.startswith("cuda")),
            devices=self._device,
            # 稠密向量是否归一化为单位向量（Qdrant cosine 检索前置条件）
            normalize_embeddings=bool(self.config.normalize_dense),
            batch_size=max(1, int(self.config.batch_size)),
            query_max_length=max(1, int(self.config.max_length)),
            passage_max_length=max(1, int(self.config.max_length)),
            # 固定契约：只取 dense + sparse，不取 colbert 向量以控制成本
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

    async def _run_serial(self, function: Callable[..., Any], *args) -> Any:
        """把同步的模型调用投递到单线程池执行，并用 shield 保护取消语义。

        取消一个 await 中的任务不会触发第二次并发模型调用：单 worker 的线程池
        始终是最终的串行边界。底层 concurrent future 会被显式跟踪，
        因此即使上层取消，close() 也会在允许下一个 client 前有界排空。
        """
        loop = asyncio.get_running_loop()
        with self._lifecycle_lock:
            if self._closed:
                raise HybridEmbeddingRuntimeError("BGE-M3 客户端已关闭")
            if self._closing or self._executor_shutdown:
                raise HybridEmbeddingRuntimeError(
                    "BGE-M3 客户端正在关闭，不再接受新推理"
                )
            try:
                future = self._executor.submit(function, *args)
            except RuntimeError as exc:
                raise HybridEmbeddingRuntimeError(
                    "BGE-M3 推理线程池已关闭"
                ) from exc
            self._active_inference_futures.add(future)
        future.add_done_callback(self._discard_inference_future)
        # wrap_future preserves run_in_executor-style asyncio semantics while
        # retaining the loop-independent concurrent future for shutdown drain.
        wrapped = asyncio.wrap_future(future, loop=loop)
        return await asyncio.shield(wrapped)

    def _discard_inference_future(self, future: ConcurrentFuture[Any]) -> None:
        """Remove a completed executor call from the lifecycle drain set."""
        with self._lifecycle_lock:
            self._active_inference_futures.discard(future)

    def _active_inference_snapshot(self) -> list[ConcurrentFuture[Any]]:
        with self._lifecycle_lock:
            return [
                future
                for future in self._active_inference_futures
                if not future.done()
            ]

    def _begin_async_operation(self) -> ConcurrentFuture[None]:
        """Register an encode operation until all async write-back has ended."""
        with self._lifecycle_lock:
            if self._closed:
                raise HybridEmbeddingRuntimeError("BGE-M3 客户端已关闭")
            if self._closing or self._executor_shutdown:
                raise HybridEmbeddingRuntimeError(
                    "BGE-M3 客户端正在关闭，不再接受新推理"
                )
            operation: ConcurrentFuture[None] = ConcurrentFuture()
            self._active_async_operations.add(operation)
            return operation

    def _finish_async_operation(
        self, operation: ConcurrentFuture[None]
    ) -> None:
        with self._lifecycle_lock:
            if not operation.done():
                operation.set_result(None)
            self._active_async_operations.discard(operation)

    def _active_work_snapshot(self) -> list[ConcurrentFuture[Any]]:
        with self._lifecycle_lock:
            return [
                future
                for future in (
                    *self._active_inference_futures,
                    *self._active_async_operations,
                )
                if not future.done()
            ]

    @property
    def accepting_requests(self) -> bool:
        """Whether a manager may safely reuse this client for new work."""
        with self._lifecycle_lock:
            return not (
                self._closed
                or self._closing
                or self._close_failed
                or self._executor_shutdown
            )

    def _drain_timeout_seconds(self) -> float:
        return max(
            0.001,
            int(
                getattr(
                    self.scheduler_config,
                    "shutdown_drain_timeout_ms",
                    30000,
                )
                or 30000
            )
            / 1000,
        )

    async def _drain_active_work(self) -> None:
        """Drain native calls and their asyncio continuations before success."""
        active = self._active_work_snapshot()
        if not active:
            return
        wrapped = [
            asyncio.shield(asyncio.wrap_future(future)) for future in active
        ]
        try:
            await asyncio.wait_for(
                asyncio.gather(*wrapped, return_exceptions=True),
                timeout=self._drain_timeout_seconds(),
            )
        except TimeoutError as exc:
            remaining = len(self._active_work_snapshot())
            if remaining == 0:
                return
            raise HybridEmbeddingRuntimeError(
                "BGE-M3 关闭排空超时："
                f"仍有 {remaining} 个推理/异步回写任务在运行；"
                "客户端已 fail-close，禁止初始化新实例"
            ) from exc

    async def _ensure_model(self):
        """懒加载模型：首次调用时加载，后续直接复用。

        使用 _init_lock + double-check，保证并发首次编码时模型只被构造一次，
        避免重复加载权重造成显存/内存浪费。
        """
        if self._model is not None:
            return self._model
        async with self._init_lock:
            if self._model is None:
                # 加载前先做运行时校验，提前暴露依赖/路径问题
                self.validate_runtime()
                model = await self._run_serial(self._build_model)
                # A native future may complete before its asyncio wrapper gets
                # CPU time.  Do not let that late delivery resurrect a client
                # after close() has begun.
                with self._lifecycle_lock:
                    if self._closing or self._closed:
                        raise HybridEmbeddingRuntimeError(
                            "BGE-M3 客户端关闭期间收到迟到的模型加载结果"
                        )
                    self._model = model
        return self._model

    def _encode_sync(self, texts: list[str], mode: str) -> dict:
        """同步编码实现：在单线程池中执行。

        根据 mode 选择 encode_queries / encode_corpus，若模型未提供这两个
        细分方法则回退到通用 encode，保证对不同 FlagEmbedding 版本兼容。
        """
        model = self._model
        if model is None:
            raise HybridEmbeddingRuntimeError("BGE-M3 尚未初始化")
        # query 走 encode_queries，document 走 encode_corpus
        method_name = "encode_queries" if mode == "query" else "encode_corpus"
        # 优先用细分方法，缺失时回退到通用 encode
        method = getattr(model, method_name, None) or model.encode
        return method(
            texts,
            batch_size=max(1, int(self.config.batch_size)),
            max_length=max(1, int(self.config.max_length)),
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

    @staticmethod
    def _rows(value: Any) -> list:
        """把 numpy 数组或类似序列统一转为 list，None/空安全处理。"""
        if hasattr(value, "tolist"):
            # numpy ndarray 转 list
            value = value.tolist()
        return list(value or [])

    def _parse(self, output: dict, expected: int) -> list[HybridEmbedding]:
        """把 BGE-M3 原始输出解析为 HybridEmbedding 值对象列表，并做契约校验。

        校验项：返回必须是 dict；dense/sparse 行数必须等于输入条数；
        lexical_weights 必须是 token->weight 映射；底层 ValueError 转为运行时错误。
        """
        if not isinstance(output, dict):
            raise HybridEmbeddingRuntimeError("BGE-M3 返回值不是 dict")
        dense_rows = self._rows(output.get("dense_vecs"))
        sparse_rows = self._rows(output.get("lexical_weights"))
        # 行数对齐校验：防止底层因批次/截断导致输出与输入不一致
        if len(dense_rows) != expected or len(sparse_rows) != expected:
            raise HybridEmbeddingRuntimeError(
                "BGE-M3 批量输出数量异常："
                f"输入 {expected}，dense {len(dense_rows)}，sparse {len(sparse_rows)}"
            )
        result = []
        # strict=True：dense 与 sparse 行数必须严格相等，否则视为契约破裂
        for dense, sparse in zip(dense_rows, sparse_rows, strict=True):
            if not hasattr(sparse, "items"):
                # lexical_weights 必须是 {token_id: weight} 形式的映射
                raise HybridEmbeddingRuntimeError("BGE-M3 lexical_weights 不是 token-weight 映射")
            try:
                # 交给 HybridEmbedding 做维度/有限性/归一化/稀疏合法性校验并构造值对象
                result.append(HybridEmbedding.from_raw(
                    dense,
                    sparse,
                    dense_size=int(self.config.dense_size),
                    normalize_dense=bool(self.config.normalize_dense),
                ))
            except ValueError as exc:
                # 把值对象层的校验失败统一包装为运行时错误
                raise HybridEmbeddingRuntimeError(str(exc)) from exc
        return result

    def _estimate_tokens(self, text: str) -> int:
        """Conservative tokenizer-free budget used only for scheduler admission.

        Chinese/non-ASCII characters are counted one-for-one; four ASCII
        characters approximate one token.  The model still performs the
        authoritative truncation at ``max_length``.
        """
        value = str(text)
        ascii_count = sum(1 for character in value if ord(character) < 128)
        non_ascii_count = len(value) - ascii_count
        estimate = non_ascii_count + math.ceil(ascii_count / 4)
        return min(max(1, int(self.config.max_length)), max(1, estimate))

    def _document_chunks(self, texts: list[str]) -> list[list[str]]:
        """Split a large offline request so online queries can run between chunks."""
        max_texts = max(
            1, int(getattr(self.scheduler_config, "max_batch_texts", 32))
        )
        max_tokens = max(
            1, int(getattr(self.scheduler_config, "max_batch_tokens", 8192))
        )
        chunks: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0
        for text in texts:
            tokens = self._estimate_tokens(text)
            if current and (
                len(current) >= max_texts
                or current_tokens + tokens > max_tokens
            ):
                chunks.append(current)
                current = []
                current_tokens = 0
            current.append(text)
            current_tokens += tokens
        if current:
            chunks.append(current)
        return chunks

    async def _dispatch_microbatch(
        self, texts: list[str], mode: str
    ) -> list[HybridEmbedding]:
        """One scheduler batch -> one serialized BGE invocation -> strict parse."""
        async with self._encode_lock:
            output = await self._run_serial(self._encode_sync, texts, mode)
        return self._parse(output, len(texts))

    async def _encode_serial(
        self, cleaned: list[str], mode: str
    ) -> list[HybridEmbedding]:
        submitted_at = time.perf_counter()
        self._serial_waiters += 1
        self._serial_metrics.submitted_request(self._serial_waiters)
        stage = "queued"
        try:
            lock_started = time.perf_counter()
            async with self._encode_lock:
                queue_wait_ms = (time.perf_counter() - lock_started) * 1000
                stage = "running"
                inference_started = time.perf_counter()
                output = await self._run_serial(self._encode_sync, cleaned, mode)
                inference_ms = (time.perf_counter() - inference_started) * 1000
            parsed = self._parse(output, len(cleaned))
            finished = time.perf_counter()
            self._serial_metrics.record_batch(
                request_count=1,
                text_count=len(cleaned),
                queue_wait_ms=[queue_wait_ms],
                inference_ms=inference_ms,
                total_ms=[(finished - submitted_at) * 1000],
            )
            return parsed
        except asyncio.CancelledError:
            self._serial_metrics.cancelled_request(stage)
            raise
        except BaseException:
            self._serial_metrics.failed_request()
            raise
        finally:
            self._serial_waiters = max(0, self._serial_waiters - 1)

    async def _encode(self, texts: list[str], mode: str) -> list[HybridEmbedding]:
        """清洗输入 -> 懒加载 -> serial 或有界 microbatch -> 严格解析。"""
        operation = self._begin_async_operation()
        try:
            # 去首尾空白并统一为字符串
            cleaned = [str(text).strip() for text in texts]
            # 拒绝空批次或包含空串的批次，避免底层模型行为未定义
            if not cleaned or any(not text for text in cleaned):
                raise ValueError("BGE-M3 不接受空文本或空批次")
            # 首次编码时加载模型（带并发去重）
            await self._ensure_model()
            if self.runtime_mode == "serial":
                return await self._encode_serial(cleaned, mode)
            if self._scheduler is None:
                raise HybridEmbeddingRuntimeError(
                    "BGE-M3 microbatch 调度器未初始化"
                )
            chunks = (
                self._document_chunks(cleaned)
                if mode == "document"
                else [cleaned]
            )
            result: list[HybridEmbedding] = []
            for chunk in chunks:
                result.extend(await self._scheduler.submit(chunk, mode))
            return result
        finally:
            self._finish_async_operation(operation)

    async def aembed_documents(self, texts: list[str]) -> list[HybridEmbedding]:
        """文档侧编码入口（encode_corpus），用于索引构建。"""
        return await self._encode(list(texts), "document")

    async def aembed_query(self, text: str) -> HybridEmbedding:
        """查询侧编码入口（encode_queries），用于在线检索。"""
        return (await self.aembed_queries([text]))[0]

    async def aembed_queries(self, texts: list[str]) -> list[HybridEmbedding]:
        """批量查询侧编码；条数和冷热缓存均不会切换到 encode_corpus。"""
        return await self._encode(list(texts), "query")

    async def warmup(self) -> dict:
        """预热：用一条固定文本跑一次 query 编码，验证整条链路可用。

        返回 status 叠加实际 dense 维度与 sparse 非零项数，供启动期健康检查展示。
        """
        embedding = await self.aembed_query("Data Agent BGE-M3 readiness")
        return {
            **self.status(),
            "dense_size": len(embedding.dense),
            "sparse_nonzero": len(embedding.sparse_indices),
        }

    def reset_runtime_metrics_epoch(self, *, reason: str) -> dict[str, Any]:
        """在完全空闲时切换观测 epoch，并保持模型、队列与缓存语义不变。"""
        with self._lifecycle_lock:
            active_inference = sum(
                not future.done() for future in self._active_inference_futures
            )
            active_async = sum(
                not future.done() for future in self._active_async_operations
            )
            if (
                self._closed
                or self._closing
                or self._close_failed
                or self._executor_shutdown
            ):
                raise HybridEmbeddingRuntimeError(
                    "BGE-M3 生命周期非 running，不能重置 metrics epoch"
                )
        before = self.status()
        queue_before = dict((before.get("scheduling") or {}).get("queue") or {})
        if active_inference or active_async:
            raise HybridEmbeddingRuntimeError(
                "BGE-M3 metrics epoch 重置前仍有活动任务："
                f"inference={active_inference}, async={active_async}"
            )
        if (
            int(queue_before.get("depth") or 0)
            or int(queue_before.get("running_requests") or 0)
            or self._encode_lock.locked()
            or self._serial_waiters
        ):
            raise HybridEmbeddingRuntimeError(
                "BGE-M3 metrics epoch 重置前 queue/running/encode 未清空"
            )
        if not before.get("loaded") or int(before.get("model_instances") or 0) != 1:
            raise HybridEmbeddingRuntimeError(
                "BGE-M3 metrics epoch 重置要求一个已驻留模型实例"
            )

        reset = (
            self._scheduler.reset_metrics_epoch(reason=reason)
            if self._scheduler is not None
            else self._serial_metrics.reset_epoch(reason=reason)
        )
        after = self.status()
        queue_after = dict((after.get("scheduling") or {}).get("queue") or {})
        operational_queue_fields = (
            "capacity",
            "depth",
            "query_depth",
            "document_depth",
            "running_requests",
        )
        queue_preserved = all(
            queue_before.get(field) == queue_after.get(field)
            for field in operational_queue_fields
        )
        model_preserved = bool(
            after.get("loaded")
            and int(after.get("model_instances") or 0) == 1
            and before.get("profile_fingerprint")
            == after.get("profile_fingerprint")
        )
        if not queue_preserved or not model_preserved:
            raise HybridEmbeddingRuntimeError(
                "BGE-M3 metrics epoch 重置意外改变 queue 或模型状态"
            )
        return {
            **reset,
            "queue_state_preserved": queue_preserved,
            "model_state_preserved": model_preserved,
            "active_inference_before": active_inference,
            "active_async_before": active_async,
            "only_observability_reset": True,
        }

    def status(self) -> dict:
        """返回当前客户端的只读状态快照，供健康检查与可观测性使用。"""
        with self._lifecycle_lock:
            active_inference = sum(
                not future.done()
                for future in self._active_inference_futures
            )
            active_async_operations = sum(
                not future.done()
                for future in self._active_async_operations
            )
            lifecycle = {
                "state": (
                    "closed"
                    if self._closed
                    else "draining"
                    if self._close_in_progress
                    else "close_failed"
                    if self._close_failed
                    else "running"
                ),
                "accepting_requests": not (
                    self._closed
                    or self._closing
                    or self._close_failed
                    or self._executor_shutdown
                ),
                "active_inference_futures": active_inference,
                "active_async_operations": active_async_operations,
                "close_failed": self._close_failed,
                "last_close_error": self._last_close_error,
                "drain_timeout_ms": int(
                    self._drain_timeout_seconds() * 1000
                ),
            }
        if self._scheduler is not None:
            scheduling = self._scheduler.status()
        else:
            scheduling = {
                "state": "closed" if self._closed else "running",
                "queue": {
                    "capacity": 1,
                    "depth": self._serial_waiters,
                    "query_depth": None,
                    "document_depth": None,
                    "running_requests": int(self._encode_lock.locked()),
                    "max_depth_seen": self._serial_metrics.max_queue_depth,
                },
                "policy": {
                    "batch_window_ms": 0,
                    "max_batch_texts": int(self.config.batch_size),
                    "max_batch_tokens": None,
                    "request_deadline_ms": None,
                    "online_query_priority": None,
                    "offline_corpus_priority": None,
                    "offline_corpus_quota": None,
                },
                "metrics": self._serial_metrics.snapshot(),
            }
        return {
            "provider": "FlagEmbedding.BGEM3FlagModel",
            "model": self.config.model,
            "revision": self.config.revision,
            "model_path": str(self.model_path),
            "model_path_exists": self.model_path.is_dir(),
            "loaded": self._model is not None,
            "device": self._device or str(self.config.device),
            "dense_size": int(self.config.dense_size),
            "embedding_mode": "dense_sparse",
            "profile_fingerprint": self.profile_fingerprint,
            "runtime_mode": self.runtime_mode,
            "runtime_profile": self.runtime_profile,
            "runtime_profile_fingerprint": self.runtime_profile_fingerprint,
            "model_instances": 1 if self._model is not None else 0,
            "lifecycle": lifecycle,
            "scheduling": scheduling,
        }

    async def close(self) -> None:
        """有界排空 scheduler 与底层推理，成功后才允许 manager 换代实例。

        若同步模型调用超过 drain timeout，无法安全终止 Python
        线程。此时显式 fail-close 并抛错；manager 必须保留该 client，
        直到旧 future 完成且再次 close() 成功，避免跨 repeat 并发。
        """
        with self._lifecycle_lock:
            if self._closed:
                return
            if self._close_in_progress:
                raise HybridEmbeddingRuntimeError(
                    "BGE-M3 客户端关闭已在进行"
                )
            self._close_in_progress = True
            self._closing = True

        try:
            if self._scheduler is not None:
                await self._scheduler.close()
            await self._drain_active_work()
            # All tracked calls have completed, so wait=True cannot strand the
            # event loop behind model inference and guarantees thread teardown.
            self._executor.shutdown(wait=True, cancel_futures=True)
        except BaseException as exc:
            # A running Python thread cannot be force-killed safely.  Prevent
            # queued executor work and poison this lifecycle until a later
            # close retry observes that the active future has completed.
            self._executor.shutdown(wait=False, cancel_futures=True)
            with self._lifecycle_lock:
                self._executor_shutdown = True
                self._close_failed = True
                self._last_close_error = f"{type(exc).__name__}: {exc}"
            raise
        else:
            with self._lifecycle_lock:
                self._executor_shutdown = True
                self._closed = True
                self._closing = False
                self._close_failed = False
                self._last_close_error = None
            self._model = None
        finally:
            with self._lifecycle_lock:
                self._close_in_progress = False
