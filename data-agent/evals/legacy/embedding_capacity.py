"""TODO-60 BGE-M3 runtime scheduling/capacity comparison.

The default run is a deterministic scheduler benchmark that is safe for CI.
``--live`` uses Data Agent's configured BGE-M3 weights on the current hardware.
Both profiles share the exact HybridEmbedding contract and differ only in
runtime scheduling.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import time
from datetime import datetime, timezone
from typing import Any

import psutil

from app.infrastructure.clients.hybrid_embedding_client import HybridEmbeddingClient
from app.config.app_config import app_config
from app.domain.retrieval.hybrid_embedding import HybridEmbedding
from evals.paths import REPORTS_ROOT as REPORT_ROOT


LATEST_PATH = REPORT_ROOT / "embedding_capacity_latest.json"
LATEST_MARKDOWN_PATH = REPORT_ROOT / "embedding_capacity_latest.md"
_MIB = 1024 * 1024
_RESOURCE_SAMPLE_INTERVAL_SECONDS = 0.01


def _mb(value: int | float) -> float:
    return round(float(value) / _MIB, 3)


def _process_snapshot(process: psutil.Process) -> dict[str, float]:
    memory = process.memory_info()
    cpu = process.cpu_times()
    return {
        "rss_mb": _mb(memory.rss),
        "cpu_seconds": round(float(cpu.user + cpu.system), 6),
    }


def _cuda_snapshot(*, reset_peak: bool = False) -> dict[str, Any]:
    """Read allocator evidence without requiring NVML or an external service."""

    if importlib.util.find_spec("torch") is None:
        return {
            "available": False,
            "reason": "torch_not_installed",
            "devices": [],
        }
    import torch

    if not torch.cuda.is_available():
        return {
            "available": False,
            "reason": "cuda_not_available",
            "devices": [],
        }

    devices = []
    for index in range(torch.cuda.device_count()):
        if reset_peak:
            torch.cuda.reset_peak_memory_stats(index)
        utilization = None
        if hasattr(torch.cuda, "utilization"):
            try:
                utilization = float(torch.cuda.utilization(index))
            except Exception:
                # NVML is optional. Allocator memory is still valid evidence.
                utilization = None
        devices.append({
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "allocated_mb": _mb(torch.cuda.memory_allocated(index)),
            "reserved_mb": _mb(torch.cuda.memory_reserved(index)),
            "peak_allocated_mb": _mb(torch.cuda.max_memory_allocated(index)),
            "peak_reserved_mb": _mb(torch.cuda.max_memory_reserved(index)),
            "utilization_percent": utilization,
        })
    return {
        "available": True,
        "reason": None,
        "devices": devices,
        "allocated_mb": round(
            sum(item["allocated_mb"] for item in devices), 3
        ),
        "reserved_mb": round(
            sum(item["reserved_mb"] for item in devices), 3
        ),
        "peak_allocated_mb": round(
            sum(item["peak_allocated_mb"] for item in devices), 3
        ),
        "peak_reserved_mb": round(
            sum(item["peak_reserved_mb"] for item in devices), 3
        ),
    }


async def _sample_process_rss(
    process: psutil.Process,
    stop: asyncio.Event,
    samples: list[float],
) -> None:
    while not stop.is_set():
        samples.append(_process_snapshot(process)["rss_mb"])
        try:
            await asyncio.wait_for(
                stop.wait(), timeout=_RESOURCE_SAMPLE_INTERVAL_SECONDS
            )
        except TimeoutError:
            pass
    samples.append(_process_snapshot(process)["rss_mb"])


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * quantile) - 1))
    return round(ordered[index], 3)


def _latencies(values: list[float]) -> dict[str, float]:
    return {
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": round(max(values), 3) if values else 0.0,
    }


class _SyntheticBGE:
    """Deterministic model-call simulator; no result is presented as GPU evidence."""

    def __init__(
        self,
        dense_size: int,
        *,
        call_overhead_ms: float = 12.0,
        per_text_ms: float = 1.2,
    ):
        self.dense_size = dense_size
        self.call_overhead_ms = call_overhead_ms
        self.per_text_ms = per_text_ms
        self.calls: list[tuple[str, list[str]]] = []
        self.active = 0
        self.max_active = 0

    def _output(self, texts: list[str], mode: str) -> dict:
        self.calls.append((mode, list(texts)))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        time.sleep((self.call_overhead_ms + self.per_text_ms * len(texts)) / 1000)
        dense_rows = []
        sparse_rows = []
        for text in texts:
            digest = hashlib.sha256(f"{mode}:{text}".encode("utf-8")).digest()
            first = int.from_bytes(digest[:2], "big") % self.dense_size
            second = int.from_bytes(digest[2:4], "big") % self.dense_size
            if second == first:
                second = (second + 1) % self.dense_size
            dense = [0.0] * self.dense_size
            dense[first] = 0.8
            dense[second] = 0.6
            dense_rows.append(dense)
            sparse_rows.append({
                int.from_bytes(digest[4:6], "big"): 0.8,
                int.from_bytes(digest[6:8], "big"): 0.3,
            })
        self.active -= 1
        return {"dense_vecs": dense_rows, "lexical_weights": sparse_rows}

    def encode_queries(self, texts, **kwargs):
        return self._output(list(texts), "query")

    def encode_corpus(self, texts, **kwargs):
        return self._output(list(texts), "document")


def _profile_config(runtime_mode: str, *, live: bool):
    config = copy.deepcopy(app_config.embedding)
    config.runtime_mode = runtime_mode
    if not live:
        config.dense_size = 32
        config.max_length = 128
        config.batch_size = 32
    return config


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / max(1e-12, left_norm * right_norm)


def _parity(
    serial: list[HybridEmbedding],
    candidate: list[HybridEmbedding],
) -> dict[str, Any]:
    if len(serial) != len(candidate):
        return {
            "passed": False,
            "count_match": False,
            "minimum_dense_cosine": 0.0,
            "sparse_index_match": False,
            "maximum_sparse_weight_delta": None,
        }
    cosines = []
    sparse_index_match = True
    max_sparse_delta = 0.0
    for left, right in zip(serial, candidate, strict=True):
        cosines.append(_cosine(left.dense, right.dense))
        if left.sparse_indices != right.sparse_indices:
            sparse_index_match = False
            continue
        max_sparse_delta = max(
            max_sparse_delta,
            max(
                (
                    abs(a - b)
                    for a, b in zip(
                        left.sparse_values, right.sparse_values, strict=True
                    )
                ),
                default=0.0,
            ),
        )
    minimum_cosine = min(cosines, default=0.0)
    return {
        "passed": (
            minimum_cosine >= 0.999
            and sparse_index_match
            and max_sparse_delta <= 1e-5
        ),
        "count_match": True,
        "minimum_dense_cosine": round(minimum_cosine, 8),
        "sparse_index_match": sparse_index_match,
        "maximum_sparse_weight_delta": round(max_sparse_delta, 8),
    }


async def _run_profile(
    runtime_mode: str,
    *,
    live: bool,
    concurrency: int,
    idle_repeats: int,
) -> dict[str, Any]:
    config = _profile_config(runtime_mode, live=live)
    synthetic = None if live else _SyntheticBGE(config.dense_size)
    process = psutil.Process(os.getpid())
    resource_before_model = _process_snapshot(process) if live else None
    client = HybridEmbeddingClient(
        config,
        model_factory=None if live else lambda: synthetic,
    )
    queries = [
        f"TODO-60 吞吐验证问题 {index}：各地区销售额"
        for index in range(concurrency)
    ]
    documents = [
        f"字段 {index}：订单事实表销售金额与地区维度"
        for index in range(min(16, max(4, concurrency)))
    ]
    resource_stop: asyncio.Event | None = None
    resource_task: asyncio.Task | None = None
    rss_samples: list[float] = []
    try:
        # Warm model loading, executor thread and scheduler task; warmup is
        # outside all measured intervals so first-thread startup is not
        # mistaken for steady-state idle latency.
        for index in range(3):
            await client.aembed_query(f"TODO-60 runtime warmup {index}")

        resource_after_warmup = _process_snapshot(process) if live else None
        cuda_after_warmup = _cuda_snapshot(reset_peak=live) if live else None
        measured_wall_started = time.perf_counter()
        measured_cpu_started = (
            resource_after_warmup["cpu_seconds"] if resource_after_warmup else 0.0
        )
        if live:
            resource_stop = asyncio.Event()
            resource_task = asyncio.create_task(
                _sample_process_rss(process, resource_stop, rss_samples),
                name=f"embedding-capacity-resource-{runtime_mode}",
            )

        idle_ms: list[float] = []
        for index in range(idle_repeats):
            started = time.perf_counter()
            await client.aembed_query(f"空闲单请求 {index}")
            idle_ms.append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        query_outputs = await asyncio.gather(
            *(client.aembed_query(query) for query in queries)
        )
        burst_ms = (time.perf_counter() - started) * 1000

        document_started = time.perf_counter()
        document_outputs = await client.aembed_documents(documents)
        document_ms = (time.perf_counter() - document_started) * 1000

        measured_wall_seconds = time.perf_counter() - measured_wall_started
        if resource_stop is not None:
            resource_stop.set()
        if resource_task is not None:
            await resource_task
            resource_task = None
        resource_end = _process_snapshot(process) if live else None
        cuda_end = _cuda_snapshot() if live else None
        if live:
            rss_samples.extend([
                resource_after_warmup["rss_mb"],
                resource_end["rss_mb"],
            ])
            process_peak = max(rss_samples)
            process_growth = process_peak - resource_after_warmup["rss_mb"]
            resource_usage = {
                "available": True,
                "sampling_interval_ms": round(
                    _RESOURCE_SAMPLE_INTERVAL_SECONDS * 1000, 3
                ),
                "sample_count": len(rss_samples),
                "process": {
                    "rss_before_model_mb": resource_before_model["rss_mb"],
                    "rss_after_warmup_mb": resource_after_warmup["rss_mb"],
                    "rss_peak_measured_mb": round(process_peak, 3),
                    "rss_end_mb": resource_end["rss_mb"],
                    "model_resident_delta_mb": round(
                        resource_after_warmup["rss_mb"]
                        - resource_before_model["rss_mb"],
                        3,
                    ),
                    "measured_peak_growth_mb": round(process_growth, 3),
                    "cpu_seconds_measured": round(
                        resource_end["cpu_seconds"] - measured_cpu_started, 6
                    ),
                    "cpu_percent_of_one_core": round(
                        (
                            resource_end["cpu_seconds"] - measured_cpu_started
                        )
                        / max(1e-6, measured_wall_seconds)
                        * 100,
                        3,
                    ),
                },
                "cuda_after_warmup": cuda_after_warmup,
                "cuda_measured": cuda_end,
            }
        else:
            resource_usage = {
                "available": False,
                "reason": "deterministic_scheduler_capacity_has_no_model_resource_evidence",
            }

        status = client.status()
        model_calls = (
            len(synthetic.calls)
            if synthetic is not None
            else status["scheduling"]["metrics"]["throughput"][
                "model_invocations"
            ]
        )
        max_model_concurrency = synthetic.max_active if synthetic is not None else 1
        return {
            "runtime_profile": client.runtime_profile,
            "semantic_profile_fingerprint": client.profile_fingerprint,
            "runtime_profile_fingerprint": client.runtime_profile_fingerprint,
            "query_outputs": query_outputs,
            "document_outputs": document_outputs,
            "measurements": {
                "concurrency": concurrency,
                "query_texts": len(queries),
                "burst_wall_ms": round(burst_ms, 3),
                "burst_texts_per_second": round(
                    len(queries) / max(0.001, burst_ms / 1000), 3
                ),
                "idle_query_ms": _latencies(idle_ms),
                "document_texts": len(documents),
                "document_wall_ms": round(document_ms, 3),
                "model_invocations_total": model_calls,
                "max_model_concurrency": max_model_concurrency,
            },
            "resource_usage": resource_usage,
            "runtime_status": status,
        }
    finally:
        if resource_stop is not None:
            resource_stop.set()
        if resource_task is not None:
            await resource_task
        await client.close()


def _hardware(*, inspect_accelerator: bool) -> dict[str, Any]:
    torch_info: dict[str, Any] = {
        "available": importlib.util.find_spec("torch") is not None,
        "version": None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "accelerator_inspected": inspect_accelerator,
    }
    try:
        torch_info["version"] = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        pass
    if inspect_accelerator and torch_info["available"]:
        import torch

        torch_info.update({
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        })
        if torch.cuda.is_available():
            torch_info["cuda_devices"] = [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "torch": torch_info,
    }


async def build_report(
    *,
    live: bool = False,
    concurrency: int = 20,
    idle_repeats: int = 5,
    write: bool = True,
) -> dict[str, Any]:
    concurrency = max(2, int(concurrency))
    # With only five samples P95 is the single maximum, so one Windows
    # scheduling hiccup can fail the deterministic CI profile.  Twenty
    # samples keep the same percentile gate while making it an actual tail
    # estimate.  Live runs retain the caller-selected hardware protocol.
    idle_repeats = max(20 if not live else 2, int(idle_repeats))
    # IsolatedAsyncioTestCase enables asyncio debug mode, whose task bookkeeping
    # can be larger than the scheduler itself on this millisecond-scale
    # synthetic workload. Capacity evidence must reflect production loop mode.
    loop = asyncio.get_running_loop()
    previous_debug = loop.get_debug()
    loop.set_debug(False)
    try:
        serial = await _run_profile(
            "serial",
            live=live,
            concurrency=concurrency,
            idle_repeats=idle_repeats,
        )
        gc.collect()
        microbatch = await _run_profile(
            "microbatch",
            live=live,
            concurrency=concurrency,
            idle_repeats=idle_repeats,
        )
    finally:
        loop.set_debug(previous_debug)
    query_parity = _parity(
        serial.pop("query_outputs"), microbatch.pop("query_outputs")
    )
    document_parity = _parity(
        serial.pop("document_outputs"), microbatch.pop("document_outputs")
    )

    serial_measurements = serial["measurements"]
    candidate_measurements = microbatch["measurements"]
    ratio = (
        candidate_measurements["burst_texts_per_second"]
        / max(0.001, serial_measurements["burst_texts_per_second"])
    )
    idle_budget = (
        serial_measurements["idle_query_ms"]["p95"]
        + float(app_config.embedding.scheduler.batch_window_ms)
        + max(1.0, serial_measurements["idle_query_ms"]["p95"] * 0.10)
    )
    gates = {
        "semantic_profile_unchanged": (
            serial["semantic_profile_fingerprint"]
            == microbatch["semantic_profile_fingerprint"]
        ),
        "query_vector_parity": query_parity["passed"],
        "document_vector_parity": document_parity["passed"],
        "single_model_concurrency": (
            serial_measurements["max_model_concurrency"] == 1
            and candidate_measurements["max_model_concurrency"] == 1
        ),
        "throughput_gain_50_percent": ratio >= 1.50,
        "idle_query_budget": (
            candidate_measurements["idle_query_ms"]["p95"] <= idle_budget
        ),
        "candidate_burst_p95_proxy": (
            candidate_measurements["burst_wall_ms"]
            <= serial_measurements["burst_wall_ms"]
        ),
        "bounded_queue": (
            microbatch["runtime_status"]["scheduling"]["queue"]["capacity"]
            == int(app_config.embedding.scheduler.queue_capacity)
            and microbatch["runtime_status"]["scheduling"]["queue"]["depth"] == 0
        ),
        "merged_batch_observed": (
            microbatch["runtime_status"]["scheduling"]["metrics"]["batching"][
                "merged_batch_count"
            ]
            > 0
        ),
        "zero_runtime_errors": all(
            profile["runtime_status"]["scheduling"]["metrics"]["requests"][
                name
            ]
            == 0
            for profile in (serial, microbatch)
            for name in ("rejected", "timed_out", "failed")
        ),
        "resource_evidence_complete": (
            not live
            or all(
                profile["resource_usage"]["available"]
                and profile["resource_usage"]["sample_count"] > 0
                for profile in (serial, microbatch)
            )
        ),
    }
    gate_passed = all(gates.values())
    comparison = {
        "burst_texts_per_second_ratio": round(ratio, 3),
        "burst_texts_per_second_delta": round(
            candidate_measurements["burst_texts_per_second"]
            - serial_measurements["burst_texts_per_second"],
            3,
        ),
        "burst_wall_ms_delta": round(
            candidate_measurements["burst_wall_ms"]
            - serial_measurements["burst_wall_ms"],
            3,
        ),
        "idle_query_p95_ms_delta": round(
            candidate_measurements["idle_query_ms"]["p95"]
            - serial_measurements["idle_query_ms"]["p95"],
            3,
        ),
        "idle_query_p95_budget_ms": round(idle_budget, 3),
        "model_invocation_delta": (
            candidate_measurements["model_invocations_total"]
            - serial_measurements["model_invocations_total"]
        ),
    }
    if live:
        serial_process = serial["resource_usage"]["process"]
        candidate_process = microbatch["resource_usage"]["process"]
        comparison["process_rss_peak_mb_delta"] = round(
            candidate_process["rss_peak_measured_mb"]
            - serial_process["rss_peak_measured_mb"],
            3,
        )
        comparison["process_rss_growth_mb_delta"] = round(
            candidate_process["measured_peak_growth_mb"]
            - serial_process["measured_peak_growth_mb"],
            3,
        )
        serial_cuda = serial["resource_usage"]["cuda_measured"]
        candidate_cuda = microbatch["resource_usage"]["cuda_measured"]
        comparison["cuda_peak_reserved_mb_delta"] = (
            round(
                candidate_cuda["peak_reserved_mb"]
                - serial_cuda["peak_reserved_mb"],
                3,
            )
            if serial_cuda["available"] and candidate_cuda["available"]
            else None
        )
    report = {
        "available": True,
        "todo": "TODO-60",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": (
            "live_bge_m3_current_hardware"
            if live
            else "deterministic_scheduler_capacity"
        ),
        "live_model_evaluated": live,
        "hardware": _hardware(inspect_accelerator=live),
        "frozen_variables": {
            "model": app_config.embedding.model,
            "revision": app_config.embedding.revision,
            "device": app_config.embedding.device,
            "precision": "fp16_if_cuda" if app_config.embedding.use_fp16 else "fp32",
            "dense_size": app_config.embedding.dense_size if live else 32,
            "normalize_dense": app_config.embedding.normalize_dense,
            "concurrency": concurrency,
            "idle_repeats": idle_repeats,
        },
        "profiles": {
            "serial_inprocess_v1": serial,
            "microbatch_inprocess_v1": microbatch,
        },
        "correctness": {
            "query": query_parity,
            "document": document_parity,
        },
        "comparison": comparison,
        "gates": gates,
        "summary": {
            "gate_passed": gate_passed,
            "selected_runtime_profile": (
                "microbatch_inprocess_v1"
                if gate_passed
                else "serial_inprocess_v1"
            ),
            "remote_service_triggered": False,
            "remote_service_decision": (
                "当前单进程/单设备的有界微批已形成受控容量收益；"
                "尚无多 GPU、跨 API worker 共享或独立扩缩容证据，不建设远端服务。"
            ),
        },
    }
    if write:
        REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        LATEST_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LATEST_MARKDOWN_PATH.write_text(
            render_markdown(report), encoding="utf-8"
        )
    return report


def render_markdown(report: dict[str, Any]) -> str:
    serial = report["profiles"]["serial_inprocess_v1"]["measurements"]
    candidate = report["profiles"]["microbatch_inprocess_v1"]["measurements"]
    status = "通过" if report["summary"]["gate_passed"] else "未通过"
    lines = [
        "# TODO-60 BGE-M3 Embedding 推理调度与吞吐报告",
        "",
        f"> 状态：**{status}** ｜ 来源：`{report['source']}` ｜ "
        f"生成时间：{report['generated_at']}",
        "",
        "## 同协议对比",
        "",
        "| Profile | 突发 texts/s | 突发 wall | 空闲 query P95 | 模型调用 | 最大模型并发 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| serial_inprocess_v1 | {serial['burst_texts_per_second']:.3f} | "
            f"{serial['burst_wall_ms']:.3f} ms | {serial['idle_query_ms']['p95']:.3f} ms | "
            f"{serial['model_invocations_total']} | {serial['max_model_concurrency']} |"
        ),
        (
            f"| microbatch_inprocess_v1 | {candidate['burst_texts_per_second']:.3f} | "
            f"{candidate['burst_wall_ms']:.3f} ms | {candidate['idle_query_ms']['p95']:.3f} ms | "
            f"{candidate['model_invocations_total']} | {candidate['max_model_concurrency']} |"
        ),
        "",
        f"- 突发吞吐倍率：**{report['comparison']['burst_texts_per_second_ratio']:.3f}×**",
        f"- query dense 最低 cosine：**{report['correctness']['query']['minimum_dense_cosine']:.8f}**",
        f"- document dense 最低 cosine：**{report['correctness']['document']['minimum_dense_cosine']:.8f}**",
        f"- 选择：`{report['summary']['selected_runtime_profile']}`",
        "",
        "## 门禁",
        "",
    ]
    lines.extend(
        f"- {'✅' if passed else '❌'} `{name}`"
        for name, passed in report["gates"].items()
    )
    if report.get("live_model_evaluated"):
        serial_resources = report["profiles"]["serial_inprocess_v1"][
            "resource_usage"
        ]
        candidate_resources = report["profiles"]["microbatch_inprocess_v1"][
            "resource_usage"
        ]
        lines.extend([
            "",
            "## 资源证据",
            "",
            "| Profile | 模型后 RSS | 测量峰值 RSS | 峰值增长 | CPU / 单核口径 | CUDA 峰值 reserved |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for name, resources in (
            ("serial_inprocess_v1", serial_resources),
            ("microbatch_inprocess_v1", candidate_resources),
        ):
            process = resources["process"]
            cuda = resources["cuda_measured"]
            cuda_peak = (
                f"{cuda['peak_reserved_mb']:.3f} MB"
                if cuda["available"]
                else "N/A"
            )
            lines.append(
                f"| {name} | {process['rss_after_warmup_mb']:.3f} MB | "
                f"{process['rss_peak_measured_mb']:.3f} MB | "
                f"{process['measured_peak_growth_mb']:.3f} MB | "
                f"{process['cpu_percent_of_one_core']:.1f}% | {cuda_peak} |"
            )
        lines.extend([
            "",
            f"- 微批相对串行的推理期 RSS 增量差："
            f"**{report['comparison']['process_rss_growth_mb_delta']:+.3f} MB**",
            f"- 同进程顺序运行下的绝对 RSS 峰值差："
            f"**{report['comparison']['process_rss_peak_mb_delta']:+.3f} MB**",
            "  （受前一 Profile 的运行库与 allocator 缓存残留影响，只作观测，"
            "不把它归因为微批开销）。",
            "- CUDA 利用率依赖 NVML；不可用时明确显示为空，不用 CPU 或吞吐反推 GPU 利用率。",
        ])
    evidence_note = (
        "> 本报告加载了配置的真实 BGE-M3，只代表当前硬件与冻结负载；"
        "发布前仍须在目标硬件按同协议重跑，不能外推为 GPU 或多 worker 容量。"
        if report.get("live_model_evaluated")
        else
        "> 本报告使用确定性调度模型，只证明调度正确性与可重复容量收益；"
        "发布前使用 `--live` 在目标硬件运行，不能把确定性结果冒充真实模型/GPU 实测。"
    )
    lines.extend([
        "",
        "## 服务化决策",
        "",
        report["summary"]["remote_service_decision"],
        "",
        evidence_note,
        "",
    ])
    return "\n".join(lines)


async def _main(args) -> int:
    report = await build_report(
        live=args.live,
        concurrency=args.concurrency,
        idle_repeats=args.idle_repeats,
        write=True,
    )
    print(render_markdown(report))
    print(f"\nJSON：{LATEST_PATH}")
    print(f"Markdown：{LATEST_MARKDOWN_PATH}")
    return 0 if report["summary"]["gate_passed"] or not args.enforce else 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Data Agent TODO-60 Embedding runtime capacity gate"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="加载配置的真实 BGE-M3；默认运行可重复的确定性调度门禁",
    )
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--idle-repeats", type=int, default=5)
    parser.add_argument("--enforce", action="store_true")
    return asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
