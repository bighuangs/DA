from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from langchain_core.callbacks import BaseCallbackHandler

from app.application.ports.llm_routing import (
    LLMRouteCatalog,
    MissingLLMRouteCatalog,
)
from app.application.ports.query_tracing import QueryTraceLLMSettings
from app.foundation.audit_compaction import (
    compact_candidate_decision,
    compact_planning_route,
    compact_query_contract,
    compact_relation_closure,
    compact_semantic_evidence,
    compact_semantic_repair,
)
from app.foundation.llm_evidence import (
    attempt_accounting,
    classify_attempt_status,
    make_llm_attempt,
    normalize_llm_attempt as normalize_llm_attempt_evidence,
    reconcile_attempts,
)
from app.foundation.observability import normalize_degradation_evidence
from app.foundation.runtime_context import current_node_id


_MISSING_ROUTE_CATALOG = MissingLLMRouteCatalog()
_DEFAULT_LLM_SETTINGS = QueryTraceLLMSettings()


def route_for_node(node_id: str | None) -> str:
    """保留原模块级 patch 点，并在未注入组合根时提供稳定节点分类。"""

    return _MISSING_ROUTE_CATALOG.route_for_node(node_id)


def task_profile(route: str | None) -> dict:
    """保留原模块级 patch 点，缺少组合根时显式返回未注册模型。"""

    return dict(_MISSING_ROUTE_CATALOG.profile_for_route(route))


def query_trace_llm_settings() -> QueryTraceLLMSettings:
    """返回未注入组合根时使用的安全默认值，并保留可替换 patch 点。"""

    return _DEFAULT_LLM_SETTINGS


def _route_for_catalog(
    catalog: LLMRouteCatalog | None,
    node_id: str | None,
) -> str:
    """优先使用显式目录；默认路径仍经过模块级 patch 点。"""

    if catalog is None:
        return route_for_node(node_id)
    return str(catalog.route_for_node(node_id))


def _profile_for_catalog(
    catalog: LLMRouteCatalog | None,
    route: str | None,
) -> dict:
    """读取路由 profile，并复制为普通字典以隔离外部可变对象。"""

    if catalog is None:
        return task_profile(route)
    return dict(catalog.profile_for_route(route))


def _normalize_degradation(
    value: dict,
    catalog: LLMRouteCatalog | None,
) -> dict:
    """补齐路由与核心路径分类后调用 Foundation 的纯规范化函数。"""

    item = dict(value or {})
    node = str(item.get("node_id") or item.get("node") or "unknown")
    return normalize_degradation_evidence(
        item,
        default_route=_route_for_catalog(catalog, node),
        default_affects_core_path=(
            node not in {"generate_chart", "generate_analysis", "generate_recommend"}
        ),
    )


def normalize_degradation(value: dict) -> dict:
    """兼容旧 patch 点：使用默认路由目录规范化降级证据。"""

    return _normalize_degradation(value, None)


def _normalize_llm_attempt(
    value: dict,
    catalog: LLMRouteCatalog | None,
) -> dict:
    """在进入纯 attempt 合同前补齐由组合根提供的路由与模型身份。"""

    item = dict(value or {})
    node = str(item.get("node_id") or item.get("step_name") or "unknown")
    route = str(item.get("route") or _route_for_catalog(catalog, node))
    model = item.get("model") or _profile_for_catalog(catalog, route).get(
        "model_name"
    )
    return normalize_llm_attempt_evidence({
        **item,
        "node_id": node,
        "route": route,
        "model": model,
    })


def normalize_llm_attempt(value: dict) -> dict:
    """兼容旧 patch 点：使用默认路由目录规范化一次调用证据。"""

    return _normalize_llm_attempt(value, None)


def _now() -> datetime:
    """返回当前 UTC 时间（去掉 tzinfo），作为链路内统一的时间基准。"""
    # 去掉 tzinfo 以保证与 DB 中 naive datetime 列兼容，避免时区比较报错
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _int(value) -> int:
    """将任意值安全转为非负整数，转换失败或为空时返回 0。"""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _short(value, limit: int = 500) -> str | None:
    """将任意值转为字符串并截断到指定长度，None 原样返回；同时剔除 NUL 字符。"""
    if value is None:
        return None
    # NUL 字符会破坏部分 DB/日志后端，替换为空格后再 strip
    text = str(value).replace("\x00", " ").strip()
    return text[:limit]


@dataclass
class StepSpan:
    """链路中单个步骤（节点）的时间跨度记录，含起止时间、状态与错误信息。"""

    name: str                          # 步骤/节点名称
    started_at: datetime               # 开始时间
    status: str = "running"            # 状态：running / success / failed
    finished_at: datetime | None = None  # 结束时间，未结束时为 None
    error: str | None = None           # 失败时的错误信息

    @property
    def duration_ms(self) -> int:
        """计算该步骤耗时（毫秒）；未结束时以当前时间作为终点，保证非负。"""
        end = self.finished_at or _now()
        return max(0, int((end - self.started_at).total_seconds() * 1000))


@dataclass
class TokenUsageRecord:
    """单次 LLM 调用的 token 用量记录，承载 token 数、耗时、估算成本与状态。"""

    step_name: str                          # 所属节点名
    model: str | None                       # 实际使用的模型名
    prompt_tokens: int                      # 输入 token 数
    completion_tokens: int                  # 输出 token 数
    total_tokens: int                       # 总 token 数
    latency_ms: int = 0                     # 本次调用耗时（毫秒）
    estimated_cost: float = 0.0             # 估算成本（按配置单价计算）
    status: str = "success"                 # 调用状态：success / timeout / cancelled / error / empty_output
    attempt_id: str | None = None           # LangChain run_id，用于与显式 attempt 去重
    token_known: bool = False                # 是否从 provider 拿到真实 token 数
    timeout_budget_ms: int | None = None    # 超时预算（毫秒），用于审计
    error_type: str | None = None           # 异常类型名，失败时记录
    retry_index: int = 0                    # 统一调用边界内的物理重试序号
    provider_request_id: str | None = None  # provider 原生请求 ID（若响应暴露）
    provider_request_id_available: bool = False
    provider_request_id_unavailable_reason: str | None = None
    token_timing_available: bool = False
    token_timing_unavailable_reason: str | None = None


class TokenUsageCallback(BaseCallbackHandler):
    """
    LangChain 回调处理器：从 provider 返回的 usage_metadata / token_usage 读取真实 token。

    核心功能：作为 BaseCallbackHandler 注入 LangGraph，在每次 LLM 调用开始时记录
             节点与起始时间戳，在调用结束时从 response 多级结构中提取真实 token 用量、
             模型名与耗时，估算成本并写入 collector.token_usages；失败时写入零 token
             的失败记录，便于后续与显式 attempt 对账。

    业务逻辑：
        1. on_llm_start / on_chat_model_start：记录 run_id → (节点, 起始 perf_counter) 映射
        2. on_llm_end：从 llm_output 与 generations.message 多级结构提取 model 与 usage，
           计算 prompt/completion/total token、估算成本，写入 TokenUsageRecord
        3. on_llm_error：弹出起始记录，写入零 token 的失败记录，含 error_type 与状态分类
    """

    def __init__(self, trace: "QueryTraceCollector"):
        self.trace = trace
        # run_id → (节点名, 起始 perf_counter) 映射，用于在 end/error 时计算耗时与归属
        self._calls: dict[str, tuple[str, float, str | None, int]] = {}

    def _start(self, run_id, **kwargs) -> None:
        # 节点优先取当前上下文的 current_node_id，回退到 collector.current_step，再回退 "llm"
        metadata = dict(kwargs.get("metadata") or {})
        self._calls[str(run_id)] = (
            current_node_id.get() or self.trace.current_step or "llm",
            time.perf_counter(),
            str(metadata.get("data_agent_attempt_id") or "") or None,
            _int(metadata.get("data_agent_retry_index")),
        )

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        # 传统 LLM 调用入口
        del prompts
        self._start(run_id, **kwargs)

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        # Chat 模型调用入口，与 on_llm_start 共用同一登记逻辑
        self._start(run_id, **kwargs)

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        """
        LLM 调用结束回调：从 response 多级结构提取 token 用量并写入 collector。

        业务逻辑：
            1. 从 llm_output 提取 model_name / model 与 token_usage / usage
            2. 遍历 generations.message，从 usage_metadata / response_metadata 补全 model 与 usage
            3. 计算 prompt / completion / total token（total 缺失时由 prompt+completion 兜底）
            4. 弹出起始记录计算耗时，按配置单价估算成本
            5. 追加 TokenUsageRecord 到 collector，token_known 标记是否拿到真实 usage
        """
        usage = {}
        model = None
        provider_request_id = None
        # 第一级：response.llm_output，部分 provider 在此直接给出 model 与 usage
        llm_output = getattr(response, "llm_output", None) or {}
        model = llm_output.get("model_name") or llm_output.get("model")
        usage.update(llm_output.get("token_usage") or llm_output.get("usage") or {})
        provider_request_id = (
            llm_output.get("request_id")
            or llm_output.get("request-id")
            or llm_output.get("x-request-id")
        )
        # 第二级：遍历 generations，从 message 的 usage_metadata / response_metadata 补全
        for generations in getattr(response, "generations", []) or []:
            for generation in generations or []:
                message = getattr(generation, "message", None)
                metadata = getattr(message, "usage_metadata", None) or {}
                response_metadata = getattr(message, "response_metadata", None) or {}
                model = model or response_metadata.get("model_name") or response_metadata.get("model")
                provider_request_id = provider_request_id or (
                    response_metadata.get("request_id")
                    or response_metadata.get("request-id")
                    or response_metadata.get("x-request-id")
                )
                # 合并优先级：response_metadata.token_usage < 已有 usage < usage_metadata
                usage = {**response_metadata.get("token_usage", {}), **usage, **metadata}
        # 兼容 input_tokens / output_tokens 命名差异
        prompt = _int(usage.get("prompt_tokens", usage.get("input_tokens")))
        completion = _int(usage.get("completion_tokens", usage.get("output_tokens")))
        total = _int(usage.get("total_tokens")) or prompt + completion
        # 弹出起始记录；若已丢失则用当前上下文兜底
        node, started, boundary_attempt_id, retry_index = self._calls.pop(
            str(run_id),
            (
                current_node_id.get() or self.trace.current_step or "llm",
                time.perf_counter(),
                None,
                0,
            ),
        )
        # 按组合根注入的每百万 token 单价估算成本
        settings = self.trace.resolved_llm_settings
        estimated_cost = (
            prompt * float(settings.input_cost_per_million or 0)
            + completion * float(settings.output_cost_per_million or 0)
        ) / 1_000_000
        self.trace.token_usages.append(TokenUsageRecord(
            step_name=node,
            model=model or settings.default_model,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            latency_ms=max(0, int((time.perf_counter() - started) * 1000)),
            estimated_cost=round(estimated_cost, 8),
            attempt_id=boundary_attempt_id or str(run_id),
            token_known=bool(usage),
            retry_index=retry_index,
            provider_request_id=(
                str(provider_request_id)[:160]
                if provider_request_id
                else "unavailable"
            ),
            provider_request_id_available=bool(provider_request_id),
            provider_request_id_unavailable_reason=(
                None
                if provider_request_id
                else "provider_callback_metadata_did_not_expose_request_id"
            ),
            token_timing_available=False,
            token_timing_unavailable_reason=(
                "non_stream_callback_has_completion_latency_only"
            ),
        ))

    def on_llm_error(self, error, *, run_id=None, **kwargs):
        """
        LLM 调用失败回调：弹出起始记录，写入零 token 的失败记录。

        业务逻辑：
            1. 弹出 run_id 对应的起始记录，计算耗时
            2. 通过 route_for_node 取该节点的路由配置，推断默认模型名
            3. 用 classify_attempt_status 将异常归类为 timeout/cancelled/error 等状态
            4. 追加零 token 的 TokenUsageRecord，记录 error_type 便于审计
        """
        node, started, boundary_attempt_id, retry_index = self._calls.pop(
            str(run_id),
            (
                current_node_id.get() or self.trace.current_step or "llm",
                time.perf_counter(),
                None,
                0,
            ),
        )
        route = self.trace.route_for_node(node)
        self.trace.token_usages.append(TokenUsageRecord(
            step_name=node,
            model=self.trace.profile_for_route(route)["model_name"],
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            latency_ms=max(0, int((time.perf_counter() - started) * 1000)),
            status=classify_attempt_status(error),
            attempt_id=boundary_attempt_id or str(run_id),
            token_known=False,
            error_type=type(error).__name__,
            retry_index=retry_index,
            provider_request_id="unavailable",
            provider_request_id_available=False,
            provider_request_id_unavailable_reason=(
                "provider_callback_failed_without_response_metadata"
            ),
            token_timing_available=False,
            token_timing_unavailable_reason=(
                "provider_callback_failed_before_token_timing"
            ),
        ))


@dataclass
class QueryTraceCollector:
    """
    单次查询链路采集器：聚合节点步骤、token 用量、降级证据、follow-up 与候选决策等运行期事实。

    核心功能：作为一次查询的全局采集容器，在 LangGraph 流式执行过程中由各节点通过
             start_step / record_* / mark_* 等方法写入运行期事实，最终由 finish 汇总为
             一份结构化 trace summary，供持久化到 query_trace 表并推送给前端。

    业务逻辑：
        1. 构造时生成 trace_id 与 started_at，并创建 TokenUsageCallback 绑定自身
        2. 执行期：start_step/end_node 维护步骤时间跨度，mark_* 记录关键里程碑耗时
        3. 执行期：record_* 收集 LLM attempt、降级、follow-up、候选决策、检索指标等证据
        4. 结束期：finish 将所有事实汇总为 trace summary，含 token 对账、成本估算、慢节点排行
    """
    chat_id: str                                       # 会话ID
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)  # 链路ID
    started_at: datetime = field(default_factory=_now)  # 链路起始时间
    steps: list[StepSpan] = field(default_factory=list)  # 节点步骤列表
    token_usages: list[TokenUsageRecord] = field(default_factory=list)  # token 用量列表
    current_step: str | None = None                     # 当前正在执行的步骤名
    error: str | None = None                            # 链路级错误信息
    finished_at: datetime | None = None                # 链路结束时间
    ttfse_ms: int | None = None                         # Time To First SSE Event（首事件耗时）
    tt_sql_ms: int | None = None                        # Time To SQL（生成 SQL 耗时）
    tt_result_ms: int | None = None                     # Time To Result（拿到结果耗时）
    degradations: list[dict] = field(default_factory=list)  # 降级证据列表
    follow_up_action: str | None = None                 # follow-up 动作（new_topic/clarify/...）
    follow_up_summary: str | None = None               # follow-up 摘要
    candidate_decision: dict = field(default_factory=dict)  # 候选决策（候选 SQL/置信度等）
    explicit_llm_attempts: list[dict] = field(default_factory=list)  # 节点显式上报的 LLM attempt
    retrieval_metrics: list[dict] = field(default_factory=list)  # 检索指标证据
    planning_route: dict = field(default_factory=dict)  # 规划实际动作、理由和 shadow
    planning_timing: dict = field(default_factory=dict)  # 非流式 planning 分段耗时
    semantic_repair: dict = field(default_factory=dict)  # TODO-64 规则、hash、violation 与耗时
    semantic_coverage: dict = field(default_factory=dict)  # TODO-92 结构槽位覆盖（无字面量）
    semantic_query_contract: dict = field(default_factory=dict)  # TODO-94 粒度、口径、阶段合同
    relation_closure: dict = field(default_factory=dict)  # TODO-94 元数据关系闭包
    request_deadline: dict = field(default_factory=dict)  # TODO-93 剩余预算（无绝对时间）
    route_catalog: LLMRouteCatalog | None = field(default=None, repr=False)
    llm_settings: QueryTraceLLMSettings | None = field(default=None, repr=False)

    def __post_init__(self):
        # dataclass 构造后创建回调处理器，绑定自身以便回调直接写入 token_usages
        self.callback = TokenUsageCallback(self)

    @property
    def resolved_llm_settings(self) -> QueryTraceLLMSettings:
        """返回本次 Trace 的不可变成本配置。"""

        return self.llm_settings or query_trace_llm_settings()

    def route_for_node(self, node_id: str | None) -> str:
        """通过显式注入目录解析节点路由。"""

        return _route_for_catalog(self.route_catalog, node_id)

    def profile_for_route(self, route: str | None) -> dict:
        """通过显式注入目录读取非敏感模型执行参数。"""

        return _profile_for_catalog(self.route_catalog, route)

    def start_step(self, name: str) -> None:
        """
        开始一个新步骤：自动结束上一个仍在 running 的步骤，并追加新 StepSpan。

        业务逻辑：
            1. name 为空则忽略，避免空步骤污染链路
            2. 若上一步仍为 running，则自动标记为 success 并补全结束时间
            3. 截断 name 到 128 字符作为 current_step，并追加新 StepSpan
        """
        if not name:
            return
        now = _now()
        # 自动收尾上一个未结束的步骤（默认视为成功）
        if self.steps and self.steps[-1].status == "running":
            self.steps[-1].status = "success"
            self.steps[-1].finished_at = now
        self.current_step = _short(name, 128)
        self.steps.append(StepSpan(name=self.current_step, started_at=now))

    def start_node(self, node_id: str, start_ms: int | None = None) -> None:
        """
        按 LangGraph 内部事件开始一个节点步骤，起始时间由相对偏移 start_ms 推算。

        业务逻辑：
            1. 截断 node_id 到 128 字符作为 current_step
            2. 起始时间 = 链路 started_at + start_ms 毫秒（start_ms 缺失时为 0，即链路起点）
            3. 追加新 StepSpan

        Args:
            node_id: 节点标识
            start_ms: 相对链路起点的毫秒偏移（来自 Graph 内部事件）
        """
        self.current_step = _short(node_id, 128)
        started_at = self.started_at + timedelta(milliseconds=max(0, int(start_ms or 0)))
        self.steps.append(StepSpan(name=self.current_step, started_at=started_at))

    def end_node(self, node_id: str, *, end_ms: int | None = None,
                 status: str = "success", error: str | None = None) -> None:
        """
        结束指定节点：从后向前找到第一个同名且仍 running 的步骤，写入结束时间/状态/错误。

        业务逻辑：
            1. 结束时间 = 链路 started_at + end_ms 毫秒
            2. 逆序遍历 steps，定位首个 name 匹配且 status==running 的步骤
            3. 写入 status、error（截断到 2000 字符）、finished_at 后立即返回

        Args:
            node_id: 节点标识
            end_ms: 相对链路起点的毫秒偏移
            status: 节点结束状态（success/failed/...）
            error: 失败时的错误信息
        """
        finished_at = self.started_at + timedelta(milliseconds=max(0, int(end_ms or 0)))
        for step in reversed(self.steps):
            if step.name == node_id and step.status == "running":
                step.status = status
                step.error = _short(error, 2000)
                step.finished_at = finished_at
                return

    def record_node_event(self, event: dict) -> None:
        """
        消费 Graph 内部节点事件，按 event 字段分发到 start_node / end_node。

        业务逻辑：
            1. 取 node_id，缺失时归一化为 "unknown"
            2. event=="node_start" → 调用 start_node，传入 start_ms
            3. event=="node_end" → 调用 end_node，传入 end_ms / status / error
        """
        node_id = str(event.get("node_id") or "unknown")
        if event.get("event") == "node_start":
            self.start_node(node_id, event.get("start_ms"))
        elif event.get("event") == "node_end":
            self.end_node(
                node_id,
                end_ms=event.get("end_ms"),
                status=str(event.get("status") or "success"),
                error=event.get("error"),
            )

    def mark_first_event(self) -> None:
        """记录首事件耗时（TTFSE），仅首次调用生效，后续不覆盖。"""
        if self.ttfse_ms is None:
            self.ttfse_ms = max(0, int((_now() - self.started_at).total_seconds() * 1000))

    def mark_sql(self) -> None:
        """记录生成 SQL 的耗时（TT_SQL），仅首次调用生效。"""
        if self.tt_sql_ms is None:
            self.tt_sql_ms = max(0, int((_now() - self.started_at).total_seconds() * 1000))

    def mark_result(self) -> None:
        """记录拿到结果的耗时（TT_RESULT），仅首次调用生效。"""
        if self.tt_result_ms is None:
            self.tt_result_ms = max(0, int((_now() - self.started_at).total_seconds() * 1000))

    def add_degradation(self, item: dict) -> None:
        """追加一条降级证据（经 normalize_degradation 归一化后存入）。"""
        if self.route_catalog is None:
            self.degradations.append(normalize_degradation(item))
            return
        self.degradations.append(_normalize_degradation(item, self.route_catalog))

    def record_llm_attempt(self, item: dict) -> None:
        """
        记录节点显式上报的 LLM attempt（超时/取消/空输出等事实）。

        核心功能：节点在 LLM 调用异常路径下显式上报 attempt 事实，与回调采集的
                 token usage 在 finish 阶段做去重对账，避免同一 provider 调用被重复计数。
        """
        if self.route_catalog is None:
            self.explicit_llm_attempts.append(normalize_llm_attempt(item))
            return
        self.explicit_llm_attempts.append(
            _normalize_llm_attempt(item, self.route_catalog)
        )

    def record_llm_attempts(self, items) -> None:
        """批量记录 LLM attempt，跳过非 dict 元素。"""
        for item in items or []:
            if isinstance(item, dict):
                self.record_llm_attempt(item)

    def canonical_llm_attempts(self) -> list[dict]:
        """
        生成去重对账后的规范 LLM attempt 列表。

        核心功能：将回调采集的 token_usages 转为 callback 源 attempt，再与节点显式
                 上报的 explicit_llm_attempts 通过 reconcile_attempts 去重对账，
                 确保同一 provider 调用只保留一条规范记录。

        Returns:
            list[dict]: 去重后的规范 LLM attempt 列表
        """
        # 将回调采集的 token usage 转为统一格式的 attempt（source="callback"）
        callback_attempts = [make_llm_attempt(
            node_id=usage.step_name,
            route=self.route_for_node(usage.step_name),
            model=usage.model,
            status=usage.status,
            latency_ms=usage.latency_ms,
            timeout_budget_ms=usage.timeout_budget_ms,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            estimated_cost=usage.estimated_cost,
            token_known=usage.token_known,
            error_type=usage.error_type,
            retry_index=usage.retry_index,
            attempt_id=usage.attempt_id,
            source="callback",
            provider_request_id=usage.provider_request_id,
            provider_request_id_available=(
                usage.provider_request_id_available
            ),
            provider_request_id_unavailable_reason=(
                usage.provider_request_id_unavailable_reason
            ),
            token_timing_available=usage.token_timing_available,
            token_timing_unavailable_reason=(
                usage.token_timing_unavailable_reason
            ),
        ) for usage in self.token_usages]
        # 与显式 attempt 对账去重，避免同一调用被重复计数
        return reconcile_attempts(callback_attempts, self.explicit_llm_attempts)

    def record_follow_up(self, item: dict) -> None:
        """
        记录 follow-up 元数据，仅保留隐私安全的动作与摘要，绝不存历史结果行。

        业务逻辑：
            1. action 截断到 32 字符
            2. summary 截断到 500 字符
        """
        self.follow_up_action = _short(item.get("action"), 32)
        self.follow_up_summary = _short(item.get("summary"), 500)

    def record_candidate_decision(self, item: dict) -> None:
        """记录候选决策（候选 SQL/置信度等），经 compact_candidate_decision 压缩后存入。"""
        self.candidate_decision = compact_candidate_decision(item)

    def record_planning_route(self, item: dict) -> None:
        """Record only the TODO-63 privacy-safe route contract."""

        self.planning_route = compact_planning_route(item)

    def record_planning_timing(self, item: dict) -> None:
        """Whitelist timing fields; never infer TTFT from a non-stream call."""

        if not isinstance(item, dict):
            return
        token_timing_available = bool(item.get("token_timing_available"))
        self.planning_timing = {
            key: item.get(key)
            for key in (
                "contract_version",
                "queue_wait_ms",
                "provider_ms",
                "node_ms",
                "time_to_first_token_ms",
                "time_to_last_token_ms",
                "completion_mode",
                "token_timing_available",
                "status",
                "recovery",
            )
        }
        self.planning_timing["token_timing_available"] = token_timing_available
        if not token_timing_available:
            self.planning_timing["time_to_first_token_ms"] = None
            self.planning_timing["time_to_last_token_ms"] = None

    def record_semantic_repair(
        self,
        repair: dict | None = None,
        violation: dict | None = None,
    ) -> None:
        """Record only TODO-64 contracts; never SQL, literals or raw questions."""

        if repair:
            compact = compact_semantic_repair(repair)
            if compact:
                self.semantic_repair = {
                    **self.semantic_repair,
                    **compact,
                }
        if violation:
            safe = {
                key: violation.get(key)
                for key in (
                    "contract_version", "violation_code", "stage", "severity",
                    "repairability", "rule_version", "repair_hint_code",
                    "safe_to_retry",
                )
                if violation.get(key) is not None
            }
            safe["evidence"] = compact_semantic_evidence(
                violation.get("evidence")
            )
            self.semantic_repair["violation"] = safe

    def record_semantic_coverage(self, item: dict) -> None:
        """Persist structural slot coverage without question/filter literals."""

        if not isinstance(item, dict):
            return
        self.semantic_coverage = {
            key: item.get(key)
            for key in (
                "schema_version", "registry_version", "question_fingerprint",
                "coverage_score", "compiler_mode", "compiler_supported",
                "raw_question_persisted", "filter_values_persisted",
            )
            if item.get(key) is not None
        }
        for key, limit in (
            ("detected_slots", 40),
            ("represented_slots", 40),
            ("unresolved_slots", 40),
            ("abstain_reasons", 12),
        ):
            self.semantic_coverage[key] = [
                str(value)[:96]
                for value in (item.get(key) or [])[:limit]
            ]

    def record_semantic_query_contract(self, item: dict) -> None:
        """Persist only the bounded TODO-94 grain/formula/stage contract."""

        if not isinstance(item, dict):
            return
        self.semantic_query_contract = compact_query_contract(item)

    def record_relation_closure(self, item: dict) -> None:
        """Persist only metadata-owned bridge and join-key evidence."""

        if not isinstance(item, dict):
            return
        self.relation_closure = compact_relation_closure(item)

    def record_request_deadline(self, item: dict) -> None:
        """Persist relative deadline evidence; absolute monotonic values are banned."""

        if not isinstance(item, dict):
            return
        phase_budgets = item.get("phase_budgets_ms")
        self.request_deadline = {
            "contract_version": item.get("contract_version"),
            "budget_ms": _int(item.get("budget_ms")),
            "remaining_ms": _int(item.get("remaining_ms")),
            "execution_reserve_ms": _int(item.get("execution_reserve_ms")),
            "phase_budgets_ms": {
                str(key)[:32]: _int(value)
                for key, value in (
                    phase_budgets.items()
                    if isinstance(phase_budgets, dict)
                    else []
                )
            },
            "absolute_timestamps_persisted": False,
        }

    def record_retrieval_metric(self, item: dict) -> None:
        """
        记录检索指标证据，仅保留有界的 rank/count 字段，绝不存向量或源文本。

        业务逻辑：
            1. 非 dict 直接忽略
            2. 从 hybrid_queries 取前 8 条，按白名单字段过滤保留非空值
            3. 顶层按白名单字段过滤，并与 hybrid_queries 合并为最终记录

        Args:
            item: 检索指标事件，含 route/raw_count/dedup_count/hybrid_queries 等字段
        """
        if not isinstance(item, dict):
            return
        # 混合查询明细：仅取前 8 条，按白名单字段过滤
        hybrid_rows = []
        for row in (item.get("hybrid_queries") or [])[:8]:
            if not isinstance(row, dict):
                continue
            hybrid_rows.append({
                key: row.get(key) for key in (
                    "retrieval_mode", "effective_mode", "fusion_method", "rrf_k",
                    "oversample_factor", "physical_limit", "dense", "sparse",
                    "intersection_entities", "dense_only_entities",
                    "sparse_only_entities", "fused_entities", "degradation_count",
                    "point_strategies", "matched_point_roles",
                ) if row.get(key) is not None
            })
        value_search_audits = []
        for row in (item.get("value_search_audits") or [])[:8]:
            if not isinstance(row, dict):
                continue
            value_search_audits.append({
                key: row.get(key) for key in (
                    "schema", "responsibility", "engine", "workspace_id",
                    "query_mode", "query_fingerprint", "index_version",
                    "allowed_table_count", "allowed_column_count",
                    "explicit_table_filter", "explicit_column_filter",
                    "hit_count", "returned_count", "latency_ms",
                    "raw_value_logged", "filter_stage", "status",
                    "active_index_version",
                ) if row.get(key) is not None
            })
        embedding_runtime = {}
        raw_embedding_runtime = item.get("embedding_runtime")
        if isinstance(raw_embedding_runtime, dict):
            embedding_runtime = {
                key: raw_embedding_runtime.get(key)
                for key in (
                    "runtime_profile", "runtime_profile_fingerprint", "status",
                    "queue_depth", "queue_capacity", "max_queue_depth",
                    "last_batch_texts", "max_batch_texts_seen",
                    "merged_batch_count", "queue_wait_p95_ms",
                    "inference_p95_ms", "total_p95_ms", "rejected",
                    "cancelled", "timed_out", "failed", "model_instances",
                    "raw_text_logged",
                )
                if raw_embedding_runtime.get(key) is not None
            }
        query_signals = {}
        raw_query_signals = item.get("query_signals")
        if isinstance(raw_query_signals, dict):
            counts = raw_query_signals.get("counts")
            safe_counts = {}
            if isinstance(counts, dict):
                safe_counts = {
                    str(key): int(value)
                    for key, value in counts.items()
                    if isinstance(value, int) and value >= 0
                }
            safe_signals = []
            for row in (raw_query_signals.get("signals") or [])[:32]:
                if not isinstance(row, dict):
                    continue
                safe_signals.append({
                    key: row.get(key)
                    for key in ("signal_id", "kind", "rank", "fingerprint")
                    if row.get(key) is not None
                })
            safe_remote = []
            for row in (raw_query_signals.get("actual_remote_signals") or [])[:8]:
                if not isinstance(row, dict):
                    continue
                safe_remote.append({
                    key: row.get(key)
                    for key in (
                        "signal_id", "kind", "rank", "fingerprint", "consumer",
                    )
                    if row.get(key) is not None
                })
            query_signals = {
                key: raw_query_signals.get(key)
                for key in (
                    "contract_version", "extractor_profile", "extractor_version",
                    "dictionary_version", "idf_version", "stopword_version",
                    "rule_version", "profile_fingerprint", "signal_fingerprint",
                    "route", "route_policy", "route_policy_version",
                    "cache_namespace_fingerprint", "local_signal_count",
                    "remote_query_count", "rerank_signal_count",
                    "query_encode_mode", "raw_text_logged",
                )
                if raw_query_signals.get(key) is not None
            } | {
                "counts": safe_counts,
                "signals": safe_signals,
                "actual_remote_signals": safe_remote,
            }
        # 顶层字段按白名单过滤，与 hybrid_queries 合并
        self.retrieval_metrics.append({
            key: item.get(key) for key in (
                "version", "route", "raw_count", "dedup_count", "selected_count", "dynamic_k",
                "cutoff_reason", "embedding_mode", "fusion_method", "cache_hit",
                "responsibility", "engine", "index_version", "index_ready",
                "workspace_scope_applied", "allowed_table_count",
                "allowed_column_count", "query_modes", "literal_source",
                "sparse_literal_inference",
                "working_source", "working_version", "redis_hit",
                "recent_loaded", "recent_selected", "mysql_exact_candidates",
                "es_candidates", "qdrant_candidates", "authoritative_recheck",
                "raw_result_rows_injected", "historical_sql_reusable",
                "owner_scope_applied",
                "keyword_count", "remote_query_count",
            ) if item.get(key) is not None
        } | {
            "hybrid_queries": hybrid_rows,
            "value_search_audits": value_search_audits,
            "embedding_runtime": embedding_runtime,
            "query_signals": query_signals,
        })

    def fail(self, error: str) -> None:
        """
        标记链路失败：记录错误信息并收尾最后一个仍在 running 的步骤为 failed。

        业务逻辑：
            1. error 截断到 2000 字符存入 self.error
            2. 若最后一个步骤仍为 running，标记为 failed、补全结束时间与错误信息
        """
        self.error = _short(error, 2000)
        if self.steps and self.steps[-1].status == "running":
            self.steps[-1].status = "failed"
            self.steps[-1].error = self.error
            self.steps[-1].finished_at = _now()

    def finish(self, status: str | None = None) -> dict:
        """
        汇总链路所有运行期事实，生成结构化 trace summary 字典。

        核心功能：收尾未结束步骤、对账 LLM attempt、汇总 token/成本/降级/慢节点/候选决策
                 等指标，产出一份可供持久化与前端展示的 trace summary。

        业务逻辑：
            1. 补全 finished_at，收尾最后一个 running 步骤（按是否有 error 决定 success/failed）
            2. 调 canonical_llm_attempts 对账得到规范 attempt 列表，并算 attempt 计数
            3. 仅对 token_known 的 attempt 汇总 prompt/completion/total token
            4. 汇总估算成本（仅 callback 源，显式 attempt 不发明零价格事实）
            5. 取耗时最长的前 5 个步骤作为 slowest_nodes
            6. 按 node_id 聚合 LLM 调用统计（模型集合、调用数、失败数、各类状态计数、token、延迟分布）
            7. 计算每节点的 latency_mean_ms / latency_p95_ms，并按失败数/token/调用数排序
            8. 组装并返回完整 summary 字典

        Args:
            status: 显式状态覆盖；None 时按 self.error 推断 success/failed

        Returns:
            dict: 完整 trace summary，含耗时、token、成本、降级、慢节点、候选决策等字段
        """
        # =============================================================================
        # 阶段 1：收尾链路与最后一个步骤
        # =============================================================================
        self.finished_at = self.finished_at or _now()
        if self.steps and self.steps[-1].status == "running":
            self.steps[-1].status = "failed" if self.error else "success"
            self.steps[-1].finished_at = self.finished_at
            if self.error:
                self.steps[-1].error = self.error

        # =============================================================================
        # 阶段 2：对账 LLM attempt 并汇总 token / 成本
        # =============================================================================
        llm_attempts = self.canonical_llm_attempts()
        accounting = attempt_accounting(llm_attempts)
        # 仅对 token_known 的 attempt 汇总，避免未知 token 被误计入总量
        total = sum(int(item.get("total_tokens") or 0) for item in llm_attempts if item.get("token_known"))
        prompt = sum(int(item.get("prompt_tokens") or 0) for item in llm_attempts if item.get("token_known"))
        completion = sum(int(item.get("completion_tokens") or 0) for item in llm_attempts if item.get("token_known"))
        # 成本只取 provider 推导的 callback 证据；显式 attempt 在 token 未知时不发明零价格事实
        cost = round(sum(float(item.get("estimated_cost") or 0) for item in llm_attempts), 8)

        # =============================================================================
        # 阶段 3：慢节点排行与总耗时
        # =============================================================================
        slowest = sorted(self.steps, key=lambda item: item.duration_ms, reverse=True)[:5]
        complete_ms = max(0, int((self.finished_at - self.started_at).total_seconds() * 1000))

        # =============================================================================
        # 阶段 4：按节点聚合 LLM 调用统计
        # =============================================================================
        usage_by_node: dict[str, dict] = {}
        for usage in llm_attempts:
            node_id = str(usage.get("node_id") or "unknown")
            route = str(usage.get("route") or self.route_for_node(node_id))
            # 首次遇到该节点时初始化聚合桶
            item = usage_by_node.setdefault(node_id, {
                "node_id": node_id,
                "route": route,
                "models": set(),
                "calls": 0,
                "failed_calls": 0,
                "succeeded": 0,
                "errors": 0,
                "timed_out": 0,
                "cancelled": 0,
                "empty_output": 0,
                "unknown_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "latencies": [],
                "queue_wait_ms": 0,
                "provider_ms": 0,
                "recovery_calls": 0,
                "outcome_classes": set(),
            })
            # 累加模型、调用数与各类状态计数
            item["models"].add(
                usage.get("model")
                or self.profile_for_route(item["route"])["model_name"]
            )
            item["calls"] += 1
            status_name = str(usage.get("status") or "error")
            item["failed_calls"] += int(status_name != "success")
            item["succeeded"] += int(status_name == "success")
            item["errors"] += int(status_name == "error")
            item["timed_out"] += int(status_name == "timeout")
            item["cancelled"] += int(status_name == "cancelled")
            item["empty_output"] += int(status_name == "empty_output")
            item["unknown_tokens"] += int(not usage.get("token_known"))
            item["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            item["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            item["total_tokens"] += int(usage.get("total_tokens") or 0)
            item["latencies"].append(int(usage.get("latency_ms") or 0))
            item["queue_wait_ms"] += int(usage.get("queue_wait_ms") or 0)
            item["provider_ms"] += int(usage.get("provider_ms") or 0)
            item["recovery_calls"] += int(bool(usage.get("recovery")))
            item["outcome_classes"].add(
                str(usage.get("outcome_class") or status_name)
            )

        # =============================================================================
        # 阶段 5：计算每节点延迟统计并排序
        # =============================================================================
        usage_rows = []
        for item in usage_by_node.values():
            # 弹出 latencies 用于计算统计量，不再保留在最终输出中
            latencies = sorted(item.pop("latencies"))
            item["models"] = sorted(item["models"])
            item["outcome_classes"] = sorted(item["outcome_classes"])
            item["latency_mean_ms"] = round(sum(latencies) / len(latencies), 1) if latencies else 0
            # p95：取排序后第 ceil(0.95*n) 个位置，保证至少 1 个样本时不越界
            item["latency_p95_ms"] = latencies[max(0, int(len(latencies) * .95 + .9999) - 1)] if latencies else 0
            usage_rows.append(item)
        # 优先按失败数、其次 token、再次调用数倒序排列，便于快速定位问题节点
        usage_rows.sort(key=lambda item: (item["failed_calls"], item["total_tokens"], item["calls"]), reverse=True)

        # =============================================================================
        # 阶段 6：组装并返回完整 summary
        # =============================================================================
        semantic_coverage = dict(self.semantic_coverage)
        if self.semantic_query_contract:
            semantic_coverage["query_contract"] = self.semantic_query_contract
        if self.relation_closure:
            semantic_coverage["relation_closure"] = self.relation_closure

        return {
            "trace_id": self.trace_id,
            "status": status or ("failed" if self.error else "success"),
            "duration_ms": complete_ms,
            "ttfse_ms": self.ttfse_ms,
            "tt_sql_ms": self.tt_sql_ms,
            "tt_result_ms": self.tt_result_ms,
            "tt_complete_ms": complete_ms,
            "step_count": len(self.steps),
            "llm_calls": accounting["attempts"],
            "failed_llm_calls": accounting["attempts"] - accounting["succeeded"],
            "failed_core_llm_calls": accounting["core_failures"],
            "failed_optional_llm_calls": accounting["optional_failures"],
            "llm_attempts": llm_attempts,
            "llm_attempt_accounting": accounting,
            "llm_attempt_evidence_complete": accounting["evidence_complete"],
            "llm_callback_missing": accounting["callback_missing"],
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "estimated_cost": cost,
            "cost_currency": self.resolved_llm_settings.cost_currency,
            "degraded_count": len(self.degradations),
            "degradations": self.degradations,
            "core_degraded_count": sum(item.get("affects_core_path") for item in self.degradations),
            "optional_degraded_count": sum(not item.get("affects_core_path") for item in self.degradations),
            "degradation_evidence_complete": all(
                item.get("evidence_complete") for item in self.degradations
            ),
            "slowest_nodes": [
                {"node_id": item.name, "duration_ms": item.duration_ms, "status": item.status}
                for item in slowest
            ],
            "follow_up_action": self.follow_up_action,
            "follow_up_summary": self.follow_up_summary,
            "candidate_decision": self.candidate_decision,
            "retrieval_metrics": self.retrieval_metrics,
            "planning_route": (
                {
                    **self.planning_route,
                    "timing": self.planning_timing,
                }
                if self.planning_route
                else {}
            ),
            "semantic_repair": self.semantic_repair,
            "semantic_coverage": semantic_coverage,
            "request_deadline": self.request_deadline,
            "candidate_count": int(self.candidate_decision.get("candidate_count") or 1),
            # 取被选中候选的 score / band，用于前端展示置信度档位
            "candidate_confidence": next((
                row.get("score") for row in self.candidate_decision.get("candidates", [])
                if row.get("label") == self.candidate_decision.get("selected", "A")
            ), None),
            "candidate_band": next((
                row.get("band") for row in self.candidate_decision.get("candidates", [])
                if row.get("label") == self.candidate_decision.get("selected", "A")
            ), None),
            "usage_by_node": usage_rows,
        }
