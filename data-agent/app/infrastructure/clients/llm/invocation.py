"""统一的可观测 LLM 调用边界。

本模块负责排队、超时分配和一次受控恢复；Provider SDK 自身保持关闭重试，
从而保证每次物理调用都生成且仅生成一条隐私安全的 attempt 记录。
"""
from __future__ import annotations

import asyncio
import copy
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from app.application.ports.llm_routing import LLMRouteCatalog
from app.config.app_config import app_config
from app.foundation.deadlines import REQUEST_DEADLINE_VERSION
from app.foundation.llm_evidence import make_llm_attempt
from app.foundation.observability import normalize_degradation_evidence


_CONNECTION_MARKERS = (
    "connection",
    "connecterror",
    "network",
    "apiconnection",
    "connecttimeout",
)
_STREAM_MARKERS = (
    "incompleteread",
    "streamerror",
    "streamclosed",
    "chunkedencoding",
    "remoteprotocol",
    "unexpected eof",
)
_RATE_LIMIT_MARKERS = ("ratelimit", "too many requests")
_SERVER_MARKERS = ("serviceunavailable", "temporarilyunavailable", "servererror")
FATAL_PROVIDER_REASON_CODES = frozenset({
    "llm.authentication_error",
    "llm.payment_required",
    "llm.permission_denied",
})
_RETRYABLE_REASONS = {
    "llm.connection_error",
    "llm.rate_limited",
    "llm.server_error",
    "llm.stream_interrupted",
    "llm.timeout",
    "llm.empty_output",
    "llm.truncated_output",
    "llm.invalid_structured_output",
}
_SEMAPHORES: dict[tuple[int, str, str, int], asyncio.Semaphore] = {}
_DEFAULT_TASK = "sql"
_DEFAULT_SENTINELS = {"", "__default__", "null", "none"}
_NODE_TASK_ROUTES = {
    "classify_query_intent": "keyword",
    "extract_keywords": "keyword",
    "recall_column": "keyword",
    "recall_metric": "keyword",
    "recall_value": "keyword",
    "filter_table": "selection",
    "filter_metric": "selection",
    "plan_sql": "planning",
    "generate_sql": "sql",
    "correct_sql": "correction",
    "judge_candidate": "candidate",
    "generate_chart": "postprocess",
    "generate_analysis": "postprocess",
    "generate_recommend": "recommend",
}


def _optional_text(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in _DEFAULT_SENTINELS else text


def route_for_node(node_id: str | None) -> str:
    """把节点映射到任务路由；保留原模块可 patch 的兼容入口。"""

    return _NODE_TASK_ROUTES.get(str(node_id or ""), _DEFAULT_TASK)


def task_profile(task: str | None) -> dict[str, Any]:
    """解析调用边界实际需要的非敏感任务执行参数。"""

    route = str(task or _DEFAULT_TASK)
    configured = dict((app_config.llm.routes or {}).get(route) or {})
    model_name = _optional_text(configured.get("model_name")) or str(
        app_config.llm.model_name
    )
    retries = configured.get("max_retries")
    return {
        "task": route,
        "model_name": model_name,
        "timeout_seconds": float(
            configured.get("timeout_seconds") or app_config.llm.timeout_seconds
        ),
        "max_retries": int(
            retries if retries is not None else app_config.llm.max_retries
        ),
        "max_concurrency": max(
            1,
            int(
                configured.get("max_concurrency")
                or app_config.query.llm_route_default_concurrency
            ),
        ),
    }


def _route(
    catalog: LLMRouteCatalog | None,
    node_id: str,
    explicit_route: str | None,
) -> str:
    if explicit_route:
        return str(explicit_route)
    return str(
        catalog.route_for_node(node_id)
        if catalog is not None
        else route_for_node(node_id)
    )


def _profile(
    catalog: LLMRouteCatalog | None,
    route: str,
) -> dict[str, Any]:
    return dict(
        catalog.profile_for_route(route)
        if catalog is not None
        else task_profile(route)
    )


@dataclass(frozen=True)
class LLMInvocationResult:
    value: Any
    attempts: list[dict[str, Any]]
    degradations: list[dict[str, Any]]


class LLMInvocationError(RuntimeError):
    """Typed failure carrying safe evidence but no provider response body."""

    def __init__(
        self,
        *,
        node_id: str,
        route: str,
        reason_code: str,
        error_type: str,
        timeout_budget_ms: int,
        attempts: list[dict[str, Any]],
        degradations: list[dict[str, Any]],
    ) -> None:
        self.node_id = node_id
        self.route = route
        self.reason_code = reason_code
        self.error_type = error_type
        self.timeout_budget_ms = timeout_budget_ms
        self.attempts = attempts
        self.degradations = degradations
        super().__init__(
            f"{reason_code} node={node_id} route={route} "
            f"budget_ms={timeout_budget_ms} error_type={error_type}"
        )

    @property
    def user_message(self) -> str:
        messages = {
            "llm.timeout": "模型调用超时，请稍后重试或缩小问题范围",
            "llm.deadline_exhausted": "本次查询的模型调用预算已耗尽，请缩小问题范围",
            "llm.queue_timeout": "模型调用排队超时，请稍后重试",
            "llm.cancelled": "模型调用已取消",
            "llm.connection_error": "模型服务暂时无法连接，请稍后重试",
            "llm.rate_limited": "模型服务当前限流，请稍后重试",
            "llm.server_error": "模型服务暂时异常，请稍后重试",
            "llm.authentication_error": "模型服务认证失败，请检查当前模型凭据配置",
            "llm.payment_required": "模型服务计费状态不可用，请检查账户余额或计费配置",
            "llm.permission_denied": "模型服务拒绝访问，请检查当前模型的访问权限",
            "llm.stream_interrupted": "模型响应流中断，请稍后重试",
            "llm.empty_output": "模型未返回有效内容，请重试或补充问题信息",
            "llm.truncated_output": "模型输出被截断，请缩小问题范围",
            "llm.invalid_structured_output": "模型未按 SQL 格式返回内容，请稍后重试",
            "llm.refusal": "模型拒绝了本次请求，请调整问题后重试",
            "llm.provider_error": "模型服务调用失败，请稍后重试",
            "llm.pre_invoke_failure": "模型调用准备失败，请检查当前配置与上下文",
            "llm.unknown_failure": "模型调用发生未知错误",
        }
        return (
            f"{messages.get(self.reason_code, messages['llm.unknown_failure'])}"
            f"（{self.reason_code}，节点 {self.node_id}）"
        )


def _http_status(exc: BaseException) -> int | None:
    for candidate in (
        getattr(exc, "status_code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
        getattr(exc, "status", None),
    ):
        try:
            if candidate is not None:
                return int(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _reason_code(exc: BaseException) -> str:
    if isinstance(exc, asyncio.CancelledError):
        return "llm.cancelled"
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return "llm.timeout"
    status = _http_status(exc)
    if status == 401:
        return "llm.authentication_error"
    if status == 402:
        return "llm.payment_required"
    if status == 403:
        return "llm.permission_denied"
    if status == 429:
        return "llm.rate_limited"
    if status is not None and status >= 500:
        return "llm.server_error"
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if any(marker in name or marker in message for marker in _STREAM_MARKERS):
        return "llm.stream_interrupted"
    if any(marker in name or marker in message for marker in _RATE_LIMIT_MARKERS):
        return "llm.rate_limited"
    if any(marker in name or marker in message for marker in _SERVER_MARKERS):
        return "llm.server_error"
    if isinstance(exc, (ConnectionError, OSError)) or any(
        marker in name for marker in _CONNECTION_MARKERS
    ):
        return "llm.connection_error"
    return "llm.provider_error"


def _status(reason_code: str) -> str:
    if reason_code in {"llm.timeout", "llm.deadline_exhausted", "llm.queue_timeout"}:
        return "timeout"
    if reason_code == "llm.cancelled":
        return "cancelled"
    if reason_code == "llm.empty_output":
        return "empty_output"
    return "error"


def _content_blocks_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, (list, tuple)):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "").lower()
        if block_type and block_type not in {"text", "output_text"}:
            continue
        text = block.get("text")
        if isinstance(text, dict):
            text = text.get("value")
        if text is None:
            text = block.get("content")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def model_output_text(value: Any) -> str:
    """Extract assistant text while retaining the original provider envelope.

    SQL nodes pass the raw LangChain message through the invocation boundary so
    finish reason, request id and token usage remain observable.  This helper
    accepts both LangChain message objects and OpenAI-compatible dictionaries.
    It never falls back to reasoning content: only the provider's final answer
    channel is eligible for execution.
    """

    if isinstance(value, str):
        return value
    content = getattr(value, "content", None)
    if content is not None:
        return _content_blocks_text(content)
    if isinstance(value, dict):
        if "content" in value:
            return _content_blocks_text(value.get("content"))
        choices = value.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0] if isinstance(choices[0], dict) else {}
            message = first.get("message") if isinstance(first, dict) else {}
            if isinstance(message, dict):
                return _content_blocks_text(message.get("content"))
            return _content_blocks_text(first.get("text"))
    return str(value or "")


def _output_issue(value: Any) -> tuple[str, str] | None:
    """Classify provider/output-parser failures without retaining response text."""

    if value is None:
        return "llm.empty_output", "empty_body"
    metadata = getattr(value, "response_metadata", None)
    if not isinstance(metadata, dict) and isinstance(value, dict):
        metadata = value.get("response_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    finish_reason = str(
        metadata.get("finish_reason")
        or metadata.get("stop_reason")
        or (
            value.get("finish_reason")
            if isinstance(value, dict)
            else ""
        )
        or ""
    ).lower()
    if finish_reason in {"length", "max_tokens", "max_output_tokens"}:
        return "llm.truncated_output", "truncation"
    refusal = (
        getattr(value, "refusal", None)
        or metadata.get("refusal")
        or (
            value.get("refusal")
            if isinstance(value, dict)
            else None
        )
    )
    if refusal:
        return "llm.refusal", "refusal"
    if isinstance(value, dict):
        if "choices" in value and not value.get("choices"):
            return "llm.empty_output", "empty_choices"
        if not value:
            return "llm.empty_output", "parsed_empty"
    if isinstance(value, (list, tuple, set)) and not value:
        return "llm.empty_output", "empty_choices"
    text = model_output_text(value)
    if (
        isinstance(value, str)
        or hasattr(value, "content")
        or (isinstance(value, dict) and ("content" in value or "choices" in value))
    ) and not text.strip():
        return "llm.empty_output", "parsed_empty"
    return None


def _provider_request_id(value: Any) -> tuple[str, bool, str | None]:
    metadata = getattr(value, "response_metadata", None)
    if not isinstance(metadata, dict) and isinstance(value, dict):
        metadata = value.get("response_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    candidate = (
        metadata.get("request_id")
        or metadata.get("request-id")
        or metadata.get("x-request-id")
        or (
            value.get("request_id")
            if isinstance(value, dict)
            else None
        )
    )
    text = str(candidate or "").strip()
    return (
        (text[:160], True, None)
        if text
        else (
            "unavailable",
            False,
            "parsed_output_did_not_retain_provider_request_id",
        )
    )


def _provider_usage(value: Any) -> tuple[int | None, int | None, int | None]:
    usage = getattr(value, "usage_metadata", None)
    if not isinstance(usage, dict) and isinstance(value, dict):
        usage = value.get("usage_metadata")
    metadata = getattr(value, "response_metadata", None)
    if not isinstance(metadata, dict) and isinstance(value, dict):
        metadata = value.get("response_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    provider_usage = metadata.get("token_usage")
    merged = {
        **(provider_usage if isinstance(provider_usage, dict) else {}),
        **(usage if isinstance(usage, dict) else {}),
    }

    def integer(*names: str) -> int | None:
        for name in names:
            raw = merged.get(name)
            try:
                if raw is not None:
                    return max(0, int(raw))
            except (TypeError, ValueError):
                continue
        return None

    prompt = integer("input_tokens", "prompt_tokens")
    completion = integer("output_tokens", "completion_tokens")
    total = integer("total_tokens")
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return prompt, completion, total


def normalize_sql_output(value: Any) -> str:
    """Remove a complete SQL Markdown fence while rejecting no extra prose."""

    text = model_output_text(value).strip()
    if not (text.startswith("```") and text.endswith("```")):
        return text
    lines = text.splitlines()
    if len(lines) < 3:
        return text
    if lines[0].strip().lower() not in {"```", "```sql", "```mysql"}:
        return text
    return "\n".join(lines[1:-1]).strip()


def sql_output_validator(value: Any) -> str | None:
    """Detect transport-shaped SQL output defects before semantic validation.

    This deliberately does not decide SQL correctness.  Authorization, AST
    safety, semantic guards and cost checks still run after every accepted
    output.  It only catches obvious refusal/prose/fence truncation so the one
    recovery opportunity is not wasted on a guaranteed parse failure.
    """

    text = model_output_text(value).strip()
    lowered = text.lower()
    refusal_prefixes = (
        "i can't",
        "i cannot",
        "i’m unable",
        "i am unable",
        "抱歉，我不能",
        "无法帮助",
    )
    if lowered.startswith(refusal_prefixes):
        return "llm.refusal"
    if text.startswith("```") and not text.endswith("```"):
        return "llm.truncated_output"
    candidate = normalize_sql_output(text)
    if not candidate.lower().startswith(("select ", "select\n", "with ")):
        return "llm.invalid_structured_output"
    # Obvious incomplete delimiters are transport/output-budget failures.  Full
    # SQL parsing remains the responsibility of validate_sql.
    if candidate.count("(") > candidate.count(")"):
        return "llm.truncated_output"
    return None


def _emit(runtime: Any, payload: dict[str, Any]) -> None:
    writer = getattr(runtime, "stream_writer", None)
    if callable(writer):
        writer(payload)


def _degradation(
    *,
    node_id: str,
    route: str,
    reason_code: str,
    core_path: bool,
    duration_ms: int,
    error_type: str,
) -> dict[str, Any]:
    return normalize_degradation_evidence({
        "node_id": node_id,
        "route": route,
        "reason_code": reason_code,
        "severity": "error" if core_path else "warning",
        "affects_core_path": core_path,
        "message": f"{reason_code} ({error_type})",
        "duration_ms": duration_ms,
    })


def _semaphore(scope: str, key: str, limit: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    cache_key = (id(loop), scope, str(key), max(1, int(limit)))
    semaphore = _SEMAPHORES.get(cache_key)
    if semaphore is None:
        semaphore = asyncio.Semaphore(cache_key[-1])
        _SEMAPHORES[cache_key] = semaphore
    return semaphore


async def _acquire(
    semaphores: list[asyncio.Semaphore],
    *,
    deadline: float,
) -> list[asyncio.Semaphore]:
    acquired: list[asyncio.Semaphore] = []
    try:
        for semaphore in semaphores:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise asyncio.TimeoutError()
            await asyncio.wait_for(semaphore.acquire(), timeout=remaining)
            acquired.append(semaphore)
        return acquired
    except BaseException:
        for semaphore in reversed(acquired):
            semaphore.release()
        raise


def _release(semaphores: list[asyncio.Semaphore]) -> None:
    for semaphore in reversed(semaphores):
        semaphore.release()


def _remaining_ms(deadline: float) -> int:
    return max(0, int((deadline - time.perf_counter()) * 1000))


def _recovery_value(recovery_payload: Any, original_payload: Any) -> Any:
    if recovery_payload is None:
        return original_payload
    if callable(recovery_payload):
        return recovery_payload(original_payload)
    return recovery_payload


def pre_invoke_failure(
    *,
    node_id: str,
    error: BaseException,
    route: str | None = None,
    timeout_seconds: float | None = None,
    core_path: bool = True,
    runtime: Any = None,
    route_catalog: LLMRouteCatalog | None = None,
) -> LLMInvocationError:
    """Create complete evidence for a failure before provider invocation."""

    resolved_route = _route(route_catalog, node_id, route)
    profile = _profile(route_catalog, resolved_route)
    budget_ms = max(
        1,
        int(float(timeout_seconds or profile["timeout_seconds"]) * 1000),
    )
    error_type = type(error).__name__ or "UnknownError"
    attempt = make_llm_attempt(
        node_id=node_id,
        route=resolved_route,
        model=profile["model_name"],
        status="error",
        latency_ms=0,
        timeout_budget_ms=budget_ms,
        token_known=False,
        error_type=error_type,
        reason_code="llm.pre_invoke_failure",
        retry_index=0,
        callback_observed=False,
        pre_invoke_failure=True,
        core_path=core_path,
        source="unified_boundary",
        deadline_before_ms=budget_ms,
        deadline_after_ms=budget_ms,
        completion_stage="pre_invoke",
        outcome_class="pre_invoke_failure",
    )
    degradation = _degradation(
        node_id=node_id,
        route=resolved_route,
        reason_code="llm.pre_invoke_failure",
        core_path=core_path,
        duration_ms=0,
        error_type=error_type,
    )
    _emit(runtime, {"llm_attempt": attempt})
    _emit(runtime, {"degradation": degradation})
    return LLMInvocationError(
        node_id=node_id,
        route=resolved_route,
        reason_code="llm.pre_invoke_failure",
        error_type=error_type,
        timeout_budget_ms=budget_ms,
        attempts=[attempt],
        degradations=[degradation],
    )


async def invoke_llm(
    runnable: Any,
    payload: Any,
    *,
    node_id: str,
    route: str | None = None,
    timeout_seconds: float | None = None,
    max_retries: int | None = None,
    core_path: bool = True,
    allow_empty: bool = False,
    config: dict[str, Any] | None = None,
    runtime: Any = None,
    request_deadline_at: float | None = None,
    recovery_payload: Any = None,
    recovery_enabled: bool = True,
    recovery_min_ms: int | None = None,
    recovery_reserve_ms: int | None = None,
    primary_min_ms: int | None = None,
    retryable_reason_codes: set[str] | None = None,
    output_validator: Callable[[Any], str | None] | None = None,
    route_catalog: LLMRouteCatalog | None = None,
) -> LLMInvocationResult:
    """Invoke one route under a shared deadline with at most one recovery.

    The configured policy may reserve a bounded tail from the first physical
    call. Recovery is useful only when the first attempt fails with an allowed
    reason and enough real deadline remains; setting the reserve to zero is the
    explicit no-recovery-tail ablation.
    Recovery reuses no hidden state: callers must provide a compressed
    ``recovery_payload`` when semantic context is needed.
    """

    resolved_route = _route(route_catalog, node_id, route)
    profile = _profile(route_catalog, resolved_route)
    total_timeout = max(
        0.001,
        float(timeout_seconds or profile["timeout_seconds"]),
    )
    retry_limit = max(
        0,
        int(profile["max_retries"] if max_retries is None else max_retries),
    )
    attempt_limit = min(2, retry_limit + 1) if recovery_enabled else 1
    started_boundary = time.perf_counter()
    local_deadline = started_boundary + total_timeout
    deadline = min(
        local_deadline,
        float(request_deadline_at)
        if request_deadline_at is not None
        else local_deadline,
    )
    # Round the configured boundary budget instead of flooring a monotonic
    # subtraction (e.g. 10 ms becoming 9 ms due to floating-point precision).
    timeout_budget_ms = max(
        1,
        int(round(max(0.001, deadline - started_boundary) * 1000)),
    )
    configured_min = max(1, int(app_config.query.llm_min_attempt_ms))
    min_attempt_ms = min(
        configured_min,
        max(1, timeout_budget_ms // 4),
    )
    effective_recovery_reserve_ms = max(
        0,
        int(
            app_config.query.llm_recovery_reserve_ms
            if recovery_reserve_ms is None
            else recovery_reserve_ms
        ),
    )
    effective_recovery_min_ms = max(
        min_attempt_ms,
        int(min_attempt_ms if recovery_min_ms is None else recovery_min_ms),
    )
    effective_primary_min_ms = max(
        min_attempt_ms,
        int(min_attempt_ms if primary_min_ms is None else primary_min_ms),
    )
    allowed_retry_reasons = (
        _RETRYABLE_REASONS
        if retryable_reason_codes is None
        else {str(item) for item in retryable_reason_codes}
    )
    attempts: list[dict[str, Any]] = []
    degradations: list[dict[str, Any]] = []
    global_semaphore = _semaphore(
        "global",
        "provider",
        max(1, int(app_config.llm.max_concurrency)),
    )
    route_semaphore = _semaphore(
        "route",
        resolved_route,
        max(1, int(profile.get("max_concurrency") or 1)),
    )

    for retry_index in range(attempt_limit):
        attempt_id = uuid.uuid4().hex
        recovery = retry_index > 0
        deadline_before_ms = _remaining_ms(deadline)
        if deadline_before_ms < min_attempt_ms:
            reason_code = "llm.deadline_exhausted"
            error_type = "DeadlineExhausted"
            attempt = make_llm_attempt(
                node_id=node_id,
                route=resolved_route,
                model=profile["model_name"],
                status="timeout",
                latency_ms=0,
                timeout_budget_ms=timeout_budget_ms,
                token_known=False,
                error_type=error_type,
                reason_code=reason_code,
                retry_index=retry_index,
                callback_observed=False,
                pre_invoke_failure=True,
                core_path=core_path,
                attempt_id=attempt_id,
                source="unified_boundary",
                deadline_before_ms=deadline_before_ms,
                deadline_after_ms=deadline_before_ms,
                completion_stage="deadline_check",
                outcome_class="deadline_exhausted",
                recovery=recovery,
                retryable=False,
                recovery_reserve_ms=0,
                physical_attempt_budget_ms=0,
                provider_request_id_unavailable_reason=(
                    "physical_provider_call_not_started"
                ),
                token_timing_unavailable_reason=(
                    "physical_provider_call_not_started"
                ),
            )
            degradation = _degradation(
                node_id=node_id,
                route=resolved_route,
                reason_code=reason_code,
                core_path=core_path,
                duration_ms=0,
                error_type=error_type,
            )
            attempts.append(attempt)
            degradations.append(degradation)
            _emit(runtime, {"llm_attempt": attempt})
            _emit(runtime, {"degradation": degradation})
            raise LLMInvocationError(
                node_id=node_id,
                route=resolved_route,
                reason_code=reason_code,
                error_type=error_type,
                timeout_budget_ms=timeout_budget_ms,
                attempts=attempts,
                degradations=degradations,
            )

        can_reserve = (
            retry_index == 0
            and attempt_limit > 1
            and effective_recovery_reserve_ms > 0
            and deadline_before_ms
            >= effective_recovery_reserve_ms + effective_primary_min_ms
        )
        reserved_ms = effective_recovery_reserve_ms if can_reserve else 0
        attempt_deadline = deadline - reserved_ms / 1000
        physical_attempt_budget_ms = max(
            0,
            int((attempt_deadline - time.perf_counter()) * 1000),
        )
        queue_started = time.perf_counter()
        acquired: list[asyncio.Semaphore] = []
        queue_wait_ms = 0
        provider_ms = 0
        completion_stage = "queue"
        cancel_stage: str | None = None
        value: Any = None
        reason_code = "llm.unknown_failure"
        outcome_class = "unknown_failure"
        error: BaseException = RuntimeError("unknown_model_failure")
        try:
            acquired = await _acquire(
                [global_semaphore, route_semaphore],
                deadline=attempt_deadline,
            )
            queue_wait_ms = max(
                0, int((time.perf_counter() - queue_started) * 1000)
            )
            provider_remaining = attempt_deadline - time.perf_counter()
            if provider_remaining <= 0:
                reason_code = "llm.queue_timeout"
                outcome_class = "queue_timeout"
                error = asyncio.TimeoutError()
                completion_stage = "queue"
            else:
                invoke_config = dict(config or {})
                metadata = dict(invoke_config.get("metadata") or {})
                metadata.update({
                    "data_agent_attempt_id": attempt_id,
                    "data_agent_retry_index": retry_index,
                    "data_agent_recovery": recovery,
                })
                invoke_config["metadata"] = metadata
                physical_payload = (
                    _recovery_value(recovery_payload, payload)
                    if recovery
                    else payload
                )
                provider_started = time.perf_counter()
                completion_stage = "provider"
                try:
                    value = await asyncio.wait_for(
                        runnable.ainvoke(physical_payload, config=invoke_config),
                        timeout=provider_remaining,
                    )
                    provider_ms = max(
                        0, int((time.perf_counter() - provider_started) * 1000)
                    )
                except BaseException:
                    provider_ms = max(
                        0, int((time.perf_counter() - provider_started) * 1000)
                    )
                    raise
                issue = None if allow_empty else _output_issue(value)
                if issue is None and output_validator is not None:
                    validation_reason = output_validator(value)
                    if validation_reason:
                        normalized = str(validation_reason)
                        if not normalized.startswith("llm."):
                            normalized = f"llm.{normalized}"
                        issue = (normalized, normalized.removeprefix("llm."))
                if issue is None:
                    (
                        provider_request_id,
                        provider_id_available,
                        provider_id_unavailable_reason,
                    ) = (
                        _provider_request_id(value)
                    )
                    prompt_tokens, completion_tokens, total_tokens = _provider_usage(value)
                    token_known = all(
                        item is not None
                        for item in (prompt_tokens, completion_tokens, total_tokens)
                    )
                    deadline_after_ms = _remaining_ms(deadline)
                    attempt = make_llm_attempt(
                        node_id=node_id,
                        route=resolved_route,
                        model=profile["model_name"],
                        status="success",
                        latency_ms=queue_wait_ms + provider_ms,
                        timeout_budget_ms=timeout_budget_ms,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        token_known=token_known,
                        reason_code="llm.success",
                        retry_index=retry_index,
                        callback_observed=False,
                        core_path=core_path,
                        attempt_id=attempt_id,
                        source="unified_boundary",
                        deadline_before_ms=deadline_before_ms,
                        deadline_after_ms=deadline_after_ms,
                        queue_wait_ms=queue_wait_ms,
                        provider_ms=provider_ms,
                        token_timing_available=False,
                        completion_stage="provider_complete",
                        provider_request_id=provider_request_id,
                        provider_request_id_available=provider_id_available,
                        provider_request_id_unavailable_reason=(
                            provider_id_unavailable_reason
                        ),
                        token_timing_unavailable_reason=(
                            "non_stream_invocation_does_not_expose_token_timing"
                        ),
                        outcome_class="success",
                        recovery=recovery,
                        retryable=False,
                        recovery_reserve_ms=reserved_ms,
                        physical_attempt_budget_ms=(
                            physical_attempt_budget_ms
                        ),
                    )
                    attempts.append(attempt)
                    _emit(runtime, {"llm_attempt": attempt})
                    return LLMInvocationResult(
                        value=value,
                        attempts=attempts,
                        degradations=degradations,
                    )
                reason_code, outcome_class = issue
                error = ValueError(outcome_class)
                completion_stage = "output_validation"
        except asyncio.CancelledError as cancelled:
            queue_wait_ms = max(
                queue_wait_ms,
                int((time.perf_counter() - queue_started) * 1000)
                if completion_stage == "queue"
                else queue_wait_ms,
            )
            cancel_stage = completion_stage
            deadline_after_ms = _remaining_ms(deadline)
            attempt = make_llm_attempt(
                node_id=node_id,
                route=resolved_route,
                model=profile["model_name"],
                status="cancelled",
                latency_ms=queue_wait_ms + provider_ms,
                timeout_budget_ms=timeout_budget_ms,
                token_known=False,
                error_type=type(cancelled).__name__,
                reason_code="llm.cancelled",
                retry_index=retry_index,
                callback_observed=False,
                core_path=core_path,
                attempt_id=attempt_id,
                source="unified_boundary",
                deadline_before_ms=deadline_before_ms,
                deadline_after_ms=deadline_after_ms,
                queue_wait_ms=queue_wait_ms,
                provider_ms=provider_ms,
                completion_stage=completion_stage,
                cancel_stage=cancel_stage,
                outcome_class="cancelled",
                recovery=recovery,
                retryable=False,
                recovery_reserve_ms=reserved_ms,
                physical_attempt_budget_ms=physical_attempt_budget_ms,
                provider_request_id_unavailable_reason=(
                    "provider_response_unavailable_after_cancellation"
                ),
                token_timing_unavailable_reason=(
                    "provider_response_unavailable_after_cancellation"
                ),
            )
            degradation = _degradation(
                node_id=node_id,
                route=resolved_route,
                reason_code="llm.cancelled",
                core_path=core_path,
                duration_ms=queue_wait_ms + provider_ms,
                error_type=type(cancelled).__name__,
            )
            attempts.append(attempt)
            degradations.append(degradation)
            _emit(runtime, {"llm_attempt": attempt})
            _emit(runtime, {"degradation": degradation})
            setattr(cancelled, "llm_attempts", attempts)
            setattr(cancelled, "degradations", degradations)
            raise
        except asyncio.TimeoutError as caught:
            queue_wait_ms = max(
                queue_wait_ms,
                int((time.perf_counter() - queue_started) * 1000)
                if completion_stage == "queue"
                else queue_wait_ms,
            )
            error = caught
            if completion_stage == "queue":
                reason_code = "llm.queue_timeout"
                outcome_class = "queue_timeout"
            else:
                reason_code = "llm.timeout"
                outcome_class = "timeout"
        except Exception as caught:
            error = caught
            reason_code = _reason_code(caught)
            outcome_class = reason_code.removeprefix("llm.")
        finally:
            _release(acquired)

        error_type = type(error).__name__ or "UnknownError"
        retryable = (
            reason_code in allowed_retry_reasons
            and retry_index + 1 < attempt_limit
            and _remaining_ms(deadline) >= effective_recovery_min_ms
        )
        deadline_after_ms = _remaining_ms(deadline)
        if value is not None and completion_stage == "output_validation":
            (
                provider_request_id,
                provider_id_available,
                provider_id_unavailable_reason,
            ) = _provider_request_id(value)
            prompt_tokens, completion_tokens, total_tokens = _provider_usage(value)
            token_known = all(
                item is not None
                for item in (prompt_tokens, completion_tokens, total_tokens)
            )
            timing_unavailable_reason = (
                "non_stream_invocation_does_not_expose_token_timing"
            )
        else:
            provider_request_id = "unavailable"
            provider_id_available = False
            provider_id_unavailable_reason = (
                f"provider_response_unavailable_after_{outcome_class}"
            )
            prompt_tokens = completion_tokens = total_tokens = None
            token_known = False
            timing_unavailable_reason = (
                f"provider_response_unavailable_after_{outcome_class}"
            )
        attempt = make_llm_attempt(
            node_id=node_id,
            route=resolved_route,
            model=profile["model_name"],
            status=_status(reason_code),
            latency_ms=queue_wait_ms + provider_ms,
            timeout_budget_ms=timeout_budget_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            token_known=token_known,
            error_type=error_type,
            reason_code=reason_code,
            retry_index=retry_index,
            callback_observed=False,
            core_path=core_path,
            attempt_id=attempt_id,
            source="unified_boundary",
            deadline_before_ms=deadline_before_ms,
            deadline_after_ms=deadline_after_ms,
            queue_wait_ms=queue_wait_ms,
            provider_ms=provider_ms,
            token_timing_available=False,
            completion_stage=completion_stage,
            cancel_stage=cancel_stage,
            outcome_class=outcome_class,
            recovery=recovery,
            retryable=retryable,
            recovery_reserve_ms=reserved_ms,
            physical_attempt_budget_ms=physical_attempt_budget_ms,
            provider_request_id=provider_request_id,
            provider_request_id_available=provider_id_available,
            provider_request_id_unavailable_reason=provider_id_unavailable_reason,
            token_timing_unavailable_reason=(
                timing_unavailable_reason
            ),
            request_deadline_version=REQUEST_DEADLINE_VERSION,
        )
        degradation = _degradation(
            node_id=node_id,
            route=resolved_route,
            reason_code=reason_code,
            core_path=core_path,
            duration_ms=queue_wait_ms + provider_ms,
            error_type=error_type,
        )
        attempts.append(attempt)
        degradations.append(degradation)
        _emit(runtime, {"llm_attempt": attempt})
        _emit(runtime, {"degradation": degradation})

        if retryable:
            # A tiny bounded backoff avoids immediate provider hammering while
            # never consuming the recovery tail itself.
            delay = min(0.1 * (2**retry_index), 0.25)
            if (
                _remaining_ms(deadline)
                > effective_recovery_min_ms + int(delay * 1000)
            ):
                await asyncio.sleep(delay)
            continue
        raise LLMInvocationError(
            node_id=node_id,
            route=resolved_route,
            reason_code=reason_code,
            error_type=error_type,
            timeout_budget_ms=timeout_budget_ms,
            attempts=attempts,
            degradations=degradations,
        ) from error

    raise pre_invoke_failure(
        node_id=node_id,
        route=resolved_route,
        timeout_seconds=total_timeout,
        core_path=core_path,
        runtime=runtime,
        route_catalog=route_catalog,
        error=RuntimeError("invocation_loop_exhausted"),
    )


async def invoke_optional_llm(
    runnable: Any,
    payload: Any,
    *,
    node_id: str,
    fallback: Any,
    route: str | None = None,
    timeout_seconds: float | None = None,
    max_retries: int | None = None,
    core_path: bool = False,
    allow_empty: bool = False,
    config: dict[str, Any] | None = None,
    runtime: Any = None,
    request_deadline_at: float | None = None,
    recovery_payload: Any = None,
    recovery_enabled: bool = True,
    output_validator: Callable[[Any], str | None] | None = None,
    route_catalog: LLMRouteCatalog | None = None,
) -> LLMInvocationResult:
    """Run an optional node while preserving its complete failure evidence."""

    try:
        return await invoke_llm(
            runnable,
            payload,
            node_id=node_id,
            route=route,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            core_path=core_path,
            allow_empty=allow_empty,
            config=config,
            runtime=runtime,
            request_deadline_at=request_deadline_at,
            recovery_payload=recovery_payload,
            recovery_enabled=recovery_enabled,
            output_validator=output_validator,
            route_catalog=route_catalog,
        )
    except LLMInvocationError as exc:
        return LLMInvocationResult(
            value=copy.deepcopy(fallback),
            attempts=exc.attempts,
            degradations=exc.degradations,
        )
