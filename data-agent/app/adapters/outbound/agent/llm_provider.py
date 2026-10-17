"""由 Bootstrap 选择的 LangChain 模型路由与客户端工厂。"""

import asyncio
import hashlib
import json
import os
import re
from builtins import ExceptionGroup
from threading import RLock
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from langchain.chat_models import init_chat_model

from app.config.app_config import app_config
from app.infrastructure.clients.llm.capabilities import (
    CAPABILITY_REGISTRY_VERSION,
    capability_for_model,
    provider_request_options as capability_request_options,
)

# 默认任务路由：当节点未显式声明任务类型时，按 SQL 生成任务对待
DEFAULT_TASK = "sql"

_DEFAULT_SENTINELS = {"", "__default__", "null", "none"}
_ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SERVICE_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_SERVICE_FINGERPRINT_HEADER = "X-DA-Expected-Service-Fingerprint"
_DIRECT_HTTPX_NETWORK_PROFILE = "httpx-trust-env-false-v1"
_CLIENT_CACHE: dict[str, Any] = {}
_CLIENT_HTTP_RESOURCES: dict[
    str,
    tuple[httpx.Client, httpx.AsyncClient],
] = {}
_PENDING_HTTP_CLOSE_TASKS: set[asyncio.Task[None]] = set()
_CLIENT_CACHE_LOCK = RLock()

# 节点 -> 任务路由表：把每个流水线节点映射到对应的任务类型，
# 不同任务可走不同模型/超时/重试/输出预算（见 app_config.llm.routes）。
# 设计意图：按节点职责分流，避免所有节点共用一个粗放配置。
#   - keyword：召回类节点（关键词扩词），调用频繁、输出短，可用快模型
#   - selection：筛选类节点（表/指标过滤），需结构化判断
#   - planning：SQL 规划，需推理能力
#   - sql：SQL 生成主任务，质量优先
#   - correction：SQL 校正，需纠错能力
#   - candidate：候选判定（TODO-45），需置信度评估
#   - postprocess：图表/解读后处理
#   - recommend：推荐问题生成
NODE_TASK_ROUTES = {
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
    """把配置占位符归一化为空串，避免把 ``null`` 当成真实地址或参数。"""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in _DEFAULT_SENTINELS else text


def _fingerprint(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_base_url(value: Any) -> str | None:
    """Normalize a provider endpoint and reject credential-bearing URLs."""
    text = _optional_text(value)
    if not text:
        return None
    if any(character.isspace() for character in text):
        raise ValueError("LLM base_url 不能包含空白字符")
    parsed = urlsplit(text)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("LLM base_url 必须是完整的 http/https 地址")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("LLM base_url 禁止携带用户名或密码")
    if parsed.query or parsed.fragment:
        raise ValueError("LLM base_url 禁止携带 query 或 fragment")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("LLM base_url 端口无效") from exc
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, path, "", ""))


def _validate_env_name(value: Any) -> str:
    env_name = _optional_text(value) or "LLM_API_KEY"
    if not _ENV_NAME_PATTERN.fullmatch(env_name):
        raise ValueError("LLM api_key_env 必须是合法的环境变量名")
    return env_name


def _normalize_service_fingerprint(value: Any) -> str:
    """规范化非敏感服务指纹，缺失时只能显式记录为 unavailable。"""
    fingerprint = _optional_text(value) or "unavailable"
    if fingerprint == "unavailable":
        return fingerprint
    if not _SERVICE_FINGERPRINT_PATTERN.fullmatch(fingerprint):
        raise ValueError(
            "LLM service_fingerprint 必须是 64 位小写十六进制或 unavailable"
        )
    return fingerprint


def _resolved_network_profile(value: Any, *, model_provider: str) -> str:
    """记录 DNS/CA 策略，并明确 OpenAI-compatible 请求不使用系统代理。"""
    configured = _optional_text(value) or "system-default"
    if model_provider != "openai":
        return configured
    components = configured.split("+")
    if _DIRECT_HTTPX_NETWORK_PROFILE not in components:
        components.append(_DIRECT_HTTPX_NETWORK_PROFILE)
    return "+".join(components)


def _validate_sealed_route_transport(profile: dict[str, Any]) -> None:
    """封存服务只允许 HTTPS，或通过本机 loopback HTTP 隧道访问。"""
    if str(profile.get("model_provider") or "").lower() != "openai":
        return
    if _normalize_service_fingerprint(
        profile.get("service_fingerprint")
    ) == "unavailable":
        return
    base_url = _normalize_base_url(profile.get("base_url"))
    if not base_url:
        raise ValueError("封存 LLM route 配置 service_fingerprint 时必须提供 base_url")
    parsed = urlsplit(base_url)
    loopback_http = (
        parsed.scheme.lower() == "http"
        and str(parsed.hostname or "").lower() in {"127.0.0.1", "localhost", "::1"}
    )
    if parsed.scheme.lower() != "https" and not loopback_http:
        raise ValueError("封存 LLM route 只允许 HTTPS 或 loopback HTTP base_url")


def _validate_profile_identity(profile: dict[str, Any]) -> None:
    """Fail fast on provider/adapter/access-mode combinations that cannot work."""
    provider = str(profile.get("provider") or "").lower()
    model_provider = str(profile.get("model_provider") or "").lower()
    model_name = str(profile.get("model_name") or "").lower()
    access_mode = str(profile.get("access_mode") or "standard").lower()
    if model_provider == "openai" and provider != "openai" and not profile.get("base_url"):
        raise ValueError(
            f"LLM provider={provider} 使用 OpenAI-compatible 适配器时必须配置 base_url"
        )
    if model_name.startswith("qwen") and model_provider != "openai":
        raise ValueError("Qwen 百炼模型必须使用 model_provider=openai 的兼容适配器")
    if model_name.startswith("qwen3.8-max-preview") and access_mode != "token_plan":
        raise ValueError(
            "qwen3.8-max-preview 仅允许 access_mode=token_plan，"
            "不能使用普通百炼凭证或端点"
        )
    registered = capability_for_model(model_name)
    hosted_deepseek = bool(
        model_name.startswith("deepseek")
        and provider != "deepseek"
        and model_provider == "openai"
    )
    if registered and not hosted_deepseek:
        identity = (
            str(registered.get("provider") or "").lower(),
            str(registered.get("model_provider") or "").lower(),
            str(registered.get("access_mode") or "standard").lower(),
        )
        actual = (provider, model_provider, access_mode)
        if actual != identity:
            raise ValueError(
                "已注册模型的 provider/adapter/access_mode 与能力合同不一致"
            )
    _validate_sealed_route_transport(profile)


def _resolve_api_key(profile: dict[str, Any]) -> str:
    """按安全配置中的环境变量名解析密钥，禁止回退到其他供应商凭证。"""
    env_name = _validate_env_name(profile.get("api_key_env"))
    value = str(os.getenv(env_name) or "").strip()
    global_env = (
        _optional_text(getattr(app_config.llm, "api_key_env", None))
        or "LLM_API_KEY"
    )
    if not value and env_name == global_env:
        value = str(app_config.llm.api_key or "").strip()
    if not value or (value.startswith("<") and value.endswith(">")):
        raise RuntimeError(
            f"LLM 路由 {profile.get('task')} 缺少环境变量 {env_name}；"
            "不会回退使用其他供应商的 API Key"
        )
    return value


def task_profile(task: str | None) -> dict[str, Any]:
    """解析某个任务的内部路由配置，并带全局兜底。

    核心实现思路：
        以全局 LLMConfig 为兜底，按任务名从 app_config.llm.routes 中读取覆盖项，
        逐字段合并后返回一个“已解析、非敏感”的配置快照。

    业务逻辑：
        1. 任务名归一化：None/空串回退到 DEFAULT_TASK。
        2. 读取该任务在 routes 中的覆盖配置（可能为空）。
        3. model_name/provider 等字段为 "__default__" 时回退全局或 provider 配置。
        4. API Key 只保留环境变量名，密钥值在构造客户端时延迟解析。
        5. 逐字段合并超时、重试、输出预算、思考模式与供应商参数。
    """
    task_name = str(task or DEFAULT_TASK)
    # 取该任务的覆盖配置；routes 未配置该任务时得到空 dict，全部回退全局
    configured = dict((app_config.llm.routes or {}).get(task_name) or {})
    configured_model = _optional_text(configured.get("model_name"))
    provider = (
        _optional_text(configured.get("provider"))
        or _optional_text(getattr(app_config.llm, "provider", None))
        or "deepseek"
    ).lower()
    providers = dict(getattr(app_config.llm, "providers", {}) or {})
    provider_config = dict(providers.get(provider) or {})
    model_provider = (
        _optional_text(configured.get("model_provider"))
        or _optional_text(provider_config.get("model_provider"))
        or provider
    ).lower()
    base_url = _normalize_base_url(
        _optional_text(configured.get("base_url"))
        or _optional_text(provider_config.get("base_url"))
        or _optional_text(getattr(app_config.llm, "base_url", None))
    )
    api_key_env = _validate_env_name(
        _optional_text(configured.get("api_key_env"))
        or _optional_text(provider_config.get("api_key_env"))
        or _optional_text(getattr(app_config.llm, "api_key_env", None))
        or "LLM_API_KEY"
    )
    access_mode = (
        _optional_text(configured.get("access_mode"))
        or _optional_text(provider_config.get("access_mode"))
        or "standard"
    ).lower()
    thinking_mode = str(
        configured.get("thinking_mode")
        or getattr(app_config.llm, "thinking_mode", "auto")
        or "auto"
    ).strip().lower()
    if thinking_mode not in {"auto", "enabled", "disabled"}:
        thinking_mode = "auto"
    reasoning_effort = (
        _optional_text(configured.get("reasoning_effort"))
        or _optional_text(getattr(app_config.llm, "reasoning_effort", None))
    ).lower()
    configured_temperature = configured.get("temperature")
    temperature = float(
        configured_temperature
        if _optional_text(configured_temperature)
        else getattr(app_config.llm, "temperature", 0.0)
    )
    model_name = configured_model or app_config.llm.model_name
    capability = capability_for_model(model_name)
    if capability and (
        str(capability.get("provider") or "").lower() != provider
        or str(capability.get("model_provider") or "").lower() != model_provider
        or str(capability.get("access_mode") or "standard").lower() != access_mode
    ):
        capability = None
    profile = {
        "task": task_name,
        # 模型名：任务覆盖优先，否则回退全局默认模型
        "model_name": model_name,
        # provider 是部署供应商身份；model_provider 是 LangChain 适配器。
        "provider": provider,
        "model_provider": model_provider,
        "base_url": base_url,
        "access_mode": access_mode,
        # 内部 profile 只记录变量名；公开清单还会将它哈希化。
        "api_key_env": api_key_env,
        # 超时：任务覆盖优先，缺失回退全局；统一转 float 便于下游比较
        "timeout_seconds": float(configured.get("timeout_seconds") or app_config.llm.timeout_seconds),
        # 重试：用 `is None` 判断而非 `or`，避免配置为 0 被误判为缺失而回退全局
        "max_retries": int(configured.get("max_retries") if configured.get("max_retries") is not None else app_config.llm.max_retries),
        # 输出预算：任务覆盖优先，缺失回退全局
        "max_completion_tokens": int(configured.get("max_completion_tokens") or app_config.llm.max_completion_tokens),
        # DeepSeek V4 的 reasoning 与最终 content 共享输出预算。SQL、纠错等
        # 结构化任务默认关闭 thinking，避免 reasoning 用尽预算后返回空正文。
        "thinking_mode": thinking_mode,
        "reasoning_effort": reasoning_effort or None,
        "max_concurrency": max(
            1,
            int(
                configured.get("max_concurrency")
                or app_config.query.llm_route_default_concurrency
            ),
        ),
        # 默认仍为 0；thinking-only 模型必须按供应商合同显式覆盖。
        "temperature": temperature,
        "candidate_id": (
            _optional_text(configured.get("candidate_id"))
            or str((capability or {}).get("candidate_id") or "unregistered")
        ),
        "model_revision": (
            _optional_text(configured.get("model_revision"))
            or str((capability or {}).get("model_revision") or "unavailable")
        ),
        "endpoint_profile": (
            _optional_text(configured.get("endpoint_profile"))
            or _optional_text(provider_config.get("endpoint_profile"))
            or str((capability or {}).get("endpoint_profile") or provider)
        ),
        "adapter_version": (
            _optional_text(configured.get("adapter_version"))
            or _optional_text(provider_config.get("adapter_version"))
            or str((capability or {}).get("adapter_version") or "generic-v1")
        ),
        # 该值由部署侧发布并通过环境变量注入，不从 API Key 或远端响应推导。
        "service_fingerprint": _normalize_service_fingerprint(
            configured.get("service_fingerprint")
        ),
        "capability_registry_version": CAPABILITY_REGISTRY_VERSION,
        "capability_lifecycle": str(
            (capability or {}).get("lifecycle") or "unregistered"
        ),
        "comparison_mode": str(
            (capability or {}).get("comparison_mode") or "unregistered"
        ),
        "promotion_eligible": bool(
            (capability or {}).get("promotion_eligible", False)
        ),
        "network_profile": _resolved_network_profile(
            configured.get("network_profile"),
            model_provider=model_provider,
        ),
    }
    _validate_profile_identity(profile)
    return profile


def public_task_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Return a report-safe profile without endpoint or credential-source values."""
    service_fingerprint = _normalize_service_fingerprint(
        profile.get("service_fingerprint")
    )
    network_profile = _resolved_network_profile(
        profile.get("network_profile"),
        model_provider=str(profile.get("model_provider") or "").lower(),
    )
    endpoint_material = str(profile.get("base_url") or (
        f"adapter-default:{profile.get('model_provider') or profile.get('provider') or ''}"
    ))
    endpoint_id = _fingerprint(endpoint_material)
    deployment_material = json.dumps(
        {
            "provider": profile.get("provider"),
            "model_provider": profile.get("model_provider"),
            "model_name": profile.get("model_name"),
            "candidate_id": profile.get("candidate_id"),
            "model_revision": profile.get("model_revision"),
            "adapter_version": profile.get("adapter_version"),
            "service_fingerprint": service_fingerprint,
            "network_profile": network_profile,
            "endpoint_id": endpoint_id,
            "endpoint_profile": profile.get("endpoint_profile"),
            # 只公开通用来源类别；具体环境变量名不进入任何 artifact/hash。
            "credential_source": "environment",
            "access_mode": profile.get("access_mode"),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    safe = {
        key: value
        for key, value in profile.items()
        if key not in {"base_url", "api_key_env"}
    }
    safe.update({
        "endpoint_id": endpoint_id,
        "deployment_id": _fingerprint(deployment_material),
        "credential_source": "environment",
        "service_fingerprint": service_fingerprint,
        "network_profile": network_profile,
    })
    return safe


def _provider_request_options(profile: dict[str, Any]) -> dict[str, Any]:
    """Build and validate provider-specific request options without reading a key."""
    _validate_profile_identity(profile)
    model_name = str(profile.get("model_name") or "").lower()
    provider = str(profile.get("provider") or "").lower()
    model_provider = str(profile.get("model_provider") or "").lower()
    thinking_mode = str(profile.get("thinking_mode") or "auto").lower()
    reasoning_effort = str(profile.get("reasoning_effort") or "").lower()
    temperature = float(profile.get("temperature") or 0.0)
    extra_body: dict[str, Any] = {}
    options: dict[str, Any] = {}

    # Registered candidates are validated and mapped in one versioned source of
    # truth.  Generic branches remain for existing non-TODO-97 routes.
    registered = capability_for_model(model_name)
    hosted_deepseek = bool(
        model_name.startswith("deepseek")
        and provider != "deepseek"
        and model_provider == "openai"
    )
    if registered and not hosted_deepseek and all((
        str(registered.get("provider") or "").lower() == provider,
        str(registered.get("model_provider") or "").lower() == model_provider,
        str(registered.get("access_mode") or "standard").lower()
        == str(profile.get("access_mode") or "standard").lower(),
    )):
        return capability_request_options(profile)

    if model_name.startswith("qwen"):
        if reasoning_effort:
            raise ValueError("非 qwen3.8-max 的 Qwen 路由不接受 reasoning_effort")
        if thinking_mode == "enabled" and temperature < 0.6:
            raise ValueError("Qwen thinking 模式 temperature 不能低于 0.6")
        if thinking_mode in {"enabled", "disabled"}:
            extra_body["enable_thinking"] = thinking_mode == "enabled"
    elif model_name.startswith("deepseek") and model_provider == "openai" and provider != "openai":
        # 百炼托管 DeepSeek 使用 DashScope OpenAI-compatible 合同。
        if thinking_mode in {"enabled", "disabled"}:
            extra_body["enable_thinking"] = thinking_mode == "enabled"
        if reasoning_effort:
            if thinking_mode == "disabled":
                raise ValueError("百炼 DeepSeek 关闭 thinking 时不能配置 reasoning_effort")
            if reasoning_effort not in {"low", "medium", "high", "xhigh", "max"}:
                raise ValueError("百炼 DeepSeek reasoning_effort 参数无效")
            extra_body["reasoning_effort"] = reasoning_effort
    elif model_name.startswith("deepseek") and model_provider == "deepseek":
        if reasoning_effort:
            if thinking_mode == "disabled":
                raise ValueError("原生 DeepSeek 关闭 thinking 时不能配置 reasoning_effort")
            if reasoning_effort not in {"low", "high", "xhigh", "max"}:
                raise ValueError("原生 DeepSeek reasoning_effort 参数无效")
            # DeepSeek OpenAI-format 把 effort 定义为标准顶层参数。
            options["reasoning_effort"] = reasoning_effort
        if thinking_mode in {"enabled", "disabled"}:
            extra_body["thinking"] = {"type": thinking_mode}

    if extra_body:
        options["extra_body"] = extra_body
    return options


def task_model_kwargs(profile: dict[str, Any]) -> dict[str, Any]:
    """Build provider kwargs without leaking credentials into manifests/tests."""
    provider_options = _provider_request_options(profile)
    kwargs: dict[str, Any] = {
        "model": profile["model_name"],
        "model_provider": profile["model_provider"],
        "temperature": profile["temperature"],
        "timeout": profile["timeout_seconds"],
        # Provider-internal retries are opaque.  The unified invocation
        # boundary owns the one bounded recovery and records every attempt.
        "max_retries": 0,
    }
    if str(profile.get("model_provider") or "").lower() == "openai":
        kwargs["max_completion_tokens"] = profile["max_completion_tokens"]
        service_fingerprint = _normalize_service_fingerprint(
            profile.get("service_fingerprint")
        )
        if service_fingerprint != "unavailable":
            # 由客户端固定到每个 OpenAI-compatible 请求，调用 payload 无覆盖入口。
            kwargs["default_headers"] = {
                _EXPECTED_SERVICE_FINGERPRINT_HEADER: service_fingerprint,
            }
    else:
        kwargs["max_tokens"] = profile["max_completion_tokens"]
    if profile.get("base_url"):
        kwargs["base_url"] = profile["base_url"]
    kwargs.update(provider_options)
    # Invalid capability/provider combinations are rejected before reading a secret.
    kwargs["api_key"] = _resolve_api_key(profile)
    return kwargs


def _client_cache_identity(
    profile: dict[str, Any],
    request_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """生成不含凭据和运行时 client 对象的客户端缓存身份材料。"""
    safe_request_kwargs = {
        key: value
        for key, value in request_kwargs.items()
        if key not in {"http_client", "http_async_client", "api_key"}
    }
    return {
        "request_kwargs": safe_request_kwargs,
        "task": profile.get("task"),
        "provider": profile.get("provider"),
        "access_mode": profile.get("access_mode"),
        "candidate_id": profile.get("candidate_id"),
        "model_revision": profile.get("model_revision"),
        "endpoint_profile": profile.get("endpoint_profile"),
        "adapter_version": profile.get("adapter_version"),
        "service_fingerprint": _normalize_service_fingerprint(
            profile.get("service_fingerprint")
        ),
        "capability_registry_version": profile.get("capability_registry_version"),
        "network_profile": _resolved_network_profile(
            profile.get("network_profile"),
            model_provider=str(profile.get("model_provider") or "").lower(),
        ),
    }


def _new_openai_http_clients() -> tuple[httpx.Client, httpx.AsyncClient]:
    """创建显式绕过 HTTP(S)_PROXY 的同步与异步传输。"""
    sync_client = httpx.Client(trust_env=False)
    try:
        async_client = httpx.AsyncClient(trust_env=False)
    except BaseException:
        sync_client.close()
        raise
    return sync_client, async_client


async def _close_http_resources(
    resources: tuple[tuple[httpx.Client, httpx.AsyncClient], ...],
    *,
    raise_errors: bool,
) -> None:
    """关闭全部 LLM HTTP 资源；单个失败不阻止其余资源回收。"""
    errors: list[Exception] = []
    for sync_client, async_client in resources:
        try:
            sync_client.close()
        except Exception as exc:  # noqa: BLE001 - 关闭阶段必须继续回收
            errors.append(exc)
        try:
            await async_client.aclose()
        except Exception as exc:  # noqa: BLE001 - 关闭阶段必须继续回收
            errors.append(exc)
    if errors and raise_errors:
        raise ExceptionGroup("关闭 LLM HTTP client 失败", errors)


def _take_cached_http_resources() -> tuple[
    tuple[httpx.Client, httpx.AsyncClient],
    ...,
]:
    with _CLIENT_CACHE_LOCK:
        _CLIENT_CACHE.clear()
        resources = tuple(_CLIENT_HTTP_RESOURCES.values())
        _CLIENT_HTTP_RESOURCES.clear()
    return resources


def _schedule_http_resource_close(
    resources: tuple[tuple[httpx.Client, httpx.AsyncClient], ...],
) -> None:
    if not resources:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_close_http_resources(resources, raise_errors=True))
        return
    task = loop.create_task(_close_http_resources(resources, raise_errors=False))
    _PENDING_HTTP_CLOSE_TASKS.add(task)
    task.add_done_callback(_PENDING_HTTP_CLOSE_TASKS.discard)


def get_task_llm(task: str = DEFAULT_TASK):
    """按完整部署与请求规格构造并缓存 LangChain Chat 模型实例。

    核心实现思路：
        缓存键覆盖 provider、endpoint、模型、参数和凭据指纹。轮换 Key 或变更
        路由时不会继续复用旧客户端，密钥本身不进入缓存键或日志。

    业务逻辑：
        1. 解析任务路由配置得到 profile；
        2. 用 profile 中的 model/timeout/max_retries/max_tokens 构造模型；
        3. API Key 按 profile.api_key_env 从环境变量延迟解析，安全清单不含密钥；
        4. 完全相同的部署规格复用实例，不同任务若规格相同也可安全共享。
    """
    profile = task_profile(task)
    kwargs = task_model_kwargs(profile)
    secret = str(kwargs.pop("api_key"))
    cache_identity = _client_cache_identity(profile, kwargs)
    serialized = json.dumps(
        cache_identity,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    cache_key = _fingerprint(f"{serialized}\0{_fingerprint(secret)}")
    with _CLIENT_CACHE_LOCK:
        cached = _CLIENT_CACHE.get(cache_key)
        if cached is None:
            http_resources = None
            init_kwargs = dict(kwargs)
            if str(profile.get("model_provider") or "").lower() == "openai":
                http_resources = _new_openai_http_clients()
                init_kwargs.update(
                    {
                        "http_client": http_resources[0],
                        "http_async_client": http_resources[1],
                    }
                )
            try:
                cached = init_chat_model(api_key=secret, **init_kwargs)
            except BaseException:
                if http_resources is not None:
                    _schedule_http_resource_close((http_resources,))
                raise
            _CLIENT_CACHE[cache_key] = cached
            if http_resources is not None:
                _CLIENT_HTTP_RESOURCES[cache_key] = http_resources
        return cached


def clear_task_llm_cache() -> None:
    """配置切换后清理 client；活动事件循环中异步完成传输关闭。"""
    _schedule_http_resource_close(_take_cached_http_resources())


async def close_task_llm_cache() -> None:
    """在应用生命周期关闭阶段确定性回收所有 LLM HTTP client。"""
    resources = _take_cached_http_resources()
    pending = tuple(_PENDING_HTTP_CLOSE_TASKS)
    if pending:
        await asyncio.gather(*pending)
    await _close_http_resources(resources, raise_errors=True)


def task_timeout(task: str) -> float:
    """返回某任务的调用超时（秒），供 optional_dependency 等限时逻辑使用。"""
    return float(task_profile(task)["timeout_seconds"])


def route_for_node(node_id: str | None) -> str:
    """把节点 id 翻译成任务路由名；未知节点回退到 DEFAULT_TASK。"""
    return NODE_TASK_ROUTES.get(str(node_id or ""), DEFAULT_TASK)


def llm_route_manifest() -> dict[str, dict[str, Any]]:
    """生成指纹安全的路由清单：包含所有已知任务 + 用户自定义任务的 profile，剔除密钥。

    核心实现思路：
        汇总“默认任务 + 路由表所有任务 + 用户在 routes 中自定义的任务”三类任务名，
        去重排序后逐个生成 profile，便于启动期/可观测性展示当前生效的路由全景。

    业务逻辑：
        1. 用集合并集合并三类任务名：DEFAULT_TASK、NODE_TASK_ROUTES.values()、routes.keys()；
        2. 排序后生成安全快照，endpoint 与凭据来源均只保留不可逆指纹；
        3. 返回 {task: profile} 字典，供健康检查/审计展示。
    """
    tasks = sorted({DEFAULT_TASK, *NODE_TASK_ROUTES.values(), *(app_config.llm.routes or {}).keys()})
    return {task: public_task_profile(task_profile(task)) for task in tasks}
