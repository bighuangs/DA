import copy
import json
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from app.adapters.outbound.agent.llm_provider import (
    _client_cache_identity,
    _normalize_service_fingerprint,
    close_task_llm_cache,
    clear_task_llm_cache,
    get_task_llm,
    public_task_profile,
    task_model_kwargs,
    task_profile,
)
from app.config.app_config import app_config
from evals.legacy import governance


BASE_SERVICE_FINGERPRINT = "0" * 64
LORA_SERVICE_FINGERPRINT = "1" * 64
EXPECTED_SERVICE_FINGERPRINT_HEADER = "X-DA-Expected-Service-Fingerprint"


@pytest.fixture(autouse=True)
def _isolated_client_cache(monkeypatch):
    monkeypatch.setenv("AUTODL_VLLM_API_KEY", "synthetic-not-a-real-key")
    clear_task_llm_cache()
    yield
    clear_task_llm_cache()


def _sealed_profile(**overrides):
    profile = {
        "task": "sql",
        "model_name": "da-nl2sql-qwen3-8b",
        "provider": "dashscope",
        "model_provider": "openai",
        "base_url": "https://sealed-eval.example.test/v1",
        "access_mode": "standard",
        "api_key_env": "AUTODL_VLLM_API_KEY",
        "timeout_seconds": 22.0,
        "max_retries": 1,
        "max_completion_tokens": 2048,
        "thinking_mode": "disabled",
        "reasoning_effort": None,
        "max_concurrency": 6,
        "temperature": 0.0,
        "candidate_id": "qwen3-8b-base",
        "model_revision": "26028140be3ee69b82b1d1450179ab71bb1121b9",
        "endpoint_profile": "autodl-vllm-gateway-v1",
        "adapter_version": "base-no-adapter",
        "service_fingerprint": BASE_SERVICE_FINGERPRINT,
        "capability_registry_version": "test-registry-v1",
        "capability_lifecycle": "sealed-eval",
        "comparison_mode": "base-vs-lora",
        "promotion_eligible": False,
        "network_profile": "autodl-ssh-tunnel-v1",
    }
    profile.update(overrides)
    return profile


def test_sql_route_maps_all_sealed_identity_environment_variables(monkeypatch):
    monkeypatch.setenv("LLM_SQL_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("LLM_SQL_CANDIDATE_ID", "qwen3-8b-lora-release-v1")
    monkeypatch.setenv(
        "LLM_SQL_MODEL_REVISION",
        "26028140be3ee69b82b1d1450179ab71bb1121b9",
    )
    monkeypatch.setenv("LLM_SQL_ADAPTER_VERSION", "adapter-release-v1")
    monkeypatch.setenv("LLM_SQL_SERVICE_FINGERPRINT", LORA_SERVICE_FINGERPRINT)

    config = OmegaConf.load("conf/app_config.yaml")
    sql_route = OmegaConf.to_container(config.llm.routes.sql, resolve=True)

    assert sql_route["timeout_seconds"] == "120"
    assert sql_route["candidate_id"] == "qwen3-8b-lora-release-v1"
    assert sql_route["model_revision"] == (
        "26028140be3ee69b82b1d1450179ab71bb1121b9"
    )
    assert sql_route["adapter_version"] == "adapter-release-v1"
    assert sql_route["service_fingerprint"] == LORA_SERVICE_FINGERPRINT


def test_task_profile_records_identity_without_resolving_service_credentials(
    monkeypatch,
):
    routes = copy.deepcopy(app_config.llm.routes)
    routes["sql"] = {
        **dict(routes["sql"]),
        "model_name": "qwen3-8b-eval",
        "provider": "dashscope",
        "base_url": "https://sealed-eval.example.test/v1",
        "api_key_env": "MISSING_SEALED_EVAL_KEY",
        "candidate_id": "qwen3-8b-lora-release-v1",
        "model_revision": "26028140be3ee69b82b1d1450179ab71bb1121b9",
        "adapter_version": "adapter-release-v1",
        "service_fingerprint": LORA_SERVICE_FINGERPRINT,
    }
    monkeypatch.setattr(app_config.llm, "routes", routes)
    monkeypatch.delenv("MISSING_SEALED_EVAL_KEY", raising=False)

    profile = task_profile("sql")
    public = public_task_profile(profile)

    assert profile["candidate_id"] == "qwen3-8b-lora-release-v1"
    assert profile["model_revision"] == (
        "26028140be3ee69b82b1d1450179ab71bb1121b9"
    )
    assert profile["adapter_version"] == "adapter-release-v1"
    assert profile["service_fingerprint"] == LORA_SERVICE_FINGERPRINT
    assert profile["network_profile"].endswith("+httpx-trust-env-false-v1")
    assert "api_key_env" not in public
    assert "base_url" not in public


@pytest.mark.parametrize(
    "value",
    [
        "A" * 64,
        "sha256:" + "a" * 64,
        "a" * 63,
        "a" * 65,
        "not-a-fingerprint",
    ],
)
def test_service_fingerprint_rejects_noncanonical_values(value):
    with pytest.raises(ValueError, match="64 位小写十六进制"):
        _normalize_service_fingerprint(value)


def test_service_fingerprint_accepts_digest_or_explicit_unavailable():
    assert _normalize_service_fingerprint(BASE_SERVICE_FINGERPRINT) == (
        BASE_SERVICE_FINGERPRINT
    )
    assert _normalize_service_fingerprint("unavailable") == "unavailable"
    assert _normalize_service_fingerprint("") == "unavailable"


def test_public_manifest_and_cache_identity_fail_closed_on_invalid_fingerprint():
    invalid = _sealed_profile(service_fingerprint="SHA256:not-canonical")

    with pytest.raises(ValueError, match="64 位小写十六进制"):
        public_task_profile(invalid)
    with pytest.raises(ValueError, match="64 位小写十六进制"):
        _client_cache_identity(invalid, {"model": invalid["model_name"]})


def test_openai_compatible_client_injects_declared_fingerprint_only():
    profile = _sealed_profile(
        default_headers={
            EXPECTED_SERVICE_FINGERPRINT_HEADER: LORA_SERVICE_FINGERPRINT,
            "Authorization": "payload-must-not-control-client-headers",
        },
    )

    kwargs = task_model_kwargs(profile)

    assert kwargs["default_headers"] == {
        EXPECTED_SERVICE_FINGERPRINT_HEADER: BASE_SERVICE_FINGERPRINT,
    }
    assert "payload-must-not-control-client-headers" not in str(kwargs)


def test_unavailable_service_fingerprint_does_not_emit_expected_header():
    kwargs = task_model_kwargs(
        _sealed_profile(service_fingerprint="unavailable"),
    )

    assert "default_headers" not in kwargs


@pytest.mark.parametrize(
    "base_url",
    [
        "http://public.example.test/v1",
        "http://10.0.0.8:6008/v1",
        "http://192.168.1.8:6008/v1",
        None,
    ],
)
def test_sealed_openai_route_rejects_missing_or_public_plaintext_endpoint(base_url):
    with pytest.raises(ValueError, match="base_url|HTTPS|loopback"):
        task_model_kwargs(_sealed_profile(base_url=base_url))


@pytest.mark.parametrize(
    "base_url",
    [
        "http://127.0.0.1:6008/v1",
        "http://localhost:6008/v1",
        "http://[::1]:6008/v1",
        "https://sealed-eval.example.test/v1",
    ],
)
def test_sealed_openai_route_accepts_https_or_exact_loopback_http(base_url):
    kwargs = task_model_kwargs(_sealed_profile(base_url=base_url))

    assert kwargs["base_url"] == base_url


@pytest.mark.parametrize(
    "base_url",
    [
        "https://user:password@sealed-eval.example.test/v1",
        "https://sealed-eval.example.test/v1?token=secret",
        "https://sealed-eval.example.test/v1#secret",
    ],
)
def test_sealed_openai_route_reuses_strict_base_url_validation(base_url):
    with pytest.raises(ValueError):
        task_model_kwargs(_sealed_profile(base_url=base_url))


@pytest.mark.asyncio
async def test_formal_clients_ignore_proxy_environment_and_close_both_paths(
    monkeypatch,
):
    profile = _sealed_profile(
        base_url="http://127.0.0.1:6008/v1",
        network_profile=(
            "autodl-ssh-tunnel-v1+httpx-trust-env-false-v1"
        ),
    )
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example.test:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.test:8443")
    monkeypatch.setattr(
        "app.adapters.outbound.agent.llm_provider.task_profile",
        lambda _task: dict(profile),
    )

    model = get_task_llm("sql")
    sync_client = model.http_client
    async_client = model.http_async_client

    assert sync_client._trust_env is False
    assert async_client._trust_env is False
    assert model.root_client._client is sync_client
    assert model.root_async_client._client is async_client
    await close_task_llm_cache()
    assert sync_client.is_closed
    assert async_client.is_closed


def test_cache_identity_records_network_policy_without_runtime_clients():
    profile = _sealed_profile(
        network_profile="system-default+httpx-trust-env-false-v1",
    )
    runtime_client = object()
    identity = _client_cache_identity(
        profile,
        {
            "model": profile["model_name"],
            "http_client": runtime_client,
            "http_async_client": runtime_client,
        },
    )
    other_identity = _client_cache_identity(
        {
            **profile,
            "network_profile": (
                "explicit-public-dns-system-ca-v1+httpx-trust-env-false-v1"
            ),
        },
        {"model": profile["model_name"]},
    )

    assert identity["network_profile"] == (
        "system-default+httpx-trust-env-false-v1"
    )
    assert "http_client" not in identity["request_kwargs"]
    assert "http_async_client" not in identity["request_kwargs"]
    assert identity != other_identity
    json.dumps(identity, sort_keys=True)


def test_service_fingerprint_changes_real_client_cache_identity(monkeypatch):
    profile = _sealed_profile()
    created = []

    def _fake_init_chat_model(**kwargs):
        client = Mock(name=f"sealed-client-{len(created)}")
        created.append((client, dict(kwargs)))
        return client

    monkeypatch.setattr(
        "app.adapters.outbound.agent.llm_provider.task_profile",
        lambda _task: dict(profile),
    )
    monkeypatch.setattr(
        "app.adapters.outbound.agent.llm_provider.init_chat_model",
        _fake_init_chat_model,
    )

    base_client = get_task_llm("sql")
    profile["service_fingerprint"] = LORA_SERVICE_FINGERPRINT
    lora_client = get_task_llm("sql")

    assert base_client is not lora_client
    assert created[0][1]["default_headers"] == {
        EXPECTED_SERVICE_FINGERPRINT_HEADER: BASE_SERVICE_FINGERPRINT,
    }
    assert created[1][1]["default_headers"] == {
        EXPECTED_SERVICE_FINGERPRINT_HEADER: LORA_SERVICE_FINGERPRINT,
    }
    request_kwargs = task_model_kwargs(profile)
    request_kwargs.pop("api_key")
    assert "synthetic-not-a-real-key" not in str(
        _client_cache_identity(profile, request_kwargs)
    )


def test_base_and_lora_identity_change_manifest_config_deployment_and_cache(
    monkeypatch,
):
    base = _sealed_profile()
    lora = _sealed_profile(
        candidate_id="qwen3-8b-lora-release-v1",
        adapter_version="adapter-release-v1",
        service_fingerprint=LORA_SERVICE_FINGERPRINT,
    )
    assert base["model_name"] == lora["model_name"]
    assert base["base_url"] == lora["base_url"]
    assert base["model_revision"] == lora["model_revision"]

    base_public = public_task_profile(base)
    lora_public = public_task_profile(lora)
    assert base_public != lora_public
    assert base_public["deployment_id"] != lora_public["deployment_id"]

    monkeypatch.setattr(
        governance,
        "llm_route_manifest",
        lambda: {"sql": base_public},
    )
    base_config = governance._safe_config_manifest()
    base_config_hash = governance.stable_hash(base_config)
    monkeypatch.setattr(
        governance,
        "llm_route_manifest",
        lambda: {"sql": lora_public},
    )
    lora_config = governance._safe_config_manifest()
    lora_config_hash = governance.stable_hash(lora_config)
    assert base_config_hash != lora_config_hash

    def _fingerprint(config, config_hash):
        fingerprint = {
            field: f"same-{field}"
            for field in governance.REQUIRED_FINGERPRINT_FIELDS
        }
        fingerprint.update(
            {
                "complete": True,
                "config_hash": config_hash,
                "model": config["llm"],
                "model_ablation": governance._model_ablation_manifest(config),
            }
        )
        return fingerprint

    attribution = governance.compare_model_ablation_fingerprints(
        _fingerprint(base_config, base_config_hash),
        _fingerprint(lora_config, lora_config_hash),
    )
    assert attribution["attribution_valid"]
    assert attribution["unexpected_differences"] == []
    assert "model.routes.sql.candidate_id" in attribution["allowed_differences"]
    assert "model.routes.sql.adapter_version" in attribution["allowed_differences"]
    assert (
        "model.routes.sql.service_fingerprint"
        in attribution["allowed_differences"]
    )
    assert "model.routes.sql.deployment_id" in attribution["allowed_differences"]

    request_kwargs = {
        "model": base["model_name"],
        "base_url": base["base_url"],
        "temperature": 0.0,
    }
    base_cache_identity = _client_cache_identity(base, request_kwargs)
    lora_cache_identity = _client_cache_identity(lora, request_kwargs)
    assert base_cache_identity != lora_cache_identity
    assert "api_key_env" not in base_cache_identity
    assert "AUTODL_VLLM_API_KEY" not in str(base_cache_identity)
