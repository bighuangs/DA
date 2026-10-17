from unittest.mock import Mock

import pytest

from app.adapters.outbound.agent.llm_provider import (
    _normalize_base_url,
    clear_task_llm_cache,
    get_task_llm,
    public_task_profile,
    task_model_kwargs,
    task_profile,
)
from app.config.app_config import LLMConfig


def _profile(**overrides):
    profile = {
        "task": "sql",
        "model_name": "deepseek-test",
        "provider": "deepseek",
        "model_provider": "deepseek",
        "base_url": None,
        "access_mode": "native",
        "api_key_env": "SYNTHETIC_LLM_KEY",
        "thinking_mode": "disabled",
        "reasoning_effort": None,
        "temperature": 0.0,
        "timeout_seconds": 22.0,
        "max_retries": 1,
        "max_completion_tokens": 2048,
        "max_concurrency": 6,
    }
    profile.update(overrides)
    return profile


@pytest.fixture(autouse=True)
def _synthetic_key(monkeypatch):
    monkeypatch.setenv("SYNTHETIC_LLM_KEY", "synthetic-not-a-real-key")
    clear_task_llm_cache()
    yield
    clear_task_llm_cache()


def test_sql_and_correction_disable_thinking_and_keep_bounded_ttl():
    sql = task_profile("sql")
    correction = task_profile("correction")

    assert sql["thinking_mode"] == "disabled"
    assert correction["thinking_mode"] == "disabled"
    assert sql["max_completion_tokens"] == 2048
    assert correction["max_completion_tokens"] == 2048
    assert sql["timeout_seconds"] == 22
    assert correction["timeout_seconds"] == 16


def test_native_deepseek_receives_native_thinking_switch():
    kwargs = task_model_kwargs(_profile())
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert kwargs["max_tokens"] == 2048
    assert "max_completion_tokens" not in kwargs


def test_native_deepseek_effort_is_a_top_level_openai_format_parameter():
    kwargs = task_model_kwargs(_profile(
        thinking_mode="enabled",
        reasoning_effort="high",
    ))
    assert kwargs["reasoning_effort"] == "high"
    assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}


def test_dashscope_qwen_uses_openai_compatible_contract():
    kwargs = task_model_kwargs(_profile(
        model_name="qwen3.7-plus",
        provider="dashscope",
        model_provider="openai",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        access_mode="standard",
    ))
    assert kwargs["extra_body"] == {"enable_thinking": False}
    assert kwargs["max_completion_tokens"] == 2048
    assert "max_tokens" not in kwargs


def test_dashscope_hosted_deepseek_uses_enable_thinking_not_native_shape():
    kwargs = task_model_kwargs(_profile(
        model_name="deepseek-v4-pro",
        provider="dashscope",
        model_provider="openai",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        access_mode="standard",
    ))
    assert kwargs["extra_body"] == {"enable_thinking": False}


def test_qwen38_formal_allows_non_thinking_and_disables_reasoning_history():
    kwargs = task_model_kwargs(_profile(
        model_name="qwen3.8-max",
        provider="dashscope",
        model_provider="openai",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        access_mode="standard",
        temperature=0.6,
    ))
    assert kwargs["extra_body"] == {
        "preserve_thinking": False,
        "enable_thinking": False,
    }


def test_qwen38_formal_thinking_puts_reasoning_effort_in_extra_body():
    kwargs = task_model_kwargs(_profile(
        model_name="qwen3.8-max",
        provider="dashscope",
        model_provider="openai",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        access_mode="standard",
        thinking_mode="enabled",
        reasoning_effort="medium",
        temperature=0.6,
    ))
    assert kwargs["extra_body"] == {
        "preserve_thinking": False,
        "enable_thinking": True,
        "reasoning_effort": "medium",
    }
    assert "reasoning_effort" not in {
        key: value for key, value in kwargs.items() if key != "extra_body"
    }


def test_qwen38_formal_rejects_silently_reset_temperature_before_key_lookup(monkeypatch):
    monkeypatch.delenv("MISSING_QWEN_KEY", raising=False)
    with pytest.raises(ValueError, match="temperature"):
        task_model_kwargs(_profile(
            model_name="qwen3.8-max",
            provider="dashscope",
            model_provider="openai",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            access_mode="standard",
            api_key_env="MISSING_QWEN_KEY",
            temperature=0.0,
        ))


def test_qwen38_preview_is_token_plan_thinking_only():
    kwargs = task_model_kwargs(_profile(
        model_name="qwen3.8-max-preview",
        provider="dashscope_token_plan",
        model_provider="openai",
        base_url="https://token-plan.example.test/compatible-mode/v1",
        access_mode="token_plan",
        thinking_mode="enabled",
        temperature=0.6,
    ))
    assert kwargs["extra_body"] == {"reasoning_effort": "xhigh"}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"thinking_mode": "disabled"}, "thinking-only"),
        ({"reasoning_effort": "high"}, "low/medium/xhigh"),
        ({"temperature": 0.59}, "temperature"),
    ],
)
def test_qwen38_preview_rejects_invalid_capability_contract(overrides, message):
    profile = _profile(
        model_name="qwen3.8-max-preview",
        provider="dashscope_token_plan",
        model_provider="openai",
        base_url="https://token-plan.example.test/compatible-mode/v1",
        access_mode="token_plan",
        thinking_mode="enabled",
        temperature=0.6,
    )
    profile.update(overrides)
    with pytest.raises(ValueError, match=message):
        task_model_kwargs(profile)


def test_preview_cannot_use_standard_bailian_access():
    with pytest.raises(ValueError, match="token_plan"):
        task_model_kwargs(_profile(
            model_name="qwen3.8-max-preview",
            provider="dashscope",
            model_provider="openai",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            access_mode="standard",
            thinking_mode="enabled",
            temperature=0.6,
        ))


@pytest.mark.parametrize("url", [
    "https://user:password@example.test/v1",
    "https://example.test/v1?api_key=secret",
    "https://example.test/v1#secret",
    "not-a-url",
])
def test_base_url_rejects_credential_bearing_or_invalid_values(url):
    with pytest.raises(ValueError):
        _normalize_base_url(url)


def test_provider_key_never_falls_back_to_another_provider(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "synthetic-deepseek-secret")
    monkeypatch.delenv("MISSING_DASHSCOPE_KEY", raising=False)
    profile = _profile(
        model_name="qwen3.7-plus",
        provider="dashscope",
        model_provider="openai",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        access_mode="standard",
        api_key_env="MISSING_DASHSCOPE_KEY",
    )
    with pytest.raises(RuntimeError) as exc_info:
        task_model_kwargs(profile)
    assert "MISSING_DASHSCOPE_KEY" in str(exc_info.value)
    assert "synthetic-deepseek-secret" not in str(exc_info.value)


def test_public_profile_and_llm_config_repr_redact_secrets_and_endpoints():
    raw = _profile(
        model_name="qwen3.7-plus",
        provider="dashscope",
        model_provider="openai",
        base_url="https://workspace-id.example.test/compatible-mode/v1",
        api_key_env="PRIVATE_DASHSCOPE_KEY_ENV",
    )
    public = public_task_profile(raw)
    rendered = str(public)
    assert "workspace-id" not in rendered
    assert "PRIVATE_DASHSCOPE_KEY_ENV" not in rendered
    assert "base_url" not in public
    assert "api_key_env" not in public
    assert public["deployment_id"].startswith("sha256:")

    config = LLMConfig(model_name="synthetic", api_key="do-not-render")
    assert "do-not-render" not in repr(config)


def test_client_cache_is_keyed_by_full_spec_and_credential(monkeypatch):
    profile = _profile()
    created = []

    def _fake_init(**kwargs):
        instance = Mock(name=f"client-{len(created)}")
        created.append((instance, kwargs["model"]))
        return instance

    monkeypatch.setattr(
        "app.adapters.outbound.agent.llm_provider.task_profile",
        lambda task: dict(profile),
    )
    monkeypatch.setattr(
        "app.adapters.outbound.agent.llm_provider.init_chat_model",
        _fake_init,
    )

    first = get_task_llm("sql")
    second = get_task_llm("sql")
    assert first is second
    assert len(created) == 1

    monkeypatch.setenv("SYNTHETIC_LLM_KEY", "rotated-synthetic-key")
    third = get_task_llm("sql")
    assert third is not first
    assert len(created) == 2

    profile["temperature"] = 0.2
    fourth = get_task_llm("sql")
    assert fourth is not third
    assert len(created) == 3


def test_non_deepseek_non_qwen_provider_gets_no_deepseek_body():
    kwargs = task_model_kwargs(_profile(
        model_name="gpt-test",
        provider="openai",
        model_provider="openai",
        base_url="https://api.openai.example.test/v1",
    ))
    assert "extra_body" not in kwargs
