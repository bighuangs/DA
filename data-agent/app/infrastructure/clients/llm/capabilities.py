"""Versioned provider/model capabilities used by routing and TODO-97 evaluation.

The registry contains no credentials or raw endpoints.  It deliberately separates
the human experiment label from the provider model id and records when a model is
only safe to *probe*, rather than assuming that a product name is callable.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_REGISTRY_VERSION = "llm-capability-registry-v1"
TODO97_CANDIDATE_SET = ("D0", "Q1", "Q3", "Q4")
_QWEN_REASONING_EFFORTS = ("low", "medium", "xhigh")


_REGISTRY: dict[str, dict[str, Any]] = {
    "D0": {
        "candidate_id": "todo97-d0-deepseek-v4-pro-native-v1",
        "model_id": "deepseek-v4-pro",
        "model_revision": "provider-alias-unresolved",
        "provider": "deepseek",
        "model_provider": "deepseek",
        "access_mode": "native",
        "endpoint_profile": "deepseek-default",
        "adapter_version": "deepseek-native-chat-v1",
        "lifecycle": "production-control",
        "comparison_mode": "strict-nonthinking-sql",
        "promotion_eligible": True,
        "thinking_modes": ("auto", "enabled", "disabled"),
        "thinking_only": False,
        "reasoning_efforts": ("low", "high", "xhigh", "max"),
        "temperature_min": 0.0,
        "temperature_min_thinking": 0.0,
        "phase1": {"thinking_mode": "disabled", "temperature": 0.0},
        "official_listing": "provider-account-control",
    },
    "Q1": {
        "candidate_id": "todo97-q1-qwen3.7-plus-standard-nonthinking-v1",
        "model_id": "qwen3.7-plus",
        "model_revision": "rolling-alias; snapshot-2026-05-26-listed",
        "provider": "dashscope",
        "model_provider": "openai",
        "access_mode": "standard",
        "endpoint_profile": "dashscope-compatible-cn",
        "adapter_version": "dashscope-openai-chat-v1",
        "lifecycle": "officially-listed",
        "comparison_mode": "strict-nonthinking-sql",
        "promotion_eligible": True,
        "thinking_modes": ("auto", "enabled", "disabled"),
        "thinking_only": False,
        "reasoning_efforts": (),
        "temperature_min": 0.0,
        "temperature_min_thinking": 0.6,
        "phase1": {"thinking_mode": "disabled", "temperature": 0.0},
        "official_listing": "verified-2026-08-07",
    },
    "Q3": {
        "candidate_id": "todo97-q3-qwen3.7-flash-standard-nonthinking-v1",
        "model_id": "qwen3.7-flash",
        "model_revision": "rolling-alias; snapshot-2026-07-15-listed",
        "provider": "dashscope",
        "model_provider": "openai",
        "access_mode": "standard",
        "endpoint_profile": "dashscope-compatible-cn",
        "adapter_version": "dashscope-openai-chat-v1",
        "lifecycle": "officially-listed-probe-required",
        "comparison_mode": "strict-nonthinking-sql",
        "promotion_eligible": True,
        "thinking_modes": ("auto", "enabled", "disabled"),
        "thinking_only": False,
        "reasoning_efforts": (),
        "temperature_min": 0.0,
        "temperature_min_thinking": 0.6,
        "phase1": {"thinking_mode": "disabled", "temperature": 0.0},
        "official_listing": "verified-2026-08-07-batch-listing",
    },
    "Q4": {
        "candidate_id": "todo97-q4-qwen3.8-max-standard-probe-v1",
        "model_id": "qwen3.8-max",
        "model_revision": "unavailable",
        "provider": "dashscope",
        "model_provider": "openai",
        "access_mode": "standard",
        "endpoint_profile": "dashscope-compatible-cn",
        "adapter_version": "dashscope-openai-chat-v1",
        "lifecycle": "official-listing-not-found-probe-only",
        "comparison_mode": "strict-nonthinking-sql-if-phase0-passes",
        "promotion_eligible": False,
        # Keep the planned request explicit so Phase 0 can determine whether
        # the account actually exposes such a formal endpoint.  A failed probe
        # is an availability result, not an SQL-quality score.
        "thinking_modes": ("auto", "enabled", "disabled"),
        "thinking_only": False,
        "reasoning_efforts": _QWEN_REASONING_EFFORTS,
        "temperature_min": 0.6,
        "temperature_min_thinking": 0.6,
        "phase1": {"thinking_mode": "disabled", "temperature": 0.6},
        "official_listing": "not-found-2026-08-07",
    },
    "Q4P": {
        "candidate_id": "todo97-q4p-qwen3.8-max-preview-token-plan-v1",
        "model_id": "qwen3.8-max-preview",
        "model_revision": "preview-unresolved",
        "provider": "dashscope_token_plan",
        "model_provider": "openai",
        "access_mode": "token_plan",
        "endpoint_profile": "dashscope-token-plan",
        "adapter_version": "dashscope-openai-chat-v1",
        "lifecycle": "preview-observation-only",
        "comparison_mode": "production-feasible-observation",
        "promotion_eligible": False,
        "thinking_modes": ("auto", "enabled"),
        "thinking_only": True,
        "reasoning_efforts": _QWEN_REASONING_EFFORTS,
        "temperature_min": 0.6,
        "temperature_min_thinking": 0.6,
        "phase1": {"thinking_mode": "enabled", "temperature": 0.6},
        "official_listing": "verified-2026-08-07-token-plan",
    },
}

for _capability in _REGISTRY.values():
    _capability.update({
        "structured_sql_output": "text-sql-validator-v1",
        "usage_metadata": "phase0-probe-required",
        "finish_reason": "phase0-probe-required",
        "provider_request_id": "phase0-probe-required",
        "timeout_control": "unified-invocation-boundary-v2",
        "cancellation_control": "unified-invocation-boundary-v2",
        "retry_control": "provider-retry-disabled; bounded-recovery-v2",
        "concurrency_control": "route-and-provider-semaphore-v1",
    })


def candidate_capability(label: str) -> dict[str, Any]:
    """Return one immutable-by-caller candidate contract."""
    normalized = str(label or "").strip().upper()
    if normalized not in _REGISTRY:
        raise ValueError(f"未知 TODO-97 候选：{label}")
    return deepcopy(_REGISTRY[normalized]) | {
        "candidate_label": normalized,
        "registry_version": CAPABILITY_REGISTRY_VERSION,
    }


def capability_for_model(model_name: str) -> dict[str, Any] | None:
    """Resolve aliases/snapshots without guessing unknown product names."""
    normalized = str(model_name or "").strip().lower()
    for label, raw in _REGISTRY.items():
        if normalized == str(raw["model_id"]).lower():
            return candidate_capability(label)
    by_specificity = sorted(
        _REGISTRY.items(),
        key=lambda item: len(str(item[1]["model_id"])),
        reverse=True,
    )
    for label, raw in by_specificity:
        model_id = str(raw["model_id"]).lower()
        if normalized.startswith(f"{model_id}-"):
            return candidate_capability(label)
    return None


def public_capability_registry(labels: tuple[str, ...] | None = None) -> dict[str, Any]:
    selected = labels or tuple(_REGISTRY)
    return {
        "schema_version": CAPABILITY_REGISTRY_VERSION,
        "candidates": {
            label: candidate_capability(label)
            for label in selected
        },
    }


def validate_profile_capability(profile: dict[str, Any]) -> dict[str, bool]:
    """Validate only documented/local parameter contracts, never availability."""
    capability = capability_for_model(str(profile.get("model_name") or ""))
    if capability is None:
        return {
            "registered_model": False,
            "provider_matches": False,
            "adapter_matches": False,
            "access_mode_matches": False,
            "thinking_mode_supported": False,
            "reasoning_effort_supported": False,
            "temperature_supported": False,
        }
    thinking_mode = str(profile.get("thinking_mode") or "auto").lower()
    effort = str(profile.get("reasoning_effort") or "").lower()
    temperature = float(profile.get("temperature") or 0.0)
    minimum = float(
        capability["temperature_min_thinking"]
        if thinking_mode == "enabled"
        else capability["temperature_min"]
    )
    allowed_efforts = set(capability.get("reasoning_efforts") or ())
    return {
        "registered_model": True,
        "provider_matches": str(profile.get("provider") or "").lower()
        == str(capability["provider"]).lower(),
        "adapter_matches": str(profile.get("model_provider") or "").lower()
        == str(capability["model_provider"]).lower(),
        "access_mode_matches": str(profile.get("access_mode") or "standard").lower()
        == str(capability["access_mode"]).lower(),
        "thinking_mode_supported": thinking_mode
        in set(capability.get("thinking_modes") or ()),
        "reasoning_effort_supported": (
            not effort or effort in allowed_efforts
        ),
        "temperature_supported": minimum <= temperature < 2.0,
    }


def provider_request_options(profile: dict[str, Any]) -> dict[str, Any]:
    """Map a registered capability to provider request options."""
    capability = capability_for_model(str(profile.get("model_name") or ""))
    if capability is None:
        return {}
    thinking_mode = str(profile.get("thinking_mode") or "auto").lower()
    effort = str(profile.get("reasoning_effort") or "").lower()
    if capability["candidate_label"] == "Q4P":
        if thinking_mode == "disabled":
            raise ValueError("qwen3.8-max-preview 是 thinking-only，不能关闭 thinking")
        if effort and effort not in set(_QWEN_REASONING_EFFORTS):
            raise ValueError(
                "qwen3.8-max-preview 的 reasoning_effort 仅支持 low/medium/xhigh"
            )
    gates = validate_profile_capability(profile)
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise ValueError(
            f"模型能力合同不满足（{capability['candidate_label']}）：{','.join(failed)}"
        )
    model_id = str(capability["model_id"]).lower()
    candidate_label = str(capability["candidate_label"])
    extra_body: dict[str, Any] = {}
    options: dict[str, Any] = {}
    if candidate_label == "Q4P":
        extra_body["reasoning_effort"] = effort or "xhigh"
    elif model_id.startswith("qwen"):
        if thinking_mode in {"enabled", "disabled"}:
            extra_body["enable_thinking"] = thinking_mode == "enabled"
        if effort:
            extra_body["reasoning_effort"] = effort
        if candidate_label == "Q4":
            extra_body["preserve_thinking"] = False
    elif model_id.startswith("deepseek"):
        if str(profile.get("model_provider") or "").lower() == "deepseek":
            if thinking_mode in {"enabled", "disabled"}:
                extra_body["thinking"] = {"type": thinking_mode}
            if effort:
                options["reasoning_effort"] = effort
        else:
            if thinking_mode in {"enabled", "disabled"}:
                extra_body["enable_thinking"] = thinking_mode == "enabled"
            if effort:
                extra_body["reasoning_effort"] = effort
    if extra_body:
        options["extra_body"] = extra_body
    return options
