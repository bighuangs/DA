import asyncio

from langchain_core.messages import AIMessage
import yaml

from evals.legacy.llm_reliability_probe import (
    CONTRACT_PATH,
    _profile_contract_gates,
    _safe_response_summary,
    run_probe,
)


class _SyntheticSQLModel:
    async def ainvoke(self, payload, config=None):
        del payload, config
        return AIMessage(
            content="WITH totals AS (SELECT 1 AS value) SELECT value FROM totals",
            response_metadata={
                "finish_reason": "stop",
                "request_id": "dev-probe-request",
                "token_usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 15,
                    "total_tokens": 35,
                },
            },
        )


def test_safe_response_summary_never_contains_response_body():
    summary = _safe_response_summary(
        AIMessage(
            content="SELECT secret_column FROM synthetic_table",
            response_metadata={"finish_reason": "stop"},
        )
    )
    assert summary["structured_output_clean"]
    assert summary["statement_type"] == "SELECT"
    assert summary["sql_sha256"].startswith("sha256:")
    assert "secret_column" not in str(summary)


def test_reliability_contract_requires_no_ttl_only_fix():
    payload = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert payload["failure_policy"]["timeout_extension_forbidden"]
    assert payload["release_gate"]["failed_core_llm_calls"] == 0
    release = next(
        item for item in payload["provider_modes"] if item.get("release_mode")
    )
    assert release["thinking_mode"] == "disabled"
    assert release["min_completion_tokens"] >= 2048


def test_probe_gate_passes_three_clean_single_attempt_calls(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "evals.legacy.llm_reliability_probe.get_task_llm",
        lambda task: _SyntheticSQLModel(),
    )
    report = asyncio.run(
        run_probe(repeat=3, persist=False, report_root=tmp_path)
    )
    assert report["status"] == "pass"
    assert report["summary"]["necessary_call_success_rate"] == 1.0
    assert report["summary"]["failed_core_llm_calls"] == 0
    assert report["summary"]["attempt_accounting"]["attempts"] == 3
    assert all(row["attempts"] == 1 for row in report["runs"])
    assert "base_url" not in report["profile"]
    assert "api_key_env" not in report["profile"]


def test_qwen38_formal_and_preview_use_different_reliability_contracts():
    common = {
        "provider": "dashscope",
        "model_provider": "openai",
        "max_completion_tokens": 2048,
        "timeout_seconds": 22,
        "temperature": 0.6,
        "reasoning_effort": None,
    }
    formal = _profile_contract_gates({
        **common,
        "model_name": "qwen3.8-max",
        "thinking_mode": "disabled",
    })
    preview = _profile_contract_gates({
        **common,
        "model_name": "qwen3.8-max-preview",
        "thinking_mode": "disabled",
    })
    assert all(formal.values())
    assert not preview["thinking_contract_valid"]
