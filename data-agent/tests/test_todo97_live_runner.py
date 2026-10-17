import asyncio
import os
import socket
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import AsyncMock, patch

from evals.legacy import todo97_live_runner


def test_dashscope_network_profile_matches_anyio_byte_hostname(monkeypatch):
    monkeypatch.setenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    class FakeResolver:
        nameservers: ClassVar[list[str]] = []
        timeout = 0
        lifetime = 0

        def resolve(self, host, record_type):
            assert host == "dashscope.aliyuncs.com"
            assert record_type == "A"
            return ["8.8.8.8"]

    def fake_original_getaddrinfo(host, port, *args, **kwargs):
        return [(host, port)]

    monkeypatch.setattr(
        todo97_live_runner.dns.resolver,
        "Resolver",
        lambda configure=False: FakeResolver(),
    )
    monkeypatch.setattr(
        todo97_live_runner,
        "_tls_works",
        lambda host, address=None: address == "8.8.8.8",
    )
    monkeypatch.setattr(
        todo97_live_runner, "_ORIGINAL_GETADDRINFO", fake_original_getaddrinfo,
    )
    monkeypatch.setattr(socket, "getaddrinfo", fake_original_getaddrinfo)

    profile = todo97_live_runner._install_dashscope_network_profile()

    assert profile == "explicit-public-dns-system-ca-v1"
    assert socket.getaddrinfo(b"dashscope.aliyuncs.com", 443) == [
        ("8.8.8.8", 443),
    ]
    assert socket.getaddrinfo(b"example.com", 443) == [
        (b"example.com", 443),
    ]


def test_qwen_candidate_resets_inherited_route_endpoints(monkeypatch):
    monkeypatch.setenv("LLM_SQL_BASE_URL", "https://stale-deepseek.example/v1")
    monkeypatch.setenv("LLM_CORRECTION_BASE_URL", "https://stale-qwen.example/v1")
    monkeypatch.setattr(
        todo97_live_runner,
        "_install_dashscope_network_profile",
        lambda: "system-dns-system-ca-v1",
    )

    candidate = todo97_live_runner._configure_candidate("Q1")

    assert candidate["candidate_label"] == "Q1"
    assert os.environ["LLM_SQL_BASE_URL"] == "__default__"
    assert os.environ["LLM_CORRECTION_BASE_URL"] == "__default__"
    assert os.environ["LLM_SQL_NETWORK_PROFILE"] == "system-dns-system-ca-v1"


def test_d0_configuration_does_not_run_dashscope_network_probe(monkeypatch):
    def unexpected_network_probe():
        raise AssertionError("D0 must not probe the DashScope endpoint")

    monkeypatch.setattr(
        todo97_live_runner,
        "_install_dashscope_network_profile",
        unexpected_network_probe,
    )

    candidate = todo97_live_runner._configure_candidate("D0")

    assert candidate["candidate_label"] == "D0"
    assert os.environ["LLM_SQL_PROVIDER"] == "deepseek"
    assert os.environ["LLM_SQL_BASE_URL"] == "__default__"
    assert os.environ["LLM_SQL_NETWORK_PROFILE"] == "system-default"


def test_experiment_reuses_one_preheated_embedding_session(monkeypatch):
    events = []

    class FakeEmbeddingSession:
        async def __aenter__(self):
            events.append("warmup")
            return self

        async def __aexit__(self, *_):
            events.append("close")

    async def fake_run_eval(**kwargs):
        events.append((kwargs["repeat_index"], kwargs["cache_state"]))
        assert kwargs["embedding_session"] is embedding_session_holder[0]
        return True

    embedding_session_holder = []

    class CapturingEmbeddingSession(FakeEmbeddingSession):
        def __init__(self):
            embedding_session_holder.append(self)

    monkeypatch.setattr(todo97_live_runner, "_validated_probe", lambda *_: {})
    args = SimpleNamespace(
        phase0_artifact="probe.json",
        workspace_id="workspace-1",
        experiment_id="experiment-1",
        candidate="D0",
        suite="easy_sql_v1",
    )
    with (
        patch(
            "evals.legacy.embedding_session.EvaluationEmbeddingSession",
            CapturingEmbeddingSession,
        ),
        patch("evals.legacy.run_eval.main", new=fake_run_eval),
    ):
        passed = asyncio.run(todo97_live_runner._run_experiment(args, {}))

    assert passed is True
    assert events == [
        "warmup",
        (1, "cold"),
        (2, "warm"),
        (3, "warm"),
        "close",
    ]


def test_experiment_does_not_start_paid_suite_when_embedding_warmup_fails(
    monkeypatch,
):
    class BrokenEmbeddingSession:
        async def __aenter__(self):
            raise RuntimeError("BGE warmup failed")

        async def __aexit__(self, *_):
            raise AssertionError("未进入会话时不应调用 __aexit__")

    run_eval = AsyncMock()
    monkeypatch.setattr(todo97_live_runner, "_validated_probe", lambda *_: {})
    args = SimpleNamespace(
        phase0_artifact="probe.json",
        workspace_id="workspace-1",
        experiment_id="experiment-1",
        candidate="D0",
        suite="easy_sql_v1",
    )
    with (
        patch(
            "evals.legacy.embedding_session.EvaluationEmbeddingSession",
            BrokenEmbeddingSession,
        ),
        patch("evals.legacy.run_eval.main", new=run_eval),
    ):
        try:
            asyncio.run(todo97_live_runner._run_experiment(args, {}))
        except RuntimeError as exc:
            assert "BGE warmup failed" in str(exc)
        else:
            raise AssertionError("预热失败必须阻止 suite")

    run_eval.assert_not_awaited()


def test_experiment_stops_after_first_ineligible_repeat(monkeypatch):
    events = []

    class FakeEmbeddingSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            events.append("close")

    async def fake_run_eval(**kwargs):
        events.append(kwargs["repeat_index"])
        return kwargs["repeat_index"] == 1

    monkeypatch.setattr(todo97_live_runner, "_validated_probe", lambda *_: {})
    args = SimpleNamespace(
        phase0_artifact="probe.json",
        workspace_id="workspace-1",
        experiment_id="experiment-1",
        candidate="D0",
        suite="easy_sql_v1",
    )
    with (
        patch(
            "evals.legacy.embedding_session.EvaluationEmbeddingSession",
            FakeEmbeddingSession,
        ),
        patch("evals.legacy.run_eval.main", new=fake_run_eval),
    ):
        passed = asyncio.run(todo97_live_runner._run_experiment(args, {}))

    assert passed is False
    assert events == [1, 2, "close"]
