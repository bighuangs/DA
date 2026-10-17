"""Single-process TODO-97 live probe/experiment runner.

The runner keeps cold/warm/warm in one worker and contains an opt-in repair for
the local Windows TUN fake-DNS path observed during the 2026-08-07 evaluation.
It never rewrites the configured provider URL: public DNS is used only to
resolve the exact hostname, while HTTPS still validates the original SNI/cert.
"""
from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import os
import socket
import ssl
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import dns.resolver
from dotenv import load_dotenv
from evals.paths import PROJECT_ROOT

TODO97_EVALUATION_SUITES = (
    "easy_sql_v1",
    "medium_sql_v1",
    "hard_sql_v1",
)
_ORIGINAL_GETADDRINFO = socket.getaddrinfo


def _tls_works(host: str, address: str | None = None) -> bool:
    try:
        with (
            socket.create_connection((address or host, 443), timeout=10) as raw,
            ssl.create_default_context().wrap_socket(
                raw, server_hostname=host,
            ),
        ):
            return True
    except OSError:
        return False


def _install_dashscope_network_profile() -> str:
    """Repair only a proven fake-DNS/TLS failure; otherwise keep system DNS."""
    base_url = str(os.getenv("DASHSCOPE_BASE_URL") or "").strip()
    host = urlsplit(base_url).hostname
    if not host:
        raise RuntimeError("DASHSCOPE_BASE_URL 缺少合法 hostname")
    # Workspace tooling may inject its own CA bundle.  Provider experiments use
    # the OS/Python default trust store so the Alibaba certificate is validated
    # independently of that unrelated workspace setting.
    os.environ.pop("SSL_CERT_FILE", None)
    os.environ.pop("REQUESTS_CA_BUNDLE", None)
    if _tls_works(host):
        return "system-dns-system-ca-v1"

    resolver = dns.resolver.Resolver(configure=False)
    resolver.nameservers = ["223.5.5.5", "119.29.29.29"]
    resolver.timeout = 4
    resolver.lifetime = 8
    answers = resolver.resolve(host, "A")
    addresses = [
        str(answer)
        for answer in answers
        if ipaddress.ip_address(str(answer)).is_global
    ]
    selected = next(
        (address for address in addresses if _tls_works(host, address)), None
    )
    if not selected:
        raise RuntimeError("公共 DNS 返回地址均未通过原始 hostname 的 TLS 校验")

    def resolved_getaddrinfo(
        requested_host: str | bytes, port: Any, *args: Any, **kwargs: Any,
    ):
        # AnyIO/httpcore resolves byte hostnames even when the original URL
        # contains text.  Normalize only for the equality check so the exact
        # DashScope hostname receives the verified address and every unrelated
        # resolution keeps its original value/type.
        comparable_host = (
            requested_host.decode("ascii", errors="ignore")
            if isinstance(requested_host, bytes)
            else str(requested_host)
        )
        target = (
            selected
            if comparable_host.lower() == host.lower()
            else requested_host
        )
        return _ORIGINAL_GETADDRINFO(target, port, *args, **kwargs)

    socket.getaddrinfo = resolved_getaddrinfo
    return "explicit-public-dns-system-ca-v1"


def _configure_candidate(label: str) -> dict[str, Any]:
    from app.infrastructure.clients.llm.capabilities import candidate_capability

    candidate = candidate_capability(label)
    recipe = candidate["phase1"]
    os.environ.update({
        "LLM_SQL_MODEL_NAME": str(candidate["model_id"]),
        "LLM_SQL_PROVIDER": str(candidate["provider"]),
        # Ignore inherited per-route/legacy strong endpoints and resolve the
        # endpoint from the selected provider profile.  Otherwise a stale
        # DeepSeek URL can silently survive when this process switches to
        # DashScope (or vice versa).
        "LLM_SQL_BASE_URL": "__default__",
        "LLM_SQL_API_KEY_ENV": (
            "DASHSCOPE_API_KEY" if label != "D0" else "LLM_API_KEY"
        ),
        "LLM_SQL_THINKING_MODE": str(recipe["thinking_mode"]),
        "LLM_SQL_REASONING_EFFORT": "null",
        "LLM_SQL_TEMPERATURE": str(recipe["temperature"]),
        "LLM_CORRECTION_MODEL_NAME": "deepseek-v4-pro",
        "LLM_CORRECTION_PROVIDER": "deepseek",
        "LLM_CORRECTION_BASE_URL": "__default__",
        "LLM_CORRECTION_API_KEY_ENV": "LLM_API_KEY",
        "LLM_CORRECTION_THINKING_MODE": "disabled",
        "LLM_CORRECTION_REASONING_EFFORT": "null",
        "LLM_CORRECTION_TEMPERATURE": "0",
    })
    if label == "D0":
        profile = "system-default"
    else:
        profile = _install_dashscope_network_profile()
    os.environ["LLM_SQL_NETWORK_PROFILE"] = profile
    return candidate


def _validated_probe(path: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    probe_candidate = payload.get("candidate") or {}
    profile = payload.get("profile") or {}
    recipe = candidate.get("phase1") or {}
    checks = {
        "schema": payload.get("schema_version") == "todo97-phase0-probe-v2",
        "registry": payload.get("capability_registry_version")
        == candidate.get("registry_version"),
        "status": payload.get("status") == "pass",
        "candidate_label": probe_candidate.get("candidate_label")
        == candidate.get("candidate_label"),
        "candidate_id": profile.get("candidate_id") == candidate.get("candidate_id"),
        "model": profile.get("model_name") == candidate.get("model_id"),
        "provider": profile.get("provider") == candidate.get("provider"),
        "adapter": profile.get("model_provider") == candidate.get("model_provider"),
        "access": profile.get("access_mode") == candidate.get("access_mode"),
        "thinking": profile.get("thinking_mode") == recipe.get("thinking_mode"),
        "temperature": float(profile.get("temperature") or 0)
        == float(recipe.get("temperature") or 0),
        "gates": bool(payload.get("gates")) and all(
            bool(value) for value in (payload.get("gates") or {}).values()
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Phase 0 artifact 未通过身份/门禁绑定：" + ",".join(failed))
    return payload


async def _run_experiment(args: argparse.Namespace, candidate: dict[str, Any]) -> bool:
    if not args.phase0_artifact:
        raise ValueError("Hard50 前必须显式绑定通过的 Phase 0 artifact")
    path = Path(args.phase0_artifact)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    _validated_probe(path, candidate)
    from evals.legacy.embedding_session import EvaluationEmbeddingSession
    from evals.legacy.run_eval import main as run_eval

    # 三个 repeat 共用同一个已预热客户端：R1 在清缓存后仍是 cold，R2/R3
    # 才能真实观察同进程 warm cache；预热失败会在任何付费用例前抛错。
    async with EvaluationEmbeddingSession() as embedding_session:
        for repeat_index, cache_state in ((1, "cold"), (2, "warm"), (3, "warm")):
            valid = await run_eval(
                workspace_id=args.workspace_id,
                experiment_id=args.experiment_id,
                candidate_label=args.candidate,
                repeat_index=repeat_index,
                cache_state=cache_state,
                evaluation_profile="core-default",
                dataset_suite=args.suite,
                embedding_session=embedding_session,
            )
            if not valid:
                return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="TODO-97 安全 live runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe = subparsers.add_parser("probe")
    probe.add_argument("--candidate", choices=("D0", "Q1", "Q3", "Q4"), required=True)
    probe.add_argument("--repeat", type=int, default=3)
    experiment = subparsers.add_parser("experiment")
    experiment.add_argument("--candidate", choices=("D0", "Q1", "Q3", "Q4"), required=True)
    experiment.add_argument("--experiment-id", required=True)
    experiment.add_argument("--phase0-artifact", required=True)
    # 公开版本必须显式选择获准评测的工作空间，禁止沿用个人开发环境身份。
    experiment.add_argument("--workspace-id", required=True)
    experiment.add_argument(
        "--suite", choices=TODO97_EVALUATION_SUITES, default="hard_sql_v1",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    candidate = _configure_candidate(args.candidate)
    if args.command == "probe":
        from evals.legacy.llm_reliability_probe import _render_markdown, run_probe

        report = asyncio.run(run_probe(
            repeat=args.repeat, candidate_label=args.candidate,
        ))
        print(_render_markdown(report))
        return 0 if report.get("status") == "pass" else 2
    return 0 if asyncio.run(_run_experiment(args, candidate)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
