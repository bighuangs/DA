"""TODO-61 query-signal extraction and routing ablation.

The default report is deterministic and dependency-free. ``--live`` additionally
runs the exact RAG contract against the configured MySQL/Qdrant/ES/BGE stack for
K0/K1/K2. Live RAG evidence is deliberately not treated as end-to-end NL2SQL
approval, so this runner never changes the active profile automatically.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import psutil
from omegaconf import OmegaConf

from app.infrastructure.clients.hybrid_embedding_client import HybridEmbeddingClient
from app.config.app_config import app_config
from app.agent_system.agents.nl2sql.services.query_signals import (
    PROFILE_VERSIONS,
    ROUTE_POLICY_VERSION,
    build_route_signal_plan,
    deterministic_value_route_variants,
    extract_query_signals,
)
from app.agent_system.governance.retrieval_budget import invalidate_retrieval_caches
from app.agent_system.agents.nl2sql.contracts.query_signals import (
    QUERY_SIGNAL_CONTRACT_VERSION,
    SIGNAL_KINDS,
    normalize_signal,
    stable_signal_hash,
)
from evals.paths import (
    LEGACY_DATASETS_ROOT,
    PROJECT_ROOT,
    REPORTS_ROOT as REPORT_ROOT,
)


CONTRACT_PATH = LEGACY_DATASETS_ROOT / "query_signal_set.yaml"
REPORT_PATH = REPORT_ROOT / "query_signal_ablation_latest.json"
REPORT_MD_PATH = REPORT_ROOT / "query_signal_ablation_latest.md"
EXPECTED_FIELDS = {
    "auxiliary_terms",
    "identifiers",
    "quoted_phrases",
    "entity_literals",
    "time_expressions",
    "numeric_constraints",
    "operators",
    "polarity",
}
MODIFIER_FIELDS = (
    "time_expressions",
    "numeric_constraints",
    "operators",
    "polarity",
)
PROFILE_MATRIX = {
    "K0": {
        "extractor": "full_query_only",
        "route_policy": "full_query_only_v1",
    },
    "K1": {
        "extractor": "jieba_tfidf_v1",
        "route_policy": "current_multi_query_v1",
    },
    "K2": {
        "extractor": "domain_signals_v1",
        "route_policy": "routed_signals_v1",
    },
    "K3": {
        "extractor": "jieba_textrank_v1",
        "route_policy": "current_multi_query_v1",
    },
}


def _load_contract() -> dict[str, Any]:
    return dict(
        OmegaConf.to_container(OmegaConf.load(CONTRACT_PATH), resolve=True) or {}
    )


def validate_contract() -> dict[str, Any]:
    payload = _load_contract()
    cases = [dict(item) for item in payload.get("cases") or []]
    ids = [str(item.get("id") or "") for item in cases]
    categories = {str(item.get("category") or "") for item in cases}
    malformed = []
    for case in cases:
        expected = dict(case.get("expected") or {})
        if (
            not case.get("id")
            or not case.get("query")
            or not case.get("category")
            or set(expected) != EXPECTED_FIELDS
        ):
            malformed.append(str(case.get("id") or "<unknown>"))
    checks = {
        "versioned": bool(payload.get("version")),
        "reviewed": payload.get("review_status") == "reviewed",
        "minimum_cases": len(cases) >= 20,
        "category_coverage": len(categories) >= 12,
        "unique_ids": len(ids) == len(set(ids)) and all(ids),
        "expected_fields_complete": not malformed,
    }
    return {
        "valid": all(checks.values()),
        "version": payload.get("version"),
        "fingerprint": stable_signal_hash(payload),
        "cases": len(cases),
        "categories": len(categories),
        "checks": checks,
        "malformed_case_ids": malformed,
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(
        0, min(len(ordered) - 1, math.ceil(len(ordered) * quantile) - 1)
    )
    return round(ordered[index], 4)


def _ratio(hit: int, total: int) -> float:
    return round(hit / total, 4) if total else 1.0


def _normalized(values: list[str] | tuple[str, ...]) -> set[str]:
    return {normalize_signal(value) for value in values if normalize_signal(value)}


def _evaluate_profile(
    label: str,
    profile: str,
    route_policy: str,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    process = psutil.Process()
    rss_before = process.memory_info().rss
    cpu_before = sum(process.cpu_times()[:2])
    cold_ms = 0.0
    warm_latencies: list[float] = []
    auxiliary_hits = auxiliary_total = auxiliary_output = auxiliary_relevant = 0
    modifier_hits = modifier_total = 0
    standalone_modifier_hits = standalone_modifier_total = 0
    standalone_retained = 0
    stable = 0
    failed_auxiliary: list[str] = []
    failed_modifiers: list[str] = []
    remote_counts: dict[str, list[int]] = {
        route: [] for route in ("column", "metric", "value", "training")
    }
    local_counts: dict[str, list[int]] = {
        route: [] for route in ("column", "metric", "value", "training")
    }

    for case_index, case in enumerate(cases):
        query = str(case["query"])
        kwargs = {
            "profile": profile,
            "workspace_id": "reviewed-workspace-a",
            "workspace_terms": list(case.get("workspace_terms") or []),
            "lexical_top_k": 10,
        }
        started = time.perf_counter()
        signals = extract_query_signals(query, **kwargs)
        latency_ms = (time.perf_counter() - started) * 1000
        if case_index == 0:
            cold_ms = latency_ms
        for _ in range(3):
            started = time.perf_counter()
            repeated = extract_query_signals(query, **kwargs)
            warm_latencies.append((time.perf_counter() - started) * 1000)
        stable += int(
            signals.signal_fingerprint == repeated.signal_fingerprint
            and signals.provenance.profile_fingerprint
            == repeated.provenance.profile_fingerprint
        )

        expected = dict(case["expected"])
        expected_aux = _normalized(list(expected["auxiliary_terms"]))
        output_aux = _normalized([
            *signals.lexical_terms,
            *signals.domain_terms,
            *signals.identifiers,
            *signals.quoted_phrases,
            *signals.entity_literals,
        ])
        case_aux_hits = len(expected_aux & output_aux)
        auxiliary_hits += case_aux_hits
        auxiliary_total += len(expected_aux)
        auxiliary_output += len(output_aux)
        auxiliary_relevant += len(expected_aux & output_aux)
        if case_aux_hits != len(expected_aux):
            failed_auxiliary.append(str(case["id"]))

        case_modifier_hit = case_modifier_total = 0
        normalized_query = normalize_signal(query)
        for field in MODIFIER_FIELDS:
            expected_values = _normalized(list(expected[field]))
            actual_values = _normalized(list(getattr(signals, field)))
            current_hits = len(expected_values & actual_values)
            case_modifier_hit += current_hits
            case_modifier_total += len(expected_values)
            standalone_modifier_hits += sum(
                value in normalized_query for value in expected_values
            )
            standalone_modifier_total += len(expected_values)
        modifier_hits += case_modifier_hit
        modifier_total += case_modifier_total
        if case_modifier_hit != case_modifier_total and profile == "domain_signals_v1":
            failed_modifiers.append(str(case["id"]))
        standalone_retained += int(
            signals.standalone_query == query
            and signals.semantic_queries[0] == query
        )

        variants = deterministic_value_route_variants(
            query, signals.retrieval_terms()
        )
        for route in remote_counts:
            plan = build_route_signal_plan(
                signals,
                route,
                route_policy=route_policy,
                remote_signal_limit=4,
                priority_terms=variants if route == "value" else (),
            )
            remote_counts[route].append(len(plan.remote_signals))
            local_counts[route].append(len(plan.local_signals))

    rss_after = process.memory_info().rss
    cpu_after = sum(process.cpu_times()[:2])
    summary = {
        "profile": profile,
        "profile_version": PROFILE_VERSIONS[profile],
        "route_policy": route_policy,
        "cases": len(cases),
        "standalone_retention": _ratio(standalone_retained, len(cases)),
        "auxiliary_recall": _ratio(auxiliary_hits, auxiliary_total),
        "auxiliary_precision": _ratio(auxiliary_relevant, auxiliary_output),
        "typed_modifier_recall": _ratio(modifier_hits, modifier_total),
        "standalone_modifier_retention": _ratio(
            standalone_modifier_hits, standalone_modifier_total
        ),
        "stability_rate": _ratio(stable, len(cases)),
        "failed_auxiliary_case_ids": sorted(set(failed_auxiliary)),
        "failed_modifier_case_ids": sorted(set(failed_modifiers)),
        "extractor_latency_ms": {
            "first_observed": round(cold_ms, 4),
            "warm_p50": _percentile(warm_latencies, 0.50),
            "warm_p95": _percentile(warm_latencies, 0.95),
            "warm_p99": _percentile(warm_latencies, 0.99),
        },
        "process": {
            "cpu_seconds_delta": round(max(0.0, cpu_after - cpu_before), 6),
            "rss_mb_delta": round(
                (rss_after - rss_before) / (1024 * 1024), 4
            ),
        },
        "routing": {
            route: {
                "remote_queries_mean": round(
                    statistics.fmean(remote_counts[route]), 3
                ),
                "remote_queries_max": max(remote_counts[route]),
                "local_signals_mean": round(
                    statistics.fmean(local_counts[route]), 3
                ),
            }
            for route in remote_counts
        },
    }
    summary["signal_layer_gate"] = bool(
        summary["standalone_retention"] == 1.0
        and summary["standalone_modifier_retention"] == 1.0
        and summary["stability_rate"] == 1.0
        and (
            profile != "domain_signals_v1"
            or summary["typed_modifier_recall"] == 1.0
        )
    )
    return summary


def _privacy_probe() -> dict[str, Any]:
    private_query = "客户甲的customer_name与机密利润率"
    first = extract_query_signals(
        private_query,
        profile="domain_signals_v1",
        workspace_id="workspace-a",
        workspace_terms=["customer_name"],
    )
    second = extract_query_signals(
        private_query,
        profile="domain_signals_v1",
        workspace_id="workspace-b",
        workspace_terms=["机密利润率"],
    )
    serialized = json.dumps(first.privacy_summary(), ensure_ascii=False)
    checks = {
        "raw_query_absent": private_query not in serialized,
        "entity_text_absent": "客户甲" not in serialized,
        "dictionary_text_absent": "customer_name" not in serialized,
        "dictionary_fingerprint_isolated": (
            first.provenance.dictionary_version
            != second.provenance.dictionary_version
        ),
        "workspace_match_isolated": (
            first.domain_terms == ("customer_name",)
            and second.domain_terms == ("机密利润率",)
        ),
        "raw_text_logged_false": (
            first.privacy_summary().get("raw_text_logged") is False
        ),
    }
    return {"checks": checks, "gate_passed": all(checks.values())}


def _phase0_contract() -> dict[str, Any]:
    checks = {
        "batch_query_api_available": hasattr(
            HybridEmbeddingClient, "aembed_queries"
        ),
        "query_signal_contract_versioned": bool(
            QUERY_SIGNAL_CONTRACT_VERSION
        ),
        "route_policy_versioned": bool(ROUTE_POLICY_VERSION),
        "signal_kinds_versioned": len(SIGNAL_KINDS) >= 9,
        "cache_namespace_versioned": bool(
            app_config.query_signals.cache_namespace_version
        ),
        "plaintext_logging_disabled": (
            app_config.query_signals.log_plaintext is False
        ),
    }
    return {"checks": checks, "gate_passed": all(checks.values())}


def build_report(
    *,
    live_profiles: dict[str, Any] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    contract = validate_contract()
    cases = [dict(item) for item in _load_contract().get("cases") or []]
    profiles = {
        label: _evaluate_profile(
            label,
            definition["extractor"],
            definition["route_policy"],
            cases,
        )
        for label, definition in PROFILE_MATRIX.items()
    }
    privacy = _privacy_probe()
    phase0 = _phase0_contract()
    active_matches = bool(
        app_config.query_signals.profile == profiles["K1"]["profile"]
        and app_config.query_signals.route_policy
        == profiles["K1"]["route_policy"]
    )
    live_evaluated = bool(live_profiles)
    live_active = (live_profiles or {}).get("K1") or {}
    live_challengers = {
        label: item
        for label, item in (live_profiles or {}).items()
        if label != "K1"
    }
    passing_live_challengers = sorted(
        label
        for label, item in live_challengers.items()
        if item.get("gate_passed")
    )
    gates = {
        "contract_valid": contract["valid"],
        "phase0_contract": phase0["gate_passed"],
        "all_profiles_keep_standalone": all(
            item["standalone_retention"] == 1.0
            for item in profiles.values()
        ),
        "all_profiles_keep_modifiers_in_standalone": all(
            item["standalone_modifier_retention"] == 1.0
            for item in profiles.values()
        ),
        "k2_typed_modifier_recall": (
            profiles["K2"]["typed_modifier_recall"] == 1.0
        ),
        "profile_stability": all(
            item["stability_rate"] == 1.0 for item in profiles.values()
        ),
        "privacy_and_tenant_isolation": privacy["gate_passed"],
        "active_profile_explicit": active_matches,
        "live_active_rag_gate": (
            bool(live_active.get("gate_passed"))
            if live_evaluated
            else True
        ),
    }
    if (
        live_evaluated
        and not passing_live_challengers
        and live_active.get("gate_passed")
    ):
        decision_reason = (
            "active K1 通过本轮真实 RAG 门禁，但没有挑战者同时通过；"
            "且尚无三次端到端 NL2SQL non-regression 证据，因此保留 active。"
        )
    elif live_evaluated and not passing_live_challengers:
        decision_reason = (
            "本轮 active K1 与所有已评测挑战者均未通过真实 RAG 门禁；"
            "结构化结果不得解释为任何方案胜出，active 仅因 fail-closed "
            "发布策略保持不变。"
        )
    elif live_evaluated:
        decision_reason = (
            f"真实 RAG 候选 {', '.join(passing_live_challengers)} 已通过召回"
            "门禁，但仍不足以替代三次端到端 NL2SQL non-regression；"
            "挑战者尚未获发布授权。"
        )
    else:
        decision_reason = (
            "尚无同协议端到端 NL2SQL 配对证据；按预注册规则保留 K1。"
        )
    report = {
        "available": True,
        "todo": "TODO-61",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": (
            "deterministic_signal_plus_live_rag"
            if live_evaluated
            else "deterministic_signal_and_route_contract"
        ),
        "contract": contract,
        "phase0": phase0,
        "privacy": privacy,
        "profiles": profiles,
        "live": {
            "evaluated": live_evaluated,
            "profiles": live_profiles or {},
        },
        "gates": gates,
        "decision": {
            "selected_profile": "jieba_tfidf_v1",
            "selected_route_policy": "current_multi_query_v1",
            "active_changed": False,
            "status": "retain_active",
            "reason": decision_reason,
            "passing_live_challengers": passing_live_challengers,
            "k2_signal_layer_delta_vs_k1": {
                "auxiliary_recall": round(
                    profiles["K2"]["auxiliary_recall"]
                    - profiles["K1"]["auxiliary_recall"],
                    4,
                ),
                "typed_modifier_recall": round(
                    profiles["K2"]["typed_modifier_recall"]
                    - profiles["K1"]["typed_modifier_recall"],
                    4,
                ),
                "column_remote_queries_mean": round(
                    profiles["K2"]["routing"]["column"][
                        "remote_queries_mean"
                    ]
                    - profiles["K1"]["routing"]["column"][
                        "remote_queries_mean"
                    ],
                    3,
                ),
            },
        },
        "summary": {
            "gate_passed": all(gates.values()),
            "cases": len(cases),
            "selected_profile": "jieba_tfidf_v1",
            "selected_route_policy": "current_multi_query_v1",
            "live_evaluated": live_evaluated,
            "raw_text_logged": False,
        },
    }
    report["report_fingerprint"] = stable_signal_hash({
        "contract": contract["fingerprint"],
        "profiles": profiles,
        "gates": gates,
        "decision": report["decision"],
    })
    if write:
        _write_report(report)
    return report


def _write_report(report: dict[str, Any]) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rows = []
    for label, item in report["profiles"].items():
        rows.append(
            "| {label} | `{profile}` | {aux:.1%} | {modifier:.1%} | "
            "{qdrant:.2f} | {value:.2f} | {p95:.3f} ms |".format(
                label=label,
                profile=item["profile"],
                aux=item["auxiliary_recall"],
                modifier=item["typed_modifier_recall"],
                qdrant=item["routing"]["column"]["remote_queries_mean"],
                value=item["routing"]["value"]["remote_queries_mean"],
                p95=item["extractor_latency_ms"]["warm_p95"],
            )
        )
    live_rows = []
    for label, item in (report.get("live") or {}).get("profiles", {}).items():
        summary = item.get("summary") or {}
        failed = ", ".join(item.get("failed_case_ids") or []) or "—"
        live_rows.append(
            "| {label} | {gate} | {recall:.1%} | {precision:.1%} | "
            "{effective:.1%} | {remote} | {failed} |".format(
                label=label,
                gate="通过" if item.get("gate_passed") else "未通过",
                recall=float(summary.get("column_recall_at_k") or 0),
                precision=float(summary.get("column_precision_at_k") or 0),
                effective=float(
                    summary.get("effective_context_column_recall") or 0
                ),
                remote=(
                    (item.get("remote_queries_mean") or {}).get("column")
                    or "—"
                ),
                failed=failed,
            )
        )
    live_markdown = ""
    if live_rows:
        live_markdown = (
            "\n\n## 真实 RAG 同协议结果\n\n"
            "| 组 | 门禁 | 原始字段 Recall | 原始字段 Precision | "
            "最终上下文 Recall | Column 远端查询均值 | 失败用例 |\n"
            "| --- | --- | ---: | ---: | ---: | ---: | --- |\n"
            + "\n".join(live_rows)
            + "\n"
        )
    markdown = (
        "# TODO-61 查询信号消融\n\n"
        f"> 门禁：{'通过' if report['summary']['gate_passed'] else '未通过'} ｜ "
        f"受控样本 {report['summary']['cases']} ｜ "
        f"真实 RAG：{'已执行' if report['summary']['live_evaluated'] else '未执行'}\n\n"
        "## 结论\n\n"
        f"- 选择：`{report['decision']['selected_profile']}` + "
        f"`{report['decision']['selected_route_policy']}`（未切换 active）。\n"
        f"- 原因：{report['decision']['reason']}\n"
        "- Phase 0 已将在线批量查询固定为 query encoder，并按 profile、"
        "route policy 和真实有序远端信号隔离缓存。\n\n"
        "## 信号与路由结果\n\n"
        "| 组 | Extractor | 辅助信号 Recall | Typed modifier Recall | "
        "Column 远端查询均值 | Value 查询均值 | Warm P95 |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |\n"
        + "\n".join(rows)
        + "\n\n"
        "报告不保存问题、词典或实体原文，只保存聚合指标、用例 ID 与指纹。\n"
        + live_markdown
    )
    REPORT_MD_PATH.write_text(markdown, encoding="utf-8")


def _live_route_summary(payload: dict, evaluated: dict, artifact: Path) -> dict:
    route_counts: dict[str, list[int]] = {}
    encode_modes: set[str] = set()
    for case in payload.get("cases") or []:
        stage = case.get("stage_evidence") or {}
        for row in stage.get("retrieval_budget") or []:
            signal = row.get("query_signals") or {}
            route = str(row.get("route") or signal.get("route") or "")
            if not route:
                continue
            route_counts.setdefault(route, []).append(
                int(
                    signal.get("remote_query_count")
                    or row.get("remote_query_count")
                    or 0
                )
            )
            if signal.get("query_encode_mode"):
                encode_modes.add(str(signal["query_encode_mode"]))
    failed_cases = [
        item
        for item in (evaluated.get("cases") or [])
        if not item.get("passed")
    ]
    failed_case_ids = [
        str(item.get("id") or item.get("golden_id") or "")
        for item in failed_cases
        if item.get("id") or item.get("golden_id")
    ]
    failed_routes = {
        str(item.get("id") or item.get("golden_id") or ""): sorted(
            route
            for route, diagnostic in (item.get("diagnostics") or {}).items()
            if route != "effective_context"
            and (diagnostic or {}).get("missing")
        )
        for item in failed_cases
    }
    return {
        "artifact": str(artifact.relative_to(PROJECT_ROOT)),
        "valid": bool(payload.get("valid")),
        "cases": int(payload.get("n") or 0),
        "gate_passed": bool((evaluated.get("summary") or {}).get("gate_passed")),
        "summary": evaluated.get("summary") or {},
        "remote_queries_mean": {
            route: round(statistics.fmean(values), 3)
            for route, values in route_counts.items()
            if values
        },
        "query_encode_modes": sorted(encode_modes),
        "failed_case_ids": failed_case_ids,
        "failed_routes": failed_routes,
        "raw_text_logged": False,
    }


async def run_live_profiles(workspace_id: str) -> dict[str, Any]:
    from app.adapters.outbound.agent.nl2sql_capabilities import build_nl2sql_settings
    from app.agent_system.agents.nl2sql.settings import override_settings
    from app.infrastructure.clients.embedding_client_manager import embedding_client_manager
    from app.infrastructure.clients.es_client_manager import es_client_manager
    from app.infrastructure.clients.mysql_client_manager import (
        dw_mysql_client_manager,
        meta_mysql_client_manager,
    )
    from app.infrastructure.clients.qdrant_client_manager import qdrant_client_manager
    from app.domain.access.policy import PolicySnapshot
    from app.infrastructure.search.index_version import retrieval_index_version
    from evals.legacy.rag_recall import (
        evaluate_live_run,
        load_contract,
        validate_contract as validate_rag_contract,
    )
    from evals.legacy.run_eval import resolve_workspace_id
    from evals.legacy.run_rag_live import (
        RUNS_DIR,
        _run_case,
        create_retrieval_graph_runner,
    )
    from evals.adapters.outbound.nl2sql import build_nl2sql_evaluation_context
    from app.infrastructure.search.elasticsearch.repositories.value_es_repository import ValueESRepository
    from app.infrastructure.persistence.mysql.repositories.dw_mysql_repository import DWMysqlRepository
    from app.infrastructure.persistence.mysql.repositories.meta_mysql_repository import MetaMysqlRepository
    from app.infrastructure.search.qdrant.repositories.column_qdrant_repository import (
        ColumnQdrantRepository,
    )
    from app.infrastructure.search.qdrant.repositories.metric_qdrant_repository import (
        MetricQdrantRepository,
    )
    from app.infrastructure.search.qdrant.repositories.training_qdrant_repository import (
        TrainingQdrantRepository,
    )

    output: dict[str, Any] = {}
    validation = validate_rag_contract()
    if not validation["valid"]:
        raise RuntimeError("RAG 契约无效，不能执行查询信号 live 消融")
    embedding_client_manager.init()
    es_client_manager.init()
    dw_mysql_client_manager.init()
    meta_mysql_client_manager.init()
    qdrant_client_manager.init()
    runner = create_retrieval_graph_runner()
    try:
        async with (
            dw_mysql_client_manager.session_factory() as dw_session,
            meta_mysql_client_manager.session_factory() as meta_session,
        ):
            selected_workspace = await resolve_workspace_id(
                meta_session, workspace_id
            )
            dw_repo = DWMysqlRepository(dw_session)
            meta_repo = MetaMysqlRepository(meta_session, selected_workspace)
            value_repo = ValueESRepository(es_client_manager.client)
            column_repo = ColumnQdrantRepository(qdrant_client_manager.client)
            metric_repo = MetricQdrantRepository(qdrant_client_manager.client)
            training_repo = TrainingQdrantRepository(qdrant_client_manager.client)
            access_policy = PolicySnapshot(
                policies=tuple(),
                user_id="query-signal-eval-user",
                username="query-signal-eval",
                workspace_id=selected_workspace,
                role="owner",
            )
            for label in ("K0", "K1", "K2"):
                definition = PROFILE_MATRIX[label]
                settings = override_settings(
                    build_nl2sql_settings(),
                    "query_signals",
                    profile=definition["extractor"],
                    route_policy=definition["route_policy"],
                )
                context = build_nl2sql_evaluation_context(
                    dw_mysql_repo=dw_repo,
                    meta_mysql_repo=meta_repo,
                    value_es_repo=value_repo,
                    column_qdrant_repo=column_repo,
                    metric_qdrant_repo=metric_repo,
                    training_qdrant_repo=training_repo,
                    embedding_client=embedding_client_manager.client,
                    access_policy=access_policy,
                    workspace_id=selected_workspace,
                    settings=settings,
                )
                invalidate_retrieval_caches()
                rows = [
                    await _run_case(dict(case), context, runner)
                    for case in load_contract().get("cases") or []
                ]
                timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                payload = {
                    "artifact_schema": "query-signal-rag-live-v1",
                    "run_id": f"{timestamp}-{label.lower()}",
                    "timestamp": dt.datetime.now().astimezone().isoformat(
                        timespec="seconds"
                    ),
                    "workspace_id": selected_workspace,
                    "query_signal_profile": definition["extractor"],
                    "query_signal_route_policy": definition["route_policy"],
                    "contract_fingerprint": validation["fingerprint"],
                    "retrieval_index_version": retrieval_index_version(),
                    "valid": all(not row.get("error") for row in rows),
                    "n": len(rows),
                    "cases": rows,
                }
                evaluated = evaluate_live_run(payload)
                RUNS_DIR.mkdir(parents=True, exist_ok=True)
                artifact = (
                    RUNS_DIR
                    / f"query_signal_{label.lower()}_{timestamp}.json"
                )
                artifact.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                output[label] = _live_route_summary(
                    payload, evaluated, artifact
                )
    finally:
        invalidate_retrieval_caches()
        await embedding_client_manager.close()
        await es_client_manager.close()
        await dw_mysql_client_manager.close()
        await meta_mysql_client_manager.close()
        await qdrant_client_manager.close()
    return output


async def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--workspace-id")
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    if args.live and not args.workspace_id:
        parser.error("--live 需要 --workspace-id")
    live_profiles = (
        await run_live_profiles(str(args.workspace_id))
        if args.live
        else None
    )
    report = build_report(live_profiles=live_profiles, write=True)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"报告：{REPORT_MD_PATH}")
    return 2 if args.enforce and not report["summary"]["gate_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
