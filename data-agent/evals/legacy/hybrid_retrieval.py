"""TODO-56 live BGE-M3 dense/sparse/entity-RRF evaluation.

This module is owned by Data Agent.  It imports no implementation from STT or
SQLBot and uses only Data Agent's embedding, repository and rerank contracts.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from omegaconf import OmegaConf
from sqlalchemy import func, select

from app.infrastructure.clients.embedding_client_manager import embedding_client_manager
from app.infrastructure.clients.mysql_client_manager import meta_mysql_client_manager
from app.infrastructure.clients.qdrant_client_manager import qdrant_client_manager
from app.agent_system.governance.retrieval_budget import (
    RERANK_VERSION,
    match_features,
    rerank_and_cut,
    route_parameters,
)
from app.infrastructure.search.index_version import (
    column_index_version,
    retrieval_index_version,
    retrieval_parameters,
    retrieval_runtime_fingerprint,
)
from evals.legacy.rag_recall import load_contract as load_rag_contract
from app.infrastructure.persistence.mysql.records.metric_info_mysql import MetricInfoMySQL
from app.infrastructure.persistence.mysql.records.training_info_mysql import TrainingInfoMySQL
from app.infrastructure.persistence.mysql.records.workspace_mysql import WorkspaceMySQL
from app.infrastructure.persistence.mysql.repositories.meta_mysql_repository import MetaMysqlRepository
from app.infrastructure.search.qdrant.repositories.column_qdrant_repository import ColumnQdrantRepository
from app.infrastructure.search.qdrant.repositories.metric_qdrant_repository import MetricQdrantRepository
from app.infrastructure.search.qdrant.repositories.training_qdrant_repository import TrainingQdrantRepository
from app.application.services.meta_admin_service import MetaAdminService
from evals.paths import LEGACY_DATASETS_ROOT, REPORTS_ROOT


CONTRACT_PATH = LEGACY_DATASETS_ROOT / "hybrid_retrieval_set.yaml"
REPORT_PATH = REPORTS_ROOT / "hybrid_retrieval_latest.json"
REPORT_MD_PATH = REPORTS_ROOT / "hybrid_retrieval_latest.md"
ROUTES = ("column", "metric", "training")
REFERENCE_COLUMN_PRECISION = 0.5738
ISOLATED_COLLECTION_PREFIX = "data-agent_hybrid_eval_"


def _plain(path: Path) -> dict:
    return dict(OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {})


def _isolated_collection_names(
    workspace_id: str,
    *,
    run_token: str | None = None,
) -> dict[str, str]:
    workspace_hash = hashlib.sha256(workspace_id.encode("utf-8")).hexdigest()[:10]
    token = run_token or hashlib.sha256(
        f"{workspace_id}:{time.time_ns()}".encode("utf-8")
    ).hexdigest()[:12]
    return {
        route: f"{ISOLATED_COLLECTION_PREFIX}{workspace_hash}_{token}_{route}"
        for route in ROUTES
    }


async def _cleanup_isolated_collections(client: Any, names: list[str]) -> dict:
    requested = sorted(set(names))
    invalid = [
        name for name in requested
        if not name.startswith(ISOLATED_COLLECTION_PREFIX)
    ]
    if invalid:
        return {
            "passed": False,
            "requested": len(requested),
            "deleted": 0,
            "residual": len(invalid),
            "errors": ["non_isolated_collection_name"],
        }

    deleted = 0
    errors: list[str] = []
    for name in requested:
        try:
            if await client.collection_exists(collection_name=name):
                await client.delete_collection(collection_name=name)
                deleted += 1
        except Exception as exc:
            errors.append(type(exc).__name__)

    residual = 0
    for name in requested:
        try:
            residual += int(await client.collection_exists(collection_name=name))
        except Exception as exc:
            errors.append(type(exc).__name__)
            residual += 1
    return {
        "passed": not errors and residual == 0,
        "requested": len(requested),
        "deleted": deleted,
        "residual": residual,
        "errors": errors,
    }


def load_contract() -> dict:
    extension = _plain(CONTRACT_PATH)
    base = [dict(item) for item in (load_rag_contract().get("cases") or [])]
    extra = [dict(item) for item in (extension.get("extra_cases") or [])]
    return {
        **extension,
        "base_cases": base,
        "cases": [*base, *extra],
        "base_case_count": len(base),
        "extra_case_count": len(extra),
    }


def validate_contract(contract: dict | None = None) -> dict:
    contract = contract or load_contract()
    errors: list[str] = []
    ids: set[str] = set()
    for index, case in enumerate(contract.get("cases") or [], 1):
        case_id = str(case.get("id") or "")
        if not case_id or not str(case.get("query") or "").strip():
            errors.append(f"第 {index} 条缺少 id/query")
        if case_id in ids:
            errors.append(f"重复 ID：{case_id}")
        ids.add(case_id)
        if not any(case.get(key) for key in (
            "relevant_columns", "relevant_metrics", "relevant_training"
        )):
            errors.append(f"{case_id} 没有任何相关实体")
    required_profiles = {"R-D1", "R-S1", "R-H1", "R-H3"}
    missing_profiles = required_profiles - set(contract.get("profiles") or {})
    if missing_profiles:
        errors.append("缺少 profile：" + ",".join(sorted(missing_profiles)))
    import hashlib
    return {
        "valid": not errors,
        "errors": errors,
        "total": len(contract.get("cases") or []),
        "base_cases": int(contract.get("base_case_count") or 0),
        "extra_cases": int(contract.get("extra_case_count") or 0),
        "fingerprint": "sha256:" + hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
    }


def _identity(item: dict, route: str) -> str:
    if route == "metric":
        return str(item.get("name") or item.get("id") or "")
    if route == "training":
        return str(item.get("question") or item.get("id") or "")
    return str(item.get("id") or item.get("entity_id") or "")


def _expected(case: dict, route: str) -> list[str]:
    key = {
        "column": "relevant_columns",
        "metric": "relevant_metrics",
        "training": "relevant_training",
    }[route]
    return [str(value) for value in (case.get(key) or [])]


def _metric(expected: list[str], rows: list[dict]) -> dict:
    identifiers = [_identity(item, item.get("_route") or "") for item in rows]
    wanted = set(expected)
    hits = [1 if identifier in wanted else 0 for identifier in identifiers]
    recall = len(wanted & set(identifiers)) / max(1, len(wanted))
    precision = sum(hits) / max(1, len(identifiers))
    first = next((index for index, hit in enumerate(hits, 1) if hit), None)
    dcg = sum(hit / math.log2(index + 1) for index, hit in enumerate(hits, 1))
    ideal = sum(
        1 / math.log2(index + 1)
        for index in range(1, min(len(wanted), len(identifiers)) + 1)
    )
    return {
        "expected": expected,
        "returned": identifiers,
        "recall_at_k": round(recall, 4),
        "precision_at_k": round(precision, 4),
        "mrr": round(1 / first if first else 0, 4),
        "ndcg": round(dcg / ideal if ideal else 1, 4),
        "missing": sorted(wanted - set(identifiers)),
    }


def _mean(values: list[float]) -> float:
    return round(sum(values) / max(1, len(values)), 4)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return round(float(ordered[position]), 1)


async def resolve_workspace_id(meta_session, explicit: str | None = None) -> str:
    """Resolve one evaluation workspace without importing the NL2SQL runner."""
    requested = (explicit or os.getenv("EVAL_WORKSPACE_ID") or "").strip()
    if requested:
        return requested

    for model in (MetricInfoMySQL, TrainingInfoMySQL):
        statement = (
            select(model.workspace_id)
            .where(model.workspace_id.is_not(None))
            .group_by(model.workspace_id)
            .order_by(func.count().desc(), model.workspace_id)
        )
        workspace_id = (await meta_session.execute(statement)).scalars().first()
        if workspace_id:
            return str(workspace_id)

    statement = select(WorkspaceMySQL.id).order_by(
        WorkspaceMySQL.created_at,
        WorkspaceMySQL.id,
    )
    workspace_id = (await meta_session.execute(statement)).scalars().first()
    if workspace_id:
        return str(workspace_id)
    raise RuntimeError(
        "评测无法确定工作空间：请先创建工作空间，或在 .env 配置 EVAL_WORKSPACE_ID"
    )


def _profile_summary(cases: list[dict], base_ids: set[str]) -> dict:
    route_values: dict[str, defaultdict[str, list]] = {
        route: defaultdict(list) for route in ROUTES
    }
    base_column_recall: list[float] = []
    base_column_precision: list[float] = []
    identifier_scores: defaultdict[str, list] = defaultdict(list)
    degradations = leakage = passed = 0
    query_latencies: list[float] = []
    raw_points = unique_entities = 0
    for case in cases:
        case_passed = True
        query_latencies.append(float(case.get("latency_ms") or 0))
        degradations += len(case.get("degradations") or [])
        leakage += int(case.get("tenant_leakage") or 0)
        for route in ROUTES:
            metric = (case.get("routes") or {}).get(route)
            if not metric or not metric.get("expected"):
                continue
            for key in ("recall_at_k", "precision_at_k", "mrr", "ndcg"):
                route_values[route][key].append(float(metric[key]))
            case_passed = case_passed and not metric["missing"]
            evidence = metric.get("evidence") or {}
            raw_points += int((evidence.get("dense") or {}).get("raw_points") or 0)
            raw_points += int((evidence.get("sparse") or {}).get("raw_points") or 0)
            unique_entities += int(evidence.get("fused_entities") or 0)
            if case["id"] in base_ids and route == "column":
                base_column_recall.append(float(metric["recall_at_k"]))
                base_column_precision.append(float(metric["precision_at_k"]))
            if set(case.get("tags") or []) & {
                "identifier", "abbreviation", "alias", "english_name"
            }:
                identifier_scores["recall_at_k"].append(float(metric["recall_at_k"]))
                identifier_scores["mrr"].append(float(metric["mrr"]))
        passed += int(case_passed)
    routes = {
        route: {
            "cases": len(route_values[route]["recall_at_k"]),
            **{
                key: _mean(route_values[route][key])
                for key in ("recall_at_k", "precision_at_k", "mrr", "ndcg")
            },
        }
        for route in ROUTES
    }
    return {
        "cases": len(cases),
        "passed_cases": passed,
        "routes": routes,
        "base_column_recall_at_k": _mean(base_column_recall),
        "base_column_precision_at_k": _mean(base_column_precision),
        "identifier_subset": {
            "recall_at_k": _mean(identifier_scores["recall_at_k"]),
            "mrr": _mean(identifier_scores["mrr"]),
            "measurements": len(identifier_scores["mrr"]),
        },
        "degradation_count": degradations,
        "tenant_leakage": leakage,
        "raw_physical_points": raw_points,
        "fused_unique_entities": unique_entities,
        "latency_ms": {
            "p50": round(median(query_latencies), 1) if query_latencies else 0.0,
            "p95": _percentile(query_latencies, .95),
        },
    }


def _local_candidates(corpora: dict[str, list[dict]], route: str, query: str) -> list[dict]:
    result = []
    for candidate in corpora[route]:
        item = dict(candidate)
        item["_route"] = route
        features = match_features(query, [query], item, route)
        if features["strong_match"] or features["keyword_hits"]:
            item["score"] = 1.0
            item["retrieval_sources"] = ["exact_alias_lexical"]
            result.append(item)
    return result


def _metric_column_candidates(
    corpora: dict[str, list[dict]],
    query: str,
) -> tuple[list[dict], list[str]]:
    """Close exact metric definitions onto their declared physical columns.

    The production merge node performs the same metadata closure after the
    column and metric routes join.  R-H3 includes it so the active-profile
    measurement covers the composed retrieval result instead of undercounting
    fields that are deterministically supplied by a recalled metric.
    """
    columns = {
        str(item.get("id") or ""): item
        for item in corpora["column"]
        if item.get("id")
    }
    selected: dict[str, dict] = {}
    matched_metrics: list[str] = []
    for metric in corpora["metric"]:
        features = match_features(query, [query], metric, "metric")
        if not (features["strong_match"] or features["keyword_hits"]):
            continue
        metric_name = str(metric.get("name") or metric.get("id") or "")
        matched_metrics.append(metric_name)
        for column_id in metric.get("relevant_columns") or []:
            source = columns.get(str(column_id))
            if source is None:
                continue
            item = dict(source)
            item["_route"] = "column"
            item["score"] = 1.0
            item["retrieval_sources"] = ["metric_relevant_column"]
            item["metric_closure"] = sorted({
                *list((selected.get(str(column_id)) or {}).get("metric_closure") or []),
                metric_name,
            })
            selected[str(column_id)] = item
    return list(selected.values()), sorted(set(matched_metrics))


async def _search_route(
    route: str,
    repository: Any,
    embedding: Any,
    *,
    workspace_id: str,
    index_version: str,
    mode: str,
    rule_rerank: bool,
    query: str,
    local_corpora: dict[str, list[dict]],
) -> tuple[list[dict], dict, list[dict], float]:
    params = route_parameters(route)
    limit = params["pool"] if rule_rerank else params["max"]
    started = time.perf_counter()
    common = {
        "score_threshold": None if rule_rerank else 0.0,
        "sparse_score_threshold": None if rule_rerank else 0.0,
        "limit": limit,
        "mode": mode,
    }
    if route == "column":
        result = await repository.search(
            embedding, index_version=index_version, **common
        )
    else:
        result = await repository.search(embedding, workspace_id, **common)
    qdrant_ms = round((time.perf_counter() - started) * 1000, 1)
    rows = [{**dict(item), "_route": route} for item in result.items]
    if rule_rerank:
        metric_columns: list[dict] = []
        matched_metrics: list[str] = []
        if route == "column":
            metric_columns, matched_metrics = _metric_column_candidates(
                local_corpora,
                query,
            )
        rows, budget = rerank_and_cut(
            route,
            query,
            [query],
            [
                *_local_candidates(local_corpora, route, query),
                *metric_columns,
                *rows,
            ],
            min_limit=params["min"],
            max_limit=params["max"],
        )
        budget.update({
            "metric_closure_count": len(metric_columns),
            "metric_closure_metrics": matched_metrics,
        })
        evidence = {**result.evidence, "rule_budget": budget}
    else:
        evidence = result.evidence
    return rows, evidence, result.degradations, qdrant_ms


def _markdown(report: dict) -> str:
    lines = [
        "# TODO-56 BGE-M3 Dense + Sparse 混合检索报告",
        "",
        f"> 生成：`{report['generated_at']}` ｜ 工作空间：`{report['workspace_id']}`"
        f" ｜ 门禁：`{'pass' if report['summary']['gate_passed'] else 'fail'}`",
        "",
        "| Profile | 字段 Recall | 字段 Precision | 标识符 Recall | 标识符 MRR | P95 | 降级 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, profile in report["profiles"].items():
        summary = profile["summary"]
        column = summary["routes"]["column"]
        identifier = summary["identifier_subset"]
        lines.append(
            f"| {name} | {column['recall_at_k']:.1%} | {column['precision_at_k']:.1%} "
            f"| {identifier['recall_at_k']:.1%} | {identifier['mrr']:.3f} "
            f"| {summary['latency_ms']['p95']:.1f} ms | {summary['degradation_count']} |"
        )
    lines.extend([
        "",
        "## 结论",
        "",
        *[f"- {item}" for item in report["summary"]["conclusions"]],
        "",
        "旧 Dense 不再是运行时分支；历史对照只来自覆盖前保存的只读制品。",
        "",
    ])
    return "\n".join(lines)


async def run(
    workspace_id: str | None = None,
    *,
    overwrite_indexes: bool = False,
    isolated_indexes: bool = False,
    only: set[str] | None = None,
    enforce: bool = False,
) -> dict:
    if overwrite_indexes and isolated_indexes:
        raise ValueError(
            "--overwrite-indexes 与 --isolated-indexes 不能同时使用"
        )
    contract = load_contract()
    validation = validate_contract(contract)
    if not validation["valid"]:
        raise RuntimeError("Hybrid 评测契约无效：" + "；".join(validation["errors"]))

    embedding_client_manager.init()
    meta_mysql_client_manager.init()
    qdrant_client_manager.init()
    isolated_collections: dict[str, str] = {}
    try:
        async with meta_mysql_client_manager.session_factory() as session:
            selected_workspace = await resolve_workspace_id(session, workspace_id)
            meta_repo = MetaMysqlRepository(session, selected_workspace)
            await meta_repo.ensure_admin_tables()
            column_repo = ColumnQdrantRepository(qdrant_client_manager.client)
            metric_repo = MetricQdrantRepository(qdrant_client_manager.client)
            training_repo = TrainingQdrantRepository(qdrant_client_manager.client)
            if isolated_indexes:
                isolated_collections = _isolated_collection_names(
                    selected_workspace
                )
                column_repo.collection_name = isolated_collections["column"]
                metric_repo.collection_name = isolated_collections["metric"]
                training_repo.collection_name = isolated_collections["training"]
            admin = MetaAdminService(
                meta_repo,
                column_repo,
                metric_repo,
                training_repo,
                embedding_client_manager.client,
            )
            if overwrite_indexes or isolated_indexes:
                await admin.rebuild_all()
            status = await admin.hybrid_status()
            if not status["ready"]:
                raise RuntimeError(
                    "BGE-M3 混合索引未就绪；评测请增加 --isolated-indexes；"
                    "只有明确要重建活动索引时才使用 --overwrite-indexes"
                )

            table_rows = await meta_repo.list_table_infos()
            columns = [column for _, values in table_rows for column in values]
            column_version = column_index_version(columns)
            metrics = await meta_repo.get_all_metric_infos()
            trainings = await meta_repo.get_all_enabled_training_infos()
            local_corpora = {
                "column": [{
                    "id": item.id,
                    "name": item.name,
                    "description": item.description or "",
                    "alias": list(item.alias or []),
                    "table_id": item.table_id,
                    "role": item.role,
                } for item in columns],
                "metric": [{
                    "id": item.id,
                    "name": item.name,
                    "workspace_id": selected_workspace,
                    "description": item.description or "",
                    "alias": list(item.alias or []),
                    "relevant_columns": list(item.relevant_columns or []),
                } for item in metrics],
                "training": [{
                    "id": str(item.id),
                    "question": item.question,
                    "sql": item.sql,
                    "workspace_id": selected_workspace,
                } for item in trainings],
            }
            repositories = {
                "column": column_repo,
                "metric": metric_repo,
                "training": training_repo,
            }
            cases = [
                dict(case) for case in contract["cases"]
                if not only or str(case["id"]) in only
            ]
            profile_cases: dict[str, list[dict]] = {
                name: [] for name in contract["profiles"]
            }
            embedding_latencies: list[float] = []
            for position, case in enumerate(cases, 1):
                print(
                    f"Hybrid RAG {position}/{len(cases)} {case['id']}: {case['query']}",
                    flush=True,
                )
                embedding_started = time.perf_counter()
                embedding = await embedding_client_manager.client.aembed_query(case["query"])
                embedding_ms = round((time.perf_counter() - embedding_started) * 1000, 1)
                embedding_latencies.append(embedding_ms)
                for profile_name, profile in contract["profiles"].items():
                    case_started = time.perf_counter()
                    routes: dict[str, dict] = {}
                    degradations: list[dict] = []
                    leakage = 0
                    for route in ROUTES:
                        expected = _expected(case, route)
                        if not expected:
                            continue
                        rows, evidence, route_degradations, qdrant_ms = await _search_route(
                            route,
                            repositories[route],
                            embedding,
                            workspace_id=selected_workspace,
                            index_version=column_version,
                            mode=str(profile["mode"]),
                            rule_rerank=bool(profile.get("rule_rerank")),
                            query=str(case["query"]),
                            local_corpora=local_corpora,
                        )
                        for item in rows:
                            if (
                                route in {"metric", "training"}
                                and item.get("workspace_id") != selected_workspace
                            ):
                                leakage += 1
                        route_metric = _metric(expected, rows)
                        route_metric.update({
                            "candidates": [{
                                key: item.get(key) for key in (
                                    "id", "name", "question", "entity_id", "rank",
                                    "dense_rank", "dense_score", "sparse_rank",
                                    "sparse_score", "fusion_rank", "fusion_score",
                                    "fusion_method", "retrieval_sources",
                                    "matched_point_roles",
                                ) if item.get(key) is not None
                            } for item in rows],
                            "evidence": evidence,
                            "qdrant_ms": qdrant_ms,
                        })
                        routes[route] = route_metric
                        degradations.extend(route_degradations)
                    profile_cases[profile_name].append({
                        "id": str(case["id"]),
                        "query": str(case["query"]),
                        "tags": list(case.get("tags") or []),
                        "routes": routes,
                        "tenant_leakage": leakage,
                        "degradations": degradations,
                        "embedding_ms": embedding_ms,
                        "latency_ms": round(
                            (time.perf_counter() - case_started) * 1000, 1
                        ),
                    })

            base_ids = {
                str(case["id"]) for case in contract["base_cases"]
                if not only or str(case["id"]) in only
            }
            profiles = {
                name: {
                    "parameters": dict(contract["profiles"][name]),
                    "summary": _profile_summary(rows, base_ids),
                    "cases": rows,
                }
                for name, rows in profile_cases.items()
            }
            dense = profiles["R-D1"]["summary"]
            hybrid = profiles["R-H1"]["summary"]
            active = profiles["R-H3"]["summary"]
            full_contract = len(cases) == validation["total"]
            isolated_cleanup = (
                await _cleanup_isolated_collections(
                    qdrant_client_manager.client,
                    list(isolated_collections.values()),
                )
                if isolated_indexes
                else None
            )
            checks = {
                "full_contract": full_contract,
                "index_ready": bool(status["ready"]),
                "dense_evidence_nonempty": dense["raw_physical_points"] > 0,
                "sparse_evidence_nonempty": (
                    profiles["R-S1"]["summary"]["raw_physical_points"] > 0
                ),
                "hybrid_evidence_nonempty": (
                    hybrid["raw_physical_points"] > 0
                    and active["raw_physical_points"] > 0
                ),
                "base_recall_non_regression": active["base_column_recall_at_k"] >= 1.0,
                "column_precision_floor": (
                    active["base_column_precision_at_k"]
                    >= REFERENCE_COLUMN_PRECISION
                ),
                "identifier_recall_non_regression": (
                    hybrid["identifier_subset"]["recall_at_k"]
                    >= dense["identifier_subset"]["recall_at_k"]
                ),
                "identifier_mrr_non_regression": (
                    hybrid["identifier_subset"]["mrr"]
                    >= dense["identifier_subset"]["mrr"]
                ),
                "tenant_isolation": active["tenant_leakage"] == 0,
                "no_active_degradation": active["degradation_count"] == 0,
                "isolated_cleanup": bool(
                    not isolated_indexes
                    or (isolated_cleanup or {}).get("passed")
                ),
            }
            gate_passed = all(checks.values())
            conclusions = [
                (
                    "R-H3 保持首轮 16 题字段 Recall 100%。"
                    if checks["base_recall_non_regression"]
                    else "R-H3 未保持首轮 16 题字段 Recall 100%，不可视为通过。"
                ),
                (
                    "R-H1 在 identifier/alias 扩展集上相对 BGE-M3 Dense 无回退。"
                    if checks["identifier_recall_non_regression"]
                    and checks["identifier_mrr_non_regression"]
                    else "Hybrid 在 identifier/alias 扩展集上出现回退，需要继续调优。"
                ),
                (
                    "运行时已直接切为 BGE-M3 dense+sparse；不存在 v1 Dense 回退。"
                ),
                (
                    "Dense、Sparse 与 Hybrid 均有真实 Qdrant point 证据。"
                    if checks["dense_evidence_nonempty"]
                    and checks["sparse_evidence_nonempty"]
                    and checks["hybrid_evidence_nonempty"]
                    else "存在仅靠本地闭包通过、向量证据为空的风险，门禁拒绝通过。"
                ),
            ]
            report = {
                "artifact_schema": "data-agent-hybrid-retrieval-v1",
                "available": True,
                "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "workspace_id": selected_workspace,
                "source": (
                    "live_bge_m3_qdrant_isolated"
                    if isolated_indexes
                    else "live_bge_m3_qdrant"
                ),
                "contract": validation,
                "active_mode": "bge_m3_dense_sparse",
                "replacement_strategy": (
                    "isolated_ephemeral_collections"
                    if isolated_indexes
                    else (
                        "overwrite_existing_collections"
                        if overwrite_indexes
                        else "reuse_existing_collections"
                    )
                ),
                "v1_fallback": False,
                "isolated_indexes": {
                    "enabled": isolated_indexes,
                    "collections": isolated_collections,
                    "cleanup": isolated_cleanup,
                },
                "index_version": retrieval_index_version(),
                "query_contract": {
                    "fingerprint": retrieval_runtime_fingerprint(RERANK_VERSION),
                    "rerank_version": RERANK_VERSION,
                    "parameters": retrieval_parameters(),
                },
                "index_status": status,
                "profiles": profiles,
                "comparison": {
                    "R-H1_vs_R-D1": {
                        "identifier_recall_delta": round(
                            hybrid["identifier_subset"]["recall_at_k"]
                            - dense["identifier_subset"]["recall_at_k"], 4
                        ),
                        "identifier_mrr_delta": round(
                            hybrid["identifier_subset"]["mrr"]
                            - dense["identifier_subset"]["mrr"], 4
                        ),
                    },
                    "R-H3_vs_R-H1": {
                        "column_recall_delta": round(
                            active["routes"]["column"]["recall_at_k"]
                            - hybrid["routes"]["column"]["recall_at_k"], 4
                        ),
                        "column_precision_delta": round(
                            active["routes"]["column"]["precision_at_k"]
                            - hybrid["routes"]["column"]["precision_at_k"], 4
                        ),
                    },
                },
                "embedding_latency_ms": {
                    "p50": round(median(embedding_latencies), 1)
                    if embedding_latencies else 0.0,
                    "p95": _percentile(embedding_latencies, .95),
                },
                "summary": {
                    "gate_passed": gate_passed,
                    "checks": checks,
                    "conclusions": conclusions,
                },
            }
            output_path = (
                REPORT_PATH if full_contract
                else REPORT_PATH.with_name("hybrid_retrieval_sample_latest.json")
            )
            report["report_path"] = str(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if full_contract:
                REPORT_MD_PATH.write_text(_markdown(report), encoding="utf-8")
            if enforce and not gate_passed:
                raise RuntimeError("TODO-56 hybrid retrieval 门禁未通过")
            return report
    finally:
        if isolated_collections and qdrant_client_manager.client is not None:
            await _cleanup_isolated_collections(
                qdrant_client_manager.client,
                list(isolated_collections.values()),
            )
        await embedding_client_manager.close()
        await meta_mysql_client_manager.close()
        await qdrant_client_manager.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data Agent TODO-56 BGE-M3 Dense/Sparse/Hybrid 真实评测"
    )
    parser.add_argument("--workspace-id", default=None)
    parser.add_argument(
        "--overwrite-indexes",
        action="store_true",
        help="显式删除并重建现有三个 Qdrant collection；不保留 v1 Dense",
    )
    parser.add_argument(
        "--isolated-indexes",
        action="store_true",
        help=(
            "在随机命名的评测专用 collection 中构建并在结束时清理；"
            "不修改活动索引"
        ),
    )
    parser.add_argument(
        "--only",
        default="",
        help="仅运行逗号分隔的 case id；抽样运行不会通过 full_contract 门禁",
    )
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    selected = {item.strip() for item in args.only.split(",") if item.strip()} or None
    result = asyncio.run(run(
        args.workspace_id,
        overwrite_indexes=args.overwrite_indexes,
        isolated_indexes=args.isolated_indexes,
        only=selected,
        enforce=args.enforce,
    ))
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Hybrid 报告：{result['report_path']}")
