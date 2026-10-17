"""TODO-57 isolated Point projection ablation for Data Agent.

The three profiles share the exact BGE-M3 model, Dense+Sparse query vectors,
entity-level RRF, rule-rerank-v6 and evaluation cases.  Only the physical
entity-to-Point projection changes.  Experimental collections are isolated
from each other and from the active production collections.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import datetime as dt
import hashlib
import json
import time
from collections import Counter, defaultdict
from statistics import median
from typing import Any

from app.infrastructure.clients.embedding_client_manager import embedding_client_manager
from app.infrastructure.clients.mysql_client_manager import meta_mysql_client_manager
from app.infrastructure.clients.qdrant_client_manager import qdrant_client_manager
from app.config.app_config import app_config
from app.infrastructure.search.projection import (
    SUPPORTED_POINT_STRATEGIES,
    ProjectionRecord,
    alias_manifest_fingerprint,
    build_column_projection_records,
    build_metric_projection_records,
    build_training_projection_records,
    load_alias_manifest,
    projection_manifest,
)
from app.agent_system.governance.retrieval_budget import RERANK_VERSION
from app.infrastructure.search.index_version import (
    column_index_version,
    retrieval_index_version,
    retrieval_parameters,
)
from evals.legacy.hybrid_retrieval import (
    ROUTES,
    _expected,
    _metric,
    _percentile,
    _profile_summary,
    _search_route,
    load_contract,
    resolve_workspace_id,
    validate_contract,
)
from app.infrastructure.persistence.mysql.repositories.meta_mysql_repository import MetaMysqlRepository
from app.infrastructure.search.qdrant.repositories.column_qdrant_repository import ColumnQdrantRepository
from app.infrastructure.search.qdrant.repositories.metric_qdrant_repository import MetricQdrantRepository
from app.infrastructure.search.qdrant.repositories.training_qdrant_repository import TrainingQdrantRepository
from evals.paths import REPORTS_ROOT as REPORT_ROOT


REPORT_PATH = REPORT_ROOT / "point_projection_latest.json"
REPORT_MD_PATH = REPORT_ROOT / "point_projection_latest.md"
IMMUTABLE_REPORT_ROOT = REPORT_ROOT / "point_projection"
STRATEGIES = tuple(SUPPORTED_POINT_STRATEGIES)
BASELINE_STRATEGY = "multi_all"
ACTIVE_PROFILE = "R-H3"


def _hash_payload(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


def point_experiment_query_contract() -> dict:
    """Fingerprint frozen query behavior while marking projection as the variable."""
    parameters = copy.deepcopy(retrieval_parameters())
    index = parameters.get("index") or {}
    index["point_strategy"] = "__experimental_variable__"
    index["alias_manifest_hash"] = "__experimental_variable__"
    parameters["index"] = index
    payload = {
        "parameters": parameters,
        "rerank_version": RERANK_VERSION,
        "active_profile": ACTIVE_PROFILE,
    }
    return {
        "fingerprint": _hash_payload(payload),
        "rerank_version": RERANK_VERSION,
        "active_profile": ACTIVE_PROFILE,
        "parameters": parameters,
    }


def _metric_content(items: list[Any], workspace_id: str) -> list[dict]:
    return sorted(({
        "workspace_id": workspace_id,
        "id": str(item.id),
        "name": item.name,
        "description": item.description or "",
        "relevant_columns": item.relevant_columns or [],
        "alias": item.alias or [],
        "formula_rule": item.formula_rule,
    } for item in items), key=lambda row: row["id"])


def _training_content(items: list[Any], workspace_id: str) -> list[dict]:
    return sorted(({
        "workspace_id": workspace_id,
        "id": str(item.id),
        "question": item.question,
        "sql": item.sql,
    } for item in items), key=lambda row: row["id"])


def _workspace_groups(items: list[Any], fallback_workspace_id: str) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for item in items:
        workspace_id = str(getattr(item, "workspace_id", None) or fallback_workspace_id)
        if workspace_id:
            grouped[workspace_id].append(item)
    return dict(grouped)


def _collection_name(workspace_id: str, strategy: str, route: str) -> str:
    workspace_hash = hashlib.sha256(workspace_id.encode("utf-8")).hexdigest()[:10]
    return f"data-agent_point_eval_{workspace_hash}_{strategy}_{route}"


def _profile_manifest(
    route_records: dict[str, list[ProjectionRecord]],
    *,
    collection_statuses: dict[str, dict],
    build_duration_ms: float,
    vector_integrity: dict,
) -> dict:
    routes = {
        route: {
            **projection_manifest(records),
            "collection": collection_statuses[route].get("collection"),
            "collection_points_count": collection_statuses[route].get("points_count"),
            "collection_compatible": collection_statuses[route].get("compatible"),
        }
        for route, records in route_records.items()
    }
    all_records = [record for records in route_records.values() for record in records]
    entity_counts = Counter(
        (
            str(record.payload.get("workspace_id") or ""),
            str(record.payload.get("entity_type") or ""),
            str(record.payload.get("entity_id") or ""),
        )
        for record in all_records
    )
    point_count = len(all_records)
    entity_count = len(entity_counts)
    point_ids = [record.point_id for record in all_records]
    return {
        "routes": routes,
        "entity_count": entity_count,
        "point_count": point_count,
        "points_per_entity": round(point_count / max(1, entity_count), 4),
        "max_points_per_entity": max(entity_counts.values(), default=0),
        "duplicate_occupancy": round(
            max(0, point_count - entity_count) / max(1, point_count), 4
        ),
        "approved_alias_points": sum(
            record.point_role == "approved_alias" for record in all_records
        ),
        "stable_ids_valid": len(point_ids) == len(set(point_ids)),
        "duplicate_point_ids": len(point_ids) - len(set(point_ids)),
        "build_duration_ms": round(build_duration_ms, 1),
        "vector_integrity": vector_integrity,
    }


def _vector_integrity(vectors: list[Any]) -> dict:
    dense_size = int(app_config.embedding.dense_size)
    invalid_dense = sum(len(vector.dense) != dense_size for vector in vectors)
    invalid_sparse = sum(
        not vector.sparse_indices
        or len(vector.sparse_indices) != len(vector.sparse_values)
        for vector in vectors
    )
    return {
        "vector_count": len(vectors),
        "expected_dense_size": dense_size,
        "invalid_dense_count": invalid_dense,
        "invalid_sparse_count": invalid_sparse,
        "valid": invalid_dense == 0 and invalid_sparse == 0,
    }


async def _embed_records(records: list[ProjectionRecord]) -> list[Any]:
    vectors: list[Any] = []
    batch_size = max(1, int(app_config.embedding.batch_size))
    texts = [record.text for record in records]
    for start in range(0, len(texts), batch_size):
        vectors.extend(
            await embedding_client_manager.client.aembed_documents(
                texts[start:start + batch_size]
            )
        )
    return vectors


async def _build_profile(
    *,
    strategy: str,
    workspace_id: str,
    table_rows: list,
    all_metrics: list[Any],
    all_trainings: list[Any],
    alias_manifest: dict,
) -> tuple[dict[str, Any], dict]:
    client = qdrant_client_manager.client
    repositories: dict[str, Any] = {
        "column": ColumnQdrantRepository(client),
        "metric": MetricQdrantRepository(client),
        "training": TrainingQdrantRepository(client),
    }
    for route, repository in repositories.items():
        repository.collection_name = _collection_name(workspace_id, strategy, route)

    manifest_hash = alias_manifest_fingerprint(alias_manifest)
    columns = [column for _, items in table_rows for column in items]
    column_version = column_index_version(
        columns,
        point_strategy=strategy,
        alias_manifest_hash=manifest_hash,
    )
    route_records: dict[str, list[ProjectionRecord]] = {
        "column": build_column_projection_records(
            table_rows,
            index_version=column_version,
            strategy=strategy,
            alias_manifest=alias_manifest,
        ),
        "metric": [],
        "training": [],
    }
    metric_versions: dict[str, str] = {}
    for candidate_workspace, items in sorted(
        _workspace_groups(all_metrics, workspace_id).items()
    ):
        version = retrieval_index_version(
            _metric_content(items, candidate_workspace),
            point_strategy=strategy,
            alias_manifest_hash=manifest_hash,
        )
        metric_versions[candidate_workspace] = version
        route_records["metric"].extend(build_metric_projection_records(
            items,
            workspace_id=candidate_workspace,
            index_version=version,
            strategy=strategy,
            alias_manifest=alias_manifest,
        ))
    training_versions: dict[str, str] = {}
    for candidate_workspace, items in sorted(
        _workspace_groups(all_trainings, workspace_id).items()
    ):
        version = retrieval_index_version(
            _training_content(items, candidate_workspace),
            point_strategy=strategy,
            alias_manifest_hash=manifest_hash,
        )
        training_versions[candidate_workspace] = version
        route_records["training"].extend(build_training_projection_records(
            items,
            workspace_id=candidate_workspace,
            index_version=version,
            strategy=strategy,
            alias_manifest=alias_manifest,
        ))

    # Rebuild the same logical records twice before touching Qdrant. This makes
    # deterministic identity a live manifest gate, not only a unit-test claim.
    second_column_ids = {
        record.point_id for record in build_column_projection_records(
            table_rows,
            index_version=column_version,
            strategy=strategy,
            alias_manifest=alias_manifest,
        )
    }
    stable_rebuild = second_column_ids == {
        record.point_id for record in route_records["column"]
    }

    started = time.perf_counter()
    all_records = [
        record for route in ROUTES for record in route_records[route]
    ]
    vectors = await _embed_records(all_records)
    integrity = _vector_integrity(vectors)
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for route in ROUTES:
        offsets[route] = (cursor, cursor + len(route_records[route]))
        cursor += len(route_records[route])

    for repository in repositories.values():
        await repository.recreate_collection()
    for route in ROUTES:
        records = route_records[route]
        start, end = offsets[route]
        route_vectors = vectors[start:end]
        if not records:
            continue
        kwargs = {
            "source_texts": [record.text for record in records],
            "point_roles": [record.point_role for record in records],
        }
        payloads = [record.payload for record in records]
        ids = [record.point_id for record in records]
        if route == "column":
            await repositories[route].upsert_column_vectors(
                route_vectors, payloads, ids, **kwargs
            )
        elif route == "metric":
            await repositories[route].upsert_metric_vectors(
                route_vectors, payloads, ids, **kwargs
            )
        else:
            await repositories[route].upsert_training_vectors(
                route_vectors, payloads, ids, **kwargs
            )
    build_duration_ms = (time.perf_counter() - started) * 1000
    statuses = {
        route: await repository.status()
        for route, repository in repositories.items()
    }
    profile_manifest = _profile_manifest(
        route_records,
        collection_statuses=statuses,
        build_duration_ms=build_duration_ms,
        vector_integrity=integrity,
    )
    profile_manifest["stable_rebuild_ids"] = stable_rebuild
    profile_manifest["index_versions"] = {
        "column": column_version,
        "metric_by_workspace": metric_versions,
        "training_by_workspace": training_versions,
    }
    return repositories, profile_manifest


def _subset_summary(cases: list[dict]) -> dict:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for case in cases:
        for tag in case.get("tags") or []:
            for route_metric in (case.get("routes") or {}).values():
                if not route_metric.get("expected"):
                    continue
                for key in ("recall_at_k", "precision_at_k", "mrr", "ndcg"):
                    grouped[str(tag)][key].append(float(route_metric.get(key) or 0))
    return {
        tag: {
            "measurements": len(values["recall_at_k"]),
            **{
                key: round(sum(values[key]) / max(1, len(values[key])), 4)
                for key in ("recall_at_k", "precision_at_k", "mrr", "ndcg")
            },
        }
        for tag, values in sorted(grouped.items())
    }


def _ambiguous_false_positives(cases: list[dict]) -> int:
    total = 0
    for case in cases:
        if "ambiguous" not in set(case.get("tags") or []):
            continue
        for route in ROUTES:
            metric = (case.get("routes") or {}).get(route) or {}
            if metric.get("expected"):
                total += max(
                    0,
                    len(metric.get("returned") or [])
                    - len(set(metric.get("returned") or []) & set(metric["expected"])),
                )
    return total


def _strategy_checks(
    profile: dict,
    baseline: dict,
    *,
    full_contract: bool,
) -> dict[str, bool]:
    summary = profile["summary"]
    reference = baseline["summary"]
    manifest = profile["manifest"]
    route_non_regression = all(
        float(summary["routes"][route]["recall_at_k"])
        >= float(reference["routes"][route]["recall_at_k"]) - 0.0001
        for route in ROUTES
    )
    return {
        "full_contract": full_contract,
        "stable_point_ids": bool(
            manifest.get("stable_ids_valid")
            and manifest.get("stable_rebuild_ids")
        ),
        "vector_integrity": bool(
            (manifest.get("vector_integrity") or {}).get("valid")
        ),
        "collection_integrity": all(
            item.get("collection_compatible")
            and int(item.get("collection_points_count") or 0)
            == int(item.get("point_count") or 0)
            for item in (manifest.get("routes") or {}).values()
        ),
        "passed_cases_non_regression": (
            int(summary.get("passed_cases") or 0)
            >= int(reference.get("passed_cases") or 0)
        ),
        "route_recall_non_regression": route_non_regression,
        "base_column_recall_non_regression": (
            float(summary.get("base_column_recall_at_k") or 0)
            >= float(reference.get("base_column_recall_at_k") or 0)
            - 0.0001
        ),
        "column_precision_non_regression": (
            float(summary.get("base_column_precision_at_k") or 0)
            >= float(reference.get("base_column_precision_at_k") or 0)
            - 0.0001
        ),
        "long_tail_recall_non_regression": (
            float((summary.get("identifier_subset") or {}).get("recall_at_k") or 0)
            >= float(
                (reference.get("identifier_subset") or {}).get(
                    "recall_at_k"
                ) or 0
            ) - 0.0001
        ),
        "long_tail_mrr_non_regression": (
            float((summary.get("identifier_subset") or {}).get("mrr") or 0)
            >= float(
                (reference.get("identifier_subset") or {}).get("mrr")
                or 0
            ) - 0.0001
        ),
        "ambiguous_false_positive_non_regression": (
            int(profile.get("ambiguous_false_positives") or 0)
            <= int(baseline.get("ambiguous_false_positives") or 0)
        ),
        "tenant_isolation": int(summary.get("tenant_leakage") or 0) == 0,
        "no_degradation": int(summary.get("degradation_count") or 0) == 0,
    }


def select_strategy(profiles: dict[str, dict], *, full_contract: bool) -> dict:
    """Apply the documented simplicity-first selection rule."""
    baseline = profiles[BASELINE_STRATEGY]
    for strategy, profile in profiles.items():
        profile["checks"] = _strategy_checks(
            profile,
            baseline,
            full_contract=full_contract,
        )
        profile["eligible"] = all(profile["checks"].values())

    canonical = profiles["single_canonical"]
    selective = profiles["mixed_selective"]
    if canonical["eligible"]:
        selected = "single_canonical"
        reason = (
            "single_canonical 在全部质量门禁上不低于 multi_all，"
            "并以最少 Point 满足召回、精度、歧义与隔离要求。"
        )
    elif (
        selective["eligible"]
        and int(selective["manifest"].get("approved_alias_points") or 0) > 0
        and float(selective["manifest"].get("points_per_entity") or 0) <= 2.0
    ):
        selected = "mixed_selective"
        reason = (
            "single_canonical 存在可复现缺口；mixed_selective 仅增加人工批准别名，"
            "在每实体不超过 2 个 Point 的约束内恢复门禁。"
        )
    else:
        selected = BASELINE_STRATEGY
        reason = (
            "简化策略尚未同时通过质量门禁，继续保留 multi_all 作为 active 基线。"
        )
    selected_profile = profiles[selected]
    return {
        "selected_strategy": selected,
        "reason": reason,
        "gate_passed": bool(selected_profile["eligible"]),
        "active_strategy_at_evaluation": app_config.point_projection.strategy,
        "requires_active_rebuild": selected != app_config.point_projection.strategy,
        "rejected": {
            strategy: [
                key for key, passed in profile["checks"].items() if not passed
            ]
            for strategy, profile in profiles.items()
            if strategy != selected
        },
    }


def _comparison(profiles: dict[str, dict], selected: str) -> dict:
    baseline = profiles[BASELINE_STRATEGY]
    result: dict[str, dict] = {}
    for strategy, profile in profiles.items():
        summary = profile["summary"]
        reference = baseline["summary"]
        result[f"{strategy}_vs_{BASELINE_STRATEGY}"] = {
            "point_count_delta": (
                int(profile["manifest"]["point_count"])
                - int(baseline["manifest"]["point_count"])
            ),
            "points_per_entity_delta": round(
                float(profile["manifest"]["points_per_entity"])
                - float(baseline["manifest"]["points_per_entity"]),
                4,
            ),
            "duplicate_occupancy_delta": round(
                float(profile["manifest"]["duplicate_occupancy"])
                - float(baseline["manifest"]["duplicate_occupancy"]),
                4,
            ),
            "column_recall_delta": round(
                float(summary["routes"]["column"]["recall_at_k"])
                - float(reference["routes"]["column"]["recall_at_k"]),
                4,
            ),
            "column_precision_delta": round(
                float(summary["routes"]["column"]["precision_at_k"])
                - float(reference["routes"]["column"]["precision_at_k"]),
                4,
            ),
            "long_tail_mrr_delta": round(
                float(summary["identifier_subset"]["mrr"])
                - float(reference["identifier_subset"]["mrr"]),
                4,
            ),
            "query_p95_delta_ms": round(
                float(summary["latency_ms"]["p95"])
                - float(reference["latency_ms"]["p95"]),
                1,
            ),
            "build_duration_delta_ms": round(
                float(profile["manifest"]["build_duration_ms"])
                - float(baseline["manifest"]["build_duration_ms"]),
                1,
            ),
        }
    result["selected"] = selected
    return result


def _markdown(report: dict) -> str:
    lines = [
        "# TODO-57 Point 投影策略消融报告",
        "",
        f"> 生成：`{report['generated_at']}` ｜ 工作空间：`{report['workspace_id']}` "
        f"｜ 选择：`{report['decision']['selected_strategy']}` "
        f"｜ 门禁：`{'pass' if report['decision']['gate_passed'] else 'fail'}`",
        "",
        "| 策略 | 实体 | Point | Point/实体 | 重复占位 | 字段 Recall | 字段 Precision | 长尾 MRR | P95 | 资格 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for strategy in STRATEGIES:
        profile = report["profiles"][strategy]
        manifest = profile["manifest"]
        summary = profile["summary"]
        lines.append(
            f"| {strategy} | {manifest['entity_count']} | {manifest['point_count']} "
            f"| {manifest['points_per_entity']:.2f} | {manifest['duplicate_occupancy']:.1%} "
            f"| {summary['routes']['column']['recall_at_k']:.1%} "
            f"| {summary['routes']['column']['precision_at_k']:.1%} "
            f"| {summary['identifier_subset']['mrr']:.3f} "
            f"| {summary['latency_ms']['p95']:.1f} ms "
            f"| {'✅' if profile['eligible'] else '❌'} |"
        )
    lines.extend([
        "",
        "## 选择结论",
        "",
        f"- {report['decision']['reason']}",
        f"- 评测时 active：`{report['decision']['active_strategy_at_evaluation']}`；"
        f"是否需要重建：`{report['decision']['requires_active_rebuild']}`。",
        "",
        "三套索引使用相同模型、查询、过滤、RRF、rule-rerank-v6 与 28 题契约，"
        "并写入彼此隔离的实验 collection。",
        "",
    ])
    return "\n".join(lines)


async def run(
    workspace_id: str | None = None,
    *,
    only: set[str] | None = None,
    enforce: bool = False,
) -> dict:
    contract = load_contract()
    validation = validate_contract(contract)
    if not validation["valid"]:
        raise RuntimeError("Point 投影评测契约无效：" + "；".join(validation["errors"]))

    embedding_client_manager.init()
    meta_mysql_client_manager.init()
    qdrant_client_manager.init()
    try:
        async with meta_mysql_client_manager.session_factory() as session:
            selected_workspace = await resolve_workspace_id(session, workspace_id)
            selected_repo = MetaMysqlRepository(session, selected_workspace)
            await selected_repo.ensure_admin_tables()
            source_repo = MetaMysqlRepository(session)
            table_rows = await selected_repo.list_table_infos()
            all_metrics = [
                item for item in await source_repo.get_all_metric_infos() if item.enabled
            ]
            all_trainings = [
                item
                for item in await source_repo.get_all_enabled_training_infos()
                if item.enabled
            ]
            selected_metrics = [
                item
                for item in all_metrics
                if str(item.workspace_id or selected_workspace) == selected_workspace
            ]
            selected_trainings = [
                item
                for item in all_trainings
                if str(item.workspace_id or selected_workspace) == selected_workspace
            ]
            columns = [column for _, values in table_rows for column in values]
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
                } for item in selected_metrics],
                "training": [{
                    "id": str(item.id),
                    "question": item.question,
                    "sql": item.sql,
                    "workspace_id": selected_workspace,
                } for item in selected_trainings],
            }
            alias_manifest = load_alias_manifest()
            repositories: dict[str, dict[str, Any]] = {}
            manifests: dict[str, dict] = {}
            warmup_started = time.perf_counter()
            # Exclude one-time tokenizer/model initialization from the first
            # strategy's build cost so sequential profile order does not bias it.
            await embedding_client_manager.client.aembed_query(
                "Data Agent Point projection evaluation warmup"
            )
            warmup_ms = round((time.perf_counter() - warmup_started) * 1000, 1)
            for strategy in STRATEGIES:
                print(f"构建隔离 Point profile：{strategy}", flush=True)
                repositories[strategy], manifests[strategy] = await _build_profile(
                    strategy=strategy,
                    workspace_id=selected_workspace,
                    table_rows=table_rows,
                    all_metrics=all_metrics,
                    all_trainings=all_trainings,
                    alias_manifest=alias_manifest,
                )

            cases = [
                dict(case)
                for case in contract["cases"]
                if not only or str(case["id"]) in only
            ]
            profile_cases: dict[str, list[dict]] = {
                strategy: [] for strategy in STRATEGIES
            }
            embedding_latencies: list[float] = []
            column_versions = {
                strategy: manifests[strategy]["index_versions"]["column"]
                for strategy in STRATEGIES
            }
            for position, case in enumerate(cases, 1):
                print(
                    f"Point RAG {position}/{len(cases)} {case['id']}: {case['query']}",
                    flush=True,
                )
                embedding_started = time.perf_counter()
                embedding = await embedding_client_manager.client.aembed_query(
                    str(case["query"])
                )
                embedding_ms = round(
                    (time.perf_counter() - embedding_started) * 1000, 1
                )
                embedding_latencies.append(embedding_ms)
                for strategy in STRATEGIES:
                    case_started = time.perf_counter()
                    routes: dict[str, dict] = {}
                    degradations: list[dict] = []
                    leakage = 0
                    for route in ROUTES:
                        expected = _expected(case, route)
                        if not expected:
                            continue
                        rows, evidence, route_degradations, qdrant_ms = (
                            await _search_route(
                                route,
                                repositories[strategy][route],
                                embedding,
                                workspace_id=selected_workspace,
                                index_version=column_versions[strategy],
                                mode="hybrid",
                                rule_rerank=True,
                                query=str(case["query"]),
                                local_corpora=local_corpora,
                            )
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
                                key: item.get(key)
                                for key in (
                                    "id", "name", "question", "entity_id", "rank",
                                    "dense_rank", "dense_score", "sparse_rank",
                                    "sparse_score", "fusion_rank", "fusion_score",
                                    "fusion_method", "retrieval_sources",
                                    "matched_point_roles", "matched_points",
                                    "point_strategy", "business_key",
                                )
                                if item.get(key) is not None
                            } for item in rows],
                            "evidence": evidence,
                            "qdrant_ms": qdrant_ms,
                        })
                        routes[route] = route_metric
                        degradations.extend(route_degradations)
                    profile_cases[strategy].append({
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
                str(case["id"])
                for case in contract["base_cases"]
                if not only or str(case["id"]) in only
            }
            profiles = {}
            for strategy in STRATEGIES:
                rows = profile_cases[strategy]
                profiles[strategy] = {
                    "manifest": manifests[strategy],
                    "summary": _profile_summary(rows, base_ids),
                    "subsets": _subset_summary(rows),
                    "ambiguous_false_positives": _ambiguous_false_positives(rows),
                    "cases": rows,
                }
            full_contract = len(cases) == validation["total"]
            decision = select_strategy(profiles, full_contract=full_contract)
            comparison = _comparison(profiles, decision["selected_strategy"])
            generated_at = dt.datetime.now().astimezone()
            timestamp = generated_at.strftime("%Y%m%d_%H%M%S")
            report = {
                "artifact_schema": "data-agent-point-projection-ablation-v1",
                "available": True,
                "generated_at": generated_at.isoformat(timespec="seconds"),
                "workspace_id": selected_workspace,
                "source": "live_bge_m3_isolated_qdrant_ablation",
                "contract": validation,
                "frozen_variable": "entity_to_point_projection_only",
                "query_contract": point_experiment_query_contract(),
                "alias_manifest": {
                    "version": alias_manifest.get("version"),
                    "fingerprint": alias_manifest_fingerprint(alias_manifest),
                    "approved_alias_count": len(
                        alias_manifest.get("approved_aliases") or []
                    ),
                },
                "profiles": profiles,
                "comparison": comparison,
                "decision": decision,
                "embedding_latency_ms": {
                    "warmup": warmup_ms,
                    "p50": round(median(embedding_latencies), 1)
                    if embedding_latencies else 0.0,
                    "p95": _percentile(embedding_latencies, .95),
                },
                "summary": {
                    "gate_passed": decision["gate_passed"],
                    "selected_strategy": decision["selected_strategy"],
                    "full_contract": full_contract,
                    "requires_active_rebuild": decision["requires_active_rebuild"],
                },
            }
            output_path = (
                REPORT_PATH
                if full_contract
                else REPORT_ROOT / "point_projection_sample_latest.json"
            )
            report["report_path"] = str(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if full_contract:
                REPORT_MD_PATH.write_text(_markdown(report), encoding="utf-8")
                IMMUTABLE_REPORT_ROOT.mkdir(parents=True, exist_ok=True)
                immutable_json = (
                    IMMUTABLE_REPORT_ROOT / f"point_projection_{timestamp}.json"
                )
                immutable_md = (
                    IMMUTABLE_REPORT_ROOT / f"point_projection_{timestamp}.md"
                )
                immutable_json.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                immutable_md.write_text(_markdown(report), encoding="utf-8")
                report["immutable_report_path"] = str(immutable_json)
            if enforce and not decision["gate_passed"]:
                raise RuntimeError("TODO-57 Point 投影消融门禁未通过")
            return report
    finally:
        await embedding_client_manager.close()
        await meta_mysql_client_manager.close()
        await qdrant_client_manager.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data Agent TODO-57 Point 投影策略隔离消融评测"
    )
    parser.add_argument("--workspace-id", default=None)
    parser.add_argument(
        "--only",
        default="",
        help="仅运行逗号分隔的 case id；抽样不会覆盖 latest 正式制品",
    )
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    selected = {
        item.strip() for item in args.only.split(",") if item.strip()
    } or None
    result = asyncio.run(run(
        args.workspace_id,
        only=selected,
        enforce=args.enforce,
    ))
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Point 投影报告：{result['report_path']}")
