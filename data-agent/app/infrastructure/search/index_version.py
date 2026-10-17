"""RAG 索引版本与候选证据的统一契约。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Iterable, Mapping
from typing import Any

from app.config.app_config import app_config
from app.infrastructure.search.projection import (
    alias_manifest_fingerprint,
    validate_point_strategy,
)


PROJECT_ROOT = Path(__file__).parents[3]


def _embedding_metadata_fingerprint() -> str:
    model_path = Path(str(app_config.embedding.model_path))
    digest = hashlib.sha256()
    found = False
    for name in ("config.json", "tokenizer_config.json", "special_tokens_map.json"):
        path = model_path / name
        if not path.is_file():
            continue
        found = True
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return f"sha256:{digest.hexdigest()}" if found else "missing-model-metadata"


def retrieval_index_parameters(
    *,
    point_strategy: str | None = None,
    alias_manifest_hash: str | None = None,
) -> dict:
    """Only parameters that change materialized Qdrant vectors/payloads.

    Query thresholds, candidate limits, RRF and context budgets are deliberately
    excluded: changing them must invalidate query caches and experiment
    fingerprints, but must not make already-published vectors invisible.
    """
    selected_strategy = validate_point_strategy(point_strategy)
    return {
        "embedding_contract": "data-agent-bge-m3-dense-sparse-v1",
        "embedding_model": app_config.embedding.model,
        "embedding_revision": app_config.embedding.revision,
        "embedding_metadata": _embedding_metadata_fingerprint(),
        "embedding_mode": "dense_sparse",
        "embedding_size": app_config.embedding.dense_size,
        "embedding_max_length": app_config.embedding.max_length,
        "normalize_dense": app_config.embedding.normalize_dense,
        "dense_vector_name": app_config.qdrant.dense_vector_name,
        "sparse_vector_name": app_config.qdrant.sparse_vector_name,
        "dense_distance": "cosine",
        "point_strategy": selected_strategy,
        "point_identity_version": app_config.point_projection.identity_version,
        "point_namespace_uuid": app_config.point_projection.namespace_uuid,
        "canonical_template_version": app_config.point_projection.canonical_template_version,
        "alias_policy_version": app_config.point_projection.alias_policy_version,
        "alias_manifest_hash": alias_manifest_hash or alias_manifest_fingerprint(),
        "max_alias_points_per_entity": (
            app_config.point_projection.max_alias_points_per_entity
        ),
        "payload_contract": "hybrid-payload-v2",
    }


def retrieval_parameters() -> dict:
    query = app_config.query
    return {
        "index": retrieval_index_parameters(),
        "column": {"threshold": query.column_recall_score_threshold, "min": query.column_recall_min_limit,
                   "sparse_threshold": query.column_sparse_score_threshold,
                   "limit": query.column_recall_limit, "pool": query.column_recall_pool_limit},
        "metric": {"threshold": query.metric_recall_score_threshold, "min": query.metric_recall_min_limit,
                   "sparse_threshold": query.metric_sparse_score_threshold,
                   "limit": query.metric_recall_limit, "pool": query.metric_recall_pool_limit},
        "training": {"threshold": query.training_recall_score_threshold, "min": query.training_recall_min_limit,
                     "sparse_threshold": query.training_sparse_score_threshold,
                     "limit": query.training_recall_limit, "pool": query.training_recall_pool_limit},
        "value": {"min": query.value_recall_min_limit, "limit": query.value_recall_limit,
                  "pool": query.value_recall_pool_limit},
        "keyword_limit": query.retrieval_keyword_limit,
        "dynamic_score_gap": query.retrieval_dynamic_score_gap,
        "context_budget": {
            "total": query.context_total_token_budget,
            "schema": query.context_schema_token_budget,
            "metric": query.context_metric_token_budget,
            "training": query.context_training_token_budget,
            "max_columns_per_table": query.context_max_columns_per_table,
        },
        "fusion": {
            "method": "entity_rrf_then_rule_rerank",
            "rrf_k": query.hybrid_rrf_k,
            "oversample_factor": query.hybrid_oversample_factor,
            "allow_sparse_only_fallback": query.hybrid_allow_sparse_only_fallback,
        },
    }


def retrieval_runtime_fingerprint(rerank_version: str) -> str:
    payload = {
        "parameters": retrieval_parameters(),
        "rerank_version": str(rerank_version),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


def retrieval_index_version(
    content: object | None = None,
    *,
    point_strategy: str | None = None,
    alias_manifest_hash: str | None = None,
) -> str:
    """Materialized content/model fingerprint, independent of query tuning."""
    meta_path = PROJECT_ROOT / "conf" / "meta_config.yaml"
    payload = {
        "index_parameters": retrieval_index_parameters(
            point_strategy=point_strategy,
            alias_manifest_hash=alias_manifest_hash,
        ),
        "meta_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest() if meta_path.exists() else "missing",
    }
    if content is not None:
        canonical = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        payload["published_content_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


def column_index_version(
    columns: Iterable[Mapping[str, Any] | Any],
    *,
    point_strategy: str | None = None,
    alias_manifest_hash: str | None = None,
) -> str:
    """返回字段集合的稳定索引版本，供构建和召回使用同一版本契约。"""
    def value(item: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
        return item.get(key, default) if isinstance(item, Mapping) else getattr(item, key, default)

    content = sorted(({
        "id": str(value(item, "id", "")),
        "name": str(value(item, "name", "")),
        "type": str(value(item, "type", "")),
        "role": str(value(item, "role", "")),
        "description": value(item, "description", "") or "",
        "alias": list(value(item, "alias", []) or []),
        "table_id": str(value(item, "table_id", "")),
        "is_display": bool(value(item, "is_display", False)),
    } for item in columns), key=lambda item: item["id"])
    return retrieval_index_version(
        content,
        point_strategy=point_strategy,
        alias_manifest_hash=alias_manifest_hash,
    )
