"""Shared Qdrant dense+sparse schema, persistence and entity-level fusion."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from app.config.app_config import app_config
from app.domain.retrieval.hybrid_embedding import HybridEmbedding
from app.foundation.observability import normalize_degradation_evidence


class HybridIndexSchemaError(RuntimeError):
    """An existing collection still uses the incompatible dense-only schema."""


@dataclass(slots=True)
class HybridSearchResult:
    items: list[dict] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    degradations: list[dict] = field(default_factory=list)


def _named_config(value: Any, name: str) -> Any | None:
    return value.get(name) if isinstance(value, Mapping) else None


async def collection_status(client: AsyncQdrantClient, collection_name: str) -> dict:
    exists = await client.collection_exists(collection_name=collection_name)
    base = {
        "collection": collection_name,
        "exists": bool(exists),
        "compatible": False,
        "dense_vector": app_config.qdrant.dense_vector_name,
        "sparse_vector": app_config.qdrant.sparse_vector_name,
        "points_count": 0,
    }
    if not exists:
        return base
    info = await client.get_collection(collection_name=collection_name)
    params = getattr(getattr(info, "config", None), "params", None)
    dense_config = _named_config(
        getattr(params, "vectors", None), app_config.qdrant.dense_vector_name
    )
    sparse_config = _named_config(
        getattr(params, "sparse_vectors", None), app_config.qdrant.sparse_vector_name
    )
    dense_size = getattr(dense_config, "size", None)
    return {
        **base,
        "compatible": bool(
            dense_config is not None
            and sparse_config is not None
            and int(dense_size or 0) == int(app_config.embedding.dense_size)
        ),
        "dense_size": dense_size,
        "points_count": int(getattr(info, "points_count", 0) or 0),
        "status": str(getattr(info, "status", "") or ""),
    }


async def create_hybrid_collection(
    client: AsyncQdrantClient, collection_name: str, *, recreate: bool = False
) -> None:
    status = await collection_status(client, collection_name)
    if status["exists"] and recreate:
        await client.delete_collection(collection_name=collection_name)
        status["exists"] = False
    if status["exists"]:
        if not status["compatible"]:
            raise HybridIndexSchemaError(
                f"{collection_name} 仍是旧 Dense Schema；"
                "请在元知识页面执行“覆盖重建 BGE-M3 混合索引”"
            )
        return
    await client.create_collection(
        collection_name=collection_name,
        vectors_config={
            app_config.qdrant.dense_vector_name: models.VectorParams(
                size=int(app_config.embedding.dense_size),
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            app_config.qdrant.sparse_vector_name: models.SparseVectorParams(
                modifier=None
            )
        },
    )


def _source_hash(text: str) -> str:
    return f"sha256:{hashlib.sha256(str(text).encode('utf-8')).hexdigest()}"


def hybrid_payload(
    payload: Mapping[str, Any], *, source_text: str, point_role: str
) -> dict:
    item = dict(payload)
    item.update({
        "entity_id": str(item.get("entity_id") or item.get("id") or ""),
        "point_role": str(point_role or "canonical"),
        "embedding_model": app_config.embedding.model,
        "embedding_revision": app_config.embedding.revision,
        "embedding_mode": "dense_sparse",
        "source_text_hash": _source_hash(source_text),
    })
    if not item["entity_id"]:
        raise ValueError("Qdrant hybrid payload 缺少 entity_id/id")
    missing = [
        field for field in ("point_strategy", "index_version")
        if not str(item.get(field) or "").strip()
    ]
    if missing:
        raise ValueError(
            "Qdrant hybrid payload 缺少当前投影契约字段：" + ", ".join(missing)
        )
    return item


async def upsert_hybrid_points(
    client: AsyncQdrantClient,
    collection_name: str,
    embeddings: Sequence[HybridEmbedding],
    payloads: Sequence[Mapping[str, Any]],
    ids: Sequence[Any],
    *,
    source_texts: Sequence[str] | None = None,
    point_roles: Sequence[str] | None = None,
    batch_size: int = 10,
) -> None:
    total = len(embeddings)
    if len(payloads) != total or len(ids) != total:
        raise ValueError("embedding/payload/id 数量不一致")
    source_texts = list(source_texts or [
        str(payload.get("name") or payload.get("question") or payload.get("id") or "")
        for payload in payloads
    ])
    point_roles = list(point_roles or ["canonical"] * total)
    if len(source_texts) != total or len(point_roles) != total:
        raise ValueError("source_text/point_role 数量不一致")
    await create_hybrid_collection(client, collection_name)
    for start in range(0, total, max(1, int(batch_size))):
        points = []
        for index in range(start, min(total, start + max(1, int(batch_size)))):
            embedding = embeddings[index]
            points.append(models.PointStruct(
                id=ids[index],
                payload=hybrid_payload(
                    payloads[index],
                    source_text=source_texts[index],
                    point_role=point_roles[index],
                ),
                vector={
                    app_config.qdrant.dense_vector_name: list(embedding.dense),
                    app_config.qdrant.sparse_vector_name: models.SparseVector(
                        indices=list(embedding.sparse_indices),
                        values=list(embedding.sparse_values),
                    ),
                },
            ))
        await client.upsert(collection_name=collection_name, points=points)


def _aggregate_leg(points: Sequence[Any], leg: str) -> tuple[dict[str, dict], dict]:
    entities: dict[str, dict] = {}
    for point_rank, point in enumerate(points, 1):
        payload = dict(getattr(point, "payload", None) or {})
        entity_id = str(payload.get("entity_id") or payload.get("id") or "")
        if not entity_id:
            continue
        missing = [
            field for field in ("point_role", "point_strategy", "index_version")
            if not str(payload.get(field) or "").strip()
        ]
        if missing:
            raise HybridIndexSchemaError(
                "检索命中未版本化 Point（缺少 "
                + ", ".join(missing)
                + "）；请覆盖重建当前混合索引"
            )
        point_row = {
            "point_id": str(getattr(point, "id", "")),
            "point_rank": point_rank,
            "point_role": str(payload["point_role"]),
            "score": float(getattr(point, "score", 0) or 0),
            "point_strategy": str(payload["point_strategy"]),
            "business_key": str(payload.get("business_key") or ""),
            "projection_hash": str(payload.get("projection_hash") or ""),
            "projected_alias": payload.get("projected_alias"),
        }
        row = entities.get(entity_id)
        if row is None:
            row = {
                **payload,
                "entity_id": entity_id,
                f"{leg}_rank": point_rank,
                f"{leg}_score": point_row["score"],
                f"{leg}_point_id": point_row["point_id"],
                f"{leg}_points": [point_row],
            }
            entities[entity_id] = row
        else:
            row[f"{leg}_points"].append(point_row)
    raw_count = len(points)
    return entities, {
        "raw_points": raw_count,
        "unique_entities": len(entities),
        "duplicate_occupancy": round(
            max(0, raw_count - len(entities)) / max(1, raw_count), 4
        ),
    }


def _leg_failure(route: str, leg: str, error: BaseException, *, critical: bool) -> dict:
    return normalize_degradation_evidence({
        "node_id": f"recall_{route}",
        "route": "retrieval",
        "reason_code": f"{leg}_retrieval_failed",
        "severity": "critical" if critical else "warning",
        "affects_core_path": True,
        "detail": f"{type(error).__name__}: {str(error)[:220]}",
    })


def _fuse_entities(
    dense_entities: Mapping[str, dict],
    sparse_entities: Mapping[str, dict],
    *,
    mode: str,
    rrf_k: int,
) -> list[dict]:
    identifiers = (
        set(dense_entities) if mode == "dense"
        else set(sparse_entities) if mode == "sparse"
        else set(dense_entities) | set(sparse_entities)
    )
    rows: list[dict] = []
    for entity_id in identifiers:
        dense = dense_entities.get(entity_id)
        sparse = sparse_entities.get(entity_id)
        base = dict(dense or sparse or {})
        base.pop("dense_points", None)
        base.pop("sparse_points", None)
        dense_rank = (dense or {}).get("dense_rank")
        sparse_rank = (sparse or {}).get("sparse_rank")
        dense_contribution = (
            1.0 / (rrf_k + int(dense_rank)) if dense_rank is not None else 0.0
        )
        sparse_contribution = (
            1.0 / (rrf_k + int(sparse_rank)) if sparse_rank is not None else 0.0
        )
        if mode == "dense":
            fusion_score = dense_contribution
            maximum = 1.0 / (rrf_k + 1)
            sources = ["dense"]
        elif mode == "sparse":
            fusion_score = sparse_contribution
            maximum = 1.0 / (rrf_k + 1)
            sources = ["sparse"]
        else:
            fusion_score = dense_contribution + sparse_contribution
            maximum = 2.0 / (rrf_k + 1)
            sources = [
                leg for leg, present in (("dense", dense), ("sparse", sparse)) if present
            ]
        points = [
            *(dense or {}).get("dense_points", []),
            *(sparse or {}).get("sparse_points", []),
        ]
        point_roles = sorted({
            str(item.get("point_role") or "") for item in points if item.get("point_role")
        })
        base.update({
            "entity_id": entity_id,
            # Representative point id for compact traces; matched_points remains
            # the authoritative entity-level evidence.
            "point_id": (
                (dense or {}).get("dense_point_id")
                or (sparse or {}).get("sparse_point_id")
                or ""
            ),
            "dense_rank": dense_rank,
            "dense_score": (dense or {}).get("dense_score"),
            "sparse_rank": sparse_rank,
            "sparse_score": (sparse or {}).get("sparse_score"),
            "fusion_score": round(fusion_score, 8),
            "fusion_method": (
                "entity_rrf" if mode == "hybrid" else f"{mode}_entity_rank"
            ),
            "score": round(fusion_score / max(maximum, 1e-12), 8),
            "matched_point_roles": point_roles,
            "matched_points": points,
            "retrieval_sources": sources,
            "index_version": str(base["index_version"]),
        })
        rows.append(base)
    rows.sort(key=lambda item: (
        -float(item.get("fusion_score") or 0),
        int(item.get("dense_rank") or 10**9),
        int(item.get("sparse_rank") or 10**9),
        str(item.get("entity_id") or ""),
    ))
    for rank, item in enumerate(rows, 1):
        item["fusion_rank"] = rank
        item["rank"] = rank
    return rows


async def search_hybrid_entities(
    client: AsyncQdrantClient,
    collection_name: str,
    embedding: HybridEmbedding,
    *,
    route: str,
    query_filter: models.Filter,
    dense_score_threshold: float | None,
    sparse_score_threshold: float | None,
    limit: int,
    mode: str = "hybrid",
) -> HybridSearchResult:
    """Query both named vectors, dedupe each leg by entity, then apply entity RRF."""
    if mode not in {"dense", "sparse", "hybrid"}:
        raise ValueError(f"未知混合检索模式：{mode}")
    await create_hybrid_collection(client, collection_name)
    oversampled = max(
        int(limit),
        int(limit) * max(1, int(app_config.query.hybrid_oversample_factor)),
    )

    async def dense_query():
        return await client.query_points(
            collection_name=collection_name,
            query=list(embedding.dense),
            using=app_config.qdrant.dense_vector_name,
            score_threshold=dense_score_threshold,
            limit=oversampled,
            query_filter=query_filter,
            with_payload=True,
        )

    async def sparse_query():
        return await client.query_points(
            collection_name=collection_name,
            query=models.SparseVector(
                indices=list(embedding.sparse_indices),
                values=list(embedding.sparse_values),
            ),
            using=app_config.qdrant.sparse_vector_name,
            score_threshold=sparse_score_threshold,
            limit=oversampled,
            query_filter=query_filter,
            with_payload=True,
        )

    dense_result: Any = None
    sparse_result: Any = None
    if mode == "dense":
        dense_result = await asyncio.gather(dense_query(), return_exceptions=True)
        dense_result = dense_result[0]
    elif mode == "sparse":
        sparse_result = await asyncio.gather(sparse_query(), return_exceptions=True)
        sparse_result = sparse_result[0]
    else:
        dense_result, sparse_result = await asyncio.gather(
            dense_query(), sparse_query(), return_exceptions=True
        )

    degradations: list[dict] = []
    dense_error = dense_result if isinstance(dense_result, BaseException) else None
    sparse_error = sparse_result if isinstance(sparse_result, BaseException) else None
    if dense_error:
        degradations.append(_leg_failure(route, "dense", dense_error, critical=True))
    if sparse_error:
        degradations.append(_leg_failure(route, "sparse", sparse_error, critical=False))

    dense_points = (
        [] if dense_result is None or dense_error
        else list(getattr(dense_result, "points", None) or [])
    )
    sparse_points = (
        [] if sparse_result is None or sparse_error
        else list(getattr(sparse_result, "points", None) or [])
    )
    dense_entities, dense_stats = _aggregate_leg(dense_points, "dense")
    sparse_entities, sparse_stats = _aggregate_leg(sparse_points, "sparse")

    effective_mode = mode
    if mode == "hybrid" and dense_error:
        if app_config.query.hybrid_allow_sparse_only_fallback and not sparse_error:
            effective_mode = "sparse"
        else:
            sparse_entities = {}
            effective_mode = "hybrid"
    elif mode == "hybrid" and sparse_error and not dense_error:
        effective_mode = "dense"

    items = _fuse_entities(
        dense_entities, sparse_entities,
        mode=effective_mode,
        rrf_k=max(1, int(app_config.query.hybrid_rrf_k)),
    )[:max(1, int(limit))]
    evidence = {
        "retrieval_mode": mode,
        "effective_mode": effective_mode,
        "fusion_method": (
            "entity_rrf" if effective_mode == "hybrid"
            else f"{effective_mode}_entity_rank"
        ),
        "rrf_k": int(app_config.query.hybrid_rrf_k),
        "oversample_factor": int(app_config.query.hybrid_oversample_factor),
        "physical_limit": oversampled,
        "dense": dense_stats,
        "sparse": sparse_stats,
        "intersection_entities": len(set(dense_entities) & set(sparse_entities)),
        "dense_only_entities": len(set(dense_entities) - set(sparse_entities)),
        "sparse_only_entities": len(set(sparse_entities) - set(dense_entities)),
        "fused_entities": len(items),
        "degradation_count": len(degradations),
        "point_strategies": sorted({
            str(item["point_strategy"]) for item in items
        }),
        "matched_point_roles": sorted({
            str(role)
            for item in items
            for role in (item.get("matched_point_roles") or [])
            if role
        }),
    }
    return HybridSearchResult(items=items, evidence=evidence, degradations=degradations)
