"""Dynamic retrieval, rule reranking, safe caches and prompt-context budgets."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
import threading
import time
import unicodedata
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

from app.domain.retrieval.hybrid_embedding import HybridEmbedding

RERANK_VERSION = "rule-rerank-v7"
CONTEXT_BUDGET_VERSION = "context-budget-v1"

_DEFAULT_QUERY_SETTINGS = {
    "retrieval_dynamic_score_gap": 0.08,
    "retrieval_cache_ttl_seconds": 120,
    "retrieval_cache_max_entries": 512,
    "retrieval_cache_enabled": True,
    "column_recall_min_limit": 3,
    "column_recall_limit": 8,
    "column_recall_pool_limit": 16,
    "column_recall_score_threshold": 0.35,
    "column_sparse_score_threshold": 0.0,
    "metric_recall_min_limit": 1,
    "metric_recall_limit": 5,
    "metric_recall_pool_limit": 10,
    "metric_recall_score_threshold": 0.58,
    "metric_sparse_score_threshold": 0.0,
    "training_recall_min_limit": 1,
    "training_recall_limit": 3,
    "training_recall_pool_limit": 6,
    "training_recall_score_threshold": 0.62,
    "training_sparse_score_threshold": 0.0,
    "value_recall_min_limit": 2,
    "value_recall_limit": 8,
    "value_recall_pool_limit": 12,
    "context_schema_token_budget": 1300,
    "context_metric_token_budget": 350,
    "context_training_token_budget": 450,
    "context_total_token_budget": 2200,
    "context_max_columns_per_table": 12,
}
_DEFAULT_EMBEDDING_SETTINGS = {
    "model": "unregistered",
    "revision": "local",
    "dense_size": 1024,
    "max_length": 512,
    "normalize_dense": True,
}


def _setting(section: Any | None, name: str, defaults: Mapping[str, Any]) -> Any:
    return defaults[name] if section is None else getattr(section, name, defaults[name])


def embedding_runtime_evidence(client: Any) -> dict[str, Any]:
    """Privacy-safe scheduler snapshot for retrieval Trace events."""
    try:
        status = client.status()
    except Exception:  # noqa: BLE001  # Trace 快照失败必须降级为不可用证据
        return {"runtime_profile": "unavailable", "status": "unavailable"}
    scheduling = dict(status.get("scheduling") or {})
    queue = dict(scheduling.get("queue") or {})
    metrics = dict(scheduling.get("metrics") or {})
    epoch = dict(metrics.get("epoch") or {})
    latency = dict(metrics.get("latency_ms") or {})
    batching = dict(metrics.get("batching") or {})
    requests = dict(metrics.get("requests") or {})
    return {
        "runtime_profile": status.get("runtime_profile"),
        "runtime_profile_fingerprint": status.get("runtime_profile_fingerprint"),
        "metrics_epoch_schema": epoch.get("schema"),
        "metrics_epoch_id": epoch.get("id"),
        "metrics_epoch_index": epoch.get("index"),
        "metrics_epoch_started_at": epoch.get("started_at"),
        "status": scheduling.get("state"),
        "queue_depth": queue.get("depth"),
        "queue_capacity": queue.get("capacity"),
        "max_queue_depth": queue.get("max_depth_seen"),
        "last_batch_texts": batching.get("last_batch_texts"),
        "max_batch_texts_seen": batching.get("max_batch_texts_seen"),
        "merged_batch_count": batching.get("merged_batch_count"),
        "queue_wait_p95_ms": dict(latency.get("queue_wait") or {}).get("p95"),
        "inference_p95_ms": dict(latency.get("inference") or {}).get("p95"),
        "total_p95_ms": dict(latency.get("total") or {}).get("p95"),
        "rejected": requests.get("rejected"),
        "cancelled": requests.get("cancelled"),
        "timed_out": requests.get("timed_out"),
        "failed": requests.get("failed"),
        "model_instances": status.get("model_instances"),
        "raw_text_logged": False,
    }


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def _terms(item: Mapping, route: str) -> list[str]:
    values: list[Any] = [item.get("name"), item.get("description")]
    if route == "training":
        values.extend((item.get("question"),))
    elif route == "value":
        values.extend((item.get("value"), item.get("column_name")))
    values.extend(item.get("alias") or [])
    return list(dict.fromkeys(normalize_text(value) for value in values if normalize_text(value)))


def _identifier(item: Mapping, route: str) -> str:
    if route == "training":
        return str(item.get("id") or item.get("question") or "")
    return str(item.get("id") or item.get("name") or "")


def match_features(query: str, keywords: Sequence[str], item: Mapping, route: str) -> dict:
    query_text = normalize_text(query)
    terms = _terms(item, route)
    name = normalize_text(item.get("name") or item.get("question") or item.get("value") or "")
    aliases = {normalize_text(value) for value in (item.get("alias") or []) if normalize_text(value)}
    exact_name = bool(name and name in query_text)
    alias_hit = any(alias in query_text for alias in aliases)
    normalized_keywords = [normalize_text(value) for value in keywords if normalize_text(value)]
    keyword_hits = sum(
        1 for keyword in normalized_keywords
        if keyword in " ".join(terms) or any(term in keyword for term in terms if len(term) >= 2)
    )
    lexical_coverage = keyword_hits / max(1, len(normalized_keywords))
    raw_score = max(0.0, float(item.get("score") or 0.0))
    vector_score = raw_score if raw_score <= 1 else raw_score / (raw_score + 10.0)
    display = bool(item.get("is_display"))
    role = str(item.get("role") or "").lower()
    table_id = str(item.get("table_id") or "").lower()
    if not table_id:
        identifier = str(item.get("id") or "")
        table_id = identifier.split(".", 1)[0].lower() if "." in identifier else ""
    metric_relation = bool(route == "metric" and item.get("relevant_columns"))
    join_key = role in {"primary_key", "foreign_key"} or str(item.get("name") or "").lower().endswith("_id")
    item_name = str(item.get("name") or "").lower()
    entity_request = any(
        marker in query_text
        for marker in ("每种", "每个", "哪些", "列表", "最高", "最低", "前", "后", "排名", "没有", "从未")
    )
    entity_display_hit = bool(
        entity_request
        and (
            (item_name == "product_name" and any(token in query_text for token in ("商品", "产品", "product")))
            or (item_name == "customer_name" and any(token in query_text for token in ("客户", "用户", "customer")))
            or (item_name == "region_name" and any(token in query_text for token in ("地区", "区域", "大区", "region")))
        )
    )
    anti_join_key_hit = bool(
        any(marker in query_text for marker in ("没有", "从未", "未产生", "未发生"))
        and (
            (item_name == "product_id" and any(token in query_text for token in ("商品", "产品", "product")))
            or (item_name == "customer_id" and any(token in query_text for token in ("客户", "用户", "customer")))
        )
    )
    semantic_hit = bool(route == "column" and (
        (item_name == "order_amount" and any(token in query_text for token in (
            "销售额", "销售总额", "成交额", "成交金额", "消费", "消费金额", "订单金额", "gmv"
        )))
        or (item_name == "order_quantity" and any(token in query_text for token in (
            "销量", "销售量", "购买数量", "件数", "售出", "卖出"
        )))
        or (item_name == "year" and re.search(r"\d{4}年", query_text))
        # Counts/rates over customers are event-population semantics on the
        # fact side.  Marking the dimension PK as an equally strong semantic
        # hit adds a second population to the prompt and has repeatedly caused
        # incorrect denominators.  Anti-join questions remain covered by the
        # dedicated two-sided key rule below, while relation closure restores
        # any dimension PK needed only for a JOIN.
        or (
            item_name == "customer_id"
            and table_id.startswith("fact_")
            and any(token in query_text for token in (
                "复购", "活跃用户", "活跃客户", "用户数", "客户数", "客户数量"
            ))
        )
        or (item_name == "order_id" and any(token in query_text for token in (
            "复购", "订单量", "订单数", "订单数量"
        )))
        or entity_display_hit
        or anti_join_key_hit
    ))
    score = (
        vector_score * 0.52
        + float(exact_name) * 0.34
        + float(alias_hit) * 0.28
        + lexical_coverage * 0.20
        + float(display) * 0.07
        + float(metric_relation) * 0.04
        + float(join_key) * 0.01
        + float(semantic_hit) * 0.32
    )
    return {
        "exact_name": exact_name,
        "alias_hit": alias_hit,
        "keyword_hits": keyword_hits,
        "lexical_coverage": round(lexical_coverage, 4),
        "vector_score": round(vector_score, 6),
        "display": display,
        "metric_relation": metric_relation,
        "join_key": join_key,
        "semantic_hit": semantic_hit,
        "strong_match": bool(exact_name or alias_hit or semantic_hit),
        "rerank_score": round(score, 6),
    }


def has_strong_match(query: str, keywords: Sequence[str], candidates: Sequence[Mapping], route: str) -> bool:
    return any(match_features(query, keywords, item, route)["strong_match"] for item in candidates)


def rerank_and_cut(
    route: str,
    query: str,
    keywords: Sequence[str],
    candidates: Sequence[Mapping],
    *,
    min_limit: int,
    max_limit: int,
    score_gap: float | None = None,
    query_settings: Any | None = None,
) -> tuple[list[dict], dict]:
    """Merge exact/lexical/vector candidates and choose an explainable dynamic K."""
    def merge_candidate_evidence(target: dict, source: Mapping) -> None:
        """Retain exact and both vector-leg evidence when one entity is deduped."""
        sources = set(target.get("retrieval_sources") or [])
        sources.update(source.get("retrieval_sources") or [])
        if target.get("_retrieval_source"):
            sources.add(str(target["_retrieval_source"]))
        if source.get("_retrieval_source"):
            sources.add(str(source["_retrieval_source"]))
        if sources:
            target["retrieval_sources"] = sorted(sources)

        matched_points: dict[str, dict] = {}
        for point in [
            *(target.get("matched_points") or []),
            *(source.get("matched_points") or []),
        ]:
            if not isinstance(point, Mapping):
                continue
            key = str(point.get("point_id") or (
                f"{point.get('point_role')}:{point.get('score')}:{point.get('point_rank')}"
            ))
            matched_points[key] = dict(point)
        if matched_points:
            target["matched_points"] = list(matched_points.values())
            target["matched_point_roles"] = sorted({
                str(point.get("point_role"))
                for point in matched_points.values()
                if point.get("point_role")
            })

        for key in (
            "entity_id", "dense_rank", "dense_score", "dense_point_id",
            "sparse_rank", "sparse_score", "sparse_point_id", "fusion_rank",
            "fusion_score", "fusion_method",
        ):
            if target.get(key) is None and source.get(key) is not None:
                target[key] = source.get(key)

    deduped: dict[str, dict] = {}
    for raw in candidates:
        identifier = _identifier(raw, route)
        if not identifier:
            continue
        item = copy.deepcopy(dict(raw))
        previous = deduped.get(identifier)
        if previous is None:
            deduped[identifier] = item
            continue
        if float(item.get("score") or 0) > float(previous.get("score") or 0):
            merge_candidate_evidence(item, previous)
            deduped[identifier] = item
        else:
            merge_candidate_evidence(previous, item)

    ranked: list[dict] = []
    for item in deduped.values():
        features = match_features(query, keywords, item, route)
        sources = set(item.pop("retrieval_sources", []) or [])
        source = item.pop("_retrieval_source", None)
        if source:
            sources.add(str(source))
        item.update({
            "match_features": features,
            "rerank_score": features["rerank_score"],
            "retrieval_sources": sorted(sources or {"vector"}),
        })
        ranked.append(item)
    ranked.sort(key=lambda item: (
        -int(bool((item.get("match_features") or {}).get("strong_match"))),
        -float(item.get("rerank_score") or 0),
        -float(item.get("score") or 0),
        _identifier(item, route),
    ))

    maximum = max(1, int(max_limit))
    minimum = max(1, min(int(min_limit), maximum))
    query_text = normalize_text(query)
    complexity_markers = (
        "同比", "环比", "趋势", "对比", "占比", "复购", "没有", "从未", "超过",
        "各", "每个", "每种", "按", "排名", "最高", "最低", "前", "后",
        "活跃", "用户数", "客户数", "订单量", "数量",
    )
    complexity = sum(1 for marker in complexity_markers if marker in query_text)
    complexity += max(0, len({normalize_text(item) for item in keywords if normalize_text(item)}) - 2) // 2
    strong_count = sum(1 for item in ranked if item["match_features"]["strong_match"])
    if strong_count:
        # Two or more independent exact/alias/semantic hits already form a
        # useful business-field set.  Do not force an exploratory vector tail
        # merely to satisfy the weak-recall minimum: metadata-owned relation
        # closure adds bridge tables and both JOIN keys later.  With only one
        # strong hit we keep the configured minimum so an unseen expression
        # still has room for one or two genuinely useful fuzzy candidates.
        if route == "column" and strong_count >= 2:
            target = min(maximum, strong_count)
            reason = "strong_semantic_compact"
        else:
            target = max(minimum, min(maximum, strong_count + 1))
            reason = (
                "single_strong_with_exploration"
                if route == "column"
                else "strong_exact_or_alias_with_exploration"
            )
    else:
        target = min(maximum, minimum + 1 + min(2, complexity))
        reason = "complexity_budget"
    if ranked and float(ranked[0].get("rerank_score") or 0) < 0.38:
        target = maximum
        reason = "weak_top_score_expand"

    gap = float(
        score_gap
        if score_gap is not None
        else _setting(
            query_settings,
            "retrieval_dynamic_score_gap",
            _DEFAULT_QUERY_SETTINGS,
        )
    )
    while target < min(maximum, len(ranked)):
        # Strong exact/alias evidence deliberately shrinks the vector tail. A close
        # but purely fuzzy candidate must not reopen the fixed-K noise problem.
        if strong_count and not ranked[target]["match_features"]["strong_match"]:
            break
        previous = float(ranked[target - 1].get("rerank_score") or 0)
        following = float(ranked[target].get("rerank_score") or 0)
        if previous - following > gap:
            break
        target += 1
        reason = f"{reason}+close_scores"

    selected = ranked[:target]
    for rank, item in enumerate(selected, 1):
        item["rank"] = rank
    evidence = {
        "version": RERANK_VERSION,
        "route": route,
        "raw_count": len(candidates),
        "dedup_count": len(ranked),
        "selected_count": len(selected),
        "dynamic_k": target,
        "min_limit": minimum,
        "max_limit": maximum,
        "strong_match_count": strong_count,
        "exploratory_tail_count": sum(
            1
            for item in selected
            if not item["match_features"]["strong_match"]
        ),
        "complexity": complexity,
        "cutoff_reason": reason,
        "pruned_count": max(0, len(ranked) - len(selected)),
    }
    return selected, evidence


class _TTLCache:
    def __init__(self) -> None:
        self._items: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        now = time.monotonic()
        with self._lock:
            row = self._items.get(key)
            if row is None:
                return None
            expires_at, value = row
            if expires_at <= now:
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return copy.deepcopy(value)

    def set(
        self,
        key: str,
        value: Any,
        *,
        ttl_seconds: int = 120,
        max_entries: int = 512,
    ) -> None:
        ttl = max(1, int(ttl_seconds))
        maximum = max(1, int(max_entries))
        with self._lock:
            self._items[key] = (time.monotonic() + ttl, copy.deepcopy(value))
            self._items.move_to_end(key)
            while len(self._items) > maximum:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


_embedding_cache = _TTLCache()
_candidate_cache = _TTLCache()
_context_cache = _TTLCache()
_embedding_flights: dict[tuple[int, str], asyncio.Task[dict[str, HybridEmbedding]]] = {}
_embedding_flights_lock = threading.RLock()
_retrieval_cache_generation = 0


def invalidate_retrieval_caches() -> None:
    global _retrieval_cache_generation
    # A cold invalidation is a generation boundary.  Do not cancel an encoder
    # already serving an existing waiter, but detach it from future callers and
    # prevent its eventual result from repopulating the new cold cache.
    with _embedding_flights_lock:
        _retrieval_cache_generation += 1
        _embedding_flights.clear()
        _embedding_cache.clear()
    _candidate_cache.clear()
    _context_cache.clear()


def _digest(payload: Mapping) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def capture_retrieval_cache_generation() -> int:
    """Capture a cold-cache generation before a retrieval request reads sources."""
    with _embedding_flights_lock:
        return _retrieval_cache_generation


def policy_scope_fingerprint(policy: Any) -> str:
    policies = []
    for item in getattr(policy, "policies", ()) or ():
        policies.append({
            "id": getattr(item, "id", None), "role": getattr(item, "role", None),
            "user_id": getattr(item, "user_id", None), "table": getattr(item, "table_pattern", None),
            "access": getattr(item, "access", None), "columns": getattr(item, "allowed_columns", None),
            "row_filter": getattr(item, "row_filter", None), "priority": getattr(item, "priority", None),
        })
    return _digest({
        "workspace_id": getattr(policy, "workspace_id", ""),
        "user_id": getattr(policy, "user_id", ""),
        "role": getattr(policy, "role", ""),
        "policies": policies,
    })


def candidate_cache_key(
    route: str,
    query: str,
    keywords: Sequence[str],
    workspace_id: str,
    policy_fingerprint: str,
    *,
    query_signal_profile: str,
    route_policy: str,
    ordered_remote_signals: Sequence[str],
    query_signal_fingerprint: str,
    cache_namespace: str,
    expansion_profile: str = "none",
    generation_token: int,
    index_version: str = "unversioned",
    model_namespace: str = "unregistered",
    query_settings: Any | None = None,
) -> str:
    return _digest({
        "kind": "candidate", "route": route, "query": normalize_text(query),
        "keywords": [
            normalize_text(value) for value in keywords if normalize_text(value)
        ],
        "ordered_remote_signals": [
            normalize_text(value) for value in ordered_remote_signals
            if normalize_text(value)
        ],
        "query_signal_profile": query_signal_profile,
        "route_policy": route_policy,
        "query_signal_fingerprint": query_signal_fingerprint,
        "cache_namespace": cache_namespace,
        "expansion_profile": expansion_profile,
        "workspace_id": workspace_id, "policy": policy_fingerprint,
        "index_version": str(index_version),
        "parameters": route_parameters(route, query_settings),
        "rerank": RERANK_VERSION,
        "model_namespace": str(model_namespace),
        # A request that started before a cold invalidation may finish later and
        # write its old candidate set.  Generation-scoped keys keep that stale
        # write unreachable without weakening route/profile namespace isolation.
        "cache_generation": int(generation_token),
    })


def candidate_cache_get(
    key: str,
    *,
    query_settings: Any | None = None,
) -> Any | None:
    if not bool(_setting(
        query_settings, "retrieval_cache_enabled", _DEFAULT_QUERY_SETTINGS
    )):
        return None
    return _candidate_cache.get(key)


def candidate_cache_set(
    key: str,
    value: Any,
    *,
    generation_token: int,
    query_settings: Any | None = None,
) -> bool:
    """CAS-write a candidate only if its source-read generation is still live."""
    if not bool(_setting(
        query_settings, "retrieval_cache_enabled", _DEFAULT_QUERY_SETTINGS
    )):
        return False
    with _embedding_flights_lock:
        if int(generation_token) != _retrieval_cache_generation:
            return False
        _candidate_cache.set(
            key,
            value,
            ttl_seconds=int(_setting(
                query_settings,
                "retrieval_cache_ttl_seconds",
                _DEFAULT_QUERY_SETTINGS,
            )),
            max_entries=int(_setting(
                query_settings,
                "retrieval_cache_max_entries",
                _DEFAULT_QUERY_SETTINGS,
            )),
        )
        return True


def _embedding_profile(
    client: Any,
    embedding_settings: Any | None = None,
) -> dict[str, Any]:
    return {
        "model": _setting(embedding_settings, "model", _DEFAULT_EMBEDDING_SETTINGS),
        "revision": _setting(embedding_settings, "revision", _DEFAULT_EMBEDDING_SETTINGS),
        "mode": "dense_sparse",
        "dense_size": _setting(embedding_settings, "dense_size", _DEFAULT_EMBEDDING_SETTINGS),
        "max_length": _setting(embedding_settings, "max_length", _DEFAULT_EMBEDDING_SETTINGS),
        "normalize_dense": _setting(
            embedding_settings, "normalize_dense", _DEFAULT_EMBEDDING_SETTINGS
        ),
        "profile_fingerprint": getattr(client, "profile_fingerprint", None),
    }


def _embedding_cache_key(
    client: Any,
    text: str,
    *,
    mode: str,
    embedding_settings: Any | None = None,
) -> str:
    """Key an exact vector independently from retrieval/rerank namespaces."""
    return _digest({
        "kind": "embedding",
        "profile": _embedding_profile(client, embedding_settings),
        "encode_mode": mode,
        "text": _embedding_text_identity(text),
    })


def _embedding_text_identity(value: str) -> str:
    """Preserve encoder-visible text while stabilizing canonical Unicode."""
    return unicodedata.normalize("NFC", str(value))


async def _load_embedding_flight(
    client: Any,
    keyed_texts: Sequence[tuple[str, str]],
    *,
    mode: str,
    generation: int,
    query_settings: Any | None,
) -> dict[str, HybridEmbedding]:
    texts = [text for _, text in keyed_texts]
    if mode == "query":
        vectors = await client.aembed_queries(texts)
    else:
        vectors = await client.aembed_documents(texts)
    if len(vectors) != len(keyed_texts):
        raise RuntimeError(
            "BGE-M3 embedding 返回数量与输入不一致："
            f"输入 {len(keyed_texts)}，输出 {len(vectors)}"
        )
    result = {
        key: vector
        for (key, _), vector in zip(keyed_texts, vectors, strict=True)
    }
    if bool(_setting(
        query_settings, "retrieval_cache_enabled", _DEFAULT_QUERY_SETTINGS
    )):
        # Hold the generation lock through all cache writes so invalidate cannot
        # clear the cache between the generation check and a stale write-back.
        with _embedding_flights_lock:
            if generation == _retrieval_cache_generation:
                for key, vector in result.items():
                    _embedding_cache.set(
                        key,
                        vector,
                        ttl_seconds=int(_setting(
                            query_settings,
                            "retrieval_cache_ttl_seconds",
                            _DEFAULT_QUERY_SETTINGS,
                        )),
                        max_entries=int(_setting(
                            query_settings,
                            "retrieval_cache_max_entries",
                            _DEFAULT_QUERY_SETTINGS,
                        )),
                    )
    return result


def _finish_embedding_flight(
    task: asyncio.Task[dict[str, HybridEmbedding]],
    *,
    loop_id: int,
    keys: Sequence[str],
) -> None:
    with _embedding_flights_lock:
        for key in keys:
            flight_key = (loop_id, key)
            if _embedding_flights.get(flight_key) is task:
                _embedding_flights.pop(flight_key, None)
    # A caller may time out after starting the shared background task.  Retrieve
    # the terminal exception here so a failed orphan never produces
    # "Task exception was never retrieved"; surviving waiters can still await
    # the same task and receive the exception normally.
    if not task.cancelled():
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):  # noqa: BLE001, S110
            pass


async def embed_texts_cached(
    client: Any,
    texts: Sequence[str],
    *,
    mode: str = "documents",
    namespace: str = "default",
    query_settings: Any | None = None,
    embedding_settings: Any | None = None,
) -> tuple[list[HybridEmbedding], dict]:
    """Batch exact-vector misses and singleflight them across route namespaces.

    ``namespace`` remains a compatibility argument for recall callers, but it
    deliberately does not participate in the vector key.  Candidate and rerank
    cache keys retain their separate query-signal profile/route namespaces.
    """
    if not texts:
        return [], {"hits": 0, "misses": 0}
    if mode not in {"query", "documents"}:
        raise ValueError(f"未知 embedding encode mode：{mode}")
    del namespace
    output: list[HybridEmbedding | None] = [None] * len(texts)
    misses: dict[str, list[int]] = {}
    for index, text in enumerate(texts):
        key = _embedding_cache_key(
            client,
            text,
            mode=mode,
            embedding_settings=embedding_settings,
        )
        cache_enabled = bool(_setting(
            query_settings, "retrieval_cache_enabled", _DEFAULT_QUERY_SETTINGS
        ))
        cached = _embedding_cache.get(key) if cache_enabled else None
        if cached is not None:
            output[index] = cached
        else:
            misses.setdefault(key, []).append(index)
    initial_miss_count = sum(len(indexes) for indexes in misses.values())
    if misses:
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        flights: dict[str, asyncio.Task[dict[str, HybridEmbedding]]] = {}
        new_keyed_texts: list[tuple[str, str]] = []
        with _embedding_flights_lock:
            # Close the narrow race where a flight populated the cache after
            # this caller's first lookup but before it joined the registry.
            for key in list(misses):
                cached = (
                    _embedding_cache.get(key)
                    if cache_enabled
                    else None
                )
                if cached is not None:
                    for index in misses.pop(key):
                        output[index] = cached
                    continue
                flight_key = (loop_id, key)
                task = _embedding_flights.get(flight_key)
                if task is not None and task.get_loop() is not loop:
                    # Event-loop ids can theoretically be reused after a loop
                    # closes.  Never await a Task owned by another loop.
                    _embedding_flights.pop(flight_key, None)
                    task = None
                if task is not None:
                    flights[key] = task
                else:
                    first_index = misses[key][0]
                    new_keyed_texts.append((
                        key,
                        _embedding_text_identity(texts[first_index]),
                    ))

            if new_keyed_texts:
                generation = _retrieval_cache_generation
                task = loop.create_task(
                    _load_embedding_flight(
                        client,
                        tuple(new_keyed_texts),
                        mode=mode,
                        generation=generation,
                        query_settings=query_settings,
                    ),
                    name="data-agent-exact-vector-singleflight",
                )
                new_keys = tuple(key for key, _ in new_keyed_texts)
                for key in new_keys:
                    _embedding_flights[(loop_id, key)] = task
                    flights[key] = task
                task.add_done_callback(
                    lambda completed, *, keys=new_keys, owner=loop_id: (
                        _finish_embedding_flight(
                            completed,
                            loop_id=owner,
                            keys=keys,
                        )
                    )
                )

        unique_tasks = list(dict.fromkeys(flights.values()))
        task_results = await asyncio.gather(*(
            asyncio.shield(task) for task in unique_tasks
        ))
        vectors_by_key: dict[str, HybridEmbedding] = {}
        for result in task_results:
            vectors_by_key.update(result)
        for key, indexes in misses.items():
            vector = vectors_by_key.get(key)
            if vector is None:
                raise RuntimeError("BGE-M3 embedding singleflight 返回了不完整批次")
            for index in indexes:
                output[index] = vector
    if any(vector is None for vector in output):
        raise RuntimeError("BGE-M3 embedding cache 返回了不完整批次")
    cache_hits = len(texts) - initial_miss_count
    return [vector for vector in output if vector is not None], {
        "hits": cache_hits,
        "misses": initial_miss_count,
    }


def route_parameters(route: str, query_settings: Any | None = None) -> dict:
    def value(name: str) -> Any:
        return _setting(query_settings, name, _DEFAULT_QUERY_SETTINGS)

    return {
        "column": {"min": value("column_recall_min_limit"), "max": value("column_recall_limit"),
                   "pool": value("column_recall_pool_limit"), "threshold": value("column_recall_score_threshold"),
                   "sparse_threshold": value("column_sparse_score_threshold")},
        "metric": {"min": value("metric_recall_min_limit"), "max": value("metric_recall_limit"),
                   "pool": value("metric_recall_pool_limit"), "threshold": value("metric_recall_score_threshold"),
                   "sparse_threshold": value("metric_sparse_score_threshold")},
        "training": {"min": value("training_recall_min_limit"), "max": value("training_recall_limit"),
                     "pool": value("training_recall_pool_limit"), "threshold": value("training_recall_score_threshold"),
                     "sparse_threshold": value("training_sparse_score_threshold")},
        "value": {"min": value("value_recall_min_limit"), "max": value("value_recall_limit"),
                  "pool": value("value_recall_pool_limit")},
    }[route]


def estimate_tokens(value: Any) -> int:
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    # Chinese text is often near one token per character; use a conservative estimate.
    chinese = len(re.findall(r"[\u4e00-\u9fff]", raw))
    return max(1, chinese + (len(raw) - chinese + 3) // 4)


def _column_priority(query: str, table: str, column: Mapping, ir: Mapping) -> tuple[int, str]:
    name = str(column.get("name") or "")
    role = str(column.get("role") or "").lower()
    required = set()
    for item in [*(ir.get("dimensions") or []), *(ir.get("measures") or [])]:
        if str(item.get("table") or "") == table:
            required.update(str(item.get(key)) for key in ("column", "entity_column", "event_column") if item.get(key))
    for join in ir.get("joins") or []:
        if join.get("left_table") == table:
            required.add(str(join.get("left_column")))
        if join.get("right_table") == table:
            required.add(str(join.get("right_column")))
    if name in required:
        return 100, "semantic_ir"
    if role in {"primary_key", "foreign_key"} or name.lower().endswith("_id"):
        return 95, "join_or_permission_key"
    if bool(column.get("is_display")):
        return 90, "display_column"
    terms = [name, column.get("description", ""), *(column.get("alias") or [])]
    if any(normalize_text(term) and normalize_text(term) in normalize_text(query) for term in terms):
        return 85, "explicit_query_term"
    if role == "measure":
        return 70, "measure"
    if role == "dimension":
        return 60, "dimension"
    return 40, "remaining_budget"


def _compact_column(column: Mapping, query: str = "") -> dict:
    item = dict(column)
    item["description"] = str(item.get("description") or "")[:160]
    item["alias"] = list(item.get("alias") or [])[:5]
    examples = list(item.get("examples") or [])
    normalized_query = normalize_text(query)
    matched_examples = [
        value for value in examples
        if normalize_text(value) and normalize_text(value) in normalized_query
    ]
    # Exact values already supplied by the user are semantic evidence rather
    # than optional samples.  Keep them through the schema budget while never
    # exposing additional stored values merely to improve recall.
    item["examples"] = matched_examples[:8] if matched_examples else examples[:3]
    return item


def _pack_uncached(query: str, table_infos: Sequence[Mapping], metric_infos: Sequence[Mapping],
                   training: Sequence[Mapping], semantic_ir: Mapping,
                   query_settings: Any | None = None) -> tuple[list[dict], list[dict], list[dict], dict]:
    before = {
        "schema": estimate_tokens(table_infos),
        "metric": estimate_tokens(metric_infos),
        "training": estimate_tokens(training),
    }
    schema_budget = int(_setting(query_settings, "context_schema_token_budget", _DEFAULT_QUERY_SETTINGS))
    metric_budget = int(_setting(query_settings, "context_metric_token_budget", _DEFAULT_QUERY_SETTINGS))
    training_budget = int(_setting(query_settings, "context_training_token_budget", _DEFAULT_QUERY_SETTINGS))
    total_budget = int(_setting(query_settings, "context_total_token_budget", _DEFAULT_QUERY_SETTINGS))
    max_columns = int(_setting(query_settings, "context_max_columns_per_table", _DEFAULT_QUERY_SETTINGS))

    packed_tables: list[dict] = []
    trimmed_columns = 0
    preserved_reasons: dict[str, int] = {}
    schema_used = 0
    for table in table_infos:
        table_name = str(table.get("name") or "")
        ranked_columns = sorted(
            [(_column_priority(query, table_name, column, semantic_ir), _compact_column(column, query))
             for column in (table.get("columns") or [])],
            key=lambda row: (-row[0][0], str(row[1].get("name") or "")),
        )
        selected: list[dict] = []
        for (priority, reason), column in ranked_columns:
            mandatory = priority >= 85
            candidate = {**dict(table), "description": str(table.get("description") or "")[:200],
                         "columns": [*selected, column]}
            tokens = estimate_tokens(candidate)
            if mandatory or (len(selected) < max_columns and schema_used + tokens <= schema_budget):
                selected.append(column)
                preserved_reasons[reason] = preserved_reasons.get(reason, 0) + 1
            else:
                trimmed_columns += 1
        if selected:
            packed = {**dict(table), "description": str(table.get("description") or "")[:200], "columns": selected}
            packed_tables.append(packed)
            schema_used += estimate_tokens(packed)

    def pack_rows(rows: Sequence[Mapping], budget: int, *, metric: bool) -> tuple[list[dict], int]:
        ordered = sorted((copy.deepcopy(dict(row)) for row in rows), key=lambda item: (
            -int(any(normalize_text(term) in normalize_text(query) for term in (
                item.get("name", ""), *(item.get("alias") or []), item.get("question", "")
            ) if normalize_text(term))),
            int(item.get("rank") or 999),
            -float(item.get("rerank_score") or item.get("score") or 0),
        ))
        selected: list[dict] = []
        for item in ordered:
            if metric:
                item["description"] = str(item.get("description") or "")[:240]
                item["alias"] = list(item.get("alias") or [])[:5]
            if estimate_tokens([*selected, item]) <= budget or not selected:
                selected.append(item)
        return selected, max(0, len(ordered) - len(selected))

    packed_metrics, trimmed_metrics = pack_rows(metric_infos, metric_budget, metric=True)
    packed_training, trimmed_training = pack_rows(training, training_budget, metric=False)
    after = {
        "schema": estimate_tokens(packed_tables),
        "metric": estimate_tokens(packed_metrics),
        "training": estimate_tokens(packed_training),
    }
    total_after = sum(after.values())
    # The per-kind budgets are primary. If serialization overhead crosses the total,
    # drop optional training rows first; never drop IR/join/display fields.
    while len(packed_training) > 1 and total_after > total_budget:
        packed_training.pop()
        trimmed_training += 1
        after["training"] = estimate_tokens(packed_training)
        total_after = sum(after.values())
    while len(packed_metrics) > 1 and total_after > total_budget:
        packed_metrics.pop()
        trimmed_metrics += 1
        after["metric"] = estimate_tokens(packed_metrics)
        total_after = sum(after.values())

    evidence = {
        "version": CONTEXT_BUDGET_VERSION,
        "budget_tokens": {"total": total_budget, "schema": schema_budget,
                          "metric": metric_budget, "training": training_budget},
        "before_tokens": {**before, "total": sum(before.values())},
        "after_tokens": {**after, "total": sum(after.values())},
        "compression_rate": round(1 - sum(after.values()) / max(1, sum(before.values())), 4),
        "trimmed": {"columns": trimmed_columns, "metrics": trimmed_metrics, "training": trimmed_training},
        "preserved_reasons": preserved_reasons,
        "over_budget_mandatory": sum(after.values()) > total_budget,
        "cache_hit": False,
    }
    return packed_tables, packed_metrics, packed_training, evidence


def pack_prompt_context(query: str, table_infos: Sequence[Mapping], metric_infos: Sequence[Mapping],
                        training: Sequence[Mapping], semantic_ir: Mapping, policy_fingerprint: str,
                        *, query_settings: Any | None = None) -> tuple[list[dict], list[dict], list[dict], dict]:
    generation_token = capture_retrieval_cache_generation()
    payload = {
        "kind": "context", "query": normalize_text(query), "tables": table_infos,
        "metrics": metric_infos, "training": [
            {"id": item.get("id"), "rank": item.get("rank"), "index_version": item.get("index_version")}
            for item in training
        ],
        "semantic_ir": semantic_ir, "policy": policy_fingerprint,
        "version": CONTEXT_BUDGET_VERSION,
        "settings": {
            name: _setting(query_settings, name, _DEFAULT_QUERY_SETTINGS)
            for name in (
                "context_schema_token_budget",
                "context_metric_token_budget",
                "context_training_token_budget",
                "context_total_token_budget",
                "context_max_columns_per_table",
            )
        },
        # See candidate_cache_key: a late pre-invalidation writer remains in the
        # bounded TTL cache but is unreachable from the new cold generation.
        "cache_generation": generation_token,
    }
    key = _digest(payload)
    cache_enabled = bool(_setting(
        query_settings, "retrieval_cache_enabled", _DEFAULT_QUERY_SETTINGS
    ))
    cached = _context_cache.get(key) if cache_enabled else None
    if cached is not None:
        tables, metrics, examples, evidence = cached
        evidence["cache_hit"] = True
        return tables, metrics, examples, evidence
    result = _pack_uncached(
        query,
        table_infos,
        metric_infos,
        training,
        semantic_ir,
        query_settings,
    )
    if cache_enabled:
        with _embedding_flights_lock:
            if generation_token == _retrieval_cache_generation:
                _context_cache.set(
                    key,
                    result,
                    ttl_seconds=int(_setting(
                        query_settings,
                        "retrieval_cache_ttl_seconds",
                        _DEFAULT_QUERY_SETTINGS,
                    )),
                    max_entries=int(_setting(
                        query_settings,
                        "retrieval_cache_max_entries",
                        _DEFAULT_QUERY_SETTINGS,
                    )),
                )
    return result
