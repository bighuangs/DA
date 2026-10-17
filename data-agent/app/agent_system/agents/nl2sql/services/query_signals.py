"""TODO-61 query-signal extraction and route policy.

This module is deterministic, workspace-safe, and has no dependency on STT or
SQLBot.  It never mutates jieba's global dictionary.  Workspace terms are
matched with a request-local longest-match pass.
"""

from __future__ import annotations

import importlib.metadata
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from jieba.analyse import extract_tags, textrank

from app.agent_system.agents.nl2sql.contracts.query_signals import (
    QUERY_SIGNAL_CONTRACT_VERSION,
    QuerySignalProvenance,
    QuerySignalsV1,
    normalize_signal,
    stable_signal_hash,
)


EXTRACTOR_VERSION = "query-signal-extractor-v2"
RULE_VERSION = "query-signal-rules-v2"
ROUTE_POLICY_VERSION = "query-signal-routing-v2"
PROFILE_VERSIONS = {
    "full_query_only": "k0-v1",
    "jieba_tfidf_v1": "k1-v1",
    "domain_signals_v1": "k2-v2",
    "jieba_textrank_v1": "k3-v1",
}
ROUTE_POLICIES = {
    "full_query_only_v1",
    "current_multi_query_v1",
    "routed_signals_v1",
}
ALLOW_POS = (
    "n", "nr", "ns", "nt", "nz", "v", "vn", "a", "an", "eng", "i", "l",
)
OPERATORS = (
    "同比增长率", "同比增长", "同比", "环比增长率", "环比增长", "环比",
    "销售贡献率", "贡献率", "占比", "比重", "份额", "比例",
    "增长百分比", "变化百分比", "增长率", "变化率", "增幅", "差额", "差值",
    "累计占比", "累计", "滚动", "移动平均", "去重", "平均", "人均",
    "每位", "每单", "客单价", "每组", "组内", "各自", "分组",
    "首次购买", "首次下单", "第一笔", "首购", "最后一次", "最后一笔", "末次",
    "复购", "回购", "留存", "流失", "交集", "并集", "同时满足", "同时",
    "分层", "分档", "分段", "最高", "最低", "排序", "排名", "趋势", "对比", "增长",
)
POLARITY = (
    "从来没有", "从未购买", "从未下单", "未产生", "未发生", "未曾",
    "尚未", "没买过", "未购买", "未下单", "不要", "排除", "剔除",
    "不含", "之外", "除外", "但未", "却未", "均未", "只看", "仅看",
    "只在", "仅在", "从未", "没有", "未支付", "已支付",
)
REGION_LITERALS = (
    "华东", "华南", "华北", "华中", "东北", "西北", "西南",
)
TIME_PATTERNS = (
    r"\d{4}年(?:\d{1,2}月(?:\d{1,2}[日号])?)?",
    r"最近\s*\d+\s*(?:天|日|周|个月|月|季度|年)",
    r"(?:上|下|本|当前)(?:周|月|季度|年)",
    r"(?:今年|去年|前年|本季度|上季度|下季度)",
)
NUMERIC_PATTERNS = (
    r"(?:前|后)\s*\d+\s*(?:名|个|条|家|种)?",
    r"(?:超过|高于|大于|不少于|至少|低于|小于|不超过|至多)\s*"
    r"\d+(?:\.\d+)?\s*(?:%|元|万元|件|单|次|人|个)?",
    r"\d+(?:\.\d+)?\s*%",
)
IDENTIFIER_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+|"
    r"[A-Z][A-Z0-9]{1,11})(?![A-Za-z0-9_])"
)
QUOTED_RE = re.compile(r"[“\"'《「『](.+?)[”\"'》」』]")


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        normalized = normalize_signal(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(text)
    return tuple(output)


def _ordered_matches(query: str, values: Sequence[str]) -> tuple[str, ...]:
    found: list[tuple[int, int, str]] = []
    normalized_query = normalize_signal(query)
    for value in values:
        normalized = normalize_signal(value)
        if not normalized:
            continue
        start = normalized_query.find(normalized)
        if start >= 0:
            found.append((start, -len(normalized), str(value).strip()))
    return _ordered_unique(value for _, _, value in sorted(found))


def _regex_matches(query: str, patterns: Sequence[str]) -> tuple[str, ...]:
    rows: list[tuple[int, str]] = []
    for pattern in patterns:
        rows.extend(
            (match.start(), match.group(0).strip())
            for match in re.finditer(pattern, query, flags=re.IGNORECASE)
        )
    return _ordered_unique(value for _, value in sorted(rows))


def workspace_dictionary_fingerprint(
    workspace_id: str,
    terms: Sequence[str],
) -> str:
    return stable_signal_hash({
        "contract": "workspace-query-dictionary-v1",
        "workspace_id": str(workspace_id or ""),
        "terms": sorted({normalize_signal(term) for term in terms if normalize_signal(term)}),
    })


def longest_workspace_matches(
    query: str,
    terms: Sequence[str],
) -> tuple[str, ...]:
    """Longest non-overlapping match without touching jieba global state."""

    normalized_query = normalize_signal(query)
    candidates: list[tuple[int, int, str]] = []
    for term in terms:
        normalized = normalize_signal(term)
        if not normalized:
            continue
        start = normalized_query.find(normalized)
        while start >= 0:
            candidates.append((start, start + len(normalized), str(term).strip()))
            start = normalized_query.find(normalized, start + 1)
    selected: list[tuple[int, int, str]] = []
    occupied: set[int] = set()
    for start, end, term in sorted(
        candidates, key=lambda row: (row[0], -(row[1] - row[0]), normalize_signal(row[2]))
    ):
        span = set(range(start, end))
        if span & occupied:
            continue
        selected.append((start, end, term))
        occupied.update(span)
    return _ordered_unique(term for _, _, term in sorted(selected))


def deterministic_value_route_variants(
    query: str,
    terms: Sequence[str],
) -> tuple[str, ...]:
    """Return deterministic ES literal variants used by production and ablation."""

    variants: list[str] = []
    for value in [query, *terms]:
        normalized = str(value or "").strip()
        for suffix in ("地区", "区域", "大区"):
            if normalized.endswith(suffix) and len(normalized) > len(suffix):
                variants.append(normalized[:-len(suffix)])
    variants.extend(
        re.findall(r"华东|华南|华北|华中|东北|西北|西南", query or "")
    )
    term_keys = {normalize_signal(value) for value in terms}
    return _ordered_unique(
        value for value in variants
        if normalize_signal(value) not in term_keys
    )


def _jieba_terms(query: str, profile: str, top_k: int) -> tuple[str, ...]:
    if profile == "full_query_only":
        return ()
    if profile == "jieba_textrank_v1":
        values = textrank(query, topK=top_k, allowPOS=ALLOW_POS)
    else:
        values = extract_tags(query, topK=top_k, allowPOS=ALLOW_POS)
    return _ordered_unique(values)


def extract_query_signals(
    query: str,
    *,
    profile: str,
    workspace_id: str = "",
    workspace_terms: Sequence[str] = (),
    dictionary_version: str | None = None,
    lexical_top_k: int = 10,
) -> QuerySignalsV1:
    standalone = str(query or "").strip()
    if not standalone:
        raise ValueError("查询信号不能从空问题构建")
    if profile not in PROFILE_VERSIONS:
        raise ValueError(f"不支持的 query-signal profile：{profile}")

    lexical_terms = _jieba_terms(standalone, profile, max(1, int(lexical_top_k)))
    if profile == "domain_signals_v1":
        identifiers = _ordered_unique(
            match.group(0) for match in IDENTIFIER_RE.finditer(standalone)
        )
        quoted_phrases = _ordered_unique(
            match.group(1) for match in QUOTED_RE.finditer(standalone)
        )
        time_expressions = _regex_matches(standalone, TIME_PATTERNS)
        numeric_constraints = tuple(
            value for value in _regex_matches(standalone, NUMERIC_PATTERNS)
            if normalize_signal(value) not in {
                normalize_signal(time_value) for time_value in time_expressions
            }
        )
        operators = _ordered_matches(standalone, OPERATORS)
        polarity = _ordered_matches(standalone, POLARITY)
        domain_terms = longest_workspace_matches(standalone, workspace_terms)
        entity_literals = _ordered_unique([
            *quoted_phrases,
            *_ordered_matches(standalone, REGION_LITERALS),
            *(
                match.group(0)
                for match in re.finditer(
                    r"(?<![A-Za-z0-9_])(?:VIP|VVIP)(?![A-Za-z0-9_])",
                    standalone,
                    flags=re.IGNORECASE,
                )
            ),
        ])
    else:
        identifiers = ()
        quoted_phrases = ()
        time_expressions = ()
        numeric_constraints = ()
        operators = ()
        polarity = ()
        domain_terms = ()
        entity_literals = ()
    if dictionary_version:
        dictionary_fingerprint = dictionary_version
    elif workspace_terms:
        dictionary_fingerprint = workspace_dictionary_fingerprint(
            workspace_id, workspace_terms
        )
    else:
        dictionary_fingerprint = "none"
    try:
        jieba_version = importlib.metadata.version("jieba")
    except importlib.metadata.PackageNotFoundError:
        jieba_version = "unknown"
    profile_payload = {
        "contract": QUERY_SIGNAL_CONTRACT_VERSION,
        "profile": profile,
        "profile_version": PROFILE_VERSIONS[profile],
        "extractor_version": EXTRACTOR_VERSION,
        "dictionary_version": dictionary_fingerprint,
        "idf_version": f"jieba-default-idf-{jieba_version}",
        "stopword_version": "none",
        "rule_version": RULE_VERSION,
        "lexical_top_k": int(lexical_top_k),
    }
    provenance = QuerySignalProvenance(
        extractor_profile=profile,
        extractor_version=f"{EXTRACTOR_VERSION}:{PROFILE_VERSIONS[profile]}",
        dictionary_version=dictionary_fingerprint,
        idf_version=f"jieba-default-idf-{jieba_version}",
        stopword_version="none",
        rule_version=RULE_VERSION,
        library_version=f"jieba-{jieba_version}",
        profile_fingerprint=stable_signal_hash(profile_payload),
    )
    signal_payload = {
        "standalone_query": normalize_signal(standalone),
        "lexical_terms": [normalize_signal(value) for value in lexical_terms],
        "domain_terms": [normalize_signal(value) for value in domain_terms],
        "identifiers": [normalize_signal(value) for value in identifiers],
        "quoted_phrases": [normalize_signal(value) for value in quoted_phrases],
        "entity_literals": [normalize_signal(value) for value in entity_literals],
        "time_expressions": [normalize_signal(value) for value in time_expressions],
        "numeric_constraints": [normalize_signal(value) for value in numeric_constraints],
        "operators": [normalize_signal(value) for value in operators],
        "polarity": [normalize_signal(value) for value in polarity],
        "profile_fingerprint": provenance.profile_fingerprint,
    }
    result = QuerySignalsV1(
        standalone_query=standalone,
        semantic_queries=(standalone,),
        lexical_terms=lexical_terms,
        domain_terms=domain_terms,
        identifiers=identifiers,
        quoted_phrases=quoted_phrases,
        entity_literals=entity_literals,
        time_expressions=time_expressions,
        numeric_constraints=numeric_constraints,
        operators=operators,
        polarity=polarity,
        provenance=provenance,
        signal_fingerprint=stable_signal_hash(signal_payload),
    )
    return result


@dataclass(frozen=True)
class QuerySignalRoutePlan:
    route: str
    route_policy: str
    local_signals: tuple[dict[str, Any], ...]
    remote_signals: tuple[dict[str, Any], ...]
    rerank_signals: tuple[dict[str, Any], ...]
    cache_namespace: str

    @property
    def local_terms(self) -> list[str]:
        return [str(item["text"]) for item in self.local_signals]

    @property
    def remote_terms(self) -> list[str]:
        return [str(item["text"]) for item in self.remote_signals]

    @property
    def rerank_terms(self) -> list[str]:
        return [str(item["text"]) for item in self.rerank_signals]


def _dedupe_items(items: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in items:
        item = dict(raw)
        normalized = normalize_signal(item.get("text"))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(item)
    return tuple(output)


def _extra_items(values: Sequence[str], kind: str = "llm_expansion") -> list[dict[str, Any]]:
    output = []
    for rank, text in enumerate(values, 1000):
        fingerprint = stable_signal_hash({
            "contract": QUERY_SIGNAL_CONTRACT_VERSION,
            "kind": kind,
            "text": normalize_signal(text),
        })
        output.append({
            "signal_id": f"qs-{fingerprint[:12]}",
            "kind": kind,
            "text": str(text),
            "rank": rank,
            "fingerprint": fingerprint,
        })
    return output


def _typed_or_extra_items(
    values: Sequence[str],
    existing: Sequence[Mapping[str, Any]],
    *,
    kind: str,
) -> list[dict[str, Any]]:
    by_text = {
        normalize_signal(row.get("text")): dict(row)
        for row in existing
        if normalize_signal(row.get("text"))
    }
    output: list[dict[str, Any]] = []
    for value in values:
        normalized = normalize_signal(value)
        if not normalized:
            continue
        output.append(
            by_text.get(normalized)
            or _extra_items([str(value)], kind=kind)[0]
        )
    return output


def build_route_signal_plan(
    signals: QuerySignalsV1,
    route: str,
    *,
    route_policy: str,
    remote_signal_limit: int,
    llm_terms: Sequence[str] = (),
    priority_terms: Sequence[str] = (),
) -> QuerySignalRoutePlan:
    if route not in {"column", "metric", "value", "training"}:
        raise ValueError(f"未知 query-signal route：{route}")
    if route_policy not in ROUTE_POLICIES:
        raise ValueError(f"未知 query-signal route policy：{route_policy}")

    rows = signals.typed_items()
    standalone = [row for row in rows if row["kind"] == "standalone_query"]
    by_kind = {
        kind: [row for row in rows if row["kind"] == kind]
        for kind in (
            "lexical_terms", "domain_terms", "identifiers", "quoted_phrases",
            "entity_literals", "time_expressions", "numeric_constraints",
            "operators", "polarity",
        )
    }
    expansions = _extra_items(llm_terms)
    priority = _typed_or_extra_items(
        priority_terms, rows, kind="deterministic_route_variant"
    )
    if route_policy == "full_query_only_v1":
        local = remote = rerank = standalone
    elif route_policy == "current_multi_query_v1":
        # Preserve the production K1 behavior while making the real order
        # explicit. Value-route deterministic entity variants are passed as
        # expansions and must remain ahead of the generic full query/tags.
        multi_query_terms = (
            [
                *priority,
                *standalone,
                *by_kind["lexical_terms"],
                *expansions,
            ]
            if route == "value"
            else [*standalone, *by_kind["lexical_terms"], *expansions]
        )
        local = multi_query_terms
        remote = standalone if route == "training" else multi_query_terms
        rerank = multi_query_terms
    elif route == "value":
        priority = [
            *by_kind["entity_literals"],
            *by_kind["quoted_phrases"],
            *by_kind["identifiers"],
            *by_kind["domain_terms"],
            *priority,
            *by_kind["lexical_terms"],
            *expansions,
            *standalone,
        ]
        local = priority
        remote = priority
        rerank = priority
    else:
        structural = (
            [*by_kind["operators"], *by_kind["polarity"]]
            if route in {"metric", "training"}
            else [*by_kind["operators"]]
        )
        local = [
            *standalone,
            *by_kind["domain_terms"],
            *by_kind["identifiers"],
            *by_kind["quoted_phrases"],
            *structural,
            *by_kind["lexical_terms"],
            *expansions,
        ]
        remote = standalone
        rerank = local

    local_rows = _dedupe_items(local)
    remote_rows = _dedupe_items(remote)[:max(1, int(remote_signal_limit))]
    rerank_rows = _dedupe_items(rerank)
    namespace = stable_signal_hash({
        "contract": QUERY_SIGNAL_CONTRACT_VERSION,
        "route_policy_version": ROUTE_POLICY_VERSION,
        "route_policy": route_policy,
        "route": route,
        "profile_fingerprint": signals.provenance.profile_fingerprint,
        "signal_fingerprint": signals.signal_fingerprint,
        "remote": [
            (item["kind"], item["fingerprint"]) for item in remote_rows
        ],
    })
    return QuerySignalRoutePlan(
        route=route,
        route_policy=route_policy,
        local_signals=local_rows,
        remote_signals=remote_rows,
        rerank_signals=rerank_rows,
        cache_namespace=f"query-signals:{namespace}",
    )


def query_signal_route_evidence(
    signals: QuerySignalsV1,
    plan: QuerySignalRoutePlan,
    *,
    query_encode_mode: str,
) -> dict[str, Any]:
    summary = signals.privacy_summary()

    def compact(rows: Sequence[Mapping[str, Any]], consumer: str) -> list[dict[str, Any]]:
        return [
            {
                "signal_id": row.get("signal_id"),
                "kind": row.get("kind"),
                "rank": row.get("rank"),
                "fingerprint": row.get("fingerprint"),
                "consumer": consumer,
            }
            for row in rows
        ]

    local_signals = compact(plan.local_signals, "local")
    remote_signals = compact(plan.remote_signals, "remote")
    rerank_signals = compact(plan.rerank_signals, "rerank")
    consumer_plan_fingerprint = stable_signal_hash({
        "contract": QUERY_SIGNAL_CONTRACT_VERSION,
        "route_policy_version": ROUTE_POLICY_VERSION,
        "route_policy": plan.route_policy,
        "route": plan.route,
        "profile_fingerprint": signals.provenance.profile_fingerprint,
        "signal_fingerprint": signals.signal_fingerprint,
        "query_encode_mode": query_encode_mode,
        "local": local_signals,
        "remote": remote_signals,
        "rerank": rerank_signals,
    })
    return {
        **summary,
        "route": plan.route,
        "route_policy": plan.route_policy,
        "route_policy_version": ROUTE_POLICY_VERSION,
        "cache_namespace_fingerprint": stable_signal_hash(plan.cache_namespace),
        "local_signal_count": len(local_signals),
        "remote_query_count": len(remote_signals),
        "rerank_signal_count": len(rerank_signals),
        "query_encode_mode": query_encode_mode,
        "actual_local_signals": local_signals,
        "actual_remote_signals": remote_signals,
        "actual_rerank_signals": rerank_signals,
        "consumer_plan_fingerprint": consumer_plan_fingerprint,
        "raw_text_logged": False,
    }
