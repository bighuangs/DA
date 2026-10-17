"""Versioned query-signal contract for TODO-61.

The complete standalone question remains the semantic source of truth.  Every
other field is an auxiliary, typed signal whose consumers are decided by a
separate route policy.  State may contain plaintext because it is request-local;
privacy-safe evidence only exposes counts and fingerprints.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


QUERY_SIGNAL_CONTRACT_VERSION = "query-signals-v1"
SIGNAL_KINDS = (
    "lexical_terms",
    "domain_terms",
    "identifiers",
    "quoted_phrases",
    "entity_literals",
    "time_expressions",
    "numeric_constraints",
    "operators",
    "polarity",
)


def normalize_signal(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).strip()
    return " ".join(normalized.split()).casefold()


def stable_signal_hash(payload: Any) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _ordered_unique(values: Sequence[Any]) -> tuple[str, ...]:
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


@dataclass(frozen=True)
class QuerySignalProvenance:
    extractor_profile: str
    extractor_version: str
    dictionary_version: str
    idf_version: str
    stopword_version: str
    rule_version: str
    library_version: str
    profile_fingerprint: str


@dataclass(frozen=True)
class QuerySignalsV1:
    standalone_query: str
    semantic_queries: tuple[str, ...]
    lexical_terms: tuple[str, ...] = ()
    domain_terms: tuple[str, ...] = ()
    identifiers: tuple[str, ...] = ()
    quoted_phrases: tuple[str, ...] = ()
    entity_literals: tuple[str, ...] = ()
    time_expressions: tuple[str, ...] = ()
    numeric_constraints: tuple[str, ...] = ()
    operators: tuple[str, ...] = ()
    polarity: tuple[str, ...] = ()
    provenance: QuerySignalProvenance = field(
        default_factory=lambda: QuerySignalProvenance(
            extractor_profile="full_query_only",
            extractor_version="query-signal-extractor-v1",
            dictionary_version="none",
            idf_version="none",
            stopword_version="none",
            rule_version="query-signal-rules-v1",
            library_version="none",
            profile_fingerprint="",
        )
    )
    signal_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not str(self.standalone_query or "").strip():
            raise ValueError("QuerySignalsV1.standalone_query 不能为空")
        if not self.semantic_queries:
            raise ValueError("QuerySignalsV1.semantic_queries 至少包含完整问题")

    def category(self, name: str) -> tuple[str, ...]:
        if name not in SIGNAL_KINDS:
            raise KeyError(name)
        return tuple(getattr(self, name))

    def typed_items(self, *, include_standalone: bool = True) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if include_standalone:
            rows.append(self._item("standalone_query", self.standalone_query, 0))
        rank = 1
        for kind in SIGNAL_KINDS:
            for text in self.category(kind):
                rows.append(self._item(kind, text, rank))
                rank += 1
        return rows

    @staticmethod
    def _item(kind: str, text: str, rank: int) -> dict[str, Any]:
        fingerprint = stable_signal_hash({
            "contract": QUERY_SIGNAL_CONTRACT_VERSION,
            "kind": kind,
            "text": normalize_signal(text),
        })
        return {
            "signal_id": f"qs-{fingerprint[:12]}",
            "kind": kind,
            "text": text,
            "rank": rank,
            "fingerprint": fingerprint,
        }

    def retrieval_terms(self) -> list[str]:
        """Return the active profile's ordered retrieval terms."""

        profile = self.provenance.extractor_profile
        if profile == "domain_signals_v1":
            ordered = [
                self.standalone_query,
                *self.domain_terms,
                *self.identifiers,
                *self.quoted_phrases,
                *self.entity_literals,
                *self.lexical_terms,
            ]
        elif profile in {"jieba_tfidf_v1", "jieba_textrank_v1"}:
            ordered = [self.standalone_query, *self.lexical_terms]
        else:
            ordered = [self.standalone_query]
        return list(_ordered_unique(ordered))

    def to_state(self) -> dict[str, Any]:
        return {
            "contract_version": QUERY_SIGNAL_CONTRACT_VERSION,
            "standalone_query": self.standalone_query,
            "semantic_queries": list(self.semantic_queries),
            **{kind: list(self.category(kind)) for kind in SIGNAL_KINDS},
            "provenance": asdict(self.provenance),
            "signal_fingerprint": self.signal_fingerprint,
        }

    def privacy_summary(self) -> dict[str, Any]:
        return {
            "contract_version": QUERY_SIGNAL_CONTRACT_VERSION,
            "extractor_profile": self.provenance.extractor_profile,
            "extractor_version": self.provenance.extractor_version,
            "dictionary_version": self.provenance.dictionary_version,
            "idf_version": self.provenance.idf_version,
            "stopword_version": self.provenance.stopword_version,
            "rule_version": self.provenance.rule_version,
            "profile_fingerprint": self.provenance.profile_fingerprint,
            "signal_fingerprint": self.signal_fingerprint,
            "counts": {
                "semantic_queries": len(self.semantic_queries),
                **{kind: len(self.category(kind)) for kind in SIGNAL_KINDS},
            },
            "signals": [
                {
                    key: row[key]
                    for key in ("signal_id", "kind", "rank", "fingerprint")
                }
                for row in self.typed_items()
            ],
            "raw_text_logged": False,
        }

    @classmethod
    def from_state(
        cls,
        payload: Mapping[str, Any] | None,
    ) -> "QuerySignalsV1":
        if not payload or payload.get("contract_version") != QUERY_SIGNAL_CONTRACT_VERSION:
            raise ValueError(
                "query_signals 必须由 extract_keywords 生成当前 query-signals-v1 契约"
            )
        provenance = QuerySignalProvenance(
            **dict(payload.get("provenance") or {})
        )
        return cls(
            standalone_query=str(payload.get("standalone_query") or ""),
            semantic_queries=_ordered_unique(payload.get("semantic_queries") or []),
            **{
                kind: _ordered_unique(payload.get(kind) or [])
                for kind in SIGNAL_KINDS
            },
            provenance=provenance,
            signal_fingerprint=str(payload.get("signal_fingerprint") or ""),
        )
