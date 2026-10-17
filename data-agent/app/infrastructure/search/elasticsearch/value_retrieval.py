"""Governed Elasticsearch value-index contracts for TODO-58.

Qdrant Dense/Sparse indexes retrieve metadata entities (columns, metrics and
training examples).  This module deliberately models a different artifact:
verified, access-controlled values that may become SQL literals.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from app.config.app_config import app_config


VALUE_RETRIEVAL_RESPONSIBILITY = "es_field_value"
RAG_RETRIEVAL_RESPONSIBILITY = "qdrant_meta_knowledge"
VALUE_SEARCH_AUDIT_SCHEMA = "value-search-audit-v1"
VALUE_QUERY_MODES = ("exact", "prefix", "full_text")

_SENSITIVE_FIELD_TOKENS = (
    "customer_name", "user_name", "full_name", "contact_name",
    "phone", "mobile", "telephone", "email", "address", "id_card",
    "identity", "passport", "bank_card", "客户姓名", "用户姓名", "联系人",
    "手机号", "电话", "邮箱", "地址", "证件", "身份证", "银行卡",
)
_UNIQUE_IDENTIFIER_TOKENS = (
    "order_id", "transaction_id", "serial_no", "serial_number", "device_id",
    "invoice_no", "tracking_no", "订单号", "流水号", "设备ID", "设备id",
    "发票号", "运单号",
)
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_PHONE_RE = re.compile(r"^(?:\+?86[- ]?)?1[3-9]\d{9}$")
_ID_CARD_RE = re.compile(r"^(?:\d{15}|\d{17}[\dXx])$")


def _value(item: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    return item.get(key, default) if isinstance(item, Mapping) else getattr(item, key, default)


def normalize_value(value: Any) -> str:
    """Canonical exact/prefix representation without mutating the source value."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip().casefold()


def value_document_id(workspace_id: str, column_id: str, normalized_value: str) -> str:
    """Stable, non-PII document identity used as both logical id and ES ``_id``."""
    digest = hashlib.sha256(
        "\x1f".join((workspace_id, column_id, normalized_value)).encode("utf-8")
    ).hexdigest()
    return f"value:{digest}"


def value_query_fingerprint(keyword: str) -> str:
    """Audit-safe query identity; raw literals are never written to trace/log."""
    return "sha256:" + hashlib.sha256(normalize_value(keyword).encode("utf-8")).hexdigest()


def build_value_search_query(
    keyword: str,
    *,
    query_mode: str,
    workspace_id: str,
    index_version: str,
    allowed_table_ids: Sequence[str] | None = None,
    allowed_column_ids: Sequence[str] | None = None,
    table_id: str | None = None,
    column_id: str | None = None,
) -> dict:
    """Compile the complete ES query so authorization cannot be post-filtered."""
    mode = str(query_mode or "full_text").strip().lower()
    if mode not in VALUE_QUERY_MODES:
        raise ValueError(f"不支持的字段值查询模式：{query_mode}")
    normalized = normalize_value(keyword)
    filters: list[dict] = [
        {"term": {"workspace_id": workspace_id}},
        {"term": {"index_version": index_version}},
        {"term": {"is_active": True}},
    ]
    if allowed_table_ids is not None:
        filters.append({"terms": {"table_id": sorted(set(allowed_table_ids))}})
    if allowed_column_ids is not None:
        filters.append({"terms": {"column_id": sorted(set(allowed_column_ids))}})
    if table_id:
        filters.append({"term": {"table_id": table_id}})
    if column_id:
        filters.append({"term": {"column_id": column_id}})
    if mode == "exact":
        lexical_query = {"term": {"normalized_value": normalized}}
    elif mode == "prefix":
        lexical_query = {"prefix": {"normalized_value": normalized}}
    else:
        lexical_query = {"match": {"value": {"query": keyword, "operator": "and"}}}
    return {"bool": {"must": [lexical_query], "filter": filters}}


def _workspace_token(workspace_id: str) -> str:
    raw = str(workspace_id or "").strip().casefold()
    if not raw:
        raise ValueError("值索引别名必须绑定 workspace_id")
    slug = re.sub(r"[^a-z0-9_-]+", "-", raw).strip("-_")[:36] or "workspace"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"{slug}-{digest}"


def value_index_alias(workspace_id: str) -> str:
    """Return the isolated active alias for one workspace."""
    base = re.sub(r"[^a-z0-9_-]+", "-", app_config.es.index_name.casefold()).strip("-")
    return f"{base}-{_workspace_token(workspace_id)}"


def value_physical_index_prefix(workspace_id: str) -> str:
    return f"{value_index_alias(workspace_id)}--"


def value_physical_index_name(workspace_id: str, index_version: str) -> str:
    short_version = str(index_version).removeprefix("sha256:")[:16] or "unversioned"
    generation = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    # generation 放在版本摘要前，使索引名逆序与发布时间逆序一致，便于安全保留历史。
    return f"{value_physical_index_prefix(workspace_id)}{generation}-{short_version}"


def _matches_any(column_id: str, patterns: Sequence[str]) -> bool:
    lowered = str(column_id or "").casefold()
    return any(fnmatch.fnmatchcase(lowered, str(pattern).casefold()) for pattern in patterns)


def sensitive_column_reason(column: Mapping[str, Any] | Any) -> str | None:
    """Conservatively classify fields that must not enter the value index."""
    column_id = str(_value(column, "id", "") or "")
    if _matches_any(column_id, app_config.es.allow_sensitive_columns):
        return None
    text = " ".join([
        column_id,
        str(_value(column, "name", "") or ""),
        str(_value(column, "description", "") or ""),
        *[str(item) for item in (_value(column, "alias", []) or [])],
    ]).casefold()
    if any(token.casefold() in text for token in _SENSITIVE_FIELD_TOKENS):
        return "sensitive_column"
    return None


def high_cardinality_reason(
    column: Mapping[str, Any] | Any,
    *,
    distinct_count: int | None = None,
) -> str | None:
    column_id = str(_value(column, "id", "") or "")
    if _matches_any(column_id, app_config.es.allow_high_cardinality_columns):
        return None
    name = str(_value(column, "name", "") or "").casefold()
    role = str(_value(column, "role", "") or "").casefold()
    if role in {"primary_key", "foreign_key"} or name.endswith("_id"):
        return "identifier_column"
    searchable = f"{column_id} {name}".casefold()
    if any(token.casefold() in searchable for token in _UNIQUE_IDENTIFIER_TOKENS):
        return "identifier_column"
    if (
        distinct_count is not None
        and distinct_count > int(app_config.es.max_distinct_values_per_column)
    ):
        return "high_cardinality"
    return None


def values_contain_sensitive_literal(values: Iterable[Any]) -> bool:
    """Detect obvious PII formats even when a column was poorly named."""
    for raw in values:
        text = str(raw or "").strip()
        compact = text.replace(" ", "")
        if (
            _EMAIL_RE.fullmatch(text)
            or _PHONE_RE.fullmatch(compact)
            or _ID_CARD_RE.fullmatch(compact)
        ):
            return True
    return False


@dataclass(frozen=True, slots=True)
class ValueColumnDecision:
    column_id: str
    allowed: bool
    reason: str
    distinct_count: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def value_column_decision(
    column: Mapping[str, Any] | Any,
    *,
    distinct_count: int | None = None,
    values: Sequence[Any] | None = None,
) -> ValueColumnDecision:
    column_id = str(_value(column, "id", "") or "")
    if not bool(_value(column, "sync", False)):
        return ValueColumnDecision(column_id, False, "sync_disabled", distinct_count)
    reason = sensitive_column_reason(column)
    if reason:
        return ValueColumnDecision(column_id, False, reason, distinct_count)
    reason = high_cardinality_reason(column, distinct_count=distinct_count)
    if reason:
        return ValueColumnDecision(column_id, False, reason, distinct_count)
    if (
        values
        and not _matches_any(column_id, app_config.es.allow_sensitive_columns)
        and values_contain_sensitive_literal(values)
    ):
        return ValueColumnDecision(column_id, False, "sensitive_value_format", distinct_count)
    return ValueColumnDecision(column_id, True, "approved_sync_field", distinct_count)


def value_index_version(
    *,
    workspace_id: str,
    columns: Sequence[Mapping[str, Any] | Any],
    content_fingerprints: Sequence[dict],
) -> str:
    """Fingerprint ES mapping, admission policy and normalized published content."""
    column_contract = sorted(({
        "id": str(_value(item, "id", "") or ""),
        "name": str(_value(item, "name", "") or ""),
        "type": str(_value(item, "type", "") or ""),
        "role": str(_value(item, "role", "") or ""),
        "table_id": str(_value(item, "table_id", "") or ""),
        "sync": bool(_value(item, "sync", False)),
    } for item in columns), key=lambda item: item["id"])
    payload = {
        "contract_version": app_config.es.value_contract_version,
        "mapping_version": app_config.es.value_mapping_version,
        "normalization_version": app_config.es.value_normalization_version,
        "analyzer": app_config.es.value_analyzer,
        "workspace_id": workspace_id,
        "max_distinct_values_per_column": int(
            app_config.es.max_distinct_values_per_column
        ),
        "allow_sensitive_columns": sorted(app_config.es.allow_sensitive_columns),
        "allow_high_cardinality_columns": sorted(
            app_config.es.allow_high_cardinality_columns
        ),
        "columns": column_contract,
        "content": sorted(content_fingerprints, key=lambda item: item["column_id"]),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()
