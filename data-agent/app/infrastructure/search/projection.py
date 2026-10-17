"""Entity-to-Point projection contracts for the BGE-M3 Qdrant indexes.

TODO-57 deliberately keeps the embedding model, Dense+Sparse vectors, entity
RRF and rule reranking fixed.  This module is the only place where the physical
Point representation may vary between the three comparison profiles.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.config.app_config import app_config


# 本模块从原 ``app/core`` 下迁入两级子包后，仓库根相应上移一层。
PROJECT_ROOT = Path(__file__).parents[3]
SUPPORTED_POINT_STRATEGIES = (
    "single_canonical",
    "multi_all",
    "mixed_selective",
)


def normalize_projection_text(value: Any) -> str:
    """Normalize projection text without flattening its structured line layout."""
    lines = []
    for raw_line in str(value or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _sha256_text(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _value(item: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    return item.get(key, default) if isinstance(item, Mapping) else getattr(item, key, default)


def _dedupe_texts(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = normalize_projection_text(value)
        identity = text.casefold()
        if not text or identity in seen:
            continue
        seen.add(identity)
        result.append(text)
    return result


def validate_point_strategy(strategy: str | None = None) -> str:
    selected = str(strategy or app_config.point_projection.strategy).strip()
    if selected not in SUPPORTED_POINT_STRATEGIES:
        raise ValueError(
            f"未知 Point 投影策略 {selected!r}；"
            f"仅支持 {', '.join(SUPPORTED_POINT_STRATEGIES)}"
        )
    return selected


def alias_manifest_path(path: str | Path | None = None) -> Path:
    configured = Path(str(path or app_config.point_projection.alias_manifest_path))
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


def load_alias_manifest(path: str | Path | None = None) -> dict:
    manifest_path = alias_manifest_path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Point 投影别名清单不存在：{manifest_path}")
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    rows = payload.get("approved_aliases") or []
    if not isinstance(rows, list):
        raise ValueError("point_projection_aliases.yaml 的 approved_aliases 必须是列表")
    normalized_rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, Mapping):
            raise ValueError(f"approved_aliases 第 {index} 项必须是对象")
        entity_type = str(row.get("entity_type") or "").strip()
        business_key = str(row.get("business_key") or "").strip()
        alias = normalize_projection_text(row.get("alias"))
        if entity_type not in {"column", "metric"} or not business_key or not alias:
            raise ValueError(
                f"approved_aliases 第 {index} 项缺少合法 entity_type/business_key/alias"
            )
        identity = (entity_type, business_key.casefold(), alias.casefold())
        if identity in seen:
            raise ValueError(f"approved_aliases 存在重复项：{business_key} / {alias}")
        seen.add(identity)
        normalized_rows.append({
            "entity_type": entity_type,
            "business_key": business_key,
            "alias": alias,
            "reason": str(row.get("reason") or "").strip(),
        })
    return {
        "version": str(payload.get("version") or "").strip(),
        "approved_aliases": normalized_rows,
        "path": str(manifest_path),
    }


def alias_manifest_fingerprint(manifest: Mapping[str, Any] | None = None) -> str:
    current = dict(manifest or load_alias_manifest())
    canonical = {
        "version": current.get("version"),
        "approved_aliases": current.get("approved_aliases") or [],
    }
    raw = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_text(raw)


def _approved_aliases(
    *,
    manifest: Mapping[str, Any],
    entity_type: str,
    business_key: str,
    available_aliases: Sequence[str],
) -> list[str]:
    available = {item.casefold(): item for item in _dedupe_texts(available_aliases)}
    approved: list[str] = []
    for row in manifest.get("approved_aliases") or []:
        if (
            str(row.get("entity_type") or "") != entity_type
            or str(row.get("business_key") or "").casefold() != business_key.casefold()
        ):
            continue
        alias = normalize_projection_text(row.get("alias"))
        matched = available.get(alias.casefold())
        # A manifest cannot inject terminology that is absent from governed metadata.
        if matched:
            approved.append(matched)
    return _dedupe_texts(approved)[:max(
        0, int(app_config.point_projection.max_alias_points_per_entity)
    )]


@dataclass(frozen=True, slots=True)
class ProjectionRecord:
    point_id: str
    text: str
    point_role: str
    payload: dict


def _point_id(
    *,
    workspace_id: str,
    entity_type: str,
    business_key: str,
    point_role: str,
    projected_alias: str,
) -> str:
    namespace = uuid.UUID(str(app_config.point_projection.namespace_uuid))
    identity = "|".join((
        str(app_config.point_projection.identity_version),
        workspace_id or "_global",
        entity_type,
        business_key,
        point_role,
        normalize_projection_text(projected_alias).casefold(),
    ))
    return str(uuid.uuid5(namespace, identity))


def _record(
    *,
    strategy: str,
    workspace_id: str,
    entity_type: str,
    business_key: str,
    entity_id: str,
    point_role: str,
    text: str,
    payload: Mapping[str, Any],
    manifest_hash: str,
    projected_alias: str = "",
) -> ProjectionRecord:
    projection_text = normalize_projection_text(text)
    if not projection_text:
        raise ValueError(f"{business_key} 的 {point_role} Point 文本为空")
    projection_hash = _sha256_text(projection_text)
    enriched = {
        **dict(payload),
        "entity_id": str(entity_id),
        "entity_type": entity_type,
        "business_key": business_key,
        "point_strategy": strategy,
        "point_identity_version": str(app_config.point_projection.identity_version),
        "canonical_template_version": str(
            app_config.point_projection.canonical_template_version
        ),
        "alias_policy_version": str(app_config.point_projection.alias_policy_version),
        "alias_manifest_hash": manifest_hash,
        "projection_hash": projection_hash,
        "projected_alias": projected_alias or None,
    }
    return ProjectionRecord(
        point_id=_point_id(
            workspace_id=workspace_id,
            entity_type=entity_type,
            business_key=business_key,
            point_role=point_role,
            projected_alias=projected_alias,
        ),
        text=projection_text,
        point_role=point_role,
        payload=enriched,
    )


def _column_canonical_text(table: Any, column: Any) -> str:
    aliases = _dedupe_texts(_value(column, "alias", []) or [])
    rows = [
        f"字段: {_value(table, 'name', '')}.{_value(column, 'name', '')}",
        f"定义: {_value(column, 'description', '') or ''}",
        f"别名: {', '.join(aliases)}" if aliases else "",
        f"数据类型: {_value(column, 'type', '')}",
        f"字段角色: {_value(column, 'role', '')}",
        f"所属表: {_value(table, 'name', '')}",
    ]
    # Examples are intentionally excluded: they may contain customer or other
    # sensitive values and are not required for entity identity retrieval.
    return normalize_projection_text("\n".join(row for row in rows if row))


def _metric_canonical_text(item: Any) -> str:
    aliases = _dedupe_texts(_value(item, "alias", []) or [])
    relevant_columns = _dedupe_texts(_value(item, "relevant_columns", []) or [])
    formula_rule = _value(item, "formula_rule")
    formula = (
        json.dumps(formula_rule, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if formula_rule else ""
    )
    rows = [
        f"指标: {_value(item, 'name', '')}",
        f"定义: {_value(item, 'description', '') or ''}",
        f"别名: {', '.join(aliases)}" if aliases else "",
        f"关联字段: {', '.join(relevant_columns)}" if relevant_columns else "",
        f"计算口径: {formula}" if formula else "",
    ]
    return normalize_projection_text("\n".join(row for row in rows if row))


def _projection_texts(
    *,
    strategy: str,
    canonical_text: str,
    name: str,
    description: str,
    aliases: Sequence[str],
    entity_label: str,
    approved_aliases: Sequence[str],
) -> list[tuple[str, str, str]]:
    if strategy == "single_canonical":
        return [(canonical_text, "canonical", "")]
    if strategy == "multi_all":
        rows: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for text, role, projected_alias in (
            [(name, "name", ""), (description, "description", "")]
            + [(alias, "alias", alias) for alias in aliases]
        ):
            normalized = normalize_projection_text(text)
            if not normalized or normalized.casefold() in seen:
                continue
            seen.add(normalized.casefold())
            rows.append((normalized, role, projected_alias))
        return rows
    rows = [(canonical_text, "canonical", "")]
    rows.extend((
        normalize_projection_text(
            f"批准别名: {alias}\n对应实体: {entity_label}\n{description}"
        ),
        "approved_alias",
        alias,
    ) for alias in approved_aliases)
    return rows


def build_column_projection_records(
    table_rows: Iterable[tuple[Any, Sequence[Any]]],
    *,
    index_version: str,
    strategy: str | None = None,
    alias_manifest: Mapping[str, Any] | None = None,
) -> list[ProjectionRecord]:
    selected = validate_point_strategy(strategy)
    manifest = dict(alias_manifest or load_alias_manifest())
    manifest_hash = alias_manifest_fingerprint(manifest)
    records: list[ProjectionRecord] = []
    for table, columns in table_rows:
        table_name = str(_value(table, "name", "") or "")
        for column in columns:
            entity_id = str(_value(column, "id", "") or "")
            column_name = str(_value(column, "name", "") or "")
            business_key = f"column:{table_name}.{column_name}"
            aliases = _dedupe_texts(_value(column, "alias", []) or [])
            payload = {
                "id": entity_id,
                "name": column_name,
                "type": str(_value(column, "type", "") or ""),
                "role": str(_value(column, "role", "") or ""),
                "description": str(_value(column, "description", "") or ""),
                "alias": aliases,
                "table_id": str(_value(column, "table_id", "") or ""),
                "table_name": table_name,
                "is_display": bool(_value(column, "is_display", False)),
                "index_version": index_version,
            }
            approved = _approved_aliases(
                manifest=manifest,
                entity_type="column",
                business_key=business_key,
                available_aliases=aliases,
            )
            for text, role, projected_alias in _projection_texts(
                strategy=selected,
                canonical_text=_column_canonical_text(table, column),
                name=column_name,
                description=payload["description"],
                aliases=aliases,
                entity_label=f"{table_name}.{column_name}",
                approved_aliases=approved,
            ):
                records.append(_record(
                    strategy=selected,
                    workspace_id="",
                    entity_type="column",
                    business_key=business_key,
                    entity_id=entity_id,
                    point_role=role,
                    text=text,
                    payload=payload,
                    manifest_hash=manifest_hash,
                    projected_alias=projected_alias,
                ))
    return records


def build_metric_projection_records(
    items: Iterable[Any],
    *,
    workspace_id: str,
    index_version: str,
    strategy: str | None = None,
    alias_manifest: Mapping[str, Any] | None = None,
) -> list[ProjectionRecord]:
    selected = validate_point_strategy(strategy)
    manifest = dict(alias_manifest or load_alias_manifest())
    manifest_hash = alias_manifest_fingerprint(manifest)
    records: list[ProjectionRecord] = []
    for item in items:
        entity_id = str(_value(item, "id", "") or "")
        name = str(_value(item, "name", "") or "")
        business_key = f"metric:{name}"
        aliases = _dedupe_texts(_value(item, "alias", []) or [])
        payload = {
            "workspace_id": workspace_id,
            "id": entity_id,
            "name": name,
            "description": str(_value(item, "description", "") or ""),
            "relevant_columns": list(_value(item, "relevant_columns", []) or []),
            "alias": aliases,
            "formula_rule": _value(item, "formula_rule"),
            "index_version": index_version,
        }
        approved = _approved_aliases(
            manifest=manifest,
            entity_type="metric",
            business_key=business_key,
            available_aliases=aliases,
        )
        for text, role, projected_alias in _projection_texts(
            strategy=selected,
            canonical_text=_metric_canonical_text(item),
            name=name,
            description=payload["description"],
            aliases=aliases,
            entity_label=name,
            approved_aliases=approved,
        ):
            records.append(_record(
                strategy=selected,
                workspace_id=workspace_id,
                entity_type="metric",
                business_key=business_key,
                entity_id=entity_id,
                point_role=role,
                text=text,
                payload=payload,
                manifest_hash=manifest_hash,
                projected_alias=projected_alias,
            ))
    return records


def build_training_projection_records(
    items: Iterable[Any],
    *,
    workspace_id: str,
    index_version: str,
    strategy: str | None = None,
    alias_manifest: Mapping[str, Any] | None = None,
) -> list[ProjectionRecord]:
    selected = validate_point_strategy(strategy)
    manifest_hash = alias_manifest_fingerprint(alias_manifest or load_alias_manifest())
    records: list[ProjectionRecord] = []
    for item in items:
        entity_id = str(_value(item, "id", "") or "")
        question = normalize_projection_text(_value(item, "question", ""))
        payload = {
            "workspace_id": workspace_id,
            "id": entity_id,
            "question": question,
            "sql": str(_value(item, "sql", "") or ""),
            "index_version": index_version,
        }
        records.append(_record(
            strategy=selected,
            workspace_id=workspace_id,
            entity_type="training",
            business_key=f"training:{entity_id}",
            entity_id=entity_id,
            point_role="question",
            text=question,
            payload=payload,
            manifest_hash=manifest_hash,
        ))
    return records


def projection_manifest(records: Sequence[ProjectionRecord]) -> dict:
    entity_ids = {
        (str(record.payload.get("workspace_id") or ""), str(record.payload["entity_id"]))
        for record in records
    }
    point_ids = [record.point_id for record in records]
    projection_hashes = [str(record.payload["projection_hash"]) for record in records]
    point_count = len(records)
    entity_count = len(entity_ids)
    return {
        "strategy": (
            str(records[0].payload.get("point_strategy")) if records else None
        ),
        "entity_count": entity_count,
        "point_count": point_count,
        "points_per_entity": round(point_count / max(1, entity_count), 4),
        "duplicate_occupancy": round(
            max(0, point_count - entity_count) / max(1, point_count), 4
        ),
        "point_roles": dict(sorted(Counter(
            record.point_role for record in records
        ).items())),
        "duplicate_point_ids": point_count - len(set(point_ids)),
        "duplicate_projection_hashes": point_count - len(set(projection_hashes)),
        "stable_ids_valid": point_count == len(set(point_ids)),
        "empty_projection_count": sum(not record.text for record in records),
        "alias_manifest_hash": (
            str(records[0].payload.get("alias_manifest_hash")) if records else None
        ),
        "identity_version": str(app_config.point_projection.identity_version),
        "canonical_template_version": str(
            app_config.point_projection.canonical_template_version
        ),
        "alias_policy_version": str(app_config.point_projection.alias_policy_version),
    }
