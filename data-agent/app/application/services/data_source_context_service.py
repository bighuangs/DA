"""统一 LangGraph 数据源上下文、上传 Schema 剖析与关联确认。"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime

from app.application.ports.data_sources import (
    DataSourceContextSettings,
    MappingQualityProbePort,
    SourceMappingStorePort,
)
from app.application.ports.metadata_indexing import SchemaSourcePort
from app.application.services.adhoc_dataset_service import AdhocDatasetService
from app.domain.access.policy import PolicySnapshot
from app.domain.data_sources.context import (
    AssetAccessContext,
    ComparisonIntent,
    DataSourceContextError,
    JOIN_INTENTS,
    MappingConfirmationRequired,
    SourceMode,
    logical_dataset_name,
    public_source_context,
    stable_fingerprint,
)
from app.domain.data_sources.models import AdhocDatasetRecord, SourceMappingRecord


_TEXT_TYPES = ("char", "text", "json", "enum")
_NUMBER_TYPES = ("int", "decimal", "numeric", "float", "double", "real")
_TIME_TYPES = ("date", "time", "year")


def _normalise_name(value: object) -> str:
    return re.sub(r"[\W_]+", "", str(value or "").lower(), flags=re.UNICODE)


def _type_family(value: object) -> str:
    text = str(value or "").lower()
    if any(token in text for token in _NUMBER_TYPES):
        return "number"
    if any(token in text for token in _TIME_TYPES):
        return "time"
    if any(token in text for token in _TEXT_TYPES):
        return "text"
    if "bool" in text or "bit" in text:
        return "boolean"
    return "unknown"


def _column_role(column: dict) -> str:
    name = str(column.get("display_name") or column.get("name") or "").lower()
    family = _type_family(column.get("type"))
    if name in {"id", "编号", "主键"} or name.endswith("_id"):
        return "dimension"
    return "measure" if family == "number" else "dimension"


def _safe_profile(item: AdhocDatasetRecord, logical_name: str) -> dict:
    columns = []
    for column in item.columns or []:
        sample_size = int(column.get("profile_sample_size") or item.row_count or 0)
        null_count = int(column.get("null_count") or 0)
        columns.append(
            {
                "name": str(column.get("name") or ""),
                "display_name": str(
                    column.get("display_name") or column.get("name") or ""
                ),
                "type": str(column.get("type") or "TEXT"),
                "type_family": _type_family(column.get("type")),
                "nullable": bool(column.get("nullable", True)),
                "null_count": null_count,
                "null_rate": round(null_count / max(1, sample_size), 4),
                "distinct_count_sample": int(column.get("distinct_count_sample") or 0),
                "profile_sample_size": sample_size,
                # examples 已在上传服务生成预览时经过统一脱敏；这里仍限制数量和长度。
                "examples": [
                    str(value)[:80] for value in (column.get("examples") or [])[:3]
                ],
                "role": _column_role(column),
            }
        )
    return {
        "schema_version": "upload-schema-profile-v1",
        "dataset_id": item.id,
        "asset_id": item.asset_id,
        "version_id": item.version_id,
        "logical_name": logical_name,
        "filename": item.filename,
        "row_count": int(item.row_count or 0),
        "column_count": int(item.column_count or 0),
        "schema_fingerprint": item.schema_fingerprint,
        "columns": columns,
        "bounded": True,
        "raw_rows_exposed": False,
    }


def _table_info(profile: dict) -> dict:
    return {
        "name": profile["logical_name"],
        "role": "upload",
        "description": (
            f"用户授权的上传数据集“{profile['filename']}”，"
            f"版本 {profile['version_id']}，共 {profile['row_count']} 行。"
        ),
        "source_kind": "managed_upload",
        "source_version_id": profile["version_id"],
        "columns": [
            {
                "name": column["name"],
                "type": column["type"],
                "role": column["role"],
                "examples": column["examples"],
                "description": f"上传文件字段：{column['display_name']}",
                "alias": [column["display_name"]],
                "is_display": column["role"] == "dimension",
            }
            for column in profile["columns"]
        ],
    }


def _mapping_score(left: dict, right: dict) -> tuple[float, list[str]]:
    left_names = {
        _normalise_name(left.get("display_name")),
        _normalise_name(left.get("name")),
        *(_normalise_name(value) for value in left.get("alias") or []),
    }
    right_names = {
        _normalise_name(right.get("display_name")),
        _normalise_name(right.get("name")),
        *(_normalise_name(value) for value in right.get("alias") or []),
    }
    left_names.discard("")
    right_names.discard("")
    reasons = []
    score = 0.0
    if left_names & right_names:
        score += 0.72
        reasons.append("name_exact")
    elif any(a in b or b in a for a in left_names for b in right_names):
        score += 0.48
        reasons.append("name_contains")
    left_family = _type_family(left.get("type"))
    right_family = _type_family(right.get("type"))
    if left_family == right_family and left_family != "unknown":
        score += 0.22
        reasons.append("type_compatible")
    if any(
        token in "".join(left_names | right_names)
        for token in ("id", "编号", "编码", "code")
    ):
        score += 0.06
        reasons.append("key_shape")
    return min(1.0, score), reasons


def _mapping_quality(left: dict, right: dict, score: float) -> dict:
    sample_size = int(left.get("profile_sample_size") or 0)
    distinct = int(left.get("distinct_count_sample") or 0)
    null_count = int(left.get("null_count") or 0)
    uniqueness = distinct / max(1, sample_size - null_count)
    null_rate = null_count / max(1, sample_size)
    return {
        "name_type_score": round(score, 4),
        "upload_uniqueness_sample": round(uniqueness, 4),
        "upload_null_rate": round(null_rate, 4),
        "type_compatible": _type_family(left.get("type"))
        == _type_family(right.get("type")),
        "coverage_verified": False,
        "fanout_verified": False,
        "grain_verified": False,
        "confirmation_required": True,
    }


@dataclass(frozen=True)
class ResolvedDataSourceContext:
    access: AssetAccessContext
    profiles: tuple[dict, ...]
    table_infos: tuple[dict, ...]
    mapping: dict

    def state_dict(self) -> dict:
        return {
            "data_source_context": public_source_context(
                self.access, self.profiles, self.mapping
            ),
            "source_table_infos": [dict(item) for item in self.table_infos],
            "source_mapping": dict(self.mapping),
        }


class DataSourceContextService:
    def __init__(
        self,
        *,
        adhoc_service: AdhocDatasetService,
        dw_repo: SchemaSourcePort,
        mapping_store: SourceMappingStorePort,
        mapping_probe: MappingQualityProbePort,
        settings: DataSourceContextSettings = DataSourceContextSettings(),
        access_policy: PolicySnapshot,
        owner_id: str,
        workspace_id: str,
    ):
        self.adhoc_service = adhoc_service
        self.dw_repo = dw_repo
        self.mapping_store = mapping_store
        self.mapping_probe = mapping_probe
        self.settings = settings
        self.access_policy = access_policy
        self.owner_id = owner_id
        self.workspace_id = workspace_id

    async def resolve(
        self,
        *,
        dataset_ids: list[str] | None = None,
        dataset_id: str | None = None,
        source_mode: str | None = None,
        compare_dw: bool = False,
        comparison_intent: str | None = None,
        mapping_confirmation_id: str | None = None,
        revision: int = 1,
    ) -> ResolvedDataSourceContext:
        ids = list(
            dict.fromkeys(
                [
                    *(dataset_ids or []),
                    *([dataset_id] if dataset_id else []),
                ]
            )
        )
        rows = await self.adhoc_service.get_datasets(ids) if ids else []
        mode = self._resolve_mode(
            source_mode=source_mode,
            dataset_count=len(rows),
            compare_dw=compare_dw,
            comparison_intent=comparison_intent,
        )
        intent = self._resolve_intent(mode, comparison_intent)

        official_snapshot = {"tables": []}
        if mode in {
            SourceMode.DW_ONLY,
            SourceMode.UPLOAD_COMPARE_DW,
            SourceMode.UPLOAD_JOIN_DW,
        }:
            official_snapshot = self.access_policy.filter_schema_snapshot(
                await self.dw_repo.get_schema_snapshot()
            )
        official_tables = frozenset(
            str(table.get("name") or "")
            for table in official_snapshot.get("tables") or []
            if table.get("name")
            and not str(table.get("name"))
            .lower()
            .startswith(("adhoc_", "da_upload_", "eval_upload_", "__da_"))
        )

        identities = []
        profiles = []
        for index, row in enumerate(rows, 1):
            logical = logical_dataset_name(index, row.version_id)
            identities.append(self.adhoc_service.asset_identity(row, logical))
            profiles.append(_safe_profile(row, logical))
        access = AssetAccessContext(
            workspace_id=self.workspace_id,
            owner_id=self.owner_id,
            source_mode=mode,
            official_tables=official_tables,
            upload_assets=tuple(identities),
            comparison_intent=intent,
            mapping_confirmation_id=mapping_confirmation_id,
            revision=max(1, int(revision)),
        )
        mapping: dict = {}
        if intent in JOIN_INTENTS:
            mapping = await self._resolve_mapping(
                rows=rows,
                profiles=profiles,
                official_snapshot=official_snapshot,
                intent=intent,
                mapping_confirmation_id=mapping_confirmation_id,
            )
        elif intent == ComparisonIntent.INDEPENDENT_AGGREGATE_COMPARE.value:
            mapping = {
                "status": "not_required",
                "strategy": "separate_execution_then_deterministic_compare",
                "reason": "独立聚合对比不建立行级 JOIN",
            }
        return ResolvedDataSourceContext(
            access=access,
            profiles=tuple(profiles),
            table_infos=tuple(_table_info(profile) for profile in profiles),
            mapping=mapping,
        )

    def _resolve_mode(
        self,
        *,
        source_mode: str | None,
        dataset_count: int,
        compare_dw: bool,
        comparison_intent: str | None,
    ) -> SourceMode:
        if source_mode:
            try:
                mode = SourceMode(source_mode)
            except ValueError as exc:
                raise DataSourceContextError("不支持的数据源模式") from exc
        elif dataset_count == 0:
            mode = SourceMode.DW_ONLY
        elif dataset_count > 1:
            mode = SourceMode.MULTI_UPLOAD
        elif comparison_intent == ComparisonIntent.UPLOAD_DW_ROW_JOIN.value:
            mode = SourceMode.UPLOAD_JOIN_DW
        elif compare_dw:
            mode = SourceMode.UPLOAD_COMPARE_DW
        else:
            mode = SourceMode.UPLOAD_ONLY
        if mode == SourceMode.DW_ONLY and dataset_count:
            raise DataSourceContextError("DW-only 模式不能携带上传数据集")
        if mode != SourceMode.DW_ONLY and dataset_count == 0:
            raise DataSourceContextError("当前数据源模式至少需要一个上传数据集")
        if mode == SourceMode.MULTI_UPLOAD and dataset_count < 2:
            raise DataSourceContextError("多上传表模式至少需要两个数据集")
        if dataset_count > self.settings.max_selected_datasets:
            raise DataSourceContextError("选择的上传数据集数量超过限制")
        return mode

    @staticmethod
    def _resolve_intent(mode: SourceMode, value: str | None) -> str:
        if value:
            try:
                return ComparisonIntent(value).value
            except ValueError as exc:
                raise DataSourceContextError("不支持的上传分析意图") from exc
        return {
            SourceMode.DW_ONLY: "dw_only",
            SourceMode.UPLOAD_ONLY: ComparisonIntent.UPLOAD_ONLY.value,
            SourceMode.UPLOAD_COMPARE_DW: ComparisonIntent.INDEPENDENT_AGGREGATE_COMPARE.value,
            SourceMode.UPLOAD_JOIN_DW: ComparisonIntent.UPLOAD_DW_ROW_JOIN.value,
            SourceMode.MULTI_UPLOAD: ComparisonIntent.MULTI_UPLOAD.value,
        }[mode]

    async def _resolve_mapping(
        self,
        *,
        rows: list[AdhocDatasetRecord],
        profiles: list[dict],
        official_snapshot: dict,
        intent: str,
        mapping_confirmation_id: str | None,
    ) -> dict:
        version_ids = [str(row.version_id) for row in rows]
        if mapping_confirmation_id:
            item = await self.mapping_store.get(mapping_confirmation_id)
            if item is None or item.status != "confirmed":
                raise DataSourceContextError(
                    "关联映射不存在、未确认或无权访问",
                    code="mapping_confirmation_invalid",
                )
            if list(item.dataset_version_ids) != version_ids:
                raise DataSourceContextError(
                    "关联映射绑定的数据集版本已变化，请重新确认",
                    code="mapping_version_mismatch",
                )
            return {
                "mapping_id": item.id,
                "status": item.status,
                "intent": item.intent,
                "mappings": list(item.mappings),
                "evidence": item.evidence,
                "fingerprint": item.fingerprint,
            }

        candidates = self._propose_candidates(profiles, official_snapshot, intent)
        fingerprint = stable_fingerprint(
            {
                "intent": intent,
                "version_ids": version_ids,
                "candidates": candidates,
            }
        )
        mapping_id = str(uuid.uuid4())
        evidence = {
            "contract": "source-mapping-evidence-v1",
            "candidate_count": len(candidates),
            "quality_gates": {
                "type_compatibility": all(
                    item["quality"]["type_compatible"] for item in candidates
                )
                if candidates
                else False,
                "coverage": False,
                "fanout": False,
                "grain": False,
            },
            "auto_join_allowed": False,
        }
        proposal = {
            "mapping_id": mapping_id,
            "status": "proposed",
            "intent": intent,
            "mappings": candidates,
            "evidence": evidence,
            "fingerprint": fingerprint,
        }
        await self.mapping_store.create(
            SourceMappingRecord(
                id=mapping_id,
                owner_id=self.owner_id,
                workspace_id=self.workspace_id,
                status="proposed",
                intent=intent,
                dataset_version_ids=tuple(version_ids),
                mappings=tuple(candidates),
                evidence=evidence,
                fingerprint=fingerprint,
            )
        )
        raise MappingConfirmationRequired(
            "该分析需要建立行级关联。系统只提供候选，不会自动 JOIN；请先确认字段映射。",
            proposal=proposal,
        )

    @staticmethod
    def _propose_candidates(
        profiles: list[dict], official_snapshot: dict, intent: str
    ) -> list[dict]:
        if not profiles:
            return []
        left_columns = profiles[0].get("columns") or []
        if intent == ComparisonIntent.MULTI_UPLOAD.value and len(profiles) > 1:
            targets = [
                {
                    **column,
                    "table_name": profiles[1]["logical_name"],
                    "source_kind": "managed_upload",
                }
                for column in profiles[1].get("columns") or []
            ]
        else:
            targets = [
                {
                    **column,
                    "table_name": table.get("name"),
                    "source_kind": "official",
                }
                for table in official_snapshot.get("tables") or []
                for column in table.get("columns") or []
            ]
        candidates = []
        for left in left_columns:
            ranked = []
            for right in targets:
                score, reasons = _mapping_score(left, right)
                if score >= 0.48:
                    ranked.append((score, reasons, right))
            if not ranked:
                continue
            score, reasons, right = sorted(
                ranked,
                key=lambda item: (
                    -item[0],
                    str(item[2].get("table_name") or ""),
                    str(item[2].get("name") or ""),
                ),
            )[0]
            candidates.append(
                {
                    "left": {
                        "table": profiles[0]["logical_name"],
                        "column": left["name"],
                        "display_name": left["display_name"],
                    },
                    "right": {
                        "table": right.get("table_name"),
                        "column": right.get("name"),
                        "display_name": right.get("display_name") or right.get("name"),
                        "source_kind": right.get("source_kind"),
                    },
                    "score": round(score, 4),
                    "reasons": reasons,
                    "quality": _mapping_quality(left, right, score),
                }
            )
        ordered = sorted(
            candidates,
            key=lambda item: (
                -float(item.get("score") or 0),
                str(item.get("left", {}).get("column") or ""),
            ),
        )
        if intent == ComparisonIntent.UNION.value:
            return [
                item
                for item in ordered
                if item["quality"]["type_compatible"] and item["score"] >= 0.7
            ][:8]
        # 行级 JOIN 只确认一条主关联键；其余候选不能被模型擅自拼进 ON。
        return ordered[:1]

    async def confirm_mapping(self, mapping_id: str) -> dict | None:
        item = await self.mapping_store.get(mapping_id)
        if item is None:
            return None
        if item.status == "rejected":
            raise DataSourceContextError(
                "已拒绝的映射不能再次确认，请重新生成候选",
                code="mapping_rejected",
            )
        rows = await self.adhoc_service.get_datasets(list(item.dataset_version_ids))
        logical_to_physical = {
            logical_dataset_name(index, row.version_id): row.table_name
            for index, row in enumerate(rows, 1)
        }
        official = self.access_policy.filter_schema_snapshot(
            await self.dw_repo.get_schema_snapshot()
        )
        official_columns = {
            (str(table.get("name") or ""), str(column.get("name") or ""))
            for table in official.get("tables") or []
            for column in table.get("columns") or []
        }
        evaluated = []
        gates = []
        for mapping in item.mappings:
            checked = await self._evaluate_mapping(
                mapping,
                logical_to_physical=logical_to_physical,
                official_columns=official_columns,
                intent=item.intent,
            )
            evaluated.append(checked)
            gates.append(bool(checked["quality"].get("gate_passed")))
        auto_join_allowed = bool(evaluated) and all(gates)
        evidence = {
            **dict(item.evidence),
            "quality_gates": {
                "type_compatibility": bool(evaluated)
                and all(value["quality"]["type_compatible"] for value in evaluated),
                "coverage": bool(evaluated)
                and all(value["quality"]["coverage_passed"] for value in evaluated),
                "fanout": bool(evaluated)
                and all(value["quality"]["fanout_passed"] for value in evaluated),
                "grain": bool(evaluated)
                and all(value["quality"]["grain_passed"] for value in evaluated),
            },
            "auto_join_allowed": auto_join_allowed,
            "evaluated_at": datetime.utcnow().isoformat(),
            "raw_values_logged": False,
        }
        status = "confirmed" if auto_join_allowed else "blocked"
        confirmed_at = datetime.utcnow() if auto_join_allowed else None
        stored = await self.mapping_store.save_confirmation(
            mapping_id,
            mappings=tuple(evaluated),
            evidence=evidence,
            status=status,
            confirmed_at=confirmed_at,
        )
        if stored is None:
            return None
        return {
            "mapping_id": stored.id,
            "status": stored.status,
            "intent": stored.intent,
            "mappings": list(stored.mappings),
            "evidence": stored.evidence,
            "fingerprint": stored.fingerprint,
            "confirmed_at": (
                stored.confirmed_at.isoformat() if stored.confirmed_at else None
            ),
        }

    async def _evaluate_mapping(
        self,
        mapping: dict,
        *,
        logical_to_physical: dict[str, str],
        official_columns: set[tuple[str, str]],
        intent: str,
    ) -> dict:
        left = dict(mapping.get("left") or {})
        right = dict(mapping.get("right") or {})
        identifier = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
        left_table = logical_to_physical.get(str(left.get("table") or ""))
        right_logical = str(right.get("table") or "")
        right_table = logical_to_physical.get(right_logical, right_logical)
        left_column = str(left.get("column") or "")
        right_column = str(right.get("column") or "")
        if not all(
            identifier.fullmatch(value or "")
            for value in (left_table, right_table, left_column, right_column)
        ):
            raise DataSourceContextError(
                "映射包含非法资源标识", code="mapping_identifier_invalid"
            )
        if (
            right_logical not in logical_to_physical
            and (right_logical, right_column) not in official_columns
        ):
            raise DataSourceContextError(
                "映射目标已不在当前授权 Schema",
                code="mapping_target_unavailable",
            )
        quality = dict(mapping.get("quality") or {})
        if intent == ComparisonIntent.UNION.value:
            quality.update(
                {
                    "coverage": 1.0,
                    "fanout": 1.0,
                    "right_uniqueness": 1.0,
                    "coverage_passed": True,
                    "fanout_passed": True,
                    "grain_passed": bool(quality.get("type_compatible")),
                    "gate_passed": bool(quality.get("type_compatible")),
                    "evaluation": "schema_alignment",
                }
            )
            return {**mapping, "quality": quality}

        stats = await self.mapping_probe.probe(
            left_table=left_table,
            left_column=left_column,
            right_table=right_table,
            right_column=right_column,
        )
        non_null = stats.right_non_null_count
        right_distinct = stats.right_distinct_count
        left_distinct = stats.left_distinct_count
        left_rows = stats.left_row_count
        matched_distinct = stats.matched_distinct_count
        joined_rows = stats.joined_row_count
        right_uniqueness = right_distinct / max(1, non_null)
        coverage = matched_distinct / max(1, left_distinct)
        fanout = joined_rows / max(1, left_rows)
        coverage_passed = coverage >= 0.8
        fanout_passed = fanout <= 1.1
        grain_passed = right_uniqueness >= 0.98
        gate_passed = (
            bool(quality.get("type_compatible"))
            and coverage_passed
            and fanout_passed
            and grain_passed
        )
        quality.update(
            {
                "coverage": round(coverage, 4),
                "fanout": round(fanout, 4),
                "right_uniqueness": round(right_uniqueness, 4),
                "coverage_passed": coverage_passed,
                "fanout_passed": fanout_passed,
                "grain_passed": grain_passed,
                "gate_passed": gate_passed,
                "evaluation": "bounded_aggregate_probe",
                "raw_values_logged": False,
            }
        )
        return {**mapping, "quality": quality}

    async def reject_mapping(self, mapping_id: str, reason: str | None) -> dict | None:
        item = await self.mapping_store.reject(mapping_id, reason)
        if item is None:
            return None
        return {
            "mapping_id": item.id,
            "status": item.status,
            "rejection_reason": item.rejection_reason,
        }

    async def mapping_evidence(self, mapping_id: str) -> dict | None:
        item = await self.mapping_store.get(mapping_id)
        if item is None:
            return None
        return {
            "mapping_id": item.id,
            "status": item.status,
            "intent": item.intent,
            "mappings": list(item.mappings),
            "evidence": item.evidence,
            "fingerprint": item.fingerprint,
            "dataset_version_ids": list(item.dataset_version_ids),
        }
