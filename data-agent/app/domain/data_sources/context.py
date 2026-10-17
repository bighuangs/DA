"""上传数据源的权威身份、上下文冻结与 SQL 资产授权。

本模块不依赖浏览器传入的表名。调用方只能把已经通过 owner/workspace/version/
status/TTL 校验的数据集版本转换为 :class:`AssetAccessContext`，再由 SQL AST
对每个物理表引用做关闭式授权。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Iterable, Mapping

import sqlglot
from sqlglot import exp

from app.domain.query.safety.sql_security import SQLSecurityError


class AssetKind(StrEnum):
    OFFICIAL = "official"
    MANAGED_UPLOAD = "managed_upload"
    ADHOC = "adhoc"
    EVALUATION = "evaluation"
    SYSTEM = "system"


class SourceMode(StrEnum):
    DW_ONLY = "dw_only"
    UPLOAD_ONLY = "upload_only"
    UPLOAD_COMPARE_DW = "upload_compare_dw"
    UPLOAD_JOIN_DW = "upload_join_dw"
    MULTI_UPLOAD = "multi_upload"


class ComparisonIntent(StrEnum):
    UPLOAD_ONLY = "upload_only"
    INDEPENDENT_AGGREGATE_COMPARE = "independent_aggregate_compare"
    LOOKUP_ENRICHMENT = "lookup_enrichment"
    UNION = "union"
    UPLOAD_DW_ROW_JOIN = "upload_dw_row_join"
    MULTI_UPLOAD = "multi_upload"


JOIN_INTENTS = {
    ComparisonIntent.LOOKUP_ENRICHMENT.value,
    ComparisonIntent.UNION.value,
    ComparisonIntent.UPLOAD_DW_ROW_JOIN.value,
    ComparisonIntent.MULTI_UPLOAD.value,
}

READY_DATASET_STATES = {"ready", "active"}
TERMINAL_DATASET_STATES = {
    "expired", "deleted", "failed", "cleanup_failed", "blocked", "archived"
}
INTERNAL_TABLE_PREFIXES = (
    "adhoc_",
    "da_upload_",
    "eval_upload_",
    "__da_",
)


class DataSourceContextError(ValueError):
    """稳定、可面向 API 返回的数据源上下文错误。"""

    code = "data_source_context_invalid"

    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        if code:
            self.code = code

    def to_dict(self) -> dict:
        return {"code": self.code, "message": str(self)}


class DatasetAccessError(DataSourceContextError):
    code = "dataset_access_denied"


class DatasetExpiredError(DataSourceContextError):
    code = "dataset_expired"


class MappingConfirmationRequired(DataSourceContextError):
    code = "mapping_confirmation_required"

    def __init__(self, message: str, *, proposal: dict):
        super().__init__(message)
        self.proposal = proposal

    def to_dict(self) -> dict:
        return {**super().to_dict(), "proposal": self.proposal}


@dataclass(frozen=True)
class AssetIdentity:
    asset_id: str
    version_id: str
    kind: AssetKind
    source: str
    catalog: str
    schema: str
    table: str
    logical_name: str
    workspace_id: str
    owner_id: str
    state: str
    schema_fingerprint: str
    policy_version: str
    expires_at: datetime | None = None

    @property
    def qualified_table(self) -> str:
        parts = [self.catalog, self.schema, self.table]
        return ".".join(part for part in parts if part)

    def public_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "version_id": self.version_id,
            "kind": self.kind.value,
            "source": self.source,
            "logical_name": self.logical_name,
            "workspace_id": self.workspace_id,
            "state": self.state,
            "schema_fingerprint": self.schema_fingerprint,
            "policy_version": self.policy_version,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


@dataclass(frozen=True)
class AssetAccessContext:
    """一次查询内不可变的资产授权快照。"""

    workspace_id: str
    owner_id: str
    source_mode: SourceMode
    official_tables: frozenset[str]
    upload_assets: tuple[AssetIdentity, ...] = ()
    comparison_intent: str | None = None
    mapping_confirmation_id: str | None = None
    revision: int = 1

    @property
    def logical_to_physical(self) -> dict[str, str]:
        return {
            asset.logical_name.lower(): asset.table
            for asset in self.upload_assets
        }

    @property
    def upload_physical_tables(self) -> frozenset[str]:
        return frozenset(asset.table.lower() for asset in self.upload_assets)

    @property
    def allowed_physical_tables(self) -> frozenset[str]:
        official = (
            frozenset(name.lower() for name in self.official_tables)
            if self.source_mode
            in {SourceMode.DW_ONLY, SourceMode.UPLOAD_COMPARE_DW, SourceMode.UPLOAD_JOIN_DW}
            else frozenset()
        )
        return official | self.upload_physical_tables

    def public_dict(self) -> dict:
        payload = {
            "schema_version": "data-source-context-v1",
            "source_mode": self.source_mode.value,
            "comparison_intent": self.comparison_intent,
            "mapping_confirmation_id": self.mapping_confirmation_id,
            "revision": self.revision,
            "datasets": [asset.public_dict() for asset in self.upload_assets],
            "official_table_count": len(self.official_tables),
        }
        payload["context_fingerprint"] = stable_fingerprint(payload)
        return payload


def stable_fingerprint(value: object) -> str:
    serialised = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return "sha256:" + hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def logical_dataset_name(index: int, version_id: str) -> str:
    suffix = re.sub(r"[^0-9a-f]", "", str(version_id).lower())[:8] or f"{index:02d}"
    return f"upload_{index}_{suffix}"


def _cte_names(tree: exp.Expression) -> set[str]:
    return {
        str(cte.alias_or_name or "").lower()
        for cte in tree.find_all(exp.CTE)
        if cte.alias_or_name
    }


def extract_physical_table_refs(sql: str) -> set[str]:
    try:
        tree = sqlglot.parse_one(str(sql or ""), read="mysql")
    except Exception as exc:
        raise SQLSecurityError(f"无法解析 SQL 资产引用：{exc}") from exc
    ctes = _cte_names(tree)
    return {
        str(table.name or "").lower()
        for table in tree.find_all(exp.Table)
        if table.name and str(table.name).lower() not in ctes
    }


def translate_logical_upload_tables(
    sql: str, access: AssetAccessContext | None
) -> str:
    """把只对模型/浏览器公开的逻辑上传表名翻译为内部物理表名。"""

    if access is None or not access.upload_assets:
        return str(sql or "")
    try:
        tree = sqlglot.parse_one(str(sql or ""), read="mysql")
    except Exception as exc:
        raise SQLSecurityError(f"无法解析 SQL 数据源：{exc}") from exc
    mapping = access.logical_to_physical
    ctes = _cte_names(tree)
    for table in tree.find_all(exp.Table):
        logical = str(table.name or "").lower()
        if not logical or logical in ctes:
            continue
        physical = mapping.get(logical)
        if physical:
            table.set("this", exp.to_identifier(physical))
            # 上传资产当前与 DW 同库，禁止模型通过伪造 catalog/schema 改变边界。
            table.set("db", None)
            table.set("catalog", None)
    return tree.sql(dialect="mysql")


def redact_physical_upload_tables(
    sql: str, access: AssetAccessContext | None
) -> str:
    """将执行 SQL 中的内部表名恢复为逻辑名，供 UI/历史展示。"""

    text = str(sql or "")
    if access is None:
        return text
    for asset in access.upload_assets:
        text = re.sub(
            rf"(?i)(?<![A-Za-z0-9_])`?{re.escape(asset.table)}`?(?![A-Za-z0-9_])",
            f"`{asset.logical_name}`",
            text,
        )
    return text


def authorize_sql_assets(sql: str, access: AssetAccessContext | None) -> set[str]:
    """对 SQL 的每个物理表引用执行关闭式三元授权。

    这里的授权事实来自服务端资产目录与已过滤的正式 Schema 快照。表名前缀只
    作为未知内部表的最后一道拒绝规则，绝不作为“允许”身份依据。
    """

    refs = extract_physical_table_refs(sql)
    if access is None:
        internal = {
            table for table in refs
            if table.startswith(INTERNAL_TABLE_PREFIXES)
        }
        if internal:
            raise SQLSecurityError("普通查询不得访问上传、评测或系统内部数据表")
        return refs

    allowed = access.allowed_physical_tables
    denied = sorted(refs - allowed)
    if denied:
        raise SQLSecurityError(
            "SQL 引用了未在本轮数据源上下文授权的表：" + ", ".join(denied)
        )
    return refs


def ensure_context_assets_current(access: AssetAccessContext) -> None:
    now = datetime.now(UTC).replace(tzinfo=None)
    for asset in access.upload_assets:
        if asset.workspace_id != access.workspace_id or asset.owner_id != access.owner_id:
            raise DatasetAccessError("上传数据集不属于当前用户或工作空间")
        if asset.state not in READY_DATASET_STATES:
            raise DatasetAccessError(
                f"上传数据集版本当前不可查询：state={asset.state}",
                code="dataset_version_unavailable",
            )
        if asset.expires_at and asset.expires_at <= now:
            raise DatasetExpiredError("上传数据集版本已过期，请重新上传或续期")


def public_source_context(
    access: AssetAccessContext,
    profiles: Iterable[Mapping] = (),
    mapping: Mapping | None = None,
) -> dict:
    payload = access.public_dict()
    payload["profiles"] = [dict(item) for item in profiles]
    payload["mapping"] = dict(mapping or {})
    return payload
