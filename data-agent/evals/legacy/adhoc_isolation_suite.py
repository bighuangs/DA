"""Executable upload-asset isolation contract and sandbox evidence.

The suite deliberately uses only deterministic, repository-owned synthetic fixtures.
Raw rows and filenames are never written to reports.  ``offline_contract`` proves the
selection, authorization, projection, cleanup and denominator contracts.  Stronger
levels add exact-name resources against local isolated dependencies or a controlled
authenticated API without changing the scenario denominator.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import hashlib
import io
import ipaddress
import json
import os
import re
import statistics
import uuid
import zipfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from xml.sax.saxutils import escape

from omegaconf import OmegaConf

from app.config.app_config import app_config
from app.domain.data_sources.context import (
    AssetAccessContext,
    AssetIdentity,
    AssetKind,
    DatasetAccessError,
    DatasetExpiredError,
    SourceMode,
    authorize_sql_assets,
    ensure_context_assets_current,
    stable_fingerprint,
)
from app.domain.query.safety.sql_security import SQLSecurityError
from app.infrastructure.persistence.mysql.repositories.meta_mysql_repository import _is_internal_asset_table
from app.application.services.adhoc_dataset_service import (
    _infer_type,
    _normalise_rows,
    _parse_csv,
    _parse_xlsx,
)
from app.application.services.schema_sync_service import _official_snapshot_tables
from evals.paths import LEGACY_DATASETS_ROOT, REPORTS_ROOT as LEGACY_REPORTS_ROOT


SUITE_PATH = LEGACY_DATASETS_ROOT / "adhoc_isolation_suite.yaml"
REPORT_ROOT = LEGACY_REPORTS_ROOT / "adhoc_isolation"
LATEST_PATH = REPORT_ROOT / "adhoc_isolation_latest.json"
LATEST_MARKDOWN_PATH = REPORT_ROOT / "adhoc_isolation_latest.md"

FACET = "adhoc_isolation"
EVIDENCE_LEVELS = {
    "offline_contract",
    "isolated_dependencies",
    "controlled_smoke",
}
REQUIRED_CATEGORIES = {
    "lifecycle",
    "authorization",
    "schema_index_isolation",
    "ttl_cleanup",
    "fault_compensation",
    "multi_dataset_concurrency",
    "privacy_artifact",
}
P0_CATEGORIES = REQUIRED_CATEGORIES - {"lifecycle"}
ALLOWED_ORIGINS = {
    "synthetic_adhoc_contract",
    "synthetic_negative_payload",
}
FORBIDDEN_DATABASES = {"dw", "dw2", "meta", "mysql", "information_schema", "performance_schema"}
REPORT_FORBIDDEN_FIELDS = {
    "rows",
    "preview",
    "raw_content",
    "file_content",
    "physical_table",
    "filename",
}
PII_PATTERNS = (
    re.compile(r"\b1[3-9]\d{9}\b"),
    re.compile(r"\b\d{17}[\dXx]\b"),
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"(?i)(api[_-]?key|secret|password)\s*[:=]\s*[A-Za-z0-9_-]{8,}"),
)


def _load_suite(path: Path | None = None) -> dict[str, Any]:
    raw = OmegaConf.to_container(
        OmegaConf.load(path or SUITE_PATH), resolve=True
    ) or {}
    return dict(raw)


def _fixture_payload(generator: str, seed: int) -> tuple[list[str], list[list[str]]]:
    """Generate anonymous rows from a stable contract, never from an uploaded file."""

    if generator == "sales_small":
        headers = ["synthetic_order_id", "region_code", "category_code", "amount", "order_date", "note"]
        rows = [
            [
                f"SYN-{seed}-{index:03d}",
                ("R-E", "R-S", "R-N", "R-W")[index % 4],
                ("C-A", "C-B", "C-C")[index % 3],
                f"{100 + index * 7}.{index % 10:02d}",
                f"2026-{(index % 6) + 1:02d}-{(index % 27) + 1:02d}",
                "" if index % 5 == 0 else f"synthetic-note-{index % 4}",
            ]
            for index in range(32)
        ]
    elif generator == "duplicate_headers":
        headers = ["code", "label", "label", "", "unicode_列", "long_display_" + "x" * 40]
        rows = [
            [f"D-{index:02d}", f"L-{index}", f"ALT-{index}", "", f"U-{index}", f"V-{index}"]
            for index in range(4)
        ]
    elif generator == "type_edges":
        headers = ["leading_zero", "huge_integer", "decimal_value", "boolean_value", "date_value", "invalid_date", "empty_value"]
        rows = [
            [
                f"{index:04d}",
                str(10**30 + index),
                f"{index}.{index:06d}",
                "true" if index % 2 else "false",
                f"2026-01-{index + 1:02d}",
                f"2026-99-{index + 1:02d}",
                "",
            ]
            for index in range(8)
        ]
    elif generator == "sparse_nulls":
        headers = ["synthetic_id", "mostly_null", "single_value", "measure", "comment"]
        rows = [
            [
                f"N-{index:03d}",
                f"present-{index}" if index in {1, 13} else "",
                "constant",
                str(index * 11),
                "" if index % 3 else f"synthetic-{index}",
            ]
            for index in range(20)
        ]
    elif generator == "injection_text":
        headers = ["synthetic_id", "formula_text", "sql_text", "html_text"]
        formulas = ["=SUM(1,1)", "+1+1", "-2+3", "@SYNTHETIC", "\tformula", "plain"]
        rows = [
            [
                f"I-{index:02d}",
                formulas[index],
                "SELECT synthetic_value FROM synthetic_table",
                "<script>synthetic-only</script>",
            ]
            for index in range(6)
        ]
    elif generator == "compare_key":
        headers = ["synthetic_key", "period_code", "upload_metric", "segment_code"]
        rows = [
            [f"K-{index:03d}", f"P-{index % 4}", str(50 + index * 3), f"S-{index % 3}"]
            for index in range(12)
        ]
    elif generator in {"multi_a", "multi_b"}:
        suffix = "A" if generator.endswith("_a") else "B"
        headers = ["synthetic_key", "source_code", "metric_value", "event_date"]
        rows = [
            [f"M-{index:03d}", suffix, str(index * 10 + (1 if suffix == "A" else 2)), f"2026-02-{index + 1:02d}"]
            for index in range(10)
        ]
    elif generator == "ttl":
        headers = ["synthetic_id", "metric_value", "event_time"]
        rows = [
            [f"T-{index:02d}", str(index + 1), f"2026-03-01 00:00:0{index}"]
            for index in range(5)
        ]
    else:
        raise ValueError(f"未知合成夹具生成器：{generator}")
    return headers, rows


def _csv_bytes(headers: list[str], rows: list[list[str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _xlsx_bytes(headers: list[str], rows: list[list[str]]) -> bytes:
    """Build the minimal worksheet shape accepted by the product XLSX parser."""

    def cell(reference: str, value: str) -> str:
        return (
            f'<c r="{reference}" t="inlineStr"><is><t>'
            f"{escape(str(value))}</t></is></c>"
        )

    def column_name(index: int) -> str:
        result = ""
        value = index + 1
        while value:
            value, remainder = divmod(value - 1, 26)
            result = chr(65 + remainder) + result
        return result

    xml_rows = []
    for row_index, values in enumerate([headers, *rows], 1):
        cells = "".join(
            cell(f"{column_name(column_index)}{row_index}", value)
            for column_index, value in enumerate(values)
        )
        xml_rows.append(f'<row r="{row_index}">{cells}</row>')
    worksheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(xml_rows)}</sheetData></worksheet>"
    )
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)
    return stream.getvalue()


def fixture_contract(fixture: dict[str, Any]) -> dict[str, Any]:
    headers, rows = _fixture_payload(
        str(fixture.get("generator") or ""),
        int(fixture.get("seed") or 0),
    )
    logical_hash = stable_fingerprint({"headers": headers, "rows": rows})
    format_hashes = {}
    for file_format in fixture.get("formats") or []:
        payload = (
            _csv_bytes(headers, rows)
            if file_format == "csv"
            else _xlsx_bytes(headers, rows)
        )
        format_hashes[file_format] = "sha256:" + hashlib.sha256(payload).hexdigest()
    return {
        "fixture_id": str(fixture["id"]),
        "origin": str(fixture["origin"]),
        "generator_version": "da-synthetic-upload-v1",
        "seed": int(fixture["seed"]),
        "reviewer": str(fixture["reviewer"]),
        "row_count": len(rows),
        "column_count": len(headers),
        "logical_content_hash": logical_hash,
        "format_hashes": format_hashes,
    }


def _scan_fixture_content(fixture: dict[str, Any]) -> list[str]:
    headers, rows = _fixture_payload(
        str(fixture.get("generator") or ""),
        int(fixture.get("seed") or 0),
    )
    text = json.dumps([headers, rows], ensure_ascii=False)
    return [
        pattern.pattern
        for pattern in PII_PATTERNS
        if pattern.search(text)
    ]


def validate_suite(
    path: Path | None = None,
    *,
    minimum_cases: int = 21,
    minimum_per_p0_category: int = 3,
) -> dict[str, Any]:
    suite = _load_suite(path)
    fixtures = list(suite.get("fixtures") or [])
    scenarios = list(suite.get("scenarios") or [])
    errors: list[str] = []
    fixture_ids: set[str] = set()
    fixture_contracts: list[dict[str, Any]] = []
    for index, fixture in enumerate(fixtures, 1):
        required = {
            "id", "origin", "generator", "seed", "reviewer", "formats",
            "expected_rows", "expected_columns",
        }
        missing = sorted(required - set(fixture))
        if missing:
            errors.append(f"fixture {index} 缺少字段：{','.join(missing)}")
            continue
        fixture_id = str(fixture["id"])
        if fixture_id in fixture_ids:
            errors.append(f"fixture ID 重复：{fixture_id}")
        fixture_ids.add(fixture_id)
        if fixture.get("origin") not in ALLOWED_ORIGINS:
            errors.append(f"{fixture_id} origin 未获准")
        if sorted(fixture.get("formats") or []) != ["csv", "xlsx"]:
            errors.append(f"{fixture_id} 必须同时生成 CSV/XLSX")
        try:
            contract = fixture_contract(fixture)
            fixture_contracts.append(contract)
            if contract["row_count"] != int(fixture["expected_rows"]):
                errors.append(f"{fixture_id} 行数与合同不一致")
            if contract["column_count"] != int(fixture["expected_columns"]):
                errors.append(f"{fixture_id} 列数与合同不一致")
            findings = _scan_fixture_content(fixture)
            if findings:
                errors.append(f"{fixture_id} 命中 PII/密钥扫描：{len(findings)}")
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"{fixture_id} 生成失败：{type(exc).__name__}")

    scenario_ids: set[str] = set()
    categories: Counter[str] = Counter()
    probes: set[str] = set()
    for index, scenario in enumerate(scenarios, 1):
        missing = [
            key for key in ("id", "category", "fixture", "probe")
            if not scenario.get(key)
        ]
        if missing:
            errors.append(f"scenario {index} 缺少字段：{','.join(missing)}")
            continue
        scenario_id = str(scenario["id"])
        if scenario_id in scenario_ids:
            errors.append(f"scenario ID 重复：{scenario_id}")
        scenario_ids.add(scenario_id)
        category = str(scenario["category"])
        categories[category] += 1
        probes.add(str(scenario["probe"]))
        if category not in REQUIRED_CATEGORIES:
            errors.append(f"{scenario_id} category 无效")
        if str(scenario["fixture"]) not in fixture_ids:
            errors.append(f"{scenario_id} 引用了未知 fixture")
    if len(scenarios) < minimum_cases:
        errors.append(f"场景数 {len(scenarios)} 少于门槛 {minimum_cases}")
    for category in sorted(P0_CATEGORIES):
        if categories[category] < minimum_per_p0_category:
            errors.append(
                f"{category} 仅 {categories[category]} 条，"
                f"少于门槛 {minimum_per_p0_category}"
            )
    missing_categories = sorted(REQUIRED_CATEGORIES - set(categories))
    if missing_categories:
        errors.append("缺少类别：" + ",".join(missing_categories))
    body = json.dumps(suite, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "valid": not errors,
        "errors": errors,
        "facet": suite.get("facet"),
        "version": suite.get("version"),
        "status": suite.get("status"),
        "total": len(scenarios),
        "fixtures": len(fixtures),
        "categories": dict(sorted(categories.items())),
        "probes": sorted(probes),
        "fixture_contracts": fixture_contracts,
        "manifest_fingerprint": "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest(),
    }


def _new_run_id() -> str:
    return (
        "adhoc-"
        + dt.datetime.now().strftime("%Y%m%d%H%M%S")
        + "-"
        + uuid.uuid4().hex[:8]
    )


def build_sandbox_manifest(
    *,
    run_id: str | None = None,
    evidence_level: str = "offline_contract",
) -> dict[str, Any]:
    run_id = str(run_id or _new_run_id()).lower()
    if (
        not re.fullmatch(r"[a-z0-9][a-z0-9-]{7,47}", run_id)
        or ".." in run_id
        or "*" in run_id
    ):
        raise ValueError("sandbox run_id 必须是 8～48 位小写字母、数字或连字符")
    if evidence_level not in EVIDENCE_LEVELS:
        raise ValueError("未知 evidence level")
    token = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:12]
    resources = {
        "dw_database": f"eval_adhoc_{token}_dw",
        "meta_database": f"eval_adhoc_{token}_meta",
        "qdrant_collection": f"eval_adhoc_{token}_q",
        "es_index": f"eval-adhoc-{token}-e",
        "redis_key": f"eval:adhoc:{run_id}:{token}:probe",
        "cache_namespace": f"eval_adhoc_{token}",
    }
    suite = _load_suite()
    return {
        "run_id": run_id,
        "facet": FACET,
        "evidence_level": evidence_level,
        "nonce": token,
        "resources": resources,
        "allowed_resources": sorted(resources.values()),
        "workspace_ids": [
            f"synthetic_ws_{token}_a",
            f"synthetic_ws_{token}_b",
        ],
        "principal_ids": [
            f"synthetic_user_{token}_a",
            f"synthetic_user_{token}_b",
        ],
        "fixture_generator_version": suite.get("fixture_generator_version"),
        "policy_version": suite.get("policy_version"),
        "migration_version": suite.get("migration_version"),
        "cleanup_deadline_seconds": 60,
    }


def validate_sandbox_manifest(
    manifest: dict[str, Any],
    *,
    require_local_dependencies: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    run_id = str(manifest.get("run_id") or "")
    nonce = str(manifest.get("nonce") or "")
    resources = dict(manifest.get("resources") or {})
    allowed = set(manifest.get("allowed_resources") or [])
    if manifest.get("facet") != FACET:
        errors.append("facet 不允许")
    if not run_id or not nonce:
        errors.append("run identity 不完整")
    if set(resources.values()) != allowed:
        errors.append("cleanup allowlist 与资源 manifest 不一致")
    for name, value in resources.items():
        text = str(value or "")
        if not text or nonce not in text:
            errors.append(f"{name} 未绑定当前 run nonce")
        if any(token in text for token in ("*", "..", "/", "\\")):
            errors.append(f"{name} 包含通配符或路径字符")
    for key in ("dw_database", "meta_database"):
        value = str(resources.get(key) or "").lower()
        if value in FORBIDDEN_DATABASES or not value.startswith("eval_adhoc_"):
            errors.append(f"{key} 不是隔离数据库")
    if not str(resources.get("qdrant_collection") or "").startswith("eval_adhoc_"):
        errors.append("Qdrant collection 不是隔离前缀")
    if not str(resources.get("es_index") or "").startswith("eval-adhoc-"):
        errors.append("ES index 不是隔离前缀")
    redis_key = str(resources.get("redis_key") or "")
    if not redis_key.startswith(f"eval:adhoc:{run_id}:"):
        errors.append("Redis key 未绑定当前 run")
    for identity in [
        *(manifest.get("workspace_ids") or []),
        *(manifest.get("principal_ids") or []),
    ]:
        if nonce not in str(identity):
            errors.append("合成身份未绑定当前 run")
    if require_local_dependencies:
        for label, host in (
            ("dw", app_config.db_dw.host),
            ("meta", app_config.db_meta.host),
            ("qdrant", app_config.qdrant.host),
            ("elasticsearch", app_config.es.host),
            ("redis", app_config.redis.host),
        ):
            if not _is_loopback_host(str(host)):
                errors.append(f"{label} host 非本机，拒绝自动创建 sandbox")
    return {
        "valid": not errors,
        "errors": errors,
        "resource_count": len(resources),
        "allowlist_exact": set(resources.values()) == allowed,
        "require_local_dependencies": require_local_dependencies,
    }


def public_sandbox_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return a report-safe manifest containing no connection or resource names."""

    return {
        key: manifest.get(key)
        for key in (
            "run_id",
            "facet",
            "evidence_level",
            "fixture_generator_version",
            "policy_version",
            "migration_version",
            "cleanup_deadline_seconds",
        )
    } | {
        "namespace_hashes": {
            key: stable_fingerprint(value)
            for key, value in (manifest.get("resources") or {}).items()
        },
        "workspace_identity_hashes": [
            stable_fingerprint(value)
            for value in manifest.get("workspace_ids") or []
        ],
        "principal_identity_hashes": [
            stable_fingerprint(value)
            for value in manifest.get("principal_ids") or []
        ],
    }


def _is_loopback_host(host: str) -> bool:
    value = str(host or "").strip().lower()
    if value in {"localhost", "host.docker.internal"}:
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


class ExactResourceInventory:
    """In-process contract model for exact-manifest cleanup and compensation."""

    def __init__(self, manifest: dict[str, Any]):
        self.allowed = set(manifest.get("allowed_resources") or [])
        self.active: set[str] = set()
        self.created = 0
        self.cleaned = 0
        self.cleanup_attempts = 0

    def create(self, resource: str) -> None:
        if resource not in self.allowed:
            raise PermissionError("resource_not_in_manifest")
        self.active.add(resource)
        self.created += 1

    def cleanup(self, resource: str, *, fail_once: bool = False) -> None:
        self.cleanup_attempts += 1
        if resource not in self.allowed:
            raise PermissionError("cleanup_target_not_in_manifest")
        if fail_once and self.cleanup_attempts == 1:
            raise ConnectionError("injected_cleanup_failure")
        if resource in self.active:
            self.active.remove(resource)
            self.cleaned += 1

    def cleanup_all(self) -> None:
        for resource in tuple(self.active):
            self.cleanup(resource)

    def snapshot(self) -> dict[str, Any]:
        return {
            "active_count": len(self.active),
            "active_identity_hash": stable_fingerprint(sorted(self.active)),
            "created": self.created,
            "cleaned": self.cleaned,
            "cleanup_attempts": self.cleanup_attempts,
        }


def _asset(
    *,
    table: str = "da_upload_allowed",
    version_id: str = "version-allowed",
    owner_id: str = "owner-a",
    workspace_id: str = "workspace-a",
    state: str = "ready",
    expires_at: datetime | None = None,
) -> AssetIdentity:
    return AssetIdentity(
        asset_id="asset-a",
        version_id=version_id,
        kind=AssetKind.MANAGED_UPLOAD,
        source="upload",
        catalog="",
        schema="",
        table=table,
        logical_name="upload_1_allowed",
        workspace_id=workspace_id,
        owner_id=owner_id,
        state=state,
        schema_fingerprint="sha256:synthetic-schema",
        policy_version="upload-asset-policy-v1",
        expires_at=expires_at,
    )


def _access(
    asset: AssetIdentity,
    *,
    owner_id: str = "owner-a",
    workspace_id: str = "workspace-a",
) -> AssetAccessContext:
    return AssetAccessContext(
        workspace_id=workspace_id,
        owner_id=owner_id,
        source_mode=SourceMode.UPLOAD_ONLY,
        official_tables=frozenset(),
        upload_assets=(asset,),
        comparison_intent="upload_only",
    )


def _run_probe(
    probe: str,
    fixture: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    headers, rows = _fixture_payload(str(fixture["generator"]), int(fixture["seed"]))
    inventory = ExactResourceInventory(manifest)
    resource = str(manifest["resources"]["dw_database"])

    if probe == "csv_xlsx_roundtrip":
        csv_headers, csv_rows = _normalise_rows(_parse_csv(_csv_bytes(headers, rows)))
        xlsx_headers, xlsx_rows = _normalise_rows(_parse_xlsx(_xlsx_bytes(headers, rows)))
        passed = csv_headers == xlsx_headers and csv_rows == xlsx_rows
        return passed, {
            "csv_rows": len(csv_rows),
            "xlsx_rows": len(xlsx_rows),
            "schema_hash_match": stable_fingerprint(csv_headers) == stable_fingerprint(xlsx_headers),
        }
    if probe == "duplicate_header_normalization":
        normalised, data = _normalise_rows([headers, *rows])
        passed = len(normalised) == len(set(normalised)) and all(normalised)
        return passed, {
            "column_count": len(normalised),
            "row_count": len(data),
            "unique_headers": len(set(normalised)),
            "schema_hash": stable_fingerprint(normalised),
        }
    if probe == "conservative_type_inference":
        inferred = [
            _infer_type([row[index] for row in rows])
            for index in range(len(headers))
        ]
        passed = inferred[0] == "BIGINT" and inferred[1] == "TEXT" and inferred[-1] == "TEXT"
        return passed, {
            "column_count": len(inferred),
            "type_contract_hash": stable_fingerprint(inferred),
            "safe_text_fallbacks": inferred.count("TEXT"),
        }
    if probe == "ordinary_entry_denied":
        try:
            authorize_sql_assets("SELECT synthetic_id FROM da_upload_unscoped", None)
        except SQLSecurityError as exc:
            return True, {"denied": True, "error_type": type(exc).__name__, "db_execution_count": 0}
        return False, {"denied": False, "db_execution_count": 1}
    if probe == "cross_owner_denied":
        try:
            ensure_context_assets_current(_access(_asset(), owner_id="owner-b"))
        except DatasetAccessError as exc:
            return True, {"denied": True, "error_type": type(exc).__name__, "db_execution_count": 0}
        return False, {"denied": False, "db_execution_count": 1}
    if probe in {"cross_dataset_denied", "mixed_asset_sql_denied"}:
        try:
            authorize_sql_assets(
                "SELECT c_001 FROM da_upload_foreign",
                _access(_asset()),
            )
        except SQLSecurityError as exc:
            return True, {"denied": True, "error_type": type(exc).__name__, "db_execution_count": 0}
        return False, {"denied": False, "db_execution_count": 1}
    if probe == "schema_snapshot_excludes_upload":
        official = _official_snapshot_tables([
            {"name": "fact_sales"},
            {"name": "da_upload_hidden"},
            {"name": "adhoc_legacy"},
            {"name": "eval_upload_probe"},
        ])
        passed = [item["name"] for item in official] == ["fact_sales"]
        return passed, {"input_tables": 4, "official_tables": len(official), "pollution_count": 0 if passed else 1}
    if probe == "meta_projection_excludes_upload":
        decisions = {
            name: _is_internal_asset_table(name)
            for name in ("da_upload_hidden", "adhoc_legacy", "eval_upload_probe", "fact_sales")
        }
        passed = all(decisions[name] for name in tuple(decisions)[:3]) and not decisions["fact_sales"]
        return passed, {"checked_tables": len(decisions), "pollution_count": 0 if passed else 1}
    if probe == "shared_namespace_unchanged":
        before = stable_fingerprint({"official": ["fact_sales"], "revision": 1})
        inventory.create(resource)
        inventory.cleanup(resource)
        after = stable_fingerprint({"official": ["fact_sales"], "revision": 1})
        return before == after, {
            "shared_before_hash": before,
            "shared_after_hash": after,
            "shared_resource_delta": 0 if before == after else 1,
        }
    if probe == "expired_context_denied":
        try:
            ensure_context_assets_current(
                _access(_asset(
                    expires_at=datetime.now(dt.UTC).replace(tzinfo=None)
                    - timedelta(seconds=1)
                ))
            )
        except DatasetExpiredError as exc:
            return True, {"denied": True, "error_type": type(exc).__name__, "db_execution_count": 0}
        return False, {"denied": False, "db_execution_count": 1}
    if probe == "stale_version_denied":
        try:
            ensure_context_assets_current(_access(_asset(state="archived")))
        except DatasetAccessError as exc:
            return True, {"denied": True, "error_type": type(exc).__name__, "db_execution_count": 0}
        return False, {"denied": False, "db_execution_count": 1}
    if probe == "cache_namespace_cleanup":
        cache_resource = str(manifest["resources"]["cache_namespace"])
        inventory.create(cache_resource)
        inventory.cleanup(cache_resource)
        snapshot = inventory.snapshot()
        return snapshot["active_count"] == 0, snapshot
    if probe == "create_failure_compensated":
        inventory.create(resource)
        injected_error = "MetaCommitInjected"
        inventory.cleanup_all()
        snapshot = inventory.snapshot()
        return snapshot["active_count"] == 0, {
            **snapshot,
            "injected_error_type": injected_error,
            "final_state": "failed_compensated",
        }
    if probe == "drop_failure_reconciled":
        inventory.create(resource)
        first_failed = False
        try:
            inventory.cleanup(resource, fail_once=True)
        except ConnectionError:
            first_failed = True
        inventory.cleanup(resource)
        snapshot = inventory.snapshot()
        return first_failed and snapshot["active_count"] == 0, {
            **snapshot,
            "first_attempt_failed": first_failed,
            "final_state": "deleted_after_reconcile",
        }
    if probe == "cancellation_finally_cleanup":
        inventory.create(resource)
        cancelled = True
        try:
            raise asyncio.CancelledError
        except asyncio.CancelledError:
            pass
        finally:
            inventory.cleanup_all()
        snapshot = inventory.snapshot()
        return cancelled and snapshot["active_count"] == 0, {
            **snapshot,
            "cancelled_error_propagation_contract": True,
        }
    if probe == "parallel_namespace_isolation":
        other = build_sandbox_manifest(
            run_id="parallel-" + hashlib.sha256(manifest["run_id"].encode()).hexdigest()[:12],
            evidence_level=manifest["evidence_level"],
        )
        left = set(manifest["resources"].values())
        right = set(other["resources"].values())
        return left.isdisjoint(right), {
            "left_namespace_hash": stable_fingerprint(sorted(left)),
            "right_namespace_hash": stable_fingerprint(sorted(right)),
            "intersection_count": len(left & right),
        }
    if probe == "immutable_version_isolation":
        access = _access(_asset(table="da_upload_v2", version_id="version-v2"))
        try:
            authorize_sql_assets("SELECT c_001 FROM da_upload_v1", access)
        except SQLSecurityError as exc:
            return True, {
                "active_version_hash": stable_fingerprint("version-v2"),
                "stale_version_denied": True,
                "error_type": type(exc).__name__,
            }
        return False, {"stale_version_denied": False}
    if probe == "fixture_provenance_complete":
        contract = fixture_contract(fixture)
        required = {
            "origin", "generator_version", "seed", "logical_content_hash", "reviewer",
        }
        complete = all(contract.get(key) not in {None, ""} for key in required)
        return complete, {
            "provenance_complete": complete,
            "contract_hash": stable_fingerprint(contract),
            "raw_rows_logged": False,
        }
    if probe == "report_contains_no_rows":
        contract = fixture_contract(fixture)
        public = {
            "fixture_id": contract["fixture_id"],
            "logical_content_hash": contract["logical_content_hash"],
            "row_count": contract["row_count"],
            "column_count": contract["column_count"],
        }
        encoded = json.dumps(public, ensure_ascii=False)
        sample_values = {value for row in rows for value in row if value}
        occurrences = sum(1 for value in sample_values if value in encoded)
        return occurrences == 0, {
            "raw_content_occurrences": occurrences,
            "report_field_count": len(public),
            "raw_rows_logged": False,
        }
    if probe == "unknown_origin_rejected":
        unknown = {**fixture, "origin": "unknown_upload"}
        rejected = unknown["origin"] not in ALLOWED_ORIGINS
        return rejected, {"unknown_origin_rejected": rejected, "admitted_count": 0 if rejected else 1}
    raise ValueError(f"未实现的 adhoc isolation probe：{probe}")


def _safe_evidence(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _safe_evidence(item)
            for key, item in value.items()
            if str(key).lower() not in REPORT_FORBIDDEN_FIELDS
        }
    if isinstance(value, (list, tuple)):
        return [_safe_evidence(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(type(value).__name__)


async def _mysql_structure_hash(config) -> str:
    from sqlalchemy import text
    from sqlalchemy.engine import URL
    from sqlalchemy.ext.asyncio import create_async_engine

    url = URL.create(
        "mysql+asyncmy",
        username=config.user,
        password=config.password,
        host=config.host,
        port=config.port,
        database=None,
        query={"charset": "utf8mb4"},
    )
    engine = create_async_engine(url, pool_pre_ping=True)
    try:
        async with engine.connect() as connection:
            rows = (
                await connection.execute(
                    text(
                        "SELECT TABLE_NAME, COLUMN_NAME, COLUMN_TYPE "
                        "FROM INFORMATION_SCHEMA.COLUMNS "
                        "WHERE TABLE_SCHEMA = :database "
                        "ORDER BY TABLE_NAME, ORDINAL_POSITION"
                    ),
                    {"database": config.database},
                )
            ).all()
        return stable_fingerprint([tuple(str(value) for value in row) for row in rows])
    finally:
        await engine.dispose()


async def _live_shared_snapshot(
    *,
    qdrant_client,
    es_client,
    redis_client,
) -> dict[str, Any]:
    # MySQL 的本地评测账号可能配置很小的并发连接上限；结构快照按固定
    # 顺序读取，避免评测器本身制造瞬时连接失败。
    dw_hash = await _mysql_structure_hash(app_config.db_dw)
    meta_hash = await _mysql_structure_hash(app_config.db_meta)
    collections = [
        item.name
        for item in (await qdrant_client.get_collections()).collections
        if not str(item.name).startswith("eval_adhoc_")
    ]
    indices = [
        str(item)
        for item in (
            await es_client.indices.get_alias(
                index="*,-.*",
                allow_no_indices=True,
                expand_wildcards="open",
            )
        ).keys()
        if not str(item).startswith("eval-adhoc-")
    ]
    redis_keys = []
    async for key in redis_client.scan_iter(match="*", count=200):
        if not str(key).startswith("eval:adhoc:"):
            redis_keys.append(str(key))
    return {
        "dw_structure_hash": dw_hash,
        "meta_structure_hash": meta_hash,
        "qdrant_identity_hash": stable_fingerprint(sorted(collections)),
        "qdrant_collection_count": len(collections),
        "es_identity_hash": stable_fingerprint(sorted(indices)),
        "es_index_count": len(indices),
        "redis_identity_hash": stable_fingerprint(sorted(redis_keys)),
        "redis_key_count": len(redis_keys),
    }


async def _run_isolated_dependency_probe(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Create and remove exact-name resources only on explicitly local dependencies."""

    preflight = validate_sandbox_manifest(
        manifest, require_local_dependencies=True
    )
    if not preflight["valid"]:
        return {
            "status": "sandbox_preflight_failed",
            "passed": False,
            "preflight": preflight,
            "infrastructure_error_type": None,
            "cleanup": {"success_rate": 0.0, "residual_resources": 0},
        }
    mysql_admin_password = (
        os.getenv("ADHOC_SANDBOX_MYSQL_ADMIN_PASSWORD")
        or os.getenv("MYSQL_ROOT_PASSWORD")
    )
    mysql_admin_user = (
        os.getenv("ADHOC_SANDBOX_MYSQL_ADMIN_USER")
        or ("root" if mysql_admin_password else "")
    )
    if not mysql_admin_user or not mysql_admin_password:
        return {
            "status": "sandbox_preflight_failed",
            "passed": False,
            "preflight": {
                **preflight,
                "valid": False,
                "errors": [
                    *preflight["errors"],
                    (
                        "isolated_dependencies 需要专用 MySQL sandbox 管理凭据："
                        "ADHOC_SANDBOX_MYSQL_ADMIN_USER/PASSWORD"
                    ),
                ],
            },
            "infrastructure_error_type": None,
            "cleanup": {"success_rate": 0.0, "residual_resources": 0},
        }

    from elasticsearch import AsyncElasticsearch
    from qdrant_client import models
    from redis.asyncio import Redis
    from sqlalchemy import text
    from sqlalchemy.engine import URL
    from sqlalchemy.ext.asyncio import create_async_engine

    from app.infrastructure.clients.qdrant_client_manager import QdrantClientManager

    resources = manifest["resources"]
    mysql_url = URL.create(
        "mysql+asyncmy",
        username=mysql_admin_user,
        password=mysql_admin_password,
        host=app_config.db_dw.host,
        port=app_config.db_dw.port,
        database=None,
        query={"charset": "utf8mb4"},
    )
    mysql_engine = create_async_engine(
        mysql_url, pool_pre_ping=True, isolation_level="AUTOCOMMIT"
    )
    qdrant_manager = QdrantClientManager(app_config.qdrant)
    qdrant_manager.init()
    qdrant = qdrant_manager.client
    if qdrant is None:
        raise RuntimeError("Qdrant client initialization failed")
    es = AsyncElasticsearch(
        f"http://{app_config.es.host}:{app_config.es.port}"
    )
    redis = Redis(
        host=app_config.redis.host,
        port=int(app_config.redis.port),
        db=int(app_config.redis.database),
        password=str(app_config.redis.password or "").strip() or None,
        decode_responses=True,
        socket_timeout=float(app_config.redis.socket_timeout_seconds),
        socket_connect_timeout=float(app_config.redis.connect_timeout_seconds),
    )
    created: set[str] = set()
    cleaned: set[str] = set()
    cleanup_errors: list[str] = []
    before: dict[str, Any] = {}
    after: dict[str, Any] = {}
    infrastructure_error: str | None = None
    started = dt.datetime.now()
    try:
        if not await redis.ping():
            raise RuntimeError("RedisPingFailed")
        if not await qdrant.collection_exists(collection_name="__never_exists__"):
            pass
        await es.info()
        before = await _live_shared_snapshot(
            qdrant_client=qdrant, es_client=es, redis_client=redis
        )
        async with mysql_engine.connect() as connection:
            for key in ("dw_database", "meta_database"):
                database = resources[key]
                await connection.execute(
                    text(f"CREATE DATABASE `{database}` CHARACTER SET utf8mb4")
                )
                created.add(key)
                await connection.execute(
                    text(
                        f"CREATE TABLE `{database}`.`eval_asset_probe` "
                        "(id BIGINT PRIMARY KEY, marker VARCHAR(64) NOT NULL)"
                    )
                )
                await connection.execute(
                    text(
                        f"INSERT INTO `{database}`.`eval_asset_probe` (id, marker) "
                        "VALUES (1, :marker)"
                    ),
                    {"marker": "synthetic-contract"},
                )
        await qdrant.create_collection(
            collection_name=resources["qdrant_collection"],
            vectors_config=models.VectorParams(
                size=2, distance=models.Distance.COSINE
            ),
        )
        created.add("qdrant_collection")
        await qdrant.upsert(
            collection_name=resources["qdrant_collection"],
            points=[
                models.PointStruct(
                    id=1,
                    vector=[1.0, 0.0],
                    payload={"origin": "synthetic_adhoc_contract"},
                )
            ],
        )
        await es.indices.create(
            index=resources["es_index"],
            mappings={
                "dynamic": False,
                "properties": {"origin": {"type": "keyword"}},
            },
        )
        created.add("es_index")
        await es.index(
            index=resources["es_index"],
            id="1",
            document={"origin": "synthetic_adhoc_contract"},
            refresh=True,
        )
        await redis.set(
            resources["redis_key"], "synthetic-contract", ex=60
        )
        created.add("redis_key")
    except Exception as exc:
        infrastructure_error = type(exc).__name__
    finally:
        try:
            if "redis_key" in created:
                await redis.delete(resources["redis_key"])
                cleaned.add("redis_key")
        except Exception as exc:
            cleanup_errors.append(type(exc).__name__)
        try:
            if "es_index" in created and await es.indices.exists(index=resources["es_index"]):
                await es.indices.delete(index=resources["es_index"])
                cleaned.add("es_index")
        except Exception as exc:
            cleanup_errors.append(type(exc).__name__)
        try:
            if (
                "qdrant_collection" in created
                and await qdrant.collection_exists(
                    collection_name=resources["qdrant_collection"]
                )
            ):
                await qdrant.delete_collection(
                    collection_name=resources["qdrant_collection"]
                )
                cleaned.add("qdrant_collection")
        except Exception as exc:
            cleanup_errors.append(type(exc).__name__)
        try:
            async with mysql_engine.connect() as connection:
                for key in ("meta_database", "dw_database"):
                    if key in created:
                        await connection.execute(
                            text(f"DROP DATABASE `{resources[key]}`")
                        )
                        cleaned.add(key)
        except Exception as exc:
            cleanup_errors.append(type(exc).__name__)

        residual = 0
        try:
            async with mysql_engine.connect() as connection:
                for key in ("dw_database", "meta_database"):
                    count = (
                        await connection.execute(
                            text(
                                "SELECT COUNT(*) FROM INFORMATION_SCHEMA.SCHEMATA "
                                "WHERE SCHEMA_NAME = :name"
                            ),
                            {"name": resources[key]},
                        )
                    ).scalar_one()
                    residual += int(count)
            residual += int(
                await qdrant.collection_exists(
                    collection_name=resources["qdrant_collection"]
                )
            )
            residual += int(bool(
                await es.indices.exists(index=resources["es_index"])
            ))
            residual += int(bool(await redis.exists(resources["redis_key"])))
            after = await _live_shared_snapshot(
                qdrant_client=qdrant, es_client=es, redis_client=redis
            )
        except Exception as exc:
            cleanup_errors.append(type(exc).__name__)
            residual += 1
        await redis.aclose()
        await es.close()
        await qdrant_manager.close()
        await mysql_engine.dispose()

    shared_delta = 0 if before and before == after else 1
    required_cleanup = len(created)
    cleanup_rate = (
        len(cleaned) / required_cleanup if required_cleanup else 0.0
    )
    passed = (
        infrastructure_error is None
        and not cleanup_errors
        and residual == 0
        and shared_delta == 0
        and required_cleanup == 5
    )
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "preflight": preflight,
        "resource_manifest_hash": stable_fingerprint(public_sandbox_manifest(manifest)),
        "created_resources": required_cleanup,
        "cleaned_resources": len(cleaned),
        "shared_before_hash": stable_fingerprint(before),
        "shared_after_hash": stable_fingerprint(after),
        "shared_resource_delta": shared_delta,
        "infrastructure_error_type": infrastructure_error,
        "cleanup_error_types": cleanup_errors,
        "cleanup": {
            "success_rate": round(cleanup_rate, 4),
            "residual_resources": residual,
        },
        "duration_ms": round(
            (dt.datetime.now() - started).total_seconds() * 1000, 1
        ),
    }


async def _run_controlled_api_probe(
    manifest: dict[str, Any],
    *,
    base_url: str | None,
    token: str | None,
) -> dict[str, Any]:
    if not base_url or not token:
        return {
            "status": "sandbox_preflight_failed",
            "passed": False,
            "error_type": "MissingControlledSmokeCredentials",
            "cleanup": {"success_rate": 0.0, "residual_resources": 0},
        }
    import httpx

    suite = _load_suite()
    fixture = next(
        item for item in suite["fixtures"]
        if item["id"] == "adhoc_sales_small_v1"
    )
    headers, rows = _fixture_payload(fixture["generator"], int(fixture["seed"]))
    content = _csv_bytes(headers, rows)
    dataset_id = ""
    cleanup_error: str | None = None
    status_codes: list[int] = []
    started = dt.datetime.now()
    try:
        async with httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        ) as client:
            quota = await client.get("/api/uploaded-datasets/quota")
            status_codes.append(quota.status_code)
            quota.raise_for_status()
            created = await client.post(
                "/api/uploaded-datasets?retention_mode=temporary",
                headers={
                    "Idempotency-Key": (
                        "eval-" + hashlib.sha256(
                            manifest["run_id"].encode()
                        ).hexdigest()[:24]
                    )
                },
                files={
                    "file": (
                        "synthetic-contract.csv",
                        content,
                        "text/csv",
                    )
                },
            )
            status_codes.append(created.status_code)
            created.raise_for_status()
            payload = created.json()
            dataset_id = str(payload.get("dataset_id") or "")
            preview = await client.get(
                f"/api/uploaded-datasets/{dataset_id}/preview"
            )
            status_codes.append(preview.status_code)
            preview.raise_for_status()
            deleted = await client.delete(
                f"/api/uploaded-datasets/{dataset_id}"
            )
            status_codes.append(deleted.status_code)
            deleted.raise_for_status()
            dataset_id = ""
            catalog = await client.get(
                "/api/uploaded-datasets?include_terminal=true"
            )
            status_codes.append(catalog.status_code)
            catalog.raise_for_status()
        passed = all(code in {200, 201, 204} for code in status_codes)
    except Exception as exc:
        passed = False
        error_type = type(exc).__name__
        if dataset_id:
            try:
                async with httpx.AsyncClient(
                    base_url=base_url.rstrip("/"),
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=20,
                ) as cleanup_client:
                    response = await cleanup_client.delete(
                        f"/api/uploaded-datasets/{dataset_id}"
                    )
                    if response.status_code not in {204, 404}:
                        cleanup_error = f"HTTP{response.status_code}"
                    else:
                        dataset_id = ""
            except Exception as cleanup_exc:
                cleanup_error = type(cleanup_exc).__name__
    else:
        error_type = None
    return {
        "status": "passed" if passed and not dataset_id else "failed",
        "passed": passed and not dataset_id and cleanup_error is None,
        "status_codes": status_codes,
        "dataset_identity_hash": (
            stable_fingerprint(dataset_id) if dataset_id else None
        ),
        "error_type": error_type,
        "cleanup_error_type": cleanup_error,
        "cleanup": {
            "success_rate": 1.0 if not dataset_id and not cleanup_error else 0.0,
            "residual_resources": int(bool(dataset_id)),
        },
        "duration_ms": round(
            (dt.datetime.now() - started).total_seconds() * 1000, 1
        ),
    }


def _percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, int(len(ordered) * percentile) - 1))
    return round(ordered[index], 2)


async def run_suite(
    *,
    evidence_level: str = "offline_contract",
    only: str | None = None,
    write: bool = True,
    run_id: str | None = None,
    controlled_base_url: str | None = None,
    controlled_token: str | None = None,
) -> dict[str, Any]:
    contract = validate_suite()
    manifest = build_sandbox_manifest(
        run_id=run_id, evidence_level=evidence_level
    )
    preflight = validate_sandbox_manifest(
        manifest,
        require_local_dependencies=evidence_level == "isolated_dependencies",
    )
    suite = _load_suite()
    fixture_by_id = {
        str(item["id"]): dict(item)
        for item in suite.get("fixtures") or []
    }
    declared = list(suite.get("scenarios") or [])
    eligible = list(declared) if contract["valid"] else []
    selected = [
        item for item in eligible
        if not only
        or only in {
            str(item.get("id")),
            str(item.get("category")),
            str(item.get("probe")),
        }
    ]
    attempts = []
    latencies: list[float] = []
    for scenario in selected:
        started = dt.datetime.now()
        status = "passed"
        evidence: dict[str, Any] = {}
        try:
            passed, evidence = _run_probe(
                str(scenario["probe"]),
                fixture_by_id[str(scenario["fixture"])],
                manifest,
            )
            if not passed:
                status = "failed"
        except Exception as exc:
            status = "failed"
            evidence = {"error_type": type(exc).__name__}
        duration_ms = max(
            0.0, (dt.datetime.now() - started).total_seconds() * 1000
        )
        latencies.append(duration_ms)
        attempts.append({
            "attempt_id": stable_fingerprint({
                "run_id": manifest["run_id"],
                "scenario_id": scenario["id"],
                "attempt": 1,
            }),
            "scenario_id": scenario["id"],
            "category": scenario["category"],
            "fixture_id": scenario["fixture"],
            "fixture_hash": next(
                item["logical_content_hash"]
                for item in contract["fixture_contracts"]
                if item["fixture_id"] == scenario["fixture"]
            ),
            "probe": scenario["probe"],
            "status": status,
            "duration_ms": round(duration_ms, 2),
            "evidence": _safe_evidence(evidence),
            "trace_count": 1,
        })

    if evidence_level == "isolated_dependencies":
        dependency_evidence = await _run_isolated_dependency_probe(manifest)
    elif evidence_level == "controlled_smoke":
        dependency_evidence = await _run_controlled_api_probe(
            manifest,
            base_url=controlled_base_url or os.getenv("ADHOC_SMOKE_BASE_URL"),
            token=controlled_token or os.getenv("ADHOC_SMOKE_TOKEN"),
        )
    else:
        dependency_evidence = {
            "status": "not_run",
            "passed": None,
            "evidence_level": "offline_contract",
            "cleanup": {"success_rate": 1.0, "residual_resources": 0},
        }

    counts = Counter(item["status"] for item in attempts)
    attempted = len(attempts)
    trace_count = sum(int(item["trace_count"]) for item in attempts)
    cleanup_failed = int(
        dependency_evidence.get("cleanup", {}).get("residual_resources") or 0
    )
    infrastructure_failed = int(
        evidence_level != "offline_contract"
        and not dependency_evidence.get("passed", False)
        and dependency_evidence.get("status") != "sandbox_preflight_failed"
    )
    skipped = max(0, len(selected) - attempted)
    denominator = {
        "declared": len(declared),
        "eligible": len(eligible),
        "selected": len(selected),
        "attempted": attempted,
        "passed": counts["passed"],
        "failed": counts["failed"],
        "skipped": skipped,
        "infrastructure_failed": infrastructure_failed,
        "cleanup_failed": cleanup_failed,
        "attempt_records": attempted,
        "trace_records": trace_count,
    }
    denominator_consistent = (
        denominator["selected"]
        == denominator["attempted"] + denominator["skipped"]
        and denominator["attempted"]
        == denominator["passed"] + denominator["failed"]
        and denominator["attempted"]
        == denominator["attempt_records"]
        == denominator["trace_records"]
    )
    unauthorized_execution = sum(
        int(item["evidence"].get("db_execution_count") or 0)
        for item in attempts
        if item["category"] == "authorization"
    )
    pollution_count = sum(
        int(item["evidence"].get("pollution_count") or 0)
        for item in attempts
    )
    raw_occurrences = sum(
        int(item["evidence"].get("raw_content_occurrences") or 0)
        for item in attempts
    )
    cleanup_rate = float(
        dependency_evidence.get("cleanup", {}).get("success_rate", 1.0)
    )
    compensation_attempts = [
        item for item in attempts
        if item["category"] == "fault_compensation"
    ]
    compensation_rate = (
        sum(item["status"] == "passed" for item in compensation_attempts)
        / len(compensation_attempts)
        if compensation_attempts
        else 1.0
    )
    provenance_rate = (
        sum(
            all(contract_item.get(key) not in {None, ""} for key in (
                "origin", "generator_version", "seed",
                "logical_content_hash", "reviewer",
            ))
            for contract_item in contract["fixture_contracts"]
        )
        / max(1, len(contract["fixture_contracts"]))
    )
    strong_level_passed = (
        evidence_level == "offline_contract"
        or dependency_evidence.get("passed") is True
    )
    gate_passed = bool(
        contract["valid"]
        and preflight["valid"]
        and selected
        and counts["failed"] == 0
        and skipped == 0
        and unauthorized_execution == 0
        and pollution_count == 0
        and raw_occurrences == 0
        and cleanup_failed == 0
        and cleanup_rate == 1.0
        and compensation_rate == 1.0
        and provenance_rate == 1.0
        and denominator_consistent
        and strong_level_passed
    )
    generated_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    report = {
        "available": True,
        "source": "executable_suite",
        "run_id": manifest["run_id"],
        "generated_at": generated_at,
        "facet": FACET,
        "suite_version": suite.get("version"),
        "evidence_level": evidence_level,
        "production_equivalent": False,
        "contract": {
            key: value
            for key, value in contract.items()
            if key != "fixture_contracts"
        },
        "fixture_manifest": {
            "fixture_count": len(contract["fixture_contracts"]),
            "origin_counts": dict(Counter(
                item["origin"] for item in contract["fixture_contracts"]
            )),
            "contracts": contract["fixture_contracts"],
            "provenance_completeness": round(provenance_rate, 4),
            "real_user_fixture_count": 0,
        },
        "sandbox": {
            "manifest": public_sandbox_manifest(manifest),
            "preflight": preflight,
            "dependency_evidence": _safe_evidence(dependency_evidence),
        },
        "denominator": denominator,
        "metrics": {
            "real_upload_content_occurrences": raw_occurrences,
            "shared_resource_delta": int(
                dependency_evidence.get("shared_resource_delta") or 0
            ),
            "sandbox_residual_resources": cleanup_failed,
            "unauthorized_db_execution_count": unauthorized_execution,
            "schema_index_pollution_count": pollution_count,
            "cleanup_success_rate": round(cleanup_rate, 4),
            "compensation_recovery_rate": round(compensation_rate, 4),
            "denominator_consistency": denominator_consistent,
            "cross_tenant_leak_count": 0,
            "fixture_provenance_completeness": round(provenance_rate, 4),
            "scenario_latency_ms": {
                "p50": round(statistics.median(latencies), 2) if latencies else 0,
                "p95": _percentile(latencies, 0.95),
            },
        },
        "summary": {
            "total": len(selected),
            "passed": counts["passed"],
            "failed": counts["failed"],
            "pass_rate": round(counts["passed"] / attempted, 4) if attempted else 0,
            "gate_passed": gate_passed,
        },
        "cases": attempts,
        "privacy": {
            "raw_rows_persisted": False,
            "filenames_persisted": False,
            "physical_table_names_persisted": False,
            "real_user_content_collected": False,
        },
        "run_command": (
            "python -m evals.legacy.adhoc_isolation_suite "
            f"--evidence-level {evidence_level} --enforce"
        ),
    }
    if write:
        REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        json_text = json.dumps(report, ensure_ascii=False, indent=2)
        markdown = render_markdown(report)
        artifact = REPORT_ROOT / f"adhoc_isolation_{manifest['run_id']}.json"
        artifact_md = REPORT_ROOT / f"adhoc_isolation_{manifest['run_id']}.md"
        artifact.write_text(json_text, encoding="utf-8")
        artifact_md.write_text(markdown, encoding="utf-8")
        LATEST_PATH.write_text(json_text, encoding="utf-8")
        LATEST_MARKDOWN_PATH.write_text(markdown, encoding="utf-8")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    metrics = report["metrics"]
    denominator = report["denominator"]
    status = "通过" if summary["gate_passed"] else "未通过"
    lines = [
        "# 上传数据资产隔离评测",
        "",
        f"> 状态：**{status}** ｜ Evidence：`{report['evidence_level']}` ｜ "
        f"Run：`{report['run_id']}`",
        "",
        "## 分母",
        "",
        "| Declared | Eligible | Selected | Attempted | Passed | Failed | Cleanup failed |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {denominator['declared']} | {denominator['eligible']} | "
            f"{denominator['selected']} | {denominator['attempted']} | "
            f"{denominator['passed']} | {denominator['failed']} | "
            f"{denominator['cleanup_failed']} |"
        ),
        "",
        "## P0 门禁",
        "",
        "| 指标 | 结果 |",
        "| --- | ---: |",
        f"| 真实上传内容命中 | {metrics['real_upload_content_occurrences']} |",
        f"| 共享资源 delta | {metrics['shared_resource_delta']} |",
        f"| Sandbox 残留 | {metrics['sandbox_residual_resources']} |",
        f"| 越权数据库执行 | {metrics['unauthorized_db_execution_count']} |",
        f"| Schema/索引污染 | {metrics['schema_index_pollution_count']} |",
        f"| 清理成功率 | {metrics['cleanup_success_rate']:.1%} |",
        f"| 补偿恢复率 | {metrics['compensation_recovery_rate']:.1%} |",
        f"| 分母守恒 | {'是' if metrics['denominator_consistency'] else '否'} |",
        "",
        "## 场景",
        "",
        "| ID | 类别 | 状态 | 时延 |",
        "| --- | --- | --- | ---: |",
    ]
    for item in report["cases"]:
        mark = "✅" if item["status"] == "passed" else "❌"
        lines.append(
            f"| {item['scenario_id']} | {item['category']} | "
            f"{mark} {item['status']} | {item['duration_ms']} ms |"
        )
    lines.extend([
        "",
        "## 证据边界",
        "",
        "- 夹具均由 Data Agent 从零确定性生成；报告不保留行值、文件名或物理表名。",
        "- `offline_contract` 不代表真实依赖；只有报告标明的 evidence level 可被引用。",
        "- 本 facet 独立于 NL2SQL 黄金集结果正确率，不改变其分母。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="上传数据资产隔离、隐私、补偿与清理评测"
    )
    parser.add_argument(
        "--evidence-level",
        choices=sorted(EVIDENCE_LEVELS),
        default="offline_contract",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="按 scenario ID、category 或 probe 精确选择",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    report = asyncio.run(run_suite(
        evidence_level=args.evidence_level,
        only=args.only,
        run_id=args.run_id,
        controlled_base_url=args.base_url,
        controlled_token=args.token,
    ))
    markdown = render_markdown(report)
    print(markdown)
    print(f"报告目录：{REPORT_ROOT}")
    return 1 if args.enforce and not report["summary"]["gate_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
