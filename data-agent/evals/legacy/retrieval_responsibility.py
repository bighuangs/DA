"""Offline governance regression for Qdrant-RAG versus ES-value responsibilities."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from typing import Any

from omegaconf import OmegaConf

from app.config.app_config import app_config
from app.infrastructure.search.projection import build_column_projection_records
from app.infrastructure.search.elasticsearch.value_retrieval import (
    RAG_RETRIEVAL_RESPONSIBILITY,
    VALUE_QUERY_MODES,
    VALUE_RETRIEVAL_RESPONSIBILITY,
    build_value_search_query,
    normalize_value,
    value_column_decision,
    value_document_id,
)
from app.infrastructure.search.elasticsearch.repositories.value_es_repository import ValueESRepository
from evals.paths import LEGACY_DATASETS_ROOT, REPORTS_ROOT as REPORT_ROOT


CONTRACT_PATH = LEGACY_DATASETS_ROOT / "value_retrieval_set.yaml"
REPORT_PATH = REPORT_ROOT / "value_retrieval_responsibility_latest.json"
REPORT_MD_PATH = REPORT_ROOT / "value_retrieval_responsibility_latest.md"
REQUIRED_CATEGORIES = {
    "exact_match", "prefix_match", "full_text_match", "tenant_isolation",
    "column_permission", "table_permission", "sensitive_field",
    "sensitive_value", "high_cardinality", "stable_identity",
    "responsibility_boundary", "version_isolation",
}


def load_contract() -> dict:
    return dict(OmegaConf.to_container(
        OmegaConf.load(CONTRACT_PATH),
        resolve=True,
    ) or {})


def validate_contract() -> dict:
    payload = load_contract()
    cases = list(payload.get("cases") or [])
    ids = [str(case.get("id") or "") for case in cases]
    categories = {str(case.get("category") or "") for case in cases}
    modes = {str(case.get("query_mode") or "") for case in cases}
    checks = {
        "versioned": bool(payload.get("version")),
        "reviewed": payload.get("review_status") == "reviewed",
        "unique_ids": len(ids) == len(set(ids)) and all(ids),
        "minimum_cases": len(cases) >= 12,
        "required_categories": REQUIRED_CATEGORIES <= categories,
        "all_query_modes": set(VALUE_QUERY_MODES) <= modes,
    }
    return {
        "valid": all(checks.values()),
        "version": payload.get("version"),
        "total": len(cases),
        "categories": sorted(categories),
        "query_modes": sorted(modes),
        "checks": checks,
    }


def _contains_filter(query: dict, field: str, expected: Any) -> bool:
    filters = ((query.get("bool") or {}).get("filter") or [])
    return any(
        (item.get("term") or {}).get(field) == expected
        or (item.get("terms") or {}).get(field) == expected
        for item in filters
    )


def build_report(*, write: bool = True) -> dict:
    contract = validate_contract()
    workspace_id = "workspace-a"
    version = "sha256:value-version"
    exact = build_value_search_query(
        " 华东 ",
        query_mode="exact",
        workspace_id=workspace_id,
        index_version=version,
        allowed_table_ids=["dim_region"],
        allowed_column_ids=["dim_region.region_name"],
    )
    prefix = build_value_search_query(
        "华",
        query_mode="prefix",
        workspace_id=workspace_id,
        index_version=version,
    )
    full_text = build_value_search_query(
        "华东地区",
        query_mode="full_text",
        workspace_id=workspace_id,
        index_version=version,
    )
    stable_a = value_document_id(workspace_id, "dim_region.region_name", "华东")
    stable_b = value_document_id(workspace_id, "dim_region.region_name", "华东")
    sensitive = value_column_decision({
        "id": "dim_customer.customer_name",
        "name": "customer_name",
        "table_id": "dim_customer",
        "sync": True,
    })
    sensitive_format = value_column_decision(
        {
            "id": "dim_contact.contact_value",
            "name": "contact_value",
            "table_id": "dim_contact",
            "sync": True,
        },
        distinct_count=1,
        values=["13800138000"],
    )
    high_cardinality = value_column_decision(
        {
            "id": "fact_order.external_code",
            "name": "external_code",
            "table_id": "fact_order",
            "sync": True,
        },
        distinct_count=int(app_config.es.max_distinct_values_per_column) + 1,
        values=[],
    )
    allowed = value_column_decision(
        {
            "id": "dim_region.region_name",
            "name": "region_name",
            "table_id": "dim_region",
            "sync": True,
        },
        distinct_count=7,
        values=["华东", "华南"],
    )
    projection = build_column_projection_records(
        [(
            {"id": "dim_customer", "name": "dim_customer"},
            [{
                "id": "dim_customer.customer_name",
                "name": "customer_name",
                "table_id": "dim_customer",
                "type": "varchar",
                "role": "dimension",
                "description": "客户名称",
                "alias": ["客户"],
                "examples": ["张三"],
            }],
        )],
        index_version="sha256:rag-version",
        strategy="single_canonical",
        alias_manifest={"version": "test", "approved_aliases": []},
    )
    mapping = ValueESRepository._mappings()
    properties = mapping.get("properties") or {}
    checks = {
        "contract_valid": contract["valid"],
        "responsibilities_separate": (
            RAG_RETRIEVAL_RESPONSIBILITY != VALUE_RETRIEVAL_RESPONSIBILITY
        ),
        "mapping_filters_complete": all(
            name in properties for name in (
                "workspace_id", "normalized_value", "table_id", "column_id",
                "index_version", "is_active",
            )
        ),
        "exact_normalized": (
            ((exact["bool"]["must"][0].get("term") or {}).get("normalized_value"))
            == normalize_value("华东")
        ),
        "prefix_compiled": "prefix" in prefix["bool"]["must"][0],
        "full_text_compiled": "match" in full_text["bool"]["must"][0],
        "workspace_filter_in_es": _contains_filter(exact, "workspace_id", workspace_id),
        "version_filter_in_es": _contains_filter(exact, "index_version", version),
        "table_scope_in_es": _contains_filter(exact, "table_id", ["dim_region"]),
        "column_scope_in_es": _contains_filter(
            exact, "column_id", ["dim_region.region_name"]
        ),
        "stable_hashed_document_id": (
            stable_a == stable_b
            and stable_a.startswith("value:")
            and "华东" not in stable_a
        ),
        "sensitive_column_blocked": (
            not sensitive.allowed and sensitive.reason == "sensitive_column"
        ),
        "sensitive_value_blocked": (
            not sensitive_format.allowed
            and sensitive_format.reason == "sensitive_value_format"
        ),
        "high_cardinality_blocked": (
            not high_cardinality.allowed
            and high_cardinality.reason == "high_cardinality"
        ),
        "explicit_sync_low_cardinality_allowed": allowed.allowed,
        "qdrant_payload_has_no_values": all(
            "examples" not in record.payload
            and "张三" not in json.dumps(record.payload, ensure_ascii=False)
            for record in projection
        ),
    }
    report = {
        "available": True,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "contract": contract,
        "responsibilities": {
            "qdrant": RAG_RETRIEVAL_RESPONSIBILITY,
            "elasticsearch": VALUE_RETRIEVAL_RESPONSIBILITY,
        },
        "checks": checks,
        "passed_checks": sum(bool(value) for value in checks.values()),
        "total_checks": len(checks),
        "gate_passed": all(checks.values()),
    }
    if write:
        REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        rows = "\n".join(
            f"| {name} | {'✅' if passed else '❌'} |"
            for name, passed in checks.items()
        )
        REPORT_MD_PATH.write_text(
            "# Sparse RAG 与 ES 字段值检索职责治理\n\n"
            f"> 状态：{'通过' if report['gate_passed'] else '未通过'} ｜ "
            f"{report['passed_checks']}/{report['total_checks']} 检查通过\n\n"
            "| 检查 | 结果 |\n| --- | --- |\n"
            f"{rows}\n",
            encoding="utf-8",
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    report = build_report(write=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n报告：{REPORT_MD_PATH}")
    return 2 if args.enforce and not report["gate_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
