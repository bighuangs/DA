"""Canonical NL2SQL dataset, scalable DW fixture and governed Meta baseline."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import sqlglot
from omegaconf import OmegaConf
from sqlalchemy import delete, select, text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlglot import exp

from app.config.app_config import app_config
from app.config.meta_config import meta_config
from app.domain.query.safety.query_guard import enforce_explain_cost
from app.domain.query.safety.sql_security import check_join_relationships, check_sql_read
from evals.legacy.governance import sample_contract_report
from app.infrastructure.persistence.mysql.records.column_metric_mysql import ColumnMetricMySQL
from app.infrastructure.persistence.mysql.records.metric_info_mysql import MetricInfoMySQL
from app.infrastructure.persistence.mysql.records.training_info_mysql import TrainingInfoMySQL
from app.infrastructure.persistence.mysql.records.workspace_mysql import WorkspaceMySQL
from app.infrastructure.persistence.mysql.repositories.meta_mysql_repository import MetaMysqlRepository
from evals.paths import BASELINES_ROOT, LEGACY_DATASETS_ROOT, PROJECT_ROOT

def _read_manifest_mapping(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name} 顶层必须是对象")
    return dict(payload)

EASY_SQL_MANIFEST_PATH = LEGACY_DATASETS_ROOT / "easy_sql_v1_manifest.yaml"
MEDIUM_SQL_MANIFEST_PATH = LEGACY_DATASETS_ROOT / "medium_sql_v1_manifest.yaml"
HARD_SQL_MANIFEST_PATH = LEGACY_DATASETS_ROOT / "hard_sql_v1_manifest.yaml"
_EASY_SQL_PATHS = (LEGACY_DATASETS_ROOT / "easy_sql_v1.yaml",)
_MEDIUM_SQL_PATHS = (LEGACY_DATASETS_ROOT / "medium_sql_v1.yaml",)
_HARD_SQL_PATHS = (LEGACY_DATASETS_ROOT / "hard_sql_v1.yaml",)
SUITE_SPECS = {
    "easy_sql_v1": {
        "paths": _EASY_SQL_PATHS,
        "case_manifest": EASY_SQL_MANIFEST_PATH,
        "expected_cases": 100,
        "promotion_status": "approved",
        "manifest": EASY_SQL_MANIFEST_PATH,
        "reference_suites": (),
        "required_tags": {"aggregation", "join", "grouping"},
        "capability_scope": "difficulty_baseline_easy",
        "llm_invocation_contract": "generate_sql_physical_attempt_per_case",
        "reference_result_contract": "non_empty_reference_with_projection",
        "baseline_eligible": True,
        "baseline_policy": "observational_three_repeat",
        "display_name": "简单 SQL 基线集",
        "difficulty": "easy",
    },
    "medium_sql_v1": {
        "paths": _MEDIUM_SQL_PATHS,
        "case_manifest": MEDIUM_SQL_MANIFEST_PATH,
        "expected_cases": 100,
        "promotion_status": "approved",
        "manifest": MEDIUM_SQL_MANIFEST_PATH,
        "reference_suites": ("easy_sql_v1",),
        "required_tags": {"aggregation", "join", "having", "conditional_aggregation"},
        "capability_scope": "difficulty_baseline_medium",
        "llm_invocation_contract": "generate_sql_physical_attempt_per_case",
        "reference_result_contract": "non_empty_reference_with_projection",
        "baseline_eligible": True,
        "baseline_policy": "observational_three_repeat",
        "display_name": "中等 SQL 基线集",
        "difficulty": "medium",
    },
    "hard_sql_v1": {
        "paths": _HARD_SQL_PATHS,
        "case_manifest": HARD_SQL_MANIFEST_PATH,
        "expected_cases": 50,
        "promotion_status": "approved",
        "manifest": HARD_SQL_MANIFEST_PATH,
        "reference_suites": ("easy_sql_v1", "medium_sql_v1"),
        "required_tags": {"cte", "window", "model_route", "multi_stage"},
        "capability_scope": "difficulty_baseline_hard",
        "llm_invocation_contract": "generate_sql_physical_attempt_per_case",
        "reference_result_contract": "non_empty_reference_with_projection",
        "baseline_eligible": True,
        "baseline_policy": "observational_three_repeat",
        "display_name": "困难 SQL 基线集",
        "difficulty": "hard",
    },
}
SUITE_PATHS = {
    name: tuple(spec["paths"])
    for name, spec in SUITE_SPECS.items()
}
BASELINE_MANIFEST_PATH = BASELINES_ROOT / "baseline_manifest.yaml"


def _case_manifest(suite: str) -> dict[str, Any]:
    if suite not in SUITE_SPECS:
        raise ValueError(f"未知数据集套件：{suite}")
    return _read_manifest_mapping(Path(SUITE_SPECS[suite]["case_manifest"]))


def _read_cases(path: Path) -> list[dict[str, Any]]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True) or []
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        defaults = dict(payload.get("defaults") or {})
        return [
            {**defaults, **dict(item)}
            for item in payload["cases"]
        ]
    raise ValueError(f"{path.name} 顶层必须是用例列表或 defaults/cases 对象")


def load_suite_cases(
    suite: str = "easy_sql_v1",
    *,
    only: str | None = None,
    tier: str | None = None,
) -> list[dict[str, Any]]:
    if suite not in SUITE_PATHS:
        raise ValueError(f"未知数据集套件：{suite}")
    cases: list[dict[str, Any]] = []
    for path in SUITE_PATHS[suite]:
        cases.extend(_read_cases(path))
    ids = [str(item.get("id") or "") for item in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"评测套件 {suite} 存在重复用例 ID")
    case_manifest = _case_manifest(suite)
    governed_ids = tuple(
        str(value) for value in case_manifest.get("case_ids") or ()
    )
    expected_cases = int(case_manifest.get("expected_cases") or 0)
    if len(governed_ids) != len(set(governed_ids)):
        raise ValueError(f"评测套件 {suite} 的能力清单存在重复用例 ID")
    if len(governed_ids) != expected_cases:
        raise ValueError(
            f"评测套件 {suite} 的能力清单用例数 {len(governed_ids)} != "
            f"expected_cases {expected_cases}"
        )
    if governed_ids:
        by_id = {str(item.get("id") or ""): item for item in cases}
        missing = [case_id for case_id in governed_ids if case_id not in by_id]
        if missing:
            raise ValueError(
                f"评测套件 {suite} 的清单引用不存在用例：" + "、".join(missing)
            )
        extras = [case_id for case_id in ids if case_id not in set(governed_ids)]
        if extras:
            raise ValueError(
                f"评测套件 {suite} 存在能力清单外用例：" + "、".join(extras)
            )
        cases = [by_id[case_id] for case_id in governed_ids]
        ids = list(governed_ids)
    if only:
        selected = [item.strip() for item in only.split(",") if item.strip()]
        unknown = sorted(set(selected) - set(ids))
        if unknown:
            raise ValueError("未知用例 ID：" + "、".join(unknown))
        selected_set = set(selected)
        cases = [item for item in cases if str(item.get("id")) in selected_set]
    if tier:
        cases = [item for item in cases if str(item.get("tier")) == tier]
    return cases


def _normalized_question(value: str) -> str:
    return "".join(str(value or "").lower().split())


def _normalized_sql(value: str) -> str:
    sql = str(value or "").strip()
    try:
        return sqlglot.parse_one(sql, read="mysql").sql(
            dialect="mysql",
            comments=False,
            pretty=False,
        ).lower()
    except Exception:  # noqa: BLE001 - normalization has a plain-text fallback
        return " ".join(sql.lower().split())


def _char_bigrams(value: str) -> set[str]:
    normalized = _normalized_question(value)
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {
        normalized[index:index + 2]
        for index in range(len(normalized) - 1)
    }


def _question_similarity(left: str, right: str) -> float:
    left_tokens, right_tokens = _char_bigrams(left), _char_bigrams(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def suite_manifest(suite: str) -> dict[str, Any]:
    if suite not in SUITE_SPECS:
        raise ValueError(f"未知数据集套件：{suite}")
    path = Path(SUITE_SPECS[suite]["manifest"])
    return _read_manifest_mapping(path)


def sql_ast_policy_report(
    cases: list[dict[str, Any]],
    suite: str = "easy_sql_v1",
) -> dict[str, Any]:
    """Validate common-SQL scope from parsed structure, independent of tags."""
    policy = dict(_case_manifest(suite).get("structural_policy") or {})
    node_types: dict[str, type[exp.Expression]] = {
        "cte": exp.CTE,
        "window": exp.Window,
        "subquery": exp.Subquery,
        "set_operation": exp.SetOperation,
    }
    forbidden_names = [
        str(value) for value in policy.get("forbidden_ast_nodes") or []
    ]
    unknown_nodes = sorted(set(forbidden_names) - set(node_types))
    if unknown_nodes:
        raise ValueError(
            "通用 SQL structural_policy 含未知 AST 节点："
            + "、".join(unknown_nodes)
        )

    violations: list[dict[str, Any]] = []
    max_joins = int(policy.get("max_joins") or 0)
    for case in cases:
        case_id = str(case.get("id") or "<unknown>")
        if policy.get("forbid_history") and case.get("history"):
            violations.append({"case_id": case_id, "reason": "history_forbidden"})
        try:
            statements = sqlglot.parse(
                str(case.get("expected_sql") or ""),
                read="mysql",
            )
        except Exception as exc:  # noqa: BLE001 - report every parser failure
            violations.append({
                "case_id": case_id,
                "reason": "sql_parse_error",
                "detail": f"{type(exc).__name__}: {exc}",
            })
            continue
        if not statements:
            violations.append({"case_id": case_id, "reason": "empty_sql"})
            continue
        if policy.get("single_statement_only") and len(statements) != 1:
            violations.append({
                "case_id": case_id,
                "reason": "multiple_statements",
                "observed": len(statements),
            })
        tree = statements[0]
        select_count = sum(1 for _ in tree.find_all(exp.Select))
        if policy.get("single_select_only") and select_count != 1:
            violations.append({
                "case_id": case_id,
                "reason": "select_count_exceeded",
                "observed": select_count,
                "maximum": 1,
            })
        join_count = sum(1 for _ in tree.find_all(exp.Join))
        if max_joins >= 0 and join_count > max_joins:
            violations.append({
                "case_id": case_id,
                "reason": "join_count_exceeded",
                "observed": join_count,
                "maximum": max_joins,
            })
        for name in forbidden_names:
            if next(tree.find_all(node_types[name]), None) is not None:
                violations.append({
                    "case_id": case_id,
                    "reason": f"forbidden_ast:{name}",
                })
    return {
        "policy": policy,
        "case_count": len(cases),
        "violations": violations,
        "passed": not violations,
    }


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def suite_freeze_report(suite: str) -> dict[str, Any]:
    manifest = suite_manifest(suite)
    entries = list((manifest.get("freeze") or {}).get("artifacts") or [])
    required = str(SUITE_SPECS[suite]["promotion_status"]) == "sealed_holdout"
    artifacts: list[dict[str, Any]] = []
    for entry in entries:
        relative = str((entry or {}).get("path") or "")
        expected = str((entry or {}).get("sha256") or "")
        path = PROJECT_ROOT / relative
        actual = _file_sha256(path) if path.is_file() else ""
        artifacts.append({
            "path": relative,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matched": bool(expected and actual and expected == actual),
        })
    return {
        "required": required,
        "frozen_at": str((manifest.get("freeze") or {}).get("frozen_at") or ""),
        "artifacts": artifacts,
        "passed": bool(
            (not required and not entries)
            or (entries and all(item["matched"] for item in artifacts))
        ),
    }


def suite_independence_report(suite: str) -> dict[str, Any]:
    """Prove a candidate/holdout suite is independent from governed references."""
    cases = load_suite_cases(suite)
    spec = SUITE_SPECS[suite]
    questions = {
        _normalized_question(str(item.get("question") or "")): str(item.get("id"))
        for item in cases
    }
    sqls = {
        _normalized_sql(str(item.get("expected_sql") or "")): str(item.get("id"))
        for item in cases
    }
    reference_cases: list[dict[str, Any]] = []
    reference_suite_by_id: dict[str, str] = {}
    for reference_suite in spec.get("reference_suites") or ():
        loaded = load_suite_cases(str(reference_suite))
        reference_cases.extend(loaded)
        reference_suite_by_id.update({
            str(item.get("id") or ""): str(reference_suite)
            for item in loaded
        })
    for path, reference_label in spec.get("reference_paths") or ():
        loaded = _read_cases(Path(path))
        reference_cases.extend(loaded)
        reference_suite_by_id.update({
            str(item.get("id") or ""): str(reference_label)
            for item in loaded
        })
    reference_questions = {
        _normalized_question(str(item.get("question") or "")): str(item.get("id"))
        for item in reference_cases
    }
    reference_sqls = {
        _normalized_sql(str(item.get("expected_sql") or "")): str(item.get("id"))
        for item in reference_cases
    }
    training_examples = baseline_training_examples()
    training_questions = {
        _normalized_question(str(item.get("question") or ""))
        for item in training_examples
    }
    training_sqls = {
        _normalized_sql(str(item.get("sql") or "")): str(item.get("question") or "")
        for item in training_examples
    }
    near_pairs: list[dict[str, Any]] = []
    ordered = list(cases)
    for index, left in enumerate(ordered):
        for right in ordered[index + 1:]:
            similarity = _question_similarity(
                str(left.get("question") or ""),
                str(right.get("question") or ""),
            )
            if similarity >= 0.92:
                near_pairs.append({
                    "left": str(left.get("id")),
                    "right": str(right.get("id")),
                    "similarity": round(similarity, 4),
                })
    cross_suite_near_pairs: list[dict[str, Any]] = []
    cross_suite_threshold = spec.get("cross_suite_similarity_max")
    if cross_suite_threshold is not None:
        for current in cases:
            for reference in reference_cases:
                similarity = _question_similarity(
                    str(current.get("question") or ""),
                    str(reference.get("question") or ""),
                )
                if similarity >= float(cross_suite_threshold):
                    cross_suite_near_pairs.append({
                        "case_id": str(current.get("id") or ""),
                        "reference_suite": reference_suite_by_id.get(
                            str(reference.get("id") or ""), ""
                        ),
                        "reference_case_id": str(reference.get("id") or ""),
                        "similarity": round(similarity, 4),
                    })
    exact_question_overlap = sorted(set(questions) & set(reference_questions))
    exact_sql_overlap = sorted(set(sqls) & set(reference_sqls))
    training_overlap = sorted(set(questions) & training_questions)
    training_sql_overlap = sorted(set(sqls) & set(training_sqls))
    return {
        "suite": suite,
        "reference_suites": [
            *list(spec.get("reference_suites") or ()),
            *[
                str(reference_label)
                for _, reference_label in spec.get("reference_paths") or ()
            ],
        ],
        "exact_question_overlap": [
            {
                "question": question,
                "case_id": questions[question],
                "reference_suite": reference_suite_by_id.get(
                    reference_questions[question], ""
                ),
                "reference_case_id": reference_questions[question],
            }
            for question in exact_question_overlap
        ],
        "exact_sql_overlap": [
            {
                "case_id": sqls[sql],
                "reference_suite": reference_suite_by_id.get(
                    reference_sqls[sql], ""
                ),
                "reference_case_id": reference_sqls[sql],
            }
            for sql in exact_sql_overlap
        ],
        "training_question_overlap": training_overlap,
        "training_sql_overlap": [
            {
                "case_id": sqls[sql],
                "training_question": training_sqls[sql],
            }
            for sql in training_sql_overlap
        ],
        "within_suite_near_duplicate_pairs": near_pairs,
        "cross_suite_near_duplicate_pairs": cross_suite_near_pairs,
        "cross_suite_similarity_max": cross_suite_threshold,
        "passed": not (
            exact_question_overlap
            or exact_sql_overlap
            or training_overlap
            or training_sql_overlap
            or near_pairs
            or cross_suite_near_pairs
        ),
    }


def baseline_training_probe_examples() -> list[dict[str, str]]:
    """返回与黄金题隔离、供确定性召回回归使用的 canonical probes。"""

    return [
        {
            "question": "2024年每月成交金额趋势",
            "sql": (
                "SELECT d.month, SUM(f.order_amount) AS total_amount "
                "FROM fact_order f JOIN dim_date d ON f.date_id = d.date_id "
                "WHERE d.year = 2024 GROUP BY d.month ORDER BY d.month"
            ),
        },
        {
            "question": "按品类统计售出件数结构",
            "sql": (
                "SELECT p.category, SUM(f.order_quantity) AS total_quantity "
                "FROM fact_order f JOIN dim_product p ON f.product_id = p.product_id "
                "GROUP BY p.category ORDER BY total_quantity DESC"
            ),
        },
        {
            "question": "2024年按地区汇总成交金额",
            "sql": (
                "SELECT r.region_name, SUM(f.order_amount) AS total_amount "
                "FROM fact_order f JOIN dim_region r ON f.region_id = r.region_id "
                "JOIN dim_date d ON f.date_id = d.date_id "
                "WHERE d.year = 2024 GROUP BY r.region_name "
                "ORDER BY total_amount DESC"
            ),
        },
        {
            "question": "2024年按大区统计订单笔数排名",
            "sql": (
                "SELECT r.region_name, COUNT(f.order_id) AS order_count "
                "FROM fact_order f JOIN dim_region r ON f.region_id = r.region_id "
                "JOIN dim_date d ON f.date_id = d.date_id WHERE d.year = 2024 "
                "GROUP BY r.region_name ORDER BY order_count DESC"
            ),
        },
    ]


def baseline_training_examples() -> list[dict[str, str]]:
    """Build a 60-example Meta corpus with zero exact golden-question overlap."""
    # 这四条是 independent retrieval probes，不是黄金题副本；完整 live corpus
    # 在其后扩展长尾实体切片，offline probe 仍保持原有的小语料职责。
    examples = baseline_training_probe_examples()
    for year in (2025, 2026):
        examples.extend([
            {
                "question": f"{year}年按省份汇总成交金额",
                "sql": (
                    "SELECT r.province, SUM(f.order_amount) AS total_amount "
                    "FROM fact_order f JOIN dim_region r ON f.region_id = r.region_id "
                    "JOIN dim_date d ON f.date_id = d.date_id "
                    f"WHERE d.year = {year} GROUP BY r.province "
                    "ORDER BY total_amount DESC"
                ),
            },
            {
                "question": f"{year}年按品牌统计售出件数",
                "sql": (
                    "SELECT p.brand, SUM(f.order_quantity) AS total_quantity "
                    "FROM fact_order f JOIN dim_product p ON f.product_id = p.product_id "
                    "JOIN dim_date d ON f.date_id = d.date_id "
                    f"WHERE d.year = {year} GROUP BY p.brand "
                    "ORDER BY total_quantity DESC"
                ),
            },
            {
                "question": f"{year}年每季度订单笔数",
                "sql": (
                    "SELECT d.quarter, COUNT(f.order_id) AS order_count "
                    "FROM fact_order f JOIN dim_date d ON f.date_id = d.date_id "
                    f"WHERE d.year = {year} GROUP BY d.quarter ORDER BY d.quarter"
                ),
            },
            {
                "question": f"{year}年不同会员等级的购买客户数",
                "sql": (
                    "SELECT c.member_level, COUNT(DISTINCT f.customer_id) AS customer_count "
                    "FROM fact_order f JOIN dim_customer c ON f.customer_id = c.customer_id "
                    "JOIN dim_date d ON f.date_id = d.date_id "
                    f"WHERE d.year = {year} GROUP BY c.member_level "
                    "ORDER BY c.member_level"
                ),
            },
        ])

    categories = (
        "手机数码", "家用电器", "鞋靴", "服饰", "食品饮料", "休闲零食",
        "运动户外", "家居日用", "美妆个护", "母婴用品", "图书文娱", "办公用品",
    )
    for category in categories:
        examples.extend([
            {
                "question": f"{category}品类在各地区的成交金额",
                "sql": (
                    "SELECT r.region_name, SUM(f.order_amount) AS total_amount "
                    "FROM fact_order f JOIN dim_product p ON f.product_id = p.product_id "
                    "JOIN dim_region r ON f.region_id = r.region_id "
                    f"WHERE p.category = '{category}' GROUP BY r.region_name "
                    "ORDER BY total_amount DESC"
                ),
            },
            {
                "question": f"{category}品类各品牌的售出件数",
                "sql": (
                    "SELECT p.brand, SUM(f.order_quantity) AS total_quantity "
                    "FROM fact_order f JOIN dim_product p ON f.product_id = p.product_id "
                    f"WHERE p.category = '{category}' GROUP BY p.brand "
                    "ORDER BY total_quantity DESC"
                ),
            },
        ])

    regions = ("华北", "东北", "华东", "华中", "华南", "西南", "西北", "港澳台")
    for region in regions:
        examples.extend([
            {
                "question": f"{region}大区各品类的订单笔数",
                "sql": (
                    "SELECT p.category, COUNT(f.order_id) AS order_count "
                    "FROM fact_order f JOIN dim_product p ON f.product_id = p.product_id "
                    "JOIN dim_region r ON f.region_id = r.region_id "
                    f"WHERE r.region_name = '{region}' GROUP BY p.category "
                    "ORDER BY order_count DESC"
                ),
            },
            {
                "question": f"{region}大区每年的成交金额趋势",
                "sql": (
                    "SELECT d.year, SUM(f.order_amount) AS total_amount "
                    "FROM fact_order f JOIN dim_region r ON f.region_id = r.region_id "
                    "JOIN dim_date d ON f.date_id = d.date_id "
                    f"WHERE r.region_name = '{region}' GROUP BY d.year ORDER BY d.year"
                ),
            },
        ])

    for level in ("青铜", "白银", "黄金", "铂金"):
        examples.extend([
            {
                "question": f"{level}会员每年的平均订单金额",
                "sql": (
                    "SELECT d.year, AVG(f.order_amount) AS avg_order_amount "
                    "FROM fact_order f JOIN dim_customer c ON f.customer_id = c.customer_id "
                    "JOIN dim_date d ON f.date_id = d.date_id "
                    f"WHERE c.member_level = '{level}' GROUP BY d.year ORDER BY d.year"
                ),
            },
            {
                "question": f"{level}会员购买最多的五个商品",
                "sql": (
                    "SELECT p.product_name, SUM(f.order_quantity) AS total_quantity "
                    "FROM fact_order f JOIN dim_customer c ON f.customer_id = c.customer_id "
                    "JOIN dim_product p ON f.product_id = p.product_id "
                    f"WHERE c.member_level = '{level}' GROUP BY p.product_name "
                    "ORDER BY total_quantity DESC LIMIT 5"
                ),
            },
        ])

    if len(examples) != 60:
        raise RuntimeError(f"Meta 训练样例生成数量异常：{len(examples)}")
    questions = [_normalized_question(item["question"]) for item in examples]
    if len(questions) != len(set(questions)):
        raise RuntimeError("Meta 训练样例存在重复问题")
    governed_cases = [
        item
        for suite in SUITE_SPECS
        for item in load_suite_cases(suite)
    ]
    golden = {
        _normalized_question(str(item["question"]))
        for item in governed_cases
    }
    overlap = sorted(set(questions) & golden)
    if overlap:
        raise RuntimeError(f"Meta 训练样例与黄金题精确重叠：{overlap}")
    golden_sqls = {
        _normalized_sql(str(item["expected_sql"]))
        for item in governed_cases
    }
    training_sqls = {_normalized_sql(str(item["sql"])) for item in examples}
    sql_overlap = sorted(training_sqls & golden_sqls)
    if sql_overlap:
        raise RuntimeError(
            f"Meta 训练样例与黄金题 SQL 规范化重叠：{len(sql_overlap)} 条"
        )
    for item in examples:
        check_sql_read(item["sql"])
        check_join_relationships(item["sql"])
    return examples


def dataset_suite_report(suite: str = "easy_sql_v1") -> dict[str, Any]:
    cases = load_suite_cases(suite)
    case_manifest = _case_manifest(suite)
    contract = sample_contract_report(cases)
    categories = sorted({str(item.get("category") or "") for item in cases})
    tags = sorted({
        str(tag)
        for item in cases
        for tag in (item.get("tags") or [])
    })
    tiers = {
        name: sum(str(item.get("tier")) == name for item in cases)
        for name in ("smoke", "regression")
    }
    difficulties = {
        name: sum(str(item.get("difficulty")) == name for item in cases)
        for name in ("easy", "medium", "hard")
    }
    spec = SUITE_SPECS[suite]
    required_categories = set(spec.get("required_categories") or ())
    required_tags = set(spec.get("required_tags") or ())
    missing_categories = sorted(required_categories - set(categories))
    missing_tags = sorted(required_tags - set(tags))
    independence = suite_independence_report(suite)
    freeze = suite_freeze_report(suite)
    ast_policy = sql_ast_policy_report(cases, suite)
    expected_cases = int(case_manifest["expected_cases"])
    structural_ready = bool(
        contract["complete"]
        and contract["all_reviewed"]
        and len(cases) == expected_cases
        and tiers["smoke"] > 0
        and tiers["regression"] > 0
        and not missing_categories
        and not missing_tags
        and independence["passed"]
        and freeze["passed"]
        and ast_policy["passed"]
    )
    review = dict(case_manifest.get("review") or {})
    runtime_validation = {
        "reference_sql_required": (
            str(review.get("reference_sql_validation") or "") == "required"
        ),
        "reference_sql_observed": False,
        "result_equivalence_required": (
            str(review.get("result_equivalence") or "") == "required"
        ),
        "result_equivalence_observed": False,
        "live_llm_run_policy": str(review.get("live_llm_run") or ""),
    }
    runtime_validation["ready"] = bool(
        (
            not runtime_validation["reference_sql_required"]
            or runtime_validation["reference_sql_observed"]
        )
        and (
            not runtime_validation["result_equivalence_required"]
            or runtime_validation["result_equivalence_observed"]
        )
    )
    structural_promotion_ready = bool(
        structural_ready and spec["promotion_status"] == "approved"
    )
    return {
        "suite": suite,
        "paths": [
            str(path.relative_to(PROJECT_ROOT)) for path in SUITE_PATHS[suite]
        ],
        "case_count": len(cases),
        "expected_case_count": expected_cases,
        "categories": categories,
        "tags": tags,
        "tiers": tiers,
        "difficulties": difficulties,
        "contract": contract,
        "independence": independence,
        "freeze": freeze,
        "ast_policy": ast_policy,
        "structural_ready": structural_ready,
        "promotion_status": str(spec["promotion_status"]),
        "llm_invocation_contract": spec.get("llm_invocation_contract"),
        "structural_promotion_ready": structural_promotion_ready,
        "runtime_validation": runtime_validation,
        "promotion_ready_scope": "structural_and_runtime_evidence",
        "promotion_ready": bool(
            structural_promotion_ready and runtime_validation["ready"]
        ),
        "missing_required_categories": missing_categories,
        "missing_required_tags": missing_tags,
    }


def baseline_manifest() -> dict[str, Any]:
    return _read_manifest_mapping(BASELINE_MANIFEST_PATH)


def resolve_order_count(
    *,
    profile: str = "medium",
    order_count: int | None = None,
) -> int:
    scale = baseline_manifest()["scale"]
    profiles = dict(scale["profiles"])
    if profile not in profiles:
        raise ValueError("profile 只允许 medium 或 large")
    selected = int(order_count if order_count is not None else profiles[profile])
    minimum = int(scale["minimum_orders"])
    maximum = int(scale["maximum_orders"])
    if not minimum <= selected <= maximum:
        raise ValueError(f"order-count 必须在 {minimum}～{maximum} 之间")
    return selected


def install_baseline_fixture(
    *,
    confirm_reset: bool,
    profile: str = "medium",
    order_count: int | None = None,
) -> int:
    """Reset the canonical ``dw2`` fixture through the project MySQL container."""
    if not confirm_reset:
        raise ValueError("安装会删除并重建正式库 dw2；请显式增加 --confirm-reset")
    selected = resolve_order_count(profile=profile, order_count=order_count)
    manifest = baseline_manifest()
    sql_path = PROJECT_ROOT / str(manifest["fixture_sql"])
    default_orders = int(manifest["scale"]["default_orders"])
    sql_body = sql_path.read_text(encoding="utf-8").replace(
        f"SET @baseline_order_count = {default_orders};",
        f"SET @baseline_order_count = {selected};",
        1,
    )
    command = [
        "docker", "exec", "-i", "mysql", "sh", "-lc",
        'exec mysql -uroot -p"$MYSQL_ROOT_PASSWORD"',
    ]
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        input=sql_body,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "Docker MySQL 执行失败").strip()
        raise RuntimeError(detail)
    return selected


def _database_url(config: Any, database: str) -> URL:
    return URL.create(
        "mysql+asyncmy",
        username=config.user,
        password=config.password,
        host=config.host,
        port=config.port,
        database=database,
        query={"charset": "utf8mb4"},
    )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(float(ordered[rank]), 3)


async def check_baseline_dw(
    *,
    benchmark_repeats: int = 2,
    suite: str = "easy_sql_v1",
) -> dict[str, Any]:
    """Validate the canonical DW and execute one suite's reference SQL."""
    if suite not in SUITE_SPECS:
        raise ValueError(f"未知数据集套件：{suite}")
    manifest = baseline_manifest()
    database = str(manifest["database"])
    expected = dict(manifest["expected"])
    scale = dict(manifest["scale"])
    engine = create_async_engine(
        _database_url(app_config.db_dw, database),
        pool_pre_ping=True,
    )
    errors: list[str] = []
    warnings: list[str] = []
    observed: dict[str, Any] = {
        "database": database,
        "row_counts": {},
        "distribution": {},
        "reference_sql": {},
    }
    try:
        async with engine.connect() as connection:
            exact_rows = dict(expected["exact_rows"])
            for table in [*exact_rows, "fact_order"]:
                count = int((await connection.execute(
                    text(f"SELECT COUNT(*) FROM `{table}`"),
                )).scalar_one())
                observed["row_counts"][table] = count
                if table in exact_rows and count != int(exact_rows[table]):
                    errors.append(f"{table} 行数 {count} != {exact_rows[table]}")
            fact_count = int(observed["row_counts"]["fact_order"])
            if not int(scale["minimum_orders"]) <= fact_count <= int(scale["maximum_orders"]):
                errors.append(
                    f"fact_order 行数 {fact_count} 不在 "
                    f"{scale['minimum_orders']}～{scale['maximum_orders']}"
                )

            date_row = (await connection.execute(text(
                "SELECT MIN(date_id) min_date, MAX(date_id) max_date FROM dim_date"
            ))).mappings().one()
            observed["date_range"] = {
                "min": int(date_row["min_date"]),
                "max": int(date_row["max_date"]),
            }
            expected_range = {
                "min": int(expected["date_range"]["min"]),
                "max": int(expected["date_range"]["max"]),
            }
            if observed["date_range"] != expected_range:
                errors.append(f"日期范围不匹配：{observed['date_range']}")

            edge = expected["required_edge_rows"]
            customer_entities = int((await connection.execute(text(
                "SELECT COUNT(*) FROM dim_customer WHERE customer_id = :value"
            ), {"value": str(edge["customer_without_orders"])})).scalar_one())
            product_entities = int((await connection.execute(text(
                "SELECT COUNT(*) FROM dim_product WHERE product_id = :value"
            ), {"value": str(edge["product_without_orders"])})).scalar_one())
            customer_orders = int((await connection.execute(text(
                "SELECT COUNT(*) FROM fact_order WHERE customer_id = :value"
            ), {"value": str(edge["customer_without_orders"])})).scalar_one())
            product_orders = int((await connection.execute(text(
                "SELECT COUNT(*) FROM fact_order WHERE product_id = :value"
            ), {"value": str(edge["product_without_orders"])})).scalar_one())
            observed["edge_rows"] = {
                "customer_without_orders": customer_orders,
                "product_without_orders": product_orders,
                "customer_dimension_rows": customer_entities,
                "product_dimension_rows": product_entities,
            }
            if customer_entities != 1 or product_entities != 1:
                errors.append(
                    "anti-join 边界实体缺失或不唯一："
                    f"customer={customer_entities}, product={product_entities}"
                )
            if customer_orders or product_orders:
                errors.append("anti-join 边界实体被事实表引用")

            orphan_count = int((await connection.execute(text(
                "SELECT COUNT(*) FROM fact_order f "
                "LEFT JOIN dim_customer c ON f.customer_id = c.customer_id "
                "LEFT JOIN dim_product p ON f.product_id = p.product_id "
                "LEFT JOIN dim_date d ON f.date_id = d.date_id "
                "LEFT JOIN dim_region r ON f.region_id = r.region_id "
                "WHERE c.customer_id IS NULL OR p.product_id IS NULL "
                "OR d.date_id IS NULL OR r.region_id IS NULL"
            ))).scalar_one())
            observed["orphan_count"] = orphan_count
            if expected.get("require_zero_orphans") and orphan_count:
                errors.append(f"事实表存在 {orphan_count} 条维度孤儿")

            null_count = int((await connection.execute(text(
                "SELECT COUNT(*) FROM fact_order WHERE "
                "order_id IS NULL OR customer_id IS NULL OR product_id IS NULL "
                "OR date_id IS NULL OR region_id IS NULL "
                "OR order_quantity IS NULL OR order_amount IS NULL"
            ))).scalar_one())
            observed["fact_null_count"] = null_count
            if expected.get("require_zero_fact_nulls") and null_count:
                errors.append(f"事实表存在 {null_count} 条关键字段空值")

            active_row = (await connection.execute(text(
                "SELECT "
                "(SELECT COUNT(DISTINCT customer_id) FROM fact_order) active_customers, "
                "(SELECT COUNT(*) FROM dim_customer) total_customers, "
                "(SELECT COUNT(DISTINCT product_id) FROM fact_order) active_products, "
                "(SELECT COUNT(*) FROM dim_product) total_products"
            ))).mappings().one()
            customer_ratio = int(active_row["active_customers"]) / max(
                1, int(active_row["total_customers"]) - 1,
            )
            product_ratio = int(active_row["active_products"]) / max(
                1, int(active_row["total_products"]) - 1,
            )
            observed["distribution"]["active_customer_ratio"] = round(customer_ratio, 6)
            observed["distribution"]["active_product_ratio"] = round(product_ratio, 6)
            if customer_ratio < float(expected["minimum_active_customer_ratio"]):
                errors.append(f"活跃客户覆盖率过低：{customer_ratio:.2%}")
            if product_ratio < float(expected["minimum_active_product_ratio"]):
                errors.append(f"活跃商品覆盖率过低：{product_ratio:.2%}")

            year_rows = (await connection.execute(text(
                "SELECT d.year, COUNT(*) order_count "
                "FROM fact_order f JOIN dim_date d ON f.date_id = d.date_id "
                "GROUP BY d.year ORDER BY d.year"
            ))).mappings().all()
            year_shares = {
                str(row["year"]): round(int(row["order_count"]) / max(1, fact_count), 6)
                for row in year_rows
            }
            observed["distribution"]["year_shares"] = year_shares
            if len(year_shares) != 3 or any(
                value < float(expected["minimum_year_share"])
                for value in year_shares.values()
            ):
                errors.append(f"年份分布覆盖不足：{year_shares}")

            top_share = float((await connection.execute(text(
                "SELECT MAX(total_amount) / NULLIF(SUM(total_amount), 0) "
                "FROM (SELECT customer_id, SUM(order_amount) total_amount "
                "FROM fact_order GROUP BY customer_id) customer_sales"
            ))).scalar_one() or 0)
            observed["distribution"]["top_customer_sales_share"] = round(top_share, 6)
            if not (
                float(expected["minimum_top_customer_share"])
                <= top_share
                <= float(expected["maximum_top_customer_share"])
            ):
                errors.append(f"头部客户销售额占比异常：{top_share:.2%}")

            invalid_dimension_values = int((await connection.execute(text(
                "SELECT "
                "(SELECT COUNT(*) FROM dim_customer "
                " WHERE gender NOT IN ('男','女','未知') "
                " OR member_level NOT IN ('青铜','白银','黄金','铂金')) + "
                "(SELECT COUNT(*) FROM dim_date "
                " WHERE month NOT BETWEEN 1 AND 12 OR day NOT BETWEEN 1 AND 31 "
                " OR quarter NOT IN ('Q1','Q2','Q3','Q4'))"
            ))).scalar_one())
            observed["invalid_dimension_values"] = invalid_dimension_values
            if invalid_dimension_values:
                errors.append(f"维度枚举/日期值存在 {invalid_dimension_values} 条异常")

            reference_failures: list[dict[str, str]] = []
            timings: list[dict[str, Any]] = []
            repeats = max(1, int(benchmark_repeats))
            suite_cases = load_suite_cases(suite)
            for case in suite_cases:
                try:
                    sql = str(case["expected_sql"])
                    check_sql_read(sql)
                    check_join_relationships(sql)
                    explain = await connection.execute(text(f"EXPLAIN {sql}"))
                    estimated_rows = enforce_explain_cost(
                        [dict(row) for row in explain.mappings().all()],
                        max_explain_rows=app_config.query.max_explain_rows,
                        enabled=app_config.query.cost_guard_enabled,
                    )
                    case_times: list[float] = []
                    row_count = 0
                    columns: set[str] = set()
                    for _ in range(repeats):
                        started = time.perf_counter()
                        result = await connection.execute(text(sql))
                        rows = result.fetchall()
                        case_times.append((time.perf_counter() - started) * 1000)
                        row_count = len(rows)
                        # ``fetchall`` exhausts the cursor; iterating ``result``
                        # here inspects remaining rows rather than projection
                        # metadata and made every non-empty reference query look
                        # as if it returned zero columns. SQLAlchemy exposes the
                        # stable projection contract through ``Result.keys()``.
                        columns = {
                            str(item).lower() for item in result.keys()
                        }
                    expected_columns = {
                        str(item).lower()
                        for item in case.get("expected_projection") or []
                    }
                    if not expected_columns.issubset(columns):
                        raise ValueError(
                            f"投影缺失 {sorted(expected_columns - columns)}，"
                            f"实际 {sorted(columns)}"
                        )
                    timings.append({
                        "id": str(case["id"]),
                        "rows": row_count,
                        "estimated_rows": estimated_rows,
                        "median_ms": _percentile(case_times, 0.50),
                        "max_ms": round(max(case_times), 3),
                    })
                except Exception as exc:  # noqa: BLE001 - aggregate validation evidence
                    reference_failures.append({
                        "id": str(case.get("id")),
                        "error": f"{type(exc).__name__}: {exc}",
                    })
            observed["reference_sql"] = {
                "suite": suite,
                "cases": len(suite_cases),
                "repeats": repeats,
                "failures": reference_failures,
                "p50_ms": _percentile(
                    [float(item["median_ms"]) for item in timings], 0.50,
                ),
                "p95_ms": _percentile(
                    [float(item["median_ms"]) for item in timings], 0.95,
                ),
                "max_ms": round(max(
                    (float(item["max_ms"]) for item in timings), default=0,
                ), 3),
                "slowest": sorted(
                    timings,
                    key=lambda item: float(item["median_ms"]),
                    reverse=True,
                )[:10],
            }
            if reference_failures:
                errors.append(f"{len(reference_failures)} 条期望 SQL 无法安全执行")
            if float(observed["reference_sql"]["p95_ms"]) > 2000:
                warnings.append(
                    f"参考 SQL P95 {observed['reference_sql']['p95_ms']}ms 超过观察线 2000ms"
                )
    except Exception as exc:  # noqa: BLE001 - CLI validation report must close resources
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        await engine.dispose()
    return {
        "fixture": manifest["id"],
        "version": manifest["version"],
        "suite": suite,
        "passed": not errors,
        "observed": observed,
        "warnings": warnings,
        "errors": errors,
    }


def _metric_id(workspace_id: str, index: int, name: str) -> str:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    return f"baseline_{workspace_id[:8]}_{index:03d}_{digest}"


async def _refresh_baseline_enum_examples(
    columns: list[Any],
    dw_repo: Any,
    *,
    max_cardinality: int = 200,
) -> dict[str, Any]:
    """Refresh only bounded, non-sensitive dimension enums from the synthetic DW.

    ``column_info.examples`` helps deterministic table closure for enum values, but
    it must not become an ungoverned copy of the value index.  Keys, measures,
    sensitive columns and high-cardinality dimensions therefore retain no samples;
    complete approved values remain Elasticsearch's responsibility.
    """
    from app.infrastructure.search.elasticsearch.value_retrieval import sensitive_column_reason

    refreshed: list[dict[str, Any]] = []
    cleared: list[dict[str, str]] = []
    limit = max(1, int(max_cardinality))
    for column in columns:
        column_id = str(column.id)
        if str(column.role or "").lower() != "dimension":
            column.examples = []
            cleared.append({"column_id": column_id, "reason": "non_dimension"})
            continue
        sensitive_reason = sensitive_column_reason(column)
        if sensitive_reason:
            column.examples = []
            cleared.append({"column_id": column_id, "reason": sensitive_reason})
            continue
        raw_values = await dw_repo.get_column_values(
            str(column.table_id),
            str(column.name),
            limit + 1,
        )
        values = sorted(
            {value for value in raw_values if value is not None},
            key=lambda value: str(value),
        )
        if str(column.name).lower() != "year" and any(
            not isinstance(value, str) for value in values
        ):
            # Enum closure currently performs substring matching.  Numeric month/day
            # samples such as 20 or 12 would spuriously match years and amounts;
            # four-digit year is the sole bounded numeric exception.
            column.examples = []
            cleared.append({"column_id": column_id, "reason": "numeric_enum"})
            continue
        if len(values) > limit:
            column.examples = []
            cleared.append({"column_id": column_id, "reason": "high_cardinality"})
            continue
        column.examples = values
        refreshed.append({"column_id": column_id, "example_count": len(values)})
    return {
        "max_cardinality": limit,
        "refreshed": refreshed,
        "cleared": cleared,
        "refreshed_column_count": len(refreshed),
        "cleared_column_count": len(cleared),
    }


def _meta_schema_coverage(table_rows: list[tuple[Any, list[Any]]]) -> dict[str, Any]:
    """Compare the live Meta schema authority with the canonical DW baseline.

    The previous Meta check only counted metrics and training examples.  A stale or
    incomplete table/column authority could therefore pass even though column RAG
    had no complete source to project.  Keep this comparison independent of
    Qdrant/ES so it remains a cheap authority check.
    """
    expected_tables = {str(item.name): item for item in meta_config.tables}
    observed_tables = {
        str(table.name): (table, {str(column.name): column for column in columns})
        for table, columns in table_rows
    }
    missing_tables = sorted(set(expected_tables) - set(observed_tables))
    extra_tables = sorted(set(observed_tables) - set(expected_tables))
    missing_columns: list[str] = []
    extra_columns: list[str] = []
    attribute_mismatches: list[dict[str, Any]] = []

    for table_name in sorted(set(expected_tables) & set(observed_tables)):
        expected_table = expected_tables[table_name]
        observed_table, observed_columns = observed_tables[table_name]
        table_fields = []
        if str(observed_table.role or "") != str(expected_table.role or ""):
            table_fields.append("role")
        if str(observed_table.description or "") != str(
            expected_table.description or ""
        ):
            table_fields.append("description")
        if table_fields:
            attribute_mismatches.append({
                "id": table_name,
                "kind": "table",
                "fields": table_fields,
            })

        expected_columns = {
            str(column.name): column for column in expected_table.columns
        }
        missing_columns.extend(
            f"{table_name}.{name}"
            for name in sorted(set(expected_columns) - set(observed_columns))
        )
        extra_columns.extend(
            f"{table_name}.{name}"
            for name in sorted(set(observed_columns) - set(expected_columns))
        )
        for column_name in sorted(set(expected_columns) & set(observed_columns)):
            expected = expected_columns[column_name]
            observed = observed_columns[column_name]
            fields = []
            if str(observed.role or "") != str(expected.role or ""):
                fields.append("role")
            if str(observed.description or "") != str(expected.description or ""):
                fields.append("description")
            if sorted(str(value) for value in (observed.alias or [])) != sorted(
                str(value) for value in (expected.alias or [])
            ):
                fields.append("alias")
            if bool(observed.sync) != bool(expected.sync):
                fields.append("sync")
            if bool(observed.is_display) != bool(expected.is_display):
                fields.append("is_display")
            if fields:
                attribute_mismatches.append({
                    "id": f"{table_name}.{column_name}",
                    "kind": "column",
                    "fields": fields,
                })

    expected_column_count = sum(
        len(table.columns) for table in expected_tables.values()
    )
    observed_column_count = sum(
        len(columns) for _, columns in observed_tables.values()
    )
    return {
        "expected_table_count": len(expected_tables),
        "observed_table_count": len(observed_tables),
        "expected_column_count": expected_column_count,
        "observed_column_count": observed_column_count,
        "missing_tables": missing_tables,
        "extra_tables": extra_tables,
        "missing_columns": sorted(missing_columns),
        "extra_columns": sorted(extra_columns),
        "attribute_mismatches": attribute_mismatches,
        "passed": not (
            missing_tables
            or extra_tables
            or missing_columns
            or extra_columns
            or attribute_mismatches
        ),
    }


def _dw_meta_physical_coverage(
    snapshot_tables: list[dict[str, Any]],
    table_rows: list[tuple[Any, list[Any]]],
) -> dict[str, Any]:
    """Check that Meta mirrors DW physical table/column identity and types.

    Roles, descriptions, aliases and retrieval flags are curated Meta semantics and
    are validated separately by ``_meta_schema_coverage``.  Only names and physical
    types belong to the DW authority; mixing both concerns made harmless comment
    differences obscure real type drift such as DECIMAL being stored as FLOAT.
    """
    expected_tables = {
        str(table["name"]): {
            str(column["name"]): str(column.get("type") or "").lower()
            for column in (table.get("columns") or [])
        }
        for table in snapshot_tables
    }
    observed_tables = {
        str(table.name): {
            str(column.name): str(column.type or "").lower()
            for column in columns
        }
        for table, columns in table_rows
    }
    missing_tables = sorted(set(expected_tables) - set(observed_tables))
    extra_tables = sorted(set(observed_tables) - set(expected_tables))
    missing_columns: list[str] = []
    extra_columns: list[str] = []
    type_mismatches: list[dict[str, str]] = []
    for table_name in sorted(set(expected_tables) & set(observed_tables)):
        expected_columns = expected_tables[table_name]
        observed_columns = observed_tables[table_name]
        missing_columns.extend(
            f"{table_name}.{name}"
            for name in sorted(set(expected_columns) - set(observed_columns))
        )
        extra_columns.extend(
            f"{table_name}.{name}"
            for name in sorted(set(observed_columns) - set(expected_columns))
        )
        for column_name in sorted(set(expected_columns) & set(observed_columns)):
            expected_type = expected_columns[column_name]
            observed_type = observed_columns[column_name]
            if expected_type != observed_type:
                type_mismatches.append({
                    "id": f"{table_name}.{column_name}",
                    "dw_type": expected_type,
                    "meta_type": observed_type,
                })
    return {
        "dw_table_count": len(expected_tables),
        "meta_table_count": len(observed_tables),
        "dw_column_count": sum(len(value) for value in expected_tables.values()),
        "meta_column_count": sum(len(value) for value in observed_tables.values()),
        "missing_tables": missing_tables,
        "extra_tables": extra_tables,
        "missing_columns": sorted(missing_columns),
        "extra_columns": sorted(extra_columns),
        "type_mismatches": type_mismatches,
        "passed": not (
            missing_tables
            or extra_tables
            or missing_columns
            or extra_columns
            or type_mismatches
        ),
    }


def _meta_content_coverage(
    metric_rows: list[Any],
    training_rows: list[Any],
) -> dict[str, Any]:
    """Compare enabled Meta metric/training content with the canonical baseline."""

    def metric_record(item: Any) -> dict[str, Any]:
        return {
            "name": str(item.name),
            "description": item.description,
            "alias": item.alias,
            "relevant_columns": item.relevant_columns,
            "formula_rule": item.formula_rule,
        }

    expected_metrics = [metric_record(item) for item in meta_config.metrics]
    observed_metrics = [metric_record(item) for item in metric_rows]
    expected_metrics_by_name = {
        item["name"]: item for item in expected_metrics
    }
    observed_metrics_by_name: dict[str, list[dict[str, Any]]] = {}
    for item in observed_metrics:
        observed_metrics_by_name.setdefault(item["name"], []).append(item)

    missing_metric_names = sorted(
        set(expected_metrics_by_name) - set(observed_metrics_by_name)
    )
    extra_metric_names = sorted(
        set(observed_metrics_by_name) - set(expected_metrics_by_name)
    )
    duplicate_metric_names = sorted(
        name
        for name, items in observed_metrics_by_name.items()
        if len(items) > 1
    )
    metric_attribute_mismatches: list[dict[str, Any]] = []
    metric_fields = (
        "description",
        "alias",
        "relevant_columns",
        "formula_rule",
    )
    for name in sorted(
        set(expected_metrics_by_name) & set(observed_metrics_by_name)
    ):
        expected_metric = expected_metrics_by_name[name]
        observed_candidates = observed_metrics_by_name[name]
        fields = [
            field
            for field in metric_fields
            if any(
                candidate[field] != expected_metric[field]
                for candidate in observed_candidates
            )
        ]
        if fields:
            metric_attribute_mismatches.append({
                "name": name,
                "fields": fields,
                "expected": {
                    field: expected_metric[field] for field in fields
                },
                "observed": [
                    {field: candidate[field] for field in fields}
                    for candidate in observed_candidates
                ],
            })

    metric_report = {
        "expected_count": len(expected_metrics),
        "observed_count": len(observed_metrics),
        "missing_names": missing_metric_names,
        "extra_names": extra_metric_names,
        "duplicate_names": duplicate_metric_names,
        "attribute_mismatches": metric_attribute_mismatches,
        "passed": not (
            missing_metric_names
            or extra_metric_names
            or duplicate_metric_names
            or metric_attribute_mismatches
        ),
    }

    expected_training_by_question = {
        str(item["question"]): str(item["sql"])
        for item in baseline_training_examples()
    }
    observed_training_by_question: dict[str, list[str]] = {}
    for item in training_rows:
        observed_training_by_question.setdefault(str(item.question), []).append(
            str(item.sql)
        )

    missing_training_questions = sorted(
        set(expected_training_by_question) - set(observed_training_by_question)
    )
    extra_training_questions = sorted(
        set(observed_training_by_question) - set(expected_training_by_question)
    )
    duplicate_training_questions = sorted(
        question
        for question, sqls in observed_training_by_question.items()
        if len(sqls) > 1
    )
    training_sql_mismatches = []
    for question in sorted(
        set(expected_training_by_question) & set(observed_training_by_question)
    ):
        expected_sql = expected_training_by_question[question]
        observed_sqls = observed_training_by_question[question]
        if any(sql != expected_sql for sql in observed_sqls):
            training_sql_mismatches.append({
                "question": question,
                "expected_sql": expected_sql,
                "observed_sqls": observed_sqls,
            })

    training_report = {
        "expected_count": len(expected_training_by_question),
        "observed_count": len(training_rows),
        "missing_questions": missing_training_questions,
        "extra_questions": extra_training_questions,
        "duplicate_questions": duplicate_training_questions,
        "sql_mismatches": training_sql_mismatches,
        "passed": not (
            missing_training_questions
            or extra_training_questions
            or duplicate_training_questions
            or training_sql_mismatches
        ),
    }
    return {
        "metrics": metric_report,
        "training_examples": training_report,
        "passed": bool(metric_report["passed"] and training_report["passed"]),
    }


async def check_baseline_meta(workspace_id: str) -> dict[str, Any]:
    manifest = baseline_manifest()
    expected = dict(manifest["meta"])
    engine = create_async_engine(
        _database_url(app_config.db_meta, app_config.db_meta.database),
        pool_pre_ping=True,
    )
    dw_engine = create_async_engine(
        _database_url(app_config.db_dw, str(manifest["database"])),
        pool_pre_ping=True,
    )
    errors: list[str] = []
    observed: dict[str, Any] = {"workspace_id": workspace_id}
    try:
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as session:
            repo = MetaMysqlRepository(session, workspace_id)
            await repo.ensure_admin_tables()
            if await session.get(WorkspaceMySQL, workspace_id) is None:
                errors.append("工作空间不存在")
            table_rows = await repo.list_table_infos()
            schema_coverage = _meta_schema_coverage(table_rows)
            from app.infrastructure.persistence.mysql.repositories.dw_mysql_repository import DWMysqlRepository

            dw_factory = async_sessionmaker(dw_engine, expire_on_commit=False)
            async with dw_factory() as dw_session:
                dw_snapshot = await DWMysqlRepository(dw_session).get_schema_snapshot()
            physical_coverage = _dw_meta_physical_coverage(
                list(dw_snapshot.get("tables") or []),
                table_rows,
            )
            metric_rows = list((await session.execute(
                select(MetricInfoMySQL).where(
                    MetricInfoMySQL.workspace_id == workspace_id,
                    MetricInfoMySQL.enabled.is_(True),
                )
            )).scalars().all())
            training_rows = list((await session.execute(
                select(TrainingInfoMySQL).where(
                    TrainingInfoMySQL.workspace_id == workspace_id,
                    TrainingInfoMySQL.enabled.is_(True),
                )
            )).scalars().all())
            content_coverage = _meta_content_coverage(
                metric_rows,
                training_rows,
            )
            metric_count = len(metric_rows)
            training_count = len(training_rows)
            duplicate_questions = int((await session.execute(text(
                "SELECT COUNT(*) FROM ("
                "SELECT question FROM training_info WHERE workspace_id = :workspace_id "
                "GROUP BY question HAVING COUNT(*) > 1"
                ") duplicated"
            ), {"workspace_id": workspace_id})).scalar_one())
            training_questions = [
                str(item.question)
                for item in training_rows
            ]
            evaluation_questions = {
                _normalized_question(str(item["question"]))
                for suite in SUITE_SPECS
                for item in load_suite_cases(suite)
            }
            evaluation_overlap = sorted({
                value
                for value in training_questions
                if _normalized_question(value) in evaluation_questions
            })
            observed.update({
                "schema_coverage": schema_coverage,
                "physical_coverage": physical_coverage,
                "content_coverage": content_coverage,
                "enabled_metrics": metric_count,
                "enabled_training_examples": training_count,
                "duplicate_training_questions": duplicate_questions,
                "evaluation_question_overlap": evaluation_overlap,
            })
            if not schema_coverage["passed"]:
                errors.append(
                    "Meta 表/字段权威与 canonical baseline 不一致："
                    + json.dumps(schema_coverage, ensure_ascii=False)
                )
            if not physical_coverage["passed"]:
                errors.append(
                    "DW 物理 Schema 与 Meta 镜像不一致："
                    + json.dumps(physical_coverage, ensure_ascii=False)
                )
            if not content_coverage["passed"]:
                errors.append(
                    "Meta 指标/训练内容与 canonical baseline 不一致："
                    + json.dumps(content_coverage, ensure_ascii=False)
                )
            if metric_count < int(expected["minimum_metrics"]):
                errors.append(
                    f"启用指标 {metric_count} < {expected['minimum_metrics']}"
                )
            if training_count != int(expected["exact_training_examples"]):
                errors.append(
                    f"启用训练样例 {training_count} != "
                    f"{expected['exact_training_examples']}"
                )
            if duplicate_questions:
                errors.append(f"训练样例存在 {duplicate_questions} 组重复问题")
            if len(evaluation_overlap) > int(
                expected["maximum_evaluation_question_overlap"]
            ):
                errors.append(
                    f"训练样例与评测题精确重叠 {len(evaluation_overlap)} 条"
                )
    except Exception as exc:  # noqa: BLE001 - CLI validation report must close resources
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        await engine.dispose()
        await dw_engine.dispose()
    return {
        "passed": not errors,
        "observed": observed,
        "errors": errors,
    }


async def sync_baseline_meta(
    workspace_id: str,
    *,
    rebuild_indexes: bool = False,
) -> dict[str, Any]:
    """Replace only one workspace's metrics/trainings with the canonical baseline."""
    engine = create_async_engine(
        _database_url(app_config.db_meta, app_config.db_meta.database),
        pool_pre_ping=True,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    index_report: dict[str, Any] = {}
    try:
        async with factory() as session:
            repo = MetaMysqlRepository(session, workspace_id)
            await repo.ensure_admin_tables()
            if await session.get(WorkspaceMySQL, workspace_id) is None:
                raise ValueError(f"工作空间不存在：{workspace_id}")

            existing_metric_ids = select(MetricInfoMySQL.id).where(
                MetricInfoMySQL.workspace_id == workspace_id,
            )
            await session.execute(delete(ColumnMetricMySQL).where(
                ColumnMetricMySQL.metric_id.in_(existing_metric_ids),
            ))
            await session.execute(delete(MetricInfoMySQL).where(
                MetricInfoMySQL.workspace_id == workspace_id,
            ))
            await session.execute(delete(TrainingInfoMySQL).where(
                TrainingInfoMySQL.workspace_id == workspace_id,
            ))

            metrics: list[MetricInfoMySQL] = []
            column_metrics: list[ColumnMetricMySQL] = []
            for index, metric in enumerate(meta_config.metrics, 1):
                metric_id = _metric_id(workspace_id, index, metric.name)
                metrics.append(MetricInfoMySQL(
                    id=metric_id,
                    name=metric.name,
                    description=metric.description,
                    relevant_columns=list(metric.relevant_columns),
                    alias=list(metric.alias),
                    workspace_id=workspace_id,
                    formula_rule=metric.formula_rule,
                    enabled=True,
                ))
                column_metrics.extend(
                    ColumnMetricMySQL(column_id=column_id, metric_id=metric_id)
                    for column_id in metric.relevant_columns
                )
            trainings = [
                TrainingInfoMySQL(
                    workspace_id=workspace_id,
                    question=str(example["question"]),
                    sql=str(example["sql"]),
                    enabled=True,
                )
                for example in baseline_training_examples()
            ]
            session.add_all([*metrics, *column_metrics, *trainings])
            await session.commit()

            if rebuild_indexes:
                from app.infrastructure.clients.embedding_client_manager import (
                    embedding_client_manager,
                )
                from app.infrastructure.clients.es_client_manager import es_client_manager
                from app.infrastructure.clients.qdrant_client_manager import qdrant_client_manager
                from app.infrastructure.search.elasticsearch.repositories.value_es_repository import ValueESRepository
                from app.infrastructure.persistence.mysql.repositories.dw_mysql_repository import DWMysqlRepository
                from app.infrastructure.search.qdrant.repositories.column_qdrant_repository import (
                    ColumnQdrantRepository,
                )
                from app.infrastructure.search.qdrant.repositories.metric_qdrant_repository import (
                    MetricQdrantRepository,
                )
                from app.infrastructure.search.qdrant.repositories.training_qdrant_repository import (
                    TrainingQdrantRepository,
                )
                from app.application.services.meta_admin_service import MetaAdminService
                from app.application.services.value_index_service import ValueIndexService

                embedding_client_manager.init()
                es_client_manager.init()
                qdrant_client_manager.init()
                dw_engine = create_async_engine(
                    _database_url(app_config.db_dw, "dw2"),
                    pool_pre_ping=True,
                )
                try:
                    service = MetaAdminService(
                        repo,
                        ColumnQdrantRepository(qdrant_client_manager.client),
                        MetricQdrantRepository(qdrant_client_manager.client),
                        TrainingQdrantRepository(qdrant_client_manager.client),
                        embedding_client_manager.client,
                        value_es_repo=ValueESRepository(es_client_manager.client),
                    )
                    columns = [
                        column
                        for _, table_columns in await repo.list_table_infos()
                        for column in table_columns
                    ]
                    dw_factory = async_sessionmaker(
                        dw_engine, expire_on_commit=False,
                    )
                    async with dw_factory() as dw_session:
                        dw_repo = DWMysqlRepository(dw_session)
                        index_report["enum_examples"] = (
                            await _refresh_baseline_enum_examples(columns, dw_repo)
                        )
                        await session.commit()
                        # Column recall filters by the exact current index version.
                        # Rebuilding only metric/training/value projections can leave
                        # a green collection with 0 queryable column points.
                        index_report["columns"] = await service.rebuild_columns()
                        index_report["metrics"] = await service.rebuild_metrics()
                        index_report["training"] = await service.rebuild_training()
                        index_report["values"] = await ValueIndexService(
                            dw_repo,
                            ValueESRepository(es_client_manager.client),
                            workspace_id=workspace_id,
                        ).rebuild(columns)
                    index_report["hybrid_status"] = await service.hybrid_status()
                    index_report["value_status"] = await service.value_index_status()
                finally:
                    await dw_engine.dispose()
                    await embedding_client_manager.close()
                    await es_client_manager.close()
                    await qdrant_client_manager.close()
    finally:
        await engine.dispose()

    checked = await check_baseline_meta(workspace_id)
    if rebuild_indexes:
        hybrid_ready = bool(
            (index_report.get("hybrid_status") or {}).get("ready")
        )
        value_ready = bool(
            (index_report.get("value_status") or {}).get("ready")
        )
        if not hybrid_ready:
            checked["errors"].append("Qdrant 三路投影版本或点数未就绪")
        if not value_ready:
            checked["errors"].append("Elasticsearch 字段值索引未就绪")
        checked["passed"] = bool(checked["passed"] and hybrid_ready and value_ready)
    return {
        **checked,
        "replaced_scope": {"workspace_id": workspace_id},
        "indexes": index_report,
    }


def render_report(
    suite_report: dict[str, Any],
    *,
    dw_report: dict[str, Any] | None = None,
    meta_report: dict[str, Any] | None = None,
) -> str:
    lines = [
        "# Data Agent 规模化评测套件",
        "",
        f"- 套件：{suite_report['suite']}",
        f"- 用例：{suite_report['case_count']}",
        f"- 类别：{len(suite_report['categories'])}",
        f"- 标签：{len(suite_report['tags'])}",
        (
            f"- 分层：smoke={suite_report['tiers']['smoke']}，"
            f"regression={suite_report['tiers']['regression']}"
        ),
        f"- 结构合同：{'通过' if suite_report['structural_ready'] else '失败'}",
        f"- 结构治理状态：{suite_report['promotion_status']}",
        (
            "- 结构候选就绪："
            + ("是" if suite_report["structural_promotion_ready"] else "否")
        ),
        (
            "- 完整运行证据就绪："
            + ("是" if suite_report["promotion_ready"] else "否")
        ),
    ]
    if suite_report["missing_required_categories"]:
        lines.append(
            "- 缺少类别：" + "、".join(suite_report["missing_required_categories"])
        )
    if suite_report["missing_required_tags"]:
        lines.append(
            "- 缺少标签：" + "、".join(suite_report["missing_required_tags"])
        )
    if dw_report is not None:
        reference = dw_report["observed"].get("reference_sql") or {}
        lines.extend([
            "",
            "## DW 数据质量与参考查询",
            "",
            f"- Fixture：{dw_report['fixture']}@{dw_report['version']}",
            f"- 校验：{'通过' if dw_report['passed'] else '失败'}",
            (
                "- 行数："
                + json.dumps(
                    dw_report["observed"].get("row_counts", {}),
                    ensure_ascii=False,
                )
            ),
            (
                f"- 参考 SQL：{reference.get('cases', 0)} 条，"
                f"P50={reference.get('p50_ms', 0)}ms，"
                f"P95={reference.get('p95_ms', 0)}ms，"
                f"max={reference.get('max_ms', 0)}ms"
            ),
        ])
        if dw_report["warnings"]:
            lines.append("- 警告：" + "；".join(dw_report["warnings"]))
        if dw_report["errors"]:
            lines.append("- 错误：" + "；".join(dw_report["errors"]))
    if meta_report is not None:
        observed = meta_report.get("observed") or {}
        lines.extend([
            "",
            "## Meta 规模与索引",
            "",
            f"- 校验：{'通过' if meta_report['passed'] else '失败'}",
            f"- 指标：{observed.get('enabled_metrics', 0)}",
            f"- 训练样例：{observed.get('enabled_training_examples', 0)}",
            (
                "- 索引："
                + json.dumps(meta_report.get("indexes") or {}, ensure_ascii=False)
            ),
        ])
        if meta_report["errors"]:
            lines.append("- 错误：" + "；".join(meta_report["errors"]))
    return "\n".join(lines)


async def _main(args: argparse.Namespace) -> int:
    installed_orders: int | None = None
    if args.install_baseline:
        installed_orders = install_baseline_fixture(
            confirm_reset=args.confirm_reset,
            profile=args.profile,
            order_count=args.order_count,
        )
    suite_report = dataset_suite_report(args.suite)
    dw_report = (
        await check_baseline_dw(
            benchmark_repeats=args.benchmark_repeats,
            suite=args.suite,
        )
        if args.check_dw else None
    )
    if dw_report is not None:
        runtime = suite_report["runtime_validation"]
        runtime["reference_sql_observed"] = bool(dw_report["passed"])
        runtime["ready"] = bool(
            (
                not runtime["reference_sql_required"]
                or runtime["reference_sql_observed"]
            )
            and (
                not runtime["result_equivalence_required"]
                or runtime["result_equivalence_observed"]
            )
        )
        suite_report["promotion_ready"] = bool(
            suite_report["structural_promotion_ready"] and runtime["ready"]
        )
    meta_report: dict[str, Any] | None = None
    if args.sync_meta:
        if not args.workspace_id:
            raise ValueError("--sync-meta 必须提供 --workspace-id")
        meta_report = await sync_baseline_meta(
            args.workspace_id,
            rebuild_indexes=args.rebuild_indexes,
        )
    elif args.check_meta:
        if not args.workspace_id:
            raise ValueError("--check-meta 必须提供 --workspace-id")
        meta_report = await check_baseline_meta(args.workspace_id)
    elif args.rebuild_indexes:
        raise ValueError("--rebuild-indexes 必须与 --sync-meta 一起使用")

    payload = {
        "installed_orders": installed_orders,
        "suite": suite_report,
        "dw": dw_report,
        "meta": meta_report,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(render_report(
            suite_report,
            dw_report=dw_report,
            meta_report=meta_report,
        ))
    passed = (
        suite_report["structural_ready"]
        and (dw_report is None or dw_report["passed"])
        and (meta_report is None or meta_report["passed"])
    )
    return 0 if passed else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="构建并验证 Data Agent 规模化评测套件",
    )
    parser.add_argument(
        "--suite",
        choices=tuple(SUITE_SPECS),
        default="easy_sql_v1",
        help="选择简单、中等或困难 SQL 治理评测集",
    )
    parser.add_argument(
        "--install-baseline",
        action="store_true",
        help="通过项目 Docker MySQL 删除并重建正式 dw2",
    )
    parser.add_argument(
        "--confirm-reset",
        action="store_true",
        help="确认允许删除并重建正式 dw2",
    )
    parser.add_argument(
        "--profile",
        choices=("medium", "large"),
        default="medium",
        help="medium=25 万订单；large=100 万订单",
    )
    parser.add_argument(
        "--order-count",
        type=int,
        default=None,
        help="覆盖 profile 的订单数，允许 10 万～100 万",
    )
    parser.add_argument(
        "--check-dw",
        action="store_true",
        help="校验规模、完整性、分布并执行所选套件的全部参考 SQL",
    )
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=2,
        help="每条参考 SQL 的执行次数",
    )
    parser.add_argument(
        "--sync-meta",
        action="store_true",
        help="替换指定工作空间的指标与 60 条训练样例",
    )
    parser.add_argument(
        "--check-meta",
        action="store_true",
        help="校验指定工作空间的 Meta 基线",
    )
    parser.add_argument(
        "--rebuild-indexes",
        action="store_true",
        help="同步 Meta 后重建字段、指标、训练 Qdrant 与 ES 字段值索引",
    )
    parser.add_argument("--workspace-id", default=None)
    parser.add_argument("--json", action="store_true")
    try:
        raise SystemExit(asyncio.run(_main(parser.parse_args())))
    except ValueError as exc:
        parser.error(str(exc))
