"""把旧 ``rag_recall_set.yaml`` 可复现地转换为正规化 CSV 三表。"""

from __future__ import annotations

import argparse
import csv
import io
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from evals.datasets.models import (
    Answerability,
    DatasetBundle,
    DatasetCase,
    RelevanceJudgment,
    ReviewStatus,
)
from evals.paths import LEGACY_DATASETS_ROOT, RAG_V1_DATASET_ROOT


DEFAULT_SOURCE = LEGACY_DATASETS_ROOT / "rag_recall_set.yaml"
DEFAULT_DESTINATION = RAG_V1_DATASET_ROOT
DATASET_ID = "da-rag-normalized-v1"
DATASET_VERSION = "draft-1"

_CASE_FIELDS = (
    "case_id",
    "dataset_id",
    "dataset_version",
    "split",
    "query",
    "answerability",
    "gold_answer",
    "slice",
    "language",
    "domain",
    "difficulty",
    "risk",
    "kb_version",
    "parent_case_id",
    "intervention",
    "changed_variable",
    "seed",
    "evaluation_scope",
    "source_dataset_version",
    "source_case_id",
    "migration_status",
)
_QREL_FIELDS = (
    "case_id",
    "document_id",
    "chunk_id",
    "relevance_grade",
    "review_status",
    "evidence_start",
    "evidence_end",
    "reviewer",
    "evidence_type",
    "source_fields",
    "source_value",
    "migration_status",
)
_CLAIM_FIELDS = (
    "case_id",
    "claim_id",
    "claim_text",
    "required",
    "support_document_id",
    "support_chunk_id",
    "importance",
    "verification_method",
    "review_status",
    "migration_status",
)


class LegacyRagMigrationError(ValueError):
    """表示旧 RAG YAML 不满足可迁移的最小结构。"""


def _load_source(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise LegacyRagMigrationError(f"无法读取旧 RAG 数据集 {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LegacyRagMigrationError("旧 RAG 数据集顶层必须是对象")
    if not isinstance(payload.get("version"), str) or not payload["version"].strip():
        raise LegacyRagMigrationError("旧 RAG 数据集缺少 version")
    if not isinstance(payload.get("cases"), list) or not payload["cases"]:
        raise LegacyRagMigrationError("旧 RAG 数据集 cases 必须是非空列表")
    return payload


def _string_list(case_id: str, case: Mapping[str, Any], field: str) -> tuple[str, ...]:
    raw = case.get(field)
    if raw is None:
        return ()
    if not isinstance(raw, list) or any(
        not isinstance(value, str) or not value.strip() for value in raw
    ):
        raise LegacyRagMigrationError(f"{case_id}.{field} 必须是非空字符串列表")
    return tuple(value.strip() for value in raw)


def _evidence_id(evidence_type: str, value: str) -> str:
    return f"{evidence_type}:{value}"


def _collect_qrels(
    case_id: str,
    case: Mapping[str, Any],
) -> tuple[list[dict[str, str]], list[RelevanceJudgment]]:
    # 同一列可同时承担 relevant/display 角色；按 Evidence Key 合并，避免重复 qrel。
    evidence: dict[tuple[str, str], dict[str, Any]] = {}
    sources = (
        ("expected_tables", "table"),
        ("relevant_columns", "column"),
        ("display_columns", "column"),
        ("relevant_metrics", "metric"),
        ("relevant_training", "training"),
        ("relevant_values", "value"),
    )
    for source_field, evidence_type in sources:
        for value in _string_list(case_id, case, source_field):
            key = (evidence_type, value)
            item = evidence.setdefault(
                key,
                {
                    "document_id": _evidence_id(evidence_type, value),
                    "evidence_type": evidence_type,
                    "source_value": value,
                    "source_fields": [],
                },
            )
            item["source_fields"].append(source_field)

    rows: list[dict[str, str]] = []
    contracts: list[RelevanceJudgment] = []
    for item in evidence.values():
        source_fields = "|".join(item["source_fields"])
        rows.append(
            {
                "case_id": case_id,
                "document_id": item["document_id"],
                "chunk_id": "",
                "relevance_grade": "3",
                "review_status": ReviewStatus.DRAFT.value,
                "evidence_start": "",
                "evidence_end": "",
                "reviewer": "",
                "evidence_type": item["evidence_type"],
                "source_fields": source_fields,
                "source_value": item["source_value"],
                "migration_status": "requires_adjudication",
            }
        )
        contracts.append(
            RelevanceJudgment(
                case_id=case_id,
                document_id=item["document_id"],
                chunk_id=None,
                relevance_grade=3,
                review_status=ReviewStatus.DRAFT,
                attributes={
                    "evidence_type": item["evidence_type"],
                    "source_fields": source_fields,
                    "source_value": item["source_value"],
                    "migration_status": "requires_adjudication",
                },
            )
        )
    if not rows:
        raise LegacyRagMigrationError(f"{case_id} 没有可迁移的 Gold Evidence")
    return rows, contracts


def _render_csv(fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def render_legacy_rag_dataset(source: Path = DEFAULT_SOURCE) -> dict[str, str]:
    """读取旧 YAML 并返回确定性 CSV 内容，不写入文件。"""

    payload = _load_source(source)
    source_version = str(payload["version"]).strip()
    case_rows: list[dict[str, str]] = []
    qrel_rows: list[dict[str, str]] = []
    case_contracts: list[DatasetCase] = []
    qrel_contracts: list[RelevanceJudgment] = []
    seen_ids: set[str] = set()

    for index, raw_case in enumerate(payload["cases"], start=1):
        if not isinstance(raw_case, dict):
            raise LegacyRagMigrationError(f"第 {index} 个 Case 必须是对象")
        case_id = str(raw_case.get("id") or "").strip()
        query = str(raw_case.get("query") or "").strip()
        if not case_id or not query:
            raise LegacyRagMigrationError(f"第 {index} 个 Case 缺少 id 或 query")
        if case_id in seen_ids:
            raise LegacyRagMigrationError(f"旧 RAG 数据集存在重复 Case ID: {case_id}")
        seen_ids.add(case_id)

        case_rows.append(
            {
                "case_id": case_id,
                "dataset_id": DATASET_ID,
                "dataset_version": DATASET_VERSION,
                "split": "development",
                "query": query,
                "answerability": Answerability.ANSWERABLE.value,
                "gold_answer": "",
                "slice": "legacy_retrieval",
                "language": "zh-CN",
                "domain": "nl2sql",
                "difficulty": "",
                "risk": "",
                "kb_version": "to-be-frozen",
                "parent_case_id": "",
                "intervention": "",
                "changed_variable": "",
                "seed": "",
                "evaluation_scope": "retrieval_only",
                "source_dataset_version": source_version,
                "source_case_id": case_id,
                "migration_status": "requires_adjudication",
            }
        )
        case_contracts.append(
            DatasetCase(
                case_id=case_id,
                dataset_id=DATASET_ID,
                dataset_version=DATASET_VERSION,
                split="development",
                query=query,
                answerability=Answerability.ANSWERABLE,
                slice_name="legacy_retrieval",
                language="zh-CN",
                domain="nl2sql",
                kb_version="to-be-frozen",
                attributes={
                    "evaluation_scope": "retrieval_only",
                    "source_dataset_version": source_version,
                    "source_case_id": case_id,
                    "migration_status": "requires_adjudication",
                },
            )
        )
        rows, contracts = _collect_qrels(case_id, raw_case)
        qrel_rows.extend(rows)
        qrel_contracts.extend(contracts)

    # 先通过统一合同校验，再生成文件。旧数据没有 Gold Answer，因此 claims 仅
    # 保留表头；生成/引用指标必须保持 SKIPPED，直到完成独立标注与裁决。
    DatasetBundle(tuple(case_contracts), tuple(qrel_contracts), ())
    return {
        "cases.csv": _render_csv(_CASE_FIELDS, case_rows),
        "relevance-judgments.csv": _render_csv(_QREL_FIELDS, qrel_rows),
        "claims.csv": _render_csv(_CLAIM_FIELDS, ()),
    }


def write_legacy_rag_dataset(
    source: Path = DEFAULT_SOURCE,
    destination: Path = DEFAULT_DESTINATION,
) -> dict[str, Path]:
    """原子写入正规化三表，并返回各产物路径。"""

    rendered = render_legacy_rag_dataset(source)
    destination.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for filename, content in rendered.items():
        target = destination / filename
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(content, encoding="utf-8", newline="")
        os.replace(temporary, target)
        written[filename] = target
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """执行迁移；``--check`` 只核对已提交产物，不产生写入。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    rendered = render_legacy_rag_dataset(args.source)
    if args.check:
        mismatches = [
            filename
            for filename, content in rendered.items()
            if not (args.destination / filename).is_file()
            or (args.destination / filename).read_text(encoding="utf-8") != content
        ]
        if mismatches:
            parser.error("迁移产物不是最新确定性输出: " + ", ".join(mismatches))
        return 0
    write_legacy_rag_dataset(args.source, args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
