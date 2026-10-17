"""正规化 CSV 数据集的无副作用加载器。"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from evals.contracts import ContractError
from evals.datasets.models import (
    Answerability,
    ClaimRecord,
    DatasetBundle,
    DatasetCase,
    DatasetValidationError,
    RelevanceJudgment,
    ReviewStatus,
)


_T = TypeVar("_T")

_CASE_REQUIRED = {
    "case_id",
    "dataset_id",
    "dataset_version",
    "split",
    "query",
    "answerability",
}
_QREL_REQUIRED = {
    "case_id",
    "document_id",
    "chunk_id",
    "relevance_grade",
    "review_status",
}
_CLAIM_REQUIRED = {
    "case_id",
    "claim_id",
    "claim_text",
    "required",
    "support_document_id",
    "support_chunk_id",
}

_CASE_KNOWN = _CASE_REQUIRED | {
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
}
_QREL_KNOWN = _QREL_REQUIRED | {
    "evidence_start",
    "evidence_end",
    "reviewer",
}
_CLAIM_KNOWN = _CLAIM_REQUIRED | {
    "importance",
    "verification_method",
}


def _optional(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _required(value: str | None, field_name: str) -> str:
    parsed = _optional(value)
    if parsed is None:
        raise ValueError(f"{field_name} 不能为空")
    return parsed


def _optional_int(value: str | None, field_name: str) -> int | None:
    parsed = _optional(value)
    if parsed is None:
        return None
    try:
        return int(parsed)
    except ValueError as exc:
        raise ValueError(f"{field_name} 必须是整数") from exc


def _required_bool(value: str | None, field_name: str) -> bool:
    parsed = _required(value, field_name).lower()
    if parsed in {"true", "1", "yes"}:
        return True
    if parsed in {"false", "0", "no"}:
        return False
    raise ValueError(f"{field_name} 必须是 true/false")


def _extra(row: dict[str, str | None], known: set[str]) -> dict[str, str]:
    return {
        key: value.strip()
        for key, value in row.items()
        if key not in known and value is not None and value.strip()
    }


class DatasetLoader:
    """从 ``cases/qrels/claims`` 三个 CSV 构建并校验 DatasetBundle。"""

    def load(self, directory: str | Path) -> DatasetBundle:
        """从约定文件名加载完整数据集。"""

        root = Path(directory)
        cases, case_issues = self._read(
            root / "cases.csv", _CASE_REQUIRED, self._parse_case
        )
        qrels, qrel_issues = self._read(
            root / "relevance-judgments.csv", _QREL_REQUIRED, self._parse_qrel
        )
        claims, claim_issues = self._read(
            root / "claims.csv", _CLAIM_REQUIRED, self._parse_claim
        )
        issues = [*case_issues, *qrel_issues, *claim_issues]
        if issues:
            raise DatasetValidationError(issues)
        return DatasetBundle(tuple(cases), tuple(qrels), tuple(claims))

    def _read(
        self,
        path: Path,
        required_headers: set[str],
        parser: Callable[[dict[str, str | None]], _T],
    ) -> tuple[list[_T], list[str]]:
        if not path.is_file():
            return [], [f"缺少数据文件: {path.name}"]

        rows: list[_T] = []
        issues: list[str] = []
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                headers = set(reader.fieldnames or ())
                missing = sorted(required_headers - headers)
                if missing:
                    return [], [f"{path.name} 缺少列: {', '.join(missing)}"]
                for line_number, row in enumerate(reader, start=2):
                    try:
                        rows.append(parser(row))
                    except (ContractError, ValueError) as exc:
                        issues.append(f"{path.name}:{line_number}: {exc}")
        except UnicodeDecodeError as exc:
            issues.append(f"{path.name} 不是合法 UTF-8 CSV: {exc}")
        return rows, issues

    @staticmethod
    def _parse_case(row: dict[str, str | None]) -> DatasetCase:
        seed = _optional_int(row.get("seed"), "seed")
        return DatasetCase(
            case_id=_required(row.get("case_id"), "case_id"),
            dataset_id=_required(row.get("dataset_id"), "dataset_id"),
            dataset_version=_required(row.get("dataset_version"), "dataset_version"),
            split=_required(row.get("split"), "split"),
            query=_required(row.get("query"), "query"),
            answerability=Answerability(
                _required(row.get("answerability"), "answerability").lower()
            ),
            gold_answer=_optional(row.get("gold_answer")),
            slice_name=_optional(row.get("slice")),
            language=_optional(row.get("language")),
            domain=_optional(row.get("domain")),
            difficulty=_optional(row.get("difficulty")),
            risk=_optional(row.get("risk")),
            kb_version=_optional(row.get("kb_version")),
            parent_case_id=_optional(row.get("parent_case_id")),
            intervention=_optional(row.get("intervention")),
            changed_variable=_optional(row.get("changed_variable")),
            seed=seed,
            attributes=_extra(row, _CASE_KNOWN),
        )

    @staticmethod
    def _parse_qrel(row: dict[str, str | None]) -> RelevanceJudgment:
        grade = _optional_int(row.get("relevance_grade"), "relevance_grade")
        if grade is None:
            raise ValueError("relevance_grade 不能为空")
        return RelevanceJudgment(
            case_id=_required(row.get("case_id"), "case_id"),
            document_id=_required(row.get("document_id"), "document_id"),
            chunk_id=_optional(row.get("chunk_id")),
            relevance_grade=grade,
            review_status=ReviewStatus(
                _required(row.get("review_status"), "review_status").lower()
            ),
            evidence_start=_optional_int(row.get("evidence_start"), "evidence_start"),
            evidence_end=_optional_int(row.get("evidence_end"), "evidence_end"),
            reviewer=_optional(row.get("reviewer")),
            attributes=_extra(row, _QREL_KNOWN),
        )

    @staticmethod
    def _parse_claim(row: dict[str, str | None]) -> ClaimRecord:
        importance_value = _optional(row.get("importance"))
        try:
            importance = float(importance_value) if importance_value is not None else 1.0
        except ValueError as exc:
            raise ValueError("importance 必须是数值") from exc
        return ClaimRecord(
            case_id=_required(row.get("case_id"), "case_id"),
            claim_id=_required(row.get("claim_id"), "claim_id"),
            claim_text=_required(row.get("claim_text"), "claim_text"),
            required=_required_bool(row.get("required"), "required"),
            support_document_id=_optional(row.get("support_document_id")),
            support_chunk_id=_optional(row.get("support_chunk_id")),
            importance=importance,
            verification_method=_optional(row.get("verification_method")),
            attributes=_extra(row, _CLAIM_KNOWN),
        )
