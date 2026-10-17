"""正规化 RAG 评估数据集合同。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Iterable, Mapping

from evals.contracts import ContractError, validate_identifier


class DatasetValidationError(ValueError):
    """汇总数据解析或跨表完整性问题。"""

    def __init__(self, issues: Iterable[str]) -> None:
        self.issues = tuple(issues)
        if not self.issues:
            self.issues = ("未知数据集校验错误",)
        super().__init__("评估集校验失败:\n- " + "\n- ".join(self.issues))


class Answerability(StrEnum):
    """证据相对于 Query 的可回答性。"""

    ANSWERABLE = "answerable"
    PARTIALLY_ANSWERABLE = "partially_answerable"
    UNANSWERABLE = "unanswerable"
    CONFLICTING = "conflicting"
    STALE = "stale"


class ReviewStatus(StrEnum):
    """人工标注审核状态。"""

    DRAFT = "draft"
    REVIEWED = "reviewed"
    ADJUDICATED = "adjudicated"


@dataclass(frozen=True, slots=True)
class DatasetCase:
    """``cases.csv`` 中的一行。"""

    case_id: str
    dataset_id: str
    dataset_version: str
    split: str
    query: str
    answerability: Answerability
    gold_answer: str | None = None
    slice_name: str | None = None
    language: str | None = None
    domain: str | None = None
    difficulty: str | None = None
    risk: str | None = None
    kb_version: str | None = None
    parent_case_id: str | None = None
    intervention: str | None = None
    changed_variable: str | None = None
    seed: int | None = None
    attributes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_identifier("case_id", self.case_id)
        validate_identifier("dataset_id", self.dataset_id)
        validate_identifier("dataset_version", self.dataset_version)
        if self.parent_case_id is not None:
            validate_identifier("parent_case_id", self.parent_case_id)
        if not self.split.strip():
            raise ContractError("split 不能为空")
        if not self.query.strip():
            raise ContractError("query 不能为空")
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))


@dataclass(frozen=True, slots=True)
class RelevanceJudgment:
    """``relevance-judgments.csv`` 中的一行 graded qrel。"""

    case_id: str
    document_id: str
    chunk_id: str | None
    relevance_grade: int
    review_status: ReviewStatus
    evidence_start: int | None = None
    evidence_end: int | None = None
    reviewer: str | None = None
    attributes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_identifier("case_id", self.case_id)
        if not self.document_id.strip():
            raise ContractError("qrel document_id 不能为空")
        if self.chunk_id is not None and not self.chunk_id.strip():
            raise ContractError("qrel chunk_id 不能是空字符串")
        if self.relevance_grade < 0:
            raise ContractError("relevance_grade 不能为负数")
        if (self.evidence_start is None) != (self.evidence_end is None):
            raise ContractError("Evidence Span 必须同时提供 start 和 end")
        if self.evidence_start is not None:
            if self.evidence_start < 0 or self.evidence_end <= self.evidence_start:
                raise ContractError("Evidence Span 必须满足 0 <= start < end")
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))

    @property
    def evidence_key(self) -> tuple[str, str | None]:
        """返回 Document/Chunk 级的稳定 Evidence Key。"""

        return self.document_id, self.chunk_id


@dataclass(frozen=True, slots=True)
class ClaimRecord:
    """``claims.csv`` 中的一行 Gold Claim。"""

    case_id: str
    claim_id: str
    claim_text: str
    required: bool
    support_document_id: str | None = None
    support_chunk_id: str | None = None
    importance: float = 1.0
    verification_method: str | None = None
    attributes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_identifier("case_id", self.case_id)
        validate_identifier("claim_id", self.claim_id)
        if not self.claim_text.strip():
            raise ContractError("claim_text 不能为空")
        if self.support_chunk_id is not None and self.support_document_id is None:
            raise ContractError("support_chunk_id 必须同时提供 support_document_id")
        if self.importance <= 0:
            raise ContractError("importance 必须大于 0")
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))

    @property
    def support_key(self) -> tuple[str, str | None] | None:
        """返回 Claim 的 Gold Evidence Key。"""

        if self.support_document_id is None:
            return None
        return self.support_document_id, self.support_chunk_id


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """三个正规化主表的只读聚合。"""

    cases: tuple[DatasetCase, ...]
    qrels: tuple[RelevanceJudgment, ...]
    claims: tuple[ClaimRecord, ...]

    def __post_init__(self) -> None:
        issues = validate_integrity(self.cases, self.qrels, self.claims)
        if issues:
            raise DatasetValidationError(issues)

    @property
    def dataset_id(self) -> str:
        return self.cases[0].dataset_id

    @property
    def dataset_version(self) -> str:
        return self.cases[0].dataset_version

    def qrels_for(self, case_id: str) -> tuple[RelevanceJudgment, ...]:
        """按 Case 读取 qrels，保留源文件顺序。"""

        return tuple(item for item in self.qrels if item.case_id == case_id)

    def claims_for(self, case_id: str) -> tuple[ClaimRecord, ...]:
        """按 Case 读取 Gold Claims，保留源文件顺序。"""

        return tuple(item for item in self.claims if item.case_id == case_id)


def validate_integrity(
    cases: tuple[DatasetCase, ...],
    qrels: tuple[RelevanceJudgment, ...],
    claims: tuple[ClaimRecord, ...],
) -> list[str]:
    """校验三个正规化表之间的主外键和 Gold Evidence 不变量。"""

    issues: list[str] = []
    if not cases:
        return ["cases.csv 至少需要一个 Case"]

    case_by_id: dict[str, DatasetCase] = {}
    dataset_versions: set[tuple[str, str]] = set()
    for case in cases:
        if case.case_id in case_by_id:
            issues.append(f"cases.csv 存在重复 case_id: {case.case_id}")
        case_by_id[case.case_id] = case
        dataset_versions.add((case.dataset_id, case.dataset_version))
    if len(dataset_versions) != 1:
        issues.append("一个 DatasetBundle 只能包含同一 dataset_id/dataset_version")

    for case in cases:
        if case.parent_case_id is not None and case.parent_case_id not in case_by_id:
            issues.append(
                f"Case {case.case_id} 的 parent_case_id 不存在: {case.parent_case_id}"
            )
        if case.parent_case_id == case.case_id:
            issues.append(f"Case {case.case_id} 不能把自身设为 parent_case_id")

    qrel_keys: set[tuple[str, str, str | None]] = set()
    qrels_by_case: dict[str, list[RelevanceJudgment]] = {}
    for qrel in qrels:
        if qrel.case_id not in case_by_id:
            issues.append(f"qrel 引用了不存在的 case_id: {qrel.case_id}")
        key = (qrel.case_id, qrel.document_id, qrel.chunk_id)
        if key in qrel_keys:
            issues.append(
                "qrels 存在重复 Evidence Key: "
                f"case={qrel.case_id}, document={qrel.document_id}, chunk={qrel.chunk_id or '-'}"
            )
        qrel_keys.add(key)
        qrels_by_case.setdefault(qrel.case_id, []).append(qrel)

    claim_keys: set[tuple[str, str]] = set()
    required_claims_by_case: dict[str, int] = {}
    for claim in claims:
        if claim.case_id not in case_by_id:
            issues.append(f"claim 引用了不存在的 case_id: {claim.case_id}")
        claim_key = (claim.case_id, claim.claim_id)
        if claim_key in claim_keys:
            issues.append(
                f"claims.csv 存在重复 Claim: case={claim.case_id}, claim={claim.claim_id}"
            )
        claim_keys.add(claim_key)
        if claim.required:
            required_claims_by_case[claim.case_id] = (
                required_claims_by_case.get(claim.case_id, 0) + 1
            )
        support = claim.support_key
        if support is not None:
            qrel_key = (claim.case_id, support[0], support[1])
            if qrel_key not in qrel_keys:
                issues.append(
                    f"Claim {claim.claim_id} 的支持 Evidence 不存在于 qrels: "
                    f"document={support[0]}, chunk={support[1] or '-'}"
                )
            else:
                matched = next(
                    qrel
                    for qrel in qrels
                    if (qrel.case_id, qrel.document_id, qrel.chunk_id) == qrel_key
                )
                if matched.relevance_grade <= 0:
                    issues.append(
                        f"Claim {claim.claim_id} 不能由 relevance_grade=0 的 Evidence 支持"
                    )

    for case in cases:
        positive_qrels = [
            qrel
            for qrel in qrels_by_case.get(case.case_id, [])
            if qrel.relevance_grade > 0
        ]
        if case.answerability in {
            Answerability.ANSWERABLE,
            Answerability.PARTIALLY_ANSWERABLE,
        }:
            if not positive_qrels:
                issues.append(f"可回答 Case {case.case_id} 缺少正相关 qrel")
            required_claim_count = required_claims_by_case.get(case.case_id, 0)
            if required_claim_count == 0:
                # 旧检索集可能只有 Gold Evidence、没有 Gold Answer。必须显式标记
                # retrieval_only 才允许没有 Claim，避免普通端到端样本静默降级。
                if case.attributes.get("evaluation_scope") != "retrieval_only":
                    issues.append(
                        f"可回答 Case {case.case_id} 缺少 required Gold Claim；"
                        "纯检索样本必须显式设置 evaluation_scope=retrieval_only"
                    )
                if case.gold_answer is not None:
                    issues.append(
                        f"可回答 Case {case.case_id} 已声明 gold_answer，"
                        "不能标记为无 Claim 的纯检索样本"
                    )
        if case.answerability is Answerability.UNANSWERABLE and positive_qrels:
            issues.append(f"不可回答 Case {case.case_id} 不能包含正相关 qrel")

    return issues
