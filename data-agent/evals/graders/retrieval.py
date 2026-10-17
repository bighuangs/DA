"""基于人工 qrels 的确定性 Retrieval 指标。"""

from __future__ import annotations

import math

from evals.contracts import MetricResult, MetricRole, MetricStatus, RetrievedEvidence
from evals.graders.base import GradeContext


class RetrievalGrader:
    """计算 Precision/Recall/Hit/MRR/MAP/nDCG@k。"""

    def __init__(self, *, k: int = 10) -> None:
        if k < 1:
            raise ValueError("k 必须大于 0")
        self.k = k

    @property
    def name(self) -> str:
        return "retrieval_qrels"

    @property
    def version(self) -> str:
        return "1.0"

    def grade(self, context: GradeContext) -> tuple[MetricResult, ...]:
        relevant = {
            (qrel.document_id, qrel.chunk_id): qrel.relevance_grade
            for qrel in context.qrels
            if qrel.relevance_grade > 0
        }
        if not relevant:
            return tuple(
                self._result(
                    context,
                    metric,
                    None,
                    MetricStatus.SKIPPED,
                    {"reason": "case_without_positive_qrels", "k": self.k},
                )
                for metric in self._metric_names()
            )

        retrieved = self._top_unique(context.response.retrieved)
        binary_hits = [
            1 if (item.document_id, item.chunk_id) in relevant else 0
            for item in retrieved
        ]
        grades = [
            relevant.get((item.document_id, item.chunk_id), 0) for item in retrieved
        ]
        hit_count = sum(binary_hits)
        precision = hit_count / self.k
        recall = hit_count / len(relevant)
        hit = 1.0 if hit_count else 0.0
        first_hit = next((index for index, value in enumerate(binary_hits, 1) if value), None)
        reciprocal_rank = 0.0 if first_hit is None else 1.0 / first_hit

        cumulative_hits = 0
        precision_sum = 0.0
        for index, is_hit in enumerate(binary_hits, 1):
            if is_hit:
                cumulative_hits += 1
                precision_sum += cumulative_hits / index
        average_precision = precision_sum / len(relevant)

        dcg = self._dcg(grades)
        ideal_grades = sorted(relevant.values(), reverse=True)[: self.k]
        ideal_dcg = self._dcg(ideal_grades)
        ndcg = dcg / ideal_dcg if ideal_dcg else 0.0

        values = {
            f"precision_at_{self.k}": precision,
            f"recall_at_{self.k}": recall,
            f"hit_at_{self.k}": hit,
            f"mrr_at_{self.k}": reciprocal_rank,
            f"map_at_{self.k}": average_precision,
            f"ndcg_at_{self.k}": ndcg,
        }
        details = {
            "k": self.k,
            "retrieved_count": len(retrieved),
            "relevant_count": len(relevant),
            "hit_count": hit_count,
        }
        return tuple(
            self._result(
                context, metric, value, MetricStatus.COMPUTED, details
            )
            for metric, value in values.items()
        )

    def _metric_names(self) -> tuple[str, ...]:
        return tuple(
            f"{name}_at_{self.k}"
            for name in ("precision", "recall", "hit", "mrr", "map", "ndcg")
        )

    def _top_unique(
        self, retrieved: tuple[RetrievedEvidence, ...]
    ) -> tuple[RetrievedEvidence, ...]:
        seen: set[tuple[str, str | None]] = set()
        result: list[RetrievedEvidence] = []
        for item in sorted(retrieved, key=lambda candidate: candidate.rank):
            key = item.document_id, item.chunk_id
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
            if len(result) == self.k:
                break
        return tuple(result)

    @staticmethod
    def _dcg(grades: list[int]) -> float:
        return sum(
            (2**grade - 1) / math.log2(index + 1)
            for index, grade in enumerate(grades, 1)
        )

    def _result(
        self,
        context: GradeContext,
        metric_name: str,
        value: float | None,
        status: MetricStatus,
        details: dict[str, object],
    ) -> MetricResult:
        return MetricResult(
            evaluation_id=context.evaluation_id,
            case_id=context.case.case_id,
            metric_name=metric_name,
            metric_version=self.version,
            role=MetricRole.DIAGNOSTIC,
            status=status,
            value=value,
            details=details,
        )
