"""评分器的稳定扩展边界。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from evals.contracts import MetricResult, TargetResponse
from evals.datasets import ClaimRecord, DatasetCase, RelevanceJudgment


@dataclass(frozen=True, slots=True)
class GradeContext:
    """评分器所需的最小逐样本上下文。"""

    evaluation_id: str
    case: DatasetCase
    response: TargetResponse
    qrels: tuple[RelevanceJudgment, ...]
    claims: tuple[ClaimRecord, ...]


class Grader(Protocol):
    """确定性评分器或已校准 Judge Adapter 的统一协议。"""

    @property
    def name(self) -> str:
        """返回稳定评分器名称。"""

    @property
    def version(self) -> str:
        """返回评分算法或 Rubric 版本。"""

    def grade(self, context: GradeContext) -> tuple[MetricResult, ...]:
        """计算逐样本原始分项，不生成不可解释总分。"""
