"""依据可观测证据执行固定的 RAG-F01 至 RAG-F08 归因。"""

from __future__ import annotations

from dataclasses import dataclass

from evals.contracts import FailureAttribution, FailureMode


@dataclass(frozen=True, slots=True)
class FailureSignals:
    """由指标、Trace 或人工复核产生的结构化失效信号。"""

    evaluator_mismatch: bool = False
    retrieval_omission: bool = False
    context_pollution: bool = False
    lost_in_the_middle: bool = False
    over_retrieval: bool = False
    unsupported_claim: bool = False
    fabricated_citation: bool = False
    generation_latency: bool = False
    earliest_event_id: str | None = None
    evidence_refs: tuple[str, ...] = ()


class FailureAttributor:
    """按最早分歧诊断顺序确定一个 Primary 和若干贡献模式。"""

    _ORDER = (
        ("evaluator_mismatch", FailureMode.EVALUATOR_ERROR),
        ("retrieval_omission", FailureMode.RETRIEVAL_OMISSION),
        ("context_pollution", FailureMode.CONTEXT_POLLUTION),
        ("over_retrieval", FailureMode.OVER_RETRIEVAL),
        ("lost_in_the_middle", FailureMode.LOST_IN_THE_MIDDLE),
        ("unsupported_claim", FailureMode.POST_RETRIEVAL_HALLUCINATION),
        ("fabricated_citation", FailureMode.FABRICATED_CITATION),
        ("generation_latency", FailureMode.GENERATION_LATENCY),
    )

    def attribute(
        self,
        *,
        evaluation_id: str,
        case_id: str,
        signals: FailureSignals,
        confidence: float = 1.0,
    ) -> FailureAttribution | None:
        """无失效信号时返回 ``None``，避免伪造归因。"""

        matches = [
            mode for field_name, mode in self._ORDER if getattr(signals, field_name)
        ]
        if not matches:
            return None
        return FailureAttribution(
            evaluation_id=evaluation_id,
            case_id=case_id,
            primary=matches[0],
            contributing=tuple(matches[1:]),
            earliest_event_id=signals.earliest_event_id,
            evidence_refs=signals.evidence_refs,
            confidence=confidence,
        )
