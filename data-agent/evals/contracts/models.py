"""评测、样本、轨迹与事件的统一数据合同。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping


class ContractError(ValueError):
    """表示评测合同违反了稳定不变量。"""


_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def validate_identifier(field_name: str, value: str) -> str:
    """校验可用于关联和产物路径分段的稳定标识符。"""

    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ContractError(
            f"{field_name} 必须是 1-128 位稳定标识符，且不能包含空白或路径分隔符"
        )
    if value in {".", ".."}:
        raise ContractError(f"{field_name} 不能是路径导航分段")
    return value


def _require_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ContractError(f"{field_name} 必须包含时区")


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(value))


class FailureMode(StrEnum):
    """RAG 全系统固定的八类基础失效模式。"""

    RETRIEVAL_OMISSION = "RAG-F01"
    CONTEXT_POLLUTION = "RAG-F02"
    LOST_IN_THE_MIDDLE = "RAG-F03"
    OVER_RETRIEVAL = "RAG-F04"
    POST_RETRIEVAL_HALLUCINATION = "RAG-F05"
    FABRICATED_CITATION = "RAG-F06"
    GENERATION_LATENCY = "RAG-F07"
    EVALUATOR_ERROR = "RAG-F08"


class EventStatus(StrEnum):
    """轨迹事件状态。"""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TraceKind(StrEnum):
    """区分被评系统轨迹与评估器自身轨迹。"""

    SYSTEM = "system"
    EVALUATOR = "evaluator"


class MetricRole(StrEnum):
    """指标在决策中的职责。"""

    PRIMARY = "primary"
    DIAGNOSTIC = "diagnostic"
    GUARDRAIL = "guardrail"


class MetricStatus(StrEnum):
    """逐样本指标计算状态。"""

    COMPUTED = "computed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class EvaluationContract:
    """一次冻结评测运行的顶层身份与版本合同。"""

    evaluation_id: str
    dataset_id: str
    dataset_version: str
    system_version: str
    schema_version: str = "1.0"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_identifier("evaluation_id", self.evaluation_id)
        validate_identifier("dataset_id", self.dataset_id)
        validate_identifier("dataset_version", self.dataset_version)
        validate_identifier("system_version", self.system_version)
        validate_identifier("schema_version", self.schema_version)
        _require_aware(self.created_at, "created_at")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class CaseContract:
    """把 Dataset Case 绑定到一次评测和一条系统 Trace。"""

    evaluation_id: str
    case_id: str
    trace_id: str

    def __post_init__(self) -> None:
        validate_identifier("evaluation_id", self.evaluation_id)
        validate_identifier("case_id", self.case_id)
        validate_identifier("trace_id", self.trace_id)


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """大体积或敏感产物的安全引用，不内嵌正文。"""

    evaluation_id: str
    relative_path: str
    uri: str
    sha256: str
    size_bytes: int
    media_type: str = "application/octet-stream"

    def __post_init__(self) -> None:
        validate_identifier("evaluation_id", self.evaluation_id)
        if not self.relative_path or self.relative_path.startswith(("/", "\\")):
            raise ContractError("relative_path 必须是非空相对路径")
        if len(self.sha256) != 64 or any(ch not in "0123456789abcdef" for ch in self.sha256):
            raise ContractError("sha256 必须是 64 位小写十六进制摘要")
        if self.size_bytes < 0:
            raise ContractError("size_bytes 不能为负数")


@dataclass(frozen=True, slots=True)
class RetrievedEvidence:
    """被评系统返回的一项可排序证据。"""

    document_id: str
    chunk_id: str | None
    rank: int
    score: float | None = None

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ContractError("document_id 不能为空")
        if self.chunk_id is not None and not self.chunk_id.strip():
            raise ContractError("chunk_id 不能是空字符串")
        if self.rank < 1:
            raise ContractError("rank 必须从 1 开始")


@dataclass(frozen=True, slots=True)
class Citation:
    """答案 Claim 与实际 Evidence 的显式绑定。"""

    claim_id: str
    document_id: str
    chunk_id: str | None = None

    def __post_init__(self) -> None:
        validate_identifier("claim_id", self.claim_id)
        if not self.document_id.strip():
            raise ContractError("document_id 不能为空")


@dataclass(frozen=True, slots=True)
class TargetResponse:
    """被评系统向 Harness 返回的最小稳定响应合同。"""

    answer: str
    retrieved: tuple[RetrievedEvidence, ...] = ()
    citations: tuple[Citation, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0 or self.completion_tokens < 0:
            raise ContractError("Token 计数不能为负数")
        if self.cost < 0:
            raise ContractError("cost 不能为负数")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class EventRecord:
    """一项可观测系统或评估器事件。"""

    evaluation_id: str
    case_id: str
    trace_id: str
    event_id: str
    event_type: str
    status: EventStatus
    started_at: datetime
    ended_at: datetime | None = None
    parent_event_id: str | None = None
    input_artifacts: tuple[ArtifactRef, ...] = ()
    output_artifacts: tuple[ArtifactRef, ...] = ()
    error_type: str | None = None
    error_message: str | None = None
    retry_count: int = 0
    attributes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("evaluation_id", "case_id", "trace_id", "event_id"):
            validate_identifier(name, getattr(self, name))
        if self.parent_event_id is not None:
            validate_identifier("parent_event_id", self.parent_event_id)
            if self.parent_event_id == self.event_id:
                raise ContractError("事件不能把自身设为父事件")
        if not self.event_type.strip():
            raise ContractError("event_type 不能为空")
        _require_aware(self.started_at, "started_at")
        if self.ended_at is not None:
            _require_aware(self.ended_at, "ended_at")
            if self.ended_at < self.started_at:
                raise ContractError("ended_at 不能早于 started_at")
        if self.status is EventStatus.RUNNING and self.ended_at is not None:
            raise ContractError("运行中的事件不能设置 ended_at")
        if self.status is not EventStatus.RUNNING and self.ended_at is None:
            raise ContractError("已结束事件必须设置 ended_at")
        if self.status is EventStatus.FAILED and not self.error_type:
            raise ContractError("失败事件必须记录 error_type")
        if self.retry_count < 0:
            raise ContractError("retry_count 不能为负数")
        object.__setattr__(self, "attributes", _freeze_mapping(self.attributes))

    @property
    def duration_ms(self) -> float | None:
        """返回事件耗时；运行中的事件返回 ``None``。"""

        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds() * 1000


@dataclass(frozen=True, slots=True)
class TraceRecord:
    """一条完整且可独立校验的评测 Trace。"""

    evaluation_id: str
    case_id: str
    trace_id: str
    kind: TraceKind
    events: tuple[EventRecord, ...]
    versions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("evaluation_id", "case_id", "trace_id"):
            validate_identifier(name, getattr(self, name))

        event_ids: set[str] = set()
        for event in self.events:
            if (
                event.evaluation_id != self.evaluation_id
                or event.case_id != self.case_id
                or event.trace_id != self.trace_id
            ):
                raise ContractError("Trace 内事件的 evaluation/case/trace 关联不一致")
            if event.event_id in event_ids:
                raise ContractError(f"Trace 内存在重复 event_id: {event.event_id}")
            event_ids.add(event.event_id)

        for event in self.events:
            if event.parent_event_id is not None and event.parent_event_id not in event_ids:
                raise ContractError(
                    f"parent_event_id 未指向同一 Trace 中的事件: {event.parent_event_id}"
                )

        for key, value in self.versions.items():
            if not key.strip() or not value.strip():
                raise ContractError("versions 的键和值不能为空")
        object.__setattr__(self, "versions", MappingProxyType(dict(self.versions)))


@dataclass(frozen=True, slots=True)
class MetricResult:
    """一项逐样本、可审计的指标结果。"""

    evaluation_id: str
    case_id: str
    metric_name: str
    metric_version: str
    role: MetricRole
    status: MetricStatus
    value: float | None
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_identifier("evaluation_id", self.evaluation_id)
        validate_identifier("case_id", self.case_id)
        validate_identifier("metric_name", self.metric_name)
        validate_identifier("metric_version", self.metric_version)
        if self.status is MetricStatus.COMPUTED and self.value is None:
            raise ContractError("已计算指标必须包含 value")
        if self.status is not MetricStatus.COMPUTED and self.value is not None:
            raise ContractError("未计算指标不能伪造 value")
        object.__setattr__(self, "details", _freeze_mapping(self.details))


@dataclass(frozen=True, slots=True)
class FailureAttribution:
    """基于最早可观测分歧点的 RAG 失败归因。"""

    evaluation_id: str
    case_id: str
    primary: FailureMode
    contributing: tuple[FailureMode, ...] = ()
    earliest_event_id: str | None = None
    evidence_refs: tuple[str, ...] = ()
    confidence: float = 1.0

    def __post_init__(self) -> None:
        validate_identifier("evaluation_id", self.evaluation_id)
        validate_identifier("case_id", self.case_id)
        if self.earliest_event_id is not None:
            validate_identifier("earliest_event_id", self.earliest_event_id)
        if self.primary in self.contributing:
            raise ContractError("primary 不能同时出现在 contributing 中")
        if len(set(self.contributing)) != len(self.contributing):
            raise ContractError("contributing 不能包含重复失效模式")
        if not 0 <= self.confidence <= 1:
            raise ContractError("confidence 必须位于 [0, 1]")
