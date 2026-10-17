"""保留失败行、评估器错误和外部导出状态的评测执行器。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol, Sequence

from evals.contracts import (
    EvaluationContract,
    EventRecord,
    EventStatus,
    FailureAttribution,
    FailureMode,
    MetricResult,
    MetricRole,
    MetricStatus,
    TargetResponse,
    TraceKind,
    TraceRecord,
)
from evals.datasets import DatasetBundle, DatasetCase
from evals.graders import GradeContext, Grader
from evals.integrations import ExportResult, TraceExporter
from evals.tracing import InMemoryTraceRecorder, TraceSink


class CaseRunStatus(StrEnum):
    """逐 Case 执行状态。"""

    SUCCEEDED = "succeeded"
    TARGET_ERROR = "target_error"
    EVALUATOR_ERROR = "evaluator_error"


class EvaluationTarget(Protocol):
    """被评系统适配器；生产系统只需实现此端口。"""

    async def evaluate(self, case: DatasetCase, trace: TraceSink) -> TargetResponse:
        """执行单个冻结 Case，并写入可观测事件。"""


@dataclass(frozen=True, slots=True)
class CaseRunResult:
    """一个 Case 的原始响应、分项指标、Trace 与错误。"""

    case_id: str
    status: CaseRunStatus
    trace: TraceRecord
    response: TargetResponse | None
    metrics: tuple[MetricResult, ...]
    failure: FailureAttribution | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationRunResult:
    """一次评测运行的完整内存结果。"""

    contract: EvaluationContract
    started_at: datetime
    ended_at: datetime
    cases: tuple[CaseRunResult, ...]
    exports: tuple[ExportResult, ...] = ()


class EvaluationHarness:
    """顺序执行冻结数据集，确保每个 Case 都产生结果行。"""

    def __init__(
        self,
        *,
        graders: Sequence[Grader],
        exporters: Sequence[TraceExporter] = (),
    ) -> None:
        self._graders = tuple(graders)
        self._exporters = tuple(exporters)

    async def run(
        self,
        *,
        contract: EvaluationContract,
        dataset: DatasetBundle,
        target: EvaluationTarget,
    ) -> EvaluationRunResult:
        """执行评测；Target、Grader 和 Exporter 失败均不会静默丢行。"""

        if (dataset.dataset_id, dataset.dataset_version) != (
            contract.dataset_id,
            contract.dataset_version,
        ):
            raise ValueError("EvaluationContract 与 DatasetBundle 版本不一致")

        started_at = datetime.now(UTC)
        results: list[CaseRunResult] = []
        exports: list[ExportResult] = []
        for case in dataset.cases:
            result = await self._run_case(contract, dataset, case, target)
            results.append(result)
            for exporter in self._exporters:
                exports.append(exporter.export(result.trace))
        return EvaluationRunResult(
            contract=contract,
            started_at=started_at,
            ended_at=datetime.now(UTC),
            cases=tuple(results),
            exports=tuple(exports),
        )

    async def _run_case(
        self,
        contract: EvaluationContract,
        dataset: DatasetBundle,
        case: DatasetCase,
        target: EvaluationTarget,
    ) -> CaseRunResult:
        trace_id = f"trace-{uuid.uuid4().hex}"
        root_event_id = f"target-{uuid.uuid4().hex}"
        recorder = InMemoryTraceRecorder(
            evaluation_id=contract.evaluation_id,
            case_id=case.case_id,
            trace_id=trace_id,
        )
        call_started_at = datetime.now(UTC)
        try:
            response = await target.evaluate(case, recorder)
        except Exception as exc:  # 被评系统错误必须保留为结果行。
            ended_at = datetime.now(UTC)
            recorder.emit(
                EventRecord(
                    evaluation_id=contract.evaluation_id,
                    case_id=case.case_id,
                    trace_id=trace_id,
                    event_id=root_event_id,
                    event_type="target.evaluate",
                    status=EventStatus.FAILED,
                    started_at=call_started_at,
                    ended_at=ended_at,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            trace = self._snapshot(contract, recorder)
            return CaseRunResult(
                case_id=case.case_id,
                status=CaseRunStatus.TARGET_ERROR,
                trace=trace,
                response=None,
                metrics=(),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

        recorder.emit(
            EventRecord(
                evaluation_id=contract.evaluation_id,
                case_id=case.case_id,
                trace_id=trace_id,
                event_id=root_event_id,
                event_type="target.evaluate",
                status=EventStatus.SUCCEEDED,
                started_at=call_started_at,
                ended_at=datetime.now(UTC),
                attributes={
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "cost": response.cost,
                },
            )
        )

        grade_context = GradeContext(
            evaluation_id=contract.evaluation_id,
            case=case,
            response=response,
            qrels=dataset.qrels_for(case.case_id),
            claims=dataset.claims_for(case.case_id),
        )
        metrics: list[MetricResult] = []
        evaluator_errors: list[tuple[str, Exception]] = []
        for grader in self._graders:
            try:
                metrics.extend(grader.grade(grade_context))
            except Exception as exc:  # 评分器失败不能被当作系统质量失败。
                evaluator_errors.append((grader.name, exc))
                metrics.append(
                    MetricResult(
                        evaluation_id=contract.evaluation_id,
                        case_id=case.case_id,
                        metric_name=grader.name,
                        metric_version=grader.version,
                        role=MetricRole.DIAGNOSTIC,
                        status=MetricStatus.ERROR,
                        value=None,
                        details={"error_type": type(exc).__name__},
                    )
                )

        trace = self._snapshot(contract, recorder)
        if evaluator_errors:
            first_name, first_error = evaluator_errors[0]
            return CaseRunResult(
                case_id=case.case_id,
                status=CaseRunStatus.EVALUATOR_ERROR,
                trace=trace,
                response=response,
                metrics=tuple(metrics),
                failure=FailureAttribution(
                    evaluation_id=contract.evaluation_id,
                    case_id=case.case_id,
                    primary=FailureMode.EVALUATOR_ERROR,
                    evidence_refs=(f"grader:{first_name}",),
                ),
                error_type=type(first_error).__name__,
                error_message=str(first_error),
            )
        return CaseRunResult(
            case_id=case.case_id,
            status=CaseRunStatus.SUCCEEDED,
            trace=trace,
            response=response,
            metrics=tuple(metrics),
        )

    @staticmethod
    def _snapshot(
        contract: EvaluationContract,
        recorder: InMemoryTraceRecorder,
    ) -> TraceRecord:
        return recorder.snapshot(
            kind=TraceKind.SYSTEM,
            versions={
                "dataset": f"{contract.dataset_id}:{contract.dataset_version}",
                "system": contract.system_version,
                "contract": contract.schema_version,
            },
        )
