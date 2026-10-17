"""把逐样本结果渲染为不掩盖 Guardrail 的 Markdown 摘要。"""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import fmean
from typing import Protocol

from evals.contracts import MetricStatus
from evals.harness import EvaluationRunResult


class ReportRenderer(Protocol):
    """耐久报告渲染协议。"""

    def render(self, result: EvaluationRunResult) -> str:
        """返回可审查文本，不自行写文件或发布。"""


class MarkdownReportRenderer:
    """渲染运行状态、分项均值、失败归因和集成状态。"""

    def render(self, result: EvaluationRunResult) -> str:
        """只聚合同名指标，不创建无依据的加权总分。"""

        status_counts = Counter(case.status.value for case in result.cases)
        failure_counts = Counter(
            case.failure.primary.value for case in result.cases if case.failure is not None
        )
        metric_values: dict[str, list[float]] = defaultdict(list)
        for case in result.cases:
            for metric in case.metrics:
                if metric.status is MetricStatus.COMPUTED and metric.value is not None:
                    metric_values[metric.metric_name].append(metric.value)

        lines = [
            f"# RAG Evaluation {result.contract.evaluation_id}",
            "",
            "## Confirmed",
            "",
            f"- Dataset: `{result.contract.dataset_id}:{result.contract.dataset_version}`",
            f"- System: `{result.contract.system_version}`",
            f"- Cases: {len(result.cases)}",
            f"- Started: `{result.started_at.isoformat()}`",
            f"- Ended: `{result.ended_at.isoformat()}`",
            "",
            "## Case status",
            "",
        ]
        lines.extend(
            f"- `{status}`: {count}" for status, count in sorted(status_counts.items())
        )
        lines.extend(["", "## Metric results", ""])
        if metric_values:
            lines.extend(
                f"- `{name}`: mean={fmean(values):.6f}, n={len(values)}"
                for name, values in sorted(metric_values.items())
            )
        else:
            lines.append("- No computed metrics")

        lines.extend(["", "## RAG failure attribution", ""])
        if failure_counts:
            lines.extend(
                f"- `{mode}`: {count}" for mode, count in sorted(failure_counts.items())
            )
        else:
            lines.append("- No attributed failures")

        lines.extend(["", "## Optional integrations", ""])
        if result.exports:
            for export in result.exports:
                lines.append(
                    f"- `{export.integration}`: {export.state.value}; "
                    f"events={export.exported_events}; reason={export.reason}"
                )
        else:
            lines.append("- No external exporter configured")

        lines.extend(
            [
                "",
                "## Unknown / not executed",
                "",
                "- Judge calibration and live production sampling are outside this run unless explicitly recorded.",
                "- This report does not infer release readiness from a weighted total score.",
                "",
            ]
        )
        return "\n".join(lines)
