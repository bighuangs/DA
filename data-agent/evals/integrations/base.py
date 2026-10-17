"""外部 Trace 集成的状态与导出协议。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from evals.contracts import TraceRecord


class IntegrationState(StrEnum):
    """区分依赖缺失、待显式启用、可用和运行错误。"""

    DISABLED = "disabled"
    AVAILABLE = "available"
    ENABLED = "enabled"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class IntegrationStatus:
    """可序列化的外部集成状态。"""

    name: str
    state: IntegrationState
    reason: str
    missing_dependencies: tuple[str, ...] = ()

    @property
    def enabled(self) -> bool:
        return self.state is IntegrationState.ENABLED


@dataclass(frozen=True, slots=True)
class ExportResult:
    """一次 Trace 导出的显式结果。"""

    integration: str
    state: IntegrationState
    exported_events: int = 0
    reason: str = ""


class TraceExporter(Protocol):
    """Harness 可选挂载的外部 Trace Exporter。"""

    @property
    def status(self) -> IntegrationStatus:
        """只探测可用性，不触发网络或全局 Tracer 配置。"""

    def export(self, trace: TraceRecord) -> ExportResult:
        """显式导出 Trace；缺依赖时返回 disabled。"""
