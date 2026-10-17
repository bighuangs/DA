"""无 import-time 副作用的可选可观测性适配器。"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import ContextManager, Protocol, cast

from evals.contracts import EventStatus, TraceRecord
from evals.integrations.base import ExportResult, IntegrationState, IntegrationStatus


DependencyProbe = Callable[[str], bool]
ModuleLoader = Callable[[str], ModuleType]


class _Span(Protocol):
    def set_attribute(self, name: str, value: str | bool | int | float) -> object: ...


class _Tracer(Protocol):
    def start_as_current_span(self, name: str) -> ContextManager[_Span]: ...


def _default_dependency_probe(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


class _OptionalAdapter:
    """把可选依赖错误收敛为状态，而不是让评测进程启动失败。"""

    name = "optional"
    required_modules: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        dependency_probe: DependencyProbe = _default_dependency_probe,
        module_loader: ModuleLoader = importlib.import_module,
    ) -> None:
        self._dependency_probe = dependency_probe
        self._module_loader = module_loader
        self._active = False
        self._error: str | None = None

    @property
    def status(self) -> IntegrationStatus:
        missing = tuple(
            module for module in self.required_modules if not self._dependency_probe(module)
        )
        if missing:
            return IntegrationStatus(
                name=self.name,
                state=IntegrationState.DISABLED,
                reason="missing_optional_dependency",
                missing_dependencies=missing,
            )
        if self._error is not None:
            return IntegrationStatus(
                name=self.name,
                state=IntegrationState.ERROR,
                reason=self._error,
            )
        if self._active:
            return IntegrationStatus(
                name=self.name,
                state=IntegrationState.ENABLED,
                reason="enabled",
            )
        return IntegrationStatus(
            name=self.name,
            state=IntegrationState.AVAILABLE,
            reason="explicit_activation_required",
        )

    def _disabled_result(self) -> ExportResult | None:
        status = self.status
        if status.state is IntegrationState.DISABLED:
            return ExportResult(
                integration=self.name,
                state=status.state,
                reason=status.reason,
            )
        return None


class OpenTelemetryTraceAdapter(_OptionalAdapter):
    """把内部 EventRecord 映射到宿主已配置的 OpenTelemetry Tracer。"""

    name = "opentelemetry"
    required_modules = ("opentelemetry.trace", "opentelemetry.sdk.trace")

    def __init__(
        self,
        *,
        tracer_name: str = "data-agent.evals",
        dependency_probe: DependencyProbe = _default_dependency_probe,
        module_loader: ModuleLoader = importlib.import_module,
    ) -> None:
        super().__init__(
            dependency_probe=dependency_probe,
            module_loader=module_loader,
        )
        self.tracer_name = tracer_name
        self._tracer: _Tracer | None = None

    def activate(self) -> IntegrationStatus:
        """显式获取 Tracer；不会替换宿主的全局 TracerProvider。"""

        disabled = self._disabled_result()
        if disabled is not None:
            return self.status
        try:
            trace_module = self._module_loader("opentelemetry.trace")
            tracer = trace_module.get_tracer(self.tracer_name)
            if not callable(getattr(tracer, "start_as_current_span", None)):
                raise TypeError("OpenTelemetry Tracer 缺少 start_as_current_span")
            self._tracer = cast(_Tracer, tracer)
            self._active = True
        except Exception as exc:  # 外部 SDK 错误需要转成可报告状态。
            self._error = f"activation_failed:{type(exc).__name__}:{exc}"
        return self.status

    def export(self, trace: TraceRecord) -> ExportResult:
        """逐事件创建 Span，并用稳定 ID 保留内部 Trace 关联。"""

        disabled = self._disabled_result()
        if disabled is not None:
            return disabled
        if not self._active:
            self.activate()
        if self.status.state is not IntegrationState.ENABLED or self._tracer is None:
            return ExportResult(
                integration=self.name,
                state=self.status.state,
                reason=self.status.reason,
            )
        try:
            for event in trace.events:
                with self._tracer.start_as_current_span(event.event_type) as span:
                    span.set_attribute("evaluation.id", event.evaluation_id)
                    span.set_attribute("evaluation.case.id", event.case_id)
                    span.set_attribute("evaluation.trace.id", event.trace_id)
                    span.set_attribute("evaluation.event.id", event.event_id)
                    span.set_attribute("evaluation.event.status", event.status.value)
                    span.set_attribute("evaluation.trace.kind", trace.kind.value)
                    if event.parent_event_id is not None:
                        span.set_attribute(
                            "evaluation.parent_event.id", event.parent_event_id
                        )
                    if event.duration_ms is not None:
                        span.set_attribute("evaluation.duration_ms", event.duration_ms)
                    for key, value in event.attributes.items():
                        if isinstance(value, (str, bool, int, float)):
                            span.set_attribute(f"evaluation.attribute.{key}", value)
                    if event.status is EventStatus.FAILED:
                        span.set_attribute("error.type", event.error_type or "Unknown")
                        if event.error_message:
                            span.set_attribute("error.message", event.error_message)
            return ExportResult(
                integration=self.name,
                state=IntegrationState.ENABLED,
                exported_events=len(trace.events),
                reason="exported",
            )
        except Exception as exc:  # 导出失败不能丢失本地评测结果。
            self._error = f"export_failed:{type(exc).__name__}:{exc}"
            return ExportResult(
                integration=self.name,
                state=IntegrationState.ERROR,
                reason=self._error,
            )


class OpenInferenceTraceAdapter(OpenTelemetryTraceAdapter):
    """在 OTel Span 上附加 OpenInference 语义约定标记。"""

    name = "openinference"
    required_modules = (
        "opentelemetry.trace",
        "opentelemetry.sdk.trace",
        "openinference.semconv.trace",
    )

    def export(self, trace: TraceRecord) -> ExportResult:
        """惰性确认语义约定模块后复用 OTel 导出路径。"""

        disabled = self._disabled_result()
        if disabled is not None:
            return disabled
        try:
            self._module_loader("openinference.semconv.trace")
        except Exception as exc:
            self._error = f"activation_failed:{type(exc).__name__}:{exc}"
            return ExportResult(
                integration=self.name,
                state=IntegrationState.ERROR,
                reason=self._error,
            )
        return super().export(trace)


@dataclass(frozen=True, slots=True)
class PhoenixConfig:
    """Phoenix OTLP 注册所需的非敏感配置。"""

    project_name: str = "data-agent-evals"
    endpoint: str | None = None


class PhoenixTraceAdapter(OpenTelemetryTraceAdapter):
    """显式注册 Phoenix OTel Provider 后导出内部 Trace。"""

    name = "phoenix"
    required_modules = ("phoenix.otel", "opentelemetry.sdk.trace")

    def __init__(
        self,
        config: PhoenixConfig | None = None,
        *,
        dependency_probe: DependencyProbe = _default_dependency_probe,
        module_loader: ModuleLoader = importlib.import_module,
    ) -> None:
        super().__init__(
            tracer_name="data-agent.evals.phoenix",
            dependency_probe=dependency_probe,
            module_loader=module_loader,
        )
        self.config = config or PhoenixConfig()

    def activate(self) -> IntegrationStatus:
        """只有调用方显式启用时才注册 Phoenix，避免导入和网络副作用。"""

        disabled = self._disabled_result()
        if disabled is not None:
            return self.status
        try:
            phoenix_otel = self._module_loader("phoenix.otel")
            kwargs: dict[str, object] = {
                "project_name": self.config.project_name,
                "auto_instrument": False,
            }
            if self.config.endpoint is not None:
                kwargs["endpoint"] = self.config.endpoint
            phoenix_otel.register(**kwargs)
            trace_module = self._module_loader("opentelemetry.trace")
            tracer = trace_module.get_tracer(self.tracer_name)
            if not callable(getattr(tracer, "start_as_current_span", None)):
                raise TypeError("OpenTelemetry Tracer 缺少 start_as_current_span")
            self._tracer = cast(_Tracer, tracer)
            self._active = True
        except Exception as exc:
            self._error = f"activation_failed:{type(exc).__name__}:{exc}"
        return self.status
