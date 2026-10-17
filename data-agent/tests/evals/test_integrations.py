from unittest.mock import Mock

import pytest

from evals.contracts import TraceKind, TraceRecord
from evals.integrations import (
    IntegrationState,
    OpenInferenceTraceAdapter,
    OpenTelemetryTraceAdapter,
    PhoenixTraceAdapter,
)


def _empty_trace() -> TraceRecord:
    return TraceRecord(
        evaluation_id="eval-1",
        case_id="case-1",
        trace_id="trace-1",
        kind=TraceKind.SYSTEM,
        events=(),
    )


@pytest.mark.parametrize(
    "adapter_type",
    [OpenTelemetryTraceAdapter, OpenInferenceTraceAdapter, PhoenixTraceAdapter],
)
def test_missing_optional_dependency_is_explicitly_disabled(adapter_type: type) -> None:
    loader = Mock(side_effect=AssertionError("disabled adapter 不应导入 SDK"))
    adapter = adapter_type(
        dependency_probe=lambda _module: False,
        module_loader=loader,
    )

    status = adapter.status
    result = adapter.export(_empty_trace())

    assert status.state is IntegrationState.DISABLED
    assert status.reason == "missing_optional_dependency"
    assert status.missing_dependencies
    assert result.state is IntegrationState.DISABLED
    assert result.exported_events == 0
    loader.assert_not_called()
