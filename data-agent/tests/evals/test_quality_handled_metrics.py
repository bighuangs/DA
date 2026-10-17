from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.adapters.inbound.http.dependencies import get_current_principal
from app.application.services.auth_service import Principal
from evals.adapters.http.dependencies import get_quality_operations_service
from evals.adapters.http.router import evaluation_router
from evals.application.dto.quality_operations import StepRecord, TraceRecord
from evals.application.services.quality_operations_service import (
    QualityOperationsService,
)


def _trace(
    trace_id: str,
    status: str,
    *,
    failure_code: str = "none",
    cache_hit: bool = False,
) -> TraceRecord:
    return TraceRecord(
        id=trace_id,
        chat_id=f"chat-{trace_id}",
        query=f"query-{trace_id}",
        status=status,
        duration_ms=100,
        tt_complete_ms=100,
        total_tokens=10,
        estimated_cost=0.001,
        run_fingerprint="version-a",
        version_run_id="run-a",
        version_complete=True,
        cache_hit=cache_hit,
        failure_code=failure_code,
        query_category="intent" if status == "handled" else "metric",
        started_at=datetime(2026, 8, 17, 12, 0),
    )


def test_handled_is_separate_expected_terminal_and_not_pipeline_failure() -> None:
    traces = [
        _trace("success", "success", cache_hit=True),
        _trace("handled", "handled", cache_hit=True),
        _trace("failed", "failed", failure_code="sql_validation"),
        _trace("running", "running", cache_hit=True),
    ]
    steps = [
        StepRecord(
            trace_id="success",
            step_name="correct_sql",
            status="success",
        ),
        # 即使异常输入带有遗留 SQL Step，handled 也不进入 SQL 指标分子。
        StepRecord(
            trace_id="handled",
            step_name="correct_sql",
            status="success",
        ),
    ]

    result = QualityOperationsService(None, "workspace-a").aggregate(
        traces,
        steps,
        [],
        [],
        days=7,
        filters={},
    )

    summary = result["summary"]
    assert summary["traces"] == 4
    assert summary["terminal_traces"] == 3
    assert summary["sql_terminal_traces"] == 2
    assert summary["successful_traces"] == 1
    assert summary["expected_terminal_traces"] == 1
    assert summary["failed_traces"] == 1
    assert summary["running_traces"] == 1
    assert summary["success_rate"] == 0.5
    assert summary["expected_terminal_rate"] == 0.3333
    assert summary["result_proxy_rate"] == 0.5
    assert summary["correction_rate"] == 0.5
    assert summary["cache_hit_rate"] == 0.5
    assert summary["rate_denominators"] == {
        "success_rate": 2,
        "expected_terminal_rate": 3,
        "result_proxy_rate": 2,
        "negative_rate": 0,
        "correction_rate": 2,
        "cache_hit_rate": 2,
        "low_confidence_rate": 2,
        "second_candidate_rate": 2,
        "adaptive_repair_rate": 0,
    }
    assert result["failure_funnel"] == [
        {"failure_code": "sql_validation", "count": 1}
    ]
    assert all(
        row["failure_code"] != "pipeline_unknown"
        for row in result["failure_funnel"]
    )

    version = result["versions"][0]
    assert version["traces"] == 4
    assert version["sql_terminal_traces"] == 2
    assert version["expected_terminal_traces"] == 1
    assert version["success_rate"] == 0.5
    assert version["expected_terminal_rate"] == 0.3333

    trend = result["trends"][0]
    assert trend["traces"] == 4
    assert trend["sql_terminal_traces"] == 2
    assert trend["expected_terminal_traces"] == 1
    assert trend["success_rate"] == 0.5
    assert trend["expected_terminal_rate"] == 0.3333


def test_handled_only_population_has_no_sql_success_or_failure_denominator() -> None:
    result = QualityOperationsService(None, "workspace-a").aggregate(
        [_trace("handled", "handled")],
        [],
        [],
        [],
        days=7,
        filters={"status": "handled"},
    )

    summary = result["summary"]
    assert summary["terminal_traces"] == 1
    assert summary["sql_terminal_traces"] == 0
    assert summary["successful_traces"] == 0
    assert summary["expected_terminal_traces"] == 1
    assert summary["failed_traces"] == 0
    assert summary["success_rate"] == 0.0
    assert summary["expected_terminal_rate"] == 1.0
    assert summary["rate_denominators"]["success_rate"] == 0
    assert summary["rate_denominators"]["expected_terminal_rate"] == 1
    assert result["failure_funnel"] == []
    assert result["versions"][0]["success_rate"] == 0.0
    assert result["versions"][0]["expected_terminal_rate"] == 1.0
    assert result["trends"][0]["success_rate"] == 0.0
    assert result["trends"][0]["expected_terminal_rate"] == 1.0


class _QualityUseCase:
    def __init__(self) -> None:
        self.statuses: list[str | None] = []

    async def overview(
        self,
        *,
        days: int,
        version: str | None,
        category: str | None,
        status: str | None,
    ) -> dict[str, object]:
        self.statuses.append(status)
        return {"status": status, "days": days}


def _test_app(service: _QualityUseCase) -> FastAPI:
    app = FastAPI()
    app.include_router(evaluation_router)
    app.dependency_overrides[get_current_principal] = lambda: Principal(
        user_id="owner-a",
        username="owner",
        display_name="Owner",
        workspace_id="workspace-a",
        workspace_name="Workspace A",
        role="owner",
    )
    app.dependency_overrides[get_quality_operations_service] = lambda: service
    return app


def test_quality_overview_http_status_filter_accepts_handled() -> None:
    service = _QualityUseCase()
    with TestClient(_test_app(service)) as client:
        response = client.get(
            "/api/admin/evaluation/quality/overview",
            params={"status": "handled"},
        )
        invalid = client.get(
            "/api/admin/evaluation/quality/overview",
            params={"status": "unexpected"},
        )

    assert response.status_code == 200
    assert response.json() == {"status": "handled", "days": 7}
    assert service.statuses == ["handled"]
    assert invalid.status_code == 422
