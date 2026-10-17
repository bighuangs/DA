"""Owner/admin API for NL2SQL evaluation attribution artifacts."""
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.application.services.auth_service import Principal
from app.adapters.inbound.http.dependencies import require_roles
from evals.adapters.http.dependencies import (
    get_adhoc_isolation_evaluation_service,
    get_evaluation_service,
    get_planning_evaluation_service,
    get_quality_operations_service,
    get_rag_evaluation_service,
    get_resilience_evaluation_service,
    get_scenario_matrix_service,
    get_semantic_repair_evaluation_service,
)
from evals.application.services.adhoc_isolation_evaluation_service import AdhocIsolationEvaluationService
from evals.application.services.evaluation_service import EvaluationService
from evals.application.services.planning_evaluation_service import PlanningEvaluationService
from evals.application.services.quality_operations_service import QualityOperationsService
from evals.application.services.rag_evaluation_service import RAGEvaluationService
from evals.application.services.resilience_evaluation_service import ResilienceEvaluationService
from evals.application.services.scenario_matrix_service import ScenarioMatrixService
from evals.application.services.semantic_repair_evaluation_service import SemanticRepairEvaluationService


evaluation_router = APIRouter(prefix="/api/admin/evaluation", tags=["evaluation"])


class BaselineReviewBody(BaseModel):
    decision: str = Field(pattern="^(approve|reject)$")
    note: str = Field(default="", max_length=500)


class FeedbackReviewBody(BaseModel):
    status: str = Field(pattern="^(pending|reviewed|dismissed)$")
    category: str | None = Field(default=None, max_length=32)
    note: str = Field(default="", max_length=1000)


class FeedbackCandidateBody(BaseModel):
    candidate_type: str = Field(pattern="^(golden|training)$")
    question: str = Field(min_length=1, max_length=500)
    sql: str = Field(min_length=1, max_length=10000)


@evaluation_router.get("/scenarios/matrix")
async def latest_scenario_matrix(
    principal: Principal = Depends(require_roles("owner", "admin")),
    service: ScenarioMatrixService = Depends(get_scenario_matrix_service),
):
    return service.latest()


@evaluation_router.get("/resilience/latest")
async def latest_resilience_report(
    principal: Principal = Depends(require_roles("owner", "admin")),
    service: ResilienceEvaluationService = Depends(
        get_resilience_evaluation_service
    ),
):
    return service.latest()


@evaluation_router.get("/adhoc-isolation/latest")
async def latest_adhoc_isolation_report(
    principal: Principal = Depends(require_roles("owner", "admin")),
    service: AdhocIsolationEvaluationService = Depends(
        get_adhoc_isolation_evaluation_service
    ),
):
    return service.latest()


@evaluation_router.get("/rag/latest")
async def latest_rag_report(
    principal: Principal = Depends(require_roles("owner", "admin")),
    service: RAGEvaluationService = Depends(get_rag_evaluation_service),
):
    return service.latest(principal.workspace_id)


@evaluation_router.get("/planning/latest")
async def latest_planning_report(
    principal: Principal = Depends(require_roles("owner", "admin")),
    service: PlanningEvaluationService = Depends(get_planning_evaluation_service),
):
    return service.latest()


@evaluation_router.post("/planning/rollback")
async def rollback_planning_profile(
    principal: Principal = Depends(require_roles("owner", "admin")),
    service: PlanningEvaluationService = Depends(get_planning_evaluation_service),
):
    return service.rollback()


@evaluation_router.get("/semantic-repair/latest")
async def latest_semantic_repair_report(
    principal: Principal = Depends(require_roles("owner", "admin")),
    service: SemanticRepairEvaluationService = Depends(
        get_semantic_repair_evaluation_service
    ),
):
    return service.latest()


@evaluation_router.post("/semantic-repair/rollback")
async def rollback_semantic_repair_profile(
    principal: Principal = Depends(require_roles("owner", "admin")),
    service: SemanticRepairEvaluationService = Depends(
        get_semantic_repair_evaluation_service
    ),
):
    return service.rollback()


@evaluation_router.get("/nl2sql/runs")
async def list_nl2sql_runs(
    limit: int = Query(default=20, ge=1, le=100),
    difficulty: str | None = Query(default=None, pattern="^(easy|medium|hard)$"),
    principal: Principal = Depends(require_roles("owner", "admin")),
    service: EvaluationService = Depends(get_evaluation_service),
):
    return await service.list_runs(
        principal.workspace_id, limit, difficulty,
    )


@evaluation_router.get("/nl2sql/baselines")
async def list_nl2sql_baselines(
    principal: Principal = Depends(require_roles("owner", "admin")),
    service: EvaluationService = Depends(get_evaluation_service),
):
    return await service.list_baselines(
        principal.workspace_id,
    )


@evaluation_router.get("/nl2sql/experiments")
async def list_nl2sql_experiments(
    principal: Principal = Depends(require_roles("owner", "admin")),
    service: EvaluationService = Depends(get_evaluation_service),
):
    return await service.list_experiments(principal.workspace_id)


@evaluation_router.get("/nl2sql/latest")
async def latest_nl2sql_attribution(
    include_passed: bool = Query(default=False),
    run_id: str | None = Query(default=None),
    difficulty: str | None = Query(default=None, pattern="^(easy|medium|hard)$"),
    principal: Principal = Depends(require_roles("owner", "admin")),
    service: EvaluationService = Depends(get_evaluation_service),
):
    return service.latest(
        principal.workspace_id, include_passed, run_id, difficulty,
    )


@evaluation_router.get("/nl2sql/governance")
async def nl2sql_governance(
    run_id: str | None = Query(default=None),
    difficulty: str | None = Query(default=None, pattern="^(easy|medium|hard)$"),
    principal: Principal = Depends(require_roles("owner", "admin")),
    service: EvaluationService = Depends(get_evaluation_service),
):
    return await service.governance(
        principal.workspace_id, run_id, difficulty,
    )


@evaluation_router.post("/nl2sql/baselines/{run_id}/review")
async def review_nl2sql_baseline(
    run_id: str,
    body: BaselineReviewBody,
    principal: Principal = Depends(require_roles("owner", "admin")),
    service: EvaluationService = Depends(get_evaluation_service),
):
    try:
        return await service.review_baseline(
            principal.workspace_id,
            run_id,
            decision=body.decision,
            reviewer=f"{principal.username} ({principal.user_id})",
            note=body.note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@evaluation_router.get("/quality/overview")
async def quality_overview(
    days: int = Query(default=7, ge=1, le=90),
    version: str | None = Query(default=None, max_length=80),
    category: str | None = Query(default=None, max_length=64),
    status: str | None = Query(
        default=None,
        pattern="^(success|handled|failed|running)$",
    ),
    principal: Principal = Depends(require_roles("owner", "admin")),
    service: QualityOperationsService = Depends(get_quality_operations_service),
):
    return await service.overview(
        days=days, version=version, category=category, status=status,
    )


@evaluation_router.get("/quality/traces/{trace_id}")
async def quality_trace_detail(
    trace_id: str,
    principal: Principal = Depends(require_roles("owner", "admin")),
    service: QualityOperationsService = Depends(get_quality_operations_service),
):
    try:
        return await service.trace_detail(trace_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@evaluation_router.patch("/quality/feedback/{feedback_id}/review")
async def review_quality_feedback(
    feedback_id: int,
    body: FeedbackReviewBody,
    principal: Principal = Depends(require_roles("owner", "admin")),
    service: QualityOperationsService = Depends(get_quality_operations_service),
):
    try:
        return await service.review_feedback(
            feedback_id, status=body.status, category=body.category, note=body.note,
            reviewer_id=principal.user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@evaluation_router.post("/quality/feedback/{feedback_id}/candidate")
async def draft_quality_candidate(
    feedback_id: int,
    body: FeedbackCandidateBody,
    principal: Principal = Depends(require_roles("owner", "admin")),
    service: QualityOperationsService = Depends(get_quality_operations_service),
):
    try:
        return await service.draft_candidate(
            feedback_id, candidate_type=body.candidate_type,
            question=body.question, sql=body.sql,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@evaluation_router.post("/quality/feedback/{feedback_id}/candidate/confirm")
async def confirm_quality_candidate(
    feedback_id: int,
    principal: Principal = Depends(require_roles("owner", "admin")),
    service: QualityOperationsService = Depends(get_quality_operations_service),
):
    try:
        return await service.confirm_candidate(
            feedback_id, confirmer_id=principal.user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
