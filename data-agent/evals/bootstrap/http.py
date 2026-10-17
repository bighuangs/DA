"""Eval HTTP 的显式组合根；仅在功能开关启用后导入。"""

from __future__ import annotations

from fastapi import Depends, FastAPI
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.inbound.http.dependencies import get_meta_session, require_roles
from app.application.services.auth_service import Principal
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
from evals.adapters.http.router import evaluation_router
from evals.adapters.outbound.runtime_profiles import (
    PlanningRuntimeProfileControl,
    SemanticRepairRuntimeProfileControl,
)
from evals.application.services.adhoc_isolation_evaluation_service import (
    AdhocIsolationEvaluationService,
)
from evals.application.services.evaluation_service import EvaluationService
from evals.application.services.planning_evaluation_service import (
    PlanningEvaluationService,
)
from evals.application.services.quality_operations_service import (
    QualityOperationsService,
)
from evals.application.services.rag_evaluation_service import RAGEvaluationService
from evals.application.services.resilience_evaluation_service import (
    ResilienceEvaluationService,
)
from evals.application.services.scenario_matrix_service import ScenarioMatrixService
from evals.application.services.semantic_repair_evaluation_service import (
    SemanticRepairEvaluationService,
)
from evals.infrastructure.persistence.mysql.evaluation_baseline_repository import (
    EvaluationBaselineRepository,
)
from evals.infrastructure.persistence.mysql.quality_operations_store import (
    MySQLQualityOperationsStore,
)


_evaluation_admin = require_roles("owner", "admin")


def provide_evaluation_service(
    session: AsyncSession = Depends(get_meta_session),
) -> EvaluationService:
    return EvaluationService(registry=EvaluationBaselineRepository(session))


def provide_quality_operations_service(
    session: AsyncSession = Depends(get_meta_session),
    principal: Principal = Depends(_evaluation_admin),
) -> QualityOperationsService:
    catalog = EvaluationService(registry=EvaluationBaselineRepository(session))
    store = MySQLQualityOperationsStore(session, principal.workspace_id)
    return QualityOperationsService(
        store,
        principal.workspace_id,
        evaluation_catalog=catalog,
    )


def provide_scenario_matrix_service() -> ScenarioMatrixService:
    return ScenarioMatrixService()


def provide_resilience_evaluation_service() -> ResilienceEvaluationService:
    return ResilienceEvaluationService()


def provide_rag_evaluation_service() -> RAGEvaluationService:
    return RAGEvaluationService()


def provide_planning_evaluation_service() -> PlanningEvaluationService:
    return PlanningEvaluationService(PlanningRuntimeProfileControl())


def provide_semantic_repair_evaluation_service(
) -> SemanticRepairEvaluationService:
    return SemanticRepairEvaluationService(SemanticRepairRuntimeProfileControl())


def provide_adhoc_isolation_evaluation_service(
) -> AdhocIsolationEvaluationService:
    return AdhocIsolationEvaluationService()


def mount_evaluation_http(app: FastAPI) -> None:
    """绑定抽象 provider 与具体实现，再挂载公开 Router。"""
    app.dependency_overrides.update(
        {
            get_evaluation_service: provide_evaluation_service,
            get_quality_operations_service: provide_quality_operations_service,
            get_scenario_matrix_service: provide_scenario_matrix_service,
            get_resilience_evaluation_service: provide_resilience_evaluation_service,
            get_rag_evaluation_service: provide_rag_evaluation_service,
            get_planning_evaluation_service: provide_planning_evaluation_service,
            get_semantic_repair_evaluation_service: (
                provide_semantic_repair_evaluation_service
            ),
            get_adhoc_isolation_evaluation_service: (
                provide_adhoc_isolation_evaluation_service
            ),
        }
    )
    app.include_router(evaluation_router)
