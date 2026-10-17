"""Eval HTTP 入站适配器声明的 Application provider ports。"""

from __future__ import annotations

from typing import NoReturn

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


def _missing(name: str) -> NoReturn:
    raise RuntimeError(f"Eval HTTP provider 未由 composition root 装配：{name}")


def get_evaluation_service() -> EvaluationService:
    return _missing("get_evaluation_service")


def get_quality_operations_service() -> QualityOperationsService:
    return _missing("get_quality_operations_service")


def get_scenario_matrix_service() -> ScenarioMatrixService:
    return _missing("get_scenario_matrix_service")


def get_resilience_evaluation_service() -> ResilienceEvaluationService:
    return _missing("get_resilience_evaluation_service")


def get_rag_evaluation_service() -> RAGEvaluationService:
    return _missing("get_rag_evaluation_service")


def get_planning_evaluation_service() -> PlanningEvaluationService:
    return _missing("get_planning_evaluation_service")


def get_semantic_repair_evaluation_service() -> SemanticRepairEvaluationService:
    return _missing("get_semantic_repair_evaluation_service")


def get_adhoc_isolation_evaluation_service() -> AdhocIsolationEvaluationService:
    return _missing("get_adhoc_isolation_evaluation_service")
