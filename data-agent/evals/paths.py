"""评测代码、数据集与运行产物的统一路径合同。"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALS_ROOT = PROJECT_ROOT / "evals"

DATASETS_ROOT = EVALS_ROOT / "datasets"
LEGACY_DATASETS_ROOT = DATASETS_ROOT / "legacy"
RAG_V1_DATASET_ROOT = DATASETS_ROOT / "rag_v1"
CONFIGS_ROOT = EVALS_ROOT / "configs"
BASELINES_ROOT = EVALS_ROOT / "baselines"

RUNTIME_ROOT = PROJECT_ROOT / "var" / "evals" / "legacy"
REPORTS_ROOT = RUNTIME_ROOT / "reports"
RUNS_ROOT = RUNTIME_ROOT / "runs"
RAG_RUNS_ROOT = RUNTIME_ROOT / "rag_runs"
PLANNING_RUNS_ROOT = RUNTIME_ROOT / "planning_runs"
MODEL_RUNS_ROOT = RUNTIME_ROOT / "model_runs"
SEMANTIC_REPAIR_RUNS_ROOT = RUNTIME_ROOT / "semantic_repair_runs"
