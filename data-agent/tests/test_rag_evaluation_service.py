import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evals.legacy.rag_recall import load_contract
from evals.application.services.rag_evaluation_service import RAGEvaluationService


class RAGEvaluationServiceTests(unittest.TestCase):
    @staticmethod
    def _live_run(run_id: str, workspace_id: str) -> dict:
        cases = []
        for expected in load_contract()["cases"]:
            cases.append({
                "id": expected["id"],
                "rag_case_id": expected["id"],
                "question": expected["query"],
                "stage_evidence": {
                    "retrieval": {
                        "columns": [],
                        "metrics": [],
                        "training": [],
                        "values": [],
                    },
                    "filtered_tables": [],
                    "context_columns": [],
                    "retrieval_budget": [],
                    "context_budget": {},
                },
            })
        return {
            "artifact_schema": "rag-live-v1",
            "run_id": run_id,
            "workspace_id": workspace_id,
            "valid": True,
            "n": len(cases),
            "cases": cases,
        }

    def test_latest_accepts_dedicated_rag_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rag_runs = root / "rag_runs"
            rag_runs.mkdir()

            dedicated = self._live_run("rag-dedicated", "workspace-a")
            (rag_runs / "rag_run_20260729_100000.json").write_text(
                json.dumps(dedicated), encoding="utf-8"
            )

            with (
                patch(
                    "evals.application.services.rag_evaluation_service.LATEST_PATH",
                    root / "missing-offline-report.json",
                ),
                patch(
                    "evals.application.services.rag_evaluation_service.RAG_RUNS_ROOT",
                    rag_runs,
                ),
            ):
                report = RAGEvaluationService().latest("workspace-a")

            self.assertTrue(report["live"]["available"])
            self.assertEqual(report["live"]["run_id"], "rag-dedicated")
            self.assertEqual(
                report["live"]["summary"]["evaluated_cases"],
                len(load_contract()["cases"]),
            )

    def test_latest_ignores_generic_nl2sql_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rag_runs = root / "rag_runs"
            nl2sql_runs = root / "runs"
            rag_runs.mkdir()
            nl2sql_runs.mkdir()
            generic = self._live_run("nl2sql-generic", "workspace-a")
            (nl2sql_runs / "run_20260729_110000.json").write_text(
                json.dumps(generic), encoding="utf-8"
            )

            with (
                patch(
                    "evals.application.services.rag_evaluation_service.LATEST_PATH",
                    root / "missing-offline-report.json",
                ),
                patch(
                    "evals.application.services.rag_evaluation_service.RAG_RUNS_ROOT",
                    rag_runs,
                ),
            ):
                report = RAGEvaluationService().latest("workspace-a")

            self.assertFalse(report["live"]["available"])
            self.assertNotIn("run_id", report["live"])


if __name__ == "__main__":
    unittest.main()
