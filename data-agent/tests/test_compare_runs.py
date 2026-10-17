import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evals.legacy import compare_runs


class CompareRunsSuiteIsolationTests(unittest.TestCase):
    @staticmethod
    def _write_run(
        root: Path,
        *,
        run_id: str,
        suite: str,
        experiment_id: str = "shared",
        repeat: int = 1,
    ) -> None:
        (root / f"run_{run_id}.json").write_text(json.dumps({
            "run_id": run_id,
            "valid": True,
            "dataset_suite": suite,
            "workspace_id": "workspace-a",
            "experiment": {
                "experiment_id": experiment_id,
                "repeat_index": repeat,
            },
            "aggregate": {
                "n": 0,
                "syntax_valid": 0,
                "exec_success": 0,
                "sql_equivalent": 0,
                "tables_correct": 0,
                "result_equivalent": 0,
                "retry_triggered": 0,
                "retry_success": 0,
                "latency_mean": 0,
                "latency_p95": 0,
            },
            "cases": [],
        }), encoding="utf-8")

    def test_run_and_experiment_loaders_filter_the_exact_suite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_run(root, run_id="easy-1", suite="easy_sql_v1")
            self._write_run(root, run_id="medium-1", suite="medium_sql_v1")
            with patch.object(compare_runs, "RUNS_DIR", root):
                runs = compare_runs.load_runs("easy_sql_v1")
                experiment = compare_runs.load_experiment_runs(
                    "shared", "easy_sql_v1",
                )

            self.assertEqual([run["run_id"] for run in runs], ["easy-1"])
            self.assertEqual(
                [run["run_id"] for run in experiment], ["easy-1"],
            )

    def test_cli_writes_suite_scoped_latest_and_immutable_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            reports = root / "reports"
            runs.mkdir()
            self._write_run(runs, run_id="easy-1", suite="easy_sql_v1")
            with (
                patch.object(compare_runs, "RUNS_DIR", runs),
                patch.object(compare_runs, "COMPARE_DIR", reports),
                patch.object(
                    sys, "argv",
                    ["compare_runs", "--suite", "easy_sql_v1"],
                ),
            ):
                compare_runs.main()

            self.assertTrue(
                (reports / "compare_easy_sql_v1_latest.md").exists(),
            )
            self.assertEqual(
                len(list(reports.glob("compare_easy_sql_v1_*.md"))), 2,
            )
            self.assertFalse((reports / "compare_latest.md").exists())

    def test_cli_experiment_filename_contains_suite_and_safe_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            reports = root / "reports"
            runs.mkdir()
            with (
                patch.object(compare_runs, "RUNS_DIR", runs),
                patch.object(compare_runs, "COMPARE_DIR", reports),
                patch.object(
                    sys, "argv",
                    [
                        "compare_runs", "--suite", "medium_sql_v1",
                        "--experiment-id", "shared/id",
                    ],
                ),
                self.assertRaises(SystemExit) as raised,
            ):
                compare_runs.main()

            self.assertEqual(raised.exception.code, 2)
            self.assertTrue(
                (
                    reports
                    / "experiment_medium_sql_v1_shared-id.md"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
