import copy
import datetime as dt
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evals.legacy import pipeline_ablation_compare as comparator
from evals.legacy.governance import (
    REQUIRED_FINGERPRINT_FIELDS,
    _pipeline_ablation_manifest,
    _safe_config_manifest,
    stable_hash,
)


def _fingerprint(config: dict) -> dict:
    profiles = {
        "query_signals": config["query"]["query_signals"],
        "planning_route": config["planning_route"],
        "semantic_repair": config["semantic_repair"],
    }
    fingerprint = {
        field: f"value-{field}" for field in REQUIRED_FINGERPRINT_FIELDS
    }
    fingerprint.update({
        "complete": True,
        "config_hash": stable_hash(config),
        "pipeline_profiles": copy.deepcopy(profiles),
        "pipeline_ablation": _pipeline_ablation_manifest(config),
    })
    return fingerprint


def _runs(
    experiment_id: str,
    fingerprint: dict,
    *,
    cache_states: tuple[str, str, str] = ("cold", "warm", "warm"),
) -> list[dict]:
    output = []
    for index, cache_state in enumerate(cache_states, start=1):
        output.append({
            "run_id": f"{experiment_id}-r{index}",
            "_artifact": f"run_{experiment_id}_r{index}.json",
            "_artifact_sha256": "sha256:" + "a" * 64,
            "valid": True,
            "n": 100,
            "dataset_suite": "easy_sql_v1",
            "workspace_id": "workspace-a",
            "experiment": {
                "experiment_id": experiment_id,
                "candidate_label": "D0",
                "repeat_index": index,
                "cache_state": cache_state,
            },
            "fingerprint": copy.deepcopy(fingerprint),
            "aggregate": {
                "first_result_equivalent": 84,
                "first_exec_success": 90,
                "result_equivalent": 90 + (1 if "candidate" in experiment_id else 0),
                "exec_success": 95,
                "tables_correct": 100,
                "table_scope_safe": 100,
                "llm_attempts": 210,
                "failed_core_llm_calls": 0,
                "tt_complete_p95": 10000.0,
                "latency_p95": 10000.0,
                "tokens_mean": 6400.0,
                "tokens_max": 6400,
                "estimated_cost_total": 0.0,
                "cost_estimate_available": False,
                "cost_currency": "USD",
                "llm_attempt_unknown_tokens": 0,
            },
            "cases": [
                {
                    "id": f"easy_{item:03d}",
                    "result_equivalent": (
                        item <= 90 + (1 if "candidate" in experiment_id else 0)
                    ),
                    "exec_success": item <= 95,
                    "tables_correct": True,
                    "table_scope_safe": True,
                    "first_sql_metrics": {
                        "result_equivalent": item <= 84,
                        "exec_success": item <= 90,
                    },
                    "llm_attempts": [
                        {
                            "attempt_index": attempt,
                            "token_known": True,
                            "estimated_cost": 0.0,
                        }
                        for attempt in range(3 if item <= 10 else 2)
                    ],
                    "failed_core_llm_calls": 0,
                    "total_tokens": 6400,
                    "tt_complete_ms": 10000.0,
                    "estimated_cost": 0.0,
                    "cost_currency": "USD",
                }
                for item in range(1, 101)
            ],
        })
    return output


def _recordable_report(runs: list[dict], *, experiment_id: str) -> dict:
    return {
        "available": True,
        "experiment_id": experiment_id,
        "candidate_label": "D0",
        "evaluation_profile": "core-default",
        "workspace_id": "workspace-a",
        "status": "recordable",
        "complete": True,
        "recordable": True,
        "qualified": False,
        "repeat_count": len(runs),
        "embedding_protocol_mode": "fresh",
        "failed_checks": ["absolute_result_floor"],
        "failed_observations": [],
        "fingerprint_differences": [],
        "worst": {"result_equivalent": 90},
        "stability_summary": {"stable_pass": 90, "stable_fail": 10},
    }


class PipelineAblationCompareTests(unittest.TestCase):
    def _factor_runs(self, factor: str) -> tuple[list[dict], list[dict]]:
        control_config = _safe_config_manifest()
        candidate_config = copy.deepcopy(control_config)
        if factor == "query_signals":
            candidate_config["query"]["query_signals"].update({
                "profile": "domain_signals_v1",
                "route_policy": "routed_signals_v1",
            })
        elif factor == "semantic_repair":
            candidate_config["semantic_repair"]["active_profile"] = (
                "registry_extended_v1"
            )
            candidate_config["planning_route"]["repair_profile"] = (
                "registry_extended_v1"
            )
        else:
            raise AssertionError(factor)
        return (
            _runs("control-exp", _fingerprint(control_config)),
            _runs("candidate-exp", _fingerprint(candidate_config)),
        )

    def test_query_and_repair_factors_generate_three_valid_pairs(self):
        for factor in ("query_signals", "semantic_repair"):
            with self.subTest(factor=factor):
                control, candidate = self._factor_runs(factor)
                with patch.object(
                    comparator,
                    "experiment_report",
                    side_effect=_recordable_report,
                ) as experiment_report:
                    report = comparator.build_factor_comparison(
                        control,
                        candidate,
                        factor=factor,
                        dataset_suite="easy_sql_v1",
                        control_experiment_id="control-exp",
                        candidate_experiment_id="candidate-exp",
                        generated_at=dt.datetime(
                            2026, 8, 14, 8, 0, tzinfo=dt.UTC,
                        ),
                    )

                self.assertEqual(2, experiment_report.call_count)
                self.assertTrue(report["attribution_valid"])
                self.assertEqual(3, len(report["pairs"]))
                self.assertTrue(report["report_fingerprint"].startswith("sha256:"))
                self.assertTrue(all(
                    pair["attribution"]["attribution_valid"]
                    for pair in report["pairs"]
                ))
                self.assertTrue(all(
                    pair["control_source"]["payload_fingerprint"].startswith(
                        "sha256:"
                    )
                    for pair in report["pairs"]
                ))

    def test_rejects_incomplete_or_nonrecordable_group(self):
        control, candidate = self._factor_runs("query_signals")
        invalid = _recordable_report(control, experiment_id="control-exp")
        invalid["recordable"] = False
        with (
            patch.object(
                comparator,
                "experiment_report",
                side_effect=[
                    invalid,
                    _recordable_report(
                        candidate,
                        experiment_id="candidate-exp",
                    ),
                ],
            ),
            self.assertRaisesRegex(
                comparator.ComparisonRejected,
                "control.experiment_not_recordable",
            ),
        ):
            comparator.build_factor_comparison(
                control,
                candidate,
                factor="query_signals",
                dataset_suite="easy_sql_v1",
                control_experiment_id="control-exp",
                candidate_experiment_id="candidate-exp",
            )

    def test_rejects_wrong_repeat_cache_protocol(self):
        control, candidate = self._factor_runs("query_signals")
        candidate[1]["experiment"]["cache_state"] = "cold"
        with (
            patch.object(
                comparator,
                "experiment_report",
                side_effect=_recordable_report,
            ),
            self.assertRaisesRegex(
                comparator.ComparisonRejected,
                "candidate.repeat_cache_protocol_mismatch",
            ),
        ):
            comparator.build_factor_comparison(
                control,
                candidate,
                factor="query_signals",
                dataset_suite="easy_sql_v1",
                control_experiment_id="control-exp",
                candidate_experiment_id="candidate-exp",
            )

    def test_rejects_an_extra_pipeline_variable(self):
        control, candidate = self._factor_runs("query_signals")
        candidate_config = _safe_config_manifest()
        candidate_config["query"]["query_signals"].update({
            "profile": "domain_signals_v1",
            "route_policy": "routed_signals_v1",
        })
        candidate_config["planning_route"]["active_profile"] = (
            "weighted_scorecard_v1"
        )
        bad_fingerprint = _fingerprint(candidate_config)
        for run in candidate:
            run["fingerprint"] = copy.deepcopy(bad_fingerprint)
        with (
            patch.object(
                comparator,
                "experiment_report",
                side_effect=_recordable_report,
            ),
            self.assertRaisesRegex(
                comparator.ComparisonRejected,
                r"pair\[1\].attribution_invalid",
            ),
        ):
            comparator.build_factor_comparison(
                control,
                candidate,
                factor="query_signals",
                dataset_suite="easy_sql_v1",
                control_experiment_id="control-exp",
                candidate_experiment_id="candidate-exp",
            )

    def test_rejects_aggregate_case_metric_tampering(self):
        control, candidate = self._factor_runs("query_signals")
        candidate[0]["aggregate"]["result_equivalent"] += 1
        with (
            patch.object(
                comparator,
                "experiment_report",
                side_effect=_recordable_report,
            ),
            self.assertRaisesRegex(
                comparator.ComparisonRejected,
                "run.aggregate_case_mismatch:result_equivalent",
            ),
        ):
            comparator.build_factor_comparison(
                control,
                candidate,
                factor="query_signals",
                dataset_suite="easy_sql_v1",
                control_experiment_id="control-exp",
                candidate_experiment_id="candidate-exp",
            )

    def test_rejects_latency_and_cost_aggregate_tampering(self):
        tamper_cases = (
            ("tt_complete_p95", 1.0, "tt_complete_p95"),
            ("latency_p95", 1.0, "latency_p95"),
            ("estimated_cost_total", 999999.0, "estimated_cost_total"),
            ("cost_estimate_available", True, "cost_estimate_available"),
            ("cost_currency", "EUR", "cost_currency"),
        )
        for field, value, error_field in tamper_cases:
            with self.subTest(field=field):
                control, candidate = self._factor_runs("query_signals")
                candidate[0]["aggregate"][field] = value
                with (
                    patch.object(
                        comparator,
                        "experiment_report",
                        side_effect=_recordable_report,
                    ),
                    self.assertRaisesRegex(
                        comparator.ComparisonRejected,
                        f"run.aggregate_case_mismatch:{error_field}",
                    ),
                ):
                    comparator.build_factor_comparison(
                        control,
                        candidate,
                        factor="query_signals",
                        dataset_suite="easy_sql_v1",
                        control_experiment_id="control-exp",
                        candidate_experiment_id="candidate-exp",
                    )

    def test_recomputes_available_priced_cost_from_attempts(self):
        control, candidate = self._factor_runs("query_signals")
        for run in [*control, *candidate]:
            for case in run["cases"]:
                case["llm_attempts"][0]["estimated_cost"] = 0.01
                case["estimated_cost"] = 0.01
            run["aggregate"]["estimated_cost_total"] = 1.0
            run["aggregate"]["cost_estimate_available"] = True
        with patch.object(
            comparator,
            "experiment_report",
            side_effect=_recordable_report,
        ):
            report = comparator.build_factor_comparison(
                control,
                candidate,
                factor="query_signals",
                dataset_suite="easy_sql_v1",
                control_experiment_id="control-exp",
                candidate_experiment_id="candidate-exp",
            )

        for pair in report["pairs"]:
            self.assertEqual(1.0, pair["control_metrics"]["estimated_cost_total"])
            self.assertTrue(pair["control_metrics"]["cost_estimate_available"])

    def test_loader_binds_exact_source_artifact_sha256(self):
        payload = {
            "run_id": "source-run",
            "dataset_suite": "easy_sql_v1",
            "experiment": {"experiment_id": "source-exp"},
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            runs_dir = Path(directory)
            path = runs_dir / "run_source.json"
            path.write_bytes(body)
            loaded = comparator.load_experiment_runs(
                runs_dir,
                experiment_id="source-exp",
                dataset_suite="easy_sql_v1",
            )

        self.assertEqual(1, len(loaded))
        self.assertEqual(
            "sha256:" + hashlib.sha256(body).hexdigest(),
            loaded[0]["_artifact_sha256"],
        )

    def test_rejects_missing_source_artifact_hash(self):
        control, candidate = self._factor_runs("query_signals")
        candidate[0].pop("_artifact_sha256")
        with (
            patch.object(
                comparator,
                "experiment_report",
                side_effect=_recordable_report,
            ),
            self.assertRaisesRegex(
                comparator.ComparisonRejected,
                "run.source_artifact_sha256_missing",
            ),
        ):
            comparator.build_factor_comparison(
                control,
                candidate,
                factor="query_signals",
                dataset_suite="easy_sql_v1",
                control_experiment_id="control-exp",
                candidate_experiment_id="candidate-exp",
            )

    def test_writes_only_timestamped_json_and_markdown(self):
        control, candidate = self._factor_runs("semantic_repair")
        with patch.object(
            comparator,
            "experiment_report",
            side_effect=_recordable_report,
        ):
            report = comparator.build_factor_comparison(
                control,
                candidate,
                factor="semantic_repair",
                dataset_suite="easy_sql_v1",
                control_experiment_id="control-exp",
                candidate_experiment_id="candidate-exp",
                generated_at=dt.datetime(
                    2026, 8, 14, 8, 0, 1, 123456, tzinfo=dt.UTC,
                ),
            )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            json_path, markdown_path = comparator.write_artifacts(
                report,
                output_dir=output_dir,
            )
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

            self.assertEqual(comparator.ARTIFACT_SCHEMA, payload["schema_version"])
            self.assertTrue(markdown_path.exists())
            self.assertIn("configured-price-but-zero-cost", markdown)
            self.assertIn("20260814_080001_123456", json_path.name)
            self.assertFalse(list(output_dir.glob("*latest*")))
            with self.assertRaises(FileExistsError):
                comparator.write_artifacts(report, output_dir=output_dir)


if __name__ == "__main__":
    unittest.main()
