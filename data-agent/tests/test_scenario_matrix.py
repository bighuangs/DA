import copy
import unittest

from evals.legacy.scenario_matrix import load_matrix, report_from_suite_status, report_from_target_status, validate_matrix
from evals.legacy.run_scenarios import _parse_unittest_statuses


class ScenarioMatrixTests(unittest.TestCase):
    def test_matrix_meets_pairwise_coverage_and_evidence_contract(self):
        report = validate_matrix()
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["total"], 56)
        self.assertEqual(set(report["capabilities"].values()), {7})
        self.assertEqual(set(report["roles"]), {"owner", "admin", "analyst", "viewer", "removed"})
        self.assertEqual(set(report["failure_layers"]), {"none", "identity", "resource", "policy", "sql", "dependency", "presentation"})

    def test_missing_automation_target_fails_contract(self):
        cases = copy.deepcopy(load_matrix())
        cases[0]["evidence"]["target"] = "tests.missing.Test.test_missing"
        report = validate_matrix(cases)
        self.assertFalse(report["valid"])
        self.assertTrue(any("证据不存在" in error for error in report["errors"]))

    def test_quality_gate_never_marks_unexecuted_suite_as_passed(self):
        report = report_from_suite_status(unit_passed=True, e2e_passed=False)
        self.assertFalse(report["summary"]["gate_passed"])
        self.assertEqual(report["summary"]["passed"], 34)
        self.assertEqual(report["summary"]["failed"], 22)

    def test_exact_target_failure_only_marks_bound_scenarios(self):
        cases = load_matrix()
        unit_targets = {case["evidence"]["target"] for case in cases if case["evidence"]["type"] == "unit"}
        e2e_targets = {case["evidence"]["target"] for case in cases if case["evidence"]["type"] == "e2e"}
        unit_status = {target: {"status": "passed"} for target in unit_targets}
        e2e_status = {target: {"status": "passed"} for target in e2e_targets}
        failed_target = next(iter(unit_targets))
        unit_status[failed_target] = {"status": "failed", "error": "assertion"}
        report = report_from_target_status(unit_status=unit_status, e2e_status=e2e_status)
        expected_failed = sum(
            case["evidence"]["type"] == "unit" and case["evidence"]["target"] == failed_target
            for case in cases
        )
        self.assertEqual(report["summary"]["failed"], expected_failed)
        self.assertEqual(report["summary"]["passed"], len(cases) - expected_failed)

    def test_verbose_unittest_parser_handles_asyncio_diagnostic_line(self):
        output = (
            "test_async (tests.test_sample.SampleTests.test_async) ... Executing <Task finished> took 0.2 seconds\n"
            "ok\n"
            "test_sync (tests.test_sample.SampleTests.test_sync) ... ok\n"
        )
        self.assertEqual(_parse_unittest_statuses(output), {
            "tests.test_sample.SampleTests.test_async": "ok",
            "tests.test_sample.SampleTests.test_sync": "ok",
        })


if __name__ == "__main__":
    unittest.main()
