import asyncio
import json
import unittest

from evals.legacy.adhoc_isolation_suite import (
    ExactResourceInventory,
    build_sandbox_manifest,
    public_sandbox_manifest,
    run_suite,
    validate_sandbox_manifest,
    validate_suite,
)
from evals.legacy.run_quality import check_adhoc_isolation


class AdhocIsolationContractTests(unittest.TestCase):
    def test_contract_has_separate_facet_nine_fixtures_and_21_cases(self):
        contract = validate_suite()
        self.assertTrue(contract["valid"], contract["errors"])
        self.assertEqual(contract["facet"], "adhoc_isolation")
        self.assertEqual(contract["fixtures"], 9)
        self.assertEqual(contract["total"], 21)
        self.assertTrue(all(count >= 3 for count in contract["categories"].values()))

    def test_every_fixture_has_complete_synthetic_provenance_and_hashes(self):
        contract = validate_suite()
        for fixture in contract["fixture_contracts"]:
            self.assertIn(
                fixture["origin"],
                {"synthetic_adhoc_contract", "synthetic_negative_payload"},
            )
            self.assertEqual(fixture["generator_version"], "da-synthetic-upload-v1")
            self.assertTrue(fixture["reviewer"])
            self.assertTrue(fixture["logical_content_hash"].startswith("sha256:"))
            self.assertEqual(set(fixture["format_hashes"]), {"csv", "xlsx"})

    def test_offline_suite_preserves_denominator_and_zero_leakage(self):
        report = asyncio.run(
            run_suite(
                evidence_level="offline_contract",
                write=False,
                run_id="unit-adhoc-isolation",
            )
        )
        self.assertTrue(report["summary"]["gate_passed"])
        self.assertEqual(report["summary"]["passed"], 21)
        self.assertEqual(report["denominator"]["attempted"], 21)
        self.assertEqual(report["denominator"]["attempt_records"], 21)
        self.assertEqual(report["denominator"]["trace_records"], 21)
        self.assertTrue(report["metrics"]["denominator_consistency"])
        self.assertEqual(report["metrics"]["unauthorized_db_execution_count"], 0)
        self.assertEqual(report["metrics"]["schema_index_pollution_count"], 0)
        self.assertEqual(report["metrics"]["sandbox_residual_resources"], 0)
        self.assertEqual(report["metrics"]["real_upload_content_occurrences"], 0)

    def test_report_never_contains_raw_rows_filename_or_physical_table(self):
        report = asyncio.run(
            run_suite(
                evidence_level="offline_contract",
                write=False,
                run_id="unit-report-redaction",
            )
        )
        encoded = json.dumps(report, ensure_ascii=False).lower()
        self.assertNotIn('"rows"', encoded)
        self.assertNotIn('"preview"', encoded)
        self.assertNotIn('"filename"', encoded)
        self.assertNotIn('"physical_table"', encoded)
        self.assertNotIn("=sum(1,1)", encoded)

    def test_preflight_rejects_broad_or_unbound_cleanup_target(self):
        manifest = build_sandbox_manifest(run_id="unit-preflight-check")
        manifest["resources"]["dw_database"] = "dw2"
        manifest["allowed_resources"] = list(manifest["resources"].values())
        result = validate_sandbox_manifest(manifest)
        self.assertFalse(result["valid"])
        self.assertTrue(any("隔离数据库" in error for error in result["errors"]))

    def test_cleanup_inventory_only_accepts_exact_manifest_resources(self):
        manifest = build_sandbox_manifest(run_id="unit-exact-cleanup")
        inventory = ExactResourceInventory(manifest)
        allowed = manifest["resources"]["dw_database"]
        inventory.create(allowed)
        with self.assertRaises(PermissionError):
            inventory.cleanup("eval_adhoc_*")
        inventory.cleanup(allowed)
        self.assertEqual(inventory.snapshot()["active_count"], 0)

    def test_public_manifest_only_contains_resource_hashes(self):
        manifest = build_sandbox_manifest(run_id="unit-public-manifest")
        public = public_sandbox_manifest(manifest)
        encoded = json.dumps(public)
        for value in manifest["resources"].values():
            self.assertNotIn(value, encoded)
        self.assertTrue(public["namespace_hashes"])

    def test_unified_quality_check_consumes_offline_gate(self):
        check = check_adhoc_isolation()
        self.assertEqual(check.status, "pass")
        self.assertEqual(check.score, 100)
        self.assertEqual(check.evidence["evidence_level"], "offline_contract")
        self.assertFalse(check.evidence["production_equivalent"])


if __name__ == "__main__":
    unittest.main()
