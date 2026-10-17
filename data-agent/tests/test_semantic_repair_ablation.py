import sys
import unittest
from unittest.mock import patch

from evals.legacy import semantic_repair_ablation as ablation


class SemanticRepairAblationTests(unittest.TestCase):
    def test_reviewed_contract_and_p0_p6_gates(self):
        contract = ablation.load_contract()
        self.assertTrue(contract["valid"])
        self.assertGreaterEqual(contract["total"], 30)
        self.assertGreaterEqual(contract["splits"]["test"], 20)

        evaluate_profile = ablation.evaluate_profile

        def deterministic_profile(profile, cases):
            result = evaluate_profile(profile, cases)
            # 此测试验证规则与决策合同；真实墙钟阈值由专用性能门禁负责。
            result["repair_latency_ms"]["p95"] = 0.0
            return result

        with patch.object(
            ablation,
            "evaluate_profile",
            side_effect=deterministic_profile,
        ):
            report = ablation.build_report(write=False)
        self.assertEqual(set(report["profiles"]), {f"P{index}" for index in range(7)})
        # 墙钟 P95 由 semantic_repair_ablation / full-quality 的专用性能门禁负责。
        # 单元测试进程可能与浏览器、数据库或评测任务共享本机资源，不能把
        # 一次调度抖动误判为确定性功能回归。
        deterministic_gates = {
            name: passed
            for name, passed in report["implementation_gates"].items()
            if name not in {
                "gate_passed",
                "active_p95_under_20ms",
                "candidate_p3_p95_under_20ms",
            }
        }
        self.assertTrue(all(deterministic_gates.values()), deterministic_gates)
        self.assertGreaterEqual(report["profiles"]["P2"]["repair_latency_ms"]["p95"], 0)
        self.assertTrue(report["implementation_gates"]["p0_p2_character_compatible"])
        self.assertEqual(1, report["profiles"]["P2"]["repair_precision"])
        self.assertEqual(0, report["profiles"]["P2"]["false_trigger_count"])
        self.assertGreater(
            report["profiles"]["P3"]["targeted_rescue_rate"],
            report["profiles"]["P2"]["targeted_rescue_rate"],
        )
        p3 = report["profiles"]["P3"]
        self.assertEqual(1, p3["repair_precision"])
        self.assertEqual(1, p3["targeted_rescue_rate"])
        self.assertEqual(0, p3["false_trigger_count"])
        self.assertEqual(0, p3["harmful_repair_count"])
        self.assertEqual(1, p3["idempotence_rate"])
        self.assertEqual(1, p3["unsafe_unchanged_rate"])
        self.assertTrue(report["decision"]["offline_candidate_qualified"])
        self.assertEqual([], report["decision"]["offline_candidate_blockers"])
        self.assertFalse(report["decision"]["release_qualified"])

    def test_enforce_rejects_non_idempotent_p3_candidate(self):
        evaluate_profile = ablation.evaluate_profile

        def non_idempotent_p3(profile, cases):
            result = evaluate_profile(profile, cases)
            # 这个用例只隔离验证幂等门禁。全套测试共享主机时，
            # 调度抖动不应成为第二个、无关的 blocker。
            result["repair_latency_ms"]["p95"] = 0.0
            if profile == "registry_extended_v1":
                result["idempotence_rate"] = 0.9167
            return result

        with patch.object(
            ablation,
            "evaluate_profile",
            side_effect=non_idempotent_p3,
        ):
            report = ablation.build_report(write=False)

        self.assertFalse(
            report["implementation_gates"]["candidate_p3_idempotence_100"]
        )
        self.assertFalse(report["implementation_gates"]["gate_passed"])
        self.assertFalse(report["decision"]["offline_candidate_qualified"])
        self.assertEqual(
            ["candidate_p3_idempotence_100"],
            report["decision"]["offline_candidate_blockers"],
        )

        with (
            patch.object(ablation, "build_report", return_value=report),
            patch.object(sys, "argv", ["semantic_repair_ablation", "--enforce"]),
        ):
            self.assertEqual(2, ablation.main())


if __name__ == "__main__":
    unittest.main()
