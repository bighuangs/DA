import unittest

from evals.legacy.conversation_memory_eval import build_report


class ConversationMemoryEvaluationTests(unittest.TestCase):
    def test_todo59_contract_has_required_coverage_and_passes(self):
        report = build_report()
        self.assertTrue(report["contract"]["valid"], report["contract"])
        self.assertGreaterEqual(report["contract"]["total"], 16)
        self.assertEqual(report["summary"]["failed"], 0, report["cases"])
        self.assertTrue(report["summary"]["gate_passed"])
        self.assertIn("security", report["contract"]["categories"])
        self.assertIn("degradation", report["contract"]["categories"])


if __name__ == "__main__":
    unittest.main()
