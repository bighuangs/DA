import json
import unittest

from evals.legacy.query_signal_ablation import build_report, validate_contract


class QuerySignalAblationTests(unittest.TestCase):
    def test_reviewed_contract_and_all_offline_gates_pass(self):
        contract = validate_contract()
        report = build_report(write=False)
        self.assertTrue(contract["valid"])
        self.assertGreaterEqual(contract["cases"], 20)
        self.assertTrue(report["summary"]["gate_passed"])
        self.assertEqual(
            report["decision"]["selected_profile"], "jieba_tfidf_v1"
        )
        self.assertFalse(report["decision"]["active_changed"])

    def test_k2_signal_gain_does_not_auto_authorize_active_switch(self):
        report = build_report(write=False)
        self.assertGreater(
            report["profiles"]["K2"]["auxiliary_recall"],
            report["profiles"]["K1"]["auxiliary_recall"],
        )
        self.assertLess(
            report["profiles"]["K2"]["routing"]["column"][
                "remote_queries_mean"
            ],
            report["profiles"]["K1"]["routing"]["column"][
                "remote_queries_mean"
            ],
        )
        self.assertEqual(report["decision"]["status"], "retain_active")

    def test_report_contains_no_reviewed_query_or_dictionary_plaintext(self):
        report = build_report(write=False)
        raw = json.dumps(report, ensure_ascii=False)
        for forbidden in (
            "order_amount与GMV有什么区别",
            "客户甲",
            "机密利润率",
        ):
            self.assertNotIn(forbidden, raw)
        self.assertFalse(report["summary"]["raw_text_logged"])

    def test_live_reason_does_not_claim_active_passed_when_all_profiles_fail(self):
        report = build_report(
            live_profiles={
                "K1": {"gate_passed": False},
                "K2": {"gate_passed": False},
            },
            write=False,
        )

        self.assertFalse(report["gates"]["live_active_rag_gate"])
        self.assertEqual(report["decision"]["passing_live_challengers"], [])
        self.assertIn("均未通过真实 RAG 门禁", report["decision"]["reason"])
        self.assertNotIn("唯一通过", report["decision"]["reason"])


if __name__ == "__main__":
    unittest.main()
