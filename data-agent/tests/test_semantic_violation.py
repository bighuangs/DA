import unittest

from app.agent_system.agents.nl2sql.services.semantic_guard import check_semantic_contract
from app.domain.query.safety.semantic_violation import (
    SemanticParseError,
    compact_evidence,
    correction_directive,
    violation_from_exception,
)


class SemanticViolationTests(unittest.TestCase):
    def test_parse_failure_has_stable_syntax_code(self):
        with self.assertRaises(SemanticParseError) as context:
            check_semantic_contract("SELECT FROM", "订单量", [])
        violation = context.exception.violation.to_dict()
        self.assertEqual("syntax.parse_error", violation["violation_code"])
        self.assertEqual("syntax", violation["stage"])
        self.assertEqual("llm", violation["repairability"])
        self.assertTrue(violation["safe_to_retry"])
        self.assertIn("sql_hash", violation["evidence"])

    def test_security_and_cost_failures_are_terminal(self):
        for name, code in (
            ("SQLSecurityError", "security.policy_violation"),
            ("QueryCostError", "cost.budget_exceeded"),
        ):
            error_type = type(name, (Exception,), {})
            violation = violation_from_exception(error_type("hidden")).to_dict()
            self.assertEqual(code, violation["violation_code"])
            self.assertEqual("reject", violation["repairability"])
            self.assertFalse(violation["safe_to_retry"])
            self.assertIn("禁止自动纠错", correction_directive(violation)["instruction"])

    def test_evidence_whitelist_drops_plaintext_and_vectors(self):
        compact = compact_evidence({
            "expected_count": 1,
            "sql_hash": "sha256:safe",
            "raw_sql": "SELECT secret FROM private",
            "question": "Alice 手机号",
            "vector": [0.1, 0.2],
        })
        self.assertEqual({"expected_count": 1, "sql_hash": "sha256:safe"}, compact)


if __name__ == "__main__":
    unittest.main()
