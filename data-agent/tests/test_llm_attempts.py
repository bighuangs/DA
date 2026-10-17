import unittest

from app.agent_system.runtime.llm_attempts import (
    attempt_accounting,
    make_llm_attempt,
    normalize_degradation,
    reconcile_attempts,
)


class LLMAttemptContractTests(unittest.TestCase):
    def test_terminal_statuses_form_a_closed_accounting_identity(self):
        attempts = [
            make_llm_attempt(node_id="generate_sql", status="success", latency_ms=10,
                             total_tokens=12, prompt_tokens=8, completion_tokens=4),
            make_llm_attempt(node_id="judge_candidate", status="error", latency_ms=11,
                             token_known=False),
            make_llm_attempt(node_id="judge_candidate", status="timeout", latency_ms=12,
                             token_known=False),
            make_llm_attempt(node_id="generate_sql", status="cancelled", latency_ms=13,
                             token_known=False),
            make_llm_attempt(node_id="generate_analysis", status="empty_output", latency_ms=14,
                             token_known=False),
        ]
        result = attempt_accounting(attempts)
        self.assertEqual(result["attempts"], 5)
        self.assertEqual(result["succeeded"], 1)
        self.assertEqual(result["failed"], 2)  # error + empty output
        self.assertEqual(result["timed_out"], 1)
        self.assertEqual(result["cancelled"], 1)
        self.assertEqual(result["core_failures"], 3)
        self.assertEqual(result["optional_failures"], 1)
        self.assertTrue(result["reconciled"])

    def test_node_status_wins_while_callback_tokens_are_preserved(self):
        callback = make_llm_attempt(
            node_id="judge_candidate", route="candidate", status="error", latency_ms=2450,
            prompt_tokens=100, completion_tokens=20, total_tokens=120,
            estimated_cost=.01, source="callback",
        )
        explicit = make_llm_attempt(
            node_id="judge_candidate", route="candidate", status="timeout", latency_ms=2500,
            timeout_budget_ms=2500, token_known=False, source="node",
        )
        merged = reconcile_attempts([callback], [explicit])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["status"], "timeout")
        self.assertEqual(merged[0]["total_tokens"], 120)
        self.assertTrue(merged[0]["token_known"])
        self.assertEqual(merged[0]["timeout_budget_ms"], 2500)

    def test_unknown_tokens_remain_null_instead_of_becoming_zero(self):
        item = make_llm_attempt(
            node_id="judge_candidate", status="timeout", latency_ms=2500,
            timeout_budget_ms=2500, token_known=False,
        )
        self.assertIsNone(item["prompt_tokens"])
        self.assertIsNone(item["completion_tokens"])
        self.assertIsNone(item["total_tokens"])
        self.assertEqual(attempt_accounting([item])["unknown_tokens"], 1)
        self.assertEqual(
            item["provider_request_id_unavailable_reason"],
            "provider_response_did_not_expose_request_id",
        )
        self.assertEqual(
            item["token_timing_unavailable_reason"],
            "non_stream_invocation_does_not_expose_token_timing",
        )
        self.assertTrue(attempt_accounting([item])["evidence_complete"])

    def test_degradation_requires_node_and_classifies_optional_postprocess(self):
        incomplete = normalize_degradation({"reason": "timeout"})
        optional = normalize_degradation({
            "node_id": "generate_analysis", "reason_code": "empty_model_output",
        })
        self.assertFalse(incomplete["evidence_complete"])
        self.assertTrue(optional["evidence_complete"])
        self.assertFalse(optional["affects_core_path"])


if __name__ == "__main__":
    unittest.main()
