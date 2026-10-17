import asyncio
import time
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

from app.adapters.outbound.agent.llm_provider import task_profile
from app.config.app_config import app_config
import app.infrastructure.clients.llm.invocation as invocation_module
from app.agent_system.runtime.llm_attempts import attempt_accounting, attempt_coverage
from app.infrastructure.clients.llm.invocation import (
    LLMInvocationError,
    FATAL_PROVIDER_REASON_CODES,
    invoke_llm,
    invoke_optional_llm,
    normalize_sql_output,
    pre_invoke_failure,
    sql_output_validator,
)


class _Runnable:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    async def ainvoke(self, payload, config=None):
        del payload, config
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return await outcome()
        return outcome


class _HTTPError(RuntimeError):
    def __init__(self, status_code):
        self.status_code = status_code
        super().__init__(f"http_status_{status_code}")


class StreamClosedError(RuntimeError):
    pass


class _GateRunnable:
    def __init__(self, started, release):
        self.started = started
        self.release = release
        self.calls = 0

    async def ainvoke(self, payload, config=None):
        del payload, config
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return "SELECT 1"


async def _never():
    await asyncio.Event().wait()


class LLMInvocationBoundaryTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        invocation_module._SEMAPHORES.clear()

    async def test_success_emits_one_complete_attempt(self):
        result = await invoke_llm(
            _Runnable("SELECT 1"),
            {},
            node_id="generate_sql",
            timeout_seconds=0.2,
            max_retries=0,
        )
        self.assertEqual(result.value, "SELECT 1")
        self.assertEqual(len(result.attempts), 1)
        attempt = result.attempts[0]
        self.assertEqual(attempt["status"], "success")
        self.assertEqual(attempt["reason_code"], "llm.success")
        self.assertEqual(attempt["retry_index"], 0)
        self.assertFalse(attempt["callback_observed"])
        self.assertEqual(attempt["provider_request_id"], "unavailable")
        self.assertFalse(attempt["provider_request_id_available"])
        self.assertEqual(
            attempt["provider_request_id_unavailable_reason"],
            "parsed_output_did_not_retain_provider_request_id",
        )
        self.assertEqual(
            attempt["token_timing_unavailable_reason"],
            "non_stream_invocation_does_not_expose_token_timing",
        )
        self.assertGreater(attempt["physical_attempt_budget_ms"], 0)
        self.assertEqual(attempt["outcome_class"], "success")
        self.assertEqual(
            attempt["request_deadline_version"],
            "request-deadline-v1",
        )
        self.assertTrue(attempt_accounting(result.attempts)["evidence_complete"])

    async def test_timeout_has_nonempty_typed_error_and_budget(self):
        with self.assertRaises(LLMInvocationError) as caught:
            await invoke_llm(
                _Runnable(_never),
                {},
                node_id="plan_sql",
                timeout_seconds=0.01,
                max_retries=0,
            )
        error = caught.exception
        self.assertEqual(error.reason_code, "llm.timeout")
        self.assertTrue(str(error))
        self.assertIn("llm.timeout", error.user_message)
        self.assertEqual(error.attempts[0]["status"], "timeout")
        self.assertEqual(error.attempts[0]["timeout_budget_ms"], 10)
        self.assertEqual(error.degradations[0]["reason_code"], "llm.timeout")

    async def test_connection_retry_is_bounded_and_visible(self):
        runnable = _Runnable(ConnectionError(), "ok")
        result = await invoke_llm(
            runnable,
            {},
            node_id="generate_sql",
            timeout_seconds=0.5,
            max_retries=1,
        )
        self.assertEqual(result.value, "ok")
        self.assertEqual(runnable.calls, 2)
        self.assertEqual(
            [(row["status"], row["retry_index"]) for row in result.attempts],
            [("error", 0), ("success", 1)],
        )
        self.assertEqual(result.attempts[0]["reason_code"], "llm.connection_error")
        self.assertTrue(result.attempts[0]["retryable"])
        self.assertTrue(result.attempts[1]["recovery"])

    async def test_timeout_policy_can_forbid_low_value_recovery(self):
        runnable = _Runnable(_never, "SELECT 1")
        with self.assertRaises(LLMInvocationError) as caught:
            await invoke_llm(
                runnable,
                {},
                node_id="generate_sql",
                timeout_seconds=0.02,
                max_retries=1,
                recovery_reserve_ms=0,
                retryable_reason_codes={"llm.empty_output"},
            )
        self.assertEqual(caught.exception.reason_code, "llm.timeout")
        self.assertEqual(runnable.calls, 1)
        self.assertFalse(caught.exception.attempts[0]["retryable"])

    async def test_production_policy_keeps_seventeen_second_sql_primary_window(self):
        self.assertEqual(app_config.query.llm_recovery_reserve_ms, 5000)
        sql_profile = task_profile("sql")
        self.assertEqual(sql_profile["max_retries"], 1)
        self.assertGreaterEqual(
            round(float(sql_profile["timeout_seconds"]) * 1000)
            - app_config.query.llm_recovery_reserve_ms,
            17000,
        )

    async def test_default_tail_is_not_reserved_when_phase_window_is_too_small(self):
        runnable = _Runnable(_never, "SELECT 1")
        with self.assertRaises(LLMInvocationError) as caught:
            await invoke_llm(
                runnable,
                {},
                node_id="generate_sql",
                timeout_seconds=0.05,
                max_retries=1,
            )
        self.assertEqual(caught.exception.reason_code, "llm.timeout")
        self.assertEqual(runnable.calls, 1)
        attempt = caught.exception.attempts[0]
        self.assertEqual(attempt["recovery_reserve_ms"], 0)
        self.assertGreaterEqual(attempt["physical_attempt_budget_ms"], 40)
        self.assertFalse(attempt["retryable"])

    async def test_timeout_recovers_once_when_primary_and_tail_are_both_viable(self):
        runnable = _Runnable(_never, "SELECT 1")
        result = await invoke_llm(
            runnable,
            {"verbose": True},
            node_id="generate_sql",
            timeout_seconds=0.2,
            max_retries=1,
            recovery_payload={"verbose": False},
            recovery_reserve_ms=80,
            recovery_min_ms=40,
            primary_min_ms=100,
            retryable_reason_codes={"llm.timeout"},
        )
        self.assertEqual(result.value, "SELECT 1")
        self.assertEqual(runnable.calls, 2)
        first, recovery = result.attempts
        self.assertEqual(first["reason_code"], "llm.timeout")
        self.assertTrue(first["retryable"])
        self.assertEqual(first["recovery_reserve_ms"], 80)
        self.assertGreaterEqual(first["physical_attempt_budget_ms"], 100)
        self.assertTrue(recovery["recovery"])
        self.assertEqual(recovery["status"], "success")
        self.assertEqual(recovery["recovery_reserve_ms"], 0)

    async def test_recovery_tail_never_squeezes_primary_below_declared_minimum(self):
        runnable = _Runnable(_never, "SELECT 1")
        with self.assertRaises(LLMInvocationError) as caught:
            await invoke_llm(
                runnable,
                {},
                node_id="generate_sql",
                timeout_seconds=0.1,
                max_retries=1,
                recovery_reserve_ms=60,
                recovery_min_ms=30,
                primary_min_ms=80,
                retryable_reason_codes={"llm.timeout"},
            )
        self.assertEqual(runnable.calls, 1)
        self.assertEqual(
            caught.exception.attempts[0]["recovery_reserve_ms"],
            0,
        )
        self.assertFalse(caught.exception.attempts[0]["retryable"])

    async def test_recovery_requires_its_declared_minimum_budget(self):
        runnable = _Runnable(None, "SELECT 1")
        with self.assertRaises(LLMInvocationError) as caught:
            await invoke_llm(
                runnable,
                {},
                node_id="generate_sql",
                timeout_seconds=0.2,
                max_retries=1,
                recovery_reserve_ms=0,
                recovery_min_ms=400,
                retryable_reason_codes={"llm.empty_output"},
            )
        self.assertEqual(caught.exception.reason_code, "llm.empty_output")
        self.assertEqual(runnable.calls, 1)
        self.assertFalse(caught.exception.attempts[0]["retryable"])

    async def test_empty_output_is_a_terminal_attempt(self):
        with self.assertRaises(LLMInvocationError) as caught:
            await invoke_llm(
                _Runnable(" "),
                {},
                node_id="generate_sql",
                timeout_seconds=0.2,
                max_retries=0,
            )
        self.assertEqual(caught.exception.reason_code, "llm.empty_output")
        self.assertEqual(caught.exception.attempts[0]["status"], "empty_output")
        self.assertEqual(
            caught.exception.attempts[0]["outcome_class"],
            "parsed_empty",
        )

    async def test_none_recovers_once_with_compressed_payload(self):
        runnable = _Runnable(None, "SELECT 1")
        with patch.object(app_config.query, "llm_recovery_reserve_ms", 0):
            result = await invoke_llm(
                runnable,
                {"verbose": True},
                node_id="generate_sql",
                timeout_seconds=0.5,
                max_retries=9,
                recovery_payload={"verbose": False},
                output_validator=sql_output_validator,
            )
        self.assertEqual(runnable.calls, 2)
        self.assertEqual(result.attempts[0]["outcome_class"], "empty_body")
        self.assertTrue(result.attempts[1]["recovery"])

    async def test_empty_choices_and_truncation_are_distinct(self):
        cases = (
            ({"choices": []}, "empty_choices", "llm.empty_output"),
            (
                {
                    "response_metadata": {"finish_reason": "length"},
                    "content": "SELECT",
                },
                "truncation",
                "llm.truncated_output",
            ),
        )
        for value, outcome, reason in cases:
            with self.subTest(outcome=outcome):
                with self.assertRaises(LLMInvocationError) as caught:
                    await invoke_llm(
                        _Runnable(value),
                        {},
                        node_id="generate_sql",
                        timeout_seconds=0.2,
                        max_retries=0,
                    )
                self.assertEqual(caught.exception.reason_code, reason)
                self.assertEqual(
                    caught.exception.attempts[0]["outcome_class"],
                    outcome,
                )

    async def test_raw_message_preserves_sql_request_id_and_usage(self):
        message = AIMessage(
            content="SELECT 1",
            response_metadata={
                "finish_reason": "stop",
                "request_id": "request-dev5",
                "token_usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 3,
                    "total_tokens": 14,
                },
            },
        )
        result = await invoke_llm(
            _Runnable(message),
            {},
            node_id="generate_sql",
            timeout_seconds=0.2,
            max_retries=0,
            output_validator=sql_output_validator,
        )
        self.assertEqual(normalize_sql_output(result.value), "SELECT 1")
        attempt = result.attempts[0]
        self.assertEqual(attempt["provider_request_id"], "request-dev5")
        self.assertTrue(attempt["provider_request_id_available"])
        self.assertTrue(attempt["token_known"])
        self.assertEqual(attempt["prompt_tokens"], 11)
        self.assertEqual(attempt["completion_tokens"], 3)
        self.assertEqual(attempt["total_tokens"], 14)

    async def test_raw_empty_message_with_length_finish_is_truncation(self):
        message = AIMessage(
            content="",
            response_metadata={
                "finish_reason": "length",
                "request_id": "request-truncated",
            },
        )
        with self.assertRaises(LLMInvocationError) as caught:
            await invoke_llm(
                _Runnable(message),
                {},
                node_id="generate_sql",
                timeout_seconds=0.2,
                max_retries=0,
                output_validator=sql_output_validator,
            )
        self.assertEqual(caught.exception.reason_code, "llm.truncated_output")
        attempt = caught.exception.attempts[0]
        self.assertEqual(attempt["outcome_class"], "truncation")
        self.assertEqual(attempt["provider_request_id"], "request-truncated")

    async def test_429_and_5xx_have_typed_bounded_recovery(self):
        for status, reason in (
            (429, "llm.rate_limited"),
            (503, "llm.server_error"),
        ):
            with self.subTest(status=status):
                runnable = _Runnable(_HTTPError(status), "SELECT 1")
                with patch.object(
                    app_config.query,
                    "llm_recovery_reserve_ms",
                    0,
                ):
                    result = await invoke_llm(
                        runnable,
                        {},
                        node_id="generate_sql",
                        timeout_seconds=0.5,
                        max_retries=5,
                    )
                self.assertEqual(runnable.calls, 2)
                self.assertEqual(result.attempts[0]["reason_code"], reason)
                self.assertTrue(result.attempts[1]["recovery"])

    async def test_auth_billing_and_permission_statuses_are_typed_and_non_retryable(self):
        for status, reason in (
            (401, "llm.authentication_error"),
            (402, "llm.payment_required"),
            (403, "llm.permission_denied"),
        ):
            with self.subTest(status=status):
                runnable = _Runnable(_HTTPError(status), "SELECT 1")
                with self.assertRaises(LLMInvocationError) as caught:
                    await invoke_llm(
                        runnable,
                        {},
                        node_id="generate_sql",
                        timeout_seconds=6,
                        max_retries=5,
                    )
                error = caught.exception
                self.assertEqual(reason, error.reason_code)
                self.assertIn(reason, FATAL_PROVIDER_REASON_CODES)
                self.assertEqual(1, runnable.calls)
                self.assertEqual(
                    error.attempts[0]["recovery_reserve_ms"],
                    app_config.query.llm_recovery_reserve_ms,
                )
                self.assertFalse(error.attempts[0]["retryable"])
                self.assertEqual(reason.removeprefix("llm."), error.attempts[0]["outcome_class"])

    async def test_fatal_provider_user_message_never_exposes_provider_body(self):
        secret = "credential-fragment-that-must-not-leak"
        provider_error = _HTTPError(402)
        provider_error.args = (secret,)
        with self.assertRaises(LLMInvocationError) as caught:
            await invoke_llm(
                _Runnable(provider_error),
                {},
                node_id="generate_sql",
                timeout_seconds=0.5,
                max_retries=0,
            )
        error = caught.exception
        self.assertEqual("llm.payment_required", error.reason_code)
        self.assertNotIn(secret, str(error))
        self.assertNotIn(secret, error.user_message)

    async def test_stream_interruption_never_recovers_more_than_once(self):
        runnable = _Runnable(
            StreamClosedError("stream closed"),
            StreamClosedError("stream closed"),
            "SELECT 1",
        )
        with patch.object(app_config.query, "llm_recovery_reserve_ms", 0):
            with self.assertRaises(LLMInvocationError) as caught:
                await invoke_llm(
                    runnable,
                    {},
                    node_id="generate_sql",
                    timeout_seconds=0.5,
                    max_retries=10,
                )
        self.assertEqual(runnable.calls, 2)
        self.assertEqual(len(caught.exception.attempts), 2)
        self.assertTrue(
            all(
                item["reason_code"] == "llm.stream_interrupted"
                for item in caught.exception.attempts
            )
        )

    async def test_expired_shared_deadline_makes_no_physical_call(self):
        runnable = _Runnable("SELECT 1")
        with self.assertRaises(LLMInvocationError) as caught:
            await invoke_llm(
                runnable,
                {},
                node_id="generate_sql",
                timeout_seconds=1,
                max_retries=1,
                request_deadline_at=time.perf_counter() - 0.01,
            )
        self.assertEqual(runnable.calls, 0)
        self.assertEqual(
            caught.exception.reason_code,
            "llm.deadline_exhausted",
        )
        self.assertEqual(
            caught.exception.attempts[0]["completion_stage"],
            "deadline_check",
        )

    async def test_optional_empty_collection_can_be_a_valid_result(self):
        result = await invoke_optional_llm(
            _Runnable([]),
            {},
            node_id="recall_column",
            route="keyword",
            fallback=["fallback"],
            timeout_seconds=0.2,
            max_retries=0,
            allow_empty=True,
        )
        self.assertEqual(result.value, [])
        self.assertEqual(result.attempts[0]["status"], "success")

    async def test_cancellation_preserves_attempt_and_degradation(self):
        task = asyncio.create_task(
            invoke_llm(
                _Runnable(_never),
                {},
                node_id="judge_candidate",
                timeout_seconds=1,
                max_retries=0,
            )
        )
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError) as caught:
            await task
        error = caught.exception
        self.assertEqual(error.llm_attempts[0]["status"], "cancelled")
        self.assertEqual(error.degradations[0]["reason_code"], "llm.cancelled")

    async def test_queue_cancellation_releases_partial_acquisitions(self):
        profile = {
            "model_name": "test-model",
            "timeout_seconds": 1,
            "max_retries": 0,
            "max_concurrency": 1,
        }
        started, release = asyncio.Event(), asyncio.Event()
        gate = _GateRunnable(started, release)
        with (
            patch.object(
                invocation_module,
                "task_profile",
                return_value=profile,
            ),
            patch.object(app_config.llm, "max_concurrency", 1),
        ):
            leader = asyncio.create_task(
                invoke_llm(
                    gate,
                    {},
                    node_id="generate_sql",
                    route="sql",
                    timeout_seconds=1,
                    max_retries=0,
                )
            )
            await started.wait()
            queued = asyncio.create_task(
                invoke_llm(
                    _Runnable("SELECT 2"),
                    {},
                    node_id="generate_sql",
                    route="sql",
                    timeout_seconds=1,
                    max_retries=0,
                )
            )
            await asyncio.sleep(0.01)
            queued.cancel()
            with self.assertRaises(asyncio.CancelledError) as caught:
                await queued
            self.assertEqual(
                caught.exception.llm_attempts[0]["cancel_stage"],
                "queue",
            )
            release.set()
            await leader
            # A third caller proves the cancelled waiter leaked no permit.
            follower = await asyncio.wait_for(
                invoke_llm(
                    _Runnable("SELECT 3"),
                    {},
                    node_id="generate_sql",
                    route="sql",
                    timeout_seconds=0.2,
                    max_retries=0,
                ),
                timeout=0.5,
            )
            self.assertEqual(follower.value, "SELECT 3")

    async def test_pre_invoke_failure_is_explicit(self):
        error = pre_invoke_failure(
            node_id="generate_sql",
            error=RuntimeError(),
            timeout_seconds=0.2,
        )
        self.assertEqual(error.reason_code, "llm.pre_invoke_failure")
        self.assertTrue(str(error))
        self.assertTrue(error.attempts[0]["pre_invoke_failure"])
        self.assertEqual(error.attempts[0]["error_type"], "RuntimeError")

    async def test_failed_node_without_attempt_fails_coverage(self):
        coverage = attempt_coverage(
            [],
            failed_nodes=["generate_sql"],
            output_nodes=[],
        )
        self.assertFalse(coverage["complete"])
        self.assertEqual(coverage["missing_failure_attempts"], ["generate_sql"])

    def test_complete_sql_fence_is_normalized_and_incomplete_fence_rejected(self):
        value = "```sql\nSELECT 1\n```"
        self.assertIsNone(sql_output_validator(value))
        self.assertEqual(normalize_sql_output(value), "SELECT 1")
        self.assertEqual(
            sql_output_validator("```sql\nSELECT 1"),
            "llm.truncated_output",
        )


if __name__ == "__main__":
    unittest.main()
