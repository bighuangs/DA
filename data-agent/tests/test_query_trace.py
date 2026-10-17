import unittest
import asyncio
import json
from types import SimpleNamespace

from app.infrastructure.persistence.mysql.repositories.query_trace_repository import build_trace_records
from evals.application.services.quality_operations_service import QualityOperationsService
from app.application.services.query_trace_service import QueryTraceCollector
from app.agent_system.runtime.llm_attempts import make_llm_attempt


class QueryTraceTest(unittest.TestCase):
    def test_handled_intent_is_not_classified_as_pipeline_failure(self):
        collector = QueryTraceCollector(chat_id="chat-handled")

        summary, trace, steps, usages = build_trace_records(
            collector,
            query="今天天气怎么样",
            chat_id="chat-handled",
            sql=None,
            status="handled",
            row_count=0,
            error=None,
        )

        self.assertEqual(summary["status"], "handled")
        self.assertEqual(trace.status, "handled")
        self.assertEqual(trace.failure_code, "none")
        self.assertEqual(steps, [])
        self.assertEqual(usages, [])

    def test_stage_and_provider_usage_are_aggregated(self):
        collector = QueryTraceCollector(chat_id="chat-1")
        collector.start_step("生成SQL")
        collector.callback.on_llm_end(SimpleNamespace(
            llm_output={
                "model_name": "test-model",
                "token_usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                },
            },
            generations=[],
        ))
        summary, trace, steps, usages = build_trace_records(
            collector,
            query="统计销售额",
            chat_id="chat-1",
            sql="SELECT 1",
            status="success",
            row_count=1,
            version_snapshot={
                "prompt_version": "prompt-v1", "knowledge_version": "knowledge-v1",
                "schema_version": "schema-v1", "index_version": "index-v1",
                "model_version": "model-v1", "run_fingerprint": "fingerprint-v1",
                "version_run_id": "run-30", "version_complete": True,
            },
            cache_hit=True,
        )

        self.assertEqual(summary["total_tokens"], 18)
        self.assertEqual(summary["prompt_tokens"], 11)
        self.assertEqual(summary["llm_calls"], 1)
        self.assertEqual(trace.total_tokens, 18)
        self.assertEqual(trace.llm_calls, 1)
        self.assertEqual(steps[0].step_name, "生成SQL")
        self.assertEqual(steps[0].attempt, 1)
        self.assertEqual(usages[0].model, "test-model")
        self.assertGreaterEqual(usages[0].latency_ms, 0)
        self.assertEqual(trace.query_category, "aggregation")
        self.assertEqual(trace.failure_code, "none")
        self.assertEqual(trace.run_fingerprint, "fingerprint-v1")
        self.assertTrue(trace.version_complete)
        self.assertTrue(trace.cache_hit)

    def test_parallel_node_spans_and_milestones_are_independent(self):
        collector = QueryTraceCollector(chat_id="chat-2")
        collector.start_node("generate_analysis", 100)
        collector.start_node("generate_recommend", 101)
        collector.end_node("generate_recommend", end_ms=140)
        collector.end_node("generate_analysis", end_ms=160)
        collector.mark_first_event()
        collector.mark_sql()
        collector.mark_result()
        collector.add_degradation({"node_id": "generate_analysis", "reason": "timeout"})
        summary = collector.finish()

        self.assertEqual([step.status for step in collector.steps], ["success", "success"])
        self.assertEqual(summary["degraded_count"], 1)
        self.assertEqual(summary["slowest_nodes"][0]["node_id"], "generate_analysis")
        self.assertIsNotNone(summary["tt_sql_ms"])

    def test_internal_node_event_maps_end_ms_as_keyword_only_argument(self):
        collector = QueryTraceCollector(chat_id="chat-events")
        collector.record_node_event({
            "event": "node_start", "node_id": "extract_keywords", "start_ms": 10,
        })
        collector.record_node_event({
            "event": "node_end", "node_id": "extract_keywords", "end_ms": 25,
            "status": "success", "error": None,
        })

        self.assertEqual(len(collector.steps), 1)
        self.assertEqual(collector.steps[0].name, "extract_keywords")
        self.assertEqual(collector.steps[0].status, "success")
        self.assertEqual(collector.steps[0].duration_ms, 15)

    def test_failure_is_recorded_on_current_step(self):
        collector = QueryTraceCollector(chat_id="chat-1")
        collector.start_step("执行SQL")
        collector.fail("数据库连接失败")
        summary, trace, steps, _ = build_trace_records(
            collector,
            query="测试",
            chat_id="chat-1",
            sql=None,
            status="failed",
            row_count=0,
            error="数据库连接失败",
        )

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(trace.error, "数据库连接失败")
        self.assertEqual(trace.failure_code, "dependency_connection")
        self.assertEqual(steps[0].status, "failed")
        self.assertEqual(steps[0].error, "数据库连接失败")

    def test_node_owned_timeout_is_counted_when_provider_callback_is_missing(self):
        collector = QueryTraceCollector(chat_id="candidate-timeout")
        collector.record_llm_attempt(make_llm_attempt(
            node_id="judge_candidate", route="candidate", status="timeout",
            latency_ms=2400, timeout_budget_ms=2500, token_known=False,
            error_type=asyncio.TimeoutError.__name__, core_path=True,
        ))
        summary, _, _, usages = build_trace_records(
            collector, query="复杂问题", chat_id="candidate-timeout", sql="SELECT 1",
            status="success", row_count=1,
        )
        self.assertEqual(summary["llm_calls"], 1)
        self.assertEqual(summary["failed_llm_calls"], 1)
        self.assertEqual(summary["failed_core_llm_calls"], 1)
        self.assertEqual(summary["llm_attempt_accounting"]["timed_out"], 1)
        self.assertFalse(summary["llm_attempts"][0]["token_known"])
        self.assertEqual(usages[0].status, "timeout")
        self.assertEqual(usages[0].route, "candidate")

    def test_retrieval_trace_keeps_point_contract_but_drops_projection_content(self):
        collector = QueryTraceCollector(chat_id="retrieval-proof")
        collector.record_retrieval_metric({
            "version": "rule-rerank-v6",
            "route": "column",
            "dynamic_k": 4,
            "fusion_method": "entity_rrf",
            "hybrid_queries": [{
                "retrieval_mode": "hybrid",
                "effective_mode": "hybrid",
                "fusion_method": "entity_rrf",
                "dense": {"raw_points": 8, "unique_entities": 8},
                "sparse": {"raw_points": 6, "unique_entities": 6},
                "fused_entities": 10,
                "point_strategies": ["single_canonical"],
                "matched_point_roles": ["canonical"],
                "source_text": "不得入库的元知识原文",
                "vector": [0.1, 0.2],
            }],
            "embedding_runtime": {
                "runtime_profile": "microbatch_inprocess_v1",
                "queue_depth": 0,
                "queue_capacity": 128,
                "last_batch_texts": 6,
                "merged_batch_count": 4,
                "total_p95_ms": 41.2,
                "failed": 0,
                "raw_text_logged": False,
                "raw_text": "不得写入 Trace 的用户问题",
            },
            "prompt": "不得入库的查询提示词",
        })
        _, trace, _, _ = build_trace_records(
            collector, query="销售额", chat_id="retrieval-proof",
            sql="SELECT 1", status="success", row_count=1,
        )

        stored = json.loads(trace.retrieval_summary)
        route = stored["routes"][0]
        query = route["hybrid_queries"][0]
        self.assertEqual(route["version"], "rule-rerank-v6")
        self.assertEqual(query["point_strategies"], ["single_canonical"])
        self.assertEqual(query["matched_point_roles"], ["canonical"])
        self.assertNotIn("prompt", route)
        self.assertNotIn("source_text", query)
        self.assertNotIn("vector", query)
        self.assertEqual(
            route["embedding_runtime"]["runtime_profile"],
            "microbatch_inprocess_v1",
        )
        self.assertEqual(route["embedding_runtime"]["last_batch_texts"], 6)
        self.assertFalse(route["embedding_runtime"]["raw_text_logged"])
        self.assertNotIn("raw_text", route["embedding_runtime"])
        self.assertEqual(
            QualityOperationsService._retrieval_evidence(trace.retrieval_summary),
            [route],
        )

    def test_query_signal_trace_keeps_contract_without_plaintext(self):
        collector = QueryTraceCollector(chat_id="query-signal-proof")
        collector.record_retrieval_metric({
            "route": "column",
            "keyword_count": 3,
            "remote_query_count": 1,
            "query_signals": {
                "contract_version": "query-signals-v1",
                "extractor_profile": "domain_signals_v1",
                "extractor_version": "query-signal-extractor-v1:k2-v1",
                "dictionary_version": "dict-fingerprint",
                "profile_fingerprint": "profile-fingerprint",
                "signal_fingerprint": "signal-fingerprint",
                "route": "column",
                "route_policy": "routed_signals_v1",
                "route_policy_version": "query-signal-routing-v1",
                "local_signal_count": 4,
                "remote_query_count": 1,
                "rerank_signal_count": 4,
                "query_encode_mode": "query",
                "counts": {"semantic_queries": 1, "identifiers": 1},
                "signals": [{
                    "signal_id": "qs-1",
                    "kind": "identifiers",
                    "rank": 1,
                    "fingerprint": "id-fingerprint",
                    "text": "customer_name",
                }],
                "actual_remote_signals": [{
                    "signal_id": "qs-2",
                    "kind": "standalone_query",
                    "rank": 0,
                    "fingerprint": "query-fingerprint",
                    "consumer": "remote",
                    "text": "Alice的大额订单",
                }],
                "raw_text_logged": False,
                "standalone_query": "Alice的大额订单",
            },
        })
        _, trace, _, _ = build_trace_records(
            collector,
            query="Alice的大额订单",
            chat_id="query-signal-proof",
            sql="SELECT 1",
            status="success",
            row_count=1,
        )
        raw = trace.retrieval_summary
        self.assertNotIn("Alice", raw)
        self.assertNotIn("customer_name", raw)
        stored = json.loads(raw)["routes"][0]["query_signals"]
        self.assertEqual(stored["extractor_profile"], "domain_signals_v1")
        self.assertEqual(stored["query_encode_mode"], "query")
        self.assertEqual(stored["actual_remote_signals"][0]["kind"], "standalone_query")
        self.assertFalse(stored["raw_text_logged"])

    def test_planning_route_trace_keeps_decision_and_drops_plaintext(self):
        collector = QueryTraceCollector(chat_id="planning-proof")
        collector.record_planning_route({
            "contract_version": "planning-route-v1",
            "profile": "keyword_length_v1",
            "decision": "plan_then_generate",
            "actual_action": "plan_then_generate",
            "score": 1,
            "confidence": 1,
            "reason_codes": ["legacy_complex_keyword"],
            "feature_schema": "planning-features-v1",
            "feature_fingerprint": "sha256-safe",
            "feature_summary": {
                "feature_schema": "planning-features-v1",
                "query_length_bucket": "13_24",
                "legacy_keyword_count": 1,
                "query": "不得保存的客户问题",
            },
            "budget_snapshot": {
                "planning_timeout_ms": 8000,
                "planning_timeout_bucket": "under_10s",
            },
            "fallback_profile": "keyword_length_v1",
            "cache_namespace": "planning:safe",
            "repair_profile": "semantic-repair-inline-v1",
            "raw_question": "不得保存的客户问题",
        })
        collector.record_planning_timing({
            "contract_version": "planning-timing-v1",
            "queue_wait_ms": 0,
            "provider_ms": 720,
            "node_ms": 731,
            "time_to_first_token_ms": 100,
            "time_to_last_token_ms": 720,
            "completion_mode": "non_stream_completion_latency",
            "token_timing_available": False,
            "status": "success",
            "prompt": "不得保存",
        })
        summary, trace, _, _ = build_trace_records(
            collector,
            query="各地区销售额同比",
            chat_id="planning-proof",
            sql="SELECT 1",
            status="success",
            row_count=1,
        )
        raw = trace.planning_summary
        self.assertNotIn("不得保存", raw)
        stored = json.loads(raw)
        self.assertEqual("plan_then_generate", stored["actual_action"])
        self.assertEqual(720, stored["timing"]["provider_ms"])
        self.assertFalse(stored["timing"]["token_timing_available"])
        # Non-stream provider calls must not claim token timing.
        self.assertIsNone(stored["timing"]["time_to_first_token_ms"])
        self.assertIsNone(stored["timing"]["time_to_last_token_ms"])
        self.assertEqual(stored, summary["planning_route"])

    def test_semantic_coverage_deadline_and_attempt_v2_survive_persistence(self):
        collector = QueryTraceCollector(chat_id="runtime-proof")
        collector.record_semantic_coverage({
            "schema_version": "semantic-coverage-v1",
            "registry_version": "semantic-composition-v1",
            "question_fingerprint": "sha256-safe",
            "coverage_score": 1.0,
            "compiler_mode": "deterministic",
            "compiler_supported": True,
            "detected_slots": ["metric:sum", "dimension:region"],
            "represented_slots": ["metric:sum", "dimension:region"],
            "unresolved_slots": [],
            "abstain_reasons": [],
            "raw_question_persisted": False,
            "filter_values_persisted": False,
            "raw_question": "不得入库的客户问题",
        })
        collector.record_request_deadline({
            "contract_version": "request-deadline-v1",
            "budget_ms": 30000,
            "remaining_ms": 24750,
            "execution_reserve_ms": 3000,
            "phase_budgets_ms": {"llm": 18000, "execution": 3000},
            "started_monotonic": 123456.78,
        })
        collector.record_llm_attempt(make_llm_attempt(
            node_id="generate_sql",
            route="generation",
            model="test-model",
            status="success",
            latency_ms=520,
            timeout_budget_ms=18000,
            deadline_before_ms=25000,
            deadline_after_ms=24480,
            queue_wait_ms=20,
            provider_ms=500,
            provider_request_id=None,
            provider_request_id_available=False,
            completion_stage="provider_complete",
            outcome_class="success",
            recovery=False,
            retryable=False,
            token_known=False,
        ))

        summary, trace, _, usages = build_trace_records(
            collector,
            query="不得写入覆盖摘要的客户问题",
            chat_id="runtime-proof",
            sql="SELECT 1",
            status="success",
            row_count=1,
        )

        coverage_raw = trace.semantic_coverage_summary
        deadline_raw = trace.request_deadline_summary
        self.assertNotIn("客户问题", coverage_raw)
        self.assertNotIn("started_monotonic", deadline_raw)
        self.assertEqual(
            QualityOperationsService._semantic_coverage_evidence(coverage_raw),
            summary["semantic_coverage"],
        )
        self.assertEqual(
            QualityOperationsService._request_deadline_evidence(deadline_raw),
            summary["request_deadline"],
        )
        usage = usages[0]
        self.assertEqual(usage.queue_wait_ms, 20)
        self.assertEqual(usage.provider_ms, 500)
        self.assertEqual(usage.deadline_before_ms, 25000)
        self.assertEqual(usage.deadline_after_ms, 24480)
        self.assertEqual(usage.outcome_class, "success")
        self.assertEqual(usage.provider_request_id, "unavailable")
        self.assertFalse(usage.provider_request_id_available)
        self.assertEqual(
            usage.provider_request_id_unavailable_reason,
            "provider_response_did_not_expose_request_id",
        )
        self.assertEqual(
            usage.token_timing_unavailable_reason,
            "non_stream_invocation_does_not_expose_token_timing",
        )
        self.assertEqual(usage.recovery_reserve_ms, 0)
        self.assertEqual(usage.physical_attempt_budget_ms, 18000)

    def test_query_contract_and_relation_closure_share_privacy_safe_trace(self):
        collector = QueryTraceCollector(chat_id="semantic-contract-proof")
        collector.record_semantic_coverage({
            "schema_version": "semantic-coverage-v1",
            "coverage_score": 1.0,
            "compiler_supported": True,
            "detected_slots": ["operator:share"],
            "represented_slots": ["operator:share"],
            "unresolved_slots": [],
        })
        collector.record_semantic_query_contract({
            "schema_version": "semantic-query-contract-v1",
            "semantic_family": "share",
            "authority": "closed",
            "contract_hash": "sha256:contract",
            "operations": ["share"],
            "output_grain": [{
                "table": "dim_region", "column": "region_name",
                "source_column": "region_name", "visibility": "output",
            }],
            "relation_plan": {
                "schema_version": "metadata-relation-closure-v1",
                "status": "closed",
                "required_columns": {
                    "fact_order": ["region_id", "order_amount"],
                    "dim_region": ["region_id", "region_name"],
                },
                "joins": [{
                    "left_table": "fact_order", "left_column": "region_id",
                    "right_table": "dim_region", "right_column": "region_id",
                    "relationship": "many_to_one", "evidence": "pk_fk_role",
                }],
            },
            "raw_question": "不得写入的客户和筛选值",
            "filter_values": ["绝密区域"],
        })
        collector.record_relation_closure({
            "schema_version": "metadata-relation-closure-v1",
            "status": "closed",
            "terminal_tables": ["fact_order", "dim_region"],
            "added_tables": ["fact_order"],
            "added_columns": {"fact_order": ["region_id"]},
            "joins": [{
                "left_table": "fact_order", "left_column": "region_id",
                "right_table": "dim_region", "right_column": "region_id",
                "relationship": "many_to_one", "evidence": "pk_fk_role",
            }],
            "raw_question": "不得写入",
        })
        summary, trace, _, _ = build_trace_records(
            collector,
            query="不得写入覆盖摘要的客户问题",
            chat_id="semantic-contract-proof",
            sql="SELECT 1",
            status="success",
            row_count=1,
        )
        raw = trace.semantic_coverage_summary
        self.assertNotIn("绝密区域", raw)
        self.assertNotIn("客户和筛选值", raw)
        stored = json.loads(raw)
        self.assertEqual(
            stored["query_contract"]["authority"], "closed"
        )
        self.assertEqual(
            stored["relation_closure"]["status"], "closed"
        )
        self.assertFalse(
            stored["query_contract"]["filter_values_persisted"]
        )
        self.assertEqual(stored, summary["semantic_coverage"])
        self.assertEqual(
            QualityOperationsService._semantic_coverage_evidence(raw),
            stored,
        )


if __name__ == "__main__":
    unittest.main()
