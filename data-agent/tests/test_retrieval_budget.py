import asyncio
import ast
import gc
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import app.agent_system.governance.retrieval_budget as retrieval_budget_module
from app.agent_system.agents.nl2sql.settings import FrozenSettingsSection
from app.agent_system.governance.retrieval_budget import (
    candidate_cache_key,
    candidate_cache_get,
    candidate_cache_set,
    capture_retrieval_cache_generation,
    embed_texts_cached,
    invalidate_retrieval_caches,
    pack_prompt_context,
    policy_scope_fingerprint,
    rerank_and_cut,
)


class RetrievalRerankTests(unittest.TestCase):
    def test_strong_matches_shrink_noise_without_losing_required_columns(self):
        candidates = [
            {"id": "dim_region.region_name", "name": "region_name", "alias": ["地区"],
             "description": "地区名称", "table_id": "dim_region", "role": "dimension",
             "is_display": True, "score": .82},
            {"id": "fact_order.order_amount", "name": "order_amount", "alias": ["销售额"],
             "description": "订单金额", "table_id": "fact_order", "role": "measure", "score": .79},
        ] + [
            {"id": f"dim_product.noise_{index}", "name": f"noise_{index}", "alias": [],
             "description": "无关字段", "table_id": "dim_product", "role": "dimension", "score": .75 - index / 100}
            for index in range(8)
        ]
        selected, evidence = rerank_and_cut(
            "column", "各地区的销售额", ["地区", "销售额"], candidates,
            min_limit=3, max_limit=8,
        )
        ids = {item["id"] for item in selected}
        self.assertIn("dim_region.region_name", ids)
        self.assertIn("fact_order.order_amount", ids)
        self.assertEqual(len(selected), 2)
        self.assertEqual(evidence["cutoff_reason"], "strong_semantic_compact")
        self.assertEqual(evidence["exploratory_tail_count"], 0)
        self.assertGreater(evidence["pruned_count"], 0)

    def test_single_strong_match_keeps_bounded_exploration(self):
        candidates = [
            {"id": "fact_order.order_amount", "name": "order_amount",
             "alias": ["成交金额"], "table_id": "fact_order",
             "role": "measure", "score": .85},
            {"id": "dim_channel.channel_name", "name": "channel_name",
             "alias": [], "table_id": "dim_channel",
             "role": "dimension", "score": .72},
            {"id": "dim_region.region_name", "name": "region_name",
             "alias": [], "table_id": "dim_region",
             "role": "dimension", "score": .70},
        ]

        selected, evidence = rerank_and_cut(
            "column", "某业务切面下的成交金额", ["成交金额", "业务切面"],
            candidates, min_limit=3, max_limit=8,
        )

        self.assertEqual(len(selected), 3)
        self.assertEqual(
            evidence["cutoff_reason"], "single_strong_with_exploration"
        )
        self.assertEqual(evidence["exploratory_tail_count"], 2)

    def test_metric_route_keeps_one_exploratory_candidate(self):
        candidates = [
            {"id": "m1", "name": "成交额", "alias": ["销售额"], "score": .9},
            {"id": "m2", "name": "订单数", "alias": ["订单笔数"], "score": .88},
            {"id": "m3", "name": "买家数", "alias": [], "score": .70},
        ]

        selected, evidence = rerank_and_cut(
            "metric", "销售额和订单笔数", ["销售额", "订单笔数"],
            candidates, min_limit=1, max_limit=5,
        )

        self.assertEqual(len(selected), 3)
        self.assertEqual(
            evidence["cutoff_reason"],
            "strong_exact_or_alias_with_exploration",
        )

    def test_weak_close_scores_expand_to_maximum(self):
        candidates = [
            {"id": f"t.c{index}", "name": f"c{index}", "alias": [], "description": "",
             "table_id": "t", "role": "dimension", "score": .70 - index / 100}
            for index in range(8)
        ]
        selected, evidence = rerank_and_cut(
            "column", "未知业务表达", ["未知业务表达"], candidates,
            min_limit=3, max_limit=8,
        )
        self.assertEqual(len(selected), 8)
        self.assertIn("expand", evidence["cutoff_reason"])


class RetrievalCacheTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        invalidate_retrieval_caches()

    async def test_embedding_cache_batches_only_misses(self):
        client = AsyncMock()
        client.aembed_documents.return_value = [[.1, .2], [.3, .4]]
        first, first_stats = await embed_texts_cached(client, ["地区", "销售额"])
        second, second_stats = await embed_texts_cached(client, ["地区", "销售额"])
        self.assertEqual(first, second)
        self.assertEqual(first_stats, {"hits": 0, "misses": 2})
        self.assertEqual(second_stats, {"hits": 2, "misses": 0})
        client.aembed_documents.assert_awaited_once()

    async def test_multiple_query_misses_never_use_document_encoder(self):
        client = AsyncMock()
        client.aembed_queries.return_value = [[.1, .2], [.3, .4]]
        vectors, stats = await embed_texts_cached(
            client,
            ["完整问题", "GMV"],
            mode="query",
            namespace="profile-k1",
        )
        self.assertEqual(vectors, [[.1, .2], [.3, .4]])
        self.assertEqual(stats, {"hits": 0, "misses": 2})
        client.aembed_queries.assert_awaited_once_with(["完整问题", "GMV"])
        client.aembed_documents.assert_not_awaited()

    async def test_concurrent_namespaces_share_one_exact_vector_flight(self):
        started = asyncio.Event()
        release = asyncio.Event()
        client = AsyncMock()
        client.profile_fingerprint = "embedding-profile-a"

        async def encode(texts):
            started.set()
            await release.wait()
            return [[.1, .2] for _ in texts]

        client.aembed_queries.side_effect = encode
        first = asyncio.create_task(embed_texts_cached(
            client, ["GMV"], mode="query", namespace="profile-k1"
        ))
        await started.wait()
        second = asyncio.create_task(embed_texts_cached(
            client, ["GMV"], mode="query", namespace="profile-k2"
        ))
        await asyncio.sleep(0)
        release.set()

        first_result, second_result = await asyncio.gather(first, second)

        self.assertEqual(first_result[0], second_result[0])
        self.assertEqual(client.aembed_queries.await_count, 1)

    async def test_cancelling_one_waiter_keeps_shared_flight_and_populates_cache(self):
        started = asyncio.Event()
        release = asyncio.Event()
        client = AsyncMock()
        client.profile_fingerprint = "embedding-profile-a"

        async def encode(texts):
            started.set()
            await release.wait()
            return [[.3, .4] for _ in texts]

        client.aembed_queries.side_effect = encode
        cancelled_waiter = asyncio.create_task(embed_texts_cached(
            client, ["GMV"], mode="query", namespace="column"
        ))
        await started.wait()
        surviving_waiter = asyncio.create_task(embed_texts_cached(
            client, ["GMV"], mode="query", namespace="metric"
        ))
        await asyncio.sleep(0)

        cancelled_waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await cancelled_waiter
        release.set()

        vectors, _ = await surviving_waiter
        cached, stats = await embed_texts_cached(
            client, ["GMV"], mode="query", namespace="training"
        )
        self.assertEqual(vectors, cached)
        self.assertEqual(stats, {"hits": 1, "misses": 0})
        self.assertEqual(client.aembed_queries.await_count, 1)

    async def test_vector_key_isolates_text_profile_and_encode_mode(self):
        first = AsyncMock()
        first.profile_fingerprint = "embedding-profile-a"
        first.aembed_queries.side_effect = [
            [[.1]],
            [[.2]],
        ]
        first.aembed_documents.return_value = [[.3]]
        second = AsyncMock()
        second.profile_fingerprint = "embedding-profile-b"
        second.aembed_queries.return_value = [[.4]]

        await embed_texts_cached(first, ["GMV"], mode="query", namespace="column")
        await embed_texts_cached(first, ["AOV"], mode="query", namespace="column")
        await embed_texts_cached(
            first, ["GMV"], mode="documents", namespace="column"
        )
        await embed_texts_cached(second, ["GMV"], mode="query", namespace="column")

        self.assertEqual(first.aembed_queries.await_count, 2)
        first.aembed_documents.assert_awaited_once_with(["GMV"])
        second.aembed_queries.assert_awaited_once_with(["GMV"])

    async def test_vector_text_identity_preserves_whitespace_and_case(self):
        client = AsyncMock()
        client.profile_fingerprint = "embedding-profile-a"
        client.aembed_queries.side_effect = [
            [[.1]],
            [[.2]],
            [[.3]],
        ]

        upper_with_space, _ = await embed_texts_cached(
            client, ["A B"], mode="query", namespace="column"
        )
        upper_without_space, _ = await embed_texts_cached(
            client, ["AB"], mode="query", namespace="metric"
        )
        lower_with_space, _ = await embed_texts_cached(
            client, ["a b"], mode="query", namespace="training"
        )

        self.assertEqual(upper_with_space, [[.1]])
        self.assertEqual(upper_without_space, [[.2]])
        self.assertEqual(lower_with_space, [[.3]])
        self.assertEqual(client.aembed_queries.await_count, 3)

    async def test_vector_text_identity_nfc_normalizes_input_and_key(self):
        client = AsyncMock()
        client.profile_fingerprint = "embedding-profile-a"
        client.aembed_queries.return_value = [[.9]]

        decomposed, _ = await embed_texts_cached(
            client, ["e\u0301"], mode="query", namespace="column"
        )
        composed, stats = await embed_texts_cached(
            client, ["é"], mode="query", namespace="metric"
        )

        self.assertEqual(decomposed, composed)
        client.aembed_queries.assert_awaited_once_with(["é"])
        self.assertEqual(stats, {"hits": 1, "misses": 0})

    async def test_partially_overlapping_batches_encode_each_text_once(self):
        first_started = asyncio.Event()
        second_started = asyncio.Event()
        release = asyncio.Event()
        client = AsyncMock()
        client.profile_fingerprint = "embedding-profile-a"
        calls: list[list[str]] = []
        vectors = {"A": [1.0], "B": [2.0], "C": [3.0]}

        async def encode(texts):
            calls.append(list(texts))
            if len(calls) == 1:
                first_started.set()
            else:
                second_started.set()
            await release.wait()
            return [vectors[text] for text in texts]

        client.aembed_queries.side_effect = encode
        first = asyncio.create_task(embed_texts_cached(
            client, ["A", "B"], mode="query", namespace="column"
        ))
        await first_started.wait()
        second = asyncio.create_task(embed_texts_cached(
            client, ["B", "C"], mode="query", namespace="metric"
        ))
        await second_started.wait()
        release.set()

        first_result, second_result = await asyncio.gather(first, second)

        self.assertEqual(first_result[0], [[1.0], [2.0]])
        self.assertEqual(second_result[0], [[2.0], [3.0]])
        self.assertEqual(calls, [["A", "B"], ["C"]])

    async def test_shared_flight_exception_reaches_all_waiters_then_retries(self):
        started = asyncio.Event()
        release = asyncio.Event()
        client = AsyncMock()
        client.profile_fingerprint = "embedding-profile-a"
        calls = 0

        async def encode(texts):
            nonlocal calls
            calls += 1
            if calls == 1:
                started.set()
                await release.wait()
                raise RuntimeError("shared embedding failure")
            return [[.4] for _ in texts]

        client.aembed_queries.side_effect = encode
        first = asyncio.create_task(embed_texts_cached(
            client, ["GMV"], mode="query", namespace="column"
        ))
        await started.wait()
        second = asyncio.create_task(embed_texts_cached(
            client, ["GMV"], mode="query", namespace="metric"
        ))
        await asyncio.sleep(0)
        release.set()

        failures = await asyncio.gather(first, second, return_exceptions=True)
        retried, stats = await embed_texts_cached(
            client, ["GMV"], mode="query", namespace="training"
        )

        self.assertTrue(all(isinstance(item, RuntimeError) for item in failures))
        self.assertEqual(retried, [[.4]])
        self.assertEqual(stats, {"hits": 0, "misses": 1})
        self.assertEqual(calls, 2)

    async def test_failed_orphan_flight_is_consumed_and_next_call_retries(self):
        started = asyncio.Event()
        release = asyncio.Event()
        failed = asyncio.Event()
        client = AsyncMock()
        client.profile_fingerprint = "embedding-profile-a"
        calls = 0

        async def encode(texts):
            nonlocal calls
            calls += 1
            if calls == 1:
                started.set()
                await release.wait()
                failed.set()
                raise RuntimeError("temporary embedding failure")
            return [[.5, .6] for _ in texts]

        client.aembed_queries.side_effect = encode
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        unhandled: list[dict] = []
        loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
        try:
            waiter = asyncio.create_task(embed_texts_cached(
                client, ["GMV"], mode="query", namespace="column"
            ))
            await started.wait()
            waiter.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiter
            release.set()
            await failed.wait()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            gc.collect()

            vectors, stats = await embed_texts_cached(
                client, ["GMV"], mode="query", namespace="metric"
            )
        finally:
            loop.set_exception_handler(previous_handler)

        self.assertEqual(vectors, [[.5, .6]])
        self.assertEqual(stats, {"hits": 0, "misses": 1})
        self.assertEqual(calls, 2)
        self.assertEqual(unhandled, [])

    async def test_cold_invalidate_detaches_inflight_and_blocks_stale_writeback(self):
        first_started = asyncio.Event()
        second_started = asyncio.Event()
        release_first = asyncio.Event()
        release_second = asyncio.Event()
        client = AsyncMock()
        client.profile_fingerprint = "embedding-profile-a"
        calls = 0

        async def encode(texts):
            nonlocal calls
            calls += 1
            if calls == 1:
                first_started.set()
                await release_first.wait()
                return [[.1] for _ in texts]
            second_started.set()
            await release_second.wait()
            return [[.2] for _ in texts]

        client.aembed_queries.side_effect = encode
        pre_cold = asyncio.create_task(embed_texts_cached(
            client, ["GMV"], mode="query", namespace="column"
        ))
        await first_started.wait()
        invalidate_retrieval_caches()
        post_cold = asyncio.create_task(embed_texts_cached(
            client, ["GMV"], mode="query", namespace="metric"
        ))
        await second_started.wait()

        release_second.set()
        post_result = await post_cold
        release_first.set()
        pre_result = await pre_cold
        cached_result, stats = await embed_texts_cached(
            client, ["GMV"], mode="query", namespace="training"
        )

        self.assertEqual(pre_result[0], [[.1]])
        self.assertEqual(post_result[0], [[.2]])
        self.assertEqual(cached_result, [[.2]])
        self.assertEqual(stats, {"hits": 1, "misses": 0})
        self.assertEqual(calls, 2)

    def test_candidate_key_isolated_by_workspace_and_policy(self):
        generation_token = capture_retrieval_cache_generation()
        first = SimpleNamespace(workspace_id="ws-a", user_id="u1", role="viewer", policies=())
        second = SimpleNamespace(workspace_id="ws-b", user_id="u1", role="viewer", policies=())
        signal_contract = {
            "query_signal_profile": "jieba_tfidf_v1",
            "route_policy": "current_multi_query_v1",
            "ordered_remote_signals": ["地区", "销售额"],
            "query_signal_fingerprint": "sha256:current-signals",
            "cache_namespace": "query-signals:current-contract",
        }
        first_key = candidate_cache_key(
            "column", "各地区销售额", ["地区"], "ws-a",
            policy_scope_fingerprint(first), **signal_contract,
            generation_token=generation_token,
        )
        second_key = candidate_cache_key(
            "column", "各地区销售额", ["地区"], "ws-b",
            policy_scope_fingerprint(second), **signal_contract,
            generation_token=generation_token,
        )
        self.assertNotEqual(first_key, second_key)

    def test_candidate_key_preserves_remote_order_and_profile(self):
        generation_token = capture_retrieval_cache_generation()
        common = {
            "route": "column",
            "query": "各地区销售额",
            "keywords": ["地区", "销售额"],
            "workspace_id": "ws-a",
            "policy_fingerprint": "policy-a",
        }
        first = candidate_cache_key(
            **common,
            query_signal_profile="jieba_tfidf_v1",
            route_policy="current_multi_query_v1",
            ordered_remote_signals=["地区", "销售额"],
            query_signal_fingerprint="sha256:current-signals",
            cache_namespace="query-signals:current-contract",
            generation_token=generation_token,
        )
        reordered = candidate_cache_key(
            **common,
            query_signal_profile="jieba_tfidf_v1",
            route_policy="current_multi_query_v1",
            ordered_remote_signals=["销售额", "地区"],
            query_signal_fingerprint="sha256:current-signals",
            cache_namespace="query-signals:current-contract",
            generation_token=generation_token,
        )
        challenger = candidate_cache_key(
            **common,
            query_signal_profile="domain_signals_v1",
            route_policy="routed_signals_v1",
            ordered_remote_signals=["地区", "销售额"],
            query_signal_fingerprint="sha256:challenger-signals",
            cache_namespace="query-signals:challenger-contract",
            generation_token=generation_token,
        )
        self.assertNotEqual(first, reordered)
        self.assertNotEqual(first, challenger)

    def test_candidate_late_writer_cannot_cross_cold_generation(self):
        arguments = {
            "route": "column",
            "query": "各地区销售额",
            "keywords": ["地区", "销售额"],
            "workspace_id": "ws-a",
            "policy_fingerprint": "policy-a",
            "query_signal_profile": "jieba_tfidf_v1",
            "route_policy": "current_multi_query_v1",
            "ordered_remote_signals": ["地区", "销售额"],
            "query_signal_fingerprint": "sha256:current-signals",
            "cache_namespace": "query-signals:current-contract",
        }
        old_token = capture_retrieval_cache_generation()
        # Simulate a request that captured its token, then read old Meta/LLM
        # sources while a cold invalidation happened before key construction.
        invalidate_retrieval_caches()
        old_key = candidate_cache_key(
            **arguments,
            generation_token=old_token,
        )
        new_token = capture_retrieval_cache_generation()
        new_key = candidate_cache_key(
            **arguments,
            generation_token=new_token,
        )

        old_written = candidate_cache_set(
            old_key,
            {"generation": "old"},
            generation_token=old_token,
        )

        self.assertNotEqual(old_key, new_key)
        self.assertFalse(old_written)
        self.assertIsNone(candidate_cache_get(new_key))
        self.assertTrue(candidate_cache_set(
            new_key,
            {"generation": "new"},
            generation_token=new_token,
        ))
        self.assertEqual(candidate_cache_get(new_key), {"generation": "new"})


class RetrievalCacheMultiLoopTests(unittest.TestCase):
    def setUp(self):
        invalidate_retrieval_caches()

    def test_simultaneous_event_loops_never_share_loop_bound_tasks(self):
        query_settings = FrozenSettingsSection.from_mapping({
            "retrieval_cache_enabled": False,
        })
        barrier = threading.Barrier(2)
        lock = threading.Lock()

        class Client:
            profile_fingerprint = "embedding-profile-a"

            def __init__(self):
                self.calls = 0

            async def aembed_queries(self, texts):
                with lock:
                    self.calls += 1
                await asyncio.to_thread(barrier.wait, 2)
                return [[.7, .8] for _ in texts]

        client = Client()

        def run(namespace):
            return asyncio.run(embed_texts_cached(
                client,
                ["GMV"],
                mode="query",
                namespace=namespace,
                query_settings=query_settings,
            ))

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(run, ("column", "metric")))

        self.assertEqual(results[0][0], results[1][0])
        self.assertEqual(client.calls, 2)


class RecallGenerationContractTests(unittest.TestCase):
    def _assert_route_contract(self, relative_path: str, function_name: str) -> None:
        source = (Path(__file__).parents[1] / relative_path).read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        )
        executable = list(function.body)
        if (
            executable
            and isinstance(executable[0], ast.Expr)
            and isinstance(executable[0].value, ast.Constant)
            and isinstance(executable[0].value.value, str)
        ):
            executable.pop(0)
        first = executable[0]
        self.assertIsInstance(first, ast.Assign)
        self.assertEqual(first.targets[0].id, "cache_generation")
        self.assertEqual(
            first.value.func.id,
            "capture_retrieval_cache_generation",
        )

        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
        for call_name in ("candidate_cache_key", "candidate_cache_set"):
            call = next(
                item
                for item in calls
                if isinstance(item.func, ast.Name) and item.func.id == call_name
            )
            token = next(
                keyword.value
                for keyword in call.keywords
                if keyword.arg == "generation_token"
            )
            self.assertIsInstance(token, ast.Name)
            self.assertEqual(token.id, "cache_generation")

    def test_column_binds_entry_generation_to_candidate_key_and_set(self):
        self._assert_route_contract(
            "app/agent_system/agents/nl2sql/nodes/recall_column.py",
            "recall_column",
        )

    def test_metric_binds_entry_generation_to_candidate_key_and_set(self):
        self._assert_route_contract(
            "app/agent_system/agents/nl2sql/nodes/recall_metric.py",
            "recall_metric",
        )

    def test_training_binds_entry_generation_to_candidate_key_and_set(self):
        self._assert_route_contract(
            "app/agent_system/agents/nl2sql/nodes/recall_training.py",
            "recall_training",
        )


class ContextBudgetTests(unittest.TestCase):
    def setUp(self):
        invalidate_retrieval_caches()

    def test_context_late_writer_cannot_cross_cold_generation(self):
        started = threading.Event()
        release = threading.Event()
        calls = 0
        original = retrieval_budget_module._pack_uncached

        def delayed_pack(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                started.set()
                if not release.wait(2):
                    raise TimeoutError("test did not release context pack")
            return original(*args, **kwargs)

        arguments = ("客户有哪些", [], [], [], {}, "policy-a")
        with patch.object(
            retrieval_budget_module,
            "_pack_uncached",
            side_effect=delayed_pack,
        ):
            with ThreadPoolExecutor(max_workers=1) as executor:
                old_request = executor.submit(pack_prompt_context, *arguments)
                self.assertTrue(started.wait(2))
                invalidate_retrieval_caches()
                release.set()
                old_result = old_request.result(timeout=5)
            new_result = pack_prompt_context(*arguments)
            cached_result = pack_prompt_context(*arguments)

        self.assertFalse(old_result[3]["cache_hit"])
        self.assertFalse(new_result[3]["cache_hit"])
        self.assertTrue(cached_result[3]["cache_hit"])
        self.assertEqual(calls, 2)

    def test_budget_preserves_ir_join_and_display_columns(self):
        query_settings = FrozenSettingsSection.from_mapping({
            "context_schema_token_budget": 150,
            "context_metric_token_budget": 350,
            "context_training_token_budget": 450,
            "context_total_token_budget": 400,
            "context_max_columns_per_table": 3,
            "retrieval_cache_enabled": True,
            "retrieval_cache_ttl_seconds": 120,
            "retrieval_cache_max_entries": 512,
        })
        with self.subTest(settings="tight_context_budget"):
            tables = [{
                "name": "dim_customer", "role": "dimension", "description": "客户维表",
                "columns": [
                    {"name": "customer_id", "role": "primary_key", "type": "int", "description": "客户编号", "alias": [], "examples": [], "is_display": False},
                    {"name": "customer_name", "role": "dimension", "type": "varchar", "description": "客户名称", "alias": ["客户"], "examples": ["Alice"], "is_display": True},
                    *[
                        {"name": f"attribute_{index}", "role": "dimension", "type": "varchar",
                         "description": "很长的低优先级属性说明" * 10, "alias": [], "examples": ["x"] * 10,
                         "is_display": False}
                        for index in range(8)
                    ],
                ],
            }]
            ir = {"dimensions": [{"table": "dim_customer", "column": "customer_name"}],
                  "measures": [], "joins": []}
            packed, _, _, evidence = pack_prompt_context(
                "客户有哪些",
                tables,
                [],
                [],
                ir,
                "policy-a",
                query_settings=query_settings,
            )
            names = {item["name"] for item in packed[0]["columns"]}
            self.assertIn("customer_id", names)
            self.assertIn("customer_name", names)
            self.assertGreater(evidence["trimmed"]["columns"], 0)
            self.assertLess(evidence["after_tokens"]["total"], evidence["before_tokens"]["total"])
            _, _, _, cached = pack_prompt_context(
                "客户有哪些", tables, [], [], ir, "policy-a",
                query_settings=query_settings,
            )
            self.assertTrue(cached["cache_hit"])
            _, _, _, isolated = pack_prompt_context(
                "客户有哪些", tables, [], [], ir, "policy-b",
                query_settings=query_settings,
            )
            self.assertFalse(isolated["cache_hit"])

    def test_budget_keeps_user_supplied_enum_values_not_unmentioned_examples(self):
        tables = [{
            "name": "dim_region",
            "role": "dimension",
            "columns": [{
                "name": "region_name",
                "role": "dimension",
                "type": "varchar",
                "description": "大区",
                "alias": ["地区"],
                "examples": ["东北", "华北", "西南", "华东", "华南"],
                "is_display": True,
            }],
        }]

        packed, _, _, _ = pack_prompt_context(
            "比较华东和华南大区", tables, [], [], {}, "policy-enum",
        )

        self.assertEqual(
            packed[0]["columns"][0]["examples"], ["华东", "华南"],
        )


if __name__ == "__main__":
    unittest.main()
