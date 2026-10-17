import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import app.bootstrap.runtime_lifecycle as lifespan_module


class LifespanDependencyProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_probe_checks_local_embedding_runtime_without_loading_weights(self):
        es = SimpleNamespace(ping=AsyncMock(return_value=True))
        qdrant = SimpleNamespace(get_collections=AsyncMock())
        embedding = SimpleNamespace(validate_runtime=Mock(return_value={
            "embedding_mode": "dense_sparse",
        }))
        sql_llm = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content="READY")))
        fast_llm = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content="READY")))

        with (
            patch.object(lifespan_module.es_client_manager, "client", es),
            patch.object(lifespan_module.qdrant_client_manager, "client", qdrant),
            patch.object(lifespan_module.embedding_client_manager, "client", embedding),
            patch.object(
                lifespan_module,
                "llm_route_manifest",
                return_value={
                    "sql": {"model_name": "strong", "deployment_id": "strong"},
                    "correction": {"model_name": "strong", "deployment_id": "strong"},
                    "keyword": {"model_name": "fast", "deployment_id": "fast"},
                },
            ),
            patch.object(lifespan_module, "get_task_llm", side_effect=[sql_llm, fast_llm]),
        ):
            await lifespan_module._probe_external_dependencies()

        es.ping.assert_awaited_once()
        qdrant.get_collections.assert_awaited_once()
        embedding.validate_runtime.assert_called_once_with()
        sql_llm.ainvoke.assert_awaited_once()
        fast_llm.ainvoke.assert_awaited_once()

    async def test_probe_shared_model_uses_widest_route_timeout(self):
        es = SimpleNamespace(ping=AsyncMock(return_value=True))
        qdrant = SimpleNamespace(get_collections=AsyncMock())
        embedding = SimpleNamespace(validate_runtime=Mock(return_value={}))
        llm = SimpleNamespace(ainvoke=AsyncMock(
            return_value=SimpleNamespace(content="READY")
        ))

        with (
            patch.object(lifespan_module.es_client_manager, "client", es),
            patch.object(lifespan_module.qdrant_client_manager, "client", qdrant),
            patch.object(lifespan_module.embedding_client_manager, "client", embedding),
            patch.object(
                lifespan_module,
                "llm_route_manifest",
                return_value={
                    "candidate": {
                        "model_name": "shared",
                        "deployment_id": "shared-deployment",
                        "timeout_seconds": 2.5,
                    },
                    "sql": {
                        "model_name": "shared",
                        "deployment_id": "shared-deployment",
                        "timeout_seconds": 18.0,
                    },
                },
            ),
            patch.object(lifespan_module, "get_task_llm", return_value=llm) as get_llm,
            patch.object(
                lifespan_module,
                "invoke_llm",
                new_callable=AsyncMock,
                return_value=SimpleNamespace(
                    value=SimpleNamespace(content="READY")
                ),
            ) as invoke,
        ):
            await lifespan_module._probe_external_dependencies()

        get_llm.assert_called_once_with("sql")
        self.assertEqual(invoke.await_args.kwargs["timeout_seconds"], 18.0)

    async def test_probe_warmup_checks_dense_and_sparse_contract_when_enabled(self):
        es = SimpleNamespace(ping=AsyncMock(return_value=True))
        qdrant = SimpleNamespace(get_collections=AsyncMock())
        embedding = SimpleNamespace(
            validate_runtime=Mock(return_value={}),
            warmup=AsyncMock(return_value={"dense_size": 1024, "sparse_nonzero": 4}),
        )
        llm = SimpleNamespace(ainvoke=AsyncMock(
            return_value=SimpleNamespace(content="READY")
        ))
        with (
            patch.object(lifespan_module.es_client_manager, "client", es),
            patch.object(lifespan_module.qdrant_client_manager, "client", qdrant),
            patch.object(lifespan_module.embedding_client_manager, "client", embedding),
            patch.object(lifespan_module.app_config.embedding, "warmup_on_startup", True),
            patch.object(
                lifespan_module,
                "llm_route_manifest",
                return_value={
                    "sql": {"model_name": "strong", "deployment_id": "strong"}
                },
            ),
            patch.object(lifespan_module, "get_task_llm", return_value=llm),
        ):
            await lifespan_module._probe_external_dependencies()

        embedding.warmup.assert_awaited_once_with()

    async def test_probe_distinguishes_same_model_with_different_credential_identity(self):
        es = SimpleNamespace(ping=AsyncMock(return_value=True))
        qdrant = SimpleNamespace(get_collections=AsyncMock())
        embedding = SimpleNamespace(validate_runtime=Mock(return_value={}))
        first_llm = object()
        second_llm = object()

        with (
            patch.object(lifespan_module.es_client_manager, "client", es),
            patch.object(lifespan_module.qdrant_client_manager, "client", qdrant),
            patch.object(lifespan_module.embedding_client_manager, "client", embedding),
            patch.object(
                lifespan_module,
                "llm_route_manifest",
                return_value={
                    "sql": {
                        "model_name": "same-model",
                        "provider": "dashscope",
                        "deployment_id": "credential-a-deployment",
                        "timeout_seconds": 18.0,
                    },
                    "correction": {
                        "model_name": "same-model",
                        "provider": "dashscope",
                        "deployment_id": "credential-b-deployment",
                        "timeout_seconds": 16.0,
                    },
                },
            ),
            patch.object(
                lifespan_module,
                "get_task_llm",
                side_effect=[first_llm, second_llm],
            ) as get_llm,
            patch.object(
                lifespan_module,
                "invoke_llm",
                new_callable=AsyncMock,
                return_value=SimpleNamespace(value=SimpleNamespace(content="READY")),
            ) as invoke,
        ):
            await lifespan_module._probe_external_dependencies()

        self.assertEqual(get_llm.call_count, 2)
        self.assertEqual(invoke.await_count, 2)

    async def test_close_attempts_embedding_after_an_earlier_close_failure(self):
        llm_close = AsyncMock()
        qdrant_close = AsyncMock(side_effect=RuntimeError("qdrant unavailable"))
        meta_close = AsyncMock()
        dw_close = AsyncMock()
        es_close = AsyncMock()
        embedding_close = AsyncMock()

        with (
            patch.object(lifespan_module, "close_task_llm_cache", llm_close),
            patch.object(lifespan_module.qdrant_client_manager, "close", qdrant_close),
            patch.object(lifespan_module.meta_mysql_client_manager, "close", meta_close),
            patch.object(lifespan_module.dw_mysql_client_manager, "close", dw_close),
            patch.object(lifespan_module.es_client_manager, "close", es_close),
            patch.object(lifespan_module.embedding_client_manager, "close", embedding_close),
            patch.object(lifespan_module.logger, "exception"),
        ):
            failures = await lifespan_module._close_external_dependencies()

        self.assertEqual([name for name, _ in failures], ["Qdrant"])
        llm_close.assert_awaited_once()
        meta_close.assert_awaited_once()
        dw_close.assert_awaited_once()
        es_close.assert_awaited_once()
        embedding_close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
