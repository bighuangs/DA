import asyncio
import unittest
from types import SimpleNamespace

from app.infrastructure.clients.embedding_client_manager import EmbeddingClientManager
from app.infrastructure.clients.hybrid_embedding_client import HybridEmbeddingRuntimeError


class _AsyncClient:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class _RetryCloseClient:
    def __init__(self):
        self.attempts = 0
        self.accepting_requests = False

    async def close(self):
        self.attempts += 1
        if self.attempts == 1:
            raise HybridEmbeddingRuntimeError("active inference drain timeout")


class EmbeddingClientManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_close_releases_local_hybrid_client(self):
        manager = EmbeddingClientManager(SimpleNamespace())
        client = _AsyncClient()
        manager.client = client

        await manager.close()

        self.assertTrue(client.closed)
        self.assertIsNone(manager.client)

    async def test_failed_close_retains_client_and_blocks_reinitialization(self):
        manager = EmbeddingClientManager(SimpleNamespace())
        client = _RetryCloseClient()
        manager.client = client

        with self.assertRaisesRegex(
            HybridEmbeddingRuntimeError,
            "drain timeout",
        ):
            await manager.close()
        self.assertIs(manager.client, client)
        with self.assertRaisesRegex(
            HybridEmbeddingRuntimeError,
            "禁止初始化新实例",
        ):
            manager.init()

        await manager.close()
        self.assertEqual(client.attempts, 2)
        self.assertIsNone(manager.client)


class EmbeddingClientManagerMultiLoopTests(unittest.TestCase):
    def test_repeated_init_close_uses_fresh_clients_across_event_loops(self):
        manager = EmbeddingClientManager(SimpleNamespace(runtime_mode="serial"))
        clients = []

        for _ in range(3):
            clients.append(manager.init())
            asyncio.run(manager.close())
            self.assertIsNone(manager.client)

        self.assertEqual(len({id(client) for client in clients}), 3)


if __name__ == "__main__":
    unittest.main()
