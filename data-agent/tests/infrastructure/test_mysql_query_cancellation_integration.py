import asyncio
import os
import unittest

from sqlalchemy import text

from app.config.app_config import app_config
from app.infrastructure.clients.mysql_client_manager import MysqlClientManager
from app.infrastructure.clients.mysql_query_cancellation import AsyncMyServerQueryCanceller
from app.infrastructure.persistence.mysql.repositories.dw_mysql_repository import (
    DWMysqlRepository,
)


@unittest.skipUnless(
    os.getenv("DA_RUN_MYSQL_CANCELLATION_INTEGRATION") == "1",
    "需要显式启用本机 MySQL 只读取消集成测试",
)
class MysqlQueryCancellationIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_server_query_is_killed_and_shared_session_recovers(self):
        """终止只读 SLEEP，并证明同一评测会话可换新连接继续执行。"""

        manager = MysqlClientManager(app_config.db_dw)
        manager.init()
        try:
            async with manager.session_factory() as session:
                warmup = await session.execute(text("SELECT 1"))
                self.assertEqual(int(warmup.scalar_one()), 1)
                repo = DWMysqlRepository(
                    session,
                    cancellation_control_timeout_seconds=2.0,
                    cancellation_drain_timeout_seconds=1.0,
                )
                target_thread_id, _ = await repo._driver_query_identity()

                with self.assertRaises(asyncio.TimeoutError):
                    await repo._await_bounded(
                        repo._execute_cancellable_statement(
                            text("SELECT SLEEP(2)")
                        ),
                        0.2,
                    )

                cancellation = repo.last_server_cancellation or {}
                self.assertTrue(cancellation.get("server_cancelled"))
                self.assertTrue(cancellation.get("query_drained"))
                self.assertTrue(cancellation.get("session_invalidated"))
                self.assertIsInstance(
                    repo._server_query_canceller,
                    AsyncMyServerQueryCanceller,
                )

                recovered = await session.execute(text("SELECT 1"))
                self.assertEqual(int(recovered.scalar_one()), 1)
                remaining = await session.execute(
                    text(
                        "SELECT COUNT(*) FROM performance_schema.processlist "
                        "WHERE ID = :thread_id"
                    ),
                    {"thread_id": target_thread_id},
                )
                self.assertEqual(int(remaining.scalar_one()), 0)

            self.assertEqual(manager.engine.pool.checkedout(), 0)
        finally:
            await manager.close()


if __name__ == "__main__":
    unittest.main()
