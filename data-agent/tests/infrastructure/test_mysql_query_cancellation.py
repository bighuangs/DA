import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.config.app_config import DBConfig
from app.foundation.concurrency import SingleFlightLeaderCancelledError
from app.infrastructure.cache.result_cache import result_cache
from app.infrastructure.clients.mysql_query_cancellation import (
    AsyncMyServerQueryCanceller,
)
from app.infrastructure.persistence.mysql.repositories.dw_mysql_repository import (
    DWMysqlRepository,
)


class _DriverConnection:
    def __init__(
        self,
        thread_id: int = 4242,
        *,
        on_close=None,
    ):
        self._thread_id = thread_id
        self._on_close = on_close
        self.close_calls = 0

    def thread_id(self) -> int:
        return self._thread_id

    def close(self) -> None:
        self.close_calls += 1
        if self._on_close is not None:
            self._on_close()


class _RawConnection:
    def __init__(self, driver_connection):
        self.driver_connection = driver_connection


class _SessionConnection:
    def __init__(self, driver_connection):
        self._raw_connection = _RawConnection(driver_connection)

    async def get_raw_connection(self):
        return self._raw_connection


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class _FakeSession:
    def __init__(
        self,
        *,
        waits_for_release: bool,
        ignores_cancellation: bool = False,
        invalidate_waits: bool = False,
    ):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()
        self.waits_for_release = waits_for_release
        self.ignores_cancellation = ignores_cancellation
        self.invalidate_waits = invalidate_waits
        self.invalidate_release = asyncio.Event()
        self.invalidations = 0
        self.active_executions = 0
        self.statement = ""
        self.driver_connection = _DriverConnection(on_close=self.release.set)
        self._connection = _SessionConnection(self.driver_connection)

    async def connection(self):
        return self._connection

    def get_bind(self):
        return SimpleNamespace(url=SimpleNamespace(database="warehouse"))

    async def execute(self, statement):
        self.statement = str(statement)
        self.started.set()
        self.active_executions += 1
        try:
            if self.waits_for_release:
                while True:
                    try:
                        await self.release.wait()
                        raise ConnectionError("target connection terminated")
                    except asyncio.CancelledError:
                        if not self.ignores_cancellation:
                            raise
                        # 模拟驱动卡死并暂时吞掉任务取消，验证清理路径仍有上界。
                        continue
            return _Result([{"value": 1}])
        finally:
            self.active_executions -= 1
            self.finished.set()

    async def invalidate(self):
        if self.active_executions:
            raise AssertionError("invalidate 不得与 execute 并发")
        self.invalidations += 1
        if self.invalidate_waits:
            await self.invalidate_release.wait()


class _FakeCanceller:
    def __init__(
        self,
        session: _FakeSession,
        *,
        error: Exception | None = None,
        releases_query: bool = True,
    ):
        self.session = session
        self.error = error
        self.releases_query = releases_query
        self.thread_ids: list[int] = []

    async def cancel(self, thread_id: int):
        self.thread_ids.append(thread_id)
        if self.error is not None:
            raise self.error
        if self.releases_query:
            self.session.release.set()


class _BlockingCanceller:
    def __init__(self):
        self.thread_ids: list[int] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def cancel(self, thread_id: int):
        self.thread_ids.append(thread_id)
        self.started.set()
        await self.release.wait()


class MysqlQueryCancellationTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _repo(session, canceller):
        return DWMysqlRepository(
            session,
            server_query_canceller=canceller,
            cancellation_control_timeout_seconds=0.05,
            cancellation_drain_timeout_seconds=0.05,
        )

    async def test_success_does_not_kill_or_invalidate(self):
        session = _FakeSession(waits_for_release=False)
        canceller = _FakeCanceller(session)
        repo = self._repo(session, canceller)

        rows = await repo.execute_sql_for_evaluation(
            "SELECT 1",
            timeout_seconds=0.1,
        )

        self.assertEqual(rows, [{"value": 1}])
        self.assertEqual(canceller.thread_ids, [])
        self.assertEqual(session.invalidations, 0)
        self.assertIsNone(repo.last_server_cancellation)

    async def test_timeout_kills_exact_thread_and_invalidates_session(self):
        session = _FakeSession(waits_for_release=True)
        canceller = _FakeCanceller(session)
        repo = self._repo(session, canceller)

        with self.assertRaises(asyncio.TimeoutError):
            await repo.execute_sql_for_evaluation(
                "SELECT 1",
                timeout_seconds=0.01,
            )

        self.assertEqual(canceller.thread_ids, [4242])
        self.assertEqual(session.invalidations, 1)
        self.assertEqual(
            repo.last_server_cancellation,
            {
                "schema": "mysql-server-cancellation-v1",
                "server_cancelled": True,
                "transport_closed": False,
                "query_drained": True,
                "session_invalidated": True,
                "error_type": None,
            },
        )

    async def test_validate_sql_explain_uses_same_cancellation_boundary(self):
        session = _FakeSession(waits_for_release=True)
        canceller = _FakeCanceller(session)
        repo = self._repo(session, canceller)

        with self.assertRaises(asyncio.TimeoutError):
            await repo.validate_sql("SELECT 1", timeout_seconds=0.01)

        self.assertIn("EXPLAIN SELECT 1", session.statement)
        self.assertEqual(canceller.thread_ids, [4242])
        self.assertEqual(session.invalidations, 1)

    async def test_cache_follower_maps_cancelled_leader_to_timeout(self):
        session = _FakeSession(waits_for_release=True)
        canceller = _FakeCanceller(session)
        repo = self._repo(session, canceller)
        result_cache.clear()

        with patch(
            "app.infrastructure.persistence.mysql.repositories."
            "dw_mysql_repository.app_config.query.cache_enabled",
            True,
        ):
            leader = asyncio.create_task(
                repo.execute_sql("SELECT 1", timeout_seconds=1.0)
            )
            await session.started.wait()
            follower = asyncio.create_task(
                repo.execute_sql("SELECT 1", timeout_seconds=0.2)
            )
            await asyncio.sleep(0)

            leader.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await leader
            with self.assertRaises(SingleFlightLeaderCancelledError) as caught:
                await follower

        self.assertIsInstance(caught.exception, TimeoutError)
        self.assertEqual(canceller.thread_ids, [4242])
        self.assertEqual(session.invalidations, 1)
        result_cache.clear()

    async def test_external_cancellation_runs_same_cleanup_and_propagates(self):
        session = _FakeSession(waits_for_release=True)
        canceller = _FakeCanceller(session)
        repo = self._repo(session, canceller)
        task = asyncio.create_task(
            repo.execute_sql_for_evaluation("SELECT 1", timeout_seconds=1.0)
        )
        await session.started.wait()

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertEqual(canceller.thread_ids, [4242])
        self.assertEqual(session.invalidations, 1)
        self.assertTrue(repo.last_server_cancellation["server_cancelled"])

    async def test_repeated_external_cancellation_cannot_interrupt_cleanup(self):
        session = _FakeSession(waits_for_release=True)
        canceller = _BlockingCanceller()
        repo = DWMysqlRepository(
            session,
            server_query_canceller=canceller,
            cancellation_control_timeout_seconds=0.02,
            cancellation_drain_timeout_seconds=0.02,
        )
        task = asyncio.create_task(
            repo.execute_sql_for_evaluation("SELECT 1", timeout_seconds=1.0)
        )
        await session.started.wait()

        task.cancel()
        await canceller.started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertEqual(canceller.thread_ids, [4242])
        self.assertEqual(session.invalidations, 1)
        self.assertTrue(repo.last_server_cancellation["query_drained"])
        self.assertTrue(repo.last_server_cancellation["session_invalidated"])
        self.assertFalse(any(
            other.get_name() in {
                "dw-mysql-read-query",
                "dw-mysql-cancellation-cleanup",
            }
            and not other.done()
            for other in asyncio.all_tasks()
        ))

    async def test_kill_failure_keeps_timeout_and_still_invalidates(self):
        session = _FakeSession(waits_for_release=True)
        canceller = _FakeCanceller(session, error=RuntimeError("control failed"))
        repo = self._repo(session, canceller)

        with self.assertRaises(asyncio.TimeoutError):
            await repo.execute_sql_for_evaluation(
                "SELECT 1",
                timeout_seconds=0.01,
            )

        self.assertEqual(canceller.thread_ids, [4242])
        self.assertEqual(session.invalidations, 1)
        self.assertFalse(repo.last_server_cancellation["server_cancelled"])
        self.assertTrue(repo.last_server_cancellation["query_drained"])
        self.assertEqual(
            repo.last_server_cancellation["error_type"],
            "RuntimeError",
        )

    async def test_stuck_query_task_does_not_exceed_cleanup_grace(self):
        session = _FakeSession(
            waits_for_release=True,
            ignores_cancellation=True,
        )
        canceller = _FakeCanceller(session, releases_query=False)
        repo = DWMysqlRepository(
            session,
            server_query_canceller=canceller,
            cancellation_control_timeout_seconds=0.02,
            cancellation_drain_timeout_seconds=0.02,
        )

        loop = asyncio.get_running_loop()
        started_at = loop.time()
        with self.assertRaises(asyncio.TimeoutError):
            await repo.execute_sql_for_evaluation(
                "SELECT 1",
                timeout_seconds=0.01,
            )
        elapsed = loop.time() - started_at

        self.assertLess(elapsed, 0.30)
        self.assertEqual(canceller.thread_ids, [4242])
        self.assertEqual(session.invalidations, 1)
        self.assertTrue(repo.last_server_cancellation["transport_closed"])
        self.assertTrue(repo.last_server_cancellation["query_drained"])
        self.assertTrue(repo.last_server_cancellation["session_invalidated"])
        self.assertIsNone(repo.last_server_cancellation["error_type"])
        self.assertEqual(session.driver_connection.close_calls, 1)
        self.assertFalse(any(
            task.get_name() == "dw-mysql-read-query" and not task.done()
            for task in asyncio.all_tasks()
        ))

    async def test_stuck_control_connection_is_bounded(self):
        session = _FakeSession(waits_for_release=True)
        canceller = _BlockingCanceller()
        repo = DWMysqlRepository(
            session,
            server_query_canceller=canceller,
            cancellation_control_timeout_seconds=0.02,
            cancellation_drain_timeout_seconds=0.02,
        )

        loop = asyncio.get_running_loop()
        started_at = loop.time()
        with self.assertRaises(asyncio.TimeoutError):
            await repo.execute_sql_for_evaluation(
                "SELECT 1",
                timeout_seconds=0.01,
            )
        elapsed = loop.time() - started_at

        self.assertLess(elapsed, 0.30)
        self.assertEqual(canceller.thread_ids, [4242])
        self.assertEqual(session.invalidations, 1)
        self.assertFalse(repo.last_server_cancellation["server_cancelled"])
        self.assertTrue(repo.last_server_cancellation["query_drained"])
        self.assertEqual(
            repo.last_server_cancellation["error_type"],
            "TimeoutError",
        )

    async def test_stuck_session_invalidation_is_bounded(self):
        session = _FakeSession(
            waits_for_release=True,
            invalidate_waits=True,
        )
        canceller = _FakeCanceller(session)
        repo = DWMysqlRepository(
            session,
            server_query_canceller=canceller,
            cancellation_control_timeout_seconds=0.02,
            cancellation_drain_timeout_seconds=0.02,
        )

        loop = asyncio.get_running_loop()
        started_at = loop.time()
        with self.assertRaises(asyncio.TimeoutError):
            await repo.execute_sql_for_evaluation(
                "SELECT 1",
                timeout_seconds=0.01,
            )
        elapsed = loop.time() - started_at

        self.assertLess(elapsed, 0.30)
        self.assertEqual(session.invalidations, 1)
        self.assertTrue(repo.last_server_cancellation["query_drained"])
        self.assertFalse(repo.last_server_cancellation["session_invalidated"])
        self.assertEqual(
            repo.last_server_cancellation["error_type"],
            "TimeoutError",
        )

    async def test_default_canceller_uses_current_session_url(self):
        session = SimpleNamespace(
            get_bind=lambda: SimpleNamespace(
                url=SimpleNamespace(
                    drivername="mysql+asyncmy",
                    host="db.internal",
                    port=3307,
                    username="reader",
                    password="test-secret",
                    database="warehouse",
                )
            )
        )
        repo = DWMysqlRepository(
            session,
            cancellation_control_timeout_seconds=0.7,
        )

        canceller = repo._server_canceller_for_session()

        self.assertIsInstance(canceller, AsyncMyServerQueryCanceller)
        self.assertEqual(canceller._config.host, "db.internal")
        self.assertEqual(canceller._config.port, 3307)
        self.assertEqual(canceller._config.user, "reader")
        self.assertEqual(canceller._config.database, "warehouse")
        self.assertEqual(canceller._operation_timeout_seconds, 0.7)

    async def test_default_canceller_rejects_non_asyncmy_session(self):
        session = SimpleNamespace(
            get_bind=lambda: SimpleNamespace(
                url=SimpleNamespace(drivername="sqlite+aiosqlite")
            )
        )
        repo = DWMysqlRepository(session)

        with self.assertRaisesRegex(RuntimeError, "mysql\\+asyncmy"):
            repo._server_canceller_for_session()

    async def test_asyncmy_canceller_uses_separate_connection_and_closes_it(self):
        connection = unittest.mock.Mock()
        connection.kill = AsyncMock()
        config = DBConfig(
            host="127.0.0.1",
            port=3306,
            user="reader",
            password="test-secret",
            database="dw",
        )
        canceller = AsyncMyServerQueryCanceller(
            config,
            operation_timeout_seconds=0.5,
        )

        with patch(
            "app.infrastructure.clients.mysql_query_cancellation.asyncmy.connect",
            new=AsyncMock(return_value=connection),
        ) as connect:
            await canceller.cancel(99)

        connect.assert_awaited_once()
        self.assertEqual(connect.await_args.kwargs["password"], "test-secret")
        connection.kill.assert_awaited_once_with(99)
        connection.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
