import asyncio
import unittest

from app.agent_system.governance.resilience import (
    AsyncCapacityGovernor,
    AsyncSingleFlight,
    CapacityExceededError,
    optional_dependency,
)
from app.foundation.concurrency import SingleFlightLeaderCancelledError
from evals.legacy.resilience_suite import certification_status, run_suite, validate_suite


class ResilienceContractTest(unittest.TestCase):
    def test_contract_has_required_categories_and_23_cases(self):
        contract = validate_suite()
        self.assertTrue(contract["valid"], contract["errors"])
        self.assertEqual(contract["total"], 23)
        self.assertEqual(
            set(contract["categories"]),
            {"dependency", "concurrency", "cancellation", "resource", "cache", "soak"},
        )

    def test_full_deterministic_suite_passes_without_claiming_long_soak(self):
        report = asyncio.run(run_suite(write=False))
        self.assertEqual(report["summary"]["passed"], 23)
        self.assertTrue(report["summary"]["gate_passed"])
        self.assertEqual(report["certification"]["status"], "not_run")
        self.assertFalse(report["certification"]["live_dependencies_evaluated"])

    def test_1800_seconds_only_certifies_in_process_evidence(self):
        self.assertEqual(certification_status(1800), "in_process_soak_certified")
        self.assertEqual(certification_status(60), "in_process_smoke_only")
        self.assertEqual(certification_status(0), "not_run")


class CapacityGovernorTest(unittest.IsolatedAsyncioTestCase):
    async def test_workspace_peak_is_bounded_and_all_leases_release(self):
        governor = AsyncCapacityGovernor(10, 2, 1)

        async def worker():
            lease = await governor.acquire("workspace-a")
            async with lease:
                await asyncio.sleep(0.005)

        await asyncio.gather(*(worker() for _ in range(8)))
        snapshot = await governor.snapshot()
        self.assertEqual(snapshot.active, 0)
        self.assertEqual(snapshot.completed, 8)
        self.assertLessEqual(snapshot.peak, 2)

    async def test_saturated_workspace_is_rejected_and_first_lease_survives(self):
        governor = AsyncCapacityGovernor(2, 1, 0.005)
        lease = await governor.acquire("workspace-a")
        with self.assertRaises(CapacityExceededError):
            await governor.acquire("workspace-a")
        self.assertEqual(await governor.workspace_active("workspace-a"), 1)
        await lease.release()
        self.assertEqual((await governor.snapshot()).active, 0)

    async def test_cancelled_lease_is_released(self):
        governor = AsyncCapacityGovernor(1, 1, 0.1)
        entered = asyncio.Event()

        async def worker():
            lease = await governor.acquire("workspace-a")
            async with lease:
                entered.set()
                await asyncio.sleep(10)

        task = asyncio.create_task(worker())
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual((await governor.snapshot()).active, 0)


class SingleFlightTest(unittest.IsolatedAsyncioTestCase):
    async def test_same_key_concurrency_executes_factory_once(self):
        flight = AsyncSingleFlight()
        calls = 0

        async def factory():
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.01)
            return [{"value": 1}]

        values = await asyncio.gather(*(flight.run("key", factory) for _ in range(12)))
        values[0][0]["value"] = 2
        self.assertEqual(calls, 1)
        self.assertTrue(all(value[0]["value"] == 1 for value in values[1:]))
        self.assertEqual(await flight.inflight(), 0)

    async def test_cancelled_leader_does_not_cancel_follower(self):
        flight = AsyncSingleFlight()
        started = asyncio.Event()

        async def factory():
            started.set()
            await asyncio.Event().wait()

        leader = asyncio.create_task(flight.run("key", factory))
        await started.wait()
        follower = asyncio.create_task(flight.run("key", factory))
        await asyncio.sleep(0)

        leader.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await leader
        with self.assertRaises(SingleFlightLeaderCancelledError):
            await follower
        self.assertEqual(await flight.inflight(), 0)

    async def test_follower_timeout_does_not_cancel_leader(self):
        flight = AsyncSingleFlight()
        started = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def factory():
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return 7

        leader = asyncio.create_task(flight.run("key", factory))
        await started.wait()
        with self.assertRaises(asyncio.TimeoutError):
            await flight.run(
                "key",
                factory,
                follower_timeout_seconds=0.01,
            )
        release.set()
        self.assertEqual(await leader, 7)
        self.assertEqual(calls, 1)

    async def test_optional_dependency_preserves_cancellation(self):
        started = asyncio.Event()

        async def slow():
            started.set()
            await asyncio.sleep(10)

        task = asyncio.create_task(optional_dependency("es", slow, fallback=[], timeout_seconds=20))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task


if __name__ == "__main__":
    unittest.main()
