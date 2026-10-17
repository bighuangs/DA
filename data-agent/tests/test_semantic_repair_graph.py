import asyncio
import unittest
from types import SimpleNamespace

from langgraph.graph import END

from app.agent_system.agents.nl2sql.graph import _route_after_execute, _route_after_validate
from app.agent_system.agents.nl2sql.nodes.correct_sql import build_correction_policy
from app.agent_system.agents.nl2sql.nodes.execute_sql import execute_sql
from app.agent_system.governance.deadline import initialize_request_deadline
from app.agent_system.agents.nl2sql.services.semantic_repair import set_runtime_profile
from app.domain.query.safety.semantic_violation import violation_from_exception


class SemanticRepairGraphTests(unittest.TestCase):
    def tearDown(self):
        set_runtime_profile(None)

    def test_security_and_cost_violations_never_enter_correction_model(self):
        for code in ("security.policy_violation", "cost.budget_exceeded"):
            route = _route_after_validate({
                "error": "blocked",
                "retry_count": 0,
                "validation_evidence": {
                    "violation": {
                        "violation_code": code,
                        "repairability": "reject",
                        "safe_to_retry": False,
                    },
                },
            })
            self.assertEqual(END, route)

    def test_retryable_violation_routes_once_to_correction(self):
        route = _route_after_validate({
            "error": "semantic mismatch",
            "retry_count": 0,
            "validation_evidence": {
                "violation": {
                    "violation_code": "semantic.grouping.missing_dimension",
                    "repairability": "llm",
                    "safe_to_retry": True,
                },
            },
        })
        self.assertEqual("correct_sql", route)

    def test_validation_timeout_is_terminal_and_never_rewrites_sql(self):
        violation = violation_from_exception(TimeoutError("EXPLAIN deadline"))

        self.assertEqual(
            violation.violation_code,
            "sql.validation_deadline_exhausted",
        )
        self.assertEqual(violation.repairability, "reject")
        self.assertFalse(violation.safe_to_retry)
        self.assertEqual(
            _route_after_validate({
                "error": "EXPLAIN deadline",
                "retry_count": 0,
                "validation_evidence": {
                    "violation": violation.to_dict(),
                },
            }),
            END,
        )

    def test_targeted_profile_builds_bounded_directive(self):
        set_runtime_profile("targeted_correction_v1")
        policy = build_correction_policy({
            "violation": {
                "violation_code": "semantic_ir.limit_mismatch",
                "repairability": "llm",
                "repair_hint_code": "restore_ir_limit",
                "safe_to_retry": True,
            },
        })
        self.assertTrue(policy["targeted"])
        self.assertFalse(policy["upstream_first"])
        self.assertIn("LIMIT", policy["instruction"])

    def test_default_profile_keeps_known_semantic_directive_and_authorized_boundary(self):
        set_runtime_profile("registry_same_rules_v1")
        policy = build_correction_policy({
            "violation": {
                "violation_code": "semantic.time.full_period_filter_required",
                "repairability": "llm",
                "repair_hint_code": "join_date_dimension_and_filter_full_year",
                "safe_to_retry": True,
            },
        })

        self.assertFalse(policy["targeted"])
        self.assertIn("完整年度", policy["instruction"])
        self.assertIn("已提供且元数据可证明", policy["boundary_instruction"])


class ExecutionBoundaryTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _runtime(repo, events):
        return SimpleNamespace(
            context={"dw_mysql_repo": repo},
            stream_writer=events.append,
        )

    async def test_execution_deadline_is_terminal_and_visible(self):
        class SlowRepo:
            def __init__(self):
                self.timeout_seconds = None

            async def execute_sql(self, sql, *, timeout_seconds=None):
                del sql
                self.timeout_seconds = timeout_seconds
                raise TimeoutError("repository deadline")

        deadline = initialize_request_deadline(budget_ms=20)
        deadline["phase_budgets_ms"]["execution"] = 5
        events = []
        repo = SlowRepo()
        output = await execute_sql(
            {
                "sql": "SELECT 1",
                "retry_count": 0,
                "table_infos": [],
                "request_deadline": deadline,
            },
            self._runtime(repo, events),
        )
        violation = output["validation_evidence"]["violation"]
        self.assertEqual(
            violation["violation_code"],
            "execution.deadline_exhausted",
        )
        self.assertEqual(violation["repairability"], "reject")
        self.assertFalse(violation["safe_to_retry"])
        self.assertEqual(_route_after_execute(output), END)
        self.assertTrue(any(event.get("error") for event in events))
        self.assertGreater(repo.timeout_seconds, 0)
        self.assertLessEqual(repo.timeout_seconds, 0.005)

    async def test_only_sql_shape_execution_errors_enter_correction(self):
        class ShapeRepo:
            async def execute_sql(self, sql, *, timeout_seconds=None):
                del sql
                del timeout_seconds
                raise RuntimeError("Unknown column 'missing'")

        output = await execute_sql(
            {
                "sql": "SELECT missing FROM fact_order",
                "retry_count": 0,
                "table_infos": [],
                "request_deadline": initialize_request_deadline(),
            },
            self._runtime(ShapeRepo(), []),
        )
        violation = output["validation_evidence"]["violation"]
        self.assertEqual(violation["repairability"], "llm")
        self.assertTrue(violation["safe_to_retry"])
        self.assertEqual(_route_after_execute(output), "correct_sql")

    async def test_dependency_execution_errors_never_rewrite_sql(self):
        class DependencyRepo:
            async def execute_sql(self, sql, *, timeout_seconds=None):
                del sql
                del timeout_seconds
                raise ConnectionError("database connection lost")

        output = await execute_sql(
            {
                "sql": "SELECT 1",
                "retry_count": 0,
                "table_infos": [],
                "request_deadline": initialize_request_deadline(),
            },
            self._runtime(DependencyRepo(), []),
        )
        violation = output["validation_evidence"]["violation"]
        self.assertEqual(violation["repairability"], "upstream")
        self.assertFalse(violation["safe_to_retry"])
        self.assertEqual(_route_after_execute(output), END)


if __name__ == "__main__":
    unittest.main()
