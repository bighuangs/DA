import unittest
from unittest.mock import patch

from app.agent_system.agents.nl2sql.nodes.build_semantic_ir import build_semantic_ir_node
from app.agent_system.agents.nl2sql.nodes.filter_table import filter_table
from app.agent_system.agents.nl2sql.nodes.generate_sql import generate_sql
from app.agent_system.agents.nl2sql.nodes.validate_sql import validate_sql
from app.adapters.outbound.agent.nl2sql_capabilities import (
    Nl2SqlLLMAdapter,
    build_nl2sql_settings,
)
from app.agent_system.agents.nl2sql.settings import override_settings
from app.agent_system.agents.nl2sql.services.model_sql_policy import (
    decide_model_sql_policy,
    deterministic_model_fallback_allowed,
    force_model_sql_policy,
)


BASE_SETTINGS = build_nl2sql_settings()


class CorrectnessFirstModelSqlPolicyTests(unittest.TestCase):

    @staticmethod
    def _closed_simple_ir():
        return {
            "intent": "aggregate",
            "semantic_family": "aggregate",
            "compiler": {
                "supported": True,
                "mode": "deterministic_sqlglot_v2",
            },
            "semantic_coverage": {
                "coverage_score": 1.0,
                "detected_slots": ["measure:order_count"],
                "unresolved_slots": [],
            },
            "dimensions": [],
            "measures": [{"logical": "order_count"}],
            "filters": [],
        }

    def test_default_routes_even_closed_simple_ir_through_model(self):
        settings = override_settings(
            BASE_SETTINGS,
            "query",
            complex_sql_llm_enabled=True,
            deterministic_sql_direct_emission_enabled=False,
            deterministic_sql_model_fallback_enabled=False,
        )
        policy = decide_model_sql_policy(
            self._closed_simple_ir(),
            query_settings=settings.query,
        )

        self.assertEqual(policy["decision"], "model_sql")
        self.assertTrue(policy["use_model"])
        self.assertEqual(policy["routing_mode"], "correctness_first_model")
        self.assertIn(
            "correctness_first_model_generation",
            policy["reason_codes"],
        )
        self.assertFalse(policy["deterministic_fallback_allowed"])

    def test_explicit_compatibility_switch_restores_direct_emission(self):
        settings = override_settings(
            BASE_SETTINGS,
            "query",
            complex_sql_llm_enabled=True,
            deterministic_sql_direct_emission_enabled=True,
        )
        policy = decide_model_sql_policy(
            self._closed_simple_ir(),
            query_settings=settings.query,
        )

        self.assertEqual(policy["decision"], "deterministic_sql")
        self.assertFalse(policy["use_model"])
        self.assertEqual(
            policy["routing_mode"], "selective_model_compatibility"
        )

    def test_compiler_fallback_requires_explicit_opt_in(self):
        ir = self._closed_simple_ir()
        ir["model_sql_policy"] = decide_model_sql_policy(
            ir,
            query_settings=BASE_SETTINGS.query,
        )

        self.assertFalse(deterministic_model_fallback_allowed(ir))
        self.assertTrue(
            deterministic_model_fallback_allowed(
                ir, runtime_override=True
            )
        )


class ForcedModelSqlPolicyTests(unittest.TestCase):
    def test_runtime_contract_overrides_deterministic_route_with_audit_evidence(self):
        forced = force_model_sql_policy(
            {
                "schema_version": "complex-model-sql-policy-v6",
                "decision": "deterministic_sql",
                "use_model": False,
                "reason_codes": [],
                "trusted_simple_compiler": True,
            },
            reason="evaluation_physical_generate_sql_contract",
        )

        self.assertEqual(forced["decision"], "model_sql")
        self.assertTrue(forced["use_model"])
        self.assertTrue(forced["forced_by_runtime_context"])
        self.assertEqual(
            forced["force_reason"],
            "evaluation_physical_generate_sql_contract",
        )
        self.assertEqual(
            forced["reason_codes"],
            ["forced:evaluation_physical_generate_sql_contract"],
        )
        self.assertTrue(forced["trusted_simple_compiler"])

    def test_runtime_contract_preserves_existing_generic_reasons_without_duplicates(self):
        original_reason = "forced:evaluation_physical_generate_sql_contract"
        forced = force_model_sql_policy(
            {
                "decision": "model_sql",
                "use_model": True,
                "reason_codes": ["multiple_semantic_operators", original_reason],
            },
            reason="evaluation_physical_generate_sql_contract",
        )

        self.assertEqual(
            forced["reason_codes"],
            ["multiple_semantic_operators", original_reason],
        )


class ForcedModelSqlRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_route_preserves_authorized_merged_table_context(self):
        class Runtime:
            context = {
                "settings": BASE_SETTINGS,
                "llm": Nl2SqlLLMAdapter(),
            }

            @staticmethod
            def stream_writer(_payload):
                return None

        tables = [
            {
                "name": "fact_order",
                "columns": [
                    {"name": "order_id", "role": "primary_key"},
                    {"name": "region_id", "role": "foreign_key"},
                ],
            },
            {
                "name": "dim_region",
                "columns": [
                    {"name": "region_id", "role": "primary_key"},
                    {"name": "province", "role": "dimension"},
                ],
            },
        ]
        result = await filter_table(
            {
                "query": "统计销售额",
                "table_infos": tables,
                "semantic_ir": {
                    "compiler": {"supported": True},
                    "model_sql_policy": {
                        "decision": "model_sql",
                        "use_model": True,
                    },
                },
            },
            Runtime(),
        )

        self.assertEqual(result["table_infos"], tables)

    async def test_budget_miss_does_not_execute_compiler_draft_by_default(self):
        class Runtime:
            context = {
                "settings": BASE_SETTINGS,
                "llm": Nl2SqlLLMAdapter(),
            }

            @staticmethod
            def stream_writer(_payload):
                return None

        state = {
            "query": "统计订单数",
            "table_infos": [],
            "metric_infos": [],
            "semantic_ir": {
                "intent": "aggregate",
                "semantic_family": "aggregate",
                "compiler": {"supported": True},
                "semantic_coverage": {
                    "coverage_score": 1.0,
                    "unresolved_slots": [],
                },
                "model_sql_policy": {
                    "decision": "model_sql",
                    "use_model": True,
                    "deterministic_fallback_allowed": False,
                },
            },
        }
        with (
            patch(
                "app.agent_system.agents.nl2sql.nodes.generate_sql.build_query_contract",
                return_value={},
            ),
            patch(
                "app.agent_system.agents.nl2sql.nodes.generate_sql.compile_semantic_ir",
                return_value="SELECT COUNT(*) FROM fact_order",
            ),
            patch(
                "app.agent_system.agents.nl2sql.nodes.generate_sql.model_phase_budget",
                return_value={
                    "allowed": False,
                    "reason_code": "budget.minimum_start_not_met",
                },
            ),
            patch(
                "app.agent_system.agents.nl2sql.nodes.generate_sql.deterministic_draft_is_safe",
                return_value=True,
            ),
        ):
            result = await generate_sql(state, Runtime())

        self.assertEqual(result["sql"], "")
        self.assertIsNotNone(result["clarification"])
        self.assertEqual(
            result["semantic_ir"]["model_sql_policy"]["actual_action"],
            "clarification",
        )
        self.assertFalse(
            result["semantic_ir"]["model_sql_policy"][
                "deterministic_fallback_allowed"
            ]
        )

    async def test_build_node_honors_only_the_trusted_runtime_override(self):
        class Runtime:
            context = {
                "settings": BASE_SETTINGS,
                "force_model_sql": True,
                "force_model_sql_reason": (
                    "evaluation_physical_generate_sql_contract"
                ),
                "model_sql_fallback_enabled": False,
            }

            @staticmethod
            def stream_writer(_payload):
                return None

        semantic_ir = {
            "intent": "aggregate",
            "semantic_family": "aggregate",
            "compiler": {"supported": True},
            "semantic_coverage": {
                "coverage_score": 1.0,
                "unresolved_slots": [],
            },
        }
        with (
            patch(
                "app.agent_system.agents.nl2sql.nodes.build_semantic_ir.build_semantic_ir",
                return_value=semantic_ir,
            ),
            patch(
                "app.agent_system.agents.nl2sql.nodes.build_semantic_ir.build_query_contract",
                return_value={},
            ),
        ):
            result = await build_semantic_ir_node(
                {
                    "query": "按类别统计数量",
                    "table_infos": [],
                    "metric_infos": [],
                    "chat_history": [],
                },
                Runtime(),
            )

        policy = result["semantic_ir"]["model_sql_policy"]
        self.assertEqual(policy["decision"], "model_sql")
        self.assertTrue(policy["use_model"])
        self.assertTrue(policy["forced_by_runtime_context"])
        self.assertEqual(
            policy["force_reason"],
            "evaluation_physical_generate_sql_contract",
        )
        self.assertFalse(policy["deterministic_fallback_allowed"])

    async def test_build_node_closes_contract_only_after_model_policy_is_attached(self):
        class Runtime:
            context = {"settings": BASE_SETTINGS}

            @staticmethod
            def stream_writer(_payload):
                return None

        semantic_ir = {
            "intent": "aggregate",
            "semantic_family": "aggregate",
            "compiler": {
                "supported": True,
                "mode": "deterministic_sqlglot_v2",
            },
            "semantic_coverage": {
                "coverage_score": 1.0,
                "detected_slots": ["measure:sum:amount:all"],
                "unresolved_slots": [],
            },
            "dimensions": [],
            "measures": [{"logical": "amount"}],
            "filters": [],
        }
        contract_policies = []

        def contract_after_policy(_query, _tables, _metrics, ir):
            policy = dict(ir.get("model_sql_policy") or {})
            contract_policies.append(policy)
            return {
                "authority": "guided" if policy.get("use_model") else "closed"
            }

        with (
            patch(
                "app.agent_system.agents.nl2sql.nodes.build_semantic_ir.build_semantic_ir",
                return_value=semantic_ir,
            ),
            patch(
                "app.agent_system.agents.nl2sql.nodes.build_semantic_ir.build_query_contract",
                side_effect=contract_after_policy,
            ),
        ):
            result = await build_semantic_ir_node(
                {
                    "query": "按部门汇总金额",
                    "table_infos": [],
                    "metric_infos": [],
                    "chat_history": [],
                },
                Runtime(),
            )

        self.assertEqual(len(contract_policies), 1)
        self.assertTrue(contract_policies[0]["use_model"])
        self.assertEqual(
            result["semantic_ir"]["query_contract"]["authority"],
            "guided",
        )

    async def test_validation_does_not_use_compiler_fallback_when_contract_disables_it(self):
        class Runtime:
            context = {
                "model_sql_fallback_enabled": False,
                "settings": BASE_SETTINGS,
            }

            @staticmethod
            def stream_writer(_payload):
                return None

        rejected = {
            "valid": False,
            "sql": "SELECT bad",
            "authorized_sql": None,
            "error": "语义不匹配",
            "error_code": "semantic.mismatch",
            "violation": {
                "repairability": "llm",
                "safe_to_retry": True,
            },
            "semantic_repair": {},
        }
        with (
            patch(
                "app.agent_system.agents.nl2sql.nodes.validate_sql.validate_candidate_sql",
                return_value=rejected,
            ),
            patch(
                "app.agent_system.agents.nl2sql.nodes.validate_sql.closed_model_draft_fallback",
            ) as fallback,
        ):
            result = await validate_sql(
                {"sql": "SELECT bad", "retry_count": 0},
                Runtime(),
            )

        fallback.assert_not_called()
        self.assertEqual(result["error"], "语义不匹配")
        self.assertIsNone(result["authorized_sql"])
        self.assertEqual(
            result["semantic_repair_ledger"][0]["phase"],
            "primary_validate",
        )
        self.assertTrue(
            result["semantic_repair_ledger"][0]["committed_at_phase"]
        )
        self.assertTrue(result["semantic_repair_ledger"][0]["selected"])

    async def test_successful_primary_validation_is_committed(self):
        class Runtime:
            context = {
                "model_sql_fallback_enabled": True,
                "settings": BASE_SETTINGS,
            }

            @staticmethod
            def stream_writer(_payload):
                return None

        accepted = {
            "valid": True,
            "sql": "SELECT 1",
            "authorized_sql": "SELECT 1",
            "error": None,
            "error_code": None,
            "violation": None,
            "semantic_repair": {},
        }
        with patch(
            "app.agent_system.agents.nl2sql.nodes.validate_sql.validate_candidate_sql",
            return_value=accepted,
        ):
            result = await validate_sql(
                {"sql": "SELECT 1", "retry_count": 0}, Runtime(),
            )

        self.assertTrue(
            result["semantic_repair_ledger"][0]["committed_at_phase"]
        )
        self.assertTrue(result["semantic_repair_ledger"][0]["selected"])

    async def test_validation_ledger_covers_deterministic_fallback_call(self):
        class Runtime:
            context = {
                "model_sql_fallback_enabled": True,
                "settings": BASE_SETTINGS,
            }

            @staticmethod
            def stream_writer(_payload):
                return None

        rejected = {
            "valid": False,
            "sql": "SELECT bad",
            "authorized_sql": None,
            "error": "语义不匹配",
            "error_code": "semantic.mismatch",
            "violation": {"repairability": "llm", "safe_to_retry": True},
            "semantic_repair": {},
        }
        accepted = {
            "valid": True,
            "sql": "SELECT 1",
            "authorized_sql": "SELECT 1",
            "error": None,
            "error_code": None,
            "violation": None,
            "semantic_repair": {},
        }
        with (
            patch(
                "app.agent_system.agents.nl2sql.nodes.validate_sql.validate_candidate_sql",
                side_effect=[rejected, accepted],
            ),
            patch(
                "app.agent_system.agents.nl2sql.nodes.validate_sql.deterministic_model_fallback_allowed",
                return_value=True,
            ),
            patch(
                "app.agent_system.agents.nl2sql.nodes.validate_sql.closed_model_draft_fallback",
                return_value=(
                    "SELECT 1",
                    {"semantic_family": "aggregate"},
                ),
            ),
        ):
            result = await validate_sql(
                {"sql": "SELECT bad", "retry_count": 0}, Runtime(),
            )

        self.assertEqual(result["sql"], "SELECT 1")
        self.assertEqual(
            [row["phase"] for row in result["semantic_repair_ledger"]],
            ["primary_validate", "deterministic_fallback_validate"],
        )
        self.assertEqual(
            [row["selected"] for row in result["semantic_repair_ledger"]],
            [False, True],
        )
        self.assertEqual(
            [
                row["committed_at_phase"]
                for row in result["semantic_repair_ledger"]
            ],
            [False, True],
        )


if __name__ == "__main__":
    unittest.main()
