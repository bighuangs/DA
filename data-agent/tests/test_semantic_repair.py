import unittest

from app.agent_system.agents.nl2sql.services.semantic_repair import (
    apply_semantic_repairs,
    registry_catalog,
    rollback_to_fallback,
    set_runtime_profile,
)


TABLES = [
    {
        "name": "customers",
        "columns": [
            {"name": "customer_id", "role": "primary_key"},
            {"name": "customer_name", "role": "dimension"},
        ],
    },
    {
        "name": "orders",
        "columns": [
            {"name": "order_id", "role": "primary_key"},
            {"name": "customer_id", "role": "foreign_key"},
            {"name": "order_amount", "role": "measure"},
            {"name": "order_date", "role": "dimension"},
        ],
    },
]


class SemanticRepairTests(unittest.TestCase):
    def tearDown(self):
        set_runtime_profile(None)

    def test_p2_repairs_unique_join_and_is_idempotent(self):
        original = (
            "SELECT c.customer_name, COUNT(*) FROM customers c "
            "JOIN orders o ON 1 = 1 GROUP BY c.customer_name"
        )
        repaired, result = apply_semantic_repairs(
            original, query="客户订单数", table_infos=TABLES,
            profile="registry_same_rules_v1",
        )
        self.assertIn("c.customer_id = o.customer_id", repaired)
        self.assertEqual(
            ["join.unique_metadata_edge"],
            [item.rule_code for item in result.applied_rules],
        )
        second, _ = apply_semantic_repairs(
            repaired, query="客户订单数", table_infos=TABLES,
            profile="registry_same_rules_v1",
        )
        self.assertEqual(repaired, second)
        self.assertTrue(all(result.invariants.values()))

    def test_p2_repairs_requested_threshold_projection_only(self):
        sql = (
            "SELECT customer_id FROM orders GROUP BY customer_id "
            "HAVING SUM(order_amount) > 1000"
        )
        repaired, result = apply_semantic_repairs(
            sql, query="累计消费金额超过1000的客户", table_infos=TABLES,
            profile="registry_same_rules_v1",
        )
        self.assertIn("SUM(order_amount) AS total_amount", repaired)
        self.assertEqual("projection.threshold_measure", result.applied_rules[0].rule_code)
        unchanged, near_miss = apply_semantic_repairs(
            sql, query="客户列表", table_infos=TABLES,
            profile="registry_same_rules_v1",
        )
        self.assertEqual(sql, unchanged)
        self.assertFalse(near_miss.changed)

    def test_ranked_limit_appends_visible_grain_as_stable_tie_breaker(self):
        sql = (
            "SELECT c.customer_name, COUNT(*) AS order_count FROM customers c "
            "JOIN orders o ON c.customer_id=o.customer_id "
            "GROUP BY c.customer_name ORDER BY order_count DESC LIMIT 5"
        )
        repaired, result = apply_semantic_repairs(
            sql,
            query="订单数最多的五位客户",
            table_infos=TABLES,
            profile="registry_same_rules_v1",
        )
        self.assertIn("ORDER BY order_count DESC, c.customer_name ASC", repaired)
        self.assertIn(
            "order.rank_limit_tie_breaker",
            [item.rule_code for item in result.applied_rules],
        )
        second, _ = apply_semantic_repairs(
            repaired,
            query="订单数最多的五位客户",
            table_infos=TABLES,
            profile="registry_same_rules_v1",
        )
        self.assertEqual(repaired, second)

    def test_non_ranked_limit_does_not_invent_tie_breaker(self):
        sql = (
            "SELECT c.customer_name FROM customers c "
            "ORDER BY c.customer_name LIMIT 5"
        )
        repaired, result = apply_semantic_repairs(
            sql,
            query="列出五位客户",
            table_infos=TABLES,
            profile="registry_same_rules_v1",
        )
        self.assertEqual(sql, repaired)
        self.assertFalse(result.changed)

    def test_ambiguous_metadata_join_is_not_repaired(self):
        ambiguous = [
            TABLES[0],
            {
                **TABLES[1],
                "columns": [
                    *TABLES[1]["columns"],
                    {"name": "account_id", "role": "foreign_key"},
                ],
            },
            {
                "name": "customers",
                "columns": [
                    *TABLES[0]["columns"],
                    {"name": "account_id", "role": "primary_key"},
                ],
            },
        ]
        sql = "SELECT * FROM customers c JOIN orders o ON 1 = 1"
        repaired, result = apply_semantic_repairs(
            sql, table_infos=ambiguous, profile="registry_same_rules_v1",
        )
        self.assertEqual(sql, repaired)
        self.assertFalse(result.changed)

    def test_extended_profile_closes_unique_order_and_limit_but_p2_does_not(self):
        sql = (
            "SELECT customer_id, SUM(order_amount) AS total_amount "
            "FROM orders GROUP BY customer_id"
        )
        semantic_ir = {
            "compiler": {"supported": True},
            "order_by": [{"target": "total_amount", "direction": "DESC"}],
            "limit": 5,
        }
        p2_sql, p2 = apply_semantic_repairs(
            sql, table_infos=TABLES, semantic_ir=semantic_ir,
            profile="registry_same_rules_v1",
        )
        p3_sql, p3 = apply_semantic_repairs(
            sql, table_infos=TABLES, semantic_ir=semantic_ir,
            profile="registry_extended_v1",
        )
        self.assertEqual(sql, p2_sql)
        self.assertFalse(p2.changed)
        self.assertIn("ORDER BY total_amount DESC", p3_sql)
        self.assertIn("LIMIT 5", p3_sql)
        self.assertEqual("ir.unique_limit_order", p3.applied_rules[-1].rule_code)

    def test_extended_profile_closes_ir_ranking_in_one_idempotent_pass(self):
        scenarios = (
            (
                "SELECT customer_id, SUM(order_amount) AS revenue "
                "FROM orders GROUP BY customer_id",
                "销售额最高的前七位客户",
                {"target": "revenue", "direction": "DESC"},
                7,
            ),
            (
                "SELECT customer_id, COUNT(*) AS order_count "
                "FROM orders GROUP BY customer_id",
                "订单数最少的前八位客户",
                {"target": "order_count", "direction": "ASC"},
                8,
            ),
        )
        for sql, query, order_by, limit in scenarios:
            with self.subTest(order_by=order_by, limit=limit):
                semantic_ir = {
                    "compiler": {"supported": True},
                    "order_by": [order_by],
                    "limit": limit,
                }
                repaired, first = apply_semantic_repairs(
                    sql,
                    query=query,
                    table_infos=TABLES,
                    semantic_ir=semantic_ir,
                    profile="registry_extended_v1",
                )
                second, _ = apply_semantic_repairs(
                    repaired,
                    query=query,
                    table_infos=TABLES,
                    semantic_ir=semantic_ir,
                    profile="registry_extended_v1",
                )

                self.assertEqual(repaired, second)
                self.assertIn("customer_id ASC", repaired)
                self.assertEqual(
                    [
                        "ir.unique_limit_order",
                        "order.rank_limit_tie_breaker",
                    ],
                    [item.rule_code for item in first.applied_rules],
                )

    def test_extended_profile_rejects_limit_over_conflicting_existing_order(self):
        semantic_ir = {
            "compiler": {"supported": True},
            "order_by": [{"target": "revenue", "direction": "DESC"}],
            "limit": 5,
        }
        scenarios = (
            (
                "SELECT customer_id, SUM(order_amount) AS revenue "
                "FROM orders GROUP BY customer_id ORDER BY revenue ASC",
                "existing_order_direction_conflicts_with_ir",
            ),
            (
                "SELECT customer_id, SUM(order_amount) AS revenue, "
                "COUNT(*) AS order_count FROM orders GROUP BY customer_id "
                "ORDER BY order_count DESC",
                "existing_order_target_conflicts_with_ir",
            ),
        )
        for sql, reason in scenarios:
            with self.subTest(reason=reason):
                repaired, result = apply_semantic_repairs(
                    sql,
                    table_infos=TABLES,
                    semantic_ir=semantic_ir,
                    profile="registry_extended_v1",
                )
                self.assertEqual(sql, repaired)
                self.assertFalse(result.changed)
                skipped = {
                    item.rule_code: item.reason
                    for item in result.skipped_rules
                }
                self.assertEqual(reason, skipped["ir.unique_limit_order"])

        matching_sql = (
            "SELECT customer_id, SUM(order_amount) AS revenue "
            "FROM orders GROUP BY customer_id ORDER BY revenue DESC"
        )
        repaired, result = apply_semantic_repairs(
            matching_sql,
            table_infos=TABLES,
            semantic_ir=semantic_ir,
            profile="registry_extended_v1",
        )
        self.assertTrue(result.changed)
        self.assertIn("LIMIT 5", repaired)

    def test_security_input_is_never_rewritten(self):
        sql = "DELETE FROM orders WHERE customer_id = 1"
        repaired, result = apply_semantic_repairs(
            sql, table_infos=TABLES, profile="registry_extended_v1",
        )
        self.assertEqual(sql, repaired)
        self.assertFalse(result.changed)
        self.assertFalse(result.to_dict()["raw_sql_logged"])

    def test_registry_and_p0_rollback_are_versioned(self):
        catalog = registry_catalog()
        self.assertEqual(5, len(catalog))
        self.assertEqual(
            {
                "join.unique_metadata_edge",
                "projection.threshold_measure",
                "order.rank_limit_tie_breaker",
            },
            {item["rule_code"] for item in catalog if item["mode"] == "active"},
        )
        runtime = rollback_to_fallback()
        self.assertEqual("inline_v1", runtime["runtime_profile"])
        self.assertTrue(runtime["rollback_without_restart"])


if __name__ == "__main__":
    unittest.main()
