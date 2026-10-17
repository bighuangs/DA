import unittest
from types import SimpleNamespace

from app.domain.access.policy import PolicySnapshot, validate_row_filter
from app.domain.query.safety.sql_security import SQLSecurityError


def policy(**overrides):
    values = {
        "id": 1,
        "priority": 100,
        "role": "viewer",
        "user_id": None,
        "table_pattern": "fact_order",
        "allowed_columns": None,
        "row_filter": None,
        "access": True,
        "enabled": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class AccessPolicyTest(unittest.TestCase):
    def test_empty_policy_set_uses_role_baseline(self):
        snapshot = PolicySnapshot(tuple(), role="viewer")
        self.assertEqual(snapshot.authorize_sql("SELECT id FROM fact_order"), "SELECT id FROM fact_order")

    def test_row_filter_is_compiled_with_user_placeholder(self):
        snapshot = PolicySnapshot(
            (policy(row_filter="tenant_id = {{user_id}}"),),
            user_id="u-1", role="viewer",
        )
        sql = snapshot.authorize_sql("SELECT region FROM fact_order")
        self.assertIn("tenant_id = 'u-1'", sql)

    def test_column_policy_rejects_explicit_and_wildcard_access(self):
        snapshot = PolicySnapshot(
            (policy(allowed_columns=["id", "region"]),), role="viewer",
        )
        with self.assertRaises(SQLSecurityError):
            snapshot.authorize_sql("SELECT amount FROM fact_order")
        with self.assertRaises(SQLSecurityError):
            snapshot.authorize_sql("SELECT * FROM fact_order")
        self.assertIn("COUNT(*)", snapshot.authorize_sql("SELECT COUNT(*) FROM fact_order"))

    def test_role_and_user_scope(self):
        viewer_policy = policy(allowed_columns=["id"])
        analyst_policy = policy(id=2, role="analyst", allowed_columns=["id", "amount"])
        viewer = PolicySnapshot((viewer_policy, analyst_policy), role="viewer")
        analyst = PolicySnapshot((viewer_policy, analyst_policy), role="analyst")
        with self.assertRaises(SQLSecurityError):
            viewer.authorize_sql("SELECT amount FROM fact_order")
        self.assertIn("amount", analyst.authorize_sql("SELECT amount FROM fact_order"))

    def test_denied_table_and_subquery_row_filter_are_rejected(self):
        snapshot = PolicySnapshot((policy(access=False, table_pattern="secret_*"),), role="viewer")
        with self.assertRaises(SQLSecurityError):
            snapshot.authorize_sql("SELECT id FROM secret_user")
        with self.assertRaises(ValueError):
            validate_row_filter("tenant_id IN (SELECT tenant_id FROM secret_user)")

    def test_nested_alias_cannot_override_outer_restricted_table(self):
        snapshot = PolicySnapshot(
            (policy(allowed_columns=["region", "amount"]),), role="viewer",
        )
        sql = (
            "SELECT o.secret FROM fact_order o "
            "WHERE EXISTS (SELECT 1 FROM dim_customer o WHERE o.id = 1)"
        )
        with self.assertRaises(SQLSecurityError):
            snapshot.authorize_sql(sql)

    def test_order_by_projection_alias_is_allowed(self):
        snapshot = PolicySnapshot(
            (policy(allowed_columns=["amount"]),), role="viewer",
        )
        sql = snapshot.authorize_sql(
            "SELECT SUM(o.amount) AS total FROM fact_order o ORDER BY total"
        )
        self.assertIn("ORDER BY total", sql)


if __name__ == "__main__":
    unittest.main()
