import unittest

from app.domain.query.safety.sql_security import SQLSecurityError, check_join_relationships, enforce_limit


class SqlResultLimitTest(unittest.TestCase):
    def test_existing_large_limit_is_still_capped(self):
        result = enforce_limit("SELECT id FROM fact_order LIMIT 999999", 1000)
        self.assertIn("LIMIT 999999", result)
        self.assertTrue(result.endswith("LIMIT 1000"))

    def test_trailing_semicolon_is_safe_inside_wrapper(self):
        result = enforce_limit("SELECT 1;", 10)
        self.assertNotIn("SELECT 1;)", result)
        self.assertTrue(result.endswith("LIMIT 10"))


class JoinRelationshipGuardTest(unittest.TestCase):
    def test_rejects_comma_join_without_relationship(self):
        with self.assertRaisesRegex(SQLSecurityError, "笛卡尔积"):
            check_join_relationships(
                "SELECT r.country, SUM(f.order_amount) "
                "FROM fact_order f, dim_region r GROUP BY r.country"
            )

    def test_accepts_explicit_join_relationship(self):
        check_join_relationships(
            "SELECT r.country, SUM(f.order_amount) FROM fact_order f "
            "JOIN dim_region r ON f.region_id = r.region_id GROUP BY r.country"
        )

    def test_accepts_legacy_where_relationship(self):
        check_join_relationships(
            "SELECT r.country FROM fact_order f, dim_region r "
            "WHERE f.region_id = r.region_id"
        )

    def test_rejects_partially_connected_three_table_query(self):
        with self.assertRaises(SQLSecurityError):
            check_join_relationships(
                "SELECT * FROM fact_order f "
                "JOIN dim_region r ON f.region_id = r.region_id, dim_product p"
            )

    def test_accepts_cross_join_to_proven_scalar_aggregate_subquery(self):
        check_join_relationships(
            "SELECT e.employee_id, benchmark.avg_hours FROM fact_shift e "
            "CROSS JOIN (SELECT AVG(hours) AS avg_hours FROM fact_shift) benchmark"
        )

    def test_accepts_cross_join_to_proven_scalar_aggregate_cte(self):
        check_join_relationships(
            "WITH shift_benchmark AS ("
            "SELECT SUM(hours) / NULLIF(COUNT(*), 0) AS avg_hours "
            "FROM fact_shift) "
            "SELECT s.store_name, b.avg_hours FROM dim_store s "
            "CROSS JOIN shift_benchmark b"
        )

    def test_rejects_cross_join_to_non_scalar_or_uncertain_source(self):
        invalid_sql = (
            (
                "WITH per_store AS (SELECT store_id, AVG(hours) AS avg_hours "
                "FROM fact_shift GROUP BY store_id) "
                "SELECT e.employee_id FROM dim_employee e CROSS JOIN per_store p"
            ),
            (
                "WITH one_shift AS (SELECT shift_id FROM fact_shift LIMIT 1) "
                "SELECT s.store_name FROM dim_store s CROSS JOIN one_shift x"
            ),
            (
                "WITH window_stats AS ("
                "SELECT AVG(hours) OVER () AS avg_hours FROM fact_shift) "
                "SELECT s.store_name FROM dim_store s CROSS JOIN window_stats w"
            ),
            (
                "WITH ambiguous_stats AS ("
                "SELECT AVG(hours) AS avg_hours, store_id FROM fact_shift) "
                "SELECT e.employee_id FROM dim_employee e CROSS JOIN ambiguous_stats a"
            ),
            (
                "WITH scalar_stats AS (SELECT AVG(hours) AS avg_hours FROM fact_shift) "
                "SELECT a.avg_hours, e.employee_id, s.store_name FROM scalar_stats a "
                "CROSS JOIN dim_employee e CROSS JOIN dim_store s"
            ),
            (
                "WITH scalar_stats AS (SELECT AVG(hours) AS avg_hours FROM fact_shift) "
                "SELECT e.employee_id, a.avg_hours, s.store_name FROM dim_employee e "
                "CROSS JOIN scalar_stats a CROSS JOIN dim_store s"
            ),
        )
        for sql in invalid_sql:
            with self.subTest(sql=sql), self.assertRaises(SQLSecurityError):
                check_join_relationships(sql)


if __name__ == "__main__":
    unittest.main()
