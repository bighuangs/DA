import unittest

from evals.legacy.attribution import attribute_case, summarize_attributions


def result(gen_sql, expected_sql, *, category="agg_join", equivalent=False, tables=False):
    return {
        "id": "case", "category": category, "gen_sql": gen_sql,
        "expected_sql": expected_sql, "result_equivalent": equivalent,
        "tables_correct": tables, "exec_success": True, "error": None,
    }


class AttributionTests(unittest.TestCase):
    def test_attributes_cartesian_join_before_generic_semantics(self):
        item = result(
            "SELECT r.country, SUM(f.order_amount) FROM fact_order f, dim_region r GROUP BY r.country",
            "SELECT r.country, SUM(f.order_amount) FROM fact_order f JOIN dim_region r ON f.region_id=r.region_id GROUP BY r.country",
            tables=True,
        )
        self.assertEqual(attribute_case(item)["code"], "join_path")

    def test_attributes_readable_entity_projection(self):
        item = result(
            "SELECT product_id, SUM(order_quantity) FROM fact_order GROUP BY product_id",
            "SELECT p.product_name, SUM(f.order_quantity) FROM fact_order f JOIN dim_product p ON f.product_id=p.product_id GROUP BY p.product_name",
        )
        self.assertEqual(attribute_case(item)["code"], "entity_projection")

    def test_attributes_time_and_multi_turn(self):
        time_item = result(
            "SELECT date_id DIV 100 ym, COUNT(*) FROM fact_order GROUP BY ym",
            "SELECT d.month, COUNT(*) FROM fact_order f JOIN dim_date d ON f.date_id=d.date_id GROUP BY d.month",
            category="time_series",
        )
        follow_up = result(
            "SELECT SUM(order_amount) FROM fact_order",
            "SELECT p.category, SUM(f.order_amount) FROM fact_order f JOIN dim_product p ON f.product_id=p.product_id GROUP BY p.category",
            category="multi_turn",
        )
        self.assertEqual(attribute_case(time_item)["code"], "time_semantics")
        self.assertEqual(attribute_case(follow_up)["code"], "multi_turn_context")

    def test_equivalent_alternative_is_not_counted_as_failure(self):
        item = result("SELECT COUNT(*) FROM dim_customer", "SELECT COUNT(*) FROM fact_order", equivalent=True)
        self.assertEqual(attribute_case(item)["code"], "acceptable_alternative")
        summary = summarize_attributions([item], "workspace")
        self.assertEqual(summary["failed_count"], 0)
        self.assertEqual(summary["root_causes"], [])

    def test_missing_generated_table_already_kept_by_retrieval_is_generation_error(self):
        item = result(
            "SELECT p.brand, SUM(f.order_amount) FROM fact_order f "
            "JOIN dim_product p ON f.product_id=p.product_id GROUP BY p.brand",
            "SELECT p.brand, SUM(f.order_amount) FROM fact_order f "
            "JOIN dim_product p ON f.product_id=p.product_id "
            "JOIN dim_date d ON f.date_id=d.date_id WHERE d.year=2028 "
            "GROUP BY p.brand",
        )
        attribution = attribute_case(
            item,
            stage_evidence={
                "recalled_tables": ["fact_order", "dim_product", "dim_date"],
                "filtered_tables": ["fact_order", "dim_product", "dim_date"],
            },
        )

        self.assertEqual(attribution["code"], "semantic_sql")
        self.assertEqual(attribution["stage"], "plan_generate")

    def test_missing_table_not_available_to_generation_remains_retrieval_error(self):
        item = result(
            "SELECT p.brand, SUM(f.order_amount) FROM fact_order f "
            "JOIN dim_product p ON f.product_id=p.product_id GROUP BY p.brand",
            "SELECT p.brand, SUM(f.order_amount) FROM fact_order f "
            "JOIN dim_product p ON f.product_id=p.product_id "
            "JOIN dim_date d ON f.date_id=d.date_id WHERE d.year=2028 "
            "GROUP BY p.brand",
        )
        attribution = attribute_case(
            item,
            stage_evidence={
                "recalled_tables": ["fact_order", "dim_product"],
                "filtered_tables": ["fact_order", "dim_product"],
            },
        )

        self.assertEqual(attribution["code"], "retrieval_table")


if __name__ == "__main__":
    unittest.main()
