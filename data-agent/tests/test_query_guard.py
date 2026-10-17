import unittest

from app.domain.query.safety.query_guard import (
    MetricFormulaError,
    QueryCostError,
    check_metric_formulas,
    enforce_explain_cost,
    estimate_explain_rows,
)


class QueryGuardTest(unittest.TestCase):
    metric = {
        "name": "GMV",
        "alias": ["成交总额"],
        "formula_rule": {
            "type": "aggregate",
            "function": "SUM",
            "column": "fact_order.order_amount",
        },
    }
    conditional_count_metric = {
        "name": "事件量",
        "alias": ["事件数"],
        "formula_rule": {
            "type": "aggregate",
            "function": "COUNT",
            "column": "fact_order.order_id",
        },
    }
    conditional_ir = {
        "semantic_family": "conditional_aggregate",
        "intent": "conditional_aggregate",
        "status": "complete",
        "compiler": {"supported": True},
        "measures": [
            {
                "table": "fact_order",
                "column": "order_id",
                "function": "COUNT",
            },
        ],
    }
    generic_customer_count = {
        "name": "客户数",
        "formula_rule": {
            "type": "distinct_count",
            "function": "COUNT",
            "column": "fact_order.customer_id",
            "distinct": True,
            "population_scope": "filtered_fact_rows",
        },
    }
    no_event_customer = {
        "name": "无事件客户",
        "formula_rule": {
            "type": "anti_join",
            "universe_column": "dim_customer.customer_id",
            "event_column": "fact_order.customer_id",
        },
    }
    anti_join_ir = {
        "semantic_family": "anti_join",
        "intent": "anti_join",
        "status": "partial",
        "compiler": {"supported": False},
        "required_tables": ["dim_customer", "fact_order"],
    }

    def test_metric_formula_accepts_table_alias(self):
        check_metric_formulas(
            "SELECT SUM(o.order_amount) AS gmv FROM fact_order o",
            "查询成交总额",
            [self.metric],
        )

    def test_metric_formula_rejects_wrong_aggregate(self):
        with self.assertRaises(MetricFormulaError):
            check_metric_formulas(
                "SELECT AVG(o.order_amount) AS gmv FROM fact_order o",
                "查询GMV",
                [self.metric],
            )

    def test_unmatched_terminology_is_not_blocked(self):
        check_metric_formulas(
            "SELECT COUNT(*) FROM fact_order",
            "统计订单数",
            [],
        )

    def test_recalled_but_unmentioned_metric_cannot_override_explicit_operation(self):
        check_metric_formulas(
            "SELECT MIN(o.order_amount) AS min_order_amount "
            "FROM fact_order o",
            "最低单笔订单金额",
            [self.metric],
        )

    def test_explain_rows_are_summed(self):
        self.assertEqual(
            estimate_explain_rows([{"rows": "120"}, {"ROWS": 30.8}, {"rows": None}]),
            150,
        )

    def test_join_cost_uses_nested_loop_product(self):
        self.assertEqual(
            estimate_explain_rows([
                {"table": "a", "rows": 1000},
                {"table": "b", "rows": 500},
            ]),
            501000,
        )

    def test_join_cost_applies_filtered_selectivity_to_downstream_probes(self):
        self.assertEqual(
            estimate_explain_rows([
                {"id": 2, "table": "customers", "rows": 100000, "filtered": 10},
                {"id": 2, "table": "orders", "rows": 100, "filtered": 25},
            ]),
            1100000,
        )

    def test_selective_cte_plan_is_not_rejected_by_raw_product(self):
        plan = [
            {"id": 1, "table": "<derived2>", "rows": 105, "filtered": 100},
            {"id": 2, "table": "<derived4>", "rows": 256705, "filtered": 10},
            {"id": 2, "table": "d", "rows": 1, "filtered": 100},
            {"id": 2, "table": "first_lines", "rows": 1, "filtered": 5},
            {"id": 2, "table": "first_product", "rows": 1, "filtered": 8.19},
            {"id": 2, "table": "later", "rows": 25, "filtered": 33.33},
            {"id": 2, "table": "later_product", "rows": 1, "filtered": 91.81},
            {"id": 4, "table": "<derived5>", "rows": 256705, "filtered": 100},
            {"id": 5, "table": "fact_order", "rows": 256705, "filtered": 100},
        ]
        estimate = estimate_explain_rows(plan)
        self.assertIsNotNone(estimate)
        self.assertLess(estimate, 1_000_000)
        self.assertEqual(
            enforce_explain_cost(plan, max_explain_rows=5_000_000),
            estimate,
        )

    def test_independent_query_blocks_are_summed_not_multiplied(self):
        self.assertEqual(
            estimate_explain_rows([
                {"id": 1, "table": "derived", "rows": 250000},
                {"id": 2, "table": "fact_order", "rows": 250000},
            ]),
            500000,
        )

    def test_metric_formula_must_be_in_final_projection(self):
        with self.assertRaises(MetricFormulaError):
            check_metric_formulas(
                "SELECT o.region FROM fact_order o WHERE o.order_amount > 0",
                "查询成交总额",
                [self.metric],
            )

    def test_metric_formula_accepts_official_projection_in_second_union_branch(self):
        check_metric_formulas(
            "SELECT 'upload' AS source_label, u.region, "
            "SUM(u.amount) AS total_amount FROM upload_1_aaaaaaaa u "
            "GROUP BY u.region UNION ALL "
            "SELECT 'dw' AS source_label, r.region_name AS region, "
            "SUM(o.order_amount) AS total_amount FROM fact_order o "
            "JOIN dim_region r ON o.region_id = r.region_id "
            "GROUP BY r.region_name",
            "对比上传文件和正式 DW 的成交总额",
            [self.metric],
        )

    def test_metric_formula_accepts_cte_alias_in_final_projection(self):
        check_metric_formulas(
            "WITH monthly AS ("
            "SELECT d.month, SUM(o.order_amount) AS total_amount "
            "FROM fact_order o JOIN dim_date d ON o.date_id = d.date_id "
            "GROUP BY d.month) "
            "SELECT month, total_amount, "
            "SUM(total_amount) OVER (ORDER BY month) AS cumulative_amount "
            "FROM monthly",
            "每月成交总额及累计值",
            [self.metric],
        )

    def test_avg_metric_accepts_sum_divided_by_order_count(self):
        average_metric = {
            "name": "AOV",
            "alias": ["客单价"],
            "formula_rule": {
                "type": "aggregate",
                "function": "AVG",
                "column": "fact_order.order_amount",
            },
        }
        check_metric_formulas(
            "SELECT SUM(o.order_amount) / "
            "NULLIF(COUNT(DISTINCT o.order_id), 0) AS aov "
            "FROM fact_order o",
            "客单价",
            [average_metric],
        )

    def test_distinct_population_rejects_non_distinct_count(self):
        metric = {
            "name": "客户数",
            "formula_rule": {
                "type": "distinct_count",
                "function": "COUNT",
                "column": "fact_order.customer_id",
                "distinct": True,
            },
        }
        check_metric_formulas(
            "SELECT COUNT(DISTINCT o.customer_id) AS customer_count "
            "FROM fact_order o",
            "客户数",
            [metric],
        )
        with self.assertRaises(MetricFormulaError):
            check_metric_formulas(
                "SELECT COUNT(o.customer_id) AS customer_count "
                "FROM fact_order o",
                "客户数",
                [metric],
            )

    def test_share_requires_registered_numerator_and_shared_total(self):
        metric = {
            "name": "销售贡献率",
            "formula_rule": {
                "type": "share",
                "function": "SUM",
                "column": "fact_order.order_amount",
            },
        }
        check_metric_formulas(
            "WITH grouped AS ("
            "SELECT o.region_id, SUM(o.order_amount) AS total_amount "
            "FROM fact_order o GROUP BY o.region_id) "
            "SELECT region_id, total_amount / "
            "NULLIF(SUM(total_amount) OVER (), 0) AS sales_share FROM grouped",
            "各区域销售贡献率",
            [metric],
        )
        with self.assertRaises(MetricFormulaError):
            check_metric_formulas(
                "WITH grouped AS ("
                "SELECT o.region_id, SUM(o.order_amount) AS total_amount, "
                "COUNT(*) AS row_count FROM fact_order o GROUP BY o.region_id) "
                "SELECT region_id, total_amount / "
                "NULLIF(SUM(row_count) OVER (), 0) AS sales_share FROM grouped",
                "各区域销售贡献率",
                [metric],
            )

    def test_activation_rate_requires_two_distinct_populations(self):
        metric = {
            "name": "商品动销率",
            "formula_rule": {
                "type": "activation_rate",
                "numerator_column": "fact_order.product_id",
                "denominator_column": "dim_product.product_id",
            },
        }
        check_metric_formulas(
            "WITH sold AS (SELECT COUNT(DISTINCT o.product_id) AS sold_count "
            "FROM fact_order o), universe AS ("
            "SELECT COUNT(DISTINCT p.product_id) AS product_count "
            "FROM dim_product p) SELECT sold_count / "
            "NULLIF(product_count, 0) AS activation_rate FROM sold, universe",
            "商品动销率",
            [metric],
        )
        with self.assertRaises(MetricFormulaError):
            check_metric_formulas(
                "SELECT COUNT(DISTINCT o.product_id) / "
                "NULLIF(COUNT(*), 0) AS activation_rate FROM fact_order o",
                "商品动销率",
                [metric],
            )

    def test_cohort_count_requires_entity_first_event_stage(self):
        metric = {
            "name": "新客户",
            "formula_rule": {
                "type": "cohort_count",
                "entity_column": "fact_order.customer_id",
                "event_column": "fact_order.date_id",
                "event_selector": "MIN",
            },
        }
        check_metric_formulas(
            "WITH first_event AS (SELECT o.customer_id, "
            "MIN(o.date_id) AS first_date FROM fact_order o "
            "GROUP BY o.customer_id) SELECT COUNT(*) AS new_customer_count "
            "FROM first_event WHERE first_date >= 20240101",
            "新客户数量",
            [metric],
        )
        with self.assertRaises(MetricFormulaError):
            check_metric_formulas(
                "SELECT COUNT(DISTINCT o.customer_id) AS new_customer_count "
                "FROM fact_order o WHERE o.date_id >= 20240101",
                "新客户数量",
                [metric],
            )

    def test_anti_join_metric_rejects_inner_join_shortcut(self):
        metric = {
            "name": "未下单客户",
            "formula_rule": {
                "type": "anti_join",
                "universe_column": "dim_customer.customer_id",
                "event_column": "fact_order.customer_id",
            },
        }
        check_metric_formulas(
            "SELECT COUNT(*) FROM dim_customer c LEFT JOIN fact_order o "
            "ON c.customer_id = o.customer_id WHERE o.customer_id IS NULL",
            "未下单客户有多少",
            [metric],
        )
        with self.assertRaises(MetricFormulaError):
            check_metric_formulas(
                "SELECT COUNT(*) FROM dim_customer c JOIN fact_order o "
                "ON c.customer_id = o.customer_id",
                "未下单客户有多少",
                [metric],
            )

    def test_conditional_count_accepts_sum_of_boolean_indicators(self):
        check_metric_formulas(
            "SELECT f.region_id, "
            "SUM(CASE WHEN f.order_amount >= 37 THEN 1 ELSE 0 END) AS upper_count, "
            "SUM(CASE WHEN f.order_amount < 37 THEN 1 ELSE 0 END) AS lower_count "
            "FROM fact_order f GROUP BY f.region_id",
            "按区域分别统计两个条件区间的事件数",
            [self.conditional_count_metric],
            self.conditional_ir,
        )

    def test_aggregate_family_does_not_enforce_unrelated_recalled_formulas(self):
        metrics = [
            self.metric,
            {
                "name": "AOV",
                "alias": ["平均订单金额"],
                "formula_rule": {
                    "type": "aggregate",
                    "function": "AVG",
                    "column": "fact_order.order_amount",
                },
            },
            {
                "name": "销量",
                "formula_rule": {
                    "type": "aggregate",
                    "function": "SUM",
                    "column": "fact_order.order_quantity",
                },
            },
            self.conditional_count_metric,
        ]
        cases = [
            (
                "按地区统计GMV",
                "SELECT f.region_id, SUM(f.order_amount) FROM fact_order f "
                "GROUP BY f.region_id",
            ),
            (
                "按地区统计平均订单金额",
                "SELECT f.region_id, AVG(f.order_amount) FROM fact_order f "
                "GROUP BY f.region_id",
            ),
            (
                "按地区统计事件数",
                "SELECT f.region_id, COUNT(f.order_id) FROM fact_order f "
                "GROUP BY f.region_id",
            ),
            (
                "按地区统计销量",
                "SELECT f.region_id, SUM(f.order_quantity) FROM fact_order f "
                "GROUP BY f.region_id",
            ),
        ]
        semantic_ir = {
            "semantic_family": "aggregate",
            "status": "complete",
            "compiler": {"supported": True},
            "required_tables": ["fact_order"],
        }
        for query, sql in cases:
            with self.subTest(query=query):
                check_metric_formulas(sql, query, metrics, semantic_ir)

    def test_conditional_count_accepts_count_case_row_identity(self):
        check_metric_formulas(
            "SELECT COUNT(CASE WHEN f.order_amount >= 80 "
            "THEN f.order_id END) AS conditional_count FROM fact_order f",
            "统计满足条件的事件数",
            [self.conditional_count_metric],
            self.conditional_ir,
        )

    def test_conditional_counts_remain_valid_across_group_dimensions(self):
        cases = [
            (
                "p.category",
                "JOIN dim_product p ON f.product_id = p.product_id",
                41,
            ),
            (
                "c.gender",
                "JOIN dim_customer c ON f.customer_id = c.customer_id",
                73,
            ),
        ]
        for dimension, join_sql, boundary in cases:
            with self.subTest(dimension=dimension):
                check_metric_formulas(
                    f"SELECT {dimension}, "
                    f"SUM(CASE WHEN f.order_amount >= {boundary} "
                    "THEN 1 ELSE 0 END) AS upper_count, "
                    f"SUM(CASE WHEN f.order_amount < {boundary} "
                    "THEN 1 ELSE 0 END) AS lower_count "
                    f"FROM fact_order f {join_sql} GROUP BY {dimension}",
                    "按分组维度分别统计两个条件区间的事件数",
                    [self.conditional_count_metric],
                    self.conditional_ir,
                )

    def test_conditional_count_accepts_cte_projection_lineage(self):
        check_metric_formulas(
            "WITH grouped AS (SELECT f.region_id, "
            "SUM(CASE WHEN f.order_amount >= 25 THEN 1 ELSE 0 END) AS event_count "
            "FROM fact_order f GROUP BY f.region_id) "
            "SELECT region_id, event_count FROM grouped",
            "按区域统计条件事件数",
            [self.conditional_count_metric],
            self.conditional_ir,
        )

    def test_conditional_count_rejects_weighted_case(self):
        for value in ("2", "f.order_amount"):
            with self.subTest(value=value), self.assertRaises(MetricFormulaError):
                check_metric_formulas(
                    "SELECT SUM(CASE WHEN f.order_amount >= 25 THEN "
                    f"{value} ELSE 0 END) FROM fact_order f",
                    "统计满足条件的事件数",
                    [self.conditional_count_metric],
                    self.conditional_ir,
                )

    def test_conditional_count_requires_closed_matching_ir(self):
        invalid_irs = [
            {**self.conditional_ir, "status": "partial"},
            {**self.conditional_ir, "compiler": {"supported": False}},
            {
                **self.conditional_ir,
                "measures": [{
                    "table": "fact_order",
                    "column": "customer_id",
                    "function": "COUNT",
                }],
            },
            {**self.conditional_ir, "semantic_family": "simple_aggregate"},
        ]
        sql = (
            "SELECT SUM(CASE WHEN f.order_amount >= 25 THEN 1 ELSE 0 END) "
            "FROM fact_order f"
        )
        for semantic_ir in invalid_irs:
            with self.subTest(semantic_ir=semantic_ir), self.assertRaises(
                MetricFormulaError
            ):
                check_metric_formulas(
                    sql,
                    "统计满足条件的事件数",
                    [self.conditional_count_metric],
                    semantic_ir,
                )

    def test_anti_join_count_accepts_preserved_universe_key(self):
        check_metric_formulas(
            "SELECT c.member_level, COUNT(DISTINCT c.customer_id) AS inactive_count "
            "FROM dim_customer c WHERE NOT EXISTS ("
            "SELECT 1 FROM fact_order f WHERE f.customer_id = c.customer_id) "
            "GROUP BY c.member_level",
            "按等级统计没有关联事件的客户数量",
            [self.no_event_customer],
            self.anti_join_ir,
        )

    def test_anti_join_count_accepts_count_star_after_null_rejection(self):
        check_metric_formulas(
            "SELECT COUNT(*) AS inactive_count FROM dim_customer c "
            "LEFT JOIN fact_order f ON c.customer_id = f.customer_id "
            "WHERE f.customer_id IS NULL",
            "没有关联事件的客户有多少",
            [self.no_event_customer],
            self.anti_join_ir,
        )

    def test_anti_join_count_rejects_nullable_fact_event_population(self):
        with self.assertRaises(MetricFormulaError):
            check_metric_formulas(
                "SELECT COUNT(DISTINCT f.customer_id) AS inactive_count "
                "FROM dim_customer c LEFT JOIN fact_order f "
                "ON c.customer_id = f.customer_id WHERE f.customer_id IS NULL",
                "统计没有关联事件的客户数量",
                [self.no_event_customer],
                self.anti_join_ir,
            )

    def test_anti_join_count_rejects_non_universe_dimension(self):
        with self.assertRaises(MetricFormulaError):
            check_metric_formulas(
                "SELECT COUNT(c.member_level) AS inactive_count "
                "FROM dim_customer c WHERE NOT EXISTS ("
                "SELECT 1 FROM fact_order f WHERE f.customer_id = c.customer_id)",
                "统计没有关联事件的客户数量",
                [self.no_event_customer],
                self.anti_join_ir,
            )

    def test_anti_join_entity_listing_does_not_require_count_projection(self):
        check_metric_formulas(
            "SELECT c.customer_id FROM dim_customer c WHERE NOT EXISTS ("
            "SELECT 1 FROM fact_order f WHERE f.customer_id = c.customer_id)",
            "列出无事件客户",
            [self.no_event_customer],
            self.anti_join_ir,
        )

    def test_anti_join_specificity_ignores_incompatible_generic_population(self):
        check_metric_formulas(
            "SELECT COUNT(DISTINCT c.customer_id) AS inactive_count "
            "FROM dim_customer c WHERE NOT EXISTS ("
            "SELECT 1 FROM fact_order f WHERE f.customer_id = c.customer_id)",
            "没有任何关联事件的客户数量",
            [self.generic_customer_count, self.no_event_customer],
            self.anti_join_ir,
        )

    def test_explicit_multi_population_audits_both_metrics(self):
        semantic_ir = {
            **self.anti_join_ir,
            "semantic_slots": {
                "requested_population_scopes": [
                    "dimension_universe_minus_events",
                    "filtered_fact_rows",
                ],
            },
        }
        check_metric_formulas(
            "SELECT COUNT(DISTINCT c.customer_id) AS inactive_count, "
            "(SELECT COUNT(DISTINCT f2.customer_id) FROM fact_order f2) "
            "AS active_count FROM dim_customer c WHERE NOT EXISTS ("
            "SELECT 1 FROM fact_order f WHERE f.customer_id = c.customer_id)",
            "无事件客户和客户数",
            [self.generic_customer_count, self.no_event_customer],
            semantic_ir,
        )

    def test_cost_guard_rejects_over_threshold(self):
        with self.assertRaises(QueryCostError):
            enforce_explain_cost([{"rows": 101}], max_explain_rows=100)

    def test_cost_guard_can_be_disabled(self):
        self.assertEqual(
            enforce_explain_cost([{"rows": 101}], max_explain_rows=100, enabled=False),
            101,
        )


if __name__ == "__main__":
    unittest.main()
