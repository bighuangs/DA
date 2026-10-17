import datetime as dt
import unittest
from decimal import Decimal

import pytest

from evals.legacy.metrics import (
    question_requires_result_order,
    result_equivalent,
    result_equivalence_evidence,
    sql_structure_equivalent,
    table_prf,
    table_contract_prf,
    tables_match_contract,
)


class EvalMetricsTests(unittest.TestCase):
    def test_table_prf_reports_partial_match(self):
        scores = table_prf(
            "SELECT * FROM fact_order f JOIN dim_product p ON f.product_id=p.product_id",
            "SELECT * FROM fact_order f JOIN dim_region r ON f.region_id=r.region_id",
        )
        self.assertEqual(scores["precision"], 0.5)
        self.assertEqual(scores["recall"], 0.5)
        self.assertEqual(scores["f1"], 0.5)

    def test_reviewed_allowed_table_path_does_not_count_as_wrong_table(self):
        sql = "SELECT COUNT(f.order_id) FROM fact_order f JOIN dim_date d ON f.date_id=d.date_id WHERE d.year=2025"
        self.assertTrue(tables_match_contract(sql, ["fact_order"], ["fact_order", "dim_date"]))
        self.assertEqual(
            table_contract_prf(sql, ["fact_order"], ["fact_order", "dim_date"]),
            {"precision": 1.0, "recall": 1.0, "f1": 1.0},
        )
        self.assertFalse(tables_match_contract(
            sql + " JOIN dim_product p ON p.product_id=f.product_id",
            ["fact_order"], ["fact_order", "dim_date"],
        ))

    def test_structure_equivalence_tolerates_aliases(self):
        generated = "SELECT p.category, SUM(f.order_amount) x FROM fact_order f JOIN dim_product p ON f.product_id=p.product_id GROUP BY p.category ORDER BY x DESC"
        expected = "SELECT product.category, SUM(orders.order_amount) total FROM fact_order orders JOIN dim_product product ON orders.product_id=product.product_id GROUP BY product.category ORDER BY total DESC"
        self.assertTrue(sql_structure_equivalent(generated, expected))

    def test_structure_equivalence_detects_missing_order(self):
        generated = "SELECT category, COUNT(*) FROM dim_product GROUP BY category"
        expected = "SELECT category, COUNT(*) c FROM dim_product GROUP BY category ORDER BY c DESC"
        self.assertFalse(sql_structure_equivalent(generated, expected))

    def test_result_equivalence_normalizes_decimal_and_float(self):
        self.assertTrue(result_equivalent(
            [{"amount": Decimal("10.5000000001")}],
            [{"amount": 10.5}],
            "SELECT amount FROM t",
        ))

    def test_result_equivalence_normalizes_serialized_decimal_scale(self):
        self.assertTrue(result_equivalent(
            [{"share": "0.00000"}],
            [{"share": "0.0000"}],
            "SELECT share FROM t",
        ))

    def test_result_equivalence_tolerates_mysql_decimal_scale_drift(self):
        self.assertTrue(result_equivalent(
            [{"member_level": "黄金", "quantity_per_customer": "126.33427"}],
            [{"member_level": "黄金", "quantity_per_customer": "126.3343"}],
            "SELECT member_level, quantity_per_customer FROM t ORDER BY member_level",
        ))

    def test_result_equivalence_rejects_material_numeric_difference(self):
        self.assertFalse(result_equivalent(
            [{"ratio": "126.3340"}],
            [{"ratio": "126.3343"}],
            "SELECT ratio FROM t",
        ))

    def test_result_equivalence_keeps_large_integer_counts_exact(self):
        self.assertFalse(result_equivalent(
            [{"order_count": 1_000_000}],
            [{"order_count": 1_000_001}],
            "SELECT order_count FROM t",
        ))

    def test_unordered_numeric_tolerance_preserves_multiset(self):
        self.assertTrue(result_equivalent(
            [
                {"segment": "B", "ratio": "20.00001"},
                {"segment": "A", "ratio": "10.000004"},
            ],
            [
                {"segment": "A", "ratio": "10.00000"},
                {"segment": "B", "ratio": "20.00000"},
            ],
            "SELECT segment, ratio FROM t",
        ))

    def test_result_equivalence_preserves_integer_like_identifiers(self):
        self.assertFalse(result_equivalent(
            [{"customer_code": "001"}],
            [{"customer_code": "1"}],
            "SELECT customer_code FROM t",
        ))

    def test_result_equivalence_normalizes_dates(self):
        value = dt.date(2025, 1, 2)
        self.assertTrue(result_equivalent([{"d": value}], [{"d": value}], "SELECT d FROM t"))

    def test_alias_fallback_preserves_projection_position(self):
        self.assertFalse(result_equivalent(
            [{"left_alias": 1, "right_alias": 2}],
            [{"a": 2, "b": 1}],
            "SELECT a, b FROM t",
        ))

    def test_unordered_results_ignore_row_order(self):
        self.assertTrue(result_equivalent(
            [{"v": 2}, {"v": 1}],
            [{"v": 1}, {"v": 2}],
            "SELECT v FROM t",
        ))

    def test_reference_order_by_does_not_make_results_order_sensitive(self):
        self.assertTrue(result_equivalent(
            [{"v": 2}, {"v": 1}],
            [{"v": 1}, {"v": 2}],
            "SELECT v FROM t ORDER BY v",
        ))

    def test_explicit_order_semantics_keep_row_order(self):
        self.assertFalse(result_equivalent(
            [{"v": 2}, {"v": 1}],
            [{"v": 1}, {"v": 2}],
            "SELECT v FROM t ORDER BY v",
            question="请按数值升序排列",
        ))

    def test_explicit_unordered_contract_overrides_reference_order_by(self):
        self.assertTrue(result_equivalent(
            [{"v": 2}, {"v": 1}],
            [{"v": 1}, {"v": 2}],
            "SELECT v FROM t ORDER BY v",
            ordered=False,
        ))

    def test_question_semantics_override_legacy_order_flag(self):
        self.assertTrue(result_equivalent(
            [{"v": 2}, {"v": 1}],
            [{"v": 1}, {"v": 2}],
            "SELECT v FROM t ORDER BY v",
            ordered=True,
            question="汇总各分组的数值",
        ))
        self.assertFalse(result_equivalent(
            [{"v": 2}, {"v": 1}],
            [{"v": 1}, {"v": 2}],
            "SELECT v FROM t ORDER BY v",
            ordered=False,
            question="按数值降序排列",
        ))

    def test_unordered_comparison_keeps_duplicate_rows_strict(self):
        self.assertFalse(result_equivalent(
            [{"v": 1}, {"v": 1}],
            [{"v": 1}, {"v": 2}],
            "SELECT v FROM t ORDER BY v",
            question="汇总各分组的数值",
        ))

    def test_existence_question_accepts_executed_witness_for_positive_count(self):
        evidence = result_equivalence_evidence(
            [{"device_id": "D7"}],
            [{"device_count": 1}],
            "SELECT COUNT(*) AS device_count FROM dim_device WHERE device_id='D7'",
            question="设备目录中编号D7的设备是否存在",
            generated_sql="SELECT device_id FROM dim_device WHERE device_id='D7'",
        )

        self.assertTrue(evidence["equivalent"])
        self.assertEqual(evidence["comparison_mode"], "existence_truth_equivalence")
        self.assertEqual(
            evidence["existence_equivalence"]["generated_representation"],
            "witness_rows",
        )
        self.assertTrue(evidence["existence_equivalence"]["reference_truth"])

    def test_existence_question_accepts_empty_witness_for_false_boolean(self):
        evidence = result_equivalence_evidence(
            [],
            [{"exists_flag": 0}],
            "SELECT EXISTS(SELECT 1 FROM dim_device WHERE device_id='missing') AS exists_flag",
            question="有没有编号为missing的设备",
            generated_sql="SELECT device_id FROM dim_device WHERE device_id='missing'",
        )

        self.assertTrue(evidence["equivalent"])
        self.assertFalse(evidence["existence_equivalence"]["generated_truth"])
        self.assertFalse(evidence["existence_equivalence"]["reference_truth"])

    def test_existence_question_rejects_different_executed_truth_values(self):
        evidence = result_equivalence_evidence(
            [],
            [{"device_count": 2}],
            "SELECT COUNT(*) AS device_count FROM dim_device WHERE active=1",
            question="是否有处于启用状态的设备",
            generated_sql="SELECT device_id FROM dim_device WHERE active=1",
        )

        self.assertFalse(evidence["equivalent"])
        self.assertTrue(evidence["existence_equivalence"]["eligible"])
        self.assertEqual(
            evidence["existence_equivalence"]["reason"],
            "existence_truth_mismatch",
        )

    def test_existence_question_does_not_collapse_two_different_witness_sets(self):
        evidence = result_equivalence_evidence(
            [{"device_id": "D7"}],
            [{"device_code": "D8"}],
            "SELECT device_code FROM dim_device WHERE active=1",
            question="是否有处于启用状态的设备",
            generated_sql="SELECT device_id FROM dim_device WHERE active=1",
        )

        self.assertFalse(evidence["equivalent"])
        self.assertFalse(evidence["existence_equivalence"]["eligible"])
        self.assertEqual(
            evidence["existence_equivalence"]["reason"],
            "requires_one_scalar_and_one_witness_projection",
        )

    def test_explicit_quantity_keeps_count_value_and_projection_contract(self):
        evidence = result_equivalence_evidence(
            [{"device_id": "D7"}],
            [{"device_count": 2}],
            "SELECT COUNT(*) AS device_count FROM dim_device WHERE active=1",
            question="是否有匹配设备，并返回匹配设备数量",
            generated_sql="SELECT device_id FROM dim_device WHERE active=1",
        )

        self.assertFalse(evidence["equivalent"])
        self.assertEqual(
            evidence["existence_equivalence"]["reason"],
            "explicit_quantity_contract",
        )

    def test_existence_normalization_requires_pure_yes_no_contract(self):
        cases = (
            (
                "是否存在启用设备；若存在，返回设备ID",
                [{"device_id": "D7"}],
                "SELECT device_id FROM dim_device WHERE active=1",
            ),
            (
                "是否存在启用设备；若存在，返回设备名称和厂商",
                [{"device_name": "A", "vendor": "V"}],
                "SELECT device_name, vendor FROM dim_device WHERE active=1",
            ),
            (
                "返回设备ID并判断是否存在启用设备",
                [{"device_id": "D7"}],
                "SELECT device_id FROM dim_device WHERE active=1",
            ),
            (
                "是否存在启用设备，并返回匹配设备计数",
                [{"device_id": "D7"}, {"device_id": "D8"}],
                "SELECT device_id FROM dim_device WHERE active=1",
            ),
        )
        for question, expected_rows, expected_sql in cases:
            with self.subTest(question=question):
                evidence = result_equivalence_evidence(
                    [{"device_count": 1}],
                    expected_rows,
                    expected_sql,
                    question=question,
                    generated_sql=(
                        "SELECT COUNT(*) AS device_count "
                        "FROM dim_device WHERE active=1"
                    ),
                )
                self.assertFalse(evidence["equivalent"])
                self.assertFalse(evidence["existence_equivalence"]["eligible"])

    def test_existence_normalization_rejects_cardinality_thresholds(self):
        for question, expected_sql in (
            (
                "是否有至少2个启用设备",
                "SELECT COUNT(*) >= 2 AS exists_flag "
                "FROM dim_device WHERE active=1",
            ),
            (
                "是否存在多个启用设备",
                "SELECT COUNT(*) > 1 AS exists_flag "
                "FROM dim_device WHERE active=1",
            ),
            (
                "至少2个启用设备是否存在",
                "SELECT COUNT(*) >= 2 AS exists_flag "
                "FROM dim_device WHERE active=1",
            ),
        ):
            with self.subTest(question=question):
                evidence = result_equivalence_evidence(
                    [{"device_id": "D7"}],
                    [{"exists_flag": 1}],
                    expected_sql,
                    question=question,
                    generated_sql=(
                        "SELECT device_id FROM dim_device WHERE active=1"
                    ),
                )
                self.assertFalse(evidence["equivalent"])
                self.assertFalse(evidence["existence_equivalence"]["eligible"])

    def test_existence_scalar_requires_canonical_truth_shape(self):
        for expected_sql in (
            "SELECT COUNT(*) = 2 AS exists_flag FROM dim_device WHERE active=1",
            "SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 1 END AS exists_flag "
            "FROM dim_device WHERE active=1",
        ):
            with self.subTest(expected_sql=expected_sql):
                evidence = result_equivalence_evidence(
                    [{"device_id": "D7"}],
                    [{"exists_flag": 1}],
                    expected_sql,
                    question="是否存在启用设备",
                    generated_sql=(
                        "SELECT device_id FROM dim_device WHERE active=1"
                    ),
                )
                self.assertFalse(evidence["equivalent"])
                self.assertFalse(evidence["existence_equivalence"]["eligible"])

    def test_existence_scalar_accepts_canonical_zero_comparisons(self):
        for expected_sql in (
            "SELECT COUNT(*) > 0 AS exists_flag FROM dim_device WHERE active=1",
            "SELECT 0 < COUNT(*) AS exists_flag FROM dim_device WHERE active=1",
            "SELECT COUNT(*) <> 0 AS exists_flag FROM dim_device WHERE active=1",
            "SELECT 0 != COUNT(*) AS exists_flag FROM dim_device WHERE active=1",
        ):
            with self.subTest(expected_sql=expected_sql):
                evidence = result_equivalence_evidence(
                    [{"device_id": "D7"}],
                    [{"exists_flag": 1}],
                    expected_sql,
                    question="是否存在启用设备",
                    generated_sql=(
                        "SELECT device_id FROM dim_device WHERE active=1"
                    ),
                )
                self.assertTrue(evidence["equivalent"])
                self.assertEqual(
                    evidence["comparison_mode"],
                    "existence_truth_equivalence",
                )

    def test_minimum_projection_accepts_only_unrequested_reference_columns(self):
        evidence = result_equivalence_evidence(
            [{"product_name": "A"}, {"product_name": "B"}],
            [
                {"product_name": "A", "category": "C1", "brand": "B1"},
                {"product_name": "B", "category": "C2", "brand": "B2"},
            ],
            "SELECT product_name, category, brand FROM dim_product "
            "ORDER BY product_name LIMIT 2",
            question="按商品名称升序列出商品目录中的前两个商品",
            generated_sql=(
                "SELECT product_name FROM dim_product "
                "ORDER BY product_name LIMIT 2"
            ),
        )

        self.assertTrue(evidence["equivalent"])
        self.assertEqual(evidence["comparison_mode"], "minimum_sufficient_projection")
        projection = evidence["minimum_sufficient_projection"]
        self.assertTrue(projection["eligible"])
        self.assertEqual(
            [item["column"] for item in projection["omitted_reference_columns"]],
            ["category", "brand"],
        )
        self.assertEqual(projection["explicitly_required_omissions"], [])

    def test_minimum_projection_rejects_explicitly_requested_dimension(self):
        evidence = result_equivalence_evidence(
            [{"product_name": "A"}],
            [{"product_name": "A", "category": "C", "brand": "B"}],
            "SELECT product_name, category, brand FROM dim_product",
            question="列出商品名称和品牌",
            generated_sql="SELECT product_name FROM dim_product",
        )

        self.assertFalse(evidence["equivalent"])
        required = evidence["minimum_sufficient_projection"][
            "explicitly_required_omissions"
        ]
        self.assertEqual([item["column"] for item in required], ["brand"])
        self.assertIn("question_names_column:brand", required[0]["request_reason_codes"])

    def test_longest_metadata_term_prevents_broad_alias_false_requirement(self):
        evidence = result_equivalence_evidence(
            [{"category": "家电"}],
            [{"category": "家电", "product_name": "A"}],
            "SELECT category, product_name FROM dim_product",
            question="按商品类别列出结果",
            generated_sql="SELECT category FROM dim_product",
        )

        self.assertTrue(evidence["equivalent"])
        omitted = evidence["minimum_sufficient_projection"][
            "omitted_reference_columns"
        ]
        self.assertEqual(omitted[0]["column"], "product_name")
        self.assertEqual(omitted[0]["request_reason_codes"], [])

    def test_independent_broad_alias_still_requires_its_projection(self):
        evidence = result_equivalence_evidence(
            [{"category": "家电"}],
            [{"category": "家电", "product_name": "A"}],
            "SELECT category, product_name FROM dim_product",
            question="列出商品及商品类别",
            generated_sql="SELECT category FROM dim_product",
        )

        self.assertFalse(evidence["equivalent"])
        reasons = evidence["minimum_sufficient_projection"][
            "explicitly_required_omissions"
        ][0]["request_reason_codes"]
        self.assertIn("question_names_column:product_name", reasons)

    def test_rank_criterion_measure_is_not_forced_into_final_projection(self):
        evidence = result_equivalence_evidence(
            [{"region_name": "华东"}],
            [{"region_name": "华东", "total_amount": "100.00"}],
            "SELECT region_name, total_amount FROM ranked_regions",
            question="找出销售额最高的地区",
            generated_sql="SELECT region_name FROM ranked_regions",
        )

        self.assertTrue(evidence["equivalent"])
        omitted = evidence["minimum_sufficient_projection"][
            "omitted_reference_columns"
        ][0]
        self.assertEqual(omitted["request_reason_codes"], [])

    def test_predicate_measure_is_not_forced_into_final_projection(self):
        evidence = result_equivalence_evidence(
            [{"region_name": "华东"}],
            [{"region_name": "华东", "total_amount": "100.00"}],
            "SELECT region_name, total_amount FROM filtered_regions",
            question="列出销售额大于100的地区",
            generated_sql="SELECT region_name FROM filtered_regions",
        )

        self.assertTrue(evidence["equivalent"])

    def test_measure_requested_as_answer_remains_required(self):
        evidence = result_equivalence_evidence(
            [{"region_name": "华东"}],
            [{"region_name": "华东", "total_amount": "100.00"}],
            "SELECT region_name, total_amount FROM region_sales",
            question="统计各地区的销售额",
            generated_sql="SELECT region_name FROM region_sales",
        )

        self.assertFalse(evidence["equivalent"])
        reasons = evidence["minimum_sufficient_projection"][
            "explicitly_required_omissions"
        ][0]["request_reason_codes"]
        self.assertIn("question_names_column:order_amount", reasons)

    def test_minimum_projection_rejects_explicitly_requested_metric(self):
        evidence = result_equivalence_evidence(
            [{"region_name": "华东"}],
            [{"region_name": "华东", "total_amount": "100.00"}],
            "SELECT r.region_name, SUM(f.order_amount) AS total_amount "
            "FROM fact_order f JOIN dim_region r ON f.region_id = r.region_id "
            "GROUP BY r.region_name",
            question="统计每个地区的销售额",
            generated_sql="SELECT region_name FROM dim_region",
        )

        self.assertFalse(evidence["equivalent"])
        required = evidence["minimum_sufficient_projection"][
            "explicitly_required_omissions"
        ]
        self.assertEqual([item["column"] for item in required], ["total_amount"])
        self.assertIn(
            "question_names_column:order_amount",
            required[0]["request_reason_codes"],
        )

    def test_count_alias_does_not_require_raw_identifier_projection(self):
        evidence = result_equivalence_evidence(
            [{"order_count": 12}],
            [{"order_count": 12, "order_id": "O-001"}],
            "SELECT order_count, order_id FROM order_summary",
            question="统计订单笔数",
            generated_sql="SELECT order_count FROM order_summary",
        )

        self.assertTrue(evidence["equivalent"])
        omitted = evidence["minimum_sufficient_projection"][
            "omitted_reference_columns"
        ][0]
        self.assertEqual(omitted["column"], "order_id")
        self.assertEqual(omitted["request_reason_codes"], [])

    def test_explicit_identifier_term_still_requires_raw_identifier_projection(self):
        evidence = result_equivalence_evidence(
            [{"order_count": 12}],
            [{"order_count": 12, "order_id": "O-001"}],
            "SELECT order_count, order_id FROM order_summary",
            question="同时返回订单ID和订单笔数",
            generated_sql="SELECT order_count FROM order_summary",
        )

        self.assertFalse(evidence["equivalent"])
        reasons = evidence["minimum_sufficient_projection"][
            "explicitly_required_omissions"
        ][0]["request_reason_codes"]
        self.assertIn("question_names_column:order_id", reasons)

    def test_minimum_projection_maps_metadata_alias_without_positional_guess(self):
        evidence = result_equivalence_evidence(
            [{"商品名称": "A"}],
            [{"product_name": "A", "brand": "B"}],
            "SELECT product_name, brand FROM dim_product",
            question="列出商品名称",
            generated_sql="SELECT product_name AS 商品名称 FROM dim_product",
        )

        self.assertTrue(evidence["equivalent"])
        self.assertEqual(
            evidence["minimum_sufficient_projection"]["column_mapping"],
            [{"generated": "商品名称", "reference": "product_name"}],
        )

    def test_minimum_projection_preserves_duplicate_multiset(self):
        self.assertFalse(result_equivalent(
            [{"product_name": "A"}, {"product_name": "A"}],
            [
                {"product_name": "A", "brand": "B1"},
                {"product_name": "B", "brand": "B2"},
            ],
            "SELECT product_name, brand FROM dim_product",
            question="列出商品名称",
            generated_sql="SELECT product_name FROM dim_product",
        ))

    def test_minimum_projection_fails_closed_for_unresolved_omission(self):
        evidence = result_equivalence_evidence(
            [{"product_name": "A"}],
            [{"product_name": "A", "fixture_marker": 1}],
            "SELECT product_name, 1 AS fixture_marker FROM dim_product",
            question="列出商品名称",
            generated_sql="SELECT product_name FROM dim_product",
        )

        self.assertFalse(evidence["equivalent"])
        required = evidence["minimum_sufficient_projection"][
            "explicitly_required_omissions"
        ]
        self.assertEqual(required[0]["request_reason_codes"], ["unresolved_semantics"])

    def test_rank_value_is_required_only_when_question_requests_rank_output(self):
        common = {
            "gen_rows": [{"product_name": "A"}],
            "expected_rows": [{"product_name": "A", "rank_value": 1}],
            "expected_sql": (
                "SELECT product_name, ROW_NUMBER() OVER (ORDER BY product_name) "
                "AS rank_value FROM dim_product"
            ),
            "generated_sql": "SELECT product_name FROM dim_product LIMIT 1",
        }
        top_membership = result_equivalence_evidence(
            **common,
            question="返回销量前一个商品",
        )
        explicit_rank = result_equivalence_evidence(
            **common,
            question="给出每个商品的销量排名",
        )

        self.assertTrue(top_membership["equivalent"])
        self.assertFalse(explicit_rank["equivalent"])
        reasons = explicit_rank["minimum_sufficient_projection"][
            "explicitly_required_omissions"
        ][0]["request_reason_codes"]
        self.assertIn("question_requests_rank_output", reasons)

    def test_having_measure_is_a_predicate_not_a_required_projection(self):
        average = result_equivalence_evidence(
            [{"category": "A"}],
            [{"category": "A", "avg_order_amount": "700.00"}],
            "SELECT category, AVG(order_amount) AS avg_order_amount "
            "FROM sales GROUP BY category HAVING AVG(order_amount) > 650",
            question="筛选平均订单金额高于650元的商品类别",
            generated_sql=(
                "SELECT category FROM sales GROUP BY category "
                "HAVING AVG(order_amount) > 650"
            ),
        )
        count = result_equivalence_evidence(
            [{"category": "A", "brand": "B"}],
            [{"category": "A", "brand": "B", "product_count": 5}],
            "SELECT category, brand, COUNT(*) AS product_count FROM products "
            "GROUP BY category, brand HAVING COUNT(*) >= 5",
            question="筛选商品数量达到5种的类别和品牌组合",
            generated_sql=(
                "SELECT category, brand FROM products GROUP BY category, brand "
                "HAVING COUNT(*) >= 5"
            ),
        )

        self.assertTrue(average["equivalent"])
        self.assertTrue(count["equivalent"])

    def test_compound_having_predicates_are_not_forced_into_projection(self):
        evidence = result_equivalence_evidence(
            [{"category": "A"}],
            [
                {
                    "category": "A",
                    "avg_order_amount": "700.00",
                    "product_count": 5,
                }
            ],
            "SELECT category, AVG(order_amount) AS avg_order_amount, "
            "COUNT(DISTINCT product_id) AS product_count FROM sales "
            "GROUP BY category HAVING AVG(order_amount) > 650 "
            "AND COUNT(DISTINCT product_id) >= 5",
            question=(
                "筛选平均订单金额高于650元且商品数量达到5种的商品类别"
            ),
            generated_sql=(
                "SELECT category FROM sales GROUP BY category "
                "HAVING AVG(order_amount) > 650 "
                "AND COUNT(DISTINCT product_id) >= 5"
            ),
        )

        self.assertTrue(evidence["equivalent"])
        omissions = evidence["minimum_sufficient_projection"][
            "omitted_reference_columns"
        ]
        self.assertEqual(
            [item["column"] for item in omissions],
            ["avg_order_amount", "product_count"],
        )
        self.assertTrue(all(not item["request_reason_codes"] for item in omissions))

    def test_independently_requested_having_measure_remains_required(self):
        evidence = result_equivalence_evidence(
            [{"category": "A"}],
            [{"category": "A", "avg_order_amount": "700.00"}],
            "SELECT category, AVG(order_amount) AS avg_order_amount "
            "FROM sales GROUP BY category",
            question="列出商品类别及其平均订单金额",
            generated_sql="SELECT category FROM sales GROUP BY category",
        )

        self.assertFalse(evidence["equivalent"])
        omissions = evidence["minimum_sufficient_projection"][
            "explicitly_required_omissions"
        ]
        self.assertEqual([item["column"] for item in omissions], ["avg_order_amount"])

    def test_change_operand_and_previous_helper_are_not_forced_into_projection(self):
        evidence = result_equivalence_evidence(
            [
                {"year": 2024, "active_customers": 10, "customer_change": None},
                {"year": 2025, "active_customers": 12, "customer_change": 2},
            ],
            [
                {
                    "year": 2024,
                    "active_customer_count": 10,
                    "previous_customer_count": None,
                    "customer_change": None,
                },
                {
                    "year": 2025,
                    "active_customer_count": 12,
                    "previous_customer_count": 10,
                    "customer_change": 2,
                },
            ],
            "SELECT year, active_customer_count, previous_customer_count, "
            "active_customer_count - previous_customer_count AS customer_change "
            "FROM annual_comparison",
            question="给出每年活跃客户数以及相对上一年的客户数变化",
            generated_sql=(
                "SELECT year, COUNT(DISTINCT customer_id) AS active_customers, "
                "COUNT(DISTINCT customer_id) - LAG(COUNT(DISTINCT customer_id)) "
                "OVER (ORDER BY year) AS customer_change FROM sales GROUP BY year"
            ),
        )

        self.assertTrue(evidence["equivalent"])
        projection = evidence["minimum_sufficient_projection"]
        self.assertEqual(
            projection["column_mapping"],
            [
                {"generated": "year", "reference": "year"},
                {
                    "generated": "active_customers",
                    "reference": "active_customer_count",
                },
                {"generated": "customer_change", "reference": "customer_change"},
            ],
        )
        self.assertEqual(projection["explicitly_required_omissions"], [])

    def test_explicit_previous_count_remains_required(self):
        evidence = result_equivalence_evidence(
            [{"year": 2025, "active_customers": 12, "customer_change": 2}],
            [
                {
                    "year": 2025,
                    "active_customer_count": 12,
                    "previous_customer_count": 10,
                    "customer_change": 2,
                }
            ],
            "SELECT year, active_customer_count, previous_customer_count, "
            "customer_change FROM annual_comparison",
            question="给出每年活跃客户数、上一年客户数以及变化值",
            generated_sql=(
                "SELECT year, COUNT(DISTINCT customer_id) AS active_customers, "
                "2 AS customer_change FROM sales GROUP BY year"
            ),
        )

        self.assertFalse(evidence["equivalent"])
        required = evidence["minimum_sufficient_projection"][
            "explicitly_required_omissions"
        ]
        self.assertEqual([item["column"] for item in required], ["previous_customer_count"])

    def test_ratio_operands_are_not_forced_into_final_projection(self):
        evidence = result_equivalence_evidence(
            [{"category": "A", "month": 1, "ratio": "0.25"}],
            [
                {
                    "category": "A",
                    "month": 1,
                    "monthly_amount": "25.00",
                    "monthly_share": "0.25",
                }
            ],
            "SELECT category, month, monthly_amount, "
            "monthly_amount / annual_amount AS monthly_share FROM monthly_sales",
            question="各商品类别每月成交金额占该类别全年成交金额的比例",
            generated_sql=(
                "SELECT category, month, SUM(order_amount) / "
                "SUM(SUM(order_amount)) OVER (PARTITION BY category) AS ratio "
                "FROM sales GROUP BY category, month"
            ),
        )

        self.assertTrue(evidence["equivalent"])
        omitted = evidence["minimum_sufficient_projection"][
            "omitted_reference_columns"
        ]
        self.assertEqual([item["column"] for item in omitted], ["monthly_amount"])
        self.assertEqual(omitted[0]["request_reason_codes"], [])

    def test_independently_requested_ratio_operand_remains_required(self):
        evidence = result_equivalence_evidence(
            [{"category": "A", "month": 1, "ratio": "0.25"}],
            [
                {
                    "category": "A",
                    "month": 1,
                    "monthly_amount": "25.00",
                    "monthly_share": "0.25",
                }
            ],
            "SELECT category, month, monthly_amount, monthly_share FROM monthly_sales",
            question="给出每月成交金额以及其占全年成交金额的比例",
            generated_sql=(
                "SELECT category, month, SUM(order_amount) / "
                "SUM(SUM(order_amount)) OVER (PARTITION BY category) AS ratio "
                "FROM sales GROUP BY category, month"
            ),
        )

        self.assertFalse(evidence["equivalent"])
        required = evidence["minimum_sufficient_projection"][
            "explicitly_required_omissions"
        ]
        self.assertEqual([item["column"] for item in required], ["monthly_amount"])

    def test_auxiliary_columns_on_both_sides_do_not_override_common_answer(self):
        evidence = result_equivalence_evidence(
            [
                {
                    "customer_id": "C1",
                    "customer_name": "甲",
                    "max_month_count": 8,
                }
            ],
            [{"customer_id": "C1", "max_month_count": 8, "rank_value": 1}],
            "SELECT customer_id, max_month_count, rank_value FROM ranked",
            question="找出连续月份数最多的客户",
            generated_sql=(
                "SELECT customer_id, customer_name, max_month_count FROM ranked"
            ),
        )

        self.assertTrue(evidence["equivalent"])
        projection = evidence["minimum_sufficient_projection"]
        self.assertEqual(projection["extra_generated_columns"], ["customer_name"])
        self.assertEqual(projection["unresolved_reference_columns"], ["rank_value"])

    def test_derived_alias_and_bucket_label_are_normalized_by_semantic_role(self):
        common = {
            "expected_rows": [
                {
                    "member_level": "银级",
                    "spend_quartile": 2,
                    "avg_customer_amount": "20.00",
                }
            ],
            "expected_sql": (
                "SELECT member_level, spend_quartile, avg_customer_amount "
                "FROM segmented"
            ),
            "question": "在每个会员等级内分为四档并返回平均成交金额",
            "generated_sql": (
                "SELECT member_level, amount_tier, avg_amount FROM segmented"
            ),
        }
        equivalent = result_equivalence_evidence(
            [
                {
                    "member_level": "银级",
                    "amount_tier": "第2档",
                    "avg_amount": "20.00",
                }
            ],
            **common,
        )
        wrong_bucket = result_equivalence_evidence(
            [
                {
                    "member_level": "银级",
                    "amount_tier": "第3档",
                    "avg_amount": "20.00",
                }
            ],
            **common,
        )

        self.assertTrue(equivalent["equivalent"])
        self.assertFalse(wrong_bucket["equivalent"])

    def test_additive_rollup_accepts_only_unrequested_reference_grain(self):
        common = {
            "gen_rows": [{"total_amount": "30.00"}],
            "expected_rows": [
                {"region_name": "北区", "category": "A", "amount": "10.00"},
                {"region_name": "北区", "category": "B", "amount": "20.00"},
            ],
            "expected_sql": (
                "SELECT region_name, category, SUM(order_amount) AS amount "
                "FROM sales GROUP BY region_name, category"
            ),
            "generated_sql": "SELECT SUM(order_amount) AS total_amount FROM sales",
        }
        filtered_grain = result_equivalence_evidence(
            **common,
            question="北区大区中非停用类别的成交金额",
        )
        requested_grain = result_equivalence_evidence(
            **common,
            question="统计各大区和各类别的成交金额",
        )

        self.assertTrue(filtered_grain["equivalent"])
        self.assertEqual(
            filtered_grain["minimum_sufficient_projection"][
                "projected_comparison_mode"
            ],
            "reference_additive_rollup",
        )
        self.assertFalse(requested_grain["equivalent"])

    def test_ambiguous_derived_alias_mapping_fails_closed(self):
        evidence = result_equivalence_evidence(
            [{"ratio": "0.5"}],
            [{"monthly_share": "0.5", "annual_share": "0.5"}],
            "SELECT monthly_share, annual_share FROM ratios",
            question="返回占比",
            generated_sql="SELECT ratio FROM ratios",
        )

        self.assertFalse(evidence["equivalent"])
        projection = evidence["minimum_sufficient_projection"]
        self.assertEqual(projection["reason"], "columns_not_uniquely_mappable")

    def test_long_time_axis_matches_proven_conditional_aggregate_wide_result(self):
        generated_rows = [
            {"segment": "A", "year": 2024, "total_amount": "10.00"},
            {"segment": "A", "year": 2026, "total_amount": "12.00"},
            {"segment": "B", "year": 2024, "total_amount": "20.00"},
            {"segment": "B", "year": 2026, "total_amount": "24.00"},
        ]
        expected_rows = [
            {"segment": "A", "amount_2024": "10.0", "amount_2026": "12.0"},
            {"segment": "B", "amount_2024": "20.0", "amount_2026": "24.0"},
        ]
        generated_sql = (
            "SELECT segment, year, SUM(order_amount) AS total_amount "
            "FROM sales WHERE year IN (2024, 2026) "
            "GROUP BY segment, year"
        )
        expected_sql = (
            "SELECT segment, "
            "SUM(CASE WHEN year = 2024 THEN order_amount ELSE 0 END) AS amount_2024, "
            "SUM(CASE WHEN year = 2026 THEN order_amount ELSE 0 END) AS amount_2026 "
            "FROM sales WHERE year IN (2024, 2026) GROUP BY segment"
        )

        evidence = result_equivalence_evidence(
            generated_rows,
            expected_rows,
            expected_sql,
            question="按分群分别给出2024年和2026年的成交金额",
            generated_sql=generated_sql,
        )

        self.assertTrue(evidence["equivalent"])
        self.assertEqual(
            "long_wide_information_equivalence", evidence["comparison_mode"],
        )
        shape = evidence["long_wide_information_equivalence"]
        self.assertEqual("year", shape["axis_column"])
        self.assertEqual(["2024", "2026"], shape["axis_values"])
        self.assertEqual("information_match", shape["reason"])

    def test_long_wide_equivalence_fails_closed_for_incomplete_or_wrong_shape(self):
        expected_rows = [
            {"segment": "A", "amount_2024": 10, "amount_2026": 12},
        ]
        expected_sql = (
            "SELECT segment, "
            "SUM(CASE WHEN year = 2024 THEN order_amount ELSE 0 END) AS amount_2024, "
            "SUM(CASE WHEN year = 2026 THEN order_amount ELSE 0 END) AS amount_2026 "
            "FROM sales GROUP BY segment"
        )
        amount_sql = (
            "SELECT segment, year, SUM(order_amount) AS total_amount "
            "FROM sales GROUP BY segment, year"
        )
        variants = [
            (
                [{"segment": "A", "year": 2024, "total_amount": 10}],
                amount_sql,
                "按分群分别给出2024年和2026年的成交金额",
            ),
            (
                [
                    {"segment": "A", "year": 2024, "total_amount": 10},
                    {"segment": "A", "year": 2026, "total_amount": 13},
                ],
                amount_sql,
                "按分群分别给出2024年和2026年的成交金额",
            ),
            (
                [
                    {"segment": "A", "year": 2024, "total_quantity": 10},
                    {"segment": "A", "year": 2026, "total_quantity": 12},
                ],
                (
                    "SELECT segment, year, SUM(order_quantity) AS total_quantity "
                    "FROM sales GROUP BY segment, year"
                ),
                "按分群分别给出2024年和2026年的成交金额",
            ),
            (
                [
                    {"segment": "A", "year": 2024, "total_amount": 10},
                    {"segment": "A", "year": 2026, "total_amount": 12},
                ],
                amount_sql,
                "按分群分别给出两个年度的成交金额",
            ),
        ]

        for generated_rows, generated_sql, question in variants:
            with self.subTest(generated_sql=generated_sql, question=question):
                evidence = result_equivalence_evidence(
                    generated_rows,
                    expected_rows,
                    expected_sql,
                    question=question,
                    generated_sql=generated_sql,
                )
                self.assertFalse(evidence["equivalent"])


@pytest.mark.parametrize(
    "question",
    [
        "按销售额降序排列各分组",
        "按年份顺序展示结果",
        "Sort regions in ascending order",
    ],
)
def test_question_requires_result_order_for_explicit_semantics(question):
    assert question_requires_result_order(question)


@pytest.mark.parametrize(
    "question",
    [
        "统计各分组的订单数量",
        "统计每个季度的销售额",
        "计算每个分组单笔记录的最高数值",
        "汇总前 3 个月的收入",
        "汇总前 3 个自然月的收入",
        "汇总前 5 个工作日的收入",
        "给出各分组的销售额排名",
        "返回销售额前 5 个分组",
        "列出规模前五大的分组",
        "找出表现倒数三名的分组",
        "Show the top 10 products",
        "查询客户顺序编号及姓名",
        "",
        None,
    ],
)
def test_question_does_not_infer_order_from_ordinary_report_semantics(question):
    assert not question_requires_result_order(question)


if __name__ == "__main__":
    unittest.main()
