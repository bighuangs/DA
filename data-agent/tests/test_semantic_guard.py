import unittest

from app.agent_system.agents.nl2sql.services.semantic_guard import (
    SemanticContractError,
    analyze_semantic_shape,
    check_semantic_contract,
    repair_invalid_metadata_joins,
    stabilize_threshold_measure_projection,
)
from tests.test_semantic_hints import TABLES


class SemanticGuardTests(unittest.TestCase):
    @staticmethod
    def _device_directory_tables():
        return [
            {
                "name": "dim_device",
                "role": "dim",
                "description": "设备目录表",
                "columns": [
                    {
                        "name": "device_id",
                        "role": "primary_key",
                        "description": "设备编号",
                        "alias": ["设备ID"],
                    },
                    {
                        "name": "vendor_id",
                        "role": "foreign_key",
                        "description": "厂商编号",
                    },
                    {
                        "name": "device_name",
                        "role": "dimension",
                        "description": "设备名称",
                        "is_display": True,
                    },
                    {
                        "name": "device_type",
                        "role": "dimension",
                        "description": "设备类型",
                        "alias": ["类型"],
                    },
                    {
                        "name": "vendor",
                        "role": "dimension",
                        "description": "厂商",
                    },
                    {
                        "name": "status",
                        "role": "dimension",
                        "description": "设备状态",
                    },
                    {
                        "name": "owner_email",
                        "role": "dimension",
                        "description": "负责人邮箱",
                        "sensitive": True,
                    },
                ],
            },
            {
                "name": "dim_vendor",
                "role": "dim",
                "description": "厂商目录表",
                "columns": [
                    {
                        "name": "vendor_id",
                        "role": "primary_key",
                        "description": "厂商编号",
                    },
                    {
                        "name": "vendor_name",
                        "role": "dimension",
                        "description": "厂商名称",
                        "is_display": True,
                    },
                    {
                        "name": "status",
                        "role": "dimension",
                        "description": "厂商状态",
                    },
                ],
            },
        ]

    @staticmethod
    def _quarter_member_tables():
        return [
            {
                "name": "fact_order", "role": "fact", "description": "订单事实表",
                "columns": [
                    {"name": "order_id", "role": "primary_key", "description": "订单编号"},
                    {"name": "date_id", "role": "foreign_key", "description": "日期外键"},
                    {"name": "customer_id", "role": "foreign_key", "description": "客户外键"},
                ],
            },
            {
                "name": "dim_date", "role": "dim", "description": "日期维度表",
                "columns": [
                    {"name": "date_id", "role": "primary_key", "description": "日期编号"},
                    {"name": "year", "role": "dimension", "description": "年份"},
                    {"name": "quarter", "role": "dimension", "description": "季度"},
                ],
            },
            {
                "name": "dim_customer", "role": "dim", "description": "客户维度表",
                "columns": [
                    {"name": "customer_id", "role": "primary_key", "description": "客户编号"},
                    {
                        "name": "member_level",
                        "role": "dimension",
                        "description": "会员等级",
                        "alias": ["会员级别"],
                    },
                ],
            },
        ]

    @staticmethod
    def _retail_contract_tables():
        return [
            {
                "name": "fact_order", "role": "fact", "description": "订单事实表",
                "columns": [
                    {"name": "order_id", "role": "primary_key", "description": "订单编号"},
                    {"name": "date_id", "role": "foreign_key", "description": "日期外键"},
                    {"name": "customer_id", "role": "foreign_key", "description": "客户外键"},
                    {"name": "product_id", "role": "foreign_key", "description": "商品外键"},
                    {"name": "region_id", "role": "foreign_key", "description": "地区外键"},
                    {
                        "name": "order_quantity", "role": "measure",
                        "description": "购买数量", "alias": ["购买件数", "商品件数"],
                    },
                    {"name": "order_amount", "role": "measure", "description": "销售额"},
                ],
            },
            {
                "name": "dim_date", "role": "dim", "description": "日期维度表",
                "columns": [
                    {"name": "date_id", "role": "primary_key", "description": "日期编号"},
                    {"name": "year", "role": "dimension", "description": "年份"},
                    {"name": "month", "role": "dimension", "description": "月份"},
                ],
            },
            {
                "name": "dim_product", "role": "dim", "description": "商品维度表",
                "columns": [
                    {
                        "name": "product_id", "role": "primary_key",
                        "description": "商品编号", "alias": ["商品ID"],
                    },
                    {
                        "name": "product_name", "role": "dimension",
                        "description": "商品名称", "is_display": True,
                    },
                    {"name": "category", "role": "dimension", "description": "品类", "alias": ["商品类别"]},
                    {"name": "brand", "role": "dimension", "description": "品牌", "alias": ["品牌名称"]},
                ],
            },
            {
                "name": "dim_customer", "role": "dim", "description": "客户维度表",
                "columns": [
                    {"name": "customer_id", "role": "primary_key", "description": "客户编号"},
                    {"name": "customer_name", "role": "dimension", "description": "客户名称"},
                    {"name": "gender", "role": "dimension", "description": "性别"},
                ],
            },
            {
                "name": "dim_region", "role": "dim", "description": "地区维度表",
                "columns": [
                    {"name": "region_id", "role": "primary_key", "description": "地区编号"},
                    {"name": "region_name", "role": "dimension", "description": "大区名称"},
                    {"name": "province", "role": "dimension", "description": "省份"},
                ],
            },
        ]

    @staticmethod
    def _threshold_entity_tables():
        """Unseen entity/measure vocabulary for threshold contract tests."""
        return [
            {
                "name": "fact_activity", "role": "fact", "description": "业务活动事实表",
                "columns": [
                    {
                        "name": "activity_id", "role": "primary_key",
                        "description": "活动编号", "type": "bigint",
                    },
                    {
                        "name": "store_id", "role": "foreign_key",
                        "description": "门店外键", "type": "bigint",
                    },
                    {
                        "name": "employee_id", "role": "foreign_key",
                        "description": "员工外键", "type": "bigint",
                    },
                    {
                        "name": "region_id", "role": "foreign_key",
                        "description": "地区外键", "type": "bigint",
                    },
                    {
                        "name": "product_id", "role": "foreign_key",
                        "description": "商品外键", "type": "bigint",
                    },
                    {
                        "name": "units", "role": "measure", "type": "decimal(12, 2)",
                        "description": "购买数量", "alias": ["购买件数", "销售件数"],
                    },
                    {
                        "name": "amount", "role": "measure", "type": "decimal(14, 2)",
                        "description": "交易金额", "alias": ["订单金额", "销售额"],
                    },
                    {
                        "name": "score", "role": "measure", "type": "decimal(8, 2)",
                        "description": "绩效分数", "alias": ["服务分数"],
                    },
                ],
            },
            {
                "name": "dim_store", "role": "dim", "description": "门店维度表",
                "columns": [
                    {"name": "store_id", "role": "primary_key", "description": "门店编号"},
                    {
                        "name": "store_name", "role": "dimension",
                        "description": "门店名称", "is_display": True,
                    },
                ],
            },
            {
                "name": "dim_employee", "role": "dim", "description": "员工维度表",
                "columns": [
                    {"name": "employee_id", "role": "primary_key", "description": "员工编号"},
                    {
                        "name": "employee_name", "role": "dimension",
                        "description": "员工名称", "is_display": True,
                    },
                ],
            },
            {
                "name": "dim_region", "role": "dim", "description": "地区维度表",
                "columns": [
                    {"name": "region_id", "role": "primary_key", "description": "地区编号"},
                    {
                        "name": "region_name", "role": "dimension",
                        "description": "地区名称", "is_display": True,
                    },
                    {
                        "name": "province", "role": "dimension",
                        "description": "省份", "alias": ["省级区域"],
                    },
                ],
            },
            {
                "name": "dim_product", "role": "dim", "description": "商品维度表",
                "columns": [
                    {"name": "product_id", "role": "primary_key", "description": "商品编号"},
                    {
                        "name": "product_name", "role": "dimension",
                        "description": "商品名称", "is_display": True,
                    },
                ],
            },
            {
                "name": "dim_period", "role": "dim", "description": "期间维度表",
                "columns": [
                    {"name": "period_id", "role": "primary_key", "description": "期间编号"},
                    {
                        "name": "month_label", "role": "dimension",
                        "description": "月份", "is_display": True,
                    },
                ],
            },
        ]

    @staticmethod
    def _workforce_anti_tables():
        return [
            {
                "name": "fact_shift", "role": "fact", "description": "排班事实表",
                "columns": [
                    {"name": "shift_id", "role": "primary_key", "description": "排班编号"},
                    {"name": "employee_id", "role": "foreign_key", "description": "员工外键"},
                    {"name": "store_id", "role": "foreign_key", "description": "门店外键"},
                    {"name": "shift_day", "role": "dimension", "description": "排班日期"},
                ],
            },
            {
                "name": "dim_employee", "role": "dim", "description": "员工维度表",
                "columns": [
                    {"name": "employee_id", "role": "primary_key", "description": "员工编号"},
                    {
                        "name": "employee_name", "role": "dimension",
                        "description": "员工名称", "is_display": True,
                    },
                ],
            },
            {
                "name": "dim_store", "role": "dim", "description": "门店维度表",
                "columns": [
                    {"name": "store_id", "role": "primary_key", "description": "门店编号"},
                    {
                        "name": "store_name", "role": "dimension",
                        "description": "门店名称", "is_display": True,
                    },
                ],
            },
        ]

    def test_row_numeric_threshold_in_having_is_model_repairable(self):
        cases = (
            (
                "购买件数至少3时，按门店统计交易金额",
                "SELECT s.store_name, SUM(f.amount) FROM fact_activity f "
                "JOIN dim_store s ON f.store_id=s.store_id "
                "GROUP BY s.store_name HAVING SUM(f.units) >= 3",
            ),
            (
                "列出订单金额超过100的记录并按员工汇总购买数量",
                "SELECT e.employee_name, SUM(f.units) FROM fact_activity f "
                "JOIN dim_employee e ON f.employee_id=e.employee_id "
                "GROUP BY e.employee_name HAVING SUM(f.amount) > 100",
            ),
            (
                "单笔服务分数不低于80时，按地区统计活动数",
                "SELECT r.region_name, COUNT(*) FROM fact_activity f "
                "JOIN dim_region r ON f.region_id=r.region_id "
                "GROUP BY r.region_name HAVING AVG(f.score) >= 75",
            ),
            (
                "销售件数至少4时，按商品统计交易金额",
                "SELECT p.product_name, SUM(f.amount), SUM(f.units) AS total_units "
                "FROM fact_activity f JOIN dim_product p ON f.product_id=p.product_id "
                "GROUP BY p.product_name HAVING total_units >= 4",
            ),
        )
        for question, sql in cases:
            with self.subTest(question=question), self.assertRaises(
                SemanticContractError,
            ) as caught:
                check_semantic_contract(sql, question, self._threshold_entity_tables())
            violation = caught.exception.violation
            self.assertEqual(
                "semantic.filter.row_threshold_in_having",
                violation.violation_code,
            )
            self.assertEqual("llm", violation.repairability)
            self.assertTrue(violation.safe_to_retry)

    def test_row_numeric_threshold_in_where_is_accepted(self):
        check_semantic_contract(
            "SELECT s.store_name, SUM(f.amount) FROM fact_activity f "
            "JOIN dim_store s ON f.store_id=s.store_id "
            "WHERE f.units >= 3 GROUP BY s.store_name",
            "购买件数至少3时，按门店统计交易金额",
            self._threshold_entity_tables(),
        )
        check_semantic_contract(
            "SELECT e.employee_name, SUM(f.units) FROM fact_activity f "
            "JOIN dim_employee e ON f.employee_id=e.employee_id "
            "WHERE f.amount > 100 GROUP BY e.employee_name",
            "列出订单金额超过100的记录并按员工汇总购买数量",
            self._threshold_entity_tables(),
        )

    def test_aggregate_measure_threshold_is_not_a_row_filter(self):
        cases = (
            (
                "累计购买件数至少3的门店",
                "s.store_name", "dim_store s", "f.store_id=s.store_id",
                "SUM(f.units) >= 3",
            ),
            (
                "合计订单金额超过100的员工",
                "e.employee_name", "dim_employee e", "f.employee_id=e.employee_id",
                "SUM(f.amount) > 100",
            ),
            (
                "平均服务分数不低于80的地区",
                "r.region_name", "dim_region r", "f.region_id=r.region_id",
                "AVG(f.score) >= 80",
            ),
        )
        for question, entity_column, dimension, join_on, having in cases:
            with self.subTest(question=question):
                check_semantic_contract(
                    f"SELECT {entity_column} FROM fact_activity f JOIN {dimension} "
                    f"ON {join_on} GROUP BY {entity_column} HAVING {having}",
                    question,
                    self._threshold_entity_tables(),
                )

    def test_aggregate_threshold_entity_count_requires_scalar_final_select(self):
        cases = (
            (
                "累计交易金额超过1000的门店有多少",
                "s.store_name", "dim_store s", "f.store_id=s.store_id",
                "SUM(f.amount) > 1000",
            ),
            (
                "有多少员工的平均绩效分数不低于80",
                "e.employee_name", "dim_employee e", "f.employee_id=e.employee_id",
                "AVG(f.score) >= 80",
            ),
            (
                "合计销售额超过5000的地区中有多少个地区",
                "r.region_name", "dim_region r", "f.region_id=r.region_id",
                "SUM(f.amount) > 5000",
            ),
            (
                "有多少省份的累计交易金额超过1000万元",
                "r.province", "dim_region r", "f.region_id=r.region_id",
                "SUM(f.amount) > 1000",
            ),
            (
                "总计购买件数至少20的商品数量",
                "p.product_name", "dim_product p", "f.product_id=p.product_id",
                "SUM(f.units) >= 20",
            ),
        )
        for question, entity_column, dimension, join_on, having in cases:
            sql = (
                f"SELECT {entity_column} FROM fact_activity f JOIN {dimension} "
                f"ON {join_on} GROUP BY {entity_column} HAVING {having}"
            )
            with self.subTest(question=question), self.assertRaises(
                SemanticContractError,
            ) as caught:
                check_semantic_contract(sql, question, self._threshold_entity_tables())
            violation = caught.exception.violation
            self.assertEqual(
                "semantic.output.scalar_count_required",
                violation.violation_code,
            )
            self.assertEqual("llm", violation.repairability)
            self.assertTrue(violation.safe_to_retry)

    def test_aggregate_threshold_entity_count_accepts_outer_scalar_count(self):
        check_semantic_contract(
            "SELECT COUNT(*) AS entity_count FROM ("
            "SELECT f.store_id FROM fact_activity f GROUP BY f.store_id "
            "HAVING SUM(f.amount) > 1000) qualified",
            "累计交易金额超过1000的门店有多少",
            self._threshold_entity_tables(),
        )
        check_semantic_contract(
            "SELECT COUNT(*) AS entity_count FROM ("
            "SELECT f.employee_id FROM fact_activity f GROUP BY f.employee_id "
            "HAVING AVG(f.score) >= 80) qualified",
            "有多少员工的平均绩效分数不低于80",
            self._threshold_entity_tables(),
        )

    def test_non_count_entity_listing_and_simple_month_count_are_unchanged(self):
        check_semantic_contract(
            "SELECT s.store_name FROM fact_activity f "
            "JOIN dim_store s ON f.store_id=s.store_id "
            "GROUP BY s.store_name HAVING SUM(f.amount) > 1000",
            "列出累计交易金额超过1000的门店",
            self._threshold_entity_tables(),
        )
        check_semantic_contract(
            "SELECT d.month, COUNT(f.order_id) FROM fact_order f "
            "JOIN dim_date d ON f.date_id=d.date_id GROUP BY d.month",
            "每月有多少订单",
            self._retail_contract_tables(),
        )
        check_semantic_contract(
            "SELECT r.province, COUNT(*) FROM fact_activity f "
            "JOIN dim_region r ON f.region_id=r.region_id GROUP BY r.province",
            "每个省份有多少活动记录",
            self._threshold_entity_tables(),
        )
        check_semantic_contract(
            "SELECT r.province, COUNT(*) FROM fact_activity f "
            "JOIN dim_region r ON f.region_id=r.region_id "
            "GROUP BY r.province HAVING SUM(f.amount) > 1000",
            "每个省份有多少活动记录，且累计交易金额超过1000",
            self._threshold_entity_tables(),
        )

    def test_filtered_derived_key_left_join_satisfies_anti_contract(self):
        check_semantic_contract(
            "SELECT p.product_name FROM dim_product p "
            "LEFT JOIN (SELECT DISTINCT f.product_id FROM fact_order f "
            "JOIN dim_date d ON f.date_id = d.date_id WHERE d.year = 2025) active "
            "ON p.product_id = active.product_id "
            "WHERE active.product_id IS NULL ORDER BY p.product_name",
            "2025年完全没有卖出的商品",
            self._retail_contract_tables(),
        )

    def test_conditional_period_absence_satisfies_anti_contract(self):
        check_semantic_contract(
            "SELECT r.province FROM fact_order f "
            "JOIN dim_region r ON f.region_id = r.region_id "
            "JOIN dim_date d ON f.date_id = d.date_id "
            "WHERE d.year IN (2024, 2026) GROUP BY r.province "
            "HAVING SUM(CASE WHEN d.year = 2024 THEN 1 ELSE 0 END) > 0 "
            "AND SUM(CASE WHEN d.year = 2026 THEN 1 ELSE 0 END) = 0 "
            "ORDER BY r.province",
            "2024年有销售但2026年没有销售的省份",
            self._retail_contract_tables(),
        )

    def test_last_event_before_period_satisfies_period_absence_contract(self):
        check_semantic_contract(
            "SELECT c.customer_name, MAX(d.year) AS last_purchase_year "
            "FROM fact_order f "
            "JOIN dim_customer c ON f.customer_id = c.customer_id "
            "JOIN dim_date d ON f.date_id = d.date_id "
            "GROUP BY c.customer_id, c.customer_name "
            "HAVING MAX(d.year) < 2026 ORDER BY c.customer_name",
            "找出2026年以前买过但2026年没有下单的客户",
            self._retail_contract_tables(),
        )

    def test_follow_up_scalar_replacement_is_not_a_grouping_axis(self):
        check_semantic_contract(
            "SELECT p.category, SUM(f.order_amount) AS total_amount "
            "FROM fact_order f "
            "JOIN dim_date d ON f.date_id = d.date_id "
            "JOIN dim_region r ON f.region_id = r.region_id "
            "JOIN dim_product p ON f.product_id = p.product_id "
            "WHERE d.year = 2025 AND r.region_name = '华南' "
            "GROUP BY p.category ORDER BY total_amount DESC",
            "2025年华东各品类销售额，换成华南地区",
            self._retail_contract_tables(),
        )

    def test_rank_projection_is_allowed_when_rank_is_requested(self):
        check_semantic_contract(
            "WITH grouped_values AS ("
            "SELECT r.region_name, p.category, SUM(f.order_amount) AS total_amount "
            "FROM fact_order f "
            "JOIN dim_region r ON f.region_id = r.region_id "
            "JOIN dim_product p ON f.product_id = p.product_id "
            "GROUP BY r.region_name, p.category), ranked AS ("
            "SELECT region_name, category, total_amount, "
            "ROW_NUMBER() OVER (PARTITION BY region_name ORDER BY total_amount DESC) "
            "AS category_rank FROM grouped_values) "
            "SELECT region_name, category, total_amount, category_rank "
            "FROM ranked WHERE category_rank <= 3 "
            "ORDER BY region_name, category_rank",
            "每个大区销售额前三品类",
            self._retail_contract_tables(),
        )

    def test_counted_month_grain_may_be_aggregated_into_bands(self):
        check_semantic_contract(
            "WITH monthly AS ("
            "SELECT d.month, SUM(f.order_amount) AS total_amount "
            "FROM fact_order f JOIN dim_date d ON f.date_id = d.date_id "
            "WHERE d.year = 2025 GROUP BY d.month), compared AS ("
            "SELECT month, total_amount, AVG(total_amount) OVER () AS average_amount "
            "FROM monthly) "
            "SELECT CASE WHEN total_amount >= average_amount THEN '高于或等于月均' "
            "ELSE '低于月均' END AS amount_band, COUNT(*) AS month_count "
            "FROM compared GROUP BY CASE WHEN total_amount >= average_amount "
            "THEN '高于或等于月均' ELSE '低于月均' END ORDER BY amount_band",
            "2025年高于和低于月均销售额的月份各有几个",
            self._retail_contract_tables(),
        )

    def test_scalar_quarter_filter_does_not_require_quarter_grouping(self):
        check_semantic_contract(
            "SELECT c.member_level, COUNT(f.order_id) AS order_count "
            "FROM fact_order f "
            "JOIN dim_date d ON f.date_id = d.date_id "
            "JOIN dim_customer c ON f.customer_id = c.customer_id "
            "WHERE d.year = 2026 AND d.quarter = 1 "
            "GROUP BY c.member_level",
            "2026年第一季度各会员等级的订单笔数",
            self._quarter_member_tables(),
        )

    def test_customer_ratio_is_measure_not_customer_name_grouping_axis(self):
        check_semantic_contract(
            "WITH customer_orders AS ("
            "SELECT f.customer_id, COUNT(DISTINCT f.order_id) AS order_count "
            "FROM fact_order f JOIN dim_date d ON f.date_id = d.date_id "
            "WHERE d.year = 2026 GROUP BY f.customer_id) "
            "SELECT c.member_level, "
            "SUM(CASE WHEN co.order_count >= 2 THEN 1 ELSE 0 END) / "
            "NULLIF(COUNT(*), 0) AS repurchase_rate "
            "FROM customer_orders co JOIN dim_customer c "
            "ON co.customer_id = c.customer_id GROUP BY c.member_level",
            "2026年按会员等级计算至少下单两次客户占下单客户的比例",
            self._quarter_member_tables(),
        )

    def test_multi_quarter_axis_still_requires_quarter_projection_and_grouping(self):
        with self.assertRaisesRegex(SemanticContractError, "季度粒度"):
            check_semantic_contract(
                "SELECT c.member_level, COUNT(f.order_id) AS order_count "
                "FROM fact_order f "
                "JOIN dim_date d ON f.date_id = d.date_id "
                "JOIN dim_customer c ON f.customer_id = c.customer_id "
                "WHERE d.year = 2026 "
                "GROUP BY c.member_level",
                "2026年各季度各会员等级的订单笔数",
                self._quarter_member_tables(),
            )

    def test_physical_measure_can_be_an_explicit_grouping_axis(self):
        tables = self._retail_contract_tables()
        check_semantic_contract(
            "SELECT f.order_quantity, COUNT(f.order_id) AS order_count "
            "FROM fact_order f GROUP BY f.order_quantity",
            "按购买件数汇总订单笔数",
            tables,
        )
        with self.assertRaisesRegex(SemanticContractError, "order_quantity"):
            check_semantic_contract(
                "SELECT COUNT(f.order_id) AS order_count FROM fact_order f",
                "按购买件数汇总订单笔数",
                tables,
            )

    def test_ordering_phrase_is_not_misread_as_grouping(self):
        tables = self._retail_contract_tables()
        product_name = next(
            column
            for table in tables if table["name"] == "dim_product"
            for column in table["columns"] if column["name"] == "product_name"
        )
        product_name["alias"] = ["商品", "产品"]
        check_semantic_contract(
            "SELECT p.product_name FROM dim_product p "
            "ORDER BY p.product_name ASC LIMIT 5",
            "按商品名称升序列出前五个商品",
            tables,
        )

        with self.assertRaisesRegex(SemanticContractError, "product_name"):
            check_semantic_contract(
                "SELECT p.product_name FROM fact_order f "
                "JOIN dim_product p ON f.product_id = p.product_id "
                "ORDER BY f.order_quantity DESC LIMIT 5",
                "按销量降序列出前五个商品",
                tables,
            )

    def test_counted_entity_does_not_hide_independent_grouping_axis(self):
        tables = self._retail_contract_tables()
        by_name = {
            column["name"]: column
            for table in tables
            for column in table["columns"]
        }
        by_name["category"].update(
            description="商品所属品类", alias=["商品类别", "品类"],
        )
        by_name["brand"].update(
            description="商品品牌名称", alias=["品牌", "品牌名称"],
        )
        by_name["region_name"].update(
            description="订单所属的大区名称", alias=["地区", "大区"],
        )
        customer_table = next(
            table for table in tables if table["name"] == "dim_customer"
        )
        customer_table["columns"].append({
            "name": "member_level",
            "role": "dimension",
            "description": "客户会员等级",
            "alias": ["会员等级"],
        })
        cases = (
            ("按会员等级统计客户数", "member_level"),
            ("按商品类别统计商品数", "category"),
            ("按地区统计订单数", "region_name"),
            ("按品牌统计商品数", "brand"),
        )
        for question, expected_column in cases:
            with self.subTest(question=question):
                with self.assertRaisesRegex(
                    SemanticContractError, expected_column,
                ):
                    check_semantic_contract(
                        "SELECT COUNT(*) AS entity_count FROM fact_order f",
                        question,
                        tables,
                    )

    def test_fact_foreign_key_can_group_an_explicit_dimension_key(self):
        tables = [
            {
                "name": "fact_event",
                "role": "fact",
                "description": "事件事实表",
                "columns": [
                    {"name": "event_id", "role": "primary_key"},
                    {"name": "product_id", "role": "foreign_key"},
                    {"name": "customer_id", "role": "foreign_key"},
                    {"name": "region_id", "role": "foreign_key"},
                    {"name": "date_id", "role": "foreign_key"},
                ],
            },
            *[
                {
                    "name": table_name,
                    "role": "dim",
                    "description": f"{entity}维度",
                    "columns": [
                        {
                            "name": key,
                            "role": "primary_key",
                            "description": f"{entity}唯一标识",
                            "alias": [f"{entity}ID", entity],
                        },
                        {
                            "name": attribute,
                            "role": "dimension",
                            "description": attribute_alias,
                            "alias": [attribute_alias],
                        },
                    ],
                }
                for table_name, entity, key, attribute, attribute_alias in (
                    ("dim_product", "商品", "product_id", "product_name", "商品名称"),
                    ("dim_customer", "客户", "customer_id", "customer_name", "客户名称"),
                    ("dim_region", "地区", "region_id", "region_name", "地区名称"),
                    ("dim_date", "日期", "date_id", "year", "年份"),
                )
            ],
        ]
        cases = (
            ("商品ID", "product_id"),
            ("客户ID", "customer_id"),
            ("地区ID", "region_id"),
            ("日期ID", "date_id"),
        )
        for key_label, column_name in cases:
            with self.subTest(key_label=key_label):
                check_semantic_contract(
                    f"SELECT f.{column_name}, COUNT(*) AS event_count "
                    f"FROM fact_event f GROUP BY f.{column_name}",
                    f"按{key_label}统计事件数",
                    tables,
                )
        with self.assertRaisesRegex(SemanticContractError, "对应维表"):
            check_semantic_contract(
                "SELECT f.date_id, COUNT(*) AS event_count "
                "FROM fact_event f GROUP BY f.date_id",
                "按日期统计关键事件数",
                tables,
            )

    def test_dimension_attribute_still_requires_its_owner_join(self):
        tables = [
            {
                "name": "fact_event",
                "role": "fact",
                "description": "事件事实表",
                "columns": [
                    {"name": "event_id", "role": "primary_key"},
                    {"name": "product_id", "role": "foreign_key"},
                    {"name": "customer_id", "role": "foreign_key"},
                    {"name": "region_id", "role": "foreign_key"},
                    {"name": "date_id", "role": "foreign_key"},
                ],
            },
            *[
                {
                    "name": table_name,
                    "role": "dim",
                    "description": f"{entity}维度",
                    "columns": [
                        {
                            "name": key,
                            "role": "primary_key",
                            "alias": [f"{entity}ID"],
                        },
                        {
                            "name": attribute,
                            "role": "dimension",
                            "description": attribute_alias,
                            "alias": [attribute_alias],
                        },
                    ],
                }
                for table_name, entity, key, attribute, attribute_alias in (
                    ("dim_product", "商品", "product_id", "product_name", "商品名称"),
                    ("dim_customer", "客户", "customer_id", "customer_name", "客户名称"),
                    ("dim_region", "地区", "region_id", "region_name", "地区名称"),
                    ("dim_date", "日期", "date_id", "year", "年份"),
                )
            ],
        ]
        cases = (
            ("商品名称", "product_name"),
            ("客户名称", "customer_name"),
            ("地区名称", "region_name"),
            ("年份", "year"),
        )
        for attribute_label, column_name in cases:
            with self.subTest(attribute_label=attribute_label):
                with self.assertRaisesRegex(
                    SemanticContractError, "对应维表",
                ):
                    check_semantic_contract(
                        f"SELECT {column_name}, COUNT(*) AS event_count "
                        f"FROM fact_event GROUP BY {column_name}",
                        f"按{attribute_label}统计事件数",
                        tables,
                    )

    def test_explicit_product_id_does_not_authorize_product_name_alias(self):
        tables = self._retail_contract_tables()
        product_name = next(
            column
            for table in tables if table["name"] == "dim_product"
            for column in table["columns"] if column["name"] == "product_name"
        )
        product_name.update(
            alias=["商品", "产品"], is_display=True,
        )
        with self.assertRaisesRegex(SemanticContractError, "不得额外投影"):
            check_semantic_contract(
                "SELECT f.product_id, p.product_name, COUNT(*) AS order_count "
                "FROM fact_order f JOIN dim_product p "
                "ON f.product_id = p.product_id "
                "GROUP BY f.product_id, p.product_name "
                "ORDER BY order_count DESC, f.product_id ASC LIMIT 10",
                "订单笔数最多的十个商品ID",
                tables,
            )

    def test_time_group_does_not_promote_counted_product_measure_to_entity(self):
        tables = self._retail_contract_tables()
        check_semantic_contract(
            "SELECT d.quarter, SUM(f.order_quantity) AS total_quantity "
            "FROM fact_order f JOIN dim_date d ON f.date_id=d.date_id "
            "WHERE d.year=2024 GROUP BY d.quarter",
            "2024年每个季度售出的商品件数",
            [
                *[table for table in tables if table["name"] != "dim_date"],
                {
                    "name": "dim_date", "role": "dim", "description": "日期维度表",
                    "columns": [
                        {"name": "date_id", "role": "primary_key", "description": "日期编号"},
                        {"name": "year", "role": "dimension", "description": "年份"},
                        {"name": "quarter", "role": "dimension", "description": "季度"},
                    ],
                },
            ],
        )

    def test_ranked_attribute_does_not_force_catalog_entity_name(self):
        check_semantic_contract(
            "SELECT p.brand, COUNT(*) AS product_count FROM dim_product p "
            "GROUP BY p.brand ORDER BY product_count DESC, p.brand ASC LIMIT 5",
            "目录内数量最多的五个品牌",
            self._retail_contract_tables(),
        )

    def test_directory_attribute_projection_comes_from_requested_metadata(self):
        tables = [
            {
                "name": "dim_device",
                "role": "dim",
                "description": "设备目录表",
                "columns": [
                    {
                        "name": "device_id",
                        "role": "primary_key",
                        "description": "设备编号",
                    },
                    {
                        "name": "device_name",
                        "role": "dimension",
                        "description": "设备名称",
                        "is_display": True,
                    },
                    {
                        "name": "device_type",
                        "role": "dimension",
                        "description": "设备类型",
                    },
                    {
                        "name": "vendor",
                        "role": "dimension",
                        "description": "厂商",
                    },
                    {
                        "name": "owner_email",
                        "role": "dimension",
                        "description": "负责人邮箱",
                        "sensitive": True,
                    },
                ],
            },
        ]
        sql = "SELECT DISTINCT device_type, vendor FROM dim_device"
        for wording in (
            "列出设备目录中的不同类型和厂商组合",
            "有哪些设备的类型和厂商组合",
        ):
            with self.subTest(wording=wording):
                check_semantic_contract(sql, wording, tables)

    def test_directory_attribute_projection_rejects_missing_and_unrequested_fields(self):
        tables = [
            {
                "name": "dim_device",
                "role": "dim",
                "description": "设备目录表",
                "columns": [
                    {"name": "device_id", "role": "primary_key", "description": "设备编号"},
                    {
                        "name": "device_name",
                        "role": "dimension",
                        "description": "设备名称",
                        "is_display": True,
                    },
                    {"name": "device_type", "role": "dimension", "description": "设备类型"},
                    {"name": "vendor", "role": "dimension", "description": "厂商"},
                    {
                        "name": "owner_email",
                        "role": "dimension",
                        "description": "负责人邮箱",
                        "sensitive": True,
                    },
                ],
            },
        ]
        question = "列出设备目录中的不同类型和厂商组合"
        for sql, expected_code in (
            (
                "SELECT DISTINCT device_name FROM dim_device",
                "semantic.projection.missing_requested_attribute",
            ),
            (
                "SELECT DISTINCT device_id FROM dim_device",
                "semantic.projection.id_only",
            ),
            (
                "SELECT DISTINCT device_type, vendor, device_name FROM dim_device",
                "semantic.projection.unrequested_extra",
            ),
            (
                "SELECT DISTINCT device_type, vendor, owner_email FROM dim_device",
                "semantic.projection.unrequested_extra",
            ),
        ):
            with self.subTest(sql=sql), self.assertRaises(
                SemanticContractError,
            ) as raised:
                check_semantic_contract(sql, question, tables)
            self.assertEqual(raised.exception.violation.violation_code, expected_code)

    def test_identifier_and_attributes_share_one_projection_contract(self):
        tables = self._device_directory_tables()
        question = "列出设备ID和厂商"
        for sql in (
            "SELECT device_id FROM dim_device",
            "SELECT owner_email FROM dim_device",
        ):
            with self.subTest(sql=sql), self.assertRaises(SemanticContractError):
                check_semantic_contract(sql, question, tables)

        with self.assertRaises(SemanticContractError):
            check_semantic_contract(
                "SELECT device_id FROM dim_device WHERE device_id='D7'",
                "列出设备编号D7对应的厂商",
                tables,
            )
        check_semantic_contract(
            "SELECT device_id, vendor FROM dim_device",
            question,
            tables,
        )

    def test_requested_directory_fields_require_direct_owner_projection(self):
        tables = self._device_directory_tables()
        with self.assertRaises(SemanticContractError):
            check_semantic_contract(
                "SELECT CONCAT(device_type, ':', vendor) AS combined "
                "FROM dim_device",
                "列出设备目录中的不同类型和厂商组合",
                tables,
            )
        with self.assertRaises(SemanticContractError):
            check_semantic_contract(
                "SELECT v.status FROM dim_device d JOIN dim_vendor v "
                "ON d.vendor_id=v.vendor_id",
                "列出设备目录中的设备状态",
                tables,
            )

    def test_unrequested_sensitive_aggregate_lineage_is_rejected(self):
        tables = self._device_directory_tables()
        with self.assertRaises(SemanticContractError):
            check_semantic_contract(
                "SELECT vendor, MAX(owner_email) AS leaked_email "
                "FROM dim_device GROUP BY vendor",
                "列出设备目录中的厂商",
                tables,
            )
        # 显式请求仍交给既有 access policy 与 masking 决定授权和脱敏。
        check_semantic_contract(
            "SELECT owner_email FROM dim_device",
            "列出设备目录中的负责人邮箱",
            tables,
        )

    def test_directory_filter_and_rank_criteria_are_not_output_fields(self):
        tables = self._device_directory_tables()
        check_semantic_contract(
            "SELECT vendor FROM dim_device WHERE device_type='传感器'",
            "列出传感器类型设备的厂商",
            tables,
        )
        check_semantic_contract(
            "SELECT vendor FROM dim_device ORDER BY device_type LIMIT 5",
            "列出按设备类型排名前五的厂商",
            tables,
        )
        check_semantic_contract(
            "SELECT vendor, COUNT(*) AS device_count "
            "FROM dim_device GROUP BY vendor",
            "列出每个厂商的设备数量",
            tables,
        )

    def test_explicit_identifier_rejects_unasked_readable_name(self):
        with self.assertRaisesRegex(SemanticContractError, "不得额外投影"):
            check_semantic_contract(
                "SELECT f.product_id, p.product_name, COUNT(*) AS order_count "
                "FROM fact_order f JOIN dim_product p ON f.product_id=p.product_id "
                "GROUP BY f.product_id, p.product_name "
                "ORDER BY order_count DESC LIMIT 10",
                "订单笔数最多的十个商品ID",
                self._retail_contract_tables(),
            )

    def test_longest_metadata_term_generalizes_to_unseen_entities(self):
        for entity, label in (("employee", "员工"), ("store", "门店")):
            key = f"{entity}_id"
            name = f"{entity}_name"
            dimension = f"dim_{entity}"
            tables = [
                {
                    "name": "fact_event", "role": "fact",
                    "columns": [
                        {
                            "name": key, "role": "foreign_key",
                            "alias": [f"{label}ID"],
                        },
                        {
                            "name": "score", "role": "measure",
                            "alias": ["绩效分数"],
                        },
                    ],
                },
                {
                    "name": dimension, "role": "dim",
                    "description": f"{label}维度",
                    "columns": [
                        {
                            "name": key, "role": "primary_key",
                            "alias": [f"{label}ID", f"{label}编号"],
                        },
                        {
                            "name": name, "role": "dimension",
                            "alias": [label, f"{label}名称"],
                            "is_display": True,
                        },
                        {
                            "name": "department", "role": "dimension",
                            "alias": ["部门"],
                        },
                    ],
                },
            ]
            with self.subTest(entity=entity, contract="ordered_directory"):
                check_semantic_contract(
                    f"SELECT e.{name} FROM {dimension} e "
                    f"ORDER BY e.{name} ASC LIMIT 5",
                    f"按{label}名称升序列出前五个{label}",
                    tables,
                )
            for wording in (
                f"列出前五个{label}",
                f"{label}名单",
                f"列出所有{label}",
            ):
                with self.subTest(
                    entity=entity, contract="pure_entity_list", wording=wording,
                ):
                    check_semantic_contract(
                        f"SELECT e.{name} FROM {dimension} e LIMIT 5",
                        wording,
                        tables,
                    )
                    with self.assertRaisesRegex(
                        SemanticContractError, "不得只投影 ID",
                    ):
                        check_semantic_contract(
                            f"SELECT e.{key} FROM {dimension} e LIMIT 5",
                            wording,
                            tables,
                        )
            with self.subTest(entity=entity, contract="explicit_id"):
                check_semantic_contract(
                    f"SELECT f.{key}, COUNT(*) AS event_count "
                    f"FROM fact_event f GROUP BY f.{key} "
                    f"ORDER BY event_count DESC, f.{key} ASC LIMIT 10",
                    f"事件数最多的十个{label}ID",
                    tables,
                )
            with self.subTest(entity=entity, contract="id_rejects_name"):
                with self.assertRaisesRegex(SemanticContractError, "不得额外投影"):
                    check_semantic_contract(
                        f"SELECT f.{key}, e.{name}, COUNT(*) AS event_count "
                        f"FROM fact_event f JOIN {dimension} e "
                        f"ON f.{key}=e.{key} GROUP BY f.{key}, e.{name} "
                        f"ORDER BY event_count DESC, f.{key} ASC LIMIT 10",
                        f"事件数最多的十个{label}ID",
                        tables,
                    )

        employee_tables = [
            {
                "name": "fact_event", "role": "fact",
                "columns": [
                    {"name": "employee_id", "role": "foreign_key", "alias": ["员工ID"]},
                    {"name": "score", "role": "measure", "alias": ["绩效分数"]},
                ],
            },
            {
                "name": "dim_employee", "role": "dim",
                "columns": [
                    {"name": "employee_id", "role": "primary_key", "alias": ["员工ID"]},
                    {"name": "employee_name", "role": "dimension", "alias": ["员工", "员工名称"]},
                    {"name": "department", "role": "dimension", "alias": ["部门"]},
                ],
            },
        ]
        check_semantic_contract(
            "SELECT e.department, COUNT(*) AS employee_count "
            "FROM dim_employee e GROUP BY e.department",
            "按部门统计员工数",
            employee_tables,
        )
        check_semantic_contract(
            "SELECT f.score, COUNT(*) AS event_count "
            "FROM fact_event f GROUP BY f.score",
            "按绩效分数统计事件数",
            employee_tables,
        )
        with self.assertRaisesRegex(SemanticContractError, "employee_name"):
            check_semantic_contract(
                "SELECT e.employee_name FROM fact_event f "
                "JOIN dim_employee e ON f.employee_id=e.employee_id "
                "ORDER BY f.score DESC LIMIT 5",
                "按绩效分数降序列出前五个员工",
                employee_tables,
            )
        check_semantic_contract(
            "SELECT e.employee_name, SUM(f.score) AS total_score "
            "FROM fact_event f JOIN dim_employee e "
            "ON f.employee_id=e.employee_id GROUP BY e.employee_name "
            "ORDER BY total_score DESC LIMIT 5",
            "按绩效分数降序列出前五个员工",
            employee_tables,
        )

    def test_pure_product_listing_and_unseen_anti_join_do_not_require_grouping(self):
        for wording in ("商品名单", "列出所有商品", "列出前五个商品"):
            with self.subTest(wording=wording):
                check_semantic_contract(
                    "SELECT p.product_name FROM dim_product p LIMIT 5",
                    wording,
                    self._retail_contract_tables(),
                )

        employee_tables = [
            {
                "name": "fact_event", "role": "fact",
                "columns": [
                    {"name": "employee_id", "role": "foreign_key", "alias": ["员工ID"]},
                ],
            },
            {
                "name": "dim_employee", "role": "dim",
                "description": "员工维度",
                "columns": [
                    {"name": "employee_id", "role": "primary_key", "alias": ["员工ID"]},
                    {
                        "name": "employee_name", "role": "dimension",
                        "alias": ["员工", "员工名称"], "is_display": True,
                    },
                ],
            },
        ]
        check_semantic_contract(
            "SELECT e.employee_name FROM dim_employee e "
            "LEFT JOIN fact_event f ON e.employee_id=f.employee_id "
            "WHERE f.employee_id IS NULL",
            "没有产生事件的员工",
            employee_tables,
        )
        with self.assertRaisesRegex(SemanticContractError, "不得只投影 ID"):
            check_semantic_contract(
                "SELECT e.employee_id FROM dim_employee e "
                "LEFT JOIN fact_event f ON e.employee_id=f.employee_id "
                "WHERE f.employee_id IS NULL",
                "没有产生事件的员工",
                employee_tables,
            )

    def test_guided_model_ir_does_not_create_self_proving_join_requirement(self):
        ir = {
            "status": "complete",
            "dimensions": [{"table": "dim_region", "column": "province"}],
            "measures": [{"table": "fact_order", "column": "order_amount"}],
            "model_sql_policy": {"use_model": True},
            "query_contract": {"authority": "guided"},
        }
        shape = analyze_semantic_shape(
            "SELECT SUM(order_amount) FROM fact_order",
            "成交金额",
            self._retail_contract_tables(),
            ir,
        )
        self.assertFalse(shape["join_required"])
        self.assertEqual(0, shape["required_table_count"])

    def test_having_amount_is_returned_when_question_requests_the_threshold_measure(self):
        sql = (
            "SELECT c.customer_name FROM fact_order f JOIN dim_customer c "
            "ON f.customer_id=c.customer_id GROUP BY c.customer_name "
            "HAVING SUM(f.order_amount) > 10000"
        )
        repaired = stabilize_threshold_measure_projection(
            sql, "累计消费金额超过10000的客户"
        )
        self.assertIn("SUM(f.order_amount) AS total_amount", repaired)
        self.assertEqual(stabilize_threshold_measure_projection(repaired, "累计消费金额超过10000的客户"), repaired)
    def test_invalid_join_is_repaired_only_with_unique_metadata_edge(self):
        bad = (
            "SELECT r.province, SUM(f.order_amount) FROM fact_order f "
            "JOIN dim_region r ON f.province=r.province GROUP BY r.province"
        )
        tables = [
            {
                "name": "fact_order", "columns": [
                    {"name": "region_id", "role": "foreign_key"},
                    {"name": "order_amount", "role": "measure"},
                ],
            },
            {
                "name": "dim_region", "columns": [
                    {"name": "region_id", "role": "primary_key"},
                    {"name": "province", "role": "dimension"},
                ],
            },
        ]
        repaired = repair_invalid_metadata_joins(bad, tables)
        self.assertIn("f.region_id = r.region_id", repaired)
        valid = bad.replace("f.province=r.province", "f.region_id=r.region_id")
        self.assertEqual(repair_invalid_metadata_joins(valid, tables), valid)

    def test_invalid_join_uses_inferred_id_edge_without_physical_fk(self):
        tables = [
            {"name": "fact_order", "columns": [
                {"name": "region_id", "role": "dimension"},
                {"name": "order_id", "role": "primary_key"},
                {"name": "order_amount", "role": "dimension"},
            ]},
            {"name": "dim_region", "columns": [
                {"name": "region_id", "role": "primary_key"},
                {"name": "province", "role": "dimension"},
            ]},
        ]
        repaired = repair_invalid_metadata_joins(
            "SELECT r.province, SUM(f.order_amount) FROM fact_order f "
            "JOIN dim_region r ON f.province=r.province GROUP BY r.province",
            tables,
        )
        self.assertIn("f.region_id = r.region_id", repaired)

    def test_month_axis_rejects_date_id_arithmetic(self):
        with self.assertRaisesRegex(SemanticContractError, "可读字段"):
            check_semantic_contract(
                "SELECT date_id DIV 100 ym, SUM(order_amount) FROM fact_order GROUP BY ym",
                "2025年各月销售额趋势", TABLES,
            )
        check_semantic_contract(
            "SELECT d.month, SUM(f.order_amount) FROM fact_order f JOIN dim_date d ON f.date_id=d.date_id GROUP BY d.month",
            "2025年各月销售额趋势", TABLES,
        )

    def test_cte_grouping_proof_is_accepted_for_window_and_top_per_group(self):
        month_sql = (
            "WITH monthly AS (SELECT d.month, SUM(f.order_amount) total_amount "
            "FROM fact_order f JOIN dim_date d ON f.date_id=d.date_id "
            "GROUP BY d.month) "
            "SELECT month, total_amount, LAG(total_amount) OVER (ORDER BY month) previous_amount "
            "FROM monthly ORDER BY month"
        )
        check_semantic_contract(
            month_sql,
            "2025年每月销售额及其环比增长率",
            TABLES,
        )
        ranking_sql = (
            "WITH grouped_values AS ("
            "SELECT r.region_name, p.product_name, SUM(f.order_amount) total_amount "
            "FROM fact_order f JOIN dim_region r ON f.region_id=r.region_id "
            "JOIN dim_product p ON f.product_id=p.product_id "
            "GROUP BY r.region_name, p.product_name), ranked AS ("
            "SELECT region_name, product_name, total_amount, "
            "ROW_NUMBER() OVER (PARTITION BY region_name ORDER BY total_amount DESC) rn "
            "FROM grouped_values) "
            "SELECT region_name, product_name, total_amount FROM ranked WHERE rn=1"
        )
        ranking_tables = [
            *TABLES,
            {
                "name": "dim_region",
                "role": "dim",
                "description": "地区维度表",
                "columns": [
                    {"name": "region_id", "role": "primary_key", "alias": ["地区ID"]},
                    {"name": "region_name", "role": "dimension", "alias": ["地区"]},
                ],
            },
        ]
        check_semantic_contract(
            ranking_sql,
            "每个地区销售额最高的商品",
            ranking_tables,
        )

    def test_union_all_counts_dimension_projected_by_the_dw_branch(self):
        tables = [
            {
                "name": "upload_1_aaaaaaaa",
                "role": "fact",
                "description": "用户上传订单",
                "columns": [
                    {
                        "name": "c_003",
                        "role": "dimension",
                        "description": "地区",
                        "alias": ["地区"],
                    },
                    {
                        "name": "c_005",
                        "role": "measure",
                        "description": "金额",
                        "alias": ["销售额"],
                    },
                ],
            },
            {
                "name": "fact_order",
                "role": "fact",
                "description": "订单事实表",
                "columns": [
                    {"name": "region_id", "role": "foreign_key"},
                    {
                        "name": "order_amount",
                        "role": "measure",
                        "description": "销售额",
                        "alias": ["销售总额"],
                    },
                ],
            },
            {
                "name": "dim_region",
                "role": "dim",
                "description": "地区维度表",
                "columns": [
                    {"name": "region_id", "role": "primary_key"},
                    {
                        "name": "region_name",
                        "role": "dimension",
                        "description": "地区名称",
                        "alias": ["地区"],
                    },
                ],
            },
        ]
        sql = (
            "SELECT 'upload' AS source_label, c_003 AS region, "
            "SUM(c_005) AS total_amount "
            "FROM upload_1_aaaaaaaa GROUP BY c_003 "
            "UNION ALL "
            "SELECT 'dw' AS source_label, r.region_name AS region, "
            "SUM(f.order_amount) AS total_amount "
            "FROM fact_order f JOIN dim_region r ON f.region_id = r.region_id "
            "GROUP BY r.region_name"
        )
        check_semantic_contract(
            sql,
            "对比上传文件和正式DW各地区销售总额",
            tables,
        )

    def test_cte_grouped_dimension_still_fails_when_final_projection_drops_it(self):
        sql = (
            "WITH monthly AS (SELECT d.month, SUM(f.order_amount) total_amount "
            "FROM fact_order f JOIN dim_date d ON f.date_id=d.date_id GROUP BY d.month) "
            "SELECT total_amount FROM monthly"
        )
        with self.assertRaisesRegex(SemanticContractError, "可读字段"):
            check_semantic_contract(
                sql,
                "2025年每月销售额趋势",
                TABLES,
            )

    def test_entity_list_rejects_id_only_projection(self):
        with self.assertRaisesRegex(SemanticContractError, "不得只投影 ID"):
            check_semantic_contract(
                "SELECT product_id, SUM(order_amount) FROM fact_order GROUP BY product_id",
                "销量最低的五个商品", TABLES,
            )

    def test_aggregated_entity_id_is_a_measure_not_an_id_projection(self):
        tables = [
            *TABLES,
            {
                "name": "dim_customer",
                "role": "dim",
                "description": "客户维度表",
                "columns": [
                    {
                        "name": "customer_id",
                        "role": "primary_key",
                        "description": "客户标识",
                        "alias": ["客户ID"],
                    },
                    {
                        "name": "customer_name",
                        "role": "dimension",
                        "description": "客户名称",
                        "alias": ["客户名", "客户"],
                    },
                ],
            },
        ]
        check_semantic_contract(
            "SELECT d.month, COUNT(DISTINCT f.customer_id) AS customer_count "
            "FROM fact_order f JOIN dim_date d ON f.date_id=d.date_id "
            "WHERE d.year=2025 GROUP BY d.month",
            "2025年每个月有多少活跃购买客户",
            tables,
        )

    def test_entity_list_rejects_unrequested_extra_projection(self):
        sql = (
            "SELECT p.brand, p.product_name, SUM(f.order_amount) AS amount "
            "FROM fact_order f JOIN dim_product p ON f.product_id=p.product_id "
            "GROUP BY p.brand, p.product_name"
        )
        with self.assertRaisesRegex(SemanticContractError, "额外投影"):
            check_semantic_contract(sql, "销量最低的五个商品", TABLES)

    def test_periodic_top_customer_share_keeps_time_grain_not_entity_rows(self):
        sql = (
            "WITH customer_monthly AS ("
            "SELECT d.month, f.customer_id, SUM(f.order_amount) AS customer_sales "
            "FROM fact_order f JOIN dim_date d ON f.date_id=d.date_id "
            "WHERE d.year=2025 GROUP BY d.month, f.customer_id), "
            "ranked AS (SELECT month, customer_id, customer_sales, "
            "ROW_NUMBER() OVER (PARTITION BY month ORDER BY customer_sales DESC) AS rn, "
            "SUM(customer_sales) OVER (PARTITION BY month) AS month_total "
            "FROM customer_monthly) "
            "SELECT month, SUM(CASE WHEN rn <= 10 THEN customer_sales ELSE 0 END) / "
            "NULLIF(MAX(month_total), 0) AS top_share "
            "FROM ranked GROUP BY month ORDER BY month"
        )
        check_semantic_contract(
            sql,
            "2025年每月销售额前十客户的销售贡献率",
            self._retail_contract_tables(),
        )

    def test_periodic_top_customer_rows_still_require_display_projection(self):
        sql = (
            "SELECT d.month, f.customer_id, SUM(f.order_amount) AS customer_sales "
            "FROM fact_order f JOIN dim_date d ON f.date_id=d.date_id "
            "GROUP BY d.month, f.customer_id"
        )
        with self.assertRaisesRegex(SemanticContractError, "展示字段"):
            check_semantic_contract(
                sql,
                "每月前十客户及其各自销售额明细",
                self._retail_contract_tables(),
            )

    def test_entity_matching_works_when_table_has_no_chinese_description(self):
        tables = [
            {**TABLES[0]},
            {**TABLES[1], "description": ""},
        ]
        with self.assertRaisesRegex(SemanticContractError, "额外投影"):
            check_semantic_contract(
                "SELECT p.brand, p.product_name FROM dim_product p",
                "没有产生订单的商品",
                tables,
            )

    def test_left_join_and_not_exists_satisfy_metadata_backed_anti_join(self):
        left_join = (
            "SELECT p.product_name FROM dim_product p "
            "LEFT JOIN fact_order f ON p.product_id=f.product_id "
            "WHERE f.product_id IS NULL"
        )
        not_exists = (
            "SELECT p.product_name FROM dim_product p WHERE NOT EXISTS ("
            "SELECT 1 FROM fact_order f WHERE f.product_id=p.product_id)"
        )
        for sql, strategy in ((left_join, "left_join_is_null"), (not_exists, "not_exists")):
            check_semantic_contract(sql, "没有产生订单的商品", TABLES)
            shape = analyze_semantic_shape(sql, "没有产生订单的商品", TABLES)
            self.assertTrue(shape["anti_join_satisfied"])
            self.assertEqual(strategy, shape["anti_join_strategy"])

    def test_not_exists_accepts_proven_derived_key_lineage(self):
        sql = (
            "SELECT COUNT(*) FROM ("
            "SELECT sh.employee_id FROM fact_shift sh "
            "JOIN dim_store st ON sh.store_id=st.store_id "
            "GROUP BY sh.employee_id) active "
            "WHERE NOT EXISTS (SELECT 1 FROM fact_shift later "
            "WHERE later.employee_id=active.employee_id)"
        )
        shape = analyze_semantic_shape(
            sql,
            "统计没有后续排班的员工数量",
            self._workforce_anti_tables(),
        )
        self.assertTrue(shape["anti_join_satisfied"])
        self.assertEqual("not_exists", shape["anti_join_strategy"])

    def test_not_exists_accepts_unique_multistage_cte_key_lineage(self):
        sql = (
            "WITH scheduled AS ("
            "SELECT DISTINCT sh.employee_id FROM fact_shift sh), "
            "qualified AS ("
            "SELECT s.employee_id AS staff_key FROM scheduled s) "
            "SELECT COUNT(*) FROM qualified q WHERE NOT EXISTS ("
            "SELECT 1 FROM fact_shift later "
            "WHERE later.employee_id=q.staff_key)"
        )
        shape = analyze_semantic_shape(
            sql,
            "统计从未再次排班的员工数量",
            self._workforce_anti_tables(),
        )
        self.assertTrue(shape["anti_join_satisfied"])
        self.assertEqual("not_exists", shape["anti_join_strategy"])

    def test_not_exists_rejects_wrong_or_ambiguous_derived_key_lineage(self):
        invalid_sql = (
            (
                "SELECT COUNT(*) FROM ("
                "SELECT sh.employee_id FROM fact_shift sh "
                "GROUP BY sh.employee_id) active "
                "WHERE NOT EXISTS (SELECT 1 FROM fact_shift later "
                "WHERE later.store_id=active.employee_id)"
            ),
            (
                "SELECT COUNT(*) FROM ("
                "SELECT COALESCE(sh.employee_id, sh.store_id) AS staff_key "
                "FROM fact_shift sh) active "
                "WHERE NOT EXISTS (SELECT 1 FROM fact_shift later "
                "WHERE later.employee_id=active.staff_key)"
            ),
            (
                "SELECT COUNT(*) FROM ("
                "SELECT sh.employee_id FROM fact_shift sh "
                "GROUP BY sh.employee_id) active "
                "WHERE NOT EXISTS (SELECT 1 FROM fact_shift later "
                "WHERE later.shift_day > '2026-01-01')"
            ),
        )
        for sql in invalid_sql:
            with self.subTest(sql=sql):
                shape = analyze_semantic_shape(
                    sql,
                    "统计没有后续排班的员工数量",
                    self._workforce_anti_tables(),
                )
                self.assertFalse(shape["anti_join_satisfied"])

    def test_null_safe_not_in_satisfies_anti_join(self):
        sql = (
            "SELECT p.product_name FROM dim_product p WHERE p.product_id NOT IN ("
            "SELECT f.product_id FROM fact_order f WHERE f.product_id IS NOT NULL)"
        )
        check_semantic_contract(sql, "没有产生订单的商品", TABLES)
        self.assertEqual(
            "null_safe_not_in",
            analyze_semantic_shape(sql, "没有产生订单的商品", TABLES)["anti_join_strategy"],
        )

    def test_missing_wrong_key_and_null_unsafe_anti_join_are_rejected(self):
        invalid_sql = (
            "SELECT p.product_name FROM dim_product p",
            (
                "SELECT p.product_name FROM dim_product p "
                "LEFT JOIN fact_order f ON p.product_id=f.date_id "
                "WHERE f.date_id IS NULL"
            ),
            (
                "SELECT p.product_name FROM dim_product p "
                "WHERE p.product_id NOT IN ("
                "SELECT f.product_id FROM fact_order f)"
            ),
        )
        for sql in invalid_sql:
            with self.subTest(sql=sql), self.assertRaisesRegex(SemanticContractError, "不存在关联事实"):
                check_semantic_contract(sql, "没有产生订单的商品", TABLES)

    def test_name_alias_does_not_override_canonical_name_column(self):
        tables = [
            {
                "name": "dim_product", "description": "商品维表", "columns": [
                    {"name": "brand", "role": "dimension", "description": "品牌", "alias": ["品牌名称"]},
                    {"name": "product_name", "role": "dimension", "description": "商品名称", "alias": ["商品名称"]},
                ],
            },
        ]
        with self.assertRaisesRegex(SemanticContractError, "额外投影"):
            check_semantic_contract(
                "SELECT p.brand, p.product_name FROM dim_product p",
                "没有产生订单的商品",
                tables,
            )

    def test_follow_up_must_keep_previous_group_and_order(self):
        history = [{"sql": "SELECT p.product_name, SUM(f.order_amount) total FROM fact_order f JOIN dim_product p ON f.product_id=p.product_id GROUP BY p.product_name ORDER BY total DESC"}]
        with self.assertRaisesRegex(SemanticContractError, "分组维度"):
            check_semantic_contract(
                "SELECT SUM(order_amount) FROM fact_order", "那销量呢", TABLES, history,
            )
        check_semantic_contract(
            "SELECT p.product_name, SUM(f.order_amount) total FROM fact_order f JOIN dim_product p ON f.product_id=p.product_id GROUP BY p.product_name ORDER BY total DESC",
            "那销售额呢", TABLES, history,
        )

    def test_explicit_group_dimension_cannot_be_dropped_during_correction(self):
        tables = [
            *TABLES,
            {
                "name": "dim_region", "role": "dim", "description": "地区维度表",
                "columns": [
                    {"name": "province", "role": "dimension", "description": "省份名称", "alias": ["省份"]},
                ],
            },
        ]
        with self.assertRaisesRegex(SemanticContractError, "明确要求"):
            check_semantic_contract(
                "SELECT SUM(order_amount) FROM fact_order",
                "各省份的销售额",
                tables,
            )
        check_semantic_contract(
            "SELECT r.province, SUM(f.order_amount) FROM fact_order f "
            "JOIN dim_region r ON f.region_id=r.region_id GROUP BY r.province",
            "各省份的销售额",
            [
                *tables,
                {
                    "name": "compat", "role": "unknown", "description": "兼容同步角色",
                    "columns": [
                        {"name": "order_amount", "role": "dimension", "description": "销售额", "alias": ["销售额"]},
                    ],
                },
            ],
        )

    def test_ranked_dimension_requires_its_metadata_table(self):
        tables = [
            {
                "name": "fact_order", "role": "fact", "columns": [
                    {"name": "region_id", "role": "foreign_key", "alias": ["地区ID"]},
                    {"name": "order_amount", "role": "measure", "alias": ["销售额"]},
                ],
            },
            {
                "name": "dim_region", "role": "dim", "columns": [
                    {"name": "region_id", "role": "primary_key", "alias": ["地区ID"]},
                    {"name": "province", "role": "dimension", "alias": ["省份"]},
                ],
            },
        ]
        with self.assertRaisesRegex(SemanticContractError, "对应维表"):
            check_semantic_contract(
                "SELECT province, SUM(order_amount) FROM fact_order GROUP BY province LIMIT 3",
                "销售额最高的三个省份",
                tables,
            )

    def test_explicit_year_requires_full_period_filter_not_one_date_key(self):
        tables = self._retail_contract_tables()
        invalid = (
            "SELECT p.category, AVG(f.order_amount) FROM fact_order f "
            "JOIN dim_product p ON f.product_id=p.product_id "
            "WHERE f.date_id=20280101 GROUP BY p.category"
        )
        with self.assertRaises(SemanticContractError) as caught:
            check_semantic_contract(
                invalid,
                "2028年各商品类别平均每笔成交金额",
                tables,
            )
        self.assertEqual(
            "semantic.time.full_period_filter_required",
            caught.exception.violation.violation_code,
        )

        valid = (
            "SELECT p.category, AVG(f.order_amount) FROM fact_order f "
            "JOIN dim_product p ON f.product_id=p.product_id "
            "JOIN dim_date d ON f.date_id=d.date_id "
            "WHERE d.year=2028 GROUP BY p.category"
        )
        check_semantic_contract(
            valid,
            "2028年各商品类别平均每笔成交金额",
            tables,
        )

    def test_explicit_year_rejects_scalar_date_lookup_but_accepts_date_set(self):
        tables = self._retail_contract_tables()
        scalar = (
            "SELECT r.province, SUM(f.order_amount) FROM fact_order f "
            "JOIN dim_region r ON f.region_id=r.region_id "
            "WHERE f.date_id=(SELECT d.date_id FROM dim_date d "
            "WHERE d.year=2029 LIMIT 1) GROUP BY r.province"
        )
        with self.assertRaises(SemanticContractError) as caught:
            check_semantic_contract(
                scalar,
                "2029年按省份汇总成交金额",
                tables,
            )
        self.assertEqual(
            "semantic.time.full_period_filter_required",
            caught.exception.violation.violation_code,
        )

        date_set = scalar.replace(
            "f.date_id=(SELECT d.date_id",
            "f.date_id IN (SELECT d.date_id",
        ).replace(" LIMIT 1", "")
        check_semantic_contract(
            date_set,
            "2029年按省份汇总成交金额",
            tables,
        )

    def test_qualified_column_must_belong_to_its_metadata_owner(self):
        tables = self._retail_contract_tables()
        with self.assertRaises(SemanticContractError) as caught:
            check_semantic_contract(
                "SELECT r.region_name, COUNT(DISTINCT f.province) "
                "FROM fact_order f JOIN dim_region r "
                "ON f.region_id=r.region_id GROUP BY r.region_name",
                "按大区统计产生过订单的省份数量",
                tables,
            )
        self.assertEqual(
            "semantic.column.wrong_owner",
            caught.exception.violation.violation_code,
        )

    def test_distinct_customer_threshold_count_requires_outer_scalar_count(self):
        tables = self._retail_contract_tables()
        invalid = (
            "SELECT COUNT(DISTINCT p.category) FROM fact_order f "
            "JOIN dim_product p ON f.product_id=p.product_id "
            "GROUP BY p.category HAVING COUNT(DISTINCT f.customer_id)>=900"
        )
        with self.assertRaises(SemanticContractError) as caught:
            check_semantic_contract(
                invalid,
                "有多少个商品类别的不同购买客户数不少于900名",
                tables,
            )
        self.assertEqual(
            "semantic.output.scalar_count_required",
            caught.exception.violation.violation_code,
        )

        with self.assertRaises(SemanticContractError) as caught:
            check_semantic_contract(
                "SELECT COUNT(*) FROM fact_order",
                "有多少个商品类别的不同购买客户数不少于900名",
                tables,
            )
        self.assertEqual(
            "semantic.output.qualified_group_count_structure_required",
            caught.exception.violation.violation_code,
        )

    def test_multi_period_separate_month_outputs_require_wide_conditional_shape(self):
        tables = self._retail_contract_tables()
        long_rows = (
            "SELECT d.year,d.month,SUM(f.order_amount) FROM fact_order f "
            "JOIN dim_date d ON f.date_id=d.date_id "
            "WHERE d.year IN (2022,2025,2028) GROUP BY d.year,d.month"
        )
        with self.assertRaises(SemanticContractError) as caught:
            check_semantic_contract(
                long_rows,
                "按月份分别给出2022、2025和2028年的成交金额",
                tables,
            )
        self.assertEqual(
            "semantic.output.period_compare_wide_required",
            caught.exception.violation.violation_code,
        )

        long_rows_with_cases = (
            "SELECT d.year,d.month,"
            "SUM(CASE WHEN d.year=2022 THEN f.order_amount ELSE 0 END) y2022,"
            "SUM(CASE WHEN d.year=2025 THEN f.order_amount ELSE 0 END) y2025,"
            "SUM(CASE WHEN d.year=2028 THEN f.order_amount ELSE 0 END) y2028 "
            "FROM fact_order f JOIN dim_date d ON f.date_id=d.date_id "
            "WHERE d.year IN (2022,2025,2028) GROUP BY d.year,d.month"
        )
        with self.assertRaises(SemanticContractError):
            check_semantic_contract(
                long_rows_with_cases,
                "按月份分别给出2022、2025和2028年的成交金额",
                tables,
            )

        wide = (
            "SELECT d.month,"
            "SUM(CASE WHEN d.year=2022 THEN f.order_amount ELSE 0 END) y2022,"
            "SUM(CASE WHEN d.year=2025 THEN f.order_amount ELSE 0 END) y2025,"
            "SUM(CASE WHEN d.year=2028 THEN f.order_amount ELSE 0 END) y2028 "
            "FROM fact_order f JOIN dim_date d ON f.date_id=d.date_id "
            "WHERE d.year IN (2022,2025,2028) GROUP BY d.month"
        )
        check_semantic_contract(
            wide,
            "按月份分别给出2022、2025和2028年的成交金额",
            tables,
        )

    def test_contiguous_multi_period_compare_accepts_closed_year_range(self):
        tables = self._retail_contract_tables()
        wide = (
            "SELECT d.month,"
            "SUM(CASE WHEN d.year=2024 THEN f.order_amount ELSE 0 END) y2024,"
            "SUM(CASE WHEN d.year=2025 THEN f.order_amount ELSE 0 END) y2025,"
            "SUM(CASE WHEN d.year=2026 THEN f.order_amount ELSE 0 END) y2026 "
            "FROM fact_order f JOIN dim_date d ON f.date_id=d.date_id "
            "WHERE d.year BETWEEN 2024 AND 2026 GROUP BY d.month"
        )
        check_semantic_contract(
            wide,
            "按月份分别给出2024、2025和2026年的成交金额",
            tables,
        )

    def test_average_of_customer_totals_requires_customer_inner_grain(self):
        tables = self._retail_contract_tables()
        # Add the requested outer grouping field to the reusable retail Meta.
        customer = next(item for item in tables if item["name"] == "dim_customer")
        customer["columns"].append({
            "name": "member_level", "role": "dimension", "description": "会员等级",
        })
        invalid = (
            "SELECT c.member_level, AVG(f.order_amount) FROM fact_order f "
            "JOIN dim_customer c ON f.customer_id=c.customer_id "
            "GROUP BY c.member_level"
        )
        with self.assertRaises(SemanticContractError) as caught:
            check_semantic_contract(
                invalid,
                "按会员等级计算客户累计成交金额的平均值",
                tables,
            )
        self.assertEqual(
            "semantic.aggregate.entity_total_average_required",
            caught.exception.violation.violation_code,
        )

        valid = (
            "WITH customer_totals AS ("
            "SELECT c.member_level,f.customer_id,SUM(f.order_amount) entity_total "
            "FROM fact_order f JOIN dim_customer c ON f.customer_id=c.customer_id "
            "GROUP BY c.member_level,f.customer_id) "
            "SELECT member_level,AVG(entity_total) FROM customer_totals "
            "GROUP BY member_level"
        )
        check_semantic_contract(
            valid,
            "按会员等级计算客户累计成交金额的平均值",
            tables,
        )

        missing_outer_group = valid.replace(
            " GROUP BY member_level",
            "",
        )
        with self.assertRaises(SemanticContractError) as caught:
            check_semantic_contract(
                missing_outer_group,
                "按会员等级计算客户累计成交金额的平均值",
                tables,
            )
        self.assertEqual(
            "semantic.aggregate.entity_total_average_required",
            caught.exception.violation.violation_code,
        )


if __name__ == "__main__":
    unittest.main()
