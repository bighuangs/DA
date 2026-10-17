import unittest
from collections import UserDict
from types import SimpleNamespace

from app.agent_system.agents.nl2sql.semantic_hints import build_semantic_contract
from app.agent_system.agents.nl2sql.nodes.filter_table import stabilize_table_selection
from app.agent_system.agents.nl2sql.nodes.merge_retrieved_info import select_semantic_metadata_columns
from app.domain.query.semantic_aliases import merge_business_aliases
from tests.semantic_fixtures import semantic_metadata


_metadata = semantic_metadata


TABLES = [
    {
        "name": "fact_order", "role": "fact", "description": "订单事实表",
        "columns": [
            {"name": "product_id", "role": "foreign_key", "description": "商品外键", "alias": ["商品ID"]},
            {"name": "date_id", "role": "foreign_key", "description": "日期外键", "alias": ["日期"]},
            {"name": "order_amount", "role": "measure", "description": "销售额", "alias": ["销售额"]},
        ],
    },
    {
        "name": "dim_product", "role": "dim", "description": "商品维度表",
        "columns": [
            {"name": "product_id", "role": "primary_key", "description": "商品标识", "alias": ["商品ID"]},
            {"name": "product_name", "role": "dimension", "description": "商品名称", "alias": ["商品名称"]},
            {"name": "brand", "role": "dimension", "description": "商品品牌", "alias": ["品牌"]},
            {"name": "category", "role": "dimension", "description": "商品品类", "alias": ["品类"]},
        ],
    },
    {
        "name": "dim_date", "role": "dim", "description": "时间维度表",
        "columns": [
            {"name": "date_id", "role": "primary_key", "description": "日期标识", "alias": ["日期ID"]},
            {"name": "month", "role": "dimension", "description": "月份", "alias": ["月", "月份"]},
        ],
    },
]


class SemanticContractTests(unittest.TestCase):
    def test_entity_question_prefers_display_name_and_join(self):
        contract = build_semantic_contract("销量最低的五个商品", TABLES)
        self.assertIn("dim_product.product_name", contract)
        self.assertIn("fact_order.product_id = dim_product.product_id", contract)
        self.assertIn("不得只返回", contract)

    def test_month_series_prefers_readable_time_dimension(self):
        contract = build_semantic_contract("2025年每月订单数", TABLES)
        self.assertIn("dim_date.month", contract)
        self.assertIn("不要用日期 ID", contract)

    def test_explicit_year_adds_full_period_filter_contract(self):
        tables = _metadata()
        contract = build_semantic_contract(
            "2028年按品牌统计成交金额",
            tables,
        )

        self.assertIn("完整年度过滤", contract)
        self.assertIn("dim_date.year", contract)
        self.assertIn("不能把全年缩成某一个 date_id", contract)

    def test_nested_entity_average_and_multi_period_shape_are_explained(self):
        tables = _metadata()
        nested = build_semantic_contract(
            "按会员等级计算客户累计成交金额的平均值",
            tables,
        )
        compared = build_semantic_contract(
            "按月份分别给出2022、2025和2028年的成交金额",
            tables,
        )

        self.assertIn("先按客户粒度", nested)
        self.assertIn("再按外层分组维度计算 AVG", nested)
        self.assertIn("条件聚合", compared)
        self.assertIn("每个年份一个结果列", compared)

    def test_ordering_only_does_not_force_group_by(self):
        contract = build_semantic_contract(
            "按商品名称升序列出前五个商品",
            TABLES,
        )
        self.assertIn("dim_product.product_name", contract)
        self.assertNotIn("必须同时出现在 SELECT 与 GROUP BY", contract)
        self.assertIn("纯目录列表无需分组", contract)

    def test_ranked_attribute_does_not_force_product_display_name(self):
        contract = build_semantic_contract(
            "商品目录内数量最多的五个品牌",
            TABLES,
        )
        self.assertIn("dim_product.brand", contract)
        self.assertNotIn("dim_product.product_name", contract)

    def test_explicit_measure_bin_is_preserved_as_grouping_axis(self):
        tables = [
            {
                "name": "fact_order", "role": "fact", "columns": [
                    {
                        "name": "order_quantity", "role": "measure",
                        "description": "购买数量", "alias": ["购买件数"],
                    },
                    {"name": "order_id", "role": "primary_key", "alias": ["订单ID"]},
                ],
            },
        ]
        contract = build_semantic_contract("按购买件数统计订单笔数", tables)
        self.assertIn("fact_order.order_quantity", contract)
        self.assertIn("物理度量字段也可以", contract)

    def test_measure_follow_up_preserves_grouping(self):
        contract = build_semantic_contract(
            "那销售额呢",
            TABLES,
            [{"query": "各品类销量", "sql": "SELECT p.category, SUM(f.order_quantity) FROM fact_order f JOIN dim_product p ON f.product_id=p.product_id GROUP BY p.category ORDER BY 2 DESC"}],
        )
        self.assertIn("只替换", contract)
        self.assertIn("GROUP BY p.category", contract)

    def test_filter_selection_restores_display_time_and_join_keys(self):
        entity = stabilize_table_selection(
            "销量最低的五个商品", TABLES,
            {"fact_order": ["order_amount", "product_id"]},
        )
        entity_map = {table["name"]: {column["name"] for column in table["columns"]} for table in entity}
        self.assertIn("product_name", entity_map["dim_product"])
        self.assertIn("product_id", entity_map["dim_product"])
        self.assertIn("product_id", entity_map["fact_order"])
        self.assertNotIn("brand", entity_map["dim_product"])

        monthly = stabilize_table_selection(
            "2025年各月销售额趋势", TABLES,
            {"fact_order": ["order_amount"]},
        )
        month_map = {table["name"]: {column["name"] for column in table["columns"]} for table in monthly}
        self.assertIn("month", month_map["dim_date"])
        self.assertIn("date_id", month_map["dim_date"])
        self.assertIn("date_id", month_map["fact_order"])

    def test_filter_selection_recovers_explicit_fields_from_empty_model_result(self):
        selected = stabilize_table_selection("各品类的销售额", TABLES, {})
        mapping = {table["name"]: {column["name"] for column in table["columns"]} for table in selected}
        self.assertIn("category", mapping["dim_product"])
        self.assertIn("order_amount", mapping["fact_order"])
        self.assertIn("product_id", mapping["dim_product"])
        self.assertIn("product_id", mapping["fact_order"])

    def test_anti_join_entity_keeps_readable_name(self):
        selected = stabilize_table_selection(
            "没有产生订单的商品", TABLES,
            {"dim_product": ["product_id"], "fact_order": ["product_id"]},
        )
        mapping = {table["name"]: {column["name"] for column in table["columns"]} for table in selected}
        self.assertIn("product_name", mapping["dim_product"])

    def test_filter_selection_keeps_previous_grouping_dimension(self):
        selected = stabilize_table_selection(
            "那销售额呢", TABLES, {"fact_order": ["order_amount"]},
            [{"sql": "SELECT p.product_name, SUM(f.order_amount) FROM fact_order f JOIN dim_product p ON f.product_id=p.product_id GROUP BY p.product_name"}],
        )
        mapping = {table["name"]: {column["name"] for column in table["columns"]} for table in selected}
        self.assertIn("product_name", mapping["dim_product"])
        self.assertIn("product_id", mapping["dim_product"])

    def test_merge_semantic_recall_adds_missing_month_and_display_columns(self):
        rows = [
            (
                SimpleNamespace(name="dim_date", description="时间维度表"),
                [
                    SimpleNamespace(name="month", description="月份", alias=["月"], role="dimension"),
                    SimpleNamespace(name="year", description="年份", alias=["年"], role="dimension"),
                ],
            ),
            (
                SimpleNamespace(name="dim_product", description="商品维度表"),
                [SimpleNamespace(name="product_name", description="商品名称", alias=["商品名称"], role="dimension")],
            ),
        ]
        monthly = select_semantic_metadata_columns("2025年每月订单数", rows)
        self.assertEqual({item.name for item in monthly}, {"month", "year"})
        product = select_semantic_metadata_columns("销量最低的五个商品", rows)
        self.assertEqual({item.name for item in product}, {"product_name"})

    def test_merge_semantic_recall_adds_standalone_year_and_quantity_measure(self):
        rows = [
            (
                SimpleNamespace(name="dim_date", description="时间维度表"),
                [SimpleNamespace(name="year", description="年份", alias=["年"], role="dimension")],
            ),
            (
                SimpleNamespace(name="fact_order", description="订单事实表"),
                [SimpleNamespace(name="order_quantity", description="购买数量", alias=["销量"], role="measure")],
            ),
        ]
        yearly = select_semantic_metadata_columns("2025年的GMV是多少", rows)
        self.assertEqual({item.name for item in yearly}, {"year"})
        quantity = select_semantic_metadata_columns("各品牌共售出多少件商品", rows)
        self.assertEqual({item.name for item in quantity}, {"order_quantity"})

    def test_merge_transaction_quantity_recall_is_action_bounded(self):
        rows = [
            (
                SimpleNamespace(name="fact_order", description="订单事实表"),
                [
                    SimpleNamespace(
                        name="order_quantity",
                        description="购买数量",
                        alias=["销量"],
                        role="measure",
                    ),
                ],
            ),
        ]
        for question in (
            "东北地区累计成交了多少件商品",
            "各品牌购买累计几件商品",
            "按会员层级一共销售了几件产品",
        ):
            with self.subTest(question=question):
                selected = select_semantic_metadata_columns(question, rows)
                self.assertEqual(
                    {item.name for item in selected},
                    {"order_quantity"},
                )

        for question in (
            "商品目录中共有几件商品",
            "按品牌统计商品目录中的商品件数",
        ):
            with self.subTest(question=question):
                self.assertEqual(
                    select_semantic_metadata_columns(question, rows),
                    [],
                )

    def test_merge_semantic_recall_adds_previous_group_for_follow_up(self):
        rows = [
            (
                SimpleNamespace(name="dim_product", description="商品维度表"),
                [
                    SimpleNamespace(name="category", description="商品品类", alias=["品类"], role="dimension"),
                    SimpleNamespace(name="brand", description="商品品牌", alias=["品牌"], role="dimension"),
                ],
            ),
        ]
        selected = select_semantic_metadata_columns(
            "那销售额呢",
            rows,
            [{"sql": "SELECT p.category, SUM(f.order_quantity) FROM fact_order f JOIN dim_product p ON f.product_id=p.product_id GROUP BY p.category"}],
        )
        self.assertEqual({item.name for item in selected}, {"category"})

        wrapped = select_semantic_metadata_columns(
            "那销售额呢",
            rows,
            [UserDict({"sql": "SELECT p.category FROM dim_product p GROUP BY p.category"})],
        )
        self.assertEqual({item.name for item in wrapped}, {"category"})

    def test_merge_semantic_recall_closes_schema_owned_enum_values(self):
        rows = [
            (
                SimpleNamespace(name="dim_product", description="商品维度表"),
                [
                    SimpleNamespace(
                        name="category",
                        description="商品品类",
                        alias=["品类"],
                        examples=["休闲零食", "家用电器"],
                        role="dimension",
                    ),
                ],
            ),
            (
                SimpleNamespace(name="dim_customer", description="客户维度表"),
                [
                    SimpleNamespace(
                        name="gender",
                        description="性别",
                        alias=["性别"],
                        examples=["男", "女"],
                        role="dimension",
                    ),
                ],
            ),
        ]

        selected = select_semantic_metadata_columns(
            "排除休闲零食后看女顾客的成交额",
            rows,
        )

        self.assertEqual(
            {item.name for item in selected},
            {"category", "gender"},
        )

    def test_merge_semantic_recall_closes_an_arbitrary_unregistered_enum_value(self):
        enum_value = "".join(("户外", "装备", "耗材"))
        rows = [(
            SimpleNamespace(name="dim_catalog", description="目录维度表"),
            [
                SimpleNamespace(
                    name="catalog_segment",
                    description="目录分群",
                    alias=["分群"],
                    examples=[enum_value],
                    role="dimension",
                ),
            ],
        )]

        selected = select_semantic_metadata_columns(
            f"只看{enum_value}的成交额",
            rows,
        )

        self.assertEqual(
            {item.name for item in selected},
            {"catalog_segment"},
        )

    def test_merge_semantic_recall_uses_v2_required_column_closure(self):
        rows = []
        table_name_by_id = {}
        for table in _metadata():
            table_id = f"table:{table['name']}"
            table_name_by_id[table_id] = table["name"]
            rows.append((
                SimpleNamespace(
                    id=table_id,
                    name=table["name"],
                    role=table["role"],
                    description=table.get("description", ""),
                ),
                [
                    SimpleNamespace(
                        id=f"{table_id}:{column['name']}",
                        table_id=table_id,
                        name=column["name"],
                        type=column.get("type", ""),
                        role=column["role"],
                        description=column.get("description", ""),
                        alias=column.get("alias", []),
                        examples=column.get("examples", []),
                        is_display=column.get("is_display", False),
                    )
                    for column in table["columns"]
                ],
            ))

        selected = select_semantic_metadata_columns(
            "订单量和销售额都按会员等级汇总",
            rows,
        )
        selected_refs = {
            (table_name_by_id[item.table_id], item.name) for item in selected
        }

        self.assertTrue({
            ("fact_order", "order_id"),
            ("fact_order", "customer_id"),
            ("fact_order", "order_amount"),
            ("dim_customer", "customer_id"),
            ("dim_customer", "member_level"),
        }.issubset(selected_refs))

    def test_merge_semantic_recall_resolves_governed_business_aliases(self):
        specs = {
            "fact_order": (
                ("order_amount", "measure"),
                ("order_quantity", "measure"),
                ("order_id", "primary_key"),
            ),
            "dim_product": (
                ("product_name", "dimension"),
                ("category", "dimension"),
            ),
        }
        rows = [
            (
                SimpleNamespace(
                    id=table_name,
                    name=table_name,
                    role="fact" if table_name.startswith("fact_") else "dim",
                    description="",
                ),
                [
                    SimpleNamespace(
                        id=f"{table_name}.{column_name}",
                        table_id=table_name,
                        name=column_name,
                        type="varchar" if role == "dimension" else "bigint",
                        role=role,
                        description="",
                        alias=merge_business_aliases(table_name, column_name),
                        examples=[],
                        is_display=column_name == "product_name",
                    )
                    for column_name, role in columns
                ],
            )
            for table_name, columns in specs.items()
        ]
        cases = (
            ("成交金额", ("fact_order", "order_amount")),
            ("单笔金额", ("fact_order", "order_amount")),
            ("成交件数", ("fact_order", "order_quantity")),
            ("购买件数", ("fact_order", "order_quantity")),
            ("订单数", ("fact_order", "order_id")),
            ("订单笔数", ("fact_order", "order_id")),
            ("商品", ("dim_product", "product_name")),
            ("产品", ("dim_product", "product_name")),
        )

        for query, expected_ref in cases:
            with self.subTest(query=query):
                selected = select_semantic_metadata_columns(query, rows)
                selected_refs = {
                    (item.table_id, item.name) for item in selected
                }
                self.assertIn(expected_ref, selected_refs)

    def test_broad_entity_alias_does_not_shadow_specific_attribute(self):
        rows = [(
            SimpleNamespace(
                id="dim_product",
                name="dim_product",
                role="dim",
                description="商品维度",
            ),
            [
                SimpleNamespace(
                    id="dim_product.product_name",
                    table_id="dim_product",
                    name="product_name",
                    type="varchar",
                    role="dimension",
                    description="商品名称",
                    alias=merge_business_aliases(
                        "dim_product", "product_name", ["商品名称"]
                    ),
                    examples=[],
                    is_display=True,
                ),
                SimpleNamespace(
                    id="dim_product.category",
                    table_id="dim_product",
                    name="category",
                    type="varchar",
                    role="dimension",
                    description="商品类别",
                    alias=["商品类别", "品类"],
                    examples=[],
                    is_display=False,
                ),
            ],
        )]

        selected = select_semantic_metadata_columns("商品类别", rows)
        selected_names = {item.name for item in selected}

        self.assertIn("category", selected_names)
        self.assertNotIn("product_name", selected_names)

if __name__ == "__main__":
    unittest.main()
