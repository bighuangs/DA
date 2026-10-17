import unittest

from app.agent_system.agents.nl2sql.nodes.filter_metric import stabilize_metric_selection


METRICS = [
    {
        "name": "GMV", "alias": ["成交总额"],
        "formula_rule": {"type": "aggregate", "function": "SUM", "column": "fact_order.order_amount"},
    },
    {
        "name": "AOV", "alias": ["平均订单金额"],
        "formula_rule": {"type": "aggregate", "function": "AVG", "column": "fact_order.order_amount"},
    },
]

POPULATION_METRICS = [
    {
        "name": "客户数",
        "alias": ["客户数量"],
        "formula_rule": {
            "type": "distinct_count",
            "function": "COUNT",
            "column": "fact_order.customer_id",
            "distinct": True,
            "population_scope": "filtered_fact_rows",
        },
    },
    {
        "name": "无事件客户",
        "alias": ["零事件客户"],
        "formula_rule": {
            "type": "anti_join",
            "universe_column": "dim_customer.customer_id",
            "event_column": "fact_order.customer_id",
        },
    },
]

AGGREGATE_METRICS = [
    {
        "name": "GMV",
        "alias": ["销售额"],
        "formula_rule": {
            "type": "aggregate",
            "function": "SUM",
            "column": "fact_order.order_amount",
        },
    },
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
        "alias": ["销售件数"],
        "formula_rule": {
            "type": "aggregate",
            "function": "SUM",
            "column": "fact_order.order_quantity",
        },
    },
    {
        "name": "订单量",
        "alias": ["订单数"],
        "formula_rule": {
            "type": "aggregate",
            "function": "COUNT",
            "column": "fact_order.order_id",
        },
    },
]

ANTI_JOIN_IR = {
    "semantic_family": "anti_join",
    "intent": "anti_join",
    "status": "partial",
    "compiler": {"supported": False},
    "required_tables": ["dim_customer", "fact_order"],
}


class MetricSelectionTests(unittest.TestCase):
    def test_sales_amount_prefers_sum_even_when_model_selected_average(self):
        selected = stabilize_metric_selection("那销售额呢", METRICS, ["AOV"])
        self.assertEqual([item["name"] for item in selected], ["GMV"])

    def test_average_order_amount_prefers_average(self):
        selected = stabilize_metric_selection("平均订单金额", METRICS, ["GMV"])
        self.assertEqual([item["name"] for item in selected], ["AOV"])

    def test_average_quantity_does_not_force_amount_metric(self):
        selected = stabilize_metric_selection("每种商品的平均购买件数", METRICS, ["AOV"])
        self.assertEqual(selected, [])

    def test_sales_quantity_rejects_model_selected_amount_metric(self):
        selected = stabilize_metric_selection("销量最低的五个商品", METRICS, ["GMV"])
        self.assertEqual(selected, [])

    def test_quantity_and_amount_can_request_amount_metric_together(self):
        selected = stabilize_metric_selection("对比各商品销量和销售额", METRICS, ["GMV"])
        self.assertEqual([item["name"] for item in selected], ["GMV"])

    def test_group_average_sales_does_not_infer_aov(self):
        selected = stabilize_metric_selection(
            "销售额高于省份平均销售额的省份", METRICS, []
        )
        self.assertEqual(selected, [])

    def test_grouped_quantity_average_does_not_bind_raw_avg_metric(self):
        average_quantity = {
            "name": "平均购买件数",
            "alias": ["平均件数"],
            "formula_rule": {
                "type": "aggregate",
                "function": "AVG",
                "column": "fact_order.order_quantity",
            },
        }
        selected = stabilize_metric_selection(
            "各品类中售出件数高于该品类品牌平均件数的品牌",
            [average_quantity],
            ["平均购买件数"],
        )
        self.assertEqual(selected, [])

    def test_generic_amount_does_not_infer_ranking_metric(self):
        ranking = {
            "name": "大客户",
            "alias": ["重点客户"],
            "formula_rule": {
                "type": "ranking",
                "function": "SUM",
                "column": "fact_order.order_amount",
            },
        }
        selected = stabilize_metric_selection(
            "累计达到销售额八成需要多少位客户", [ranking], []
        )
        self.assertEqual(selected, [])

    def test_family_specific_metric_shadows_generic_fact_population(self):
        selected = stabilize_metric_selection(
            "按等级汇总没有任何事件的客户数量",
            POPULATION_METRICS,
            [],
            ANTI_JOIN_IR,
        )
        self.assertEqual([item["name"] for item in selected], ["无事件客户"])

    def test_llm_generic_selection_cannot_override_population_family(self):
        selected = stabilize_metric_selection(
            "没有任何关联事件的客户数量",
            POPULATION_METRICS,
            ["客户数"],
            ANTI_JOIN_IR,
        )
        self.assertEqual([item["name"] for item in selected], ["无事件客户"])

    def test_plain_fact_population_keeps_generic_metric(self):
        selected = stabilize_metric_selection(
            "按等级统计客户数量",
            POPULATION_METRICS,
            [],
            {"semantic_family": "simple_aggregate"},
        )
        self.assertEqual([item["name"] for item in selected], ["客户数"])

    def test_explicit_multi_population_keeps_both_authorities(self):
        semantic_ir = {
            **ANTI_JOIN_IR,
            "semantic_slots": {
                "requested_population_scopes": [
                    "dimension_universe_minus_events",
                    "filtered_fact_rows",
                ],
            },
        }
        selected = stabilize_metric_selection(
            "无事件客户和客户数",
            POPULATION_METRICS,
            ["无事件客户", "客户数"],
            semantic_ir,
        )
        self.assertEqual(
            [item["name"] for item in selected],
            ["无事件客户", "客户数"],
        )

    def test_aggregate_family_does_not_promote_unselected_recalled_metrics(self):
        cases = [
            ("按地区统计销售额", ["GMV"]),
            ("按地区统计平均订单金额", ["AOV"]),
            ("按地区统计订单数", ["订单量"]),
            ("按品类统计销量", ["销量"]),
            ("对比销售额和销量", ["GMV", "销量"]),
        ]
        semantic_ir = {
            "semantic_family": "aggregate",
            "status": "complete",
            "compiler": {"supported": True},
            "required_tables": ["fact_order"],
        }
        for query, expected in cases:
            with self.subTest(query=query):
                selected = stabilize_metric_selection(
                    query,
                    AGGREGATE_METRICS,
                    [],
                    semantic_ir,
                )
                self.assertEqual(
                    [item["name"] for item in selected],
                    expected,
                )

    def test_wrong_entity_structural_metric_is_not_promoted(self):
        selected = stabilize_metric_selection(
            "列出没有关联事件的商品",
            [POPULATION_METRICS[1]],
            [],
            {
                **ANTI_JOIN_IR,
                "required_tables": ["dim_product", "fact_order"],
            },
        )
        self.assertEqual(selected, [])


if __name__ == "__main__":
    unittest.main()
