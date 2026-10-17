import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from app.agent_system.agents.nl2sql.follow_up import (
    FollowUpContractError,
    resolve_follow_up,
    semantic_state_from_sql,
    validate_follow_up_sql,
)
from app.domain.query.follow_up_semantics import (
    semantic_state_from_sql as domain_semantic_state_from_sql,
)
from app.domain.query.result_references import project_single_result_references
from app.application.services.query_service import _latest_result_references
from evals.legacy.multi_turn_eval import evaluate_cases, load_cases
from app.agent_system.agents.nl2sql.nodes.rewrite_follow_up import rewrite_follow_up
from app.agent_system.agents.nl2sql.semantic_ir import build_semantic_ir, compile_semantic_ir
from app.infrastructure.persistence.mysql.records.chat_history_mysql import ChatHistoryMySQL
from app.infrastructure.persistence.mysql.records.query_trace_mysql import QueryTraceMySQL
from app.infrastructure.persistence.mysql.repositories.chat_history_repository import ChatHistoryRepository


PREVIOUS_QUERY = "各品类的销量总和"
PREVIOUS_SQL = (
    "SELECT p.category, SUM(f.order_quantity) AS total_sales FROM fact_order f "
    "JOIN dim_product p ON f.product_id=p.product_id GROUP BY p.category ORDER BY total_sales DESC"
)


def history(sql=PREVIOUS_SQL, query=PREVIOUS_QUERY):
    return [{
        "query": query, "standalone_question": query, "sql": sql,
        "semantic_state": semantic_state_from_sql(query, sql),
    }]


class FollowUpRewriteTests(unittest.TestCase):
    def test_agent_path_reexports_domain_semantic_state_extractor(self):
        self.assertIs(semantic_state_from_sql, domain_semantic_state_from_sql)

    def test_eval_set_has_twenty_passing_combinations_without_llm(self):
        cases = load_cases()
        self.assertGreaterEqual(len(cases), 20)
        report = evaluate_cases(cases)
        self.assertEqual(report["passed"], report["cases"], report["results"])
        self.assertEqual(report["deterministic_llm_calls"], 0)
        self.assertLessEqual(report["p95_ms"], 3000)

    def test_measure_replacement_closes_to_standalone_question(self):
        output = resolve_follow_up("那销售额呢", history())
        self.assertEqual(output["action"], "replace")
        self.assertEqual(output["standalone_question"], "各品类的销售额总和")
        self.assertEqual(output["preserve"]["dimensions"], ["category"])

    def test_rewritten_eval_028_compiles_through_semantic_ir(self):
        output = resolve_follow_up("那销售额呢", history())
        tables = [
            {
                "name": "fact_order",
                "columns": [
                    {"name": "product_id", "role": "foreign_key", "description": "商品ID"},
                    {"name": "order_quantity", "role": "measure", "description": "销量", "alias": ["销售量"]},
                    {"name": "order_amount", "role": "measure", "description": "销售额", "alias": ["销售总额"]},
                ],
            },
            {
                "name": "dim_product",
                "columns": [
                    {"name": "product_id", "role": "primary_key", "description": "商品ID"},
                    {"name": "category", "role": "dimension", "description": "品类", "alias": ["商品品类"]},
                ],
            },
        ]
        ir = build_semantic_ir(output["standalone_question"], tables, [], history())
        sql = compile_semantic_ir(ir)
        self.assertTrue(ir["compiler"]["supported"], ir)
        self.assertIn("SUM(f.order_amount)", sql)
        self.assertIn("GROUP BY p.category", sql)
        self.assertNotIn("order_quantity", sql)

    def test_dimension_replacement_does_not_preserve_old_group(self):
        output = resolve_follow_up("改成按地区", history())
        self.assertEqual(output["standalone_question"], "各地区的销量总和")
        self.assertEqual(output["preserve"]["dimensions"], [])

    def test_chart_change_is_presentation_not_filter(self):
        output = resolve_follow_up("改为饼图", history())

        self.assertEqual(output["action"], "replace")
        self.assertEqual(output["standalone_question"], "各品类的销量总和，改为饼图")
        self.assertEqual(
            output["operations"],
            [{"op": "replace", "slot": "presentation", "to": "pie"}],
        )
        self.assertEqual(output["preserve"]["dimensions"], ["category"])
        self.assertIn("变更 展示", output["summary"])

        # 展示偏好不改变 SQL，原查询仍应满足跨轮继承契约。
        validate_follow_up_sql(PREVIOUS_SQL, output)

    def test_explicit_year_follow_up_replaces_previous_year(self):
        prior_sql = (
            "SELECT p.category, SUM(f.order_amount) AS total_amount FROM fact_order f "
            "JOIN dim_product p ON f.product_id=p.product_id "
            "JOIN dim_date d ON f.date_id=d.date_id WHERE d.year=2025 "
            "GROUP BY p.category ORDER BY total_amount DESC"
        )
        output = resolve_follow_up(
            "那2024年呢",
            history(sql=prior_sql, query="2025年各品类的销售额"),
        )
        self.assertEqual(output["action"], "replace")
        self.assertEqual(output["standalone_question"], "2024年各品类的销售额")
        self.assertNotIn("2025", output["standalone_question"])
        self.assertEqual(
            next(item for item in output["operations"] if item["slot"] == "time")["to"],
            ["2024年"],
        )

    def test_explicit_complete_question_with_history_is_new_topic(self):
        output = resolve_follow_up("复购率是多少", history())
        self.assertEqual(output["action"], "new_topic")
        self.assertEqual(output["standalone_question"], "复购率是多少")

    def test_orphan_and_result_set_reference_require_clarification(self):
        self.assertTrue(resolve_follow_up("那销售额呢", [])["needs_clarification"])
        unsafe = resolve_follow_up("这些客户的销售额呢", history())
        self.assertEqual(unsafe["action"], "clarify")
        self.assertTrue(unsafe["needs_clarification"])

    def test_single_region_result_is_projected_without_raw_result_rows(self):
        references = project_single_result_references(
            [{"region_name": "华北", "total_sales_amount": 1726767806.09}],
            row_count=1,
        )

        self.assertEqual(
            references,
            {
                "region": {
                    "entity": "region",
                    "column": "region_name",
                    "value": "华北",
                }
            },
        )
        self.assertNotIn("total_sales_amount", str(references))
        self.assertEqual(
            project_single_result_references(
                [{"region_name": "华北"}, {"region_name": "华东"}],
                row_count=2,
            ),
            {},
        )

    def test_query_service_uses_only_the_immediately_previous_single_result(self):
        rows = [
            SimpleNamespace(
                result=json.dumps([{"region_name": "华东"}], ensure_ascii=False),
                row_count=1,
            ),
            SimpleNamespace(
                result=json.dumps(
                    [{"region_name": "华北", "total_sales_amount": 10}],
                    ensure_ascii=False,
                ),
                row_count=1,
            ),
        ]

        references = _latest_result_references(rows)

        self.assertEqual(references["region"]["value"], "华北")
        self.assertNotIn("total_sales_amount", str(references))

    def test_singular_region_reference_binds_latest_value_and_drops_ranking_shape(self):
        prior = history(
            query="该地区各品牌销售额",
            sql=(
                "SELECT p.brand, SUM(f.order_amount) AS total_sales_amount "
                "FROM fact_order f JOIN dim_product p ON f.product_id=p.product_id "
                "JOIN dim_region r ON f.region_id=r.region_id "
                "WHERE r.region_name='华东' GROUP BY p.brand"
            ),
        ) + history(
            query="销售额第二的地区",
            sql=(
                "SELECT r.region_name, SUM(f.order_amount) AS total_sales_amount "
                "FROM fact_order f JOIN dim_region r ON f.region_id=r.region_id "
                "GROUP BY r.region_name ORDER BY total_sales_amount DESC "
                "LIMIT 1 OFFSET 1"
            ),
        )
        output = resolve_follow_up(
            "该地区各品牌销售额",
            prior,
            result_references={
                "region": {
                    "entity": "region",
                    "column": "region_name",
                    "value": "华北",
                }
            },
        )

        self.assertEqual(output["action"], "replace")
        self.assertEqual(output["standalone_question"], "华北地区各品牌销售额")
        self.assertEqual(output["preserve"]["dimensions"], [])
        self.assertFalse(output["preserve"]["order"])
        self.assertIsNone(output["preserve"]["limit"])
        self.assertNotIn("华东", str(output))

        validate_follow_up_sql(
            "SELECT p.brand, SUM(f.order_amount) FROM fact_order f "
            "JOIN dim_product p ON f.product_id=p.product_id "
            "JOIN dim_region r ON f.region_id=r.region_id "
            "WHERE r.region_name='华北' GROUP BY p.brand",
            output,
        )
        with self.assertRaises(FollowUpContractError):
            validate_follow_up_sql(
                "SELECT p.brand, SUM(f.order_amount) FROM fact_order f "
                "JOIN dim_product p ON f.product_id=p.product_id "
                "JOIN dim_region r ON f.region_id=r.region_id "
                "WHERE r.region_name='华东' GROUP BY p.brand",
                output,
            )

    def test_singular_region_reference_without_anchor_requires_clarification(self):
        output = resolve_follow_up("该地区各品牌销售额", history())

        self.assertEqual(output["action"], "clarify")
        self.assertTrue(output["needs_clarification"])
        self.assertIn("地区名称", output["clarification"])
        customer = resolve_follow_up("该客户的销售额", history())
        self.assertEqual(customer["action"], "clarify")
        self.assertIn("客户名称", customer["clarification"])

    def test_intra_query_group_references_are_not_cross_turn_results(self):
        complete_questions = (
            "各大区成交金额最高的客户占该大区成交金额的比例",
            "每个大区找出相对该大区月均金额最高的月份",
            "按订单所属大区计算复购率，复购客户指该大区内多次下单客户",
            "每个客户的订单金额占该客户全部订单金额的比例",
            "各商品销量占该商品所属类别总销量的比例",
        )

        for question in complete_questions:
            with self.subTest(question=question):
                output = resolve_follow_up(question, [])
                self.assertEqual(output["action"], "new_topic")
                self.assertFalse(output["needs_clarification"])
                self.assertEqual(output["standalone_question"], question)

    def test_local_group_reference_never_injects_previous_region_anchor(self):
        question = "各大区内每个类别的成交金额占该大区总成交金额的比例"

        output = resolve_follow_up(
            question,
            history(),
            result_references={
                "region": {
                    "entity": "region",
                    "column": "region_name",
                    "value": "华北",
                }
            },
        )

        self.assertEqual(output["action"], "new_topic")
        self.assertEqual(output["standalone_question"], question)
        self.assertNotIn("华北", str(output))

    def test_semantic_state_masks_filter_literals(self):
        state = semantic_state_from_sql(
            "只看华东的品类销量",
            PREVIOUS_SQL.replace("GROUP BY", "WHERE p.region = '华东' GROUP BY"),
        )
        self.assertNotIn("华东", str(state["filters"]))
        self.assertIn("category", state["dimensions"])
        self.assertEqual(state["authorized_tables"], ["dim_product", "fact_order"])

    def test_follow_up_contract_rejects_dropped_group(self):
        patch = resolve_follow_up("那销售额呢", history())
        validate_follow_up_sql(
            "SELECT p.category, SUM(f.order_amount) FROM fact_order f JOIN dim_product p "
            "ON f.product_id=p.product_id GROUP BY p.category ORDER BY 2 DESC",
            patch,
        )
        with self.assertRaises(FollowUpContractError):
            validate_follow_up_sql("SELECT SUM(order_amount) FROM fact_order", patch)

    def test_follow_up_contract_rejects_changed_filter_and_limit(self):
        prior_sql = (
            "SELECT p.category, SUM(f.order_quantity) AS total_sales FROM fact_order f "
            "JOIN dim_product p ON f.product_id=p.product_id "
            "WHERE f.date_id BETWEEN 20250101 AND 20251231 "
            "GROUP BY p.category ORDER BY total_sales DESC LIMIT 10"
        )
        patch = resolve_follow_up("那销售额呢", history(sql=prior_sql, query="2025年各品类销量前10"))
        with self.assertRaises(FollowUpContractError):
            validate_follow_up_sql(
                "SELECT p.category, SUM(f.order_amount) AS total_amount FROM fact_order f "
                "JOIN dim_product p ON f.product_id=p.product_id WHERE p.region_id = 1 "
                "GROUP BY p.category ORDER BY total_amount DESC LIMIT 10",
                patch,
            )
        with self.assertRaises(FollowUpContractError):
            validate_follow_up_sql(
                "SELECT p.category, SUM(f.order_amount) AS total_amount FROM fact_order f "
                "JOIN dim_product p ON f.product_id=p.product_id "
                "WHERE f.date_id BETWEEN 20250101 AND 20251231 "
                "GROUP BY p.category ORDER BY total_amount DESC LIMIT 5",
                patch,
            )

    def test_follow_up_contract_allows_appended_order_tie_breaker(self):
        patch = {
            "action": "append",
            "needs_clarification": False,
            "preserve": {
                "dimensions": ["member_level"],
                "filters": False,
                "filter_columns": [],
                "order": True,
                "order_directions": ["ASC"],
                "limit": None,
            },
        }
        validate_follow_up_sql(
            "SELECT c.member_level, c.gender, COUNT(DISTINCT f.customer_id) AS customer_count "
            "FROM fact_order f JOIN dim_customer c ON f.customer_id = c.customer_id "
            "GROUP BY c.member_level, c.gender "
            "ORDER BY c.member_level ASC, customer_count DESC",
            patch,
        )

    def test_reset_drops_previous_topic(self):
        output = resolve_follow_up("新问题：各地区销售额", history())
        self.assertEqual(output["action"], "reset")
        self.assertEqual(output["standalone_question"], "各地区销售额")

    def test_storage_models_include_semantic_and_trace_audit_columns(self):
        self.assertTrue({"standalone_question", "follow_up_action", "semantic_state"}.issubset(ChatHistoryMySQL.__table__.columns.keys()))
        self.assertTrue({"follow_up_action", "follow_up_summary"}.issubset(QueryTraceMySQL.__table__.columns.keys()))


class FollowUpRewriteNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_history_lookup_is_scoped_by_chat_owner_and_workspace(self):
        session = MagicMock()
        session.execute = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        session.execute.return_value = result

        repo = ChatHistoryRepository(session, owner_id="alice", workspace_id="workspace-a")
        await repo.get_recent_history("chat-a", 5)

        statement = session.execute.await_args.args[0]
        sql = str(statement)
        self.assertIn("chat_history.chat_id", sql)
        self.assertIn("chat_history.owner_id", sql)
        self.assertIn("chat_history.workspace_id", sql)

    async def test_node_streams_privacy_safe_audit_and_rewrites_query(self):
        events = []
        output = await rewrite_follow_up(
            {"query": "那销售额呢", "chat_history": history()},
            SimpleNamespace(stream_writer=events.append),
        )
        self.assertEqual(output["query"], "各品类的销售额总和")
        self.assertTrue(any(event.get("stage") == "闭合多轮指代" for event in events))
        audit = next(event["follow_up"] for event in events if event.get("follow_up"))
        self.assertNotIn("result", str(audit).lower())

    async def test_node_keeps_original_chart_preference_for_postprocessing(self):
        events = []
        output = await rewrite_follow_up(
            {"query": "改为饼图", "chat_history": history()},
            SimpleNamespace(stream_writer=events.append),
        )

        self.assertEqual(output["original_query"], "改为饼图")
        self.assertEqual(output["query"], "各品类的销量总和，改为饼图")
        self.assertEqual(
            output["follow_up"]["operations"],
            [{"op": "replace", "slot": "presentation", "to": "pie"}],
        )

    async def test_node_stops_orphan_follow_up_with_readable_error(self):
        events = []
        output = await rewrite_follow_up(
            {"query": "那销售额呢", "chat_history": []},
            SimpleNamespace(stream_writer=events.append),
        )
        self.assertIn("没有可继承", output["error"])
        self.assertTrue(any(event.get("error") for event in events))

    async def test_node_binds_singular_region_reference_from_state_contract(self):
        events = []
        output = await rewrite_follow_up(
            {
                "query": "该地区各品牌销售额",
                "chat_history": history(
                    query="销售额第二的地区",
                    sql=(
                        "SELECT region_name, SUM(order_amount) AS total_amount "
                        "FROM fact_order GROUP BY region_name "
                        "ORDER BY total_amount DESC LIMIT 1 OFFSET 1"
                    ),
                ),
                "result_references": {
                    "region": {
                        "entity": "region",
                        "column": "region_name",
                        "value": "华北",
                    }
                },
            },
            SimpleNamespace(stream_writer=events.append),
        )

        self.assertEqual(output["query"], "华北地区各品牌销售额")
        self.assertIsNone(output["error"])
        self.assertTrue(any(event.get("stage") == "闭合多轮指代" for event in events))


if __name__ == "__main__":
    unittest.main()
