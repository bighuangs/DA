import unittest

from app.application.services.schema_sync_service import (
    SchemaSyncService,
    _embed_documents_in_batches,
    diff_schema,
)


class SchemaSyncDiffTest(unittest.TestCase):
    def test_diff_reports_table_and_column_changes(self):
        current = [{
            "name": "fact_order",
            "role": "fact",
            "description": "订单",
            "columns": [{
                "name": "order_amount",
                "type": "decimal(10,2)",
                "role": "measure",
                "description": "金额",
            }, {
                "name": "old_flag",
                "type": "tinyint",
                "role": "dimension",
                "description": "旧字段",
            }],
        }]
        incoming = [{
            "name": "fact_order",
            "role": "fact",
            "description": "订单事实",
            "columns": [{
                "name": "order_amount",
                "type": "decimal(18,2)",
                "role": "measure",
                "description": "订单金额",
            }, {
                "name": "order_date",
                "type": "date",
                "role": "dimension",
                "description": "下单日期",
            }],
        }, {
            "name": "dim_region",
            "role": "dim",
            "description": "地区",
            "columns": [],
        }]

        result = diff_schema(incoming, current)

        self.assertEqual(result["added_tables"], ["dim_region"])
        self.assertEqual(result["removed_tables"], [])
        self.assertEqual(result["added_columns"], ["fact_order.order_date"])
        self.assertEqual(result["removed_columns"], ["fact_order.old_flag"])
        self.assertEqual(result["changed_columns"][0]["id"], "fact_order.order_amount")
        self.assertEqual(result["counts"]["changed_tables"], 1)

    def test_identical_schema_has_empty_diff(self):
        schema = [{
            "name": "dim_region",
            "role": "dim",
            "description": "地区",
            "columns": [{
                "name": "region_id",
                "type": "bigint",
                "role": "primary_key",
                "description": "主键",
            }],
        }]
        result = diff_schema(schema, schema)
        self.assertEqual(result["counts"], {
            "added_tables": 0,
            "removed_tables": 0,
            "changed_tables": 0,
            "added_columns": 0,
            "removed_columns": 0,
            "changed_columns": 0,
        })


class SchemaSyncEmbeddingTest(unittest.IsolatedAsyncioTestCase):
    async def test_merge_refreshes_physical_type_without_overwriting_curated_semantics(self):
        service = SchemaSyncService(None, None, None, None, None)
        current = [{
            "name": "fact_order",
            "role": "fact",
            "description": "人工订单语义",
            "columns": [{
                "name": "order_amount",
                "type": "float",
                "role": "measure",
                "description": "人工金额语义",
                "alias": ["GMV", "成交金额", "GMV"],
                "examples": [100],
                "sync": True,
                "is_display": False,
            }],
        }]
        incoming = [{
            "name": "fact_order",
            "role": "unknown",
            "description": "DW 物理注释",
            "columns": [{
                "id": "fact_order.order_amount",
                "name": "order_amount",
                "type": "decimal(14,2)",
                "role": "dimension",
                "description": "DW 字段注释",
                "alias": ["订单金额", "成交金额"],
            }],
        }]

        tables, columns = await service._merge_schema(incoming, current)

        self.assertEqual(tables[0].role, "fact")
        self.assertEqual(tables[0].description, "人工订单语义")
        self.assertEqual(columns[0].type, "decimal(14,2)")
        self.assertEqual(columns[0].role, "measure")
        self.assertEqual(columns[0].description, "人工金额语义")
        self.assertEqual(
            columns[0].alias,
            ["GMV", "成交金额", "订单金额", "单笔金额"],
        )
        self.assertEqual(columns[0].examples, [100])
        self.assertTrue(columns[0].sync)

    async def test_embedding_requests_are_batched(self):
        class FakeEmbedding:
            def __init__(self):
                self.calls = []

            async def aembed_documents(self, texts):
                self.calls.append(texts)
                return [[float(index)] for index, _ in enumerate(texts)]

        client = FakeEmbedding()
        texts = [f"field-{index}" for index in range(35)]
        vectors = await _embed_documents_in_batches(client, texts, batch_size=32)

        self.assertEqual([len(call) for call in client.calls], [32, 3])
        self.assertEqual(len(vectors), len(texts))


if __name__ == "__main__":
    unittest.main()
