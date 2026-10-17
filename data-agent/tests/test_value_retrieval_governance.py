import unittest
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.config.app_config import app_config
from app.infrastructure.search.elasticsearch.value_retrieval import (
    build_value_search_query,
    normalize_value,
    value_column_decision,
    value_document_id,
    value_index_alias,
    value_index_version,
    value_physical_index_prefix,
)
from app.infrastructure.search.elasticsearch.repositories.value_es_repository import ValueESRepository
from app.infrastructure.search.elasticsearch.value_index_adapter import (
    ElasticsearchValueIndexAdapter,
)
from app.application.ports.metadata_indexing import MetadataIndexSettings
from app.application.services.value_index_service import ValueIndexService


class ValueRetrievalContractTests(unittest.TestCase):
    def test_search_query_applies_tenant_version_and_access_filters_in_es(self):
        query = build_value_search_query(
            " 华东 ",
            query_mode="exact",
            workspace_id="workspace-a",
            index_version="sha256:v1",
            allowed_table_ids=["dim_region"],
            allowed_column_ids=["dim_region.region_name"],
        )
        filters = query["bool"]["filter"]
        self.assertIn({"term": {"workspace_id": "workspace-a"}}, filters)
        self.assertIn({"term": {"index_version": "sha256:v1"}}, filters)
        self.assertIn({"term": {"is_active": True}}, filters)
        self.assertIn({"terms": {"table_id": ["dim_region"]}}, filters)
        self.assertIn(
            {"terms": {"column_id": ["dim_region.region_name"]}},
            filters,
        )
        self.assertEqual(
            query["bool"]["must"],
            [{"term": {"normalized_value": normalize_value("华东")}}],
        )

    def test_document_id_is_stable_and_does_not_contain_literal(self):
        first = value_document_id(
            "workspace-a", "dim_region.region_name", normalize_value("华东")
        )
        second = value_document_id(
            "workspace-a", "dim_region.region_name", normalize_value(" 华东 ")
        )
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("value:"))
        self.assertNotIn("华东", first)

    def test_sensitive_and_high_cardinality_fields_are_default_denied(self):
        sensitive = value_column_decision({
            "id": "dim_customer.customer_name",
            "name": "customer_name",
            "table_id": "dim_customer",
            "sync": True,
        })
        high_cardinality = value_column_decision(
            {
                "id": "fact_order.external_code",
                "name": "external_code",
                "table_id": "fact_order",
                "sync": True,
            },
            distinct_count=3,
            values=["a", "b", "c"],
        )
        self.assertFalse(sensitive.allowed)
        with patch.object(app_config.es, "max_distinct_values_per_column", 2):
            high_cardinality = value_column_decision(
                {
                    "id": "fact_order.external_code",
                    "name": "external_code",
                    "table_id": "fact_order",
                    "sync": True,
                },
                distinct_count=3,
                values=["a", "b", "c"],
            )
        self.assertFalse(high_cardinality.allowed)
        self.assertEqual(high_cardinality.reason, "high_cardinality")


class ValueIndexPublicationTests(unittest.IsolatedAsyncioTestCase):
    async def test_rebuild_only_publishes_approved_normalized_values(self):
        dw_repo = SimpleNamespace(get_column_values=AsyncMock())

        async def values_for(_table, column, _limit):
            if column == "region_name":
                return ["华东", " 华东 ", "华南"]
            return ["13800138000"]

        dw_repo.get_column_values.side_effect = values_for
        value_repo = SimpleNamespace(publish_workspace=AsyncMock(
            return_value={"ready": True, "alias": "value-workspace-a"}
        ))
        columns = [
            SimpleNamespace(
                id="dim_region.region_name",
                name="region_name",
                table_id="dim_region",
                type="varchar",
                role="dimension",
                sync=True,
                description="地区名称",
                alias=["地区"],
            ),
            SimpleNamespace(
                id="dim_customer.customer_name",
                name="customer_name",
                table_id="dim_customer",
                type="varchar",
                role="dimension",
                sync=True,
                description="客户姓名",
                alias=["客户"],
            ),
        ]
        publisher = ElasticsearchValueIndexAdapter(
            dw_repo=dw_repo,
            value_repo=value_repo,
            settings=MetadataIndexSettings(
                max_distinct_values_per_column=(
                    app_config.es.max_distinct_values_per_column
                ),
            ),
        )
        result = await ValueIndexService(
            publisher,
            workspace_id="workspace-a",
        ).rebuild(columns)

        self.assertTrue(result["ready"])
        self.assertEqual(result["admitted_column_count"], 1)
        self.assertEqual(result["blocked_column_count"], 1)
        self.assertEqual(dw_repo.get_column_values.await_count, 1)
        documents = value_repo.publish_workspace.await_args.kwargs["values"]
        expected_version = value_index_version(
            workspace_id="workspace-a",
            columns=columns,
            content_fingerprints=[{
                "column_id": "dim_region.region_name",
                "document_count": 2,
                "content_hash": "sha256:" + hashlib.sha256(
                    "华东\x1e华南".encode("utf-8")
                ).hexdigest(),
            }],
        )
        self.assertEqual([item["value"] for item in documents], ["华东", "华南"])
        self.assertEqual(
            value_repo.publish_workspace.await_args.kwargs["index_version"],
            expected_version,
        )
        self.assertTrue(all(item["index_version"] == expected_version for item in documents))
        self.assertTrue(all(item["workspace_id"] == "workspace-a" for item in documents))
        self.assertTrue(all("华东" not in item["id"] for item in documents))
        self.assertTrue(all(item["is_active"] for item in documents))


class ValueRepositorySafetyTests(unittest.IsolatedAsyncioTestCase):
    def _repository(self):
        indices = SimpleNamespace(
            create=AsyncMock(),
            refresh=AsyncMock(),
            update_aliases=AsyncMock(),
            delete=AsyncMock(),
            get_mapping=AsyncMock(),
        )
        client = SimpleNamespace(
            indices=indices,
            bulk=AsyncMock(return_value={"errors": False}),
            count=AsyncMock(return_value={"count": 1}),
            search=AsyncMock(),
        )
        return ValueESRepository(client), client

    async def test_history_prune_failure_does_not_delete_activated_index(self):
        repo, client = self._repository()
        repo._active_indices = AsyncMock(return_value=["old-index"])
        repo._prune_history = AsyncMock(side_effect=RuntimeError("delete failed"))
        document = {
            "id": "value:hash",
            "workspace_id": "workspace-a",
            "value": "华东",
            "normalized_value": "华东",
            "type": "varchar",
            "column_id": "dim_region.region_name",
            "column_name": "region_name",
            "table_id": "dim_region",
            "table_name": "dim_region",
            "index_version": "sha256:v1",
            "indexed_at": "2026-07-24T00:00:00+00:00",
            "source_updated_at": None,
            "is_active": True,
            "responsibility": "es_field_value",
        }

        with patch(
            "app.infrastructure.search.elasticsearch.repositories.value_es_repository.value_physical_index_name",
            return_value="new-active-index",
        ):
            result = await repo.publish_workspace(
                workspace_id="workspace-a",
                index_version="sha256:v1",
                values=[document],
            )

        self.assertEqual(result["history_prune_status"], "degraded")
        client.indices.update_aliases.assert_awaited_once()
        client.indices.delete.assert_not_awaited()

    async def test_failed_rollback_verification_compensates_original_alias(self):
        repo, client = self._repository()
        prefix = value_physical_index_prefix("workspace-a")
        target = f"{prefix}target"
        repo.history = AsyncMock(return_value=[{
            "physical_index": target,
            "index_version": "sha256:target",
        }])
        repo._active_indices = AsyncMock(return_value=["old-active-index"])
        repo.active_manifest = AsyncMock(return_value={
            "ready": False,
            "physical_index": target,
            "index_version": "sha256:target",
        })

        with self.assertRaisesRegex(RuntimeError, "回退后校验失败"):
            await repo.activate_retained(
                workspace_id="workspace-a",
                physical_index=target,
            )

        self.assertEqual(client.indices.update_aliases.await_count, 2)
        restore_actions = client.indices.update_aliases.await_args_list[1].kwargs[
            "actions"
        ]
        self.assertIn(
            {"remove": {"index": target, "alias": value_index_alias("workspace-a")}},
            restore_actions,
        )
        self.assertTrue(any(
            action.get("add", {}).get("index") == "old-active-index"
            for action in restore_actions
        ))

    async def test_active_manifest_fails_closed_when_alias_has_multiple_targets(self):
        repo, client = self._repository()
        repo._active_indices = AsyncMock(return_value=["index-a", "index-b"])
        client.indices.get_mapping.return_value = {
            "index-b": {
                "mappings": {
                    "_meta": {
                        "contract_version": app_config.es.value_contract_version,
                        "mapping_version": app_config.es.value_mapping_version,
                        "responsibility": "es_field_value",
                        "index_version": "sha256:v1",
                    }
                }
            }
        }
        client.search.return_value = {"hits": {"total": {"value": 0}, "hits": []}}

        status = await repo.active_manifest("workspace-a")

        self.assertFalse(status["ready"])
        self.assertIn("active_alias_target_count", status["drift_reasons"])

    async def test_search_exception_does_not_echo_raw_literal(self):
        repo, client = self._repository()
        client.search.side_effect = RuntimeError("request body leaked 华东")

        with self.assertRaises(RuntimeError) as raised:
            await repo.search_with_evidence(
                "华东",
                workspace_id="workspace-a",
                query_mode="exact",
                active_manifest={
                    "ready": True,
                    "alias": "value-workspace-a",
                    "index_version": "sha256:v1",
                },
                allowed_table_ids=["dim_region"],
                allowed_column_ids=["dim_region.region_name"],
            )

        self.assertNotIn("华东", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
