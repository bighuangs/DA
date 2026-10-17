import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from pydantic import ValidationError

from app.adapters.inbound.http.schemas.query_schema import ExecSqlSchema, QuerySchema
from app.adapters.inbound.http.schemas.upload_schema import ConversationDataContextSchema
from app.domain.data_sources.context import (
    AssetAccessContext,
    AssetIdentity,
    AssetKind,
    DatasetAccessError,
    DatasetExpiredError,
    SourceMode,
    authorize_sql_assets,
    ensure_context_assets_current,
    redact_physical_upload_tables,
    translate_logical_upload_tables,
)
from app.domain.query.safety.source_analysis_guard import validate_source_analysis_sql
from app.domain.query.safety.sql_security import SQLSecurityError
from app.infrastructure.persistence.mysql.repositories.meta_mysql_repository import _is_internal_asset_table
from app.application.services.adhoc_dataset_service import reconciled_terminal_status
from app.application.services.schema_sync_service import _official_snapshot_tables
from app.domain.query.source_comparison import deterministic_comparison


def asset(
    table="da_upload_a",
    logical="upload_1_aaaaaaaa",
    *,
    version_id="version-a",
    owner_id="owner-a",
    workspace_id="workspace-a",
    state="ready",
    expires_at=None,
):
    return AssetIdentity(
        asset_id="asset-a",
        version_id=version_id,
        kind=AssetKind.MANAGED_UPLOAD,
        source="upload",
        catalog="dw2",
        schema="dw2",
        table=table,
        logical_name=logical,
        workspace_id=workspace_id,
        owner_id=owner_id,
        state=state,
        schema_fingerprint="sha256:schema",
        policy_version="upload-asset-policy-v1",
        expires_at=expires_at,
    )


def access(
    *assets,
    mode=SourceMode.UPLOAD_ONLY,
    intent="upload_only",
    mapping_id=None,
):
    return AssetAccessContext(
        workspace_id="workspace-a",
        owner_id="owner-a",
        source_mode=mode,
        official_tables=frozenset({"fact_sales", "dim_region"}),
        upload_assets=tuple(assets),
        comparison_intent=intent,
        mapping_confirmation_id=mapping_id,
    )


class UploadAssetAuthorizationTests(unittest.TestCase):
    def test_logical_name_translates_to_internal_table_without_catalog_escape(self):
        context = access(asset())
        translated = translate_logical_upload_tables(
            "SELECT c_001 FROM dw2.fake.upload_1_aaaaaaaa",
            context,
        )
        self.assertIn("da_upload_a", translated)
        self.assertNotIn("fake", translated)
        self.assertNotIn("upload_1_aaaaaaaa", translated)

    def test_physical_table_is_redacted_for_ui_and_history(self):
        context = access(asset())
        public_sql = redact_physical_upload_tables(
            "SELECT c_001 FROM `da_upload_a`",
            context,
        )
        self.assertIn("upload_1_aaaaaaaa", public_sql)
        self.assertNotIn("da_upload_a", public_sql)

    def test_ordinary_sql_entry_rejects_internal_upload_tables(self):
        with self.assertRaises(SQLSecurityError):
            authorize_sql_assets("SELECT c_001 FROM da_upload_a", None)

    def test_exact_asset_context_rejects_another_upload_table(self):
        with self.assertRaises(SQLSecurityError):
            authorize_sql_assets(
                "SELECT c_001 FROM da_upload_b",
                access(asset()),
            )

    def test_upload_only_context_rejects_official_dw_table(self):
        with self.assertRaises(SQLSecurityError):
            authorize_sql_assets(
                "SELECT region FROM fact_sales",
                access(asset()),
            )

    def test_compare_context_allows_only_selected_upload_and_filtered_dw(self):
        context = access(
            asset(),
            mode=SourceMode.UPLOAD_COMPARE_DW,
            intent="independent_aggregate_compare",
        )
        refs = authorize_sql_assets(
            "SELECT c_001 FROM da_upload_a UNION ALL "
            "SELECT region FROM fact_sales",
            context,
        )
        self.assertEqual(refs, {"da_upload_a", "fact_sales"})
        with self.assertRaises(SQLSecurityError):
            authorize_sql_assets(
                "SELECT c_001 FROM da_upload_b UNION ALL "
                "SELECT region FROM fact_sales",
                context,
            )
        with self.assertRaises(SQLSecurityError):
            authorize_sql_assets(
                "SELECT c_001 FROM da_upload_a UNION ALL "
                "SELECT customer_name FROM dim_customer",
                context,
            )

    def test_owner_workspace_and_ttl_are_rechecked_at_execution(self):
        with self.assertRaises(DatasetAccessError):
            ensure_context_assets_current(
                access(asset(owner_id="owner-b"))
            )
        with self.assertRaises(DatasetExpiredError):
            ensure_context_assets_current(
                access(asset(
                    expires_at=datetime.now(UTC).replace(tzinfo=None)
                    - timedelta(seconds=1)
                ))
            )
        with self.assertRaises(DatasetAccessError):
            ensure_context_assets_current(access(asset(state="archived")))


class SourceIntentGuardTests(unittest.TestCase):
    def test_upload_analysis_forbids_select_star(self):
        with self.assertRaises(SQLSecurityError):
            validate_source_analysis_sql(
                "SELECT * FROM upload_1_aaaaaaaa",
                access=access(asset()),
                mapping=None,
            )

    def test_independent_compare_forbids_upload_dw_row_join(self):
        context = access(
            asset(),
            mode=SourceMode.UPLOAD_COMPARE_DW,
            intent="independent_aggregate_compare",
        )
        with self.assertRaises(SQLSecurityError):
            validate_source_analysis_sql(
                "SELECT u.c_001, d.region "
                "FROM upload_1_aaaaaaaa u "
                "JOIN fact_sales d ON u.c_001 = d.region",
                access=context,
                mapping=None,
            )

    def test_independent_compare_requires_two_labelled_aggregate_union_all_branches(self):
        context = access(
            asset(),
            mode=SourceMode.UPLOAD_COMPARE_DW,
            intent="independent_aggregate_compare",
        )
        valid = (
            "SELECT 'upload' AS source_label, c_001 AS region, "
            "SUM(c_002) AS amount FROM upload_1_aaaaaaaa GROUP BY c_001 "
            "UNION ALL "
            "SELECT 'dw' AS source_label, region, "
            "SUM(amount) AS amount FROM fact_sales GROUP BY region"
        )
        evidence = validate_source_analysis_sql(
            valid,
            access=context,
            mapping=None,
        )
        self.assertTrue(evidence["independent_aggregate"]["union_all"])
        self.assertEqual(
            sorted(evidence["independent_aggregate"]["branch_types"]),
            ["dw", "upload"],
        )

        invalid = (
            (
                "SELECT 'upload' AS source_label, SUM(c_002) AS amount "
                "FROM upload_1_aaaaaaaa"
            ),
            valid.replace("UNION ALL", "UNION"),
            valid.replace("'upload' AS source_label", "c_001 AS source_label"),
            valid.replace("'dw' AS source_label", "'upload' AS source_label"),
        )
        for sql in invalid:
            with self.subTest(sql=sql), self.assertRaises(SQLSecurityError):
                validate_source_analysis_sql(
                    sql,
                    access=context,
                    mapping=None,
                )

    def test_join_requires_matching_confirmation_and_quality_gates(self):
        context = access(
            asset(),
            mode=SourceMode.UPLOAD_JOIN_DW,
            intent="upload_dw_row_join",
            mapping_id="mapping-a",
        )
        sql = (
            "SELECT u.c_001, d.region FROM upload_1_aaaaaaaa u "
            "JOIN dim_region d ON u.c_001 = d.region"
        )
        with self.assertRaises(SQLSecurityError):
            validate_source_analysis_sql(sql, access=context, mapping=None)
        blocked = {
            "mapping_id": "mapping-a",
            "status": "confirmed",
            "mappings": [],
            "evidence": {"auto_join_allowed": False},
        }
        with self.assertRaises(SQLSecurityError):
            validate_source_analysis_sql(sql, access=context, mapping=blocked)

    def test_confirmed_mapping_allows_only_registered_on_pair(self):
        context = access(
            asset(),
            mode=SourceMode.UPLOAD_JOIN_DW,
            intent="upload_dw_row_join",
            mapping_id="mapping-a",
        )
        mapping = {
            "mapping_id": "mapping-a",
            "status": "confirmed",
            "mappings": [{
                "left": {"table": "upload_1_aaaaaaaa", "column": "c_001"},
                "right": {"table": "dim_region", "column": "region"},
            }],
            "evidence": {"auto_join_allowed": True},
        }
        evidence = validate_source_analysis_sql(
            "SELECT u.c_001, d.region FROM upload_1_aaaaaaaa u "
            "JOIN dim_region d ON u.c_001 = d.region",
            access=context,
            mapping=mapping,
        )
        self.assertTrue(evidence["mapping_gate_passed"])
        with self.assertRaises(SQLSecurityError):
            validate_source_analysis_sql(
                "SELECT u.c_001, d.region FROM upload_1_aaaaaaaa u "
                "JOIN dim_region d ON u.c_002 = d.region",
                access=context,
                mapping=mapping,
            )

    def test_union_intent_requires_explicit_union(self):
        context = access(
            asset(),
            mode=SourceMode.UPLOAD_JOIN_DW,
            intent="union",
            mapping_id="mapping-a",
        )
        mapping = {
            "mapping_id": "mapping-a",
            "status": "confirmed",
            "mappings": [],
            "evidence": {"auto_join_allowed": True},
        }
        with self.assertRaises(SQLSecurityError):
            validate_source_analysis_sql(
                "SELECT c_001 FROM upload_1_aaaaaaaa",
                access=context,
                mapping=mapping,
            )


class ProjectionAndComparisonTests(unittest.TestCase):
    def test_cleanup_reconciliation_restores_the_original_terminal_meaning(self):
        now = datetime.now(UTC).replace(tzinfo=None)
        self.assertEqual(
            reconciled_terminal_status(
                SimpleNamespace(error="materialization failed", expires_at=now),
                now=now,
            ),
            "failed",
        )
        self.assertEqual(
            reconciled_terminal_status(
                SimpleNamespace(error=None, expires_at=now - timedelta(seconds=1)),
                now=now,
            ),
            "expired",
        )
        self.assertEqual(
            reconciled_terminal_status(
                SimpleNamespace(error=None, expires_at=now + timedelta(hours=1)),
                now=now,
            ),
            "deleted",
        )

    def test_official_schema_and_meta_filters_exclude_internal_assets(self):
        snapshot = _official_snapshot_tables([
            {"name": "fact_sales"},
            {"name": "adhoc_old"},
            {"name": "da_upload_new"},
            {"name": "eval_upload_fixture"},
            {"name": "__da_system"},
        ])
        self.assertEqual(snapshot, [{"name": "fact_sales"}])
        self.assertTrue(_is_internal_asset_table("da_upload_new"))
        self.assertFalse(_is_internal_asset_table("fact_sales"))

    def test_deterministic_comparator_computes_delta_without_row_join(self):
        result = deterministic_comparison([
            {"source_label": "dw", "amount": 100},
            {"source_label": "upload", "amount": 125},
        ])
        self.assertEqual(result["status"], "compared")
        self.assertEqual(result["mode"], "scalar")
        self.assertEqual(result["delta"], 25)
        self.assertEqual(result["delta_pct"], 25)
        self.assertFalse(result["row_level_join_used"])

    def test_deterministic_comparator_aligns_grouped_dimensions_and_missing_keys(self):
        result = deterministic_comparison([
            {"source_label": "dw", "region": "华东", "total_amount": 100},
            {"source_label": "dw", "region": "华南", "total_amount": 80},
            {"source_label": "upload", "region": "华东", "total_amount": 125},
            {"source_label": "upload", "region": "华北", "total_amount": 40},
        ])
        self.assertEqual(result["status"], "compared")
        self.assertEqual(result["mode"], "grouped")
        self.assertEqual(result["dimensions"], ["region"])
        self.assertEqual(result["compared_key_count"], 3)
        self.assertEqual(result["matched_key_count"], 1)
        self.assertEqual(result["baseline_missing_count"], 1)
        self.assertEqual(result["comparison_missing_count"], 1)
        east = next(
            item
            for item in result["comparisons"]
            if item["dimensions"]["region"] == "华东"
        )
        self.assertEqual(east["delta"], 25)
        self.assertEqual(east["delta_pct"], 25)

    def test_deterministic_comparator_uses_official_dw_as_chinese_labelled_baseline(self):
        result = deterministic_comparison([
            {"source_label": "上传文件", "region": "华东", "amount": 125},
            {"source_label": "正式DW", "region": "华东", "amount": 100},
        ])
        self.assertEqual(result["baseline_source"], "正式DW")
        self.assertEqual(result["comparison_source"], "上传文件")
        self.assertEqual(result["delta"], 25)

    def test_comparator_requires_exactly_two_labelled_sources(self):
        self.assertEqual(
            deterministic_comparison([{"amount": 1}])["reason"],
            "source_label_missing",
        )
        self.assertEqual(
            deterministic_comparison([
                {"source_label": "a", "amount": 1},
                {"source_label": "b", "amount": 2},
                {"source_label": "c", "amount": 3},
            ])["reason"],
            "requires_exactly_two_sources",
        )

    def test_api_contract_limits_context_to_two_datasets_and_known_modes(self):
        with self.assertRaises(ValidationError):
            QuerySchema(
                query="测试",
                dataset_ids=["a", "b", "c"],
                source_mode="multi_upload",
            )
        with self.assertRaises(ValidationError):
            ConversationDataContextSchema(
                source_mode="unsafe_mode",
                dataset_ids=[],
            )
        payload = ExecSqlSchema(
            sql="SELECT c_001 FROM upload_1_aaaaaaaa",
            dataset_ids=["dataset-a"],
            source_mode="upload_only",
            comparison_intent="upload_only",
        )
        self.assertEqual(payload.dataset_ids, ["dataset-a"])


if __name__ == "__main__":
    unittest.main()
