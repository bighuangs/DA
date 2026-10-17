import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from evals.legacy.dataset_suite import (
    SUITE_PATHS,
    SUITE_SPECS,
    _dw_meta_physical_coverage,
    _meta_content_coverage,
    _meta_schema_coverage,
    _normalized_sql,
    _refresh_baseline_enum_examples,
    baseline_manifest,
    baseline_training_examples,
    dataset_suite_report,
    install_baseline_fixture,
    load_suite_cases,
    resolve_order_count,
    sql_ast_policy_report,
    suite_manifest,
)
from app.config.meta_config import meta_config


class EvaluationDatasetSuiteTests(unittest.TestCase):
    @staticmethod
    def _canonical_meta_rows():
        return [
            (
                SimpleNamespace(
                    name=table.name,
                    role=table.role,
                    description=table.description,
                ),
                [
                    SimpleNamespace(
                        name=column.name,
                        role=column.role,
                        description=column.description,
                        alias=list(column.alias),
                        sync=bool(column.sync),
                        is_display=bool(column.is_display),
                    )
                    for column in table.columns
                ],
            )
            for table in meta_config.tables
        ]

    @staticmethod
    def _canonical_metric_rows():
        return [
            SimpleNamespace(
                name=metric.name,
                description=metric.description,
                alias=list(metric.alias),
                relevant_columns=list(metric.relevant_columns),
                formula_rule=metric.formula_rule,
            )
            for metric in meta_config.metrics
        ]

    @staticmethod
    def _canonical_training_rows():
        return [
            SimpleNamespace(
                question=example["question"],
                sql=example["sql"],
            )
            for example in baseline_training_examples()
        ]

    def test_meta_schema_coverage_requires_all_canonical_tables_and_columns(self):
        report = _meta_schema_coverage(self._canonical_meta_rows())

        self.assertTrue(report["passed"], report)
        self.assertEqual(report["observed_table_count"], 5)
        self.assertEqual(report["observed_column_count"], 24)

    def test_meta_schema_coverage_reports_missing_and_drifted_columns(self):
        rows = self._canonical_meta_rows()
        rows[0][1].pop()
        rows[1][1][0].alias = ["错误别名"]

        report = _meta_schema_coverage(rows)

        self.assertFalse(report["passed"])
        self.assertEqual(len(report["missing_columns"]), 1)
        self.assertIn(
            rows[1][1][0].name,
            report["attribute_mismatches"][0]["id"],
        )
        self.assertIn("alias", report["attribute_mismatches"][0]["fields"])

    def test_dw_meta_physical_coverage_isolates_type_drift(self):
        snapshot = [{
            "name": "fact_order",
            "columns": [{"name": "order_amount", "type": "decimal(14,2)"}],
        }]
        rows = [(
            SimpleNamespace(name="fact_order"),
            [SimpleNamespace(name="order_amount", type="decimal(14,2)")],
        )]

        self.assertTrue(_dw_meta_physical_coverage(snapshot, rows)["passed"])
        rows[0][1][0].type = "float"
        report = _dw_meta_physical_coverage(snapshot, rows)

        self.assertFalse(report["passed"])
        self.assertEqual(report["type_mismatches"], [{
            "id": "fact_order.order_amount",
            "dw_type": "decimal(14,2)",
            "meta_type": "float",
        }])

    def test_meta_content_coverage_accepts_exact_canonical_content(self):
        report = _meta_content_coverage(
            self._canonical_metric_rows(),
            self._canonical_training_rows(),
        )

        self.assertTrue(report["passed"], report)
        self.assertEqual(
            report["metrics"]["expected_count"],
            len(meta_config.metrics),
        )
        self.assertEqual(report["training_examples"]["expected_count"], 60)

    def test_meta_content_coverage_reports_metric_and_training_drift(self):
        metrics = self._canonical_metric_rows()
        trainings = self._canonical_training_rows()
        missing_metric_name = metrics[-1].name
        metrics.pop()
        metrics[0].description = "漂移后的描述"
        metrics[0].alias = ["漂移后的别名"]
        metrics[0].relevant_columns = ["fact_order.order_quantity"]
        metrics[0].formula_rule = {"type": "drifted"}
        metrics.append(SimpleNamespace(
            name="额外指标",
            description="额外指标",
            alias=[],
            relevant_columns=[],
            formula_rule=None,
        ))
        mismatched_question = trainings[0].question
        trainings[0].sql = "SELECT 1"
        missing_question = trainings[-1].question
        trainings.pop()
        trainings.append(SimpleNamespace(
            question="额外训练问题",
            sql="SELECT 2",
        ))

        report = _meta_content_coverage(metrics, trainings)

        self.assertFalse(report["passed"])
        self.assertEqual(report["metrics"]["missing_names"], [missing_metric_name])
        self.assertEqual(report["metrics"]["extra_names"], ["额外指标"])
        self.assertEqual(
            report["metrics"]["attribute_mismatches"][0]["fields"],
            ["description", "alias", "relevant_columns", "formula_rule"],
        )
        self.assertEqual(
            report["training_examples"]["missing_questions"],
            [missing_question],
        )
        self.assertEqual(
            report["training_examples"]["extra_questions"],
            ["额外训练问题"],
        )
        self.assertEqual(
            report["training_examples"]["sql_mismatches"][0]["question"],
            mismatched_question,
        )

    def test_default_and_explicit_easy_suite_share_one_hundred_governed_cases(self):
        default_cases = load_suite_cases()
        easy = load_suite_cases("easy_sql_v1")

        self.assertEqual(len(easy), 100)
        self.assertEqual(
            [item["id"] for item in default_cases],
            [item["id"] for item in easy],
        )
        self.assertEqual(easy[0]["id"], "easy_001")
        self.assertEqual(easy[-1]["id"], "easy_100")
        manifest = suite_manifest("easy_sql_v1")
        self.assertEqual(
            [item["id"] for item in easy],
            list(manifest["case_ids"]),
        )
        self.assertNotIn("case_ids", SUITE_SPECS["easy_sql_v1"])

    def test_three_difficulty_contracts_are_frozen_independent_and_structurally_ready(self):
        expected_difficulties = {
            "easy_sql_v1": {"easy": 100, "medium": 0, "hard": 0},
            "medium_sql_v1": {"easy": 0, "medium": 100, "hard": 0},
            "hard_sql_v1": {"easy": 0, "medium": 0, "hard": 50},
        }
        expected_tiers = {
            "easy_sql_v1": {"smoke": 10, "regression": 90},
            "medium_sql_v1": {"smoke": 10, "regression": 90},
            "hard_sql_v1": {"smoke": 10, "regression": 40},
        }
        for suite, difficulties in expected_difficulties.items():
            with self.subTest(suite=suite):
                cases = load_suite_cases(suite)
                report = dataset_suite_report(suite)
                manifest = suite_manifest(suite)
                self.assertEqual(len(cases), sum(difficulties.values()))
                self.assertEqual(
                    len(manifest["case_ids"]), sum(difficulties.values()),
                )
                self.assertEqual(report["tiers"], expected_tiers[suite])
                self.assertEqual(report["difficulties"], difficulties)
                self.assertTrue(report["structural_ready"], report)
                self.assertTrue(report["structural_promotion_ready"])
                self.assertFalse(report["runtime_validation"]["ready"])
                self.assertFalse(report["promotion_ready"])
                self.assertTrue(report["freeze"]["passed"])
                self.assertTrue(report["independence"]["passed"])
                self.assertTrue(report["ast_policy"]["passed"])

    def test_all_formal_suites_require_physical_model_invocation(self):
        for suite in ("easy_sql_v1", "medium_sql_v1", "hard_sql_v1"):
            with self.subTest(suite=suite):
                spec = SUITE_SPECS[suite]
                self.assertEqual(
                    spec["llm_invocation_contract"],
                    "generate_sql_physical_attempt_per_case",
                )
                self.assertEqual(
                    dataset_suite_report(suite)["llm_invocation_contract"],
                    "generate_sql_physical_attempt_per_case",
                )
                self.assertTrue(spec["baseline_eligible"])

    def test_ast_policy_rejects_untagged_multi_step_long_tail(self):
        base = dict(load_suite_cases("easy_sql_v1")[0])
        candidates = [
            {
                **base,
                "id": "bad_cte",
                "tags": [],
                "expected_sql": "WITH x AS (SELECT 1 AS n) SELECT n FROM x",
            },
            {
                **base,
                "id": "bad_window",
                "tags": [],
                "expected_sql": "SELECT ROW_NUMBER() OVER (ORDER BY order_id) FROM fact_order",
            },
            {
                **base,
                "id": "bad_subquery",
                "tags": [],
                "expected_sql": "SELECT * FROM (SELECT order_id FROM fact_order) x",
            },
            {
                **base,
                "id": "bad_set",
                "tags": [],
                "expected_sql": "SELECT 1 UNION SELECT 2",
            },
            {
                **base,
                "id": "bad_history",
                "tags": [],
                "history": [{"query": "此前问题", "sql": "SELECT 1"}],
                "expected_sql": "SELECT order_id FROM fact_order",
            },
            {
                **base,
                "id": "bad_joins",
                "tags": [],
                "expected_sql": (
                    "SELECT f.order_id FROM fact_order f "
                    "JOIN dim_date d ON f.date_id=d.date_id "
                    "JOIN dim_region r ON f.region_id=r.region_id "
                    "JOIN dim_product p ON f.product_id=p.product_id "
                    "JOIN dim_customer c ON f.customer_id=c.customer_id"
                ),
            },
        ]
        report = sql_ast_policy_report(candidates, "easy_sql_v1")
        reasons = {item["reason"] for item in report["violations"]}

        self.assertFalse(report["passed"])
        self.assertIn("forbidden_ast:cte", reasons)
        self.assertIn("forbidden_ast:window", reasons)
        self.assertIn("forbidden_ast:subquery", reasons)
        self.assertIn("forbidden_ast:set_operation", reasons)
        self.assertIn("history_forbidden", reasons)
        self.assertIn("join_count_exceeded", reasons)

    def test_explicit_filters_keep_unique_stable_case_ids(self):
        combined = load_suite_cases("easy_sql_v1")
        self.assertEqual(len({item["id"] for item in combined}), 100)
        smoke = load_suite_cases("easy_sql_v1", tier="smoke")
        self.assertEqual(len(smoke), 10)
        selected = load_suite_cases(
            "easy_sql_v1", only="easy_001,easy_100",
        )
        self.assertEqual(
            [item["id"] for item in selected],
            ["easy_001", "easy_100"],
        )
        with self.assertRaisesRegex(ValueError, "未知用例"):
            load_suite_cases("easy_sql_v1", only="missing")

    def test_loader_rejects_cases_outside_capability_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path = root / "cases.yaml"
            manifest_path = root / "manifest.yaml"
            cases_path.write_text(
                yaml.safe_dump([
                    {"id": "easy_001"},
                    {"id": "tail_unregistered"},
                ]),
                encoding="utf-8",
            )
            manifest_path.write_text(
                yaml.safe_dump({
                    "expected_cases": 1,
                    "case_ids": ["easy_001"],
                }),
                encoding="utf-8",
            )
            SUITE_SPECS["manifest_guard_test"] = {
                "case_manifest": manifest_path,
            }
            SUITE_PATHS["manifest_guard_test"] = (cases_path,)
            try:
                with self.assertRaisesRegex(ValueError, "能力清单外用例"):
                    load_suite_cases("manifest_guard_test")
            finally:
                SUITE_PATHS.pop("manifest_guard_test", None)
                SUITE_SPECS.pop("manifest_guard_test", None)

    def test_manifest_replaces_only_the_canonical_synthetic_dw(self):
        manifest = baseline_manifest()
        self.assertEqual(manifest["database"], "dw2")
        self.assertEqual(manifest["data_policy"]["reset_scope"], "dw2")
        self.assertTrue(manifest["data_policy"]["synthetic_only"])
        self.assertFalse(manifest["data_policy"]["contains_personal_data"])
        self.assertEqual(resolve_order_count(profile="medium"), 250000)
        self.assertEqual(resolve_order_count(profile="large"), 1000000)

    def test_meta_training_corpus_has_sixty_non_leaking_examples(self):
        examples = baseline_training_examples()
        self.assertEqual(len(examples), 60)
        golden_questions = {
            "".join(str(item["question"]).lower().split())
            for suite in SUITE_SPECS
            for item in load_suite_cases(suite)
        }
        training_questions = {
            "".join(str(item["question"]).lower().split())
            for item in examples
        }
        golden_sqls = {
            _normalized_sql(str(item["expected_sql"]))
            for suite in SUITE_SPECS
            for item in load_suite_cases(suite)
        }
        training_sqls = {
            _normalized_sql(str(item["sql"]))
            for item in examples
        }
        self.assertEqual(len(training_questions), 60)
        self.assertFalse(training_questions & golden_questions)
        self.assertFalse(training_sqls & golden_sqls)
        self.assertEqual(
            {
                suite: dataset_suite_report(suite)["independence"][
                    "training_sql_overlap"
                ]
                for suite in SUITE_SPECS
            },
            {suite: [] for suite in SUITE_SPECS},
        )

    def test_current_manifests_match_their_governed_case_counts(self):
        root = Path(__file__).resolve().parents[1]
        gates = yaml.safe_load(
            (root / "evals" / "configs" / "quality_gates.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertLessEqual(gates["thresholds"]["golden_min_cases"], 50)
        self.assertEqual(
            {
                suite: suite_manifest(suite)["expected_cases"]
                for suite in SUITE_SPECS
            },
            {"easy_sql_v1": 100, "medium_sql_v1": 100, "hard_sql_v1": 50},
        )

    def test_fixture_install_requires_explicit_reset_and_scale_bounds(self):
        with self.assertRaisesRegex(ValueError, "confirm-reset"):
            install_baseline_fixture(confirm_reset=False)
        with self.assertRaisesRegex(ValueError, "100000～1000000"):
            install_baseline_fixture(
                confirm_reset=True,
                order_count=99999,
            )


class EvaluationMetaExampleRefreshTests(unittest.IsolatedAsyncioTestCase):
    async def test_refresh_keeps_only_bounded_non_sensitive_dimension_enums(self):
        class FakeDW:
            values = {
                ("dim_date", "quarter"): ["Q2", "Q1", "Q2"],
                ("dim_date", "month"): [1, 2, 12],
                ("dim_product", "product_name"): ["A", "B", "C", "D"],
            }

            async def get_column_values(self, table_name, column_name, limit=10):
                return self.values[(table_name, column_name)][:limit]

        columns = [
            SimpleNamespace(
                id="dim_date.quarter", table_id="dim_date", name="quarter",
                role="dimension", examples=["stale"],
            ),
            SimpleNamespace(
                id="dim_product.product_name", table_id="dim_product",
                name="product_name", role="dimension", examples=["stale"],
            ),
            SimpleNamespace(
                id="dim_date.month", table_id="dim_date", name="month",
                role="dimension", examples=[1, 6, 12],
            ),
            SimpleNamespace(
                id="dim_customer.customer_name", table_id="dim_customer",
                name="customer_name", role="dimension", examples=["private"],
            ),
            SimpleNamespace(
                id="fact_order.order_amount", table_id="fact_order",
                name="order_amount", role="measure", examples=[999],
            ),
        ]

        report = await _refresh_baseline_enum_examples(
            columns, FakeDW(), max_cardinality=3,
        )

        self.assertEqual(columns[0].examples, ["Q1", "Q2"])
        self.assertEqual(columns[1].examples, [])
        self.assertEqual(columns[2].examples, [])
        self.assertEqual(columns[3].examples, [])
        self.assertEqual(columns[4].examples, [])
        self.assertEqual(report["refreshed_column_count"], 1)
        self.assertEqual(report["cleared_column_count"], 4)


if __name__ == "__main__":
    unittest.main()
