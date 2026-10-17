import unittest
from pathlib import Path

from app.config.meta_config import meta_config
from app.domain.query.semantic_aliases import (
    BUSINESS_SEMANTIC_ALIASES,
    business_aliases_for,
    merge_business_aliases,
)


class BusinessSemanticAliasesTest(unittest.TestCase):
    def test_registry_lookup_is_case_insensitive_and_read_only(self):
        self.assertEqual(
            business_aliases_for("FACT_ORDER", "Order_Amount"),
            ("成交金额", "单笔金额"),
        )
        with self.assertRaises(TypeError):
            BUSINESS_SEMANTIC_ALIASES[("fact_order", "new_column")] = ("新词",)

    def test_merge_preserves_curated_order_and_stably_deduplicates(self):
        aliases = merge_business_aliases(
            "fact_order",
            "order_amount",
            ["GMV", "成交金额", "GMV"],
            ["订单金额", " 成交金额 "],
        )

        self.assertEqual(
            aliases,
            ["GMV", "成交金额", "订单金额", "单笔金额"],
        )
        self.assertEqual(
            merge_business_aliases("unknown", "field", ["人工词"]),
            ["人工词"],
        )

    def test_canonical_meta_config_exposes_governed_aliases(self):
        columns = {
            (table.name, column.name): column
            for table in meta_config.tables
            for column in table.columns
        }
        for physical_ref, governed_aliases in BUSINESS_SEMANTIC_ALIASES.items():
            with self.subTest(physical_ref=physical_ref):
                self.assertTrue(
                    set(governed_aliases).issubset(
                        set(columns[physical_ref].alias)
                    )
                )

    def test_repeatable_migration_covers_the_registry(self):
        migration = (
            Path(__file__).parents[1]
            / "migrations"
            / "20260808_add_business_semantic_aliases.sql"
        ).read_text(encoding="utf-8")
        update_statements = [
            statement
            for statement in migration.split(";")
            if statement.lstrip().startswith("UPDATE")
        ]

        expected_statement_count = sum(
            len(aliases) for aliases in BUSINESS_SEMANTIC_ALIASES.values()
        )
        self.assertEqual(len(update_statements), expected_statement_count)
        for (table_name, column_name), aliases in BUSINESS_SEMANTIC_ALIASES.items():
            for alias in aliases:
                with self.subTest(
                    physical_ref=f"{table_name}.{column_name}", alias=alias
                ):
                    matches = [
                        statement
                        for statement in update_statements
                        if f"`table_id` = '{table_name}'" in statement
                        and f"`name` = '{column_name}'" in statement
                        and f"JSON_QUOTE('{alias}')" in statement
                        and "JSON_ARRAY_APPEND" in statement
                    ]
                    self.assertEqual(len(matches), 1)


if __name__ == "__main__":
    unittest.main()
