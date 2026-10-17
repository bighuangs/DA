import unittest
from types import SimpleNamespace

from app.adapters.inbound.http.schemas.meta_schema import MaskingRuleSchema
from app.application.services.masking_service import (
    MaskingPolicy,
    MaskingRule,
    compile_rules,
    mask_rows,
    mask_sql_literals,
    masking_policy,
    masking_rules_fingerprint,
)


class MaskingServiceTest(unittest.TestCase):
    def test_heuristic_masks_pii_but_keeps_metrics(self):
        rows = mask_rows([
            {"phone": "13812345678", "email": "alice@example.com", "total_amount": 1234},
        ], rules=())
        self.assertEqual(rows[0]["phone"], "138****5678")
        self.assertEqual(rows[0]["email"], "a***@example.com")
        self.assertEqual(rows[0]["total_amount"], 1234)

    def test_explicit_rule_supports_wildcard_and_hash(self):
        rules = compile_rules([{
            "column_pattern": "customer.*",
            "data_kind": "generic",
            "strategy": "hash",
            "priority": 1,
        }])
        result = mask_rows(
            [{"customer.code": "C-001", "amount": 10}],
            rules=rules,
        )
        self.assertEqual(len(result[0]["customer.code"]), 12)
        self.assertEqual(result[0]["amount"], 10)

    def test_sql_storage_sanitizes_obvious_literals(self):
        sql = "SELECT * FROM customer WHERE email = 'alice@example.com' AND status = 'ok'"
        stored = mask_sql_literals(sql)
        self.assertNotIn("alice@example.com", stored)
        self.assertIn("status = 'ok'", stored)

    def test_business_name_columns_are_not_treated_as_person_names(self):
        result = mask_rows([{
            "customer_name": "张三",
            "product_name": "分析平台",
            "region_name": "华东",
        }], rules=())
        self.assertEqual(result[0]["customer_name"], "张*")
        self.assertEqual(result[0]["product_name"], "分析平台")
        self.assertEqual(result[0]["region_name"], "华东")

    def test_field_tokens_use_boundaries_instead_of_arbitrary_substrings(self):
        result = mask_rows([{
            "hotel_name": "Grand Hotel",
            "mail_count": 12,
            "identity_score": 98,
        }], rules=())
        self.assertEqual(result[0]["hotel_name"], "Grand Hotel")
        self.assertEqual(result[0]["mail_count"], 12)
        self.assertEqual(result[0]["identity_score"], 98)

    def test_value_format_inference_is_applied_to_unknown_columns(self):
        result = mask_rows([
            {"contact": "alice@example.com"},
            {"contact": "bob@example.com"},
        ], rules=())
        self.assertEqual(result[0]["contact"], "a***@example.com")
        self.assertEqual(result[1]["contact"], "b***@example.com")

    def test_business_identifiers_are_not_inferred_as_bank_cards(self):
        result = mask_rows([
            {"order_number": "4532015112830366"},
            {"order_number": "6011000990139424"},
        ], rules=())
        self.assertEqual(result[0]["order_number"], "4532015112830366")
        self.assertEqual(result[1]["order_number"], "6011000990139424")

    def test_heuristic_switch_disables_name_and_value_inference(self):
        result = mask_rows([
            {"customer_name": "张三", "phone": "13812345678", "contact": "alice@example.com"},
            {"customer_name": "李四", "phone": "13912345678", "contact": "bob@example.com"},
        ], rules=(), policy=MaskingPolicy(heuristic_enabled=False))
        self.assertEqual(result[0]["customer_name"], "张三")
        self.assertEqual(result[0]["phone"], "13812345678")
        self.assertEqual(result[0]["contact"], "alice@example.com")

    def test_keep_rule_explicitly_overrides_builtin_masking(self):
        rules = (MaskingRule("customer_name", data_kind="name", strategy="keep", priority=1),)
        result = mask_rows([{"customer_name": "张三"}], rules=rules)
        self.assertEqual(result[0]["customer_name"], "张三")
        self.assertEqual(MaskingRuleSchema(column_pattern="customer_name", strategy="keep").strategy, "keep")

    def test_policy_is_visible_and_rule_fingerprint_changes_with_rules(self):
        policy = masking_policy()
        self.assertTrue(policy["heuristic_enabled"])
        self.assertTrue(any(item["data_kind"] == "name" for item in policy["builtins"]))
        self.assertNotEqual(
            masking_rules_fingerprint(()),
            masking_rules_fingerprint((MaskingRule("customer_name", strategy="keep"),)),
        )

    def test_database_orm_rule_is_compiled_without_mapping_interface(self):
        orm_rule = SimpleNamespace(
            column_pattern="customer_name",
            data_kind="name",
            strategy="keep",
            enabled=True,
            priority=1,
            description="测试放行",
        )
        compiled = compile_rules([orm_rule])
        self.assertEqual(compiled, (
            MaskingRule("customer_name", "name", "keep", True, 1, "测试放行"),
        ))
        self.assertEqual(mask_rows([{"customer_name": "张三"}], rules=compiled)[0]["customer_name"], "张三")


if __name__ == "__main__":
    unittest.main()
