import json
import unittest

from app.infrastructure.persistence.mysql.repositories.query_trace_repository import build_trace_records
from evals.application.services.quality_operations_service import QualityOperationsService
from app.application.services.query_trace_service import QueryTraceCollector


class SemanticRepairTraceTests(unittest.TestCase):
    def test_trace_persists_only_privacy_safe_repair_contract(self):
        collector = QueryTraceCollector(chat_id="semantic-repair-proof")
        collector.record_semantic_repair(repair={
            "contract_version": "semantic-repair-result-v1",
            "registry_version": "semantic-repair-registry-v1",
            "rule_set_version": "semantic-repair-rules-v1",
            "profile": "registry_same_rules_v1",
            "changed": True,
            "original_sql_hash": "sha256:before",
            "repaired_sql_hash": "sha256:after",
            "applied_rules": [{
                "rule_code": "join.unique_metadata_edge",
                "rule_version": "1",
                "status": "applied",
                "changed_ast_nodes": ["join.on"],
                "proof": {"candidate_edge_count": 1, "raw_sql": "SELECT secret"},
            }],
            "repair_ms": .8,
            "fallback_used": False,
            "invariants": {"single_bounded_pass": True},
            "raw_sql_logged": False,
            "raw_sql": "SELECT customer_name FROM customers",
        })
        collector.record_semantic_repair(violation={
            "contract_version": "semantic-violation-v1",
            "violation_code": "semantic.join.missing_predicate",
            "stage": "semantic_contract",
            "repairability": "deterministic",
            "safe_to_retry": True,
            "evidence": {"sql_hash": "sha256:before", "question": "Alice"},
            "message": "不得保存",
        })
        summary, trace, _, _ = build_trace_records(
            collector,
            query="Alice 的客户订单",
            chat_id="semantic-repair-proof",
            sql="SELECT customer_name FROM customers",
            status="success",
            row_count=1,
        )
        raw = trace.semantic_repair_summary
        self.assertNotIn("Alice", raw)
        self.assertNotIn("SELECT secret", raw)
        self.assertNotIn("不得保存", raw)
        stored = json.loads(raw)
        self.assertEqual("registry_same_rules_v1", stored["profile"])
        self.assertEqual(
            "semantic.join.missing_predicate",
            stored["violation"]["violation_code"],
        )
        self.assertFalse(stored["raw_sql_logged"])
        self.assertEqual(stored, summary["semantic_repair"])
        self.assertEqual(
            stored,
            QualityOperationsService._semantic_repair_evidence(raw),
        )


if __name__ == "__main__":
    unittest.main()
