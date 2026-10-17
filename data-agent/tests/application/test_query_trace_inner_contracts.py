"""Query Trace 内层合同与旧运行语义的定向回归。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import app.application.services.query_trace_service as query_trace_module
from app.agent_system.agents.nl2sql.contracts.planning_route import (
    PlanningRouteFeaturesV1,
    compact_planning_route as legacy_compact_planning_route,
)
from app.agent_system.agents.nl2sql.relation_graph import (
    compact_relation_closure as legacy_compact_relation_closure,
)
from app.agent_system.agents.nl2sql.semantic_contract import (
    compact_query_contract as legacy_compact_query_contract,
)
from app.agent_system.agents.nl2sql.services.candidate_confidence import (
    compact_candidate_decision as legacy_compact_candidate_decision,
)
from app.agent_system.agents.nl2sql.services.semantic_repair import (
    compact_semantic_repair as legacy_compact_semantic_repair,
)
from app.agent_system.runtime.performance import current_node_id as legacy_node_id
from app.application.ports.query_tracing import QueryTraceLLMSettings
from app.application.services.query_trace_service import QueryTraceCollector
from app.domain.query.safety.semantic_violation import (
    compact_evidence as legacy_compact_semantic_evidence,
)
from app.foundation.audit_compaction import (
    PLANNING_FEATURE_FIELDS,
    compact_candidate_decision,
    compact_planning_route,
    compact_query_contract,
    compact_relation_closure,
    compact_semantic_evidence,
    compact_semantic_repair,
)
from app.foundation.runtime_context import current_node_id


class _RouteCatalog:
    def route_for_node(self, node_id: str | None) -> str:
        return f"route:{node_id or 'unknown'}"

    def profile_for_route(self, route: str | None) -> dict:
        return {
            "task": str(route or "unknown"),
            "model_name": "catalog-model",
            "timeout_seconds": 1.0,
            "max_retries": 0,
            "max_concurrency": 1,
        }


def _candidate_payload() -> dict:
    return {
        "decision_mode": "semantic_review",
        "semantic_review_triggered": True,
        "semantic_review_eligible": True,
        "trigger_reason": "semantic_review_required",
        "selected": "B",
        "selection_reason": "semantic_review_improved",
        "candidate_count": 2,
        "extra_call_used": True,
        "candidates": [{
            "label": "A",
            "valid": True,
            "score": 0.7,
            "band": "medium",
            "weaknesses": ["join"],
            "actionable_weaknesses": ["join"],
            "validation": {"error_code": "none"},
            "sql_features": {"join_count": 1},
            "query_contract_audit": {
                "schema_version": "query-contract-audit-v1",
                "valid": False,
                "violations": [{"code": "grain"}],
                "contract_hash": "sha256:contract",
            },
            "sql": "不得进入 Trace",
        }],
    }


def _planning_payload() -> dict:
    return {
        "contract_version": "planning-route-v1",
        "profile": "semantic_ir_v1",
        "decision": "plan_then_generate",
        "actual_action": "plan_then_generate",
        "score": 1.23456,
        "confidence": 0.98765,
        "reason_codes": ["semantic_ir_supported"],
        "feature_schema": "planning-features-v1",
        "feature_fingerprint": "sha256:feature",
        "feature_summary": {
            "feature_schema": "planning-features-v1",
            "query_length_bucket": "13_24",
            "has_aggregation": True,
            "raw_question": "不得进入 Trace",
        },
        "budget_snapshot": {
            "remaining_deadline_ms": 9000,
            "planning_timeout_ms": 3000,
            "planning_timeout_bucket": "under_5s",
            "absolute_deadline": 123.4,
        },
        "shadow_decisions": [{
            "profile": "shadow-v1",
            "decision": "direct_generate",
            "score": 0.1,
            "confidence": 0.2,
            "reason_codes": ["shadow"],
        }],
    }


def _relation_payload() -> dict:
    return {
        "schema_version": "metadata-relation-closure-v1",
        "status": "closed",
        "required_columns": {"fact_order": ["region_id", "amount"]},
        "terminal_tables": ["fact_order", "dim_region"],
        "added_tables": ["dim_region"],
        "added_columns": {"dim_region": ["region_id"]},
        "joins": [{
            "left_table": "fact_order",
            "left_column": "region_id",
            "right_table": "dim_region",
            "right_column": "region_id",
            "relationship": "many_to_one",
            "evidence": "pk_fk_role",
            "confidence": 1.0,
            "filter_value": "不得进入 Trace",
        }],
        "ambiguous_paths": [{
            "target_table": "dim_region",
            "candidate_path_count": 1,
            "hop_count": 1,
            "raw_path": "不得进入 Trace",
        }],
        "edge_count": 1,
    }


def _repair_payload() -> dict:
    return {
        "contract_version": "semantic-repair-result-v1",
        "registry_version": "registry-v1",
        "rule_set_version": "rules-v1",
        "profile": "same-rules-v1",
        "changed": True,
        "original_sql_hash": "sha256:before",
        "repaired_sql_hash": "sha256:after",
        "applied_rules": [{
            "rule_code": "join.unique_metadata_edge",
            "rule_version": "1",
            "status": "applied",
            "proof": {
                "candidate_edge_count": 1,
                "sql_hash": "sha256:before",
                "raw_sql": "不得进入 Trace",
            },
        }],
        "attempted_rule_codes": ["join.unique_metadata_edge"],
        "repair_ms": 0.8,
        "fallback_used": False,
        "invariants": {"single_pass": True},
    }


def test_compaction_contracts_match_legacy_exports() -> None:
    relation = _relation_payload()
    query_contract = {
        "schema_version": "semantic-query-contract-v1",
        "semantic_family": "share",
        "authority": "closed",
        "confidence": 1.0,
        "contract_hash": "sha256:contract",
        "output_grain": [{"table": "dim_region", "column": "region_name"}],
        "metric_contracts": [{"name": "sales", "type": "sum"}],
        "filter_contract": {"mode": "registered"},
        "operator_contract": {"ratio_required": True},
        "operations": ["share"],
        "stage_plan": [{"stage": "aggregate"}],
        "relation_plan": relation,
        "unresolved_slots": [],
        "raw_question": "不得进入 Trace",
    }
    evidence = {
        "candidate_edge_count": 1,
        "changed_ast_nodes": ["join.on"],
        "sql_hash": "sha256:sql",
        "raw_sql": "不得进入 Trace",
    }

    assert compact_candidate_decision(
        _candidate_payload()
    ) == legacy_compact_candidate_decision(_candidate_payload())
    assert compact_planning_route(
        _planning_payload()
    ) == legacy_compact_planning_route(_planning_payload())
    assert compact_relation_closure(
        relation
    ) == legacy_compact_relation_closure(relation)
    assert compact_query_contract(
        query_contract
    ) == legacy_compact_query_contract(query_contract)
    assert compact_semantic_repair(
        _repair_payload()
    ) == legacy_compact_semantic_repair(_repair_payload())
    assert compact_semantic_evidence(
        evidence
    ) == legacy_compact_semantic_evidence(evidence)


def test_planning_feature_whitelist_matches_legacy_contract() -> None:
    assert PLANNING_FEATURE_FIELDS == {
        *PlanningRouteFeaturesV1.__dataclass_fields__,
        "feature_schema",
    }


def test_callback_uses_injected_route_and_cost_contracts() -> None:
    collector = QueryTraceCollector(
        chat_id="injected-trace",
        route_catalog=_RouteCatalog(),
        llm_settings=QueryTraceLLMSettings(
            default_model="settings-model",
            input_cost_per_million=2.0,
            output_cost_per_million=4.0,
            cost_currency="CNY",
        ),
    )
    token = current_node_id.set("custom_node")
    try:
        collector.callback.on_llm_start({}, [], run_id="success-run")
        collector.callback.on_llm_end(
            SimpleNamespace(
                llm_output={
                    "token_usage": {
                        "prompt_tokens": 1_000_000,
                        "completion_tokens": 500_000,
                        "total_tokens": 1_500_000,
                    },
                },
                generations=[],
            ),
            run_id="success-run",
        )
        collector.callback.on_llm_start({}, [], run_id="error-run")
        collector.callback.on_llm_error(
            asyncio.TimeoutError(),
            run_id="error-run",
        )
    finally:
        current_node_id.reset(token)

    summary = collector.finish()
    assert collector.token_usages[0].model == "settings-model"
    assert collector.token_usages[0].estimated_cost == 4.0
    assert collector.token_usages[1].model == "catalog-model"
    assert collector.token_usages[1].status == "timeout"
    assert summary["estimated_cost"] == 4.0
    assert summary["cost_currency"] == "CNY"
    assert summary["llm_attempts"][0]["route"] == "route:custom_node"


def test_default_path_retains_module_patch_points(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        query_trace_module,
        "route_for_node",
        lambda node_id: f"patched:{node_id}",
    )
    monkeypatch.setattr(
        query_trace_module,
        "task_profile",
        lambda route: {"model_name": f"model:{route}"},
    )
    collector = QueryTraceCollector(chat_id="patched-trace")
    collector.record_llm_attempt({
        "node_id": "custom_node",
        "status": "timeout",
        "latency_ms": 10,
        "timeout_budget_ms": 20,
        "token_known": False,
    })

    attempt = collector.explicit_llm_attempts[0]
    assert attempt["route"] == "patched:custom_node"
    assert attempt["model"] == "model:patched:custom_node"


def test_runtime_context_old_path_preserves_contextvar_identity() -> None:
    assert legacy_node_id is current_node_id
