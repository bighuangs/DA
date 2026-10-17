"""Deterministic semantic coverage tests for the current difficulty suites."""
from pathlib import Path

import pytest

from app.agent_system.agents.nl2sql.semantic_coverage import SEMANTIC_COVERAGE_VERSION
from app.agent_system.agents.nl2sql.semantic_ir import build_semantic_ir, compile_semantic_ir
from app.agent_system.agents.nl2sql.semantic_ir_v2 import (
    COMPILER_MODE_V2,
    validate_compiled_semantic_ir,
)
from tests.semantic_fixtures import difficulty_sql_cases, semantic_metadata


@pytest.mark.parametrize(
    "case",
    difficulty_sql_cases(),
    ids=lambda item: item["id"],
)
def test_current_difficulty_sets_compile_only_closed_semantic_ir(case):
    ir = build_semantic_ir(
        case["question"],
        semantic_metadata(),
        [],
        case.get("history") or [],
    )
    coverage = ir["semantic_coverage"]
    sql = compile_semantic_ir(ir)

    assert coverage["schema_version"] == SEMANTIC_COVERAGE_VERSION
    if ir["compiler"]["supported"]:
        assert coverage["unresolved_slots"] == []
        assert set(coverage["detected_slots"]) == set(
            coverage["represented_slots"]
        )
        assert coverage["coverage_score"] == 1.0
        # This unit contract checks that a deterministic route is internally
        # closed and auditable. Reference-table/result accuracy remains an
        # observational live-baseline metric and is not forced green here.
        assert ir["compiler"]["mode"] in {
            COMPILER_MODE_V2, "deterministic_ast",
        }
        assert sql
        if ir["compiler"]["mode"] == COMPILER_MODE_V2:
            validate_compiled_semantic_ir(sql, ir)
    else:
        assert ir["compiler"]["mode"] == "llm_fallback"
        assert coverage["unresolved_slots"] or coverage["abstain_reasons"]
        assert sql is None


def test_each_current_difficulty_has_deterministic_and_model_routed_cases():
    by_suite: dict[str, list[dict]] = {}
    for case in difficulty_sql_cases():
        by_suite.setdefault(case["dataset_suite"], []).append(case)

    assert set(by_suite) == {
        "easy_sql_v1", "medium_sql_v1", "hard_sql_v1",
    }
    expected_counts = {
        "easy_sql_v1": 100,
        "medium_sql_v1": 100,
        "hard_sql_v1": 50,
    }
    for suite, cases in by_suite.items():
        decisions = [
            build_semantic_ir(
                case["question"], semantic_metadata(), [],
                case.get("history") or [],
            )["compiler"]["supported"]
            for case in cases
        ]
        assert len(cases) == expected_counts[suite], suite
        assert any(decisions), suite
        assert not all(decisions), suite


def test_unregistered_multi_step_request_abstains_instead_of_using_tail_template():
    question = (
        "先按客户计算每笔订单间隔，再按首购品类队列比较连续三个月的"
        "复购贡献累计占比"
    )
    ir = build_semantic_ir(question, semantic_metadata(), [], [])

    assert ir["status"] == "partial"
    assert ir["compiler"]["supported"] is False
    assert ir["compiler"]["mode"] == "llm_fallback"
    assert any(
        item.startswith("operator:unregistered_")
        for item in ir["semantic_coverage"]["unresolved_slots"]
    )
    assert compile_semantic_ir(ir) is None


def test_runtime_semantic_modules_have_no_benchmark_or_reference_project_coupling():
    root = Path(__file__).resolve().parents[1]
    source = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in (
            "app/agent_system/agents/nl2sql/semantic_coverage.py",
            "app/agent_system/agents/nl2sql/semantic_ir_v2.py",
        )
    )
    for forbidden in (
        "challenge_",
        "generalization_",
        "difficulty_sql_cases",
        "SQLBot",
        r"D:\private\unrelated-project",
    ):
        assert forbidden not in source
