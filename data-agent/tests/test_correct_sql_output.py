from app.agent_system.agents.nl2sql.nodes.correct_sql import (
    retain_valid_correction,
    semantic_ir_for_correction_prompt,
)


def test_empty_correction_keeps_previous_sql():
    previous = "SELECT COUNT(*) FROM fact_order"
    sql, accepted = retain_valid_correction(previous, "  ")
    assert sql == previous
    assert accepted is False


def test_non_empty_correction_is_trimmed_and_accepted():
    sql, accepted = retain_valid_correction("SELECT 1", "  SELECT 2  ")
    assert sql == "SELECT 2"
    assert accepted is True


def test_guided_ir_details_are_withheld_from_correction_prompt():
    visible = semantic_ir_for_correction_prompt({
        "compiler": {"supported": True},
        "dimensions": [{"column": "wrong_dimension"}],
        "measures": [{"column": "wrong_measure"}],
        "semantic_slots": {"partition_keys": ["wrong_dimension"]},
        "query_contract": {"authority": "guided"},
        "model_sql_policy": {"use_model": True},
    })

    assert visible["status"] == "guided_ir_withheld"
    assert visible["query_contract"]["authority"] == "guided"
    assert "dimensions" not in visible
    assert "measures" not in visible
    assert "semantic_slots" not in visible


def test_closed_ir_remains_available_for_deterministic_ablation():
    original = {
        "compiler": {"supported": True},
        "dimensions": [{"column": "region_name"}],
        "query_contract": {"authority": "closed"},
    }

    assert semantic_ir_for_correction_prompt(original) == original
