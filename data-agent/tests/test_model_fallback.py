from app.infrastructure.clients.llm.fallback import (
    MODEL_CLARIFICATION_VERSION,
    build_model_clarification,
)


def test_model_clarification_is_structured_and_contains_no_query_or_sql():
    value = build_model_clarification(
        node_id="generate_sql",
        reason_code="llm.timeout",
        semantic_ir={
            "semantic_coverage": {
                "unresolved_slots": [
                    "operator:set",
                    "filterop:category:negated",
                ]
            }
        },
        retryable=True,
    )
    assert value["schema_version"] == MODEL_CLARIFICATION_VERSION
    assert value["action"] == "clarify_then_retry"
    assert value["retryable"] is True
    assert value["contains_query_or_sql"] is False
    assert value["unresolved_slots"] == [
        "operator:set",
        "filterop:category:negated",
    ]
