"""Evidence contract for governed NL2SQL suites that require model calls.

Some governed suites may legitimately complete deterministic cases without an
LLM. A suite with the physical-call contract has a different purpose: every
reviewed case must reach the physical ``generate_sql`` provider boundary. This
module prevents a compiler, budget, or validation fallback from being
misreported as model evidence.
"""
from __future__ import annotations

import time
from collections import Counter
from typing import Any

from app.agent_system.runtime.llm_attempts import ATTEMPT_STATUSES, LLM_ATTEMPT_SCHEMA

GENERATE_SQL_PHYSICAL_ATTEMPT_PER_CASE = (
    "generate_sql_physical_attempt_per_case"
)
NON_EMPTY_REFERENCE_WITH_PROJECTION = "non_empty_reference_with_projection"
_DETERMINISTIC_ACTIONS = {
    "deterministic_draft",
    "validated_deterministic_draft",
}
_PROVIDER_BOUNDARY_STAGES = {
    "provider",
    "provider_complete",
    "output_validation",
}


def _physical_generate_sql_attempts(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Return attempts that crossed the SQL provider invocation boundary."""

    return [
        attempt
        for attempt in (result.get("llm_attempts") or [])
        if str(attempt.get("node_id") or "") == "generate_sql"
        and str(attempt.get("route") or "") == "sql"
        and attempt.get("pre_invoke_failure") is False
        and int(attempt.get("physical_attempt_budget_ms") or 0) > 0
        and str(attempt.get("completion_stage") or "")
        in _PROVIDER_BOUNDARY_STAGES
    ]


async def validate_reference_result_contract(
    cases: list[dict[str, Any]],
    dw_repo: Any,
    contract: str | None,
) -> dict[str, Any]:
    """Execute reviewed references before any paid model call.

    Only row counts, column names, timings and failure classes are persisted;
    reference values never enter the governance artifact.
    """

    if contract is None:
        return {
            "schema_version": "reference-result-preflight-v1",
            "contract": contract,
            "required": False,
            "passed": True,
            "cases": [],
        }
    if contract != NON_EMPTY_REFERENCE_WITH_PROJECTION:
        return {
            "schema_version": "reference-result-preflight-v1",
            "contract": contract,
            "required": True,
            "supported": False,
            "passed": False,
            "failure_reason": "unknown_contract",
            "cases": [],
        }

    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        case_id = str(case.get("id") or f"case_{index + 1}")
        started = time.perf_counter()
        try:
            session = getattr(dw_repo, "session", None)
            if session is not None and hasattr(session, "rollback"):
                await session.rollback()
            executor = getattr(dw_repo, "execute_sql_for_evaluation", None)
            if callable(executor):
                result_rows = await executor(
                    str(case.get("expected_sql") or ""),
                    max_rows=10_000,
                )
            else:
                result_rows = await dw_repo.execute_sql(
                    str(case.get("expected_sql") or "")
                )
            columns = sorted({
                str(key)
                for row in result_rows
                if isinstance(row, dict)
                for key in row
            })
            expected_projection = sorted(
                str(value) for value in (case.get("expected_projection") or [])
            )
            missing_columns = sorted(set(expected_projection) - set(columns))
            row_count = len(result_rows)
            rows.append({
                "case_id": case_id,
                "executed": True,
                "row_count": row_count,
                "non_empty": row_count > 0,
                "columns": columns,
                "expected_projection": expected_projection,
                "missing_projection_columns": missing_columns,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "values_persisted": False,
                "passed": bool(row_count > 0 and not missing_columns),
            })
        except Exception as exc:  # noqa: BLE001 - fail-closed audit records type only
            rows.append({
                "case_id": case_id,
                "executed": False,
                "row_count": 0,
                "non_empty": False,
                "columns": [],
                "expected_projection": sorted(
                    str(value)
                    for value in (case.get("expected_projection") or [])
                ),
                "missing_projection_columns": [],
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "values_persisted": False,
                "error_type": type(exc).__name__,
                "passed": False,
            })
    return {
        "schema_version": "reference-result-preflight-v1",
        "contract": contract,
        "required": True,
        "supported": True,
        "passed": bool(rows) and all(row["passed"] for row in rows),
        "expected_cases": len(cases),
        "executed_cases": sum(row["executed"] for row in rows),
        "non_empty_cases": sum(row["non_empty"] for row in rows),
        "projection_matched_cases": sum(
            row["executed"] and not row["missing_projection_columns"]
            for row in rows
        ),
        "cases": rows,
    }


def evaluate_llm_invocation_contract(
    results: list[dict[str, Any]],
    contract: str | None,
    *,
    expected_case_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Audit whether a model-boundary result set contains genuine model calls.

    Provider errors and timeouts still count as physical attempts: they are
    valid observations of the live boundary.  Pre-invocation failures,
    deterministic compiler output, and closed-draft fallbacks do not.
    """

    if contract is None:
        return {
            "schema_version": "model-boundary-invocation-audit-v1",
            "contract": contract,
            "required": False,
            "passed": True,
        }
    if contract != GENERATE_SQL_PHYSICAL_ATTEMPT_PER_CASE:
        return {
            "schema_version": "model-boundary-invocation-audit-v1",
            "contract": contract,
            "required": True,
            "supported": False,
            "passed": False,
            "failure_reason": "unknown_contract",
        }

    case_rows: list[dict[str, Any]] = []
    policy_cases: list[str] = []
    physical_cases: list[str] = []
    fallback_cases: list[str] = []
    compiler_emitted_cases: list[str] = []
    missing_attempt_id_cases: list[str] = []
    invalid_attempt_statuses: list[dict[str, str]] = []
    invalid_attempt_schemas: list[dict[str, str]] = []
    invalid_policy_cases: list[str] = []
    invalid_stage_status_pairs: list[dict[str, str]] = []
    attempt_ids: list[str] = []
    per_node: dict[tuple[str, str], dict[str, Any]] = {}

    for index, result in enumerate(results):
        case_id = str(result.get("id") or f"case_{index + 1}")
        evidence = dict(result.get("stage_evidence") or {})
        semantic_ir = dict(evidence.get("semantic_ir") or {})
        policy = dict(semantic_ir.get("model_sql_policy") or {})
        compiler = dict(semantic_ir.get("compiler") or {})
        fallback = dict(evidence.get("model_sql_fallback") or {})
        physical = _physical_generate_sql_attempts(result)
        attempts = list(result.get("llm_attempts") or [])

        if policy.get("use_model") is True:
            policy_cases.append(case_id)
        if not (
            policy.get("use_model") is True
            and str(policy.get("decision") or "") == "model_sql"
        ):
            invalid_policy_cases.append(case_id)
        if physical:
            physical_cases.append(case_id)
        deterministic_action = str(policy.get("actual_action") or "")
        used_fallback = bool(fallback) or deterministic_action in _DETERMINISTIC_ACTIONS
        if used_fallback:
            fallback_cases.append(case_id)
        compiler_emitted = compiler.get("emitted") is True
        if compiler_emitted:
            compiler_emitted_cases.append(case_id)

        for attempt in attempts:
            schema = str(attempt.get("schema") or "")
            if schema != LLM_ATTEMPT_SCHEMA:
                invalid_attempt_schemas.append({
                    "case_id": case_id,
                    "schema": schema or "missing",
                })
            attempt_id = str(attempt.get("attempt_id") or "").strip()
            if attempt_id:
                attempt_ids.append(attempt_id)
            else:
                missing_attempt_id_cases.append(case_id)
            status = str(attempt.get("status") or "")
            if status not in ATTEMPT_STATUSES:
                invalid_attempt_statuses.append({
                    "case_id": case_id,
                    "status": status or "missing",
                })
            completion_stage = str(attempt.get("completion_stage") or "")
            if (
                (completion_stage == "provider_complete" and status != "success")
                or (
                    completion_stage in {"provider", "output_validation"}
                    and status == "success"
                )
            ):
                invalid_stage_status_pairs.append({
                    "case_id": case_id,
                    "completion_stage": completion_stage or "missing",
                    "status": status or "missing",
                })
            node_id = str(attempt.get("node_id") or "unknown")
            route = str(attempt.get("route") or "unknown")
            key = (node_id, route)
            row = per_node.setdefault(key, {
                "node_id": node_id,
                "route": route,
                "attempts": 0,
                "case_ids": set(),
                "statuses": Counter(),
                "total_tokens": 0,
                "unknown_tokens": 0,
            })
            row["attempts"] += 1
            row["case_ids"].add(case_id)
            row["statuses"][status or "missing"] += 1
            if attempt.get("token_known") is True:
                row["total_tokens"] += int(attempt.get("total_tokens") or 0)
            else:
                row["unknown_tokens"] += 1

        case_rows.append({
            "case_id": case_id,
            "policy_use_model": policy.get("use_model") is True,
            "policy_decision": str(policy.get("decision") or "unknown"),
            "physical_generate_sql_attempts": len(physical),
            "physical_attempt_ids": [
                str(item.get("attempt_id"))
                for item in physical
                if item.get("attempt_id")
            ],
            "physical_statuses": [
                str(item.get("status") or "missing") for item in physical
            ],
            "fallback_used": used_fallback,
            "compiler_emitted": compiler_emitted,
        })

    duplicate_attempt_ids = sorted(
        attempt_id
        for attempt_id, count in Counter(attempt_ids).items()
        if count > 1
    )
    per_node_rows = []
    for row in per_node.values():
        per_node_rows.append({
            **row,
            "unique_cases": len(row["case_ids"]),
            "case_ids": sorted(row["case_ids"]),
            "statuses": dict(sorted(row["statuses"].items())),
        })
    per_node_rows.sort(key=lambda item: (item["node_id"], item["route"]))

    expected_cases = [row["case_id"] for row in case_rows]
    observed_id_counts = Counter(expected_cases)
    duplicate_case_ids = sorted(
        case_id for case_id, count in observed_id_counts.items() if count > 1
    )
    governed_ids = [str(value) for value in (expected_case_ids or [])]
    missing_result_cases = (
        sorted(set(governed_ids) - set(expected_cases))
        if expected_case_ids is not None else []
    )
    unexpected_result_cases = (
        sorted(set(expected_cases) - set(governed_ids))
        if expected_case_ids is not None else []
    )
    complete_expected_set = bool(
        expected_case_ids is None
        or (
            len(expected_cases) == len(governed_ids)
            and not duplicate_case_ids
            and not missing_result_cases
            and not unexpected_result_cases
        )
    )
    missing_policy_cases = sorted(set(expected_cases) - set(policy_cases))
    missing_physical_cases = sorted(set(expected_cases) - set(physical_cases))
    passed = bool(results) and not any((
        missing_policy_cases,
        missing_physical_cases,
        fallback_cases,
        compiler_emitted_cases,
        missing_attempt_id_cases,
        invalid_attempt_statuses,
        invalid_attempt_schemas,
        invalid_policy_cases,
        invalid_stage_status_pairs,
        duplicate_attempt_ids,
        duplicate_case_ids,
        missing_result_cases,
        unexpected_result_cases,
    ))
    return {
        "schema_version": "model-boundary-invocation-audit-v1",
        "attempt_schema": LLM_ATTEMPT_SCHEMA,
        "contract": contract,
        "required": True,
        "supported": True,
        "passed": passed,
        "complete_expected_set": complete_expected_set,
        "expected_cases": len(expected_cases),
        "policy_model_cases": len(policy_cases),
        "physical_generate_sql_cases": len(physical_cases),
        "physical_generate_sql_attempts": sum(
            row["physical_generate_sql_attempts"] for row in case_rows
        ),
        "missing_policy_cases": missing_policy_cases,
        "missing_physical_attempt_cases": missing_physical_cases,
        "fallback_cases": sorted(set(fallback_cases)),
        "compiler_emitted_cases": sorted(set(compiler_emitted_cases)),
        "missing_attempt_id_cases": sorted(set(missing_attempt_id_cases)),
        "invalid_attempt_statuses": invalid_attempt_statuses,
        "invalid_attempt_schemas": invalid_attempt_schemas,
        "invalid_policy_cases": sorted(set(invalid_policy_cases)),
        "invalid_stage_status_pairs": invalid_stage_status_pairs,
        "duplicate_attempt_ids": duplicate_attempt_ids,
        "duplicate_case_ids": duplicate_case_ids,
        "missing_result_cases": missing_result_cases,
        "unexpected_result_cases": unexpected_result_cases,
        "per_node": per_node_rows,
        "cases": case_rows,
    }
