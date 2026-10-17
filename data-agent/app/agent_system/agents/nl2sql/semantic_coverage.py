"""Bounded semantic coverage evidence for compiler routing.

This module deliberately does not compile SQL.  It records the reusable
semantic slots discovered by ``semantic-ir-v2`` and checks whether an emitted
candidate represents them.  Unregistered multi-step requests are marked as
unresolved so they can be handled by a model, clarification, or an explicit
human rule instead of growing a benchmark-shaped template registry.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

import sqlglot
from sqlglot import exp

from app.agent_system.agents.nl2sql.semantic_ir_v2 import (
    build_semantic_ir_v2,
    semantic_question_cues,
    validate_compiled_semantic_ir,
)


SEMANTIC_COVERAGE_VERSION = "semantic-coverage-v1"
COVERAGE_REGISTRY_VERSION = "semantic-coverage-bounded-v3"

_FAMILY_OPERATORS: dict[str, tuple[str, ...]] = {
    "conditional_aggregate": ("operator:conditional_aggregate",),
    "having_aggregate": ("operator:having",),
    "qualifying_group_count": (
        "operator:having", "operator:count_qualified_groups",
        "output:scalar",
    ),
    "distribution_rank": (
        "operator:percentile_rank", "operator:partitioned_rank",
    ),
    "period_compare": ("operator:period_compare",),
    "window_mom": ("operator:period_growth", "operator:period_delta"),
    "window_running_total": ("operator:running_sum",),
    "group_share": ("operator:share",),
    "top_share": ("operator:share", "operator:bounded_rank"),
    "top_per_group": ("operator:top_per_group", "operator:bounded_rank"),
    "active_both_periods": ("operator:set",),
    "cohort_new": ("operator:cohort",),
    "repeat_rate": ("operator:repeat_rate",),
    "segmentation": ("operator:segmentation",),
    "above_average_customer": ("operator:above_average",),
    "above_average_brand": ("operator:above_average",),
    "entity_aggregate_average": ("operator:entity_aggregate_average",),
    "having_threshold": ("operator:having",),
    "inventory_status": ("operator:anti",),
    "anti_join": ("operator:anti",),
}

# These cues are intentionally broad.  They do not claim support; they stop an
# unrecognised multi-stage question from being mistaken for a simple aggregate.
_UNREGISTERED_MULTI_STEP_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("operator:unregistered_multi_step", re.compile(r"先.{0,40}(?:再|然后|之后|最后)")),
    ("operator:unregistered_cohort", re.compile(r"(?:留存|流失|生命周期|队列|cohort)")),
    ("operator:unregistered_set", re.compile(r"(?:交集|并集|差集|同时购买|既.+又)")),
    ("operator:unregistered_window", re.compile(r"(?:连续\d+|移动平均|累计占比|累计贡献)")),
    ("operator:unregistered_nested_rank", re.compile(r"每(?:个|一).{0,16}(?:前\d+|最高的?\d*|最低的?\d*)")),
)


def _normalized(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _operand_hash(logical: str, value: Any) -> str:
    return hashlib.sha256(
        f"{logical}\0{_normalized(value)}".encode("utf-8")
    ).hexdigest()


def _ir_slots(ir: Mapping[str, Any] | None) -> tuple[
    list[str], list[str], list[dict[str, Any]], list[str]
]:
    if not ir:
        return [], [], [], []
    dimensions = [
        str(item.get("logical") or item.get("column") or "")
        for item in ir.get("dimensions") or []
        if isinstance(item, Mapping)
        and (item.get("logical") or item.get("column"))
    ]
    measures = [
        {
            "logical": str(item.get("logical") or item.get("column") or ""),
            "function": str(item.get("function") or "").upper(),
            "distinct": bool(item.get("distinct")),
        }
        for item in ir.get("measures") or []
        if isinstance(item, Mapping)
        and (item.get("logical") or item.get("column"))
    ]
    filters: list[str] = []
    slots: list[str] = [f"dimension:{item}" for item in dimensions]
    slots.extend(
        "measure:"
        f"{item['function'].lower()}:{item['logical']}:"
        f"{'distinct' if item['distinct'] else 'all'}"
        for item in measures
    )
    for item in ir.get("filters") or []:
        if not isinstance(item, Mapping):
            continue
        logical = str(item.get("logical") or "")
        if not logical:
            continue
        filters.append(logical)
        slots.append(f"filter:{logical}")
        operator = str(item.get("operator") or "")
        polarity = "negative" if operator in {"neq", "not_in"} else "positive"
        value = item.get("value")
        values = list(value) if isinstance(value, (list, tuple)) else [value]
        slots.extend(
            f"filteroperand:{logical}:{polarity}:{_operand_hash(logical, operand)}"
            for operand in values
        )
        if polarity == "negative":
            slots.append(f"filterop:{logical}:negated")
        if len(values) > 1:
            slots.append(f"filtercard:{logical}:multi")
    family = str(ir.get("semantic_family") or ir.get("intent") or "")
    slots.extend(_FAMILY_OPERATORS.get(family, ()))
    return list(dict.fromkeys(slots)), dimensions, measures, list(dict.fromkeys(filters))


def _unregistered_slots(query: str) -> list[str]:
    normalized = _normalized(query)
    return [slot for slot, pattern in _UNREGISTERED_MULTI_STEP_CUES if pattern.search(normalized)]


def _question_cue_slots(
    query: str,
    table_infos: Sequence[Mapping[str, Any]] | None,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Return question-derived slots without relying on compiled IR output."""

    cues = semantic_question_cues(query, table_infos)
    slots = [f"dimension:{item}" for item in cues.get("dimensions") or []]
    measures = [dict(item) for item in cues.get("measures") or []]
    slots.extend(
        "measure:"
        f"{str(item.get('function') or '').lower()}:"
        f"{str(item.get('logical') or '')}:"
        f"{'distinct' if item.get('distinct') else 'all'}"
        for item in measures
        if item.get("logical") and item.get("function")
    )
    for item in cues.get("comparisons") or []:
        logical = str(item.get("logical") or "")
        operator = str(item.get("operator") or "")
        if not logical or not operator:
            continue
        scope = "having" if item.get("aggregate_scope") else "row"
        slots.append(
            f"comparison:{scope}:{logical}:{operator}:"
            f"{_operand_hash(logical, item.get('value'))}"
        )
    slots.extend(f"operator:{item}" for item in cues.get("operations") or [])
    if cues.get("counted_group"):
        slots.append(f"grain:{cues['counted_group']}")
    if cues.get("output_shape") == "scalar":
        slots.append("output:scalar")
    return list(dict.fromkeys(slots)), list(cues.get("dimensions") or []), measures


def _payload(
    *,
    query: str,
    detected: Sequence[str],
    represented: Sequence[str],
    dimensions: Sequence[str],
    measures: Sequence[Mapping[str, Any]],
    filters: Sequence[str],
    compiled: bool = False,
) -> dict[str, Any]:
    detected_unique = list(dict.fromkeys(str(item) for item in detected))
    represented_set = set(str(item) for item in represented)
    represented_unique = [item for item in detected_unique if item in represented_set]
    unresolved = [item for item in detected_unique if item not in represented_set]
    return {
        "schema_version": SEMANTIC_COVERAGE_VERSION,
        "question_fingerprint": _hash(_normalized(query)),
        "registry_version": COVERAGE_REGISTRY_VERSION,
        "detected_slots": detected_unique,
        "represented_slots": represented_unique,
        "unresolved_slots": unresolved,
        "dimensions": list(dimensions),
        "measures": [dict(item) for item in measures],
        "filters": list(filters),
        "time_constraints": [
            item for item in filters
            if item in {"year", "year_range", "quarter", "month"}
        ],
        "comparisons": [item for item in detected_unique if item in {
            "operator:period_compare", "operator:period_growth", "operator:share",
        }],
        "set_operations": [item for item in detected_unique if item in {
            "operator:set", "operator:anti", "operator:unregistered_set",
        }],
        "window_operations": [item for item in detected_unique if item in {
            "operator:top_per_group", "operator:period_growth",
            "operator:running_sum", "operator:unregistered_window",
            "operator:unregistered_nested_rank",
        }],
        "cohort_operations": [
            item for item in detected_unique
            if item in {"operator:cohort", "operator:unregistered_cohort"}
        ],
        "follow_up_operations": [],
        "slot_hashes": [_hash(item) for item in detected_unique],
        "coverage_score": (
            1.0 if compiled and not detected_unique
            else round(len(represented_unique) / max(1, len(detected_unique)), 4)
        ),
        "abstain_reasons": (
            ["semantic_slots_unresolved"] if unresolved
            else ([] if detected_unique or compiled else ["no_registered_semantic_slots"])
        ),
    }


def _coverage_ir(
    query: str,
    table_infos: Sequence[Mapping[str, Any]] | None,
) -> Mapping[str, Any] | None:
    return build_semantic_ir_v2(query, list(table_infos or []), [], [])


def detect_semantic_coverage(
    query: str,
    table_infos: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Return detected semantic slots with none represented yet."""

    ir = _coverage_ir(query, table_infos)
    detected, dimensions, measures, filters = _ir_slots(ir)
    cue_slots, cue_dimensions, cue_measures = _question_cue_slots(
        query, table_infos,
    )
    detected.extend(cue_slots)
    dimensions = list(dict.fromkeys([*dimensions, *cue_dimensions]))
    measure_index = {
        (
            str(item.get("logical") or ""),
            str(item.get("function") or "").upper(),
            bool(item.get("distinct")),
        ): dict(item)
        for item in [*measures, *cue_measures]
    }
    measures = list(measure_index.values())
    detected.extend(_unregistered_slots(query))
    payload = _payload(
        query=query,
        detected=detected,
        represented=[],
        dimensions=dimensions,
        measures=measures,
        filters=filters,
    )
    payload["audit_source"] = "bounded_question_slot_detector"
    return payload


def _physical_column(ir: Mapping[str, Any], logical: str) -> str:
    resolved = (ir.get("resolved_fields") or {}).get(logical)
    if isinstance(resolved, Mapping) and resolved.get("column"):
        return str(resolved["column"])
    return logical


def _comparison_polarity(node: exp.Expression) -> str:
    if isinstance(node, exp.NEQ):
        return "negative"
    parent = node.parent
    while parent is not None:
        if isinstance(parent, exp.Not):
            return "negative"
        parent = parent.parent
    return "positive"


def _sql_operand_keys(
    tree: exp.Expression,
    ir: Mapping[str, Any],
) -> set[tuple[str, str, str]]:
    physical_to_logical = {
        _physical_column(ir, logical): logical
        for logical in (ir.get("resolved_fields") or {})
    }
    output: set[tuple[str, str, str]] = set()
    for node in tree.walk():
        if not isinstance(node, (
            exp.EQ, exp.NEQ, exp.In, exp.GT, exp.GTE, exp.LT, exp.LTE,
            exp.Between,
        )):
            continue
        polarity = _comparison_polarity(node)
        for column in node.find_all(exp.Column):
            logical = physical_to_logical.get(column.name)
            if not logical:
                continue
            for literal in node.find_all(exp.Literal):
                output.add((logical, polarity, _operand_hash(logical, literal.this)))
    return output


def _sql_comparison_keys(
    tree: exp.Expression,
    ir: Mapping[str, Any],
) -> set[tuple[str, str, str]]:
    physical_to_logical = {
        _physical_column(ir, logical): logical
        for logical in (ir.get("resolved_fields") or {})
    }
    node_operators: tuple[tuple[type[exp.Expression], str], ...] = (
        (exp.EQ, "eq"), (exp.NEQ, "neq"), (exp.GT, "gt"),
        (exp.GTE, "gte"), (exp.LT, "lt"), (exp.LTE, "lte"),
    )
    output: set[tuple[str, str, str]] = set()
    for node_type, operator in node_operators:
        for node in tree.find_all(node_type):
            for column in node.find_all(exp.Column):
                logical = physical_to_logical.get(column.name)
                if not logical:
                    continue
                for literal in node.find_all(exp.Literal):
                    output.add((
                        logical,
                        operator,
                        _operand_hash(logical, literal.this),
                    ))
    return output


def audit_compiled_semantic_coverage(
    query: str,
    table_infos: Sequence[Mapping[str, Any]] | None,
    sql: str,
) -> dict[str, Any]:
    """Verify an emitted SQL candidate against bounded question-derived slots."""

    ir = _coverage_ir(query, table_infos)
    detected, dimensions, measures, filters = _ir_slots(ir)
    cue_slots, cue_dimensions, cue_measures = _question_cue_slots(
        query, table_infos,
    )
    detected.extend(cue_slots)
    dimensions = list(dict.fromkeys([*dimensions, *cue_dimensions]))
    measure_index = {
        (
            str(item.get("logical") or ""),
            str(item.get("function") or "").upper(),
            bool(item.get("distinct")),
        ): dict(item)
        for item in [*measures, *cue_measures]
    }
    measures = list(measure_index.values())
    detected.extend(_unregistered_slots(query))
    represented: list[str] = []
    try:
        tree = sqlglot.parse_one(str(sql or ""), read="mysql")
    except Exception:
        payload = _payload(
            query=query, detected=detected, represented=[], dimensions=dimensions,
            measures=measures, filters=filters,
        )
        payload["audit_source"] = "compiled_sql_parse_failed"
        return payload

    registered_contract_valid = False
    if ir is not None and (ir.get("compiler") or {}).get("supported"):
        try:
            validate_compiled_semantic_ir(sql, ir)
            registered_contract_valid = True
        except Exception:
            registered_contract_valid = False

    grouped_or_selected = {
        column.name
        for node_type in (exp.Group, exp.Select)
        for node in tree.find_all(node_type)
        for column in node.find_all(exp.Column)
    }
    predicate_columns = {
        column.name
        for node_type in (exp.Where, exp.Having)
        for node in tree.find_all(node_type)
        for column in node.find_all(exp.Column)
    }
    aggregates = list(tree.find_all(exp.AggFunc))
    operands = _sql_operand_keys(tree, ir or {})
    comparisons = _sql_comparison_keys(tree, ir or {})
    having_node = tree.find(exp.Having)
    having_comparisons = (
        _sql_comparison_keys(having_node, ir or {})
        if having_node is not None else set()
    )
    top_select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    top_level_scalar_count = bool(
        top_select is not None
        and top_select.args.get("group") is None
        and any(expression.find(exp.Count) for expression in top_select.expressions)
    )
    percent_rank_type = getattr(exp, "PercentRank", None)
    has_percent_rank = bool(
        percent_rank_type is not None and tree.find(percent_rank_type)
    )
    has_partitioned_window = any(
        window.args.get("partition_by")
        for window in tree.find_all(exp.Window)
    )
    has_negative_shape = bool(
        tree.find(exp.NEQ)
        or tree.find(exp.Not)
        or (
            any(str(join.args.get("side") or "").upper() == "LEFT" for join in tree.find_all(exp.Join))
            and tree.find(exp.Is)
        )
    )

    for slot in detected:
        if slot.startswith("dimension:"):
            logical = slot.split(":", 1)[1]
            if _physical_column(ir or {}, logical) in grouped_or_selected:
                represented.append(slot)
        elif slot.startswith("filter:"):
            logical = slot.split(":", 1)[1]
            if _physical_column(ir or {}, logical) in predicate_columns:
                represented.append(slot)
        elif slot.startswith("filteroperand:"):
            _, logical, polarity, digest = slot.split(":", 3)
            if (logical, polarity, digest) in operands or (
                polarity == "negative"
                and has_negative_shape
                and (logical, "positive", digest) in operands
            ):
                represented.append(slot)
        elif slot.startswith("filterop:"):
            if has_negative_shape:
                represented.append(slot)
        elif slot.startswith("filtercard:"):
            logical = slot.split(":", 2)[1]
            count = sum(1 for item in operands if item[0] == logical)
            if count > 1:
                represented.append(slot)
        elif slot.startswith("comparison:"):
            _, scope, logical, operator, digest = slot.split(":", 4)
            evidence = having_comparisons if scope == "having" else comparisons
            if (logical, operator, digest) in evidence:
                represented.append(slot)
        elif slot.startswith("grain:"):
            logical = slot.split(":", 1)[1]
            if _physical_column(ir or {}, logical) in grouped_or_selected:
                represented.append(slot)
        elif slot == "output:scalar":
            if top_level_scalar_count:
                represented.append(slot)
        elif slot.startswith("measure:"):
            _, function, logical, distinct = slot.split(":", 3)
            if function in {
                "ratio", "count_distinct", "count_distinct_sold",
                "count_distinct_unsold",
            } and registered_contract_valid:
                represented.append(slot)
                continue
            physical = _physical_column(ir or {}, logical)
            for aggregate in aggregates:
                if str(aggregate.sql_name()).lower() != function:
                    continue
                if physical not in {column.name for column in aggregate.find_all(exp.Column)}:
                    continue
                if distinct == "distinct" and aggregate.find(exp.Distinct) is None:
                    continue
                represented.append(slot)
                break
        elif slot == "operator:conditional_aggregate":
            if any(
                aggregate.find(exp.Case) is not None
                for aggregate in aggregates
            ):
                represented.append(slot)
        elif slot == "operator:having":
            if having_node is not None:
                represented.append(slot)
        elif slot == "operator:count_qualified_groups":
            if (
                top_level_scalar_count
                and tree.find(exp.Subquery) is not None
                and having_node is not None
            ):
                represented.append(slot)
        elif slot == "operator:percentile_rank":
            if has_percent_rank:
                represented.append(slot)
        elif slot == "operator:partitioned_rank":
            if has_percent_rank and has_partitioned_window:
                represented.append(slot)
        elif slot.startswith("operator:") and not slot.startswith("operator:unregistered_"):
            if registered_contract_valid:
                represented.append(slot)

    payload = _payload(
        query=query,
        detected=detected,
        represented=represented,
        dimensions=dimensions,
        measures=measures,
        filters=filters,
        compiled=True,
    )
    payload["audit_source"] = "bounded_compiled_sql_ast"
    return payload
