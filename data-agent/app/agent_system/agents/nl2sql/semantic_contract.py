"""Unified semantic contract shared by retrieval, planning and SQL generation.

Historically each stage reconstructed a different view of the question: column
retrieval cared about fields, planning emitted prose, and SQL validation cared
about AST shape.  That split is especially harmful for ratios and nested
aggregates because the visible grain, inner grain and denominator scope are
different concepts.  This module creates one bounded, privacy-safe contract
from live metadata, registered metrics and the existing semantic IR.

The contract is guidance when semantic coverage is partial and authoritative
when a compiler has proved full coverage.  It never imports evaluation data or
reference-project code.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

import sqlglot
from sqlglot import exp

from app.agent_system.agents.nl2sql.relation_graph import (
    close_relation_requirements,
    compact_relation_closure,
)
from app.agent_system.agents.nl2sql.semantic_coverage import detect_semantic_coverage


SEMANTIC_QUERY_CONTRACT_VERSION = "semantic-query-contract-v1"


def _items(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _normalized(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def _split_ref(value: Any) -> tuple[str, str] | None:
    parts = str(value or "").strip().split(".", 1)
    return (parts[0], parts[1]) if len(parts) == 2 and all(parts) else None


def _metadata_index(
    table_infos: Sequence[Mapping[str, Any]] | None,
) -> tuple[
    dict[tuple[str, str], tuple[str, str]],
    dict[str, list[tuple[str, str]]],
]:
    exact: dict[tuple[str, str], tuple[str, str]] = {}
    by_column: dict[str, list[tuple[str, str]]] = {}
    for table in table_infos or []:
        table_name = str(table.get("name") or "").strip()
        if not table_name:
            continue
        for column in table.get("columns") or []:
            column_name = str(column.get("name") or "").strip()
            if not column_name:
                continue
            ref = (table_name, column_name)
            exact[(table_name.lower(), column_name.lower())] = ref
            by_column.setdefault(column_name.lower(), []).append(ref)
    return exact, by_column


def _resolve_ref(
    table: Any,
    column: Any,
    exact: Mapping[tuple[str, str], tuple[str, str]],
    by_column: Mapping[str, Sequence[tuple[str, str]]],
) -> tuple[str, str] | None:
    table_name = str(table or "").lower()
    column_name = str(column or "").lower()
    if table_name and column_name and (table_name, column_name) in exact:
        return exact[(table_name, column_name)]
    candidates = list(by_column.get(column_name, ()))
    return candidates[0] if len(candidates) == 1 else None


def _logical_ref(
    logical: str,
    exact: Mapping[tuple[str, str], tuple[str, str]],
    by_column: Mapping[str, Sequence[tuple[str, str]]],
) -> tuple[str, str] | None:
    candidates = [logical]
    if logical.endswith("_key"):
        candidates.append(f"{logical[:-4]}_id")
    for candidate in candidates:
        rows = list(by_column.get(candidate.lower(), ()))
        if len(rows) == 1:
            return rows[0]
        # A fact FK and a dimension PK can share one logical name.  Prefer the
        # dimension-side field for a visible grain, but do not guess between
        # two same-role candidates.
        dimension_rows = [
            row for row in rows if row[0].lower().startswith("dim_")
        ]
        if len(dimension_rows) == 1:
            return dimension_rows[0]
    return None


def _direct_metric(
    query: str,
    metric: Mapping[str, Any],
) -> bool:
    q = _normalized(query)
    return any(
        len(term) >= 2 and term in q
        for term in (
            _normalized(metric.get("name")),
            *(_normalized(item) for item in (metric.get("alias") or [])),
        )
        if term
    )


def _metric_contract(
    metric: Mapping[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    rule = dict(metric.get("formula_rule") or {})
    rule_type = str(rule.get("type") or "definition").lower()
    relevant = [str(item) for item in (metric.get("relevant_columns") or [])]
    columns = [
        value for value in (
            rule.get("column"), rule.get("entity_column"),
            rule.get("event_column"), rule.get("numerator_column"),
            rule.get("denominator_column"), rule.get("universe_column"),
            rule.get("period_column"), rule.get("partition_column"),
            rule.get("alignment_column"), *relevant,
        )
        if _split_ref(value)
    ]
    output: dict[str, Any] = {
        "name": str(metric.get("name") or ""),
        "source": source,
        "type": rule_type,
        "columns": list(dict.fromkeys(str(item) for item in columns)),
    }
    for key in (
        "function", "distinct", "dimension", "direction", "default_limit",
        "result_alias", "entity_column", "event_column", "repeat_threshold",
        "event_selector", "numerator", "denominator", "numerator_scope",
        "denominator_scope", "population_scope", "partition_scope",
        "zero_denominator", "observation_window", "inactivity_window",
        "source_table", "column", "numerator_column", "denominator_column",
        "universe_column", "period_column", "partition_column",
        "alignment_column", "anti_join_strategy",
        "include_zero_event_entities",
    ):
        if rule.get(key) is not None:
            output[key] = rule[key]
    if rule_type == "repeat_rate":
        output.setdefault("numerator", "entities_with_event_count_gte_threshold")
        output.setdefault("denominator", "entities_with_events_in_same_filtered_scope")
        output.setdefault("population_scope", "filtered_event_population")
        output.setdefault("zero_denominator", "null")
    elif rule_type in {"share", "activation_rate"}:
        output.setdefault("denominator_scope", "same_base_filters_before_group_limit")
        output.setdefault("zero_denominator", "null")
    elif rule_type in {"period_growth", "growth_rate"}:
        output.setdefault("numerator", "current_period_minus_previous_period")
        output.setdefault("denominator", "previous_period")
        output.setdefault("zero_denominator", "null")
    return output


def _ir_metric_contracts(
    semantic_ir: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for measure in semantic_ir.get("measures") or []:
        if not isinstance(measure, Mapping):
            continue
        ref = None
        if measure.get("table") and measure.get("column"):
            ref = f"{measure['table']}.{measure['column']}"
        row = {
            "name": str(
                measure.get("alias") or measure.get("logical") or "measure"
            ),
            "source": "semantic_ir",
            "type": "aggregate",
            "function": str(measure.get("function") or "").upper(),
            "distinct": bool(measure.get("distinct")),
            "columns": [ref] if ref else [],
        }
        output.append(row)
    return output


def _operator_contracts(
    operations: Sequence[str],
    semantic_slots: Mapping[str, Any],
    output_grain: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    operation_set = set(operations)
    partition_keys = list(semantic_slots.get("partition_keys") or [])
    ratio = bool(operation_set & {
        "share", "partitioned_share", "repeat_rate", "activation_rate",
        "per_entity_ratio", "period_growth",
    })
    denominator_scope = None
    if "repeat_rate" in operation_set:
        denominator_scope = "entities_with_events_after_shared_base_filters"
    elif "partitioned_share" in operation_set:
        denominator_scope = "all_rows_in_each_partition_after_shared_base_filters"
    elif "share" in operation_set:
        denominator_scope = "all_rows_after_shared_base_filters_before_rank_or_limit"
    elif "per_entity_ratio" in operation_set:
        denominator_scope = "distinct_entities_after_shared_base_filters"
    elif "average_order_value" in operation_set:
        denominator_scope = "registered_metric_definition"
    elif "period_growth" in operation_set:
        denominator_scope = "previous_comparable_period"
    return {
        "ratio_required": ratio,
        "denominator_scope": denominator_scope,
        "partition_keys": partition_keys,
        "visible_grain": [
            str(item.get("logical") or item.get("column") or "")
            for item in output_grain
            if item.get("logical") or item.get("column")
        ],
        "zero_denominator": (
            semantic_slots.get("zero_denominator") or "null"
            if ratio else None
        ),
    }


def _stage_plan(
    operations: Sequence[str],
    output_grain: Sequence[Mapping[str, Any]],
    semantic_slots: Mapping[str, Any],
) -> list[dict[str, Any]]:
    ops = set(operations)
    stages: list[dict[str, Any]] = [{
        "order": 1,
        "stage": "base_scope",
        "grain": "authorized_fact_rows",
        "contract": "apply shared scalar/time filters before aggregation",
    }]

    def add(stage: str, grain: Any, contract: str) -> None:
        stages.append({
            "order": len(stages) + 1,
            "stage": stage,
            "grain": grain,
            "contract": contract,
        })

    if ops & {"cohort", "set", "anti", "repeat_rate", "repeat_customer_set"}:
        add(
            "entity_qualification",
            semantic_slots.get("entity_key") or "business_entity",
            "derive one qualifying row/flag per entity without changing the requested population",
        )
    if ops & {
        "nested_monthly_average", "above_average", "multi_having",
        "entity_aggregate_average",
        "per_entity_ratio", "average_order_value", "share",
        "partitioned_share", "top_per_group", "bounded_rank",
        "running_sum", "period_growth", "period_delta", "dual_rank",
    }:
        inner_grain = [
            *list(semantic_slots.get("partition_keys") or []),
            semantic_slots.get("rank_dimension"),
            semantic_slots.get("calendar_grain"),
            semantic_slots.get("aggregate_grain"),
        ]
        add(
            "inner_aggregate",
            [item for item in inner_grain if item] or "metric_base_grain",
            "aggregate once at the finest required semantic grain",
        )
    if ops & {
        "top_per_group", "bounded_rank", "running_sum", "period_growth",
        "period_delta", "dual_rank", "partitioned_share",
        "cumulative_rank_share", "cumulative_share_threshold",
    }:
        add(
            "window_or_comparison",
            list(semantic_slots.get("partition_keys") or []),
            "compute window/rank/delta only after the inner aggregate and keep partition scope explicit",
        )
    if ops & {
        "top_per_group", "bounded_rank", "having", "multi_having",
        "above_average", "entity_aggregate_average", "cumulative_share_threshold",
    }:
        add(
            "qualification",
            "derived_rows",
            "filter aggregate/window results in an outer scope, never in the base WHERE",
        )
    add(
        "final_projection",
        [
            str(item.get("logical") or item.get("column") or "")
            for item in output_grain
            if item.get("logical") or item.get("column")
        ] or "scalar",
        "return exactly the requested visible grain and requested measures",
    )
    return stages


def build_query_contract(
    query: str,
    table_infos: Sequence[Mapping[str, Any]] | None,
    metric_infos: Sequence[Mapping[str, Any]] | None = None,
    semantic_ir: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a bounded semantic contract from current, authorized evidence."""

    ir = dict(semantic_ir or {})
    model_routed = bool(
        isinstance(ir.get("model_sql_policy"), Mapping)
        and ir["model_sql_policy"].get("use_model")
    )
    # A model route is an explicit abstention by the deterministic compiler.
    # Its parsed grain/measure hints may still be useful to the model, but they
    # must not become an authoritative validation contract.  In particular,
    # using the same parser both to build and to certify the IR creates a
    # circular proof: a missed user requirement is invisible to both passes.
    # Keep independently registered metrics and relationship evidence, while
    # treating compiler-derived output shape as advisory outside this contract.
    ir_contract_trusted = not model_routed
    raw_coverage = dict(
        ir.get("semantic_coverage")
        or detect_semantic_coverage(query, table_infos)
    )
    if model_routed:
        # ``semantic_coverage`` is produced by the same bounded parser as the
        # IR.  Reusing its slots in a guided contract would let a parser error
        # regain hard validation authority (for example, a spurious ``share``
        # slot forcing a ratio).  Keep only the privacy-safe question identity;
        # direct metric-name matches and metadata relations below are the
        # independently grounded parts of a model-route contract.
        coverage = {
            "question_fingerprint": raw_coverage.get("question_fingerprint"),
            "detected_slots": [],
            "unresolved_slots": [],
            "dimensions": [],
            "measures": [],
            "filters": [],
            "audit_source": "model_route_independent_contract",
        }
    else:
        coverage = raw_coverage
    exact, by_column = _metadata_index(table_infos)

    output_grain: list[dict[str, Any]] = []
    semantic_slots = dict(ir.get("semantic_slots") or {})
    declared_outputs = [
        str(item).lower()
        for item in (semantic_slots.get("output_columns") or [])
        if str(item).strip()
    ]
    for dimension in (ir.get("dimensions") or []) if ir_contract_trusted else []:
        if not isinstance(dimension, Mapping):
            continue
        logical = str(dimension.get("logical") or dimension.get("column") or "")
        ref = _resolve_ref(
            dimension.get("table"), dimension.get("column"), exact, by_column
        )
        if ref:
            output_name = ref[1]
            if declared_outputs and output_name.lower() not in declared_outputs:
                derived = next(
                    (
                        name for name in declared_outputs
                        if logical and (
                            name.endswith(f"_{logical.lower()}")
                            or name.startswith(f"{logical.lower()}_")
                        )
                    ),
                    None,
                )
                if derived is None:
                    # The physical dimension is an intermediate/filter grain,
                    # not part of the final visible result.
                    continue
                output_name = derived
            output_grain.append({
                "logical": logical or ref[1],
                "table": ref[0],
                "column": output_name,
                "source_column": ref[1],
                "derived": output_name.lower() != ref[1].lower(),
                "visibility": "output",
            })
        elif logical and (
            not declared_outputs or logical.lower() in declared_outputs
        ):
            output_grain.append({
                "logical": logical,
                "column": logical,
                "derived": True,
                "visibility": "output",
            })
    # Only fall back to question-derived dimensions when no IR owns the output
    # shape.  A closed scalar IR with ``dimensions=[]`` must not accidentally
    # promote a time filter (for example a year range) into final GROUP BY.
    if not output_grain and ir_contract_trusted and not (
        ir.get("semantic_family") is not None and "dimensions" in ir
    ):
        for logical in coverage.get("dimensions") or []:
            ref = _logical_ref(str(logical), exact, by_column)
            if ref:
                output_grain.append({
                    "logical": str(logical),
                    "table": ref[0],
                    "column": ref[1],
                    "source_column": ref[1],
                    "derived": False,
                    "visibility": "output",
                })
    output_grain = list({
        (
            str(item.get("table") or "derived").lower(),
            str(item.get("column") or "").lower(),
        ): item
        for item in output_grain
    }.values())

    metric_contracts = _ir_metric_contracts(ir) if ir_contract_trusted else []
    selected_metric_names = {
        str((ir.get("metric") or {}).get("name") or "").lower()
    }
    for metric in metric_infos or []:
        if not isinstance(metric, Mapping):
            continue
        direct = _direct_metric(query, metric)
        # A model-routed query deliberately treats the parser/IR as advisory.
        # Do not let an IR-selected registry entry regain formula authority via
        # this secondary path: only a metric named directly in the user's
        # question is independent evidence.  Closed deterministic contracts
        # retain the existing IR selection behaviour for ablation/compiler use.
        selected_by_ir = (
            ir_contract_trusted
            and str(metric.get("name") or "").lower() in selected_metric_names
        )
        if direct or selected_by_ir:
            metric_contracts.append(
                _metric_contract(
                    metric,
                    source="registered_metric_direct" if direct else "registered_metric_ir",
                )
            )
    unique_metrics: dict[tuple[str, str, str], dict[str, Any]] = {}
    for metric in metric_contracts:
        key = (
            str(metric.get("name") or "").lower(),
            str(metric.get("type") or "").lower(),
            ",".join(str(item).lower() for item in metric.get("columns") or []),
        )
        unique_metrics[key] = metric
    metric_contracts = list(unique_metrics.values())

    required_columns: dict[str, set[str]] = {}
    if ir_contract_trusted:
        for table, columns in dict(ir.get("required_columns") or {}).items():
            required_columns.setdefault(str(table), set()).update(
                str(column) for column in columns or []
            )
    for item in output_grain:
        if item.get("table") and item.get("source_column"):
            required_columns.setdefault(str(item["table"]), set()).add(
                str(item["source_column"])
            )
    for metric in metric_contracts:
        for raw_ref in metric.get("columns") or []:
            ref = _split_ref(raw_ref)
            if ref:
                required_columns.setdefault(ref[0], set()).add(ref[1])
    relation_plan = close_relation_requirements(
        table_infos,
        required_columns,
    )

    detected_slots = [str(item) for item in coverage.get("detected_slots") or []]
    operations = list(dict.fromkeys(
        item.split(":", 1)[1]
        for item in detected_slots
        if item.startswith("operator:") and ":" in item
    ))
    metric_operation_map = {
        "repeat_rate": "repeat_rate",
        "share": "share",
        "activation_rate": "activation_rate",
        "period_growth": "period_growth",
        "growth_rate": "period_growth",
        "cohort_count": "cohort",
        "inactivity_cohort": "cohort",
        "anti_join": "anti",
        "ranking": "bounded_rank",
    }
    operations = [
        item
        for item in dict.fromkeys([
            *operations,
            *(
                metric_operation_map.get(
                    str(metric.get("type") or "").lower()
                )
                for metric in metric_contracts
            ),
        ])
        if item
    ]
    operator_contract = _operator_contracts(
        operations, semantic_slots, output_grain
    )
    compiler = dict(ir.get("compiler") or {})
    unresolved_slots = [
        str(item) for item in coverage.get("unresolved_slots") or []
    ]
    closed = bool(
        ir_contract_trusted
        and not model_routed
        and compiler.get("supported")
        and not unresolved_slots
        and relation_plan.get("status") == "closed"
    )
    confidence = (
        1.0 if closed
        else 0.82 if detected_slots and relation_plan.get("status") == "closed"
        else 0.65
    )
    payload: dict[str, Any] = {
        "schema_version": SEMANTIC_QUERY_CONTRACT_VERSION,
        "semantic_family": str(
            ir.get("semantic_family") or ir.get("intent") or "unknown"
        ),
        "authority": "closed" if closed else "guided",
        "confidence": confidence,
        "question_fingerprint": coverage.get("question_fingerprint"),
        "output_grain": output_grain,
        "metric_contracts": metric_contracts,
        "filter_contract": {
            "fields": list(coverage.get("filters") or []),
            "shared_base_scope": True,
            "filter_before_aggregation": True,
            "filter_values_repeated": False,
        },
        "operator_contract": operator_contract,
        "operations": operations,
        "stage_plan": _stage_plan(operations, output_grain, semantic_slots),
        "relation_plan": relation_plan,
        "required_tables": list(relation_plan.get("required_columns") or {}),
        "unresolved_slots": unresolved_slots,
        "raw_question_persisted": False,
        "filter_values_persisted": False,
    }
    digest_source = {
        key: payload[key]
        for key in (
            "schema_version", "semantic_family", "authority", "output_grain",
            "metric_contracts", "filter_contract", "operator_contract",
            "operations", "stage_plan", "relation_plan", "unresolved_slots",
        )
    }
    payload["contract_hash"] = hashlib.sha256(
        json.dumps(
            digest_source,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return payload


def compact_query_contract(
    contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the model/trace-safe subset of a semantic query contract."""

    value = dict(contract or {})
    relation = dict(value.get("relation_plan") or {})
    return {
        "schema_version": value.get("schema_version"),
        "semantic_family": value.get("semantic_family"),
        "authority": value.get("authority"),
        "confidence": value.get("confidence"),
        "contract_hash": value.get("contract_hash"),
        "output_grain": _items(value.get("output_grain"))[:12],
        "metric_contracts": _items(value.get("metric_contracts"))[:12],
        "filter_contract": dict(value.get("filter_contract") or {}),
        "operator_contract": dict(value.get("operator_contract") or {}),
        "operations": [str(item) for item in _items(value.get("operations"))[:16]],
        "stage_plan": _items(value.get("stage_plan"))[:10],
        "relation_plan": compact_relation_closure(relation),
        "unresolved_slots": [
            str(item)[:120] for item in _items(value.get("unresolved_slots"))[:16]
        ],
        "raw_question_persisted": False,
        "filter_values_persisted": False,
    }


def _render_metric_formula(metric: Mapping[str, Any]) -> str:
    """Render one registered rule as an explicit grain/population equation."""

    name = str(metric.get("name") or "metric")
    rule_type = str(metric.get("type") or "definition").lower()
    columns = list(metric.get("columns") or [])
    column = str(metric.get("column") or (columns[0] if columns else "registered_column"))
    function = str(metric.get("function") or "").upper()
    if rule_type == "distinct_count":
        return f"{name}=COUNT(DISTINCT {column})"
    if rule_type == "share":
        return (
            f"{name}={function or 'SUM'}({column}) / 同一基础过滤下全体分组总量"
            f"（分母作用域={metric.get('denominator_scope')}，先于排名/LIMIT）"
        )
    if rule_type == "activation_rate":
        return (
            f"{name}=COUNT(DISTINCT {metric.get('numerator_column')}) / "
            f"COUNT(DISTINCT {metric.get('denominator_column')})"
            f"（分母作用域={metric.get('denominator_scope')}）"
        )
    if rule_type == "repeat_rate":
        return (
            f"{name}=事件数>={metric.get('repeat_threshold', 2)}的去重实体数 / "
            "同一过滤范围内有事件的全部去重实体数"
        )
    if rule_type == "period_growth":
        return (
            f"{name}=(本期{function or 'SUM'}({column})-可比上期值) / "
            "NULLIF(可比上期值,0)"
        )
    if rule_type == "cohort_count":
        return (
            f"{name}=先按{metric.get('entity_column')}求"
            f"{metric.get('event_selector', 'MIN')}({metric.get('event_column')})，"
            "再按该首次/末次事件落入观察期筛选并计数"
        )
    if rule_type == "anti_join":
        return (
            f"{name}=以{metric.get('universe_column')}为完整总体，对"
            f"{metric.get('event_column')}执行NOT EXISTS或LEFT JOIN IS NULL"
        )
    if rule_type == "inactivity_cohort":
        return (
            f"{name}=按{metric.get('entity_column')}求MAX({metric.get('event_column')})，"
            "再使用用户给定的不活跃窗口判定"
        )
    if function:
        distinct = "DISTINCT " if metric.get("distinct") else ""
        return f"{name}={function}({distinct}{column})"
    return f"{name}={rule_type}({','.join(columns) or 'registered definition'})"


def render_query_contract(contract: Mapping[str, Any] | None) -> str:
    """Render the compact contract as short, imperative SQL guidance."""

    value = compact_query_contract(contract)
    lines = [
        f"合同版本 {value.get('schema_version')}；权威级别 {value.get('authority')}。"
    ]
    grain = value.get("output_grain") or []
    if grain:
        lines.append(
            "最终可见粒度只能是："
            + "、".join(
                (
                    f"{item.get('table')}.{item.get('source_column')}"
                    f"→{item.get('column')}"
                    if item.get("derived") and item.get("table")
                    else str(item.get("column"))
                    if item.get("derived")
                    else f"{item.get('table')}.{item.get('column')}"
                )
                for item in grain
            )
            + "；过滤维度、分母实体和中间粒度不得泄漏到最终 GROUP BY。"
        )
    metrics = value.get("metric_contracts") or []
    if metrics:
        lines.append(
            "指标口径："
            + "；".join(_render_metric_formula(metric) for metric in metrics)
            + "。"
        )
    operator_contract = value.get("operator_contract") or {}
    if operator_contract.get("ratio_required"):
        lines.append(
            "比例必须使用 NULLIF 保护分母；分母作用域="
            + str(operator_contract.get("denominator_scope"))
            + "，且排名/LIMIT 不得提前缩小分母。"
        )
    joins = (value.get("relation_plan") or {}).get("joins") or []
    if joins:
        lines.append(
            "关系闭包："
            + "；".join(
                f"{item.get('left_table')}.{item.get('left_column')}="
                f"{item.get('right_table')}.{item.get('right_column')}"
                for item in joins
            )
            + "；禁止省略桥接表或猜测连接。"
        )
    stages = value.get("stage_plan") or []
    if stages:
        lines.append(
            "计算阶段必须按顺序执行："
            + " → ".join(str(item.get("stage")) for item in stages)
            + "。"
        )
    unresolved = value.get("unresolved_slots") or []
    if unresolved:
        lines.append(
            "未闭合槽位：" + "、".join(unresolved)
            + "；不得静默降级成更简单的全局聚合。"
        )
    return "\n".join(lines)


def _nearest_select(node: exp.Expression | None) -> exp.Select | None:
    """Return the SQL scope that owns ``node`` without crossing a subquery."""

    current = node
    while current is not None:
        if isinstance(current, exp.Select):
            return current
        current = current.parent
    return None


def _direct_source_aliases(select: exp.Select | None) -> list[str]:
    """Return only FROM/JOIN aliases that belong to one SELECT scope."""

    if select is None:
        return []
    relations: list[exp.Expression] = []
    from_clause = select.args.get("from_")
    if isinstance(from_clause, exp.From) and from_clause.this is not None:
        relations.append(from_clause.this)
    relations.extend(
        join.this
        for join in (select.args.get("joins") or [])
        if isinstance(join, exp.Join) and join.this is not None
    )
    return list(dict.fromkeys(
        str(relation.alias_or_name or "").lower()
        for relation in relations
        if relation.alias_or_name
    ))


def _build_sql_column_lineage(
    tree: exp.Expression,
    cte_names: set[str],
) -> tuple[
    dict[str, set[str]],
    dict[str, dict[str, set[tuple[str, str]]]],
    dict[str, set[tuple[str, str]]],
]:
    """Resolve projected CTE/subquery columns back to physical table columns.

    The query-contract audit needs physical relationship evidence even when a
    valid SQL aggregates a fact table first and joins the derived entity key to
    a dimension later.  SQLGlot exposes the AST but does not attach lineage, so
    this bounded resolver follows direct projections through named CTEs and
    subqueries.  It deliberately does not infer arithmetic-expression lineage
    as a join key.
    """

    physical_aliases: dict[str, set[str]] = {}
    cte_references: list[tuple[str, str]] = []
    for table in tree.find_all(exp.Table):
        table_name = str(table.name or "").lower()
        alias = str(table.alias_or_name or table.name or "").lower()
        if not table_name or not alias:
            continue
        if table_name in cte_names:
            cte_references.append((alias, table_name))
            continue
        physical_aliases.setdefault(alias, set()).add(table_name)
        physical_aliases.setdefault(table_name, set()).add(table_name)

    derived_selects: dict[str, exp.Select] = {}
    for cte in tree.find_all(exp.CTE):
        alias = str(cte.alias_or_name or "").lower()
        selected = cte.this if isinstance(cte.this, exp.Select) else cte.this.find(exp.Select)
        if alias and selected is not None:
            derived_selects[alias] = selected
    for subquery in tree.find_all(exp.Subquery):
        alias = str(subquery.alias_or_name or "").lower()
        selected = (
            subquery.this
            if isinstance(subquery.this, exp.Select)
            else subquery.this.find(exp.Select)
        )
        if alias and selected is not None:
            derived_selects[alias] = selected

    lineage: dict[str, dict[str, set[tuple[str, str]]]] = {}
    unique_entity_refs: dict[str, set[tuple[str, str]]] = {}

    def resolve_column(
        column: exp.Column,
        owner: exp.Select | None,
    ) -> set[tuple[str, str]]:
        name = str(column.name or "").lower()
        qualifier = str(column.table or "").lower()
        if not name:
            return set()
        if qualifier:
            output = {
                (table_name, name)
                for table_name in physical_aliases.get(qualifier, set())
            }
            output.update(
                lineage.get(qualifier, {}).get(name, set())
            )
            return output
        sources = _direct_source_aliases(owner)
        possible: set[tuple[str, str]] = set()
        for source in sources:
            possible.update(
                (table_name, name)
                for table_name in physical_aliases.get(source, set())
            )
            possible.update(lineage.get(source, {}).get(name, set()))
        # An unqualified column is safe to attribute only when its SELECT has
        # one source or all possible paths converge on the same physical ref.
        if len(sources) == 1 or len(possible) == 1:
            return possible
        return set()

    # CTEs can reference earlier CTEs, so converge over a small, bounded number
    # of passes instead of relying on declaration order.
    for _ in range(max(2, len(derived_selects) + 1)):
        changed = False
        for alias, selected in derived_selects.items():
            projected: dict[str, set[tuple[str, str]]] = {}
            for projection in selected.expressions:
                output_name = str(projection.alias_or_name or "").lower()
                if not output_name:
                    continue
                origins: set[tuple[str, str]] = set()
                columns = list(projection.find_all(exp.Column))
                if isinstance(projection, exp.Column):
                    columns.insert(0, projection)
                for column in columns:
                    origins.update(resolve_column(column, selected))
                if origins:
                    projected[output_name] = origins
            if projected and projected != lineage.get(alias):
                lineage[alias] = projected
                changed = True

            group = selected.args.get("group")
            group_origins: set[tuple[str, str]] = set()
            if isinstance(group, exp.Group):
                for expression in group.expressions:
                    columns = list(expression.find_all(exp.Column))
                    if isinstance(expression, exp.Column):
                        columns.insert(0, expression)
                    for column in columns:
                        group_origins.update(resolve_column(column, selected))
            if not group_origins and selected.args.get("distinct"):
                projected_origins = set().union(*projected.values()) if projected else set()
                if len(projected) == 1:
                    group_origins = projected_origins
            # COUNT(*) over a derived relation represents a physical entity
            # count only when that relation proves one-row-per-single-key.
            unique = group_origins if len(group_origins) == 1 else set()
            if unique != unique_entity_refs.get(alias, set()):
                unique_entity_refs[alias] = unique
                changed = True

        for alias, cte_name in cte_references:
            inherited = lineage.get(cte_name)
            if inherited and inherited != lineage.get(alias):
                lineage[alias] = {
                    key: set(value) for key, value in inherited.items()
                }
                changed = True
            inherited_unique = unique_entity_refs.get(cte_name, set())
            if inherited_unique != unique_entity_refs.get(alias, set()):
                unique_entity_refs[alias] = set(inherited_unique)
                changed = True
        if not changed:
            break

    return physical_aliases, lineage, unique_entity_refs


def _column_origins(
    column: exp.Column,
    physical_aliases: Mapping[str, set[str]],
    lineage: Mapping[str, Mapping[str, set[tuple[str, str]]]],
) -> set[tuple[str, str]]:
    """Resolve one candidate column in its own SELECT scope."""

    name = str(column.name or "").lower()
    qualifier = str(column.table or "").lower()
    if not name:
        return set()
    if qualifier:
        output = {
            (table_name, name)
            for table_name in physical_aliases.get(qualifier, set())
        }
        output.update(lineage.get(qualifier, {}).get(name, set()))
        return output
    owner = _nearest_select(column)
    sources = _direct_source_aliases(owner)
    output: set[tuple[str, str]] = set()
    for source in sources:
        output.update(
            (table_name, name)
            for table_name in physical_aliases.get(source, set())
        )
        output.update(lineage.get(source, {}).get(name, set()))
    return output if len(sources) == 1 or len(output) == 1 else set()


def _count_star_unique_origins(
    aggregate: exp.AggFunc,
    unique_entity_refs: Mapping[str, set[tuple[str, str]]],
) -> set[tuple[str, str]]:
    """Return proven physical keys represented by COUNT(*) in this scope."""

    if not isinstance(aggregate, exp.Count):
        return set()
    if any(True for _ in aggregate.find_all(exp.Column)):
        return set()
    sources = _direct_source_aliases(_nearest_select(aggregate))
    if len(sources) != 1:
        return set()
    return set(unique_entity_refs.get(sources[0], set()))


def _repeat_threshold_matches(
    expression: exp.Expression,
    count_aliases: set[str],
    threshold: int,
) -> bool:
    """Recognize ``event_count >= N`` and its algebraic ``> N-1`` form."""

    comparisons = (
        (exp.GTE, threshold),
        (exp.GT, threshold - 1),
    )
    for node_type, expected in comparisons:
        for comparison in expression.find_all(node_type):
            left = comparison.this
            right = comparison.expression
            left_is_count = isinstance(left, exp.Count) or any(
                str(column.name or "").lower() in count_aliases
                for column in left.find_all(exp.Column)
            )
            if not left_is_count or not isinstance(right, exp.Literal):
                continue
            try:
                if float(right.this) == float(expected):
                    return True
            except (TypeError, ValueError):
                continue
    return False


def _has_repeat_population_shape(
    tree: exp.Expression,
    threshold: int,
) -> bool:
    """Prove the two-stage population contract of a repeat-rate SQL."""

    count_aliases: set[str] = set()
    entity_count_stages: list[exp.Select] = []
    for selected in tree.find_all(exp.Select):
        if selected.args.get("group") is None:
            continue
        aliases = {
            str(projection.alias_or_name or "").lower()
            for projection in selected.expressions
            if projection.find(exp.Count) is not None
        }
        if aliases:
            count_aliases.update(aliases)
            entity_count_stages.append(selected)
    if not entity_count_stages:
        return False

    having_threshold = any(
        _repeat_threshold_matches(having, count_aliases, threshold)
        for having in tree.find_all(exp.Having)
    )
    for division in tree.find_all(exp.Div):
        numerator = division.this
        denominator = division.expression
        denominator_has_population = denominator.find(exp.Count) is not None
        numerator_has_population = numerator.find(exp.AggFunc) is not None
        numerator_has_threshold = _repeat_threshold_matches(
            numerator, count_aliases, threshold
        )
        if (
            denominator_has_population
            and numerator_has_population
            and (numerator_has_threshold or having_threshold)
        ):
            return True
    return False


def _has_null_safe_not_in(tree: exp.Expression) -> bool:
    """Recognize a NOT IN anti-set only when its projected key excludes NULL."""

    for negation in tree.find_all(exp.Not):
        inclusion = negation.this
        if not isinstance(inclusion, exp.In):
            continue
        query = inclusion.args.get("query")
        inner = query.this if isinstance(query, exp.Subquery) else query
        if not isinstance(inner, exp.Select) or len(inner.expressions) != 1:
            continue
        projected = next(inner.expressions[0].find_all(exp.Column), None)
        if projected is None:
            continue
        where = inner.args.get("where")
        if where is None:
            continue
        projected_table = str(projected.table or "").lower()
        projected_name = str(projected.name or "").lower()
        for not_node in where.find_all(exp.Not):
            is_node = not_node.this
            column = is_node.this if isinstance(is_node, exp.Is) else None
            if not (
                isinstance(column, exp.Column)
                and isinstance(is_node.expression, exp.Null)
            ):
                continue
            if (
                str(column.name or "").lower() == projected_name
                and (
                    not projected_table
                    or not column.table
                    or str(column.table).lower() == projected_table
                )
            ):
                return True
    return False


def _has_period_exclusion_aggregate(tree: exp.Expression) -> bool:
    """Recognize one-row-per-entity inactivity proofs such as MAX(period) < X."""

    for having in tree.find_all(exp.Having):
        for comparison in having.find_all(exp.LT, exp.LTE):
            if (
                comparison.this.find(exp.Max) is not None
                and isinstance(comparison.expression, exp.Literal)
            ):
                return True
        for comparison in having.find_all(exp.GT, exp.GTE):
            if (
                isinstance(comparison.this, exp.Literal)
                and comparison.expression.find(exp.Max) is not None
            ):
                return True
    return False


def _has_anti_population_shape(tree: exp.Expression) -> bool:
    """Accept equivalent, independently guarded anti-population SQL shapes."""

    has_not_exists = any(
        node.find(exp.Exists) is not None
        for node in tree.find_all(exp.Not)
    )
    has_left_null = bool(
        any(
            str(join.args.get("side") or "").upper() == "LEFT"
            for join in tree.find_all(exp.Join)
        )
        and any(
            node.find(exp.Null) is not None
            for node in tree.find_all(exp.Is)
        )
    )
    return bool(
        has_not_exists
        or has_left_null
        or _has_null_safe_not_in(tree)
        or _has_period_exclusion_aggregate(tree)
    )


def _has_event_selector_shape(tree: exp.Expression) -> bool:
    """Prove first/last-event selection by aggregate or ranked-event form."""

    if tree.find(exp.Min) is not None or tree.find(exp.Max) is not None:
        return True

    rank_aliases: set[str] = set()
    for selected in tree.find_all(exp.Select):
        for projection in selected.expressions:
            window = projection.find(exp.Window)
            if not (
                window is not None
                and isinstance(window.this, exp.RowNumber)
                and bool(window.args.get("partition_by"))
                and window.args.get("order") is not None
            ):
                continue
            alias = str(projection.alias_or_name or "").lower()
            if alias:
                rank_aliases.add(alias)
    if not rank_aliases:
        return False

    for equality in tree.find_all(exp.EQ):
        pairs = (
            (equality.this, equality.expression),
            (equality.expression, equality.this),
        )
        for column, literal in pairs:
            if not isinstance(column, exp.Column) or not isinstance(literal, exp.Literal):
                continue
            try:
                is_first = float(literal.this) == 1.0
            except (TypeError, ValueError):
                is_first = False
            if is_first and str(column.name or "").lower() in rank_aliases:
                return True
    return False


def audit_sql_against_query_contract(
    sql: str,
    contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Audit a candidate SQL against cross-stage semantic invariants.

    This audit is intentionally structural.  It does not compare SQL text and
    it does not attempt to prove arbitrary business logic.  It verifies the
    reusable invariants that caused the largest blind-test failures: physical
    relationship closure, final visible grain, registered aggregate inputs and
    required ratio/window/cohort shapes.
    """

    value = compact_query_contract(contract)
    operations = set(value.get("operations") or [])
    violations: list[dict[str, Any]] = []
    try:
        tree = sqlglot.parse_one(str(sql or ""), read="mysql")
    except Exception as exc:
        return {
            "schema_version": "semantic-query-contract-audit-v1",
            "valid": False,
            "violations": [{
                "code": "query_contract.sql_parse_failed",
                "detail": type(exc).__name__,
            }],
            "contract_hash": value.get("contract_hash"),
        }

    cte_names = {
        str(cte.alias_or_name or "").lower()
        for cte in tree.find_all(exp.CTE)
        if cte.alias_or_name
    }
    physical_aliases, column_lineage, unique_entity_refs = (
        _build_sql_column_lineage(tree, cte_names)
    )
    actual_tables: set[str] = set()
    for table in tree.find_all(exp.Table):
        table_name = str(table.name or "").lower()
        if not table_name or table_name in cte_names:
            continue
        actual_tables.add(table_name)

    relation = dict(value.get("relation_plan") or {})
    required_tables = {
        str(name).lower()
        for name in dict(relation.get("required_columns") or {})
    }
    missing_tables = sorted(required_tables - actual_tables)
    if missing_tables:
        violations.append({
            "code": "query_contract.missing_required_table",
            "detail": missing_tables,
        })

    equality_edges: set[tuple[tuple[str, str], tuple[str, str]]] = set()
    for equality in tree.find_all(exp.EQ):
        left = equality.left
        right = equality.right
        if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
            continue
        left_origins = _column_origins(
            left, physical_aliases, column_lineage
        )
        right_origins = _column_origins(
            right, physical_aliases, column_lineage
        )
        for left_origin in left_origins:
            for right_origin in right_origins:
                equality_edges.add(tuple(sorted((
                    left_origin, right_origin,
                ))))
    for join in relation.get("joins") or []:
        expected = tuple(sorted((
            (
                str(join.get("left_table") or "").lower(),
                str(join.get("left_column") or "").lower(),
            ),
            (
                str(join.get("right_table") or "").lower(),
                str(join.get("right_column") or "").lower(),
            ),
        )))
        if (
            expected[0][0] in actual_tables
            and expected[1][0] in actual_tables
            and expected not in equality_edges
            and not (operations & {"cohort", "set"})
        ):
            violations.append({
                "code": "query_contract.missing_relation_edge",
                "detail": [".".join(item) for item in expected],
            })

    final_select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    final_projection: set[str] = set()
    if final_select is not None:
        for projection in final_select.expressions:
            if projection.alias_or_name:
                final_projection.add(str(projection.alias_or_name).lower())
            final_projection.update(
                str(column.name or "").lower()
                for column in projection.find_all(exp.Column)
                if column.name
            )
    for grain in value.get("output_grain") or []:
        column = str(grain.get("column") or "").lower()
        logical = str(grain.get("logical") or "").lower()
        if column and not ({column, logical} & final_projection):
            violations.append({
                "code": "query_contract.missing_output_grain",
                "detail": f"{grain.get('table')}.{grain.get('column')}",
            })

    for metric in value.get("metric_contracts") or []:
        function = str(metric.get("function") or "").upper()
        refs = [
            _split_ref(item) for item in (metric.get("columns") or [])
        ]
        refs = [item for item in refs if item]
        if function not in {"SUM", "COUNT", "AVG", "MIN", "MAX"} or not refs:
            continue
        if (
            metric.get("source") == "semantic_ir"
            and operations
            & {"cohort", "set", "anti", "segmentation", "multi_having"}
        ):
            # The IR measure describes the final business result.  These
            # families often implement it as COUNT(*) over a qualifying CTE or
            # a bucket summary rather than the literal inner source column.
            continue
        represented = False
        for aggregate in tree.find_all(exp.AggFunc):
            if aggregate.sql_name().upper() != function:
                continue
            aggregate_columns: set[tuple[str | None, str]] = set()
            for column in aggregate.find_all(exp.Column):
                origins = _column_origins(
                    column, physical_aliases, column_lineage
                )
                if origins:
                    aggregate_columns.update(origins)
                else:
                    aggregate_columns.add((
                        None, str(column.name or "").lower(),
                    ))
            count_star_origins = _count_star_unique_origins(
                aggregate, unique_entity_refs
            )
            if any(
                (table.lower(), column.lower()) in aggregate_columns
                or (None, column.lower()) in aggregate_columns
                or (table.lower(), column.lower()) in count_star_origins
                for table, column in refs
            ):
                if (
                    metric.get("distinct")
                    and aggregate.find(exp.Distinct) is None
                    and not count_star_origins
                ):
                    continue
                represented = True
                break
        if not represented:
            violations.append({
                "code": "query_contract.metric_formula_missing",
                "detail": {
                    "name": metric.get("name"),
                    "function": function,
                    "columns": [".".join(item) for item in refs],
                },
            })

    if (value.get("operator_contract") or {}).get("ratio_required") and tree.find(exp.Div) is None:
        violations.append({
            "code": "query_contract.ratio_shape_missing",
            "detail": sorted(operations),
        })
    if (value.get("operator_contract") or {}).get("ratio_required") and not any(
        isinstance(division.expression, exp.Nullif)
        or division.expression.find(exp.Nullif) is not None
        for division in tree.find_all(exp.Div)
    ):
        violations.append({
            "code": "query_contract.zero_denominator_guard_missing",
            "detail": "NULLIF",
        })
    if "share" in operations and value.get("output_grain") and not any(
        window.find(exp.Sum) is not None
        for window in tree.find_all(exp.Window)
    ):
        violations.append({
            "code": "query_contract.share_denominator_scope_missing",
            "detail": "window_total_before_rank_or_limit",
        })
    if "activation_rate" in operations:
        distinct_counts = [
            count
            for count in tree.find_all(exp.Count)
            if count.find(exp.Distinct) is not None
        ]
        if len(distinct_counts) < 2:
            violations.append({
                "code": "query_contract.activation_population_missing",
                "detail": "distinct_event_population_over_dimension_universe",
            })
    repeat_threshold = next((
        int(metric.get("repeat_threshold") or 2)
        for metric in value.get("metric_contracts") or []
        if str(metric.get("type") or "").lower() == "repeat_rate"
    ), 2)
    if (
        "repeat_rate" in operations
        and not _has_repeat_population_shape(tree, repeat_threshold)
    ):
        violations.append({
            "code": "query_contract.repeat_population_missing",
            "detail": "entity_event_threshold_over_filtered_population",
        })
    if "anti" in operations:
        if not _has_anti_population_shape(tree):
            violations.append({
                "code": "query_contract.anti_population_missing",
                "detail": (
                    "not_exists_or_left_join_is_null_or_null_safe_not_in_or_"
                    "entity_period_exclusion"
                ),
            })
    if "period_delta" in operations and tree.find(exp.Sub) is None:
        violations.append({
            "code": "query_contract.period_delta_missing",
            "detail": "subtraction",
        })
    if "period_growth" in operations and (
        tree.find(exp.Div) is None or tree.find(exp.Sub) is None
    ):
        violations.append({
            "code": "query_contract.period_growth_shape_missing",
            "detail": "delta_over_previous_period",
        })
    if "top_per_group" in operations:
        partitioned_rank = any(
            isinstance(window.this, (exp.RowNumber, exp.Rank, exp.DenseRank))
            and bool(window.args.get("partition_by"))
            for window in tree.find_all(exp.Window)
        )
        if not partitioned_rank:
            violations.append({
                "code": "query_contract.partitioned_rank_missing",
                "detail": (value.get("operator_contract") or {}).get("partition_keys"),
            })
    if "running_sum" in operations:
        running_sum = any(
            isinstance(window.this, exp.Sum) and window.args.get("order") is not None
            for window in tree.find_all(exp.Window)
        )
        if not running_sum:
            violations.append({
                "code": "query_contract.running_sum_missing",
                "detail": "ordered_sum_window",
            })
    if "cohort" in operations and not _has_event_selector_shape(tree):
        violations.append({
            "code": "query_contract.cohort_event_selector_missing",
            "detail": "first_or_last_event_aggregate_or_partitioned_row_number",
        })

    return {
        "schema_version": "semantic-query-contract-audit-v1",
        "valid": not violations,
        "violations": violations,
        "contract_hash": value.get("contract_hash"),
        "authority": value.get("authority"),
        "actual_tables": sorted(actual_tables),
        "raw_sql_persisted": False,
    }
