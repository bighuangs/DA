"""指标口径与查询成本守卫。

该模块只放与外部存储无关的校验逻辑，便于在 SQL 校验节点和单元测试中复用。
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping

import sqlglot
from sqlglot import exp


class MetricFormulaError(Exception):
    """生成 SQL 未遵循已注册指标口径。"""


class QueryCostError(Exception):
    """SQL 预计扫描成本超过系统阈值。"""


_DIMENSION_UNIVERSE_POPULATION = "dimension_universe_minus_events"
_FILTERED_FACT_POPULATION = "filtered_fact_rows"
_STRUCTURAL_RULE_TYPES = {
    "activation_rate",
    "anti_join",
    "cohort_count",
    "growth_rate",
    "inactivity_cohort",
    "period_growth",
    "repeat_rate",
    "share",
}


def _metric_rule_type(metric: Mapping) -> str:
    return str((metric.get("formula_rule") or {}).get("type") or "aggregate").lower()


def _semantic_family(semantic_ir: Mapping | None) -> str:
    ir = semantic_ir or {}
    return str(ir.get("semantic_family") or ir.get("intent") or "").lower()


def _requested_population_scopes(semantic_ir: Mapping | None) -> set[str]:
    """Return explicitly closed population scopes, or the family default.

    A multi-measure IR may declare more than one population.  In that case no
    formula is shadowed merely because another requested output uses a different
    population.  Older IRs do not carry this optional slot; anti-join still has
    one unambiguous preserved-universe default.
    """

    ir = semantic_ir or {}
    slots = ir.get("semantic_slots") or {}
    raw = (
        slots.get("requested_population_scopes")
        or slots.get("population_scopes")
        or ir.get("population_scopes")
        or []
    )
    if isinstance(raw, str):
        raw = [raw]
    scopes = {str(item).lower() for item in raw if str(item).strip()}
    if not scopes and _semantic_family(ir) == "anti_join":
        scopes.add(_DIMENSION_UNIVERSE_POPULATION)
    return scopes


def _metric_population_scope(metric: Mapping) -> str:
    rule = metric.get("formula_rule") or {}
    rule_type = _metric_rule_type(metric)
    if rule_type == "anti_join":
        return _DIMENSION_UNIVERSE_POPULATION
    return str(rule.get("population_scope") or "").lower()


def _rule_entity_columns(metric: Mapping) -> set[str]:
    """Return governed entity-column leaves for population conflict checks."""

    rule = metric.get("formula_rule") or {}
    values = [
        rule.get("column"),
        rule.get("entity_column"),
        rule.get("universe_column"),
        rule.get("event_column"),
    ]
    return {
        str(value).strip("`").lower().rsplit(".", 1)[-1]
        for value in values
        if str(value or "").strip()
    }


def _rule_tables(metric: Mapping) -> set[str]:
    rule = metric.get("formula_rule") or {}
    values = [
        rule.get("column"),
        rule.get("entity_column"),
        rule.get("universe_column"),
        rule.get("event_column"),
    ]
    output: set[str] = set()
    for value in values:
        parts = str(value or "").strip("`").lower().split(".", 1)
        if len(parts) == 2 and all(parts):
            output.add(parts[0])
    return output


def _metric_matches_ir_tables(metric: Mapping, semantic_ir: Mapping | None) -> bool:
    governed = _rule_tables(metric)
    required = {
        str(item).strip("`").lower()
        for item in ((semantic_ir or {}).get("required_tables") or [])
        if str(item).strip()
    }
    return bool(governed and required and governed.issubset(required))


def _metric_term_spans(query: str, metric: Mapping) -> list[tuple[int, int]]:
    text = str(query or "").lower()
    spans: list[tuple[int, int]] = []
    for term in (metric.get("name", ""), *(metric.get("alias") or [])):
        needle = str(term or "").strip().lower()
        if not needle:
            continue
        start = 0
        while True:
            index = text.find(needle, start)
            if index < 0:
                break
            spans.append((index, index + len(needle)))
            start = index + max(1, len(needle))
    return spans


def _has_independent_surface_request(
    query: str,
    generic: Mapping,
    specialized: Mapping,
) -> bool:
    """Whether a generic metric is named outside a specialized metric phrase."""

    generic_spans = _metric_term_spans(query, generic)
    specialized_spans = _metric_term_spans(query, specialized)
    if not generic_spans or not specialized_spans:
        return False
    return any(
        all(end <= other_start or start >= other_end
            for other_start, other_end in specialized_spans)
        for start, end in generic_spans
    )


def select_metric_authorities(
    query: str,
    metric_infos: Iterable[Mapping],
    selection: Iterable[str] | None = None,
    semantic_ir: Mapping | None = None,
    *,
    allow_surface_evidence: bool = True,
) -> list[Mapping]:
    """Select formula authorities using surface, semantic family and population.

    Retrieval relevance alone is not formula authority.  A family-compatible
    registered metric is promoted when it is recalled, while a generic
    filtered-fact metric cannot govern a preserved-universe anti-join merely
    because its short display name is a substring of the request.
    """

    metrics = list(metric_infos or [])
    selected_names = {str(item) for item in (selection or [])}
    authority_evidence = (semantic_ir or {}).get("metric_authority") or {}
    selected_names.update(
        str(item) for item in (authority_evidence.get("selected_names") or [])
    )
    direct = {
        id(metric): bool(_metric_term_spans(query, metric))
        for metric in metrics
    }
    family = _semantic_family(semantic_ir)
    family_matches = [
        metric for metric in metrics
        if family and _metric_rule_type(metric) == family
    ]
    ir_metric_name = str(((semantic_ir or {}).get("metric") or {}).get("name") or "")
    structural_matches = [
        metric for metric in family_matches
        if family in _STRUCTURAL_RULE_TYPES
        and _metric_matches_ir_tables(metric, semantic_ir)
    ]
    auto_promoted = structural_matches if len(structural_matches) == 1 else []
    specialized = [
        metric for metric in family_matches
        if direct[id(metric)]
        or str(metric.get("name") or "") in selected_names
        or (ir_metric_name and str(metric.get("name") or "") == ir_metric_name)
        or metric in auto_promoted
    ]
    requested_scopes = _requested_population_scopes(semantic_ir)
    explicit_multi_population = len(requested_scopes) > 1

    candidates = [
        metric for metric in metrics
        if (allow_surface_evidence and direct[id(metric)])
        or str(metric.get("name") or "") in selected_names
        or (ir_metric_name and str(metric.get("name") or "") == ir_metric_name)
        or metric in specialized
    ]
    result: list[Mapping] = []
    for metric in candidates:
        population = _metric_population_scope(metric)
        if (
            family == "anti_join"
            and population == _FILTERED_FACT_POPULATION
            and not explicit_multi_population
        ):
            competitors = [
                item for item in specialized
                if _rule_entity_columns(item) & _rule_entity_columns(metric)
            ]
            independently_requested = any(
                _has_independent_surface_request(query, metric, item)
                for item in competitors
            )
            if not independently_requested:
                # Even when the specialized metric was not recalled, the
                # generic fact population is inapplicable to an anti-join.
                continue
        if metric not in result:
            result.append(metric)

    # Put the semantic-family authority first without disturbing peer order.
    return sorted(
        result,
        key=lambda item: 0 if item in specialized else 1,
    )


def estimate_explain_rows(explain_rows: Iterable[Mapping]) -> int | None:
    """Estimate examined rows without multiplying independent query blocks.

    MySQL ``rows`` values in the same ``id``/``select_id`` describe one nested
    loop.  For each loop, downstream probes are driven by the rows surviving the
    preceding table's ``filtered`` percentage.  Ignoring ``filtered`` and simply
    multiplying every raw estimate badly overstates selective joins and
    semijoins; multiplying independent CTE/subquery blocks is worse still.

    We therefore sum examined rows along each filtered loop, sum independent
    blocks, and retain the raw-row sum as a conservative floor.  This still
    rejects genuinely expansive joins while avoiding duplicate/materialisation
    false positives for valid cohort queries.
    """
    total = 0
    grouped_loops: dict[str, list[tuple[int, float]]] = {}
    found = False
    for row in explain_rows or []:
        value = row.get("rows")
        if value is None:
            value = row.get("ROWS")
        if value is None:
            value = row.get("Rows")
        if value is None:
            continue
        try:
            estimate = max(0, int(float(value)))
            total += estimate
            if row.get("table") is not None or row.get("TABLE") is not None:
                block = (
                    row.get("id")
                    if row.get("id") is not None
                    else row.get("select_id")
                )
                if block is None:
                    block = "__implicit__"
                filtered = row.get("filtered")
                if filtered is None:
                    filtered = row.get("FILTERED")
                if filtered is None:
                    filtered = row.get("Filtered")
                try:
                    selectivity = min(1.0, max(0.0, float(filtered) / 100.0))
                except (TypeError, ValueError):
                    selectivity = 1.0
                grouped_loops.setdefault(str(block), []).append(
                    (estimate, selectivity)
                )
            found = True
        except (TypeError, ValueError):
            continue
    if not found:
        return None
    block_cost = 0.0
    for estimates in grouped_loops.values():
        survivors = 1.0
        for estimate, selectivity in estimates:
            examined = survivors * estimate
            block_cost += examined
            survivors = examined * selectivity
    return max(total, int(math.ceil(block_cost)))


def enforce_explain_cost(
    explain_rows: Iterable[Mapping],
    *,
    max_explain_rows: int,
    enabled: bool = True,
) -> int | None:
    """按 EXPLAIN 预计扫描行数执行成本守卫，返回估算值。"""
    estimated_rows = estimate_explain_rows(explain_rows)
    if enabled and max_explain_rows > 0 and estimated_rows is not None:
        if estimated_rows > max_explain_rows:
            raise QueryCostError(
                f"预计扫描行数 {estimated_rows:,}，超过阈值 {max_explain_rows:,}"
            )
    return estimated_rows


def _table_aliases(tree: exp.Expression) -> dict[str, str]:
    """构造 SQL 别名到真实表名的映射。"""
    aliases: dict[str, str] = {}
    for table in tree.find_all(exp.Table):
        aliases[table.alias_or_name.lower()] = table.name.lower()
        aliases[table.name.lower()] = table.name.lower()
    return aliases


def _column_matches(column: exp.Column, expected_column: str, aliases: dict[str, str]) -> bool:
    """判断 AST 字段是否对应指标规则中的字段，允许 SQL 使用表别名。"""
    expected_parts = [part.strip().strip("`").lower() for part in expected_column.split(".")]
    expected_name = expected_parts[-1]
    if column.name.strip("`").lower() != expected_name:
        return False
    if len(expected_parts) == 1 or not column.table:
        return True
    actual_table = aliases.get(column.table.strip("`").lower(), column.table.lower())
    return actual_table == expected_parts[-2]


def _aggregate_matches(tree: exp.Expression, function: str, column: str, aliases: dict[str, str]) -> bool:
    expected_function = function.upper()
    for aggregate in tree.find_all(exp.AggFunc):
        try:
            actual_function = aggregate.sql_name().upper()
        except Exception:
            continue
        if actual_function != expected_function:
            continue
        if any(_column_matches(item, column, aliases) for item in aggregate.find_all(exp.Column)):
            return True
    return False


def _count_matches_expected_table(
    tree: exp.Expression,
    expected_column: str,
    aliases: dict[str, str],
) -> bool:
    """Accept a row/order count from the same source as an AVG numerator."""

    expected_parts = [
        part.strip().strip("`").lower()
        for part in expected_column.split(".")
    ]
    expected_table = expected_parts[-2] if len(expected_parts) > 1 else ""
    for aggregate in tree.find_all(exp.Count):
        columns = list(aggregate.find_all(exp.Column))
        if not columns:
            return True
        for item in columns:
            actual_table = aliases.get(
                item.table.strip("`").lower(), item.table.lower()
            ) if item.table else ""
            if not expected_table or not actual_table or actual_table == expected_table:
                return True
    return False


def _average_ratio_matches(
    tree: exp.Expression,
    expected_column: str,
    aliases: dict[str, str],
) -> bool:
    """Recognise AVG(x) expressed as SUM(x) / COUNT(rows or order ids)."""

    return any(
        _aggregate_matches(node.this, "SUM", expected_column, aliases)
        and _count_matches_expected_table(
            node.expression, expected_column, aliases
        )
        for node in tree.find_all(exp.Div)
    )


def _distinct_count_matches(
    tree: exp.Expression,
    expected_column: str,
    aliases: dict[str, str],
) -> bool:
    return any(
        aggregate.find(exp.Distinct) is not None
        and any(
            _column_matches(column, expected_column, aliases)
            for column in aggregate.find_all(exp.Column)
        )
        for aggregate in tree.find_all(exp.Count)
    )


def _closed_conditional_count_ir(
    rule: Mapping,
    semantic_ir: Mapping | None,
) -> bool:
    """Require a complete IR that binds this COUNT to conditional outputs."""

    ir = semantic_ir or {}
    if (
        _semantic_family(ir) != "conditional_aggregate"
        or str(ir.get("status") or "").lower() != "complete"
        or (ir.get("compiler") or {}).get("supported") is not True
        or str(rule.get("function") or "").upper() != "COUNT"
    ):
        return False
    expected = str(rule.get("column") or "").strip("`").lower().split(".")
    if not expected or not expected[-1]:
        return False
    expected_table = expected[-2] if len(expected) > 1 else ""
    expected_column = expected[-1]
    return any(
        str(measure.get("function") or "").upper() == "COUNT"
        and str(measure.get("column") or "").strip("`").lower()
        == expected_column
        and (
            not expected_table
            or str(measure.get("table") or "").strip("`").lower()
            == expected_table
        )
        for measure in (ir.get("measures") or [])
    )


def _numeric_literal_is(node: exp.Expression | None, expected: int) -> bool:
    if not isinstance(node, exp.Literal) or node.is_string:
        return False
    try:
        return float(node.this) == float(expected)
    except (TypeError, ValueError):
        return False


def _case_predicates(case: exp.Case) -> list[exp.If]:
    return [item for item in (case.args.get("ifs") or []) if isinstance(item, exp.If)]


def _conditional_count_matches(
    tree: exp.Expression,
    rule: Mapping,
    aliases: dict[str, str],
    semantic_ir: Mapping | None,
) -> bool:
    """Recognise bounded SQL encodings of a registered conditional row count."""

    if not _closed_conditional_count_ir(rule, semantic_ir):
        return False
    expected_column = str(rule.get("column") or "")
    expected_parts = expected_column.strip("`").lower().split(".")
    expected_table = expected_parts[-2] if len(expected_parts) > 1 else ""
    expected_table_present = not expected_table or expected_table in set(aliases.values())

    for aggregate in tree.find_all(exp.Sum):
        case = aggregate.this
        if not isinstance(case, exp.Case):
            continue
        predicates = _case_predicates(case)
        if (
            expected_table_present
            and predicates
            and all(_numeric_literal_is(item.args.get("true"), 1)
                    for item in predicates)
            and _numeric_literal_is(case.args.get("default"), 0)
        ):
            return True

    for aggregate in tree.find_all(exp.Count):
        # COUNT(DISTINCT CASE ...) changes the registered row population.
        if aggregate.find(exp.Distinct) is not None:
            continue
        case = aggregate.this
        if not isinstance(case, exp.Case):
            continue
        default = case.args.get("default")
        if default is not None and not isinstance(default, exp.Null):
            continue
        predicates = _case_predicates(case)
        if not predicates:
            continue
        valid_values = True
        for item in predicates:
            value = item.args.get("true")
            if _numeric_literal_is(value, 1):
                valid_values = valid_values and expected_table_present
            elif isinstance(value, exp.Column):
                valid_values = valid_values and _column_matches(
                    value, expected_column, aliases
                )
            else:
                valid_values = False
        if valid_values:
            return True
    return False


def _formula_aliases(
    tree: exp.Expression,
    *,
    function: str,
    column: str,
    aliases: dict[str, str],
    distinct: bool = False,
) -> set[str]:
    """Return projection aliases proven to own one registered aggregate."""

    output: set[str] = set()
    for select in tree.find_all(exp.Select):
        for projection in select.expressions:
            expression = (
                projection.this
                if isinstance(projection, exp.Alias)
                else projection
            )
            if not _aggregate_matches(
                expression, function, column, aliases
            ):
                continue
            if distinct and expression.find(exp.Distinct) is None:
                continue
            name = str(projection.alias_or_name or "").strip("`").lower()
            if name:
                output.add(name)
    return output


def _expression_owns_formula(
    expression: exp.Expression,
    *,
    function: str,
    column: str,
    aliases: dict[str, str],
    formula_aliases: set[str],
    distinct: bool = False,
) -> bool:
    if _aggregate_matches(expression, function, column, aliases):
        return not distinct or expression.find(exp.Distinct) is not None
    return any(
        str(item.name or "").strip("`").lower() in formula_aliases
        for item in expression.find_all(exp.Column)
    )


def _null_safe_denominator(expression: exp.Expression) -> bool:
    return (
        isinstance(expression, exp.Nullif)
        or expression.find(exp.Nullif) is not None
    )


def _share_matches(
    tree: exp.Expression,
    rule: Mapping,
    aliases: dict[str, str],
) -> bool:
    function = str(rule.get("function") or "SUM").upper()
    column = str(rule.get("column") or "")
    if not column:
        return False
    formula_aliases = _formula_aliases(
        tree, function=function, column=column, aliases=aliases
    )
    for division in tree.find_all(exp.Div):
        if not _null_safe_denominator(division.expression):
            continue
        if not _expression_owns_formula(
            division.this,
            function=function,
            column=column,
            aliases=aliases,
            formula_aliases=formula_aliases,
        ):
            continue
        denominator = division.expression
        has_window_total = any(
            window.find(exp.Sum) is not None
            and (
                not formula_aliases
                or any(
                    str(item.name or "").strip("`").lower()
                    in formula_aliases
                    for item in window.find_all(exp.Column)
                )
                or _aggregate_matches(window, function, column, aliases)
            )
            for window in denominator.find_all(exp.Window)
        )
        has_registered_total = _expression_owns_formula(
            denominator,
            function=function,
            column=column,
            aliases=aliases,
            formula_aliases=formula_aliases,
        )
        if has_window_total or has_registered_total:
            return True
    return False


def _activation_rate_matches(
    tree: exp.Expression,
    rule: Mapping,
    aliases: dict[str, str],
) -> bool:
    numerator = str(rule.get("numerator_column") or "")
    denominator = str(rule.get("denominator_column") or "")
    if not numerator or not denominator:
        return False
    numerator_aliases = _formula_aliases(
        tree,
        function="COUNT",
        column=numerator,
        aliases=aliases,
        distinct=True,
    )
    denominator_aliases = _formula_aliases(
        tree,
        function="COUNT",
        column=denominator,
        aliases=aliases,
        distinct=True,
    )
    return any(
        _null_safe_denominator(division.expression)
        and _expression_owns_formula(
            division.this,
            function="COUNT",
            column=numerator,
            aliases=aliases,
            formula_aliases=numerator_aliases,
            distinct=True,
        )
        and _expression_owns_formula(
            division.expression,
            function="COUNT",
            column=denominator,
            aliases=aliases,
            formula_aliases=denominator_aliases,
            distinct=True,
        )
        for division in tree.find_all(exp.Div)
    )


def _period_growth_matches(
    tree: exp.Expression,
    rule: Mapping,
    aliases: dict[str, str],
) -> bool:
    function = str(rule.get("function") or "SUM").upper()
    column = str(rule.get("column") or "")
    if not column or not _aggregate_matches(tree, function, column, aliases):
        return False
    return any(
        division.this.find(exp.Sub) is not None
        and _null_safe_denominator(division.expression)
        and (
            tree.find(exp.Lag) is not None
            or len(list(tree.find_all(exp.Case))) >= 2
            or len(list(tree.find_all(exp.Select))) >= 2
        )
        for division in tree.find_all(exp.Div)
    )


def _anti_join_matches(
    tree: exp.Expression,
    rule: Mapping,
    aliases: dict[str, str],
) -> bool:
    universe = str(rule.get("universe_column") or "")
    event = str(rule.get("event_column") or "")
    columns = list(tree.find_all(exp.Column))
    if not (
        universe
        and event
        and any(_column_matches(item, universe, aliases) for item in columns)
        and any(_column_matches(item, event, aliases) for item in columns)
    ):
        return False
    not_exists = any(
        node.find(exp.Exists) is not None for node in tree.find_all(exp.Not)
    )
    left_join = any(
        str(join.args.get("side") or "").upper() == "LEFT"
        for join in tree.find_all(exp.Join)
    )
    null_rejection = any(
        node.find(exp.Null) is not None for node in tree.find_all(exp.Is)
    )
    return not_exists or (left_join and null_rejection)


def _cohort_count_matches(
    tree: exp.Expression,
    rule: Mapping,
    aliases: dict[str, str],
) -> bool:
    entity = str(rule.get("entity_column") or "")
    event = str(rule.get("event_column") or "")
    selector = str(rule.get("event_selector") or "MIN").upper()
    if not entity or not event or selector not in {"MIN", "MAX"}:
        return False
    has_selector = _aggregate_matches(tree, selector, event, aliases)
    has_entity_grain = any(
        any(
            _column_matches(column, entity, aliases)
            for column in group.find_all(exp.Column)
        )
        for group in tree.find_all(exp.Group)
    )
    return bool(
        has_selector
        and has_entity_grain
        and list(tree.find_all(exp.Count))
        and len(list(tree.find_all(exp.Select))) >= 2
    )


def _inactivity_cohort_matches(
    tree: exp.Expression,
    rule: Mapping,
    aliases: dict[str, str],
) -> bool:
    entity = str(rule.get("entity_column") or "")
    event = str(rule.get("event_column") or "")
    has_last_event = bool(event) and _aggregate_matches(
        tree, "MAX", event, aliases
    )
    has_entity_grain = bool(entity) and any(
        any(
            _column_matches(column, entity, aliases)
            for column in group.find_all(exp.Column)
        )
        for group in tree.find_all(exp.Group)
    )
    has_time_comparison = any(
        str(function.name or "").upper()
        in {"DATEDIFF", "TIMESTAMPDIFF", "DATE_SUB"}
        for function in tree.find_all(exp.Anonymous)
    ) or any(
        node.find(exp.Literal) is not None
        for node_type in (exp.GT, exp.GTE, exp.LT, exp.LTE)
        for node in tree.find_all(node_type)
    )
    return has_last_event and has_entity_grain and has_time_comparison


def _formula_matches(tree: exp.Expression, rule: Mapping, aliases: dict[str, str]) -> bool:
    rule_type = str(rule.get("type", "aggregate")).lower()
    if rule_type == "aggregate":
        function = rule.get("function")
        column = rule.get("column")
        if not (function and column):
            return False
        if _aggregate_matches(tree, str(function), str(column), aliases):
            return True
        return bool(
            str(function).upper() == "AVG"
            and _average_ratio_matches(tree, str(column), aliases)
        )
    if rule_type == "ratio":
        numerator = rule.get("numerator") or {}
        denominator = rule.get("denominator") or {}
        return any(
            _formula_matches(node.this, numerator, aliases)
            and _formula_matches(node.expression, denominator, aliases)
            for node in tree.find_all(exp.Div)
        )
    if rule_type == "ranking":
        function = rule.get("function")
        column = rule.get("column")
        return bool(function and column) and _aggregate_matches(tree, str(function), str(column), aliases)
    if rule_type == "distinct_count":
        column = str(rule.get("column") or "")
        return bool(column) and _distinct_count_matches(
            tree, column, aliases
        )
    if rule_type == "entity_aggregate":
        function = str(rule.get("function") or "")
        column = str(rule.get("column") or "")
        return bool(function and column) and _aggregate_matches(
            tree, function, column, aliases
        )
    if rule_type == "share":
        return _share_matches(tree, rule, aliases)
    if rule_type == "activation_rate":
        return _activation_rate_matches(tree, rule, aliases)
    if rule_type in {"period_growth", "growth_rate"}:
        return _period_growth_matches(tree, rule, aliases)
    if rule_type == "anti_join":
        return _anti_join_matches(tree, rule, aliases)
    if rule_type == "cohort_count":
        return _cohort_count_matches(tree, rule, aliases)
    if rule_type == "inactivity_cohort":
        return _inactivity_cohort_matches(tree, rule, aliases)
    if rule_type == "repeat_rate":
        entity = str(rule.get("entity_column") or "")
        event = str(rule.get("event_column") or "")
        threshold = int(rule.get("repeat_threshold") or 2)
        has_entity_group = any(
            any(_column_matches(column, entity, aliases) for column in group.find_all(exp.Column))
            for group in tree.find_all(exp.Group)
        )
        has_event_count = any(
            aggregate.sql_name().upper() == "COUNT"
            and any(_column_matches(column, event, aliases) for column in aggregate.find_all(exp.Column))
            for aggregate in tree.find_all(exp.AggFunc)
        )
        has_threshold = any(
            isinstance(node.expression, exp.Literal)
            and str(node.expression.this) == str(threshold)
            for node in tree.find_all(exp.GTE)
        )
        has_null_safe_division = any(
            division.expression.find(exp.Nullif) is not None
            for division in tree.find_all(exp.Div)
        )
        return has_entity_group and has_event_count and has_threshold and has_null_safe_division
    return False


def _result_selects(tree: exp.Expression) -> list[exp.Select]:
    """Return direct user-visible SELECT scopes, including every UNION branch."""
    if isinstance(tree, (exp.Union, exp.Intersect, exp.Except)):
        return [
            *_result_selects(tree.this),
            *_result_selects(tree.expression),
        ]
    if isinstance(tree, exp.Subquery):
        return _result_selects(tree.this)
    if isinstance(tree, exp.Select):
        return [tree]
    final = tree.find(exp.Select)
    return [final] if final is not None else []


def _projection_definitions(
    tree: exp.Expression,
) -> dict[str, list[exp.Expression]]:
    """Index named SELECT expressions for conservative CTE/subquery lineage."""

    definitions: dict[str, list[exp.Expression]] = {}
    for select in tree.find_all(exp.Select):
        for projection in select.expressions:
            name = str(projection.alias_or_name or "").strip("`").lower()
            if not name:
                continue
            expression = (
                projection.this
                if isinstance(projection, exp.Alias)
                else projection
            )
            definitions.setdefault(name, []).append(expression)
    return definitions


def _formula_matches_with_lineage(
    expression: exp.Expression,
    rule: Mapping,
    aliases: dict[str, str],
    definitions: Mapping[str, list[exp.Expression]],
    *,
    visited: frozenset[str] = frozenset(),
) -> bool:
    """Resolve formula-bearing aliases exposed through CTEs/subqueries.

    A metric remains user-visible when the final SELECT projects a CTE alias
    whose upstream definition contains the registered aggregate.  This keeps
    the original final-projection guard while avoiding false negatives for
    correct window and ranked-query shapes.
    """

    if _formula_matches(expression, rule, aliases):
        return True
    for column in expression.find_all(exp.Column):
        name = column.name.strip("`").lower()
        if not name or name in visited:
            continue
        for definition in definitions.get(name, []):
            if definition is expression:
                continue
            if _formula_matches_with_lineage(
                definition,
                rule,
                aliases,
                definitions,
                visited=visited | {name},
            ):
                return True
    return False


def _conditional_count_matches_with_lineage(
    expression: exp.Expression,
    rule: Mapping,
    aliases: dict[str, str],
    definitions: Mapping[str, list[exp.Expression]],
    semantic_ir: Mapping | None,
    *,
    visited: frozenset[str] = frozenset(),
) -> bool:
    if _conditional_count_matches(expression, rule, aliases, semantic_ir):
        return True
    for column in expression.find_all(exp.Column):
        name = column.name.strip("`").lower()
        if not name or name in visited:
            continue
        for definition in definitions.get(name, []):
            if definition is expression:
                continue
            if _conditional_count_matches_with_lineage(
                definition,
                rule,
                aliases,
                definitions,
                semantic_ir,
                visited=visited | {name},
            ):
                return True
    return False


def _anti_join_count_requested(
    query: str,
    semantic_ir: Mapping | None,
) -> bool:
    ir = semantic_ir or {}
    if any(
        "COUNT" in str(measure.get("function") or "").upper()
        for measure in (ir.get("measures") or [])
    ):
        return True
    text = str(query or "").lower()
    return bool(re.search(
        r"(?:how\s+many|\bcount\b|\u591a\u5c11|\u51e0(?:\u4e2a|\u4f4d|\u540d|\u7b14|\u4ef6)?|"
        r"(?:\u5ba2\u6237|\u7528\u6237|\u987e\u5ba2|\u5546\u54c1|\u4ea7\u54c1|\u8ba2\u5355|\u8bb0\u5f55|\u5b9e\u4f53|\u4eba)"
        r"(?:\u603b)?(?:\u6570|\u6570\u91cf|\u4eba\u6570|\u4e2a\u6570|\u7b14\u6570|\u4ef6\u6570))",
        text,
    ))


def _nearest_select(node: exp.Expression) -> exp.Select | None:
    parent = node.parent
    while parent is not None:
        if isinstance(parent, exp.Select):
            return parent
        parent = parent.parent
    return None


def _anti_join_population_matches(
    tree: exp.Expression,
    rule: Mapping,
    aliases: dict[str, str],
) -> bool:
    """Require an inactive count to count preserved-universe rows, not NULL facts."""

    universe = str(rule.get("universe_column") or "")
    event = str(rule.get("event_column") or "")
    if not universe or not event:
        return False
    counts: list[exp.Count] = []
    for select in _result_selects(tree):
        for projection in select.expressions:
            counts.extend(
                item for item in projection.find_all(exp.Count)
                if _nearest_select(item) is select
            )
    if not counts:
        return False

    valid_universe_count = False
    for count in counts:
        if isinstance(count.this, exp.Star):
            valid_universe_count = True
            continue
        columns = list(count.find_all(exp.Column))
        if any(_column_matches(column, event, aliases) for column in columns):
            return False
        if not columns or not all(
            _column_matches(column, universe, aliases) for column in columns
        ):
            return False
        valid_universe_count = True
    return valid_universe_count


def _metric_is_relational_operand(query: str, names: Iterable[object]) -> bool:
    """Whether a registered metric qualifies/ranks rows without being output.

    A metric formula may legitimately live in an inner HAVING, ranking CTE or
    ordering scope while the user-visible result is an entity count/list or a
    different derived metric.  Direct requests such as “查询 GMV” still require
    the formula in the final projection.
    """

    text = str(query or "").lower()
    terms = [str(item).strip().lower() for item in names if str(item).strip()]
    if not any(term in text for term in terms):
        return False
    qualifies_entities = bool(
        re.search(
            r"(?:客户|顾客|用户|品牌|品类|类别|商品|产品)"
            r"[^，。；;]{0,24}(?:多少|数量|人数|个数|有哪些|名单)",
            text,
        )
        or re.search(
            r"(?:多少|几个|几位|列出|找出)"
            r"[^，。；;]{0,24}(?:客户|顾客|用户|品牌|品类|类别|商品|产品)",
            text,
        )
    )
    has_relational_operator = any(token in text for token in (
        "至少", "不少于", "超过", "高于", "低于", "达到",
        "最高", "最低", "最多", "最少", "领先", "垫底",
        "前", "倒数", "排名", "排序", "累计",
    ))
    return qualifies_entities or has_relational_operator


def check_metric_formulas(
    sql: str,
    query: str,
    metric_infos: Iterable[Mapping],
    semantic_ir: Mapping | None = None,
) -> None:
    """检查问题命中的已注册指标是否使用了登记的计算口径。

    当前配置支持 ``aggregate`` 和 ``ratio`` 两种规则。未登记规则的纯术语或
    历史指标不会被误拦截；命中有规则的指标但 SQL 缺少对应聚合时抛错，交给
    现有纠错节点重新生成 SQL。
    """
    metrics = select_metric_authorities(
        query,
        metric_infos,
        semantic_ir=semantic_ir,
    )
    if not metrics:
        return
    try:
        tree = sqlglot.parse_one(sql, read="mysql")
    except Exception as exc:
        # 语法错误由 SQL 安全/EXPLAIN 校验负责，这里不重复包装。
        raise MetricFormulaError(f"指标口径校验前 SQL 解析失败：{exc}") from exc

    query_text = (query or "").lower()
    result_selects = _result_selects(tree)
    if not result_selects:
        raise MetricFormulaError("指标口径校验未找到最终 SELECT 投影")
    projection_scopes = [
        (projection, _table_aliases(select))
        for select in result_selects
        for projection in select.expressions
    ]
    aliases = _table_aliases(tree)
    definitions = _projection_definitions(tree)
    for metric in metrics:
        rule = metric.get("formula_rule")
        if not rule:
            continue
        rule_type = str(rule.get("type") or "aggregate").lower()
        names = [metric.get("name", ""), *(metric.get("alias") or [])]
        # Retrieval relevance is not metric authority.  A recalled metric may
        # share the same physical column with a different user-requested
        # operation (for example registered SUM(GMV) versus explicit MIN/MAX
        # single-order amount).  Enforce a registered formula only when the
        # user directly names that metric or one of its governed aliases.
        hit = any(
            name and str(name).lower() in query_text
            for name in names
        ) or (
            bool(_semantic_family(semantic_ir))
            and rule_type == _semantic_family(semantic_ir)
        )
        if not hit:
            continue
        whole_tree_rule_types = {
            "repeat_rate", "share", "activation_rate", "period_growth",
            "growth_rate", "cohort_count", "anti_join",
            "inactivity_cohort",
        }
        matches = (
            _formula_matches(tree, rule, aliases)
            if rule_type in whole_tree_rule_types
            else any(
                _formula_matches_with_lineage(
                    projection,
                    rule,
                    projection_aliases or aliases,
                    definitions,
                )
                or _conditional_count_matches_with_lineage(
                    projection,
                    rule,
                    aliases,
                    definitions,
                    semantic_ir,
                )
                for projection, projection_aliases in projection_scopes
            )
        )
        population_mismatch = False
        if (
            matches
            and rule_type == "anti_join"
            and _anti_join_count_requested(query_text, semantic_ir)
        ):
            matches = _anti_join_population_matches(tree, rule, aliases)
            population_mismatch = not matches
        if (
            not matches
            and not population_mismatch
            and rule_type != "repeat_rate"
            and _metric_is_relational_operand(query_text, names)
        ):
            # The formula is still mandatory, but may be consumed by an inner
            # relational operator rather than exposed as a final column.
            matches = _formula_matches(tree, rule, aliases)
        if not matches:
            metric_name = metric.get("name") or "未命名指标"
            raise MetricFormulaError(
                f"指标“{metric_name}”应遵循公式 {rule}，当前 SQL 未检测到对应计算"
            )
