"""
NL2SQL 评测指标计算（TODO-23）。

一句话总述：对生成 SQL 与期望 SQL 计算多维指标——语法正确率、SQL 等价率、表选择准确率、
            结果一致率、执行成功率等，供 evaluator 聚合为整体与分类别报告。

设计说明：
    - SQL 等价为"近似等价"：sqlglot 归一化（去注释/统一格式/小写）后字符串相等，
      容许空格/换行/注释差异，但别名差异会判为不等价（保守估计，真值 ≥ 该值）。
    - 结果一致按行集比对：期望 SQL 也在 DW 执行，比较两份结果集；
      仅当用户问题明确要求排序、排名或 Top/Bottom 时才按行序敏感比对。
      Reference SQL 为了产物可复现而带有 ORDER BY，不构成业务顺序契约。
"""
import datetime as dt
import math
import re
from decimal import Decimal, InvalidOperation
from typing import Any

import sqlglot
from sqlglot import exp

from app.config.meta_config import meta_config
from app.domain.query.safety.sql_security import extract_table_names


RESULT_EQUIVALENCE_CONTRACT_VERSION = 9


def syntax_valid(sql: str) -> bool:
    """生成 SQL 能否被 sqlglot 解析（语法正确性）。"""
    if not sql:
        return False
    try:
        sqlglot.parse_one(sql, read="mysql")
        return True
    except Exception:
        return False


def _normalize_sql(sql: str) -> str:
    """归一化 SQL 字符串：去注释、统一输出、小写，便于近似等价比对。"""
    try:
        tree = sqlglot.parse_one(sql, read="mysql")
        return tree.sql(dialect="mysql", comments=False, pad=0, indent=0).lower()
    except Exception:
        return sql.strip().lower()


def sql_equivalent(gen_sql: str, expected_sql: str) -> bool:
    """两 SQL 归一化后字符串是否相等（近似等价，保守估计）。"""
    return _normalize_sql(gen_sql) == _normalize_sql(expected_sql)


def extract_tables(sql: str) -> set:
    """提取 SQL 涉及的表名集合（复用 sql_security 的 sqlglot 实现）。"""
    try:
        return set(extract_table_names(sql))
    except Exception:
        return set()


def tables_correct(gen_sql: str, expected_sql: str) -> bool:
    """生成 SQL 与期望 SQL 涉及的表名集合是否一致。"""
    return extract_tables(gen_sql) == extract_tables(expected_sql)


def tables_match_contract(gen_sql: str, required_tables: list[str] | None,
                          allowed_tables: list[str] | None) -> bool:
    """Accept an explicitly reviewed alternative table path."""
    generated = {name.lower() for name in extract_tables(gen_sql)}
    required = {str(name).lower() for name in (required_tables or [])}
    allowed = {str(name).lower() for name in (allowed_tables or [])}
    if not required or not allowed:
        return False
    return required.issubset(generated) and generated.issubset(allowed)


def table_prf(gen_sql: str, expected_sql: str) -> dict[str, float]:
    """Return table precision/recall/F1 instead of hiding partial matches behind one bool."""
    generated, expected = extract_tables(gen_sql), extract_tables(expected_sql)
    if not generated and not expected:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    overlap = len(generated & expected)
    precision = overlap / len(generated) if generated else 0.0
    recall = overlap / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def table_contract_prf(gen_sql: str, required_tables: list[str] | None,
                       allowed_tables: list[str] | None) -> dict[str, float]:
    generated = {name.lower() for name in extract_tables(gen_sql)}
    required = {str(name).lower() for name in (required_tables or [])}
    allowed = {str(name).lower() for name in (allowed_tables or [])}
    if not required or not allowed:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = len(generated & allowed) / len(generated) if generated else 0.0
    recall = len(generated & required) / len(required)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def sql_structure_equivalent(gen_sql: str, expected_sql: str) -> bool:
    """Compare stable query structure while tolerating aliases and formatting.

    This is intentionally stricter than result equality and looser than normalized SQL equality.
    It checks tables and the presence/count of joins, aggregates, grouping, ordering, limits,
    predicates, CTEs and set operations.
    """
    try:
        generated = sqlglot.parse_one(gen_sql, read="mysql")
        expected = sqlglot.parse_one(expected_sql, read="mysql")
    except Exception:
        return False

    def signature(tree: exp.Expression) -> tuple:
        limit = tree.args.get("limit")
        limit_value = None
        if limit is not None and limit.expression is not None:
            limit_value = str(limit.expression)
        return (
            tuple(sorted(name.lower() for name in extract_tables(tree.sql(dialect="mysql")))),
            sum(1 for _ in tree.find_all(exp.Join)),
            tuple(sorted(node.sql_name().upper() for node in tree.find_all(exp.AggFunc))),
            sum(1 for _ in tree.find_all(exp.Group)),
            sum(1 for _ in tree.find_all(exp.Order)),
            sum(1 for _ in tree.find_all(exp.Where)),
            sum(1 for _ in tree.find_all(exp.Having)),
            sum(1 for _ in tree.find_all(exp.CTE)),
            sum(1 for _ in tree.find_all(exp.SetOperation)),
            limit_value,
        )

    return signature(generated) == signature(expected)


_EXPLICIT_RESULT_ORDER_PATTERNS = (
    # Direct ordering/ranking language.  “顺序” alone is intentionally not a
    # signal because it is also common in business field names; require a verb
    # or an explicit “按……顺序” construction for that word.
    re.compile(
        r"(?:升序|降序|从高到低|由高到低|从低到高|由低到高|"
        r"从大到小|由大到小|从小到大|由小到大|"
        r"递增(?:顺序)?|递减(?:顺序)?|倒序|正序|排序|排列)"
    ),
    re.compile(r"(?:按|依|以).{0,32}?顺序"),
    re.compile(r"(?:输出|结果|展示).{0,16}?顺序"),
    # English SQL questions and mixed Chinese/English prompts.
    re.compile(
        r"\b(?:sort(?:ed|ing)?|ascending|descending)\b|"
        r"\border(?:ed)?\s+by\b|"
        r"\bin\s+(?:ascending|descending)\s+order\b",
        re.IGNORECASE,
    ),
)


def question_requires_result_order(question: str | None) -> bool:
    """Return whether row order is an explicit part of the user's request.

    SQL text and legacy dataset flags are deliberately excluded from this
    decision.  They describe an implementation/fixture, not necessarily the
    observable business answer requested by the user.
    """

    text = re.sub(r"\s+", " ", str(question or "")).strip()
    return bool(text) and any(
        pattern.search(text) for pattern in _EXPLICIT_RESULT_ORDER_PATTERNS
    )


_DECIMAL_TEXT_RE = re.compile(
    r"^[+-]?(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?$|^[+-]?\d+[eE][+-]?\d+$"
)
_NUMERIC_REL_TOL = 1e-6
_NUMERIC_ABS_TOL = 1e-9
_NUMERIC_SLOT = "<numeric>"


def _normal_value(value: Any) -> Any:
    """Normalize DB scalar types without losing numeric meaning."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        # The query repository serialises DECIMAL as text so browsers do not
        # lose precision.  MySQL may choose a different display scale for two
        # mathematically identical expressions (``0.0000`` vs ``0.00000``).
        # Canonicalise only text that is unambiguously decimal/scientific;
        # integer-looking identifiers such as ``001`` deliberately stay text.
        text = value.strip()
        if _DECIMAL_TEXT_RE.fullmatch(text):
            try:
                return round(float(Decimal(text)), 9)
            except (ArithmeticError, ValueError):
                pass
        return value
    if isinstance(value, (float, Decimal)):
        number = float(value)
        if math.isnan(number):
            return "NaN"
        if math.isinf(number):
            return "Infinity" if number > 0 else "-Infinity"
        return round(number, 9)
    if isinstance(value, (dt.date, dt.datetime, dt.time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def _row_key(row: dict) -> tuple:
    """Stable named-row key with case-insensitive column names."""
    return tuple((str(k).lower(), _normal_value(row.get(k))) for k in sorted(row.keys(), key=lambda item: str(item).lower()))


def _numeric_for_tolerance(value: Any) -> float | None:
    """Return an analytical number without coercing integer-like identifiers.

    Decimal/scientific strings are numerical because the query repository
    serialises MySQL DECIMAL values as text.  Integer-looking strings stay text,
    so identifiers such as ``001`` remain exact.  Plain Python integers are
    admitted for comparisons with a Decimal/float counterpart, but two unequal
    integers are still compared exactly by :func:`_values_equivalent`.
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return float(value)
    if isinstance(value, (float, Decimal)):
        number = float(value)
    elif isinstance(value, str) and _DECIMAL_TEXT_RE.fullmatch(value.strip()):
        try:
            number = float(Decimal(value.strip()))
        except (ArithmeticError, ValueError):
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def _values_equivalent(left: Any, right: Any) -> bool:
    """Compare one result value, tolerating only insignificant numeric drift."""

    normalized_left = _normal_value(left)
    normalized_right = _normal_value(right)
    if normalized_left == normalized_right:
        return True
    # Distinct integer values can be identifiers or exact counts and must never
    # become equal merely because their magnitude makes the relative delta small.
    if (
        isinstance(left, int) and not isinstance(left, bool)
        and isinstance(right, int) and not isinstance(right, bool)
    ):
        return False
    left_number = _numeric_for_tolerance(left)
    right_number = _numeric_for_tolerance(right)
    return bool(
        left_number is not None
        and right_number is not None
        and math.isclose(
            left_number,
            right_number,
            rel_tol=_NUMERIC_REL_TOL,
            abs_tol=_NUMERIC_ABS_TOL,
        )
    )


def _row_items(row: dict, *, positional: bool) -> list[tuple[str, Any]]:
    if positional:
        return [(str(index), value) for index, value in enumerate(row.values())]
    return [
        (str(key).lower(), row.get(key))
        for key in sorted(row.keys(), key=lambda item: str(item).lower())
    ]


def _rows_equivalent(left: dict, right: dict, *, positional: bool) -> bool:
    left_items = _row_items(left, positional=positional)
    right_items = _row_items(right, positional=positional)
    return (
        len(left_items) == len(right_items)
        and all(
            left_key == right_key and _values_equivalent(left_value, right_value)
            for (left_key, left_value), (right_key, right_value)
            in zip(left_items, right_items)
        )
    )


def _row_tolerance_group(row: dict, *, positional: bool) -> tuple:
    """Group unordered rows by exact non-numeric content before tolerant match."""

    return tuple(
        (
            key,
            _NUMERIC_SLOT
            if _numeric_for_tolerance(value) is not None
            else _normal_value(value),
        )
        for key, value in _row_items(row, positional=positional)
    )


def _row_numeric_sort_key(row: dict, *, positional: bool) -> tuple:
    return tuple(
        number
        for _, value in _row_items(row, positional=positional)
        if (number := _numeric_for_tolerance(value)) is not None
    )


def _unordered_rows_equivalent(
    generated: list[dict], expected: list[dict], *, positional: bool,
) -> bool:
    """Compare an unordered multiset with bounded numeric-tolerance grouping."""

    generated_groups: dict[tuple, list[dict]] = {}
    expected_groups: dict[tuple, list[dict]] = {}
    for row in generated:
        generated_groups.setdefault(
            _row_tolerance_group(row, positional=positional), []
        ).append(row)
    for row in expected:
        expected_groups.setdefault(
            _row_tolerance_group(row, positional=positional), []
        ).append(row)
    if generated_groups.keys() != expected_groups.keys():
        return False
    for group, generated_rows in generated_groups.items():
        expected_rows = expected_groups[group]
        if len(generated_rows) != len(expected_rows):
            return False
        generated_rows.sort(
            key=lambda row: _row_numeric_sort_key(row, positional=positional)
        )
        expected_rows.sort(
            key=lambda row: _row_numeric_sort_key(row, positional=positional)
        )
        if not all(
            _rows_equivalent(left, right, positional=positional)
            for left, right in zip(generated_rows, expected_rows)
        ):
            return False
    return True


def _compare_same_projection(
    gen_rows: list[dict], expected_rows: list[dict], *, order_sensitive: bool,
) -> tuple[bool, str]:
    """Compare equal-width result projections and return an audit mode."""

    if len(gen_rows) != len(expected_rows):
        return False, "row_count_mismatch"
    if order_sensitive:
        gen_keys = [_row_key(row) for row in gen_rows]
        expected_keys = [_row_key(row) for row in expected_rows]
        if gen_keys == expected_keys:
            return True, "ordered_named_exact"
        if all(
            _rows_equivalent(left, right, positional=False)
            for left, right in zip(gen_rows, expected_rows)
        ):
            return True, "ordered_named_tolerance"
        if all(
            _rows_equivalent(left, right, positional=True)
            for left, right in zip(gen_rows, expected_rows)
        ):
            return True, "ordered_positional_alias"
        return False, "ordered_value_mismatch"

    gen_sorted = sorted((_row_key(row) for row in gen_rows), key=repr)
    expected_sorted = sorted((_row_key(row) for row in expected_rows), key=repr)
    if gen_sorted == expected_sorted:
        return True, "unordered_named_exact"
    if _unordered_rows_equivalent(gen_rows, expected_rows, positional=False):
        return True, "unordered_named_tolerance"
    if _unordered_rows_equivalent(gen_rows, expected_rows, positional=True):
        return True, "unordered_positional_alias"
    return False, "unordered_value_mismatch"


_EXISTENCE_QUESTION_RE = re.compile(
    r"(?:是否\s*(?:存在|有)|有没有|有无|\b(?:does|do|is|are)\b.{0,32}\bexist)",
    re.IGNORECASE,
)
_EXPLICIT_QUANTITY_OUTPUT_RE = re.compile(
    r"(?:"
    r"多少(?:个|位|名|家|种|条|笔|件)?|"
    r"(?:返回|给出|显示|展示|输出|统计|计算|告诉).{0,16}"
    r"(?:数量|数目|总数|个数|人数|笔数|件数|条数|记录数|客户数|用户数|商品数|产品数|订单数)|"
    r"(?:数量|数目|总数|个数|人数|笔数|件数|条数|记录数|客户数|用户数|商品数|产品数|订单数|计数|合计数)"
    r".{0,8}(?:是多少|为多少|有多少|结果)|"
    r"\b(?:how\s+many|counts?|number\s+of|total\s+count)\b"
    r")",
    re.IGNORECASE,
)
_EXISTENCE_CARDINALITY_RE = re.compile(
    r"(?:"
    r"(?:至少|至多|不少于|不多于|多于|少于|超过|低于|恰好|正好|仅有|只有)"
    r"\s*(?:\d+|[一二两三四五六七八九十百]+)\s*"
    r"(?:个|位|名|家|种|条|笔|件|台|项|组|批)?|"
    r"(?:多|若干)(?:个|位|名|家|种|条|笔|件|台|项|组|批)|"
    r"\b(?:at\s+least|at\s+most|more\s+than|fewer\s+than|exactly)\s+\d+\b"
    r")",
    re.IGNORECASE,
)
_EXISTENCE_OUTPUT_VERB_RE = re.compile(
    r"(?:返回|列出|显示|展示|给出|输出|包含|"
    r"\b(?:return|list|show|display|include|output)\b)",
    re.IGNORECASE,
)
_BOOLEAN_RESULT_ONLY_RE = re.compile(
    r"^[\s，,：:；;]*(?:判断|判断结果|结果|结论|布尔值|真假|yes\s*/?\s*no)"
    r"[\s。.!！?？]*$",
    re.IGNORECASE,
)


def _final_projection(sql: str) -> list[exp.Expression] | None:
    """返回最终可见投影；解析失败或集合操作时关闭式拒绝。"""

    try:
        tree = sqlglot.parse_one(str(sql or ""), read="mysql")
    except Exception:
        return None
    if not isinstance(tree, exp.Select):
        return None
    return list(tree.expressions)


def _unwrap_projection(expression: exp.Expression) -> exp.Expression:
    current = expression
    while isinstance(current, (exp.Alias, exp.Paren)):
        current = current.this
    return current


def _is_zero_literal(expression: exp.Expression) -> bool:
    value = _unwrap_projection(expression)
    if not isinstance(value, exp.Literal) or value.is_string:
        return False
    try:
        return Decimal(str(value.this)) == 0
    except (InvalidOperation, ValueError):
        return False


def _is_direct_count(expression: exp.Expression) -> bool:
    return isinstance(_unwrap_projection(expression), exp.Count)


def _existence_question_reason(question: str) -> str:
    """只允许单一 yes/no 存在意图进入跨表示归一。"""

    query = re.sub(r"\s+", " ", str(question or "")).strip()
    existence_match = _EXISTENCE_QUESTION_RE.search(query)
    if existence_match is None:
        return "not_an_existence_question"
    if _EXPLICIT_QUANTITY_OUTPUT_RE.search(query):
        return "explicit_quantity_contract"
    if _EXISTENCE_CARDINALITY_RE.search(query):
        return "cardinality_threshold_contract"
    for output_match in _EXISTENCE_OUTPUT_VERB_RE.finditer(query):
        requested = query[output_match.end():]
        if (
            output_match.start() >= existence_match.start()
            and _BOOLEAN_RESULT_ONLY_RE.fullmatch(requested)
        ):
            continue
        return "explicit_non_boolean_output_contract"
    return "pure_existence_question"


def _existence_scalar_kind(sql: str) -> str | None:
    """证明单列 SQL 是 COUNT 或由 COUNT/EXISTS 产生的布尔存在值。"""

    projections = _final_projection(sql)
    if projections is None or len(projections) != 1:
        return None
    expression = _unwrap_projection(projections[0])
    if isinstance(expression, exp.Count):
        return "count"
    if isinstance(expression, exp.Exists):
        return "boolean"
    left = expression.this if isinstance(
        expression, (exp.GT, exp.LT, exp.NEQ),
    ) else None
    right = expression.expression if isinstance(
        expression, (exp.GT, exp.LT, exp.NEQ),
    ) else None
    if isinstance(expression, exp.GT) and left is not None and right is not None:
        if _is_direct_count(left) and _is_zero_literal(right):
            return "boolean"
    if isinstance(expression, exp.LT) and left is not None and right is not None:
        if _is_zero_literal(left) and _is_direct_count(right):
            return "boolean"
    if isinstance(expression, exp.NEQ) and left is not None and right is not None and (
        (_is_direct_count(left) and _is_zero_literal(right))
        or (_is_zero_literal(left) and _is_direct_count(right))
    ):
        return "boolean"
    return None


def _is_witness_projection(sql: str) -> bool:
    """证明 SQL 返回实体见证行，而不是另一个聚合或常量标志。"""

    projections = _final_projection(sql)
    if not projections:
        return False
    if any(
        projection.find(exp.AggFunc) is not None
        or projection.find(exp.Exists) is not None
        for projection in projections
    ):
        return False
    return any(
        projection.find(exp.Column) is not None
        or projection.find(exp.Star) is not None
        for projection in projections
    )


def _count_truth(rows: list[dict]) -> bool | None:
    if len(rows) != 1 or len(rows[0]) != 1:
        return None
    value = next(iter(rows[0].values()))
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not re.fullmatch(r"[+]?[0-9]+", text):
            return None
        number = int(text)
    elif isinstance(value, (int, Decimal)):
        if isinstance(value, Decimal) and value != value.to_integral_value():
            return None
        number = int(value)
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        number = int(value)
    else:
        return None
    return number > 0 if number >= 0 else None


def _boolean_truth(rows: list[dict]) -> bool | None:
    if len(rows) != 1 or len(rows[0]) != 1:
        return None
    value = next(iter(rows[0].values()))
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
        return None
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
    return None


def _existence_equivalence_evidence(
    gen_rows: list[dict],
    expected_rows: list[dict],
    *,
    generated_sql: str | None,
    expected_sql: str,
    question: str,
) -> dict[str, Any]:
    """只在存在性题的“标量存在值 ↔ 见证行”之间归一化真值。"""

    evidence: dict[str, Any] = {
        "attempted": False,
        "eligible": False,
        "equivalent": False,
        "reason": "not_an_existence_question",
        "generated_representation": None,
        "reference_representation": None,
        "generated_truth": None,
        "reference_truth": None,
    }
    question_reason = _existence_question_reason(question)
    if question_reason == "not_an_existence_question":
        return evidence
    evidence["attempted"] = True
    if question_reason != "pure_existence_question":
        evidence["reason"] = question_reason
        return evidence

    generated_kind = _existence_scalar_kind(str(generated_sql or ""))
    reference_kind = _existence_scalar_kind(expected_sql)
    generated_witness = _is_witness_projection(str(generated_sql or ""))
    reference_witness = _is_witness_projection(expected_sql)
    if bool(generated_kind) == bool(reference_kind):
        evidence["reason"] = "requires_one_scalar_and_one_witness_projection"
        return evidence
    if generated_kind and not reference_witness:
        evidence["reason"] = "reference_is_not_a_witness_projection"
        return evidence
    if reference_kind and not generated_witness:
        evidence["reason"] = "generated_is_not_a_witness_projection"
        return evidence

    generated_representation = generated_kind or "witness_rows"
    reference_representation = reference_kind or "witness_rows"
    generated_truth = (
        _count_truth(gen_rows)
        if generated_kind == "count"
        else _boolean_truth(gen_rows)
        if generated_kind == "boolean"
        else bool(gen_rows)
    )
    reference_truth = (
        _count_truth(expected_rows)
        if reference_kind == "count"
        else _boolean_truth(expected_rows)
        if reference_kind == "boolean"
        else bool(expected_rows)
    )
    evidence.update({
        "generated_representation": generated_representation,
        "reference_representation": reference_representation,
        "generated_truth": generated_truth,
        "reference_truth": reference_truth,
    })
    if generated_truth is None or reference_truth is None:
        evidence["reason"] = "invalid_scalar_existence_result"
        return evidence
    evidence.update({
        "eligible": True,
        "equivalent": generated_truth is reference_truth,
        "reason": (
            "existence_truth_match"
            if generated_truth is reference_truth
            else "existence_truth_mismatch"
        ),
    })
    return evidence


def _normalized_identifier(value: Any) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value or "").lower())


def _identifier_tokens(value: Any) -> set[str]:
    return set(_identifier_token_sequence(value))


def _identifier_token_sequence(value: Any) -> list[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value or ""))
    return [
        token.lower()
        for token in re.split(r"[^0-9A-Za-z\u4e00-\u9fff]+", text)
        if token
    ]


def _config_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _build_projection_vocabulary() -> tuple[
    dict[str, set[str]], dict[str, set[str]], list[dict[str, Any]],
]:
    """Build stable semantic vocabulary from governed metadata, not fixtures."""

    column_terms: dict[str, set[str]] = {}
    for table in meta_config.tables:
        for column in table.columns:
            column_name = str(column.name).lower()
            terms = column_terms.setdefault(column_name, set())
            terms.add(str(column.name))
            terms.update(str(alias) for alias in (column.alias or []))

    metric_terms: dict[str, set[str]] = {}
    metric_signatures: list[dict[str, Any]] = []
    for metric in meta_config.metrics:
        metric_name = str(metric.name)
        terms = {metric_name, *(str(alias) for alias in (metric.alias or []))}
        formula = metric.formula_rule or {}
        result_alias = str(_config_value(formula, "result_alias", "") or "")
        if result_alias:
            terms.add(result_alias)
        metric_terms[metric_name] = terms
        column = str(_config_value(formula, "column", "") or "")
        function = str(_config_value(formula, "function", "") or "").upper()
        if not column:
            relevant = list(metric.relevant_columns or [])
            if len(relevant) == 1:
                column = str(relevant[0])
        metric_signatures.append({
            "name": metric_name,
            "column": column.rsplit(".", 1)[-1].lower(),
            "function": function,
            "formula_type": str(_config_value(formula, "type", "") or ""),
            "result_alias": result_alias,
        })
    return column_terms, metric_terms, metric_signatures


_COLUMN_TERMS, _METRIC_TERMS, _METRIC_SIGNATURES = (
    _build_projection_vocabulary()
)

# Generic language that is stable across SQL suites but is intentionally not
# injected into the retrieval metadata.  These terms only help interpret a
# result-column label and whether the user asked to see it.
_COMMON_COLUMN_TERMS = {
    "category": {"类别"},
    "customer_name": {"客户", "用户", "买家"},
}
for _column_name, _terms in _COMMON_COLUMN_TERMS.items():
    _COLUMN_TERMS.setdefault(_column_name, set()).update(_terms)

_OUTPUT_TOKEN_COLUMNS = {
    "amount": "order_amount",
    "revenue": "order_amount",
    "gmv": "order_amount",
    "quantity": "order_quantity",
    "qty": "order_quantity",
    "category": "category",
    "brand": "brand",
    "region": "region_name",
    "province": "province",
    "country": "country",
    "gender": "gender",
    "year": "year",
    "quarter": "quarter",
    "month": "month",
}

_DERIVED_TOKEN_ALIASES = {
    "count": "count",
    "cnt": "count",
    "number": "count",
    "num": "count",
    "total": "total",
    "sum": "total",
    "avg": "average",
    "average": "average",
    "mean": "average",
    "min": "minimum",
    "minimum": "minimum",
    "lowest": "minimum",
    "max": "maximum",
    "maximum": "maximum",
    "highest": "maximum",
    "rate": "ratio",
    "ratio": "ratio",
    "share": "ratio",
    "percentage": "ratio",
    "percent": "ratio",
    "pct": "ratio",
    "rank": "rank",
    "ranking": "rank",
    "percentile": "rank",
    "quartile": "bucket",
    "decile": "bucket",
    "bucket": "bucket",
    "band": "bucket",
    "tier": "bucket",
    "stddev": "dispersion",
    "std": "dispersion",
    "deviation": "dispersion",
    "variance": "dispersion",
    "concentration": "concentration",
    "orders": "count",
    "customers": "count",
    "products": "count",
    "records": "count",
    "previous": "previous",
    "prev": "previous",
    "prior": "previous",
    "current": "current",
    "growth": "growth",
    "change": "change",
    "difference": "change",
    "diff": "change",
}

_DERIVED_QUESTION_PATTERNS = {
    "count": re.compile(
        r"(?:多少|几(?:个|名|笔|条|种)|数量|数目|总数|计数|笔数|人数|"
        r"(?:客户|用户|订单|商品|产品|记录|品牌|类别)数|"
        r"\b(?:count|number\s+of|how\s+many)\b)",
        re.IGNORECASE,
    ),
    "total": re.compile(
        r"(?:合计|总计|总额|总量|总和|累计|\b(?:sum|total)\b)",
        re.IGNORECASE,
    ),
    "average": re.compile(r"(?:平均|均值|\b(?:avg|average|mean)\b)", re.IGNORECASE),
    "minimum": re.compile(r"(?:最低|最小|\b(?:min|minimum|lowest)\b)", re.IGNORECASE),
    "maximum": re.compile(r"(?:最高|最大|\b(?:max|maximum|highest)\b)", re.IGNORECASE),
    "ratio": re.compile(
        r"(?:占比|比例|比率|百分比|贡献率|增长率|[\u4e00-\u9fff]{1,12}率|"
        r"\b(?:rate|ratio|share|percent(?:age)?)\b)",
        re.IGNORECASE,
    ),
    "previous": re.compile(r"(?:上期|上一期|上月|上季|上年|上一年|去年|同期|\b(?:previous|prior)\b)", re.IGNORECASE),
    "current": re.compile(r"(?:本期|当期|当前|本月|本季|本年|今年|\bcurrent\b)", re.IGNORECASE),
    "growth": re.compile(r"(?:增长|同比|环比|\bgrowth\b)", re.IGNORECASE),
    "change": re.compile(r"(?:变化|差值|差额|增减|\b(?:change|difference|diff)\b)", re.IGNORECASE),
    "bucket": re.compile(r"(?:分为?\s*[四十]档|分档|分组档位|\b(?:quartile|decile|bucket|band)\b)", re.IGNORECASE),
    "dispersion": re.compile(r"(?:标准差|方差|离散度|\b(?:stddev|standard\s+deviation|variance)\b)", re.IGNORECASE),
    "concentration": re.compile(r"(?:集中度|\bconcentration\b)", re.IGNORECASE),
}

_EXPLICIT_RANK_OUTPUT = re.compile(
    r"(?:(?:返回|列出|显示|展示|给出|包含|输出).{0,16}?(?:排名|名次)|"
    r"(?:排名|名次)(?:列|字段|值|是多少|情况|结果|明细)|"
    r"\b(?:show|return|include|display).{0,24}?\b(?:rank|ranking)\b)",
    re.IGNORECASE,
)

_IDENTIFIER_TERM = re.compile(r"(?:id|编号|标识|编码|号码|号)$", re.IGNORECASE)
_COUNT_SEMANTIC_TERM = re.compile(
    r"(?:数|数量|笔数|件数|人数|个数|\bcount\b)$",
    re.IGNORECASE,
)

_CRITERION_AFTER_TERM = re.compile(
    # An aggregate operator can precede its governed measure label, e.g.
    # ``平均 + 订单金额 + 超过``.  Keep the bridge local and punctuation-bounded
    # so an independent output in the next clause is not swallowed as a filter.
    r"^\s*[^，,。；;]{0,16}?(?:大于|小于|高于|低于|超过|少于|不少于|不多于|不低于|不高于|"
    r"达到|未达到|等于|不等于|为|是|>=|<=|<>|!=|=|>|<)"
)
_RANK_CRITERION_AFTER_TERM = re.compile(
    r"^\s*(?:最高|最低|最大|最小|最多|最少|前\s*[一二三四五六七八九十百\d]+|"
    r"后\s*[一二三四五六七八九十百\d]+|倒数\s*[一二三四五六七八九十百\d]+)"
)
_ORDER_CRITERION_AFTER_TERM = re.compile(
    r"^\s*(?:升序|降序|从高到低|从低到高|由高到低|由低到高|排序|排列)"
)
_DERIVED_QUALIFIER_AFTER_TERM = re.compile(
    r"^\s*(?:的\s*)?(?:"
    r"(?:相对.{0,20})?(?:同比|环比|增长率|占比|比例|比率|贡献率|"
    r"集中度|百分位排名|变化|差值|差额|增减)|"
    r"[^，,。；;及和与]{0,12}(?:变化|差值|差额|增减)|"
    r"占[^，,。；;]{0,24}(?:占比|比例|比率|百分比|份额)"
    r")"
)


def _term_spans(question: str, term: str) -> list[tuple[int, int]]:
    """Locate a governed term while retaining spans for longest-term disambiguation."""

    needle = str(term or "").strip()
    if not needle:
        return []
    escaped = re.escape(needle)
    if re.fullmatch(r"[0-9A-Za-z_ ]+", needle):
        escaped = escaped.replace(r"\ ", r"\s+")
        pattern = rf"(?<![0-9A-Za-z_]){escaped}(?![0-9A-Za-z_])"
    else:
        pattern = escaped
    return [match.span() for match in re.finditer(pattern, question, re.IGNORECASE)]


def _maximal_question_concepts(question: str) -> dict[str, set[str]]:
    """Return concepts named by maximal governed terms in the question.

    A broad alias cannot prove output intent when the same text is covered by a
    longer, different metadata term: e.g. ``商品`` inside ``商品类别`` or
    ``地区`` inside ``地区ID``.  Independent occurrences remain valid.
    """

    matches: list[tuple[int, int, str, str]] = []
    for column_name, terms in _COLUMN_TERMS.items():
        for term in terms:
            matches.extend(
                (start, end, f"column:{column_name}", str(term))
                for start, end in _term_spans(question, term)
            )
    for metric_name, terms in _METRIC_TERMS.items():
        for term in terms:
            matches.extend(
                (start, end, f"metric:{metric_name}", str(term))
                for start, end in _term_spans(question, term)
            )
    maximal: dict[str, set[str]] = {}
    for start, end, concept, term in matches:
        shadowed = any(
            other_start <= start
            and other_end >= end
            and (other_end - other_start) > (end - start)
            for other_start, other_end, _, _ in matches
        )
        if not shadowed:
            maximal.setdefault(concept, set()).add(term)
    return maximal


def _span_has_output_role(
    question: str,
    start: int,
    end: int,
    descriptor: dict[str, Any],
) -> bool:
    """Distinguish an answer field from a filter/order/rank criterion."""

    head = question[max(0, start - 16):start]
    tail = question[end:end + 48]
    if _CRITERION_AFTER_TERM.match(tail):
        return False
    if _RANK_CRITERION_AFTER_TERM.match(tail):
        # ``销售额最高的地区`` asks for the region membership, whereas
        # ``最高销售额`` puts the operator before the requested measure.
        return False
    if re.search(r"(?:按|依|以)\s*$", head) and _ORDER_CRITERION_AFTER_TERM.match(tail):
        return False
    descriptor_concepts = set(descriptor["concepts"])
    if (
        _DERIVED_QUALIFIER_AFTER_TERM.match(tail)
        and not descriptor_concepts.intersection({
            "derived:ratio",
            "derived:growth",
            "derived:concentration",
            "derived:rank",
        })
    ):
        return False
    # A field feeding a comparison to an average is a criterion unless the
    # question independently names that average as an output elsewhere.
    if re.search(r"(?:大于|小于|高于|低于|超过|不少于|不高于).{0,8}$", head):
        return False
    if re.match(r"^\s*(?:中|内|范围内)", tail):
        return False
    # A governed dimension modified by an exclusion is a filter value, not an
    # answer grain.  The bound deliberately stops at clause punctuation.
    if re.search(
        r"(?:^|[，,；;\s中内])(?:非|不含|排除|剔除)[^，,。；;]{0,12}$",
        head,
    ):
        return False
    return True


def _terms_have_output_role(
    question: str,
    terms: set[str],
    descriptor: dict[str, Any],
) -> bool:
    return any(
        _span_has_output_role(question, start, end, descriptor)
        for term in terms
        for start, end in _term_spans(question, term)
    )


def _pattern_has_output_role(
    question: str,
    pattern: re.Pattern,
    descriptor: dict[str, Any],
) -> bool:
    return any(
        _span_has_output_role(question, *match.span(), descriptor)
        for match in pattern.finditer(question)
    )


def _label_concepts(label: str) -> set[str]:
    normalized = _normalized_identifier(label)
    concepts: set[str] = set()
    for column_name, terms in _COLUMN_TERMS.items():
        if normalized in {_normalized_identifier(term) for term in terms}:
            concepts.add(f"column:{column_name}")
    for metric_name, terms in _METRIC_TERMS.items():
        if normalized in {_normalized_identifier(term) for term in terms}:
            concepts.add(f"metric:{metric_name}")
    for token in _identifier_tokens(label):
        if token in _DERIVED_TOKEN_ALIASES:
            concepts.add(f"derived:{_DERIVED_TOKEN_ALIASES[token]}")
        if token in _OUTPUT_TOKEN_COLUMNS:
            concepts.add(f"column:{_OUTPUT_TOKEN_COLUMNS[token]}")
        if token in _COLUMN_TERMS:
            concepts.add(f"column:{token}")
    return concepts


def _label_output_roles(label: str, concepts: set[str]) -> set[str]:
    """Return the semantic head of a result label.

    Compound aliases contain both the answer role and lineage qualifiers.  For
    example, ``region_avg_province_amount`` depends on region and province, but
    its answer role is an average amount.  Treating every token as an output
    dimension creates false ambiguity when mapping equivalent aliases.
    """

    token_roles: list[str] = []
    derived_roles: list[str] = []
    for token in _identifier_token_sequence(label):
        role = None
        if token in _DERIVED_TOKEN_ALIASES:
            role = f"derived:{_DERIVED_TOKEN_ALIASES[token]}"
            derived_roles.append(role)
        elif token in _OUTPUT_TOKEN_COLUMNS:
            role = f"column:{_OUTPUT_TOKEN_COLUMNS[token]}"
        elif token in _COLUMN_TERMS:
            role = f"column:{token}"
        if role:
            token_roles.append(role)
    if not token_roles:
        return set(concepts)
    final_role = token_roles[-1]
    if final_role.startswith("derived:"):
        # ``previous_customer_count`` and ``current_customer_count`` are both
        # counts, but their temporal qualifier is part of the answer role.  If
        # it is discarded, an unqualified active-customer count becomes
        # ambiguously mappable to either column and a correct minimal answer is
        # rejected.  Retain only directional qualifiers here; lineage operators
        # such as SUM/AVG remain governed by the final semantic head.
        directional = {
            role
            for role in derived_roles
            if role in {"derived:previous", "derived:current"}
        }
        return {final_role, *directional}
    # Aggregate/ratio operators qualify a trailing measure noun and remain part
    # of its answer role (``avg_amount`` versus ``total_amount``).
    return {final_role, *derived_roles}


def _select_descriptors(sql: str, result_columns: list[str]) -> list[dict[str, Any]]:
    try:
        tree = sqlglot.parse_one(sql, read="mysql")
        selections = list(tree.selects)
    except Exception:
        selections = []
    aligned = len(selections) == len(result_columns)
    descriptors: list[dict[str, Any]] = []
    for index, output_name in enumerate(result_columns):
        selection = selections[index] if aligned else None
        columns = {
            str(column.name).lower()
            for column in selection.find_all(exp.Column)
        } if selection is not None else set()
        aggregates = {
            str(aggregate.sql_name()).upper()
            for aggregate in selection.find_all(exp.AggFunc)
        } if selection is not None else set()
        rank_output = bool(
            selection is not None
            and any(
                selection.find(node_type) is not None
                for node_type in (exp.RowNumber, exp.Rank, exp.DenseRank)
            )
        )
        concepts = _label_concepts(output_name)
        for column_name in columns:
            if column_name in _COLUMN_TERMS:
                concepts.add(f"column:{column_name}")
        for signature in _METRIC_SIGNATURES:
            signature_column = str(signature["column"] or "")
            signature_function = str(signature["function"] or "")
            alias_match = bool(
                signature["result_alias"]
                and _normalized_identifier(signature["result_alias"])
                == _normalized_identifier(output_name)
            )
            expression_match = bool(
                signature_column
                and signature_column in columns
                and signature_function in aggregates
                and signature["formula_type"] in {"aggregate", "distinct_count"}
            )
            if alias_match or expression_match:
                concepts.add(f"metric:{signature['name']}")
        for aggregate in aggregates:
            derived = {
                "COUNT": "count",
                "SUM": "total",
                "AVG": "average",
                "MIN": "minimum",
                "MAX": "maximum",
            }.get(aggregate)
            if derived:
                concepts.add(f"derived:{derived}")
        if rank_output:
            concepts.add("derived:rank")
        descriptors.append({
            "output_name": str(output_name),
            "columns": sorted(columns),
            "aggregates": sorted(aggregates),
            "concepts": sorted(concepts),
            "output_roles": sorted(_label_output_roles(output_name, concepts)),
            "expression_known": selection is not None,
            "literal": isinstance(selection, (exp.Literal, exp.Null, exp.Boolean)),
            "rank_output": rank_output,
        })
    return descriptors


def _result_columns(rows: list[dict]) -> list[str] | None:
    if not rows:
        return []
    columns = [str(column) for column in rows[0].keys()]
    normalized = [_normalized_identifier(column) for column in columns]
    if len(normalized) != len(set(normalized)):
        return None
    expected_set = set(normalized)
    if any(
        {_normalized_identifier(column) for column in row.keys()} != expected_set
        for row in rows
    ):
        return None
    return columns


def _descriptor_match_score(
    generated: dict[str, Any], expected: dict[str, Any],
) -> int:
    if _normalized_identifier(generated["output_name"]) == _normalized_identifier(
        expected["output_name"]
    ):
        return 100
    generated_columns = set(generated["columns"])
    expected_columns = set(expected["columns"])
    generated_aggregates = set(generated["aggregates"])
    expected_aggregates = set(expected["aggregates"])
    generated_roles = set(generated.get("output_roles") or generated["concepts"])
    expected_roles = set(expected.get("output_roles") or expected["concepts"])
    generated_derived = {
        concept for concept in generated_roles if concept.startswith("derived:")
    }
    expected_derived = {
        concept for concept in expected_roles if concept.startswith("derived:")
    }
    directional_roles = {"derived:previous", "derived:current"}
    if (
        generated_derived.intersection(directional_roles)
        != expected_derived.intersection(directional_roles)
    ):
        return 0
    # Different final operators over the same source measure are not aliases.
    # One side may legitimately expose no operator in a CTE alias, but two
    # explicit and disjoint operators must fail closed.
    if (
        generated_derived
        and expected_derived
        and not generated_derived.intersection(expected_derived)
    ):
        return 0
    if generated_columns and generated_columns == expected_columns:
        if generated_aggregates == expected_aggregates:
            return 90
        if not generated_aggregates and not expected_aggregates:
            return 85
        return 0
    role_overlap = generated_roles.intersection(expected_roles)
    if role_overlap:
        return 80 + min(5, len(role_overlap))
    generated_concepts = set(generated["concepts"])
    expected_concepts = set(expected["concepts"])
    strong_overlap = {
        concept for concept in generated_concepts & expected_concepts
        if concept.startswith(("column:", "metric:"))
    }
    if strong_overlap and generated_aggregates == expected_aggregates:
        return 70
    return 0


def _map_projection_pairs(
    generated: list[dict[str, Any]], expected: list[dict[str, Any]],
) -> tuple[dict[int, int], list[str], list[str]]:
    """Map only mutually unique semantic pairs, leaving extras auditable.

    A final answer can be correct when either SQL emits auxiliary columns.  We
    therefore do not require either projection to be a strict subset.  Exact
    names are reserved first; remaining pairs must be the unique best match in
    both directions so an ambiguous alias never receives a positional guess.
    """

    mapping: dict[int, int] = {}
    used_expected: set[int] = set()
    expected_by_name = {
        _normalized_identifier(descriptor["output_name"]): index
        for index, descriptor in enumerate(expected)
    }
    for generated_index, generated_descriptor in enumerate(generated):
        expected_index = expected_by_name.get(
            _normalized_identifier(generated_descriptor["output_name"])
        )
        if expected_index is not None and expected_index not in used_expected:
            mapping[generated_index] = expected_index
            used_expected.add(expected_index)

    while True:
        remaining_generated = [
            index for index in range(len(generated)) if index not in mapping
        ]
        remaining_expected = [
            index for index in range(len(expected)) if index not in used_expected
        ]
        proposals: list[tuple[int, int]] = []
        for generated_index in remaining_generated:
            scores = {
                expected_index: _descriptor_match_score(
                    generated[generated_index], expected[expected_index],
                )
                for expected_index in remaining_expected
            }
            best_score = max(scores.values(), default=0)
            best_expected = [
                index for index, score in scores.items() if score == best_score
            ]
            if best_score <= 0 or len(best_expected) != 1:
                continue
            expected_index = best_expected[0]
            reverse_scores = {
                candidate: _descriptor_match_score(
                    generated[candidate], expected[expected_index],
                )
                for candidate in remaining_generated
            }
            reverse_best = max(reverse_scores.values(), default=0)
            reverse_candidates = [
                index
                for index, score in reverse_scores.items()
                if score == reverse_best
            ]
            if reverse_best == best_score and reverse_candidates == [generated_index]:
                proposals.append((generated_index, expected_index))
        if not proposals:
            break
        for generated_index, expected_index in proposals:
            if generated_index not in mapping and expected_index not in used_expected:
                mapping[generated_index] = expected_index
                used_expected.add(expected_index)

    unmatched_generated = [
        generated[index]["output_name"]
        for index in range(len(generated))
        if index not in mapping
    ]
    unmatched_expected = [
        expected[index]["output_name"]
        for index in range(len(expected))
        if index not in used_expected
    ]
    return mapping, unmatched_generated, unmatched_expected


def _descriptor_request_reasons(
    descriptor: dict[str, Any], question: str,
) -> list[str]:
    reasons: set[str] = set()
    concepts = set(descriptor["concepts"])
    explicit_concepts = _maximal_question_concepts(question)
    for concept in concepts:
        kind, _, name = concept.partition(":")
        if (
            kind == "column"
            and not descriptor["rank_output"]
            and concept in explicit_concepts
            and _terms_have_output_role(
                question, explicit_concepts[concept], descriptor,
            )
        ):
            matched_terms = explicit_concepts[concept]
            count_alias_for_identifier = bool(
                name.endswith("_id")
                and not descriptor["aggregates"]
                and matched_terms
                and all(
                    _COUNT_SEMANTIC_TERM.search(term)
                    and not _IDENTIFIER_TERM.search(term)
                    for term in matched_terms
                )
            )
            if not count_alias_for_identifier:
                reasons.add(f"question_names_column:{name}")
        elif (
            kind == "metric"
            and concept in explicit_concepts
            and _terms_have_output_role(
                question, explicit_concepts[concept], descriptor,
            )
        ):
            reasons.add(f"question_names_metric:{name}")
        elif kind == "derived":
            if name == "rank":
                if _EXPLICIT_RANK_OUTPUT.search(question):
                    reasons.add("question_requests_rank_output")
            elif (
                pattern := _DERIVED_QUESTION_PATTERNS.get(name)
            ) is not None and _pattern_has_output_role(
                question, pattern, descriptor,
            ):
                reasons.add(f"question_requests_derived:{name}")
    output_name = str(descriptor["output_name"])
    output_spans = _term_spans(question, output_name)
    governed_terms = {
        term
        for terms in (*_COLUMN_TERMS.values(), *_METRIC_TERMS.values())
        for term in terms
    }
    governed_spans = [
        span
        for term in governed_terms
        for span in _term_spans(question, term)
    ]
    output_is_maximal = bool(output_spans) and any(
        not any(
            other_start <= start
            and other_end >= end
            and (other_end - other_start) > (end - start)
            for other_start, other_end in governed_spans
        )
        for start, end in output_spans
    )
    if (
        len(_normalized_identifier(output_name)) >= 2
        and output_is_maximal
        and _terms_have_output_role(question, {output_name}, descriptor)
    ):
        reasons.add("question_names_output_alias")
    return sorted(reasons)


def _reason_semantic_concept(reason: str) -> str | None:
    prefixes = {
        "question_names_column:": "column:",
        "question_names_metric:": "metric:",
        "question_requests_derived:": "derived:",
    }
    if reason == "question_requests_rank_output":
        return "derived:rank"
    for prefix, concept_prefix in prefixes.items():
        if reason.startswith(prefix):
            return f"{concept_prefix}{reason.removeprefix(prefix)}"
    return None


def _uncovered_request_reasons(
    descriptor: dict[str, Any], question: str, covered_concepts: set[str],
) -> list[str]:
    """Keep only question roles not already served by a mapped answer field."""

    reasons = _descriptor_request_reasons(descriptor, question)
    return [
        reason
        for reason in reasons
        if (
            (concept := _reason_semantic_concept(reason)) is None
            or concept not in covered_concepts
        )
    ]


_BUCKET_ORDINAL_RE = re.compile(
    r"^\s*第?\s*([0-9]+|[一二三四五六七八九十]+)\s*"
    r"(?:档|组|层|级|等级|象限|分位)?\s*$"
)
_CHINESE_DIGITS = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def _bucket_ordinal(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, Decimal) and value == value.to_integral_value():
        return int(value)
    match = _BUCKET_ORDINAL_RE.fullmatch(str(value or ""))
    if not match:
        return None
    token = match.group(1)
    if token.isdigit():
        return int(token)
    if token in _CHINESE_DIGITS:
        return _CHINESE_DIGITS[token]
    if token.startswith("十"):
        return 10 + _CHINESE_DIGITS.get(token[1:], 0)
    if token.endswith("十"):
        return _CHINESE_DIGITS.get(token[:-1], 0) * 10
    if "十" in token:
        tens, ones = token.split("十", 1)
        return _CHINESE_DIGITS.get(tens, 0) * 10 + _CHINESE_DIGITS.get(ones, 0)
    return None


def _projection_value(
    value: Any, generated: dict[str, Any], expected: dict[str, Any],
) -> Any:
    concepts = {
        *generated.get("concepts", []),
        *expected.get("concepts", []),
        *generated.get("output_roles", []),
        *expected.get("output_roles", []),
    }
    if "derived:bucket" in concepts:
        ordinal = _bucket_ordinal(value)
        if ordinal is not None:
            return f"<bucket:{ordinal}>"
    return value


def _project_mapped_rows(
    rows: list[dict],
    descriptors: list[dict[str, Any]],
    other_descriptors: list[dict[str, Any]],
    pairs: list[tuple[int, int]],
    *,
    generated_side: bool,
) -> list[dict]:
    projected: list[dict] = []
    for row in rows:
        normalized_row = {
            _normalized_identifier(key): value for key, value in row.items()
        }
        item: dict[str, Any] = {}
        for position, (generated_index, expected_index) in enumerate(pairs):
            descriptor_index = generated_index if generated_side else expected_index
            descriptor = descriptors[descriptor_index]
            other = other_descriptors[
                expected_index if generated_side else generated_index
            ]
            value = normalized_row[
                _normalized_identifier(descriptor["output_name"])
            ]
            generated_descriptor, expected_descriptor = (
                (descriptor, other) if generated_side else (other, descriptor)
            )
            item[f"field_{position}"] = _projection_value(
                value, generated_descriptor, expected_descriptor,
            )
        projected.append(item)
    return projected


def _decimal_for_rollup(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return Decimal(str(value))
    except (ArithmeticError, ValueError):
        return None


def _reference_rollup_rows(
    expected_rows: list[dict],
    expected_descriptors: list[dict[str, Any]],
    generated_descriptors: list[dict[str, Any]],
    pairs: list[tuple[int, int]],
) -> list[dict] | None:
    """Roll a finer reference grain into an explicitly coarser final answer.

    Only additive totals may be rolled up.  Counts, averages, ratios, ranks and
    unknown expressions stay strict because they are not generally additive.
    """

    additive_positions: set[int] = set()
    for position, (generated_index, expected_index) in enumerate(pairs):
        generated_roles = set(
            generated_descriptors[generated_index].get("output_roles") or []
        )
        expected_roles = set(
            expected_descriptors[expected_index].get("output_roles") or []
        )
        combined_roles = generated_roles | expected_roles
        derived_roles = {
            role for role in combined_roles if role.startswith("derived:")
        }
        if "derived:total" in derived_roles and not derived_roles.intersection({
            "derived:average",
            "derived:ratio",
            "derived:count",
            "derived:rank",
            "derived:bucket",
            "derived:growth",
            "derived:change",
        }):
            additive_positions.add(position)
    if not additive_positions:
        return None

    projected = _project_mapped_rows(
        expected_rows,
        expected_descriptors,
        generated_descriptors,
        pairs,
        generated_side=False,
    )
    groups: dict[tuple, dict[str, Any]] = {}
    for row in projected:
        key = tuple(
            (field, _normal_value(value))
            for position, (field, value) in enumerate(row.items())
            if position not in additive_positions
        )
        aggregate = groups.setdefault(
            key,
            {
                field: (Decimal(0) if position in additive_positions else value)
                for position, (field, value) in enumerate(row.items())
            },
        )
        for position in additive_positions:
            field = f"field_{position}"
            number = _decimal_for_rollup(row[field])
            if number is None:
                return None
            aggregate[field] += number
    return list(groups.values())


_LONG_WIDE_AXIS_COLUMNS = {"year", "quarter", "month"}


def _select_expressions(sql: str) -> list[exp.Expression]:
    try:
        return list(sqlglot.parse_one(sql, read="mysql").selects)
    except Exception:
        return []


def _conditional_axis_literal(
    selection: exp.Expression,
    axis_column: str,
) -> Any | None:
    """Return one CASE-governing literal for a conditional aggregate."""

    values: list[Any] = []
    for condition in selection.find_all(exp.EQ):
        pairs = (
            (condition.this, condition.expression),
            (condition.expression, condition.this),
        )
        for column, literal in pairs:
            if (
                isinstance(column, exp.Column)
                and str(column.name or "").lower() == axis_column
                and isinstance(literal, exp.Literal)
            ):
                value: Any = literal.this
                if literal.is_number:
                    try:
                        value = int(str(value))
                    except ValueError:
                        try:
                            value = Decimal(str(value))
                        except (ArithmeticError, ValueError):
                            pass
                values.append(value)
    unique = list(dict.fromkeys(values))
    return unique[0] if len(unique) == 1 else None


def _row_output_value(row: dict, output_name: str) -> Any:
    normalized = {
        _normalized_identifier(key): value for key, value in row.items()
    }
    return normalized[_normalized_identifier(output_name)]


def _long_wide_information_equivalence(
    gen_rows: list[dict],
    expected_rows: list[dict],
    *,
    generated_sql: str | None,
    expected_sql: str,
    question: str,
    order_sensitive: bool,
) -> dict[str, Any]:
    """Prove a long time-axis result equals a conditional-aggregate wide result.

    The normalization is intentionally narrow: one explicit time axis, one
    aggregate value, at least two conditional output columns, uniquely mapped
    grouping keys, and a complete value multiset.  The axis literals must be
    named in the user question.  This accepts a presentation-shape difference,
    never a missing value, different measure, or different aggregation.
    """

    evidence: dict[str, Any] = {
        "attempted": False,
        "eligible": False,
        "equivalent": False,
        "reason": "shape_not_supported",
        "axis_column": None,
        "axis_values": [],
        "group_column_mapping": [],
        "generated_value_column": None,
        "reference_value_columns": [],
    }
    if order_sensitive or not gen_rows or not expected_rows:
        evidence["reason"] = (
            "explicit_order_contract" if order_sensitive else "empty_result"
        )
        return evidence
    generated_columns = _result_columns(gen_rows)
    expected_columns = _result_columns(expected_rows)
    if not generated_columns or not expected_columns:
        evidence["reason"] = "ambiguous_result_schema"
        return evidence
    generated_descriptors = _select_descriptors(
        str(generated_sql or ""), generated_columns,
    )
    expected_descriptors = _select_descriptors(expected_sql, expected_columns)
    generated_selections = _select_expressions(str(generated_sql or ""))
    expected_selections = _select_expressions(expected_sql)
    if (
        len(generated_selections) != len(generated_descriptors)
        or len(expected_selections) != len(expected_descriptors)
    ):
        evidence["reason"] = "projection_expression_mismatch"
        return evidence

    axis_indexes = [
        index
        for index, descriptor in enumerate(generated_descriptors)
        if not descriptor["aggregates"]
        and len(descriptor["columns"]) == 1
        and descriptor["columns"][0] in _LONG_WIDE_AXIS_COLUMNS
    ]
    value_indexes = [
        index
        for index, descriptor in enumerate(generated_descriptors)
        if descriptor["aggregates"]
    ]
    if len(axis_indexes) != 1 or len(value_indexes) != 1:
        evidence["reason"] = "requires_one_axis_and_one_value"
        return evidence
    axis_index = axis_indexes[0]
    value_index = value_indexes[0]
    axis_column = generated_descriptors[axis_index]["columns"][0]
    value_descriptor = generated_descriptors[value_index]
    generated_key_indexes = [
        index for index in range(len(generated_descriptors))
        if index not in {axis_index, value_index}
    ]
    if any(generated_descriptors[index]["aggregates"] for index in generated_key_indexes):
        evidence["reason"] = "generated_key_is_aggregate"
        return evidence

    expected_wide_indexes = [
        index
        for index, descriptor in enumerate(expected_descriptors)
        if descriptor["aggregates"] and axis_column in descriptor["columns"]
    ]
    if len(expected_wide_indexes) < 2:
        evidence["reason"] = "requires_multiple_conditional_outputs"
        return evidence
    expected_key_indexes = [
        index for index in range(len(expected_descriptors))
        if index not in set(expected_wide_indexes)
    ]
    if any(expected_descriptors[index]["aggregates"] for index in expected_key_indexes):
        evidence["reason"] = "reference_has_other_aggregate"
        return evidence

    generated_keys = [generated_descriptors[index] for index in generated_key_indexes]
    expected_keys = [expected_descriptors[index] for index in expected_key_indexes]
    key_mapping, unmatched_generated, unmatched_expected = _map_projection_pairs(
        generated_keys, expected_keys,
    )
    if (
        unmatched_generated
        or unmatched_expected
        or len(key_mapping) != len(generated_keys)
        or len(key_mapping) != len(expected_keys)
    ):
        evidence["reason"] = "group_columns_not_uniquely_mappable"
        return evidence

    value_columns = set(value_descriptor["columns"])
    value_aggregates = set(value_descriptor["aggregates"])
    axis_values: list[Any] = []
    for index in expected_wide_indexes:
        descriptor = expected_descriptors[index]
        if (
            set(descriptor["columns"]) - {axis_column} != value_columns
            or set(descriptor["aggregates"]) != value_aggregates
        ):
            evidence["reason"] = "measure_or_aggregate_mismatch"
            return evidence
        axis_value = _conditional_axis_literal(
            expected_selections[index], axis_column,
        )
        if axis_value is None or str(axis_value) not in question:
            evidence["reason"] = "axis_literal_not_explicit_in_question"
            return evidence
        axis_values.append(axis_value)
    if len({_normal_value(value) for value in axis_values}) != len(axis_values):
        evidence["reason"] = "duplicate_axis_literal"
        return evidence

    canonical_generated: list[dict[str, Any]] = []
    for row in gen_rows:
        item = {
            f"group_{position}": _row_output_value(
                row, generated_descriptors[index]["output_name"],
            )
            for position, index in enumerate(generated_key_indexes)
        }
        item["axis"] = _row_output_value(
            row, generated_descriptors[axis_index]["output_name"],
        )
        item["value"] = _row_output_value(
            row, value_descriptor["output_name"],
        )
        canonical_generated.append(item)

    canonical_expected: list[dict[str, Any]] = []
    for row in expected_rows:
        key_values = {
            f"group_{position}": _row_output_value(
                row,
                expected_keys[key_mapping[position]]["output_name"],
            )
            for position in range(len(generated_keys))
        }
        for axis_value, index in zip(axis_values, expected_wide_indexes):
            canonical_expected.append({
                **key_values,
                "axis": axis_value,
                "value": _row_output_value(
                    row, expected_descriptors[index]["output_name"],
                ),
            })

    equivalent = _unordered_rows_equivalent(
        canonical_generated, canonical_expected, positional=False,
    )
    evidence.update({
        "attempted": True,
        "eligible": True,
        "equivalent": equivalent,
        "reason": "information_match" if equivalent else "expanded_values_mismatch",
        "axis_column": axis_column,
        "axis_values": [str(value) for value in axis_values],
        "group_column_mapping": [
            {
                "generated": generated_keys[generated_index]["output_name"],
                "reference": expected_keys[expected_index]["output_name"],
            }
            for generated_index, expected_index in sorted(key_mapping.items())
        ],
        "generated_value_column": value_descriptor["output_name"],
        "reference_value_columns": [
            expected_descriptors[index]["output_name"]
            for index in expected_wide_indexes
        ],
    })
    return evidence


def _projection_equivalence_evidence(
    gen_rows: list[dict],
    expected_rows: list[dict],
    *,
    generated_sql: str | None,
    expected_sql: str,
    question: str,
    order_sensitive: bool,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "attempted": False,
        "eligible": False,
        "reason": "no_semantic_projection_match",
        "generated_columns": [],
        "reference_columns": [],
        "column_mapping": [],
        "omitted_reference_columns": [],
        "extra_generated_columns": [],
        "explicitly_required_omissions": [],
        "unresolved_generated_columns": [],
        "unresolved_reference_columns": [],
        "rows_equivalent_after_projection": False,
    }
    generated_columns = _result_columns(gen_rows)
    expected_columns = _result_columns(expected_rows)
    if generated_columns is None or expected_columns is None:
        evidence["reason"] = "inconsistent_or_ambiguous_result_schema"
        return evidence
    evidence["generated_columns"] = generated_columns
    evidence["reference_columns"] = expected_columns
    if not gen_rows or not expected_rows or not generated_columns or not expected_columns:
        if len(gen_rows) != len(expected_rows):
            evidence["reason"] = "row_count_mismatch"
        return evidence

    evidence["attempted"] = True
    generated_descriptors = _select_descriptors(
        str(generated_sql or ""), generated_columns,
    )
    expected_descriptors = _select_descriptors(expected_sql, expected_columns)
    mapping, unmatched_generated, unmatched_expected = _map_projection_pairs(
        generated_descriptors, expected_descriptors,
    )
    evidence["unresolved_generated_columns"] = unmatched_generated
    evidence["unresolved_reference_columns"] = unmatched_expected
    evidence["extra_generated_columns"] = unmatched_generated
    if not mapping:
        evidence["reason"] = "columns_not_uniquely_mappable"
        return evidence

    pairs = sorted(mapping.items())
    evidence["column_mapping"] = [
        {
            "generated": generated_descriptors[generated_index]["output_name"],
            "reference": expected_descriptors[expected_index]["output_name"],
        }
        for generated_index, expected_index in pairs
    ]
    omitted_indexes = sorted(set(range(len(expected_descriptors))) - set(mapping.values()))
    omitted = []
    required = []
    covered_concepts = {
        concept
        for expected_index in mapping.values()
        for concept in (
            *expected_descriptors[expected_index]["concepts"],
            *expected_descriptors[expected_index].get("output_roles", []),
        )
    }
    for index in omitted_indexes:
        descriptor = expected_descriptors[index]
        reasons = _uncovered_request_reasons(
            descriptor, question, covered_concepts,
        )
        resolvable = bool(descriptor["concepts"] or descriptor["literal"])
        item = {
            "column": descriptor["output_name"],
            "semantic_concepts": descriptor["concepts"],
            "request_reason_codes": reasons,
            "semantics_resolved": resolvable,
        }
        omitted.append(item)
        if reasons or not resolvable:
            if not resolvable:
                item = {**item, "request_reason_codes": ["unresolved_semantics"]}
            required.append(item)
    evidence["omitted_reference_columns"] = omitted
    evidence["explicitly_required_omissions"] = required
    if required:
        evidence["reason"] = "question_requires_omitted_reference_columns"
        return evidence

    projected_generated_rows = _project_mapped_rows(
        gen_rows,
        generated_descriptors,
        expected_descriptors,
        pairs,
        generated_side=True,
    )
    projected_expected_rows = _project_mapped_rows(
        expected_rows,
        expected_descriptors,
        generated_descriptors,
        pairs,
        generated_side=False,
    )
    equivalent, comparison_mode = _compare_same_projection(
        projected_generated_rows,
        projected_expected_rows,
        order_sensitive=order_sensitive,
    )
    if (
        not equivalent
        and not order_sensitive
        and len(gen_rows) < len(expected_rows)
        and all(
            not set(descriptor.get("output_roles") or []).intersection({
                "derived:average",
                "derived:ratio",
                "derived:count",
                "derived:rank",
                "derived:bucket",
                "derived:growth",
                "derived:change",
            })
            for index, descriptor in enumerate(expected_descriptors)
            if index not in mapping.values()
        )
    ):
        rolled_up_rows = _reference_rollup_rows(
            expected_rows,
            expected_descriptors,
            generated_descriptors,
            pairs,
        )
        if rolled_up_rows is not None:
            equivalent, comparison_mode = _compare_same_projection(
                projected_generated_rows,
                rolled_up_rows,
                order_sensitive=False,
            )
            if equivalent:
                comparison_mode = "reference_additive_rollup"
    evidence.update({
        "eligible": True,
        "reason": (
            "minimum_sufficient_projection_match"
            if equivalent else "projected_values_mismatch"
        ),
        "rows_equivalent_after_projection": equivalent,
        "projected_comparison_mode": comparison_mode,
    })
    return evidence


def result_equivalence_evidence(
    gen_rows: list[dict],
    expected_rows: list[dict],
    expected_sql: str,
    ordered: bool | None = None,
    *,
    question: str | None = None,
    generated_sql: str | None = None,
) -> dict[str, Any]:
    """Evaluate final results with a conservative minimum-projection contract.

    Both projections are reduced to mutually unique semantic answer fields.
    Auxiliary columns may exist on either side, but a reference field explicitly
    requested by the question can never be dropped.  Ordinal bucket labels and
    a strictly additive reference rollup are normalized only under resolved
    semantic roles.  No fixture ID or question literal is special-cased.
    """

    order_sensitive = (
        question_requires_result_order(question)
        if question is not None
        else bool(ordered)
    )
    equivalent, comparison_mode = _compare_same_projection(
        gen_rows, expected_rows, order_sensitive=order_sensitive,
    )
    evidence: dict[str, Any] = {
        "contract_version": RESULT_EQUIVALENCE_CONTRACT_VERSION,
        "equivalent": equivalent,
        "comparison_mode": comparison_mode,
        "order_sensitive": order_sensitive,
        "existence_equivalence": {
            "attempted": False,
            "eligible": False,
            "equivalent": False,
            "reason": "not_evaluated",
        },
        "minimum_sufficient_projection": {
            "attempted": False,
            "eligible": False,
            "reason": "direct_result_match" if equivalent else "not_evaluated",
        },
        "long_wide_information_equivalence": {
            "attempted": False,
            "eligible": False,
            "equivalent": False,
            "reason": "not_evaluated",
        },
    }
    if equivalent:
        return evidence
    existence = _existence_equivalence_evidence(
        gen_rows,
        expected_rows,
        generated_sql=generated_sql,
        expected_sql=expected_sql,
        question=str(question or ""),
    )
    evidence["existence_equivalence"] = existence
    if existence["equivalent"]:
        evidence["equivalent"] = True
        evidence["comparison_mode"] = "existence_truth_equivalence"
        return evidence
    long_wide = _long_wide_information_equivalence(
        gen_rows,
        expected_rows,
        generated_sql=generated_sql,
        expected_sql=expected_sql,
        question=str(question or ""),
        order_sensitive=order_sensitive,
    )
    evidence["long_wide_information_equivalence"] = long_wide
    if long_wide["equivalent"]:
        evidence["equivalent"] = True
        evidence["comparison_mode"] = "long_wide_information_equivalence"
        return evidence
    projection = _projection_equivalence_evidence(
        gen_rows,
        expected_rows,
        generated_sql=generated_sql,
        expected_sql=expected_sql,
        question=str(question or ""),
        order_sensitive=order_sensitive,
    )
    evidence["minimum_sufficient_projection"] = projection
    if projection["rows_equivalent_after_projection"]:
        evidence["equivalent"] = True
        evidence["comparison_mode"] = "minimum_sufficient_projection"
    return evidence


def result_equivalent(
    gen_rows: list[dict],
    expected_rows: list[dict],
    expected_sql: str,
    ordered: bool | None = None,
    *,
    question: str | None = None,
    generated_sql: str | None = None,
) -> bool:
    """
    比对两份结果集是否一致。

    业务逻辑：
        1. 行数不等直接判否
        2. question 提供时，仅由题面显式顺序语义决定是否逐行比对；旧数据集
           result_ordered 标志与 Reference SQL 的 ORDER BY 都不能覆盖题意
        3. question 未提供时，ordered 可作为已解析好的显式契约；缺省按无序多集
        4. 先按"小写列名=值"比对；不等则按"仅值"兜底比对，容许别名大小写/命名差异

    ``expected_sql`` 保留在签名中以兼容已有调用，但不再用于推断顺序语义。
    """
    return bool(result_equivalence_evidence(
        gen_rows,
        expected_rows,
        expected_sql,
        ordered,
        question=question,
        generated_sql=generated_sql,
    )["equivalent"])
