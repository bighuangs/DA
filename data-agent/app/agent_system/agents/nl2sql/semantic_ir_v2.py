"""Versioned semantic slots and deterministic SQLGlot compiler for broad NL2SQL.

The original progressive compiler intentionally handled only one dimension and
one measure.  That was a useful safe starting point, but it left comparison,
window, ratio, cohort and subquery questions entirely to a probabilistic model.
This module closes those reusable semantic families without coupling the runtime
to evaluation case identifiers or golden SQL artifacts.

The compiler is deliberately conservative:

* a family is selected from query semantics, never from a case id;
* every physical field must exist in the current workspace metadata;
* every required relationship key must be present;
* user literals are represented as SQLGlot literals;
* the generated statement is parsed and re-emitted by SQLGlot before use.

If any required slot is missing, the returned IR remains auditable but its
compiler is disabled, so the normal constrained LLM path can take over.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import sqlglot
from sqlglot import exp

from app.domain.query.semantic_aliases import (
    business_aliases_for,
    stable_merge_aliases,
)


IR_VERSION_V2 = "semantic-ir-v2"
COMPILER_MODE_V2 = "deterministic_sqlglot_v2"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*(?:年|全年)")
_NUMBER_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)(?!\d)")
_FOLLOW_UP_RE = re.compile(r"(?:^那|^那么|呢$|改成|换成|再看|再算)")

_COMPARISON_OPERATORS: tuple[tuple[str, str], ...] = (
    ("不低于", "gte"), ("不少于", "gte"), ("至少", "gte"),
    ("达到", "gte"), (">=", "gte"), ("≥", "gte"),
    ("不高于", "lte"), ("不超过", "lte"), ("至多", "lte"),
    ("<=", "lte"), ("≤", "lte"),
    ("超过", "gt"), ("高于", "gt"), ("大于", "gt"), (">", "gt"),
    ("低于", "lt"), ("少于", "lt"), ("小于", "lt"), ("不足", "lt"), ("<", "lt"),
    ("等于", "eq"), ("=", "eq"),
)
_COMPARISON_PATTERN = re.compile(
    "(" + "|".join(
        re.escape(item[0])
        for item in sorted(_COMPARISON_OPERATORS, key=lambda item: -len(item[0]))
    ) + r")(\d+(?:\.\d+)?)(亿|万)?(?:元|块|笔|单|次|件|个)?"
)
_COMPARISON_OPERATOR_MAP = dict(_COMPARISON_OPERATORS)
_AGGREGATE_SCOPE_CUES = ("累计", "总计", "合计", "汇总", "平均")
_TRANSACTION_ACTION_PATTERN = r"(?:成交|购买|销售|售出|卖出)"
_TRANSACTION_CLAUSE_GAP = r"[^，,。；;！!？?]"
_ITEM_COUNT_UNIT_PATTERN = r"(?:件(?:商品|产品)?|个(?:商品|产品))"
_TRANSACTION_QUANTITY_QUESTION_RE = re.compile(
    rf"{_TRANSACTION_ACTION_PATTERN}{_TRANSACTION_CLAUSE_GAP}{{0,18}}"
    rf"(?:多少|几){_TRANSACTION_CLAUSE_GAP}{{0,5}}{_ITEM_COUNT_UNIT_PATTERN}"
)
_TRANSACTION_QUANTITY_NOUN_RE = re.compile(
    rf"{_TRANSACTION_ACTION_PATTERN}{_TRANSACTION_CLAUSE_GAP}{{0,18}}"
    r"(?:商品|产品)?(?:件数|销量|销售量)"
)
_AGGREGATE_TRANSACTION_QUANTITY_RE = re.compile(
    rf"(?:累计|合计|总计|共计|一共|共)"
    rf"{_TRANSACTION_CLAUSE_GAP}{{0,10}}{_TRANSACTION_ACTION_PATTERN}"
    rf"{_TRANSACTION_CLAUSE_GAP}{{0,18}}"
    rf"(?:多少|几|\d+|[零一二两三四五六七八九十百千万]+)?"
    rf"{_TRANSACTION_CLAUSE_GAP}{{0,4}}{_ITEM_COUNT_UNIT_PATTERN}"
)
_CATALOG_CONTEXT_RE = re.compile(
    r"(?:商品|产品).{0,8}(?:目录|清单|主数据|档案)"
    r"|(?:目录|清单|主数据|档案).{0,8}(?:商品|产品)"
)


# Logical fields are a Data Agent semantic contract, not an evaluation fixture.
# Exact table/column names are the primary binding; aliases provide a narrow
# compatibility path for metadata that uses localized descriptions.
FIELD_SPECS: dict[str, dict[str, Any]] = {
    "order_id": {
        "table": "fact_order", "column": "order_id",
        "terms": (
            "订单id", "订单编号", "订单数", "订单总数", "订单数量",
            "订单量", "订单笔数", "多少笔订单",
        ),
    },
    "customer_id": {
        "table": "fact_order", "column": "customer_id",
        "terms": (
            "客户id", "客户编号", "购买客户数", "不同购买客户数",
            "去重客户数", "客户数量", "客户数",
        ),
    },
    "product_id": {
        "table": "fact_order", "column": "product_id",
        "terms": ("商品id", "产品id"),
    },
    "region_id": {
        "table": "fact_order", "column": "region_id",
        "terms": ("地区id", "区域id"),
    },
    "date_id": {
        "table": "fact_order", "column": "date_id",
        "terms": ("日期id", "日期编号"),
    },
    "order_amount": {
        "table": "fact_order", "column": "order_amount",
        "terms": (
            "销售额", "销售总额", "订单金额", "成交金额", "成交额",
            "消费金额", "单笔金额", "gmv",
        ),
    },
    "order_quantity": {
        "table": "fact_order", "column": "order_quantity",
        "terms": ("销量", "销售量", "购买数量", "购买件数", "售出件数"),
    },
    "date_key": {
        "table": "dim_date", "column": "date_id",
        "terms": ("日期id", "日期"),
    },
    "year": {
        "table": "dim_date", "column": "year",
        "terms": ("年份", "年"),
    },
    "quarter": {
        "table": "dim_date", "column": "quarter",
        "terms": ("季度", "季"),
    },
    "month": {
        "table": "dim_date", "column": "month",
        "terms": ("月份", "月"),
    },
    "day": {
        "table": "dim_date", "column": "day",
        "terms": ("日", "号", "日期中的日"),
    },
    "region_key": {
        "table": "dim_region", "column": "region_id",
        "terms": ("地区id", "区域id"),
    },
    "region_name": {
        "table": "dim_region", "column": "region_name",
        "terms": ("地区", "区域", "销售区域", "大区", "地区名称"),
    },
    "province": {
        "table": "dim_region", "column": "province",
        "terms": ("省份", "省"),
    },
    "country": {
        "table": "dim_region", "column": "country",
        "terms": ("国家",),
    },
    "product_key": {
        "table": "dim_product", "column": "product_id",
        "terms": ("商品id", "产品id"),
    },
    "product_name": {
        "table": "dim_product", "column": "product_name",
        "terms": ("商品", "产品", "商品名称", "产品名称"),
    },
    "category": {
        "table": "dim_product", "column": "category",
        "terms": ("品类", "类别", "商品品类", "产品品类", "商品类别", "产品类别"),
    },
    "brand": {
        "table": "dim_product", "column": "brand",
        "terms": ("品牌", "品牌名称"),
    },
    "customer_key": {
        "table": "dim_customer", "column": "customer_id",
        "terms": ("客户id", "客户编号"),
    },
    "customer_name": {
        "table": "dim_customer", "column": "customer_name",
        "terms": ("客户", "用户", "客户名称"),
    },
    "member_level": {
        "table": "dim_customer", "column": "member_level",
        "terms": ("会员等级", "会员级别"),
    },
    "gender": {
        "table": "dim_customer", "column": "gender",
        "terms": ("性别",),
    },
}


def _metadata_column_terms(
    logical: str,
    table_infos: Sequence[Mapping[str, Any]] | None = None,
) -> list[str]:
    """Return vocabulary owned by the logical field's physical Meta column.

    The physical table/column binding remains the authority.  Curated live Meta
    terms and the governed fallback registry extend the same binding, so query
    family detection, retrieval closure and measure selection cannot silently
    drift from one another when a business synonym is added.
    """

    spec = FIELD_SPECS[logical]
    metadata_groups: list[Any] = []
    for table in table_infos or []:
        if str(table.get("name") or "").casefold() != str(spec["table"]).casefold():
            continue
        for column in table.get("columns") or []:
            if str(column.get("name") or "").casefold() != str(spec["column"]).casefold():
                continue
            metadata_groups.extend([
                column.get("description"),
                column.get("alias") or [],
            ])
    terms = stable_merge_aliases(
        spec.get("terms") or (),
        business_aliases_for(str(spec["table"]), str(spec["column"])),
        *metadata_groups,
    )
    return [str(term) for term in terms if len(_normalize_text(term)) >= 2]


def _shared_semantic_suffixes(terms: Sequence[str]) -> list[str]:
    """Derive repeated business nouns such as ``金额`` from governed terms.

    This is intentionally conservative: a suffix must contain at least two CJK
    characters and occur in at least three distinct terms.  It lets natural
    ellipsis (``金额达到...`` after terms such as ``订单金额`` and
    ``成交金额``) resolve without adding one benchmark phrase to the parser.
    """

    normalized = list(dict.fromkeys(_normalize_text(term) for term in terms))
    counts: dict[str, int] = {}
    for term in normalized:
        for length in range(2, min(5, len(term)) + 1):
            suffix = term[-length:]
            if not re.fullmatch(r"[\u4e00-\u9fff]+", suffix):
                continue
            counts[suffix] = counts.get(suffix, 0) + 1
    candidates = [suffix for suffix, count in counts.items() if count >= 3]
    return [
        suffix
        for suffix in sorted(candidates, key=lambda item: (-len(item), item))
        if not any(suffix != other and suffix in other for other in candidates)
    ]


def semantic_terms_for_logical(
    logical: str,
    table_infos: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[str, ...]:
    """Expose the de-duplicated, metadata-backed vocabulary for one field."""

    terms = _metadata_column_terms(logical, table_infos)
    return tuple(stable_merge_aliases(terms, _shared_semantic_suffixes(terms)))


def _query_has_logical(
    query: str,
    logical: str,
    table_infos: Sequence[Mapping[str, Any]] | None = None,
) -> bool:
    q = _normalize_text(query)
    return any(
        _normalize_text(term) in q
        for term in semantic_terms_for_logical(logical, table_infos)
    )


def transaction_item_quantity_requested(
    query: str,
    table_infos: Sequence[Mapping[str, Any]] | None = None,
) -> bool:
    """Identify an item-quantity measure only inside a transaction context.

    Governed metadata aliases remain authoritative.  The structural fallback
    allows aggregate/aspect words between a transaction action and its item
    count unit, while requiring both signals in the same bounded clause.
    Catalog cardinality without a transaction action therefore does not
    acquire an order measure.
    """

    q = _normalize_text(query)
    structured_transaction = any(
        pattern.search(q)
        for pattern in (
            _TRANSACTION_QUANTITY_QUESTION_RE,
            _TRANSACTION_QUANTITY_NOUN_RE,
            _AGGREGATE_TRANSACTION_QUANTITY_RE,
        )
    )
    if structured_transaction:
        return True
    # Shared metadata suffixes such as "件数" are useful ellipsis, but they
    # do not override an explicit catalog-cardinality context without a sale or
    # purchase action.
    if _CATALOG_CONTEXT_RE.search(q) and not re.search(
        _TRANSACTION_ACTION_PATTERN,
        q,
    ):
        return False
    return _query_has_logical(q, "order_quantity", table_infos)


def _numeric_comparisons(
    query: str,
    table_infos: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Extract explicit numeric predicates and whether they apply post-aggregate."""

    q = _normalize_text(query)
    output: list[dict[str, Any]] = []
    for logical in ("order_amount", "order_quantity", "order_id", "customer_id"):
        terms = sorted(
            semantic_terms_for_logical(logical, table_infos),
            key=lambda item: -len(_normalize_text(item)),
        )
        for term in terms:
            normalized_term = _normalize_text(term)
            start = q.find(normalized_term)
            if start < 0:
                continue
            prefix = q[max(0, start - 4):start]
            # A bounded tail supports two coordinated comparisons that share
            # one field mention, without absorbing a later unrelated clause.
            tail = q[start + len(normalized_term):]
            tail = re.split(r"[，,;；。？?]", tail, maxsplit=1)[0][:64]
            other_field_positions = [
                position
                for other_logical in (
                    "order_amount", "order_quantity", "order_id", "customer_id",
                )
                if other_logical != logical
                for other_term in semantic_terms_for_logical(
                    other_logical, table_infos,
                )
                if (
                    position := tail.find(_normalize_text(other_term))
                ) >= 0
            ]
            if other_field_positions:
                tail = tail[:min(other_field_positions)]
            for match in _COMPARISON_PATTERN.finditer(tail):
                numeric = float(match.group(2))
                multiplier = {"万": 10_000, "亿": 100_000_000}.get(match.group(3), 1)
                value: int | float = numeric * multiplier
                if float(value).is_integer():
                    value = int(value)
                output.append({
                    "logical": logical,
                    "operator": _COMPARISON_OPERATOR_MAP[match.group(1)],
                    "value": value,
                    # Counts of business entities are aggregates by definition;
                    # unlike an amount or quantity there is no meaningful
                    # row-level ``customer_id >= N`` interpretation.
                    "aggregate_scope": (
                        logical in {"order_id", "customer_id"}
                        or any(cue in prefix for cue in _AGGREGATE_SCOPE_CUES)
                    ),
                })
            break
    unique: dict[tuple[str, str, int | float, bool], dict[str, Any]] = {}
    for item in output:
        key = (
            str(item["logical"]), str(item["operator"]), item["value"],
            bool(item["aggregate_scope"]),
        )
        unique[key] = item
    return list(unique.values())


_COUNTABLE_GROUP_LOGICALS: tuple[str, ...] = (
    "region_name", "province", "country", "category", "brand",
    "product_name", "customer_name", "member_level", "gender",
)


def _counted_group_logical(
    query: str,
    table_infos: Sequence[Mapping[str, Any]] | None = None,
) -> str | None:
    q = _normalize_text(query)
    count_prefixes = ("有多少", "共有多少", "共多少", "多少个", "多少种")
    for logical in _COUNTABLE_GROUP_LOGICALS:
        for term in semantic_terms_for_logical(logical, table_infos):
            normalized = _normalize_text(term)
            position = q.find(normalized)
            if position < 0:
                continue
            prefix = q[max(0, position - 8):position]
            if any(cue in prefix for cue in count_prefixes):
                return logical
    return None


def _is_percentile_request(query: str) -> bool:
    q = _normalize_text(query)
    return any(token in q for token in ("百分位排名", "百分位秩", "percent_rank"))


def _is_entity_total_average_request(
    query: str,
    table_infos: Sequence[Mapping[str, Any]] | None = None,
) -> bool:
    """Detect an average over per-entity aggregates, not an average fact row.

    This is a semantic-role pattern: an entity (currently the governed customer
    entity), a cumulative/total modifier, an additive amount and an outer
    average must all be explicit.  It therefore generalizes across the outer
    grouping dimension without matching ordinary AOV questions.
    """

    q = _normalize_text(query)
    entity = any(token in q for token in ("客户", "用户", "顾客"))
    total = any(token in q for token in ("累计", "累积", "合计", "总计", "总额"))
    average = any(token in q for token in ("平均值", "均值", "平均"))
    amount = _query_has_logical(q, "order_amount", table_infos)
    return entity and total and average and amount


def _counted_dimension_measure(
    query: str,
    table_infos: Sequence[Mapping[str, Any]] | None = None,
) -> str | None:
    """Return a dimension attribute used as a distinct-count measure."""

    q = _normalize_text(query)
    patterns = {
        "province": ("省份数", "省份数量", "多少省份"),
        "region_name": ("地区数", "地区数量", "大区数", "大区数量", "多少地区", "多少大区"),
        "category": ("品类数", "品类数量", "类别数", "类别数量"),
        "brand": ("品牌数", "品牌数量"),
    }
    for logical, cues in patterns.items():
        if any(cue in q for cue in cues) and _query_has_logical(
            q, logical, table_infos,
        ):
            return logical
    return None

TABLE_RELATIONS: tuple[tuple[str, str, str, str], ...] = (
    ("fact_order", "date_id", "dim_date", "date_id"),
    ("fact_order", "region_id", "dim_region", "region_id"),
    ("fact_order", "product_id", "dim_product", "product_id"),
    ("fact_order", "customer_id", "dim_customer", "customer_id"),
)

_TABLE_ALIASES = {
    "fact_order": "f",
    "dim_date": "d",
    "dim_region": "r",
    "dim_product": "p",
    "dim_customer": "c",
}

def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def _column_text(column: Mapping[str, Any]) -> str:
    return " ".join(
        str(value)
        for value in (
            column.get("name", ""),
            column.get("description", ""),
            *(column.get("alias") or []),
        )
        if value
    ).lower()


def _metadata_index(
    table_infos: Sequence[Mapping[str, Any]] | None,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (
            str(table.get("name") or "").lower(),
            str(column.get("name") or "").lower(),
        ): column
        for table in (table_infos or [])
        for column in (table.get("columns") or [])
        if table.get("name") and column.get("name")
    }


def _resolve_fields(
    table_infos: Sequence[Mapping[str, Any]] | None,
    logical_fields: Sequence[str],
) -> tuple[dict[str, dict[str, str]], list[str]]:
    index = _metadata_index(table_infos)
    resolved: dict[str, dict[str, str]] = {}
    missing: list[str] = []
    tables = list(table_infos or [])
    for logical in logical_fields:
        spec = FIELD_SPECS[logical]
        exact = index.get((spec["table"].lower(), spec["column"].lower()))
        if exact is not None:
            resolved[logical] = {
                "table": spec["table"],
                "column": str(exact.get("name") or spec["column"]),
            }
            continue
        # Narrow fallback: remain inside the expected physical table and require
        # a metadata alias/description term match.
        candidates = []
        for table in tables:
            if str(table.get("name") or "").lower() != spec["table"].lower():
                continue
            for column in table.get("columns") or []:
                text = _column_text(column)
                if any(_normalize_text(term) in _normalize_text(text) for term in spec["terms"]):
                    candidates.append(column)
        if len(candidates) == 1:
            resolved[logical] = {
                "table": spec["table"],
                "column": str(candidates[0].get("name") or ""),
            }
        else:
            missing.append(logical)
    return resolved, missing


def _years(query: str) -> list[int]:
    values = [int(item) for item in _YEAR_RE.findall(query)]
    for group in re.findall(
        r"(?<!\d)"
        r"((?:19|20)\d{2}"
        r"(?:(?:和|与|、|,|，)(?:19|20)\d{2})+)"
        r"(?:\s*年)?",
        query,
    ):
        values.extend(
            int(item)
            for item in re.findall(r"(?:19|20)\d{2}", group)
        )
    for start, end in re.findall(
        r"(?<!\d)((?:19|20)\d{2})\s*(?:到|至|-)\s*"
        r"((?:19|20)\d{2})(?:\s*年)?"
        r"(?!\d|\s*(?:元|块|笔|单|次|件))",
        query,
    ):
        values.extend((int(start), int(end)))
    # The suffix-regex may discover the last, explicitly suffixed year before
    # the coordinated-list regex.  Preserve the user's textual order rather
    # than detector order so output columns remain stable for 3+ periods.
    return sorted(
        dict.fromkeys(values),
        key=lambda year: query.find(str(year)),
    )


def _extract_limit(query: str, default: int | None = None) -> int | None:
    match = re.search(r"(?:前|后|top\s*)(\d{1,4})", query, flags=re.IGNORECASE)
    if match:
        return max(1, min(int(match.group(1)), 1000))
    cn = {
        "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
        "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    }
    match = re.search(r"(?:前|后)([一二两三四五六七八九十])", query)
    if match:
        return cn.get(match.group(1), default)
    match = re.search(
        r"(?:最高|最低|最多|最少)(?:的)?([一二两三四五六七八九十])个",
        query,
    )
    return cn.get(match.group(1), default) if match else default


def _extract_dimension_literal(query: str, suffix: str) -> str | None:
    """Extract a user-provided entity literal immediately before a dimension.

    Generic quantifiers are rejected, so "各地区" remains a grouping request
    while "华东地区" becomes an equality filter.
    """
    suffix_boundary = r"(?!别)" if suffix == "品类" else ""
    match = re.search(
        rf"([\u4e00-\u9fffA-Za-z0-9_-]{{1,24}}?){re.escape(suffix)}"
        rf"{suffix_boundary}",
        query,
    )
    if not match:
        return None
    value = match.group(1)
    # Trim common clause prefixes greedily captured by the broad CJK range.
    for prefix in (
        "统计", "查询", "看看", "以及", "并且", "和", "的", "年",
        "再按", "也按", "改成", "换成", "改为", "调整为",
        "只看", "仅看", "限定", "筛选", "排除", "不含", "不要",
        *(("地区", "区域", "大区") if suffix in {"品类", "类别"} else ()),
    ):
        if prefix in value:
            value = value.split(prefix)[-1]
    value = re.sub(r"^(?:各|每个|每种|所有|全部|不同|按)", "", value)
    if (
        not value
        or value in {"各", "每个", "每种", "所有", "全部", "不同"}
        or (suffix in {"品类", "类别"} and value in {"商品", "产品"})
        or re.search(r"(?:多少|几)(?:个|种)?$", value)
    ):
        return None
    return value


def _extract_excluded_literal(query: str) -> str | None:
    match = re.search(r"除(.{1,30}?)外", query) or re.search(
        r"(?:排除|不含|不要)"
        r"([\u4e00-\u9fffA-Za-z0-9_-]{1,24})"
        r"(?:地区|区域|大区|品类|类别|品牌)?"
        r"(?:$|[，,。；;])",
        query,
    )
    if not match:
        return None
    value = re.sub(r"(?:品类|类别|地区|区域|品牌)$", "", match.group(1)).strip()
    return value or None


def _family(
    query: str,
    table_infos: Sequence[Mapping[str, Any]] | None = None,
) -> str | None:
    q = _normalize_text(query)
    years = _years(q)
    if not q:
        return None
    comparisons = _numeric_comparisons(q, table_infos)
    compared_counts: dict[str, int] = {}
    for item in comparisons:
        logical = str(item["logical"])
        compared_counts[logical] = compared_counts.get(logical, 0) + 1
    if _is_percentile_request(q):
        return "distribution_rank"
    if _is_entity_total_average_request(q, table_infos):
        return "entity_aggregate_average"
    if (
        any(count >= 2 for count in compared_counts.values())
        and _query_has_logical(q, "order_id", table_infos)
    ):
        return "conditional_aggregate"
    if any(bool(item["aggregate_scope"]) for item in comparisons):
        if _counted_group_logical(q, table_infos):
            return "qualifying_group_count"
        return "having_aggregate"
    if (
        ("商品总数" in q or "产品总数" in q)
        and ("卖出过" in q or "售出过" in q)
        and ("未卖出" in q or "未售出" in q)
    ):
        return "inventory_status"
    if re.search(r"(?:从未|没有产生|没有.*订单|未售出|未卖出|哪些日期没有订单)", q):
        return "anti_join"
    if "复购率" in q or "回购率" in q:
        return "repeat_rate"
    if (
        len(years) >= 2
        and any(token in q for token in ("分别", "比较", "对比"))
        and _query_has_logical(q, "order_amount", table_infos)
    ):
        return "period_compare"
    if "环比" in q:
        return "window_mom"
    if ("年内累计" in q or "累计销售额" in q) and re.search(r"每月|按月|各月", q):
        return "window_running_total"
    if re.search(r"(?:前|top).{0,4}\d*.*客户.*(?:贡献|占)", q) or (
        "客户" in q and "总销售额的多少" in q
    ):
        return "top_share"
    if ("占总销售额" in q or "占比" in q or "比例" in q) and any(
        token in q for token in ("品类", "品牌", "地区", "会员等级")
    ):
        return "group_share"
    if (
        ("每个地区" in q and "最高" in q and any(token in q for token in ("商品", "产品")))
        or ("每个季度" in q and "最高" in q and "品牌" in q)
        or ("每年" in q and "最高" in q and "月份" in q)
    ):
        return "top_per_group"
    if len(years) >= 2 and "都下过订单" in q:
        return "active_both_periods"
    if "首次下单日期" in q:
        return "first_event"
    if "首次下单月份" in q and "新客户" in q:
        return "cohort_new"
    if "分成" in q and ("档" in q or "段" in q) and "订单金额" in q:
        return "segmentation"
    if "高于客户平均累计消费" in q:
        return "above_average_customer"
    if "高于品牌平均销售额" in q:
        return "above_average_brand"
    if re.search(r"累计消费金额?超过\d+", q):
        return "having_threshold"
    # Broad but still structured aggregation/ranking/time-series family.
    if (
        any(
            _query_has_logical(q, logical, table_infos)
            for logical in ("order_amount", "order_id")
        )
        or transaction_item_quantity_requested(q, table_infos)
        or any(token in q for token in (
            "客户有多少", "客户数", "有过下单行为", "多少笔订单",
            "售出", "卖出", "购买件数", "大客户", "产生过订单",
        ))
        or _counted_dimension_measure(q, table_infos) is not None
    ):
        return "aggregate"
    return None


def semantic_metadata_requirements(
    query: str,
    chat_history: Sequence[Mapping[str, Any]] | None = None,
    table_infos: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, set[str]]:
    """Return the bounded metadata closure needed by a recognized semantic family."""
    q = _normalize_text(query)
    family = _family(q, table_infos)

    logical: set[str] = set()
    for measure_logical in ("order_amount", "order_id"):
        if _query_has_logical(q, measure_logical, table_infos):
            logical.add(measure_logical)
    if transaction_item_quantity_requested(q, table_infos):
        logical.add("order_quantity")
    if "下单" in q:
        logical.add("order_id")
    logical.update(
        str(item["logical"])
        for item in _numeric_comparisons(q, table_infos)
    )

    if _years(q) or any(token in q for token in (
        "每年", "月份", "每月", "按月", "季度", "每季",
        "月均", "平均每月", "每月平均",
    )):
        logical.update({"date_id", "date_key", "year"})
    if any(token in q for token in (
        "月份", "每月", "按月", "各月", "月均", "平均每月", "每月平均",
    )):
        logical.add("month")
    if "季度" in q or "每季" in q:
        logical.add("quarter")
    if "几号" in q or "按日" in q:
        logical.add("day")

    if any(
        _query_has_logical(q, name, table_infos)
        for name in ("region_name", "province", "country")
    ):
        logical.update({"region_id", "region_key"})
    if _query_has_logical(q, "region_name", table_infos):
        logical.add("region_name")
    if _query_has_logical(q, "province", table_infos):
        logical.add("province")
    if _query_has_logical(q, "country", table_infos):
        logical.add("country")

    if any(
        _query_has_logical(q, name, table_infos)
        for name in ("product_name", "category", "brand")
    ):
        logical.update({"product_id", "product_key"})
    if _query_has_logical(q, "category", table_infos):
        logical.add("category")
    if _query_has_logical(q, "brand", table_infos):
        logical.add("brand")

    customer_count_only = bool(
        re.search(r"(?:有过下单行为的)?客户(?:有多少|数)", q)
        and not any(token in q for token in ("会员等级", "会员级别", "每个客户", "客户排名", "大客户"))
    )
    if any(token in q for token in ("客户", "用户", "会员", "性别")):
        logical.add("customer_id")
        if not customer_count_only:
            logical.add("customer_key")
    elif "复购" in q:
        logical.add("customer_id")
    if any(token in q for token in ("客户", "用户", "大客户")) and not (
        re.search(r"客户(?:有多少|数)", q) and not any(token in q for token in ("每个客户", "客户排名", "大客户"))
    ):
        logical.add("customer_name")
    explicit_member_level = re.search(
        r"(?P<level>[\u4e00-\u9fff]{1,8})(?:会员等级|会员级别|会员)",
        q,
    )
    if (
        "会员等级" in q
        or "会员级别" in q
        or "会员层级" in q
        or (
            explicit_member_level is not None
            and explicit_member_level.group("level") not in {
                "各", "每个", "不同", "全部", "所有", "活跃", "客户", "用户",
            }
            and not any(
                token in explicit_member_level.group("level")
                for token in ("年", "月", "季度", "销售", "订单", "按")
            )
        )
    ):
        logical.add("member_level")
    if (
        any(token in q for token in ("性别", "男女", "女性", "男性"))
        or re.search(r"(?:男|女)(?:顾客|客户|用户|会员)", q)
    ):
        logical.add("gender")

    # Close explicit output/inner-group roles independently of family minima.
    # This is what keeps a readable entity and its relationship keys in recall
    # even when the same sentence also contains a filter-only measure.
    requested_dimensions = set(_dimension_logicals(q, table_infos))
    logical.update(requested_dimensions)
    if requested_dimensions & {"year", "quarter", "month", "day"}:
        logical.update({"date_id", "date_key"})
    if requested_dimensions & {"region_name", "province", "country"}:
        logical.update({"region_id", "region_key"})
    if requested_dimensions & {"product_name", "category", "brand"}:
        logical.update({"product_id", "product_key"})
    if requested_dimensions & {"customer_name", "member_level", "gender"}:
        logical.update({"customer_id", "customer_key"})

    if (
        any(token in q for token in (
            "首笔订单到末笔订单", "首单到末单", "首次下单到最后下单",
            "第一笔订单到最后一笔订单",
        ))
        and any(token in q for token in ("跨度", "间隔", "相隔", "持续"))
    ):
        logical.update({
            "order_id", "date_id", "customer_id", "customer_key",
            "customer_name",
        })

    exact_special = {
        "inventory_status": {"order_id", "product_id", "product_key"},
        "repeat_rate": (
            {"order_id", "customer_id", "customer_key", "member_level"}
            if "会员等级" in q or "会员级别" in q
            else {"order_id", "customer_id"}
        ),
        "period_compare": {"order_id", "order_amount", "date_id", "date_key", "year"},
        "window_mom": {"order_id", "order_amount", "date_id", "date_key", "year", "month"},
        "window_running_total": {"order_id", "order_amount", "date_id", "date_key", "year", "month"},
        "top_share": {"order_id", "order_amount", "customer_id"},
        "active_both_periods": {"order_id", "customer_id", "date_id", "date_key", "year"},
        "first_event": {"order_id", "customer_id", "customer_key", "customer_name", "date_id"},
        "cohort_new": {"order_id", "customer_id", "date_id", "date_key", "year", "month"},
        "segmentation": {"order_id", "order_amount"},
        "above_average_customer": {
            "order_id", "customer_id", "customer_key", "customer_name", "order_amount",
        },
        "above_average_brand": {
            "order_id", "product_id", "product_key", "brand", "order_amount",
        },
        "entity_aggregate_average": {
            "order_id", "customer_id", "customer_key", "order_amount",
        },
        "having_threshold": {
            "order_id", "customer_id", "customer_key", "customer_name", "order_amount",
        },
    }
    if family in exact_special:
        # A semantic-family minimum is additive.  Replacing the independently
        # detected fields here used to erase explicit axes and filters, for
        # example ``gender`` and ``year`` in “2025 年男女客户各自复购率”.
        # Preserve those explicit semantic fields and their join closure, but
        # do not retain a generic entity mention such as “客户” as a display
        # dimension when the family only needs the fact-side customer id.
        detected_logical = set(logical)
        logical = set(exact_special[family])
        semantic_context = detected_logical & {
            "year", "quarter", "month", "day",
            "region_name", "province", "country",
            "category", "brand", "member_level", "gender",
        }
        logical.update(semantic_context)
        if semantic_context & {"year", "quarter", "month", "day"}:
            logical.update({"date_id", "date_key"})
        if semantic_context & {"region_name", "province", "country"}:
            logical.update({"region_id", "region_key"})
        if semantic_context & {"category", "brand"}:
            logical.update({"product_id", "product_key"})
        if semantic_context & {"member_level", "gender"}:
            logical.update({"customer_id", "customer_key"})
    if family == "anti_join":
        # The anti side is always proved by the nullable fact event.  Resolving
        # only the preserved dimension key would make the compiler emit
        # ``f.order_id IS NULL`` without that field being part of the audited
        # metadata contract.
        logical.add("order_id")
        if "日期" in q:
            logical.update({"date_id", "date_key", "year"})
        elif any(token in q for token in ("客户", "用户")):
            logical.update({"customer_id", "customer_key", "customer_name"})
        else:
            logical.update({"product_id", "product_key", "product_name"})

    requirements: dict[str, set[str]] = {}
    for name in logical:
        spec = FIELD_SPECS[name]
        requirements.setdefault(spec["table"], set()).add(spec["column"])
    return requirements


def _dimension_logicals(
    query: str,
    table_infos: Sequence[Mapping[str, Any]] | None = None,
) -> list[str]:
    q = _normalize_text(query)
    output: list[str] = []
    # Calendar dimensions come first to preserve natural chronological output.
    if "每年" in q or "按年" in q:
        output.append("year")
    if any(token in q for token in ("每季度", "每个季度", "各季度", "按季度", "每季")):
        if "year" not in output and "每个季度" in q and not _years(q):
            output.append("year")
        output.append("quarter")
    if any(token in q for token in ("每月", "各月", "按月", "月份")):
        output.append("month")
    if any(token in q for token in (
        "各地区", "每个地区", "按地区", "各销售区域", "按销售区域",
        "各区域", "每个区域", "按区域", "各大区", "每个大区", "按大区",
    )) or (
        any(token in q for token in ("地区", "销售区域"))
        and any(token in q for token in ("最高", "最低", "最多", "最少", "排名"))
    ):
        output.append("region_name")
    if _query_has_logical(q, "province", table_infos) and (
        re.search(r"(?:各|按|每个)省份", q)
        or (
            "省份" in q
            and any(token in q for token in ("最高", "最低", "最多", "最少", "排名"))
        )
    ):
        output.append("province")
    if "各国家" in q or re.search(r"以国家(?:为维度|作为分组|分组)", q):
        output.append("country")
    if any(token in q for token in (
        "各品类", "每个品类", "每种品类", "按品类",
        "各类别", "每个类别", "每种类别", "按类别",
        "各商品类别", "每个商品类别", "按商品类别",
        "各产品类别", "每个产品类别", "按产品类别",
        "各商品品类", "每个商品品类", "按商品品类",
        "各产品品类", "每个产品品类", "按产品品类",
        "类别内", "所属类别内", "所属品类内",
    )):
        output.append("category")
    if any(token in q for token in ("各品牌", "每个品牌", "按品牌")) or (
        "品牌" in q and any(token in q for token in ("最高", "最低", "最多", "最少", "排名"))
    ):
        output.append("brand")
    if (
        ("商品" in q or "产品" in q)
        and any(token in q for token in ("各商品", "各产品", "每种", "每个", "最高", "最低", "前", "后", "排名"))
        and not any(token in q for token in (
            "商品总数", "产品总数", "商品类别", "产品类别",
            "商品品类", "产品品类", "商品品牌", "产品品牌",
        ))
    ):
        output.append("product_name")
    if (
        ("客户" in q or "用户" in q)
        and any(token in q for token in ("每个", "排名", "前", "后", "大客户", "超过", "高于"))
        and "客户有多少" not in q
    ):
        output.append("customer_name")
    if "会员等级" in q or "会员级别" in q or "会员层级" in q:
        output.append("member_level")
    if _query_has_logical(q, "gender", table_infos) or any(
        token in q for token in ("男女", "男性", "女性")
    ):
        output.append("gender")
    if any(token in q for token in ("列出", "哪些", "名单", "列表")):
        output.extend(
            logical
            for logical in _COUNTABLE_GROUP_LOGICALS
            if _query_has_logical(q, logical, table_infos)
        )
    return list(dict.fromkeys(output))


def _measure_specs(
    query: str,
    table_infos: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    q = _normalize_text(query)
    output: list[dict[str, Any]] = []
    comparisons = _numeric_comparisons(q, table_infos)
    row_filter_fields = {
        str(item["logical"])
        for item in comparisons
        if not bool(item["aggregate_scope"])
    }
    aggregate_filter_fields = {
        str(item["logical"])
        for item in comparisons
        if bool(item["aggregate_scope"])
    }
    two_metrics = (
        _query_has_logical(q, "order_id", table_infos)
        and _query_has_logical(q, "order_amount", table_infos)
        and "order_amount" not in row_filter_fields
    )
    if two_metrics or _query_has_logical(q, "order_id", table_infos):
        output.append({
            "logical": "order_id", "function": "COUNT", "alias": "order_count",
            # 订单事实表允许一笔订单对应多行明细；通用“订单数/笔数”
            # 合同统计业务订单，而不是物理行数。
            "distinct": True,
        })
    if re.search(
        r"(?:有过下单行为的)?(?:不同|去重|购买|活跃)?客户(?:有多少|数|数量)",
        q,
    ) or (
        "按会员等级分别有多少" in q
    ):
        output.append({
            "logical": "customer_id", "function": "COUNT",
            "alias": "active_customer_count" if "会员等级" not in q else "customer_count",
            "distinct": True,
        })
    counted_dimension = _counted_dimension_measure(q, table_infos)
    if counted_dimension:
        output.append({
            "logical": counted_dimension,
            "function": "COUNT",
            "alias": f"{counted_dimension}_count",
            "distinct": True,
        })
    explicit_non_amount_measure = (
        transaction_item_quantity_requested(q, table_infos)
        or any(token in q for token in (
            "订单数", "订单总数", "订单数量", "订单量", "多少笔订单",
        ))
    )
    amount_requested = (
        _query_has_logical(q, "order_amount", table_infos)
        and (
            "order_amount" not in row_filter_fields
            or "order_amount" in aggregate_filter_fields
        )
    ) or ("大客户" in q and not explicit_non_amount_measure)
    if amount_requested:
        function = "AVG" if any(token in q for token in (
            "平均订单金额", "平均每笔", "每笔平均", "平均单笔",
            "单笔平均", "aov",
        )) else "SUM"
        if "gmv" in q:
            alias = "gmv"
        elif "aov" in q:
            alias = "aov"
        elif function == "AVG":
            alias = "avg_amount"
        else:
            alias = "total_amount"
        output.append({
            "logical": "order_amount", "function": function, "alias": alias,
            "distinct": False,
        })
    quantity_requested = transaction_item_quantity_requested(q, table_infos)
    if quantity_requested and "order_quantity" not in row_filter_fields:
        function = "AVG" if "平均" in q else "SUM"
        if function == "AVG":
            alias = "avg_quantity"
        elif any(token in q for token in (
            "成交", "售出", "卖出", "件商品", "件产品",
        )):
            alias = "total_quantity"
        else:
            alias = "total_sales"
        output.append({
            "logical": "order_quantity", "function": function, "alias": alias,
            "distinct": False,
        })
    # Deduplicate a cue such as "订单量和销售额" deterministically.
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for item in output:
        unique[(item["logical"], item["function"])] = item
    return list(unique.values())


def semantic_question_cues(
    query: str,
    table_infos: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Detect explicit question roles independently from compiler closure.

    Coverage auditing consumes this function directly instead of trusting that
    family selection and compilation already noticed every user cue.
    """

    q = _normalize_text(query)
    family = _family(q, table_infos)
    comparisons = _numeric_comparisons(q, table_infos)
    operations: list[str] = []
    if family == "conditional_aggregate":
        operations.append("conditional_aggregate")
    if family in {"having_aggregate", "qualifying_group_count"}:
        operations.append("having")
    if family == "qualifying_group_count":
        operations.append("count_qualified_groups")
    if family == "entity_aggregate_average":
        operations.append("entity_aggregate_average")
    if _is_percentile_request(q):
        operations.append("percentile_rank")
        if any(token in q for token in ("所属", "组内", "类别内", "品类内")):
            operations.append("partitioned_rank")
    return {
        "family": family,
        "dimensions": _dimension_logicals(q, table_infos),
        "measures": _measure_specs(q, table_infos),
        "comparisons": comparisons,
        "operations": list(dict.fromkeys(operations)),
        "counted_group": _counted_group_logical(q, table_infos),
        "output_shape": (
            "scalar" if family == "qualifying_group_count" else None
        ),
    }


def _required_logicals(
    family: str,
    query: str,
    dimensions: Sequence[str],
    measures: Sequence[Mapping[str, Any]],
    table_infos: Sequence[Mapping[str, Any]] | None = None,
) -> list[str]:
    requirements = semantic_metadata_requirements(
        query,
        table_infos=table_infos,
    )
    logical = {
        name
        for name, spec in FIELD_SPECS.items()
        if spec["column"] in requirements.get(spec["table"], set())
    }
    logical.update(dimensions)
    logical.update(str(item["logical"]) for item in measures)
    return sorted(logical)


def _relation_closure(
    required_tables: set[str],
    resolved: Mapping[str, Mapping[str, str]],
) -> tuple[list[dict[str, str]], list[str]]:
    if len(required_tables) <= 1:
        return [], []
    joins: list[dict[str, str]] = []
    missing: list[str] = []
    available = {
        (item["table"], item["column"])
        for item in resolved.values()
    }
    for left_table, left_column, right_table, right_column in TABLE_RELATIONS:
        if right_table not in required_tables:
            continue
        if "fact_order" not in required_tables:
            missing.append(f"relation:{left_table}->{right_table}")
            continue
        if (left_table, left_column) not in available or (right_table, right_column) not in available:
            missing.append(f"relation:{left_table}.{left_column}={right_table}.{right_column}")
            continue
        joins.append({
            "left_table": left_table,
            "left_column": left_column,
            "right_table": right_table,
            "right_column": right_column,
            "relationship": "many_to_one",
        })
    expected_dims = required_tables - {"fact_order"}
    joined_dims = {item["right_table"] for item in joins}
    missing.extend(f"relation:fact_order->{table}" for table in sorted(expected_dims - joined_dims))
    return joins, list(dict.fromkeys(missing))


def _base_ir(query: str, family: str) -> dict[str, Any]:
    return {
        "schema_version": IR_VERSION_V2,
        "query": query,
        "status": "partial",
        "intent": family,
        "semantic_family": family,
        "dimensions": [],
        "measures": [],
        "filters": [],
        "time": None,
        "joins": [],
        "order_by": [],
        "limit": None,
        "context_refs": [],
        "metric": None,
        "confidence": 0.0,
        "required_tables": [],
        "required_columns": {},
        "entity_projection": {},
        "semantic_slots": {},
        "compiler": {
            "supported": False,
            "mode": "llm_fallback",
            "template": family,
            "reason": "语义槽位尚未闭合",
        },
    }


def build_semantic_ir_v2(
    query: str,
    table_infos: Sequence[Mapping[str, Any]] | None,
    metric_infos: Sequence[Mapping[str, Any]] | None = None,
    chat_history: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Build an auditable v2 IR for a recognized reusable semantic family."""
    query_text = str(query or "").strip()
    family = _family(query_text, table_infos)
    if family is None:
        return None

    ir = _base_ir(query_text, family)
    q = _normalize_text(query_text)
    metric_hits = [
        metric
        for metric in (metric_infos or [])
        if metric.get("formula_rule")
        and any(
            _normalize_text(term) in q
            for term in (metric.get("name", ""), *(metric.get("alias") or []))
            if term
        )
    ]
    explicit_measure_override = (
        transaction_item_quantity_requested(q, table_infos)
        or any(token in q for token in (
            "订单数", "订单总数", "订单数量", "订单量", "多少笔订单",
        ))
    )
    if len(metric_hits) == 1 and not explicit_measure_override:
        rule_type = str((metric_hits[0].get("formula_rule") or {}).get("type") or "")
        ir["metric"] = {"name": metric_hits[0].get("name"), "rule_type": rule_type}
    years = _years(q)
    dimensions = _dimension_logicals(q, table_infos)
    measures = _measure_specs(q, table_infos)
    numeric_comparisons = _numeric_comparisons(q, table_infos)

    # Family-specific dimensions/measures and slots.
    slots: dict[str, Any] = {
        "years": years,
        "calendar_grain": (
            "month" if "month" in dimensions
            else "quarter" if "quarter" in dimensions
            else "year" if "year" in dimensions
            else None
        ),
        "zero_denominator": "null",
        "empty_result": "preserve_sql_aggregate_semantics",
    }
    if family == "conditional_aggregate":
        predicate_logical = next(
            (
                str(item["logical"])
                for item in numeric_comparisons
                if not bool(item["aggregate_scope"])
            ),
            "order_amount",
        )
        conditions = [
            dict(item)
            for item in numeric_comparisons
            if str(item["logical"]) == predicate_logical
            and not bool(item["aggregate_scope"])
        ]
        measures = [{
            "logical": "order_id",
            "function": "COUNT",
            "alias": "conditional_order_count",
            "distinct": True,
        }]
        slots.update({
            "conditional_field": predicate_logical,
            "conditional_predicates": conditions,
            "conditional_measure": "order_id",
            "output_shape": "grouped_conditional_counts",
        })
    elif family == "qualifying_group_count":
        aggregate_condition = next(
            (item for item in numeric_comparisons if bool(item["aggregate_scope"])),
            None,
        )
        aggregate_grain = _counted_group_logical(q, table_infos)
        dimensions = []
        measures = []
        qualifying_measure = (
            str(aggregate_condition["logical"])
            if aggregate_condition else "order_amount"
        )
        slots.update({
            "aggregate_grain": aggregate_grain,
            "qualifying_measure": qualifying_measure,
            "qualifying_aggregate": (
                "COUNT_DISTINCT"
                if qualifying_measure == "customer_id"
                else "SUM"
            ),
            "having_operator": (
                str(aggregate_condition["operator"])
                if aggregate_condition else None
            ),
            "having_threshold": (
                aggregate_condition["value"] if aggregate_condition else None
            ),
            "outer_aggregate": "count_qualified_groups",
            "output_shape": "scalar",
            "output_columns": [
                f"{aggregate_grain}_count" if aggregate_grain else "group_count"
            ],
        })
    elif family == "having_aggregate":
        aggregate_condition = next(
            (item for item in numeric_comparisons if bool(item["aggregate_scope"])),
            None,
        )
        slots.update({
            "aggregate_grain": dimensions[0] if dimensions else None,
            "qualifying_measure": (
                str(aggregate_condition["logical"])
                if aggregate_condition else "order_amount"
            ),
            "having_operator": (
                str(aggregate_condition["operator"])
                if aggregate_condition else None
            ),
            "having_threshold": (
                aggregate_condition["value"] if aggregate_condition else None
            ),
            "output_shape": "qualified_groups",
        })
    elif family == "distribution_rank":
        partition = "category" if "category" in dimensions else None
        entity_grain = next(
            (item for item in dimensions if item != partition),
            None,
        )
        slots.update({
            "aggregate_grain": entity_grain,
            "partition_keys": [partition] if partition else [],
            "window_function": "percent_rank",
            "rank_direction": (
                "ASC" if any(token in q for token in ("从低到高", "升序")) else "DESC"
            ),
            "output_shape": "partitioned_distribution_rank",
        })
    elif family == "period_compare":
        dimensions = ["month"] if any(token in q for token in ("月份", "按月", "每月")) else []
        measures = [{"logical": "order_amount", "function": "SUM", "alias": "total_amount"}]
        slots.update({
            "comparison_periods": years,
            "comparison_mode": "conditional_aggregation",
            "output_shape": "wide_period_columns",
        })
    elif family in {"window_mom", "window_running_total"}:
        dimensions = ["month"]
        measures = [{"logical": "order_amount", "function": "SUM", "alias": "total_amount"}]
        slots.update({
            "window_function": "lag" if family == "window_mom" else "running_sum",
            "partition_keys": [],
            "order_keys": ["month"],
        })
    elif family == "group_share":
        dimensions = dimensions or ["category"]
        measures = [{"logical": "order_amount", "function": "SUM", "alias": "total_amount"}]
        slots.update({
            "numerator": "group_sales",
            "denominator": "post_filter_total_sales",
            "ratio_scope": "all_groups_after_filters",
        })
    elif family == "top_share":
        dimensions = []
        measures = [{"logical": "order_amount", "function": "SUM", "alias": "total_amount"}]
        slots.update({
            "entity_key": "customer_id",
            "rank_type": "row_number",
            "rank_direction": "DESC",
            "top_n": _extract_limit(q, 10),
            "numerator": "top_n_entity_sales",
            "denominator": "all_entity_sales",
        })
    elif family == "top_per_group":
        if "每个地区" in q:
            dimensions = ["region_name", "product_name"]
            partition = ["region_name"]
            rank_dimension = "product_name"
        elif "每个季度" in q:
            dimensions = ["year", "quarter", "brand"]
            partition = ["year", "quarter"]
            rank_dimension = "brand"
        else:
            dimensions = ["year", "month"]
            partition = ["year"]
            rank_dimension = "month"
        measures = [{"logical": "order_amount", "function": "SUM", "alias": "total_amount"}]
        slots.update({
            "partition_keys": partition,
            "rank_dimension": rank_dimension,
            "rank_type": "row_number",
            "rank_direction": "DESC",
            "top_n": _extract_limit(q, 3 if "三个" in q else 1),
            "tie_breaker": rank_dimension,
        })
    elif family == "active_both_periods":
        dimensions = []
        measures = [{"logical": "customer_id", "function": "COUNT", "alias": "customer_count"}]
        slots.update({
            "comparison_periods": years[:2],
            "set_requirement": "present_in_all_periods",
            "subquery_grain": "customer_id",
        })
    elif family == "first_event":
        dimensions = ["customer_name"]
        measures = [{"logical": "date_id", "function": "MIN", "alias": "first_order_date"}]
        slots.update({"entity_key": "customer_id", "event_key": "date_id", "event_selector": "first"})
    elif family == "cohort_new":
        dimensions = ["month"]
        measures = [{"logical": "customer_id", "function": "COUNT", "alias": "new_customer_count"}]
        slots.update({
            "entity_key": "customer_id",
            "event_key": "date_id",
            "event_selector": "first",
            "cohort_key": "month",
            "observation_year": years[0] if years else None,
        })
    elif family == "repeat_rate":
        dimensions = ["member_level"] if "会员等级" in q or "会员级别" in q else []
        measures = [{"logical": "customer_id", "function": "RATIO", "alias": "repurchase_rate"}]
        slots.update({
            "entity_key": "customer_id",
            "event_key": "order_id",
            "repeat_threshold": 2,
            "numerator": "customers_with_event_count_gte_threshold",
            "denominator": "customers_in_scope",
        })
    elif family == "segmentation":
        dimensions = []
        measures = [{"logical": "order_id", "function": "COUNT", "alias": "order_count"}]
        numbers = list(dict.fromkeys(float(item) for item in _NUMBER_RE.findall(q)))
        slots.update({
            "segment_field": "order_amount",
            "boundaries": numbers[:2],
            "boundary_policy": ["lt", "lte", "else"],
            "labels": ["低金额", "中金额", "高金额"],
        })
    elif family == "above_average_customer":
        dimensions = ["customer_name"]
        measures = [{"logical": "order_amount", "function": "SUM", "alias": "total_amount"}]
        slots.update({
            "aggregate_grain": "customer_id",
            "subquery_grain": "customer_id",
            "outer_filter": "group_total_gt_average_group_total",
        })
    elif family == "above_average_brand":
        dimensions = ["brand"]
        measures = [{"logical": "order_amount", "function": "SUM", "alias": "total_amount"}]
        slots.update({
            "aggregate_grain": "brand",
            "subquery_grain": "brand",
            "outer_filter": "group_total_gt_average_group_total",
        })
    elif family == "entity_aggregate_average":
        # The visible dimensions stay outside; customer_id is an internal
        # grouping grain used to build one additive total per entity.
        dimensions = [
            item for item in dimensions if item not in {"customer_name"}
        ]
        measures = [{
            "logical": "order_amount",
            "function": "AVG_OF_SUM",
            "alias": "avg_entity_total",
            "distinct": False,
        }]
        slots.update({
            "aggregate_grain": "customer_id",
            "subquery_grain": "customer_id",
            "inner_aggregate": "SUM",
            "outer_aggregate": "AVG",
            "output_shape": "grouped_average_of_entity_totals",
        })
    elif family == "having_threshold":
        dimensions = ["customer_name"]
        measures = [{"logical": "order_amount", "function": "SUM", "alias": "total_amount"}]
        numbers = [float(item) for item in _NUMBER_RE.findall(q)]
        slots.update({"having_operator": "gt", "having_threshold": numbers[-1] if numbers else None})
    elif family == "inventory_status":
        dimensions = []
        measures = [
            {"logical": "product_key", "function": "COUNT_DISTINCT", "alias": "total_products"},
            {"logical": "product_key", "function": "COUNT_DISTINCT_SOLD", "alias": "sold_products"},
            {"logical": "product_key", "function": "COUNT_DISTINCT_UNSOLD", "alias": "unsold_products"},
        ]
        slots.update({"preserved_entity": "product", "relationship_state": "sold_and_unsold"})
    elif family == "anti_join":
        dimensions = (
            ["date_key"] if "日期" in q
            else ["customer_name"] if any(token in q for token in ("客户", "用户"))
            else ["product_name"]
        )
        measures = []
        slots.update({
            "preserved_entity": dimensions[0],
            "anti_join_strategy": "left_join_is_null",
            "observation_year": years[0] if years else None,
        })

    logical_requirements = _required_logicals(
        family, q, dimensions, measures, table_infos,
    )
    for slot_logical in (
        slots.get("aggregate_grain"),
        slots.get("qualifying_measure"),
        slots.get("conditional_field"),
        *(slots.get("partition_keys") or []),
    ):
        if slot_logical and str(slot_logical) in FIELD_SPECS:
            logical_requirements.append(str(slot_logical))
    logical_requirements = sorted(set(logical_requirements))
    resolved, missing = _resolve_fields(table_infos, logical_requirements)

    def dimension_item(logical: str) -> dict[str, Any]:
        field = resolved[logical]
        return {
            "table": field["table"],
            "column": field["column"],
            "logical": logical,
            "source": "semantic_slot_v2",
        }

    def measure_item(item: Mapping[str, Any]) -> dict[str, Any]:
        field = resolved[str(item["logical"])]
        return {
            "table": field["table"],
            "column": field["column"],
            "logical": item["logical"],
            "function": item["function"],
            "alias": item["alias"],
            "distinct": bool(item.get("distinct")),
            "source": "semantic_slot_v2",
        }

    if not missing:
        ir["dimensions"] = [dimension_item(item) for item in dimensions]
        ir["measures"] = [measure_item(item) for item in measures]

    filters: list[dict[str, Any]] = []
    if family not in {"period_compare", "active_both_periods"} and len(years) == 1:
        filters.append({"logical": "year", "operator": "eq", "value": years[0]})
    if family not in {
        "conditional_aggregate", "qualifying_group_count", "having_aggregate",
    }:
        filters.extend(
            {
                "logical": str(item["logical"]),
                "operator": str(item["operator"]),
                "value": item["value"],
            }
            for item in numeric_comparisons
            if not bool(item["aggregate_scope"])
        )
    excluded = _extract_excluded_literal(query_text)
    if excluded:
        logical = "category" if "品类" in q or "类别" in q else "region_name"
        filters.append({"logical": logical, "operator": "neq", "value": excluded})
    elif "各品类" not in q and "每个品类" not in q:
        category = _extract_dimension_literal(query_text, "品类")
        if category:
            filters.append({"logical": "category", "operator": "eq", "value": category})
    if "各地区" not in q and "每个地区" not in q:
        region = _extract_dimension_literal(query_text, "地区")
        if region:
            filters.append({"logical": "region_name", "operator": "eq", "value": region})

    # Filter fields must be resolved too. Values remain in runtime state and are
    # excluded from persisted compact evidence.
    filter_logicals = [str(item["logical"]) for item in filters]
    filter_resolved, filter_missing = _resolve_fields(table_infos, filter_logicals)
    resolved.update(filter_resolved)
    missing.extend(filter_missing)
    ir["filters"] = filters

    required_tables = {item["table"] for item in resolved.values()}
    joins, relation_missing = _relation_closure(required_tables, resolved)
    missing.extend(relation_missing)
    missing = list(dict.fromkeys(missing))

    required_columns: dict[str, list[str]] = {}
    for field in resolved.values():
        required_columns.setdefault(field["table"], []).append(field["column"])
    ir["required_columns"] = {
        table: sorted(set(columns)) for table, columns in sorted(required_columns.items())
    }
    ir["required_tables"] = sorted(required_tables)
    ir["joins"] = joins
    ir["semantic_slots"] = slots
    ir["time"] = {
        "years": years,
        "grain": slots.get("calendar_grain"),
    } if years or slots.get("calendar_grain") else None
    ir["limit"] = (
        slots.get("top_n")
        if family in {"top_share", "top_per_group"}
        else _extract_limit(
            q,
            1 if family == "aggregate" and any(
                token in q for token in ("最高", "最低", "最多", "最少")
            ) else None,
        )
    )
    direction = (
        "ASC"
        if any(token in q for token in ("最低", "最少", "最短", "升序", "从低到高"))
        else "DESC"
    )
    if family == "aggregate" and measures:
        if dimensions:
            if dimensions[0] in {"year", "quarter", "month"}:
                ir["order_by"] = [{"target": item, "direction": "ASC"} for item in dimensions]
            elif len(dimensions) > 1:
                ir["order_by"] = [
                    {"target": dimensions[0], "direction": "ASC"},
                    {"target": measures[-1]["alias"], "direction": direction},
                ]
            else:
                ir["order_by"] = [{"target": measures[-1]["alias"], "direction": direction}]
    elif family in {"group_share", "above_average_customer", "above_average_brand"}:
        ir["order_by"] = [{"target": "total_amount", "direction": "DESC"}]
    elif family == "first_event":
        ir["order_by"] = [
            {"target": "first_order_date", "direction": "ASC"},
            {"target": "customer_name", "direction": "ASC"},
        ]

    display_logicals = {"region_name", "product_name", "customer_name"}
    ir["entity_projection"] = {
        item: {
            "id": {
                "region_name": "region_id",
                "product_name": "product_id",
                "customer_name": "customer_id",
            }[item],
            "display": item,
        }
        for item in dimensions if item in display_logicals
    }

    if missing:
        ir["compiler"]["reason"] = f"语义槽位或关系闭包缺失：{missing}"
        ir["compiler"]["missing_slots"] = missing
        return ir
    if family == "aggregate" and not measures:
        ir["compiler"]["reason"] = "未解析到唯一度量"
        ir["compiler"]["missing_slots"] = ["measure"]
        return ir
    if family == "period_compare" and len(years) < 2:
        ir["compiler"]["reason"] = "周期比较至少需要两个明确周期"
        ir["compiler"]["missing_slots"] = ["comparison_periods"]
        return ir
    if family == "segmentation" and len(slots.get("boundaries") or []) < 2:
        ir["compiler"]["reason"] = "分段语义缺少两个明确边界"
        ir["compiler"]["missing_slots"] = ["segment_boundaries"]
        return ir
    if family in {"cohort_new"} and slots.get("observation_year") is None:
        ir["compiler"]["reason"] = "cohort 统计缺少观察年份"
        ir["compiler"]["missing_slots"] = ["observation_year"]
        return ir

    ir["status"] = "complete"
    ir["confidence"] = 0.995
    ir["resolved_fields"] = resolved
    ir["compiler"] = {
        "supported": True,
        "mode": COMPILER_MODE_V2,
        "template": family,
        "reason": "语义槽位、字段与关系闭包完整",
        "contract_version": "semantic-compiler-contract-v2",
    }
    # Fail closed before the IR can be advertised as deterministic. A family
    # template may be syntactically available while still omitting a table or
    # operator required by the resolved IR (for example a scoped repeat-rate
    # request). Such requests belong on the model path, not a partial template.
    try:
        candidate_sql = compile_semantic_ir_v2(ir)
        if not candidate_sql:
            raise ValueError("semantic_ir_v2.empty_compilation")
        validate_compiled_semantic_ir(candidate_sql, ir)
    except (ValueError, sqlglot.errors.ParseError) as exc:
        ir["status"] = "partial"
        ir["confidence"] = 0.0
        ir["compiler"] = {
            **ir["compiler"],
            "supported": False,
            "mode": "llm_fallback",
            "reason": "编译结果未满足语义 IR 结构合同",
            "validation_error": str(exc),
        }
    return ir


def _field(ir: Mapping[str, Any], logical: str, alias: str | None = None) -> str:
    item = (ir.get("resolved_fields") or {}).get(logical)
    if not isinstance(item, Mapping):
        raise ValueError(f"missing resolved semantic field: {logical}")
    table = str(item.get("table") or "")
    column = str(item.get("column") or "")
    if not _IDENTIFIER_RE.fullmatch(table) or not _IDENTIFIER_RE.fullmatch(column):
        raise ValueError(f"unsafe semantic identifier: {logical}")
    prefix = alias if alias is not None else _TABLE_ALIASES.get(table)
    return f"{prefix}.{column}" if prefix else column


def _name(ir: Mapping[str, Any], logical: str) -> str:
    item = (ir.get("resolved_fields") or {}).get(logical)
    column = str((item or {}).get("column") or "")
    if not _IDENTIFIER_RE.fullmatch(column):
        raise ValueError(f"unsafe semantic identifier: {logical}")
    return column


def _literal(value: Any) -> str:
    if isinstance(value, bool):
        return exp.Boolean(this=value).sql(dialect="mysql")
    if isinstance(value, (int, float)):
        return exp.Literal.number(value).sql(dialect="mysql")
    return exp.Literal.string(str(value)).sql(dialect="mysql")


def _fact_from_and_joins(ir: Mapping[str, Any]) -> str:
    required = set(ir.get("required_tables") or [])
    sql = "fact_order f"
    for table in ("dim_date", "dim_region", "dim_product", "dim_customer"):
        if table not in required:
            continue
        relation = next(
            (
                item for item in ir.get("joins") or []
                if item.get("left_table") == "fact_order" and item.get("right_table") == table
            ),
            None,
        )
        if relation is None:
            raise ValueError(f"missing relation closure for {table}")
        alias = _TABLE_ALIASES[table]
        left = str(relation["left_column"])
        right = str(relation["right_column"])
        if not _IDENTIFIER_RE.fullmatch(left) or not _IDENTIFIER_RE.fullmatch(right):
            raise ValueError("unsafe relation identifier")
        sql += f" JOIN {table} {alias} ON f.{left} = {alias}.{right}"
    return sql


def _where_sql(ir: Mapping[str, Any]) -> str:
    conditions: list[str] = []
    for item in ir.get("filters") or []:
        logical = str(item.get("logical") or "")
        operator = str(item.get("operator") or "")
        column = _field(ir, logical)
        value = _literal(item.get("value"))
        operators = {"eq": "=", "neq": "<>", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
        if operator not in operators:
            raise ValueError(f"unsupported semantic filter: {operator}")
        conditions.append(f"{column} {operators[operator]} {value}")
    return f" WHERE {' AND '.join(conditions)}" if conditions else ""


def _comparison_sql(operator: str) -> str:
    operators = {
        "eq": "=", "neq": "<>", "gt": ">", "gte": ">=",
        "lt": "<", "lte": "<=",
    }
    if operator not in operators:
        raise ValueError(f"unsupported semantic comparison: {operator}")
    return operators[operator]


def _normalize_compiled_sql(sql: str) -> str:
    tree = sqlglot.parse_one(sql, read="mysql")
    # Deterministic compilation is SELECT-only by construction. Keep the guard
    # here so a future template cannot accidentally broaden this boundary.
    if not isinstance(tree, (exp.Select, exp.Union)) and tree.find(exp.Select) is None:
        raise ValueError("semantic compiler emitted a non-select statement")
    return tree.sql(dialect="mysql")


def compile_semantic_ir_v2(ir: Mapping[str, Any] | None) -> str | None:
    """Compile a complete v2 IR and re-emit it through SQLGlot."""
    if not ir or not (ir.get("compiler") or {}).get("supported"):
        return None
    if (ir.get("compiler") or {}).get("mode") != COMPILER_MODE_V2:
        return None
    family = str(ir.get("semantic_family") or ir.get("intent") or "")
    slots = dict(ir.get("semantic_slots") or {})
    fact_from = _fact_from_and_joins(ir)

    if family == "conditional_aggregate":
        dimensions = [
            (str(item["logical"]), _field(ir, str(item["logical"])))
            for item in ir.get("dimensions") or []
        ]
        predicate_field = _field(ir, str(slots["conditional_field"]))
        counted_field = _field(ir, str(slots.get("conditional_measure") or "order_id"))
        conditions = list(slots.get("conditional_predicates") or [])
        projections = [column for _, column in dimensions]
        for index, condition in enumerate(conditions, start=1):
            operator = _comparison_sql(str(condition.get("operator") or ""))
            projections.append(
                "COUNT(DISTINCT CASE WHEN "
                f"{predicate_field} {operator} {_literal(condition.get('value'))} "
                f"THEN {counted_field} END) AS condition_{index}_count"
            )
        sql = f"SELECT {', '.join(projections)} FROM {fact_from}{_where_sql(ir)}"
        if dimensions:
            group_columns = ", ".join(column for _, column in dimensions)
            sql += f" GROUP BY {group_columns} ORDER BY {group_columns}"
        return _normalize_compiled_sql(sql)

    if family in {"having_aggregate", "qualifying_group_count"}:
        grain_logical = str(slots.get("aggregate_grain") or "")
        measure_logical = str(slots.get("qualifying_measure") or "")
        grain = _field(ir, grain_logical)
        measure = _field(ir, measure_logical)
        operator = _comparison_sql(str(slots.get("having_operator") or ""))
        threshold = _literal(slots.get("having_threshold"))
        aggregate = str(slots.get("qualifying_aggregate") or "SUM")
        if aggregate == "COUNT_DISTINCT":
            having_value = f"COUNT(DISTINCT {measure})"
        elif aggregate == "SUM":
            having_value = f"SUM({measure})"
        else:
            raise ValueError(f"unsupported qualifying aggregate: {aggregate}")
        inner = (
            f"SELECT {grain} FROM {fact_from}{_where_sql(ir)} "
            f"GROUP BY {grain} HAVING {having_value} {operator} {threshold}"
        )
        if family == "qualifying_group_count":
            alias = f"{grain_logical}_count"
            return _normalize_compiled_sql(
                f"SELECT COUNT(*) AS {alias} FROM ({inner}) qualified_groups"
            )
        sql = (
            f"SELECT {grain}, SUM({measure}) AS total_amount "
            f"FROM {fact_from}{_where_sql(ir)} GROUP BY {grain} "
            f"HAVING SUM({measure}) {operator} {threshold} "
            f"ORDER BY total_amount DESC, {grain} ASC"
        )
        return _normalize_compiled_sql(sql)

    if family == "period_compare":
        projections = [
            (
                f"SUM(CASE WHEN {_field(ir, 'year')} = {int(year)} "
                f"THEN {_field(ir, 'order_amount')} ELSE 0 END) AS sales_{int(year)}"
            )
            for year in slots["comparison_periods"]
        ]
        dimensions = list(ir.get("dimensions") or [])
        dim_sql = [f"{_field(ir, str(item['logical']))}" for item in dimensions]
        year_values = ", ".join(str(int(item)) for item in slots["comparison_periods"])
        sql = (
            f"SELECT {', '.join([*dim_sql, *projections])} FROM {fact_from} "
            f"WHERE {_field(ir, 'year')} IN ({year_values})"
        )
        if dimensions:
            sql += f" GROUP BY {', '.join(dim_sql)} ORDER BY {', '.join(dim_sql)}"
        return _normalize_compiled_sql(sql)

    if family == "entity_aggregate_average":
        dimensions = [
            (str(item["logical"]), _field(ir, str(item["logical"])))
            for item in ir.get("dimensions") or []
        ]
        entity = _field(ir, str(slots.get("aggregate_grain") or "customer_id"))
        amount = _field(ir, "order_amount")
        inner_dimensions = [column for _, column in dimensions]
        inner_projection = [*inner_dimensions, entity]
        inner = (
            f"SELECT {', '.join(inner_projection)}, SUM({amount}) AS entity_total "
            f"FROM {fact_from}{_where_sql(ir)} "
            f"GROUP BY {', '.join(inner_projection)}"
        )
        outer_dimensions = [
            _name(ir, logical) for logical, _ in dimensions
        ]
        projections = [*outer_dimensions, "AVG(entity_total) AS avg_entity_total"]
        sql = (
            f"WITH entity_totals AS ({inner}) "
            f"SELECT {', '.join(projections)} FROM entity_totals"
        )
        if outer_dimensions:
            sql += (
                f" GROUP BY {', '.join(outer_dimensions)}"
                f" ORDER BY {', '.join(outer_dimensions)}"
            )
        return _normalize_compiled_sql(sql)

    if family in {"window_mom", "window_running_total"}:
        month = _field(ir, "month")
        amount = _field(ir, "order_amount")
        year = next((item.get("value") for item in ir.get("filters") or [] if item.get("logical") == "year"), None)
        where = f" WHERE {_field(ir, 'year')} = {int(year)}" if year is not None else ""
        cte = (
            f"SELECT {month} AS month, SUM({amount}) AS total_amount "
            f"FROM {fact_from}{where} GROUP BY {month}"
        )
        if family == "window_running_total":
            sql = (
                "WITH monthly_sales AS (" + cte + ") "
                "SELECT month, total_amount, "
                "SUM(total_amount) OVER (ORDER BY month) AS cumulative_amount "
                "FROM monthly_sales ORDER BY month"
            )
        else:
            sql = (
                "WITH monthly_sales AS (" + cte + "), "
                "with_previous AS (SELECT month, total_amount, "
                "LAG(total_amount) OVER (ORDER BY month) AS previous_amount "
                "FROM monthly_sales) "
                "SELECT month, total_amount, "
                "(total_amount - previous_amount) / NULLIF(previous_amount, 0) "
                "AS mom_growth_rate FROM with_previous ORDER BY month"
            )
        return _normalize_compiled_sql(sql)

    if family == "group_share":
        dims = [_field(ir, str(item["logical"])) for item in ir.get("dimensions") or []]
        amount = _field(ir, "order_amount")
        sql = (
            f"SELECT {', '.join(dims)}, SUM({amount}) AS total_amount, "
            f"SUM({amount}) / NULLIF(SUM(SUM({amount})) OVER (), 0) AS sales_share "
            f"FROM {fact_from}{_where_sql(ir)} GROUP BY {', '.join(dims)} "
            "ORDER BY total_amount DESC"
        )
        return _normalize_compiled_sql(sql)

    if family == "top_share":
        n = int(slots["top_n"])
        sql = (
            "WITH customer_sales AS ("
            f"SELECT {_field(ir, 'customer_id')} AS customer_id, "
            f"SUM({_field(ir, 'order_amount')}) AS total_amount "
            f"FROM {fact_from} GROUP BY {_field(ir, 'customer_id')}"
            "), ranked AS ("
            "SELECT customer_id, total_amount, "
            "ROW_NUMBER() OVER (ORDER BY total_amount DESC, customer_id) AS rn "
            "FROM customer_sales) "
            f"SELECT SUM(CASE WHEN rn <= {n} THEN total_amount ELSE 0 END) / "
            "NULLIF(SUM(total_amount), 0) "
            f"AS top{n}_sales_share FROM ranked"
        )
        return _normalize_compiled_sql(sql)

    if family == "top_per_group":
        partition = [str(item) for item in slots["partition_keys"]]
        rank_dimension = str(slots["rank_dimension"])
        group_logicals = [*partition, rank_dimension]
        aliases = [logical for logical in group_logicals]
        group_select = ", ".join(
            f"{_field(ir, logical)} AS {logical}" for logical in group_logicals
        )
        group_by = ", ".join(_field(ir, logical) for logical in group_logicals)
        partition_sql = ", ".join(partition)
        n = int(slots["top_n"])
        sql = (
            f"WITH grouped_values AS (SELECT {group_select}, "
            f"SUM({_field(ir, 'order_amount')}) AS total_amount "
            f"FROM {fact_from}{_where_sql(ir)} GROUP BY {group_by}), "
            f"ranked AS (SELECT {', '.join(aliases)}, total_amount, "
            f"ROW_NUMBER() OVER (PARTITION BY {partition_sql} "
            f"ORDER BY total_amount DESC, {rank_dimension}) AS rn FROM grouped_values) "
            f"SELECT {', '.join(aliases)}, total_amount FROM ranked "
            f"WHERE rn <= {n} ORDER BY {partition_sql}, rn"
        )
        return _normalize_compiled_sql(sql)

    if family == "active_both_periods":
        periods = [int(item) for item in slots["comparison_periods"]]
        values = ", ".join(str(item) for item in periods)
        sql = (
            "SELECT COUNT(*) AS customer_count FROM ("
            f"SELECT {_field(ir, 'customer_id')} AS customer_id FROM {fact_from} "
            f"WHERE {_field(ir, 'year')} IN ({values}) "
            f"GROUP BY {_field(ir, 'customer_id')} "
            f"HAVING COUNT(DISTINCT {_field(ir, 'year')}) = {len(periods)}"
            ") active_all_periods"
        )
        return _normalize_compiled_sql(sql)

    if family == "first_event":
        sql = (
            f"SELECT {_field(ir, 'customer_name')}, "
            f"MIN({_field(ir, 'date_id')}) AS first_order_date "
            f"FROM {fact_from} GROUP BY {_field(ir, 'customer_name')} "
            f"ORDER BY first_order_date, {_field(ir, 'customer_name')}"
        )
        return _normalize_compiled_sql(sql)

    if family == "cohort_new":
        year = int(slots["observation_year"])
        # first_orders is intentionally aggregated before the date join so the
        # cohort is based on each customer's first event, not events in the year.
        sql = (
            "WITH first_orders AS ("
            f"SELECT {_field(ir, 'customer_id', alias='')} AS customer_id, "
            f"MIN({_field(ir, 'date_id', alias='')}) AS first_order_date "
            "FROM fact_order GROUP BY "
            f"{_field(ir, 'customer_id', alias='')}) "
            f"SELECT {_field(ir, 'month')} AS month, COUNT(*) AS new_customer_count "
            "FROM first_orders fo JOIN dim_date d "
            f"ON fo.first_order_date = {_field(ir, 'date_key')} "
            f"WHERE {_field(ir, 'year')} = {year} "
            f"GROUP BY {_field(ir, 'month')} ORDER BY {_field(ir, 'month')}"
        )
        return _normalize_compiled_sql(sql)

    if family == "repeat_rate":
        threshold = int(slots["repeat_threshold"])
        customer_id = _name(ir, "customer_id")
        order_id = _name(ir, "order_id")
        cte = (
            f"SELECT {customer_id}, COUNT({order_id}) AS order_count "
            f"FROM fact_order GROUP BY {customer_id}"
        )
        ratio = (
            f"SUM(CASE WHEN co.order_count >= {threshold} THEN 1 ELSE 0 END) / "
            "NULLIF(COUNT(*), 0) AS repurchase_rate"
        )
        if any(item.get("logical") == "member_level" for item in ir.get("dimensions") or []):
            sql = (
                f"WITH customer_orders AS ({cte}) "
                f"SELECT {_field(ir, 'member_level')}, {ratio} "
                "FROM customer_orders co JOIN dim_customer c "
                f"ON co.customer_id = {_field(ir, 'customer_key')} "
                f"GROUP BY {_field(ir, 'member_level')} ORDER BY {_field(ir, 'member_level')}"
            )
        else:
            plain_ratio = ratio.replace("co.order_count", "order_count")
            sql = (
                f"WITH customer_orders AS ({cte}) "
                f"SELECT {plain_ratio} FROM customer_orders"
            )
        return _normalize_compiled_sql(sql)

    if family == "segmentation":
        low, high = slots["boundaries"]
        amount = _field(ir, "order_amount", alias="")
        order_id = _field(ir, "order_id", alias="")
        labels = slots["labels"]
        segment = (
            f"CASE WHEN {amount} < {_literal(low)} THEN {_literal(labels[0])} "
            f"WHEN {amount} <= {_literal(high)} THEN {_literal(labels[1])} "
            f"ELSE {_literal(labels[2])} END"
        )
        sql = (
            f"SELECT {segment} AS amount_segment, COUNT({order_id}) AS order_count "
            "FROM fact_order GROUP BY amount_segment ORDER BY amount_segment"
        )
        return _normalize_compiled_sql(sql)

    if family == "above_average_customer":
        sql = (
            "WITH entity_sales AS ("
            f"SELECT {_field(ir, 'customer_id', alias='')} AS customer_id, "
            f"SUM({_field(ir, 'order_amount', alias='')}) AS total_amount "
            "FROM fact_order GROUP BY "
            f"{_field(ir, 'customer_id', alias='')}) "
            f"SELECT {_field(ir, 'customer_name')}, es.total_amount "
            "FROM entity_sales es JOIN dim_customer c "
            f"ON es.customer_id = {_field(ir, 'customer_key')} "
            "WHERE es.total_amount > (SELECT AVG(total_amount) FROM entity_sales) "
            "ORDER BY es.total_amount DESC"
        )
        return _normalize_compiled_sql(sql)

    if family == "above_average_brand":
        sql = (
            "WITH entity_sales AS ("
            f"SELECT {_field(ir, 'brand')} AS brand, "
            f"SUM({_field(ir, 'order_amount')}) AS total_amount "
            f"FROM {fact_from} GROUP BY {_field(ir, 'brand')}) "
            "SELECT brand, total_amount FROM entity_sales "
            "WHERE total_amount > (SELECT AVG(total_amount) FROM entity_sales) "
            "ORDER BY total_amount DESC"
        )
        return _normalize_compiled_sql(sql)

    if family == "having_threshold":
        threshold = slots["having_threshold"]
        amount = _field(ir, "order_amount")
        customer = _field(ir, "customer_name")
        sql = (
            f"SELECT {customer}, SUM({amount}) AS total_amount FROM {fact_from} "
            f"GROUP BY {customer} HAVING SUM({amount}) > {_literal(threshold)} "
            "ORDER BY total_amount DESC"
        )
        return _normalize_compiled_sql(sql)

    if family == "inventory_status":
        product = _field(ir, "product_key")
        order = _field(ir, "order_id")
        sql = (
            f"SELECT COUNT(DISTINCT {product}) AS total_products, "
            f"COUNT(DISTINCT CASE WHEN {order} IS NOT NULL THEN {product} END) AS sold_products, "
            f"COUNT(DISTINCT CASE WHEN {order} IS NULL THEN {product} END) AS unsold_products "
            "FROM dim_product p LEFT JOIN fact_order f "
            f"ON {_field(ir, 'product_key')} = {_field(ir, 'product_id')}"
        )
        return _normalize_compiled_sql(sql)

    if family == "anti_join":
        preserved = str(slots["preserved_entity"])
        if preserved == "date_key":
            base_table, base_alias, key, fact_key = "dim_date", "d", "date_key", "date_id"
        elif preserved == "customer_name":
            base_table, base_alias, key, fact_key = "dim_customer", "c", "customer_key", "customer_id"
        else:
            base_table, base_alias, key, fact_key = "dim_product", "p", "product_key", "product_id"
        projection = _field(ir, preserved)
        sql = (
            f"SELECT {projection} FROM {base_table} {base_alias} "
            f"LEFT JOIN fact_order f ON {_field(ir, key)} = {_field(ir, fact_key)} "
            f"WHERE {_field(ir, 'order_id')} IS NULL"
        )
        observation_year = slots.get("observation_year")
        if observation_year is not None and preserved == "date_key":
            sql = sql.replace(
                f"WHERE {_field(ir, 'order_id')} IS NULL",
                f"WHERE {_field(ir, 'year')} = {int(observation_year)} "
                f"AND {_field(ir, 'order_id')} IS NULL",
            )
        sql += f" ORDER BY {projection}"
        return _normalize_compiled_sql(sql)

    if family == "aggregate":
        dims = [
            (str(item["logical"]), _field(ir, str(item["logical"])))
            for item in ir.get("dimensions") or []
        ]
        projections = [column for _, column in dims]
        measure_expressions: list[tuple[str, str]] = []
        for item in ir.get("measures") or []:
            column = _field(ir, str(item["logical"]))
            function = str(item["function"]).upper()
            distinct = bool(item.get("distinct"))
            if function not in {"SUM", "AVG", "COUNT", "MIN", "MAX"}:
                raise ValueError(f"unsupported aggregate function: {function}")
            argument = f"DISTINCT {column}" if distinct else column
            expression = f"{function}({argument}) AS {item['alias']}"
            projections.append(expression)
            measure_expressions.append((str(item["alias"]), expression))
        sql = f"SELECT {', '.join(projections)} FROM {fact_from}{_where_sql(ir)}"
        if dims:
            sql += f" GROUP BY {', '.join(column for _, column in dims)}"
        for item in ir.get("order_by") or []:
            target = str(item["target"])
            resolved_target = next((column for logical, column in dims if logical == target), target)
            direction = "ASC" if str(item.get("direction")).upper() == "ASC" else "DESC"
            sql += (" ORDER BY " if " ORDER BY " not in sql else ", ") + f"{resolved_target} {direction}"
        if ir.get("limit"):
            sql += f" LIMIT {int(ir['limit'])}"
        return _normalize_compiled_sql(sql)

    return None


def validate_compiled_semantic_ir(sql: str, ir: Mapping[str, Any]) -> None:
    """Validate family-defining AST features for a v2 deterministic candidate."""
    if (ir.get("compiler") or {}).get("mode") != COMPILER_MODE_V2:
        return
    tree = sqlglot.parse_one(sql, read="mysql")
    family = str(ir.get("semantic_family") or "")
    required_tables = set(ir.get("required_tables") or [])
    cte_names = {
        str(item.alias_or_name or "").lower()
        for item in tree.find_all(exp.CTE)
    }
    physical_tables = {
        table.name
        for table in tree.find_all(exp.Table)
        if table.name.lower() not in cte_names
    }
    if not required_tables.issubset(physical_tables):
        raise ValueError(
            f"semantic_ir_v2.missing_table:{sorted(required_tables - physical_tables)}"
        )

    checks = {
        "conditional_aggregate": lambda: bool(tree.find(exp.Case)) and bool(
            tree.find(exp.Count)
        ),
        "qualifying_group_count": lambda: bool(tree.find(exp.Subquery))
        and bool(tree.find(exp.Count))
        and bool(tree.find(exp.Having)),
        "having_aggregate": lambda: bool(tree.find(exp.Having)),
        "period_compare": lambda: len(list(tree.find_all(exp.Case))) >= 2,
        "window_mom": lambda: bool(tree.find(exp.Lag)) and bool(tree.find(exp.Nullif)),
        "window_running_total": lambda: bool(tree.find(exp.Window)),
        "group_share": lambda: bool(tree.find(exp.Div)) and bool(tree.find(exp.Nullif)),
        "top_share": lambda: bool(tree.find(exp.RowNumber)) and bool(tree.find(exp.Div)),
        "top_per_group": lambda: bool(tree.find(exp.RowNumber)) and any(
            window.args.get("partition_by") for window in tree.find_all(exp.Window)
        ),
        "active_both_periods": lambda: bool(tree.find(exp.Count)) and bool(tree.find(exp.Having)),
        "first_event": lambda: bool(tree.find(exp.Min)),
        "cohort_new": lambda: bool(tree.find(exp.Min)) and bool(tree.find(exp.CTE)),
        "repeat_rate": lambda: bool(tree.find(exp.Case)) and bool(tree.find(exp.Nullif)),
        "segmentation": lambda: bool(tree.find(exp.Case)),
        "above_average_customer": lambda: bool(tree.find(exp.Avg)) and bool(tree.find(exp.CTE)),
        "above_average_brand": lambda: bool(tree.find(exp.Avg)) and bool(tree.find(exp.CTE)),
        "entity_aggregate_average": lambda: bool(tree.find(exp.Avg))
        and bool(tree.find(exp.Sum))
        and bool(tree.find(exp.CTE)),
        "having_threshold": lambda: bool(tree.find(exp.Having)),
        "inventory_status": lambda: len(list(tree.find_all(exp.Case))) >= 2,
        "anti_join": lambda: any(
            str(join.args.get("side") or "").upper() == "LEFT"
            for join in tree.find_all(exp.Join)
        ) and bool(tree.find(exp.Null)),
    }
    predicate = checks.get(family)
    if predicate is not None and not predicate():
        raise ValueError(f"semantic_ir_v2.family_contract_failed:{family}")
