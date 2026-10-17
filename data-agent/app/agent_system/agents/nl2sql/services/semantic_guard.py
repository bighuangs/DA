"""面向高置信 NL2SQL Contract 的确定性业务语义检查。"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import Scope, traverse_scope

from app.domain.query.safety.semantic_violation import (
    SemanticContractError,
    SemanticIRError,
    SemanticParseError,
)
from app.agent_system.agents.nl2sql.semantic_contract import (
    audit_sql_against_query_contract,
)
from app.agent_system.agents.nl2sql.semantic_ir_v2 import (
    COMPILER_MODE_V2,
    validate_compiled_semantic_ir,
)
from app.agent_system.agents.nl2sql.services.model_sql_policy import use_model_sql


_FOLLOW_UP_RE = re.compile(r"(?:^那|^那么|呢$|改成|换成|再看|再算)")
_ANTI_JOIN_QUERY_RE = re.compile(
    r"(?:没有|从来没有|从未|未曾|未产生|未下单|"
    r"无(?:任何)?(?:订单|下单|购买|交易))"
)
_STRONG_ENTITY_LIST_RE = re.compile(
    r"(?:哪些|名单|列表|明细|排行|排名|最高|最低|最多|最少|"
    r"前\s*\d*|后\s*\d*|超过|没有)"
)
_EXPLICIT_LIST_VERB_RE = re.compile(
    r"(?:列出|展示|显示|返回|给出)(?:全部|所有)?"
)
_PER_TIME_RE = re.compile(
    r"每(?:个)?(?:年|季度|季|月|周|星期|天|日|小时)"
)
_COUNT_QUESTION_RE = re.compile(
    r"(?:有)?多少(?:个|位|名|家|种)?|"
    r"(?:客户|用户|商品|产品|订单|品牌|品类|地区|省份)(?:数|数量)"
)
_COUNTABLE_ENTITY_TERMS = (
    "客户", "用户", "商品", "产品", "订单", "品牌", "品类", "地区", "省份",
    "月份", "季度", "年份",
)
_EXPLICIT_YEAR_RE = re.compile(
    r"(?<!\d)((?:19|20)\d{2})\s*(?:年|全年)"
)


def _parse(sql: str) -> exp.Expression:
    try:
        return sqlglot.parse_one(sql, read="mysql")
    except Exception as exc:
        raise SemanticParseError(
            f"语义校验前 SQL 解析失败：{exc}",
            evidence={"sql_hash": _safe_sql_hash(sql)},
        ) from exc


def _metadata_schemas(
    table_infos: Sequence[Mapping],
) -> dict[str, set[str]]:
    return {
        str(table.get("name") or "").lower(): {
            str(column.get("name") or "").lower()
            for column in (table.get("columns") or [])
            if column.get("name")
        }
        for table in table_infos
        if table.get("name")
    }


def _check_qualified_column_ownership(
    tree: exp.Expression,
    table_infos: Sequence[Mapping],
) -> None:
    """Reject a physical alias qualified with a column absent from its Meta."""

    schemas = _metadata_schemas(table_infos)
    aliases, ctes = _physical_aliases(tree)
    for column in tree.find_all(exp.Column):
        alias = str(column.table or "").lower()
        name = str(column.name or "").lower()
        if not alias or alias in ctes or alias not in aliases:
            continue
        table = aliases[alias]
        if name in schemas.get(table, set()):
            continue
        owners = sorted(
            candidate
            for candidate, columns in schemas.items()
            if name in columns
        )
        # A partial metadata payload cannot prove ownership.  The SQL engine
        # remains the syntax authority; this guard fires only when Meta names
        # at least one unique physical owner for the column.
        if not owners:
            continue
        # The fact side of a proven same-name PK/FK edge may be omitted from a
        # partial prompt payload.  Do not reinterpret such a join key as a
        # dimension-owner error; non-key business attributes remain guarded.
        if name.endswith("_id"):
            continue
        raise SemanticContractError(
            f"字段 {alias}.{name} 不属于元数据表 {table}；"
            f"可确认的字段所有者为 {owners or ['none']}，必须使用正确别名和关系连接",
            violation_code="semantic.column.wrong_owner",
            repairability="llm",
            repair_hint_code="replace_with_metadata_column_owner",
            evidence={
                "actual_table": table,
                "column": name,
                "expected_tables": owners,
            },
        )


def _safe_sql_hash(sql: str) -> str:
    from app.domain.query.safety.semantic_violation import stable_hash

    return stable_hash(sql)


def _final_select(tree: exp.Expression) -> exp.Select | None:
    return tree if isinstance(tree, exp.Select) else tree.find(exp.Select)


def _projected_columns(tree: exp.Expression) -> set[str]:
    # UNION/INTERSECT/EXCEPT 的各分支共同构成用户可见结果。只读取第一
    # 个 SELECT 会把第二来源中真实投影的业务维度误判为缺失；普通 CTE
    # 仍只检查最终 SELECT，继续防止上游分组后在最终结果中丢失维度。
    if isinstance(tree, (exp.Union, exp.Intersect, exp.Except)):
        selects = list(_set_operation_selects(tree))
    else:
        select = _final_select(tree)
        selects = [select] if select is not None else []
    return {
        column.name.lower()
        for select in selects
        for projection in select.expressions
        for column in projection.find_all(exp.Column)
    }


def _set_operation_selects(tree: exp.Expression):
    """Yield only the direct result branches of a top-level set operation."""
    if isinstance(tree, (exp.Union, exp.Intersect, exp.Except)):
        yield from _set_operation_selects(tree.this)
        yield from _set_operation_selects(tree.expression)
    elif isinstance(tree, exp.Subquery):
        yield from _set_operation_selects(tree.this)
    elif isinstance(tree, exp.Select):
        yield tree


def _plain_projection_columns(tree: exp.Expression) -> set[str]:
    """Columns returned as dimensions, excluding columns nested in aggregates."""
    select = _final_select(tree)
    if select is None:
        return set()
    columns: set[str] = set()
    for projection in select.expressions:
        if projection.find(exp.AggFunc) is not None:
            continue
        columns.update(column.name.lower() for column in projection.find_all(exp.Column))
    return columns


def _aggregated_columns(tree: exp.Expression) -> set[str]:
    return {
        column.name.lower()
        for aggregate in tree.find_all(exp.AggFunc)
        for column in aggregate.find_all(exp.Column)
    }


def _aggregate_projection_aliases(tree: exp.Expression) -> set[str]:
    """Aliases whose lineage is an aggregate in an upstream SELECT."""
    aliases: set[str] = set()
    for select in tree.find_all(exp.Select):
        for projection in select.expressions:
            if projection.find(exp.AggFunc) is None:
                continue
            alias = str(projection.alias_or_name or "").lower()
            if alias:
                aliases.add(alias)
    return aliases


def _window_projection_aliases(tree: exp.Expression) -> set[str]:
    """Aliases whose value is produced by an upstream window expression."""
    aliases: set[str] = set()
    for select in tree.find_all(exp.Select):
        for projection in select.expressions:
            if projection.find(exp.Window) is None:
                continue
            alias = str(projection.alias_or_name or "").lower()
            if alias:
                aliases.add(alias)
    return aliases


def _entity_listing_requested(query_text: str) -> bool:
    """Distinguish entity rows from counts grouped by time or dimensions.

    “哪些客户/销量前五商品” needs readable entity fields.  In contrast,
    “每个月有多少活跃客户” returns a time axis plus an aggregate, even though
    it contains both “每个” and an entity noun.  Treating both shapes as an
    entity list makes a valid month projection look like an unrelated field.
    """
    # A bounded entity set can be an intermediate population rather than the
    # user-visible grain.  For example, “每月前十客户的销售贡献率” asks for one
    # ratio per month; forcing customer_name into the final projection both
    # changes the denominator and destroys the requested time grain.  Keep
    # explicit row-list wording (及其/各自/名单/明细/分别) on the entity path.
    periodic_ranked_share = bool(
        _PER_TIME_RE.search(query_text)
        and re.search(r"(?:前|后)(?:\d+|[一二两三四五六七八九十百]+)", query_text)
        and any(token in query_text for token in (
            "贡献率", "贡献占比", "合计占比", "合计比例", "销售占比", "成交占比",
        ))
        and not any(token in query_text for token in (
            "名单", "明细", "分别", "各自", "及其", "哪些",
        ))
    )
    if periodic_ranked_share:
        return False
    # “把每件商品按活跃季度数分成三档，并统计商品数”以实体为中间
    # 分桶粒度，最终结果是分群汇总而不是商品明细。若把“每件商品”机械
    # 识别成实体列表，会错误要求最终结果投影 product_name。
    segmentation_summary = bool(
        re.search(r"(?:每个|每件|每位|每名).{0,48}(?:分成|划为|分为|分档|分段)", query_text)
        and re.search(r"(?:统计|汇总|计算).{0,20}(?:数|数量|平均|均值|占比)", query_text)
    )
    if segmentation_summary:
        return False
    cohort_summary = bool(
        re.search(
            r"(?:客户|用户).{0,24}按(?:首次|首笔|首单).{0,12}(?:年|月).{0,20}"
            r"(?:统计|汇总|计算).{0,12}(?:人数|客户数|用户数)",
            query_text,
        )
    )
    if cohort_summary:
        return False
    if _STRONG_ENTITY_LIST_RE.search(query_text):
        return True
    if (
        _EXPLICIT_LIST_VERB_RE.search(query_text)
        and not _COUNT_QUESTION_RE.search(query_text)
    ):
        return True
    has_per_entity_shape = "每个" in query_text or "每种" in query_text
    return bool(
        has_per_entity_shape
        and not _PER_TIME_RE.search(query_text)
        and not _COUNT_QUESTION_RE.search(query_text)
    )


def _counted_entity_terms(query_text: str) -> set[str]:
    """Return entity nouns used as a measure rather than a grouping axis."""
    return {
        term
        for term in _COUNTABLE_ENTITY_TERMS
        if re.search(rf"多少[^，。；]*{re.escape(term)}", query_text)
        or re.search(rf"{re.escape(term)}(?:数|数量|人数)", query_text)
        or re.search(rf"{re.escape(term)}(?:总)?件数", query_text)
        or re.search(
            rf"{re.escape(term)}[^，。；]{{0,12}}(?:各)?(?:有)?(?:多少|几个)",
            query_text,
        )
        or re.search(
            rf"{re.escape(term)}[^，。；]{{0,16}}占[^，。；]{{0,16}}"
            r"(?:比例|比重|占比)",
            query_text,
        )
    }


def _group_columns(tree: exp.Expression) -> set[str]:
    select = _final_select(tree)
    group = select.args.get("group") if select is not None else None
    if group is None:
        return set()
    return {column.name.lower() for column in group.find_all(exp.Column)}


def _group_columns_across_scopes(tree: exp.Expression) -> set[str]:
    """Return grouping dimensions proved by any upstream SELECT scope.

    A final SELECT over a CTE commonly projects dimensions that were grouped in
    the CTE and then ranked/windowed outside it. Requiring the outer SELECT to
    repeat ``GROUP BY`` rejects valid window and top-per-group plans. The final
    projection check remains separate, so a dimension grouped upstream but
    dropped from the user-facing result still fails the contract.
    """
    grouped: set[str] = set()
    for select in tree.find_all(exp.Select):
        group = select.args.get("group")
        if group is not None:
            grouped.update(column.name.lower() for column in group.find_all(exp.Column))
    return grouped


def _column_text(column: Mapping) -> str:
    return " ".join(str(value) for value in (
        column.get("name", ""), column.get("description", ""), *(column.get("alias") or [])
    ) if value).lower()


_ORDER_ONLY_SUFFIX_RE = re.compile(r"^(?:升序|降序|排序|顺序|排列|排名|排行)")
_RANK_CUE_RE = re.compile(
    r"(?:最高|最低|最多|最少|排名|排行|"
    r"(?:前|后)\s*(?:\d{1,4}|[一二两三四五六七八九十百]{1,4})|"
    r"(?:最高|最低|最多|最少)(?:的)?\s*"
    r"(?:\d{1,4}|[一二两三四五六七八九十百]{1,4})?(?:个|位|名|家|种)?)"
)
_AGGREGATE_MEASURE_CUE_RE = re.compile(
    r"(?:总数|数量|人数|笔数|件数|次数|金额|总额|销售额|成交额|销量|"
    r"均值|平均|合计|总计|占比|比例|比率|率)"
)
_ENTITY_LIST_BEFORE_RE = re.compile(
    r"(?:哪些|所有|全部|列出|展示|显示|返回|给出|"
    r"没有|无任何|从未|未曾|超过|高于|低于|大于|少于)"
    r"[^，。；]{0,24}$"
)
# Governed translations for the three stable physical DW entity stems.  They
# are used only when a caller supplies skeletal metadata without aliases; all
# unseen entities continue to be resolved from their own Meta vocabulary.
_PHYSICAL_ENTITY_FALLBACK_TERMS = {
    "product": ("商品", "产品"),
    "customer": ("客户", "用户"),
    "region": ("地区", "区域", "大区"),
}


def _metadata_terms(column: Mapping) -> list[str]:
    """Return stable longest-first terms for one metadata column."""
    values = {
        str(value).strip().lower()
        for value in (
            column.get("name", ""),
            column.get("description", ""),
            *(column.get("alias") or []),
        )
        if str(value or "").strip()
    }
    return sorted((value for value in values if len(value) >= 2), key=lambda value: (-len(value), value))


def _all_metadata_terms(table_infos: Sequence[Mapping] | None) -> tuple[str, ...]:
    """Return one longest-first vocabulary derived only from supplied metadata."""
    return tuple(sorted({
        term
        for table in table_infos or []
        for column in (table.get("columns") or [])
        for term in _metadata_terms(column)
    }, key=lambda value: (-len(value), value)))


def _table_entity_terms(table: Mapping) -> tuple[str, ...]:
    """Derive entity nouns from display metadata and dimension-table names."""
    display_terms = {
        term
        for column in (table.get("columns") or [])
        if column.get("is_display")
        or str(column.get("name") or "").lower().endswith("_name")
        for term in _metadata_terms(column)
    }
    terms = set(display_terms)
    for term in display_terms:
        stem = re.sub(r"(?:_?name|名称|名字|姓名|称谓)$", "", term).strip()
        if len(stem) >= 2:
            terms.add(stem)
    table_name = str(table.get("name") or "").strip().lower()
    if table_name.startswith("dim_") and len(table_name) > 4:
        physical_entity = table_name[4:]
        terms.add(physical_entity)
        terms.update(_PHYSICAL_ENTITY_FALLBACK_TERMS.get(physical_entity, ()))
    description = str(table.get("description") or "").strip().lower()
    entity_description = re.sub(
        r"(?:维度表?|信息表|主数据表?|目录表?)$", "", description,
    ).strip()
    if len(entity_description) >= 2:
        terms.add(entity_description)
    return tuple(sorted(terms, key=lambda value: (-len(value), value)))


def _metadata_term_is_shadowed(
    query: str,
    start: int,
    end: int,
    term: str,
    known_terms: Sequence[str] = (),
) -> bool:
    """Ignore a short alias when a longer metadata term owns the occurrence.

    The rule is deliberately schema-agnostic.  For example, if metadata contains
    both ``员工`` and ``员工名称``/``员工ID``, the occurrence of ``员工`` inside
    either longer term is not independent evidence for an employee-name axis.
    The same longest-match contract applies to unseen entities without growing
    a hard-coded product/customer vocabulary.
    """
    needle = str(term or "").strip().lower()
    if not needle:
        return False
    for candidate in known_terms:
        longer = str(candidate or "").strip().lower()
        if len(longer) <= len(needle) or needle not in longer:
            continue
        first_start = max(0, end - len(longer))
        last_start = min(start, len(query) - len(longer))
        for candidate_start in range(first_start, last_start + 1):
            candidate_end = candidate_start + len(longer)
            if (
                candidate_start <= start
                and candidate_end >= end
                and query.startswith(longer, candidate_start)
            ):
                return True
    return False


def _metadata_term_occurs(
    query: str, term: str, known_terms: Sequence[str] = (),
) -> bool:
    """Match one metadata term without letting a broad entity alias overreach."""
    needle = str(term or "").strip().lower()
    if not needle:
        return False
    escaped = re.escape(needle)
    if re.fullmatch(r"[0-9a-z_ ]+", needle):
        escaped = escaped.replace(r"\ ", r"\s+")
        pattern = re.compile(rf"(?<![0-9a-z_]){escaped}(?![0-9a-z_])")
    else:
        pattern = re.compile(escaped)
    return any(
        not _metadata_term_is_shadowed(
            query, match.start(), match.end(), needle, known_terms,
        )
        for match in pattern.finditer(query)
    )


def _column_names_counted_entity(column: Mapping, counted: set[str]) -> bool:
    """Whether the column itself, rather than its description, is being counted."""
    return bool(counted & set(_metadata_terms(column)))


def _fact_fk_satisfies_explicit_dimension_key(
    query: str,
    owner_table: str,
    column_name: str,
    used_tables: set[str],
    table_infos: Sequence[Mapping],
) -> bool:
    """Allow a fact FK when the user explicitly requests the equivalent dim key."""
    tables_by_name = {
        str(table.get("name") or "").lower(): table
        for table in table_infos or []
    }
    owner = tables_by_name.get(str(owner_table or "").lower())
    if not owner or str(owner.get("role") or "").lower() not in {
        "dim", "dimension",
    }:
        return False
    owner_column = next((
        column for column in (owner.get("columns") or [])
        if str(column.get("name") or "").lower() == column_name
    ), None)
    if not owner_column or str(owner_column.get("role") or "").lower() not in {
        "primary_key", "unique_key",
    }:
        return False
    key_wording = re.search(
        r"(?:(?<![0-9a-z_])id(?![0-9a-z_])|编号|标识|编码|"
        r"主键|外键|键值|键(?:分组|统计|汇总|维度|字段|$))",
        query,
        re.IGNORECASE,
    )
    known_terms = _all_metadata_terms(table_infos)
    if not key_wording or not any(
        _metadata_term_occurs(query, term, known_terms)
        for term in _metadata_terms(owner_column)
    ):
        return False
    return any(
        table_name in used_tables
        and str(table.get("role") or "").lower() in {"fact", "fact_table"}
        and any(
            str(column.get("name") or "").lower() == column_name
            and str(column.get("role") or "").lower() == "foreign_key"
            for column in (table.get("columns") or [])
        )
        for table_name, table in tables_by_name.items()
    )


def _ordered_listing_term(
    query: str, term: str, known_terms: Sequence[str] = (),
) -> bool:
    """Recognise ``按 <field> 升/降序列出`` as presentation ordering."""
    needle = str(term or "").strip().lower()
    if not needle:
        return False
    for match in re.finditer(re.escape(needle), query):
        if _metadata_term_is_shadowed(
            query, match.start(), match.end(), needle, known_terms,
        ):
            continue
        before = query[max(0, match.start() - 4):match.start()]
        after = query[match.end():match.end() + 32]
        if (
            re.search(r"按\s*$", before)
            and re.match(r"\s*(?:升序|降序|排序|顺序|排列)", after)
            and re.search(r"(?:列出|展示|显示|返回|给出)", query)
        ):
            return True
    return False


def _measure_controls_ranking(
    query: str, column: Mapping, known_terms: Sequence[str] = (),
) -> bool:
    """Return whether a measure term, rather than a catalogue field, sets rank."""
    for term in _metadata_terms(column):
        needle = str(term or "").strip().lower()
        for match in re.finditer(re.escape(needle), query):
            if _metadata_term_is_shadowed(
                query, match.start(), match.end(), needle, known_terms,
            ):
                continue
            before = query[max(0, match.start() - 20):match.start()]
            after = query[match.end():match.end() + 20]
            if (
                _RANK_CUE_RE.search(before)
                or _RANK_CUE_RE.search(after)
                or re.search(r"按\s*$", before)
                and _ORDER_ONLY_SUFFIX_RE.match(after.lstrip())
            ):
                return True
    return False


def _aggregate_ranking_requested(
    query: str,
    table_infos: Sequence[Mapping] | None,
    known_terms: Sequence[str],
) -> bool:
    """Whether a ranked result is controlled by an aggregate measure.

    A bare ``前五个实体`` is a bounded directory listing and needs no
    ``GROUP BY``.  A rank controlled by a metadata measure or by a generic
    aggregate noun such as count/amount/average does require an entity grain.
    """
    if not _RANK_CUE_RE.search(query):
        return False
    if any(
        str(column.get("role") or "").lower() == "measure"
        and _measure_controls_ranking(query, column, known_terms)
        for table in table_infos or []
        for column in (table.get("columns") or [])
    ):
        return True
    return bool(_AGGREGATE_MEASURE_CUE_RE.search(query))


def _entity_term_is_visible(
    query: str,
    term: str,
    known_terms: Sequence[str] = (),
) -> bool:
    """Prove that a metadata entity term belongs in the visible result."""
    needle = str(term or "").strip().lower()
    if not needle:
        return False
    for match in re.finditer(re.escape(needle), query):
        if _metadata_term_is_shadowed(
            query, match.start(), match.end(), needle, known_terms,
        ):
            continue
        before = query[max(0, match.start() - 36):match.start()]
        after = query[match.end():match.end() + 16]
        if (
            _ENTITY_LIST_BEFORE_RE.search(before)
            or _RANK_CUE_RE.search(before)
            or re.search(r"^(?:名单|列表|明细|排行|排名|及其|各自)", after)
        ):
            return True
    return (
        _ordered_listing_term(query, needle, known_terms)
        or _direct_grouping_term(query, needle, known_terms)
    )


def _direct_grouping_term(
    query: str, term: str, known_terms: Sequence[str] = (),
) -> bool:
    pattern = re.compile(
        rf"(?P<prefix>按|各|每个|每种|每位|每名|逐)\s*(?P<term>{re.escape(term)})"
    )
    for match in pattern.finditer(query):
        if _metadata_term_is_shadowed(
            query, match.start("term"), match.end("term"), term, known_terms,
        ):
            continue
        # “按商品名称升序列出”中的“按”声明展示顺序，不是 GROUP BY。
        # 若用户同时需要聚合，会通过另一个明确分组短语或排名目标被识别。
        if match.group("prefix") == "按" and _ORDER_ONLY_SUFFIX_RE.match(
            query[match.end("term"):].lstrip()
        ):
            continue
        return True
    return False


def _ranked_or_listed_term(
    query: str, term: str, known_terms: Sequence[str] = (),
) -> bool:
    """Whether a dimension/key term is the visible entity of a ranked/list query."""
    for match in re.finditer(re.escape(term), query):
        if _metadata_term_is_shadowed(
            query, match.start(), match.end(), term, known_terms,
        ):
            continue
        before = query[max(0, match.start() - 28):match.start()]
        after = query[match.end():match.end() + 12]
        if _RANK_CUE_RE.search(before) or re.search(
            r"(?:哪些|没有|无任何|从未|未曾)[^，。；]{0,20}$", before,
        ):
            return True
        if re.search(r"^(?:名单|列表|明细|排行|排名|及其|各自)", after):
            return True
    return False


def explicit_grouping_entries(
    query: str, table_infos: Sequence[Mapping] | None,
) -> set[tuple[str, str]]:
    """Infer only independently grounded grouping axes from question + metadata.

    Unlike the old global marker check, a cue must be grammatically adjacent to
    the exact metadata term.  Physical measures may therefore be explicit bins
    (``按购买件数``), while ranked/list targets remain restricted to dimensions
    and keys.  This independent scanner is intentionally not derived from IR.
    """
    query_text = str(query or "").strip().lower()
    known_terms = _all_metadata_terms(table_infos)
    counted = _counted_entity_terms(query_text)
    catalog_order_listing = any(
        str(column.get("role") or "").lower() != "measure"
        and any(
            _ordered_listing_term(query_text, term, known_terms)
            for term in _metadata_terms(column)
        )
        for table in table_infos or []
        for column in (table.get("columns") or [])
    )
    measure_controls_ranking = any(
        str(column.get("role") or "").lower() == "measure"
        and _measure_controls_ranking(query_text, column, known_terms)
        for table in table_infos or []
        for column in (table.get("columns") or [])
    )
    aggregate_controls_ranking = (
        measure_controls_ranking
        or _aggregate_ranking_requested(query_text, table_infos, known_terms)
    )
    pure_catalog_order_listing = (
        catalog_order_listing and not aggregate_controls_ranking
    )
    entries: set[tuple[str, str]] = set()
    for table in table_infos or []:
        table_name = str(table.get("name") or "").lower()
        for column in table.get("columns") or []:
            column_name = str(column.get("name") or "").lower()
            role = str(column.get("role") or "").lower()
            if not column_name:
                continue
            terms = _metadata_terms(column)
            direct = any(
                _direct_grouping_term(query_text, term, known_terms)
                for term in terms
            )
            ranked = role in {
                "dimension", "primary_key", "foreign_key", "unique_key",
            } and any(
                _ranked_or_listed_term(query_text, term, known_terms)
                for term in terms
            )
            if (
                ranked
                and not direct
                and (
                    pure_catalog_order_listing
                    or _column_names_counted_entity(column, counted)
                    or not aggregate_controls_ranking
                )
            ):
                ranked = False
            if direct or ranked:
                entries.add((table_name, column_name))
    return entries


def _cte_aliases(tree: exp.Expression) -> set[str]:
    return {
        str(cte.alias_or_name or "").lower()
        for cte in tree.find_all(exp.CTE)
        if cte.alias_or_name
    }


def _physical_aliases(tree: exp.Expression) -> tuple[dict[str, str], set[str]]:
    """Return SQL alias -> physical table and CTE names separately.

    SQLGlot exposes a CTE reference as ``exp.Table`` too.  Treating it as a
    second physical table made single-source formulas such as repeat rate look
    disconnected merely because the final SELECT reads the CTE.
    """
    ctes = _cte_aliases(tree)
    aliases: dict[str, str] = {}
    for table in tree.find_all(exp.Table):
        name = str(table.name or "").lower()
        if not name or name in ctes:
            continue
        aliases[str(table.alias_or_name or name).lower()] = name
        aliases[name] = name
    return aliases, ctes


def _direct_table_aliases(select: exp.Select, ctes: set[str]) -> set[str]:
    aliases: set[str] = set()
    from_clause = select.args.get("from_")
    source = from_clause.this if from_clause is not None else None
    if isinstance(source, exp.Table) and str(source.name or "").lower() not in ctes:
        aliases.add(str(source.alias_or_name or source.name).lower())
    for join in select.args.get("joins") or []:
        table = join.this
        if isinstance(table, exp.Table) and str(table.name or "").lower() not in ctes:
            aliases.add(str(table.alias_or_name or table.name).lower())
    return aliases


def _metadata_schemas(table_infos: Sequence[Mapping] | None) -> dict[str, dict[str, str]]:
    return {
        str(table.get("name") or "").lower(): {
            str(column.get("name") or "").lower(): str(column.get("role") or "").lower()
            for column in (table.get("columns") or [])
            if column.get("name")
        }
        for table in (table_infos or [])
        if table.get("name")
    }


def _metadata_relation(
    left: exp.Column,
    right: exp.Column,
    aliases: Mapping[str, str],
    schemas: Mapping[str, Mapping[str, str]],
) -> bool:
    left_alias = str(left.table or "").lower()
    right_alias = str(right.table or "").lower()
    if not left_alias or not right_alias or left_alias == right_alias:
        return False
    left_table = aliases.get(left_alias)
    right_table = aliases.get(right_alias)
    left_name = str(left.name or "").lower()
    right_name = str(right.name or "").lower()
    if not left_table or not right_table or left_table == right_table or left_name != right_name:
        return False
    left_role = str((schemas.get(left_table) or {}).get(left_name) or "").lower()
    right_role = str((schemas.get(right_table) or {}).get(right_name) or "").lower()
    explicit = {left_role, right_role} == {"primary_key", "foreign_key"}
    inferred = (
        left_name.endswith("_id")
        and bool(left_role or right_role)
        and ((left_role == "primary_key") != (right_role == "primary_key"))
    )
    return explicit or inferred


def _projection_comes_from_preserved_side(select: exp.Select, preserved: set[str], rejected: set[str]) -> bool:
    projected = [
        column
        for projection in select.expressions
        for column in projection.find_all(exp.Column)
    ]
    if not projected:
        # ``COUNT(*)`` is the scalar form of an anti query (“无订单日期有多少
        # 天”). It counts preserved-side rows after the nullable side is proved
        # absent and therefore does not expose or depend on rejected-side data.
        return any(
            projection.find(exp.Count) is not None
            for projection in select.expressions
        )
    aliases = {str(column.table or "").lower() for column in projected if column.table}
    return not (aliases & rejected) and (not aliases or bool(aliases & preserved))


def _left_join_anti_contract(
    tree: exp.Expression,
    aliases: Mapping[str, str],
    ctes: set[str],
    schemas: Mapping[str, Mapping[str, str]],
) -> bool:
    for select in tree.find_all(exp.Select):
        from_clause = select.args.get("from_")
        source = from_clause.this if from_clause is not None else None
        if not isinstance(source, exp.Table):
            continue
        left_aliases = {str(source.alias_or_name or source.name).lower()}
        where = select.args.get("where")
        for join in select.args.get("joins") or []:
            right = join.this
            if not isinstance(right, exp.Table):
                continue
            right_alias = str(right.alias_or_name or right.name).lower()
            if str(join.args.get("side") or "").upper() != "LEFT":
                left_aliases.add(right_alias)
                continue
            right_join_columns: set[str] = set()
            on = join.args.get("on")
            for equality in on.find_all(exp.EQ) if on is not None else []:
                if not isinstance(equality.this, exp.Column) or not isinstance(equality.expression, exp.Column):
                    continue
                pair = (equality.this, equality.expression)
                if not _metadata_relation(*pair, aliases, schemas):
                    continue
                for column in pair:
                    if str(column.table or "").lower() == right_alias:
                        right_join_columns.add(str(column.name or "").lower())
            null_checked = False
            if right_join_columns and where is not None:
                right_table = aliases.get(right_alias, str(right.name or "").lower())
                primary_keys = {
                    name for name, role in (schemas.get(right_table) or {}).items()
                    if role == "primary_key"
                }
                for item in where.find_all(exp.Is):
                    column = item.this
                    if not isinstance(column, exp.Column) or not isinstance(item.expression, exp.Null):
                        continue
                    if str(column.table or "").lower() != right_alias:
                        continue
                    if str(column.name or "").lower() in (right_join_columns | primary_keys):
                        null_checked = True
                        break
            if (
                right_join_columns
                and null_checked
                and _projection_comes_from_preserved_side(select, left_aliases, {right_alias})
            ):
                return True
            left_aliases.add(right_alias)
    return False


def _derived_left_join_anti_contract(
    tree: exp.Expression,
    aliases: Mapping[str, str],
    schemas: Mapping[str, Mapping[str, str]],
) -> bool:
    """Prove anti joins whose nullable side is a filtered derived key set.

    A time-scoped absence query commonly joins a dimension to
    ``SELECT DISTINCT fact.fk ... WHERE year = ?``.  The existing proof only
    accepted physical tables on the right side and therefore rejected this
    metadata-safe form even though the projected key lineage is explicit.
    """
    for select in tree.find_all(exp.Select):
        from_clause = select.args.get("from_")
        source = from_clause.this if from_clause is not None else None
        if not isinstance(source, exp.Table):
            continue
        preserved_aliases = {
            str(source.alias_or_name or source.name).lower()
        }
        where = select.args.get("where")
        if where is None:
            continue
        for join in select.args.get("joins") or []:
            right = join.this
            if (
                str(join.args.get("side") or "").upper() != "LEFT"
                or not isinstance(right, exp.Subquery)
                or not isinstance(right.this, exp.Select)
            ):
                continue
            derived_alias = str(right.alias_or_name or "").lower()
            if not derived_alias:
                continue
            inner = right.this
            projected_sources: dict[str, exp.Column] = {}
            for projection in inner.expressions:
                columns = list(projection.find_all(exp.Column))
                if len(columns) != 1:
                    continue
                output_name = str(projection.alias_or_name or columns[0].name).lower()
                projected_sources[output_name] = columns[0]

            proven_keys: set[str] = set()
            on = join.args.get("on")
            for equality in on.find_all(exp.EQ) if on is not None else []:
                if not (
                    isinstance(equality.this, exp.Column)
                    and isinstance(equality.expression, exp.Column)
                ):
                    continue
                pair = (equality.this, equality.expression)
                derived = next(
                    (
                        column for column in pair
                        if str(column.table or "").lower() == derived_alias
                    ),
                    None,
                )
                outer = next(
                    (
                        column for column in pair
                        if str(column.table or "").lower() in preserved_aliases
                    ),
                    None,
                )
                if derived is None or outer is None:
                    continue
                inner_source = projected_sources.get(
                    str(derived.name or "").lower()
                )
                if (
                    inner_source is not None
                    and _metadata_relation(outer, inner_source, aliases, schemas)
                ):
                    proven_keys.add(str(derived.name or "").lower())

            null_checked = any(
                isinstance(item.this, exp.Column)
                and isinstance(item.expression, exp.Null)
                and str(item.this.table or "").lower() == derived_alias
                and str(item.this.name or "").lower() in proven_keys
                for item in where.find_all(exp.Is)
            )
            if (
                proven_keys
                and null_checked
                and _projection_comes_from_preserved_side(
                    select,
                    preserved_aliases,
                    {derived_alias},
                )
            ):
                return True
    return False


def _conditional_aggregation_anti_contract(
    tree: exp.Expression,
    query_text: str,
) -> str | None:
    """Recognize bounded, explicit period-absence aggregation contracts."""
    years = list(dict.fromkeys(
        int(value)
        for value in re.findall(r"(?:19|20)\d{2}", query_text)
    ))
    present: list[int] = []
    absent: list[int] = []
    for year in years:
        marker = str(year)
        index = query_text.find(marker)
        left = query_text[max(0, index - 10):index] if index >= 0 else ""
        right = query_text[index + 4:index + 18] if index >= 0 else ""
        negative = bool(
            re.search(r"(?:未|没|没有|不曾|从未)", left[-6:])
            or re.search(
                r"^(?:年)?(?:均)?(?:未|没|没有|不曾|从未|无)",
                right,
            )
        )
        (absent if negative else present).append(year)

    # The HAVING may live inside a derived table whose outer SELECT only
    # counts qualified entities.  Inspect each grouping scope rather than only
    # the user-facing SELECT.
    for select in tree.find_all(exp.Select):
        if (
            select.args.get("group") is None
            or select.args.get("having") is None
        ):
            continue
        having_sql = (
            select.args["having"].sql(dialect="mysql").lower().replace("`", "")
        )
        if present and absent:
            patterns = [
                *[
                    rf"sum\s*\(\s*case\s+when\s+[\w.]*year\s*=\s*{year}"
                    rf"\s+then\s+1\s+else\s+0\s+end\s*\)\s*>\s*0"
                    for year in present
                ],
                *[
                    rf"sum\s*\(\s*case\s+when\s+[\w.]*year\s*=\s*{year}"
                    rf"\s+then\s+1\s+else\s+0\s+end\s*\)\s*=\s*0"
                    for year in absent
                ],
            ]
            if all(re.search(pattern, having_sql) for pattern in patterns):
                return "conditional_period_absence"
        if (
            years
            and "以前" in query_text
            and re.search(r"(?:没有|未)", query_text)
            and re.search(
                rf"max\s*\(\s*[\w.]*year\s*\)\s*<\s*{years[-1]}",
                having_sql,
            )
        ):
            return "last_event_before_period"
    return None


def _resolve_scope_column_domain(
    column: exp.Column,
    scope: Scope,
    schemas: Mapping[str, Mapping[str, str]],
    seen: set[tuple[int, str, str]] | None = None,
) -> tuple[str, str, str] | None:
    """Resolve one column through direct derived/CTE passthrough projections.

    This deliberately refuses computed expressions, duplicate output names and
    unqualified columns in multi-source scopes.  A returned tuple is therefore
    a unique physical ``(table, column, metadata_role)`` domain, not a guessed
    name match.
    """
    column_name = str(column.name or "").lower()
    alias = str(column.table or "").lower()
    if not column_name:
        return None
    try:
        selected_sources = scope.selected_sources
    except Exception:
        return None
    if not alias:
        if len(selected_sources) != 1:
            return None
        alias = str(next(iter(selected_sources))).lower()
    marker = (id(scope), alias, column_name)
    visited = seen if seen is not None else set()
    if marker in visited:
        return None
    visited.add(marker)
    selected = selected_sources.get(alias)
    if selected is None:
        return None
    source = selected[1]
    if isinstance(source, exp.Table):
        table_name = str(source.name or "").lower()
        role = str((schemas.get(table_name) or {}).get(column_name) or "").lower()
        return (table_name, column_name, role) if role else None
    if not isinstance(source, Scope) or not isinstance(source.expression, exp.Select):
        return None
    matches = [
        projection
        for projection in source.expression.expressions
        if str(projection.alias_or_name or "").lower() == column_name
    ]
    if len(matches) != 1:
        return None
    projection = matches[0]
    value = projection.this if isinstance(projection, exp.Alias) else projection
    if not isinstance(value, exp.Column):
        return None
    return _resolve_scope_column_domain(value, source, schemas, visited)


def _same_metadata_key_domain(
    left: tuple[str, str, str] | None,
    right: tuple[str, str, str] | None,
) -> bool:
    """Prove identical physical keys or an explicit same-name PK/FK edge."""
    if left is None or right is None:
        return False
    left_table, left_column, left_role = left
    right_table, right_column, right_role = right
    key_roles = {"primary_key", "foreign_key"}
    if left_column != right_column or not {left_role, right_role} <= key_roles:
        return False
    if left_table == right_table:
        return True
    return {left_role, right_role} == key_roles


def _not_exists_anti_contract(
    tree: exp.Expression,
    schemas: Mapping[str, Mapping[str, str]],
) -> bool:
    outer = _final_select(tree)
    if outer is None:
        return False
    try:
        scopes = {id(scope.expression): scope for scope in traverse_scope(tree)}
    except Exception:
        return False
    outer_scope = scopes.get(id(outer))
    if outer_scope is None:
        return False
    outer_aliases = {
        str(alias).lower() for alias in outer_scope.selected_sources
    }
    for item in tree.find_all(exp.Not):
        exists = item.this
        inner = exists.this if isinstance(exists, exp.Exists) else None
        if not isinstance(inner, exp.Select):
            continue
        inner_scope = scopes.get(id(inner))
        if inner_scope is None:
            continue
        inner_aliases = {
            str(alias).lower() for alias in inner_scope.selected_sources
        }
        correlated = False
        for equality in inner.find_all(exp.EQ):
            # Do not borrow an equality from a deeper nested SELECT.
            owner = equality.find_ancestor(exp.Select)
            if owner is not inner:
                continue
            if not (
                isinstance(equality.this, exp.Column)
                and isinstance(equality.expression, exp.Column)
            ):
                continue
            pair = (equality.this, equality.expression)
            inner_column = next((
                column for column in pair
                if str(column.table or "").lower() in inner_aliases
                and str(column.table or "").lower() not in outer_aliases
            ), None)
            outer_column = next((
                column for column in pair
                if str(column.table or "").lower() in outer_aliases
                and str(column.table or "").lower() not in inner_aliases
            ), None)
            if inner_column is None or outer_column is None:
                continue
            if _same_metadata_key_domain(
                _resolve_scope_column_domain(inner_column, inner_scope, schemas),
                _resolve_scope_column_domain(outer_column, outer_scope, schemas),
            ):
                correlated = True
                break
        if correlated and _projection_comes_from_preserved_side(outer, outer_aliases, inner_aliases):
            return True
    return False


def _not_in_anti_contract(
    tree: exp.Expression,
    aliases: Mapping[str, str],
    ctes: set[str],
    schemas: Mapping[str, Mapping[str, str]],
) -> bool:
    outer = _final_select(tree)
    if outer is None:
        return False
    outer_aliases = _direct_table_aliases(outer, ctes)
    for item in tree.find_all(exp.Not):
        inclusion = item.this
        query = inclusion.args.get("query") if isinstance(inclusion, exp.In) else None
        inner = query.this if isinstance(query, exp.Subquery) else None
        outer_column = inclusion.this if isinstance(inclusion, exp.In) else None
        if not isinstance(inner, exp.Select) or not isinstance(outer_column, exp.Column):
            continue
        inner_columns = [
            column for projection in inner.expressions for column in projection.find_all(exp.Column)
        ]
        if len(inner_columns) != 1:
            continue
        inner_column = inner_columns[0]
        inner_aliases = _direct_table_aliases(inner, ctes)
        outer_table = aliases.get(str(outer_column.table or "").lower(), "")
        inner_table = aliases.get(str(inner_column.table or "").lower(), "")
        same_scalar_domain = bool(
            outer_table
            and outer_table == inner_table
            and str(outer_column.name or "").lower()
            == str(inner_column.name or "").lower()
        )
        if not (
            _metadata_relation(outer_column, inner_column, aliases, schemas)
            or same_scalar_domain
        ):
            continue
        inner_role = str((schemas.get(inner_table) or {}).get(str(inner_column.name or "").lower()) or "")
        where = inner.args.get("where")
        explicit_not_null = any(
            isinstance(not_node.this, exp.Is)
            and isinstance(not_node.this.expression, exp.Null)
            and isinstance(not_node.this.this, exp.Column)
            and str(not_node.this.this.table or "").lower() == str(inner_column.table or "").lower()
            and str(not_node.this.this.name or "").lower() == str(inner_column.name or "").lower()
            for not_node in (where.find_all(exp.Not) if where is not None else [])
        )
        null_safe = inner_role == "primary_key" or explicit_not_null
        if (
            null_safe
            and _projection_comes_from_preserved_side(outer, outer_aliases, inner_aliases)
        ):
            return True
    return False


def analyze_semantic_shape(
    sql: str,
    query: str,
    table_infos: Sequence[Mapping] | None,
    semantic_ir: Mapping | None = None,
) -> dict:
    """Return privacy-safe relationship evidence for validation and confidence.

    The function never guesses through missing metadata.  An anti-join is marked
    complete only when its correlation uses a metadata-backed PK/FK edge.
    """
    anti_requested = bool(_ANTI_JOIN_QUERY_RE.search(str(query or "").lower()))
    try:
        tree = sqlglot.parse_one(str(sql or ""), read="mysql")
    except Exception:
        return {
            "parsed": False,
            "physical_table_count": 0,
            "cte_count": 0,
            "required_table_count": 0,
            "join_required": False,
            "join_contract_satisfied": False,
            "anti_join_requested": anti_requested,
            "anti_join_satisfied": False,
            "anti_join_strategy": None,
        }
    aliases, ctes = _physical_aliases(tree)
    physical_tables = set(aliases.values())
    ir = dict(semantic_ir or {})
    routing = dict(ir.get("model_sql_policy") or {})
    contract = dict(ir.get("query_contract") or {})
    # Model-routed guided IR is non-authoritative. Reusing its table slots here
    # would let the same parser both create and "prove" a join obligation and
    # bias the independent candidate reviewer toward the old parse.
    guided_model_route = bool(
        routing.get("use_model") and contract.get("authority") != "closed"
    )
    required_tables = {
        str(item.get("table") or "").lower()
        for item in [*(ir.get("dimensions") or []), *(ir.get("measures") or [])]
        if isinstance(item, Mapping) and item.get("table")
    } if not guided_model_route else set()
    semantic_source_known = bool(ir.get("status") == "complete" and required_tables)
    relation_table_count = len(required_tables) if semantic_source_known else len(physical_tables)
    join_required = relation_table_count > 1
    join_count = sum(1 for _ in tree.find_all(exp.Join))
    join_satisfied = not join_required or join_count >= relation_table_count - 1
    schemas = _metadata_schemas(table_infos)
    strategy = None
    if anti_requested:
        if _left_join_anti_contract(tree, aliases, ctes, schemas):
            strategy = "left_join_is_null"
        elif _derived_left_join_anti_contract(tree, aliases, schemas):
            strategy = "derived_left_join_is_null"
        elif _not_exists_anti_contract(tree, schemas):
            strategy = "not_exists"
        elif _not_in_anti_contract(tree, aliases, ctes, schemas):
            strategy = "null_safe_not_in"
        else:
            strategy = _conditional_aggregation_anti_contract(
                tree,
                str(query or "").lower(),
            )
    return {
        "parsed": True,
        "physical_table_count": len(physical_tables),
        "cte_count": len(ctes),
        "required_table_count": len(required_tables),
        "join_required": join_required,
        "join_contract_satisfied": join_satisfied,
        "anti_join_requested": anti_requested,
        "anti_join_satisfied": bool(strategy),
        "anti_join_strategy": strategy,
    }


def repair_invalid_metadata_joins(sql: str, table_infos: Sequence[Mapping] | None) -> str:
    """Repair only invalid/tautological joins when metadata yields one FK/PK edge.

    Valid joins are never rewritten. This is intentionally narrow: the current ON
    clause must reference a column absent from its table metadata or be ``1 = 1``,
    and exactly one matching foreign-key/primary-key column must exist.
    """
    tree = _parse(sql)
    schemas: dict[str, dict[str, str]] = {
        str(table.get("name") or "").lower(): {
            str(column.get("name") or "").lower(): str(column.get("role") or "").lower()
            for column in (table.get("columns") or [])
        }
        for table in (table_infos or [])
    }
    alias_to_table = {
        table.alias_or_name.lower(): table.name.lower()
        for table in tree.find_all(exp.Table)
    }
    changed = False
    selects = list(tree.find_all(exp.Select))
    if isinstance(tree, exp.Select) and tree not in selects:
        selects.insert(0, tree)
    for select in selects:
        from_clause = select.args.get("from_")
        source = from_clause.this if from_clause is not None else None
        if not isinstance(source, exp.Table):
            continue
        left_aliases = [source.alias_or_name.lower()]
        for join in select.args.get("joins") or []:
            right = join.this
            if not isinstance(right, exp.Table):
                continue
            right_alias = right.alias_or_name.lower()
            right_table = alias_to_table.get(right_alias, right.name.lower())
            candidates: list[tuple[str, str, str, str]] = []
            for left_alias in left_aliases:
                left_table = alias_to_table.get(left_alias, left_alias)
                for column_name, left_role in schemas.get(left_table, {}).items():
                    right_role = schemas.get(right_table, {}).get(column_name)
                    explicit_edge = {left_role, right_role} == {"primary_key", "foreign_key"}
                    inferred_edge = (
                        right_role is not None
                        and
                        column_name.endswith("_id")
                        and ((left_role == "primary_key") != (right_role == "primary_key"))
                    )
                    if explicit_edge or inferred_edge:
                        candidates.append((left_alias, column_name, right_alias, column_name))

            on = join.args.get("on")
            has_expected = False
            for equality in on.find_all(exp.EQ) if on is not None else []:
                if not isinstance(equality.this, exp.Column) or not isinstance(equality.expression, exp.Column):
                    continue
                pair = {
                    ((equality.this.table or "").lower(), equality.this.name.lower()),
                    ((equality.expression.table or "").lower(), equality.expression.name.lower()),
                }
                if any(pair == {(la, lc), (ra, rc)} for la, lc, ra, rc in candidates):
                    has_expected = True
                    break
            invalid_column = any(
                column.table
                and column.name.lower() not in schemas.get(
                    alias_to_table.get(column.table.lower(), column.table.lower()), {}
                )
                for column in (on.find_all(exp.Column) if on is not None else [])
            )
            tautology = (
                isinstance(on, exp.EQ)
                and isinstance(on.this, exp.Literal)
                and isinstance(on.expression, exp.Literal)
                and on.this.this == on.expression.this
            )
            if not has_expected and len(candidates) == 1 and (invalid_column or tautology or on is None):
                left_alias, left_column, _, right_column = candidates[0]
                join.set(
                    "on",
                    exp.EQ(
                        this=exp.column(left_column, table=left_alias),
                        expression=exp.column(right_column, table=right_alias),
                    ),
                )
                changed = True
            left_aliases.append(right_alias)
    return tree.sql(dialect="mysql") if changed else sql


def stabilize_threshold_measure_projection(sql: str, query: str) -> str:
    """Project the aggregate used by an explicitly requested amount threshold.

    Models occasionally generated a valid HAVING clause but returned only the
    entity name.  That answers *which* customers while silently omitting the
    requested cumulative amount and breaks result-contract comparison.  Reuse the
    already generated HAVING aggregate; do not invent a formula or a filter.
    """
    query_text = (query or "").strip().lower()
    if not (
        any(token in query_text for token in ("累计消费", "消费金额", "累计金额"))
        and any(token in query_text for token in ("超过", "大于", "高于", "低于", "少于"))
    ):
        return sql
    tree = _parse(sql)
    select = _final_select(tree)
    having = select.args.get("having") if select is not None else None
    if select is None or having is None:
        return sql
    aggregate = next((
        item for item in having.find_all(exp.AggFunc)
        if any(column.name.lower() == "order_amount" for column in item.find_all(exp.Column))
    ), None)
    if aggregate is None:
        return sql
    already_projected = any(
        projection.find(exp.AggFunc) is not None
        and any(column.name.lower() == "order_amount" for column in projection.find_all(exp.Column))
        for projection in select.expressions
    )
    if already_projected:
        return sql
    select.append("expressions", aggregate.copy().as_("total_amount"))
    return tree.sql(dialect="mysql")


def stabilize_rank_limit_ties(sql: str, query: str) -> str:
    """Append deterministic visible-grain tie breakers to bounded rankings.

    SQL ``ORDER BY metric LIMIT N`` selects an arbitrary subset when several
    groups share the cutoff metric.  The primary ranking remains unchanged;
    visible GROUP BY columns are appended ascending only when the question is a
    ranked/top/bottom request and the SQL already contains both ORDER BY and
    LIMIT.  No table, filter, projection, or ranking direction is invented.
    """
    query_text = str(query or "").strip().lower()
    if not _RANK_CUE_RE.search(query_text):
        return sql
    try:
        tree = sqlglot.parse_one(str(sql or ""), read="mysql")
    except Exception:
        return sql
    select = _final_select(tree)
    if select is None or select.args.get("limit") is None:
        return sql
    order = select.args.get("order")
    group = select.args.get("group")
    if order is None or group is None:
        return sql

    visible = _plain_projection_columns(tree)
    ordered_columns = {
        column.name.lower()
        for expression in order.expressions
        for column in expression.find_all(exp.Column)
    }
    tie_breakers: list[exp.Column] = []
    for expression in group.expressions:
        if not isinstance(expression, exp.Column):
            continue
        name = str(expression.name or "").lower()
        if name and name in visible and name not in ordered_columns:
            tie_breakers.append(expression.copy())
            ordered_columns.add(name)
    if not tie_breakers:
        return sql
    for column in tie_breakers:
        order.append(
            "expressions",
            exp.Ordered(this=column, desc=False, nulls_first=True),
        )
    return tree.sql(dialect="mysql")


def _matching_display_columns(query: str, table_infos: Sequence[Mapping]) -> set[str]:
    """Resolve readable entity projections from metadata, not entity vocabularies."""
    display: set[str] = set()
    known_terms = _all_metadata_terms(table_infos)
    explicit_entries = explicit_grouping_entries(query, table_infos)
    for table in table_infos:
        table_name = str(table.get("name") or "").lower()
        columns = list(table.get("columns") or [])
        display_columns = {
            str(column.get("name") or "").lower()
            for column in columns
            if column.get("is_display")
            or str(column.get("name") or "").lower().endswith("_name")
        }
        if not display_columns:
            display_columns = {
                str(column.get("name") or "").lower()
                for column in columns
                if str(column.get("role") or "").lower() == "dimension"
                and (
                    "名称" in _column_text(column)
                    or str(column.get("name") or "").lower().endswith("name")
                )
            }
        if not display_columns:
            continue
        entity_terms = _table_entity_terms(table)
        direct_visible = any(
            _entity_term_is_visible(query, term, known_terms)
            for term in entity_terms
        )
        ranked_visible = any(
            _ranked_or_listed_term(query, term, known_terms)
            for term in entity_terms
        )
        other_visible_axes = {
            column_name
            for owner, column_name in explicit_entries
            if owner == table_name and column_name not in display_columns
        }
        # A table/entity word can merely identify the catalogue source while a
        # different attribute is the visible ranked target.  Prefer that
        # metadata-backed attribute unless the entity itself has direct list,
        # anti-join, ordering or pre-rank evidence.
        if direct_visible or (ranked_visible and not other_visible_axes):
            display.update(display_columns)
    return display


_ATTRIBUTE_FILTER_AFTER_RE = re.compile(
    r"^\s*(?:为|是|等于|不等于|属于|不属于|在|不在|包含|不含|"
    r"大于|小于|高于|低于|超过|少于|至少|至多|不低于|不高于|"
    r"=|!=|<>|>=|<=|>|<)"
)
_ATTRIBUTE_VALUE_FILTER_AFTER_RE = re.compile(
    r"^\s*(?:['\"]?[0-9a-z][0-9a-z_.-]*['\"]?)\s*"
    r"(?:的|对应|所属|相关|记录)",
    re.IGNORECASE,
)


def _scoped_attribute_terms(table: Mapping, column: Mapping) -> tuple[str, ...]:
    """从 Meta 别名派生实体作用域内的属性简称，不维护题目词表。"""

    terms = set(_metadata_terms(column))
    entity_terms = _table_entity_terms(table)
    for term in tuple(terms):
        for entity_term in entity_terms:
            if not term.startswith(entity_term):
                continue
            suffix = term[len(entity_term):].lstrip("_- ")
            if len(suffix) >= 2:
                terms.add(suffix)
    return tuple(sorted(terms, key=lambda value: (-len(value), value)))


def _attribute_term_has_output_role(
    query: str,
    term: str,
    known_terms: Sequence[str],
    entity_terms: Sequence[str] = (),
) -> bool:
    """确认属性词是列表输出，而不是过滤、排序或排名条件。"""

    needle = str(term or "").strip().lower()
    if not needle:
        return False
    for match in re.finditer(re.escape(needle), query):
        if _metadata_term_is_shadowed(
            query, match.start(), match.end(), needle, known_terms,
        ):
            continue
        before = query[max(0, match.start() - 40):match.start()]
        after = query[match.end():match.end() + 32]
        if (
            _ATTRIBUTE_FILTER_AFTER_RE.match(after)
            or _ATTRIBUTE_VALUE_FILTER_AFTER_RE.match(after)
        ):
            continue
        if (
            re.search(r"(?:按|依|以)\s*$", before)
            and _ORDER_ONLY_SUFFIX_RE.match(after.lstrip())
        ):
            continue
        if any(
            re.match(rf"^\s*{re.escape(entity_term)}(?:的|中|内)", after)
            for entity_term in entity_terms
        ):
            continue
        if _entity_term_is_visible(query, needle, known_terms):
            return True
    return False


def _requested_metadata_fields(
    query: str,
    table_infos: Sequence[Mapping],
) -> dict[str, set[str]]:
    """解析实体目录题中明确要求直接返回的 Meta 标识与属性列。"""

    query_text = str(query or "").strip().lower()
    if not _entity_listing_requested(query_text):
        return {}
    scoped_terms = {
        term
        for table in table_infos
        for column in (table.get("columns") or [])
        for term in _scoped_attribute_terms(table, column)
    }
    known_terms = tuple(sorted(
        {*_all_metadata_terms(table_infos), *scoped_terms},
        key=lambda value: (-len(value), value),
    ))
    requested: dict[str, set[str]] = {}
    for table in table_infos:
        table_name = str(table.get("name") or "").lower()
        columns = list(table.get("columns") or [])
        if not table_name or not columns:
            continue
        entity_terms = _table_entity_terms(table)
        entity_context = any(
            _metadata_term_occurs(query_text, term, known_terms)
            for term in entity_terms
        ) or any(
            _metadata_term_occurs(query_text, term, known_terms)
            for column in columns
            for term in _metadata_terms(column)
        )
        if not entity_context:
            continue
        for column in columns:
            role = str(column.get("role") or "").lower()
            column_name = str(column.get("name") or "").lower()
            if (
                not column_name
                or role == "measure"
            ):
                continue
            terms = set(_scoped_attribute_terms(table, column))
            terms -= set(entity_terms)
            if any(
                _attribute_term_has_output_role(
                    query_text,
                    term,
                    known_terms,
                    entity_terms,
                )
                for term in terms
            ):
                requested.setdefault(table_name, set()).add(column_name)
    return requested


def _visible_projection_expression(
    projection: exp.Expression,
) -> exp.Expression:
    return projection.this if isinstance(projection, exp.Alias) else projection


def _projection_scope(
    tree: exp.Expression,
) -> tuple[exp.Select | None, Scope | None]:
    """返回最终 SELECT 及其 SQLGlot scope；无法证明时关闭式返回空。"""

    selected = _final_select(tree)
    if selected is None:
        return None, None
    try:
        for scope in traverse_scope(tree):
            if scope.expression is selected:
                return selected, scope
    except Exception:
        return selected, None
    return selected, None


def _scope_sources(scope: Scope) -> dict[str, object]:
    sources: dict[str, object] = {}
    try:
        for alias, selected in scope.selected_sources.items():
            source = selected[1] if isinstance(selected, tuple) else selected
            sources[str(alias or "").lower()] = source
    except Exception:
        return {}
    return sources


def _scope_output_projection(
    scope: Scope,
    column_name: str,
) -> exp.Expression | None:
    selected = _final_select(scope.expression)
    if selected is None:
        return None
    name = str(column_name or "").lower()
    matches = [
        projection
        for projection in selected.expressions
        if str(projection.alias_or_name or "").lower() == name
    ]
    return matches[0] if len(matches) == 1 else None


def _source_exposes_column(
    source: object,
    column_name: str,
    schemas: Mapping[str, Mapping[str, str]],
) -> bool:
    name = str(column_name or "").lower()
    if isinstance(source, exp.Table):
        return name in schemas.get(str(source.name or "").lower(), {})
    if isinstance(source, Scope):
        return _scope_output_projection(source, name) is not None
    return False


def _local_projection_columns(
    expression: exp.Expression,
    selected: exp.Select,
) -> list[exp.Column]:
    if isinstance(expression, exp.Column):
        return [expression]
    return [
        column
        for column in expression.find_all(exp.Column)
        if column.find_ancestor(exp.Select) is selected
    ]


def _resolve_projection_column(
    scope: Scope,
    column: exp.Column,
    schemas: Mapping[str, Mapping[str, str]],
    *,
    direct_only: bool,
    seen: set[tuple[int, str, str, bool]] | None = None,
) -> set[tuple[str, str]]:
    """沿最终投影与 CTE 解析物理 owner；歧义或表达式折叠均不猜测。"""

    name = str(column.name or "").lower()
    qualifier = str(column.table or "").lower()
    if not name:
        return set()
    visited = seen if seen is not None else set()
    marker = (id(scope), qualifier, name, direct_only)
    if marker in visited:
        return set()
    visited.add(marker)

    sources = _scope_sources(scope)
    if qualifier:
        source = sources.get(qualifier)
        candidates = [source] if source is not None else []
    else:
        candidates = [
            source
            for source in sources.values()
            if _source_exposes_column(source, name, schemas)
        ]
        unique = {id(source): source for source in candidates}
        candidates = list(unique.values())
    if len(candidates) != 1:
        return set()
    source = candidates[0]
    if isinstance(source, exp.Table):
        table_name = str(source.name or "").lower()
        return (
            {(table_name, name)}
            if name in schemas.get(table_name, {})
            else set()
        )
    if not isinstance(source, Scope):
        return set()

    projection = _scope_output_projection(source, name)
    if projection is None:
        return set()
    expression = _visible_projection_expression(projection)
    if direct_only and not isinstance(expression, exp.Column):
        return set()
    selected = _final_select(source.expression)
    if selected is None:
        return set()
    resolved: set[tuple[str, str]] = set()
    for nested in _local_projection_columns(expression, selected):
        resolved.update(
            _resolve_projection_column(
                source,
                nested,
                schemas,
                direct_only=direct_only,
                seen=visited,
            )
        )
    return resolved


def _visible_projection_evidence(
    tree: exp.Expression,
    table_infos: Sequence[Mapping],
) -> dict[str, object]:
    """提取最终可见字段的直接 owner 与完整敏感 lineage 证据。"""

    selected, scope = _projection_scope(tree)
    evidence: dict[str, object] = {
        "direct_entries": set(),
        "direct_names": set(),
        "lineage_entries": set(),
        "has_star": False,
    }
    if selected is None or scope is None:
        return evidence
    schemas = _metadata_schemas(table_infos)
    direct_entries: set[tuple[str, str]] = set()
    direct_names: set[str] = set()
    lineage_entries: set[tuple[str, str]] = set()
    for projection in selected.expressions:
        expression = _visible_projection_expression(projection)
        if isinstance(expression, exp.Star) or (
            isinstance(expression, exp.Column)
            and isinstance(expression.this, exp.Star)
        ):
            evidence["has_star"] = True
        if isinstance(expression, exp.Column):
            direct_names.add(str(expression.name or "").lower())
            direct_entries.update(
                _resolve_projection_column(
                    scope,
                    expression,
                    schemas,
                    direct_only=True,
                )
            )
        for column in _local_projection_columns(expression, selected):
            lineage_entries.update(
                _resolve_projection_column(
                    scope,
                    column,
                    schemas,
                    direct_only=False,
                )
            )
    evidence.update({
        "direct_entries": direct_entries,
        "direct_names": direct_names,
        "lineage_entries": lineage_entries,
    })
    return evidence


def _requested_entry_is_visible(
    requested: tuple[str, str],
    direct_entries: set[tuple[str, str]],
    *,
    query: str,
    used_tables: set[str],
    table_infos: Sequence[Mapping],
) -> bool:
    if requested in direct_entries:
        return True
    owner, column = requested
    return bool(
        any(actual_column == column for _, actual_column in direct_entries)
        and _fact_fk_satisfies_explicit_dimension_key(
            query,
            owner,
            column,
            used_tables,
            table_infos,
        )
    )


_QUERY_NUMERIC_COMPARISON_RE = re.compile(
    r"[^，。；]{0,8}?(?P<operator>至少|不少于|不低于|大于等于|达到|"
    r"超过|大于|高于|多于|至多|不超过|不高于|小于等于|"
    r"少于|小于|低于|不足|等于)\s*"
    r"(?P<value>[+-]?\d[\d,]*(?:\.\d+)?)"
)
_QUERY_OPERATOR = {
    "至少": ">=", "不少于": ">=", "不低于": ">=", "大于等于": ">=", "达到": ">=",
    "超过": ">", "大于": ">", "高于": ">", "多于": ">",
    "至多": "<=", "不超过": "<=", "不高于": "<=", "小于等于": "<=",
    "少于": "<", "小于": "<", "低于": "<", "不足": "<", "等于": "=",
}
_AGGREGATE_PREFIX_RE = re.compile(
    r"(?:累计|累积|合计|总计|总共|汇总|平均|均值|月均|季均|年均|人均)\s*$"
)
_ROW_LEVEL_PREFIX_RE = re.compile(
    r"(?:单笔|每笔|每条|每行|每次|每单|每个订单|每条记录)[^，。；]{0,8}$"
)
_ROW_LEVEL_SUFFIX_RE = re.compile(
    r"^\s*(?:元|件|笔|次|条|个)?\s*"
    r"(?:时|的?(?:记录|订单|交易|明细|行)(?:中|内)?)"
)
_AGGREGATE_THRESHOLD_RE = re.compile(
    r"(?:累计|累积|合计|总计|总共|汇总|平均|均值|月均|季均|年均|人均|"
    r"总额|总量|订单数|订单数量|订单笔数|交易数|交易次数|事件数|记录数)"
    r"[^，。；]{0,24}"
    r"(?:至少|不少于|不低于|大于等于|达到|超过|大于|高于|多于|"
    r"至多|不超过|不高于|小于等于|少于|小于|低于|不足|等于)"
    r"\s*[+-]?\d[\d,]*(?:\.\d+)?"
    r"|(?:至少|不少于|不低于|超过|大于|高于|多于)\s*"
    r"[+-]?\d[\d,]*(?:\.\d+)?\s*(?:笔|次|条)\s*"
    r"(?:订单|交易|事件|记录)"
    r"|(?:不同|去重|购买|活跃)?(?:客户|用户)(?:数|数量)"
    r"[^，。；]{0,24}"
    r"(?:至少|不少于|不低于|大于等于|达到|超过|大于|高于|多于|"
    r"至多|不超过|不高于|小于等于|少于|小于|低于|不足|等于)"
    r"\s*[+-]?\d[\d,]*(?:\.\d+)?"
)
_NUMERIC_SQL_COMPARISONS = (exp.GT, exp.GTE, exp.LT, exp.LTE, exp.EQ)
_SQL_COMPARISON_OPERATOR = {
    exp.GT: ">", exp.GTE: ">=", exp.LT: "<", exp.LTE: "<=", exp.EQ: "=",
}
_REVERSED_SQL_OPERATOR = {">": "<", ">=": "<=", "<": ">", "<=": ">=", "=": "="}


def _normalized_number(value: str) -> str | None:
    """Return one stable decimal spelling for query and SQL thresholds."""
    try:
        number = Decimal(str(value or "").replace(",", ""))
    except (InvalidOperation, ValueError):
        return None
    normalized = format(number.normalize(), "f")
    return "0" if normalized in {"-0", "+0"} else normalized


def _metadata_term_spans(
    query: str, term: str, known_terms: Sequence[str],
):
    """Yield non-shadowed spans for one Meta term."""
    needle = str(term or "").strip().lower()
    if not needle:
        return
    escaped = re.escape(needle)
    if re.fullmatch(r"[0-9a-z_ ]+", needle):
        escaped = escaped.replace(r"\ ", r"\s+")
        pattern = re.compile(rf"(?<![0-9a-z_]){escaped}(?![0-9a-z_])")
    else:
        pattern = re.compile(escaped)
    for match in pattern.finditer(query):
        if not _metadata_term_is_shadowed(
            query, match.start(), match.end(), needle, known_terms,
        ):
            yield match.start(), match.end()


def _numeric_fact_measure_columns(
    table_infos: Sequence[Mapping],
) -> list[tuple[str, Mapping]]:
    """Select row-level numeric candidates from fact Meta only."""
    numeric_type = re.compile(
        r"(?:tinyint|smallint|mediumint|bigint|int|integer|decimal|numeric|"
        r"float|double|real|number)"
    )
    output: list[tuple[str, Mapping]] = []
    for table in table_infos:
        if str(table.get("role") or "").lower() != "fact":
            continue
        table_name = str(table.get("name") or "").lower()
        for column in table.get("columns") or []:
            role = str(column.get("role") or "").lower()
            data_type = str(
                column.get("type") or column.get("data_type") or ""
            ).lower()
            if role == "measure" or numeric_type.search(data_type):
                output.append((table_name, column))
    return output


def _explicit_row_numeric_thresholds(
    query: str, table_infos: Sequence[Mapping],
) -> set[tuple[str, str, str]]:
    """Resolve high-confidence row predicates as (column, operator, value).

    The row contract requires either an explicit per-row prefix (``单笔``) or
    a row-result suffix (``时``/``的记录``). Aggregate modifiers immediately
    before the Meta term keep cumulative/average questions on the HAVING path.
    """
    known_terms = _all_metadata_terms(table_infos)
    thresholds: set[tuple[str, str, str]] = set()
    for _, column in _numeric_fact_measure_columns(table_infos):
        column_name = str(column.get("name") or "").lower()
        if not column_name:
            continue
        for term in _metadata_terms(column):
            for start, end in _metadata_term_spans(query, term, known_terms):
                prefix = query[max(0, start - 16):start]
                after = query[end:end + 40]
                comparison = _QUERY_NUMERIC_COMPARISON_RE.match(after)
                if comparison is None:
                    continue
                between = after[:comparison.start("operator")]
                aggregate_prefix = bool(
                    _AGGREGATE_PREFIX_RE.search(prefix)
                    or re.search(
                        r"(?:累计|累积|合计|总计|总共|汇总|平均|均值|"
                        r"月均|季均|年均|人均)",
                        between,
                    )
                )
                tail = after[comparison.end():comparison.end() + 18]
                row_level = bool(
                    _ROW_LEVEL_PREFIX_RE.search(prefix)
                    or _ROW_LEVEL_SUFFIX_RE.match(tail)
                    or re.search(r"(?:单笔|每笔|每条|每行|每次|每单)", term)
                )
                value = _normalized_number(comparison.group("value"))
                if aggregate_prefix or not row_level or value is None:
                    continue
                thresholds.add((
                    column_name,
                    _QUERY_OPERATOR[comparison.group("operator")],
                    value,
                ))
    return thresholds


def _numeric_constant(expression: exp.Expression | None) -> str | None:
    """Read a literal numeric constant without treating arithmetic as one."""
    current = expression
    while isinstance(current, (exp.Paren, exp.Cast)):
        current = current.this
    sign = ""
    if isinstance(current, exp.Neg):
        sign = "-"
        current = current.this
    if not isinstance(current, exp.Literal) or current.is_string:
        return None
    return _normalized_number(sign + str(current.this))


def _comparison_sides(
    predicate: exp.Expression,
) -> tuple[exp.Expression, str, str] | None:
    """Return (value expression, normalized operator, numeric literal)."""
    operator = _SQL_COMPARISON_OPERATOR.get(type(predicate))
    if operator is None:
        return None
    right = _numeric_constant(predicate.expression)
    if right is not None:
        return predicate.this, operator, right
    left = _numeric_constant(predicate.this)
    if left is not None:
        return predicate.expression, _REVERSED_SQL_OPERATOR[operator], left
    return None


def _aggregate_alias_columns(select: exp.Select) -> dict[str, set[str]]:
    aliases: dict[str, set[str]] = {}
    for projection in select.expressions:
        if projection.find(exp.AggFunc) is None:
            continue
        alias = str(projection.alias_or_name or "").lower()
        if alias:
            aliases[alias] = {
                column.name.lower() for column in projection.find_all(exp.Column)
            }
    return aliases


def _sql_numeric_thresholds(
    tree: exp.Expression,
) -> tuple[set[tuple[str, str, str]], set[tuple[str, str, str]]]:
    """Return direct WHERE and aggregate HAVING threshold signatures."""
    where_thresholds: set[tuple[str, str, str]] = set()
    having_thresholds: set[tuple[str, str, str]] = set()
    for select in tree.find_all(exp.Select):
        where = select.args.get("where")
        if where is not None:
            for predicate in where.walk():
                if not isinstance(predicate, _NUMERIC_SQL_COMPARISONS):
                    continue
                sides = _comparison_sides(predicate)
                if sides is None:
                    continue
                expression, operator, value = sides
                if expression.find(exp.AggFunc) is not None:
                    continue
                where_thresholds.update(
                    (column.name.lower(), operator, value)
                    for column in expression.find_all(exp.Column)
                )
        having = select.args.get("having")
        if having is None:
            continue
        aggregate_aliases = _aggregate_alias_columns(select)
        for predicate in having.walk():
            if not isinstance(predicate, _NUMERIC_SQL_COMPARISONS):
                continue
            sides = _comparison_sides(predicate)
            if sides is None:
                continue
            expression, operator, value = sides
            aggregate_columns = {
                column.name.lower()
                for aggregate in expression.walk()
                if isinstance(aggregate, exp.AggFunc)
                for column in aggregate.find_all(exp.Column)
            }
            for column in expression.find_all(exp.Column):
                aggregate_columns.update(
                    aggregate_aliases.get(column.name.lower(), set())
                )
            having_thresholds.update(
                (column_name, operator, value)
                for column_name in aggregate_columns
            )
    return where_thresholds, having_thresholds


def _metadata_entity_count_requested(
    query: str, table_infos: Sequence[Mapping],
) -> set[str]:
    """Return Meta-owned dimension targets explicitly counted by the question.

    A counted business entity may be represented by the table display entity
    (store/employee) or by a dimension attribute on that table (province/tier).
    The suffix grammar is deliberately end-anchored: ``地区有多少订单`` counts
    orders *by* region, whereas ``地区有多少`` counts regions themselves.
    """
    known_terms = _all_metadata_terms(table_infos)
    counted: set[str] = set()
    quantifier = r"(?:个|家|名|位|种)?"
    for table in table_infos:
        if str(table.get("role") or "").lower() != "dim":
            continue
        table_name = str(table.get("name") or "").lower()
        terms = set(_table_entity_terms(table))
        terms.update(
            term
            for column in table.get("columns") or []
            if column.get("is_display")
            or str(column.get("role") or "").lower() == "dimension"
            for term in _metadata_terms(column)
        )
        for term in sorted(terms, key=lambda value: (-len(value), value)):
            for start, end in _metadata_term_spans(query, term, known_terms):
                prefix = query[max(0, start - 20):start]
                suffix = query[end:]
                escaped_term = re.escape(term)
                if (
                    re.search(rf"(?:有|共计|统计)?多少{quantifier}\s*$", prefix)
                    or re.fullmatch(
                        rf"(?:中|里|当中)?(?:有)?多少{quantifier}"
                        rf"(?:\s*(?:该|这些)?{escaped_term})?\s*[，。；？?]*",
                        suffix,
                    )
                    or re.fullmatch(
                        r"(?:总)?(?:数|数量)(?:是多少|为多少)?\s*[，。；？?]*",
                        suffix,
                    )
                ):
                    counted.add(table_name)
                    break
    return counted


def _final_is_scalar_count(tree: exp.Expression) -> bool:
    select = _final_select(tree)
    if (
        select is None
        or select.args.get("group") is not None
        or len(select.expressions) != 1
    ):
        return False
    projection = select.expressions[0]
    value = projection.this if isinstance(projection, exp.Alias) else projection
    return isinstance(value, exp.Count) and value.find(exp.Window) is None


def _check_threshold_output_contracts(
    tree: exp.Expression,
    query: str,
    table_infos: Sequence[Mapping],
) -> None:
    """Enforce row-filter placement and aggregate-entity scalar shape."""
    row_thresholds = _explicit_row_numeric_thresholds(query, table_infos)
    if row_thresholds:
        where_thresholds, having_thresholds = _sql_numeric_thresholds(tree)
        misplaced = {
            column
            for column, operator, value in row_thresholds
            if (column, operator, value) not in where_thresholds
            and any(
                actual_column == column
                for actual_column, _, _ in having_thresholds
            )
        }
        if misplaced:
            raise SemanticContractError(
                "问题明确要求对事实表单行数值做过滤；SQL 不得只在 HAVING 中比较该字段的聚合值，"
                f"应在聚合前的 WHERE 中保留对应比较 {sorted(misplaced)}",
                violation_code="semantic.filter.row_threshold_in_having",
                repairability="llm",
                repair_hint_code="move_row_threshold_to_where",
                evidence={
                    "expected_columns": sorted(misplaced),
                    "actual_columns": sorted({item[0] for item in having_thresholds}),
                    "sql_hash": _safe_sql_hash(tree.sql(dialect="mysql")),
                },
            )

    counted_entities = _metadata_entity_count_requested(query, table_infos)
    if (
        counted_entities
        and _AGGREGATE_THRESHOLD_RE.search(query)
    ):
        if _final_is_scalar_count(tree):
            final = _final_select(tree)
            qualified_scopes = [
                select for select in tree.find_all(exp.Select)
                if select is not final
                and select.args.get("group") is not None
                and select.args.get("having") is not None
            ]
            if qualified_scopes:
                return
            raise SemanticContractError(
                "满足聚合阈值的实体数量必须来自内层资格集合；"
                "需要内层 GROUP BY / HAVING，最终层才能 COUNT(*)",
                violation_code=(
                    "semantic.output.qualified_group_count_structure_required"
                ),
                repairability="llm",
                repair_hint_code="build_qualified_groups_then_count",
                evidence={
                    "expected_tables": sorted(counted_entities),
                    "expected_count": 1,
                    "actual_count": 0,
                },
            )
        final = _final_select(tree)
        raise SemanticContractError(
            "问题询问满足聚合阈值的维度实体数量；Final SELECT 必须返回单个 scalar COUNT，"
            "实体级 GROUP BY / HAVING 只能位于内层资格子查询",
            violation_code="semantic.output.scalar_count_required",
            repairability="llm",
            repair_hint_code="wrap_qualified_entities_with_scalar_count",
            evidence={
                "expected_columns": ["scalar_count"],
                "actual_columns": sorted(_plain_projection_columns(tree)),
                "expected_tables": sorted(counted_entities),
                "expected_count": 1,
                "actual_count": len(final.expressions) if final is not None else 0,
                "sql_hash": _safe_sql_hash(tree.sql(dialect="mysql")),
            },
        )


def _explicit_year_values(query: str) -> list[int]:
    values = [int(value) for value in _EXPLICIT_YEAR_RE.findall(query)]
    for group in re.findall(
        r"(?<!\d)((?:19|20)\d{2}"
        r"(?:(?:和|与|、|,|，)(?:19|20)\d{2})+)"
        r"(?:\s*年)?",
        query,
    ):
        values.extend(int(value) for value in re.findall(r"(?:19|20)\d{2}", group))
    return sorted(
        dict.fromkeys(values),
        key=lambda year: query.find(str(year)),
    )


def _year_dimension_owners(
    table_infos: Sequence[Mapping],
) -> set[str]:
    owners: set[str] = set()
    for table in table_infos:
        for column in table.get("columns") or []:
            name = str(column.get("name") or "").lower()
            text = _column_text(column)
            if name == "year" or "年份" in text:
                owners.add(str(table.get("name") or "").lower())
    return owners


def _sql_has_full_period_year_constraint(
    tree: exp.Expression,
    years: Sequence[int],
    table_infos: Sequence[Mapping],
) -> bool:
    """Prove that every requested year constrains a date-dimension year field."""

    if not years:
        return True
    aliases, ctes = _physical_aliases(tree)
    year_owners = _year_dimension_owners(table_infos)
    if not year_owners:
        return False
    seen: set[int] = set()
    for column in tree.find_all(exp.Column):
        if str(column.name or "").lower() != "year":
            continue
        alias = str(column.table or "").lower()
        if alias in ctes or aliases.get(alias) not in year_owners:
            continue
        parent = column.parent
        if isinstance(parent, exp.EQ):
            other = parent.expression if parent.this is column else parent.this
            if isinstance(other, exp.Literal) and not other.is_string:
                try:
                    seen.add(int(other.this))
                except (TypeError, ValueError):
                    pass
        elif isinstance(parent, exp.In) and parent.this is column:
            for literal in parent.expressions:
                if isinstance(literal, exp.Literal) and not literal.is_string:
                    try:
                        seen.add(int(literal.this))
                    except (TypeError, ValueError):
                        pass
        elif isinstance(parent, exp.Between) and parent.this is column:
            low = parent.args.get("low")
            high = parent.args.get("high")
            if (
                isinstance(low, exp.Literal)
                and isinstance(high, exp.Literal)
                and not low.is_string
                and not high.is_string
            ):
                try:
                    lower, upper = int(low.this), int(high.this)
                except (TypeError, ValueError):
                    continue
                requested = set(years)
                bounded = set(range(lower, upper + 1))
                # A closed range is equivalent only when it introduces no
                # unrequested periods.  This proves contiguous comparisons
                # while rejecting BETWEEN over sparse requested years.
                if requested == bounded:
                    seen.update(bounded)
    return set(years).issubset(seen)


def _check_explicit_year_contract(
    tree: exp.Expression,
    query: str,
    table_infos: Sequence[Mapping],
) -> None:
    years = _explicit_year_values(query)
    # Partial retrieval metadata cannot prove how a year maps physically.
    # Do not reject on vocabulary alone; the complete production Meta path
    # supplies a unique year owner and activates this contract.
    if not years or not _year_dimension_owners(table_infos):
        return
    # Set/absence questions have a different, independently checked period
    # contract (MAX(year)<Y, conditional presence/absence, etc.).
    if re.search(r"(?:以前|之前|没有|未|从未|都下过|同时)", query):
        return
    if _sql_has_full_period_year_constraint(tree, years, table_infos):
        # A scalar subquery that selects one date_id (typically LIMIT 1) still
        # narrows a year to one day even though its inner predicate says year.
        scalar_date_lookup = any(
            (
                isinstance(equality.expression, exp.Subquery)
                and isinstance(equality.this, exp.Column)
                and str(equality.this.name or "").lower() == "date_id"
            )
            or (
                isinstance(equality.this, exp.Subquery)
                and isinstance(equality.expression, exp.Column)
                and str(equality.expression.name or "").lower() == "date_id"
            )
            for equality in tree.find_all(exp.EQ)
        )
        if not scalar_date_lookup:
            return
    raise SemanticContractError(
        "问题中的显式年份表示完整年度；SQL 必须用日期维度 year 约束全部年份，"
        "不能用单个 date_id、带 LIMIT 1 的标量日期子查询或省略年份过滤",
        violation_code="semantic.time.full_period_filter_required",
        repairability="llm",
        repair_hint_code="join_date_dimension_and_filter_full_year",
        evidence={
            "expected_years": years,
            "expected_columns": [
                f"{table.get('name')}.year"
                for table in table_infos
                if str(table.get("name") or "").lower()
                in _year_dimension_owners(table_infos)
            ],
            "sql_hash": _safe_sql_hash(tree.sql(dialect="mysql")),
        },
    )


def _check_period_compare_output_shape(
    tree: exp.Expression,
    query: str,
) -> None:
    years = _explicit_year_values(query)
    if (
        len(years) < 2
        or "分别" not in query
        or not any(token in query for token in (
            "销售额", "成交额", "成交金额", "订单金额", "消费金额",
        ))
    ):
        return
    represented = {
        year
        for case in tree.find_all(exp.Case)
        for year in years
        if re.search(
            rf"(?:^|[.\s])year\s*=\s*{year}(?:\D|$)",
            case.sql(dialect="mysql").lower().replace("`", ""),
        )
    }
    final = _final_select(tree)
    final_groups = {
        str(column.name or "").lower()
        for group in [final.args.get("group") if final is not None else None]
        if group is not None
        for column in group.find_all(exp.Column)
    }
    if set(years).issubset(represented) and "year" not in final_groups:
        return
    raise SemanticContractError(
        "问题要求同一展示粒度分别返回多个年份；SQL 必须使用条件聚合形成宽表，"
        "每个年份对应一个结果列，不能返回 year × grain 的长表",
        violation_code="semantic.output.period_compare_wide_required",
        repairability="llm",
        repair_hint_code="pivot_periods_with_conditional_aggregation",
        evidence={
            "expected_years": years,
            "represented_years": sorted(represented),
            "expected_count": len(years),
        },
    )


def _check_entity_total_average_contract(
    tree: exp.Expression,
    query: str,
) -> None:
    if not (
        any(token in query for token in ("客户", "用户", "顾客"))
        and any(token in query for token in ("累计", "累积", "合计", "总计", "总额"))
        and any(token in query for token in ("平均值", "均值", "平均"))
        and any(token in query for token in (
            "销售额", "成交额", "成交金额", "订单金额", "消费金额",
        ))
    ):
        return
    aggregate_aliases: set[str] = set()
    per_entity_sum = False
    for select in tree.find_all(exp.Select):
        grouped = {
            str(column.name or "").lower()
            for group in [select.args.get("group")]
            if group is not None
            for column in group.find_all(exp.Column)
        }
        for projection in select.expressions:
            if projection.find(exp.Sum) is None or "customer_id" not in grouped:
                continue
            alias = str(projection.alias_or_name or "").lower()
            if alias:
                aggregate_aliases.add(alias)
                per_entity_sum = True
    outer_average = any(
        any(
            str(column.name or "").lower() in aggregate_aliases
            for column in average.find_all(exp.Column)
        )
        for average in tree.find_all(exp.Avg)
    )
    final = _final_select(tree)
    final_group_columns = {
        str(column.name or "").lower()
        for group in [final.args.get("group") if final is not None else None]
        if group is not None
        for column in group.find_all(exp.Column)
    }
    explicit_outer_columns = {
        column
        for _, column in explicit_grouping_entries(query, [])
    }
    # ``explicit_grouping_entries`` needs Meta for localized names, so use the
    # stable governed outer axes as a narrow fallback for this contract.
    requested_outer = {
        column for token, column in (
            ("会员等级", "member_level"), ("会员级别", "member_level"),
            ("客户性别", "gender"), ("性别", "gender"),
            ("大区", "region_name"), ("地区", "region_name"),
            ("商品类别", "category"), ("品类", "category"),
        ) if token in query
    } | explicit_outer_columns
    outer_grain_valid = not requested_outer or requested_outer.issubset(
        final_group_columns
    )
    if per_entity_sum and outer_average and outer_grain_valid:
        return
    raise SemanticContractError(
        "问题要求客户累计值的平均；必须先按 customer_id SUM 金额形成每客户一行，"
        "再在外层按请求维度 AVG 该派生累计值",
        violation_code="semantic.aggregate.entity_total_average_required",
        repairability="llm",
        repair_hint_code="aggregate_per_entity_then_average",
        evidence={
            "inner_grain": "customer_id",
            "inner_aggregate": "SUM",
            "outer_aggregate": "AVG",
        },
    )


def check_semantic_contract(
    sql: str,
    query: str,
    table_infos: Sequence[Mapping] | None,
    chat_history: Sequence[Mapping] | None = None,
) -> None:
    """Enforce only high-confidence semantics; ambiguous questions remain LLM decisions."""
    query_text = (query or "").strip().lower()
    tables = list(table_infos or [])
    tree = _parse(sql)
    projected = _projected_columns(tree)
    grouped = _group_columns_across_scopes(tree)

    # These contracts are inferred from the raw question plus supplied Meta and
    # verified against the generated AST. They intentionally run before guided
    # IR checks so a model-repairable semantic error enters the existing
    # ``correct_sql`` path instead of being treated as closed legacy authority.
    _check_qualified_column_ownership(tree, tables)
    _check_explicit_year_contract(tree, query_text, tables)
    _check_period_compare_output_shape(tree, query_text)
    _check_entity_total_average_contract(tree, query_text)
    _check_threshold_output_contracts(tree, query_text, tables)

    time_rules = (
        (re.compile(r"每月|各月|按月|月度.*(?:趋势|变化|分布)"), ("month", "月份", "月"), "月"),
        (re.compile(r"每季|各季|按季|季度.*(?:趋势|变化|分布)"), ("quarter", "季度"), "季度"),
    )
    for pattern, terms, label in time_rules:
        if not pattern.search(query_text):
            continue
        candidates = {
            str(column.get("name") or "").lower()
            for table in tables
            for column in (table.get("columns") or [])
            if any(term in _column_text(column) for term in terms)
            and str(column.get("role") or "").lower() == "dimension"
        }
        final = _final_select(tree)
        upstream_aggregate_aliases = _aggregate_projection_aliases(tree)
        nested_rollup = bool(
            candidates & grouped
            and final is not None
            and any(
                {
                    column.name.lower()
                    for column in average.find_all(exp.Column)
                }
                & upstream_aggregate_aliases
                for average in final.find_all(exp.Avg)
            )
        )
        if candidates and not (
            candidates & grouped
            and (candidates & projected or nested_rollup)
        ):
            raise SemanticContractError(
                f"问题要求{label}粒度，SQL 必须投影并分组元数据中的可读字段 {sorted(candidates)}，"
                "不能使用日期 ID 截取、除法或拼接值代替",
                violation_code="semantic.time.readable_axis_missing",
                repairability="upstream",
                repair_hint_code="use_metadata_readable_time_axis",
                evidence={
                    "expected_columns": sorted(candidates),
                    "actual_columns": sorted(projected & grouped),
                },
            )
        break

    # Grouping evidence must be grammatically adjacent to the metadata term
    # (``按购买件数``), or make a dimension/key the visible ranked target.
    # A global ``按``/``最高`` elsewhere in the sentence is not sufficient.
    explicit_entries = explicit_grouping_entries(query_text, tables)
    if explicit_entries:
        aggregated = _aggregated_columns(tree)
        explicit_entries = {entry for entry in explicit_entries if entry[1] not in aggregated}
        # A literal-constrained dimension is a scalar filter, even when the
        # predicate is IN/NOT IN rather than equality.  It must not be promoted
        # to a GROUP BY obligation merely because the question also contains
        # an explicit grouped axis.
        scalar_filter_columns = {
            str(column.name or "").lower()
            for select in tree.find_all(exp.Select)
            for where in [select.args.get("where")]
            if where is not None
            for predicate in where.walk()
            if isinstance(predicate, (exp.EQ, exp.NEQ, exp.In, exp.Between))
            and predicate.find(exp.Literal) is not None
            for column in predicate.find_all(exp.Column)
        }
        explicit_entries = {
            entry for entry in explicit_entries
            if entry[1] not in scalar_filter_columns
        }
        # “商品类别/产品品类” names the category axis.  Generic aliases such
        # as “商品” on product_name must not create a second display-grain.
        if (
            any(token in query_text for token in (
                "商品类别", "产品类别", "商品品类", "产品品类",
                "商品品牌", "产品品牌",
            ))
            and not any(token in query_text for token in (
                "商品名称", "产品名称", "商品名", "产品名",
            ))
        ):
            explicit_entries = {
                entry for entry in explicit_entries if entry[1] != "product_name"
            }
        grouped_customer_summary = bool(
            (
                any(token in query_text for token in (
                    "平均活跃天数", "平均跨度", "平均间隔", "平均天数",
                ))
                and tree.find(exp.Avg) is not None
            )
            or (
                any(token in query_text for token in (
                    "分为", "划为", "分段", "分档",
                ))
                and tree.find(exp.Case) is not None
                and tree.find(exp.Count) is not None
            )
        )
        if grouped_customer_summary:
            explicit_entries = {
                entry for entry in explicit_entries
                if entry[1] not in {"customer_name", "customer_id"}
            }
        if re.search(r"第[一二三四1234]季度", query_text):
            explicit_entries = {
                entry for entry in explicit_entries if entry[1] != "quarter"
            }
        if (
            any(token in query_text for token in (
                "首次下单年份", "首次购买年份", "首单年份", "首笔交易年份",
            ))
            and "first_year" in projected
            and "first_year" in grouped
        ):
            explicit_entries = {
                entry for entry in explicit_entries if entry[1] != "year"
            }
        # Cohort dimensions are derived readable axes.  A query grouped by the
        # first-purchase month/year is correctly represented by cohort_month /
        # cohort_year; requiring the raw dim_date column in the final SELECT
        # rejects valid CTE-based cohort SQL.
        derived_cohort_axes = {
            axis
            for axis in ("month", "year")
            if f"cohort_{axis}" in projected and f"cohort_{axis}" in grouped
        }
        if derived_cohort_axes:
            explicit_entries = {
                entry for entry in explicit_entries
                if entry[1] not in derived_cohort_axes
            }
        # In “先筛出一批客户，再按会员等级汇总成交额”, customers define an
        # intermediate population, not the visible grouping grain.
        intermediate_customer_population = bool(
            re.search(r"(?:先|首先).{0,28}(?:客户|用户).{0,16}(?:再|然后|随后)", query_text)
            and re.search(r"(?:统计|汇总|计算|分析)", query_text)
        )
        if intermediate_customer_population:
            explicit_entries = {
                entry for entry in explicit_entries
                if entry[1] not in {"customer_name", "customer_id"}
            }
        scalar_rank_share = bool(
            re.search(
                r"(?:前|后)(?:\d{1,4}|[一二两三四五六七八九十百]{1,4})",
                query_text,
            )
            and any(token in query_text for token in (
                "合计占比", "贡献占比", "合计比重",
            ))
            and not _plain_projection_columns(tree)
        )
        if scalar_rank_share:
            explicit_entries = {
                entry for entry in explicit_entries if entry[1] not in grouped
            }
        grouped_rank_share = bool(
            re.search(
                r"(?:前|后|最高的|最低的|最多的|最少的)"
                r"(?:\d{1,4}|[一二两三四五六七八九十百]{1,4})?",
                query_text,
            )
            and any(token in query_text for token in (
                "合计占", "合计贡献", "共同占", "总共占",
            ))
            and tree.find(exp.Div) is not None
        )
        if grouped_rank_share:
            explicit_entries = {
                entry for entry in explicit_entries
                if entry[1] in projected or entry[1] not in grouped
            }
        if (
            any(token in query_text for token in (
                "首次购买", "首次交易", "首次下单", "最后购买",
                "最后一次交易", "末次交易",
            ))
            and any(token in query_text for token in (
                "人数", "客户数", "用户数",
            ))
        ):
            explicit_entries = {
                entry for entry in explicit_entries
                if entry[1] not in {"customer_name", "customer_id"}
            }
        per_customer_ratio = bool(
            any(token in query_text for token in (
                "每位下单客户", "每位购买客户", "每位活跃客户",
                "平均每个客户", "平均每名客户", "平均每位客户", "人均",
            ))
            and any(
                count.find(exp.Distinct) is not None
                and any(
                    column.name.lower() == "customer_id"
                    for column in count.find_all(exp.Column)
                )
                for count in tree.find_all(exp.Count)
            )
        )
        if per_customer_ratio:
            explicit_entries = {
                entry for entry in explicit_entries
                if entry[1] not in {"customer_name", "customer_id"}
            }
        explicit_dimensions = {column for _, column in explicit_entries}
        explicit_dimensions -= _aggregated_columns(tree)
        missing = explicit_dimensions - (projected & grouped)
        if missing:
            raise SemanticContractError(
                f"问题明确要求按元数据维度 {sorted(explicit_dimensions)} 分组；"
                f"SQL 缺少投影或 GROUP BY 字段 {sorted(missing)}",
                violation_code="semantic.grouping.missing_dimension",
                repairability="llm",
                repair_hint_code="add_explicit_group_dimension",
                evidence={
                    "expected_columns": sorted(explicit_dimensions),
                    "missing_columns": sorted(missing),
                },
            )
        used_tables = {table.name.lower() for table in tree.find_all(exp.Table)}
        owners_by_column: dict[str, set[str]] = {}
        for table_name, column_name in explicit_entries:
            owners_by_column.setdefault(column_name, set()).add(table_name)
        missing_owners = {}
        for column_name, owners in owners_by_column.items():
            if len(owners) != 1 or owners & used_tables:
                continue
            owner = next(iter(owners))
            if _fact_fk_satisfies_explicit_dimension_key(
                query_text, owner, column_name, used_tables, tables,
            ):
                continue
            missing_owners[owner] = column_name
        if missing_owners:
            details = [f"{table}.{column}" for table, column in sorted(missing_owners.items())]
            raise SemanticContractError(
                f"问题指定的维度属于元数据表 {details}；SQL 必须引用对应维表并通过 PK/FK 连接",
                violation_code="semantic.join.missing_predicate",
                repairability="upstream",
                repair_hint_code="restore_required_dimension_table",
                evidence={
                    "expected_tables": sorted(missing_owners),
                    "missing_columns": sorted(missing_owners.values()),
                },
            )

    entity_request = _entity_listing_requested(query_text)
    if entity_request:
        requested_by_table = _requested_metadata_fields(query_text, tables)
        requested_entries = {
            (table_name, column)
            for table_name, columns in requested_by_table.items()
            for column in columns
        }
        requested_names = {column for _, column in requested_entries}
        metadata_roles = _metadata_schemas(tables)
        requested_ids = {
            entry
            for entry in requested_entries
            if metadata_roles.get(entry[0], {}).get(entry[1]) in {
                "primary_key", "foreign_key", "unique_key",
            }
            or entry[1].endswith("_id")
        }
        display = _matching_display_columns(query_text, tables)
        # 当目录问题明确要求属性列时，该实体词只限定 Meta 作用域，不再
        # 隐式授权同表的展示名称；显式要求名称时它已在 requested 中。
        for table in tables:
            table_name = str(table.get("name") or "").lower()
            if not requested_by_table.get(table_name):
                continue
            display -= {
                str(column.get("name") or "").lower()
                for column in (table.get("columns") or [])
                if column.get("is_display")
                or str(column.get("name") or "").lower().endswith("_name")
            }
        display_entries = {
            (str(table.get("name") or "").lower(), str(column.get("name") or "").lower())
            for table in tables
            for column in (table.get("columns") or [])
            if str(column.get("name") or "").lower() in display
        }
        expected_visible = display | requested_names
        projection_evidence = _visible_projection_evidence(tree, tables)
        direct_entries = set(projection_evidence["direct_entries"])
        direct_names = set(projection_evidence["direct_names"])
        lineage_entries = set(projection_evidence["lineage_entries"])
        used_tables = {
            str(table.name or "").lower()
            for table in tree.find_all(exp.Table)
            if str(table.name or "").lower() not in _cte_aliases(tree)
        }

        # 未显式请求标识时，纯 ID 投影优先归类为 id_only；否则更宽泛的
        # missing_requested_attribute 会掩盖最接近根因的修复提示。
        id_projection = {
            name
            for name in direct_names
            if name.endswith("_id")
        }
        if (
            expected_visible
            and id_projection
            and not requested_ids
            and not (expected_visible & direct_names)
        ):
            raise SemanticContractError(
                f"业务实体查询不得只投影 ID {sorted(id_projection)}；"
                f"应使用题面要求的元数据展示字段或属性 {sorted(expected_visible)}",
                violation_code="semantic.projection.id_only",
                repairability="llm",
                repair_hint_code="replace_id_with_display_field",
                evidence={
                    "expected_columns": sorted(expected_visible),
                    "actual_columns": sorted(id_projection),
                },
            )

        missing_requested_entries = {
            entry
            for entry in requested_entries
            if not _requested_entry_is_visible(
                entry,
                direct_entries,
                query=query_text,
                used_tables=used_tables,
                table_infos=tables,
            )
        }
        if missing_requested_entries:
            missing_labels = [
                f"{table}.{column}"
                for table, column in sorted(missing_requested_entries)
            ]
            raise SemanticContractError(
                "实体目录结果缺少题面明确要求的 owner-qualified 元数据字段 "
                f"{missing_labels}",
                violation_code="semantic.projection.missing_requested_attribute",
                repairability="llm",
                repair_hint_code="restore_requested_metadata_attributes",
                evidence={
                    "expected_columns": [
                        f"{table}.{column}"
                        for table, column in sorted(requested_entries)
                    ],
                    "actual_columns": [
                        f"{table}.{column}"
                        for table, column in sorted(direct_entries)
                    ],
                    "missing_columns": missing_labels,
                },
            )

        sensitive_entries = {
            (str(table.get("name") or "").lower(), str(column.get("name") or "").lower())
            for table in tables
            for column in (table.get("columns") or [])
            if column.get("sensitive") and column.get("name")
        }
        unrequested_sensitive = (
            sensitive_entries - requested_entries
            if projection_evidence["has_star"]
            else lineage_entries & sensitive_entries - requested_entries
        )
        if unrequested_sensitive:
            unexpected_sensitive = [
                f"{table}.{column}"
                for table, column in sorted(unrequested_sensitive)
            ]
            raise SemanticContractError(
                "实体目录结果不得通过表达式、聚合或窗口函数额外返回敏感字段 "
                f"{unexpected_sensitive}；显式请求仍由 access policy 与 masking 治理",
                violation_code="semantic.projection.unrequested_extra",
                repairability="llm",
                repair_hint_code="remove_unrequested_sensitive_projection",
                evidence={
                    "unexpected_columns": unexpected_sensitive,
                    "sensitive_lineage": True,
                },
            )

        if expected_visible:
            allowed_window_aliases = (
                _window_projection_aliases(tree)
                if re.search(r"(?:排行|排名|最高|最低|前|后)", query_text)
                else set()
            )
            allowed_entries = requested_entries | display_entries
            allowed_names = expected_visible
            unexpected_entries = {
                entry
                for entry in direct_entries
                if entry not in allowed_entries
                and not any(
                    requested[1] == entry[1]
                    and _requested_entry_is_visible(
                        requested,
                        {entry},
                        query=query_text,
                        used_tables=used_tables,
                        table_infos=tables,
                    )
                    for requested in requested_ids
                )
            }
            unexpected_names = (
                direct_names
                - allowed_names
                - _aggregate_projection_aliases(tree)
                - allowed_window_aliases
            )
            if unexpected_entries or unexpected_names:
                unexpected = sorted({
                    *(f"{table}.{column}" for table, column in unexpected_entries),
                    *unexpected_names,
                })
                raise SemanticContractError(
                    f"实体列表不得额外投影；只能返回题面要求的元数据字段 {sorted(allowed_names)}；"
                    f"发现未要求的额外投影 {sorted(unexpected)}",
                    violation_code="semantic.projection.unrequested_extra",
                    repairability="llm",
                    repair_hint_code="remove_unrequested_projection",
                    evidence={
                        "expected_columns": sorted(allowed_names),
                        "actual_columns": sorted(direct_names),
                    },
                )

    semantic_shape = analyze_semantic_shape(sql, query_text, tables)
    if semantic_shape["anti_join_requested"] and not semantic_shape["anti_join_satisfied"]:
        has_not_in = any(isinstance(item.this, exp.In) for item in tree.find_all(exp.Not))
        has_anti_structure = (
            has_not_in
            or any(isinstance(item.this, exp.Exists) for item in tree.find_all(exp.Not))
            or any(
                str(join.args.get("side") or "").upper() == "LEFT"
                for join in tree.find_all(exp.Join)
            )
        )
        violation_code = (
            "semantic.anti_join.null_unsafe"
            if has_not_in
            else "semantic.anti_join.invalid_key"
            if has_anti_structure
            else "semantic.anti_join.missing"
        )
        raise SemanticContractError(
            "问题要求返回不存在关联事实的实体；SQL 必须使用元数据 PK/FK 关联的 "
            "LEFT JOIN ... IS NULL、NOT EXISTS 或空值安全 NOT IN",
            violation_code=violation_code,
            repairability="llm",
            repair_hint_code=violation_code.replace("semantic.", "repair_").replace(".", "_"),
            evidence={
                "anti_join_strategy": semantic_shape.get("anti_join_strategy"),
                "candidate_edge_count": 0,
            },
        )

    history = list(chat_history or [])
    if history and _FOLLOW_UP_RE.search(query_text):
        previous_sql = str(history[-1].get("sql") or "").strip()
        if previous_sql:
            previous = _parse(previous_sql)
            previous_groups = _group_columns(previous)
            if previous_groups and not previous_groups.issubset(grouped):
                raise SemanticContractError(
                    f"省略追问必须继承上一轮分组维度 {sorted(previous_groups)}，当前分组为 {sorted(grouped)}",
                    violation_code="semantic.follow_up.missing_group",
                    repairability="llm",
                    repair_hint_code="restore_follow_up_group",
                    evidence={
                        "expected_columns": sorted(previous_groups),
                        "actual_columns": sorted(grouped),
                        "history_contract": True,
                    },
                )
            previous_select = _final_select(previous)
            current_select = _final_select(tree)
            for key, label in (("where", "过滤条件"), ("order", "排序"), ("limit", "限制")):
                if previous_select is not None and previous_select.args.get(key) is not None:
                    if current_select is None or current_select.args.get(key) is None:
                        raise SemanticContractError(
                            f"省略追问必须继承上一轮{label}",
                            violation_code={
                                "where": "semantic.follow_up.missing_filter",
                                "order": "semantic.follow_up.missing_order",
                                "limit": "semantic.follow_up.missing_limit",
                            }[key],
                            repairability="llm",
                            repair_hint_code=f"restore_follow_up_{key}",
                            evidence={"history_contract": True},
                        )


def check_semantic_ir(sql: str, semantic_ir: Mapping | None) -> None:
    """Verify deterministic dimensions, measures, ordering and limits against IR."""
    ir = semantic_ir or {}
    query_contract = dict(ir.get("query_contract") or {})
    if query_contract.get("authority") == "closed":
        contract_audit = audit_sql_against_query_contract(sql, query_contract)
        if not contract_audit.get("valid"):
            violations = list(contract_audit.get("violations") or [])
            raise SemanticIRError(
                "semantic_query_contract_mismatch: "
                + ",".join(
                    str(item.get("code") or "unknown") for item in violations[:6]
                ),
                violation_code="semantic_ir.shared_query_contract_mismatch",
                repairability="llm",
                repair_hint_code="restore_shared_query_contract",
                evidence={
                    "contract_hash": contract_audit.get("contract_hash"),
                    "violation_codes": [
                        str(item.get("code") or "unknown")
                        for item in violations[:8]
                    ],
                },
            )
    if not (ir.get("compiler") or {}).get("supported"):
        return
    # Model-routed SQL is checked by the shared query contract above.  Long-tail
    # requests intentionally have no benchmark-shaped deterministic template
    # to compare byte-for-byte.
    if use_model_sql(ir):
        return
    if (ir.get("compiler") or {}).get("mode") == COMPILER_MODE_V2:
        try:
            validate_compiled_semantic_ir(sql, ir)
        except Exception as exc:
            raise SemanticIRError(
                f"semantic_ir_v2_mismatch: {type(exc).__name__}",
                violation_code="semantic_ir.v2_contract_mismatch",
                repairability="upstream",
                repair_hint_code="rebuild_semantic_ir_v2",
                evidence={
                    "semantic_family": str(ir.get("semantic_family") or ""),
                    "compiler_supported": True,
                    "reason_code": str(exc)[:160],
                },
            ) from exc
        return
    tree = _parse(sql)
    used_tables = {table.name.lower() for table in tree.find_all(exp.Table)}
    required_tables = {
        str(item.get("table") or "").lower()
        for item in [*(ir.get("dimensions") or []), *(ir.get("measures") or [])]
        if item.get("table")
    }
    # CTE aliases are tables to SQLGlot too, therefore required physical tables
    # are checked as a subset rather than requiring exact equality.
    if not required_tables.issubset(used_tables):
        missing_tables = sorted(required_tables - used_tables)
        raise SemanticIRError(
            f"semantic_ir_mismatch: 缺少表 {missing_tables}",
            violation_code="semantic_ir.missing_table",
            repairability="upstream",
            repair_hint_code="restore_ir_required_tables",
            evidence={
                "expected_tables": sorted(required_tables),
                "missing_tables": missing_tables,
                "compiler_supported": True,
            },
        )

    if ir.get("intent") == "repeat_rate":
        # The registered formula guard performs the detailed CTE/ratio check.
        return

    projected = _projected_columns(tree)
    grouped = _group_columns(tree)
    for dimension in ir.get("dimensions") or []:
        column = str(dimension.get("column") or "").lower()
        if column not in projected or column not in grouped:
            raise SemanticIRError(
                f"semantic_ir_mismatch: 维度 {column} 必须投影并分组",
                violation_code="semantic_ir.missing_dimension",
                repairability="llm",
                repair_hint_code="restore_ir_dimension",
                evidence={
                    "expected_columns": [column],
                    "actual_columns": sorted(projected & grouped),
                    "compiler_supported": True,
                },
            )

    aliases = {
        table.alias_or_name.lower(): table.name.lower() for table in tree.find_all(exp.Table)
    }
    for measure in ir.get("measures") or []:
        function = str(measure.get("function") or "").upper()
        expected = f"{measure.get('table')}.{measure.get('column')}"
        if function and not any(
            aggregate.sql_name().upper() == function
            and any(
                column.name.lower() == str(measure.get("column") or "").lower()
                and aliases.get((column.table or "").lower(), (column.table or "").lower())
                == str(measure.get("table") or "").lower()
                for column in aggregate.find_all(exp.Column)
            )
            for aggregate in tree.find_all(exp.AggFunc)
        ):
            raise SemanticIRError(
                f"semantic_ir_mismatch: 缺少口径 {function}({expected})",
                violation_code="semantic_ir.missing_measure",
                repairability="upstream",
                repair_hint_code="restore_ir_measure",
                evidence={
                    "expected_count": 1,
                    "actual_count": 0,
                    "ir_proof_hash": _safe_sql_hash(f"{function}:{expected}"),
                    "compiler_supported": True,
                },
            )

    select = _final_select(tree)
    if select is None:
        raise SemanticParseError(
            "semantic_ir_invalid: 未找到最终 SELECT",
            evidence={"sql_hash": _safe_sql_hash(sql)},
        )
    expected_limit = ir.get("limit")
    if expected_limit is not None:
        limit = select.args.get("limit")
        actual = limit.expression.this if limit is not None and isinstance(limit.expression, exp.Literal) else None
        if str(actual) != str(expected_limit):
            raise SemanticIRError(
                f"semantic_ir_mismatch: LIMIT 应为 {expected_limit}",
                violation_code="semantic_ir.limit_mismatch",
                repairability="deterministic" if actual is None else "llm",
                repair_hint_code="restore_ir_limit",
                evidence={
                    "expected_limit": expected_limit,
                    "actual_limit": actual,
                    "compiler_supported": True,
                },
            )
    if ir.get("order_by") and select.args.get("order") is None:
        expected_order = list(ir.get("order_by") or [])
        raise SemanticIRError(
            "semantic_ir_mismatch: 缺少 IR 指定的排序",
            violation_code="semantic_ir.order_mismatch",
            repairability="deterministic" if len(expected_order) == 1 else "llm",
            repair_hint_code="restore_ir_order",
            evidence={
                "expected_count": len(expected_order),
                "actual_count": 0,
                "direction": (
                    str(expected_order[0].get("direction") or "")
                    if len(expected_order) == 1 and isinstance(expected_order[0], Mapping)
                    else None
                ),
                "compiler_supported": True,
            },
        )
