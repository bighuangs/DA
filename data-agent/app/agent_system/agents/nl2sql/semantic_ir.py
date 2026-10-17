"""基于元数据的语义 IR 与保守的 SQLGlot AST 编译器（TODO-42）。

本模块提供两条确定性能力，避免对 LLM 的依赖：
    1. build_semantic_ir：仅依据 query 中显式命中的元数据词与注册指标规则，
       构建一份版本化的语义 IR（intent/dimensions/measures/joins/limit 等）。
    2. compile_semantic_ir：把“结构完整且连接唯一”的 IR 用 SQLGlot AST 节点
       确定性编译为 MySQL SQL，供 generate_sql 短路 LLM 调用。

设计要点：
    - 只接受“显式元数据词命中”作为维度/度量证据，拒绝模糊向量候选，
      保证编译出的 SQL 可审计、可重复。
    - 渐进式编译白名单仅含 aggregate/ranking/repeat_rate 三种意图，
      其余意图一律回退 LLM 路径，避免越界编译。
    - 多表连接必须存在唯一 PK/FK 路径才编译，否则回退 LLM，保证 JOIN 正确性。
"""
from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence

from sqlglot import exp

from app.agent_system.agents.nl2sql.follow_up import resolve_follow_up
from app.agent_system.agents.nl2sql.semantic_ir_v2 import (
    IR_VERSION_V2,
    build_semantic_ir_v2,
    compile_semantic_ir_v2,
    transaction_item_quantity_requested,
)
from app.agent_system.agents.nl2sql.semantic_coverage import (
    audit_compiled_semantic_coverage,
    detect_semantic_coverage,
)

# IR 版本号，进入索引/缓存指纹，便于发现 IR 契约漂移
IR_VERSION = IR_VERSION_V2
# 多轮省略/替换追问 cue：命中即认为当前问题依赖历史，交给 LLM 处理
_FOLLOW_UP_RE = re.compile(r"(?:^那|^那么|呢$|改成|换成|再看|再算)")
# 订单计数口径 cue：命中即把排名度量从 SUM(order_amount) 改写为 COUNT(order_id)
_ORDER_COUNT_RE = re.compile(r"(?:订单(?:数量|数|量|总数|笔数)|下单次数)")
# 不支持的简单意图关键词：命中即回退 LLM（过滤/时间/比例/比较等复杂语义）
_UNSUPPORTED_SIMPLE = re.compile(
    r"同比|环比|占比|比例|平均|去重|不同|没有|从未|超过|低于|高于|介于|"
    r"\d{4}\s*年|本月|上月|今年|去年|每天|每月|各月|每季|各季|趋势|对比"
)
# 中文数字 -> 阿拉伯数字，用于解析“前十”中的“十”
_CHINESE_NUMBERS = {
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def _column_terms(column: Mapping) -> list[str]:
    """收集字段的全部检索词：列名 + 描述 + 别名，统一小写去空白。"""
    return [
        str(value).strip().lower()
        for value in (column.get("name", ""), column.get("description", ""), *(column.get("alias") or []))
        if value
    ]


def _term_in_query(term: str, query: str) -> bool:
    """匹配元数据术语是否出现在 query 中，兼容“销售额→销售总额”一类修饰写法。

    核心实现思路：精确子串匹配为主，对以“额”结尾的术语额外尝试去掉“额”补“总额”，
    避免“销售总额”提问漏命中“销售额”元数据。
    """
    normalized = str(term or "").strip().lower()
    return bool(normalized) and (
        normalized in query
        # “销售额”也匹配“销售总额”：去掉末尾“额”后补“总额”再判断
        or (normalized.endswith("额") and f"{normalized[:-1]}总额" in query)
    )


def _is_measure(column: Mapping) -> bool:
    """判断字段是否为度量：role=measure，或 fact_ 表的 _amount/_quantity 后缀列。

    核心实现思路：Schema 自动同步无法从 MySQL 类型推断数仓语义，可能把数值事实字段
    标成维度。这里用 fact_ 表 + 金额/数量后缀作为确定性兼容兜底；维度表永不走此兜底。
    """
    role = str(column.get("role") or "").lower()
    table = str(column.get("table") or "").lower()
    name = str(column.get("name") or "").lower()
    # Schema auto-sync cannot infer warehouse semantics from MySQL types and may
    # label numeric fact fields as dimensions. Common fact measure suffixes are a
    # deterministic compatibility fallback; dimension tables never use it.
    return role == "measure" or (
        table.startswith("fact_") and name.endswith(("_amount", "_quantity"))
    )


def _is_dimension(column: Mapping) -> bool:
    """判断字段是否为维度：role=dimension 且非度量 且 表名以 dim_ 开头。"""
    return (
        str(column.get("role") or "").lower() == "dimension"
        and not _is_measure(column)
        and str(column.get("table") or "").lower().startswith("dim_")
    )


def _flatten_columns(table_infos: Sequence[Mapping]) -> list[dict]:
    """把 table_infos 中嵌套的 columns 展平为一维列表，每项带上所属表名。"""
    return [
        {"table": str(table.get("name") or ""), **dict(column)}
        for table in table_infos
        for column in (table.get("columns") or [])
    ]


def _split_ref(value: str | None) -> tuple[str, str] | None:
    """把 "table.column" 形式的引用拆为 (table, column)，非法格式返回 None。"""
    parts = str(value or "").split(".", 1)
    return (parts[0], parts[1]) if len(parts) == 2 and all(parts) else None


def _direct_metric(query: str, metric_infos: Sequence[Mapping]) -> Mapping | None:
    """在 query 中直接命中且唯一带 formula_rule 的注册指标。

    核心实现思路：先用指标名/别名在 query 中做子串匹配，再从中筛出带 formula_rule
    的指标；仅当有且仅有一个带规则的命中时才返回，避免多义指标误触发确定性编译。
    """
    query_text = query.lower()
    # 命中：指标名或任一别名出现在 query 中
    hits = [
        metric for metric in metric_infos
        if any(
            term and str(term).lower() in query_text
            for term in (metric.get("name", ""), *(metric.get("alias") or []))
        )
    ]
    # 仅保留带 formula_rule 的命中，保证有可编译的口径
    ruled = [metric for metric in hits if metric.get("formula_rule")]
    # 唯一才返回，避免多义指标误触发
    return ruled[0] if len(ruled) == 1 else None


def _extract_limit(query: str, default: int | None = None) -> int | None:
    """从 query 中解析 LIMIT：优先阿拉伯数字“前10”，其次中文“前十”，否则用 default。"""
    # 阿拉伯数字：前/后/top N，限制在 1..1000 防止超大 LIMIT
    match = re.search(r"(?:前|后|top\s*)(\d{1,4})", query, flags=re.IGNORECASE)
    if match:
        return max(1, min(int(match.group(1)), 1000))
    # 中文数字：前/后 + 一..十
    match = re.search(r"(?:前|后)([一二两三四五六七八九十])", query)
    if match:
        return _CHINESE_NUMBERS[match.group(1)]
    return default


def _result_alias(column: str, function: str) -> str:
    """为聚合列生成稳定的中文友好别名，供前端展示与 order_by 引用。

    核心实现思路：对常见事实表列（order_id/order_amount/order_quantity）按函数
    映射到固定别名（order_count/total_amount/total_sales 等），其余列用
    {function}_{column} 兜底，保证别名可预测且与指标口径一致。
    """
    name = column.lower()
    # COUNT(order_id) -> order_count，其余 COUNT(col) -> col_count
    if function == "COUNT":
        return "order_count" if name == "order_id" else f"{name}_count"
    # order_amount：SUM -> total_amount，AVG -> average_amount
    if name == "order_amount":
        return "total_amount" if function == "SUM" else "average_amount"
    # order_quantity：SUM -> total_sales，AVG -> average_sales
    if name == "order_quantity":
        return "total_sales" if function == "SUM" else "average_sales"
    # 兜底：{function}_{column}
    return f"{function.lower()}_{name}"


def _find_joins(required_tables: set[str], columns: Sequence[Mapping]) -> list[dict] | None:
    """在所需表之间寻找唯一的 PK/FK 连接路径。

    核心实现思路：仅支持两表连接，按字段名匹配 PK/FK 对，只有恰好一对候选时
    才返回，否则返回 None 表示无唯一路径，回退 LLM。

    业务逻辑：
        1. 表数<=1 无需连接，返回空列表
        2. 表数!=2 不支持，返回 None
        3. 按字段名分组，找在两表中都出现且角色为 PK+FK 的字段
        4. 候选恰好 1 个才返回，否则 None（多义路径回退 LLM）
    """
    if len(required_tables) <= 1:
        return []
    # 仅支持两表连接，多表回退 LLM
    if len(required_tables) != 2:
        return None
    left, right = sorted(required_tables)
    candidates: list[dict] = []
    # 按字段名分组：同名字段在两表中各出现一次的候选
    by_name: dict[str, list[Mapping]] = {}
    for column in columns:
        if column.get("table") in required_tables:
            by_name.setdefault(str(column.get("name") or "").lower(), []).append(column)
    for name, rows in by_name.items():
        # 必须恰好两行且分属两张表
        if len(rows) != 2 or {str(row.get("table")) for row in rows} != required_tables:
            continue
        roles = {str(row.get("role") or "").lower() for row in rows}
        # 角色为 PK+FK，或字段名以 _id 结尾且含 PK
        if roles == {"primary_key", "foreign_key"} or (
            name.endswith("_id") and "primary_key" in roles
        ):
            row_left = next(row for row in rows if row.get("table") == left)
            row_right = next(row for row in rows if row.get("table") == right)
            candidates.append({
                "left_table": left, "left_column": str(row_left.get("name")),
                "right_table": right, "right_column": str(row_right.get("name")),
            })
    # 唯一路径才返回，多义回退 LLM
    return candidates if len(candidates) == 1 else None


def _base_ir(query: str) -> dict:
    """构造语义 IR 的初始骨架，所有字段均为“未解析”默认值。"""
    return {
        "schema_version": IR_VERSION,
        "query": query,
        "status": "partial",
        "intent": "unknown",
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
        # 编译器默认不支持，等待结构补全后翻转
        "compiler": {"supported": False, "mode": "llm_fallback", "reason": "语义证据不足"},
    }


def build_semantic_ir(
    query: str,
    table_infos: Sequence[Mapping] | None,
    metric_infos: Sequence[Mapping] | None,
    chat_history: Sequence[Mapping] | None = None,
) -> dict:
    """仅依据显式命中的元数据词与注册指标规则，构建一份版本化语义 IR。

    核心实现思路：以 _base_ir 为骨架，按“注册指标规则 → 显式元数据词命中”两条
    路径补全 intent/dimensions/measures/joins/limit，最后校验表召回完整性与
    连接唯一性，决定 compiler.supported 是否翻转为 True。

    业务逻辑：
        1. 空问题或显式省略追问 -> 直接回退 LLM（保留 reason）
        2. 注册指标路径：唯一命中带 formula_rule 的指标时，按 rule_type
           （ranking/repeat_rate）填充维度/度量/排序/limit
        3. 显式元数据词路径：未命中指标时，用精确词匹配找维度+度量，
           要求恰好 1 维度 + 1 度量且含分组/排名 cue 才编译
        4. 校验所需表已全部召回，且两表间存在唯一 PK/FK 连接路径
        5. 意图在白名单（aggregate/ranking/repeat_rate）内才标记 supported=True
    """
    tables = list(table_infos or [])
    original_query = str(query or "").strip()
    effective_query = original_query
    follow_up_operations: list[str] = []
    follow_up_resolution: dict = {}
    if chat_history:
        follow_up_resolution = resolve_follow_up(original_query, chat_history)
        if (
            follow_up_resolution.get("action") != "new_topic"
            and not follow_up_resolution.get("needs_clarification")
        ):
            effective_query = str(
                follow_up_resolution.get("standalone_question") or original_query
            ).strip()
            follow_up_operations = [
                f"{str(item.get('op') or 'unknown')}:{str(item.get('slot') or 'unknown')}"
                for item in (follow_up_resolution.get("operations") or [])
            ]

    def finalize(candidate: dict) -> dict:
        """Attach one safe coverage contract to every deterministic/fallback path."""

        output = copy.deepcopy(candidate)
        coverage = dict(output.get("semantic_coverage") or {})
        if not coverage:
            compiled = None
            if (output.get("compiler") or {}).get("supported"):
                compiled = compile_semantic_ir(output)
            coverage = (
                audit_compiled_semantic_coverage(
                    effective_query,
                    tables,
                    compiled,
                )
                if compiled
                else detect_semantic_coverage(effective_query, tables)
            )
        if follow_up_operations:
            coverage["follow_up_operations"] = list(follow_up_operations)
            output["context_refs"] = [
                {
                    "type": "follow_up",
                    "action": str(
                        follow_up_resolution.get("action") or "inherit"
                    ),
                }
            ]
        output["semantic_coverage"] = coverage
        if (
            list(coverage.get("unresolved_slots") or [])
            and (output.get("compiler") or {}).get("supported")
        ):
            output["status"] = "partial"
            output["compiler"] = {
                **dict(output.get("compiler") or {}),
                "supported": False,
                "mode": "llm_fallback",
                "reason": "semantic coverage incomplete",
                "missing_slots": list(coverage.get("unresolved_slots") or []),
            }
        output["compiler"] = {
            **dict(output.get("compiler") or {}),
            "coverage_version": coverage.get("schema_version"),
        }
        return output

    # First preserve an already closed v2 contract. Closure is no longer
    # inferred from table/column availability: the compiled candidate must
    # represent every independently detected high-confidence semantic slot.
    v2_ir = build_semantic_ir_v2(
        effective_query,
        tables,
        metric_infos,
        [] if effective_query != original_query else chat_history,
    )
    v2_sql = compile_semantic_ir_v2(v2_ir)
    v2_coverage = (
        audit_compiled_semantic_coverage(
            effective_query,
            tables,
            v2_sql,
        )
        if v2_sql
        else None
    )
    if v2_ir is not None and v2_coverage and not v2_coverage.get("unresolved_slots"):
        closed_v2 = copy.deepcopy(v2_ir)
        closed_v2["semantic_coverage"] = v2_coverage
        closed_v2["compiler"] = {
            **dict(closed_v2.get("compiler") or {}),
            "coverage_version": v2_coverage.get("schema_version"),
            "coverage_reason": "compiled SQL represents all detected slots",
        }
        return finalize(closed_v2)

    # A v2 candidate with a semantic gap stays partial.  Do not grow a second
    # benchmark-shaped compiler registry to fit rare multi-stage questions;
    # the existing planning/model or clarification path owns that long tail.
    if v2_ir is not None:
        partial_v2 = copy.deepcopy(v2_ir)
        if v2_coverage:
            partial_v2["semantic_coverage"] = v2_coverage
            partial_v2["status"] = "partial"
            partial_v2["compiler"] = {
                **dict(partial_v2.get("compiler") or {}),
                "supported": False,
                "mode": "llm_fallback",
                "reason": "semantic coverage incomplete",
                "missing_slots": list(v2_coverage.get("unresolved_slots") or []),
                "coverage_version": v2_coverage.get("schema_version"),
            }
        return finalize(partial_v2)

    ir = _base_ir(effective_query)
    # 展平所有字段，每项带所属表名，便于后续按词匹配
    columns = _flatten_columns(tables)
    query_text = effective_query.lower()
    # 空问题：交由 LLM 处理
    if not query_text:
        ir["compiler"]["reason"] = "空问题交由 LLM 处理"
        return finalize(ir)
    # 历史记录本身不代表当前问题依赖历史。完整、可独立理解的问题仍应使用
    # 确定性 IR；只有显式省略/替换追问才保留给多轮 LLM 路径。
    if (
        chat_history
        and effective_query == original_query
        and _FOLLOW_UP_RE.search(query_text)
    ):
        ir["compiler"]["reason"] = "多轮省略语义无法安全闭合，保留 LLM 路径"
        return finalize(ir)

    # 路径 1：注册指标规则
    metric = _direct_metric(query_text, list(metric_infos or []))
    if metric:
        rule = dict(metric.get("formula_rule") or {})
        rule_type = str(rule.get("type") or "").lower()
        ir["metric"] = {"name": metric.get("name"), "rule_type": rule_type}
        if rule_type == "ranking":
            dimension = _split_ref(rule.get("dimension"))
            measure = _split_ref(rule.get("column"))
            if dimension and measure:
                # “大客户中谁的订单数量最多”显式改变了排名度量，不能继续按
                # 大客户默认的 SUM(order_amount) 回答。订单计数使用事实表主键，
                # 并清除 metric 绑定，避免后续公式守卫再次强制金额口径。
                order_count_override = bool(_ORDER_COUNT_RE.search(query_text))
                order_count_applied = False
                if order_count_override:
                    # 找到事实表的 order_id 列，把度量改写为 COUNT(order_id)
                    order_column = next(
                        (
                            column for column in columns
                            if str(column.get("table")) == measure[0]
                            and str(column.get("name")) == "order_id"
                        ),
                        None,
                    )
                    if order_column:
                        measure = (measure[0], "order_id")
                        # 清除 metric 绑定，避免后续公式守卫再次强制金额口径
                        ir["metric"] = None
                        order_count_applied = True
                ir["dimensions"] = [{"table": dimension[0], "column": dimension[1], "source": "metric_rule"}]
                ir["measures"] = [{
                    "table": measure[0], "column": measure[1],
                    # 订单计数用 COUNT，否则用规则指定的函数（默认 SUM）
                    "function": "COUNT" if order_count_applied else str(rule.get("function") or "SUM").upper(),
                    "alias": "order_count" if order_count_applied else str(rule.get("result_alias") or _result_alias(measure[1], "SUM")),
                    "source": "explicit_query_override" if order_count_applied else "metric_rule",
                }]
                ir["intent"] = "ranking"
                # “谁/最多/最高/第一”时 limit 默认 1，否则用规则默认值（默认 10）
                default_limit = 1 if order_count_applied and any(token in query_text for token in ("谁", "最多", "最高", "第一")) else int(rule.get("default_limit") or 10)
                ir["limit"] = _extract_limit(query_text, default_limit)
                ir["order_by"] = [{
                    "target": ir["measures"][0]["alias"],
                    "direction": str(rule.get("direction") or "DESC").upper(),
                }]
        elif rule_type == "repeat_rate":
            entity = _split_ref(rule.get("entity_column"))
            event = _split_ref(rule.get("event_column"))
            # 复购率要求实体列与事件列同属一张源表
            source_table = str(rule.get("source_table") or (entity or ("", ""))[0])
            if entity and event and source_table == entity[0] == event[0]:
                ir["intent"] = "repeat_rate"
                ir["measures"] = [{
                    "table": source_table,
                    "entity_column": entity[1],
                    "event_column": event[1],
                    "repeat_threshold": int(rule.get("repeat_threshold") or 2),
                    "alias": str(rule.get("result_alias") or "repurchase_rate"),
                    "source": "metric_rule",
                }]

    # 路径 2：显式元数据词命中（仅在指标路径未确定 intent 时走）
    if ir["intent"] == "unknown":
        # 含过滤/时间/比例/比较等复杂语义 -> 回退 LLM
        if _UNSUPPORTED_SIMPLE.search(query_text):
            ir["compiler"]["reason"] = "包含过滤、时间、比例或比较语义，保留 LLM 路径"
            return finalize(ir)
        # 精确元数据词命中的维度：dim_ 表 + role=dimension + 词出现在 query
        dimensions = [
            column for column in columns
            if _is_dimension(column)
            and any(len(term) >= 2 and _term_in_query(term, query_text) for term in _column_terms(column))
        ]
        # 精确元数据词命中的度量
        measures = [
            column for column in columns
            if _is_measure(column)
            and any(len(term) >= 2 and _term_in_query(term, query_text) for term in _column_terms(column))
        ]
        # Exact metadata terms must identify one dimension and one measure. This
        # deliberately excludes fuzzy vector-only candidates.
        # 按 (table, name) 去重，避免同表同列多次命中导致计数偏大
        unique_dimensions = {(str(item["table"]), str(item["name"])): item for item in dimensions}
        unique_measures = {(str(item["table"]), str(item["name"])): item for item in measures}
        # 交易动作与件数单位可能不逐字命中元数据的数量别名。
        # 仅在没有显式 measure 且事实表中恰有一个
        # order_quantity 时补齐，避免把订单笔数 COUNT(order_id) 当成商品件数。
        quantity_cue = transaction_item_quantity_requested(query_text, tables)
        if not unique_measures and quantity_cue:
            quantity_columns = {
                (str(item["table"]), str(item["name"])): item
                for item in columns
                if _is_measure(item) and str(item.get("name") or "").lower() == "order_quantity"
            }
            # 恰好一个 order_quantity 才补齐，避免多义
            if len(quantity_columns) == 1:
                unique_measures = quantity_columns
        # 恰好 1 维度 + 1 度量 + 分组/排名 cue 才编译
        if len(unique_dimensions) == 1 and len(unique_measures) == 1 and any(
            token in query_text for token in ("各", "每个", "每种", "按", "总额", "总和", "排名")
        ):
            dimension = next(iter(unique_dimensions.values()))
            measure = next(iter(unique_measures.values()))
            function = "SUM"
            # 含“排名/前/后”为排名意图，否则为聚合意图
            ir["intent"] = "ranking" if any(token in query_text for token in ("排名", "前", "后")) else "aggregate"
            ir["dimensions"] = [{
                "table": dimension["table"], "column": dimension["name"], "source": "exact_metadata_term"
            }]
            ir["measures"] = [{
                "table": measure["table"], "column": measure["name"], "function": function,
                "alias": _result_alias(str(measure["name"]), function),
                # 数量 cue 补齐的度量标记为 explicit_quantity_cue，否则为精确元数据命中
                "source": "explicit_quantity_cue" if quantity_cue and not measures else "exact_metadata_term",
            }]
            ir["order_by"] = [{"target": ir["measures"][0]["alias"], "direction": "DESC"}]
            ir["limit"] = _extract_limit(query_text)

    # 校验：IR 所需表是否已全部召回
    required_tables = {
        str(item.get("table"))
        for item in [*ir["dimensions"], *ir["measures"]]
        if item.get("table")
    }
    available = {str(table.get("name") or "") for table in tables}
    # 所需表未完整召回 -> 回退 LLM，避免基于缺失 schema 编译
    if not required_tables or not required_tables.issubset(available):
        ir["compiler"]["reason"] = f"IR 所需表未完整召回：{sorted(required_tables - available)}"
        return finalize(ir)
    # 校验：两表间是否存在唯一 PK/FK 连接路径
    joins = _find_joins(required_tables, columns)
    if joins is None:
        ir["compiler"]["reason"] = "所需表之间没有唯一 PK/FK 连接路径"
        return finalize(ir)
    ir["joins"] = joins
    # 意图必须在渐进式编译白名单内才支持
    if ir["intent"] not in {"aggregate", "ranking", "repeat_rate"}:
        ir["compiler"]["reason"] = "当前意图不在渐进式编译白名单"
        return finalize(ir)

    # 结构完整且连接唯一：翻转 compiler 状态为支持
    ir["status"] = "complete"
    # 命中注册指标置信度 1.0，显式元数据词命中 0.96
    ir["confidence"] = 1.0 if metric else 0.96
    ir["compiler"] = {"supported": True, "mode": "deterministic_ast", "reason": "结构完整且连接唯一"}
    return finalize(ir)


def _table_aliases(table_names: set[str]) -> dict[str, str]:
    """为涉及的表分配稳定的短别名，优先用预置别名（fact_order->f 等），冲突时回退 tN。

    核心实现思路：预置别名提升 SQL 可读性，已用别名集合保证不冲突，回退别名按
    排序后的表序号生成 t1/t2，保证多次编译别名稳定一致。
    """
    # 预置别名：事实表 f，维度表 r/c/p/d，提升生成 SQL 可读性
    preferred = {"fact_order": "f", "dim_region": "r", "dim_customer": "c", "dim_product": "p", "dim_date": "d"}
    aliases: dict[str, str] = {}
    used: set[str] = set()
    # 按 sorted 顺序分配，保证多次编译别名稳定
    for index, table in enumerate(sorted(table_names)):
        candidate = preferred.get(table, f"t{index + 1}")
        # 冲突时回退到 tN
        while candidate in used:
            candidate = f"t{index + 1}"
        aliases[table] = candidate
        used.add(candidate)
    return aliases


def _aggregate(function: str, column: exp.Column) -> exp.Expression:
    """把函数名映射为 SQLGlot 聚合 AST 节点（SUM/AVG/COUNT/MAX/MIN）。"""
    functions = {"SUM": exp.Sum, "AVG": exp.Avg, "COUNT": exp.Count, "MAX": exp.Max, "MIN": exp.Min}
    return functions[function](this=column)


def compile_semantic_ir(ir: Mapping | None) -> str | None:
    """把一份结构完整且受支持的 IR 用 SQLGlot AST 节点编译为 MySQL SQL。

    核心实现思路：按 intent 分支编译——repeat_rate 用 CTE 先聚合事件计数再算
    复购比例；aggregate/ranking 用标准 SELECT+JOIN+GROUP BY+ORDER BY+LIMIT。
    全程用 AST 节点构造，避免字符串拼接注入风险。

    业务逻辑：
        1. IR 不支持编译时返回 None（交由 LLM）
        2. repeat_rate：CTE 按 entity 分组 COUNT(event) -> CASE 标记复购 ->
           外层 SUM(复购标记)/COUNT(*) 得复购率
        3. aggregate/ranking：选维度列 + 聚合度量列 -> JOIN 连接表 ->
           GROUP BY 维度 -> ORDER BY 度量 -> LIMIT
        4. 返回 tree.sql(dialect="mysql")
    """
    # v2 semantic families use their own SQLGlot compiler and structural
    # contract.  Legacy IRs continue through the original progressive branch.
    compiled_v2 = compile_semantic_ir_v2(ir)
    if compiled_v2:
        return compiled_v2

    # 不支持的 IR 直接返回 None，交由 generate_sql 走 LLM
    if not ir or not (ir.get("compiler") or {}).get("supported"):
        return None
    # 复购率：需要 CTE 两层聚合
    if ir.get("intent") == "repeat_rate":
        measure = ir["measures"][0]
        threshold = int(measure["repeat_threshold"])
        count_alias = "order_count"
        # 内层 CTE：按 entity 分组统计每个实体的事件数
        cte = (
            exp.select(
                exp.column(measure["entity_column"]),
                exp.Count(this=exp.column(measure["event_column"])).as_(count_alias),
            )
            .from_(measure["table"])
            .group_by(measure["entity_column"])
        )
        # 复购标记：事件数 >= 阈值 则 1，否则 0
        repeated = exp.Case(
            ifs=[exp.If(
                this=exp.GTE(this=exp.column(count_alias), expression=exp.Literal.number(threshold)),
                true=exp.Literal.number(1),
            )],
            default=exp.Literal.number(0),
        )
        # 外层：SUM(复购标记) / NULLIF(COUNT(*), 0) 得复购率，避免除零
        ratio = exp.Div(
            this=exp.Sum(this=repeated),
            expression=exp.Nullif(
                this=exp.Count(this=exp.Star()), expression=exp.Literal.number(0)
            ),
        ).as_(measure["alias"])
        # 用 CTE customer_orders 包装内层聚合
        tree = exp.select(ratio).from_("customer_orders").with_("customer_orders", as_=cte)
        return tree.sql(dialect="mysql")

    # aggregate / ranking 分支
    dimensions = list(ir.get("dimensions") or [])
    measures = list(ir.get("measures") or [])
    # 收集所有涉及的表名
    table_names = {
        str(item["table"]) for item in [*dimensions, *measures]
    }
    # 分配表别名
    aliases = _table_aliases(table_names)
    # 基表优先选事实表，否则取排序后的首张表，保证 FROM 稳定
    base_table = next((table for table in table_names if table.startswith("fact_")), sorted(table_names)[0])
    base = exp.Table(
        this=exp.to_identifier(base_table),
        alias=exp.TableAlias(this=exp.to_identifier(aliases[base_table])),
    )
    # 投影列：维度列 + 聚合度量列（带别名）
    projections: list[exp.Expression] = [
        exp.column(item["column"], table=aliases[item["table"]]) for item in dimensions
    ]
    projections.extend(
        _aggregate(
            str(item["function"]).upper(),
            exp.column(item["column"], table=aliases[item["table"]]),
        ).as_(item["alias"])
        for item in measures
    )
    tree = exp.select(*projections).from_(base)
    # 依次 JOIN 所需表：已连接集合 joined 决定哪一侧作为 source/target
    joined = {base_table}
    for join in ir.get("joins") or []:
        left_table = join["left_table"]
        right_table = join["right_table"]
        # target 是尚未连接的那一侧，source 是已连接的一侧
        target = right_table if left_table in joined else left_table
        source = left_table if target == right_table else right_table
        source_column = join["left_column"] if source == left_table else join["right_column"]
        target_column = join["right_column"] if target == right_table else join["left_column"]
        tree = tree.join(
            exp.Table(
                this=exp.to_identifier(target),
                alias=exp.TableAlias(this=exp.to_identifier(aliases[target])),
            ),
            on=exp.EQ(
                this=exp.column(source_column, table=aliases[source]),
                expression=exp.column(target_column, table=aliases[target]),
            ),
        )
        joined.add(target)
    # 有维度则 GROUP BY 维度列
    if dimensions:
        tree = tree.group_by(*[
            exp.column(item["column"], table=aliases[item["table"]]) for item in dimensions
        ])
    # ORDER BY：DESC/ASC 按 IR 指定
    for order in ir.get("order_by") or []:
        expression = exp.column(order["target"])
        tree = tree.order_by(expression.desc() if order.get("direction") == "DESC" else expression.asc())
    # LIMIT
    if ir.get("limit"):
        tree = tree.limit(int(ir["limit"]))
    return tree.sql(dialect="mysql")


def restrict_table_infos_to_ir(table_infos: Sequence[Mapping], ir: Mapping) -> list[dict]:
    """只保留 IR 业务字段及其唯一连接键，供 prompt/守卫使用。

    核心实现思路：当 IR 已确定性编译时，筛表可跳过 LLM，直接按 IR 的维度/度量/连接键
    裁剪 table_infos，既缩小 prompt 规模，又保证 prompt 与编译出的 SQL 字段一致，
    避免模型看到多余字段而引入干扰。

    业务逻辑：
        1. 从 IR 的 dimensions/measures 收集每张表所需的业务字段
           （column/entity_column/event_column）
        2. 从 IR 的 joins 收集每张表的连接键（left_column/right_column）
        3. 遍历 table_infos：只保留在 required 中的表，且只保留 required 字段
        4. 字段全被删的表不保留，返回裁剪后的列表
    """
    required: dict[str, set[str]] = {
        str(table): {str(column) for column in columns}
        for table, columns in (ir.get("required_columns") or {}).items()
    }
    # 收集维度/度量业务字段：兼容 ranking 的 column 与 repeat_rate 的 entity/event_column
    for item in [*(ir.get("dimensions") or []), *(ir.get("measures") or [])]:
        table = item.get("table")
        if not table:
            continue
        required.setdefault(str(table), set()).update(
            str(item[key]) for key in ("column", "entity_column", "event_column") if item.get(key)
        )
    # 收集连接键：PK/FK 列也必须保留，否则 JOIN 缺字段
    for join in ir.get("joins") or []:
        required.setdefault(join["left_table"], set()).add(join["left_column"])
        required.setdefault(join["right_table"], set()).add(join["right_column"])
    output: list[dict] = []
    for table in table_infos:
        name = str(table.get("name") or "")
        if name not in required:
            continue
        item = copy.deepcopy(dict(table))
        # 只保留 required 字段
        item["columns"] = [
            column for column in (item.get("columns") or [])
            if str(column.get("name")) in required[name]
        ]
        # 字段全被删的表不保留，避免空表进入下游
        if item["columns"]:
            output.append(item)
    return output
