import logging
import re
from collections.abc import Mapping

import sqlglot
from sqlglot import exp

from langgraph.runtime import Runtime

from app.agent_system.agents.nl2sql.context import DataAgentContext
from app.agent_system.agents.nl2sql.relation_graph import (
    close_relation_requirements,
    compact_relation_closure,
)
from app.agent_system.agents.nl2sql.semantic_ir_v2 import (
    semantic_metadata_requirements,
    transaction_item_quantity_requested,
)
from app.agent_system.agents.nl2sql.state import DataAgentState, MetricInfoState, ColumnInfoState, TableInfoState
from app.agent_system.agents.nl2sql.source_context import requires_dw_retrieval
from app.agent_system.agents.nl2sql.context import ColumnMetadata, TableMetadata
from app.agent_system.agents.nl2sql.contracts.retrieval import (
    ColumnCandidate,
    MetricCandidate,
    ValueCandidate,
)

logger = logging.getLogger(__name__)


def _metadata_term_in_query(term: str, query: str) -> bool:
    """判断元数据术语是否出现在 query 中，兼容“销售额”->“销售总额”一类扩展写法。"""
    normalized = str(term or "").strip().lower()
    return bool(normalized) and (
        normalized in query
        # “销售额”也匹配“销售总额”：去掉末尾“额”后补“总额”再判断
        or (normalized.endswith("额") and f"{normalized[:-1]}总额" in query)
    )


def _history_group_columns(chat_history: list[dict] | None) -> set[tuple[str, str]]:
    """从历史 SQL 的 GROUP BY 中提取 (表名, 分组列) 对，供省略问题继承分组维度。

    核心实现思路：遍历历史轮次，解析每条 SQL 的 GROUP BY，把列的表别名还原为
    真实表名，返回 (table_name, column_name) 集合，供补召回时保留对应分组列。
    """
    required: set[tuple[str, str]] = set()
    for entry in chat_history or []:
        previous_sql = str(entry.get("sql") or "") if isinstance(entry, Mapping) else ""
        if not previous_sql:
            continue
        try:
            tree = sqlglot.parse_one(previous_sql, read="mysql")
        except Exception:
            # 历史 SQL 解析失败跳过该轮，不影响其他轮
            continue
        # 表别名 -> 真实表名（小写），用于还原 GROUP BY 列的归属表
        aliases = {table.alias_or_name.lower(): table.name.lower() for table in tree.find_all(exp.Table)}
        select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
        group = select.args.get("group") if select is not None else None
        if group is None:
            continue
        for column in group.find_all(exp.Column):
            table_name = aliases.get((column.table or "").lower())
            if table_name:
                required.add((table_name, column.name.lower()))
    return required


def _metadata_value(item, name: str, default=None):
    """Read one field from either an ORM row or a mapping fixture."""

    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _plain_table_infos(table_rows) -> list[dict]:
    """Project authoritative Meta rows into the compiler's read-only shape.

    The projection exists only long enough to derive a bounded column closure;
    it is never written to the prompt or returned as the selected schema.
    """

    return [
        {
            "name": str(_metadata_value(table, "name", "") or ""),
            "role": str(_metadata_value(table, "role", "") or ""),
            "description": str(
                _metadata_value(table, "description", "") or ""
            ),
            "columns": [
                {
                    "name": str(_metadata_value(column, "name", "") or ""),
                    "type": str(_metadata_value(column, "type", "") or ""),
                    "role": str(_metadata_value(column, "role", "") or ""),
                    "description": str(
                        _metadata_value(column, "description", "") or ""
                    ),
                    "alias": list(_metadata_value(column, "alias", []) or []),
                    "examples": list(
                        _metadata_value(column, "examples", []) or []
                    ),
                    "is_display": bool(
                        _metadata_value(column, "is_display", False)
                    ),
                }
                for column in columns
            ],
        }
        for table, columns in table_rows
        if _metadata_value(table, "name", "")
    ]


def _maximal_explicit_metadata_terms(query: str, table_rows) -> set[str]:
    """Return matching metadata terms after suppressing broader substrings.

    A short entity alias remains useful for a standalone entity request, but it
    must not compete with a more specific governed term in the same utterance
    (for example an entity word that is the prefix of an attribute name).
    """

    matched: set[str] = set()
    query_text = str(query or "").lower()
    for _table, columns in table_rows:
        for column in columns:
            for raw_term in (
                _metadata_value(column, "name", ""),
                _metadata_value(column, "description", ""),
                *(_metadata_value(column, "alias", []) or []),
            ):
                term = str(raw_term or "").strip().lower()
                if len(term) >= 2 and _metadata_term_in_query(term, query_text):
                    matched.add(term)
    return {
        term
        for term in matched
        if not any(
            term != other and len(other) > len(term) and term in other
            for other in matched
        )
    }


def select_semantic_metadata_columns(
    query: str,
    table_rows,
    chat_history: list[dict] | None = None,
) -> list[ColumnMetadata]:
    """在筛表前从当前工作空间元数据中选择高置信语义所需的字段。

    核心实现思路：根据 query 与历史分组维度，从 table_rows 中确定性挑选
    “用户显式命名的维度/度量 + 实体展示字段 + 时间粒度字段 + 历史分组列 + 数量度量”，
    供 merge_retrieved_info 补召回，弥补向量召回只命中事实表 id 的不足。

    业务逻辑：
        1. 计算 query 的月/季/年/数量/实体请求标志与历史分组列
        2. 遍历每张表的每个字段，判定是否命中以下任一保留条件：
           - explicit：受管字段的名/描述/别名出现在 query 中
           - display：实体表的可读展示字段（is_display 或 _name 或含“名称”）
           - time_grain：维度且匹配月/季/年请求
           - history_group：属于上一轮 GROUP BY 的分组列
           - quantity_measure：数量请求下 fact_ 表的 order_quantity 列
        3. 命中任一即加入 selected 列表返回
    """
    query_text = (query or "").lower()
    # v2 semantic families declare a bounded field closure.  Only those
    # required columns are added to retrieval; complete Meta never leaks into
    # the prompt.  The final access-policy filter remains authoritative.
    plain_table_infos = _plain_table_infos(table_rows)
    contract_requirements = semantic_metadata_requirements(
        query,
        chat_history,
        plain_table_infos,
    )
    # Required business fields are not sufficient if their owning tables do
    # not form a connected graph.  Close the unique live-metadata path here so
    # bridge tables and both join keys enter retrieval before any model sees
    # the schema.  Ambiguous paths remain unresolved and are never guessed.
    relation_closure = close_relation_requirements(
        plain_table_infos,
        contract_requirements,
    )
    contract_requirements = {
        str(table_name): set(str(column) for column in column_names)
        for table_name, column_names in dict(
            relation_closure.get("required_columns") or {}
        ).items()
    }
    explicit_metadata_terms = _maximal_explicit_metadata_terms(query, table_rows)
    month_request = bool(re.search(r"每月|各月|按月|月度.*(?:趋势|变化|分布)", query_text))
    quarter_request = bool(re.search(r"每季|各季|按季|季度.*(?:趋势|变化|分布)", query_text))
    year_request = bool(re.search(r"\d{4}\s*年", query_text))
    # 年粒度：单独“2025年”不强制补 year 列，避免误补
    year_grain_required = year_request
    # 交易件数请求与 IR/coverage 共用同一受管判定，避免各阶段漂移。
    quantity_request = transaction_item_quantity_requested(
        query,
        plain_table_infos,
    )
    # 实体列表/排名请求（排除 id/编号类）
    entity_request = any(token in query_text for token in (
        "每种", "每个", "哪些", "列表", "最高", "最低", "前", "后", "排名", "超过", "没有"
    )) and not any(token in query_text for token in ("id", "编号", "标识"))
    entity_variants = [
        variants for variants in (
            ("商品", "产品", "product"),
            ("客户", "用户", "customer"),
            ("地区", "区域", "大区", "region"),
        )
        if any(token in query_text for token in variants)
    ]
    # 历史分组列：用于省略问题继承上一轮分组维度
    history_groups = _history_group_columns(chat_history)
    selected: list[ColumnMetadata] = []
    for table, columns in table_rows:
        table_text = f"{getattr(table, 'name', '')} {getattr(table, 'description', '')}".lower()
        # 当前表是否为涉及的实体表
        entity_table = entity_request and any(
            any(token in table_text for token in variants) for variants in entity_variants
        )
        # 优先用 is_display 标记的展示字段
        explicit_display = {
            str(getattr(column, "name", "") or "")
            for column in columns if bool(getattr(column, "is_display", False))
        }
        # 回退：无 is_display 时用 _name 后缀
        preferred_display = explicit_display or {
            str(getattr(column, "name", "") or "")
            for column in columns
            if str(getattr(column, "name", "") or "").lower().endswith("_name")
        }
        for column in columns:
            name = str(getattr(column, "name", "") or "")
            description = str(getattr(column, "description", "") or "")
            aliases = getattr(column, "alias", None) or []
            examples = getattr(column, "examples", None) or []
            # 字段检索文本：列名 + 描述 + 别名，统一小写
            text = " ".join([name, description, *(str(item) for item in aliases)]).lower()
            role = str(getattr(column, "role", "") or "").lower()
            contract_required = name in contract_requirements.get(
                str(getattr(table, "name", "") or ""), set()
            )
            # explicit：任一受管业务字段的名/描述/别名都可补回。
            # 主键也可能承载 COUNT 口径（如“订单数”），不应因
            # role 不是 dimension/measure 而在进入生成上下文前丢失。
            explicit = role in {
                "dimension", "measure", "primary_key", "foreign_key"
            } and any(
                len(term) >= 2
                and term.strip().lower() in explicit_metadata_terms
                for term in [name.lower(), description.lower(), *(str(item).lower() for item in aliases)]
            )
            # Schema-owned enum examples close the common RAG chicken-and-egg:
            # a value such as a category/member level may be present in the
            # utterance even though the column name itself is not.  Only exact
            # example substrings of length >= 2 are admitted; one-character
            # gender values use a bounded linguistic context instead of a
            # global substring match.
            example_value = role == "dimension" and any(
                len(str(item).strip()) >= 2
                and str(item).strip().lower() in query_text
                for item in examples
            )
            gender_value = (
                role == "dimension"
                and name.lower() == "gender"
                and bool(
                    re.search(
                        r"(?:男|女)(?:性|顾客|客户|用户|会员)",
                        query_text,
                    )
                )
            )
            # display：实体表的可读展示字段
            display = entity_table and (
                name in preferred_display or (not preferred_display and "名称" in text)
            )
            # time_grain：维度且匹配月/季/年请求
            time_grain = role == "dimension" and (
                (month_request and (name.lower() == "month" or "月份" in text))
                or (quarter_request and (name.lower() == "quarter" or "季度" in text))
                or (year_grain_required and (name.lower() == "year" or "年份" in text))
            )
            # history_group：属于上一轮 GROUP BY 的分组列
            history_group = (str(getattr(table, "name", "") or "").lower(), name.lower()) in history_groups
            # quantity_measure：数量请求下 fact_ 表的 order_quantity 列
            quantity_measure = (
                quantity_request
                and str(getattr(table, "name", "") or "").lower().startswith("fact_")
                and name.lower() == "order_quantity"
            )
            # 命中任一保留条件即加入 selected
            if (
                contract_required
                or explicit
                or example_value
                or gender_value
                or display
                or time_grain
                or history_group
                or quantity_measure
            ):
                selected.append(column)
    return selected


async def merge_retrieved_info(state:DataAgentState, runtime: Runtime[DataAgentContext]):
    """
    合并字段/指标/字段值三类召回结果，组装成生成 SQL 所需的表与指标结构。

    核心功能：以字段召回结果为基底，把指标关联的相关字段、字段值对应的字段
             补入字段集合，再按表分组补充各表的主键/外键字段，最终组装成
             TableInfoState 列表与 MetricInfoState 列表，作为 filter 与
             generate_sql 节点的输入。期间在 Qdrant / MySQL / State 三种
             字段模型间做相互转换。

    业务逻辑：
        1. 从 state 取出 recall_columns / recall_metrics / recall_values / chat_history，
           从 runtime 取出 meta_mysql_repo（元数据库访问对象）
        2. 以 recall_columns 建立字段字典（key: column_id）
        3. 补充指标关联字段：缺失的 relevant_columns 从元数据库查回并补入
        4. 补充字段值对应字段：缺失的从元数据库查回，并把命中值追加到 examples
        5. 历史/长期记忆只作为当前检索与语义闭合线索，不直接补表；
           所有表字段必须由当前召回、当前 Meta 与当前 Policy 重新确认
        6. 按 table_id 将字段分组，并补充每张表的主键/外键字段
        7. 逐表查询表元信息，组装 TableInfoState 列表（meta库不存在的表跳过）
        8. 将指标召回结果转换为 MetricInfoState 列表
        9. 返回 {"table_infos": ..., "metric_infos": ...} 更新到 state

    Args:
        state: 流水线状态，包含三类召回结果 recall_columns/recall_metrics/recall_values
        runtime: LangGraph 运行时，携带 meta_mysql_repo 等上下文

    Returns:
        dict: {"table_infos": list[TableInfoState],
               "metric_infos": list[MetricInfoState]}
    """
    runtime.stream_writer({"stage": "合并召回"})
    # 取出相关的数据：三类召回结果（字段/指标/字段值）与元数据库访问对象
    recall_columns: list[ColumnCandidate] = state["recall_columns"]
    recall_metrics: list[MetricCandidate] = state["recall_metrics"]
    recall_values: list[ValueCandidate] = state["recall_values"]
    meta_mysql_repo = runtime.context["meta_mysql_repo"]

    # =============================================================================
    # 阶段 1：收集最完整的字段信息集合（以字段召回为基底，逐层补充）
    # =============================================================================

    # 1.1 根据recall_columns来创建：以 column_id 为键建立字段字典，便于后续按 id 去重补入
    column_infos_dict: dict[str, ColumnCandidate] = {item["id"]: item for item in recall_columns}
    # 1.2 补充1：recall_metrics对应的字段
    # 指标会关联若干相关字段，这些字段可能未被字段召回命中，需从元数据库补回
    for metric in recall_metrics:
        for relevant_column_id in metric["relevant_columns"]:
            if relevant_column_id not in column_infos_dict:
                column_info_mysql: ColumnMetadata | None = await meta_mysql_repo.get_colun_info_by_id(relevant_column_id)
                if column_info_mysql is not None:
                    column_infos_dict[relevant_column_id] = _convert_column_info_mysql_to_qdrant(column_info_mysql)
                else:
                    logger.warning("指标相关字段 %s 不在当前 Schema，保留 fallback 路径", relevant_column_id)
    # 1.3. 补充2： 字段值对应的字段  内部还需要将当前字段值添加到字段的examples中
    # 字段值召回命中的字段可能未被字段召回命中，需补回；同时把命中值作为示例追加进 examples，帮助大模型理解字段取值
    for recall_value in recall_values:
        column_id = recall_value["column_id"]
        if column_id not in column_infos_dict:
            column_info_mysql: ColumnMetadata | None = await meta_mysql_repo.get_colun_info_by_id(column_id)
            if column_info_mysql is None:
                logger.warning("字段值召回的字段 %s 已不在当前 Schema，跳过", column_id)
                continue
            column_infos_dict[column_id] = _convert_column_info_mysql_to_qdrant(column_info_mysql)
        column_infos_dict[column_id].setdefault("examples", [])
        # 仅当命中值尚未在 examples 中时追加，避免重复
        if recall_value["value"] not in column_infos_dict[column_id]["examples"]:
            column_infos_dict[column_id]["examples"].append(recall_value["value"])
        # Preserve only privacy-safe query-owned match shape. This tells the SQL
        # model whether one user phrase names an exact value or a shared prefix,
        # without exposing any stored value that the user did not provide.
        hints = column_infos_dict[column_id].setdefault("value_match_hints", [])
        seen = {
            (str(item.get("query_term") or ""), str(item.get("operator") or ""))
            for item in hints if isinstance(item, dict)
        }
        for hint in recall_value.get("value_match_hints") or []:
            key = (
                str(hint.get("query_term") or ""),
                str(hint.get("operator") or ""),
            )
            if key not in seen:
                hints.append(dict(hint))
                seen.add(key)
    # 1.4 安全边界：不再把历史 SQL 的表/全部字段直接并入当前上下文。
    # TODO-44 已先把省略问题闭合为 standalone question，当前四路 RAG 必须据此
    # 重新召回。历史与长期记忆不能扩大当前 Schema/Policy 白名单。
    chat_history = state.get("chat_history", [])

    # 高置信语义补召回：向量召回可能只命中事实表 date_id/entity_id。
    # 在筛表前从当前工作空间元数据补入可读时间粒度和实体展示字段；后续 PK/FK
    # 阶段会自动补齐连接键，权限快照仍在最终 table_infos 上统一过滤。
    current_query = state.get("query", "")
    meta_table_rows = None
    if requires_dw_retrieval(state):
        # Run the bounded selector for every DW question.  The compositional
        # registry has a conservative catch-all and can discover a required
        # field even when the legacy lexical pre-check does not recognize a new
        # semantic family (for example cohort/set operations).  An unrelated
        # question selects nothing, so this does not widen the model context.
        meta_table_rows = await meta_mysql_repo.list_table_infos()
        semantic_columns = select_semantic_metadata_columns(
            current_query, meta_table_rows, chat_history
        )
        for column in semantic_columns:
            if column.id not in column_infos_dict:
                column_infos_dict[column.id] = _convert_column_info_mysql_to_qdrant(column)

        # Close relationships once more over the *actual* union of column,
        # metric and value retrieval.  This catches a bridge that could not be
        # known from lexical semantics alone (for example two dimensions
        # selected by independent metric/value routes).  Only join-key columns
        # are added; unrelated business columns remain outside the prompt.
        table_name_by_id = {
            str(_metadata_value(table, "id", "")): str(
                _metadata_value(table, "name", "")
            )
            for table, _columns in meta_table_rows
        }
        required_columns: dict[str, set[str]] = {}
        for column in column_infos_dict.values():
            table_name = (
                str(column.get("table_name") or "")
                or table_name_by_id.get(str(column.get("table_id") or ""), "")
            )
            if table_name and column.get("name"):
                required_columns.setdefault(table_name, set()).add(
                    str(column["name"])
                )
        relation_closure = close_relation_requirements(
            _plain_table_infos(meta_table_rows),
            required_columns,
        )
        columns_by_ref = {
            (
                str(_metadata_value(table, "name", "")).lower(),
                str(_metadata_value(column, "name", "")).lower(),
            ): column
            for table, columns in meta_table_rows
            for column in columns
        }
        for table_name, column_names in dict(
            relation_closure.get("required_columns") or {}
        ).items():
            for column_name in column_names:
                column = columns_by_ref.get(
                    (str(table_name).lower(), str(column_name).lower())
                )
                if column is not None and column.id not in column_infos_dict:
                    column_infos_dict[column.id] = _convert_column_info_mysql_to_qdrant(
                        column
                    )
        safe_relation_evidence = compact_relation_closure(relation_closure)
        runtime.stream_writer({"relation_closure": safe_relation_evidence})

    # 对字段信息列表按表分组存储：dict[table_id, list[ColumnCandidate]]
    # 便于后续以"表"为单位补充主外键并组装 TableInfoState
    table_column_infos_dict:dict[str, list[ColumnCandidate]] = {}
    # column_infos_dict: dict[str, ColumnCandidate]
    for column_info in column_infos_dict.values():
        if column_info["table_id"] not in table_column_infos_dict:
            table_column_infos_dict[column_info["table_id"]] = []
        table_column_infos_dict[column_info["table_id"]].append(column_info)

    table_infos: list[TableInfoState] = []

    # =============================================================================
    # 阶段 2：逐表补充主外键字段并组装 TableInfoState
    # =============================================================================

    # 补充3：将相关表的主键和外键字段添加到列表中
    # 主外键是表连接的关键，即便未被召回也必须提供给大模型，否则无法生成 JOIN
    for table_id in table_column_infos_dict:
        # 当前表已收集的字段（table_column_infos_dict[table_id] 与下面 column_infos 同一对象）
        column_infos = table_column_infos_dict[table_id]
        # 本表已存在的字段 id 集合，用于去重，避免主外键与已召回字段重复
        existing_ids = {c["id"] for c in column_infos}

        # 去meta库取当前表的主键和外键字段
        pf_column_infos: list[ColumnMetadata] = list(
            await meta_mysql_repo.get_column_infos_by_table_id(table_id)
        )
        for pf_column_info in pf_column_infos:
            # 双重去重：按本表已存在 id 集合判断，避免依赖全局 column_infos_dict 的隐式状态
            if pf_column_info.id not in existing_ids:
                pf_column_qdrant = _convert_column_info_mysql_to_qdrant(pf_column_info)
                # 同时写入本表列表与全局字段字典，保持两者一致
                column_infos.append(pf_column_qdrant)
                column_infos_dict[pf_column_info.id] = pf_column_qdrant
                existing_ids.add(pf_column_info.id)

        # 2. 生成TableInfoState的对象列表
        # 2.1 查询当前表信息：取表名、角色、描述等元数据
        table_info_mysql: TableMetadata | None = await meta_mysql_repo.get_table_info_by_table_id(table_id)
        # 防护：历史SQL提取的表名可能在meta库中不存在（如已下线表），跳过避免 None 访问
        if table_info_mysql is None:
            logger.warning(f"表 {table_id} 在meta库中不存在，跳过组装 TableInfoState")
            continue
        # 2.2 得到当前表的所字段状态信息对象列表：显式从 dict 取最新列表（含刚补入的主外键），
        #     不再依赖循环变量 column_infos 的引用语义，避免后续重构引入遗漏 bug
        columns: list[ColumnInfoState] = [
            _convert_column_info_qdrant_to_state(item)
            for item in table_column_infos_dict[table_id]
        ]
        # 2.3 创建TableInfoState对象, 添加到列表
        table_infos.append(TableInfoState(
            name=table_info_mysql.name,
            role=table_info_mysql.role,
            description=table_info_mysql.description,
            columns=columns
        ))

    # 处理指标列表：将 Qdrant 指标模型转换为 State 指标模型
    metric_infos = [
        _convert_metric_info_qdrant_to_state(item)
        for item in recall_metrics
    ]

    # 上传表从本轮权威数据源上下文直接 overlay；它们从不写入或读取全局
    # Meta/Qdrant/ES，因此不会污染正式数据面的召回集合。
    source_tables = [
        dict(item) for item in (state.get("source_table_infos") or [])
    ]
    existing_names = {
        str(item.get("name") or "").lower() for item in table_infos
    }
    table_infos.extend(
        item for item in source_tables
        if str(item.get("name") or "").lower() not in existing_names
    )

    # 权限过滤发生在 Prompt 组装前，避免模型看到不可见表/列或依赖受限指标。
    access_policy = runtime.context.get("access_policy")
    if access_policy:
        table_infos = access_policy.filter_table_infos(table_infos)
        metric_infos = access_policy.filter_metric_infos(metric_infos)

    logger.info(f"合并表信息完成 count={len(table_infos)}")
    logger.info(f"合并指标信息完成 count={len(metric_infos)}")

    return {
        "table_infos": table_infos,
        "metric_infos": metric_infos,
        **(
            {"relation_closure": safe_relation_evidence}
            if requires_dw_retrieval(state) and meta_table_rows is not None
            else {}
        ),
    }

def _convert_column_info_qdrant_to_state(column:ColumnCandidate)->ColumnInfoState:
    """
    将 Qdrant 字段模型转换为 State 字段模型（剥离 id/table_id 等内部字段）。

    核心功能：从向量库召回的字段对象中提取生成 SQL 所需的业务字段
             （名称、类型、角色、示例、描述、别名），重新封装为流水线
             内部使用的 ColumnInfoState。

    Args:
        column: Qdrant 字段信息对象

    Returns:
        ColumnInfoState: 流水线内部使用的字段状态对象
    """
    return ColumnInfoState(
        name=column['name'],
        type=column['type'],
        role=column["role"],
        examples=column.get("examples", []),
        description=column["description"],
        alias=column["alias"],
        is_display=bool(column.get("is_display", False)),
        value_match_hints=list(column.get("value_match_hints") or []),
    )


def _convert_metric_info_qdrant_to_state(recall_metric:MetricCandidate):
    """
    将 Qdrant 指标模型转换为 State 指标模型。

    核心功能：从向量库召回的指标对象中提取生成 SQL 所需的业务字段
             （名称、描述、关联字段、别名），重新封装为流水线内部使用的
             MetricInfoState。

    Args:
        recall_metric: Qdrant 指标信息对象

    Returns:
        MetricInfoState: 流水线内部使用的指标状态对象
    """
    return MetricInfoState(
        name=recall_metric['name'],
        description=recall_metric["description"],
        relevant_columns=recall_metric["relevant_columns"],
        alias=recall_metric["alias"],
        formula_rule=recall_metric.get("formula_rule"),
    )

def _convert_column_info_mysql_to_qdrant(column_info_mysql:ColumnMetadata)->ColumnCandidate:
    """
    将元数据能力返回的字段视图转换为 Agent 自有候选 DTO。

    核心功能：把从元数据端口查回的字段对象统一为 Agent 字段候选，使其能与
             字段召回结果在同一个字段字典中无差别存储与处理。

    Args:
        column_info_mysql: 元数据端口返回的字段只读视图

    Returns:
        ColumnCandidate: 与向量召回结果同构的 Agent 字段候选
    """
    return ColumnCandidate(
        id=column_info_mysql.id,
        name=column_info_mysql.name,
        description=column_info_mysql.description,
        role=column_info_mysql.role,
        type=column_info_mysql.type,
        examples=list(column_info_mysql.examples or ()),
        table_id=column_info_mysql.table_id,
        alias=list(column_info_mysql.alias or ()),
        is_display=bool(column_info_mysql.is_display),
    )
