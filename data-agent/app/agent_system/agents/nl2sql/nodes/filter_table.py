import copy
import logging
import re

import sqlglot
from sqlglot import exp
import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent_system.agents.nl2sql.context import DataAgentContext
from app.agent_system.agents.nl2sql.relation_graph import close_relation_requirements
from app.agent_system.agents.nl2sql.llm import get_task_llm, task_timeout
from app.agent_system.agents.nl2sql.state import DataAgentState
from app.agent_system.agents.nl2sql.semantic_ir import restrict_table_infos_to_ir
from app.agent_system.agents.nl2sql.services.model_sql_policy import use_model_sql
from app.agent_system.agents.nl2sql.llm import invoke_optional_llm
from app.agent_system.runtime.llm_output import parse_json_response
from app.agent_system.governance.deadline import route_deadline_at
from app.agent_system.agents.nl2sql.prompts.prompt_loader import load_prompt
from app.agent_system.agents.nl2sql.source_context import source_mode

logger = logging.getLogger(__name__)

llm = None


def stabilize_table_selection(query: str, table_infos: list[dict], selection: dict,
                              chat_history: list[dict] | None = None) -> list[dict]:
    """在 LLM 过滤之后，确定性保留高置信语义字段与主外键连接路径。

    核心实现思路：LLM 筛表会选最小业务字段集，但可能误删连接键、展示字段、
    时间粒度与多轮分组维度。本函数用规则把这些“结构性必需”字段确定性补回，
    避免后续 generate_sql 因缺连接键而无法生成 JOIN。

    业务逻辑：
        1. 把模型选择结果规整为 {表名: {字段名集合}}
        2. 判定 query 是否请求实体列表/排名、时间粒度（月/季/年）
        3. 第一遍遍历表：用户显式命名的字段强制保留；实体表的展示字段（_name）保留；
           时间粒度字段（month/quarter/year）按请求保留
        4. 第二遍遍历：实体表中未被用户命名的额外维度字段剔除，避免“五个商品”
           被误加品牌等无关分组维度
        5. 多轮继承：从上一轮 SQL 的 GROUP BY 中提取分组维度并保留，保证省略问题
           的分组维度不丢失
        6. 主外键补全：对已选表按字段名匹配 PK/FK 对，把连接键补回
        7. 按 required 字典裁剪 table_infos，只保留有字段的表
    """
    query_text = (query or "").lower()
    # 规整模型选择结果：{表名: {字段名集合}}，空列表也兼容
    required: dict[str, set[str]] = {
        str(table_name): {str(column) for column in (columns or [])}
        for table_name, columns in (selection or {}).items()
    }

    # 实体列表/排名类问题：命中“每种/排名/前/后”且未要求 id/编号
    entity_request = any(token in query_text for token in (
        "每种", "每个", "哪些", "列表", "最高", "最低", "前", "后", "排名", "超过", "没有"
    )) and not any(token in query_text for token in ("id", "编号", "标识"))
    # 实体类型变体：商品/客户/地区，命中任一即认为涉及该实体
    entity_variants = [
        variants for variants in (
            ("商品", "产品", "product"),
            ("客户", "用户", "customer"),
            ("地区", "区域", "大区", "region"),
        )
        if any(token in query_text for token in variants)
    ]
    # 时间粒度请求：月/季/年趋势类
    month_request = bool(re.search(r"每月|各月|按月|月度.*(?:趋势|变化|分布)", query_text))
    quarter_request = bool(re.search(r"每季|各季|按季|季度.*(?:趋势|变化|分布)", query_text))
    year_request = bool(re.search(r"\d{4}\s*年", query_text))
    # 年粒度仅在同时有月/季粒度时才必需（避免单独“2025年”误补 year 列）
    year_grain_required = year_request and (month_request or quarter_request)

    # Entity ranking/list questions should not silently acquire extra grouping
    # dimensions. Keep the readable display field and dimensions explicitly named
    # by the user; discard model-added attributes such as brand in "五个商品".
    entity_table_names: set[str] = set()
    display_by_table: dict[str, set[str]] = {}

    # 第一遍：保留用户显式命名字段、实体展示字段、时间粒度字段
    for table in table_infos:
        table_name = str(table.get("name") or "")
        table_text = f"{table_name} {table.get('description', '')}".lower()
        # 优先用 is_display 标记的展示字段，否则回退 _name 后缀
        preferred_display = {
            str(column.get("name") or "")
            for column in (table.get("columns") or [])
            if str(column.get("name") or "").lower().endswith("_name")
        }
        for column in table.get("columns") or []:
            column_name = str(column.get("name") or "")
            # 字段的检索词集合：列名 + 描述 + 别名，统一小写
            column_terms = [str(value).strip().lower() for value in (
                column_name, column.get("description", ""), *(column.get("alias") or [])
            ) if value]
            column_text = " ".join(column_terms)
            # A field explicitly named by the user is stronger evidence than the
            # filter model. Preserve it even when that model returns an empty map.
            # 用户显式命名的字段（长度>=2 的检索词出现在 query 中）强制保留
            if any(len(term) >= 2 and term in query_text for term in column_terms):
                required.setdefault(table_name, set()).add(column_name)
            # 实体列表问题：保留实体表的展示字段（_name 或含“名称”）
            if entity_request and any(any(token in table_text for token in variants) for variants in entity_variants):
                entity_table_names.add(table_name)
                if column_name in preferred_display or (not preferred_display and "名称" in column_text):
                    required.setdefault(table_name, set()).add(column_name)
                    display_by_table.setdefault(table_name, set()).add(column_name)
            role = str(column.get("role") or "").lower()
            # 时间粒度字段：按月/季/年请求保留对应维度列
            if role == "dimension" and (
                (month_request and (column_name.lower() == "month" or "月份" in column_text))
                or (quarter_request and (column_name.lower() == "quarter" or "季度" in column_text))
                or (year_grain_required and (column_name.lower() == "year" or "年份" in column_text))
            ):
                required.setdefault(table_name, set()).add(column_name)

    # 第二遍：剔除实体表中未被用户命名的额外维度字段，避免“五个商品”被误加品牌等
    for table in table_infos:
        table_name = str(table.get("name") or "")
        if table_name not in entity_table_names:
            continue
        for column in table.get("columns") or []:
            column_name = str(column.get("name") or "")
            if column_name not in required.get(table_name, set()):
                continue
            # 只处理维度字段，度量字段不受此规则影响
            if str(column.get("role") or "").lower() != "dimension":
                continue
            # 展示字段（_name）不受剔除
            if column_name in display_by_table.get(table_name, set()):
                continue
            # 该维度字段的检索词未出现在 query 中则剔除
            terms = [
                str(value).strip().lower()
                for value in (column_name, column.get("description", ""), *(column.get("alias") or []))
                if value and len(str(value).strip()) >= 2
            ]
            if not any(term in query_text for term in terms):
                required[table_name].discard(column_name)

    # A measure-only follow-up still needs the previous turn's grouping table/columns.
    # 多轮继承：从上一轮 SQL 的 GROUP BY 提取分组维度并保留，保证省略问题的分组不丢失
    history = list(chat_history or [])
    if history:
        previous_sql = str(history[-1].get("sql") or "").strip()
        try:
            tree = sqlglot.parse_one(previous_sql, read="mysql") if previous_sql else None
        except Exception:
            tree = None
        if tree is not None:
            # 表别名 -> 真实表名 映射，用于把 GROUP BY 列的表别名还原为表名
            aliases = {
                table.alias_or_name.lower(): table.name
                for table in tree.find_all(exp.Table)
            }
            previous_select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
            group = previous_select.args.get("group") if previous_select is not None else None
            if group is not None:
                for column in group.find_all(exp.Column):
                    table_name = aliases.get((column.table or "").lower())
                    if table_name:
                        required.setdefault(table_name, set()).add(column.name)

    # The filter model selects business columns, not a relationship graph.  Use
    # the same metadata-owned closure as retrieval so an omitted bridge table
    # or join key cannot disconnect otherwise valid selected dimensions.
    relation_closure = close_relation_requirements(table_infos, required)
    for table_name, column_names in dict(
        relation_closure.get("required_columns") or {}
    ).items():
        required.setdefault(str(table_name), set()).update(
            str(column_name) for column_name in column_names
        )

    selected_names = set(required)
    # Restore matching FK/PK pairs for every selected table pair.
    # 主外键补全：对已选表按字段名匹配 PK/FK 对，把连接键补回，保证能生成 JOIN
    keys: dict[str, list[tuple[str, str, dict]]] = {}
    for table in table_infos:
        table_name = str(table.get("name") or "")
        if table_name not in selected_names:
            continue
        for column in table.get("columns") or []:
            role = str(column.get("role") or "").lower()
            # 主键/外键 或 字段名以 _id 结尾的视为连接键候选
            if role in {"primary_key", "foreign_key"} or str(column.get("name") or "").lower().endswith("_id"):
                keys.setdefault(str(column.get("name") or "").lower(), []).append((table_name, role, column))
    # 同名键在多张表中出现且至少有一个是主键，则把该键在所有相关表中补回
    for candidates in keys.values():
        roles = {role for _, role, _ in candidates}
        if "primary_key" in roles and len(candidates) > 1:
            for table_name, _, column in candidates:
                required.setdefault(table_name, set()).add(str(column.get("name")))

    # 按 required 裁剪 table_infos：只保留有字段的表，并删减表内未选字段
    filtered: list[dict] = []
    for table in table_infos:
        table_name = str(table.get("name") or "")
        if table_name not in required:
            continue
        item = copy.deepcopy(table)
        item["columns"] = [
            column for column in item.get("columns") or []
            if str(column.get("name")) in required[table_name]
        ]
        # 字段全被删的表也不保留，避免空表进入下游
        if item["columns"]:
            filtered.append(item)
    return filtered


# 利用大模型对table_infos进行过滤生成：{表名：[字段名1， 字段名2]}
async def filter_table(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    settings = runtime.context["settings"]
    """
    利用大模型过滤召回的表与字段，只保留与提问真正相关的部分。

    核心功能：将用户提问与合并后的表信息（表+字段）交给大模型判断，让其
             输出"需要的表名 → 需要的字段名列表"的字典；再据此对
             table_infos 做原地删减——既删表也删表内字段，以缩小后续
             生成 SQL 的上下文规模、降低干扰。

    业务逻辑：
        1. 向前端推送当前流水线阶段"过滤表"
        2. 从 state 取出 query 与 table_infos
        3. 构造 prompt 链（模板 → llm → JsonOutputParser），将 table_infos
           用 yaml 序列化后连同 query 一起送入大模型
        4. 大模型返回形如 {表名: [字段名...]} 的字典 result
        5. 遍历 table_infos：不在 result 中的表整张删除，在 result 中的表
           再逐字段删除不在 result[表名] 列表中的字段
        6. 返回 {"table_infos": 过滤后的表信息列表} 更新到 state

    Args:
        state: 流水线状态，包含 query 与 table_infos
        runtime: LangGraph 运行时，携带上下文与流式写入器

    Returns:
        dict: {"table_infos": list[TableInfoState]} —— 过滤后的表信息列表

    Raises:
        Exception: 过滤过程中任意异常都向上抛出，便于调试定位
    """
    runtime.stream_writer({"stage": "过滤表"})
    try:
        query = state["query"]
        table_infos = state["table_infos"]
        upload_tables = [
            copy.deepcopy(table) for table in table_infos
            if table.get("source_kind") == "managed_upload"
        ]
        if source_mode(state) in {"upload_only", "multi_upload"}:
            # 上传-only 不调用筛表模型，也不能让 DW 语义规则删除上传列。
            # 受控上限最多两表/百列，完整保留比误删连接键更安全。
            logger.info(
                "上传数据源确定性筛表完成 mode=%s count=%s",
                source_mode(state),
                len(upload_tables),
            )
            return {"table_infos": upload_tables}

        semantic_ir = state.get("semantic_ir", {})
        if (
            semantic_ir.get("compiler", {}).get("supported")
            and use_model_sql(semantic_ir, query_settings=settings.query)
        ):
            # Complex SQL generation needs the bounded recalled context, not a
            # rule compiler's narrower table projection.  The SQL model still
            # cannot use anything outside these authorized recalled tables.
            logger.info(
                "复杂语义模型路径保留受控召回表 count=%s", len(table_infos)
            )
            return {"table_infos": table_infos}
        if semantic_ir.get("compiler", {}).get("supported"):
            table_infos = restrict_table_infos_to_ir(table_infos, semantic_ir)
            logger.info("高置信语义 IR 已跳过筛表 LLM count=%s", len(table_infos))
            return {"table_infos": table_infos}

        if not settings.query.selection_llm_enabled:
            deterministic_selection = {
                str(table.get("name") or ""): [
                    str(column.get("name") or "")
                    for column in (table.get("columns") or []) if column.get("name")
                ]
                for table in table_infos if table.get("name")
            }
            table_infos = stabilize_table_selection(
                query, table_infos, deterministic_selection, state.get("chat_history", [])
            )
            logger.info("确定性筛表完成 count=%s", len(table_infos))
            return {"table_infos": table_infos}

        # 1. 调用模型，过滤掉不需要表和字段，生成需要的表名和字段名的列表的字典
        prompt_template = PromptTemplate(
            template=load_prompt("filter_table_info"),
            input_variables=["query", "table_infos"],
        )
        output_parser = StrOutputParser()
        chain = (
            prompt_template
            | (llm or get_task_llm(runtime, "selection"))
            | output_parser
        )
        # table_infos 是一个 list[TableInfoState]，每个元素是嵌套了 columns 的字典。
        # 不能直接把 Python 对象塞进模板，必须先序列化成文本，这里选了 yaml 而不是 json
        invocation = await invoke_optional_llm(
            chain,
            {
                "query": query,
                "table_infos": yaml.dump(
                    [{
                        "name": table.get("name"),
                        "description": table.get("description"),
                        "columns": [{
                            key: column.get(key) for key in ("name", "description", "role", "alias", "is_display")
                            if column.get(key) not in (None, [], "")
                        } for column in (table.get("columns") or [])],
                    } for table in table_infos],
                    allow_unicode=True,
                    sort_keys=False,
                )
            },
            node_id="filter_table",
            route="selection",
            fallback="",
            timeout_seconds=task_timeout(runtime, "selection"),
            core_path=True,
            runtime=runtime,
            request_deadline_at=route_deadline_at(
                state,
                "selection",
                timeout_seconds=task_timeout(runtime, "selection"),
            ),
            recovery_enabled=False,
        )
        degradation = None
        if invocation.degradations:
            # 筛选是可降级步骤：保留已召回候选，再交给语义守卫和 SQL 校验。
            # 不记录原始模型文本，避免 Prompt/实体值进入日志。
            result = {
                str(table.get("name") or ""): [
                    str(column.get("name") or "") for column in (table.get("columns") or [])
                    if column.get("name")
                ]
                for table in table_infos if table.get("name")
            }
        else:
            try:
                result = parse_json_response(invocation.value, dict)
            except Exception as exc:
                result = {
                    str(table.get("name") or ""): [
                        str(column.get("name") or "") for column in (table.get("columns") or [])
                        if column.get("name")
                    ]
                    for table in table_infos if table.get("name")
                }
                degradation = {
                    "node_id": "filter_table",
                    "reason": f"structured_output_fallback:{type(exc).__name__}",
                }
                runtime.stream_writer({"degradation": degradation})
        # {表名1：[字段名1， 字段名2]}

        # 2. 应用模型选择，同时确定性保留连接键、展示字段、时间轴和多轮分组。
        table_infos = stabilize_table_selection(
            query, table_infos, result, state.get("chat_history", [])
        )
        selected_names = {
            str(item.get("name") or "").lower() for item in table_infos
        }
        # 比较/关联模式下上传 Schema 是显式选择的必需来源，筛表模型不能静默
        # 删除；正式 DW 表仍由 RAG + 筛表最小化。
        table_infos.extend(
            table for table in upload_tables
            if str(table.get("name") or "").lower() not in selected_names
        )

        logger.info(f"过滤表完成 count={len(table_infos)}")

        return {
            "table_infos": table_infos,
            "llm_attempts": invocation.attempts,
            **({
                "degradations": [
                    *invocation.degradations,
                    *([degradation] if degradation else []),
                ]
            } if invocation.degradations or degradation else {}),
        }
    except Exception as e:
        logger.error(f"过滤表信息失败：{str(e)}")
        raise  #抛出全部的错误  方便调试
        # raise e  # 只是抛出当前的
