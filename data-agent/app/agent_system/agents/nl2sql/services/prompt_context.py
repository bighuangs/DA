"""按任务节点压缩 Prompt 载荷，并产出可审计的 Token 估算证据。

本模块为 generate_sql / plan_sql / generate_recommend / judge_candidate 等
需要调用 LLM 的节点提供统一的 Prompt 载荷压缩入口，把 state 中的
table_infos / metric_infos / recall_training / chat_history / memory_context
按节点需要裁剪、截断、序列化为 YAML 字符串，并估算压缩前后的 Token 数，
写入 context_budget.model_prompts[node_id] 供可观测性展示。

设计要点：
    - 压缩是“无损关键信息、有损冗余”：保留 name/role/type/alias 等结构字段，
      截断长描述与过多 examples，去掉空值，保证 Prompt 既小又足够生成 SQL。
    - Token 估算用 estimate_tokens（中文按 1 字 ≈ 1 token，英文按 4 字符 ≈ 1 token），
      仅作预算与裁剪决策，不追求精确。
    - build_candidate_prompt_payload 是 judge_candidate 专用：只保留主候选 A
      实际用到的表/列与连接键，进一步缩小备选生成的 Prompt 规模。
"""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence

import yaml
from app.agent_system.agents.nl2sql.settings import Nl2SqlSettings
from app.agent_system.agents.nl2sql.services.memory_prompt_context import (
    memory_context_for_prompt,
)
from app.agent_system.agents.nl2sql.services.model_sql_policy import model_semantic_summary
from app.agent_system.governance.retrieval_budget import estimate_tokens


# Prompt 载荷版本号，进入 context_budget 证据，便于发现压缩契约漂移
PROMPT_PAYLOAD_VERSION = "prompt-payload-v1"


def _short(value, limit: int = 120) -> str:
    """把长文本截断到配置上限，超长则截断并加省略号“…”。

    核心实现思路：默认上限取 app_config.query.prompt_description_max_chars，
    截断时去尾空白再加“…”，保证截断后文本整洁且明确表示已截断。
    """
    maximum = max(1, int(limit))
    text = str(value or "").strip()
    return text if len(text) <= maximum else text[:maximum].rstrip() + "…"


def compact_table_infos(
    tables: Sequence[Mapping], *, query: str = "",
    description_max_chars: int = 120,
    example_limit: int = 0,
) -> list[dict]:
    """压缩表信息列表：保留结构字段，截断长描述，限制每列 examples 数量。

    核心实现思路：对每张表只保留 name/description/columns，对每个字段只保留
    name/type/role/is_display/alias/description 等结构字段并去掉空值，描述超长截断，
    examples 按配置上限截断，既缩小 Prompt 又保留生成 SQL 所需的结构信息。

    业务逻辑：
        1. 每张表保留 name + 截断后的 description + columns 列表
        2. 每个字段保留 6 个结构字段（去掉空值），description 截断
        3. examples 优先取 column.examples，其次 example_values，按上限截断
    """
    result = []
    # 每列 examples 数量上限，0 表示不输出 examples
    example_limit = max(0, int(example_limit))
    normalized_query = "".join(str(query or "").lower().split())
    for table in tables or []:
        item = {
            "name": table.get("name"),
            "description": _short(table.get("description"), description_max_chars),
            "columns": [],
        }
        for column in table.get("columns") or []:
            # 只保留 6 个结构字段，去掉 None/空串/空列表/空字典
            compact = {
                key: copy.deepcopy(column.get(key))
                for key in (
                    "name", "type", "role", "is_display", "alias", "description",
                    "value_match_hints",
                )
                if column.get(key) not in (None, "", [], {})
            }
            # description 单独截断
            if "description" in compact:
                compact["description"] = _short(
                    compact["description"], description_max_chars
                )
            # examples 优先取 examples 字段，其次 example_values
            examples = list(column.get("examples") or column.get("example_values") or [])
            # Values already present in the user's own question are safe and
            # semantically essential normalization evidence (for example a
            # colloquial label that maps to a canonical enum value).  Preserve
            # only those exact substrings even when the deployment disables
            # general example exposure.  Unmentioned stored examples remain
            # governed by ``prompt_column_example_limit`` and are never leaked
            # by this exception.
            matched_examples = [
                value for value in examples
                if str(value).strip()
                and "".join(str(value).lower().split()) in normalized_query
            ]
            if matched_examples:
                compact["examples"] = matched_examples[: max(8, example_limit)]
            elif example_limit and examples:
                compact["examples"] = examples[:example_limit]
            item["columns"].append(compact)
        result.append(item)
    return result


def compact_metric_infos(
    metrics: Sequence[Mapping],
    *,
    description_max_chars: int = 120,
) -> list[dict]:
    """压缩指标信息列表：保留 name/alias/description/formula/formula_rule/relevant_columns，description 截断。

    核心实现思路：对每个指标只保留生成 SQL 所需的口径字段，description 单独截断，
    其余字段深拷贝，去掉空值，保证指标口径完整且 Prompt 紧凑。
    """
    return [{
        # description 截断，其余字段深拷贝
        key: (
            _short(metric.get(key), description_max_chars)
            if key == "description"
            else copy.deepcopy(metric.get(key))
        )
        for key in ("name", "alias", "description", "formula", "formula_rule", "relevant_columns")
        if metric.get(key) not in (None, "", [], {})
    } for metric in (metrics or [])]


def compact_training_examples(
    training: Sequence[Mapping],
    *,
    limit: int = 2,
) -> list[dict]:
    """压缩训练样例列表：按上限截断条数，每条只保留 question/query/sql。

    核心实现思路：训练样例作为 few-shot 示例注入 Prompt，条数受
    prompt_training_limit 限制，每条只保留问题与 SQL，去掉 id/score 等内部字段。
    """
    # 训练样例条数上限
    limit = max(0, int(limit))
    return [{
        key: item.get(key) for key in ("question", "query", "sql") if item.get(key)
    } for item in list(training or [])[:limit]]


def compact_history(
    history: Sequence[Mapping],
    *,
    limit: int = 2,
) -> list[dict]:
    """压缩多轮历史：取最近 N 轮，每轮只保留 query/standalone_question/sql/follow_up_action。

    核心实现思路：历史用于让 LLM 感知上下文，取最近 prompt_history_limit 轮，
    每轮只保留问题与 SQL 与闭合动作，去掉结果行等敏感/冗余字段。
    """
    # 历史轮数上限，0 表示不注入历史
    limit = max(0, int(limit))
    # 取最近 limit 轮（旧→新排列，切片末尾）
    rows = list(history or [])[-limit:] if limit else []
    return [{
        key: item.get(key)
        for key in ("query", "standalone_question", "sql", "follow_up_action")
        if item.get(key) not in (None, "", [], {})
    } for item in rows]


def build_model_prompt_payload(
    state: Mapping,
    node_id: str,
    *,
    settings: Nl2SqlSettings | None = None,
) -> tuple[dict, dict]:
    """为指定节点构建压缩后的 YAML 载荷与累计的 Token 预算证据。

    核心功能：把 state 中的五类上下文（表/指标/训练样例/历史/记忆）按节点压缩为
    YAML 字符串，估算压缩前后 Token 数，把证据追加到 context_budget.model_prompts[node_id]，
    供可观测性展示每个 LLM 节点的 Prompt 预算与压缩率。

    业务逻辑：
        1. 取出原始五类上下文（深拷贝避免污染 state）
        2. 分别压缩：表/指标/训练样例/历史/记忆
        3. estimate_tokens 估算压缩前后 Token 数
        4. 组装 evidence（版本/节点/前后 token/压缩率/各类条数）
        5. 把 evidence 追加到 state.context_budget.model_prompts[node_id]
        6. 把压缩后的载荷序列化为 YAML 字符串返回

    Args:
        state: 流水线状态，含 table_infos/metric_infos/recall_training/chat_history/memory_context/context_budget
        node_id: 当前节点名（如 "generate_sql"），作为 model_prompts 的键

    Returns:
        tuple[dict, dict]:
            - payload: 各上下文的 YAML 字符串字典，直接注入 Prompt 模板
            - current_budget: 更新后的 context_budget，写回 state
    """
    # 1. 取出原始五类上下文（list/dict 包一层避免修改原 state）
    original = {
        "table_infos": list(state.get("table_infos") or []),
        "metric_infos": list(state.get("metric_infos") or []),
        "training_examples": list(state.get("recall_training") or []),
        "chat_history": list(state.get("chat_history") or []),
        "memory_context": dict(state.get("memory_context") or {}),
    }
    # 2. 分别压缩：表/指标/训练样例/历史/记忆
    query_settings = settings.query if settings is not None else None
    description_max_chars = int(
        getattr(query_settings, "prompt_description_max_chars", 120)
    )
    example_limit = int(getattr(query_settings, "prompt_column_example_limit", 0))
    compact = {
        "table_infos": compact_table_infos(
            original["table_infos"], query=str(state.get("query") or ""),
            description_max_chars=description_max_chars,
            example_limit=example_limit,
        ),
        "metric_infos": compact_metric_infos(
            original["metric_infos"],
            description_max_chars=description_max_chars,
        ),
        "training_examples": compact_training_examples(
            original["training_examples"],
            limit=int(getattr(query_settings, "prompt_training_limit", 2)),
        ),
        "chat_history": compact_history(
            original["chat_history"],
            limit=int(getattr(query_settings, "prompt_history_limit", 2)),
        ),
        # memory_context_for_prompt：暴露有界、显式不可信的记忆子集给 Prompt
        "memory_context": memory_context_for_prompt(
            original["memory_context"],
            token_budget=int(
                getattr(
                    settings.memory if settings is not None else None,
                    "context_token_budget",
                    700,
                )
            ),
        ),
    }
    # 3. 估算压缩前后 Token 数
    before = estimate_tokens(original)
    after = estimate_tokens(compact)
    # 4. 组装 evidence：版本/节点/前后 token/压缩率/各类条数
    evidence = {
        "version": PROMPT_PAYLOAD_VERSION,
        "node_id": node_id,
        "before_tokens": before,
        "after_tokens": after,
        # 压缩率 = 1 - after/before，before 为 0 时记 0.0 避免除零
        "compression_rate": round(max(0.0, 1 - after / before), 4) if before else 0.0,
        "history_items": len(compact["chat_history"]),
        # 记忆子项条数，供可观测性展示记忆注入规模
        "memory_preferences": len(compact["memory_context"]["confirmed_preferences"]),
        "memory_profile_facts": len(
            compact["memory_context"].get("confirmed_profile_facts") or []
        ),
        "memory_goals": len(compact["memory_context"].get("long_term_goals") or []),
        "memory_summaries": len(
            compact["memory_context"].get("selected_summaries") or []
        ),
        "memory_recent_messages": len(
            compact["memory_context"].get("recent_messages") or []
        ),
        "memory_episodes": len(compact["memory_context"]["relevant_episodes"]),
        "training_items": len(compact["training_examples"]),
        "column_examples_per_field": example_limit,
    }
    # 5. 把 evidence 追加到 state.context_budget.model_prompts[node_id]
    # 深拷贝当前预算，避免修改原 state；按节点累计，便于多节点对比
    current_budget = copy.deepcopy(dict(state.get("context_budget") or {}))
    model_prompts = dict(current_budget.get("model_prompts") or {})
    model_prompts[node_id] = evidence
    current_budget["model_prompts"] = model_prompts
    # 6. 把压缩后的载荷序列化为 YAML 字符串，直接注入 Prompt 模板
    payload = {
        key: yaml.dump(value, allow_unicode=True, sort_keys=False)
        for key, value in compact.items()
    }
    return payload, current_budget


def build_candidate_prompt_payload(
    state: Mapping, weaknesses: Sequence[str], node_id: str = "judge_candidate",
    *, settings: Nl2SqlSettings | None = None,
) -> tuple[dict, dict]:
    """从当前已授权上下文构建独立语义复核 Prompt 载荷。

    核心功能：复核器必须能独立从用户原话重建需求，因此要传入 state 中
    已经经过召回、租户/列权限与 token 装包的全部 table_infos，不得再按
    候选 A 或可能有误的 semantic_ir 二次缩窄表列范围。

    业务逻辑：
        1. 对全部受控 table_infos 仅做字段级安全压缩，不按 A/IR 删表删列
        2. 保留当前受控 metric_infos 的全部口径摘要
        3. 通过统一 authority 视图压缩 semantic_ir；model+guided 路由不向复核器
           暴露同源解析槽，避免“独立复核”被错误 IR 锚定
        4. 估算压缩前后 Token，组装 evidence 追加到 context_budget
        5. 序列化为 YAML 字符串返回

    Args:
        state: 流水线状态，含 sql/table_infos/metric_infos/semantic_ir/db_info/context_budget
        weaknesses: 主候选 A 的可操作弱点列表，注入备选 Prompt 引导修正
        node_id: 默认 "judge_candidate"，作为 model_prompts 的键

    Returns:
        tuple[dict, dict]:
            - payload: 备选 Prompt 的 YAML 字符串字典
            - current_budget: 更新后的 context_budget
    """
    tables = list(state.get("table_infos") or [])
    ir = dict(state.get("semantic_ir") or {})
    # table_infos 在 merge_retrieved_info 已按权限过滤，并在 pack_context
    # 受 token 预算装包。这里仅删减描述噪声，不再以可能有误的 A/IR
    # 当作二次召回器。
    query_settings = settings.query if settings is not None else None
    description_max_chars = int(
        getattr(query_settings, "prompt_description_max_chars", 120)
    )
    compact_tables = compact_table_infos(
        tables,
        query=str(state.get("query") or ""),
        description_max_chars=description_max_chars,
        example_limit=int(getattr(query_settings, "prompt_column_example_limit", 0)),
    )
    compact_metrics = compact_metric_infos(
        state.get("metric_infos") or [],
        description_max_chars=description_max_chars,
    )
    # 复用生成节点相同的 authority 视图。生产默认的 model+guided 路由中，
    # 维度、度量、连接、排序等槽来自未经独立证明的 parser，不能再次注入
    # reviewer 形成相关性偏差；closed/显式消融路径仍完整保留这些结构证据。
    compact_ir = model_semantic_summary(
        ir,
        query_settings=query_settings,
    )
    # 7. 估算压缩前后 Token
    original = {
        "table_infos": tables,
        "metric_infos": list(state.get("metric_infos") or []),
        "semantic_ir": ir,
        "db_info": dict(state.get("db_info") or {}),
    }
    compact = {
        "table_infos": compact_tables,
        "metric_infos": compact_metrics,
        "semantic_ir": compact_ir,
        "db_info": dict(state.get("db_info") or {}),
        # 主候选弱点注入备选 Prompt，引导针对性修正
        "weaknesses": list(weaknesses),
    }
    before, after = estimate_tokens(original), estimate_tokens(compact)
    evidence = {
        "version": PROMPT_PAYLOAD_VERSION,
        "node_id": node_id,
        # 表列范围与主链路受控上下文一致，只去除冗余属性。
        "profile": "candidate-semantic-review",
        "before_tokens": before,
        "after_tokens": after,
        "compression_rate": round(max(0.0, 1 - after / before), 4) if before else 0.0,
        # 候选载荷规模指标，便于审计备选 Prompt 是否过小/过大
        "tables": len(compact_tables),
        "columns": sum(len(table.get("columns") or []) for table in compact_tables),
        "metrics": len(compact_metrics),
        "semantic_authority": str(
            (compact_ir.get("query_contract") or {}).get("authority") or "guided"
        ),
        "ir_hints_withheld": bool(compact_ir.get("ir_hints_withheld")),
    }
    # 8. 把 evidence 追加到 context_budget.model_prompts[node_id]
    current_budget = copy.deepcopy(dict(state.get("context_budget") or {}))
    model_prompts = dict(current_budget.get("model_prompts") or {})
    model_prompts[node_id] = evidence
    current_budget["model_prompts"] = model_prompts
    return {
        key: yaml.dump(value, allow_unicode=True, sort_keys=False)
        for key, value in compact.items()
    }, current_budget
