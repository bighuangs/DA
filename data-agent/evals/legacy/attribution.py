"""Deterministic NL2SQL failure attribution and targeted regression planning."""
from __future__ import annotations

from collections import Counter
from typing import Any

import sqlglot
from sqlglot import exp

from app.domain.query.safety.sql_security import find_unlinked_table_groups
from evals.legacy.metrics import extract_tables, syntax_valid


CAUSES: dict[str, dict[str, Any]] = {
    "pass": {"label": "结果通过", "stage": "result", "action": "保持现有行为并纳入防回归集。"},
    "acceptable_alternative": {"label": "可接受 SQL 路径", "stage": "evaluation_contract", "action": "在黄金集声明 required/allowed tables，避免把等价路径误报为错误。"},
    "infrastructure": {"label": "基础设施失败", "stage": "infrastructure", "action": "先修复数据库、向量库或模型连接，再重跑；不得计入模型正确率。"},
    "pipeline": {"label": "流水线失败", "stage": "pipeline", "action": "查看 error、retry_count 与步骤快照，修复失败节点后定向回归。"},
    "syntax": {"label": "SQL 语法失败", "stage": "generate_validate", "action": "收紧生成格式并将校验错误反馈给 correct_sql。"},
    "join_path": {"label": "连接路径缺失", "stage": "validate_sql", "action": "补充 PK/FK 连接上下文，并由 AST 守卫拒绝未关联多表查询。"},
    "multi_turn_context": {"label": "多轮上下文丢失", "stage": "plan_generate", "action": "继承上一轮维度、过滤、排序和限制，仅替换追问明确改变的度量。"},
    "time_semantics": {"label": "时间粒度错误", "stage": "filter_plan_generate", "action": "优先召回时间维表，并使用年月/季度可读字段作为投影、分组和排序轴。"},
    "entity_projection": {"label": "实体展示字段错误", "stage": "retrieval_filter_generate", "action": "将实体 ID 与展示名称写入语义契约，列表和 Top/Bottom 默认返回名称。"},
    "retrieval_table": {"label": "缺少必要表", "stage": "retrieval_filter", "action": "检查字段召回和筛表证据，补召回别名、展示字段及连接维表。"},
    "semantic_sql": {"label": "SQL 语义不一致", "stage": "plan_generate", "action": "对比投影、聚合、过滤、分组、排序和限制，补定向样例或业务口径。"},
}


def _projection_names(sql: str) -> set[str]:
    try:
        tree = sqlglot.parse_one(sql, read="mysql")
    except Exception:
        return set()
    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if select is None:
        return set()
    names: set[str] = set()
    for projection in select.expressions:
        for column in projection.find_all(exp.Column):
            names.add(column.name.lower())
    return names


def _evidence(result: dict, code: str, stage_evidence: dict | None) -> list[str]:
    evidence = []
    generated_tables = sorted(extract_tables(result.get("gen_sql") or ""))
    expected_tables = sorted(extract_tables(result.get("expected_sql") or ""))
    if generated_tables != expected_tables:
        evidence.append(f"生成表={generated_tables}，期望表={expected_tables}")
    if code == "join_path":
        evidence.append("同一查询块包含多个数据源，但关联图不连通")
    elif code == "multi_turn_context":
        evidence.append("追问结果未保留上一轮的分组或维表")
    elif code == "time_semantics":
        evidence.append("时间序列未使用期望的可读时间粒度字段")
    elif code == "entity_projection":
        evidence.append("生成 SQL 投影 ID，而期望 SQL 投影业务名称")
    if stage_evidence:
        filtered = stage_evidence.get("filtered_tables") or []
        if filtered:
            evidence.append(f"筛表阶段保留={filtered}")
        attempts = stage_evidence.get("sql_attempts") or []
        if len(attempts) > 1:
            evidence.append(f"SQL 共生成/纠正 {len(attempts)} 版")
    return evidence


def attribute_case(result: dict, case: dict | None = None, stage_evidence: dict | None = None) -> dict:
    """Assign one primary root cause using observable execution and AST evidence."""
    case = case or result
    if result.get("failure_type") == "infrastructure":
        code, confidence = "infrastructure", 1.0
    elif result.get("result_equivalent"):
        code, confidence = ("pass", 1.0) if result.get("tables_correct") else ("acceptable_alternative", 0.95)
    elif result.get("error") or not result.get("exec_success"):
        code, confidence = "pipeline", 0.9
    elif not syntax_valid(result.get("gen_sql") or ""):
        code, confidence = "syntax", 1.0
    elif find_unlinked_table_groups(result.get("gen_sql") or ""):
        code, confidence = "join_path", 1.0
    elif str(case.get("category") or result.get("category")) == "multi_turn":
        code, confidence = "multi_turn_context", 0.98
    else:
        generated_sql = result.get("gen_sql") or ""
        expected_sql = result.get("expected_sql") or ""
        generated_tables = extract_tables(generated_sql)
        expected_tables = extract_tables(expected_sql)
        generated_projection = _projection_names(generated_sql)
        expected_projection = _projection_names(expected_sql)
        if str(case.get("category") or result.get("category")) == "time_series" and (
            generated_tables != expected_tables or generated_projection != expected_projection
        ):
            code, confidence = "time_semantics", 0.95
        elif (
            any(name.endswith("_id") for name in generated_projection)
            and any(name.endswith("_name") for name in expected_projection)
        ):
            code, confidence = "entity_projection", 0.98
        elif not expected_tables.issubset(generated_tables):
            missing_tables = expected_tables - generated_tables
            prompt_tables = {
                str(item)
                for item in ((stage_evidence or {}).get("filtered_tables") or [])
            }
            if missing_tables and missing_tables.issubset(prompt_tables):
                # The model had the required table in its authorized prompt
                # context but omitted it from SQL.  Calling this retrieval
                # failure sends remediation to the wrong owner stage.
                code, confidence = "semantic_sql", 0.9
            else:
                code, confidence = "retrieval_table", 0.9
        else:
            code, confidence = "semantic_sql", 0.75

    definition = CAUSES[code]
    return {
        "code": code,
        "label": definition["label"],
        "stage": definition["stage"],
        "confidence": confidence,
        "evidence": _evidence(result, code, stage_evidence),
        "recommendation": definition["action"],
    }


def enrich_results(results: list[dict]) -> list[dict]:
    for result in results:
        result["attribution"] = attribute_case(
            result,
            result,
            result.get("stage_evidence"),
        )
    return results


def summarize_attributions(results: list[dict], workspace_id: str | None = None) -> dict:
    """Aggregate root causes and create deterministic, runnable regression slices."""
    enrich_results(results)
    failed = [item for item in results if not item.get("result_equivalent")]
    counter = Counter(item["attribution"]["code"] for item in failed)
    summaries = []
    actions = []
    for code, count in sorted(counter.items(), key=lambda pair: (-pair[1], pair[0])):
        cases = [str(item.get("id")) for item in failed if item["attribution"]["code"] == code]
        definition = CAUSES[code]
        summaries.append({
            "code": code,
            "label": definition["label"],
            "stage": definition["stage"],
            "count": count,
            "case_ids": cases,
            "recommendation": definition["action"],
        })
        only = ",".join(cases)
        command = f"python -m evals.legacy.run_eval --only {only}"
        if workspace_id:
            command += f" --workspace-id {workspace_id}"
        actions.append({
            "cause": code,
            "title": f"定向修复：{definition['label']}",
            "case_ids": cases,
            "owner_stage": definition["stage"],
            "action": definition["action"],
            "regression_command": command,
        })
    return {
        "failed_count": len(failed),
        "passed_count": len(results) - len(failed),
        "root_causes": summaries,
        "targeted_actions": actions,
    }
