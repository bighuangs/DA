"""TODO-88 上传分析意图与关联 SQL 的确定性守卫。"""
from __future__ import annotations

import sqlglot
from sqlglot import exp

from app.domain.data_sources.context import AssetAccessContext, JOIN_INTENTS
from app.domain.query.safety.sql_security import SQLSecurityError


def _alias_map(tree: exp.Expression) -> dict[str, str]:
    return {
        str(table.alias_or_name or table.name).lower(): str(table.name).lower()
        for table in tree.find_all(exp.Table)
        if table.name
    }


def _column_ref(column: exp.Column, aliases: dict[str, str]) -> tuple[str, str]:
    table = aliases.get(str(column.table or "").lower(), str(column.table or "").lower())
    return table, str(column.name or "").lower()


def _normalised_mapping_pairs(mapping: dict) -> set[frozenset[tuple[str, str]]]:
    pairs = set()
    for item in mapping.get("mappings") or []:
        left = item.get("left") or {}
        right = item.get("right") or {}
        left_ref = (
            str(left.get("table") or "").lower(),
            str(left.get("column") or "").lower(),
        )
        right_ref = (
            str(right.get("table") or "").lower(),
            str(right.get("column") or "").lower(),
        )
        if all(left_ref) and all(right_ref):
            pairs.add(frozenset((left_ref, right_ref)))
    return pairs


def _set_operation_selects(tree: exp.Expression) -> list[exp.Select]:
    if isinstance(tree, (exp.Union, exp.Intersect, exp.Except)):
        return [
            *_set_operation_selects(tree.this),
            *_set_operation_selects(tree.expression),
        ]
    if isinstance(tree, exp.Subquery):
        return _set_operation_selects(tree.this)
    return [tree] if isinstance(tree, exp.Select) else []


def _source_label(select: exp.Select) -> str | None:
    for projection in select.expressions:
        if str(projection.alias_or_name or "").lower() != "source_label":
            continue
        expression = projection.this if isinstance(projection, exp.Alias) else projection
        if isinstance(expression, exp.Literal) and expression.is_string:
            return str(expression.this)
    return None


def validate_source_analysis_sql(
    sql: str,
    *,
    access: AssetAccessContext | None,
    mapping: dict | None,
) -> dict:
    if access is None or not access.upload_assets:
        return {"checked": False, "reason": "dw_only"}
    try:
        tree = sqlglot.parse_one(str(sql or ""), read="mysql")
    except Exception as exc:
        raise SQLSecurityError(f"上传分析 SQL 无法解析：{exc}") from exc
    if any(True for _ in tree.find_all(exp.Star)):
        raise SQLSecurityError("上传分析禁止 SELECT *，必须显式列出字段")

    intent = str(access.comparison_intent or "upload_only")
    mapping = dict(mapping or {})
    joins = list(tree.find_all(exp.Join))
    independent_evidence = None
    if intent == "independent_aggregate_compare":
        if not isinstance(tree, exp.Union) or tree.args.get("distinct") is not False:
            raise SQLSecurityError(
                "独立聚合对比必须使用 UNION ALL 对齐上传来源与正式 DW"
            )
        branches = _set_operation_selects(tree)
        if len(branches) != 2:
            raise SQLSecurityError("独立聚合对比必须且只能包含两个来源分支")
        upload_names = set(access.logical_to_physical)
        branch_types = []
        labels = []
        for branch in branches:
            table_names = {
                str(table.name or "").lower()
                for table in branch.find_all(exp.Table)
                if table.name
            }
            has_upload = bool(table_names & upload_names)
            has_dw = bool(table_names - upload_names)
            if has_upload and has_dw:
                raise SQLSecurityError(
                    "独立聚合对比的单个分支不得混合上传表与 DW 表"
                )
            if not table_names or not any(True for _ in branch.find_all(exp.AggFunc)):
                raise SQLSecurityError("独立聚合对比的每个来源分支都必须先完成聚合")
            branch_types.append("upload" if has_upload else "dw")
            label = _source_label(branch)
            if not label:
                raise SQLSecurityError(
                    "独立聚合对比的每个来源分支必须投影常量 source_label"
                )
            labels.append(label)
        if sorted(branch_types) != ["dw", "upload"]:
            raise SQLSecurityError("独立聚合对比必须包含一个上传分支和一个 DW 分支")
        if len(set(labels)) != 2:
            raise SQLSecurityError("独立聚合对比的 source_label 必须能区分两个来源")
        independent_evidence = {
            "branch_count": len(branches),
            "branch_types": branch_types,
            "source_labels": labels,
            "union_all": True,
        }

    if intent == "independent_aggregate_compare" and joins:
        aliases = _alias_map(tree)
        upload_names = set(access.logical_to_physical)
        for equality in tree.find_all(exp.EQ):
            if not isinstance(equality.left, exp.Column) or not isinstance(
                equality.right, exp.Column
            ):
                continue
            left_table, _ = _column_ref(equality.left, aliases)
            right_table, _ = _column_ref(equality.right, aliases)
            if (left_table in upload_names) != (right_table in upload_names):
                raise SQLSecurityError(
                    "独立聚合对比禁止上传表与 DW 行级 JOIN；"
                    "请分别聚合后使用 UNION ALL 对齐"
                )
    if intent in JOIN_INTENTS:
        if (
            mapping.get("status") != "confirmed"
            or mapping.get("mapping_id") != access.mapping_confirmation_id
        ):
            raise SQLSecurityError("行级关联缺少当前版本的已确认字段映射")
        if not mapping.get("evidence", {}).get("auto_join_allowed", False):
            raise SQLSecurityError(
                "关联质量门禁未通过（覆盖率/唯一性/扇出/粒度），禁止执行 JOIN"
            )
        aliases = _alias_map(tree)
        allowed_pairs = _normalised_mapping_pairs(mapping)
        for join in joins:
            on = join.args.get("on")
            if on is None:
                raise SQLSecurityError("上传关联禁止无 ON 条件的 JOIN")
            equalities = list(on.find_all(exp.EQ))
            if not equalities:
                raise SQLSecurityError("上传关联必须使用已确认字段的等值条件")
            for equality in equalities:
                if not isinstance(equality.left, exp.Column) or not isinstance(
                    equality.right, exp.Column
                ):
                    raise SQLSecurityError("上传关联 ON 仅允许字段间等值匹配")
                pair = frozenset((
                    _column_ref(equality.left, aliases),
                    _column_ref(equality.right, aliases),
                ))
                if pair not in allowed_pairs:
                    raise SQLSecurityError("SQL 使用了未确认的关联字段")
    if intent == "union" and not any(True for _ in tree.find_all(exp.Union)):
        raise SQLSecurityError("UNION 意图必须使用显式 UNION ALL 对齐来源")
    return {
        "checked": True,
        "intent": intent,
        "join_count": len(joins),
        "mapping_confirmation_id": access.mapping_confirmation_id,
        "mapping_gate_passed": True,
        **({"independent_aggregate": independent_evidence} if independent_evidence else {}),
    }
