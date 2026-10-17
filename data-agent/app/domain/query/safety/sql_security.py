"""
SQL 只读安全校验模块。

一句话总述：对 LLM 生成的 SQL 做只读硬拦截与危险操作黑名单校验，
任何写操作 / DDL / 危险函数 / 危险模式一律拒绝并抛出 SQLSecurityError。

核心功能：使用 sqlglot 将 SQL 解析为 AST，按多层规则做防御性校验——
        1) 根语句白名单：仅允许 SELECT / UNION / WITH(CTE) / EXPLAIN / DESCIBE 等只读根；
        2) 写操作 / DDL 黑名单：遍历 AST 检测 Insert/Update/Delete/Drop/Create/Alter/
           Truncate/Merge/Command 等节点；
        3) 危险函数黑名单：检测 LOAD_FILE、SLEEP、BENCHMARK、xp_cmdshell 等可被用于
           注入或拒绝服务的函数；
        4) 危险模式正则：INTO OUTFILE / INTO DUMPFILE / LOAD DATA 等文件读写与批量加载。

业务逻辑：
    1. 用 sqlglot.parse_one 将 SQL 解析为 AST（MySQL 方言）
    2. 检查根语句类型是否在 ALLOWED_ROOT_KEYS 白名单内
    3. 遍历 AST 检测写操作 / DDL 节点
    4. 遍历 AST 检测危险函数调用
    5. 正则扫描原始 SQL 文本检测危险模式
    6. 任一校验失败抛出 SQLSecurityError；全部通过则静默返回
"""
import re
from functools import lru_cache

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import Scope, traverse_scope


class SQLSecurityError(Exception):
    """SQL 安全校验失败异常，携带具体违规原因。"""


# 根语句白名单：sqlglot Expression.key 取值
# Select=普通查询，Union=联合查询，Subquery=括号包裹的子查询作为根，
# Describe=DESCRIBE/DESC，Show=SHOW，SetOperation 涵盖 UNION/INTERSECT/EXCEPT
ALLOWED_ROOT_KEYS = {
    "select",
    "union",
    "intersect",
    "except",
    "subquery",
    "describe",
    "show",
    "setoperation",
}

# 写操作 / DDL / 危险语句节点黑名单（按 Expression 类）
WRITE_DDL_NODE_TYPES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.TruncateTable,
    exp.Merge,
    exp.Command,  # sqlglot 兜底捕获的任意命令（如 GRANT/REPLACE/KILL）
)

# 危险函数黑名单（按方言扩展）：可被用于数据外泄 / 拒绝服务 / 系统命令
DANGEROUS_FUNCTIONS = {
    # MySQL
    "load_file", "into_outfile", "into_dumpfile",
    "sleep", "benchmark", "get_lock", "release_lock",
    "sys_exec", "sys_eval", "exec",
    # PostgreSQL
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_sleep",
    "pg_terminate_backend", "lo_import", "lo_export", "copy",
    # SQL Server
    "xp_cmdshell", "xp_filelist", "xp_dirtree", "sp_executesql",
    # Oracle
    "dbms_xmlquery", "utl_file", "utl_http",
}

# 危险模式正则：文件读写 / 批量加载 / 系统命令（在 AST 之外做文本级兜底）
DANGEROUS_PATTERNS = [
    re.compile(r"\binto\s+(outfile|dumpfile)\b", re.IGNORECASE),
    re.compile(r"\bload\s+data\b", re.IGNORECASE),
    re.compile(r"\bload_file\s*\(", re.IGNORECASE),
    re.compile(r"\bxp_cmdshell\b", re.IGNORECASE),
    re.compile(r"\bcopy\s+.+\bto\b", re.IGNORECASE),  # PG COPY ... TO
    re.compile(r"\bcopy\s+.+\bfrom\b", re.IGNORECASE),  # PG COPY ... FROM
]


@lru_cache(maxsize=1)
def _dangerous_func_names() -> tuple[str, ...]:
    """返回小写危险函数名元组，供 AnonymousFunc 匹配。"""
    return tuple(f.lower() for f in DANGEROUS_FUNCTIONS)


def check_sql_read(sql: str) -> None:
    """
    校验 SQL 是否为只读安全语句；若违反则抛出 SQLSecurityError。

    业务逻辑：
        1. 解析 SQL 为 sqlglot AST（MySQL 方言）；解析失败视为语法非法，拒绝
        2. 根语句白名单校验：key 不在白名单内则拒绝
        3. 遍历 AST 检测写操作 / DDL 节点
        4. 遍历 AST 检测危险函数调用（AnonymousFunc / Func 节点匹配黑名单）
        5. 文本级正则扫描检测危险模式（兜底防护，避免 AST 解析遗漏）
        6. 全部通过则静默返回

    Args:
        sql: 待校验的 SQL 字符串（LLM 生成）

    Raises:
        SQLSecurityError: 任一规则命中时抛出，message 描述具体违规类型
    """
    if not sql or not sql.strip():
        raise SQLSecurityError("SQL 为空")

    # ── 阶段 1：解析 AST ──────────────────────────────────────────
    try:
        tree = sqlglot.parse_one(sql, read="mysql")
    except Exception as e:
        # 解析失败：可能是 LLM 生成了非 SQL 文本，按语法非法拒绝
        raise SQLSecurityError(f"SQL 解析失败，疑似非合法 SQL：{e}") from e

    # ── 阶段 2：根语句白名单校验 ──────────────────────────────────
    root_key = tree.key.lower()
    if root_key not in ALLOWED_ROOT_KEYS:
        raise SQLSecurityError(
            f"仅允许只读查询（SELECT/UNION/WITH），检测到根语句类型：{tree.__class__.__name__}"
        )

    # ── 阶段 3：写操作 / DDL 节点检测 ─────────────────────────────
    for node in tree.find_all(WRITE_DDL_NODE_TYPES):
        raise SQLSecurityError(
            f"检测到写操作/DDL语句：{node.__class__.__name__}，已拒绝"
        )

    # ── 阶段 4：危险函数检测 ─────────────────────────────────────
    dangerous_names = _dangerous_func_names()
    # AnonymousFunc：sqlglot 把未知函数解析为该类型（含 this=函数名）
    for node in tree.find_all(exp.Anonymous):
        func_name = str(node.name).lower()
        if func_name in dangerous_names:
            raise SQLSecurityError(f"检测到危险函数调用：{node.name}，已拒绝")
    # 已知函数子类：按 sql_name 兜底匹配
    for node in tree.find_all(exp.Func):
        sql_name_fn = getattr(node, "sql_name", None)
        if sql_name_fn:
            try:
                name = sql_name_fn().lower()
            except Exception:
                continue
            if name in dangerous_names:
                raise SQLSecurityError(f"检测到危险函数调用：{name}，已拒绝")

    # ── 阶段 5：危险模式文本级正则兜底 ───────────────────────────
    for pattern in DANGEROUS_PATTERNS:
        match = pattern.search(sql)
        if match:
            raise SQLSecurityError(f"检测到危险模式：{match.group(0)}，已拒绝")


def extract_table_names(sql: str) -> set[str]:
    """
    从 SQL 中提取所有真实表名（排除 CTE 定义的临时表名）。

    一句话总述：用 sqlglot 解析 SQL，收集所有 Table 节点的表名，但剔除
                WITH ... AS (...) 定义的 CTE 名，避免把 LLM 自建的临时表
                误判为越权表。

    业务逻辑：
        1. 解析 SQL 为 AST（MySQL 方言）；解析失败抛 SQLSecurityError
        2. 收集所有 CTE 节点定义的临时表名（WITH t AS (...) 中的 t）
        3. 遍历所有 Table 节点，取其 name，排除 CTE 名后加入结果集
        4. 返回真实表名集合（小写归一不在此处做，保留原始大小写供调用方比对）

    Args:
        sql: 待解析的 SQL 字符串

    Returns:
        set[str]: SQL 中引用的真实表名集合（不含 CTE 临时表名）

    Raises:
        SQLSecurityError: SQL 解析失败时抛出
    """
    if not sql or not sql.strip():
        raise SQLSecurityError("SQL 为空")
    try:
        tree = sqlglot.parse_one(sql, read="mysql")
    except Exception as e:
        raise SQLSecurityError(f"SQL 解析失败，疑似非合法 SQL：{e}") from e

    # 收集 CTE 临时表名：WITH t AS (...) SELECT * FROM t 中的 t 不算越权表
    cte_names: set[str] = set()
    for cte in tree.find_all(exp.CTE):
        # sqlglot CTE 节点：alias_or_name 给出 CTE 名；兼容不同版本取 this.name
        name = getattr(cte, "alias_or_name", None) or (
            cte.this.name if cte.this is not None else None
        )
        if name:
            cte_names.add(name)

    # 收集真实表名：排除 CTE 名，避免误判
    tables: set[str] = set()
    for tbl in tree.find_all(exp.Table):
        name = tbl.name
        if name and name not in cte_names:
            tables.add(name)
    return tables


def _relationship_edges(expression: exp.Expression, aliases: set[str]) -> set[tuple[str, str]]:
    """Collect cross-source equality predicates from JOIN ON and WHERE clauses."""
    edges: set[tuple[str, str]] = set()
    predicates: list[exp.Expression] = []
    where = expression.args.get("where")
    if where is not None:
        predicates.append(where)
    for join in expression.args.get("joins") or []:
        if str(join.args.get("kind") or "").upper() == "CROSS":
            continue
        on = join.args.get("on")
        if on is not None:
            predicates.append(on)
    for predicate in predicates:
        for equality in predicate.find_all(exp.EQ):
            left = equality.this
            right = equality.expression
            if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                continue
            left_alias = (left.table or "").lower()
            right_alias = (right.table or "").lower()
            if left_alias in aliases and right_alias in aliases and left_alias != right_alias:
                edges.add(tuple(sorted((left_alias, right_alias))))

    # USING is an explicit relationship even though sqlglot stores only identifiers.
    seen_sources: list[str] = []
    from_ = expression.args.get("from_")
    if from_ is not None and isinstance(from_.this, (exp.Table, exp.Subquery)):
        seen_sources.append(from_.this.alias_or_name.lower())
    for join in expression.args.get("joins") or []:
        target = getattr(join.this, "alias_or_name", "").lower()
        if join.args.get("using") and target and seen_sources:
            edges.add(tuple(sorted((seen_sources[-1], target))))
        if target:
            seen_sources.append(target)
    return edges


def _is_proven_single_row_aggregate_source(source: object) -> bool:
    """Whether a derived/CTE source is provably one-row pure aggregation.

    The proof is intentionally narrower than general SQL cardinality analysis:
    it accepts only a single SELECT with no grouping/HAVING/window and requires
    every projection to be aggregate-derived without aggregate-external columns.
    Ordinary ``LIMIT 1`` queries and uncertain computed sources remain blocked.
    """
    if not isinstance(source, Scope) or not isinstance(source.expression, exp.Select):
        return False
    select = source.expression
    if (
        select.args.get("group") is not None
        or select.args.get("having") is not None
        or select.args.get("qualify") is not None
        or not select.expressions
    ):
        return False
    for projection in select.expressions:
        value = projection.this if isinstance(projection, exp.Alias) else projection
        if (
            value.find(exp.Select) is not None
            or value.find(exp.Subquery) is not None
            or value.find(exp.Window) is not None
        ):
            return False
        aggregates = list(value.find_all(exp.AggFunc))
        if not aggregates:
            return False
        aggregate_columns = {
            id(column)
            for aggregate in aggregates
            for column in aggregate.find_all(exp.Column)
        }
        if any(
            id(column) not in aggregate_columns
            for column in value.find_all(exp.Column)
        ):
            return False
    return True


def _proven_cross_join_edges(scope: Scope) -> set[tuple[str, str]] | None:
    """Return connectivity edges for safe scalar CROSS JOINs, else ``None``."""
    expression = scope.expression
    if not isinstance(expression, exp.Select):
        return set()
    from_clause = expression.args.get("from_")
    from_source = from_clause.this if from_clause is not None else None
    seen: list[str] = []
    if isinstance(from_source, (exp.Table, exp.Subquery)):
        source_alias = str(from_source.alias_or_name or "").lower()
        if source_alias:
            seen.append(source_alias)
    edges: set[tuple[str, str]] = set()
    for join in expression.args.get("joins") or []:
        target = str(getattr(join.this, "alias_or_name", "") or "").lower()
        if str(join.args.get("kind") or "").upper() == "CROSS":
            selected = scope.selected_sources.get(target) if target else None
            if (
                not seen
                or selected is None
                or not _is_proven_single_row_aggregate_source(selected[1])
            ):
                return None
            edges.add(tuple(sorted((seen[0], target))))
        if target:
            seen.append(target)
    return edges


def find_unlinked_table_groups(sql: str) -> list[list[str]]:
    """Return disconnected multi-source query blocks that can create cartesian products.

    Each SELECT scope is checked independently. CTE/subquery internals are therefore not
    confused with their parent query. Explicit CROSS JOIN is accepted only when its
    right-hand source is a statically proven one-row pure aggregate scope.
    """
    if not sql or not sql.strip():
        return []
    try:
        tree = sqlglot.parse_one(sql, read="mysql")
    except Exception:
        return []
    risks: list[list[str]] = []
    for scope in traverse_scope(tree):
        aliases = {str(alias).lower() for alias in scope.selected_sources}
        if len(aliases) <= 1:
            continue
        cross_edges = _proven_cross_join_edges(scope)
        if cross_edges is None:
            risks.append(sorted(aliases))
            continue
        edges = _relationship_edges(scope.expression, aliases) | cross_edges
        connected = {next(iter(aliases))}
        changed = True
        while changed:
            changed = False
            for left, right in edges:
                if left in connected and right not in connected:
                    connected.add(right)
                    changed = True
                elif right in connected and left not in connected:
                    connected.add(left)
                    changed = True
        if connected != aliases:
            risks.append(sorted(aliases - connected))
    return risks


def check_join_relationships(sql: str) -> None:
    """Reject disconnected multi-table SELECT scopes before EXPLAIN/execution."""
    risks = find_unlinked_table_groups(sql)
    if risks:
        groups = "; ".join(", ".join(group) for group in risks)
        raise SQLSecurityError(
            f"检测到未关联的多表查询或笛卡尔积风险：{groups}；请使用 PK/FK 关联谓词连接所有数据源"
        )


def check_table_whitelist(sql: str, allowed_tables, *, reject_empty: bool = False) -> None:
    """
    校验 SQL 中所有真实表名都在白名单内；越权则抛 SQLSecurityError。

    一句话总述：不信任 LLM 返回的表名，从真实 SQL 用 sqlglot 提取表名，
                与本次召回允许的表名集合求差集，任何未授权表直接拒绝。

    业务逻辑：
        1. 调用 extract_table_names 取出 SQL 中所有真实表名
        2. 把 allowed_tables 归一为集合（过滤空值）
        3. 若白名单为空，则跳过校验（无可比对的白名单时不阻断，依赖只读校验兜底）
        4. 求差集 used - allowed，若存在越权表名则抛 SQLSecurityError 列出违规表

    Args:
        sql: 待校验的 SQL 字符串（LLM 生成）
        allowed_tables: 本次允许访问的表名可迭代对象（通常来自 state["table_infos"]）

    Raises:
        SQLSecurityError: SQL 引用了未授权表时抛出，message 列出违规表名
    """
    used = extract_table_names(sql)
    allowed = {str(t) for t in (allowed_tables or []) if t}
    # 普通调用保留空白名单兼容行为；权限策略场景可显式拒绝空白名单。
    if not allowed:
        if reject_empty and used:
            raise SQLSecurityError("本轮没有允许访问的表，已拒绝执行")
        return
    violators = used - allowed
    if violators:
        raise SQLSecurityError(
            f"SQL 引用了未授权的表：{sorted(violators)}（允许的表：{sorted(allowed)}）"
        )


def enforce_limit(sql: str, limit: int) -> str:
    """
    若 SQL 不含 LIMIT，则包装为子查询并附加 LIMIT {limit}，防止大结果集拖垮服务。

    一句话总述：用 sqlglot 解析 SQL 检测是否已存在 Limit 节点，已有则原样返回，
                没有则包装成 `SELECT * FROM ({sql}) AS _da_sub LIMIT {limit}`。

    业务逻辑：
        1. 空 SQL 或 limit<=0 直接返回原 SQL
        2. sqlglot 解析；解析失败则原样返回（由 check_sql_read 兜底校验合法性）
        3. AST 中存在 exp.Limit 节点 → 已有 LIMIT，原样返回
        4. 否则包装为子查询 + LIMIT 返回（包装后仍是只读 SELECT，不影响只读安全校验）

    Args:
        sql: 待执行的 SQL 字符串
        limit: 强制行数上限

    Returns:
        str: 带 LIMIT 的安全 SQL（已含 LIMIT 则不变）
    """
    if not sql or limit <= 0:
        return sql
    try:
        tree = sqlglot.parse_one(sql, read="mysql")
    except Exception:
        return sql
    if not isinstance(tree, (exp.Select, exp.SetOperation, exp.Subquery)):
        return sql
    normalized = sql.strip().removesuffix(";").strip()
    return f"SELECT * FROM ({normalized}) AS _da_sub LIMIT {limit}"
