"""行级与列级权限编译器。

策略的安全边界在应用侧完成：

* Schema 注入前过滤不可见表和字段，减少模型误用机会；
* SQL AST 校验时拒绝越权表、列和通配符；
* 通过校验后把行过滤条件编译进每个直接引用目标表的 SELECT；
* 实际执行使用编译后的 SQL，不能依赖 Prompt 或前端按钮。

没有配置策略时保持兼容，默认允许当前工作空间已有表和列。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from fnmatch import fnmatchcase

import sqlglot
from sqlglot import exp

from app.domain.query.safety.sql_security import SQLSecurityError, check_sql_read, extract_table_names


# 策略角色白名单用于约束配置值；all 表示策略对任意请求角色生效。
ROLE_VALUES = {"all", "owner", "admin", "analyst", "viewer"}

# 行过滤模板只允许替换四种经过 SQL 字面量转义的请求身份字段。
PLACEHOLDER_RE = re.compile(r"\{\{(user_id|username|workspace_id|role)\}\}")


def _literal(value: str) -> str:
    """
    将身份属性安全编码为 SQL 单引号字符串字面量。

    核心功能：把输入转换为字符串，并将内部单引号按 SQL 规则翻倍转义，
    最后包裹单引号，供受控行过滤占位符替换使用。

    业务逻辑：
        1. 将输入值转换为字符串
        2. 把每个单引号替换为两个单引号
        3. 在转义结果两侧添加单引号

    Args:
        value: 用户、工作空间或角色属性值。

    Returns:
        str: 可安全嵌入 SQL 谓词的字符串字面量。
    """
    return "'" + str(value).replace("'", "''") + "'"


def render_row_filter(template: str, *, user_id: str, username: str,
                      workspace_id: str, role: str) -> str:
    """
    将行级权限模板中的身份占位符替换为安全 SQL 字面量。

    核心功能：预先转义当前请求的用户 ID、用户名、工作空间和角色，再通过受限正则
    仅替换已知占位符；模板中的其他 SQL 谓词结构保持不变。

    业务逻辑：
        1. 分别将四个身份属性编码为 SQL 字面量
        2. 建立占位符名称到安全值的映射
        3. 扫描模板中的允许占位符
        4. 用匹配分组对应的安全值完成替换

    Args:
        template: 已通过谓词校验的行过滤模板。
        user_id: 当前用户 ID。
        username: 当前用户名。
        workspace_id: 当前工作空间 ID。
        role: 当前请求角色。

    Returns:
        str: 已渲染身份值的 SQL 行过滤谓词。
    """
    # 所有动态值先转换为安全字面量，替换回调不接触原始输入。
    values = {
        "user_id": _literal(user_id),
        "username": _literal(username),
        "workspace_id": _literal(workspace_id),
        "role": _literal(role),
    }

    # 正则只识别白名单中的四种占位符，不执行任意模板表达式。
    return PLACEHOLDER_RE.sub(lambda match: values[match.group(1)], template)


def validate_row_filter(value: str | None) -> str | None:
    """
    校验行过滤配置是无副作用的单一谓词，而不是可执行的完整 SQL。

    核心功能：拒绝多语句和 SQL 注释，将允许的身份占位符替换为样例字面量后嵌入
    ``SELECT 1 WHERE`` 解析；随后禁止子查询、表引用及 CTE，并复用只读安全检查。

    业务逻辑：
        1. 规范化输入；空值表示没有行过滤
        2. 拒绝分号及行内、块级 SQL 注释标记
        3. 用固定样例字面量替换允许的身份占位符
        4. 将条件嵌入最小 SELECT 并按 MySQL 方言解析
        5. 要求 WHERE 中存在非空谓词
        6. 拒绝谓词中的 SELECT、子查询、表和 CTE 节点
        7. 执行只读 SQL 安全检查并返回原始规范化模板

    Args:
        value: 待校验的行过滤模板；允许为空。

    Returns:
        str | None: 合法时返回去除首尾空白的模板，空输入返回 ``None``。

    Raises:
        ValueError: 模板包含多语句、注释、非法语法、空谓词或额外查询结构。
        SQLSecurityError: 包装后的检查 SQL 违反只读安全规则。
    """
    # 空字符串和 None 都表示该策略不附加行级过滤。
    text = str(value or "").strip()
    if not text:
        return None

    # 禁止语句分隔符与 SQL 注释，避免模板截断或追加额外语句。
    if ";" in text or "--" in text or "/*" in text or "*/" in text:
        raise ValueError("行过滤条件不能包含多语句或 SQL 注释")

    # 使用固定非敏感样例替换占位符，使模板可以在保存策略时独立解析验证。
    sample = PLACEHOLDER_RE.sub("'__da_placeholder__'", text)
    try:
        tree = sqlglot.parse_one(f"SELECT 1 WHERE {sample}", read="mysql")
    except Exception as exc:
        # 将底层解析异常转换为配置层 ValueError，同时保留异常链供诊断。
        raise ValueError(f"行过滤条件不是合法谓词：{exc}") from exc
    where = tree.args.get("where")

    # 包装 SQL 必须实际生成非空 WHERE 表达式，防止空配置绕过后续编译。
    if where is None or not where.this:
        raise ValueError("行过滤条件不能为空")

    # 行过滤只能引用当前目标表字段，不允许通过子查询或额外表扩大数据访问。
    if any(isinstance(node, (exp.Select, exp.Subquery, exp.Table, exp.CTE))
           for node in where.find_all(exp.Expression)):
        raise ValueError("行过滤条件不能包含子查询或额外表引用")

    # 最后复用统一只读检查，确保谓词未引入写操作或其他危险结构。
    check_sql_read(f"SELECT 1 WHERE {sample}")
    return text


@dataclass(frozen=True)
class PolicySnapshot:
    """
    封装当前请求身份与已启用访问策略，并提供不可变的授权判断视图。

    快照将角色、用户和工作空间身份固定在一次请求内，统一负责表/列策略匹配、
    Prompt 元数据裁剪、SQL AST 作用域解析及行级过滤编译。
    """

    policies: tuple
    user_id: str = ""
    username: str = ""
    workspace_id: str = ""
    role: str = "viewer"

    def _matching(self, table_name: str) -> list:
        """
        查找对指定表和当前请求身份生效的策略，并按优先级稳定排序。

        核心功能：依次应用角色、用户和表名通配模式过滤；匹配结果按 priority 升序、
        id 升序排列，为后续拒绝优先、列限制求交和行条件累积提供稳定输入。

        业务逻辑：
            1. 规范化目标表名并初始化结果
            2. 遍历快照中的全部策略
            3. 跳过角色既非 ``all`` 也不匹配当前角色的策略
            4. 跳过绑定其他用户的策略
            5. 使用不区分大小写的 glob 模式匹配表名
            6. 按优先级和策略 ID 排序后返回

        Args:
            table_name: 待授权的物理表名。

        Returns:
            list: 对当前身份和目标表生效的策略列表。
        """
        table = str(table_name or "")
        result = []

        # 遍历快照策略，逐项应用角色、用户和表模式三个作用域条件。
        for policy in self.policies:
            pattern = str(policy.table_pattern or "*").strip() or "*"
            role = str(policy.role or "all").lower()

            # all 角色对所有请求生效，其余策略只匹配当前请求角色。
            if role not in ("all", self.role.lower()):
                continue

            # 设置 user_id 的策略为用户专属策略，其他用户不得继承。
            if policy.user_id and str(policy.user_id) != self.user_id:
                continue

            # 表模式采用大小写无关 glob 匹配，支持精确表名和通配范围。
            if fnmatchcase(table.lower(), pattern.lower()):
                result.append(policy)

        # priority 越小越靠前，id 用于同优先级下形成确定顺序。
        return sorted(result, key=lambda item: (int(item.priority or 100), int(item.id or 0)))

    def table_rule(self, table_name: str) -> dict:
        """
        合并指定表的全部匹配策略，生成最终表级授权规则。

        核心功能：任意拒绝策略都会拒绝整表；多个 allowed_columns 采用集合交集，
        使组合策略只会收紧权限；所有非空 row_filter 按顺序累积，稍后以 AND 编译。

        业务逻辑：
            1. 获取当前身份下匹配目标表的策略
            2. 若任一策略 access 为假，则设置整表拒绝
            3. 遍历策略并规范化允许列和行过滤
            4. 收集每个非空列限制集合
            5. 收集每个非空行过滤模板
            6. 对所有列限制求交；没有限制时保持 ``None`` 表示全部列
            7. 返回拒绝标记、允许列集合和行过滤列表

        Args:
            table_name: 待合并策略的物理表名。

        Returns:
            dict: 包含 ``denied``、``allowed_columns`` 和 ``row_filters`` 的规则。
        """
        matching = self._matching(table_name)

        # 拒绝优先：只要一条匹配策略显式拒绝，就不能被其他允许策略覆盖。
        denied = any(not bool(item.access) for item in matching)
        restrictions = []
        filters = []

        # 遍历所有匹配策略，分别收集列级限制集合与行级谓词。
        for item in matching:
            columns = [str(column).split(".")[-1] for column in (item.allowed_columns or []) if str(column).strip()]

            # 策略显式声明列时，去除可选表前缀并统一小写后加入限制集合。
            if columns:
                restrictions.append({column.lower() for column in columns})

            # 非空行过滤保持策略排序顺序，后续编译时逐项与现有 WHERE 求 AND。
            if str(item.row_filter or "").strip():
                filters.append(str(item.row_filter).strip())
        allowed = None

        # 多条列限制取交集而非并集，确保策略叠加只能收紧、不能扩大访问范围。
        for restriction in restrictions:
            allowed = restriction if allowed is None else allowed & restriction
        return {"denied": denied, "allowed_columns": allowed, "row_filters": filters}

    def is_table_allowed(self, table_name: str) -> bool:
        """
        判断当前请求是否允许访问指定表。

        核心功能：复用合并后的表规则，只依据拒绝优先结果作出布尔判断。

        业务逻辑：
            1. 合并目标表的匹配策略
            2. 对 ``denied`` 标记取反

        Args:
            table_name: 待检查的物理表名。

        Returns:
            bool: 未被任一匹配策略拒绝时为真。
        """
        return not self.table_rule(table_name)["denied"]

    def is_column_allowed(self, table_name: str, column_name: str) -> bool:
        """
        判断当前请求是否允许访问指定表中的字段。

        核心功能：先应用整表拒绝，再检查合并后的列集合；没有列限制时允许全部字段，
        有限制时仅允许大小写规范化后位于交集中的字段。

        业务逻辑：
            1. 合并目标表规则
            2. 若整表被拒绝，则字段也被拒绝
            3. 若 allowed_columns 为 ``None``，则允许任意字段
            4. 否则检查小写字段名是否位于允许集合

        Args:
            table_name: 字段所属物理表。
            column_name: 待检查字段名。

        Returns:
            bool: 表和字段均通过权限规则时为真。
        """
        rule = self.table_rule(table_name)
        return not rule["denied"] and (
            rule["allowed_columns"] is None
            or str(column_name).lower() in rule["allowed_columns"]
        )

    def filter_table_infos(self, table_infos: list[dict]) -> list[dict]:
        """
        按当前权限过滤注入 Prompt 的表及字段元数据，并返回新的列表。

        核心功能：隐藏整表拒绝的元数据，对列受限表仅保留交集允许字段；复制表字典
        后替换 columns，避免在过滤过程中修改调用方持有的原对象。

        业务逻辑：
            1. 初始化可见表列表
            2. 遍历输入表元数据并合并对应策略
            3. 若整表被拒绝，则跳过该表
            4. 若存在列限制，则过滤字段；否则保留全部字段
            5. 浅复制表对象并写入新的字段列表
            6. 返回可安全提供给模型的元数据

        Args:
            table_infos: 原始表及字段元数据列表。

        Returns:
            list[dict]: 仅包含当前请求可见表和字段的新列表。
        """
        visible = []

        # 遍历每张候选表，在进入 Prompt 之前应用与执行侧一致的权限规则。
        for table in table_infos or []:
            name = str(table.get("name") or "")
            rule = self.table_rule(name)

            # 整表拒绝时不暴露表名或字段，降低模型误用不可见资源的机会。
            if rule["denied"]:
                continue

            # 无列限制时保留全部字段；有限制时只保留交集允许列。
            columns = [
                column for column in (table.get("columns") or [])
                if rule["allowed_columns"] is None
                or str(column.get("name") or "").lower() in rule["allowed_columns"]
            ]

            # 复制表映射后替换字段列表，避免修改输入元数据中的 columns。
            copied = dict(table)
            copied["columns"] = columns
            visible.append(copied)
        return visible

    def filter_metric_infos(self, metric_infos: list[dict]) -> list[dict]:
        """
        过滤引用不可访问字段的指标元数据。

        核心功能：解析每个指标的 relevant_columns，只有所有相关字段都通过列权限时
        才向模型暴露该指标，防止指标公式间接泄露受限列。

        业务逻辑：
            1. 初始化可见指标列表
            2. 遍历指标并读取相关字段标识
            3. 将每个 ``table.column`` 标识拆分为表名和字段名
            4. 要求所有相关字段均通过授权
            5. 将完全可访问的指标加入结果

        Args:
            metric_infos: 原始指标元数据列表。

        Returns:
            list[dict]: 相关字段全部可访问的指标列表。
        """
        visible = []

        # 逐个指标检查其完整字段依赖；空依赖按 all(empty)=True 保持兼容。
        for metric in metric_infos or []:
            relevant = list(metric.get("relevant_columns") or [])

            # 任一相关字段被拒绝都会隐藏整个指标，避免暴露受限计算口径。
            if all(self.is_column_allowed(*self._split_column_id(column_id)) for column_id in relevant):
                visible.append(metric)
        return visible

    def filter_schema_snapshot(self, snapshot: dict) -> dict:
        """
        过滤 Schema 快照中的表结构，同时保留其他快照元数据。

        核心功能：复制顶层快照映射，并使用 ``filter_table_infos`` 替换 tables 字段，
        使调用方获得与 Prompt 权限一致的新快照。

        业务逻辑：
            1. 将空快照归一化并复制顶层映射
            2. 过滤原快照中的 tables
            3. 将安全表列表写入副本并返回

        Args:
            snapshot: 包含 tables 及其他元数据的 Schema 快照。

        Returns:
            dict: 表结构已按当前权限裁剪的新快照。
        """
        result = dict(snapshot or {})
        result["tables"] = self.filter_table_infos(snapshot.get("tables") or [])
        return result

    @staticmethod
    def _split_column_id(column_id: str) -> tuple[str, str]:
        """
        将字段标识拆分为表名与字段名。

        核心功能：以最后一个点号为界解析 ``table.column``；没有点号时保留原文本
        作为表名并返回空字段名，以维持现有权限检查的关闭式行为。

        业务逻辑：
            1. 将输入规范化为字符串
            2. 若不存在点号，则返回原文本和空字段
            3. 否则从右侧拆分一次

        Args:
            column_id: 待拆分的字段标识。

        Returns:
            tuple[str, str]: 表名和字段名。
        """
        text = str(column_id or "")

        # 无限定符时无法可靠识别字段归属，按既有契约返回空字段部分。
        if "." not in text:
            return text, ""
        return text.rsplit(".", 1)

    def _render_filter(self, template: str, table_name: str, alias: str | None) -> exp.Expression:
        """
        渲染单个行过滤模板，并把字段限定到目标表当前作用域的别名。

        核心功能：使用快照中的身份属性替换模板占位符，解析为 MySQL 谓词 AST；
        对未限定字段或仍使用原表名的字段，统一改写为查询中的实际表别名。

        业务逻辑：
            1. 将请求身份安全渲染进行过滤模板
            2. 按 MySQL 方言解析谓词表达式
            3. 选择查询别名；没有别名时使用物理表名
            4. 遍历谓词中的字段节点
            5. 对未限定或以原表名限定的字段写入当前作用域限定符
            6. 返回可插入 WHERE 的表达式

        Args:
            template: 已验证的行过滤模板。
            table_name: 策略对应的物理表名。
            alias: 当前 SELECT 中该表的可选别名。

        Returns:
            exp.Expression: 已渲染并正确限定字段的谓词 AST。

        Raises:
            sqlglot.errors.ParseError: 渲染后的谓词无法按 MySQL 方言解析。
        """
        # 身份值由受控占位符渲染函数转义，避免直接拼接原始用户属性。
        rendered = render_row_filter(
            template,
            user_id=self.user_id,
            username=self.username,
            workspace_id=self.workspace_id,
            role=self.role,
        )

        # 已验证模板在替换真实值后再次解析为 AST，后续通过节点操作安全组合。
        predicate = sqlglot.parse_one(rendered, read="mysql")
        qualifier = alias or table_name

        # 遍历谓词字段，把无前缀或原表名前缀改写为当前查询作用域中的实际别名。
        for column in predicate.find_all(exp.Column):
            current = str(column.table or "")
            if not current or current.lower() == table_name.lower():
                column.set("table", exp.to_identifier(qualifier))
        return predicate

    @staticmethod
    def _nearest_select(node: exp.Expression) -> exp.Select | None:
        """
        沿 AST 父链查找包围指定节点的最近 SELECT。

        核心功能：从节点父级开始逐层向上移动，在遇到第一个 SELECT 时返回，
        用于建立字段、表与嵌套查询之间的词法作用域关系。

        业务逻辑：
            1. 从节点父级开始
            2. 循环向上检查每个祖先
            3. 遇到 SELECT 时立即返回
            4. 到达根节点仍未找到时返回 ``None``

        Args:
            node: 任意 sqlglot AST 节点。

        Returns:
            exp.Select | None: 最近的外层 SELECT，不存在时为 ``None``。
        """
        parent = node.parent

        # 沿父链逐层上溯，确保优先返回最内层直接包含该节点的 SELECT。
        while parent is not None:
            if isinstance(parent, exp.Select):
                return parent
            parent = parent.parent
        return None

    @staticmethod
    def _direct_tables(select: exp.Select) -> list[exp.Table]:
        """
        返回由指定 SELECT 直接引用、而非子查询内部引用的表。

        核心功能：遍历 SELECT 子树中的所有表节点，仅保留最近外层 SELECT 正是当前
        SELECT 的表，从而隔离嵌套查询的授权和行过滤作用域。

        业务逻辑：
            1. 遍历当前 SELECT 子树的全部 Table 节点
            2. 查找每个表最近的外层 SELECT
            3. 仅保留直接属于当前 SELECT 的表

        Args:
            select: 待分析的 SELECT 节点。

        Returns:
            list[exp.Table]: 当前词法作用域直接引用的表节点。
        """
        return [table for table in select.find_all(exp.Table)
                if PolicySnapshot._nearest_select(table) is select]

    @staticmethod
    def _all_selects(tree: exp.Expression) -> list[exp.Select]:
        """
        收集 SQL AST 中的全部 SELECT 作用域。

        核心功能：将生成器结果固化为列表，供行过滤编译逐个处理主查询与子查询。

        业务逻辑：
            1. 遍历 AST 中所有 SELECT 节点
            2. 将结果转换为列表

        Args:
            tree: SQL AST 根表达式。

        Returns:
            list[exp.Select]: 主查询及所有嵌套 SELECT。
        """
        return list(tree.find_all(exp.Select))

    @classmethod
    def _scope_alias_map(cls, select: exp.Select | None) -> dict[str, str]:
        """
        构建当前 SELECT 直接表的限定符到物理表名映射。

        核心功能：同时登记物理表名和 alias_or_name 的小写形式，但只处理当前词法
        作用域的直接表，防止子查询中的同名别名覆盖外层表。

        业务逻辑：
            1. 初始化空别名映射
            2. 若 SELECT 为空则直接返回
            3. 遍历当前 SELECT 的直接表
            4. 分别登记物理表名和查询别名到原表名

        Args:
            select: 当前 SELECT 作用域；允许为 ``None``。

        Returns:
            dict[str, str]: 小写限定符到物理表名的映射。
        """
        aliases: dict[str, str] = {}

        # 无 SELECT 作用域时不存在可解析的局部别名。
        if select is None:
            return aliases

        # 仅登记直接表，避免把更内层子查询的别名泄漏到当前作用域。
        for table in cls._direct_tables(select):
            aliases[str(table.name).lower()] = str(table.name)
            aliases[str(table.alias_or_name).lower()] = str(table.name)
        return aliases

    @classmethod
    def _resolve_qualifier(cls, column: exp.Column) -> str:
        """
        按 SQL 词法作用域由内向外把字段限定符解析为物理表名。

        核心功能：从字段所在 SELECT 开始逐层查询别名映射；内层未解析时继续查找
        外层 SELECT，从而支持相关子查询引用外层表。

        业务逻辑：
            1. 读取字段原始限定符
            2. 找到字段最近的 SELECT
            3. 循环查询当前作用域的别名映射
            4. 命中时返回物理表名
            5. 未命中则移动到更外层 SELECT
            6. 全部作用域均未命中时返回原限定符

        Args:
            column: 带或不带表限定符的字段 AST。

        Returns:
            str: 解析后的物理表名，无法解析时为原限定符。
        """
        qualifier = str(column.table or "")
        select = cls._nearest_select(column)

        # 从最内层 SELECT 开始，逐级向外解析相关子查询可能引用的外层别名。
        while select is not None:
            resolved = cls._scope_alias_map(select).get(qualifier.lower())
            if resolved:
                return resolved
            select = cls._nearest_select(select)
        return qualifier

    @classmethod
    def _is_projection_alias_reference(cls, column: exp.Column) -> bool:
        """
        判断无表前缀字段是否是 ORDER/GROUP/HAVING 中的 SELECT 投影别名。

        核心功能：遵循 MySQL 允许排序、分组和 HAVING 引用投影别名的语义，
        避免把别名误判为物理字段并错误触发列权限拒绝。

        业务逻辑：
            1. 带表限定符的字段不是投影别名引用
            2. 获取字段最近的 SELECT；缺失时返回假
            3. 沿父链查找字段是否位于 ORDER、GROUP 或 HAVING 中
            4. 若不在允许别名的上下文中，则返回假
            5. 收集当前 SELECT 的全部投影别名
            6. 检查字段名是否匹配任一别名

        Args:
            column: 待判断的字段 AST 节点。

        Returns:
            bool: 字段是合法投影别名引用时为真。
        """
        # 显式带表前缀的节点表示物理字段引用，不可能只是投影别名。
        if column.table:
            return False
        select = cls._nearest_select(column)

        # 不属于任何 SELECT 的字段没有可供匹配的投影列表。
        if select is None:
            return False
        parent = column.parent
        alias_context = False

        # 沿父链查找字段是否位于 MySQL 允许引用投影别名的三个子句中。
        while parent is not None and parent is not select:
            if isinstance(parent, (exp.Order, exp.Group, exp.Having)):
                alias_context = True
                break
            parent = parent.parent

        # SELECT 投影本身、WHERE、JOIN 等上下文不能把无前缀字段视为投影别名。
        if not alias_context:
            return False

        # 只收集当前 SELECT 的显式别名，并以小写形式进行匹配。
        aliases = {str(item.alias).lower() for item in select.expressions if item.alias}
        return str(column.name or "").lower() in aliases

    def authorize_sql(self, sql: str) -> str:
        """
        校验 SQL 的表级与列级权限，编译行过滤并返回唯一允许执行的 SQL。

        核心功能：先执行只读安全检查和 AST 解析，再按词法作用域验证物理表、
        通配符及字段访问；全部通过后，把每张直接引用表的行过滤谓词以 AND 写入
        对应 SELECT 的 WHERE，最终由修改后的 AST 生成实际执行文本。

        业务逻辑：
            1. 验证 SQL 只读性并按 MySQL 方言解析 AST
            2. 提取全部物理表，拒绝任一整表越权访问
            3. 检查 ``*``：允许 ``COUNT(*)``，列受限表必须使用显式字段
            4. 遍历普通字段，解析限定符、投影别名及嵌套 SELECT 作用域
            5. 拒绝明确限定字段、单表无前缀字段及多表歧义字段的越权访问
            6. 遍历每个 SELECT，为其直接表渲染并合并全部行过滤谓词
            7. 使用固定 MySQL 方言序列化授权后的 AST

        Args:
            sql: 待授权并编译行级权限的只读 SQL。

        Returns:
            str: 已通过表/列授权且写入行过滤的可执行 SQL。

        Raises:
            SQLSecurityError: SQL 解析失败，或访问被拒绝的表、字段或通配符。
        """
        # =============================================================================
        # 阶段 1：执行只读检查并解析 SQL AST
        # =============================================================================

        # 权限编译前先复用统一 SQL 安全门禁，拒绝写操作和危险语句。
        check_sql_read(sql)
        try:
            tree = sqlglot.parse_one(sql, read="mysql")
        except Exception as exc:
            # 将解析异常统一转换为安全错误，避免未授权 SQL 进入后续执行路径。
            raise SQLSecurityError(f"权限校验解析 SQL 失败：{exc}") from exc

        # =============================================================================
        # 阶段 2：验证全部物理表的访问权限
        # =============================================================================

        # 表名提取覆盖主查询和子查询；任一表被拒绝都会终止整个 SQL。
        used_tables = extract_table_names(sql)
        for table_name in used_tables:
            if not self.is_table_allowed(table_name):
                raise SQLSecurityError(f"数据权限拒绝访问表：{table_name}")

        # =============================================================================
        # 阶段 3：验证通配符不会绕过列级限制
        # =============================================================================

        # 遍历 AST 中全部星号，区分聚合计数、限定通配符和无前缀 SELECT *。
        for star in tree.find_all(exp.Star):

            # COUNT(*) 不直接暴露具体字段，可在表级权限通过后保留。
            if isinstance(star.parent, exp.Count):
                continue

            # ``alias.*`` 需解析到物理表；该表存在任何列限制时必须改用显式列。
            if isinstance(star.parent, exp.Column) and star.parent.table:
                table_name = self._resolve_qualifier(star.parent)
                rule = self.table_rule(table_name)
                if rule["allowed_columns"] is not None:
                    raise SQLSecurityError(f"列权限拒绝通配符访问：{table_name}.*，请改为显式列名")
            elif any(self.table_rule(table)["allowed_columns"] is not None for table in used_tables):
                # 无限定 SELECT * 可能展开任一列受限表，因此只要存在限制就关闭式拒绝。
                raise SQLSecurityError("列权限拒绝 SELECT *，请改为显式列名")

        # =============================================================================
        # 阶段 4：按词法作用域验证普通字段访问
        # =============================================================================

        # 遍历所有 Column 节点，分别处理星号包装、投影别名、限定和未限定字段。
        for column in tree.find_all(exp.Column):

            # ``table.*`` 已在星号阶段统一验证，避免将 ``*`` 当作普通字段重复检查。
            if str(column.name or "") == "*":
                continue

            # ORDER/GROUP/HAVING 中的合法投影别名不是物理列，无需再次做列权限判断。
            if self._is_projection_alias_reference(column):
                continue
            qualifier = str(column.table or "").lower()

            # 带限定符字段先按内外层作用域解析到物理表，再进行精确列授权。
            if qualifier:
                table_name = self._resolve_qualifier(column)
                if not self.is_column_allowed(table_name, column.name):
                    raise SQLSecurityError(f"列权限拒绝访问：{table_name}.{column.name}")
                continue

            # 无限定字段仅在当前 SELECT 的直接表内解析，不把子查询表混入外层作用域。
            select = self._nearest_select(column)
            direct_tables = [str(item.name) for item in self._direct_tables(select)] if select else list(used_tables)
            direct_tables = list(dict.fromkeys(direct_tables))

            # 单表作用域可唯一确定字段归属，直接检查该表的列权限。
            if len(direct_tables) == 1 and not self.is_column_allowed(direct_tables[0], column.name):
                raise SQLSecurityError(f"列权限拒绝访问：{direct_tables[0]}.{column.name}")

            # 多表作用域的无前缀字段存在归属歧义；任一列受限表不允许该字段时即拒绝。
            if len(direct_tables) > 1 and any(
                not self.is_column_allowed(table, column.name)
                for table in direct_tables
                if self.table_rule(table)["allowed_columns"] is not None
            ):
                raise SQLSecurityError(f"列权限拒绝访问未限定字段：{column.name}")

        # =============================================================================
        # 阶段 5：为每个 SELECT 的直接表编译行级过滤
        # =============================================================================

        # 每个 SELECT 只处理其直接引用表。LEFT JOIN 同样在 WHERE 中收紧语义，
        # 确保不会因过滤条件位于 ON 或 WHERE 的差异而泄漏被策略拒绝的行。
        for select in self._all_selects(tree):
            predicates = []

            # 遍历当前作用域直接表，按表别名渲染其全部匹配行过滤模板。
            for table in self._direct_tables(select):
                rule = self.table_rule(str(table.name))
                alias = str(table.alias_or_name) if table.alias else None
                predicates.extend(self._render_filter(item, str(table.name), alias)
                                  for item in rule["row_filters"])

            # 按策略顺序逐个与现有 WHERE 求 AND；无原 WHERE 时直接使用首个谓词。
            for predicate in predicates:
                current = select.args.get("where")
                condition = predicate if current is None else exp.And(
                    this=current.this,
                    expression=predicate,
                )
                select.set("where", exp.Where(this=condition))

        # =============================================================================
        # 阶段 6：序列化唯一允许执行的 SQL
        # =============================================================================

        # 固定使用 MySQL 方言输出，实际执行方必须使用此授权结果而非原始 SQL。
        return tree.sql(dialect="mysql")
