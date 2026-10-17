import asyncio
import hashlib
import logging
import re
from contextlib import suppress
from inspect import isawaitable
from typing import Any, Awaitable, TypeVar

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.app_config import DBConfig, app_config
from app.domain.query.safety.query_guard import enforce_explain_cost
from app.foundation.capabilities import (
    CapabilityVersionProvider,
    MissingCapabilityVersionProvider,
)
from app.infrastructure.cache.result_cache import result_cache
from app.infrastructure.clients.mysql_query_cancellation import (
    AsyncMyServerQueryCanceller,
    MySQLServerQueryCanceller,
)
from app.foundation.concurrency import singleflight
from app.domain.query.safety.sql_security import check_join_relationships, check_sql_read, enforce_limit
from app.domain.query.result_normalization import normalize_rows
from app.infrastructure.security.masking import (
    MaskingRule,
    mask_rows,
    masking_rules_fingerprint,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")


class DWMysqlRepository:
    def __init__(self, session: AsyncSession, masking_rules: tuple[MaskingRule, ...] = (),
                 cache_scope: str = "unscoped",
                 capability_versions: CapabilityVersionProvider | None = None,
                 server_query_canceller: MySQLServerQueryCanceller | None = None,
                 cancellation_control_timeout_seconds: float | None = None,
                 cancellation_drain_timeout_seconds: float | None = None):
        self.session = session
        self.masking_rules = masking_rules
        self.cache_scope = cache_scope
        self.capability_versions = (
            capability_versions or MissingCapabilityVersionProvider()
        )
        self.last_cache_hit = False
        self.last_server_cancellation: dict[str, object] | None = None
        self._cancellation_control_timeout_seconds = float(
            cancellation_control_timeout_seconds
            if cancellation_control_timeout_seconds is not None
            else app_config.query.mysql_cancel_control_timeout_seconds
        )
        self._cancellation_drain_timeout_seconds = float(
            cancellation_drain_timeout_seconds
            if cancellation_drain_timeout_seconds is not None
            else app_config.query.mysql_cancel_drain_timeout_seconds
        )
        if self._cancellation_control_timeout_seconds <= 0:
            raise ValueError("MySQL 取消控制超时必须大于 0")
        if self._cancellation_drain_timeout_seconds <= 0:
            raise ValueError("MySQL 查询收敛超时必须大于 0")
        self._server_query_canceller = server_query_canceller
        result_cache.max_entries = max(1, app_config.query.cache_max_entries)

    @staticmethod
    def _default_execution_timeout_seconds() -> float:
        """返回没有请求级截止时间时的产品 SQL 执行上限。"""

        phase_budgets = app_config.query.request_phase_budgets_ms or {}
        return float(phase_budgets.get("execution", 6000)) / 1000.0

    @staticmethod
    async def _await_bounded(operation: Awaitable[T], timeout_seconds: float) -> T:
        """以正数秒级预算等待操作；超时会等待内部取消清理完成。"""

        timeout = float(timeout_seconds)
        if timeout <= 0:
            if hasattr(operation, "close"):
                operation.close()
            raise ValueError("MySQL 查询超时必须大于 0")
        return await asyncio.wait_for(operation, timeout=timeout)

    async def _driver_query_identity(self) -> tuple[int, Any]:
        """读取服务端 ID，并保留可在故障时强制断开的 asyncmy 连接。"""

        session_connection = await self.session.connection()
        raw_connection = await session_connection.get_raw_connection()
        driver_connection = getattr(raw_connection, "driver_connection", None)
        thread_id_getter = getattr(driver_connection, "thread_id", None)
        if not callable(thread_id_getter):
            raise RuntimeError("MySQL 驱动不支持服务端查询取消合同")
        thread_id = int(thread_id_getter())
        if thread_id <= 0:
            raise RuntimeError("MySQL 驱动返回了无效 thread_id")
        return thread_id, driver_connection

    def _server_canceller_for_session(self) -> MySQLServerQueryCanceller:
        """构造与当前 Session URL 同源的控制连接，避免误杀另一实例同号会话。"""

        if self._server_query_canceller is not None:
            return self._server_query_canceller
        url = self.session.get_bind().url
        drivername = str(getattr(url, "drivername", ""))
        if drivername != "mysql+asyncmy":
            raise RuntimeError("MySQL 取消控制要求 mysql+asyncmy Session")
        host = str(getattr(url, "host", "") or "").strip()
        user = str(getattr(url, "username", "") or "").strip()
        database = str(getattr(url, "database", "") or "").strip()
        if not host or not user or not database:
            raise RuntimeError("MySQL Session URL 缺少取消控制所需连接身份")
        self._server_query_canceller = AsyncMyServerQueryCanceller(
            DBConfig(
                host=host,
                port=int(getattr(url, "port", None) or 3306),
                user=user,
                password=str(getattr(url, "password", "") or ""),
                database=database,
            ),
            operation_timeout_seconds=self._cancellation_control_timeout_seconds,
        )
        return self._server_query_canceller

    @staticmethod
    def _consume_query_task(query_task: asyncio.Task[Any]) -> None:
        """消费后台任务终态，避免延迟退出产生未检索异常告警。"""

        with suppress(asyncio.CancelledError, Exception):
            query_task.result()

    async def _close_driver_transport(self, driver_connection: Any) -> None:
        """关闭目标连接 transport，作为控制连接 KILL 后的本地强制兜底。"""

        close = getattr(driver_connection, "close", None)
        if not callable(close):
            raise RuntimeError("MySQL 驱动不支持强制关闭目标 transport")
        result = close()
        if isawaitable(result):
            await result

    @staticmethod
    def _remaining_cleanup_seconds(
        cleanup_deadline: float,
        preferred_seconds: float,
        *,
        reserve_seconds: float = 0.0,
    ) -> float:
        """把取消、排空和连接淘汰限制在一个共享清理宽限期内。"""

        remaining = cleanup_deadline - asyncio.get_running_loop().time()
        usable = remaining - max(0.0, reserve_seconds)
        return max(0.001, min(float(preferred_seconds), usable))

    async def _drain_cancelled_query(
        self,
        *,
        thread_id: int,
        driver_connection: Any,
        query_task: asyncio.Task[Any],
    ) -> None:
        """终止服务端会话、收敛业务协程并淘汰原 SQLAlchemy 连接。"""

        outcome: dict[str, object] = {
            "schema": "mysql-server-cancellation-v1",
            "server_cancelled": False,
            "transport_closed": False,
            "query_drained": False,
            "session_invalidated": False,
            "error_type": None,
        }
        cleanup_budget = (
            self._cancellation_control_timeout_seconds
            + self._cancellation_drain_timeout_seconds
        )
        cleanup_deadline = asyncio.get_running_loop().time() + cleanup_budget
        invalidate_reserve = min(0.25, max(0.01, cleanup_budget / 4.0))
        drain_budget = self._cancellation_drain_timeout_seconds
        try:
            await asyncio.wait_for(
                self._server_canceller_for_session().cancel(thread_id),
                timeout=self._remaining_cleanup_seconds(
                    cleanup_deadline,
                    self._cancellation_control_timeout_seconds,
                    reserve_seconds=drain_budget + invalidate_reserve,
                ),
            )
            outcome["server_cancelled"] = True
        except Exception as error:  # noqa: BLE001  # 原始取消语义必须优先返回
            outcome["error_type"] = type(error).__name__
            logger.error(
                "MySQL 服务端查询取消失败 error_type=%s",
                type(error).__name__,
            )

        try:
            await asyncio.wait_for(
                asyncio.shield(query_task),
                timeout=self._remaining_cleanup_seconds(
                    cleanup_deadline,
                    drain_budget / 2.0,
                    reserve_seconds=(drain_budget / 2.0) + invalidate_reserve,
                ),
            )
            outcome["query_drained"] = True
        except asyncio.TimeoutError:
            pass
        except asyncio.CancelledError:
            if not query_task.done():
                raise
            outcome["query_drained"] = True
        except Exception:  # noqa: BLE001  # KILL 后驱动异常即表示查询协程已退出
            outcome["query_drained"] = True

        if not query_task.done():
            query_task.cancel()
            done, _ = await asyncio.wait(
                {query_task},
                timeout=self._remaining_cleanup_seconds(
                    cleanup_deadline,
                    drain_budget / 4.0,
                    reserve_seconds=(drain_budget / 4.0) + invalidate_reserve,
                ),
            )
            outcome["query_drained"] = bool(done)

        if not query_task.done():
            try:
                await self._close_driver_transport(driver_connection)
                outcome["transport_closed"] = True
            except Exception as error:  # noqa: BLE001  # 继续保留原始超时语义
                if outcome["error_type"] is None:
                    outcome["error_type"] = type(error).__name__
                logger.error(
                    "MySQL 目标 transport 强制关闭失败 error_type=%s",
                    type(error).__name__,
                )
            query_task.cancel()
            done, _ = await asyncio.wait(
                {query_task},
                timeout=self._remaining_cleanup_seconds(
                    cleanup_deadline,
                    drain_budget,
                    reserve_seconds=invalidate_reserve,
                ),
            )
            outcome["query_drained"] = bool(done)

        if query_task.done():
            outcome["query_drained"] = True
            self._consume_query_task(query_task)
        else:
            query_task.add_done_callback(self._consume_query_task)
            if outcome["error_type"] is None:
                outcome["error_type"] = "QueryTaskDrainTimeout"
            logger.critical(
                "MySQL 查询协程在 KILL 与 transport close 后仍未退出；"
                "禁止并发 invalidate 当前 Session"
            )

        # AsyncSession 不允许与仍在运行的 execute 并发操作；只有查询协程已经
        # 收敛后才淘汰连接。asyncmy transport close 是该不变量的最终兜底。
        if query_task.done():
            try:
                await asyncio.wait_for(
                    self.session.invalidate(),
                    timeout=self._remaining_cleanup_seconds(
                        cleanup_deadline,
                        self._cancellation_control_timeout_seconds,
                    ),
                )
                outcome["session_invalidated"] = True
            except Exception as error:  # noqa: BLE001  # 不覆盖原始 timeout/cancelled
                if outcome["error_type"] is None:
                    outcome["error_type"] = type(error).__name__
                logger.error(
                    "MySQL 超时连接淘汰失败 error_type=%s",
                    type(error).__name__,
                )
        self.last_server_cancellation = outcome

    async def _execute_cancellable_statement(self, statement: Any) -> Any:
        """执行一个可在调用方取消时同步终止服务端会话的语句。"""

        self.last_server_cancellation = None
        thread_id, driver_connection = await self._driver_query_identity()
        query_task = asyncio.create_task(
            self.session.execute(statement),
            name="dw-mysql-read-query",
        )
        try:
            return await asyncio.shield(query_task)
        except asyncio.CancelledError as cancellation:
            # shield 保留底层查询，先通过独立连接准确终止，再把原始取消语义上抛。
            cleanup_task = asyncio.create_task(
                self._drain_cancelled_query(
                    thread_id=thread_id,
                    driver_connection=driver_connection,
                    query_task=query_task,
                ),
                name="dw-mysql-cancellation-cleanup",
            )
            # 调用方可能重复 cancel 当前请求；清理任务必须继续拥有 Session，
            # 完成 KILL/drain/invalidate 后才恢复最初的取消语义。
            while not cleanup_task.done():
                try:
                    await asyncio.shield(cleanup_task)
                except asyncio.CancelledError:
                    continue
            await cleanup_task
            raise cancellation

    async def get_column_types(self, table_name: str)->dict[str,str]:
        """获取当前表的所有字段的类型"""
        # 定义查询sql
        sql = f"show columns from {table_name}"
        # 执行sql
        result = await self.session.execute(text(sql))
        # 整理数据并返回
        return {row.Field:row.Type for row in result.all()}

    async def get_schema_snapshot(self, *, include_internal: bool = False) -> dict:
        """读取正式 DW Schema。

        默认在 information_schema 查询层排除上传/评测内部表；调用侧仍需再按
        资产目录和访问策略过滤，形成存储层 + 服务层双重隔离。
        """
        internal_filter = "" if include_internal else (
            " AND LEFT(TABLE_NAME, 6) <> 'adhoc_'"
            " AND LEFT(TABLE_NAME, 10) <> 'da_upload_'"
            " AND LEFT(TABLE_NAME, 12) <> 'eval_upload_'"
            " AND LEFT(TABLE_NAME, 5) <> '__da_' "
        )
        table_result = await self.session.execute(text(
            "SELECT TABLE_NAME, TABLE_COMMENT, TABLE_TYPE "
            "FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_TYPE = 'BASE TABLE' "
            + internal_filter +
            "ORDER BY TABLE_NAME"
        ))
        column_result = await self.session.execute(text(
            "SELECT TABLE_NAME, COLUMN_NAME, COLUMN_TYPE, COLUMN_KEY, "
            "COLUMN_COMMENT, ORDINAL_POSITION "
            "FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = DATABASE() "
            + internal_filter +
            "ORDER BY TABLE_NAME, ORDINAL_POSITION"
        ))
        foreign_result = await self.session.execute(text(
            "SELECT TABLE_NAME, COLUMN_NAME "
            "FROM information_schema.KEY_COLUMN_USAGE "
            "WHERE TABLE_SCHEMA = DATABASE() AND REFERENCED_TABLE_NAME IS NOT NULL "
            + internal_filter
        ))
        foreign_keys = {
            (str(row["TABLE_NAME"]), str(row["COLUMN_NAME"]))
            for row in foreign_result.mappings().all()
        }
        tables: dict[str, dict] = {}
        for row in table_result.mappings().all():
            name = str(row["TABLE_NAME"])
            tables[name] = {
                "id": name,
                "name": name,
                "role": _infer_table_role(name),
                "description": row.get("TABLE_COMMENT") or "",
                "columns": [],
            }
        for row in column_result.mappings().all():
            table_name = str(row["TABLE_NAME"])
            if table_name not in tables:
                continue
            column_name = str(row["COLUMN_NAME"])
            column_key = str(row.get("COLUMN_KEY") or "")
            role = "primary_key" if column_key == "PRI" else (
                "foreign_key" if (table_name, column_name) in foreign_keys else "dimension"
            )
            tables[table_name]["columns"].append({
                "id": f"{table_name}.{column_name}",
                "name": column_name,
                "type": str(row.get("COLUMN_TYPE") or ""),
                "role": role,
                "description": row.get("COLUMN_COMMENT") or "",
                "alias": [],
                "examples": [],
                "sync": False,
                "table_id": table_name,
            })
        return {
            "source": self.session.get_bind().url.database,
            "asset_scope": "official_only" if not include_internal else "all",
            "tables": list(tables.values()),
        }

    async def get_column_values(self, table_name:str, column_name:str, limit:int=10) ->list:
        """查询指定字段的前limit个字段值"""
        # 定义查询sql
        sql = f"select distinct {column_name} from {table_name} limit {limit}"
        # 执行sql
        result = await self.session.execute(text(sql))
        # 整理数据并返回
        return result.scalars().all()

    async def get_db_infos(self) ->dict[str,str]:
        result = await self.session.execute(text("select version()"))
        version = result.scalar()
        dialect = self.session.get_bind().dialect.name
        return {"version":version, "dialect":dialect}

    async def validate_sql(
        self,
        sql: str,
        *,
        timeout_seconds: float | None = None,
    ):
        """
        校验 SQL 语法是否合法（仅校验，不执行）。

        业务逻辑：
            1. 先做只读安全校验（check_sql_read）：拦截写操作 / DDL / 危险函数
            2. 拒绝未通过关联谓词连通的多表查询
            3. 再用 EXPLAIN 做语法与执行计划校验
            4. 按预计扫描行数执行查询成本守卫
        """
        check_sql_read(sql)
        check_join_relationships(sql)
        effective_timeout = (
            self._default_execution_timeout_seconds()
            if timeout_seconds is None
            else float(timeout_seconds)
        )
        result = await self._await_bounded(
            self._execute_cancellable_statement(text(f"EXPLAIN {sql}")),
            effective_timeout,
        )
        explain_rows = [dict(row) for row in result.mappings().all()]
        estimated_rows = enforce_explain_cost(
            explain_rows,
            max_explain_rows=app_config.query.max_explain_rows,
            enabled=app_config.query.cost_guard_enabled,
        )
        return {"explain_rows": explain_rows, "estimated_rows": estimated_rows}

    async def execute_sql(
        self,
        sql: str,
        *,
        timeout_seconds: float | None = None,
    ):
        """
        执行一条查询的 SQL。

        业务逻辑：
            1. 先做只读安全校验（check_sql_read）：防御深度，防止绕过校验节点的恶意 SQL
            2. 再次拒绝未关联多表查询，防止绕过 validate_sql 直接执行
            3. 强制附加行数上限（enforce_limit）：SQL 无 LIMIT 时包装子查询追加 LIMIT，
               防止大结果集拖垮服务（上限取自 app_config.query.limit_rows，默认 1000）
            3. 再真正执行查询并返回字典列表
        """
        check_sql_read(sql)
        check_join_relationships(sql)
        safe_sql = enforce_limit(sql, app_config.query.limit_rows)
        effective_timeout = (
            self._default_execution_timeout_seconds()
            if timeout_seconds is None
            else float(timeout_seconds)
        )
        if effective_timeout <= 0:
            raise ValueError("MySQL 查询超时必须大于 0")
        # 相同 SQL 的不同模型候选必须使用不同缓存命名空间；版本由组合根注入，
        # 仓储不再反向读取 Agent 的路由实现。
        model_namespace = str(
            self.capability_versions.version_for("llm.sql")
            or "unregistered:llm.sql"
        )
        cache_key = hashlib.sha256(
            "|".join([
                "dw",
                self.cache_scope,
                masking_rules_fingerprint(self.masking_rules),
                str(self.session.get_bind().url.database or ""),
                str(app_config.query.limit_rows),
                model_namespace,
                re.sub(r"\s+", " ", safe_sql.strip()).lower(),
            ]).encode("utf-8")
        ).hexdigest()
        if app_config.query.cache_enabled:
            cached = result_cache.get(cache_key, app_config.query.cache_ttl_seconds)
            if cached is not None:
                self.last_cache_hit = True
                return cached
        self.last_cache_hit = False
        async def load_rows():
            # A second lookup closes the race between the first miss and leader election.
            cached_after_wait = result_cache.get(cache_key, app_config.query.cache_ttl_seconds)
            if cached_after_wait is not None:
                return cached_after_wait
            result = await self._await_bounded(
                self._execute_cancellable_statement(text(safe_sql)),
                effective_timeout,
            )
            rows = mask_rows(
                normalize_rows([dict(row) for row in result.mappings().all()]),
                rules=self.masking_rules,
            )
            result_cache.set(cache_key, rows, app_config.query.cache_ttl_seconds)
            return rows

        if app_config.query.cache_enabled:
            return await singleflight.run(
                cache_key,
                load_rows,
                follower_timeout_seconds=effective_timeout,
            )
        result = await self._await_bounded(
            self._execute_cancellable_statement(text(safe_sql)),
            effective_timeout,
        )
        return mask_rows(
            normalize_rows([dict(row) for row in result.mappings().all()]),
            rules=self.masking_rules,
        )

    async def execute_sql_for_evaluation(
        self,
        sql: str,
        *,
        max_rows: int = 10_000,
        timeout_seconds: float | None = None,
    ) -> list[dict]:
        """Execute one read-only query for complete-result evaluation.

        The product query path intentionally caps browser results at
        ``query.limit_rows``.  Reusing that truncated payload for golden-result
        comparison can create both false passes and false regressions when the
        true result has more rows, especially when the cut-off falls inside an
        ORDER BY tie.  The evaluation path therefore re-executes generated and
        reference SQL under a larger, explicit hard cap and rejects an
        incomplete denominator instead of silently comparing prefixes.

        This method is deliberately not used by API routes.  It keeps the same
        read-only and join-connectivity guards as normal execution, bypasses the
        product result cache, and fetches one sentinel row so callers can prove
        that the observed result is complete.
        """
        check_sql_read(sql)
        check_join_relationships(sql)
        if max_rows <= 0:
            raise ValueError("评测完整结果行数上限必须大于 0")
        normalized_sql = str(sql or "").strip().removesuffix(";").strip()
        bounded_sql = (
            "SELECT * FROM ("
            f"{normalized_sql}"
            f") AS _da_eval_sub LIMIT {int(max_rows) + 1}"
        )
        effective_timeout = (
            app_config.query.evaluation_sql_timeout_seconds
            if timeout_seconds is None
            else float(timeout_seconds)
        )
        result = await self._await_bounded(
            self._execute_cancellable_statement(text(bounded_sql)),
            effective_timeout,
        )
        raw_rows = [dict(row) for row in result.mappings().all()]
        if len(raw_rows) > max_rows:
            raise ValueError(
                f"评测结果超过完整性上限 {max_rows} 行；已拒绝使用截断结果计分"
            )
        return mask_rows(
            normalize_rows(raw_rows),
            rules=self.masking_rules,
        )

    """
    序列化：将某种语言的对象或数组转为特定格式的字符串（json/yaml）
    反序列化：将特定格式的字符串转换为某种语言的对象或数组
    """


def _infer_table_role(table_name: str) -> str:
    """按常见命名给新表提供可解释的默认角色，后续可在后台修正。"""
    name = table_name.lower()
    if name.startswith("fact_") or name.startswith("f_"):
        return "fact"
    if name.startswith("dim_") or name.startswith("d_"):
        return "dim"
    return "unknown"
