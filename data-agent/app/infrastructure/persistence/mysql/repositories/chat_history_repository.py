import json
from datetime import datetime

from sqlalchemy import Select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.observability.logging import logger
from app.infrastructure.persistence.mysql.records.chat_history_mysql import ChatHistoryMySQL
from app.infrastructure.security.masking import mask_sql_literals


class ChatHistoryRepository:
    """
    多轮对话历史的持久层仓库（meta MySQL）。

    一句话总述：封装 chat_history 表的读写，供 Service 层加载历史与存回合（含结果数据）。

    业务逻辑：
        1. get_recent_history：按 chat_id 取最近 limit 轮，升序返回（旧→新），
           便于 generate_sql 节点按时间顺序注入 prompt
        2. save_turn：把当前一轮 (chat_id, query, sql, result) 写入 chat_history，
           result 以 JSON 字符串持久化（Decimal/datetime 用 default=str 兜底）
    """

    def __init__(self, session: AsyncSession, owner_id: str | None = None, workspace_id: str | None = None):
        """
        构造对话历史仓库，绑定 SQLAlchemy 异步会话与租户上下文。

        Args:
            session: SQLAlchemy AsyncSession，所有读写均通过该会话执行
            owner_id: 调用者用户 ID，非空时作为查询过滤条件，实现用户级数据隔离
            workspace_id: 工作空间 ID，非空时作为查询过滤条件，实现工作空间级隔离
        """
        self.session = session
        self.owner_id = owner_id
        self.workspace_id = workspace_id

    async def get_recent_history(
        self,
        chat_id: str,
        limit: int = 5,
        *,
        after: datetime | None = None,
    ) -> list[ChatHistoryMySQL]:
        """
        获取指定会话最近 limit 轮历史（按时间升序，旧→新）。

        业务逻辑：
            1. 按 chat_id 过滤
            2. 按创建时间降序取最近 limit 条（最新的在前）
            3. 反转为升序返回，便于按对话顺序注入 prompt

        Args:
            chat_id: 会话ID
            limit: 取最近几轮（默认 5）

        Returns:
            list[ChatHistoryMySQL]：按时间升序排列的历史记录列表
        """
        # =============================================================================
        # 阶段 1：构造基础查询语句（按 chat_id 过滤）
        # =============================================================================
        statement = Select(ChatHistoryMySQL).where(ChatHistoryMySQL.chat_id == chat_id)
        # 租户隔离：owner_id / workspace_id 非空时追加过滤，避免跨租户读到历史
        if self.owner_id:
            statement = statement.where(ChatHistoryMySQL.owner_id == self.owner_id)
        if self.workspace_id:
            statement = statement.where(ChatHistoryMySQL.workspace_id == self.workspace_id)
        if after is not None:
            statement = statement.where(ChatHistoryMySQL.created_at > after)
        # =============================================================================
        # 阶段 2：按创建时间降序取最近 limit 条，并反转为升序返回
        # =============================================================================
        safe_limit = max(1, min(int(limit), 100))
        result = await self.session.execute(
            statement.order_by(
                desc(ChatHistoryMySQL.created_at), desc(ChatHistoryMySQL.id)
            ).limit(safe_limit)
        )
        rows = result.scalars().all()
        # 数据库取的是降序（最新在前），反转为升序（旧→新）便于按顺序注入
        return list(reversed(rows))

    async def save_turn(self, chat_id: str, query: str, sql: str, result: list = None,
                        *, standalone_question: str | None = None,
                        follow_up_action: str | None = None,
                        semantic_state: dict | None = None,
                        commit: bool = True) -> ChatHistoryMySQL:
        """
        保存一轮对话（用户提问 + 生成的SQL + 查询结果）到 chat_history 表。

        业务逻辑：
            1. 把 result 序列化为 JSON 字符串（default=str 兜底 Decimal/datetime 等）
            2. 计算 row_count
            3. 构造 ChatHistoryMySQL 对象并 add 到当前 session
            4. commit 提交，并记录 INFO 日志便于核对持久化是否真的发生

        Args:
            chat_id: 会话ID
            query: 用户自然语言提问
            sql: 大模型生成的最终 SQL
            result: 该轮查询结果行（list[dict]），可为 None（如执行失败）
        """
        # =============================================================================
        # 阶段 1：序列化结果并计算行数
        # =============================================================================
        # default=str 兜底 Decimal/datetime 等非原生 JSON 类型；ensure_ascii=False 保留中文
        result_json = json.dumps(result, ensure_ascii=False, default=str) if result is not None else None
        row_count = len(result) if result is not None else None
        # =============================================================================
        # 阶段 2：构造 ORM 对象并加入会话
        # =============================================================================
        # mask_sql_literals 对 SQL 中的字面量做脱敏，避免敏感值落库
        # standalone_question / follow_up_action 做长度截断，防止超长文本溢出列定义
        row = ChatHistoryMySQL(
            chat_id=chat_id, query=query, sql=mask_sql_literals(sql),
            result=result_json, row_count=row_count, owner_id=self.owner_id,
            workspace_id=self.workspace_id,
            standalone_question=(standalone_question or query)[:2000],
            follow_up_action=(follow_up_action or "new_topic")[:32],
            semantic_state=semantic_state,
        )
        self.session.add(row)
        # =============================================================================
        # 阶段 3：提交事务并记录日志
        # =============================================================================
        if commit:
            await self.session.commit()
        else:
            await self.session.flush()
        logger.info(f"保存对话历史成功：chat_id={chat_id} row_count={row_count}")
        return row
