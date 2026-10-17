from sqlalchemy import JSON, String, Text, DateTime, Integer, Index, func
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.mysql.records.base import Base


class ChatHistoryMySQL(Base):
    """
    多轮对话历史持久化模型（meta 库）。

    每一轮问答对应一行：记录 chat_id、用户提问、生成的 SQL、查询结果、结果行数、创建时间。
    - generate_sql 节点据此注入最近 N 轮 (query, sql) 作为上下文；
    - result / row_count 保留该轮查询的真实结果数据，供历史回看、结果复现与后续分析节点使用。
    """
    __tablename__ = "chat_history"
    __table_args__ = (
        Index(
            "ix_chat_history_scope_chat_order",
            "owner_id", "workspace_id", "chat_id", "created_at", "id",
        ),
    )

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
        comment="自增主键"
    )
    chat_id: Mapped[str] = mapped_column(
        String(64),
        index=True,
        comment="会话ID（一次多轮对话共享一个 chat_id）"
    )
    owner_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    workspace_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    query: Mapped[str] = mapped_column(
        Text,
        comment="用户自然语言提问"
    )
    sql: Mapped[str] = mapped_column(
        Text,
        comment="大模型生成的 SQL（成功执行的最终 SQL）"
    )
    result: Mapped[str] = mapped_column(
        Text,
        nullable=True,
        comment="该轮查询的结果数据（JSON 字符串，含 Decimal/datetime 的 default=str 序列化）"
    )
    row_count: Mapped[int] = mapped_column(
        Integer,
        nullable=True,
        comment="结果行数（便于快速统计与展示）"
    )
    standalone_question: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="TODO-44 闭合后的独立问题"
    )
    follow_up_action: Mapped[str | None] = mapped_column(
        String(32), nullable=True, comment="new_topic/inherit/replace/append/remove/reset"
    )
    semantic_state: Mapped[dict | None] = mapped_column(
        JSON, nullable=True, comment="不含历史结果值的成功轮结构化语义状态"
    )
    created_at: Mapped[str] = mapped_column(
        DateTime,
        server_default=func.now(),
        comment="创建时间"
    )
