from sqlalchemy import String, Text, DateTime, Integer, func
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.mysql.records.base import Base


class FeedbackMySQL(Base):
    """
    用户反馈持久化模型（meta 库）。

    每条回答可被用户标记 👍(score=1)/👎(score=-1) 并附可选评论；
    差评（score=-1）定期人工 review，提炼为 training 样例（TODO-8）补回知识库，
    形成"反馈→评测→知识库"的质量闭环（TODO-23）。
    """
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="自增主键"
    )
    chat_id: Mapped[str] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        comment="会话ID（关联 chat_history.chat_id）"
    )
    owner_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    workspace_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    query: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="用户自然语言提问"
    )
    sql: Mapped[str] = mapped_column(
        Text,
        nullable=True,
        comment="该轮生成的 SQL（便于差评定位）"
    )
    score: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="反馈分：1=赞，-1=踩"
    )
    comment: Mapped[str] = mapped_column(
        Text,
        nullable=True,
        comment="可选评论（如差评原因）"
    )
    review_status: Mapped[str] = mapped_column(String(32), default="pending", server_default="pending", index=True)
    review_category: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    reviewed_at: Mapped[str | None] = mapped_column(DateTime, nullable=True)
    source_trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    candidate_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    candidate_status: Mapped[str] = mapped_column(String(32), default="none", server_default="none", index=True)
    candidate_question: Mapped[str | None] = mapped_column(Text, nullable=True)
    candidate_sql: Mapped[str | None] = mapped_column(Text, nullable=True)
    candidate_training_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    candidate_confirmed_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    candidate_confirmed_at: Mapped[str | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[str] = mapped_column(
        DateTime,
        server_default=func.now(),
        comment="创建时间"
    )
