from sqlalchemy import String, Text, Boolean, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.mysql.records.base import Base


class TrainingInfoMySQL(Base):
    """
    SQL 样例库（Data Training）持久化模型（meta 库）。

    每行是一对"自然语言问题 → 目标 SQL"样例，供召回节点 recall_training
    做向量近邻检索后，作为 few-shot 示例注入 generate_sql 的 prompt，
    引导大模型对相似问题生成更贴合业务口径的 SQL。
    """
    __tablename__ = "training_info"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
        comment="自增主键"
    )
    workspace_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    question: Mapped[str] = mapped_column(
        String(255),
        comment="自然语言问题样例"
    )
    sql: Mapped[str] = mapped_column(
        Text,
        comment="该问题对应的目标 SQL"
    )
    enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        comment="是否启用（禁用的样例不参与召回与索引构建）"
    )
    created_at: Mapped[str] = mapped_column(
        DateTime,
        server_default=func.now(),
        comment="创建时间"
    )
