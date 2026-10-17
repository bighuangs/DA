"""
用户反馈数据访问层（meta 库）。

一句话总述：封装 feedback 表的写入与查询，供反馈采集端点与后台 review 使用。
"""
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.persistence.mysql.records.feedback_mysql import FeedbackMySQL


class FeedbackRepository:
    def __init__(self, session: AsyncSession, owner_id: str | None = None, workspace_id: str | None = None):
        self.session = session
        self.owner_id = owner_id
        self.workspace_id = workspace_id

    async def create(self, chat_id: Optional[str], query: str, sql: Optional[str],
                     score: int, comment: Optional[str] = None) -> dict:
        """写入一条反馈。"""
        fb = FeedbackMySQL(
            chat_id=chat_id,
            query=query,
            sql=sql,
            score=score,
                    comment=comment, owner_id=self.owner_id, workspace_id=self.workspace_id,
        )
        self.session.add(fb)
        await self.session.flush()
        await self.session.refresh(fb)
        await self.session.commit()
        return {
            "id": fb.id,
            "chat_id": fb.chat_id,
            "query": fb.query,
            "sql": fb.sql,
            "score": fb.score,
            "comment": fb.comment,
            "created_at": str(fb.created_at) if fb.created_at else None,
        }

    async def list(self, score: Optional[int] = None, limit: int = 100) -> list[dict]:
        """查询反馈列表；可按 score 过滤（如只看差评 score=-1）。"""
        stmt = select(FeedbackMySQL).order_by(FeedbackMySQL.created_at.desc()).limit(limit)
        if self.workspace_id:
            stmt = stmt.where(FeedbackMySQL.workspace_id == self.workspace_id)
        if score is not None:
            stmt = stmt.where(FeedbackMySQL.score == score)
        result = await self.session.execute(stmt)
        rows = result.scalars().all()
        return [
            {
                "id": r.id,
                "chat_id": r.chat_id,
                "query": r.query,
                "sql": r.sql,
                "score": r.score,
                "comment": r.comment,
                "created_at": str(r.created_at) if r.created_at else None,
            }
            for r in rows
        ]
