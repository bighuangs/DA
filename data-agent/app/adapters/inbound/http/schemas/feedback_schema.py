from typing import Optional

from pydantic import BaseModel


class FeedbackSchema(BaseModel):
    """用户反馈提交请求体（TODO-23）。"""
    chat_id: Optional[str] = None
    query: str
    sql: Optional[str] = None
    score: int                # 1=赞，-1=踩
    comment: Optional[str] = None
