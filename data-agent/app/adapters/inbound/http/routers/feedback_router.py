"""
用户反馈路由器（TODO-23 反馈闭环）。

一句话总述：提供反馈采集与查询端点；前端在回答卡上点👍👎调用 POST /api/feedback，
            差评定期人工 review 补入 training 样例库，形成质量闭环。
"""
from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.adapters.inbound.http.dependencies import get_feedback_service
from app.adapters.inbound.http.schemas.feedback_schema import FeedbackSchema
from app.adapters.inbound.http.dependencies import get_current_principal
from app.application.services.auth_service import Principal
from app.application.services.feedback_service import FeedbackService
from app.adapters.inbound.http.dependencies import require_roles

feedback_router = APIRouter()


@feedback_router.post("/api/feedback")
async def submit_feedback(
    body: FeedbackSchema,
    service: FeedbackService = Depends(get_feedback_service),
    principal: Principal = Depends(get_current_principal),
):
    """提交一条用户反馈（👍/👎 + 可选评论）。"""
    return await service.submit(
        chat_id=body.chat_id, query=body.query, sql=body.sql,
        score=body.score, comment=body.comment,
    )


@feedback_router.get("/api/feedback")
async def list_feedback(
    score: Optional[int] = Query(default=None, description="按分数过滤：1=赞，-1=踩"),
    limit: int = Query(default=100, ge=1, le=1000),
    service: FeedbackService = Depends(get_feedback_service),
    principal: Principal = Depends(require_roles("owner", "admin")),
):
    """查询反馈列表（后台 review 用，可只看差评 score=-1）。"""
    return await service.list_feedback(score=score, limit=limit)
