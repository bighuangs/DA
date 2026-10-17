"""
按需分析路由器：数据预测 + 数据异常检测（SSE 流式）。

一句话总述：提供两个独立于主问数流水线的按需端点，前端在图表/表格回答上
            点击"数据预测"或"异常检测"按钮时调用，把当前 query/sql/chart/result
            回传后端，流式返回预测数据行或异常检测报告。
"""
import json

from fastapi import APIRouter, Depends
from starlette.responses import StreamingResponse

from app.adapters.inbound.http.schemas.analysis_schema import AnalysisRequest
from app.application.services.analysis_service import AnalysisService
from app.application.services.masking_service import mask_rows
from app.adapters.inbound.http.dependencies import (
    get_analysis_service,
    get_current_principal,
    get_masking_policy,
    get_masking_rules,
)
from app.application.services.auth_service import Principal
from app.application.services.masking_service import MaskingRule
from app.domain.privacy.masking import MaskingPolicy

analysis_router = APIRouter()


async def _sse_wrap(gen):
    """把 analysis_service 的 dict chunk 序列化为 SSE `data: {json}\\n\\n` 行。"""
    async for chunk in gen:
        yield f"data: {json.dumps(chunk, ensure_ascii=False, default=str)}\n\n"


@analysis_router.post("/api/predict")
async def predict(req: AnalysisRequest,
                  principal: Principal = Depends(get_current_principal),
                  policy: MaskingPolicy = Depends(get_masking_policy),
                  rules: tuple[MaskingRule, ...] = Depends(get_masking_rules),
                  service: AnalysisService = Depends(get_analysis_service)):
    """
    数据预测（SSE 流式）。

    业务逻辑：
        1. 接收 {query, sql, chart, result}（前端从上一轮回答中回传）
        2. 调用 analysis_service.predict 异步生成器
        3. 流式返回 {"stage":"数据预测"} → {"predict":[...]} 或 {"predict_failed":"..."}
    """
    return StreamingResponse(
        _sse_wrap(service.predict(
            req.query,
            req.sql,
            req.chart or {},
            mask_rows(req.result, rules=rules, policy=policy),
        )),
        media_type="text/event-stream",
    )


@analysis_router.post("/api/anomaly")
async def anomaly(req: AnalysisRequest,
                  principal: Principal = Depends(get_current_principal),
                  policy: MaskingPolicy = Depends(get_masking_policy),
                  rules: tuple[MaskingRule, ...] = Depends(get_masking_rules),
                  service: AnalysisService = Depends(get_analysis_service)):
    """
    数据异常检测（SSE 流式）。

    业务逻辑：
        1. 接收 {query, sql, chart, result}
        2. 调用 analysis_service.anomaly 异步生成器
        3. 流式返回 {"stage":"数据异常检测"} → {"anomaly":"..."}
    """
    return StreamingResponse(
        _sse_wrap(service.anomaly(
            req.query,
            req.sql,
            req.chart or {},
            mask_rows(req.result, rules=rules, policy=policy),
        )),
        media_type="text/event-stream",
    )
