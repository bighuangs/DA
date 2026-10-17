from typing import Optional

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    """按需分析请求体（数据预测 / 数据异常检测共用）。"""
    query: str = Field(min_length=1, max_length=2000)
    sql: str = Field(min_length=1, max_length=20000)
    chart: Optional[dict] = None   # 图表配置（可能为 table）
    result: list[dict] = Field(default_factory=list, max_length=5000)
