"""按需数据分析的可选模型能力端口。"""

from __future__ import annotations

from typing import Protocol


class AnalysisModelPort(Protocol):
    """确定性分析无法覆盖时使用的模型增强能力。"""

    async def predict(
        self,
        *,
        query: str,
        sql: str,
        chart: dict,
        result: list[dict],
    ) -> str: ...

    async def anomaly(
        self,
        *,
        query: str,
        sql: str,
        chart: dict,
        result: list[dict],
    ) -> str: ...


__all__ = ["AnalysisModelPort"]
