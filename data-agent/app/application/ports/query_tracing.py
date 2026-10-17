"""查询 Trace 的运行配置端口。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QueryTraceLLMSettings:
    """一次 Trace 使用的不可变模型成本与展示配置。"""

    default_model: str = "unregistered"
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    cost_currency: str = "USD"


__all__ = ["QueryTraceLLMSettings"]
