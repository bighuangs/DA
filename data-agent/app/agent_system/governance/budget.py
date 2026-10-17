"""端到端执行预算合同。"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class BudgetPolicy:
    """限制一次执行可消耗的时间、模型调用和数据规模。"""

    total_timeout_ms: int = 30_000
    max_node_attempts: int = 64
    max_model_calls: int = 8
    max_input_tokens: int = 32_000
    max_output_tokens: int = 8_000
    max_result_rows: int = 1_000
    completion_reserve_ms: int = 500

    def __post_init__(self) -> None:
        for name in (
            "total_timeout_ms",
            "max_node_attempts",
            "max_model_calls",
            "max_input_tokens",
            "max_output_tokens",
            "max_result_rows",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} 必须大于 0")
        if not 0 <= self.completion_reserve_ms < self.total_timeout_ms:
            raise ValueError("completion_reserve_ms 必须位于 [0, total_timeout_ms) 区间")

    @property
    def total_timeout_seconds(self) -> float:
        return self.total_timeout_ms / 1_000


class BudgetExceeded(RuntimeError):
    """某个治理预算已耗尽。"""

    def __init__(self, dimension: str) -> None:
        super().__init__(f"Agent 执行预算已耗尽：{dimension}")
        self.dimension = dimension


@dataclass(frozen=True, slots=True)
class BudgetUsage:
    """不可变预算账本；每次消费都会产生新快照。"""

    node_attempts: int = 0
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    result_rows: int = 0

    def consume(
        self,
        policy: BudgetPolicy,
        *,
        node_attempts: int = 0,
        model_calls: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        result_rows: int = 0,
    ) -> BudgetUsage:
        increments = {
            "node_attempts": node_attempts,
            "model_calls": model_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "result_rows": result_rows,
        }
        if any(value < 0 for value in increments.values()):
            raise ValueError("预算消费量不能为负数")
        candidate = replace(
            self,
            **{
                name: getattr(self, name) + increment
                for name, increment in increments.items()
            },
        )
        limits = {
            "node_attempts": policy.max_node_attempts,
            "model_calls": policy.max_model_calls,
            "input_tokens": policy.max_input_tokens,
            "output_tokens": policy.max_output_tokens,
            "result_rows": policy.max_result_rows,
        }
        for name, limit in limits.items():
            if getattr(candidate, name) > limit:
                raise BudgetExceeded(name)
        return candidate

    def to_dict(self) -> dict[str, int]:
        return {
            "node_attempts": self.node_attempts,
            "model_calls": self.model_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "result_rows": self.result_rows,
        }
