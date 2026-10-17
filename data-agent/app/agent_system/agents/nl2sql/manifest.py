"""NL2SQL Agent 的能力与合同清单。"""

from __future__ import annotations

from dataclasses import dataclass

from app.agent_system.contracts.context import INVOCATION_CONTEXT_VERSION
from app.agent_system.contracts.state import STATE_V2_VERSION
from app.agent_system.contracts import OUTPUT_EVENT_VERSION, QUERY_INPUT_VERSION


@dataclass(frozen=True, slots=True)
class AgentManifest:
    """描述 Agent 身份、合同和必需能力，不持有运行时对象。"""

    agent_id: str
    version: str
    description: str
    input_contract_version: str
    output_contract_version: str
    state_contract_version: str
    context_contract_version: str
    required_capabilities: tuple[str, ...]
    supported_operations: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "agent_id",
            "version",
            "description",
            "input_contract_version",
            "output_contract_version",
            "state_contract_version",
            "context_contract_version",
        ):
            if not str(getattr(self, name) or "").strip():
                raise ValueError(f"{name} 不能为空")
        object.__setattr__(
            self,
            "required_capabilities",
            tuple(dict.fromkeys(self.required_capabilities)),
        )
        object.__setattr__(
            self,
            "supported_operations",
            tuple(dict.fromkeys(self.supported_operations)),
        )


DEFAULT_NL2SQL_MANIFEST = AgentManifest(
    agent_id="data-agent.nl2sql",
    version="2.0.0",
    description="在授权数据边界内把自然语言问题转换为 SQL 并返回分析结果",
    input_contract_version=QUERY_INPUT_VERSION,
    output_contract_version=OUTPUT_EVENT_VERSION,
    state_contract_version=STATE_V2_VERSION,
    context_contract_version=INVOCATION_CONTEXT_VERSION,
    required_capabilities=(
        "embedding",
        "retrieval",
        "metadata",
        "model",
        "sql_execution",
    ),
    supported_operations=(
        "retrieve_metadata",
        "generate_sql",
        "validate_sql",
        "execute_sql",
        "generate_analysis",
    ),
)
