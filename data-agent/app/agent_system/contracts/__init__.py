"""Agent 输入、输出与执行快照合同。"""

from app.agent_system.contracts.execution import (
    ERROR_CONTRACT_VERSION,
    EXECUTION_SNAPSHOT_VERSION,
    OUTPUT_EVENT_VERSION,
    QUERY_INPUT_VERSION,
    Error,
    ExecutionError,
    ExecutionSnapshot,
    ExecutionStatus,
    OutputEvent,
    OutputEventType,
    QueryInput,
)
from app.agent_system.contracts.context import (
    INVOCATION_CONTEXT_VERSION,
    InvocationContext,
)
from app.agent_system.contracts.state import (
    STATE_V2_VERSION,
    StateV2,
    deserialize_state,
    new_state,
    serialize_state,
    validate_state,
)

__all__ = [
    "ERROR_CONTRACT_VERSION",
    "EXECUTION_SNAPSHOT_VERSION",
    "OUTPUT_EVENT_VERSION",
    "QUERY_INPUT_VERSION",
    "Error",
    "ExecutionError",
    "ExecutionSnapshot",
    "ExecutionStatus",
    "INVOCATION_CONTEXT_VERSION",
    "InvocationContext",
    "OutputEvent",
    "OutputEventType",
    "QueryInput",
    "STATE_V2_VERSION",
    "StateV2",
    "deserialize_state",
    "new_state",
    "serialize_state",
    "validate_state",
]
