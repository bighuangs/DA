"""NL2SQL Agent 的纯定义层。"""

from app.agent_system.contracts.context import (
    INVOCATION_CONTEXT_VERSION,
    InvocationContext,
)
from app.agent_system.agents.nl2sql.manifest import (
    DEFAULT_NL2SQL_MANIFEST,
    AgentManifest,
)
from app.agent_system.contracts.state import (
    STATE_V2_VERSION,
    StateV2,
    deserialize_state,
    new_state,
    serialize_state,
)

__all__ = [
    "AgentManifest",
    "DEFAULT_NL2SQL_MANIFEST",
    "INVOCATION_CONTEXT_VERSION",
    "InvocationContext",
    "STATE_V2_VERSION",
    "StateV2",
    "deserialize_state",
    "new_state",
    "serialize_state",
]
