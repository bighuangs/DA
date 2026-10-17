"""会话领域对象与稳定值语义。"""

from app.domain.conversation.errors import ConversationUnavailableError
from app.domain.conversation.fingerprints import memory_query_fingerprint

__all__ = ["ConversationUnavailableError", "memory_query_fingerprint"]
