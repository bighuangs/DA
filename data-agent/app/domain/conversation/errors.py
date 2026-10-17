"""会话生命周期领域错误。"""


class ConversationUnavailableError(ValueError):
    """会话不存在、已归档、已重置或正被删除。"""


__all__ = ["ConversationUnavailableError"]
