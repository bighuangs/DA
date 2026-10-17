"""把进程配置转换为 Application 可注入的不可变快照。"""

from app.infrastructure.config.conversation import load_conversation_settings

__all__ = ["load_conversation_settings"]
