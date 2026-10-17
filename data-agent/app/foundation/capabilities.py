"""运行能力版本的内层合同。"""

from __future__ import annotations

from typing import Protocol


class CapabilityVersionProvider(Protocol):
    """按能力名称返回可用于隔离缓存与追踪的稳定版本。"""

    def version_for(self, capability: str) -> str:
        """返回不含凭据的版本标识。"""


class MissingCapabilityVersionProvider:
    """组合根尚未注入版本提供器时的显式安全降级。"""

    def version_for(self, capability: str) -> str:
        return f"unregistered:{str(capability or 'unknown')}"


__all__ = ["CapabilityVersionProvider", "MissingCapabilityVersionProvider"]
