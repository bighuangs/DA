"""执行版本快照端口。"""

from __future__ import annotations

from typing import Mapping, Protocol

from app.foundation.capabilities import CapabilityVersionProvider


class ExecutionVersionProvider(CapabilityVersionProvider, Protocol):
    """在一次运行开始时冻结可复现版本，而不是读取最新评测结果。"""

    def snapshot(self, workspace_id: str) -> Mapping[str, object]:
        """返回当前运行的不可变版本快照。"""


class MissingExecutionVersionProvider:
    """显式表示组合根没有注入版本提供器的安全降级实现。"""

    def snapshot(self, workspace_id: str) -> Mapping[str, object]:
        return {
            "snapshot_schema_version": "execution-snapshot-v1",
            "workspace_id": str(workspace_id or ""),
            "prompt_version": "",
            "knowledge_version": "",
            "schema_version": "",
            "index_version": "",
            "model_version": "",
            "run_fingerprint": "",
            "version_complete": False,
            "degradation": "execution_version_provider_not_configured",
        }

    def version_for(self, capability: str) -> str:
        """返回显式的未配置能力版本。"""

        return f"unregistered:{str(capability or 'unknown')}"
