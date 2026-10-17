"""评测大产物的存储协议。"""

from __future__ import annotations

from typing import Mapping, Protocol

from evals.contracts import ArtifactRef


class ArtifactStore(Protocol):
    """可替换为本地、对象存储或受控外部服务的产物接口。"""

    def write_bytes(
        self,
        evaluation_id: str,
        relative_path: str,
        data: bytes,
        *,
        media_type: str = "application/octet-stream",
    ) -> ArtifactRef:
        """原子写入二进制产物。"""

    def write_json(
        self,
        evaluation_id: str,
        relative_path: str,
        payload: Mapping[str, object] | list[object],
    ) -> ArtifactRef:
        """以稳定 UTF-8 JSON 写入产物。"""

    def read_bytes(self, evaluation_id: str, relative_path: str) -> bytes:
        """读取指定评测下的产物。"""
