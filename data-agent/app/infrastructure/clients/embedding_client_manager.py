from typing import Optional

from app.infrastructure.clients.hybrid_embedding_client import (
    HybridEmbeddingClient,
    HybridEmbeddingRuntimeError,
)
from app.config.app_config import EmbeddingConfig, app_config


class EmbeddingClientManager:
    """Lifecycle wrapper for Data Agent's own local BGE-M3 runtime."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.client: Optional[HybridEmbeddingClient] = None
        self._last_close_error: str | None = None

    def init(self):
        if self._last_close_error is not None:
            raise HybridEmbeddingRuntimeError(
                "上一个 BGE-M3 客户端未完成排空，"
                "禁止初始化新实例；请等待并重试 close()："
                f"{self._last_close_error}"
            )
        if self.client is None:
            self.client = HybridEmbeddingClient(self.config)
        elif not getattr(self.client, "accepting_requests", True):
            raise HybridEmbeddingRuntimeError(
                "当前 BGE-M3 客户端正在关闭或已 fail-close，"
                "禁止复用或替换实例"
            )
        return self.client

    async def close(self):
        client = self.client
        if client is None:
            return
        try:
            await client.close()
        except BaseException as exc:
            # Retain the poisoned client so init() cannot create a second model
            # while an executor future from the first one is still alive.
            self._last_close_error = f"{type(exc).__name__}: {exc}"
            raise
        else:
            if self.client is client:
                self.client = None
            self._last_close_error = None

# 创建客户端管理器
embedding_client_manager = EmbeddingClientManager(app_config.embedding)
