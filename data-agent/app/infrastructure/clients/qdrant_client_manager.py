from typing import Optional

from qdrant_client import AsyncQdrantClient

from app.config.app_config import QdrantConfig, app_config


class QdrantClientManager:
    """
    进行向量存储和搜索的qdrant的客户端器类
    """
    def __init__(self, config: QdrantConfig):
        self.config = config
        self.client: Optional[AsyncQdrantClient] = None

    def _get_url(self):
        return f"http://{self.config.host}:{self.config.port}"

    def init(self):
        self.client = AsyncQdrantClient(
            url=self._get_url(),
            # httpx 还会读取 Windows 系统代理；本机/内网数据库连接默认直连。
            trust_env=bool(getattr(self.config, "trust_env", False)),
            # lifespan 会立即执行真实 get_collections 探活；构造期的同步
            # compatibility probe 无法继承上述代理设置，会产生虚假 502 warning。
            check_compatibility=False,
        )

    async def close(self):
        if self.client is not None:
            await self.client.close()
        self.client = None


# 创建客户端管理器
qdrant_client_manager = QdrantClientManager(app_config.qdrant)
