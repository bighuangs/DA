from typing import Optional
from elasticsearch import AsyncElasticsearch

from app.config.app_config import ESConfig, app_config


class ESClientManager:
    def __init__(self, config: ESConfig):
        self.config = config
        self.client: Optional[AsyncElasticsearch] = None

    def init(self):
        self.client = AsyncElasticsearch(self._get_url())

    def _get_url(self):
        return f"http://{self.config.host}:{self.config.port}"

    async def close(self):
        if self.client is not None:
            await self.client.close()
        self.client = None

es_client_manager = ESClientManager(app_config.es)
