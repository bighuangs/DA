"""Async Redis lifecycle for TODO-59 short-term conversation memory."""
from __future__ import annotations

from typing import Any

from redis.asyncio import Redis

from app.config.app_config import RedisConfig, app_config


class RedisClientManager:
    """Own one bounded asyncio Redis pool and expose privacy-safe health state."""

    def __init__(self, config: RedisConfig):
        self.config = config
        self.client: Redis | None = None
        self._last_error_type: str | None = None

    def init(self) -> None:
        password = str(self.config.password or "").strip() or None
        self.client = Redis(
            host=self.config.host,
            port=int(self.config.port),
            db=int(self.config.database),
            password=password,
            decode_responses=True,
            socket_timeout=float(self.config.socket_timeout_seconds),
            socket_connect_timeout=float(self.config.connect_timeout_seconds),
            max_connections=max(1, int(self.config.max_connections)),
            health_check_interval=30,
        )

    async def ping(self) -> bool:
        if self.client is None:
            self._last_error_type = "NotInitialized"
            return False
        try:
            healthy = bool(await self.client.ping())
        except Exception as exc:
            self._last_error_type = type(exc).__name__
            return False
        self._last_error_type = None
        return healthy

    def status(self) -> dict[str, Any]:
        return {
            "configured": True,
            "initialized": self.client is not None,
            "host": self.config.host,
            "port": int(self.config.port),
            "database": int(self.config.database),
            "required_on_startup": bool(self.config.required_on_startup),
            "last_error_type": self._last_error_type,
        }

    async def close(self) -> None:
        client, self.client = self.client, None
        if client is not None:
            await client.aclose()


redis_client_manager = RedisClientManager(app_config.redis)
