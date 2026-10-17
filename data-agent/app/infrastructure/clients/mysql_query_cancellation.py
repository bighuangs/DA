"""MySQL 服务端查询取消控制面。"""

from __future__ import annotations

from typing import Protocol

import asyncmy

from app.config.app_config import DBConfig


class MySQLServerQueryCanceller(Protocol):
    """使用独立连接终止指定 MySQL 会话的最小端口。"""

    async def cancel(self, thread_id: int) -> None: ...


class AsyncMyServerQueryCanceller:
    """通过不共享业务连接池的 asyncmy 控制连接终止目标会话。"""

    def __init__(self, config: DBConfig, *, operation_timeout_seconds: float):
        if operation_timeout_seconds <= 0:
            raise ValueError("MySQL 取消控制超时必须大于 0")
        self._config = config
        self._operation_timeout_seconds = float(operation_timeout_seconds)

    async def cancel(self, thread_id: int) -> None:
        """终止一个已确认的 MySQL 连接 ID，不复用被取消的业务连接。"""

        target_thread_id = int(thread_id)
        if target_thread_id <= 0:
            raise ValueError("MySQL thread_id 必须大于 0")

        connection = await asyncmy.connect(
            host=self._config.host,
            port=self._config.port,
            user=self._config.user,
            password=self._config.password,
            database=self._config.database,
            charset="utf8mb4",
            autocommit=True,
            connect_timeout=self._operation_timeout_seconds,
            read_timeout=self._operation_timeout_seconds,
        )
        try:
            # asyncmy 的 kill() 发送 KILL <thread_id>。目标连接随后必须被淘汰，
            # 不能在协议状态未知时重新放回 SQLAlchemy 连接池。
            await connection.kill(target_thread_id)
        finally:
            connection.close()


__all__ = ["AsyncMyServerQueryCanceller", "MySQLServerQueryCanceller"]
