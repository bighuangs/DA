"""查询结果 TTL/LRU 缓存。

当前项目尚未接入用户、空间和行列权限，因此缓存只在单进程、单 DW 数据源内使用。
调用方必须把数据源和关键查询配置纳入 key；Schema 发布成功后由服务主动清空缓存。
"""

from __future__ import annotations

import copy
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float


class ResultCache:
    def __init__(self, max_entries: int = 256):
        self.max_entries = max(1, max_entries)
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str, ttl_seconds: int) -> Any | None:
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.expires_at <= now or entry.expires_at - now > ttl_seconds:
                if entry is not None:
                    self._entries.pop(key, None)
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return copy.deepcopy(entry.value)

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return
        with self._lock:
            self._entries[key] = _CacheEntry(
                value=copy.deepcopy(value),
                expires_at=time.monotonic() + ttl_seconds,
            )
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "hits": self.hits,
                "misses": self.misses,
            }


result_cache = ResultCache()
