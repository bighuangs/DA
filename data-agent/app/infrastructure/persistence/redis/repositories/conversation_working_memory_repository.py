"""Tenant-scoped Redis working-memory storage with atomic CAS and sliding TTL."""
from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.parse import quote

from redis.asyncio import Redis


WORKING_MEMORY_KEY_VERSION = "working-memory-key-v2"
_CAS_SCRIPT = """
local current = redis.call('HGET', KEYS[1], 'version')
if not current then current = '-1' end
if tonumber(current) ~= tonumber(ARGV[1]) then
  return {0, tonumber(current)}
end
redis.call('HSET', KEYS[1],
  'version', ARGV[2],
  'payload', ARGV[3],
  'updated_at', ARGV[4],
  'schema_version', ARGV[5])
redis.call('EXPIRE', KEYS[1], ARGV[6])
return {1, tonumber(ARGV[2])}
"""

_FORCE_SCRIPT = """
redis.call('HSET', KEYS[1],
  'version', ARGV[1],
  'payload', ARGV[2],
  'updated_at', ARGV[3],
  'schema_version', ARGV[4])
redis.call('EXPIRE', KEYS[1], ARGV[5])
return tonumber(ARGV[1])
"""


def _scope_part(value: str, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"working memory 缺少 {name}")
    if len(text) > 128:
        raise ValueError(f"working memory {name} 超过 128 字符")
    return quote(text, safe="-_.")


def working_memory_key(workspace_id: str, owner_id: str, chat_id: str) -> str:
    """Build a key that cannot collide across workspace, owner or chat."""
    return "da:wm:v2:{workspace}:{owner}:{chat}".format(
        workspace=_scope_part(workspace_id, "workspace_id"),
        owner=_scope_part(owner_id, "owner_id"),
        chat=_scope_part(chat_id, "chat_id"),
    )


@dataclass(frozen=True, slots=True)
class WorkingMemoryCASResult:
    updated: bool
    version: int


class ConversationWorkingMemoryRepository:
    """Redis is a reconstructable hot projection; MySQL remains authoritative."""

    def __init__(
        self,
        client: Redis,
        *,
        owner_id: str,
        workspace_id: str,
        ttl_seconds: int,
    ):
        if client is None:
            raise RuntimeError("Redis client 尚未初始化")
        self.client = client
        _scope_part(owner_id, "owner_id")
        _scope_part(workspace_id, "workspace_id")
        self.owner_id = str(owner_id).strip()
        self.workspace_id = str(workspace_id).strip()
        self.ttl_seconds = max(60, int(ttl_seconds))

    def key(self, chat_id: str) -> str:
        return working_memory_key(self.workspace_id, self.owner_id, chat_id)

    async def get(self, chat_id: str, *, renew_ttl: bool = True) -> dict | None:
        key = self.key(chat_id)
        row = await self.client.hgetall(key)
        if not row:
            return None
        if renew_ttl:
            await self.client.expire(key, self.ttl_seconds)
        payload = json.loads(row.get("payload") or "{}")
        if not isinstance(payload, dict):
            raise ValueError("Redis working memory payload 不是对象")
        return {
            **payload,
            "version": int(row.get("version") or 0),
            "schema_version": row.get("schema_version"),
            "updated_at": row.get("updated_at"),
            "ttl_seconds": int(await self.client.ttl(key)),
        }

    async def compare_and_set(
        self,
        chat_id: str,
        *,
        expected_version: int,
        payload: dict,
        schema_version: str,
        updated_at: str,
    ) -> WorkingMemoryCASResult:
        next_version = int(expected_version) + 1
        result = await self.client.eval(
            _CAS_SCRIPT,
            1,
            self.key(chat_id),
            int(expected_version),
            next_version,
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str),
            updated_at,
            str(schema_version),
            self.ttl_seconds,
        )
        values = list(result or [0, expected_version])
        return WorkingMemoryCASResult(bool(int(values[0])), int(values[1]))

    async def replace(
        self,
        chat_id: str,
        *,
        version: int,
        payload: dict,
        schema_version: str,
        updated_at: str,
    ) -> int:
        return int(await self.client.eval(
            _FORCE_SCRIPT,
            1,
            self.key(chat_id),
            max(0, int(version)),
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str),
            updated_at,
            str(schema_version),
            self.ttl_seconds,
        ))

    async def delete(self, chat_id: str) -> bool:
        return bool(await self.client.delete(self.key(chat_id)))

    async def status(self, chat_id: str) -> dict:
        key = self.key(chat_id)
        exists, ttl = await self.client.exists(key), await self.client.ttl(key)
        return {
            "key_version": WORKING_MEMORY_KEY_VERSION,
            "exists": bool(exists),
            "ttl_seconds": int(ttl),
        }
