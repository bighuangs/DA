"""Versioned Elasticsearch BM25 projection for long-term conversation memory."""
from __future__ import annotations

import time
from typing import Any

from elasticsearch import AsyncElasticsearch, NotFoundError

from app.config.app_config import app_config
from app.domain.conversation.fingerprints import memory_query_fingerprint


def _response_body(value: Any) -> dict:
    return dict(getattr(value, "body", value) or {})


class ConversationMemoryESRepository:
    """ES returns candidate IDs only; MySQL is always read afterwards."""

    def __init__(
        self,
        client: AsyncElasticsearch,
        *,
        projection_version: str | None = None,
        manage_alias: bool = True,
    ):
        if client is None:
            raise RuntimeError("Elasticsearch client 尚未初始化")
        self.client = client
        self.base_name = app_config.memory.es_index_name
        self.projection_version = str(
            projection_version or app_config.memory.es_projection_version
        )
        self.manage_alias = bool(manage_alias)

    @property
    def physical_index(self) -> str:
        version = self.projection_version.lower().replace("_", "-")
        return f"{self.base_name}-{version}"

    @property
    def read_alias(self) -> str:
        return f"{self.base_name}-active"

    async def ensure_index(self) -> dict:
        physical = self.physical_index
        if not await self.client.indices.exists(index=physical):
            await self.client.indices.create(
                index=physical,
                settings={"number_of_shards": 1, "number_of_replicas": 0},
                mappings={
                    "dynamic": "strict",
                    "properties": {
                        "memory_id": {"type": "keyword"},
                        "owner_id": {"type": "keyword"},
                        "workspace_id": {"type": "keyword"},
                        "chat_id": {"type": "keyword"},
                        "memory_type": {"type": "keyword"},
                        "content_masked": {
                            "type": "text",
                            "analyzer": "ik_max_word",
                            "search_analyzer": "ik_smart",
                        },
                        "identifiers": {"type": "keyword"},
                        "status": {"type": "keyword"},
                        "verification_status": {"type": "keyword"},
                        "projection_version": {"type": "keyword"},
                        "created_at": {"type": "date"},
                        "valid_to": {"type": "date"},
                        "expires_at": {"type": "date"},
                    },
                },
            )
        if not self.manage_alias:
            return await self.status()
        if not await self.client.indices.exists_alias(name=self.read_alias):
            await self.client.indices.put_alias(
                index=physical, name=self.read_alias, is_write_index=True
            )
        else:
            aliases = _response_body(
                await self.client.indices.get_alias(name=self.read_alias)
            )
            targets = sorted(aliases)
            if targets != [physical]:
                raise RuntimeError(
                    f"conversation memory alias 漂移：{targets}，期望 {[physical]}"
                )
        return await self.status()

    async def upsert(self, item) -> None:
        await self.ensure_index()
        payload = dict(item.structured_payload or {})
        await self.client.index(
            index=(self.read_alias if self.manage_alias else self.physical_index),
            id=item.memory_id,
            document={
                "memory_id": item.memory_id,
                "owner_id": item.owner_id,
                "workspace_id": item.workspace_id,
                "chat_id": item.chat_id,
                "memory_type": item.memory_type,
                "content_masked": item.content_masked,
                "identifiers": list(payload.get("table_ids") or []),
                "status": item.status,
                "verification_status": item.verification_status,
                "projection_version": self.projection_version,
                "created_at": item.created_at,
                "valid_to": item.valid_to,
                "expires_at": item.expires_at,
            },
            refresh=False,
        )

    async def delete(self, memory_id: str) -> None:
        await self.ensure_index()
        try:
            await self.client.delete(
                index=(self.read_alias if self.manage_alias else self.physical_index),
                id=memory_id,
                refresh=False,
            )
        except NotFoundError:
            # Per-target outbox delivery is at-least-once.
            pass

    async def search(
        self,
        query: str,
        *,
        owner_id: str,
        workspace_id: str,
        limit: int,
    ) -> tuple[list[str], dict]:
        await self.ensure_index()
        started = time.perf_counter()
        response = _response_body(await self.client.search(
            index=self.read_alias,
            size=max(1, min(int(limit), 50)),
            query={
                "bool": {
                    "filter": [
                        {"term": {"owner_id": owner_id}},
                        {"term": {"workspace_id": workspace_id}},
                        {"term": {"status": "active"}},
                        {
                            "terms": {
                                "verification_status": [
                                    "verified", "confirmed", "executed"
                                ]
                            }
                        },
                        {"term": {"projection_version": self.projection_version}},
                    ],
                    "must": [{
                        "multi_match": {
                            "query": str(query),
                            "fields": ["content_masked^2", "identifiers^3"],
                            "type": "best_fields",
                        }
                    }],
                }
            },
            source=False,
        ))
        hits = list(((response.get("hits") or {}).get("hits") or []))
        identifiers = [str(hit.get("_id") or "") for hit in hits if hit.get("_id")]
        return identifiers, {
            "engine": "elasticsearch",
            "route": "memory_es",
            "query_fingerprint": memory_query_fingerprint(query),
            "candidate_count": len(identifiers),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "projection_version": self.projection_version,
            "workspace_scope_applied": True,
            "owner_scope_applied": True,
            "raw_text_logged": False,
            "status": "ok",
        }

    async def status(self) -> dict:
        exists = bool(await self.client.indices.exists(index=self.physical_index))
        alias_exists = bool(await self.client.indices.exists_alias(name=self.read_alias))
        count = 0
        if exists:
            body = _response_body(await self.client.count(index=self.physical_index))
            count = int(body.get("count") or 0)
        return {
            "engine": "elasticsearch",
            "physical_index": self.physical_index,
            "alias": self.read_alias,
            "exists": exists,
            "alias_ready": alias_exists,
            "document_count": count,
            "projection_version": self.projection_version,
        }
