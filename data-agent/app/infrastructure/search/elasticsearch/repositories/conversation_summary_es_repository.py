"""Dedicated BM25 projection for chat-scoped summary segments."""
from __future__ import annotations

import time
from typing import Any, Sequence

from elasticsearch import AsyncElasticsearch, NotFoundError

from app.config.app_config import app_config
from app.domain.conversation.fingerprints import memory_query_fingerprint


def _body(value: Any) -> dict:
    return dict(getattr(value, "body", value) or {})


class ConversationSummaryESRepository:
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
        self.base_name = app_config.memory.summary_es_index_name
        self.projection_version = str(
            projection_version
            or app_config.memory.summary_es_projection_version
        )
        self.manage_alias = bool(manage_alias)

    @property
    def physical_index(self) -> str:
        version = self.projection_version.lower().replace("_", "-")
        return f"{self.base_name}-{version}"

    @property
    def read_alias(self) -> str:
        return f"{self.base_name}-active"

    async def read_alias_ready(self) -> bool:
        """Return true only when the alias targets this exact physical version."""
        if not await self.client.indices.exists_alias(name=self.read_alias):
            return False
        try:
            aliases = _body(
                await self.client.indices.get_alias(name=self.read_alias)
            )
        except NotFoundError:
            return False
        return sorted(aliases) == [self.physical_index]

    async def ensure_index(self) -> dict:
        physical = self.physical_index
        if not await self.client.indices.exists(index=physical):
            await self.client.indices.create(
                index=physical,
                settings={"number_of_shards": 1, "number_of_replicas": 0},
                mappings={
                    "dynamic": "strict",
                    "properties": {
                        "summary_id": {"type": "keyword"},
                        "owner_id": {"type": "keyword"},
                        "workspace_id": {"type": "keyword"},
                        "chat_id": {"type": "keyword"},
                        "level": {"type": "integer"},
                        "content_masked": {
                            "type": "text",
                            "analyzer": "ik_max_word",
                            "search_analyzer": "ik_smart",
                        },
                        "signals": {"type": "keyword"},
                        "signal_records": {
                            "type": "nested",
                            "properties": {
                                "signal_type": {"type": "keyword"},
                                "normalized_value": {"type": "keyword"},
                                "weight": {"type": "double"},
                                "source_order": {"type": "integer"},
                                "source_message_ids": {"type": "long"},
                            },
                        },
                        "signal_text": {
                            "type": "text",
                            "analyzer": "ik_max_word",
                            "search_analyzer": "ik_smart",
                        },
                        "status": {"type": "keyword"},
                        "source_content_hash": {"type": "keyword"},
                        "projection_version": {"type": "keyword"},
                        "source_start_message_id": {"type": "long"},
                        "source_end_message_id": {"type": "long"},
                        "created_at": {"type": "date"},
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
            aliases = _body(await self.client.indices.get_alias(name=self.read_alias))
            targets = sorted(aliases)
            if targets != [physical]:
                raise RuntimeError(
                    f"conversation summary alias 漂移：{targets}，期望 {[physical]}"
                )
        return await self.status()

    async def upsert(self, segment, signals: Sequence[object]) -> None:
        await self.ensure_index()
        values = [str(getattr(row, "normalized_value", "") or "") for row in signals]
        signal_records = [
            {
                "signal_type": str(getattr(row, "signal_type", "") or ""),
                "normalized_value": str(
                    getattr(row, "normalized_value", "") or ""
                ),
                "weight": float(getattr(row, "weight", 1.0) or 1.0),
                "source_order": int(
                    getattr(row, "source_order", index) or index
                ),
                "source_message_ids": [
                    int(value)
                    for value in list(
                        getattr(row, "source_message_ids", []) or []
                    )
                ],
            }
            for index, row in enumerate(signals)
        ]
        await self.client.index(
            index=self.physical_index,
            id=segment.summary_id,
            document={
                "summary_id": segment.summary_id,
                "owner_id": segment.owner_id,
                "workspace_id": segment.workspace_id,
                "chat_id": segment.chat_id,
                "level": int(segment.level or 0),
                "content_masked": segment.content_masked,
                "signals": values,
                "signal_records": signal_records,
                "signal_text": " ".join(values),
                "status": segment.status,
                "source_content_hash": segment.source_content_hash,
                "projection_version": self.projection_version,
                "source_start_message_id": int(segment.source_start_message_id),
                "source_end_message_id": int(segment.source_end_message_id),
                "created_at": segment.created_at,
            },
            refresh=False,
        )

    async def delete(self, summary_id: str) -> None:
        await self.ensure_index()
        try:
            await self.client.delete(
                index=self.physical_index, id=str(summary_id), refresh=False
            )
        except NotFoundError:
            pass

    async def promote_alias(self) -> dict:
        """Atomically move the read alias after a version backfill completes."""
        original = self.manage_alias
        self.manage_alias = False
        try:
            await self.ensure_index()
        finally:
            self.manage_alias = original
        actions: list[dict] = []
        if await self.client.indices.exists_alias(name=self.read_alias):
            aliases = _body(
                await self.client.indices.get_alias(name=self.read_alias)
            )
            if sorted(aliases) == [self.physical_index]:
                return await self.status()
            actions.extend(
                {"remove": {"index": index, "alias": self.read_alias}}
                for index in sorted(aliases)
                if index != self.physical_index
            )
        actions.append({
            "add": {
                "index": self.physical_index,
                "alias": self.read_alias,
                "is_write_index": True,
            }
        })
        await self.client.indices.update_aliases(actions=actions)
        return await self.status()

    async def search(
        self,
        query: str,
        *,
        owner_id: str,
        workspace_id: str,
        chat_id: str,
        limit: int,
    ) -> tuple[list[str], dict]:
        await self.ensure_index()
        started = time.perf_counter()
        if not await self.read_alias_ready():
            return [], {
                "engine": "elasticsearch",
                "route": "summary_es",
                "query_fingerprint": memory_query_fingerprint(query),
                "candidate_count": 0,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "projection_version": self.projection_version,
                "owner_scope_applied": True,
                "workspace_scope_applied": True,
                "chat_scope_applied": True,
                "raw_text_logged": False,
                "status": "backfill_pending",
            }
        response = _body(await self.client.search(
            index=self.read_alias,
            size=max(1, min(int(limit), 50)),
            query={
                "bool": {
                    "filter": [
                        {"term": {"owner_id": owner_id}},
                        {"term": {"workspace_id": workspace_id}},
                        {"term": {"chat_id": chat_id}},
                        {"term": {"status": "active"}},
                        {"term": {"projection_version": self.projection_version}},
                    ],
                    "must": [{
                        "multi_match": {
                            "query": str(query),
                            "fields": ["content_masked^2", "signal_text^3", "signals^4"],
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
            "route": "summary_es",
            "query_fingerprint": memory_query_fingerprint(query),
            "candidate_count": len(identifiers),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "projection_version": self.projection_version,
            "owner_scope_applied": True,
            "workspace_scope_applied": True,
            "chat_scope_applied": True,
            "raw_text_logged": False,
            "status": "ok",
        }

    async def status(self) -> dict:
        exists = bool(await self.client.indices.exists(index=self.physical_index))
        alias_exists = bool(await self.client.indices.exists_alias(name=self.read_alias))
        alias_ready = await self.read_alias_ready() if alias_exists else False
        count = 0
        if exists:
            count = int(_body(await self.client.count(index=self.physical_index)).get("count") or 0)
        return {
            "engine": "elasticsearch",
            "physical_index": self.physical_index,
            "alias": self.read_alias,
            "exists": exists,
            "alias_ready": alias_ready,
            "document_count": count,
            "projection_version": self.projection_version,
        }
