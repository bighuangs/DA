"""Live infrastructure smoke test for TODO-59 without invoking an LLM."""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

from app.infrastructure.clients.es_client_manager import es_client_manager
from app.infrastructure.clients.mysql_client_manager import meta_mysql_client_manager
from app.infrastructure.clients.qdrant_client_manager import qdrant_client_manager
from app.infrastructure.clients.redis_client_manager import redis_client_manager
from app.config.app_config import app_config
from app.application.services.conversation_memory_policy import extract_memory_candidates, iso_now
from app.domain.retrieval.hybrid_embedding import HybridEmbedding
from app.infrastructure.search.elasticsearch.repositories.conversation_memory_es_repository import (
    ConversationMemoryESRepository,
)
from app.infrastructure.persistence.mysql.repositories.conversation_memory_repository import (
    ConversationMemoryRepository,
)
from app.infrastructure.persistence.mysql.repositories.meta_mysql_repository import MetaMysqlRepository
from app.infrastructure.search.qdrant.repositories.conversation_memory_qdrant_repository import (
    ConversationMemoryQdrantRepository,
)
from app.infrastructure.persistence.redis.repositories.conversation_working_memory_repository import (
    ConversationWorkingMemoryRepository,
)


async def run_smoke() -> dict:
    suffix = uuid.uuid4().hex
    owner_id = f"smoke-owner-{suffix[:8]}"
    workspace_id = f"smoke-workspace-{suffix[:8]}"
    chat_id = f"smoke-chat-{suffix[:8]}"
    memory_id = str(uuid.uuid4())
    results: dict[str, dict] = {}

    meta_mysql_client_manager.init()
    es_client_manager.init()
    qdrant_client_manager.init()
    redis_client_manager.init()
    es_repo = ConversationMemoryESRepository(es_client_manager.client)
    qdrant_repo = ConversationMemoryQdrantRepository(
        qdrant_client_manager.client
    )
    redis_repo = ConversationWorkingMemoryRepository(
        redis_client_manager.client,
        owner_id=owner_id,
        workspace_id=workspace_id,
        ttl_seconds=120,
    )
    try:
        if not await redis_client_manager.ping():
            raise RuntimeError("Redis PING 失败")
        cas = await redis_repo.compare_and_set(
            chat_id,
            expected_version=-1,
            payload={
                "structured_state": {"dimensions": ["region"]},
                "recent_message_ids": [],
                "rolling_summary": "smoke",
                "last_success_question": "smoke",
            },
            schema_version=app_config.memory.context_schema_version,
            updated_at=iso_now(),
        )
        redis_value = await redis_repo.get(chat_id)
        if not cas.updated or redis_value is None:
            raise RuntimeError("Redis CAS/get 合同失败")
        results["redis"] = {
            "status": "passed",
            "cas_version": cas.version,
            "ttl_seconds": redis_value["ttl_seconds"],
        }

        async with meta_mysql_client_manager.session_factory() as session:
            await MetaMysqlRepository(session).ensure_admin_tables()
        async with meta_mysql_client_manager.session_factory() as session:
            repository = ConversationMemoryRepository(
                session, owner_id=owner_id, workspace_id=workspace_id
            )
            values = extract_memory_candidates(
                owner_id=owner_id,
                workspace_id=workspace_id,
                chat_id=chat_id,
                query="各地区销售额",
                standalone_question="各地区销售额",
                follow_up_action="new_topic",
                semantic_state={
                    "dimensions": ["region"],
                    "measures": ["sales_amount"],
                },
                sql=(
                    "SELECT region, SUM(amount) "
                    "FROM fact_sales GROUP BY region"
                ),
                user_message_id=1,
                assistant_message_id=2,
                schema_version="smoke-schema",
            )[0]
            values["memory_id"] = memory_id
            item, created = await repository.upsert_memory_item(values)
            visible = await repository.list_authoritative_by_ids([memory_id])
            foreign = await ConversationMemoryRepository(
                session,
                owner_id=f"{owner_id}-foreign",
                workspace_id=workspace_id,
            ).list_authoritative_by_ids([memory_id])
            if not created or not visible or foreign:
                raise RuntimeError("MySQL 权威层 scope 合同失败")
            results["mysql"] = {
                "status": "passed",
                "authority_visible": len(visible),
                "foreign_visible": len(foreign),
            }
            # This smoke row must never survive the verification transaction.
            await session.rollback()

        await es_repo.ensure_index()
        projection_item = SimpleNamespace(
            memory_id=memory_id,
            owner_id=owner_id,
            workspace_id=workspace_id,
            chat_id=chat_id,
            memory_type="query_episode",
            content_masked="TODO59 地区销售额 smoke",
            structured_payload={"table_ids": ["fact_sales"]},
            status="active",
            verification_status="executed",
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            valid_to=None,
            expires_at=None,
        )
        await es_repo.upsert(projection_item)
        await es_client_manager.client.indices.refresh(index=es_repo.read_alias)
        es_ids, es_evidence = await es_repo.search(
            "TODO59 地区销售额",
            owner_id=owner_id,
            workspace_id=workspace_id,
            limit=5,
        )
        if memory_id not in es_ids:
            raise RuntimeError("Elasticsearch memory projection 未命中")
        results["elasticsearch"] = {
            "status": "passed",
            "candidate_count": len(es_ids),
            "projection_version": es_evidence["projection_version"],
        }

        await qdrant_repo.ensure_collection()
        dense = [0.0] * int(app_config.embedding.dense_size)
        dense[0] = 1.0
        embedding = HybridEmbedding(
            dense=tuple(dense),
            sparse_indices=(42,),
            sparse_values=(1.0,),
        )
        await qdrant_repo.upsert(projection_item, embedding)
        qdrant_ids, qdrant_evidence, degradations = await qdrant_repo.search(
            embedding,
            query="TODO59 地区销售额",
            owner_id=owner_id,
            workspace_id=workspace_id,
            limit=5,
        )
        if memory_id not in qdrant_ids or degradations:
            raise RuntimeError("Qdrant memory projection 未命中或发生降级")
        results["qdrant"] = {
            "status": "passed",
            "candidate_count": len(qdrant_ids),
            "projection_version": qdrant_evidence["projection_version"],
        }
    finally:
        try:
            await redis_repo.delete(chat_id)
        except Exception:
            pass
        try:
            await es_repo.delete(memory_id)
            await es_client_manager.client.indices.refresh(index=es_repo.read_alias)
        except Exception:
            pass
        try:
            await qdrant_repo.delete(memory_id)
        except Exception:
            pass
        await redis_client_manager.close()
        await qdrant_client_manager.close()
        await es_client_manager.close()
        await meta_mysql_client_manager.close()
    return {
        "gate_passed": all(
            value.get("status") == "passed" for value in results.values()
        ) and set(results) == {"redis", "mysql", "elasticsearch", "qdrant"},
        "services": results,
    }


def main() -> int:
    report = asyncio.run(run_smoke())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
