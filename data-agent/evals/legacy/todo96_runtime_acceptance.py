"""Destructive-scoped TODO-96 acceptance against local asyncmy/InnoDB.

This executable is intentionally opt-in.  It writes synthetic rows under
run-specific owner/workspace identifiers, uses run-specific Elasticsearch and
Qdrant names, verifies cleanup, and refuses non-loopback/non-meta2 targets
unless the caller explicitly overrides the safety boundary.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import sys
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from qdrant_client import models as qdrant_models
from sqlalchemy import delete, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.inbound.worker.memory_projection_worker import (
    MemoryProjectionWorker,
)
from app.bootstrap.worker import (
    enqueue_summary_projection_backfill,
    summary_projection_backfill_status,
)
from app.application.services.conversation_summary_service import (
    ConversationSummaryService,
)
from app.config.app_config import app_config
from app.infrastructure.clients.embedding_client_manager import embedding_client_manager
from app.infrastructure.clients.es_client_manager import es_client_manager
from app.infrastructure.clients.mysql_client_manager import meta_mysql_client_manager
from app.infrastructure.clients.qdrant_client_manager import qdrant_client_manager
from app.infrastructure.config.conversation import load_conversation_settings
from app.infrastructure.persistence.mysql.conversation_adapters import (
    SqlConversationMemoryUnitOfWork,
)
from app.infrastructure.persistence.mysql.records.conversation_memory_outbox_mysql import (
    ConversationMemoryOutboxMySQL,
)
from app.infrastructure.persistence.mysql.records.conversation_message_mysql import ConversationMessageMySQL
from app.infrastructure.persistence.mysql.records.conversation_session_mysql import ConversationSessionMySQL
from app.infrastructure.persistence.mysql.records.conversation_summary_segment_mysql import (
    ConversationSummarySegmentMySQL,
)
from app.infrastructure.persistence.mysql.records.conversation_summary_signal_mysql import (
    ConversationSummarySignalMySQL,
)
from app.infrastructure.persistence.mysql.records.conversation_summary_state_mysql import (
    ConversationSummaryStateMySQL,
)
from app.infrastructure.search.elasticsearch.repositories.conversation_summary_es_repository import (
    ConversationSummaryESRepository,
)
from app.infrastructure.persistence.mysql.repositories.conversation_history_repository import (
    ConversationHistoryRepository,
)
from app.infrastructure.persistence.mysql.repositories.conversation_memory_repository import (
    ConversationMemoryRepository,
)
from app.infrastructure.persistence.mysql.repositories.conversation_summary_repository import (
    ConversationSummaryRepository,
)
from app.infrastructure.search.qdrant.repositories.conversation_summary_qdrant_repository import (
    ConversationSummaryQdrantRepository,
)
from evals.paths import REPORTS_ROOT


SAFE_CAPACITY_LIMIT = 5_000
ABSOLUTE_CAPACITY_LIMIT = 50_000
SAFE_CANARY_LIMIT = 40
ABSOLUTE_CANARY_LIMIT = 200


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _response_body(value: Any) -> dict:
    return dict(getattr(value, "body", value) or {})


class AcceptanceFailure(RuntimeError):
    """A runtime acceptance gate did not hold."""


class Todo96RuntimeAcceptance:
    CONFIG_FIELDS = (
        "summary_enabled",
        "summary_build_enabled",
        "summary_projection_enabled",
        "stable_fact_enabled",
        "summary_es_index_name",
        "summary_es_projection_version",
        "summary_qdrant_collection_name",
        "summary_qdrant_projection_version",
        "projection_batch_size",
        "projection_lease_seconds",
        "summary_turn_high_watermark",
        "summary_turn_low_watermark",
        "summary_recent_protected_turns",
        "summary_retrieval_always",
        "summary_top_k",
    )

    def __init__(
        self,
        *,
        run_id: str,
        capacity_segments: int,
        canary_segments: int,
        lock_hold_ms: int,
        allow_nonlocal: bool,
    ) -> None:
        self.run_id = run_id.lower().replace("_", "-")
        self.capacity_segments = int(capacity_segments)
        self.canary_segments = int(canary_segments)
        self.lock_hold_ms = int(lock_hold_ms)
        self.allow_nonlocal = bool(allow_nonlocal)
        script_path = Path(__file__).resolve()
        self.report: dict[str, Any] = {
            "contract": "todo96-runtime-acceptance-v1",
            "run_id": self.run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "execution": {
                "command": [sys.executable, *sys.argv],
                "script_path": str(script_path),
                "script_sha256": hashlib.sha256(
                    script_path.read_bytes()
                ).hexdigest(),
            },
            "checks": [],
        }
        self._original_config = {
            name: getattr(app_config.memory, name) for name in self.CONFIG_FIELDS
        }
        self.es_base = f"data-agent-todo96-canary-{self.run_id}"
        self.es_version = f"todo96-es-{self.run_id}"
        self.qdrant_collection = f"data_agent_todo96_canary_{self.run_id}"
        self.qdrant_version = f"todo96-qd-{self.run_id}"
        self._external_initialized = False
        self._baseline_outbox: dict[str, int] = {}
        self._owners: set[str] = set()

    def uid(self, label: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"todo96:{self.run_id}:{label}"))

    def check(self, name: str, condition: bool, **evidence: Any) -> None:
        item = {"name": name, "passed": bool(condition), **evidence}
        self.report["checks"].append(item)
        if not condition:
            raise AcceptanceFailure(name)

    def _configure_isolated_projection(self) -> None:
        values = {
            "summary_es_index_name": self.es_base,
            "summary_es_projection_version": self.es_version,
            "summary_qdrant_collection_name": self.qdrant_collection,
            "summary_qdrant_projection_version": self.qdrant_version,
            "projection_batch_size": 4,
            "projection_lease_seconds": 30,
            "summary_turn_high_watermark": 2,
            "summary_turn_low_watermark": 1,
            "summary_recent_protected_turns": 1,
            "summary_retrieval_always": True,
            "summary_top_k": 10,
            "summary_enabled": False,
            "summary_build_enabled": False,
            "summary_projection_enabled": False,
            "stable_fact_enabled": False,
        }
        for name, value in values.items():
            setattr(app_config.memory, name, value)

    def _restore_config(self) -> None:
        for name, value in self._original_config.items():
            setattr(app_config.memory, name, value)

    async def _outbox_distribution(self) -> dict[str, int]:
        async with meta_mysql_client_manager.session_factory() as session:
            rows = (await session.execute(
                select(
                    ConversationMemoryOutboxMySQL.status,
                    func.count(ConversationMemoryOutboxMySQL.event_id),
                ).group_by(ConversationMemoryOutboxMySQL.status)
            )).all()
        return {str(status): int(count) for status, count in rows}

    async def _summary_counts(self) -> dict[str, int]:
        async with meta_mysql_client_manager.session_factory() as session:
            return {
                "segments": int(await session.scalar(
                    select(func.count()).select_from(
                        ConversationSummarySegmentMySQL
                    )
                ) or 0),
                "signals": int(await session.scalar(
                    select(func.count()).select_from(
                        ConversationSummarySignalMySQL
                    )
                ) or 0),
                "states": int(await session.scalar(
                    select(func.count()).select_from(
                        ConversationSummaryStateMySQL
                    )
                ) or 0),
            }

    async def preflight(self) -> None:
        config = app_config.db_meta
        target = {
            "host": str(config.host),
            "port": int(config.port),
            "database": str(config.database),
            "user": str(config.user),
            "password_present": bool(str(config.password or "")),
        }
        is_loopback = target["host"] in {"127.0.0.1", "localhost", "::1"}
        if not self.allow_nonlocal:
            self.check("safety_target_is_local_meta2", (
                is_loopback and target["database"] == "meta2"
            ), target={key: value for key, value in target.items() if key != "user"})

        engine = meta_mysql_client_manager.engine
        async with engine.connect() as connection:
            server = dict((await connection.execute(text(
                "SELECT VERSION() AS version, "
                "@@version_comment AS version_comment, "
                "@@transaction_isolation AS tx_isolation, "
                "@@default_storage_engine AS default_engine, "
                "DATABASE() AS database_name"
            ))).mappings().one())
            table_rows = list((await connection.execute(text(
                "SELECT TABLE_NAME AS name, ENGINE AS engine, "
                "TABLE_COLLATION AS collation "
                "FROM information_schema.tables "
                "WHERE table_schema=DATABASE() AND table_name IN "
                "('conversation_memory_outbox','conversation_summary_segment',"
                "'conversation_summary_signal','conversation_summary_state') "
                "ORDER BY table_name"
            ))).mappings().all())
            lease_rows = list((await connection.execute(text(
                "SELECT COLUMN_NAME AS name, COLUMN_TYPE AS type, "
                "IS_NULLABLE AS nullable "
                "FROM information_schema.columns "
                "WHERE table_schema=DATABASE() "
                "AND table_name='conversation_memory_outbox' "
                "AND column_name IN ('claimed_at','lease_until','worker_id') "
                "ORDER BY ordinal_position"
            ))).mappings().all())
            column_rows = list((await connection.execute(text(
                "SELECT TABLE_NAME AS name, COUNT(*) AS column_count "
                "FROM information_schema.columns "
                "WHERE table_schema=DATABASE() AND table_name IN "
                "('conversation_summary_segment','conversation_summary_signal',"
                "'conversation_summary_state') GROUP BY table_name "
                "ORDER BY table_name"
            ))).mappings().all())
            index_rows = list((await connection.execute(text(
                "SELECT TABLE_NAME AS name, COUNT(DISTINCT INDEX_NAME) AS index_count "
                "FROM information_schema.statistics "
                "WHERE table_schema=DATABASE() AND table_name IN "
                "('conversation_summary_segment','conversation_summary_signal',"
                "'conversation_summary_state') GROUP BY table_name "
                "ORDER BY table_name"
            ))).mappings().all())
            innodb_support = (await connection.execute(text(
                "SELECT SUPPORT FROM information_schema.engines "
                "WHERE ENGINE='InnoDB'"
            ))).scalar_one()

        migration = Path("migrations/20260804_todo96_conversation_summary.sql")
        rollback = Path(
            "migrations/rollback/20260804_todo96_conversation_summary.down.sql"
        )
        collation_alignment = Path(
            "migrations/20260805_todo96_summary_collation_alignment.sql"
        )
        migration_hashes = {
            "upgrade_sha256": hashlib.sha256(migration.read_bytes()).hexdigest(),
            "rollback_sha256": hashlib.sha256(rollback.read_bytes()).hexdigest(),
            "collation_alignment_sha256": hashlib.sha256(
                collation_alignment.read_bytes()
            ).hexdigest(),
        }
        table_map = {str(row["name"]): str(row["engine"]) for row in table_rows}
        collation_map = {
            str(row["name"]): str(row["collation"]) for row in table_rows
        }
        column_map = {
            str(row["name"]): int(row["column_count"]) for row in column_rows
        }
        index_map = {
            str(row["name"]): int(row["index_count"]) for row in index_rows
        }
        lease_map = {
            str(row["name"]): [str(row["type"]), str(row["nullable"])]
            for row in lease_rows
        }
        self.check(
            "driver_is_real_asyncmy",
            engine.dialect.name == "mysql" and engine.dialect.driver == "asyncmy",
            dialect=engine.dialect.name,
            driver=engine.dialect.driver,
        )
        self.check(
            "mysql_repeatable_read_innodb",
            str(server["tx_isolation"]).upper() == "REPEATABLE-READ"
            and str(server["default_engine"]).upper() == "INNODB"
            and str(innodb_support).upper() in {"YES", "DEFAULT"},
            tx_isolation=str(server["tx_isolation"]),
            default_engine=str(server["default_engine"]),
        )
        self.check(
            "migration_tables_are_innodb",
            len(table_map) == 4
            and all(str(value).upper() == "INNODB" for value in table_map.values()),
            tables=table_map,
        )
        self.check(
            "migration_collations_align_with_existing_outbox",
            len(set(collation_map.values())) == 1,
            collations=collation_map,
        )
        self.check(
            "migration_lease_columns_match",
            lease_map == {
                "claimed_at": ["datetime", "YES"],
                "lease_until": ["datetime", "YES"],
                "worker_id": ["varchar(80)", "YES"],
            },
            columns=lease_map,
        )
        self.check(
            "migration_column_counts_match",
            column_map == {
                "conversation_summary_segment": 25,
                "conversation_summary_signal": 15,
                "conversation_summary_state": 10,
            },
            counts=column_map,
        )
        self.check(
            "migration_index_counts_match",
            index_map == {
                "conversation_summary_segment": 4,
                "conversation_summary_signal": 4,
                "conversation_summary_state": 3,
            },
            counts=index_map,
        )
        self._baseline_outbox = await self._outbox_distribution()
        summary_counts = await self._summary_counts()
        non_completed = sum(
            count for status, count in self._baseline_outbox.items()
            if status != "completed"
        )
        self.check(
            "shared_outbox_has_no_claimable_work",
            non_completed == 0,
            distribution=self._baseline_outbox,
        )
        self.check(
            "summary_authority_empty_before_synthetic_run",
            not any(summary_counts.values()),
            counts=summary_counts,
        )
        self.check(
            "rollout_defaults_all_off",
            all(not bool(self._original_config[name]) for name in (
                "summary_enabled",
                "summary_build_enabled",
                "summary_projection_enabled",
                "stable_fact_enabled",
            )),
            values={
                name: bool(self._original_config[name]) for name in (
                    "summary_enabled",
                    "summary_build_enabled",
                    "summary_projection_enabled",
                    "stable_fact_enabled",
                )
            },
        )
        self.report["environment"] = {
            "target": {
                "host": target["host"],
                "port": target["port"],
                "database": target["database"],
            },
            "server": server,
            "asyncmy_version": importlib.metadata.version("asyncmy"),
            "sqlalchemy_version": importlib.metadata.version("sqlalchemy"),
            "migration": migration_hashes,
            "tables": [dict(row) for row in table_rows],
        }

    async def _cleanup_owners(self, owners: Iterable[str]) -> None:
        values = tuple(dict.fromkeys(str(owner) for owner in owners if owner))
        if not values:
            return
        async with meta_mysql_client_manager.session_factory() as session:
            for model in (
                ConversationMemoryOutboxMySQL,
                ConversationSummarySignalMySQL,
                ConversationSummarySegmentMySQL,
                ConversationSummaryStateMySQL,
                ConversationMessageMySQL,
                ConversationSessionMySQL,
            ):
                await session.execute(delete(model).where(model.owner_id.in_(values)))
            await session.commit()

    async def lock_and_generation_fence(self) -> None:
        owner = self.uid("lock-owner")
        workspace = self.uid("lock-workspace")
        chat_id = f"todo96-lock-{self.run_id}"[:64]
        self._owners.add(owner)
        async with meta_mysql_client_manager.session_factory() as session:
            session.add(ConversationSessionMySQL(
                chat_id=chat_id,
                owner_id=owner,
                workspace_id=workspace,
                title="TODO-96 lock canary",
                preview="lock",
                status="active",
                last_turn_status="succeeded",
                turn_count=3,
                version=1,
            ))
            for index in range(6):
                session.add(ConversationMessageMySQL(
                    owner_id=owner,
                    workspace_id=workspace,
                    chat_id=chat_id,
                    role="user" if index % 2 == 0 else "assistant",
                    content_masked=f"锁竞争消息 {index}",
                    request_id=f"{self.run_id}-lock-{index}",
                    trace_id=f"{self.run_id}-lock-{index}",
                    status="success",
                ))
            await session.commit()

        app_config.memory.summary_enabled = True
        app_config.memory.summary_build_enabled = True
        app_config.memory.summary_projection_enabled = True
        _, memory_settings = load_conversation_settings()
        builder_started = asyncio.Event()
        builder_connection: dict[str, int] = {}

        async def stale_builder() -> tuple[dict, float]:
            async with meta_mysql_client_manager.session_factory() as session:
                uow = SqlConversationMemoryUnitOfWork(
                    session,
                    owner_id=owner,
                    workspace_id=workspace,
                    settings=memory_settings,
                )
                builder_connection["id"] = int(await session.scalar(
                    text("SELECT CONNECTION_ID()")
                ))
                builder_started.set()
                started = time.perf_counter()
                result = await ConversationSummaryService(
                    owner_id=owner,
                    workspace_id=workspace,
                    settings=memory_settings,
                ).build_if_needed(
                    uow,
                    chat_id=chat_id,
                    expected_session_version=1,
                )
                await uow.commit()
                return result, (time.perf_counter() - started) * 1000

        async with meta_mysql_client_manager.session_factory() as delete_session:
            delete_connection = int(await delete_session.scalar(
                text("SELECT CONNECTION_ID()")
            ))
            installed = await ConversationHistoryRepository(
                delete_session,
                owner_id=owner,
                workspace_id=workspace,
            ).begin_delete(chat_id)
            task = asyncio.create_task(stale_builder())
            await builder_started.wait()
            await asyncio.sleep(self.lock_hold_ms / 1000)
            blocked_before_release = not task.done()
            lock_wait_rows: int | None = None
            try:
                async with meta_mysql_client_manager.session_factory() as observer:
                    lock_wait_rows = int(await observer.scalar(text(
                        "SELECT COUNT(*) FROM performance_schema.data_lock_waits w "
                        "JOIN performance_schema.threads t "
                        "ON w.REQUESTING_THREAD_ID=t.THREAD_ID "
                        "WHERE t.PROCESSLIST_ID=:connection_id"
                    ), {"connection_id": builder_connection["id"]}) or 0)
            except Exception:
                lock_wait_rows = None
            await delete_session.commit()
            result, waited_ms = await asyncio.wait_for(task, timeout=30)

        async with meta_mysql_client_manager.session_factory() as session:
            segment_count = int(await session.scalar(
                select(func.count()).select_from(
                    ConversationSummarySegmentMySQL
                ).where(ConversationSummarySegmentMySQL.owner_id == owner)
            ) or 0)
            outbox_count = int(await session.scalar(
                select(func.count()).select_from(
                    ConversationMemoryOutboxMySQL
                ).where(ConversationMemoryOutboxMySQL.owner_id == owner)
            ) or 0)
        self.check("delete_fence_installed", installed)
        self.check(
            "second_connection_blocked_on_innodb_row_lock",
            blocked_before_release
            and delete_connection != builder_connection.get("id")
            and waited_ms >= self.lock_hold_ms * 0.75,
            delete_connection=delete_connection,
            builder_connection=builder_connection.get("id"),
            configured_hold_ms=self.lock_hold_ms,
            observed_wait_ms=round(waited_ms, 3),
            performance_schema_wait_rows=lock_wait_rows,
        )
        self.check(
            "stale_builder_rejected_after_delete_generation",
            result.get("status") == "stale_generation"
            and int(result.get("segments_created") or 0) == 0
            and segment_count == 0
            and outbox_count == 0,
            result=result,
            segment_count=segment_count,
            outbox_count=outbox_count,
        )
        self.report["dual_connection_lock"] = {
            "delete_connection": delete_connection,
            "builder_connection": builder_connection.get("id"),
            "waited_ms": round(waited_ms, 3),
            "performance_schema_wait_rows": lock_wait_rows,
            "result": result,
        }
        await self._cleanup_owners((owner,))

    async def build_only_canary(self) -> None:
        owner = self.uid("build-owner")
        workspace = self.uid("build-workspace")
        chat_id = f"todo96-build-{self.run_id}"[:64]
        self._owners.add(owner)
        app_config.memory.summary_enabled = True
        app_config.memory.summary_build_enabled = True
        app_config.memory.summary_projection_enabled = False
        _, memory_settings = load_conversation_settings()
        async with meta_mysql_client_manager.session_factory() as session:
            session.add(ConversationSessionMySQL(
                chat_id=chat_id,
                owner_id=owner,
                workspace_id=workspace,
                title="TODO-96 build canary",
                preview="build",
                status="active",
                last_turn_status="succeeded",
                turn_count=3,
                version=1,
            ))
            for index in range(6):
                session.add(ConversationMessageMySQL(
                    owner_id=owner,
                    workspace_id=workspace,
                    chat_id=chat_id,
                    role="user" if index % 2 == 0 else "assistant",
                    content_masked=(
                        f"查询销售额 canary {index}"
                        if index % 2 == 0
                        else f"销售额查询完成 canary {index}"
                    ),
                    request_id=f"{self.run_id}-build-{index}",
                    trace_id=f"{self.run_id}-build-{index}",
                    status="success",
                ))
            await session.commit()
        async with meta_mysql_client_manager.session_factory() as session:
            uow = SqlConversationMemoryUnitOfWork(
                session,
                owner_id=owner,
                workspace_id=workspace,
                settings=memory_settings,
            )
            result = await ConversationSummaryService(
                owner_id=owner,
                workspace_id=workspace,
                settings=memory_settings,
            ).build_if_needed(
                uow,
                chat_id=chat_id,
                expected_session_version=1,
            )
            await uow.commit()
            segment_count = int(await session.scalar(
                select(func.count()).select_from(
                    ConversationSummarySegmentMySQL
                ).where(ConversationSummarySegmentMySQL.owner_id == owner)
            ) or 0)
            outbox_count = int(await session.scalar(
                select(func.count()).select_from(
                    ConversationMemoryOutboxMySQL
                ).where(ConversationMemoryOutboxMySQL.owner_id == owner)
            ) or 0)
        self.check(
            "mysql_build_only_canary_created_grounded_segment",
            result.get("status") == "created"
            and int(result.get("segments_created") or 0) >= 1
            and int(result.get("llm_calls") or 0) == 0
            and segment_count >= 1
            and outbox_count == 0,
            result=result,
            segment_count=segment_count,
            outbox_count=outbox_count,
        )
        self.report["build_only_canary"] = result
        await self._cleanup_owners((owner,))

    async def _run_pair(
        self,
        action: Callable[[AsyncSession, str], Awaitable[Any]],
    ) -> list[dict[str, Any]]:
        barrier = asyncio.Barrier(2)

        async def one(label: str) -> dict[str, Any]:
            async with meta_mysql_client_manager.session_factory() as session:
                connection_id = int(await session.scalar(
                    text("SELECT CONNECTION_ID()")
                ))
                await barrier.wait()
                started = time.perf_counter()
                value = await action(session, label)
                await session.commit()
                return {
                    "label": label,
                    "connection_id": connection_id,
                    "value": value,
                    "elapsed_ms": round(
                        (time.perf_counter() - started) * 1000, 3
                    ),
                }

        return list(await asyncio.wait_for(asyncio.gather(
            one("a"), one("b")
        ), timeout=30))

    async def atomic_creation_competition(self) -> None:
        owner = self.uid("atomic-owner")
        workspace = self.uid("atomic-workspace")
        memory_id = self.uid("atomic-memory")
        chat_id = f"todo96-atomic-{self.run_id}"[:64]
        self._owners.add(owner)

        async def enqueue_action(session: AsyncSession, _label: str) -> str:
            row = await ConversationMemoryRepository(
                session,
                owner_id=owner,
                workspace_id=workspace,
            ).enqueue_projection(
                memory_id=memory_id,
                target="summary_es",
                operation="upsert",
                projection_version=self.es_version,
            )
            return str(row.event_id)

        enqueue_results = await self._run_pair(enqueue_action)

        async def state_action(session: AsyncSession, _label: str) -> int:
            row = await ConversationSummaryRepository(
                session,
                owner_id=owner,
                workspace_id=workspace,
            ).ensure_state(
                chat_id,
                schema_version="summary-segment-v1",
                for_update=True,
            )
            return int(row.id)

        state_results = await self._run_pair(state_action)
        async with meta_mysql_client_manager.session_factory() as session:
            event_count = int(await session.scalar(
                select(func.count()).select_from(
                    ConversationMemoryOutboxMySQL
                ).where(ConversationMemoryOutboxMySQL.owner_id == owner)
            ) or 0)
            state_count = int(await session.scalar(
                select(func.count()).select_from(
                    ConversationSummaryStateMySQL
                ).where(ConversationSummaryStateMySQL.owner_id == owner)
            ) or 0)
        self.check(
            "dual_connection_atomic_outbox_unique_key",
            len({row["connection_id"] for row in enqueue_results}) == 2
            and len({row["value"] for row in enqueue_results}) == 1
            and event_count == 1,
            results=enqueue_results,
            row_count=event_count,
        )
        self.check(
            "dual_connection_atomic_summary_state_first_create",
            len({row["connection_id"] for row in state_results}) == 2
            and len({row["value"] for row in state_results}) == 1
            and state_count == 1,
            results=state_results,
            row_count=state_count,
        )
        self.report["atomic_creation"] = {
            "outbox": enqueue_results,
            "summary_state": state_results,
        }
        await self._cleanup_owners((owner,))

    async def _enqueue_event(
        self,
        *,
        owner: str,
        workspace: str,
        label: str,
        target: str,
        operation: str,
    ) -> str:
        async with meta_mysql_client_manager.session_factory() as session:
            row = await ConversationMemoryRepository(
                session,
                owner_id=owner,
                workspace_id=workspace,
            ).enqueue_projection(
                memory_id=self.uid(label),
                target=target,
                operation=operation,
                projection_version=(
                    self.es_version if target == "summary_es"
                    else self.qdrant_version
                ),
            )
            event_id = str(row.event_id)
            await session.commit()
            return event_id

    async def worker_lease_competition(self) -> None:
        owner = self.uid("lease-owner")
        workspace = self.uid("lease-workspace")
        self._owners.add(owner)
        event_ids = []
        for index in range(8):
            event_ids.append(await self._enqueue_event(
                owner=owner,
                workspace=workspace,
                label=f"lease-batch-{index}",
                target="summary_es",
                operation="delete",
            ))
        barrier = asyncio.Barrier(2)

        async def claim(worker_id: str) -> list[dict[str, Any]]:
            async with meta_mysql_client_manager.session_factory() as session:
                await barrier.wait()
                rows = await ConversationMemoryRepository.claim_pending_outbox(
                    session,
                    limit=4,
                    max_attempts=8,
                    worker_id=worker_id,
                    lease_seconds=30,
                    owner_allowlist=(owner,),
                )
                return [{
                    "event_id": row.event_id,
                    "attempt_count": int(row.attempt_count),
                    "worker_id": row.worker_id,
                } for row in rows]

        batch_a, batch_b = await asyncio.gather(
            claim(f"todo96-worker-a-{self.run_id}"),
            claim(f"todo96-worker-b-{self.run_id}"),
        )
        ids_a = {row["event_id"] for row in batch_a}
        ids_b = {row["event_id"] for row in batch_b}
        self.check(
            "two_workers_skip_locked_claims_are_disjoint",
            not (ids_a & ids_b)
            and (ids_a | ids_b) == set(event_ids)
            and all(row["attempt_count"] == 1 for row in batch_a + batch_b),
            worker_a=batch_a,
            worker_b=batch_b,
        )
        for worker_id, rows in (
            (f"todo96-worker-a-{self.run_id}", batch_a),
            (f"todo96-worker-b-{self.run_id}", batch_b),
        ):
            async with meta_mysql_client_manager.session_factory() as session:
                for row in rows:
                    completed = await ConversationMemoryRepository.complete_outbox(
                        session,
                        row["event_id"],
                        worker_id=worker_id,
                    )
                    self.check(
                        f"lease_owner_completed_{row['event_id']}", completed
                    )

        crash_event = await self._enqueue_event(
            owner=owner,
            workspace=workspace,
            label="lease-crash",
            target="summary_es",
            operation="delete",
        )
        worker_a = f"todo96-crash-a-{self.run_id}"
        worker_b = f"todo96-crash-b-{self.run_id}"
        async with meta_mysql_client_manager.session_factory() as session:
            claimed = await ConversationMemoryRepository.claim_pending_outbox(
                session,
                limit=1,
                max_attempts=8,
                worker_id=worker_a,
                lease_seconds=30,
                owner_allowlist=(owner,),
            )
            self.check(
                "crash_event_claimed_by_first_worker",
                len(claimed) == 1 and claimed[0].event_id == crash_event,
            )
        async with meta_mysql_client_manager.session_factory() as session:
            await session.execute(update(ConversationMemoryOutboxMySQL).where(
                ConversationMemoryOutboxMySQL.event_id == crash_event
            ).values(lease_until=_utcnow() - timedelta(seconds=1)))
            await session.commit()
        async with meta_mysql_client_manager.session_factory() as session:
            reclaimed = await ConversationMemoryRepository.claim_pending_outbox(
                session,
                limit=1,
                max_attempts=8,
                worker_id=worker_b,
                lease_seconds=30,
                owner_allowlist=(owner,),
            )
            reclaimed_attempt = int(reclaimed[0].attempt_count)
        async with meta_mysql_client_manager.session_factory() as session:
            stale_complete = await ConversationMemoryRepository.complete_outbox(
                session,
                crash_event,
                worker_id=worker_a,
            )
            current_complete = await ConversationMemoryRepository.complete_outbox(
                session,
                crash_event,
                worker_id=worker_b,
            )
        self.check(
            "expired_lease_reclaimed_and_stale_owner_rejected",
            reclaimed_attempt == 2
            and not stale_complete
            and current_complete,
            reclaimed_attempt=reclaimed_attempt,
            stale_complete=stale_complete,
            current_complete=current_complete,
        )

        retry_event = await self._enqueue_event(
            owner=owner,
            workspace=workspace,
            label="lease-retry",
            target="summary_qdrant",
            operation="delete",
        )
        retry_a = f"todo96-retry-a-{self.run_id}"
        retry_b = f"todo96-retry-b-{self.run_id}"
        async with meta_mysql_client_manager.session_factory() as session:
            claimed = await ConversationMemoryRepository.claim_pending_outbox(
                session,
                limit=1,
                max_attempts=8,
                worker_id=retry_a,
                lease_seconds=30,
                owner_allowlist=(owner,),
            )
            self.check(
                "retry_event_claimed_by_first_worker",
                len(claimed) == 1 and claimed[0].event_id == retry_event,
            )
        async with meta_mysql_client_manager.session_factory() as session:
            failed = await ConversationMemoryRepository.fail_outbox(
                session,
                retry_event,
                "SyntheticRetry",
                1,
                worker_id=retry_a,
            )
            await session.execute(update(ConversationMemoryOutboxMySQL).where(
                ConversationMemoryOutboxMySQL.event_id == retry_event
            ).values(next_retry_at=_utcnow() - timedelta(seconds=1)))
            await session.commit()
        async with meta_mysql_client_manager.session_factory() as session:
            reclaimed = await ConversationMemoryRepository.claim_pending_outbox(
                session,
                limit=1,
                max_attempts=8,
                worker_id=retry_b,
                lease_seconds=30,
                owner_allowlist=(owner,),
            )
            retry_attempt = int(reclaimed[0].attempt_count)
            retry_completed = await ConversationMemoryRepository.complete_outbox(
                session,
                retry_event,
                worker_id=retry_b,
            )
        self.check(
            "failed_event_retried_by_second_worker",
            failed and retry_attempt == 2 and retry_completed,
            retry_attempt=retry_attempt,
        )

        processing_event = await self._enqueue_event(
            owner=owner,
            workspace=workspace,
            label="lease-processing-requeue",
            target="summary_es",
            operation="upsert",
        )
        processing_worker = f"todo96-processing-{self.run_id}"
        async with meta_mysql_client_manager.session_factory() as session:
            rows = await ConversationMemoryRepository.claim_pending_outbox(
                session,
                limit=1,
                max_attempts=8,
                worker_id=processing_worker,
                lease_seconds=30,
                owner_allowlist=(owner,),
            )
            self.check(
                "processing_event_claimed",
                len(rows) == 1 and rows[0].event_id == processing_event,
            )
        async with meta_mysql_client_manager.session_factory() as session:
            row = await ConversationMemoryRepository(
                session,
                owner_id=owner,
                workspace_id=workspace,
            ).enqueue_projection(
                memory_id=self.uid("lease-processing-requeue"),
                target="summary_es",
                operation="upsert",
                projection_version=self.es_version,
                requeue_existing=True,
            )
            preserved = {
                "status": row.status,
                "worker_id": row.worker_id,
                "attempt_count": int(row.attempt_count),
            }
            await session.commit()
        async with meta_mysql_client_manager.session_factory() as session:
            completed = await ConversationMemoryRepository.complete_outbox(
                session,
                processing_event,
                worker_id=processing_worker,
            )
        self.check(
            "backfill_requeue_preserves_live_processing_lease",
            preserved == {
                "status": "processing",
                "worker_id": processing_worker,
                "attempt_count": 1,
            } and completed,
            preserved=preserved,
        )
        self.report["worker_lease_competition"] = {
            "skip_locked_worker_a": batch_a,
            "skip_locked_worker_b": batch_b,
            "crash_reclaim_attempt": reclaimed_attempt,
            "retry_attempt": retry_attempt,
            "processing_requeue": preserved,
        }
        await self._cleanup_owners((owner,))

    def _segment(
        self,
        *,
        label: str,
        owner: str,
        workspace: str,
        chat_id: str,
        index: int,
        content: str,
    ) -> ConversationSummarySegmentMySQL:
        digest = hashlib.sha256(
            f"{self.run_id}:{label}:{index}:{content}".encode("utf-8")
        ).hexdigest()
        return ConversationSummarySegmentMySQL(
            summary_id=self.uid(f"{label}-summary-{index}"),
            owner_id=owner,
            workspace_id=workspace,
            chat_id=chat_id,
            level=index % 3,
            source_start_message_id=index * 2 + 1,
            source_end_message_id=index * 2 + 2,
            source_message_ids=[index * 2 + 1, index * 2 + 2],
            child_summary_ids=[],
            source_content_hash=f"sha256:{digest}",
            content_masked=content,
            structured_facts=[],
            open_items=[],
            source_token_count=12,
            summary_token_count=6,
            summary_schema_version="summary-segment-v1",
            extractor_version="summary-extractive-v1",
            projection_version=(
                f"es:{self.es_version}|qdrant:{self.qdrant_version}"
            ),
            status="active",
        )

    async def backfill_capacity(self) -> None:
        owner = self.uid("capacity-owner")
        workspace = self.uid("capacity-workspace")
        chat_id = f"todo96-capacity-{self.run_id}"[:64]
        self._owners.add(owner)
        started = time.perf_counter()
        async with meta_mysql_client_manager.session_factory() as session:
            session.add_all([
                self._segment(
                    label="capacity",
                    owner=owner,
                    workspace=workspace,
                    chat_id=chat_id,
                    index=index,
                    content=f"TODO-96 capacity summary {index}",
                )
                for index in range(self.capacity_segments)
            ])
            await session.commit()
        seed_seconds = time.perf_counter() - started

        async with meta_mysql_client_manager.session_factory() as session:
            started = time.perf_counter()
            first = await enqueue_summary_projection_backfill(
                session,
                batch_size=500,
            )
            await session.commit()
            first_seconds = time.perf_counter() - started
        async with meta_mysql_client_manager.session_factory() as session:
            started = time.perf_counter()
            second = await enqueue_summary_projection_backfill(
                session,
                batch_size=500,
            )
            await session.commit()
            second_seconds = time.perf_counter() - started
            event_count = int(await session.scalar(
                select(func.count()).select_from(
                    ConversationMemoryOutboxMySQL
                ).where(ConversationMemoryOutboxMySQL.owner_id == owner)
            ) or 0)
            distinct_by_target = {
                str(target): int(count)
                for target, count in (await session.execute(select(
                    ConversationMemoryOutboxMySQL.target,
                    func.count(func.distinct(
                        ConversationMemoryOutboxMySQL.memory_id
                    )),
                ).where(
                    ConversationMemoryOutboxMySQL.owner_id == owner
                ).group_by(
                    ConversationMemoryOutboxMySQL.target
                ))).all()
            }
            pending_status = await summary_projection_backfill_status(session)

        expected_events = self.capacity_segments * 2
        self.check(
            "backfill_crosses_500_row_page_boundary",
            self.capacity_segments > 1000
            and math.ceil(self.capacity_segments / 500) >= 3,
            active_segments=self.capacity_segments,
            page_size=500,
            expected_pages=math.ceil(self.capacity_segments / 500),
        )
        self.check(
            "backfill_first_pass_complete_cardinality",
            first == {
                "active_segments": self.capacity_segments,
                "requested_events": expected_events,
            }
            and event_count == expected_events
            and distinct_by_target == {
                "summary_es": self.capacity_segments,
                "summary_qdrant": self.capacity_segments,
            },
            result=first,
            event_count=event_count,
            distinct_by_target=distinct_by_target,
        )
        self.check(
            "backfill_second_pass_is_row_idempotent",
            second == first and event_count == expected_events,
            result=second,
            event_count=event_count,
        )
        self.check(
            "backfill_not_ready_before_delivery",
            not bool(pending_status.get("ready")),
            status=pending_status,
        )

        async with meta_mysql_client_manager.session_factory() as session:
            await session.execute(update(ConversationMemoryOutboxMySQL).where(
                ConversationMemoryOutboxMySQL.owner_id == owner
            ).values(
                status="completed",
                last_error=None,
                next_retry_at=None,
                claimed_at=None,
                lease_until=None,
                worker_id=None,
                completed_at=_utcnow(),
            ))
            await session.commit()
            completed_status = await summary_projection_backfill_status(session)
        self.check(
            "backfill_readiness_counts_nonzero_completed_set",
            completed_status == {
                "active_segments": self.capacity_segments,
                "summary_es_completed": self.capacity_segments,
                "summary_qdrant_completed": self.capacity_segments,
                "ready": True,
            },
            status=completed_status,
        )
        self.report["backfill_capacity"] = {
            "active_segments": self.capacity_segments,
            "requested_events": expected_events,
            "page_size": 500,
            "pages": math.ceil(self.capacity_segments / 500),
            "seed_seconds": round(seed_seconds, 3),
            "first_enqueue_seconds": round(first_seconds, 3),
            "first_events_per_second": round(
                expected_events / max(first_seconds, 0.001), 3
            ),
            "second_requeue_seconds": round(second_seconds, 3),
            "second_events_per_second": round(
                expected_events / max(second_seconds, 0.001), 3
            ),
            "pending_status": pending_status,
            "completed_status": completed_status,
            "scope": "database_enqueue_and_readiness_only",
        }
        await self._cleanup_owners((owner,))

    async def _external_counts(self) -> dict[str, Any]:
        es_repo = ConversationSummaryESRepository(
            es_client_manager.client,
            projection_version=self.es_version,
            manage_alias=False,
        )
        if await es_client_manager.client.indices.exists(
            index=es_repo.physical_index
        ):
            await es_client_manager.client.indices.refresh(
                index=es_repo.physical_index
            )
            es_count = int(_response_body(await es_client_manager.client.count(
                index=es_repo.physical_index
            )).get("count") or 0)
        else:
            es_count = 0
        alias_targets: list[str] = []
        if await es_client_manager.client.indices.exists_alias(
            name=es_repo.read_alias
        ):
            alias_targets = sorted(_response_body(
                await es_client_manager.client.indices.get_alias(
                    name=es_repo.read_alias
                )
            ))
        qdrant_count = 0
        if await qdrant_client_manager.client.collection_exists(
            collection_name=self.qdrant_collection
        ):
            qdrant_count = int((await qdrant_client_manager.client.count(
                collection_name=self.qdrant_collection,
                count_filter=qdrant_models.Filter(must=[
                    qdrant_models.FieldCondition(
                        key="projection_version",
                        match=qdrant_models.MatchValue(value=self.qdrant_version),
                    )
                ]),
                exact=True,
            )).count)
        return {
            "es_physical_index": es_repo.physical_index,
            "es_document_count": es_count,
            "es_alias": es_repo.read_alias,
            "es_alias_targets": alias_targets,
            "qdrant_collection": self.qdrant_collection,
            "qdrant_projection_count": qdrant_count,
        }

    async def _canary_outbox_state(self, owners: Iterable[str]) -> dict[str, int]:
        values = tuple(owners)
        async with meta_mysql_client_manager.session_factory() as session:
            rows = (await session.execute(select(
                ConversationMemoryOutboxMySQL.status,
                func.count(ConversationMemoryOutboxMySQL.event_id),
            ).where(
                ConversationMemoryOutboxMySQL.owner_id.in_(values)
            ).group_by(
                ConversationMemoryOutboxMySQL.status
            ))).all()
        return {str(status): int(count) for status, count in rows}

    async def _drain_two_workers(
        self,
        workers: tuple[MemoryProjectionWorker, MemoryProjectionWorker],
        owners: Iterable[str],
    ) -> dict[str, Any]:
        rounds: list[dict[str, Any]] = []
        started = time.perf_counter()
        for round_index in range(20):
            before = await self._canary_outbox_state(owners)
            backlog = sum(
                count for status, count in before.items()
                if status != "completed"
            )
            if backlog == 0:
                break
            round_started = time.perf_counter()
            counts = await asyncio.gather(
                workers[0].process_once(),
                workers[1].process_once(),
            )
            after = await self._canary_outbox_state(owners)
            rounds.append({
                "round": round_index + 1,
                "claimed": [int(value) for value in counts],
                "seconds": round(time.perf_counter() - round_started, 3),
                "before": before,
                "after": after,
            })
            if not any(counts) and any(
                status != "completed" and count > 0
                for status, count in after.items()
            ):
                await asyncio.sleep(2.2)
        await workers[0].process_once()
        final = await self._canary_outbox_state(owners)
        return {
            "seconds": round(time.perf_counter() - started, 3),
            "rounds": rounds,
            "final": final,
            "workers": [worker.status() for worker in workers],
        }

    async def external_canary(self) -> None:
        owners = [self.uid("canary-owner-a"), self.uid("canary-owner-b")]
        workspaces = [
            self.uid("canary-workspace-a"),
            self.uid("canary-workspace-b"),
        ]
        self._owners.update(owners)
        chat_id = f"todo96-canary-{self.run_id}"[:64]
        scope_ids: dict[tuple[str, str], list[str]] = {}
        async with meta_mysql_client_manager.session_factory() as session:
            for index in range(self.canary_segments):
                owner = owners[index % len(owners)]
                workspace = workspaces[(index // len(owners)) % len(workspaces)]
                segment = self._segment(
                    label="external-canary",
                    owner=owner,
                    workspace=workspace,
                    chat_id=chat_id,
                    index=index,
                    content=f"TODO-96 canary 销售额汇总 {index}",
                )
                session.add(segment)
                session.add(ConversationSummarySignalMySQL(
                    signal_id=self.uid(f"external-canary-signal-{index}"),
                    summary_id=segment.summary_id,
                    owner_id=owner,
                    workspace_id=workspace,
                    chat_id=chat_id,
                    signal_type="metric",
                    normalized_value="销售额",
                    weight=2.0,
                    source_order=0,
                    source_message_ids=list(segment.source_message_ids),
                    extractor_version="summary-signals-rule-v1",
                    sensitivity="masked",
                    status="active",
                ))
                scope_ids.setdefault((owner, workspace), []).append(
                    segment.summary_id
                )
            await session.commit()

        es_client_manager.init()
        qdrant_client_manager.init()
        embedding = embedding_client_manager.init()
        self._external_initialized = True
        es_repo = ConversationSummaryESRepository(
            es_client_manager.client,
            projection_version=self.es_version,
            manage_alias=False,
        )
        qdrant_repo = ConversationSummaryQdrantRepository(
            qdrant_client_manager.client,
            projection_version=self.qdrant_version,
        )
        await es_repo.ensure_index()
        await qdrant_repo.ensure_collection()
        warm_started = time.perf_counter()
        query_embedding = await embedding.aembed_query(
            "TODO-96 canary 销售额"
        )
        warm_seconds = time.perf_counter() - warm_started
        self.check(
            "real_bge_m3_embedding_contract",
            len(query_embedding.dense) == int(app_config.embedding.dense_size)
            and bool(query_embedding.sparse_indices),
            dense_size=len(query_embedding.dense),
            sparse_nonzero=len(query_embedding.sparse_indices),
            cold_start_seconds=round(warm_seconds, 3),
        )

        app_config.memory.summary_enabled = True
        app_config.memory.summary_build_enabled = True
        app_config.memory.summary_projection_enabled = True
        async with meta_mysql_client_manager.session_factory() as session:
            enqueue_result = await enqueue_summary_projection_backfill(
                session,
                batch_size=3,
            )
            await session.commit()
        self.check(
            "external_canary_enqueued_two_targets_per_segment",
            enqueue_result == {
                "active_segments": self.canary_segments,
                "requested_events": self.canary_segments * 2,
            },
            result=enqueue_result,
        )
        workers = (
            MemoryProjectionWorker(owner_allowlist=owners),
            MemoryProjectionWorker(owner_allowlist=owners),
        )
        drain = await self._drain_two_workers(workers, owners)
        async with meta_mysql_client_manager.session_factory() as session:
            ready = await summary_projection_backfill_status(session)
            attempts = list((await session.execute(select(
                ConversationMemoryOutboxMySQL.attempt_count
            ).where(
                ConversationMemoryOutboxMySQL.owner_id.in_(tuple(owners)),
                ConversationMemoryOutboxMySQL.operation == "upsert",
            ))).scalars().all())
        projected_counts = await self._external_counts()
        self.check(
            "two_real_workers_both_processed_events",
            all(worker.processed > 0 for worker in workers)
            and all(worker.failed == 0 for worker in workers),
            workers=drain["workers"],
        )
        self.check(
            "real_worker_drain_has_no_backlog_or_retry",
            drain["final"] == {"completed": self.canary_segments * 2}
            and attempts
            and max(int(value) for value in attempts) == 1,
            final=drain["final"],
            maximum_attempt=max(int(value) for value in attempts),
        )
        self.check(
            "mysql_nonzero_backfill_ready",
            ready == {
                "active_segments": self.canary_segments,
                "summary_es_completed": self.canary_segments,
                "summary_qdrant_completed": self.canary_segments,
                "ready": True,
            },
            status=ready,
        )
        self.check(
            "physical_projection_counts_and_alias_reconcile",
            projected_counts["es_document_count"] == self.canary_segments
            and projected_counts["qdrant_projection_count"]
            == self.canary_segments
            and projected_counts["es_alias_targets"]
            == [projected_counts["es_physical_index"]],
            counts=projected_counts,
        )

        async def retrieve_scope(
            owner_id: str,
            workspace_id: str,
            scoped_chat_id: str,
            *,
            limit: int,
        ) -> dict[str, Any]:
            async with meta_mysql_client_manager.session_factory() as session:
                mysql_ids, mysql_evidence = await ConversationSummaryRepository(
                    session,
                    owner_id=owner_id,
                    workspace_id=workspace_id,
                ).search_signal_ids(
                    scoped_chat_id,
                    ["销售额"],
                    limit=limit,
                )
            es_ids, es_evidence = await es_repo.search(
                "销售额",
                owner_id=owner_id,
                workspace_id=workspace_id,
                chat_id=scoped_chat_id,
                limit=limit,
            )
            qdrant_ids, qdrant_evidence, degradations = await qdrant_repo.search(
                query_embedding,
                query="销售额",
                owner_id=owner_id,
                workspace_id=workspace_id,
                chat_id=scoped_chat_id,
                limit=limit,
            )
            return {
                "mysql_ids": set(mysql_ids),
                "es_ids": set(es_ids),
                "qdrant_ids": set(qdrant_ids),
                "mysql_status": mysql_evidence.get("status"),
                "es_status": es_evidence.get("status"),
                "qdrant_status": qdrant_evidence.get("status"),
                "degradations": degradations,
            }

        scope_results: list[dict[str, Any]] = []
        negative_scope_results: list[dict[str, Any]] = []
        wrong_owner = self.uid("canary-wrong-owner")
        wrong_workspace = self.uid("canary-wrong-workspace")
        wrong_chat = f"todo96-wrong-chat-{self.run_id}"[:64]
        for (owner, workspace), expected in sorted(scope_ids.items()):
            expected_set = set(expected)
            retrieved = await retrieve_scope(
                owner,
                workspace,
                chat_id,
                limit=max(20, len(expected_set)),
            )
            result = {
                "owner": owner,
                "workspace": workspace,
                "expected": len(expected_set),
                "mysql": len(retrieved["mysql_ids"]),
                "es": len(retrieved["es_ids"]),
                "qdrant": len(retrieved["qdrant_ids"]),
                "mysql_status": retrieved["mysql_status"],
                "es_status": retrieved["es_status"],
                "qdrant_status": retrieved["qdrant_status"],
                "degradations": retrieved["degradations"],
            }
            scope_results.append(result)
            self.check(
                f"scope_isolation_{owner[:8]}_{workspace[:8]}",
                retrieved["mysql_ids"] == expected_set
                and retrieved["es_ids"] == expected_set
                and retrieved["qdrant_ids"] == expected_set
                and not retrieved["degradations"],
                result=result,
            )
            negative_variants = {
                "wrong_owner": (wrong_owner, workspace, chat_id),
                "wrong_workspace": (owner, wrong_workspace, chat_id),
                "wrong_chat": (owner, workspace, wrong_chat),
            }
            for label, (negative_owner, negative_workspace, negative_chat) in (
                negative_variants.items()
            ):
                negative = await retrieve_scope(
                    negative_owner,
                    negative_workspace,
                    negative_chat,
                    limit=max(20, len(expected_set)),
                )
                negative_result = {
                    "scope_owner": owner,
                    "scope_workspace": workspace,
                    "variant": label,
                    "mysql": len(negative["mysql_ids"]),
                    "es": len(negative["es_ids"]),
                    "qdrant": len(negative["qdrant_ids"]),
                    "degradations": negative["degradations"],
                }
                negative_scope_results.append(negative_result)
                self.check(
                    f"negative_{label}_{owner[:8]}_{workspace[:8]}",
                    not negative["mysql_ids"]
                    and not negative["es_ids"]
                    and not negative["qdrant_ids"]
                    and not negative["degradations"],
                    result=negative_result,
                )

        delete_requested = 0
        for (owner, workspace), expected in scope_ids.items():
            async with meta_mysql_client_manager.session_factory() as session:
                revoked = await ConversationSummaryRepository(
                    session,
                    owner_id=owner,
                    workspace_id=workspace,
                ).revoke_chat(chat_id)
                self.check(
                    f"scope_revoke_{owner[:8]}_{workspace[:8]}",
                    set(revoked) == set(expected),
                    revoked_count=len(revoked),
                )
                outbox_repo = ConversationMemoryRepository(
                    session,
                    owner_id=owner,
                    workspace_id=workspace,
                )
                for summary_id in revoked:
                    delete_requested += len(
                        await outbox_repo.enqueue_projection_deletes(
                            memory_id=summary_id,
                            target="summary_es",
                            current_projection_version=self.es_version,
                        )
                    )
                    delete_requested += len(
                        await outbox_repo.enqueue_projection_deletes(
                            memory_id=summary_id,
                            target="summary_qdrant",
                            current_projection_version=self.qdrant_version,
                        )
                    )
                await session.commit()
        delete_drain = await self._drain_two_workers(workers, owners)
        deleted_counts = await self._external_counts()
        async with meta_mysql_client_manager.session_factory() as session:
            active_after_revoke = int(await session.scalar(
                select(func.count()).select_from(
                    ConversationSummarySegmentMySQL
                ).where(
                    ConversationSummarySegmentMySQL.owner_id.in_(tuple(owners)),
                    ConversationSummarySegmentMySQL.status == "active",
                )
            ) or 0)
            completed_deletes = int(await session.scalar(
                select(func.count()).select_from(
                    ConversationMemoryOutboxMySQL
                ).where(
                    ConversationMemoryOutboxMySQL.owner_id.in_(tuple(owners)),
                    ConversationMemoryOutboxMySQL.operation == "delete",
                    ConversationMemoryOutboxMySQL.status == "completed",
                )
            ) or 0)
        self.check(
            "revoke_delete_drained_all_projection_versions",
            delete_requested == self.canary_segments * 2
            and completed_deletes == delete_requested
            and delete_drain["final"] == {
                "completed": self.canary_segments * 4
            },
            requested=delete_requested,
            completed=completed_deletes,
            outbox=delete_drain["final"],
        )
        self.check(
            "external_delete_has_zero_residual_documents",
            active_after_revoke == 0
            and deleted_counts["es_document_count"] == 0
            and deleted_counts["qdrant_projection_count"] == 0,
            active_after_revoke=active_after_revoke,
            counts=deleted_counts,
        )
        self.report["external_canary"] = {
            "segments": self.canary_segments,
            "embedding_cold_start_seconds": round(warm_seconds, 3),
            "upsert_drain": drain,
            "ready": ready,
            "projected_counts": projected_counts,
            "scope_results": scope_results,
            "negative_scope_results": negative_scope_results,
            "delete_requested": delete_requested,
            "delete_drain": delete_drain,
            "deleted_counts": deleted_counts,
        }
        await self._cleanup_owners(owners)

    async def _cleanup_external(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "es_index_deleted": False,
            "qdrant_collection_deleted": False,
        }
        errors: dict[str, dict[str, str]] = {}
        if self.run_id not in self.es_base or self.run_id not in self.qdrant_collection:
            raise RuntimeError("拒绝清理非 run-scoped 外部资源")
        if es_client_manager.client is not None:
            try:
                es_repo = ConversationSummaryESRepository(
                    es_client_manager.client,
                    projection_version=self.es_version,
                    manage_alias=False,
                )
                if await es_client_manager.client.indices.exists(
                    index=es_repo.physical_index
                ):
                    await es_client_manager.client.indices.delete(
                        index=es_repo.physical_index
                    )
                    result["es_index_deleted"] = True
                result["es_index_absent"] = not bool(
                    await es_client_manager.client.indices.exists(
                        index=es_repo.physical_index
                    )
                )
                result["es_alias_absent"] = not bool(
                    await es_client_manager.client.indices.exists_alias(
                        name=es_repo.read_alias
                    )
                )
            except BaseException as exc:
                errors["elasticsearch"] = {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
        if qdrant_client_manager.client is not None:
            try:
                if await qdrant_client_manager.client.collection_exists(
                    collection_name=self.qdrant_collection
                ):
                    await qdrant_client_manager.client.delete_collection(
                        collection_name=self.qdrant_collection
                    )
                    result["qdrant_collection_deleted"] = True
                result["qdrant_collection_absent"] = not bool(
                    await qdrant_client_manager.client.collection_exists(
                        collection_name=self.qdrant_collection
                    )
                )
            except BaseException as exc:
                errors["qdrant"] = {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
        if errors:
            result["errors"] = errors
        return result

    async def final_reconciliation(self) -> None:
        await self._cleanup_owners(self._owners)
        current_outbox = await self._outbox_distribution()
        summary_counts = await self._summary_counts()
        self.check(
            "database_cleanup_restored_outbox_baseline",
            current_outbox == self._baseline_outbox,
            baseline=self._baseline_outbox,
            current=current_outbox,
        )
        self.check(
            "database_cleanup_removed_all_synthetic_summary_rows",
            not any(summary_counts.values()),
            counts=summary_counts,
        )

    async def run(self) -> dict[str, Any]:
        meta_mysql_client_manager.init()
        self._configure_isolated_projection()
        failure: BaseException | None = None
        try:
            await self.preflight()
            await self.lock_and_generation_fence()
            await self.build_only_canary()
            await self.atomic_creation_competition()
            await self.worker_lease_competition()
            await self.backfill_capacity()
            await self.external_canary()
            await self.final_reconciliation()
            self.report["status"] = "passed"
        except BaseException as exc:
            failure = exc
            self.report["status"] = "failed"
            self.report["failure"] = {
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        finally:
            cleanup_errors: dict[str, dict[str, str]] = {}
            try:
                await self._cleanup_owners(self._owners)
            except BaseException as exc:
                cleanup_errors["database"] = {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            if self._external_initialized:
                try:
                    external_cleanup = await self._cleanup_external()
                    self.report["external_cleanup"] = external_cleanup
                    cleanup_errors.update(external_cleanup.get("errors", {}))
                except BaseException as exc:
                    cleanup_errors["external"] = {
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
            try:
                self._restore_config()
            except BaseException as exc:
                cleanup_errors["config_restore"] = {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            self.report["flags_restored"] = {
                name: bool(getattr(app_config.memory, name))
                for name in (
                    "summary_enabled",
                    "summary_build_enabled",
                    "summary_projection_enabled",
                    "stable_fact_enabled",
                )
            }
            for name, manager in (
                ("embedding_close", embedding_client_manager),
                ("qdrant_close", qdrant_client_manager),
                ("elasticsearch_close", es_client_manager),
                ("mysql_close", meta_mysql_client_manager),
            ):
                try:
                    await manager.close()
                except BaseException as exc:
                    cleanup_errors[name] = {
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
            if cleanup_errors:
                self.report["cleanup_errors"] = cleanup_errors
                self.report["status"] = "failed"
                if failure is None:
                    failure = RuntimeError(
                        "TODO-96 acceptance cleanup failed: "
                        + ", ".join(sorted(cleanup_errors))
                    )
            self.report["finished_at"] = datetime.now(timezone.utc).isoformat()
        if failure is not None:
            raise AcceptanceFailure(json.dumps(
                self.report,
                ensure_ascii=False,
                default=str,
            )) from failure
        return self.report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required acknowledgement that synthetic DB/external writes will run.",
    )
    parser.add_argument("--capacity-segments", type=int, default=1001)
    parser.add_argument("--canary-segments", type=int, default=8)
    parser.add_argument("--lock-hold-ms", type=int, default=500)
    parser.add_argument(
        "--report-dir",
        default=str(REPORTS_ROOT),
        help="Directory for the complete raw acceptance JSON report.",
    )
    parser.add_argument(
        "--allow-nonlocal",
        action="store_true",
        help="Allow a target other than loopback meta2 (unsafe; explicit only).",
    )
    parser.add_argument(
        "--allow-large",
        action="store_true",
        help=(
            "Allow capacity/canary sizes above the safe defaults, while still "
            "enforcing absolute safety limits."
        ),
    )
    return parser.parse_args()


def _persist_report(report: dict[str, Any], report_dir: str) -> Path:
    run_id = "".join(
        char if char.isalnum() or char == "-" else "-"
        for char in str(report.get("run_id") or "unknown")
    )[:96]
    directory = Path(report_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"todo96_runtime_{run_id}.json"
    report["report_path"] = str(target)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


async def _main() -> int:
    arguments = _arguments()
    if not arguments.execute:
        raise SystemExit("Refusing writes without --execute")
    if arguments.capacity_segments < 1001:
        raise SystemExit("--capacity-segments must be >= 1001")
    if arguments.canary_segments < 4 or arguments.canary_segments % 4:
        raise SystemExit("--canary-segments must be a positive multiple of 4")
    if arguments.capacity_segments > ABSOLUTE_CAPACITY_LIMIT:
        raise SystemExit(
            f"--capacity-segments must be <= {ABSOLUTE_CAPACITY_LIMIT}"
        )
    if arguments.canary_segments > ABSOLUTE_CANARY_LIMIT:
        raise SystemExit(
            f"--canary-segments must be <= {ABSOLUTE_CANARY_LIMIT}"
        )
    if not arguments.allow_large and (
        arguments.capacity_segments > SAFE_CAPACITY_LIMIT
        or arguments.canary_segments > SAFE_CANARY_LIMIT
    ):
        raise SystemExit(
            "large acceptance sizes require --allow-large "
            f"(safe limits: capacity={SAFE_CAPACITY_LIMIT}, "
            f"canary={SAFE_CANARY_LIMIT})"
        )
    if arguments.lock_hold_ms < 250:
        raise SystemExit("--lock-hold-ms must be >= 250")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S") + "-" + uuid.uuid4().hex[:6]
    acceptance = Todo96RuntimeAcceptance(
        run_id=run_id,
        capacity_segments=arguments.capacity_segments,
        canary_segments=arguments.canary_segments,
        lock_hold_ms=arguments.lock_hold_ms,
        allow_nonlocal=arguments.allow_nonlocal,
    )
    exit_code = 0
    try:
        report = await acceptance.run()
    except AcceptanceFailure as exc:
        try:
            report = json.loads(str(exc))
        except json.JSONDecodeError:
            report = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        exit_code = 1
    try:
        _persist_report(report, arguments.report_dir)
    except BaseException as exc:
        report["report_write_failure"] = {
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        report["status"] = "failed"
        exit_code = 1
    print("TODO96_ACCEPTANCE_JSON_BEGIN")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("TODO96_ACCEPTANCE_JSON_END")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
