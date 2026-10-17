"""Layered conversation memory orchestration for the NL2SQL request path."""
from __future__ import annotations

import asyncio
import logging
from collections import Counter
from typing import Any, Mapping

from app.application.ports.conversation import (
    ConversationEmbeddingPort,
    ConversationMemoryPersistence,
    ConversationMemorySearchPort,
    ConversationMemoryVectorSearchPort,
    ConversationUnavailableError,
    NullProjectionNotifier,
    ProjectionNotifier,
    WorkingMemoryPort,
)
from app.application.services.conversation_memory_policy import (
    build_working_payload,
    extract_explicit_stable_facts,
    extract_memory_candidates,
    extract_summary_signals,
    estimate_tokens,
    iso_now,
    lexical_relevance,
    new_empty_memory_context,
    rrf_fuse,
    summary_retrieval_needed,
)
from app.application.settings.conversation import (
    DEFAULT_MEMORY_SETTINGS,
    ConversationMemorySettings,
)
from app.application.services.conversation_summary_service import ConversationSummaryService
from app.domain.privacy.masking import mask_free_text


logger = logging.getLogger(__name__)


def _safe_degradation(node_id: str, reason_code: str, exc: BaseException) -> dict:
    return {
        "node_id": node_id,
        "route": "memory",
        "reason_code": reason_code,
        "severity": "warning",
        "affects_core_path": False,
        "detail": type(exc).__name__,
        "evidence_complete": True,
    }


class ConversationMemoryService:
    """Build and persist a scoped memory context without trusting projections."""

    def __init__(
        self,
        *,
        owner_id: str,
        workspace_id: str,
        persistence: ConversationMemoryPersistence,
        embedding_client: ConversationEmbeddingPort,
        es_repository: ConversationMemorySearchPort,
        qdrant_repository: ConversationMemoryVectorSearchPort,
        summary_es_repository: ConversationMemorySearchPort | None = None,
        summary_qdrant_repository: ConversationMemoryVectorSearchPort | None = None,
        working_memory: WorkingMemoryPort | None = None,
        notifier: ProjectionNotifier | None = None,
        settings: ConversationMemorySettings = DEFAULT_MEMORY_SETTINGS,
    ):
        self.owner_id = str(owner_id or "").strip()
        self.workspace_id = str(workspace_id or "").strip()
        if not self.owner_id or not self.workspace_id:
            raise ValueError("ConversationMemoryService 必须绑定 owner/workspace")
        self.persistence = persistence
        self.embedding_client = embedding_client
        self.es_repository = es_repository
        self.qdrant_repository = qdrant_repository
        self.summary_es_repository = summary_es_repository
        self.summary_qdrant_repository = summary_qdrant_repository
        self.working_memory = working_memory
        self.notifier = notifier or NullProjectionNotifier()
        self.settings = settings

    def _redis_repo(self) -> WorkingMemoryPort | None:
        return self.working_memory

    async def start_turn(
        self, *, chat_id: str, query: str, trace_id: str
    ) -> int:
        """Write the user message before Agent execution; failure is not hidden."""
        created_stable = False
        async with self.persistence.transaction() as uow:
            repo = uow.memory
            history_repo = uow.history
            existing_session = await history_repo.get_session(
                chat_id,
                include_archived=True,
                include_deleted=True,
                for_update=True,
            )
            if existing_session is not None and existing_session.status != "active":
                raise ConversationUnavailableError("会话已归档或正在删除")
            row = await repo.append_message(
                chat_id=chat_id,
                role="user",
                content_masked=mask_free_text(
                    query,
                    policy=self.settings.masking_policy,
                    max_chars=self.settings.message_content_max_chars,
                ),
                request_id=trace_id,
                trace_id=trace_id,
                status="pending",
            )
            await history_repo.start_turn(
                chat_id=chat_id,
                turn_id=trace_id,
                user_text=query,
                trace_id=trace_id,
                source_message_id=int(row.id),
            )
            if self.settings.stable_fact_enabled:
                for values in extract_explicit_stable_facts(
                    owner_id=self.owner_id,
                    workspace_id=self.workspace_id,
                    chat_id=chat_id,
                    query=query,
                    user_message_id=int(row.id),
                    settings=self.settings,
                ):
                    item, created = await repo.upsert_memory_item(values)
                    if not created:
                        continue
                    created_stable = True
                    replaced = await repo.supersede_active_fact_slot(
                        memory_type=item.memory_type,
                        fact_slot=str(
                            (item.structured_payload or {}).get("fact_slot") or ""
                        ),
                        superseded_by_id=item.memory_id,
                    )
                    for old_memory_id in replaced:
                        for target, version in (
                            ("es", self.settings.es_projection_version),
                            ("qdrant", self.settings.qdrant_projection_version),
                        ):
                            await repo.enqueue_projection_deletes(
                                memory_id=old_memory_id,
                                target=target,
                                current_projection_version=version,
                            )
                    for target, version in (
                        ("es", self.settings.es_projection_version),
                        ("qdrant", self.settings.qdrant_projection_version),
                    ):
                        await repo.enqueue_projection(
                            memory_id=item.memory_id,
                            target=target,
                            operation="upsert",
                            projection_version=version,
                        )
            await uow.commit()
            message_id = int(row.id)
        if created_stable:
            self.notifier.notify()
        return message_id

    async def _working_from_mysql(
        self, chat_id: str
    ) -> tuple[dict, list[Any], str, int]:
        async with self.persistence.transaction() as uow:
            repo = uow.memory
            checkpoint = await repo.get_checkpoint(chat_id)
            messages = await repo.recent_messages(
                chat_id, self.settings.recent_message_limit
            )
        structured_state = dict(
            getattr(checkpoint, "structured_state", None) or {}
        )
        payload = build_working_payload(
            semantic_state=structured_state,
            rolling_summary=str(
                getattr(checkpoint, "rolling_summary", None) or ""
            ),
            recent_message_ids=[int(row.id) for row in messages],
            last_success_question=str(
                structured_state.get("standalone_question") or ""
            ),
            settings=self.settings,
        )
        return (
            payload,
            messages,
            "mysql_checkpoint" if checkpoint else "mysql_messages",
            int(getattr(checkpoint, "version", 0) or 0),
        )

    def _bounded_recent_messages(self, rows: list[Any]) -> tuple[list[dict], int]:
        selected: list[dict] = []
        used = 0
        budget = max(100, int(self.settings.recent_token_budget))
        for row in reversed(rows):
            item = {
                "message_id": int(row.id),
                "role": str(row.role),
                "content": str(row.content_masked),
                "created_at": str(row.created_at),
            }
            cost = estimate_tokens(item)
            if selected and used + cost > budget:
                continue
            selected.append(item)
            used += cost
        selected.reverse()
        return selected, max(0, len(rows) - len(selected))

    async def _load_summary_context(
        self,
        *,
        chat_id: str,
        query: str,
        recent_message_ids: set[int],
        omitted_recent: int,
    ) -> tuple[list[dict], list[dict], dict, list[dict]]:
        """Gated three-route retrieval followed by MySQL authority and range de-dup."""
        if not self.settings.summary_enabled:
            return [], [], {"status": "disabled"}, []
        query_signals = [
            str(item.get("normalized_value") or "")
            for item in extract_summary_signals(
                query,
                source_message_ids=[],
                limit=self.settings.summary_signal_limit,
                settings=self.settings,
            )
        ]
        projection_backfill: dict = {"ready": False, "status": "disabled"}
        projection_backfill_error: Exception | None = None
        async with self.persistence.transaction() as uow:
            repo = uow.summary
            active = await repo.list_active_segments(chat_id, limit=200)
            mysql_ids, mysql_evidence = await repo.search_signal_ids(
                chat_id,
                query_signals,
                limit=self.settings.summary_top_k * 3,
            )
            if self.settings.summary_projection_enabled:
                try:
                    projection_backfill = dict(
                        await uow.summary_projection_backfill_status()
                    )
                    projection_backfill["status"] = (
                        "ready" if projection_backfill.get("ready") else "pending"
                    )
                except Exception as exc:
                    projection_backfill_error = exc
                    projection_backfill = {
                        "ready": False,
                        "status": "degraded",
                        "error_type": type(exc).__name__,
                    }
        needed, reason = summary_retrieval_needed(
            query,
            has_active_summary=bool(active),
            omitted_recent=omitted_recent,
            force=bool(self.settings.summary_retrieval_always),
        )
        if not needed and mysql_ids:
            needed, reason = True, "summary_signal_match"
        if not needed:
            return [], [], {
                "status": reason,
                "trigger_reason": reason,
                "active_segment_count": len(active),
                "query_signal_count": len(query_signals),
                "mysql": mysql_evidence,
                "projection_backfill": projection_backfill,
                "selected_count": 0,
                "recent_messages_in_prompt": True,
            }, []

        degradations: list[dict] = []
        if projection_backfill_error is not None:
            degradations.append(
                _safe_degradation(
                    "summary_memory_read",
                    "summary_projection_readiness_failed",
                    projection_backfill_error,
                )
            )
        es_ids: list[str] = []
        qdrant_ids: list[str] = []
        es_evidence: dict = {"status": "unavailable"}
        qdrant_evidence: dict = {"status": "unavailable"}

        async def es_route():
            if self.summary_es_repository is None:
                raise RuntimeError("summary ES repository 未配置")
            return await asyncio.wait_for(
                self.summary_es_repository.search(
                    query,
                    owner_id=self.owner_id,
                    workspace_id=self.workspace_id,
                    chat_id=chat_id,
                    limit=self.settings.summary_top_k * 3,
                ),
                timeout=float(self.settings.retrieval_timeout_seconds),
            )

        async def qdrant_route():
            if self.summary_qdrant_repository is None:
                raise RuntimeError("summary Qdrant repository 未配置")
            embedding = await self.embedding_client.aembed_query(query)
            return await self.summary_qdrant_repository.search(
                embedding,
                query=query,
                owner_id=self.owner_id,
                workspace_id=self.workspace_id,
                chat_id=chat_id,
                limit=self.settings.summary_top_k * 3,
            )

        if (
            self.settings.summary_projection_enabled
            and bool(projection_backfill.get("ready"))
        ):
            es_result, qdrant_result = await asyncio.gather(
                es_route(), qdrant_route(), return_exceptions=True
            )
            if isinstance(es_result, BaseException):
                es_evidence = {
                    "status": "degraded", "error_type": type(es_result).__name__
                }
                degradations.append(
                    _safe_degradation(
                        "summary_memory_read", "summary_es_retrieval_failed", es_result
                    )
                )
            else:
                es_ids, es_evidence = es_result
            if isinstance(qdrant_result, BaseException):
                qdrant_evidence = {
                    "status": "degraded", "error_type": type(qdrant_result).__name__
                }
                degradations.append(
                    _safe_degradation(
                        "summary_memory_read",
                        "summary_qdrant_retrieval_failed",
                        qdrant_result,
                    )
                )
            else:
                qdrant_ids, qdrant_evidence, qdrant_degradations = qdrant_result
                degradations.extend(qdrant_degradations)
        elif self.settings.summary_projection_enabled:
            es_evidence = {"status": "backfill_pending"}
            qdrant_evidence = {"status": "backfill_pending"}
        else:
            es_evidence = {"status": "disabled"}
            qdrant_evidence = {"status": "disabled"}

        fused_ids, provenance = rrf_fuse(
            {
                "summary_mysql_signals": mysql_ids,
                "summary_es_bm25": es_ids,
                "summary_qdrant_hybrid": qdrant_ids,
            },
            rrf_k=self.settings.fusion_rrf_k,
            limit=self.settings.summary_top_k * 2,
        )
        async with self.persistence.transaction() as uow:
            repo = uow.summary
            authoritative = await repo.list_authoritative_by_ids(chat_id, fused_ids)
            signals_by_id = await repo.list_signals_for_summaries(
                row.summary_id for row in authoritative
            )
        source_by_id = {
            row["memory_id"]: "+".join(row.get("sources") or [])
            for row in provenance
        }
        selected: list[dict] = []
        selected_ranges: list[set[int]] = []
        rejected_overlap = 0
        for row in authoritative:
            source_ids = {int(value) for value in list(row.source_message_ids or [])}
            if source_ids & recent_message_ids:
                rejected_overlap += 1
                continue
            if any(
                source_ids and existing
                and len(source_ids & existing) / max(1, min(len(source_ids), len(existing))) >= .8
                for existing in selected_ranges
            ):
                rejected_overlap += 1
                continue
            selected_ranges.append(source_ids)
            selected.append({
                "summary_id": row.summary_id,
                "level": int(row.level or 0),
                "content": row.content_masked,
                "source_message_ids": sorted(source_ids),
                "source_start_message_id": int(row.source_start_message_id),
                "source_end_message_id": int(row.source_end_message_id),
                "confidence": 1.0,
                "source": source_by_id.get(row.summary_id, "mysql_authority"),
                "structured_facts": list(row.structured_facts or []),
                "signals": [
                    {
                        "signal_type": signal.signal_type,
                        "normalized_value": signal.normalized_value,
                        "weight": float(signal.weight or 0),
                        "source_order": int(signal.source_order or 0),
                        "source_message_ids": list(signal.source_message_ids or []),
                    }
                    for signal in signals_by_id.get(row.summary_id, [])
                ],
            })
            if len(selected) >= int(self.settings.summary_top_k):
                break
        return selected, provenance, {
            "status": "selected" if selected else "no_authoritative_match",
            "trigger_reason": reason,
            "active_segment_count": len(active),
            "query_signal_count": len(query_signals),
            "mysql_candidates": len(mysql_ids),
            "es_candidates": len(es_ids),
            "qdrant_candidates": len(qdrant_ids),
            "selected_count": len(selected),
            "rejected_recent_or_lineage_overlap": rejected_overlap,
            "fusion_method": "summary_id_rrf",
            "authoritative_recheck": True,
            "recent_messages_in_prompt": True,
            "mysql": mysql_evidence,
            "es": es_evidence,
            "qdrant": qdrant_evidence,
            "projection_backfill": projection_backfill,
        }, degradations

    async def load_context(self, *, chat_id: str, query: str) -> dict:
        if not self.settings.enabled:
            return new_empty_memory_context(
                evidence={"status": "disabled"},
                settings=self.settings,
            )

        degradations: list[dict] = []
        redis_repo = self._redis_repo()
        working: dict | None = None
        redis_hit = False
        if redis_repo is not None:
            try:
                working = await redis_repo.get(chat_id)
                redis_hit = working is not None
            except Exception as exc:
                degradations.append(
                    _safe_degradation(
                        "memory_read", "redis_working_memory_failed", exc
                    )
                )
        payload, messages, working_source, mysql_version = await self._working_from_mysql(
            chat_id
        )
        redis_stale = bool(
            working is not None
            and (
                int(working.get("version") or 0) != int(mysql_version)
                or str(working.get("schema_version") or "")
                != str(self.settings.context_schema_version)
                or str(working.get("source_hash") or "")
                != str(payload.get("source_hash") or "")
            )
        )
        if working is None or redis_stale:
            working = {**payload, "version": mysql_version}
            if redis_repo is not None:
                try:
                    await redis_repo.replace(
                        chat_id,
                        version=mysql_version,
                        payload=payload,
                        schema_version=self.settings.context_schema_version,
                        updated_at=iso_now(),
                    )
                    working_source = (
                        "mysql_authority_reconciled_stale_redis"
                        if redis_stale
                        else f"{working_source}_rebuilt_to_redis"
                    )
                except Exception as exc:
                    degradations.append(
                        _safe_degradation(
                            "memory_read", "redis_rebuild_failed", exc
                        )
                    )
        else:
            working_source = "redis"
            known_ids = {
                int(value) for value in working.get("recent_message_ids") or []
            }
            if known_ids:
                messages = [row for row in messages if int(row.id) in known_ids]

        recent_messages, omitted_recent = self._bounded_recent_messages(messages)
        (
            selected_summaries,
            summary_provenance,
            summary_evidence,
            summary_degradations,
        ) = await self._load_summary_context(
            chat_id=chat_id,
            query=query,
            recent_message_ids={
                int(item["message_id"]) for item in recent_messages
            },
            omitted_recent=omitted_recent,
        )
        degradations.extend(summary_degradations)
        async with self.persistence.transaction() as uow:
            repo = uow.memory
            candidates = await repo.list_active_candidates(
                memory_types=(
                    "preference", "correction", "query_episode",
                    "profile_fact", "long_term_goal",
                ),
                limit=80,
            )

        preferences = [
            row for row in candidates if row.memory_type == "preference"
        ][:3]
        profile_facts = [
            row for row in candidates if row.memory_type == "profile_fact"
        ][:3]
        long_term_goals = [
            row for row in candidates if row.memory_type == "long_term_goal"
        ][:3]
        exact_ranked = sorted(
            (
                (lexical_relevance(query, row.content_masked), row)
                for row in candidates
                if row.memory_type not in {
                    "preference", "profile_fact", "long_term_goal"
                }
            ),
            key=lambda pair: (
                -pair[0],
                -float(pair[1].confidence or 0),
                str(pair[1].memory_id),
            ),
        )
        exact_ids = [row.memory_id for score, row in exact_ranked if score > 0][
            : self.settings.long_term_top_k
        ]

        es_ids: list[str] = []
        qdrant_ids: list[str] = []
        es_evidence: dict = {"status": "skipped_no_memory"}
        qdrant_evidence: dict = {"status": "skipped_no_memory"}
        if self.settings.long_term_enabled and candidates:
            async def es_route():
                return await asyncio.wait_for(
                    self.es_repository.search(
                        query,
                        owner_id=self.owner_id,
                        workspace_id=self.workspace_id,
                        limit=self.settings.long_term_top_k * 3,
                    ),
                    timeout=float(self.settings.retrieval_timeout_seconds),
                )

            async def qdrant_route():
                embedding = await self.embedding_client.aembed_query(query)
                return await self.qdrant_repository.search(
                    embedding,
                    query=query,
                    owner_id=self.owner_id,
                    workspace_id=self.workspace_id,
                    limit=self.settings.long_term_top_k * 3,
                )

            es_result, qdrant_result = await asyncio.gather(
                es_route(), qdrant_route(), return_exceptions=True
            )
            if isinstance(es_result, BaseException):
                es_evidence = {
                    "status": "degraded", "error_type": type(es_result).__name__
                }
                degradations.append(
                    _safe_degradation(
                        "memory_read", "memory_es_retrieval_failed", es_result
                    )
                )
            else:
                es_ids, es_evidence = es_result
            if isinstance(qdrant_result, BaseException):
                qdrant_evidence = {
                    "status": "degraded",
                    "error_type": type(qdrant_result).__name__,
                }
                degradations.append(
                    _safe_degradation(
                        "memory_read",
                        "memory_qdrant_retrieval_failed",
                        qdrant_result,
                    )
                )
            else:
                qdrant_ids, qdrant_evidence, qdrant_degradations = qdrant_result
                degradations.extend(qdrant_degradations)

        fused_ids, provenance = rrf_fuse(
            {"mysql_exact": exact_ids, "es_bm25": es_ids, "qdrant_hybrid": qdrant_ids},
            rrf_k=self.settings.fusion_rrf_k,
            limit=self.settings.long_term_top_k,
        )
        async with self.persistence.transaction() as uow:
            repo = uow.memory
            authoritative = await repo.list_authoritative_by_ids(fused_ids)
        selected_by_id = {row.memory_id: row for row in authoritative}
        source_by_id = {
            row["memory_id"]: "+".join(row.get("sources") or [])
            for row in provenance
        }

        def reference(row) -> dict:
            return {
                "memory_id": row.memory_id,
                "memory_type": row.memory_type,
                "content": row.content_masked,
                "confidence": float(row.confidence or 0),
                "source": source_by_id.get(row.memory_id, "mysql_authority"),
                "created_at": str(row.created_at),
                "structured_payload": dict(row.structured_payload or {}),
            }

        relevant = [
            reference(selected_by_id[memory_id])
            for memory_id in fused_ids
            if memory_id in selected_by_id
            and selected_by_id[memory_id].memory_type in {
                "correction", "query_episode"
            }
        ]
        preference_refs = [reference(row) for row in preferences]
        profile_refs = [reference(row) for row in profile_facts]
        goal_refs = [reference(row) for row in long_term_goals]
        evidence = {
            "version": self.settings.context_schema_version,
            "route": "conversation_memory",
            "responsibility": "conversation_memory",
            "working_source": working_source,
            "working_version": int(working.get("version") or mysql_version),
            "redis_hit": redis_hit,
            "redis_stale_reconciled": redis_stale,
            "recent_loaded": len(messages),
            "recent_selected": len(recent_messages),
            "mysql_exact_candidates": len(exact_ids),
            "es_candidates": len(es_ids),
            "qdrant_candidates": len(qdrant_ids),
            "selected_count": (
                len(relevant) + len(preference_refs) + len(profile_refs)
                + len(goal_refs) + len(selected_summaries)
            ),
            "fusion_method": "memory_id_rrf",
            "rrf_k": int(self.settings.fusion_rrf_k),
            "workspace_scope_applied": True,
            "owner_scope_applied": True,
            "authoritative_recheck": True,
            "raw_result_rows_injected": False,
            "historical_sql_reusable": False,
            "es": es_evidence,
            "qdrant": qdrant_evidence,
            "summary": summary_evidence,
            "degradations": degradations,
        }
        return {
            "schema_version": self.settings.context_schema_version,
            "working_state": {
                "structured_state": dict(working.get("structured_state") or {}),
                "rolling_summary": str(working.get("rolling_summary") or ""),
                "last_success_question": str(
                    working.get("last_success_question") or ""
                ),
            },
            "recent_messages": recent_messages,
            "selected_summaries": selected_summaries,
            "confirmed_preferences": preference_refs,
            "confirmed_profile_facts": profile_refs,
            "long_term_goals": goal_refs,
            "relevant_episodes": relevant,
            "provenance": provenance + summary_provenance,
            "omitted_counts": {
                "recent_messages": omitted_recent,
                "summaries": max(
                    0,
                    int(summary_evidence.get("active_segment_count") or 0)
                    - len(selected_summaries),
                ),
                "long_term_candidates": max(
                    0, len(set(exact_ids + es_ids + qdrant_ids)) - len(relevant)
                ),
            },
            "evidence": evidence,
        }

    async def complete_turn(
        self,
        *,
        chat_id: str,
        query: str,
        sql: str,
        row_count: int,
        result: list,
        standalone_question: str,
        follow_up_action: str,
        semantic_state: dict,
        trace_id: str,
        user_message_id: int,
        memory_context: Mapping,
        schema_version: str | None,
    ) -> dict:
        """Atomically persist success episode, checkpoint, memory and outbox."""
        async with self.persistence.transaction() as uow:
            repo = uow.memory
            history_repo = uow.history
            conversation = await history_repo.get_session(
                chat_id,
                include_archived=True,
                include_deleted=True,
                for_update=True,
            )
            if conversation is None or conversation.status != "active":
                raise ConversationUnavailableError(
                    "会话已在本轮执行期间重置、归档或删除"
                )
            message_claimed = await repo.set_message_status(
                user_message_id,
                "success",
                expected_statuses=("pending",),
            )
            if not message_claimed:
                raise ConversationUnavailableError(
                    "会话已在本轮执行期间重置或删除"
                )
            assistant = await repo.append_message(
                chat_id=chat_id,
                role="assistant",
                content_masked=f"查询成功，返回 {max(0, int(row_count))} 行。",
                request_id=f"{trace_id}:assistant",
                trace_id=trace_id,
                status="success",
            )
            history = await uow.chat_history.save_turn(
                chat_id,
                query,
                sql,
                result,
                standalone_question=standalone_question,
                follow_up_action=follow_up_action,
                semantic_state=semantic_state,
                commit=False,
            )
            await history_repo.complete_turn(
                chat_id=chat_id,
                turn_id=trace_id,
                source_history_id=int(history.id),
                row_count=row_count,
            )
            summary_session_version = int(conversation.version or 0)
            recent = await repo.recent_messages(
                chat_id, self.settings.recent_message_limit
            )
            recent_ids = [int(row.id) for row in recent]
            authoritative_checkpoint = await repo.get_checkpoint(
                chat_id, for_update=True
            )
            previous_version = int(
                getattr(authoritative_checkpoint, "version", 0) or 0
            )
            previous_summary = str(
                getattr(authoritative_checkpoint, "rolling_summary", "") or ""
            )
            summary_lines = [
                line for line in previous_summary.splitlines() if line.strip()
            ][-3:]
            summary_lines.append(f"成功问题：{standalone_question}")
            rolling_summary = mask_free_text(
                "\n".join(summary_lines),
                policy=self.settings.masking_policy,
                max_chars=self.settings.memory_content_max_chars,
            )
            checkpoint = await repo.upsert_checkpoint(
                chat_id=chat_id,
                version=previous_version + 1,
                structured_state={
                    **dict(semantic_state or {}),
                    "standalone_question": standalone_question,
                },
                rolling_summary=rolling_summary,
                recent_message_ids=recent_ids,
                schema_version=self.settings.context_schema_version,
                ttl_seconds=self.settings.working_ttl_seconds,
            )
            candidates = extract_memory_candidates(
                owner_id=self.owner_id,
                workspace_id=self.workspace_id,
                chat_id=chat_id,
                query=query,
                standalone_question=standalone_question,
                follow_up_action=follow_up_action,
                semantic_state=semantic_state,
                sql=sql,
                user_message_id=user_message_id,
                assistant_message_id=int(assistant.id),
                schema_version=schema_version,
                settings=self.settings,
            )
            created_ids: list[str] = []
            superseded_ids: list[str] = []
            outbox_events_created = 0
            for values in candidates:
                item, created = await repo.upsert_memory_item(values)
                if not created:
                    continue
                created_ids.append(item.memory_id)
                if item.memory_type == "preference":
                    replaced = await repo.supersede_active_preferences(
                        preference_slot=str(
                            (item.structured_payload or {}).get(
                                "preference_slot"
                            ) or ""
                        ),
                        superseded_by_id=item.memory_id,
                    )
                    superseded_ids.extend(replaced)
                    for old_memory_id in replaced:
                        for target, projection_version in (
                            ("es", self.settings.es_projection_version),
                            (
                                "qdrant",
                                self.settings.qdrant_projection_version,
                            ),
                        ):
                            await repo.enqueue_projection_deletes(
                                memory_id=old_memory_id,
                                target=target,
                                current_projection_version=projection_version,
                            )
                            outbox_events_created += 1
                await repo.enqueue_projection(
                    memory_id=item.memory_id,
                    target="es",
                    operation="upsert",
                    projection_version=self.settings.es_projection_version,
                )
                outbox_events_created += 1
                await repo.enqueue_projection(
                    memory_id=item.memory_id,
                    target="qdrant",
                    operation="upsert",
                    projection_version=self.settings.qdrant_projection_version,
                )
                outbox_events_created += 1
            await uow.commit()

        # Summary is a derived layer.  Its failure must never roll back or
        # relabel an already successful SQL turn.  Segment + signals + their
        # projection outbox remain atomic inside this independent transaction.
        summary_write: dict = {"status": "disabled", "segments_created": 0}
        try:
            async with self.persistence.transaction() as summary_uow:
                summary_write = await ConversationSummaryService(
                    owner_id=self.owner_id,
                    workspace_id=self.workspace_id,
                    settings=self.settings,
                ).build_if_needed(
                    summary_uow,
                    chat_id=chat_id,
                    expected_session_version=summary_session_version,
                )
                await summary_uow.commit()
        except Exception as exc:
            summary_write = {
                "status": "degraded",
                "segments_created": 0,
                "error_type": type(exc).__name__,
                "affects_query_success": False,
            }
            logger.warning(
                "conversation summary 构建降级 workspace=%s chat=%s error_type=%s",
                self.workspace_id,
                chat_id,
                type(exc).__name__,
            )

        redis_status = "unavailable"
        redis_repo = self._redis_repo()
        if redis_repo is not None:
            payload = build_working_payload(
                semantic_state={
                    **dict(semantic_state or {}),
                    "standalone_question": standalone_question,
                },
                rolling_summary=rolling_summary,
                recent_message_ids=recent_ids,
                last_success_question=standalone_question,
                settings=self.settings,
            )
            try:
                cas = await redis_repo.compare_and_set(
                    chat_id,
                    expected_version=previous_version,
                    payload=payload,
                    schema_version=self.settings.context_schema_version,
                    updated_at=iso_now(),
                )
                if not cas.updated:
                    latest_payload, _, _, latest_version = await self._working_from_mysql(
                        chat_id
                    )
                    await redis_repo.replace(
                        chat_id,
                        version=latest_version,
                        payload=latest_payload,
                        schema_version=self.settings.context_schema_version,
                        updated_at=iso_now(),
                    )
                    redis_status = "reconciled_from_mysql"
                else:
                    redis_status = "updated"
            except Exception as exc:
                redis_status = f"degraded:{type(exc).__name__}"
                logger.warning(
                    "Redis working memory 更新降级 workspace=%s chat=%s error_type=%s",
                    self.workspace_id,
                    chat_id,
                    type(exc).__name__,
                )
        self.notifier.notify()
        return {
            "mysql_committed": True,
            "checkpoint_version": int(checkpoint.version),
            "memory_items_created": len(created_ids),
            "memory_items_superseded": len(superseded_ids),
            "outbox_events_created": outbox_events_created,
            "redis_status": redis_status,
            "summary": summary_write,
        }

    async def complete_handled_turn(
        self,
        user_message_id: int | None,
        *,
        chat_id: str,
        trace_id: str,
        response: Mapping[str, object],
    ) -> dict:
        """完成无 SQL 的策略响应，不生成查询记忆或检索投影。"""

        if user_message_id is None:
            raise ConversationUnavailableError("已处理回合缺少用户消息")
        assistant_summary = str(response.get("message") or "").strip()
        if not assistant_summary:
            raise ValueError("已处理回合必须包含用户可见消息")
        safe_summary = mask_free_text(
            assistant_summary,
            policy=self.settings.masking_policy,
            max_chars=self.settings.message_content_max_chars,
        )
        async with self.persistence.transaction() as uow:
            conversation = await uow.history.get_session(
                chat_id,
                include_archived=True,
                include_deleted=True,
                for_update=True,
            )
            if conversation is None or conversation.status != "active":
                raise ConversationUnavailableError(
                    "会话已在本轮执行期间重置、归档或删除"
                )
            message_claimed = await uow.memory.set_message_status(
                user_message_id,
                "success",
                expected_statuses=("pending",),
            )
            if not message_claimed:
                raise ConversationUnavailableError(
                    "会话已在本轮执行期间重置或删除"
                )
            await uow.memory.append_message(
                chat_id=chat_id,
                role="assistant",
                content_masked=safe_summary,
                request_id=f"{trace_id}:assistant",
                trace_id=trace_id,
                status="success",
            )
            await uow.history.complete_handled_turn(
                chat_id=chat_id,
                turn_id=trace_id,
                assistant_summary=safe_summary,
            )
            await uow.commit()
        return {
            "mysql_committed": True,
            "replay_fidelity": "message_only",
            "memory_items_created": 0,
            "outbox_events_created": 0,
        }

    async def fail_turn(
        self,
        user_message_id: int | None,
        *,
        chat_id: str | None = None,
        trace_id: str | None = None,
        status: str = "failed",
        summary: str | None = None,
    ) -> None:
        if not user_message_id and not (chat_id and trace_id):
            return
        async with self.persistence.transaction() as uow:
            repo = uow.memory
            if user_message_id:
                await repo.set_message_status(
                    user_message_id,
                    status,
                    expected_statuses=("pending",),
                )
            if chat_id and trace_id:
                await uow.history.fail_turn(
                    chat_id=chat_id,
                    turn_id=trace_id,
                    status=status,
                    assistant_summary=summary,
                )
            await uow.commit()

    async def status(self, chat_id: str) -> dict:
        redis = {"available": False, "exists": False, "ttl_seconds": -2}
        repo = self._redis_repo()
        if repo is not None:
            try:
                redis = {"available": True, **await repo.status(chat_id)}
            except Exception as exc:
                redis = {
                    "available": False,
                    "exists": False,
                    "ttl_seconds": -2,
                    "error_type": type(exc).__name__,
                }
        async with self.persistence.transaction() as uow:
            memory_repo = uow.memory
            summary_repo = uow.summary
            checkpoint = await memory_repo.get_checkpoint(chat_id)
            memories = await memory_repo.list_chat_memories(chat_id, limit=100)
            scope_memories = await memory_repo.list_active_candidates(
                memory_types=(
                    "preference", "correction", "query_episode",
                    "profile_fact", "long_term_goal",
                ),
                limit=100,
            )
            outbox = await memory_repo.outbox_status()
            summary_status = await summary_repo.status(chat_id)

        async def projection_status(name: str, call) -> dict:
            try:
                return await asyncio.wait_for(
                    call(),
                    timeout=float(self.settings.retrieval_timeout_seconds),
                )
            except Exception as exc:
                return {
                    "engine": name,
                    "status": "degraded",
                    "error_type": type(exc).__name__,
                }

        es_projection, qdrant_projection, summary_es, summary_qdrant = await asyncio.gather(
            projection_status("elasticsearch", self.es_repository.status),
            projection_status("qdrant", self.qdrant_repository.status),
            projection_status(
                "summary_elasticsearch",
                self.summary_es_repository.status
                if self.summary_es_repository is not None
                else lambda: asyncio.sleep(
                    0, result={"status": "unavailable"}
                ),
            ),
            projection_status(
                "summary_qdrant",
                self.summary_qdrant_repository.status
                if self.summary_qdrant_repository is not None
                else lambda: asyncio.sleep(
                    0, result={"status": "unavailable"}
                ),
            ),
        )
        counts = Counter(row.memory_type for row in scope_memories)
        return {
            "chat_id": chat_id,
            "schema_version": self.settings.context_schema_version,
            "working_memory": redis,
            "checkpoint": {
                "exists": checkpoint is not None,
                "version": int(getattr(checkpoint, "version", 0) or 0),
                "updated_at": str(getattr(checkpoint, "updated_at", "") or ""),
            },
            "long_term": {
                "active_count": sum(counts.values()),
                "current_chat_count": sum(
                    row.status == "active" for row in memories
                ),
                "by_type": dict(counts),
                "items": [
                    {
                        "memory_id": row.memory_id,
                        "memory_type": row.memory_type,
                        "content": row.content_masked,
                        "source_chat_id": row.chat_id,
                        "status": row.status,
                        "verification_status": row.verification_status,
                        "created_at": str(row.created_at),
                    }
                    for row in scope_memories[:20]
                ],
            },
            "summary_memory": summary_status,
            "projection": {
                "outbox": outbox,
                "worker": self.notifier.status(),
                "elasticsearch": es_projection,
                "qdrant": qdrant_projection,
                "summary_elasticsearch": summary_es,
                "summary_qdrant": summary_qdrant,
            },
        }

    async def reset(
        self,
        chat_id: str,
        *,
        include_long_term: bool = False,
        strict_redis: bool = False,
    ) -> dict:
        redis_deleted = False
        redis_error: Exception | None = None
        redis_repo = self._redis_repo()
        async with self.persistence.transaction() as uow:
            repo = uow.memory
            summary_repo = uow.summary
            # Lock the session row before touching any derived state.  A turn
            # that started earlier either commits before this lock (and is
            # cleaned below) or observes its source message changed from
            # pending after this transaction and cannot resurrect memory.
            await uow.history.reset_context(chat_id)
            checkpoint_deleted = await repo.delete_checkpoint(chat_id)
            working_messages = await repo.reset_working_transcript(chat_id)
            revoked: list[str] = []
            revoked_summaries = await summary_repo.revoke_chat(chat_id)
            for summary_id in revoked_summaries:
                await repo.enqueue_projection_deletes(
                    memory_id=summary_id,
                    target="summary_es",
                    current_projection_version=(
                        self.settings.summary_es_projection_version
                    ),
                )
                await repo.enqueue_projection_deletes(
                    memory_id=summary_id,
                    target="summary_qdrant",
                    current_projection_version=(
                        self.settings.summary_qdrant_projection_version
                    ),
                )
            if include_long_term:
                revoked = await repo.revoke_chat_memories(chat_id)
                for memory_id in revoked:
                    await repo.enqueue_projection_deletes(
                        memory_id=memory_id,
                        target="es",
                        current_projection_version=(
                            self.settings.es_projection_version
                        ),
                    )
                    await repo.enqueue_projection_deletes(
                        memory_id=memory_id,
                        target="qdrant",
                        current_projection_version=(
                            self.settings.qdrant_projection_version
                        ),
                    )
            # Keep the session fence until Redis has been invalidated.  New
            # turns may begin only after this transaction commits, so their
            # fresh generation cannot be deleted by this reset.
            if redis_repo is not None:
                try:
                    redis_deleted = await redis_repo.delete(chat_id)
                except Exception as exc:
                    redis_error = exc
                    if strict_redis:
                        await uow.rollback()
                        raise RuntimeError(
                            "Redis working memory 删除未完成"
                        ) from exc
            await uow.commit()
        if revoked or revoked_summaries:
            self.notifier.notify()
        return {
            "chat_id": chat_id,
            "redis_deleted": redis_deleted,
            "checkpoint_deleted": checkpoint_deleted,
            "working_messages_reset": working_messages,
            "long_term_revoked": len(revoked),
            "summaries_revoked": len(revoked_summaries),
            "redis_error_type": (
                type(redis_error).__name__ if redis_error is not None else None
            ),
        }
