"""会话记忆的确定性摘要构建与层级编排。"""
from __future__ import annotations

from collections.abc import Sequence

from app.application.ports.conversation import (
    ConversationMemoryRepositoryPort,
    ConversationMemoryUnitOfWork,
    ConversationSummaryRepositoryPort,
)
from app.application.services.conversation_memory_policy import (
    build_extractive_summary_segment,
    build_hierarchical_summary_segment,
    estimate_tokens,
)
from app.application.settings.conversation import (
    DEFAULT_MEMORY_SETTINGS,
    ConversationMemorySettings,
)


class ConversationSummaryService:
    """在成功回合后构建仅追加、可追溯到来源的摘要。"""

    def __init__(
        self,
        *,
        owner_id: str,
        workspace_id: str,
        settings: ConversationMemorySettings = DEFAULT_MEMORY_SETTINGS,
    ):
        self.owner_id = str(owner_id or "").strip()
        self.workspace_id = str(workspace_id or "").strip()
        if not self.owner_id or not self.workspace_id:
            raise ValueError("ConversationSummaryService 必须绑定 owner/workspace")
        self.settings = settings

    @staticmethod
    def _source_token_count(messages: Sequence[object]) -> int:
        return estimate_tokens([
            str(getattr(row, "content_masked", "") or "") for row in messages
        ])

    def _complete_prefix(self, messages: Sequence[object]) -> list[object]:
        maximum_messages = max(2, int(self.settings.summary_segment_max_messages))
        maximum_tokens = max(100, int(self.settings.summary_segment_max_tokens))
        selected: list[object] = []
        used = 0
        for row in list(messages)[:maximum_messages]:
            if str(getattr(row, "status", "success")) != "success":
                break
            cost = estimate_tokens(str(getattr(row, "content_masked", "") or ""))
            if selected and used + cost > maximum_tokens:
                break
            selected.append(row)
            used += cost
        last_assistant = -1
        for index, row in enumerate(selected):
            if str(getattr(row, "role", "")) == "assistant":
                last_assistant = index
        return selected[:last_assistant + 1] if last_assistant >= 1 else []

    async def build_if_needed(
        self,
        uow: ConversationMemoryUnitOfWork,
        *,
        chat_id: str,
        expected_session_version: int,
        max_l0_segments: int = 3,
    ) -> dict:
        if not (
            self.settings.enabled
            and self.settings.summary_enabled
            and self.settings.summary_build_enabled
        ):
            return {"status": "disabled", "segments_created": 0}

        # 所有能创建或撤销摘要状态的写入者都按“会话、摘要状态”的固定顺序加锁。
        # 预期版本取自成功回合仍持有会话行时；重置、删除或后续回合会推进版本，
        # 因而旧派生任务无法越过该边界提交仍为 active 的摘要片段。
        conversation = await uow.history.get_session(
            chat_id,
            include_archived=True,
            include_deleted=True,
            for_update=True,
        )
        if (
            conversation is None
            or conversation.status != "active"
            or int(conversation.version or 0) != int(expected_session_version)
        ):
            return {
                "status": "stale_generation",
                "trigger_reason": "session_generation_changed",
                "segments_created": 0,
                "segment_ids": [],
                "outbox_events_created": 0,
                "llm_calls": 0,
            }

        summary_repo = uow.summary
        outbox_repo = uow.memory
        state = await summary_repo.ensure_state(
            chat_id,
            schema_version=self.settings.summary_schema_version,
            for_update=True,
        )
        created_ids: list[str] = []
        outbox_events = 0
        trigger_reason = "below_watermark"

        for _ in range(max(1, int(max_l0_segments))):
            messages = await summary_repo.list_uncovered_summary_messages(
                chat_id,
                after_message_id=state.covered_through_message_id,
            )
            # 遇到首个在途回合后停止统计与摘要，避免后发请求先完成并把仅追加水位
            # 永久推进到先发请求之后。
            committed_prefix: list[object] = []
            for row in messages:
                if str(getattr(row, "status", "success")) != "success":
                    break
                committed_prefix.append(row)
            messages = committed_prefix
            user_turns = sum(
                str(getattr(row, "role", "")) == "user" for row in messages
            )
            token_count = self._source_token_count(messages)
            turn_trigger = user_turns >= int(
                self.settings.summary_turn_high_watermark
            )
            token_trigger = token_count >= int(
                self.settings.summary_token_high_watermark
            )
            if not (turn_trigger or token_trigger):
                break
            trigger_reason = (
                "turn_and_token_high_watermark"
                if turn_trigger and token_trigger
                else "turn_high_watermark"
                if turn_trigger
                else "token_high_watermark"
            )
            protected_turns = max(
                int(self.settings.summary_recent_protected_turns),
                int(self.settings.summary_turn_low_watermark),
            )
            protected_messages = protected_turns * 2
            eligible = (
                messages[:-protected_messages]
                if len(messages) > protected_messages
                else []
            )
            selected = self._complete_prefix(eligible)
            if not selected:
                trigger_reason = "incomplete_turn_prefix"
                break
            semantic_facts = await summary_repo.semantic_facts_for_messages(
                [int(getattr(row, "id")) for row in selected]
            )
            values, signals = build_extractive_summary_segment(
                owner_id=self.owner_id,
                workspace_id=self.workspace_id,
                chat_id=chat_id,
                messages=selected,
                structured_facts=semantic_facts,
                settings=self.settings,
            )
            segment, created = await summary_repo.insert_segment_with_signals(
                values, signals
            )
            if created:
                created_ids.append(segment.summary_id)
                if self.settings.summary_projection_enabled:
                    for target, version in (
                        ("summary_es", self.settings.summary_es_projection_version),
                        (
                            "summary_qdrant",
                            self.settings.summary_qdrant_projection_version,
                        ),
                    ):
                        await outbox_repo.enqueue_projection(
                            memory_id=segment.summary_id,
                            target=target,
                            operation="upsert",
                            projection_version=version,
                        )
                        outbox_events += 1
            await summary_repo.advance_state(
                state,
                covered_through_message_id=int(segment.source_end_message_id),
                source_hash=segment.source_content_hash,
                schema_version=self.settings.summary_schema_version,
            )

        hierarchy_created = await self._build_hierarchy(
            summary_repo=summary_repo,
            outbox_repo=outbox_repo,
            chat_id=chat_id,
        )
        created_ids.extend(hierarchy_created)
        if self.settings.summary_projection_enabled:
            outbox_events += len(hierarchy_created) * 2
        remaining = await summary_repo.list_uncovered_summary_messages(
            chat_id, after_message_id=state.covered_through_message_id
        )
        return {
            "status": "created" if created_ids else trigger_reason,
            "trigger_reason": trigger_reason,
            "segments_created": len(created_ids),
            "segment_ids": created_ids,
            "outbox_events_created": outbox_events,
            "covered_through_message_id": state.covered_through_message_id,
            "watermark_version": int(state.version or 0),
            "coverage_lag_messages": len(remaining),
            "coverage_lag_turns": sum(
                str(getattr(row, "role", "")) == "user" for row in remaining
            ),
            "coverage_lag_tokens": self._source_token_count(remaining),
            "builder_mode": "deterministic_extractive_post_result",
            "llm_calls": 0,
        }

    async def _build_hierarchy(
        self,
        *,
        summary_repo: ConversationSummaryRepositoryPort,
        outbox_repo: ConversationMemoryRepositoryPort,
        chat_id: str,
    ) -> list[str]:
        group_size = max(2, int(self.settings.summary_hierarchy_group_size))
        max_level = max(0, int(self.settings.summary_max_level))
        created_ids: list[str] = []
        for level in range(max_level):
            all_segments = await summary_repo.list_active_segments(chat_id, limit=500)
            referenced = {
                str(child_id)
                for row in all_segments
                if int(row.level or 0) == level + 1
                for child_id in list(row.child_summary_ids or [])
            }
            children = sorted(
                (
                    row for row in all_segments
                    if int(row.level or 0) == level and row.summary_id not in referenced
                ),
                key=lambda row: (int(row.source_start_message_id), row.summary_id),
            )
            if len(children) < group_size:
                continue
            group = children[:group_size]
            values, signals = build_hierarchical_summary_segment(
                owner_id=self.owner_id,
                workspace_id=self.workspace_id,
                chat_id=chat_id,
                children=group,
                settings=self.settings,
            )
            segment, created = await summary_repo.insert_segment_with_signals(
                values, signals
            )
            if not created:
                continue
            created_ids.append(segment.summary_id)
            if self.settings.summary_projection_enabled:
                for target, version in (
                    ("summary_es", self.settings.summary_es_projection_version),
                    (
                        "summary_qdrant",
                        self.settings.summary_qdrant_projection_version,
                    ),
                ):
                    await outbox_repo.enqueue_projection(
                        memory_id=segment.summary_id,
                        target=target,
                        operation="upsert",
                        projection_version=version,
                    )
        return created_ids
