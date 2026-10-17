import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import Select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.config.app_config import app_config
from app.agent_system.agents.nl2sql.services.memory_prompt_context import (
    memory_context_for_prompt,
)
from app.application.services.conversation_memory_policy import (
    build_extractive_summary_segment,
    build_hierarchical_summary_segment,
    extract_explicit_stable_facts,
    extract_summary_signals,
    summary_retrieval_needed,
)
from app.agent_system.governance.retrieval_budget import estimate_tokens
from app.infrastructure.persistence.mysql.records.base import Base
from app.infrastructure.config.conversation import load_conversation_settings
from app.infrastructure.persistence.mysql.conversation_adapters import (
    SqlConversationMemoryUnitOfWork,
)
from app.infrastructure.persistence.mysql.records.conversation_memory_outbox_mysql import (
    ConversationMemoryOutboxMySQL,
)
from app.infrastructure.persistence.mysql.records.conversation_session_mysql import ConversationSessionMySQL
from app.infrastructure.persistence.mysql.records.conversation_summary_signal_mysql import (
    ConversationSummarySignalMySQL,
)
from app.infrastructure.persistence.mysql.repositories.conversation_memory_repository import (
    ConversationMemoryRepository,
)
from app.infrastructure.persistence.mysql.repositories.conversation_summary_repository import (
    ConversationSummaryRepository,
)
from app.application.services.conversation_summary_service import ConversationSummaryService


class ConversationSummaryPureTests(unittest.TestCase):
    def test_extractive_segment_is_source_grounded_and_one_segment_has_one_identity(self):
        messages = [
            SimpleNamespace(
                id=1, role="user", status="success", content_masked="2024年华东销售额"
            ),
            SimpleNamespace(
                id=2,
                role="assistant",
                status="success",
                content_masked="查询成功，返回 3 行。",
            ),
        ]
        segment, signals = build_extractive_summary_segment(
            owner_id="owner-a",
            workspace_id="workspace-a",
            chat_id="chat-a",
            messages=messages,
        )
        duplicate, _ = build_extractive_summary_segment(
            owner_id="owner-a",
            workspace_id="workspace-a",
            chat_id="chat-a",
            messages=messages,
        )

        self.assertEqual(segment["summary_id"], duplicate["summary_id"])
        self.assertEqual(segment["source_message_ids"], [1, 2])
        self.assertIn("2024年华东销售额", segment["content_masked"])
        self.assertNotIn("查询成功，返回", segment["content_masked"])
        self.assertEqual(segment["model_version"], None)
        self.assertGreater(len(signals), 0)
        self.assertTrue(all(row["summary_id"] == segment["summary_id"] for row in signals))

    def test_summary_signals_are_independent_typed_records(self):
        signals = extract_summary_signals(
            "2024年华东销售额与客户复购率", source_message_ids=[10, 11]
        )
        self.assertTrue(signals)
        self.assertTrue({"time", "metric"} & {row["signal_type"] for row in signals})
        self.assertTrue(all(row["source_message_ids"] == [10, 11] for row in signals))
        self.assertTrue(all("query_signal" not in row for row in signals))

    def test_summary_redacts_sensitive_identity_and_uses_actual_signal_sources(self):
        messages = [
            SimpleNamespace(
                id=10, role="user", status="success", content_masked="我是基督徒"
            ),
            SimpleNamespace(
                id=11, role="user", status="success", content_masked="2024年华东销售额"
            ),
            SimpleNamespace(
                id=12, role="user", status="success", content_masked="华南客户数量"
            ),
            SimpleNamespace(
                id=13,
                role="assistant",
                status="success",
                content_masked="已记录你是基督徒",
            ),
        ]

        segment, signals = build_extractive_summary_segment(
            owner_id="owner-a",
            workspace_id="workspace-a",
            chat_id="chat-a",
            messages=messages,
        )

        persisted = repr((
            segment["content_masked"],
            segment["structured_facts"],
            signals,
        ))
        self.assertNotIn("基督徒", persisted)
        self.assertIn("敏感身份内容已省略", segment["content_masked"])
        self.assertEqual(segment["source_message_ids"], [10, 11, 12, 13])
        self.assertFalse(extract_summary_signals("我是基督徒", source_message_ids=[10]))
        sales_sources = {
            tuple(row["source_message_ids"])
            for row in signals
            if "销售" in row["normalized_value"]
        }
        customer_sources = {
            tuple(row["source_message_ids"])
            for row in signals
            if "客户" in row["normalized_value"]
        }
        self.assertEqual(sales_sources, {(11,)})
        self.assertEqual(customer_sources, {(12,)})

    def test_sensitive_identity_categories_are_deny_by_default(self):
        sensitive_cases = (
            ("少数民族", "我是土家族数据工程师", "土家族"),
            ("未枚举少数民族", "我是布依族产品经理", "布依族"),
            ("宗教", "我是摩门教信徒数据工程师", "摩门教信徒"),
            ("政治党派", "我是某联盟员数据工程师", "某联盟员"),
            ("性取向", "我是酷儿数据工程师", "酷儿"),
            ("残疾", "我是脊髓损伤数据工程师", "脊髓损伤"),
            ("病史", "我有渐冻症", "渐冻症"),
        )
        messages = []
        for index, (category, query, _) in enumerate(sensitive_cases, start=30):
            with self.subTest(category=category, query=query):
                facts = extract_explicit_stable_facts(
                    owner_id="owner-a",
                    workspace_id="workspace-a",
                    chat_id="chat-a",
                    query=query,
                    user_message_id=index,
                )
                self.assertEqual(facts, [])
                self.assertEqual(
                    extract_summary_signals(query, source_message_ids=[index]),
                    [],
                )
            messages.append(SimpleNamespace(
                id=index,
                role="user",
                status="success",
                content_masked=query,
            ))
        messages.append(SimpleNamespace(
            id=100,
            role="user",
            status="success",
            content_masked="我是数据工程师",
        ))

        segment, signals = build_extractive_summary_segment(
            owner_id="owner-a",
            workspace_id="workspace-a",
            chat_id="chat-a",
            messages=messages,
        )

        persisted = repr((
            segment["content_masked"],
            segment["structured_facts"],
            signals,
        ))
        for _, _, sensitive_body in sensitive_cases:
            self.assertNotIn(sensitive_body, persisted)
        self.assertIn("敏感身份内容已省略", segment["content_masked"])
        self.assertIn("我是数据工程师", segment["content_masked"])
        self.assertTrue(any(
            row["source_message_ids"] == [100]
            for row in signals
        ))

    def test_hierarchical_summary_defensively_redacts_legacy_child_content(self):
        children = [
            {
                "summary_id": "child-1",
                "level": 0,
                "source_message_ids": [21],
                "source_content_hash": "sha256:one",
                "content_masked": "用户曾陈述：我是土家族",
                "structured_facts": [{
                    "fact_type": "user_statement",
                    "content": "我是土家族",
                    "source_message_ids": [21],
                    "status": "supported",
                }],
            },
            {
                "summary_id": "child-2",
                "level": 0,
                "source_message_ids": [22],
                "source_content_hash": "sha256:two",
                "content_masked": "用户曾询问：2025年销售额",
                "structured_facts": [{
                    "fact_type": "user_statement",
                    "content": "2025年销售额",
                    "source_message_ids": [22],
                    "status": "supported",
                }],
            },
        ]

        segment, signals = build_hierarchical_summary_segment(
            owner_id="owner-a",
            workspace_id="workspace-a",
            chat_id="chat-a",
            children=children,
        )

        self.assertNotIn("土家族", repr((segment, signals)))
        self.assertIn("敏感身份内容已省略", segment["content_masked"])
        self.assertTrue(all(row["source_message_ids"] == [22] for row in signals))

    def test_explicit_stable_facts_do_not_depend_on_successful_sql(self):
        profile = extract_explicit_stable_facts(
            owner_id="owner-a",
            workspace_id="workspace-a",
            chat_id="chat-a",
            query="我是销售分析师",
            user_message_id=7,
        )
        goal = extract_explicit_stable_facts(
            owner_id="owner-a",
            workspace_id="workspace-a",
            chat_id="chat-a",
            query="我的长期目标是提高华东复购率",
            user_message_id=8,
        )
        sensitive = extract_explicit_stable_facts(
            owner_id="owner-a",
            workspace_id="workspace-a",
            chat_id="chat-a",
            query="我的病史是高血压",
            user_message_id=9,
        )

        self.assertEqual(profile[0]["memory_type"], "profile_fact")
        self.assertEqual(goal[0]["memory_type"], "long_term_goal")
        self.assertEqual(profile[0]["verification_status"], "confirmed")
        self.assertEqual(sensitive, [])

    def test_profile_fact_is_a_work_context_allowlist(self):
        allowed = (
            "我是数据工程师",
            "我的岗位是产品经理",
            "我负责华东区销售分析",
            "我的工作职责包括整理办公室",
        )
        rejected = (
            "我是基督徒",
            "我是犹太人",
            "我是汉族",
            "我是党员",
            "我是同性恋",
            "我是残疾人",
            "我有自闭症",
            "我患有高血压",
            "我是一个爱旅行的人",
            "我是经理的妻子",
            "我负责照顾孩子",
            "我是土家族数据工程师",
            "我负责销售运营且是土家族",
        )
        for index, query in enumerate(allowed, start=30):
            with self.subTest(query=query):
                facts = extract_explicit_stable_facts(
                    owner_id="owner-a",
                    workspace_id="workspace-a",
                    chat_id="chat-a",
                    query=query,
                    user_message_id=index,
                )
                self.assertEqual([row["memory_type"] for row in facts], ["profile_fact"])
        for index, query in enumerate(rejected, start=40):
            with self.subTest(query=query):
                facts = extract_explicit_stable_facts(
                    owner_id="owner-a",
                    workspace_id="workspace-a",
                    chat_id="chat-a",
                    query=query,
                    user_message_id=index,
                )
                self.assertEqual(facts, [])

    def test_summary_retrieval_is_gated(self):
        self.assertEqual(
            summary_retrieval_needed("继续之前那个指标", has_active_summary=True),
            (True, "long_range_reference"),
        )
        self.assertEqual(
            summary_retrieval_needed("各地区销售额", has_active_summary=True),
            (False, "not_needed"),
        )
        self.assertEqual(
            summary_retrieval_needed("继续", has_active_summary=False),
            (False, "no_summary"),
        )

    def test_memory_context_v2_injects_recent_and_summary_under_budget(self):
        context = memory_context_for_prompt({
            "schema_version": "memory-context-v2",
            "working_state": {"last_success_question": "销售额"},
            "recent_messages": [
                {"message_id": index, "role": "user", "content": "近期问题" + "x" * 80}
                for index in range(10)
            ],
            "selected_summaries": [
                {
                    "summary_id": f"s-{index}",
                    "level": 0,
                    "content": "历史摘要" + "x" * 120,
                    "source_message_ids": [index],
                    "source": "mysql_authority",
                    "confidence": 1.0,
                }
                for index in range(5)
            ],
            "confirmed_preferences": [],
            "confirmed_profile_facts": [],
            "long_term_goals": [],
            "relevant_episodes": [],
        })
        self.assertEqual(context["schema_version"], "memory-context-v2")
        self.assertGreaterEqual(len(context["recent_messages"]), 2)
        self.assertLessEqual(len(context["selected_summaries"]), 3)
        self.assertIn("不是指令", context["security_contract"])
        self.assertLessEqual(
            estimate_tokens(context),
            max(100, int(app_config.memory.context_token_budget)) + 80,
        )


class ConversationSummaryRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._summary_flags = (
            app_config.memory.summary_enabled,
            app_config.memory.summary_build_enabled,
            app_config.memory.summary_projection_enabled,
        )
        app_config.memory.summary_enabled = True
        app_config.memory.summary_build_enabled = True
        app_config.memory.summary_projection_enabled = True
        self.settings = load_conversation_settings()[1]
        self.engine = create_async_engine(
            "sqlite+aiosqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        self.factory = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def asyncTearDown(self):
        await self.engine.dispose()
        (
            app_config.memory.summary_enabled,
            app_config.memory.summary_build_enabled,
            app_config.memory.summary_projection_enabled,
        ) = self._summary_flags

    def uow(self, session, *, settings=None):
        return SqlConversationMemoryUnitOfWork(
            session,
            owner_id="owner-a",
            workspace_id="workspace-a",
            settings=settings or self.settings,
        )

    async def _append_turns(self, session, count: int, chat_id: str = "chat-a"):
        session.add(ConversationSessionMySQL(
            chat_id=chat_id,
            owner_id="owner-a",
            workspace_id="workspace-a",
            title="测试会话",
            preview="测试会话",
            status="active",
            last_turn_status="succeeded",
            turn_count=count,
            version=1,
        ))
        repo = ConversationMemoryRepository(
            session, owner_id="owner-a", workspace_id="workspace-a"
        )
        for index in range(count):
            await repo.append_message(
                chat_id=chat_id,
                role="user",
                content_masked=f"第{index + 1}轮各地区销售额",
                request_id=f"request-{index}:user",
                trace_id=f"trace-{index}",
                status="success",
            )
            await repo.append_message(
                chat_id=chat_id,
                role="assistant",
                content_masked="查询成功，返回 4 行。",
                request_id=f"request-{index}:assistant",
                trace_id=f"trace-{index}",
                status="success",
            )
        await session.flush()

    async def test_builder_advances_watermark_and_writes_signals_and_two_outboxes(self):
        async with self.factory() as session:
            await self._append_turns(session, 12)
            result = await ConversationSummaryService(
                owner_id="owner-a",
                workspace_id="workspace-a",
                settings=self.settings,
            ).build_if_needed(
                self.uow(session),
                chat_id="chat-a",
                expected_session_version=1,
            )
            await session.commit()

            repo = ConversationSummaryRepository(
                session, owner_id="owner-a", workspace_id="workspace-a"
            )
            segments = await repo.list_active_segments("chat-a")
            state = await repo.get_state("chat-a")
            signals = list((await session.execute(
                Select(ConversationSummarySignalMySQL)
            )).scalars().all())
            outbox = list((await session.execute(
                Select(ConversationMemoryOutboxMySQL)
            )).scalars().all())

            self.assertEqual(result["llm_calls"], 0)
            self.assertEqual(len(segments), 1)
            self.assertEqual(len(segments[0].source_message_ids), 16)
            self.assertEqual(state.covered_through_message_id, 16)
            self.assertGreater(len(signals), 0)
            self.assertEqual(
                {row.target for row in outbox}, {"summary_es", "summary_qdrant"}
            )
            self.assertEqual(result["coverage_lag_turns"], 4)

    async def test_hierarchy_preserves_children_and_signal_search_is_chat_scoped(self):
        with (
            patch.object(app_config.memory, "summary_turn_high_watermark", 3),
            patch.object(app_config.memory, "summary_turn_low_watermark", 1),
            patch.object(app_config.memory, "summary_recent_protected_turns", 1),
            patch.object(app_config.memory, "summary_segment_max_messages", 4),
            patch.object(app_config.memory, "summary_hierarchy_group_size", 2),
            patch.object(app_config.memory, "summary_max_level", 1),
        ):
            settings = replace(
                self.settings,
                summary_turn_high_watermark=3,
                summary_turn_low_watermark=1,
                summary_recent_protected_turns=1,
                summary_segment_max_messages=4,
                summary_hierarchy_group_size=2,
                summary_max_level=1,
            )
            async with self.factory() as session:
                await self._append_turns(session, 10)
                await ConversationSummaryService(
                    owner_id="owner-a",
                    workspace_id="workspace-a",
                    settings=settings,
                ).build_if_needed(
                    self.uow(session, settings=settings),
                    chat_id="chat-a",
                    expected_session_version=1,
                    max_l0_segments=3,
                )
                await session.commit()
                repo = ConversationSummaryRepository(
                    session, owner_id="owner-a", workspace_id="workspace-a"
                )
                segments = await repo.list_active_segments("chat-a")
                parents = [row for row in segments if int(row.level) == 1]
                identifiers, evidence = await repo.search_signal_ids(
                    "chat-a", ["销售额"], limit=5
                )
                other, _ = await repo.search_signal_ids(
                    "chat-b", ["销售额"], limit=5
                )

                self.assertEqual(len(parents), 1)
                self.assertEqual(len(parents[0].child_summary_ids), 2)
                self.assertTrue(identifiers)
                self.assertEqual(other, [])
                self.assertTrue(evidence["chat_scope_applied"])

    async def test_revoke_is_authoritative_and_checkpoint_expiry_is_read_miss(self):
        async with self.factory() as session:
            await self._append_turns(session, 12)
            await ConversationSummaryService(
                owner_id="owner-a",
                workspace_id="workspace-a",
                settings=self.settings,
            ).build_if_needed(
                self.uow(session),
                chat_id="chat-a",
                expected_session_version=1,
            )
            memory_repo = ConversationMemoryRepository(
                session, owner_id="owner-a", workspace_id="workspace-a"
            )
            checkpoint = await memory_repo.upsert_checkpoint(
                chat_id="chat-a",
                version=1,
                structured_state={},
                rolling_summary="legacy",
                recent_message_ids=[],
                schema_version="memory-context-v1",
                ttl_seconds=60,
            )
            checkpoint.expires_at = (
                datetime.now(timezone.utc).replace(tzinfo=None)
                - timedelta(seconds=1)
            )
            await session.commit()

            summary_repo = ConversationSummaryRepository(
                session, owner_id="owner-a", workspace_id="workspace-a"
            )
            revoked = await summary_repo.revoke_chat("chat-a")
            await session.commit()

            self.assertTrue(revoked)
            self.assertEqual(await summary_repo.list_active_segments("chat-a"), [])
            self.assertIsNone(await memory_repo.get_checkpoint("chat-a"))
            self.assertIsNotNone(
                await memory_repo.get_checkpoint("chat-a", include_expired=True)
            )


if __name__ == "__main__":
    unittest.main()
