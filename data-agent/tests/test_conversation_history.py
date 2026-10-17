import json
import unittest
from datetime import UTC, datetime, timedelta

from sqlalchemy import Select, delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.application.services.conversation_history_service import ConversationHistoryService
from app.domain.access.policy import PolicySnapshot
from app.infrastructure.persistence.mysql.records.base import Base
from app.infrastructure.persistence.mysql.records.chat_history_mysql import ChatHistoryMySQL
from app.infrastructure.persistence.mysql.records.workspace_member_mysql import WorkspaceMemberMySQL
from app.infrastructure.persistence.mysql.conversation_adapters import (
    SqlConversationHistoryStore,
)
from app.infrastructure.persistence.mysql.repositories.auth_repository import AuthRepository
from app.infrastructure.persistence.mysql.repositories.conversation_history_repository import (
    ConversationHistoryRepository,
    ConversationUnavailableError,
)


class ConversationHistoryTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def now() -> datetime:
        return datetime.now(UTC).replace(tzinfo=None)

    async def asyncSetUp(self):
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

    @staticmethod
    def repo(session, owner="owner-a", workspace="workspace-a"):
        return ConversationHistoryRepository(
            session,
            owner_id=owner,
            workspace_id=workspace,
        )

    @staticmethod
    def service(session, owner="owner-a", workspace="workspace-a"):
        return ConversationHistoryService(
            SqlConversationHistoryStore(
                session,
                owner_id=owner,
                workspace_id=workspace,
            ),
            access_policy=PolicySnapshot(policies=()),
        )

    async def test_pending_success_directory_and_transcript_are_scoped(self):
        async with self.factory() as session:
            repo = self.repo(session)
            await repo.start_turn(
                chat_id="chat-a",
                turn_id="trace-a",
                user_text="查看客户姓名",
                trace_id="trace-a",
            )
            history = ChatHistoryMySQL(
                chat_id="chat-a",
                owner_id="owner-a",
                workspace_id="workspace-a",
                query="查看客户姓名",
                sql="SELECT customer_name FROM dim_customer",
                result=json.dumps(
                    [{"customer_name": "张三"}, {"customer_name": "李四"}],
                    ensure_ascii=False,
                ),
                row_count=2,
                created_at=self.now(),
            )
            session.add(history)
            await session.flush()
            await repo.complete_turn(
                chat_id="chat-a",
                turn_id="trace-a",
                source_history_id=history.id,
                row_count=2,
            )
            await session.commit()

            service = self.service(session)
            directory = await service.list_sessions()
            self.assertEqual([item["chat_id"] for item in directory["items"]], ["chat-a"])
            self.assertEqual(directory["items"][0]["turn_count"], 1)
            transcript = await service.get_turns("chat-a")
            turn = transcript["items"][0]
            self.assertEqual(turn["status"], "succeeded")
            self.assertEqual(turn["sql"], "SELECT customer_name FROM dim_customer")
            self.assertNotEqual(
                turn["result"]["rows"][0]["customer_name"], "张三"
            )
            self.assertIn("chart", turn["unavailable_parts"])

            foreign_owner = self.service(session, owner="owner-b")
            foreign_workspace = self.service(session, workspace="workspace-b")
            self.assertEqual((await foreign_owner.list_sessions())["items"], [])
            self.assertEqual(
                (await foreign_workspace.list_sessions())["items"], []
            )
            self.assertIsNone(await foreign_owner.get_turns("chat-a"))

    async def test_transcript_reconstructs_explicit_line_chart_from_safe_preview(self):
        async with self.factory() as session:
            repo = self.repo(session)
            await repo.start_turn(
                chat_id="chat-line",
                turn_id="trace-line",
                user_text="用折线图展示月度销售额趋势",
                trace_id="trace-line",
            )
            history = ChatHistoryMySQL(
                chat_id="chat-line",
                owner_id="owner-a",
                workspace_id="workspace-a",
                query="用折线图展示月度销售额趋势",
                sql=(
                    "SELECT d.month, SUM(f.order_amount) AS total_amount "
                    "FROM fact_order f JOIN dim_date d ON f.date_id=d.date_id "
                    "GROUP BY d.month ORDER BY d.month"
                ),
                result=json.dumps(
                    [
                        {"month": 1, "total_amount": 632396550.30},
                        {"month": 2, "total_amount": 577492829.67},
                    ],
                    ensure_ascii=False,
                ),
                row_count=2,
                created_at=self.now(),
            )
            session.add(history)
            await session.flush()
            await repo.complete_turn(
                chat_id="chat-line",
                turn_id="trace-line",
                source_history_id=history.id,
                row_count=2,
            )
            await session.commit()

            transcript = await self.service(session).get_turns("chat-line")
            turn = transcript["items"][0]

            self.assertEqual(turn["chart"]["type"], "line")
            self.assertEqual(turn["chart"]["axis"]["x"]["value"], "month")
            self.assertNotIn("chart", turn["unavailable_parts"])

    async def test_handled_turn_is_successful_message_only_transcript(self):
        async with self.factory() as session:
            repo = self.repo(session)
            await repo.start_turn(
                chat_id="chat-guidance",
                turn_id="trace-guidance",
                user_text="今天天气怎么样",
                trace_id="trace-guidance",
            )
            await repo.complete_handled_turn(
                chat_id="chat-guidance",
                turn_id="trace-guidance",
                assistant_summary="我主要用于销售、订单、客户、商品、地区等业务数据分析。",
            )
            conversation = await repo.get_session("chat-guidance")
            completed_version = conversation.version
            await repo.complete_handled_turn(
                chat_id="chat-guidance",
                turn_id="trace-guidance",
                assistant_summary="重复完成不应改写原响应。",
            )
            self.assertEqual(conversation.version, completed_version)
            await session.commit()

            transcript = await self.service(session).get_turns("chat-guidance")
            turn = transcript["items"][0]

            self.assertEqual(turn["status"], "succeeded")
            self.assertEqual(turn["replay_fidelity"], "message_only")
            self.assertIn("销售", turn["assistant_summary"])
            self.assertIsNone(turn["sql"])
            self.assertIsNone(turn["result"])

    async def test_failed_turn_cannot_be_rewritten_as_handled(self):
        async with self.factory() as session:
            repo = self.repo(session)
            await repo.start_turn(
                chat_id="chat-failed-guidance",
                turn_id="trace-failed-guidance",
                user_text="今天天气怎么样",
                trace_id="trace-failed-guidance",
            )
            await repo.fail_turn(
                chat_id="chat-failed-guidance",
                turn_id="trace-failed-guidance",
                status="failed",
            )

            with self.assertRaises(ConversationUnavailableError):
                await repo.complete_handled_turn(
                    chat_id="chat-failed-guidance",
                    turn_id="trace-failed-guidance",
                    assistant_summary="不应覆盖失败状态。",
                )

    async def test_cursor_archive_delete_and_continuation_guards(self):
        async with self.factory() as session:
            repo = self.repo(session)
            for index in range(3):
                await repo.start_turn(
                    chat_id=f"chat-{index}",
                    turn_id=f"trace-{index}",
                    user_text=f"问题 {index}",
                    trace_id=f"trace-{index}",
                )
                row = await repo.get_session(f"chat-{index}")
                row.last_message_at = self.now() + timedelta(seconds=index)
            await session.commit()

            service = self.service(session)
            first = await service.list_sessions(limit=2)
            self.assertEqual(len(first["items"]), 2)
            self.assertTrue(first["has_more"])
            second = await service.list_sessions(
                limit=2, cursor=first["next_cursor"]
            )
            self.assertEqual(len(second["items"]), 1)
            self.assertFalse(second["has_more"])

            archived = await service.update_session(
                "chat-2", status="archived"
            )
            self.assertEqual(archived["status"], "archived")
            self.assertEqual(
                [item["chat_id"] for item in (await service.list_sessions())["items"]],
                ["chat-1", "chat-0"],
            )
            self.assertEqual(
                [item["chat_id"] for item in (
                    await service.list_sessions(status="archived")
                )["items"]],
                ["chat-2"],
            )
            with self.assertRaises(ConversationUnavailableError):
                await repo.start_turn(
                    chat_id="chat-2",
                    turn_id="trace-new",
                    user_text="继续",
                )

            self.assertTrue(await service.delete_session("chat-1"))
            self.assertIsNone(await service.get_turns("chat-1"))
            with self.assertRaises(ConversationUnavailableError):
                await repo.start_turn(
                    chat_id="chat-1",
                    turn_id="trace-after-delete",
                    user_text="尝试恢复已删除会话",
                )

    async def test_context_reset_keeps_transcript_snapshot(self):
        async with self.factory() as session:
            repo = self.repo(session)
            await repo.start_turn(
                chat_id="chat-reset",
                turn_id="trace-reset",
                user_text="各地区销售额",
            )
            history = ChatHistoryMySQL(
                chat_id="chat-reset",
                owner_id="owner-a",
                workspace_id="workspace-a",
                query="各地区销售额",
                sql="SELECT 1",
                result="[{\"value\": 1}]",
                row_count=1,
                created_at=self.now(),
            )
            session.add(history)
            await session.flush()
            await repo.complete_turn(
                chat_id="chat-reset",
                turn_id="trace-reset",
                source_history_id=history.id,
                row_count=1,
            )
            self.assertTrue(await repo.reset_context("chat-reset"))
            await session.commit()

            remaining = (
                await session.execute(
                    Select(ChatHistoryMySQL).where(
                        ChatHistoryMySQL.chat_id == "chat-reset"
                    )
                )
            ).scalars().all()
            self.assertEqual(len(remaining), 1)
            self.assertIsNotNone((await repo.get_session("chat-reset")).context_reset_at)

    async def test_deleting_fence_rejects_new_and_late_turns(self):
        async with self.factory() as session:
            repo = self.repo(session)
            await repo.start_turn(
                chat_id="chat-delete-fence",
                turn_id="trace-old",
                user_text="在途问题",
            )
            await session.commit()
            self.assertTrue(await repo.begin_delete("chat-delete-fence"))
            await session.commit()

            with self.assertRaises(ConversationUnavailableError):
                await repo.start_turn(
                    chat_id="chat-delete-fence",
                    turn_id="trace-new",
                    user_text="删除期间新问题",
                )
            with self.assertRaises(ConversationUnavailableError):
                await repo.complete_turn(
                    chat_id="chat-delete-fence",
                    turn_id="trace-old",
                    source_history_id=1,
                    row_count=1,
                )

    async def test_last_workspace_prefers_authorized_server_value(self):
        async with self.factory() as session:
            repo = AuthRepository(session)
            user = await repo.create_user(
                username="alice",
                display_name="Alice",
                password_hash="hash",
            )
            first = await repo.create_workspace(name="第一空间", owner_id=user.id)
            second = await repo.create_workspace(name="第二空间", owner_id=user.id)
            await session.commit()
            user.last_workspace_id = second.id
            await session.commit()
            self.assertEqual(await repo.preferred_workspace_id(user), second.id)

            await session.execute(
                delete(WorkspaceMemberMySQL).where(
                    WorkspaceMemberMySQL.workspace_id == second.id,
                    WorkspaceMemberMySQL.user_id == user.id,
                )
            )
            await session.commit()
            self.assertEqual(await repo.preferred_workspace_id(user), first.id)


if __name__ == "__main__":
    unittest.main()
