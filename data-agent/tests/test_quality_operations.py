import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from evals.application.services.quality_operations_service import QualityOperationsService
from app.infrastructure.persistence.mysql.records.feedback_mysql import FeedbackMySQL
from app.infrastructure.persistence.mysql.records.query_trace_mysql import QueryStepLogMySQL, QueryTraceMySQL, TokenUsageMySQL
from app.infrastructure.persistence.mysql.records.training_info_mysql import TrainingInfoMySQL
from evals.infrastructure.persistence.mysql.quality_operations_store import (
    MySQLQualityOperationsStore,
)


def obj(**values):
    return SimpleNamespace(**values)


class QualityOperationsTests(unittest.TestCase):
    def test_aggregate_is_exact_and_privacy_minimized(self):
        traces = [
            obj(id="t1", chat_id="c1", query="查询 alice@example.com 的订单 13800138000", status="success",
                failure_code="none", query_category="metric", run_fingerprint="fp", version_run_id="run-full",
                version_complete=True, total_tokens=100, estimated_cost=.01, tt_complete_ms=100,
                duration_ms=100, cache_hit=True, started_at=datetime(2026, 7, 16, 9, 0)),
            obj(id="t2", chat_id="c2", query="查询客户 123456789", status="failed",
                failure_code="none", query_category="ranking", run_fingerprint="fp", version_run_id="run-full",
                version_complete=True, total_tokens=300, estimated_cost=.03, tt_complete_ms=300,
                duration_ms=300, cache_hit=False, started_at=datetime(2026, 7, 16, 10, 0)),
        ]
        steps = [
            obj(trace_id="t1", step_name="generate_sql", status="success", duration_ms=60),
            obj(trace_id="t2", step_name="correct_sql", status="failed", duration_ms=200),
        ]
        usages = [obj(step_name="generate_sql", total_tokens=80), obj(step_name="correct_sql", total_tokens=250)]
        feedback = [
            obj(id=1, chat_id="c1", query="alice@example.com 订单", sql="SELECT * FROM t WHERE phone='13800138000'",
                score=-1, comment="客户 123456789 不正确", created_at=datetime(2026, 7, 16), review_status="pending",
                review_category=None, review_note=None, source_trace_id=None, candidate_type=None,
                candidate_status="none", candidate_question=None, candidate_sql=None, candidate_training_id=None),
            obj(id=2, chat_id="outside-filter", query="另一个问题", sql=None, score=-1, comment="展示问题",
                created_at=datetime(2026, 7, 16), review_status="pending", review_category=None,
                review_note=None, source_trace_id=None, candidate_type=None, candidate_status="none",
                candidate_question=None, candidate_sql=None, candidate_training_id=None),
        ]
        result = QualityOperationsService(None, "workspace-a").aggregate(
            traces, steps, usages, feedback, days=7, filters={}, metric_chat_ids={"c1", "c2"},
            approved=[{
                "run_id": "run-full", "dataset_suite": "easy_sql_v1",
            }], evaluation_runs=[],
        )
        self.assertEqual(result["summary"]["success_rate"], .5)
        self.assertEqual(result["summary"]["negative_rate"], 1.0)
        self.assertEqual(result["summary"]["result_proxy_rate"], 0.0)
        self.assertEqual(result["summary"]["p50_ms"], 100)
        self.assertEqual(result["summary"]["p95_ms"], 300)
        self.assertEqual(result["summary"]["tokens_mean"], 200)
        self.assertEqual(result["summary"]["correction_rate"], .5)
        self.assertEqual(result["summary"]["cache_hit_rate"], .5)
        self.assertEqual(result["failure_funnel"], [{"failure_code": "pipeline_unknown", "count": 1}])
        self.assertEqual(len(result["feedback_queue"]), 2)
        rendered = str(result)
        self.assertNotIn("alice@example.com", rendered)
        self.assertNotIn("13800138000", rendered)
        self.assertNotIn("123456789", rendered)
        self.assertIn("<email>", rendered)
        self.assertTrue(result["versions"][0]["baseline_bound"])
        self.assertEqual(
            result["versions"][0]["baseline_difficulties"], ["easy"],
        )
        self.assertTrue(result["coverage"]["exact"])
        self.assertFalse(result["coverage"]["truncated"])

    def test_retired_suite_approval_does_not_bind_current_version(self):
        trace = obj(
            id="t1", chat_id="c1", query="查询销售额", status="success",
            failure_code="none", query_category="metric", run_fingerprint="fp",
            version_run_id="legacy-run", version_complete=True,
            total_tokens=10, estimated_cost=0, tt_complete_ms=10,
            duration_ms=10, cache_hit=False,
            started_at=datetime(2026, 8, 6, 10, 0),
        )
        result = QualityOperationsService(None, "workspace-a").aggregate(
            [trace], [], [], [], days=7, filters={}, metric_chat_ids={"c1"},
            approved=[{
                "run_id": "legacy-run",
                "dataset_suite": "common_sql_v1",
                "status": "approved",
            }],
            evaluation_runs=[],
        )
        self.assertFalse(result["versions"][0]["baseline_bound"])
        self.assertEqual(result["versions"][0]["baseline_difficulties"], [])


class QualityCandidateTransactionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine(
            "sqlite+aiosqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
        )
        self.factory = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as connection:
            await connection.run_sync(FeedbackMySQL.__table__.create)
            await connection.run_sync(TrainingInfoMySQL.__table__.create)
            await connection.run_sync(QueryTraceMySQL.__table__.create)
            await connection.run_sync(QueryStepLogMySQL.__table__.create)
            await connection.run_sync(TokenUsageMySQL.__table__.create)

    async def asyncTearDown(self):
        await self.engine.dispose()

    async def test_training_confirmation_is_atomic_and_idempotent(self):
        async with self.factory() as session:
            item = FeedbackMySQL(
                chat_id="chat", owner_id="owner", workspace_id="workspace-a",
                query="问题", sql="SELECT 1", score=-1, review_status="reviewed",
                candidate_type="training", candidate_status="draft",
                candidate_question="标准问题", candidate_sql="SELECT 1",
            )
            session.add(item)
            await session.commit()
            service = QualityOperationsService(
                MySQLQualityOperationsStore(session, "workspace-a"),
                "workspace-a",
            )
            first = await service.confirm_candidate(item.id, confirmer_id="admin-a")
            second = await service.confirm_candidate(item.id, confirmer_id="admin-a")
            count = int((await session.execute(select(func.count()).select_from(TrainingInfoMySQL))).scalar() or 0)
            self.assertEqual(count, 1)
            self.assertEqual(first["candidate_status"], "confirmed")
            self.assertEqual(second["candidate_training_id"], first["candidate_training_id"])

    async def test_review_and_draft_validation_stay_in_application_use_case(self):
        async with self.factory() as session:
            item = FeedbackMySQL(
                chat_id="chat-review",
                owner_id="owner",
                workspace_id="workspace-a",
                query="问题",
                sql="SELECT 1",
                score=-1,
                review_status="pending",
            )
            session.add(item)
            await session.commit()
            service = QualityOperationsService(
                MySQLQualityOperationsStore(session, "workspace-a"),
                "workspace-a",
            )

            with self.assertRaisesRegex(ValueError, "有效分类"):
                await service.review_feedback(
                    item.id,
                    status="reviewed",
                    category=None,
                    note="",
                    reviewer_id="admin-a",
                )
            reviewed = await service.review_feedback(
                item.id,
                status="reviewed",
                category="sql_issue",
                note="需要修订",
                reviewer_id="admin-a",
            )
            drafted = await service.draft_candidate(
                item.id,
                candidate_type="golden",
                question="标准问题",
                sql="SELECT 1",
            )

            self.assertEqual(reviewed["review_status"], "reviewed")
            self.assertEqual(drafted["candidate_status"], "draft")

    async def test_overview_filters_feedback_by_matching_trace_in_sql(self):
        async with self.factory() as session:
            session.add_all([
                QueryTraceMySQL(
                    id="trace-v1", chat_id="chat-v1", workspace_id="workspace-a",
                    query="版本一", status="success", run_fingerprint="version-1",
                    query_category="metric", started_at=datetime.now(),
                ),
                QueryTraceMySQL(
                    id="trace-v2", chat_id="chat-v2", workspace_id="workspace-a",
                    query="版本二", status="success", run_fingerprint="version-2",
                    query_category="metric", started_at=datetime.now(),
                ),
                FeedbackMySQL(
                    chat_id="chat-v1", workspace_id="workspace-a", query="版本一",
                    score=-1,
                ),
                FeedbackMySQL(
                    chat_id="chat-v2", workspace_id="workspace-a", query="版本二",
                    score=-1,
                ),
            ])
            await session.commit()
            approved_all = [{
                "workspace_id": "workspace-a", "run_id": "easy-r1",
                "dataset_suite": "easy_sql_v1", "status": "approved",
            }]
            catalog = SimpleNamespace(
                approved_entries=AsyncMock(return_value=approved_all),
                list_runs=AsyncMock(return_value=[]),
            )
            result = await QualityOperationsService(
                MySQLQualityOperationsStore(session, "workspace-a"),
                "workspace-a",
                evaluation_catalog=catalog,
            ).overview(version="version-1")
            catalog.approved_entries.assert_awaited_once_with("workspace-a")
            self.assertEqual(result["coverage"]["trace"]["matched"], 1)
            self.assertEqual(result["coverage"]["feedback"]["matched"], 1)
            self.assertEqual(result["summary"]["feedback_total"], 1)


if __name__ == "__main__":
    unittest.main()
