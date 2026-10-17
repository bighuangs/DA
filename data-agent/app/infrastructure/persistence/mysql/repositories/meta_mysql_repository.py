from typing import Optional
from datetime import datetime, timedelta

from sqlalchemy import Select, delete, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.persistence.mysql.records.column_info_mysql import ColumnInfoMySQL
from app.infrastructure.persistence.mysql.records.column_metric_mysql import ColumnMetricMySQL
from app.infrastructure.persistence.mysql.records.metric_info_mysql import MetricInfoMySQL
from app.infrastructure.persistence.mysql.records.table_info_mysql import TableInfoMySQL
from app.infrastructure.persistence.mysql.records.training_info_mysql import TrainingInfoMySQL
from app.infrastructure.persistence.mysql.records.schema_sync_log_mysql import SchemaSyncLogMySQL
from app.infrastructure.persistence.mysql.records.query_trace_mysql import QueryTraceMySQL, QueryStepLogMySQL, TokenUsageMySQL
from app.infrastructure.persistence.mysql.records.adhoc_dataset_mysql import AdhocDatasetMySQL
from app.infrastructure.persistence.mysql.records.masking_rule_mysql import MaskingRuleMySQL
from app.infrastructure.persistence.mysql.records.user_mysql import UserMySQL
from app.infrastructure.persistence.mysql.records.workspace_mysql import WorkspaceMySQL
from app.infrastructure.persistence.mysql.records.workspace_member_mysql import WorkspaceMemberMySQL
from app.infrastructure.persistence.mysql.records.access_policy_mysql import AccessPolicyMySQL
from app.infrastructure.persistence.mysql.records.chat_history_mysql import ChatHistoryMySQL
from app.infrastructure.persistence.mysql.records.feedback_mysql import FeedbackMySQL
from app.infrastructure.persistence.mysql.records.dashboard_mysql import DashboardMySQL
from app.infrastructure.persistence.mysql.records.evaluation_baseline_mysql import EvaluationBaselineMySQL
from app.infrastructure.persistence.mysql.records.conversation_message_mysql import ConversationMessageMySQL
from app.infrastructure.persistence.mysql.records.conversation_working_memory_checkpoint_mysql import (
    ConversationWorkingMemoryCheckpointMySQL,
)
from app.infrastructure.persistence.mysql.records.conversation_memory_item_mysql import ConversationMemoryItemMySQL
from app.infrastructure.persistence.mysql.records.conversation_memory_outbox_mysql import ConversationMemoryOutboxMySQL
from app.infrastructure.persistence.mysql.records.conversation_summary_segment_mysql import ConversationSummarySegmentMySQL
from app.infrastructure.persistence.mysql.records.conversation_summary_signal_mysql import ConversationSummarySignalMySQL
from app.infrastructure.persistence.mysql.records.conversation_summary_state_mysql import ConversationSummaryStateMySQL
from app.infrastructure.persistence.mysql.records.conversation_session_mysql import ConversationSessionMySQL
from app.infrastructure.persistence.mysql.records.conversation_turn_mysql import ConversationTurnMySQL
from app.infrastructure.persistence.mysql.records.conversation_data_context_mysql import (
    ConversationDataContextMySQL,
    ConversationTurnDataContextMySQL,
    DatasetResourceLedgerMySQL,
    SourceMappingMySQL,
)
from app.infrastructure.persistence.mysql.records.base import Base


# These models are imported here deliberately so ``Base.metadata.create_all``
# sees the complete metadata graph in local/dev bootstrap flows.
_REGISTERED_MODELS = (
    ChatHistoryMySQL,
    AdhocDatasetMySQL,
    UserMySQL,
    WorkspaceMySQL,
    WorkspaceMemberMySQL,
    AccessPolicyMySQL,
    DashboardMySQL,
    EvaluationBaselineMySQL,
    ConversationMessageMySQL,
    ConversationWorkingMemoryCheckpointMySQL,
    ConversationMemoryItemMySQL,
    ConversationMemoryOutboxMySQL,
    ConversationSummarySegmentMySQL,
    ConversationSummarySignalMySQL,
    ConversationSummaryStateMySQL,
    ConversationSessionMySQL,
    ConversationTurnMySQL,
    ConversationDataContextMySQL,
    ConversationTurnDataContextMySQL,
    DatasetResourceLedgerMySQL,
    SourceMappingMySQL,
)


_INTERNAL_ASSET_PREFIXES = ("adhoc_", "da_upload_", "eval_upload_", "__da_")


def _is_internal_asset_table(value: object) -> bool:
    return str(value or "").lower().startswith(_INTERNAL_ASSET_PREFIXES)


class MetaMysqlRepository:
    def __init__(self, session: AsyncSession, workspace_id: str | None = None):
        self.session = session
        self.workspace_id = workspace_id

    async def cleanup_sensitive_history(self, retention_days: int) -> dict[str, int]:
        """按配置清理查询历史、链路日志、Token 记录和反馈中的敏感上下文。"""
        if retention_days <= 0:
            return {}
        cutoff = datetime.utcnow() - timedelta(days=retention_days)
        trace_ids = select(QueryTraceMySQL.id).where(QueryTraceMySQL.created_at < cutoff)
        counts: dict[str, int] = {}
        for name, statement in (
            ("steps", delete(QueryStepLogMySQL).where(QueryStepLogMySQL.trace_id.in_(trace_ids))),
            ("tokens", delete(TokenUsageMySQL).where(TokenUsageMySQL.trace_id.in_(trace_ids))),
            ("traces", delete(QueryTraceMySQL).where(QueryTraceMySQL.created_at < cutoff)),
            # Conversation transcript is the provenance authority for summary
            # and stable facts.  It is governed by explicit chat deletion and
            # must not be physically removed by trace-history retention.
            ("feedback", delete(FeedbackMySQL).where(FeedbackMySQL.created_at < cutoff)),
        ):
            result = await self.session.execute(statement)
            counts[name] = int(result.rowcount or 0)
        await self.session.commit()
        return counts

    # 向table_info表中插入多条数据
    def save_table_infos(self, table_infos: list[TableInfoMySQL]):
        self.session.add_all(table_infos)

    # 向column_info表中插入多条数据
    def save_column_infos(self, column_infos: list[ColumnInfoMySQL]):
        self.session.add_all(column_infos)

    # 向metric_info表中插入多条数据
    def save_metric_infos(self, metric_infos: list[MetricInfoMySQL]):
        self.session.add_all(metric_infos)

    # 向column_metric表中插入多条数据
    def save_column_metrics(self, column_metrics: list[ColumnMetricMySQL]):
        self.session.add_all(column_metrics)

    async def get_colun_info_by_id(self, column_id) ->ColumnInfoMySQL:
        item = await self.session.get(ColumnInfoMySQL, column_id)
        return None if item and _is_internal_asset_table(item.table_id) else item

    async def get_column_infos_by_table_id(self, table_id: str)->list[ColumnInfoMySQL]:
        if _is_internal_asset_table(table_id):
            return []
        result = await self.session.execute(
            Select(ColumnInfoMySQL)
                .where(ColumnInfoMySQL.table_id==table_id)
                .where(or_(
                    ColumnInfoMySQL.role.in_(["primary_key", "foreign_key"]),
                    ColumnInfoMySQL.name.like("%\\_id", escape="\\"),
                ))
        )

        return result.scalars().all()

    async def get_table_info_by_table_id(self, table_id: str)->TableInfoMySQL:
        if _is_internal_asset_table(table_id):
            return None
        return await self.session.get(TableInfoMySQL, table_id)

    # ==================== SQL 样例库（training_info） ====================

    def save_training_infos(self, training_infos: list[TrainingInfoMySQL]):
        """批量写入训练样例到 training_info 表。"""
        self.session.add_all(training_infos)

    async def get_all_enabled_training_infos(self) -> list[TrainingInfoMySQL]:
        """获取全部启用的训练样例，供构建 Qdrant 向量索引使用。"""
        statement = Select(TrainingInfoMySQL).where(TrainingInfoMySQL.enabled == True)  # noqa: E712
        if self.workspace_id:
            statement = statement.where(TrainingInfoMySQL.workspace_id == self.workspace_id)
        result = await self.session.execute(statement)
        return result.scalars().all()

    # ==================== 指标库（metric_info）子串匹配辅助 ====================

    async def get_all_metric_infos(self) -> list[MetricInfoMySQL]:
        """获取全部指标（含纯术语型，relevant_columns 为空者），供 recall_metric 子串匹配使用。"""
        statement = Select(MetricInfoMySQL).where(MetricInfoMySQL.enabled == True)  # noqa: E712
        if self.workspace_id:
            statement = statement.where(MetricInfoMySQL.workspace_id == self.workspace_id)
        result = await self.session.execute(statement.order_by(MetricInfoMySQL.name))
        return result.scalars().all()

    # ==================== 元知识管理后台（TODO-24） ====================

    async def list_metric_infos(self, kind: Optional[str] = None,
                                search: Optional[str] = None) -> list[MetricInfoMySQL]:
        """列出指标；kind=terminology 时只返回纯术语型指标。"""
        stmt = select(MetricInfoMySQL).order_by(MetricInfoMySQL.name)
        if self.workspace_id:
            stmt = stmt.where(MetricInfoMySQL.workspace_id == self.workspace_id)
        if kind == "terminology":
            stmt = stmt.where(MetricInfoMySQL.relevant_columns == [])
        elif kind == "metric":
            stmt = stmt.where(MetricInfoMySQL.relevant_columns != [])
        if search:
            stmt = stmt.where(MetricInfoMySQL.name.like(f"%{search}%"))
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_metric_info(self, metric_id: str) -> Optional[MetricInfoMySQL]:
        item = await self.session.get(MetricInfoMySQL, metric_id)
        if item and self.workspace_id and item.workspace_id != self.workspace_id:
            return None
        return item

    async def create_metric_info(self, *, metric_id: str, name: str,
                                 description: str, relevant_columns: list,
                                 alias: list, enabled: bool,
                                 formula_rule: dict | None = None) -> MetricInfoMySQL:
        item = MetricInfoMySQL(
            id=metric_id,
            name=name,
            description=description,
            relevant_columns=relevant_columns,
            alias=alias,
            formula_rule=formula_rule,
            enabled=enabled,
            workspace_id=self.workspace_id,
        )
        self.session.add(item)
        self.session.add_all([
            ColumnMetricMySQL(column_id=column_id, metric_id=metric_id)
            for column_id in relevant_columns
        ])
        await self.session.commit()
        await self.session.refresh(item)
        return item

    async def update_metric_info(self, item: MetricInfoMySQL, *, name: str,
                                 description: str, relevant_columns: list,
                                 alias: list, enabled: bool,
                                 formula_rule: dict | None = None) -> MetricInfoMySQL:
        item.name = name
        item.description = description
        item.relevant_columns = relevant_columns
        item.alias = alias
        item.enabled = enabled
        # 后台旧客户端未传该字段时保留已有口径，避免一次普通编辑清空规则。
        if formula_rule is not None:
            item.formula_rule = formula_rule
        await self.session.execute(
            delete(ColumnMetricMySQL).where(ColumnMetricMySQL.metric_id == item.id)
        )
        self.session.add_all([
            ColumnMetricMySQL(column_id=column_id, metric_id=item.id)
            for column_id in relevant_columns
        ])
        await self.session.commit()
        await self.session.refresh(item)
        return item

    async def delete_metric_info(self, item: MetricInfoMySQL) -> None:
        await self.session.execute(
            delete(ColumnMetricMySQL).where(ColumnMetricMySQL.metric_id == item.id)
        )
        await self.session.delete(item)
        await self.session.commit()

    async def list_training_infos(self, search: Optional[str] = None) -> list[TrainingInfoMySQL]:
        stmt = select(TrainingInfoMySQL).order_by(TrainingInfoMySQL.created_at.desc(), TrainingInfoMySQL.id.desc())
        if self.workspace_id:
            stmt = stmt.where(TrainingInfoMySQL.workspace_id == self.workspace_id)
        if search:
            stmt = stmt.where(TrainingInfoMySQL.question.like(f"%{search}%"))
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_training_info(self, training_id: int) -> Optional[TrainingInfoMySQL]:
        item = await self.session.get(TrainingInfoMySQL, training_id)
        if item and self.workspace_id and item.workspace_id != self.workspace_id:
            return None
        return item

    async def create_training_info(self, *, question: str, sql: str,
                                   enabled: bool) -> TrainingInfoMySQL:
        item = await self.add_training_info(question=question, sql=sql, enabled=enabled)
        await self.session.commit()
        await self.session.refresh(item)
        return item

    async def add_training_info(self, *, question: str, sql: str,
                                enabled: bool) -> TrainingInfoMySQL:
        """Stage a training row without committing; services can compose one transaction."""
        item = TrainingInfoMySQL(question=question, sql=sql, enabled=enabled, workspace_id=self.workspace_id)
        self.session.add(item)
        await self.session.flush()
        return item

    async def update_training_info(self, item: TrainingInfoMySQL, *, question: str,
                                   sql: str, enabled: bool) -> TrainingInfoMySQL:
        item.question = question
        item.sql = sql
        item.enabled = enabled
        await self.session.commit()
        await self.session.refresh(item)
        return item

    async def delete_training_info(self, item: TrainingInfoMySQL) -> None:
        await self.session.delete(item)
        await self.session.commit()

    async def list_table_infos(self) -> list[tuple[TableInfoMySQL, list[ColumnInfoMySQL]]]:
        tables = (await self.session.execute(select(TableInfoMySQL).order_by(TableInfoMySQL.name))).scalars().all()
        columns = (await self.session.execute(select(ColumnInfoMySQL).order_by(ColumnInfoMySQL.table_id, ColumnInfoMySQL.name))).scalars().all()
        tables = [table for table in tables if not _is_internal_asset_table(table.name)]
        columns = [
            column for column in columns
            if not _is_internal_asset_table(column.table_id)
        ]
        grouped: dict[str, list[ColumnInfoMySQL]] = {}
        for column in columns:
            grouped.setdefault(column.table_id, []).append(column)
        return [(table, grouped.get(table.id, [])) for table in tables]

    async def list_masking_rules(self) -> list[MaskingRuleMySQL]:
        statement = select(MaskingRuleMySQL)
        if self.workspace_id:
            statement = statement.where(MaskingRuleMySQL.workspace_id == self.workspace_id)
        result = await self.session.execute(
            statement.order_by(MaskingRuleMySQL.priority, MaskingRuleMySQL.id)
        )
        return result.scalars().all()

    async def get_masking_rule(self, rule_id: int) -> Optional[MaskingRuleMySQL]:
        item = await self.session.get(MaskingRuleMySQL, rule_id)
        if item and self.workspace_id and item.workspace_id != self.workspace_id:
            return None
        return item

    async def create_masking_rule(self, *, column_pattern: str, data_kind: str,
                                  strategy: str, enabled: bool, priority: int,
                                  description: str) -> MaskingRuleMySQL:
        item = MaskingRuleMySQL(
            workspace_id=self.workspace_id,
            column_pattern=column_pattern, data_kind=data_kind, strategy=strategy,
            enabled=enabled, priority=priority, description=description,
        )
        self.session.add(item)
        await self.session.commit()
        await self.session.refresh(item)
        return item

    async def update_masking_rule(self, item: MaskingRuleMySQL, *, column_pattern: str,
                                  data_kind: str, strategy: str, enabled: bool,
                                  priority: int, description: str) -> MaskingRuleMySQL:
        item.column_pattern = column_pattern
        item.data_kind = data_kind
        item.strategy = strategy
        item.enabled = enabled
        item.priority = priority
        item.description = description
        await self.session.commit()
        await self.session.refresh(item)
        return item

    async def delete_masking_rule(self, item: MaskingRuleMySQL) -> None:
        await self.session.delete(item)
        await self.session.commit()

    async def replace_schema_infos(
        self,
        table_infos: list[TableInfoMySQL],
        column_infos: list[ColumnInfoMySQL],
    ) -> None:
        """发布一次 Schema 快照，并清理已删除字段对应的指标关联。"""
        metrics = await self.get_all_metric_infos()
        valid_columns = {column.id for column in column_infos}
        for metric in metrics:
            metric.relevant_columns = [
                column_id for column_id in (metric.relevant_columns or [])
                if column_id in valid_columns
            ]
        await self.session.execute(delete(ColumnMetricMySQL))
        await self.session.execute(delete(ColumnInfoMySQL))
        await self.session.execute(delete(TableInfoMySQL))
        self.session.add_all(table_infos)
        self.session.add_all(column_infos)
        self.session.add_all([
            ColumnMetricMySQL(column_id=column_id, metric_id=metric.id)
            for metric in metrics
            for column_id in (metric.relevant_columns or [])
        ])
        await self.session.commit()

    async def save_schema_sync_log(self, log: SchemaSyncLogMySQL) -> SchemaSyncLogMySQL:
        self.session.add(log)
        await self.session.commit()
        await self.session.refresh(log)
        return log

    async def list_schema_sync_logs(self, limit: int = 10) -> list[SchemaSyncLogMySQL]:
        result = await self.session.execute(
            select(SchemaSyncLogMySQL)
            .order_by(SchemaSyncLogMySQL.id.desc())
            .limit(max(1, min(limit, 50)))
        )
        return result.scalars().all()

    async def ensure_admin_tables(self) -> None:
        """Create the current Meta schema for a fresh development database."""
        await self.session.run_sync(
            lambda sync_session: Base.metadata.create_all(sync_session.connection())
        )
