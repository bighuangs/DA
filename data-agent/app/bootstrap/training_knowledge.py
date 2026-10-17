"""SQL 训练知识导入的 CLI 组合根。"""

from __future__ import annotations

import asyncio
import os

from app.application.ports.training_knowledge import TrainingSeed
from app.application.services.training_knowledge_service import TrainingKnowledgeService
from app.config.meta_config import meta_config
from app.infrastructure.clients.embedding_client_manager import embedding_client_manager
from app.infrastructure.clients.mysql_client_manager import meta_mysql_client_manager
from app.infrastructure.clients.qdrant_client_manager import qdrant_client_manager
from app.infrastructure.observability.logging import logger
from app.infrastructure.persistence.mysql.repositories.meta_mysql_repository import (
    MetaMysqlRepository,
)
from app.infrastructure.persistence.mysql.training_knowledge_store import (
    MysqlTrainingKnowledgeStore,
)
from app.infrastructure.search.metadata_index_adapter import MetadataIndexAdapter
from app.infrastructure.search.qdrant.repositories.column_qdrant_repository import (
    ColumnQdrantRepository,
)
from app.infrastructure.search.qdrant.repositories.metric_qdrant_repository import (
    MetricQdrantRepository,
)
from app.infrastructure.search.qdrant.repositories.training_qdrant_repository import (
    TrainingQdrantRepository,
)


async def run_training_knowledge_import(
    *,
    workspace_id: str,
) -> None:
    """装配导入用例并管理本地资源生命周期。"""

    normalized_workspace_id = str(workspace_id or "").strip()
    if not normalized_workspace_id:
        raise ValueError("init_training_knowledge 必须显式设置 WORKSPACE_ID")

    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost,::1")
    embedding_client_manager.init()
    meta_mysql_client_manager.init()
    qdrant_client_manager.init()
    try:
        async with meta_mysql_client_manager.session_factory() as session:
            meta_repo = MetaMysqlRepository(session, normalized_workspace_id)
            projection = MetadataIndexAdapter(
                meta_repo=meta_repo,
                column_repo=ColumnQdrantRepository(qdrant_client_manager.client),
                metric_repo=MetricQdrantRepository(qdrant_client_manager.client),
                training_repo=TrainingQdrantRepository(qdrant_client_manager.client),
                embedding_client=embedding_client_manager.client,
            )
            result = await TrainingKnowledgeService(
                store=MysqlTrainingKnowledgeStore(session),
                projection=projection,
            ).replace_and_rebuild(
                workspace_id=normalized_workspace_id,
                seeds=tuple(
                    TrainingSeed(question=item.question, sql=item.sql)
                    for item in meta_config.trainings
                ),
            )
            logger.info(
                "SQL 训练知识导入完成: workspace_id={}, persisted={}, indexed={}",
                result.workspace_id,
                result.persisted_count,
                result.indexed_count,
            )
    finally:
        await meta_mysql_client_manager.close()
        await qdrant_client_manager.close()
        await embedding_client_manager.close()


def main() -> None:
    """从环境变量读取 workspace 并运行导入。"""

    asyncio.run(
        run_training_knowledge_import(
            workspace_id=str(os.getenv("WORKSPACE_ID") or ""),
        )
    )


__all__ = ["main", "run_training_knowledge_import"]
