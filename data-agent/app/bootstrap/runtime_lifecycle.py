"""FastAPI 应用启动、运行与关闭阶段的资源编排。"""

from contextlib import asynccontextmanager
import os
import secrets

from fastapi import FastAPI

from app.adapters.outbound.agent.llm_provider import (
    close_task_llm_cache,
    get_task_llm,
    llm_route_manifest,
    task_profile,
)
from app.infrastructure.clients.embedding_client_manager import embedding_client_manager
from app.infrastructure.clients.es_client_manager import es_client_manager
from app.infrastructure.clients.mysql_client_manager import dw_mysql_client_manager, meta_mysql_client_manager
from app.infrastructure.clients.qdrant_client_manager import qdrant_client_manager
from app.infrastructure.clients.redis_client_manager import redis_client_manager
from app.infrastructure.persistence.mysql.repositories.meta_mysql_repository import MetaMysqlRepository
from app.application.services.adhoc_dataset_service import AdhocDatasetService
from app.infrastructure.config.data_sources import adhoc_dataset_settings
from app.infrastructure.persistence.mysql.adhoc_dataset_adapter import (
    MysqlAdhocDatasetAdapter,
)
from app.config.app_config import app_config
from app.infrastructure.observability.logging import logger
from app.infrastructure.clients.llm.invocation import invoke_llm
from app.infrastructure.search.elasticsearch.repositories.conversation_memory_es_repository import ConversationMemoryESRepository
from app.infrastructure.search.elasticsearch.repositories.conversation_summary_es_repository import ConversationSummaryESRepository
from app.infrastructure.search.qdrant.repositories.conversation_memory_qdrant_repository import (
    ConversationMemoryQdrantRepository,
)
from app.infrastructure.search.qdrant.repositories.conversation_summary_qdrant_repository import (
    ConversationSummaryQdrantRepository,
)
from app.bootstrap.worker import (
    enqueue_summary_projection_backfill,
    memory_projection_worker,
)


async def _probe_external_dependencies() -> None:
    """Run small real requests so startup proves required services are usable.

    Client constructors are lazy for HTTP services and SQLAlchemy engines are lazy for
    MySQL.  A successful constructor is therefore not readiness evidence.  This
    probe verifies ES, Qdrant, Redis, the BGE-M3 runtime contract and every distinct
    LLM route model.  It does not require business indexes/collections to exist:
    those belong to the separate knowledge-build lifecycle.  The DW engine remains
    lazy here; its first real DW SQL normally happens in later knowledge building or
    query execution.  Meta MySQL is exercised by startup maintenance after this probe.
    """
    es_client = es_client_manager.client
    qdrant_client = qdrant_client_manager.client
    embedding_client = embedding_client_manager.client
    if es_client is None or qdrant_client is None or embedding_client is None:
        raise RuntimeError("外部客户端尚未初始化，无法执行启动探活")

    if not await es_client.ping():
        raise RuntimeError("Elasticsearch 启动探活失败")
    await qdrant_client.get_collections()
    redis_ready = await redis_client_manager.ping()
    if not redis_ready and app_config.redis.required_on_startup:
        raise RuntimeError("Redis 启动探活失败")
    if not redis_ready:
        logger.warning(
            "Redis 启动探活失败：短期 working memory 将从 MySQL 重建并降级运行"
        )

    embedding_client.validate_runtime()
    if app_config.embedding.warmup_on_startup:
        probe = await embedding_client.warmup()
        if probe["dense_size"] != app_config.embedding.dense_size or probe["sparse_nonzero"] <= 0:
            raise RuntimeError(f"BGE-M3 启动预热合同异常：{probe}")

    # Routes may select different providers/endpoints/models. Probe every distinct
    # resolved deployment once. Multiple routes can share one deployment but carry
    # very different request budgets
    # (for example candidate=2.5s and sql=18s). Choosing the first alphabetic route
    # made startup readiness depend on the shortest task budget, so a healthy model
    # could randomly prevent the whole application from starting. For each physical
    # model, use the representative route with the widest configured deadline.
    # 同名模型可能由不同供应商或不同地域端点托管，不能只按 model_name
    # 去重，否则会漏掉某个真实外部依赖的启动探活。
    probe_routes: dict[str, tuple[str, dict]] = {}
    for task, profile in llm_route_manifest().items():
        model_name = str(profile["model_name"])
        # deployment_id 覆盖 provider、adapter、endpoint、model、access mode 与
        # credential source；这些原始值不会进入公开清单或日志。
        route_identity = str(profile["deployment_id"])
        timeout_seconds = float(
            profile.get("timeout_seconds")
            or task_profile(task)["timeout_seconds"]
        )
        existing = probe_routes.get(route_identity)
        existing_timeout = (
            float(
                existing[1].get("timeout_seconds")
                or task_profile(existing[0])["timeout_seconds"]
            )
            if existing
            else -1.0
        )
        if existing is None or timeout_seconds > existing_timeout:
            probe_routes[route_identity] = (task, profile)

    for _deployment_id, (task, profile) in probe_routes.items():
        provider = str(profile.get("provider") or profile.get("model_provider") or "")
        model_name = str(profile["model_name"])
        invocation = await invoke_llm(
            get_task_llm(task),
            "Reply with exactly READY.",
            node_id=f"startup_probe_{task}",
            route=task,
            timeout_seconds=float(
                profile.get("timeout_seconds")
                or task_profile(task)["timeout_seconds"]
            ),
            max_retries=0,
            core_path=True,
        )
        response = invocation.value
        content = getattr(response, "content", response)
        if content is None or not str(content).strip():
            raise RuntimeError(
                "LLM 启动探活返回空响应："
                f"task={task}, provider={provider}, model={model_name}"
            )

    logger.info(
        "外部依赖启动探活完成：ES、Qdrant、Redis、BGE-M3 runtime 及 {} 个 LLM 部署",
        len(probe_routes),
    )


async def _close_external_dependencies() -> list[tuple[str, Exception]]:
    """Attempt every close operation, even if an earlier one fails."""
    failures: list[tuple[str, Exception]] = []
    for name, close in (
        ("LLM", close_task_llm_cache),
        ("Qdrant", qdrant_client_manager.close),
        ("Redis", redis_client_manager.close),
        ("Meta MySQL", meta_mysql_client_manager.close),
        ("DW MySQL", dw_mysql_client_manager.close),
        ("Elasticsearch", es_client_manager.close),
        ("Embedding", embedding_client_manager.close),
    ):
        try:
            await close()
        except Exception as exc:
            failures.append((name, exc))
            logger.exception("关闭 {} 失败；将继续关闭其余资源", name)
    return failures


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 应用生命周期管理入口，负责启动期资源初始化与关闭期资源释放。

    核心功能：作为 FastAPI 的 lifespan 上下文管理器，在应用启动前校验鉴权密钥安全性、
    依次初始化各类外部客户端（embedding/ES/MySQL/Qdrant），并基于 meta MySQL 会话
    执行管理员表初始化、评测基线导入及敏感历史清理等运维动作；通过 yield 将控制权
    交还给 FastAPI 运行主循环，最终在 finally 阶段按安全顺序关闭所有需要释放的客户端。

    业务逻辑：
        1. 校验 AUTH_SECRET_KEY：若鉴权开启且密钥仍为默认值或长度不足 32，生产环境直接抛错，开发环境则生成临时随机密钥并告警
        2. 初始化外部客户端：依次创建 embedding、ES、dw/meta MySQL、Qdrant、Redis 管理器资源
        3. 对 ES、Qdrant、Redis、Embedding runtime 与每个不同的 LLM 路由模型执行探活；DW 此时不执行 SQL
        4. 执行运维数据库操作：在 meta MySQL 会话中建管理员表、导入评测基线、清理过期敏感历史
        5. 清理过期临时数据集：调用 AdhocDatasetService 清理已过期的 adhoc 数据集
        6. yield 交还控制权：进入 FastAPI 正常服务运行期
        7. 关闭资源：按安全顺序逐项关闭；一项失败也继续尝试其余资源

    Args:
        app: FastAPI 应用实例，由框架传入，供后续扩展使用
    """
    # =============================================================================
    # 阶段 1：鉴权密钥安全性校验
    # =============================================================================
    # 若鉴权开启且密钥仍为占位默认值或长度不足 32 位，则视为不安全
    if (app_config.auth.enabled and
            (app_config.auth.secret_key == "dev-only-change-me"
             or len(app_config.auth.secret_key) < 32)):
        # 生产环境强制要求显式配置强密钥，否则直接中断启动
        if os.getenv("APP_ENV", "development").lower() == "production":
            raise RuntimeError("生产环境 AUTH_SECRET_KEY 必须设置为至少 32 个字符的随机密钥")
        # 开发环境兜底：生成临时随机密钥，避免使用默认值；重启后旧令牌会失效
        app_config.auth.secret_key = secrets.token_urlsafe(48)
        logger.warning("未配置 AUTH_SECRET_KEY：开发环境已使用临时随机密钥，重启后现有登录令牌会失效")

    lifecycle_failed = False
    try:
        # =============================================================================
        # 阶段 2：外部客户端初始化
        # =============================================================================
        # 依次初始化各基础设施客户端管理器，建立到对应服务的连接/配置
        embedding_client_manager.init()
        es_client_manager.init()
        dw_mysql_client_manager.init()
        meta_mysql_client_manager.init()
        qdrant_client_manager.init()
        redis_client_manager.init()
        await _probe_external_dependencies()

        # =============================================================================
        # 阶段 3：运维数据库操作（基于 meta MySQL 会话）
        # =============================================================================
        # 复用 meta MySQL 会话执行若干一次性运维任务，会话退出时自动归还连接
        async with meta_mysql_client_manager.session_factory() as session:
            # 建表/迁移：确保管理员相关表结构存在
            repo = MetaMysqlRepository(session)
            await repo.ensure_admin_tables()
            # 历史清理：按配置保留天数清理过期的敏感字段脱敏历史
            await repo.cleanup_sensitive_history(app_config.masking.history_retention_days)

        # 长期记忆使用独立 ES alias 与 Qdrant collection；仅建立空 Schema，
        # 不加载 BGE-M3 模型，也不与元知识/字段值索引混用。
        if app_config.memory.enabled and app_config.memory.long_term_enabled:
            await ConversationMemoryESRepository(
                es_client_manager.client
            ).ensure_index()
            await ConversationMemoryQdrantRepository(
                qdrant_client_manager.client
            ).ensure_collection()
        if (
            app_config.memory.enabled
            and app_config.memory.summary_enabled
            and app_config.memory.summary_projection_enabled
        ):
            await ConversationSummaryESRepository(
                es_client_manager.client,
                manage_alias=False,
            ).ensure_index()
            await ConversationSummaryQdrantRepository(
                qdrant_client_manager.client
            ).ensure_collection()
            async with meta_mysql_client_manager.session_factory() as session:
                backfill = await enqueue_summary_projection_backfill(session)
                await session.commit()
            memory_projection_worker.summary_backfill = {
                "status": "queued",
                **backfill,
            }
        if app_config.memory.enabled:
            memory_projection_worker.start()
            memory_projection_worker.notify()

        # =============================================================================
        # 阶段 4：临时数据集过期清理
        # =============================================================================
        # 清理已过期的 adhoc 数据集，回收存储与元数据（system 身份执行）
        await AdhocDatasetService(
            storage=MysqlAdhocDatasetAdapter(),
            owner_id="system",
            workspace_id="system",
            settings=adhoc_dataset_settings(),
        ).cleanup_expired()

        # =============================================================================
        # 阶段 5：交还控制权，进入 FastAPI 服务运行期
        # =============================================================================
        # yield 之前的部分对应"启动期"，yield 之后对应"关闭期"
        yield
    except BaseException:
        # Preserve the original startup/request-lifecycle error.  Cleanup failures
        # are logged below but must not obscure the cause that stopped the app.
        lifecycle_failed = True
        raise
    finally:
        # =============================================================================
        # 阶段 6：按安全顺序释放资源；单项失败不会跳过后续关闭
        # =============================================================================
        await memory_projection_worker.stop()
        failures = await _close_external_dependencies()
        if failures and not lifecycle_failed:
            names = ", ".join(name for name, _ in failures)
            raise RuntimeError(f"应用关闭时资源释放失败：{names}") from failures[0][1]
