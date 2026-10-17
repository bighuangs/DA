from typing import Optional

from sqlalchemy.engine import URL

from app.config.app_config import DBConfig, app_config
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, async_sessionmaker



class MysqlClientManager:
    """
    MySQL 数据库异步客户端管理器，统一封装引擎与会话工厂的创建与生命周期管理。

    核心功能：基于 SQLAlchemy 的 asyncmy 异步驱动构建 AsyncEngine，并通过
             async_sessionmaker 生成可复用的 AsyncSession 工厂，供上层业务
             （如 dw 数据仓库、meta 元数据库）进行 ORM 与原生 SQL 操作。

    业务逻辑：
        1. 在 __init__ 中保存 DBConfig 配置，并将引擎与会话工厂初始化为 None
        2. _get_url 按 asyncmy 驱动格式拼装 MySQL 异步连接字符串
        3. init 创建带连接池的 AsyncEngine，并基于引擎创建 async_sessionmaker
        4. close 释放引擎占用的所有连接池资源
        5. 模块级单例 dw / meta 两个管理器分别对应不同业务库
    """

    def __init__(self, config: DBConfig):
        """
        初始化 MySQL 客户端管理器，仅保存配置，不建立连接。

        Args:
            config: DBConfig 配置对象，包含 host/port/user/password/database 等连接信息
        """
        self.config = config
        # 引擎与会话工厂延迟到 init() 中创建，便于在事件循环启动后再初始化异步资源
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker] = None

    def _get_url(self):
        """
        拼装 MySQL 异步连接 URL。

        Returns:
            str: 形如 mysql+asyncmy://user:pwd@host:port/db?charset=utf8mb4 的连接字符串，
                 使用 asyncmy 异步驱动并指定 utf8mb4 字符集以支持完整 Unicode（含 emoji）
        """
        return URL.create(
            "mysql+asyncmy",
            username=self.config.user,
            password=self.config.password,
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            query={"charset": "utf8mb4"},
        )

    def init(self):
        """
        创建异步引擎与会话工厂，完成连接池与 ORM 会话基础设施的初始化。

        核心功能：通过 create_async_engine 建立带连接池的 AsyncEngine，再用
                 async_sessionmaker 包装引擎生成会话工厂，统一控制事务、刷新与
                 提交后对象过期等行为。

        业务逻辑：
            1. 调用 _get_url 获取连接字符串
            2. 创建 AsyncEngine，配置连接池大小、溢出上限与连接健康检查
            3. 创建 async_sessionmaker，配置自动开启事务、提交后对象不过期、查询前自动刷新
        """
        # =============================================================================
        # 阶段 1：创建异步引擎（带连接池）
        # =============================================================================
        self.engine = create_async_engine(
            self._get_url(),
            pool_size=10,          # 连接池大小：初始创建的常驻连接数，用完归还；超过则创建临时连接，用完即断
            max_overflow=15,       # 最大临时连接数：超过该上限则等待，超时内有连接归还可正常使用，否则报错
            pool_pre_ping=True,    # 取连接前做健康检查，连接失效时自动重建，避免数据库操作意外失败
        )

        # =============================================================================
        # 阶段 2：创建会话工厂（统一会话行为配置）
        # =============================================================================
        self.session_factory = async_sessionmaker(
            self.engine,
            autobegin=True,           # 默认 True，自动开启事务但不会自动提交，需显式 commit
            expire_on_commit=False,   # 提交后 ORM 对象不过期，仍可访问其属性（避免二次查询数据库）
            autoflush=True,           # 查询前自动将未提交的改动 flush 到暂存区，保证同事务内可立即查到新数据
        )

    async def close(self):
        """
        关闭并释放引擎持有的所有连接池资源。

        业务逻辑：
            1. 调用 engine.dispose() 关闭连接池中的全部连接
            2. 等待异步释放完成后返回
        """
        if self.engine is not None:
            await self.engine.dispose()
        self.engine = None
        self.session_factory = None


# 创建操作 dw（数据仓库）数据库的客户端管理器单例
dw_mysql_client_manager = MysqlClientManager(app_config.db_dw)

# 创建操作 meta（元数据）数据库的客户端管理器单例
meta_mysql_client_manager = MysqlClientManager(app_config.db_meta)
