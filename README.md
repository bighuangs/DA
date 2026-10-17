# 掌柜问数 · Data Agent

掌柜问数是一个面向结构化数据与经营指标分析的智能问数系统，主要解决业务人员查数依赖手工编写 SQL 的问题。用户可以用自然语言提问，系统检索相关表、字段、指标口径与 SQL 样例，生成并校验 SQL，执行查询，再返回结果、图表和分析。

项目以 **Text2SQL + RAG + LangGraph** 为核心，同时提供元数据管理、工作空间权限、会话记忆、文件问数、反馈与评测代码。本仓库发布前后端实现和必要配置模板，不携带真实业务数据、模型权重、运行凭据或历史评测结果；接入自己的数据源和指标知识后才能完成实际问数。

## 核心能力

- **受控 SQL 查询**：意图拦截、追问补全、语义规划、SQL AST/权限/语义校验、执行计划检查和有界纠错。
- **四路知识检索**：分别召回字段、指标、字段值和 SQL 样例；结合关系闭包、Semantic IR 与上下文预算组织生成依据。
- **元数据与指标维护**：Schema 差异预览和同步、指标/样例管理、检索索引构建、字段值索引治理。
- **多轮交互**：会话历史、追问、MySQL 权威记忆记录、Redis 热状态，以及 Elasticsearch/Qdrant 派生记忆索引。
- **结果呈现**：流式进度、SQL 与结果表、图表、分析、推荐问题、仪表盘和导出。
- **权限与治理**：用户及工作空间、角色权限、数据访问策略、脱敏、Trace 与反馈。
- **受管文件问数**：CSV/XLSX 解析、数据集版本和生命周期管理；启用前需要单独规划上传数据的存储与权限。

这些是代码包含的能力范围，不代表所有部署环境、业务口径或安全场景均已完成生产验收。

## 查询流程

```text
用户问题与会话上下文
          ↓
意图安全检查 → 追问补全 → 查询意图分类
          ↓
解析数据源与当前访问范围
          ↓
提取查询信号 → 字段 / 指标 / 字段值 / SQL 样例召回
          ↓
结构化合并与关系闭包 → Semantic IR → 上下文筛选和预算
          ↓
按需规划 → 生成 SQL → 校验与候选复核
          ↓
执行查询（可修复失败进入有界纠错；不可修复失败终止）
          ↓
返回结果 → 并行生成图表、分析和推荐问题
```

权限检查、执行限制与错误终止由程序控制。检索内容、历史记忆和模型输出不是授权依据；SQL 可执行也不等于业务口径一定正确。

## 代码结构

```text
.
├── data-agent/                 # Python 后端
│   ├── app/
│   │   ├── adapters/           # HTTP 等入站接口与出站适配
│   │   ├── application/        # 用例、服务及端口
│   │   ├── domain/             # 业务规则、查询安全与领域合同
│   │   ├── agent_system/       # NL2SQL Graph、运行时与记忆能力
│   │   ├── infrastructure/     # 数据库、检索、模型及可观测适配
│   │   ├── bootstrap/          # 应用装配与生命周期
│   │   └── config/             # 配置加载与类型定义
│   ├── conf/                  # 运行配置与空白业务知识模板
│   ├── prompts/               # 运行时提示词
│   ├── migrations/            # 需要按现有数据库状态审核的迁移 SQL
│   ├── tests/                 # 单元与集成测试代码
│   ├── evals/                 # 评测代码，不包含完整评测数据和历史制品
│   └── main.py                # 后端启动入口
└── date-agent-frontend/        # Vue 前端，沿用当前目录名
    ├── src/                   # 页面、组件与交互逻辑
    ├── public/                # 静态资源
    ├── e2e/                   # Mock E2E
    ├── e2e-real/              # 真实后端 E2E
    └── scripts/               # 前端测试启动脚本
```

组件说明：[后端](data-agent/README.md) · [前端](date-agent-frontend/README.md)

## 环境要求

| 依赖 | 用途与要求 |
| --- | --- |
| Python 3.12+、uv | 后端；以 `pyproject.toml` 和 `uv.lock` 为准 |
| Node.js 22.12+、npm | 前端；当前 Vite 锁定版本要求 Node.js `^20.19.0` 或 `>=22.12.0` |
| MySQL 8 | Meta 应用数据库与业务数据仓库，可按实际部署隔离实例和账号 |
| Qdrant 1.16 系列 | 字段、指标、样例和记忆的 Dense/Sparse 检索 |
| Elasticsearch 8 | 字段值和记忆检索；默认字段值分析器需要匹配版本的 IK 插件 |
| Redis | 可重建的会话热状态；是否强制就绪由配置决定 |
| BGE-M3 模型文件 | 后端通过 FlagEmbedding 在进程内加载，权重由部署方单独准备 |
| 可用的 LLM 服务 | SQL 生成、纠错、分析等；模型名、端点和凭据需自行配置 |

后端使用 FastAPI、LangGraph、SQLAlchemy、sqlglot；前端使用 Vue 3、Vite、ECharts。具体版本以锁文件为准。当前仓库不提供一键基础设施部署，也不包含数据库备份或模型下载包。

## 安装与配置

### 1. 安装依赖

```bash
git clone https://github.com/bighuangs/DA.git
cd DA/data-agent
uv sync --locked
```

如需 Phoenix/OpenTelemetry 等可选观测集成，再安装：

```bash
uv sync --locked --extra observability
```

### 2. 配置后端

首次部署时，将 `data-agent/.env.example` 复制为同目录的 `.env`，填写自己的配置。不要覆盖已有环境文件，不要将 `.env` 提交到 Git。

| 配置 | 说明 |
| --- | --- |
| `DB_META_HOST/PORT/USER/DATABASE/PASSWORD` | Meta 数据库连接；默认库名 `da_meta`、账号名 `da_meta_app` |
| `DB_DW_HOST/PORT/USER/DATABASE/PASSWORD` | 数据仓库连接；默认库名 `da_warehouse`、账号名 `da_reader` |
| `LLM_PROVIDER`、`LLM_MODEL_NAME` | 供应商适配与实际可用模型名；模板占位模型不能直接调用 |
| `LLM_BASE_URL`、`LLM_API_KEY` | 模型服务端点与凭据；OpenAI-compatible 服务需填写匹配的端点 |
| `AUTH_SECRET_KEY` | 至少 32 个字符的高强度随机认证密钥 |
| `MASKING_HASH_SALT` | 独立的随机脱敏盐，不应与认证密钥共用 |
| `BGE_M3_PATH`、`BGE_M3_DEVICE` | 本地模型目录与设备，例如 `cpu` 或可用 CUDA 设备 |
| `APP_ENV`、`CORS_ALLOWED_ORIGINS` | 环境标识和明确允许访问的前端来源 |

Elasticsearch、Qdrant、Redis 以及模型任务路由等其余选项以 `.env.example` 和 `conf/app_config.yaml` 为准。系统环境变量优先于本地 `.env`。

`conf/meta_config.yaml` 和别名配置是空白业务模板。需要根据自己的物理 Schema、指标定义、字段角色、别名和表关系补充知识，不应直接使用未知来源或未经审核的 SQL 样例。

### 3. 准备数据库、检索服务与模型

1. 由数据库管理员创建 Meta 库、业务仓库和专用账号。普通问数的数据仓库账号应只读；Meta 账号需要应用持久化权限，首次建表还需要相应 DDL 权限。
2. 准备 Elasticsearch、Qdrant 和 Redis。Elasticsearch 默认使用 `ik_max_word`，须安装与服务版本相容的 IK 插件；仅启动原生 ES 并不满足该配置。
3. 将 BGE-M3 权重放在 `BGE_M3_PATH` 指定的位置。当前实现由后端进程加载模型，无需另起旧式 TEI Embedding 容器。
4. 准备有效的模型端点、模型名称和凭据；根据部署环境评估模型显存、CPU 内存、调用时延与成本。

新建空 Meta 库的表会在应用启动时通过当前 ORM 元数据创建。**这不是完整的自动迁移系统**：已有数据库的列变更和索引变更需要先备份、检查迁移 SQL，再按实际版本执行。

上传文件功能会创建和维护受管数据表。只读数仓账号不具备这些权限；如需启用，必须先审核当前存储适配路径，规划隔离存储、专用权限与配额，不能直接向生产数仓只读账号追加广泛写权限。

### 4. 启动前后端并初始化业务知识

后端，在 `data-agent` 目录执行：

```bash
uv run python -m main
```

默认 API 地址为 `http://127.0.0.1:8000`，API 文档为 `/docs`。入口默认监听 `0.0.0.0:8000`；请用防火墙或反向代理限制暴露范围。

**启动会产生真实外部访问和数据库写入**：应用检查依赖、创建缺失 Meta 表、执行生命周期维护，并对每个不同的 LLM 部署发送小型探活请求，可能产生 API 费用。不要在未确认连接目标时直接指向生产数据库或外部模型服务。

另开终端启动前端：

```bash
cd DA/date-agent-frontend
npm ci
npm run dev
```

打开 `http://127.0.0.1:5173`。Vite 默认将 `/api` 代理到 `http://localhost:8000`；需要其他地址时设置进程环境变量 `VITE_API_PROXY_TARGET`。

首次使用建议按顺序完成：

1. 注册并确认当前工作空间。
2. 在元数据管理中预览数仓 Schema 差异，再执行同步。
3. 补充指标口径、字段语义、表关系、访问策略及审核通过的 SQL 样例。
4. 构建相关检索索引，确认字段值治理与敏感字段策略。
5. 用已知结果的简单查询核对 SQL、权限与统计结果，再扩大问题范围。

索引重建、Schema 同步和样例维护会写入 Meta 或检索服务，应在正确的工作空间和授权范围内操作。

## 验证与评测

发布前在本地已有 Python 依赖环境中完成了 565 个 Python 文件的语法解析、发布版配置与 8 条模型路由的加载，以及下列定向测试：68 项测试和 18 个子测试通过；另有 6 项评测脚本测试通过，共 74 项测试。前端使用锁文件安装依赖后构建通过；图表相关产物仍有大于 500 KB 的体积警告。未连接真实数据库、未启动业务服务、未调用模型，也未执行全量集成或业务准确率评测。这些结果不代表其他部署环境已通过验收。

在 `data-agent` 中运行不依赖完整业务评测集的定向规则测试：

```bash
uv run python -m pytest -q tests/test_sql_security.py tests/test_access_policy.py tests/test_query_guard.py tests/test_relation_graph.py tests/test_prompt_context.py tests/test_semantic_ir.py
```

在 `date-agent-frontend` 目录执行前端构建：

```bash
npm ci
npm run build
```

如需浏览器回归：

```bash
npx playwright install chromium
npm run test:e2e:mock
```

`npm run test:e2e:real` 需要真实后端和对应测试环境，不应默认指向生产服务。Mock 测试只验证页面交互合同，不能证明真实 SQL、检索和模型质量。

仓库保留评测代码，但不发布完整业务数据集与历史报告。部分测试和评测入口依赖额外 fixture、数据集、基础设施或模型服务，不能直接把全量测试作为零配置的可运行承诺。评测 HTTP 接口默认关闭；启用 `EVALUATION_HTTP_ENABLED` 前应准备相应资产和权限。

业务验收应同时检查表字段召回、指标口径、SQL 执行结果、拒答/澄清、安全隔离、时延和成本；RAG 分数或 SQL 语法通过率都不能单独替代结果正确率。

## 部署与安全边界

- 将数据仓库账号保持为最小权限只读，并在真实数据库边界落实权限；应用中的检查不能替代数据库授权。
- Meta、Redis、检索索引、日志和 Trace 也可能包含敏感信息，需实施认证、网络隔离、保留期限、备份和删除策略。
- 外部模型可能接收问题、受控 Schema、样例和结果上下文。生产接入前需确认数据分类、脱敏、供应商协议与部署地域，必要时使用符合要求的私有推理服务。
- Redis 与检索索引是可重建的派生状态，不能替代 MySQL 权威记录的备份与恢复。
- 本地缓存、限流和容量控制不自动等同于多实例全局控制；多副本部署需要额外验证一致性、并发与恢复行为。
- `npm run dev` 是开发服务器。生产前端需要构建后托管，并配置 HTTPS、API 反向代理和访问控制。
- 会话摘要、稳定事实提取等能力存在独立开关，代码存在不等于默认开启；应按迁移、数据边界和实际验证结果启用。
- 不承诺自然语言到 SQL 的绝对正确性。重要经营和财务结果应与权威报表、指标口径及人工复核建立对账机制。

本仓库未指定开源许可证；公开可见不等于授予任意使用、再分发或商业使用许可。第三方依赖及模型仍适用各自的许可证和使用条款。
