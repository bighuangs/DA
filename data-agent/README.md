# 京析·数语 · 后端

个人电商经营智能问数与分析平台“京析·数语”的后端，基于 Python 3.12+、FastAPI 和 LangGraph，为销售统计、商品排行、区域对比与客户消费分析提供 Text-to-SQL、业务知识检索、会话记忆和结果分析能力。代码包沿用 `data-agent` 命名。完整项目介绍、个人职责、成果、环境、初始化和安全说明见[项目 README](../README.md)。

## 目录职责

- `app/adapters`：HTTP 入口和出站适配。
- `app/application`：业务用例、服务和接口端口。
- `app/domain`：领域合同、语义规则和查询安全。
- `app/agent_system`：NL2SQL 工作流、运行时及记忆能力。
- `app/infrastructure`：模型、MySQL、Redis、Qdrant、Elasticsearch 和观测适配。
- `app/bootstrap`：应用工厂、依赖装配与生命周期。
- `conf`、`prompts`：运行配置、业务知识模板与提示词。
- `migrations`、`tests`、`evals`：迁移、测试及评测代码；不包含完整业务评测资产。

## 常用命令

在本目录执行：

```bash
uv sync --locked
uv run python -m pytest -q tests/test_sql_security.py tests/test_query_intent_policy.py
```

配置 `.env`、准备数据库/检索服务和 BGE-M3 权重后再启动：

```bash
uv run python -m main
```

入口为 `main.py`，默认端口 `8000`，API 文档地址 `/docs`。启动会探测外部模型并执行 Meta 建表等维护操作，可能产生费用和数据写入；请先阅读根 README，不要将未审核配置直接用于生产。

`conf/meta_config.yaml` 为待填充业务模板，运行前需接入自己的数仓和指标知识。空库建表不替代已有库的版本迁移；普通数仓查询应使用只读账号，受管上传需另行审核隔离存储与写权限。

评测 HTTP 默认关闭。完整测试和真实问数评测需要自行准备相应数据集、依赖服务和模型配置，不能由上述定向单测替代。
