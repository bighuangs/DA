"""NL2SQL Graph 所需具体模型与索引版本能力适配器。"""

from __future__ import annotations

from typing import Any

from app.agent_system.agents.nl2sql.settings import (
    FrozenSettingsSection,
    Nl2SqlSettings,
    SeedMetricSettings,
)
from app.config.app_config import app_config
from app.config.meta_config import meta_config

from app.adapters.outbound.agent.llm_provider import (
    get_task_llm,
    llm_route_manifest,
    task_timeout,
)
from app.infrastructure.clients.llm.invocation import (
    invoke_llm,
    invoke_optional_llm,
    pre_invoke_failure,
)
from app.infrastructure.search.index_version import (
    column_index_version,
    retrieval_index_version,
)


class Nl2SqlLLMAdapter:
    """把供应商模型和统一调用治理实现映射为 Agent capability。"""

    def model(self, task: str) -> Any:
        return get_task_llm(task)

    def timeout(self, task: str) -> float:
        return task_timeout(task)

    def route_manifest(self) -> dict[str, dict[str, Any]]:
        return llm_route_manifest()

    async def invoke(self, runnable: Any, payload: Any, **kwargs: Any) -> Any:
        return await invoke_llm(runnable, payload, **kwargs)

    async def invoke_optional(
        self,
        runnable: Any,
        payload: Any,
        **kwargs: Any,
    ) -> Any:
        return await invoke_optional_llm(runnable, payload, **kwargs)

    def pre_invoke_failure(self, **kwargs: Any) -> Any:
        return pre_invoke_failure(**kwargs)


class Nl2SqlIndexVersionAdapter:
    """把物化索引版本实现映射为 Agent 只读 capability。"""

    def column_version(self, columns) -> str:
        return column_index_version(columns)

    def retrieval_version(self) -> str:
        return retrieval_index_version()


def build_nl2sql_settings() -> Nl2SqlSettings:
    """在组合边界冻结 Agent 所需配置，排除凭据与连接信息。"""

    llm_values = {
        "routes": {
            str(route): {
                "timeout_seconds": float(
                    (profile or {}).get(
                        "timeout_seconds",
                        app_config.llm.timeout_seconds,
                    )
                )
            }
            for route, profile in (app_config.llm.routes or {}).items()
        },
        "timeout_seconds": float(app_config.llm.timeout_seconds),
    }
    embedding_values = {
        name: getattr(app_config.embedding, name)
        for name in (
            "model",
            "revision",
            "dense_size",
            "max_length",
            "normalize_dense",
        )
    }
    seed_metrics = tuple(
        SeedMetricSettings.create(
            name=item.name,
            description=item.description,
            relevant_columns=item.relevant_columns,
            alias=item.alias,
            formula_rule=item.formula_rule,
        )
        for item in meta_config.metrics
    )
    return Nl2SqlSettings(
        query=FrozenSettingsSection.from_mapping(vars(app_config.query)),
        query_signals=FrozenSettingsSection.from_mapping(vars(app_config.query_signals)),
        embedding=FrozenSettingsSection.from_mapping(embedding_values),
        memory=FrozenSettingsSection.from_mapping({
            "context_token_budget": app_config.memory.context_token_budget,
        }),
        planning_route=FrozenSettingsSection.from_mapping(vars(app_config.planning_route)),
        semantic_repair=FrozenSettingsSection.from_mapping(vars(app_config.semantic_repair)),
        llm=FrozenSettingsSection.from_mapping(llm_values),
        seed_metrics=seed_metrics,
    )


__all__ = [
    "Nl2SqlIndexVersionAdapter",
    "Nl2SqlLLMAdapter",
    "build_nl2sql_settings",
]
