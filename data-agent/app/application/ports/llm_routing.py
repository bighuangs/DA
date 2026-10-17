"""LLM 任务路由的应用端口。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol


class LLMRouteCatalog(Protocol):
    """为调用边界提供节点路由与非敏感执行参数。"""

    def route_for_node(self, node_id: str | None) -> str:
        """把节点标识映射为稳定任务路由。"""

    def profile_for_route(self, route: str | None) -> Mapping[str, Any]:
        """返回模型名、超时、重试和并发等执行参数。"""


DEFAULT_LLM_NODE_ROUTES = {
    "classify_query_intent": "keyword",
    "extract_keywords": "keyword",
    "recall_column": "keyword",
    "recall_metric": "keyword",
    "recall_value": "keyword",
    "filter_table": "selection",
    "filter_metric": "selection",
    "plan_sql": "planning",
    "generate_sql": "sql",
    "correct_sql": "correction",
    "judge_candidate": "candidate",
    "generate_chart": "postprocess",
    "generate_analysis": "postprocess",
    "generate_recommend": "recommend",
}


@dataclass(frozen=True, slots=True)
class MissingLLMRouteCatalog:
    """组合根未注入路由时保留节点分类，并显式标记未知模型。"""

    default_model: str = "unregistered"

    def route_for_node(self, node_id: str | None) -> str:
        return DEFAULT_LLM_NODE_ROUTES.get(str(node_id or ""), "sql")

    def profile_for_route(self, route: str | None) -> Mapping[str, Any]:
        return {
            "task": str(route or "sql"),
            "model_name": self.default_model,
            "timeout_seconds": 30.0,
            "max_retries": 0,
            "max_concurrency": 1,
        }


__all__ = [
    "DEFAULT_LLM_NODE_ROUTES",
    "LLMRouteCatalog",
    "MissingLLMRouteCatalog",
]
