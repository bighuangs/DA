"""按需预测与异常说明的 LLM 出站适配器。"""

from __future__ import annotations

import json

import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from app.adapters.outbound.agent.llm_provider import get_task_llm, task_timeout
from app.agent_system.agents.nl2sql.prompts.prompt_loader import load_prompt
from app.application.ports.analysis_model import AnalysisModelPort
from app.domain.query.analysis_statistics import compute_analysis_statistics
from app.infrastructure.clients.llm.invocation import invoke_optional_llm


class LLMAnalysisModel(AnalysisModelPort):
    """通过统一 LLM Invocation 边界提供非核心分析增强。"""

    async def predict(
        self,
        *,
        query: str,
        sql: str,
        chart: dict,
        result: list[dict],
    ) -> str:
        result_columns = list(result[0].keys()) if result else []
        chain = (
            PromptTemplate(
                template=load_prompt("generate_predict"),
                input_variables=[
                    "query",
                    "sql",
                    "chart",
                    "result_columns",
                    "row_count",
                    "result_json",
                ],
            )
            | get_task_llm("postprocess")
            | StrOutputParser()
        )
        invocation = await invoke_optional_llm(
            chain,
            {
                "query": query,
                "sql": sql,
                "chart": yaml.dump(chart, allow_unicode=True, sort_keys=False),
                "result_columns": yaml.dump(
                    result_columns, allow_unicode=True, sort_keys=False
                ),
                "row_count": str(len(result)),
                "result_json": json.dumps(
                    result[:30], ensure_ascii=False, default=str
                ),
            },
            node_id="analysis_predict",
            route="postprocess",
            fallback="",
            timeout_seconds=task_timeout("postprocess"),
            core_path=False,
        )
        return str(invocation.value or "")

    async def anomaly(
        self,
        *,
        query: str,
        sql: str,
        chart: dict,
        result: list[dict],
    ) -> str:
        result_columns = list(result[0].keys()) if result else []
        result_sample = [
            {
                key: str(value)[:120] if value is not None else None
                for key, value in row.items()
            }
            for row in result[:10]
        ]
        chain = (
            PromptTemplate(
                template=load_prompt("generate_anomaly"),
                input_variables=[
                    "query",
                    "sql",
                    "chart",
                    "result_columns",
                    "row_count",
                    "stats",
                    "result_sample",
                ],
            )
            | get_task_llm("postprocess")
            | StrOutputParser()
        )
        invocation = await invoke_optional_llm(
            chain,
            {
                "query": query,
                "sql": sql,
                "chart": yaml.dump(chart, allow_unicode=True, sort_keys=False),
                "result_columns": yaml.dump(
                    result_columns, allow_unicode=True, sort_keys=False
                ),
                "row_count": str(len(result)),
                "stats": yaml.dump(
                    compute_analysis_statistics(result),
                    allow_unicode=True,
                    sort_keys=False,
                ),
                "result_sample": yaml.dump(
                    result_sample, allow_unicode=True, sort_keys=False
                ),
            },
            node_id="analysis_anomaly",
            route="postprocess",
            fallback="",
            timeout_seconds=task_timeout("postprocess"),
            core_path=False,
        )
        return str(invocation.value or "")


__all__ = ["LLMAnalysisModel"]
