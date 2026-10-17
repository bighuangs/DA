"""NL2SQL 的模型能力端口、调用结果合同与纯输出校验。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol


DEFAULT_TASK = "sql"
NODE_TASK_ROUTES = {
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
MODEL_CLARIFICATION_VERSION = "model-clarification-v1"


@dataclass(frozen=True)
class LLMInvocationResult:
    """一次受治理调用的值与隐私安全证据。"""

    value: Any
    attempts: list[dict[str, Any]]
    degradations: list[dict[str, Any]]


class LLMInvocationError(RuntimeError):
    """不携带供应商响应正文的稳定 Agent 错误。"""

    def __init__(
        self,
        *,
        node_id: str,
        route: str,
        reason_code: str,
        error_type: str,
        timeout_budget_ms: int,
        attempts: list[dict[str, Any]],
        degradations: list[dict[str, Any]],
    ) -> None:
        self.node_id = node_id
        self.route = route
        self.reason_code = reason_code
        self.error_type = error_type
        self.timeout_budget_ms = timeout_budget_ms
        self.attempts = attempts
        self.degradations = degradations
        super().__init__(
            f"{reason_code} node={node_id} route={route} "
            f"budget_ms={timeout_budget_ms} error_type={error_type}"
        )

    @property
    def user_message(self) -> str:
        messages = {
            "llm.timeout": "模型调用超时，请稍后重试或缩小问题范围",
            "llm.deadline_exhausted": "本次查询的模型调用预算已耗尽，请缩小问题范围",
            "llm.queue_timeout": "模型调用排队超时，请稍后重试",
            "llm.cancelled": "模型调用已取消",
            "llm.connection_error": "模型服务暂时无法连接，请稍后重试",
            "llm.rate_limited": "模型服务当前限流，请稍后重试",
            "llm.server_error": "模型服务暂时异常，请稍后重试",
            "llm.authentication_error": "模型服务认证失败，请检查当前模型凭据配置",
            "llm.payment_required": "模型服务计费状态不可用，请检查账户余额或计费配置",
            "llm.permission_denied": "模型服务拒绝访问，请检查当前模型的访问权限",
            "llm.stream_interrupted": "模型响应流中断，请稍后重试",
            "llm.empty_output": "模型未返回有效内容，请重试或补充问题信息",
            "llm.truncated_output": "模型输出被截断，请缩小问题范围",
            "llm.invalid_structured_output": "模型未按 SQL 格式返回内容，请稍后重试",
            "llm.refusal": "模型拒绝了本次请求，请调整问题后重试",
            "llm.provider_error": "模型服务调用失败，请稍后重试",
            "llm.pre_invoke_failure": "模型调用准备失败，请检查当前配置与上下文",
            "llm.unknown_failure": "模型调用发生未知错误",
        }
        message = messages.get(self.reason_code, messages["llm.unknown_failure"])
        return f"{message}（{self.reason_code}，节点 {self.node_id}）"


class Nl2SqlLLMCapability(Protocol):
    """由 outbound adapter 实现、由每次 Graph Invocation 注入的模型能力。"""

    def model(self, task: str) -> Any: ...

    def timeout(self, task: str) -> float: ...

    def route_manifest(self) -> Mapping[str, Mapping[str, Any]]: ...

    async def invoke(self, runnable: Any, payload: Any, **kwargs: Any) -> Any: ...

    async def invoke_optional(
        self, runnable: Any, payload: Any, **kwargs: Any
    ) -> Any: ...

    def pre_invoke_failure(self, **kwargs: Any) -> Any: ...


def route_for_node(node_id: str | None) -> str:
    """把节点 id 翻译为稳定任务路由。"""

    return NODE_TASK_ROUTES.get(str(node_id or ""), DEFAULT_TASK)


def _capability(runtime: Any) -> Nl2SqlLLMCapability:
    context = getattr(runtime, "context", None) or {}
    capability = context.get("llm")
    if capability is None:
        raise RuntimeError("NL2SQL Execution 缺少 llm capability")
    return capability


def get_task_llm(runtime: Any, task: str = DEFAULT_TASK) -> Any:
    """从当前 Invocation 的能力上下文取得已装配模型。"""

    return _capability(runtime).model(task)


def task_timeout(runtime: Any, task: str) -> float:
    return float(_capability(runtime).timeout(task))


def llm_route_manifest(runtime: Any) -> dict[str, dict[str, Any]]:
    return {
        str(route): dict(profile)
        for route, profile in _capability(runtime).route_manifest().items()
    }


def _result(value: Any) -> LLMInvocationResult:
    if isinstance(value, LLMInvocationResult):
        return value
    return LLMInvocationResult(
        value=getattr(value, "value", None),
        attempts=list(getattr(value, "attempts", ()) or ()),
        degradations=list(getattr(value, "degradations", ()) or ()),
    )


def _error(value: BaseException) -> LLMInvocationError:
    if isinstance(value, LLMInvocationError):
        return value
    required = (
        "node_id",
        "route",
        "reason_code",
        "error_type",
        "timeout_budget_ms",
        "attempts",
        "degradations",
    )
    if not all(hasattr(value, name) for name in required):
        raise value
    return LLMInvocationError(
        node_id=str(getattr(value, "node_id")),
        route=str(getattr(value, "route")),
        reason_code=str(getattr(value, "reason_code")),
        error_type=str(getattr(value, "error_type")),
        timeout_budget_ms=int(getattr(value, "timeout_budget_ms")),
        attempts=list(getattr(value, "attempts") or ()),
        degradations=list(getattr(value, "degradations") or ()),
    )


async def invoke_llm(
    runnable: Any,
    payload: Any,
    *,
    runtime: Any,
    **kwargs: Any,
) -> LLMInvocationResult:
    """经注入能力执行核心模型调用并稳定化错误合同。"""

    try:
        return _result(
            await _capability(runtime).invoke(
                runnable,
                payload,
                runtime=runtime,
                **kwargs,
            )
        )
    except BaseException as exc:
        translated = _error(exc)
        if translated is exc:
            raise
        raise translated from exc


async def invoke_optional_llm(
    runnable: Any,
    payload: Any,
    *,
    runtime: Any,
    **kwargs: Any,
) -> LLMInvocationResult:
    """经注入能力执行可降级模型调用。"""

    try:
        return _result(
            await _capability(runtime).invoke_optional(
                runnable,
                payload,
                runtime=runtime,
                **kwargs,
            )
        )
    except BaseException as exc:
        translated = _error(exc)
        if translated is exc:
            raise
        raise translated from exc


def pre_invoke_failure(*, runtime: Any, **kwargs: Any) -> LLMInvocationError:
    """在物理调用开始前生成与统一边界一致的失败证据。"""

    result = _capability(runtime).pre_invoke_failure(runtime=runtime, **kwargs)
    return _error(result)


def _content_blocks_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, (list, tuple)):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "").lower()
        if block_type and block_type not in {"text", "output_text"}:
            continue
        text = block.get("text")
        if isinstance(text, dict):
            text = text.get("value")
        if text is None:
            text = block.get("content")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def model_output_text(value: Any) -> str:
    """只提取供应商最终 answer channel，不回退到 reasoning。"""

    if isinstance(value, str):
        return value
    content = getattr(value, "content", None)
    if content is not None:
        return _content_blocks_text(content)
    if isinstance(value, dict):
        if "content" in value:
            return _content_blocks_text(value.get("content"))
        choices = value.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0] if isinstance(choices[0], dict) else {}
            message = first.get("message") if isinstance(first, dict) else {}
            if isinstance(message, dict):
                return _content_blocks_text(message.get("content"))
            return _content_blocks_text(first.get("text"))
    return str(value or "")


def normalize_sql_output(value: Any) -> str:
    """移除完整 SQL Markdown fence，不接受额外说明文字。"""

    text = model_output_text(value).strip()
    if not (text.startswith("```") and text.endswith("```")):
        return text
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip().lower() not in {
        "```",
        "```sql",
        "```mysql",
    }:
        return text
    return "\n".join(lines[1:-1]).strip()


def sql_output_validator(value: Any) -> str | None:
    """在语义校验前识别拒答、截断与显然非 SQL 输出。"""

    text = model_output_text(value).strip()
    lowered = text.lower()
    if lowered.startswith((
        "i can't",
        "i cannot",
        "i’m unable",
        "i am unable",
        "抱歉，我不能",
        "无法帮助",
    )):
        return "llm.refusal"
    if text.startswith("```") and not text.endswith("```"):
        return "llm.truncated_output"
    candidate = normalize_sql_output(text)
    if not candidate.lower().startswith(("select ", "select\n", "with ")):
        return "llm.invalid_structured_output"
    if candidate.count("(") > candidate.count(")"):
        return "llm.truncated_output"
    return None


def build_model_clarification(
    *,
    node_id: str,
    reason_code: str,
    semantic_ir: Mapping[str, Any] | None = None,
    retryable: bool,
) -> dict[str, Any]:
    """生成不复制 query/SQL 的可执行澄清响应。"""

    coverage = dict((semantic_ir or {}).get("semantic_coverage") or {})
    missing = [
        str(item)
        for item in (coverage.get("unresolved_slots") or [])
        if str(item)
    ][:12]
    suggestions = [
        "明确要统计的指标及聚合方式",
        "明确分组维度、时间范围和筛选条件",
    ]
    if any("operator:" in item for item in missing):
        suggestions.append("把比较、集合、窗口或排序要求单独写明")
    return {
        "schema_version": MODEL_CLARIFICATION_VERSION,
        "node_id": str(node_id),
        "reason_code": str(reason_code),
        "action": "clarify_then_retry",
        "retryable": bool(retryable),
        "message": (
            "当前未能在安全时限内完成完整 SQL。请补充或缩小条件后重试，"
            "系统不会执行丢失语义槽位的简化查询。"
        ),
        "unresolved_slots": missing,
        "suggestions": suggestions,
        "contains_query_or_sql": False,
    }


def clarification_message(value: Mapping[str, Any] | None) -> str:
    item = dict(value or {})
    return str(
        item.get("message")
        or "当前查询需要补充完整的指标、维度、时间或筛选条件。"
    )


__all__ = [
    "DEFAULT_TASK",
    "LLMInvocationError",
    "LLMInvocationResult",
    "MODEL_CLARIFICATION_VERSION",
    "NODE_TASK_ROUTES",
    "Nl2SqlLLMCapability",
    "build_model_clarification",
    "clarification_message",
    "get_task_llm",
    "invoke_llm",
    "invoke_optional_llm",
    "llm_route_manifest",
    "model_output_text",
    "normalize_sql_output",
    "pre_invoke_failure",
    "route_for_node",
    "sql_output_validator",
    "task_timeout",
]
