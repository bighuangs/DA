from langgraph.runtime import Runtime

from app.agent_system.agents.nl2sql.context import DataAgentContext
from app.agent_system.agents.nl2sql.state import DataAgentState


async def resolve_data_source_context(
    state: DataAgentState, runtime: Runtime[DataAgentContext]
):
    """冻结并公开本轮实际使用的数据源上下文。"""
    runtime.stream_writer({"stage": "解析数据源上下文"})
    context = dict(state.get("data_source_context") or {})
    if not context:
        context = {
            "schema_version": "data-source-context-v1",
            "source_mode": "dw_only",
            "datasets": [],
            "revision": 1,
        }
    runtime.stream_writer({"source_context_resolved": context})
    return {"data_source_context": context}


async def profile_upload_schema(
    state: DataAgentState, runtime: Runtime[DataAgentContext]
):
    """把有界、已脱敏的上传 Schema profile 纳入统一 Trace/SSE。"""
    profiles = list((state.get("data_source_context") or {}).get("profiles") or [])
    if not profiles:
        return {}
    runtime.stream_writer({"stage": "剖析上传 Schema"})
    runtime.stream_writer({
        "upload_schema_profile": {
            "profile_count": len(profiles),
            "profiles": profiles,
            "raw_rows_exposed": False,
        }
    })
    return {"upload_profiles": profiles}


async def propose_source_mapping(
    state: DataAgentState, runtime: Runtime[DataAgentContext]
):
    """公开已确认映射或“不需要映射”的权威状态。"""
    mapping = dict(state.get("source_mapping") or {})
    if not mapping:
        return {}
    runtime.stream_writer({"stage": "核对数据源关联"})
    runtime.stream_writer({"source_mapping": mapping})
    return {"source_mapping": mapping}
