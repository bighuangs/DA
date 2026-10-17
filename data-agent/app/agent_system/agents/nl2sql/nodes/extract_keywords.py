import logging

from langgraph.runtime import Runtime

from app.agent_system.agents.nl2sql.context import DataAgentContext, serialized_meta_call
from app.agent_system.agents.nl2sql.state import DataAgentState
from app.agent_system.agents.nl2sql.services.query_signals import extract_query_signals
from app.agent_system.agents.nl2sql.contracts.query_signals import stable_signal_hash
from app.agent_system.agents.nl2sql.source_context import requires_dw_retrieval

logger = logging.getLogger(__name__)


def _append_term(target: list[str], value: object) -> None:
    text = str(value or "").strip()
    if text and text not in target:
        target.append(text)


async def _authorized_workspace_terms(
    runtime: Runtime[DataAgentContext],
) -> tuple[list[str], str]:
    """Build a request-local dictionary from this workspace's authorized Meta rows."""

    context = runtime.context
    policy = context.get("access_policy")
    terms: list[str] = []
    table_rows = await serialized_meta_call(
        context, context["meta_mysql_repo"].list_table_infos
    )
    for table, columns in table_rows:
        table_id = str(getattr(table, "id", "") or getattr(table, "name", ""))
        table_name = str(getattr(table, "name", "") or table_id)
        if policy and not policy.is_table_allowed(table_id):
            continue
        _append_term(terms, table_name)
        for column in columns:
            column_name = str(getattr(column, "name", "") or "")
            if policy and not policy.is_column_allowed(table_id, column_name):
                continue
            _append_term(terms, column_name)
            for alias in getattr(column, "alias", None) or []:
                _append_term(terms, alias)

    metric_rows = await serialized_meta_call(
        context, context["meta_mysql_repo"].get_all_metric_infos
    )
    for metric in metric_rows:
        relevant = list(getattr(metric, "relevant_columns", None) or [])
        if policy and not all(
            policy.is_column_allowed(*policy._split_column_id(column_id))
            for column_id in relevant
        ):
            continue
        _append_term(terms, getattr(metric, "name", ""))
        for alias in getattr(metric, "alias", None) or []:
            _append_term(terms, alias)

    workspace_id = str(
        context.get("workspace_id")
        or getattr(policy, "workspace_id", "")
        or getattr(context["meta_mysql_repo"], "workspace_id", "")
    )
    version = stable_signal_hash({
        "contract": "authorized-workspace-dictionary-v1",
        "workspace_id": workspace_id,
        "policy": getattr(policy, "role", ""),
        "terms": sorted(terms),
    })
    return terms, version


async def extract_keywords(
    state: DataAgentState,
    runtime: Runtime[DataAgentContext],
):
    settings = runtime.context["settings"]
    """Extract the current typed, versioned query-signal contract."""
    runtime.stream_writer({"stage": "提取关键字"})
    query = state["query"]
    context = runtime.context
    policy = context.get("access_policy")
    workspace_id = str(
        context.get("workspace_id")
        or getattr(policy, "workspace_id", "")
        or getattr(context["meta_mysql_repo"], "workspace_id", "")
    )
    profile = settings.query_signals.profile
    workspace_terms: list[str] = []
    dictionary_version: str | None = None
    degradations: list[dict] = []
    if (
        profile == "domain_signals_v1"
        and settings.query_signals.workspace_dictionary_enabled
    ):
        try:
            if requires_dw_retrieval(state):
                workspace_terms, dictionary_version = await _authorized_workspace_terms(
                    runtime
                )
            else:
                workspace_terms = [
                    str(value)
                    for profile_item in (
                        (state.get("data_source_context") or {}).get("profiles")
                        or []
                    )
                    for column in profile_item.get("columns") or []
                    for value in (
                        column.get("display_name"),
                        column.get("name"),
                    )
                    if value
                ]
                dictionary_version = stable_signal_hash({
                    "contract": "upload-schema-dictionary-v1",
                    "workspace_id": workspace_id,
                    "schema_fingerprints": [
                        item.get("schema_fingerprint")
                        for item in (
                            (state.get("data_source_context") or {}).get(
                                "profiles"
                            )
                            or []
                        )
                    ],
                })
        except Exception as exc:
            # A broken dictionary must not remove the full question. K2 degrades
            # to the approved K1 extractor without borrowing another tenant's data.
            profile = "jieba_tfidf_v1"
            degradation = {
                "dependency": "query_signal_dictionary",
                "status": "fallback",
                "fallback": profile,
                "error_type": type(exc).__name__,
                "raw_text_logged": False,
            }
            degradations.append(degradation)
            runtime.stream_writer({"degradation": degradation})
            logger.warning(
                "查询信号工作空间词典加载失败，回退 profile=%s error_type=%s",
                profile,
                type(exc).__name__,
            )

    signals = extract_query_signals(
        query,
        profile=profile,
        workspace_id=workspace_id,
        workspace_terms=workspace_terms,
        dictionary_version=dictionary_version,
        lexical_top_k=settings.query_signals.lexical_top_k,
    )
    summary = signals.privacy_summary()
    logger.info(
        "查询信号提取完成 profile=%s signal_count=%s fingerprint=%s",
        profile,
        sum(summary["counts"].values()),
        signals.signal_fingerprint[:12],
    )
    return {
        "query_signals": signals.to_state(),
        "degradations": degradations,
    }
