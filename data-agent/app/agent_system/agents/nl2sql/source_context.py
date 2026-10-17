from collections.abc import Mapping


DW_RETRIEVAL_MODES = {"dw_only", "upload_compare_dw", "upload_join_dw"}


def source_mode(state: Mapping) -> str:
    return str(
        (state.get("data_source_context") or {}).get("source_mode") or "dw_only"
    )


def requires_dw_retrieval(state: Mapping) -> bool:
    return source_mode(state) in DW_RETRIEVAL_MODES


def skipped_retrieval_evidence(route: str) -> dict:
    return {
        "route": route,
        "responsibility": "official_dw_rag",
        "status": "skipped",
        "reason": "upload_only_context",
        "selected_count": 0,
        "global_dw_calls": 0,
        "evidence_complete": True,
    }
