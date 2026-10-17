from typing import NotRequired, TypedDict

class ValueInfoES(TypedDict):
    id: str
    workspace_id: str
    value: str
    normalized_value: str
    type: str
    column_id: str
    column_name: str
    table_id: str
    table_name: str
    index_version: str
    indexed_at: str
    is_active: bool
    source_updated_at: NotRequired[str | None]
    responsibility: NotRequired[str]
    score: NotRequired[float]
    point_id: NotRequired[str]
    document_id: NotRequired[str]
    rank: NotRequired[int]
    rerank_score: NotRequired[float]
    match_features: NotRequired[dict]
    retrieval_sources: NotRequired[list[str]]
    retrieval_source: NotRequired[str]
    query_mode: NotRequired[str]
    source_index: NotRequired[str]
    # Privacy-safe evidence copied only from the current user question. Stored
    # values are never included; the hint describes exact/prefix match shape.
    value_match_hints: NotRequired[list[dict]]
