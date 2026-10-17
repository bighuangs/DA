from typing import NotRequired, TypedDict

# 定义qdrant中保存的字段信息的模型
class ColumnInfoQdrant(TypedDict):
    id: str
    name: str
    type:str
    role: str
    # 真实字段值只允许进入 ES；旧 Point 可能仍携带 examples，读取时仅作兼容。
    examples: NotRequired[list]
    description:str
    alias: list
    table_id: str
    table_name: NotRequired[str]
    is_display: NotRequired[bool]
    point_id: NotRequired[str]
    rank: NotRequired[int]
    index_version: NotRequired[str]
    score: NotRequired[float]
    rerank_score: NotRequired[float]
    match_features: NotRequired[dict]
    retrieval_sources: NotRequired[list[str]]
    entity_id: NotRequired[str]
    point_role: NotRequired[str]
    embedding_model: NotRequired[str]
    embedding_revision: NotRequired[str]
    embedding_mode: NotRequired[str]
    source_text_hash: NotRequired[str]
    entity_type: NotRequired[str]
    business_key: NotRequired[str]
    point_strategy: NotRequired[str]
    point_identity_version: NotRequired[str]
    canonical_template_version: NotRequired[str]
    alias_policy_version: NotRequired[str]
    alias_manifest_hash: NotRequired[str]
    projection_hash: NotRequired[str]
    projected_alias: NotRequired[str | None]
    value_match_hints: NotRequired[list[dict]]
    dense_rank: NotRequired[int | None]
    dense_score: NotRequired[float | None]
    sparse_rank: NotRequired[int | None]
    sparse_score: NotRequired[float | None]
    fusion_rank: NotRequired[int]
    fusion_score: NotRequired[float]
    fusion_method: NotRequired[str]
    matched_point_roles: NotRequired[list[str]]
    matched_points: NotRequired[list[dict]]
