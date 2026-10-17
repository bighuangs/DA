from typing import NotRequired, TypedDict


class TrainingInfoQdrant(TypedDict):
    """
    Qdrant 训练样例向量库的 payload 结构。

    id: 样例唯一标识（与 meta 库 training_info.id 对应，转 str 作为 Qdrant point id）
    question: 自然语言问题（同时是向量化文本）
    sql: 目标 SQL（召回后作为 few-shot 示例的"答案"）
    """
    workspace_id: str
    id: str
    question: str
    sql: str
    score: NotRequired[float]
    point_id: NotRequired[str]
    rank: NotRequired[int]
    index_version: NotRequired[str]
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
    dense_rank: NotRequired[int | None]
    dense_score: NotRequired[float | None]
    sparse_rank: NotRequired[int | None]
    sparse_score: NotRequired[float | None]
    fusion_rank: NotRequired[int]
    fusion_score: NotRequired[float]
    fusion_method: NotRequired[str]
    matched_point_roles: NotRequired[list[str]]
    matched_points: NotRequired[list[dict]]
