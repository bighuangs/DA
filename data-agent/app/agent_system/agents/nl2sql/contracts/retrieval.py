"""NL2SQL 检索阶段使用的基础设施无关候选 DTO。"""

from __future__ import annotations

from typing import NotRequired, TypedDict


class RetrievalEvidence(TypedDict, total=False):
    """候选在混合检索与重排阶段产生的可选证据。"""

    point_id: str
    rank: int
    score: float
    rerank_score: float
    match_features: dict[str, object]
    retrieval_sources: list[str]
    entity_id: str
    point_role: str
    dense_rank: int | None
    dense_score: float | None
    sparse_rank: int | None
    sparse_score: float | None
    fusion_rank: int
    fusion_score: float
    fusion_method: str
    matched_point_roles: list[str]
    matched_points: list[dict[str, object]]


class ColumnCandidate(RetrievalEvidence):
    """字段检索候选；不暴露 Qdrant Point 或 ORM 类型。"""

    id: str
    name: str
    type: str
    role: str
    description: str
    alias: list[str]
    table_id: str
    examples: NotRequired[list[object]]
    table_name: NotRequired[str]
    is_display: NotRequired[bool]
    index_version: NotRequired[str]
    value_match_hints: NotRequired[list[dict[str, object]]]
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


class MetricCandidate(RetrievalEvidence):
    """指标或术语检索候选。"""

    workspace_id: str
    id: str
    name: str
    description: str
    relevant_columns: list[str]
    alias: list[str]
    formula_rule: NotRequired[dict[str, object] | None]
    index_version: NotRequired[str]
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


class ValueCandidate(RetrievalEvidence):
    """字段值检索候选；只承载当前授权范围内的结果。"""

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
    document_id: NotRequired[str]
    retrieval_source: NotRequired[str]
    query_mode: NotRequired[str]
    source_index: NotRequired[str]
    value_match_hints: NotRequired[list[dict[str, object]]]


__all__ = ["ColumnCandidate", "MetricCandidate", "ValueCandidate"]
