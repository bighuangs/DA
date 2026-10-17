"""会话目录、记忆与摘要用例使用的不可变设置。"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.privacy.masking import MaskingPolicy


@dataclass(frozen=True, slots=True)
class ConversationHistorySettings:
    """会话目录和 transcript 水合边界。"""

    directory_page_size: int = 20
    transcript_page_size: int = 30
    result_preview_rows: int = 100
    result_preview_bytes: int = 200_000
    title_max_chars: int = 80
    retention_days: int = 30
    masking_policy: MaskingPolicy = field(default_factory=MaskingPolicy)


@dataclass(frozen=True, slots=True)
class ConversationMemorySettings:
    """一次会话记忆运行使用的完整配置快照。"""

    enabled: bool = True
    working_ttl_seconds: int = 86_400
    recent_message_limit: int = 20
    recent_token_budget: int = 900
    long_term_enabled: bool = True
    long_term_top_k: int = 5
    es_projection_version: str = "conversation-memory-lexical-v1"
    qdrant_projection_version: str = "conversation-memory-vector-v2"
    extractor_version: str = "memory-rule-extractor-v1"
    context_schema_version: str = "memory-context-v2"
    fusion_rrf_k: int = 60
    retrieval_timeout_seconds: float = 5.0
    message_content_max_chars: int = 2_000
    memory_content_max_chars: int = 1_000
    summary_enabled: bool = False
    summary_build_enabled: bool = False
    summary_projection_enabled: bool = False
    summary_turn_high_watermark: int = 12
    summary_turn_low_watermark: int = 4
    summary_token_high_watermark: int = 1_800
    summary_token_low_watermark: int = 700
    summary_recent_protected_turns: int = 4
    summary_segment_max_messages: int = 16
    summary_segment_max_tokens: int = 1_200
    summary_content_max_chars: int = 2_400
    summary_signal_limit: int = 12
    summary_top_k: int = 3
    summary_hierarchy_group_size: int = 4
    summary_max_level: int = 2
    summary_retrieval_always: bool = False
    summary_extractor_version: str = "summary-extractive-v1"
    summary_signal_extractor_version: str = "summary-signals-rule-v1"
    summary_schema_version: str = "summary-segment-v1"
    summary_es_projection_version: str = "conversation-summary-lexical-v2"
    summary_qdrant_projection_version: str = "conversation-summary-vector-v2"
    stable_fact_enabled: bool = False
    stable_fact_extractor_version: str = "stable-fact-rule-v1"
    embedding_model: str = "BAAI/bge-m3"
    embedding_revision: str = "local"
    masking_policy: MaskingPolicy = field(default_factory=MaskingPolicy)


DEFAULT_HISTORY_SETTINGS = ConversationHistorySettings()
DEFAULT_MEMORY_SETTINGS = ConversationMemorySettings()


__all__ = [
    "ConversationHistorySettings",
    "ConversationMemorySettings",
    "DEFAULT_HISTORY_SETTINGS",
    "DEFAULT_MEMORY_SETTINGS",
]
