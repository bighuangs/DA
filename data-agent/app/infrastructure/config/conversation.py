"""会话 Application settings 的配置适配器。"""

from __future__ import annotations

import os

from app.application.settings.conversation import (
    ConversationHistorySettings,
    ConversationMemorySettings,
)
from app.config.app_config import app_config
from app.domain.privacy.masking import MaskingPolicy


def load_conversation_settings() -> tuple[
    ConversationHistorySettings,
    ConversationMemorySettings,
]:
    """冻结当前进程配置，避免用例执行期间读取可变全局对象。"""

    masking_policy = MaskingPolicy(
        enabled=bool(app_config.masking.enabled),
        heuristic_enabled=bool(app_config.masking.heuristic_enabled),
        default_strategy=str(app_config.masking.default_strategy),
        hash_salt=str(os.getenv("MASKING_HASH_SALT") or app_config.auth.secret_key),
    )
    history = ConversationHistorySettings(
        directory_page_size=int(app_config.chat.directory_page_size),
        transcript_page_size=int(app_config.chat.transcript_page_size),
        result_preview_rows=int(app_config.chat.history_result_preview_rows),
        result_preview_bytes=int(app_config.chat.history_result_preview_bytes),
        title_max_chars=int(app_config.chat.title_max_chars),
        retention_days=int(app_config.masking.history_retention_days),
        masking_policy=masking_policy,
    )
    memory = app_config.memory
    embedding = app_config.embedding
    return history, ConversationMemorySettings(
        enabled=bool(memory.enabled),
        working_ttl_seconds=int(memory.working_ttl_seconds),
        recent_message_limit=int(memory.recent_message_limit),
        recent_token_budget=int(memory.recent_token_budget),
        long_term_enabled=bool(memory.long_term_enabled),
        long_term_top_k=int(memory.long_term_top_k),
        es_projection_version=str(memory.es_projection_version),
        qdrant_projection_version=str(memory.qdrant_projection_version),
        extractor_version=str(memory.extractor_version),
        context_schema_version=str(memory.context_schema_version),
        fusion_rrf_k=int(memory.fusion_rrf_k),
        retrieval_timeout_seconds=float(memory.retrieval_timeout_seconds),
        message_content_max_chars=int(memory.message_content_max_chars),
        memory_content_max_chars=int(memory.memory_content_max_chars),
        summary_enabled=bool(memory.summary_enabled),
        summary_build_enabled=bool(memory.summary_build_enabled),
        summary_projection_enabled=bool(memory.summary_projection_enabled),
        summary_turn_high_watermark=int(memory.summary_turn_high_watermark),
        summary_turn_low_watermark=int(memory.summary_turn_low_watermark),
        summary_token_high_watermark=int(memory.summary_token_high_watermark),
        summary_token_low_watermark=int(memory.summary_token_low_watermark),
        summary_recent_protected_turns=int(memory.summary_recent_protected_turns),
        summary_segment_max_messages=int(memory.summary_segment_max_messages),
        summary_segment_max_tokens=int(memory.summary_segment_max_tokens),
        summary_content_max_chars=int(memory.summary_content_max_chars),
        summary_signal_limit=int(memory.summary_signal_limit),
        summary_top_k=int(memory.summary_top_k),
        summary_hierarchy_group_size=int(memory.summary_hierarchy_group_size),
        summary_max_level=int(memory.summary_max_level),
        summary_retrieval_always=bool(memory.summary_retrieval_always),
        summary_extractor_version=str(memory.summary_extractor_version),
        summary_signal_extractor_version=str(memory.summary_signal_extractor_version),
        summary_schema_version=str(memory.summary_schema_version),
        summary_es_projection_version=str(memory.summary_es_projection_version),
        summary_qdrant_projection_version=str(memory.summary_qdrant_projection_version),
        stable_fact_enabled=bool(memory.stable_fact_enabled),
        stable_fact_extractor_version=str(memory.stable_fact_extractor_version),
        embedding_model=str(embedding.model),
        embedding_revision=str(embedding.revision),
        masking_policy=masking_policy,
    )


__all__ = ["load_conversation_settings"]
