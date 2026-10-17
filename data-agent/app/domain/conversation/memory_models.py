"""Versioned DTOs shared by memory storage, Agent state, Trace and API."""
from __future__ import annotations

from typing import NotRequired, TypedDict


MEMORY_CONTEXT_SCHEMA_VERSION = "memory-context-v2"
SUMMARY_SEGMENT_SCHEMA_VERSION = "summary-segment-v1"
SUMMARY_SIGNALS_SCHEMA_VERSION = "summary-signals-v1"


class MemoryReference(TypedDict):
    memory_id: str
    memory_type: str
    content: str
    confidence: float
    source: str
    created_at: str
    structured_payload: NotRequired[dict]


class SummarySignal(TypedDict):
    signal_type: str
    normalized_value: str
    weight: float
    source_order: int
    source_message_ids: list[int]


class SummaryReference(TypedDict):
    summary_id: str
    level: int
    content: str
    source_message_ids: list[int]
    source_start_message_id: int
    source_end_message_id: int
    confidence: float
    source: str
    structured_facts: NotRequired[list[dict]]
    signals: NotRequired[list[SummarySignal]]


class MemoryContext(TypedDict):
    schema_version: str
    working_state: dict
    recent_messages: list[dict]
    selected_summaries: list[SummaryReference]
    confirmed_preferences: list[MemoryReference]
    confirmed_profile_facts: list[MemoryReference]
    long_term_goals: list[MemoryReference]
    relevant_episodes: list[MemoryReference]
    provenance: list[dict]
    omitted_counts: dict
    evidence: dict
