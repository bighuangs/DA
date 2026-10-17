"""Serialize access to the request-scoped SQLAlchemy AsyncSession.

LangGraph runs retrieval routes concurrently, while one AsyncSession cannot
provision or execute multiple statements at the same time. Vector/ES work stays
parallel; only the short Meta MySQL reads share this lock.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar


T = TypeVar("T")


async def serialized_meta_call(context: dict[str, Any], operation: Callable[[], Awaitable[T]]) -> T:
    lock = context.get("_meta_repo_lock")
    if lock is None:
        lock = asyncio.Lock()
        context["_meta_repo_lock"] = lock
    async with lock:
        return await operation()
