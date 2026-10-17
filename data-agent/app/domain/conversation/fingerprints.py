"""会话记忆检索使用的稳定领域指纹。"""

from __future__ import annotations

import hashlib


def memory_query_fingerprint(query: str) -> str:
    """对规范化查询生成稳定且不暴露原文的 SHA-256 指纹。"""

    normalized = " ".join(str(query or "").strip().lower().split())
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


__all__ = ["memory_query_fingerprint"]
