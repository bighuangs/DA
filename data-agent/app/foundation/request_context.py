"""使用 ``ContextVar`` 隔离每个异步请求的关联标识。"""

from __future__ import annotations

from contextvars import ContextVar, Token


_request_id: ContextVar[str] = ContextVar("request_id", default="")


def set_req_id(request_id: str) -> Token[str]:
    """设置当前异步上下文的请求标识。"""

    return _request_id.set(request_id)


def get_req_id() -> str:
    """返回当前异步上下文的请求标识。"""

    return _request_id.get()


__all__ = ["get_req_id", "set_req_id"]
