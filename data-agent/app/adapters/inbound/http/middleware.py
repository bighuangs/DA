"""FastAPI 请求级边界与进程内保护中间件。"""

from __future__ import annotations

import os
import time
import uuid
from collections import defaultdict, deque

from fastapi import FastAPI, Request
from starlette.responses import JSONResponse

from app.foundation.request_context import set_req_id


def register_http_middleware(app: FastAPI) -> None:
    """为单个 FastAPI 实例注册请求 ID、体积上限和进程内速率限制。"""
    request_windows: dict[str, deque[float]] = defaultdict(deque)
    request_limit_per_minute = max(
        1, int(os.getenv("REQUESTS_PER_MINUTE", "120"))
    )
    max_request_bytes = max(
        1024,
        int(os.getenv("MAX_REQUEST_BYTES", str(12 * 1024 * 1024))),
    )

    @app.middleware("http")
    async def enforce_request_boundaries(request: Request, call_next):
        set_req_id(str(uuid.uuid4()))

        content_length = request.headers.get("content-length")
        if (
            content_length
            and content_length.isdigit()
            and int(content_length) > max_request_bytes
        ):
            return JSONResponse(
                status_code=413,
                content={"detail": "请求体超过大小上限"},
            )

        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = request_windows[client]
        while window and window[0] <= now - 60:
            window.popleft()
        if len(window) >= request_limit_per_minute:
            return JSONResponse(
                status_code=429,
                content={"detail": "请求过于频繁，请稍后重试"},
                headers={"Retry-After": "60"},
            )
        window.append(now)
        return await call_next(request)
