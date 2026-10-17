"""跨业务边界稳定且无外部依赖的基础类型。"""

from app.foundation.clock import Clock, SystemClock
from app.foundation.errors import ApplicationError, ErrorContract
from app.foundation.ids import new_run_id, new_trace_id

__all__ = [
    "ApplicationError",
    "Clock",
    "ErrorContract",
    "SystemClock",
    "new_run_id",
    "new_trace_id",
]
