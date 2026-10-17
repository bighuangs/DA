"""集中生成不携带业务语义的运行标识。"""

from __future__ import annotations

import uuid


def new_run_id() -> str:
    """生成一次 Agent Execution 的稳定标识。"""

    return uuid.uuid4().hex


def new_trace_id() -> str:
    """生成一次可观测链路标识。"""

    return uuid.uuid4().hex
