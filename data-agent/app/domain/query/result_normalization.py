"""跨 Repository、API 与导出共用的查询结果值归一化。"""

from __future__ import annotations

import math
from datetime import date, datetime, time
from decimal import Decimal


def normalize_result_value(value):
    """归一化查询值，避免 Decimal/超大整数经过 JSON 或 JS Number 后丢精度。"""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, int) and abs(value) > 9_007_199_254_740_991:
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def normalize_rows(rows: list[dict]) -> list[dict]:
    """归一化结果行，同时把列键稳定转换为字符串。"""
    return [
        {str(key): normalize_result_value(value) for key, value in row.items()}
        for row in rows
    ]
