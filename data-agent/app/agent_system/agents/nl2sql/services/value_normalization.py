"""字段值召回使用的纯文本规范化规则。"""

from __future__ import annotations

import re
import unicodedata
from typing import Any


def normalize_value(value: Any) -> str:
    """生成稳定的精确/前缀匹配表示，不修改原始值。"""

    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip().casefold()


__all__ = ["normalize_value"]
