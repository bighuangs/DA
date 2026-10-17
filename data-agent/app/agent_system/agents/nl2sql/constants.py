"""NL2SQL Graph 与节点共享的轻量常量。"""

# SQL 校正达到上限后必须终止，避免无界模型循环。
MAX_RETRY = 3

__all__ = ["MAX_RETRY"]
