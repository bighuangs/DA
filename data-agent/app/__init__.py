"""Data Agent 生产应用包。

依赖由 ``foundation``、``domain`` 向 ``application`` 和 ``agent_system`` 单向扩展，
HTTP/Worker 位于 ``adapters``，具体外部实现位于 ``infrastructure``，并统一由
``bootstrap`` 组合。离线评测独立放在顶层 ``evals`` 包中。
"""
