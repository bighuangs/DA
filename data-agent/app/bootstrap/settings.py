"""仅描述进程装配行为的启动配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BootstrapSettings:
    """控制组合根选择，不承载业务或基础设施配置。"""

    application_name: str = "data-agent"
    evaluation_http_enabled: bool = False

    @classmethod
    def from_environment(cls) -> "BootstrapSettings":
        """读取启动开关；读取过程不会导入或创建外部客户端。"""
        return cls(
            application_name=(
                os.getenv("APPLICATION_NAME", "data-agent").strip() or "data-agent"
            ),
            evaluation_http_enabled=(
                os.getenv("EVALUATION_HTTP_ENABLED", "").strip().lower()
                in {"1", "true", "yes", "on"}
            ),
        )
