"""NL2SQL 调用期间只读的事实与策略上下文。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


INVOCATION_CONTEXT_VERSION = "nl2sql-invocation-context/v2"


@dataclass(frozen=True, slots=True)
class InvocationContext:
    """不可变调用上下文；客户端、会话、锁和密钥必须由 Runtime 持有。"""

    request_id: str
    actor_id: str
    workspace_id: str
    roles: tuple[str, ...] = ()
    allowed_data_sources: tuple[str, ...] = ()
    conversation_id: str | None = None
    trace_id: str | None = None
    locale: str = "zh-CN"
    timezone: str = "Asia/Shanghai"
    policy_version: str = "access-policy/v1"
    analysis_enabled: bool = False
    evaluation_mode: bool = False
    feature_flags: frozenset[str] = field(default_factory=frozenset)
    version: str = INVOCATION_CONTEXT_VERSION

    def __post_init__(self) -> None:
        if self.version != INVOCATION_CONTEXT_VERSION:
            raise ValueError(f"不支持的 InvocationContext 版本：{self.version!r}")
        for name in (
            "request_id",
            "actor_id",
            "workspace_id",
            "locale",
            "timezone",
            "policy_version",
        ):
            normalized = str(getattr(self, name) or "").strip()
            if not normalized:
                raise ValueError(f"{name} 不能为空")
            object.__setattr__(self, name, normalized)
        for name in ("conversation_id", "trace_id"):
            value = getattr(self, name)
            object.__setattr__(self, name, str(value).strip() or None if value else None)
        object.__setattr__(
            self,
            "roles",
            tuple(dict.fromkeys(str(item).strip() for item in self.roles if str(item).strip())),
        )
        object.__setattr__(
            self,
            "allowed_data_sources",
            tuple(
                dict.fromkeys(
                    str(item).strip()
                    for item in self.allowed_data_sources
                    if str(item).strip()
                )
            ),
        )
        object.__setattr__(
            self,
            "feature_flags",
            frozenset(
                str(item).strip() for item in self.feature_flags if str(item).strip()
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "request_id": self.request_id,
            "actor_id": self.actor_id,
            "workspace_id": self.workspace_id,
            "roles": list(self.roles),
            "allowed_data_sources": list(self.allowed_data_sources),
            "conversation_id": self.conversation_id,
            "trace_id": self.trace_id,
            "locale": self.locale,
            "timezone": self.timezone,
            "policy_version": self.policy_version,
            "analysis_enabled": self.analysis_enabled,
            "evaluation_mode": self.evaluation_mode,
            "feature_flags": sorted(self.feature_flags),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> InvocationContext:
        if value.get("version") != INVOCATION_CONTEXT_VERSION:
            raise ValueError(f"不支持的 InvocationContext 版本：{value.get('version')!r}")
        return cls(
            version=str(value["version"]),
            request_id=str(value.get("request_id") or ""),
            actor_id=str(value.get("actor_id") or ""),
            workspace_id=str(value.get("workspace_id") or ""),
            roles=tuple(value.get("roles") or ()),
            allowed_data_sources=tuple(value.get("allowed_data_sources") or ()),
            conversation_id=value.get("conversation_id"),
            trace_id=value.get("trace_id"),
            locale=str(value.get("locale") or ""),
            timezone=str(value.get("timezone") or ""),
            policy_version=str(value.get("policy_version") or ""),
            analysis_enabled=bool(value.get("analysis_enabled", False)),
            evaluation_mode=bool(value.get("evaluation_mode", False)),
            feature_flags=frozenset(value.get("feature_flags") or ()),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_json(cls, value: str) -> InvocationContext:
        decoded = json.loads(value)
        if not isinstance(decoded, dict):
            raise ValueError("InvocationContext JSON 必须是对象")
        return cls.from_dict(decoded)
