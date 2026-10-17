"""Agent 操作权限的纯策略判定。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class PermissionSubject(Protocol):
    """权限判定所需的最小调用事实。"""

    actor_id: str
    workspace_id: str
    roles: tuple[str, ...]
    allowed_data_sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PermissionDecision:
    allowed: bool
    reason_code: str
    policy_version: str


@dataclass(frozen=True, slots=True)
class PermissionPolicy:
    """在能力调用前校验角色、操作和数据源范围。"""

    version: str = "agent-permission-policy/v1"
    allowed_roles: frozenset[str] = field(default_factory=frozenset)
    allowed_operations: frozenset[str] = field(default_factory=frozenset)
    require_explicit_data_source: bool = True

    def __post_init__(self) -> None:
        if not str(self.version or "").strip():
            raise ValueError("version 不能为空")
        object.__setattr__(
            self,
            "allowed_roles",
            frozenset(str(item).strip() for item in self.allowed_roles if str(item).strip()),
        )
        object.__setattr__(
            self,
            "allowed_operations",
            frozenset(
                str(item).strip() for item in self.allowed_operations if str(item).strip()
            ),
        )

    def evaluate(
        self,
        subject: PermissionSubject,
        *,
        operation: str,
        data_source: str | None = None,
    ) -> PermissionDecision:
        normalized_operation = str(operation or "").strip()
        if not normalized_operation:
            return PermissionDecision(False, "operation.missing", self.version)
        if self.allowed_roles and not self.allowed_roles.intersection(subject.roles):
            return PermissionDecision(False, "role.denied", self.version)
        if self.allowed_operations and normalized_operation not in self.allowed_operations:
            return PermissionDecision(False, "operation.denied", self.version)
        normalized_source = str(data_source or "").strip()
        if self.require_explicit_data_source and not normalized_source:
            return PermissionDecision(False, "data_source.missing", self.version)
        if normalized_source and normalized_source not in subject.allowed_data_sources:
            return PermissionDecision(False, "data_source.denied", self.version)
        return PermissionDecision(True, "allowed", self.version)
