"""应用边界统一错误合同。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ErrorContract:
    """可安全返回给入站适配器的版本化错误。"""

    code: str
    message: str
    retryable: bool = False
    details: Mapping[str, Any] | None = None
    schema_version: str = "error-v1"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.details:
            payload["details"] = dict(self.details)
        return payload


class ApplicationError(Exception):
    """携带脱敏错误合同的应用异常。"""

    def __init__(self, contract: ErrorContract) -> None:
        super().__init__(contract.message)
        self.contract = contract
