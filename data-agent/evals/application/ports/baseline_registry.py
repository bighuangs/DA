"""NL2SQL baseline 治理持久化端口。"""

from __future__ import annotations

from typing import Any, Protocol


class EvaluationBaselineRegistry(Protocol):
    def get(self, workspace_id: str, run_id: str) -> Any: ...

    def approved(self, workspace_id: str, dataset_suite: str) -> Any: ...

    def approved_all(self, workspace_id: str) -> Any: ...

    def review(
        self,
        run: dict,
        *,
        decision: str,
        reviewer: str,
        note: str,
    ) -> Any: ...
