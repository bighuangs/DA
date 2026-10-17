"""``query-intent-safety-v1`` 的严格、无副作用加载器。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

from evals.contracts import ContractError
from evals.contracts.query_intent_safety import (
    QueryIntentCaseOrigin,
    QueryIntentCategory,
    QueryIntentDecision,
    QueryIntentExpectedOutcome,
    QueryIntentHistoryRole,
    QueryIntentHistoryTurn,
    QueryIntentReasonCode,
    QueryIntentResponseKind,
    QueryIntentSafetyCase,
    QueryIntentSafetyDataset,
    QueryIntentSafetyManifest,
    QueryIntentSlice,
)


class QueryIntentSafetyDatasetError(ValueError):
    """数据文件格式或完整性不满足冻结合同时抛出。"""

    def __init__(self, issues: Sequence[str]) -> None:
        self.issues = tuple(issues) or ("未知查询意图安全数据集错误",)
        super().__init__("查询意图安全数据集校验失败:\n- " + "\n- ".join(self.issues))


_EnumT = TypeVar("_EnumT")

_MANIFEST_KEYS = {
    "schema_version",
    "dataset_id",
    "dataset_version",
    "split",
    "case_file",
    "contains_hidden_labels",
    "known_failure_case_ids",
    "required_slices",
}
_CASE_KEYS = {
    "case_id",
    "dataset_id",
    "dataset_version",
    "split",
    "origin",
    "query",
    "history",
    "slices",
    "expected",
    "known_failure_reference",
}
_HISTORY_KEYS = {"role", "content"}
_EXPECTED_KEYS = {
    "category",
    "decision",
    "response_kind",
    "reason_code",
    "terminal",
    "sql_generation_allowed",
}


def _object(value: Any, context: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} 必须是 JSON object")
    actual_keys = set(value)
    missing = sorted(keys - actual_keys)
    unknown = sorted(actual_keys - keys)
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"缺少字段: {', '.join(missing)}")
        if unknown:
            parts.append(f"未知字段: {', '.join(unknown)}")
        raise ValueError(f"{context} " + "；".join(parts))
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} 必须是非空字符串")
    return value.strip()


def _bool(value: Any, context: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{context} 必须是 boolean")
    return value


def _list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} 必须是 array")
    return value


def _enum(enum_type: type[_EnumT], value: Any, context: str) -> _EnumT:
    parsed = _string(value, context)
    try:
        return enum_type(parsed)  # type: ignore[call-arg]
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_type)  # type: ignore[attr-defined]
        raise ValueError(f"{context} 必须是以下值之一: {allowed}") from exc


def _optional_response_kind(value: Any) -> QueryIntentResponseKind | None:
    if value is None:
        return None
    return _enum(QueryIntentResponseKind, value, "expected.response_kind")


def _parse_manifest(value: Any) -> QueryIntentSafetyManifest:
    row = _object(value, "manifest", _MANIFEST_KEYS)
    return QueryIntentSafetyManifest(
        schema_version=_string(row["schema_version"], "manifest.schema_version"),
        dataset_id=_string(row["dataset_id"], "manifest.dataset_id"),
        dataset_version=_string(row["dataset_version"], "manifest.dataset_version"),
        split=_string(row["split"], "manifest.split"),
        case_file=_string(row["case_file"], "manifest.case_file"),
        contains_hidden_labels=_bool(
            row["contains_hidden_labels"], "manifest.contains_hidden_labels"
        ),
        known_failure_case_ids=tuple(
            _string(item, "manifest.known_failure_case_ids[]")
            for item in _list(
                row["known_failure_case_ids"], "manifest.known_failure_case_ids"
            )
        ),
        required_slices=tuple(
            _enum(QueryIntentSlice, item, "manifest.required_slices[]")
            for item in _list(row["required_slices"], "manifest.required_slices")
        ),
    )


def _parse_history(value: Any) -> tuple[QueryIntentHistoryTurn, ...]:
    turns: list[QueryIntentHistoryTurn] = []
    for index, raw_turn in enumerate(_list(value, "history")):
        row = _object(raw_turn, f"history[{index}]", _HISTORY_KEYS)
        turns.append(
            QueryIntentHistoryTurn(
                role=_enum(
                    QueryIntentHistoryRole,
                    row["role"],
                    f"history[{index}].role",
                ),
                content=_string(row["content"], f"history[{index}].content"),
            )
        )
    return tuple(turns)


def _parse_expected(value: Any) -> QueryIntentExpectedOutcome:
    row = _object(value, "expected", _EXPECTED_KEYS)
    return QueryIntentExpectedOutcome(
        category=_enum(QueryIntentCategory, row["category"], "expected.category"),
        decision=_enum(QueryIntentDecision, row["decision"], "expected.decision"),
        response_kind=_optional_response_kind(row["response_kind"]),
        reason_code=_enum(
            QueryIntentReasonCode, row["reason_code"], "expected.reason_code"
        ),
        terminal=_bool(row["terminal"], "expected.terminal"),
        sql_generation_allowed=_bool(
            row["sql_generation_allowed"], "expected.sql_generation_allowed"
        ),
    )


def _parse_case(value: Any) -> QueryIntentSafetyCase:
    row = _object(value, "case", _CASE_KEYS)
    known_failure_reference = row["known_failure_reference"]
    if known_failure_reference is not None:
        known_failure_reference = _string(
            known_failure_reference, "known_failure_reference"
        )
    return QueryIntentSafetyCase(
        case_id=_string(row["case_id"], "case_id"),
        dataset_id=_string(row["dataset_id"], "dataset_id"),
        dataset_version=_string(row["dataset_version"], "dataset_version"),
        split=_string(row["split"], "split"),
        origin=_enum(QueryIntentCaseOrigin, row["origin"], "origin"),
        query=_string(row["query"], "query"),
        history=_parse_history(row["history"]),
        slices=tuple(
            _enum(QueryIntentSlice, item, "slices[]")
            for item in _list(row["slices"], "slices")
        ),
        expected=_parse_expected(row["expected"]),
        known_failure_reference=known_failure_reference,
    )


class QueryIntentSafetyDatasetLoader:
    """加载公开 development 集，不执行模型、网络或生产代码。"""

    def load(self, directory: str | Path) -> QueryIntentSafetyDataset:
        root = Path(directory)
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise QueryIntentSafetyDatasetError(("缺少数据文件: manifest.json",))

        try:
            manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            manifest = _parse_manifest(manifest_raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ContractError, ValueError) as exc:
            raise QueryIntentSafetyDatasetError((f"manifest.json: {exc}",)) from exc

        cases_path = root / manifest.case_file
        if not cases_path.is_file():
            raise QueryIntentSafetyDatasetError((f"缺少数据文件: {manifest.case_file}",))

        cases: list[QueryIntentSafetyCase] = []
        issues: list[str] = []
        try:
            lines = cases_path.read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise QueryIntentSafetyDatasetError((f"{manifest.case_file}: {exc}",)) from exc
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                issues.append(
                    f"{manifest.case_file}:{line_number}: JSONL 不允许空行"
                )
                continue
            try:
                cases.append(_parse_case(json.loads(line)))
            except (json.JSONDecodeError, ContractError, ValueError) as exc:
                issues.append(f"{manifest.case_file}:{line_number}: {exc}")

        if issues:
            raise QueryIntentSafetyDatasetError(issues)
        try:
            return QueryIntentSafetyDataset(manifest=manifest, cases=tuple(cases))
        except ContractError as exc:
            raise QueryIntentSafetyDatasetError((str(exc),)) from exc


__all__ = [
    "QueryIntentSafetyDatasetError",
    "QueryIntentSafetyDatasetLoader",
]
