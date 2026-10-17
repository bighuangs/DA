"""NL2SQL Execution 使用的不可变非敏感配置快照。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class FrozenSettingsSection:
    """以属性方式读取的不可变配置分区。"""

    _values: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "FrozenSettingsSection":
        return cls(MappingProxyType({
            str(key): _freeze(value) for key, value in values.items()
        }))

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def as_mapping(self) -> Mapping[str, Any]:
        return self._values


@dataclass(frozen=True, slots=True)
class SeedMetricSettings:
    """供指标召回补全口径的只读种子指标。"""

    name: str
    description: str
    relevant_columns: tuple[str, ...]
    alias: tuple[str, ...]
    formula_rule: Mapping[str, Any] | None

    @classmethod
    def create(
        cls,
        *,
        name: str,
        description: str,
        relevant_columns: Sequence[str],
        alias: Sequence[str],
        formula_rule: Mapping[str, Any] | None,
    ) -> "SeedMetricSettings":
        return cls(
            name=str(name),
            description=str(description),
            relevant_columns=tuple(str(value) for value in relevant_columns),
            alias=tuple(str(value) for value in alias),
            formula_rule=(
                MappingProxyType(dict(_freeze(formula_rule)))
                if formula_rule is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class Nl2SqlSettings:
    """由组合根为一次 Graph Runtime 装配的非敏感配置清单。"""

    query: FrozenSettingsSection
    query_signals: FrozenSettingsSection
    embedding: FrozenSettingsSection
    memory: FrozenSettingsSection
    planning_route: FrozenSettingsSection
    semantic_repair: FrozenSettingsSection
    llm: FrozenSettingsSection
    seed_metrics: tuple[SeedMetricSettings, ...] = ()


def override_settings(
    settings: Nl2SqlSettings,
    section: str,
    **values: Any,
) -> Nl2SqlSettings:
    """生成单个分区带显式覆盖的新快照，供评测与测试装配复用。"""

    current = getattr(settings, section)
    if not isinstance(current, FrozenSettingsSection):
        raise ValueError(f"不支持覆盖 settings 分区：{section}")
    merged = {**dict(current.as_mapping()), **values}
    return replace(
        settings,
        **{section: FrozenSettingsSection.from_mapping(merged)},
    )


__all__ = [
    "FrozenSettingsSection",
    "Nl2SqlSettings",
    "SeedMetricSettings",
    "override_settings",
]
