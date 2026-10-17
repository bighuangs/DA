"""混合检索向量值对象。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HybridEmbedding:
    """一个已校验的稠密向量与学习型稀疏向量组合。"""

    dense: tuple[float, ...]
    sparse_indices: tuple[int, ...]
    sparse_values: tuple[float, ...]

    @classmethod
    def from_raw(
        cls,
        dense: Sequence[float],
        lexical_weights: Mapping[object, object],
        *,
        dense_size: int,
        normalize_dense: bool,
    ) -> "HybridEmbedding":
        """校验模型原始输出，并构造不可变的领域值对象。"""

        dense_values = [float(value) for value in dense]
        if len(dense_values) != int(dense_size):
            raise ValueError(
                f"BGE-M3 dense 维度异常：期望 {dense_size}，实际 {len(dense_values)}"
            )
        if not all(math.isfinite(value) for value in dense_values):
            raise ValueError("BGE-M3 dense 包含 NaN/Inf")
        norm = math.sqrt(sum(value * value for value in dense_values))
        if norm <= 0:
            raise ValueError("BGE-M3 dense 不能是零向量")
        if normalize_dense:
            dense_values = [value / norm for value in dense_values]

        sparse_items: list[tuple[int, float]] = []
        for raw_index, raw_value in lexical_weights.items():
            try:
                index = int(raw_index)
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError("BGE-M3 sparse index/value 类型异常") from exc
            if index < 0:
                raise ValueError("BGE-M3 sparse index 必须为非负整数")
            if not math.isfinite(value):
                raise ValueError("BGE-M3 sparse 包含 NaN/Inf")
            if value > 0:
                sparse_items.append((index, value))
        sparse_items.sort(key=lambda item: item[0])
        if not sparse_items:
            raise ValueError("BGE-M3 sparse 为空或全部为 0")
        if any(
            left[0] == right[0]
            for left, right in zip(sparse_items, sparse_items[1:])
        ):
            raise ValueError("BGE-M3 sparse index 必须唯一")
        return cls(
            dense=tuple(dense_values),
            sparse_indices=tuple(item[0] for item in sparse_items),
            sparse_values=tuple(item[1] for item in sparse_items),
        )

    def as_dict(self) -> dict:
        """转换为基础类型，供缓存和外部适配器序列化。"""

        return {
            "dense": list(self.dense),
            "sparse_indices": list(self.sparse_indices),
            "sparse_values": list(self.sparse_values),
        }
