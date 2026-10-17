"""默认写入 ``var/evals`` 的安全本地产物存储。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Mapping

from evals.contracts import ArtifactRef, ContractError, validate_identifier


class ArtifactPathError(ValueError):
    """表示调用方尝试越过评测产物根目录。"""


class LocalArtifactStore:
    """支持路径约束与同目录原子替换的本地 ArtifactStore。"""

    def __init__(self, root: str | Path = Path("var") / "evals") -> None:
        self.root = Path(root).expanduser().resolve(strict=False)

    def write_bytes(
        self,
        evaluation_id: str,
        relative_path: str,
        data: bytes,
        *,
        media_type: str = "application/octet-stream",
    ) -> ArtifactRef:
        """使用临时文件、``fsync`` 和 ``os.replace`` 原子提交产物。"""

        if not isinstance(data, bytes):
            raise TypeError("data 必须是 bytes")
        target = self._resolve_target(evaluation_id, relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._assert_contained(target.parent, self.root / evaluation_id)

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, target)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        digest = hashlib.sha256(data).hexdigest()
        stored_relative_path = (Path(evaluation_id) / Path(relative_path)).as_posix()
        return ArtifactRef(
            evaluation_id=evaluation_id,
            relative_path=stored_relative_path,
            uri=target.as_uri(),
            sha256=digest,
            size_bytes=len(data),
            media_type=media_type,
        )

    def write_json(
        self,
        evaluation_id: str,
        relative_path: str,
        payload: Mapping[str, object] | list[object],
    ) -> ArtifactRef:
        """用稳定键序和 UTF-8 编码写入 JSON。"""

        data = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return self.write_bytes(
            evaluation_id,
            relative_path,
            data,
            media_type="application/json",
        )

    def read_bytes(self, evaluation_id: str, relative_path: str) -> bytes:
        """在相同路径约束下读取产物。"""

        return self._resolve_target(evaluation_id, relative_path).read_bytes()

    def _resolve_target(self, evaluation_id: str, relative_path: str) -> Path:
        try:
            validate_identifier("evaluation_id", evaluation_id)
        except ContractError as exc:
            raise ArtifactPathError(str(exc)) from exc
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", evaluation_id):
            raise ArtifactPathError("evaluation_id 不是安全的路径分段")
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ArtifactPathError("relative_path 必须是非空字符串")
        if "\x00" in relative_path:
            raise ArtifactPathError("relative_path 不能包含空字节")

        candidate_path = Path(relative_path)
        if candidate_path.is_absolute() or candidate_path.drive:
            raise ArtifactPathError("relative_path 不能是绝对路径")
        if any(part in {".", ".."} for part in candidate_path.parts):
            raise ArtifactPathError("relative_path 不能包含路径导航分段")

        run_root = (self.root / evaluation_id).resolve(strict=False)
        target = (run_root / candidate_path).resolve(strict=False)
        self._assert_contained(target, run_root)
        if target == run_root:
            raise ArtifactPathError("relative_path 必须指向文件")
        return target

    @staticmethod
    def _assert_contained(candidate: Path, root: Path) -> None:
        resolved_root = root.resolve(strict=False)
        resolved_candidate = candidate.resolve(strict=False)
        if not resolved_candidate.is_relative_to(resolved_root):
            raise ArtifactPathError("产物路径越过了当前 evaluation 根目录")
