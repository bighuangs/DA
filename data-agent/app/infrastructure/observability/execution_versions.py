"""从真实发布配置构建请求执行版本快照。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from app.config.app_config import app_config
from app.infrastructure.search.index_version import retrieval_index_version

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def stable_hash(payload: object) -> str:
    """对 JSON 可序列化结构生成稳定 SHA-256。"""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _file_hashes(root: Path, pattern: str) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): "sha256:"
        + hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob(pattern))
        if path.is_file()
    }


def _without_secret_fields(values: Mapping[str, Any]) -> dict[str, Any]:
    """移除凭据及凭据来源字段，仅保留会影响部署语义的配置。"""

    return {
        str(key): value
        for key, value in values.items()
        if "key" not in str(key).lower() and "secret" not in str(key).lower()
    }


class RuntimeExecutionVersionProvider:
    """从当前配置与发布资产冻结请求版本，不依赖离线 Eval 目录。"""

    def __init__(self, project_root: Path | None = None) -> None:
        self._project_root = (project_root or PROJECT_ROOT).resolve()

    def snapshot(self, workspace_id: str) -> Mapping[str, Any]:
        prompt_hashes = _file_hashes(self._project_root / "prompts", "*.prompt")
        knowledge_hashes = _file_hashes(self._project_root / "conf", "*.yaml")
        components: dict[str, Any] = {
            "snapshot_schema_version": "execution-snapshot-v1",
            "workspace_id": str(workspace_id or ""),
            "prompt_version": stable_hash(prompt_hashes),
            "knowledge_version": stable_hash(knowledge_hashes),
            "schema_version": app_config.memory.context_schema_version,
            "index_version": retrieval_index_version(),
            "model_version": app_config.llm.model_name,
            "planning_contract_version": app_config.planning_route.contract_version,
            "semantic_repair_contract_version": app_config.semantic_repair.contract_version,
        }
        components["run_fingerprint"] = stable_hash(components)
        components["version_complete"] = all(
            components.get(field)
            for field in (
                "prompt_version",
                "knowledge_version",
                "schema_version",
                "index_version",
                "model_version",
            )
        )
        return components

    def version_for(self, capability: str) -> str:
        """返回当前能力的发布配置指纹，不读取凭据值。"""

        name = str(capability or "unknown")
        if name == "llm.sql":
            route = dict((app_config.llm.routes or {}).get("sql") or {})
            providers = {
                str(provider): _without_secret_fields(dict(settings or {}))
                for provider, settings in dict(app_config.llm.providers or {}).items()
            }
            return stable_hash({
                "capability": name,
                "global_model_name": app_config.llm.model_name,
                "global_provider": app_config.llm.provider,
                "global_base_url": app_config.llm.base_url,
                "route": _without_secret_fields(route),
                "providers": providers,
            })
        return stable_hash({"capability": name, "snapshot": self.snapshot("")})
