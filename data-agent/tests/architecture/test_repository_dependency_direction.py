"""Repository 依赖方向与退役路径门禁。"""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INFRASTRUCTURE_ROOT = PROJECT_ROOT / "app" / "infrastructure"


def _repository_files() -> list[Path]:
    return sorted(
        path
        for path in INFRASTRUCTURE_ROOT.rglob("*.py")
        if "repositories" in path.parts or path.name.endswith("repository.py")
    )


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_repositories_do_not_import_application_or_agent_system() -> None:
    violations = {
        path.relative_to(PROJECT_ROOT).as_posix(): sorted(
            name
            for name in _imports(path)
            if name == "app.application"
            or name.startswith("app.application.")
            or name == "app.agent_system"
            or name.startswith("app.agent_system.")
        )
        for path in _repository_files()
    }
    assert not {path: names for path, names in violations.items() if names}


def test_repository_files_do_not_use_retired_hybrid_embedding_path() -> None:
    retired = "app.agent_system.agents.nl2sql.contracts.hybrid_embedding"
    violations = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in _repository_files()
        if retired in _imports(path)
    ]
    assert violations == []


def test_retired_hybrid_embedding_module_is_removed() -> None:
    """退役的 Agent 私有合同路径不应继续成为源码事实来源。"""

    retired_module = (
        PROJECT_ROOT
        / "app"
        / "agent_system"
        / "agents"
        / "nl2sql"
        / "contracts"
        / "hybrid_embedding.py"
    )
    assert not retired_module.exists()
