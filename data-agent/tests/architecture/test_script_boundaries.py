"""工程脚本只调用正式 Bootstrap/Application 入口。"""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_scripts_do_not_compose_concrete_infrastructure() -> None:
    forbidden = ("app.config", "app.infrastructure", "sqlalchemy")
    violations = {
        path.relative_to(PROJECT_ROOT).as_posix(): sorted(
            module
            for module in _imports(path)
            if module == forbidden[0]
            or module.startswith(f"{forbidden[0]}.")
            or module == forbidden[1]
            or module.startswith(f"{forbidden[1]}.")
            or module == forbidden[2]
            or module.startswith(f"{forbidden[2]}.")
        )
        for path in SCRIPTS_ROOT.glob("*.py")
    }
    assert not {path: modules for path, modules in violations.items() if modules}
