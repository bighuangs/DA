"""Worker Adapter 与生产组合根的可执行边界门禁。"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKER_ROOT = PROJECT_ROOT / "app" / "adapters" / "inbound" / "worker"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_inbound_workers_use_application_ports_only() -> None:
    violations = {
        path.relative_to(PROJECT_ROOT).as_posix(): sorted(
            name
            for name in _imports(path)
            if (
                name == "sqlalchemy"
                or name.startswith("sqlalchemy.")
                or (
                    name == "app"
                    or name.startswith("app.")
                )
                and not (
                    name == "app.application"
                    or name.startswith("app.application.")
                )
            )
        )
        for path in WORKER_ROOT.rglob("*.py")
    }

    assert not {path: names for path, names in violations.items() if names}


def test_inbound_worker_has_no_module_global_runtime_instance() -> None:
    path = WORKER_ROOT / "memory_projection_worker.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[int] = []
    for node in tree.body:
        value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
        if not isinstance(value, ast.Call):
            continue
        function = value.func
        if isinstance(function, ast.Name) and function.id == "MemoryProjectionWorker":
            violations.append(node.lineno)

    assert violations == []


def test_worker_console_entry_points_to_bootstrap() -> None:
    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert project["project"]["scripts"]["data-agent-memory-worker"] == (
        "app.bootstrap.worker:main"
    )
