"""用于架构测试的 Python 静态导入分析。"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ImportReference:
    path: Path
    module: str
    line: int


def _module_name(project_root: Path, path: Path) -> tuple[str, bool]:
    relative = path.relative_to(project_root).with_suffix("")
    parts = list(relative.parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    return ".".join(parts), is_package


def _resolve_import_from(
    current_module: str,
    is_package: bool,
    node: ast.ImportFrom,
) -> str:
    if node.level == 0:
        return node.module or ""
    package_parts = current_module.split(".")
    if not is_package:
        package_parts = package_parts[:-1]
    parent_steps = node.level - 1
    if parent_steps:
        package_parts = package_parts[:-parent_steps]
    if node.module:
        package_parts.extend(node.module.split("."))
    return ".".join(package_parts)


def imports_from(project_root: Path, path: Path) -> list[ImportReference]:
    current_module, is_package = _module_name(project_root, path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    references: list[ImportReference] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            references.append(ImportReference(
                path=path,
                module=_resolve_import_from(current_module, is_package, node),
                line=node.lineno,
            ))
        elif isinstance(node, ast.Import):
            references.extend(
                ImportReference(path=path, module=alias.name, line=node.lineno)
                for alias in node.names
            )
    return references


def package_imports(project_root: Path, package: Path) -> list[ImportReference]:
    if not package.exists():
        return []
    references: list[ImportReference] = []
    for path in sorted(package.rglob("*.py")):
        references.extend(imports_from(project_root, path))
    return references


def matches_prefix(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == prefix or module.startswith(f"{prefix}.") for prefix in prefixes)
