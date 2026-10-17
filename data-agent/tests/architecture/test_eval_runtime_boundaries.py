"""评测 Harness 与具体生产 Target 的依赖边界。"""

from __future__ import annotations

from pathlib import Path

from tests.architecture._imports import package_imports


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_generic_evaluation_harness_does_not_import_production_app() -> None:
    references = package_imports(PROJECT_ROOT, PROJECT_ROOT / "evals" / "harness")
    violations = [
        f"{item.path.relative_to(PROJECT_ROOT)}:{item.line} -> {item.module}"
        for item in references
        if item.module == "app" or item.module.startswith("app.")
    ]

    assert violations == []


def test_production_nl2sql_target_is_an_outbound_eval_adapter() -> None:
    target = PROJECT_ROOT / "evals" / "adapters" / "outbound" / "nl2sql.py"

    assert target.is_file()
    modules = {item.module for item in package_imports(PROJECT_ROOT, target.parent)}
    assert "app.adapters.outbound.agent.nl2sql_runtime" in modules
    assert "app.agent_system.agents.nl2sql.runtime" in modules
