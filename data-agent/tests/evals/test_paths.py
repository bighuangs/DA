from __future__ import annotations

import ast

from evals.paths import (
    BASELINES_ROOT,
    CONFIGS_ROOT,
    LEGACY_DATASETS_ROOT,
    PROJECT_ROOT,
    REPORTS_ROOT,
    RUNS_ROOT,
    RUNTIME_ROOT,
)


def test_evaluation_paths_have_separate_code_data_and_runtime_roots() -> None:
    assert LEGACY_DATASETS_ROOT == PROJECT_ROOT / "evals" / "datasets" / "legacy"
    assert CONFIGS_ROOT == PROJECT_ROOT / "evals" / "configs"
    assert BASELINES_ROOT == PROJECT_ROOT / "evals" / "baselines"
    assert RUNTIME_ROOT == PROJECT_ROOT / "var" / "evals" / "legacy"
    assert REPORTS_ROOT == RUNTIME_ROOT / "reports"
    assert RUNS_ROOT == RUNTIME_ROOT / "runs"


def test_python_sources_do_not_reintroduce_legacy_eval_root() -> None:
    forbidden_fragments = (
        'PROJECT_ROOT / "eval"',
        "PROJECT_ROOT / 'eval'",
        'ROOT / "eval"',
        "ROOT / 'eval'",
        '"eval/reports',
        "'eval/reports",
        '"eval/runs',
        "'eval/runs",
        '"app/eval/',
        "'app/eval/",
        "app.eval.",
    )
    evals_root = PROJECT_ROOT / "evals"
    violations: list[str] = []
    for path in sorted(evals_root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
        for fragment in forbidden_fragments:
            if fragment in source:
                violations.append(f"{path.relative_to(PROJECT_ROOT)}: {fragment}")

    assert not violations, "发现旧 eval 根路径引用:\n" + "\n".join(violations)
