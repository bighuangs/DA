from __future__ import annotations

from app.infrastructure.observability.execution_versions import (
    RuntimeExecutionVersionProvider,
    stable_hash,
)


def test_stable_hash_is_order_independent() -> None:
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})


def test_runtime_snapshot_does_not_require_eval_artifacts(tmp_path) -> None:
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "query.prompt").write_text(
        "query={query}", encoding="utf-8"
    )
    (tmp_path / "conf").mkdir()
    (tmp_path / "conf" / "meta_config.yaml").write_text(
        "tables: []\n", encoding="utf-8"
    )

    snapshot = RuntimeExecutionVersionProvider(tmp_path).snapshot("workspace-1")

    assert snapshot["workspace_id"] == "workspace-1"
    assert snapshot["snapshot_schema_version"] == "execution-snapshot-v1"
    assert str(snapshot["prompt_version"]).startswith("sha256:")
    assert str(snapshot["knowledge_version"]).startswith("sha256:")
    assert str(snapshot["run_fingerprint"]).startswith("sha256:")
    assert snapshot["version_complete"] is True
