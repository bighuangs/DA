import hashlib
from pathlib import Path

import pytest

from evals.artifacts import ArtifactPathError, LocalArtifactStore


def test_default_root_is_var_evals_without_creating_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    store = LocalArtifactStore()

    assert store.root == (tmp_path / "var" / "evals").resolve()
    assert not store.root.exists()


def test_write_is_atomic_and_returns_verified_reference(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    data = "第一版".encode()

    first = store.write_bytes("eval-1", "traces/case-1.jsonl", data)
    second_data = "第二版".encode()
    second = store.write_bytes("eval-1", "traces/case-1.jsonl", second_data)

    assert store.read_bytes("eval-1", "traces/case-1.jsonl") == second_data
    assert first.sha256 == hashlib.sha256(data).hexdigest()
    assert second.sha256 == hashlib.sha256(second_data).hexdigest()
    assert second.relative_path == "eval-1/traces/case-1.jsonl"
    assert not list((tmp_path / "artifacts").rglob("*.tmp"))


@pytest.mark.parametrize(
    "relative_path",
    [
        "../escape.json",
        "nested/../../escape.json",
        "/absolute/escape.json",
        "C:\\escape.json",
        ".",
    ],
)
def test_rejects_path_escape(tmp_path: Path, relative_path: str) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")

    with pytest.raises(ArtifactPathError):
        store.write_bytes("eval-1", relative_path, b"unsafe")


def test_rejects_evaluation_id_as_path(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")

    with pytest.raises(ArtifactPathError):
        store.write_bytes("../eval-1", "report.json", b"unsafe")
