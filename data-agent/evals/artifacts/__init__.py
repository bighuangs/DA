"""评测产物存储边界与本地实现。"""

from evals.artifacts.base import ArtifactStore
from evals.artifacts.local import ArtifactPathError, LocalArtifactStore

__all__ = ["ArtifactPathError", "ArtifactStore", "LocalArtifactStore"]
