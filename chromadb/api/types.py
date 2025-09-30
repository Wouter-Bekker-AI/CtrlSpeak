"""Type aliases used by the chromadb stub."""
from __future__ import annotations

from typing import Iterable, List

Documents = Iterable[str]
Embeddings = List[List[float]]

__all__ = ["Documents", "Embeddings"]
