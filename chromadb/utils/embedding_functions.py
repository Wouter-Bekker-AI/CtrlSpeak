"""Embedding function protocol for the chromadb stub."""
from __future__ import annotations

from typing import Generic, Iterable, TypeVar

T = TypeVar("T")


class EmbeddingFunction(Generic[T]):
    """Match the call signature expected by utils.vector_memory."""

    def __call__(self, input: T) -> Iterable[Iterable[float]]:  # pragma: no cover - protocol
        raise NotImplementedError
