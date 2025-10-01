"""Lightweight in-repo stub of the Chroma client API used by tests."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .api.models.Collection import Collection


# Registry of collections per persistence path so new clients share state.
_CLIENT_REGISTRY: Dict[Path, Dict[str, Collection]] = {}


class PersistentClient:
    """Minimal stub that persists collections on disk per data path."""

    def __init__(self, *, path: str) -> None:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        self._root = root
        self._collections = _CLIENT_REGISTRY.setdefault(root, {})
        if not self._collections:
            for file in sorted(root.glob("*.json")):
                name = file.stem
                self._collections[name] = Collection(name=name, storage_dir=root)

    def list_collections(self) -> List[Collection]:
        return list(self._collections.values())

    def get_collection(self, name: str, embedding_function=None) -> Collection:
        collection = self._collections[name]
        if embedding_function is not None:
            collection.embedding_function = embedding_function
        return collection

    def get_or_create_collection(
        self,
        *,
        name: str,
        metadata: Optional[dict] = None,
        embedding_function=None,
    ) -> Collection:
        collection = self._collections.get(name)
        if collection is None:
            collection = Collection(
                name=name,
                metadata=dict(metadata or {}),
                embedding_function=embedding_function,
                storage_dir=self._root,
            )
            self._collections[name] = collection
        else:
            if metadata is not None:
                collection.set_metadata(dict(metadata))
            if embedding_function is not None:
                collection.embedding_function = embedding_function
        return collection

    def delete_collection(self, name: str) -> None:
        collection = self._collections.pop(name, None)
        if collection is not None:
            collection.drop_storage()

    def persist(self) -> None:  # pragma: no cover - interface compatibility
        return None


__all__ = ["PersistentClient"]
