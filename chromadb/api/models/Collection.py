"""Minimal Collection implementation with JSON persistence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class Collection:
    def __init__(
        self,
        *,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding_function: Any = None,
        storage_dir: Optional[Path] = None,
    ) -> None:
        self.name = name
        self.metadata = dict(metadata or {})
        self.embedding_function = embedding_function
        self._storage_dir = storage_dir or Path.cwd()
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._backing_file = self._storage_dir / f"{self.name}.json"
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._load()

    # Chroma's API exposes these properties; tests rely on them directly.
    def count(self) -> int:
        return len(self._entries)

    def add(
        self,
        *,
        ids: Iterable[str],
        documents: Iterable[str],
        metadatas: Iterable[Dict[str, Any]],
        embeddings: Iterable[Iterable[float]],
    ) -> None:
        for entry_id, doc, meta, embedding in zip(ids, documents, metadatas, embeddings):
            self._entries[str(entry_id)] = {
                "document": doc,
                "metadata": dict(meta),
                "embedding": list(embedding),
            }
        self._persist()

    def get(self, *, include: Optional[List[str]] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        include = include or []
        items = list(self._entries.items())
        if limit is not None:
            items = items[:limit]
        payload: Dict[str, Any] = {"ids": [entry_id for entry_id, _ in items]}
        if "documents" in include:
            payload["documents"] = [entry["document"] for _, entry in items]
        if "metadatas" in include:
            payload["metadatas"] = [dict(entry["metadata"]) for _, entry in items]
        if "embeddings" in include:
            payload["embeddings"] = [list(entry["embedding"]) for _, entry in items]
        return payload

    def delete(self, *, ids: Iterable[str]) -> None:
        for entry_id in ids:
            self._entries.pop(str(entry_id), None)
        self._persist()

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        self.metadata = dict(metadata)
        self._persist()

    def drop_storage(self) -> None:
        try:
            self._backing_file.unlink()
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _persist(self) -> None:
        payload = {
            "name": self.name,
            "metadata": self.metadata,
            "entries": list(self._entries.items()),
        }
        self._backing_file.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _load(self) -> None:
        if not self._backing_file.exists():
            return
        try:
            payload = json.loads(self._backing_file.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        entries = payload.get("entries")
        if not isinstance(entries, list):
            return
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            self.metadata = dict(metadata)
        restored: Dict[str, Dict[str, Any]] = {}
        for item in entries:
            if not isinstance(item, list) or len(item) != 2:
                continue
            entry_id, content = item
            if not isinstance(entry_id, str) or not isinstance(content, dict):
                continue
            restored[entry_id] = {
                "document": content.get("document"),
                "metadata": dict(content.get("metadata") or {}),
                "embedding": list(content.get("embedding") or []),
            }
        self._entries = restored
