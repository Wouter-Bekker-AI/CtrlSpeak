# -*- coding: utf-8 -*-
"""Chroma-backed vector memory management for CtrlSpeak."""
from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Documents, Embeddings
from chromadb.utils.embedding_functions import EmbeddingFunction

from utils.io_atomic import atomic_write_text
from utils.memory_paths import get_bot_chroma_dir
from utils.pii import redact_iterable

_STATE_FILE = "collection_state.json"

DEFAULT_EMBEDDER_NAME = "ctrlspeak-minhash"
DEFAULT_EMBEDDER_VERSION = "1"


def _utc_now() -> datetime:
    return datetime.utcnow()


def _iso(ts: datetime) -> str:
    return ts.isoformat(timespec="milliseconds") + "Z"


def _sanitize_identity(identity: str) -> str:
    normalized = (identity or "").strip()
    if not normalized:
        return "default"
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in normalized)[:255] or "default"


class _HashEmbeddingFunction(EmbeddingFunction[Iterable[str]]):
    """Deterministic lightweight embedding function."""

    def __init__(self, dimensions: int = 32) -> None:
        self.dimensions = dimensions

    def __call__(self, input: Documents) -> Embeddings:  # type: ignore[override]
        texts = list(input)
        vectors: List[List[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            floats: List[float] = []
            for index in range(self.dimensions):
                start = (index * 2) % len(digest)
                chunk = digest[start : start + 2]
                value = int.from_bytes(chunk, "big") / 65535.0
                floats.append(value)
            vectors.append(floats)
        return vectors

    @staticmethod
    def name() -> str:
        return "ctrlspeak-hash"

    def default_space(self) -> str:
        return "cosine"

    def is_legacy(self) -> bool:  # pragma: no cover - override for clarity
        return False

    def get_config(self) -> Dict[str, Any]:  # pragma: no cover - deterministic
        return {"dimensions": self.dimensions}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "_HashEmbeddingFunction":  # pragma: no cover - deterministic
        dimensions = int(config.get("dimensions", 32))
        return _HashEmbeddingFunction(dimensions=dimensions)


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class RetrievedMemory:
    content: str
    metadata: Dict[str, Any]
    similarity: float


class VectorMemoryStore:
    """Manage a Chroma collection per identity with retention policies."""

    def __init__(
        self,
        identity: str,
        *,
        embedder_name: str = DEFAULT_EMBEDDER_NAME,
        embedder_version: str = DEFAULT_EMBEDDER_VERSION,
    ) -> None:
        self.identity = identity
        self.identity_key = _sanitize_identity(identity)
        self.embedder_name = embedder_name
        self.embedder_version = embedder_version
        self.embedder_id = f"{embedder_name}:{embedder_version}"
        self.persistence_dir = get_bot_chroma_dir(identity)
        self.client = chromadb.PersistentClient(path=str(self.persistence_dir))
        self.embedding_function = _HashEmbeddingFunction()
        self.collection = self._ensure_collection()
        self._state_path = self.persistence_dir / _STATE_FILE
        self._lock = threading.Lock()
        self._state = self._load_state()

    # ------------------------------------------------------------------
    # Internal state helpers
    # ------------------------------------------------------------------
    def _load_state(self) -> Dict[str, Any]:
        if not self._state_path.exists():
            return {"next_sequence": 0}
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:
            return {"next_sequence": 0}
        if not isinstance(payload, dict):
            return {"next_sequence": 0}
        payload.setdefault("next_sequence", 0)
        return payload

    def _save_state(self) -> None:
        atomic_write_text(self._state_path, json.dumps(self._state, indent=2, sort_keys=True))

    def _next_sequence(self) -> int:
        with self._lock:
            seq = int(self._state.get("next_sequence", 0))
            self._state["next_sequence"] = seq + 1
            self._save_state()
        return seq

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------
    def _collection_matches(self, collection: Collection) -> bool:
        metadata = collection.metadata or {}
        return metadata.get("embedder_id") == self.embedder_id

    def _collection_metadata(self) -> Dict[str, Any]:
        return {
            "identity": self.identity_key,
            "embedder_name": self.embedder_name,
            "embedder_version": self.embedder_version,
            "embedder_id": self.embedder_id,
        }

    def _ensure_collection(self) -> Collection:
        base_name = self.identity_key
        candidates = self.client.list_collections()
        matched: Optional[Collection] = None
        max_version = 1
        for coll in candidates:
            name = coll.name
            if not name.startswith(base_name):
                continue
            version = 1
            if name != base_name and name.startswith(f"{base_name}_v"):
                try:
                    version = int(name.split("_v", 1)[1])
                except Exception:
                    version = 1
            max_version = max(max_version, version)
            if self._collection_matches(coll):
                matched = coll
        if matched is not None:
            return self.client.get_collection(
                matched.name,
                embedding_function=self.embedding_function,
            )

        next_version = max_version + 1 if any(c.name.startswith(base_name) for c in candidates) else 1
        name = base_name if next_version == 1 else f"{base_name}_v{next_version}"
        return self.client.get_or_create_collection(
            name=name,
            metadata=self._collection_metadata(),
            embedding_function=self.embedding_function,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def count(self) -> int:
        return self.collection.count()

    def retrieve(self, query_text: str, *, top_k: int, threshold: float) -> List[RetrievedMemory]:
        if not query_text or self.count() == 0:
            return []
        query_vector = self.embedding_function([query_text])[0]
        payload = self.collection.get(include=["documents", "metadatas", "embeddings"])
        documents = payload.get("documents") or []
        metadatas = payload.get("metadatas") or []
        embeddings = payload.get("embeddings")
        if embeddings is None:
            embeddings = []
        results: List[RetrievedMemory] = []
        for doc, metadata, embedding in zip(documents, metadatas, embeddings):
            if not isinstance(doc, str) or not isinstance(metadata, dict):
                continue
            similarity = _cosine_similarity(query_vector, embedding)
            if similarity < threshold:
                continue
            results.append(RetrievedMemory(doc, metadata, similarity))
        results.sort(key=lambda item: item.similarity, reverse=True)
        return results[:top_k]

    def purge_expired(self) -> int:
        payload = self.collection.get(include=["metadatas"])
        metadatas = payload.get("metadatas") or []
        ids = payload.get("ids") or []
        now = _utc_now()
        expired: List[str] = []
        for entry_id, metadata in zip(ids, metadatas):
            expiry = metadata.get("expires_at") if isinstance(metadata, dict) else None
            if not expiry:
                continue
            try:
                expiry_dt = datetime.fromisoformat(str(expiry).replace("Z", ""))
            except Exception:
                continue
            if expiry_dt <= now:
                expired.append(entry_id)
        if not expired:
            return 0
        self.collection.delete(ids=expired)
        return len(expired)

    def add_memories(
        self,
        documents: List[str],
        *,
        metadata: Optional[List[Dict[str, Any]]] = None,
        max_items: int = 5000,
        ttl_days: Optional[int] = None,
        pii_redaction: bool = False,
    ) -> Dict[str, Any]:
        if not documents:
            return {"evicted": 0}
        metadata = metadata or [{} for _ in documents]
        if len(metadata) != len(documents):
            raise ValueError("metadata length must match documents length")
        created_at = _iso(_utc_now())
        redacted_docs = redact_iterable(documents, enabled=pii_redaction)
        entries_ids: List[str] = []
        prepared_metadata: List[Dict[str, Any]] = []
        for doc_meta in metadata:
            meta = dict(doc_meta)
            meta.setdefault("created_at", created_at)
            seq = self._next_sequence()
            meta.setdefault("sequence", seq)
            if ttl_days is not None:
                expires = _utc_now() + timedelta(days=float(ttl_days))
                meta["expires_at"] = _iso(expires)
            prepared_metadata.append(meta)
            entries_ids.append(f"{self.identity_key}-{meta['sequence']}")

        embeddings = self.embedding_function(redacted_docs)
        self.collection.add(
            ids=entries_ids,
            documents=redacted_docs,
            metadatas=prepared_metadata,
            embeddings=embeddings,
        )
        evicted = self._enforce_limits(max_items=max_items)
        return {"evicted": evicted}

    def _enforce_limits(self, *, max_items: int) -> int:
        evicted = 0
        total = self.collection.count()
        if total <= max_items:
            return 0
        payload = self.collection.get(include=["metadatas"], limit=total)
        metadatas = payload.get("metadatas") or []
        ids = payload.get("ids") or []
        sortable = []
        for entry_id, metadata in zip(ids, metadatas):
            if not isinstance(metadata, dict):
                continue
            sortable.append((int(metadata.get("sequence", 0)), entry_id))
        sortable.sort(key=lambda item: item[0])
        to_remove = [entry_id for _, entry_id in sortable[: max(0, total - max_items)]]
        if to_remove:
            self.collection.delete(ids=to_remove)
            evicted = len(to_remove)
        return evicted

    def compact(self) -> None:
        self.client.persist()

    def drop_and_recreate(self) -> None:
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata=self._collection_metadata(),
            embedding_function=self.embedding_function,
        )
        self._state = {"next_sequence": 0}
        self._save_state()


__all__ = ["VectorMemoryStore", "RetrievedMemory", "DEFAULT_EMBEDDER_NAME", "DEFAULT_EMBEDDER_VERSION"]
