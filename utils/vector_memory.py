# -*- coding: utf-8 -*-
"""Chroma-backed vector memory management for CtrlSpeak."""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Documents, Embeddings
from chromadb.utils.embedding_functions import EmbeddingFunction

from utils.config_paths import get_logger
from utils.io_atomic import atomic_write_text
from utils.memory_paths import get_bot_chroma_dir
from utils.pii import redact_iterable

_STATE_FILE = "collection_state.json"

DEFAULT_EMBEDDER_NAME = "ctrlspeak-minhash"
DEFAULT_EMBEDDER_VERSION = "2"

PROFILE_COLLECTION_SUFFIX = "_profile"
PROFILE_SCHEMA_VERSION = 1
DEFAULT_PROFILE_EMBEDDER_NAME = "semantic-e5-small"
DEFAULT_PROFILE_EMBEDDER_VERSION = "1"
PROFILE_MODEL_NAME = "intfloat/e5-small-v2"
PROFILE_RERANKER_MODEL = "BAAI/bge-reranker-base"

logger = get_logger(__name__)


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
    """Deterministic hashed bag-of-words embedding."""

    def __init__(self, dimensions: int = 128, *, ngram_min: int = 1, ngram_max: int = 2) -> None:
        self.dimensions = max(1, int(dimensions))
        self.ngram_min = max(1, int(min(ngram_min, ngram_max)))
        self.ngram_max = max(self.ngram_min, int(max(ngram_min, ngram_max)))

    def _iter_features(self, text: str) -> Iterable[str]:
        tokens = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text.lower())
        if not tokens:
            return []
        features: List[str] = []
        for size in range(self.ngram_min, self.ngram_max + 1):
            if len(tokens) < size:
                continue
            for index in range(len(tokens) - size + 1):
                features.append(" ".join(tokens[index : index + size]))
        return features or tokens

    def __call__(self, input: Documents) -> Embeddings:  # type: ignore[override]
        vectors: List[List[float]] = []
        for raw in input:
            text = str(raw or "")
            features = list(self._iter_features(text))
            if not features:
                vectors.append([0.0] * self.dimensions)
                continue
            values = [0.0] * self.dimensions
            for feature in features:
                digest = hashlib.blake2s(feature.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if (digest[4] & 0x01) == 0 else -1.0
                values[index] += sign
            norm = math.sqrt(sum(value * value for value in values))
            if norm:
                values = [value / norm for value in values]
            vectors.append(values)
        return vectors

    @staticmethod
    def name() -> str:
        return "ctrlspeak-hash"

    def default_space(self) -> str:
        return "cosine"

    def is_legacy(self) -> bool:  # pragma: no cover - override for clarity
        return False

    def get_config(self) -> Dict[str, Any]:  # pragma: no cover - deterministic
        return {
            "dimensions": self.dimensions,
            "ngram_min": self.ngram_min,
            "ngram_max": self.ngram_max,
        }

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "_HashEmbeddingFunction":  # pragma: no cover - deterministic
        dimensions = int(config.get("dimensions", 128))
        ngram_min = int(config.get("ngram_min", 1))
        ngram_max = int(config.get("ngram_max", max(ngram_min, 2)))
        return _HashEmbeddingFunction(dimensions=dimensions, ngram_min=ngram_min, ngram_max=ngram_max)


class SemanticEmbeddingFunction(EmbeddingFunction[Iterable[str]]):
    """Sentence-embedding wrapper with graceful fallback."""

    def __init__(
        self,
        model_name: str = "intfloat/e5-small-v2",
        *,
        device: Optional[str] = None,
        normalize: bool = True,
        fallback_dimensions: int = 384,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.normalize = bool(normalize)
        self.fallback_dimensions = max(1, int(fallback_dimensions))
        self._model = None
        self._model_lock = threading.Lock()
        self._fallback = _HashEmbeddingFunction(dimensions=self.fallback_dimensions, ngram_min=1, ngram_max=2)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:  # pragma: no cover - optional dependency
                logger.debug("Semantic embedding unavailable (sentence-transformers import failed): %s", exc)
                self._model = None
                return
            try:
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as exc:  # pragma: no cover - runtime download failure
                logger.warning("Failed to load semantic embedding model '%s': %s", self.model_name, exc)
                self._model = None

    def __call__(self, input: Documents) -> Embeddings:  # type: ignore[override]
        self._load_model()
        if self._model is None:
            return self._fallback(input)
        try:
            embeddings = self._model.encode(
                list(input),
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                device=self.device,
            )
        except Exception as exc:  # pragma: no cover - runtime issues
            logger.warning("Semantic embedding encode failed, falling back to hash: %s", exc)
            return self._fallback(input)
        return embeddings.tolist()

    @staticmethod
    def name() -> str:  # pragma: no cover - metadata only
        return "ctrlspeak-semantic"

    def default_space(self) -> str:  # pragma: no cover - metadata only
        return "cosine"

    def is_legacy(self) -> bool:  # pragma: no cover - metadata only
        return False

    def get_config(self) -> Dict[str, Any]:  # pragma: no cover - deterministic metadata
        return {
            "model_name": self.model_name,
            "device": self.device,
            "normalize": self.normalize,
            "fallback_dimensions": self.fallback_dimensions,
        }


class _CrossEncoderReranker:
    """Optional cross-encoder reranker for short profile facts."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", *, device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._lock = threading.Lock()
        self._load_attempted = False

    def _ensure_model(self) -> None:
        if self._model is not None or self._load_attempted:
            return
        with self._lock:
            if self._model is not None or self._load_attempted:
                return
            self._load_attempted = True
            try:
                from sentence_transformers import CrossEncoder
            except Exception as exc:  # pragma: no cover - optional dependency
                logger.debug("Cross-encoder reranker unavailable: %s", exc)
                self._model = None
                return
            try:
                self._model = CrossEncoder(self.model_name, device=self.device)
            except Exception as exc:  # pragma: no cover - runtime load failure
                logger.warning("Failed to load cross-encoder '%s': %s", self.model_name, exc)
                self._model = None

    def rerank(self, query: str, candidates: Sequence[Tuple[str, Dict[str, Any], float]]) -> List[Tuple[str, Dict[str, Any], float]]:
        self._ensure_model()
        if self._model is None or not candidates:
            return list(candidates)
        try:
            pairs = [(query, doc) for doc, _meta, _score in candidates]
            scores = self._model.predict(pairs)
        except Exception as exc:  # pragma: no cover - runtime encode issues
            logger.debug("Cross-encoder prediction failed, skipping rerank: %s", exc)
            return list(candidates)
        scored = []
        for (doc, meta, _score), new_score in zip(candidates, scores):
            scored.append((doc, meta, float(new_score)))
        scored.sort(key=lambda item: item[2], reverse=True)
        return scored

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
        self.profile_embedder_name = DEFAULT_PROFILE_EMBEDDER_NAME
        self.profile_embedder_version = DEFAULT_PROFILE_EMBEDDER_VERSION
        self.profile_embedder_id = f"{self.profile_embedder_name}:{self.profile_embedder_version}"
        self.profile_embedding_function = SemanticEmbeddingFunction(model_name=PROFILE_MODEL_NAME)
        self._profile_reranker = _CrossEncoderReranker(model_name=PROFILE_RERANKER_MODEL)
        self.profile_collection = self._ensure_profile_collection()
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
            "profile_schema_version": PROFILE_SCHEMA_VERSION,
        }

    def _profile_collection_matches(self, collection: Collection) -> bool:
        metadata = collection.metadata or {}
        return (
            metadata.get("embedder_id") == self.profile_embedder_id
            and int(metadata.get("profile_schema_version", 0)) == PROFILE_SCHEMA_VERSION
            and metadata.get("collection_kind") == "profile"
        )

    def _profile_collection_metadata(self) -> Dict[str, Any]:
        return {
            "identity": self.identity_key,
            "embedder_name": self.profile_embedder_name,
            "embedder_version": self.profile_embedder_version,
            "embedder_id": self.profile_embedder_id,
            "collection_kind": "profile",
            "profile_schema_version": PROFILE_SCHEMA_VERSION,
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

    def _ensure_profile_collection(self) -> Collection:
        base_name = f"{self.identity_key}{PROFILE_COLLECTION_SUFFIX}"
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
            if self._profile_collection_matches(coll):
                matched = coll
        if matched is not None:
            return self.client.get_collection(
                matched.name,
                embedding_function=self.profile_embedding_function,
            )

        next_version = max_version + 1 if any(c.name.startswith(base_name) for c in candidates) else 1
        name = base_name if next_version == 1 else f"{base_name}_v{next_version}"
        return self.client.get_or_create_collection(
            name=name,
            metadata=self._profile_collection_metadata(),
            embedding_function=self.profile_embedding_function,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def count(self) -> int:
        return self.collection.count()

    @staticmethod
    def _normalize_profile_attribute(attribute: str) -> str:
        return (attribute or "").strip().lower()

    @staticmethod
    def _profile_document(attribute: str, value: str) -> str:
        readable_attribute = attribute.replace("_", " ")
        return f"The user's {readable_attribute} is {value}.".strip()

    def _load_profile_entries(self) -> List[Dict[str, Any]]:
        payload = self.profile_collection.get(
            include=["ids", "documents", "metadatas"],
        )
        ids = payload.get("ids") or []
        documents = payload.get("documents") or []
        metadatas = payload.get("metadatas") or []
        entries: List[Dict[str, Any]] = []
        for entry_id, document, metadata in zip(ids, documents, metadatas):
            entries.append(
                {
                    "id": entry_id,
                    "document": document or "",
                    "metadata": metadata or {},
                }
            )
        return entries

    @staticmethod
    def _filter_profile_entries(
        entries: Iterable[Dict[str, Any]],
        *,
        user_id: str,
        attribute: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        normalized_user = str(user_id or "user").strip() or "user"
        normalized_attribute = (attribute or "").strip().lower()
        normalized_status = (status or "").strip().lower()
        filtered: List[Dict[str, Any]] = []
        for entry in entries:
            metadata = entry.get("metadata") or {}
            entry_user = str(metadata.get("user_id", "")).strip() or "user"
            if entry_user != normalized_user:
                continue
            if normalized_attribute:
                entry_attribute = str(metadata.get("attribute", "")).strip().lower()
                if entry_attribute != normalized_attribute:
                    continue
            if normalized_status:
                entry_status = str(metadata.get("status", "")).strip().lower()
                if entry_status != normalized_status:
                    continue
            filtered.append(entry)
        filtered.sort(
            key=lambda item: str((item.get("metadata") or {}).get("updated_at", "")),
            reverse=True,
        )
        return filtered

    @staticmethod
    def _profile_history_suffix(metadata: Dict[str, Any], fallback: str) -> str:
        for key in ("updated_at", "valid_from", "created_at"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return value.replace(":", "-")
        return fallback.replace(":", "-")

    def upsert_profile_slot(
        self,
        user_id: str,
        attribute: str,
        value: str,
        source: str,
        *,
        valid_from: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        normalized_user = str(user_id or "user").strip() or "user"
        normalized_attribute = self._normalize_profile_attribute(attribute)
        normalized_value = str(value or "").strip()
        if not normalized_attribute:
            raise ValueError("attribute is required for profile upsert")
        if not normalized_value:
            raise ValueError("value is required for profile upsert")
        now = _utc_now()
        created_at = _iso(now)
        valid_from_iso = _iso(valid_from) if isinstance(valid_from, datetime) else created_at
        document = self._profile_document(normalized_attribute, normalized_value)
        slot_id = f"{normalized_user}:{normalized_attribute}"
        collection = self.profile_collection

        existing_entries = [
            entry for entry in self._load_profile_entries() if entry.get("id") == slot_id
        ]
        if existing_entries:
            existing_doc = existing_entries[0].get("document", "")
            existing_meta_raw = existing_entries[0].get("metadata") or {}
            existing_meta = dict(existing_meta_raw)
            existing_meta["status"] = "superseded"
            existing_meta["superseded_at"] = created_at
            existing_meta.setdefault("user_id", normalized_user)
            existing_meta.setdefault("attribute", normalized_attribute)
            history_suffix = self._profile_history_suffix(existing_meta, created_at)
            history_id = f"{slot_id}#{history_suffix}"
            history_embedding = self.profile_embedding_function([existing_doc])[0]
            collection.add(
                ids=[history_id],
                documents=[existing_doc],
                metadatas=[existing_meta],
                embeddings=[history_embedding],
            )
            collection.delete(ids=[slot_id])

        metadata = {
            "identity": self.identity_key,
            "user_id": normalized_user,
            "entity": "user",
            "attribute": normalized_attribute,
            "value": normalized_value,
            "status": "current",
            "source": source or "unspecified",
            "created_at": created_at,
            "updated_at": created_at,
            "valid_from": valid_from_iso,
            "embedder_id": self.profile_embedder_id,
            "profile_schema_version": PROFILE_SCHEMA_VERSION,
        }
        embedding = self.profile_embedding_function([document])[0]
        collection.add(
            ids=[slot_id],
            documents=[document],
            metadatas=[metadata],
            embeddings=[embedding],
        )
        return metadata

    def read_profile_slot(self, user_id: str, attribute: str) -> Optional[Dict[str, Any]]:
        normalized_user = str(user_id or "user").strip() or "user"
        normalized_attribute = self._normalize_profile_attribute(attribute)
        if not normalized_attribute:
            return None
        entries = self._filter_profile_entries(
            self._load_profile_entries(),
            user_id=normalized_user,
            attribute=normalized_attribute,
            status="current",
        )
        if not entries:
            return None
        first = entries[0]
        return {
            "id": first.get("id"),
            "document": first.get("document", ""),
            "metadata": first.get("metadata", {}),
        }

    def read_all_profile(self, user_id: str) -> List[Dict[str, Any]]:
        normalized_user = str(user_id or "user").strip() or "user"
        entries = self._filter_profile_entries(
            self._load_profile_entries(),
            user_id=normalized_user,
            status="current",
        )
        return [
            {
                "id": entry.get("id"),
                "document": entry.get("document", ""),
                "metadata": entry.get("metadata", {}),
            }
            for entry in entries
        ]

    def query_profile(
        self,
        query_text: str,
        user_id: str,
        *,
        attribute: Optional[str] = None,
        k: int = 3,
        rerank: bool = False,
    ) -> List[RetrievedMemory]:
        query = str(query_text or "").strip()
        if not query:
            return []
        normalized_user = str(user_id or "user").strip() or "user"
        normalized_attribute = self._normalize_profile_attribute(attribute) if attribute else None
        entries = self._filter_profile_entries(
            self._load_profile_entries(),
            user_id=normalized_user,
            attribute=normalized_attribute,
            status="current",
        )
        if not entries:
            return []
        query_vector = self.profile_embedding_function([query])[0]
        candidates: List[Tuple[str, Dict[str, Any], float]] = []
        for entry in entries:
            document = str(entry.get("document", ""))
            metadata = dict(entry.get("metadata") or {})
            doc_vector = self.profile_embedding_function([document])[0]
            similarity = _cosine_similarity(query_vector, doc_vector)
            candidates.append((document, metadata, similarity))
        candidates.sort(key=lambda item: item[2], reverse=True)
        if rerank and len(candidates) > 1:
            candidates = self._profile_reranker.rerank(query, candidates)
        memories: List[RetrievedMemory] = []
        for doc, meta, similarity in candidates[: max(1, int(k))]:
            memories.append(RetrievedMemory(doc, meta, similarity))
        return memories

    def retrieve(
        self,
        query_text: str,
        *,
        top_k: int,
        threshold: float,
        category_thresholds: Optional[Dict[str, float]] = None,
        fallback_categories: Optional[Dict[str, int]] = None,
    ) -> List[RetrievedMemory]:
        if not query_text or self.count() == 0:
            return []

        category_thresholds = {
            str(key): float(value)
            for key, value in (category_thresholds or {}).items()
        }
        fallback_limits = {
            str(key): max(0, int(value))
            for key, value in (fallback_categories or {}).items()
        }

        threshold = float(threshold)
        query_vector = self.embedding_function([query_text])[0]

        effective_top_k = max(0, int(top_k))
        fallback_total = sum(fallback_limits.values()) if fallback_limits else 0
        initial_limit = effective_top_k + fallback_total
        if initial_limit <= 0:
            return []

        include_fields = ["documents", "metadatas", "distances"]

        def similarity_from_distance(distance: Any) -> float:
            try:
                value = 1.0 - float(distance)
            except Exception:
                return 0.0
            return max(min(value, 1.0), -1.0)

        def make_key(metadata: Dict[str, Any], document: str) -> Any:
            sequence = metadata.get("sequence") if isinstance(metadata, dict) else None
            created = metadata.get("created_at") if isinstance(metadata, dict) else None
            category = metadata.get("category") if isinstance(metadata, dict) else None
            doc_hash = metadata.get("doc_hash") if isinstance(metadata, dict) else None
            return (sequence, created, category, doc_hash, document)

        results: List[RetrievedMemory] = []
        fallback_pool: Dict[str, List[RetrievedMemory]] = {key: [] for key in fallback_limits}
        seen_keys: set = set()

        def ingest_payload(payload: Dict[str, Any]) -> None:
            if not payload:
                return
            docs_list = payload.get("documents") or []
            metas_list = payload.get("metadatas") or []
            dist_list = payload.get("distances") or []
            if not docs_list or not metas_list:
                return
            # Chroma responses return lists per query; we only issue single queries.
            documents_inner = docs_list[0] if isinstance(docs_list[0], list) else docs_list
            metadatas_inner = metas_list[0] if isinstance(metas_list[0], list) else metas_list
            distances_inner_raw = dist_list[0] if dist_list and isinstance(dist_list[0], list) else dist_list
            distances_inner = list(distances_inner_raw) if isinstance(distances_inner_raw, IterableABC) else []
            if len(distances_inner) < len(documents_inner):
                distances_inner.extend([0.0] * (len(documents_inner) - len(distances_inner)))

            for doc, metadata, distance in zip(documents_inner, metadatas_inner, distances_inner):
                if not isinstance(doc, str) or not isinstance(metadata, dict):
                    continue
                key = make_key(metadata, doc)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                similarity = similarity_from_distance(distance)
                category = str(metadata.get("category", "")) if metadata else ""
                category_threshold = category_thresholds.get(category, threshold)
                memory = RetrievedMemory(doc, metadata, similarity)
                if similarity >= category_threshold:
                    results.append(memory)
                elif category in fallback_limits:
                    fallback_pool.setdefault(category, []).append(memory)

        payload = self.collection.query(
            query_embeddings=[query_vector],
            n_results=initial_limit,
            include=include_fields,
        )
        ingest_payload(payload)

        if fallback_limits:
            for category, limit in fallback_limits.items():
                if limit <= 0:
                    continue
                while True:
                    category_results = [
                        item for item in results if str(item.metadata.get("category", "")) == category
                    ]
                    pool = fallback_pool.get(category, [])
                    needed = limit - len(category_results)
                    if needed <= 0 or len(pool) >= needed:
                        break
                    additional_needed = needed - len(pool)
                    if additional_needed <= 0:
                        break
                    extra_payload = self.collection.query(
                        query_embeddings=[query_vector],
                        n_results=additional_needed,
                        include=include_fields,
                        where={"category": category},
                    )
                    before = len(fallback_pool.get(category, []))
                    ingest_payload(extra_payload)
                    after = len(fallback_pool.get(category, []))
                    if after <= before:
                        break

        if fallback_pool:
            for category, pool in fallback_pool.items():
                pool.sort(key=lambda item: item.similarity, reverse=True)

        # Sort primary matches by similarity before enforcing fallbacks.
        results.sort(key=lambda item: item.similarity, reverse=True)

        if fallback_limits:
            selected_ids = set()
            quota_selected: List[RetrievedMemory] = []
            for category, limit in fallback_limits.items():
                if limit <= 0:
                    continue
                category_matches = [
                    item
                    for item in results
                    if str(item.metadata.get("category", "")) == category
                ]
                if len(category_matches) < limit:
                    pool = fallback_pool.get(category, [])
                    needed = limit - len(category_matches)
                    category_matches.extend(pool[:needed])
                for item in category_matches[:limit]:
                    quota_selected.append(item)
                    selected_ids.add(id(item))

            remaining = [item for item in results if id(item) not in selected_ids]
            remaining.sort(key=lambda item: item.similarity, reverse=True)

            combined: List[RetrievedMemory] = []
            for item in quota_selected:
                if item not in combined:
                    combined.append(item)
            for item in remaining:
                if len(combined) >= top_k:
                    break
                combined.append(item)

            results = combined

        results.sort(key=lambda item: item.similarity, reverse=True)
        if len(results) > top_k:
            results = results[:top_k]
        return results

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


__all__ = [
    "VectorMemoryStore",
    "RetrievedMemory",
    "DEFAULT_EMBEDDER_NAME",
    "DEFAULT_EMBEDDER_VERSION",
    "DEFAULT_PROFILE_EMBEDDER_NAME",
    "DEFAULT_PROFILE_EMBEDDER_VERSION",
    "PROFILE_SCHEMA_VERSION",
]
