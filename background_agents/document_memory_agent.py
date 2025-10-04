# -*- coding: utf-8 -*-
"""Helpers for preloading documentation into vector memory."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from utils.config_paths import get_app_base_dir, get_data_dir, get_logger
from utils.io_atomic import atomic_write_text
from utils.memory_settings import load_identity_settings
from utils.vector_memory import VectorMemoryStore

_LOGGER = get_logger(__name__)

_DOCUMENT_CATEGORY = "documentation"
_TRACKER_DIR = "doc_memory"
_COOLDOWN = timedelta(hours=24)
_DEFAULT_CHARS_PER_CHUNK = 1200


def _sanitize_identity(identity: str) -> str:
    normalized = (identity or "").strip()
    if not normalized:
        return "reception"
    allowed = [ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in normalized]
    return ("".join(allowed) or "reception")[:255]


def _tracker_path(identity: str) -> Path:
    root = get_data_dir() / _TRACKER_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{_sanitize_identity(identity)}.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _from_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        cleaned = value.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned)
    except Exception:
        return None


def _load_tracker(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _LOGGER.debug("Failed to read documentation tracker at %s", path, exc_info=True)
        return {}


def _write_tracker(path: Path, payload: Dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


_SOURCE_ALLOWLIST = (
    Path("README.md"),
    Path("docs/bot_integration.md"),
    Path("docs/tooling.md"),
    Path("docs/user_flow.md"),
)


def _iter_markdown_files(base_dir: Path) -> Iterable[Path]:
    for relative in _SOURCE_ALLOWLIST:
        path = base_dir / relative
        if path.exists() and path.is_file():
            yield path


def _read_markdown_sources(base_dir: Path) -> List[Tuple[Path, str]]:
    sources: List[Tuple[Path, str]] = []
    for path in _iter_markdown_files(base_dir):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            _LOGGER.exception("Failed to read documentation file %s", path)
            continue
        if not text.strip():
            continue
        sources.append((path, text))
    return sources


def _chunk_text(text: str, *, max_chars: int = _DEFAULT_CHARS_PER_CHUNK) -> List[str]:
    stripped = text.strip()
    if not stripped:
        return []
    words = stripped.split()
    if not words:
        return []

    chunks: List[str] = []
    current: List[str] = []
    length = 0
    for word in words:
        word_len = len(word)
        separator = 1 if current else 0
        if current and length + word_len + separator > max_chars:
            chunks.append(" ".join(current))
            current = []
            length = 0
        current.append(word)
        length += word_len + separator
    if current:
        chunks.append(" ".join(current))
    return chunks


def _build_metadata(
    source: Path,
    doc_hash: str,
    total_chunks: int,
) -> Iterable[Dict[str, Any]]:
    relative: str
    base = get_app_base_dir()
    try:
        relative = str(source.relative_to(base))
    except ValueError:
        relative = str(source)
    for index in range(total_chunks):
        yield {
            "category": _DOCUMENT_CATEGORY,
            "source": relative,
            "chunk": index + 1,
            "chunks": total_chunks,
            "doc_hash": doc_hash,
        }


def _aggregate_documents(base_dir: Path) -> Tuple[List[str], List[Dict[str, Any]], str]:
    collected = _read_markdown_sources(base_dir)
    if not collected:
        return [], [], ""

    hash_builder = hashlib.sha256()
    documents: List[str] = []
    metadata: List[Dict[str, Any]] = []

    for path, text in collected:
        identifier = f"{path.as_posix()}\n".encode("utf-8")
        hash_builder.update(identifier)
        hash_builder.update(text.encode("utf-8"))
        chunks = _chunk_text(text)
        if not chunks:
            continue
        documents.extend(chunks)
        metadata.extend(_build_metadata(path, "", len(chunks)))

    doc_hash = hash_builder.hexdigest()
    for entry in metadata:
        entry["doc_hash"] = doc_hash
    return documents, metadata, doc_hash


def refresh_document_memory(
    identity: str,
    *,
    force: bool = False,
    reason: str | None = None,
) -> bool:
    """Refresh documentation embeddings for ``identity`` if necessary."""

    base_dir = get_app_base_dir()
    documents, metadata, doc_hash = _aggregate_documents(base_dir)
    tracker_path = _tracker_path(identity)
    tracker = _load_tracker(tracker_path)
    last_hash = tracker.get("doc_hash") if isinstance(tracker, dict) else None
    last_refresh = _from_iso(tracker.get("last_refreshed")) if isinstance(tracker, dict) else None

    if not documents:
        print("[DocMemory] No documentation sources found; skipping refresh.")
        now_iso = _to_iso(_utc_now())
        _write_tracker(tracker_path, {"last_refreshed": now_iso, "doc_hash": doc_hash})
        return True

    now = _utc_now()
    needs_refresh = False
    refresh_reason: str | None = None

    if force:
        needs_refresh = True
        refresh_reason = "Forced refresh requested"
    elif not last_refresh:
        needs_refresh = True
        refresh_reason = "No previous documentation refresh recorded"
    elif now - last_refresh >= _COOLDOWN:
        needs_refresh = True
        elapsed_hours = (now - last_refresh).total_seconds() / 3600.0
        refresh_reason = f"Docs last refreshed {elapsed_hours:.1f}h ago"

    if not needs_refresh and last_hash != doc_hash:
        needs_refresh = True
        refresh_reason = "Documentation content changed"

    if not needs_refresh:
        print("[DocMemory] Documentation memory already current; skipping refresh.")
        return True

    if reason:
        refresh_reason = f"{refresh_reason or 'Refreshing documentation memory'} ({reason})"

    if refresh_reason:
        print(f"[DocMemory] {refresh_reason}; injecting documentation now...")
    else:
        print("[DocMemory] Refreshing documentation memory...")

    settings = load_identity_settings(identity)
    if not settings.get("store_vector_memory", True):
        print(
            f"[DocMemory] Vector memory disabled for identity '{identity}'; "
            "skipping documentation injection.",
        )
        _write_tracker(tracker_path, {"last_refreshed": _to_iso(now), "doc_hash": doc_hash})
        return True

    max_items = int(settings.get("max_vector_items", 5000) or 5000)
    ttl_days = settings.get("vector_ttl_days")
    pii_redaction = bool(settings.get("pii_redaction", False))

    try:
        store = VectorMemoryStore(identity)
        try:
            store.collection.delete(where={"category": _DOCUMENT_CATEGORY})  # type: ignore[arg-type]
        except TypeError:
            try:
                payload = store.collection.get(include=["ids", "metadatas"])
            except Exception:
                payload = {}
            ids = []
            for entry_id, meta in zip(payload.get("ids", []), payload.get("metadatas", [])):
                if isinstance(meta, dict) and meta.get("category") == _DOCUMENT_CATEGORY:
                    ids.append(entry_id)
            if ids:
                try:
                    store.collection.delete(ids=ids)
                except Exception:
                    _LOGGER.debug(
                        "Fallback deletion failed for documentation entries on %s", identity, exc_info=True
                    )
        except Exception:
            _LOGGER.debug("No existing documentation entries to delete for %s", identity, exc_info=True)

        result = store.add_memories(
            documents,
            metadata=metadata,
            max_items=max_items,
            ttl_days=ttl_days,
            pii_redaction=pii_redaction,
        )
        evicted = result.get("evicted", 0)
        print(
            "[DocMemory] Documentation injection complete "
            f"({len(documents)} chunks, evicted={evicted}).",
        )
    except Exception:
        _LOGGER.exception("Failed to refresh documentation memory for %s", identity)
        print("[DocMemory] Documentation refresh failed; check logs for details.")
        return False

    _write_tracker(
        tracker_path,
        {"last_refreshed": _to_iso(now), "doc_hash": doc_hash},
    )
    return True


__all__ = ["refresh_document_memory"]
