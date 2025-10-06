"""
Database utilities for ChromaDB.
"""

import os
from pathlib import Path

import chromadb

_client = None


def _resolve_persist_directory() -> Path:
    """Return the directory ChromaDB should use for persistence."""
    env_path = os.environ.get("CHROMA_PERSIST_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()

    project_root = Path(__file__).resolve().parents[1]
    return project_root / "chroma_storage"


def initialize_database() -> None:
    """Initialize the global ChromaDB client if needed."""
    global _client
    if _client is not None:
        return

    persist_path = _resolve_persist_directory()
    persist_path.mkdir(parents=True, exist_ok=True)
    _client = chromadb.PersistentClient(path=str(persist_path))


def get_client():
    """Return the initialized ChromaDB client."""
    initialize_database()
    return _client


def close_database() -> None:
    """Release the ChromaDB client reference (noop for persistent client)."""
    global _client
    _client = None
