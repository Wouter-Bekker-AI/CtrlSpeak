# -*- coding: utf-8 -*-
"""Per-identity memory configuration helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from utils.config_paths import get_config_dir
from utils.io_atomic import atomic_write_text


_IDENTITIES_DIR = "identities"
_MEMORY_SETTINGS_FILE = "memory.json"

DEFAULT_IDENTITY_SETTINGS: Dict[str, Any] = {
    "store_vector_memory": True,
    "store_screenshots": True,
    "retrieval_top_k": 5,
    "retrieval_threshold": 0.75,
    "max_vector_items": 5000,
    "vector_ttl_days": None,
    "pii_redaction": False,
}


def _sanitize(identity: str) -> str:
    cleaned = (identity or "").strip()
    if not cleaned:
        return "reception"
    sanitized = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in cleaned)
    return sanitized[:255] or "reception"


def get_identity_settings_dir(identity: str) -> Path:
    root = get_config_dir() / _IDENTITIES_DIR / _sanitize(identity)
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_identity_settings_path(identity: str) -> Path:
    return get_identity_settings_dir(identity) / _MEMORY_SETTINGS_FILE


def load_identity_settings(identity: str) -> Dict[str, Any]:
    path = get_identity_settings_path(identity)
    if not path.exists():
        return dict(DEFAULT_IDENTITY_SETTINGS)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(DEFAULT_IDENTITY_SETTINGS)
    settings: Dict[str, Any] = dict(DEFAULT_IDENTITY_SETTINGS)
    if isinstance(payload, dict):
        settings.update(payload)
    return settings


def save_identity_settings(identity: str, overrides: Dict[str, Any]) -> None:
    data = dict(DEFAULT_IDENTITY_SETTINGS)
    data.update(overrides)
    path = get_identity_settings_path(identity)
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True))


__all__ = [
    "DEFAULT_IDENTITY_SETTINGS",
    "get_identity_settings_dir",
    "get_identity_settings_path",
    "load_identity_settings",
    "save_identity_settings",
]
