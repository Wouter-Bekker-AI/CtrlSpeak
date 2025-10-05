# -*- coding: utf-8 -*-
"""Helpers for resolving per-identity memory paths."""
from __future__ import annotations

import re
from pathlib import Path

from utils.config_paths import get_data_dir


_MEM_ROOT_NAME = "bot_memory"
_CONVERSATION_DIR = "conversation"
_SCREENSHOTS_DIR = "screenshots"
_CHROMA_DIR = "chroma"
_TRACES_DIR = "traces"
_LOCKS_DIR = ".locks"
_CONVERSATION_LOG = "conversation.jsonl"
_PROFILE_EXPORT = "profile_snapshot.json"


def _sanitize_identity(identity: str) -> str:
    normalized = (identity or "").strip()
    if not normalized:
        return "reception"
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized)
    return sanitized[:255] or "reception"


def get_bot_memory_dir(identity: str) -> Path:
    base = get_data_dir() / _MEM_ROOT_NAME / _sanitize_identity(identity)
    for sub in (_CONVERSATION_DIR, _SCREENSHOTS_DIR, _CHROMA_DIR, _TRACES_DIR):
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base


def get_bot_conversation_dir(identity: str) -> Path:
    return get_bot_memory_dir(identity) / _CONVERSATION_DIR


def get_bot_conversation_log(identity: str) -> Path:
    return get_bot_conversation_dir(identity) / _CONVERSATION_LOG


def get_bot_screenshots_dir(identity: str) -> Path:
    return get_bot_memory_dir(identity) / _SCREENSHOTS_DIR


def get_bot_chroma_dir(identity: str) -> Path:
    return get_bot_memory_dir(identity) / _CHROMA_DIR


def get_bot_traces_dir(identity: str) -> Path:
    return get_bot_memory_dir(identity) / _TRACES_DIR


def get_identity_lock_path(identity: str) -> Path:
    root = get_data_dir() / _LOCKS_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{_sanitize_identity(identity)}.lock"


def get_bot_profile_export_path(identity: str) -> Path:
    return get_bot_memory_dir(identity) / _PROFILE_EXPORT
