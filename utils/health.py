# -*- coding: utf-8 -*-
"""Health diagnostics for CtrlSpeak memory orchestration."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from utils.config_paths import get_config_dir, get_data_dir, get_logs_dir
from utils.memory_lock import IdentityLock, IdentityLockError, probe_lock_path
from utils.memory_paths import get_bot_chroma_dir, get_identity_lock_path
from utils.vector_memory import VectorMemoryStore


def _check_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        test_file = path / ".__ctrlspeak_health"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def run_health_check(identity: str = "default") -> Dict[str, object]:
    data_root = get_data_dir()
    config_root = get_config_dir()
    logs_root = get_logs_dir()

    status = "ok"
    details: Dict[str, object] = {
        "data_root": str(data_root),
        "config_root": str(config_root),
        "logs_root": str(logs_root),
    }

    details["data_root_writable"] = _check_writable(data_root)
    details["logs_root_writable"] = _check_writable(logs_root)
    if not details["data_root_writable"] or not details["logs_root_writable"]:
        status = "error"

    lock_path = get_identity_lock_path(identity)
    lock_info = {
        "path": str(lock_path),
        "available": probe_lock_path(lock_path),
    }
    try:
        lock = IdentityLock(identity)
        lock.acquire(timeout=0.0)
        lock.release()
        lock_info["acquired"] = True
    except IdentityLockError:
        lock_info["acquired"] = False
        status = "error"
    except Exception as exc:
        lock_info["error"] = str(exc)
        status = "error"
    details["identity_lock"] = lock_info

    chroma_dir = get_bot_chroma_dir(identity)
    details["chroma_dir"] = str(chroma_dir)
    try:
        store = VectorMemoryStore(identity)
        collection_meta = store.collection.metadata or {}
        details["embedder_id"] = collection_meta.get("embedder_id")
        details["vector_count"] = store.count()
    except Exception as exc:
        details["chroma_error"] = str(exc)
        status = "error"

    return {
        "status": status,
        "identity": identity,
        "details": details,
    }


__all__ = ["run_health_check"]
