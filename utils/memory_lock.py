# -*- coding: utf-8 -*-
"""Cross-process identity locking helpers."""
from __future__ import annotations

import atexit
from pathlib import Path
from typing import Optional

import portalocker

from utils.memory_paths import get_identity_lock_path


class IdentityLockError(RuntimeError):
    """Raised when acquiring the identity lock fails."""


class IdentityLock:
    """Manage an exclusive file lock for a bot identity."""

    def __init__(self, identity: str) -> None:
        self.identity = identity
        self.lock_path = get_identity_lock_path(identity)
        self._lock: Optional[portalocker.Lock] = None
        self._registered = False

    def acquire(self, timeout: float = 0.0) -> None:
        if self._lock is not None:
            return
        try:
            lock = portalocker.Lock(
                str(self.lock_path),
                timeout=timeout,
                flags=portalocker.LOCK_EX,
            )
            lock.acquire()
        except portalocker.LockException as exc:
            raise IdentityLockError(f"Identity '{self.identity}' is already in use") from exc
        self._lock = lock
        if not self._registered:
            atexit.register(self.release)
            self._registered = True

    def release(self) -> None:
        lock = self._lock
        if lock is None:
            return
        try:
            lock.release()
        except portalocker.LockException:
            pass
        finally:
            try:
                lock.close()
            except Exception:
                pass
            self._lock = None

    def __enter__(self) -> "IdentityLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def try_acquire_identity_lock(identity: str, timeout: float = 0.0) -> IdentityLock:
    lock = IdentityLock(identity)
    lock.acquire(timeout=timeout)
    return lock


def probe_lock_path(lock_path: Path) -> bool:
    """Return ``True`` if ``lock_path`` can be locked (i.e. currently free)."""

    try:
        with portalocker.Lock(str(lock_path), timeout=0, flags=portalocker.LOCK_EX | portalocker.LOCK_NB):
            return True
    except portalocker.LockException:
        return False
