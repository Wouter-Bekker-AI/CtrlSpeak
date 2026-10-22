"""One expiring, process-private failed recording; never automatic retry.

Retention is RAM-only: no persistent audio cache, manifest, token or transcript.
On retry a normal private recording temp file is used and the usual lifecycle
owns its cleanup. Closing CtrlSpeak also discards recovery.
"""
from __future__ import annotations

import os
from pathlib import Path
import threading
import time
from typing import Callable


class RecordingRecovery:
    def __init__(self, *, ttl_seconds: float = 900, max_bytes: int = 100 * 1024 * 1024,
                 clock: Callable[[], float] = time.monotonic):
        self.ttl_seconds = ttl_seconds
        self.max_bytes = max_bytes
        self.clock = clock
        self._lock = threading.RLock()
        self._audio: bytearray | None = None
        self._expires = 0.0
        self._timer: threading.Timer | None = None
        self._generation = 0

    def _clear(self):
        self._generation += 1
        if self._timer:
            self._timer.cancel()
            self._timer = None
        if self._audio is not None:
            # Best-effort overwrite of our mutable buffer before releasing it.
            for offset in range(0, len(self._audio), 65536):
                end = min(offset + 65536, len(self._audio))
                self._audio[offset:end] = b"\0" * (end - offset)
            self._audio = None
        self._expires = 0.0

    def discard(self):
        with self._lock:
            self._clear()

    def retain(self, path: Path) -> bool:
        with self._lock:
            self._clear()
            if path.is_symlink() or not 44 < path.stat().st_size <= self.max_bytes:
                return False
            with path.open("rb") as handle:
                audio = bytearray(handle.read(self.max_bytes + 1))
            if len(audio) > self.max_bytes:
                return False
            self._audio = audio
            self._expires = self.clock() + self.ttl_seconds
            generation = self._generation
            def expire():
                with self._lock:
                    if generation == self._generation:
                        self._clear()
            self._timer = threading.Timer(self.ttl_seconds, expire)
            self._timer.daemon = True
            self._timer.start()
            return True

    def seconds_remaining(self) -> float:
        with self._lock:
            remaining = self._expires - self.clock()
            if remaining <= 0:
                self._clear()
                return 0
            return remaining if self._audio is not None else 0

    def take(self, make_path: Callable[[], Path]) -> Path | None:
        with self._lock:
            if not self.seconds_remaining():
                return None
            path = make_path()
            created = False
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                created = True
                with os.fdopen(fd, "wb") as handle:
                    handle.write(self._audio)
            except Exception:
                if created:
                    path.unlink(missing_ok=True)
                raise
            self._clear()
            return path
