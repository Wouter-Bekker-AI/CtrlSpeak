# -*- coding: utf-8 -*-
"""Atomic file helpers used across CtrlSpeak."""
from __future__ import annotations

import errno
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable


class AtomicWriteError(RuntimeError):
    """Raised when an atomic write operation fails."""


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _replace_with_retry(src: Path | str, dst: Path | str, *, attempts: int = 6, delay: float = 0.05) -> None:
    """Replace ``dst`` with ``src`` handling transient Windows locking issues."""

    source = Path(src)
    target = Path(dst)
    last_exc: OSError | None = None

    retry_errnos = {errno.EACCES}
    if hasattr(errno, "EBUSY"):
        retry_errnos.add(errno.EBUSY)
    if hasattr(errno, "ETXTBSY"):
        retry_errnos.add(errno.ETXTBSY)  # type: ignore[arg-type]

    for attempt in range(1, attempts + 1):
        try:
            os.replace(source, target)
            return
        except PermissionError as exc:  # Windows WinError 32
            last_exc = exc
        except OSError as exc:
            if exc.errno not in retry_errnos:
                raise
            last_exc = exc

        if attempt < attempts:
            time.sleep(delay * attempt)

    if last_exc is None:
        # Should not happen, but guard against None for mypy/typing
        last_exc = PermissionError(f"Failed to replace {source} -> {target}")
    raise last_exc


def _finalize_atomic_write(temp_path: Path, destination: Path, context: str) -> None:
    try:
        _replace_with_retry(temp_path, destination)
    except Exception as exc:  # pragma: no cover - validated via callers
        try:
            if temp_path.exists():
                os.unlink(temp_path)
        except Exception:
            pass
        raise AtomicWriteError(f"{context}: {exc}") from exc


def atomic_write_text(path: Path | str, text: str, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` atomically."""

    destination = Path(path)
    _ensure_parent(destination)

    temp_file = NamedTemporaryFile("w", encoding=encoding, dir=str(destination.parent), delete=False)
    temp_path = Path(temp_file.name)
    try:
        temp_file.write(text)
        temp_file.flush()
        os.fsync(temp_file.fileno())
    finally:
        try:
            temp_file.close()
        except Exception:
            pass

    _finalize_atomic_write(temp_path, destination, f"Failed to atomically write to {destination}")


def atomic_write_bytes(path: Path | str, payload: bytes) -> None:
    destination = Path(path)
    _ensure_parent(destination)

    temp_file = NamedTemporaryFile("wb", dir=str(destination.parent), delete=False)
    temp_path = Path(temp_file.name)
    try:
        temp_file.write(payload)
        temp_file.flush()
        os.fsync(temp_file.fileno())
    finally:
        try:
            temp_file.close()
        except Exception:
            pass

    _finalize_atomic_write(temp_path, destination, f"Failed to atomically write to {destination}")


def _rotate_chain(path: Path, keep: int) -> None:
    """Rotate existing log files, keeping ``keep`` historical copies."""
    if keep < 1:
        raise ValueError("keep must be at least 1")

    base = path.name
    parent = path.parent

    oldest = parent / f"{base}.{keep}"
    if oldest.exists():
        oldest.unlink()

    for index in range(keep - 1, 0, -1):
        src = parent / f"{base}.{index}"
        if src.exists():
            _replace_with_retry(src, parent / f"{base}.{index + 1}")

    _replace_with_retry(path, parent / f"{base}.1")


def atomic_rotate(path: Path | str, max_bytes: int, keep: int = 5, *, force: bool = False) -> bool:
    """Rotate ``path`` when it exceeds ``max_bytes``."""

    target = Path(path)
    if not target.exists():
        return False

    if not force:
        try:
            if target.stat().st_size <= max_bytes:
                return False
        except FileNotFoundError:
            return False

    _ensure_parent(target)
    _rotate_chain(target, keep)
    return True


def atomic_append_bytes(
    path: Path | str,
    payload: bytes,
    *,
    max_bytes: int | None = None,
    keep: int = 5,
) -> None:
    """Append ``payload`` to ``path`` atomically, optionally rotating beforehand."""

    if not payload:
        return

    destination = Path(path)
    _ensure_parent(destination)

    existing_bytes: bytes = b""
    if destination.exists():
        try:
            current_size = destination.stat().st_size
        except FileNotFoundError:
            current_size = 0
        if max_bytes is not None and current_size + len(payload) > max_bytes:
            atomic_rotate(destination, max_bytes, keep=keep, force=True)
            current_size = 0
        if current_size:
            with destination.open("rb") as src:
                existing_bytes = src.read()
            if existing_bytes and not existing_bytes.endswith(b"\n"):
                existing_bytes += b"\n"

    temp_file = NamedTemporaryFile("wb", dir=str(destination.parent), delete=False)
    temp_path = Path(temp_file.name)
    try:
        if existing_bytes:
            temp_file.write(existing_bytes)
        temp_file.write(payload)
        temp_file.flush()
        os.fsync(temp_file.fileno())
    finally:
        try:
            temp_file.close()
        except Exception:
            pass

    _finalize_atomic_write(temp_path, destination, f"Failed to append to {destination}")


def atomic_append_lines(
    path: Path | str,
    lines: Iterable[str],
    *,
    max_bytes: int | None = None,
    keep: int = 5,
    encoding: str = "utf-8",
) -> None:
    """Append newline-terminated ``lines`` atomically to ``path``."""

    payload = "".join(line if line.endswith("\n") else f"{line}\n" for line in lines)
    if not payload:
        return
    atomic_append_bytes(Path(path), payload.encode(encoding), max_bytes=max_bytes, keep=keep)
