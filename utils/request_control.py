"""Per-request cancellation and progress, without process-global request state."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import threading
from typing import Callable


@dataclass(frozen=True)
class RequestControl:
    cancel: threading.Event
    progress: Callable[[str], None] | None = None


_local = threading.local()


def current_request_control() -> RequestControl | None:
    return getattr(_local, "control", None)


@contextmanager
def request_control(cancel: threading.Event, progress: Callable[[str], None] | None = None):
    previous = current_request_control()
    _local.control = RequestControl(cancel, progress)
    try:
        yield _local.control
    finally:
        _local.control = previous
