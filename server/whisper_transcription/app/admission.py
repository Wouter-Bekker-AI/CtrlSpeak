"""Non-queuing admission and safe, transcript-free request telemetry."""
from __future__ import annotations

import logging
import math
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("ctrlspeak_whisper_transcription.requests")
REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def milliseconds(start: float) -> float:
    return round(max(0.0, time.monotonic() - start) * 1000, 3)


def safe_timings(value: object) -> dict[str, float]:
    """Never propagate an arbitrary worker object or transcript into diagnostics."""
    if not isinstance(value, dict):
        return {}
    result = {}
    for key in ("receive_duration_ms", "upload_spool_duration_ms", "queue_duration_ms",
                "inference_duration_ms", "cleanup_duration_ms", "server_duration_ms"):
        number = value.get(key)
        if (not isinstance(number, bool) and isinstance(number, (float, int))
                and math.isfinite(number) and 0 <= number <= 86_400_000):
            result[key] = round(float(number), 3)
    return result


class Admission:
    def __init__(self, capacity: int, stalled_after_seconds: float) -> None:
        self.capacity = capacity
        self.stalled_after_seconds = stalled_after_seconds
        self._active: dict[str, float] = {}
        self._lock = threading.Lock()

    def acquire(self) -> "Lease | None":
        with self._lock:
            if len(self._active) >= self.capacity:
                return None
            key = uuid.uuid4().hex
            self._active[key] = time.monotonic()
        return Lease(self, key)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            active = len(self._active)
            age = milliseconds(min(self._active.values())) if active else 0.0
        busy = active >= self.capacity
        return {"capacity": self.capacity, "active": active, "queued": 0,
                "available": max(0, self.capacity - active),
                "oldest_active_duration_ms": age,
                "status": "stalled" if busy and age >= self.stalled_after_seconds * 1000
                else "busy" if busy else "ready"}


class Lease:
    def __init__(self, admission: Admission, key: str) -> None:
        self.admission = admission
        self.key = key
        self.transferred = False

    def release(self) -> None:
        with self.admission._lock:
            self.admission._active.pop(self.key, None)


class WorkOwner:
    """Keep the slot/file alive even if the awaiting asyncio task is cancelled.

    Python's synchronous inference cannot be killed. Its thread, not the
    caller's socket lifetime, is authoritative for resource release.
    """
    def __init__(self, lease: Lease, path: Path):
        self.lease, self.path = lease, path
        self._lock = threading.Lock()
        self._running = 0
        self._closing = False

    def call(self, function, *args, **kwargs):
        with self._lock:
            if self._closing:
                raise RuntimeError("request ended before inference started")
            self._running += 1
        try:
            return function(*args, **kwargs)
        finally:
            with self._lock:
                self._running -= 1
                if self._closing and not self._running:
                    self._release()

    def close(self):
        with self._lock:
            self._closing = True
            if not self._running:
                self._release()

    def _release(self):
        try:
            self.path.unlink(missing_ok=True)
        finally:
            self.lease.release()


class RequestTelemetryMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        state = scope.setdefault("state", {})
        headers = dict(scope.get("headers", []))
        candidate = headers.get(b"x-ctrlspeak-request-id", b"").decode("ascii", "ignore")
        state["request_id"] = candidate if REQUEST_ID.fullmatch(candidate) else uuid.uuid4().hex
        state["request_started_at"] = time.monotonic()
        state["cancel_event"] = threading.Event()
        state["receive_duration_ms"] = 0.0
        state["received_bytes"] = 0
        status = 499

        async def measured_receive():
            message = await receive()
            if message["type"] == "http.request":
                state["received_bytes"] += len(message.get("body", b""))
                if not message.get("more_body", False):
                    state["receive_duration_ms"] = milliseconds(state["request_started_at"])
            return message

        async def measured_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message["headers"] = list(message.get("headers", [])) + [
                    (b"x-ctrlspeak-request-id", state["request_id"].encode("ascii")),
                ]
            await send(message)

        try:
            await self.app(scope, measured_receive, measured_send)
        except BaseException:
            state["cancel_event"].set()
            raise
        finally:
            if scope.get("path") in {"/v1/transcribe", "/v1/worker/transcribe"}:
                LOGGER.info(
                    "transcription_request request_id=%s status=%s category=%s "
                    "received_bytes=%s receive_duration_ms=%s server_duration_ms=%s",
                    state["request_id"], status, state.get("failure_category", "none"),
                    state["received_bytes"], state["receive_duration_ms"],
                    milliseconds(state["request_started_at"]),
                )
