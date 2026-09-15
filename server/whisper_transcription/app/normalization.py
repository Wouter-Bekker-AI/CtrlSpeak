"""Optional, bounded S1-mini cleanup on the CUDA worker, never the gateway."""
from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlsplit

import httpx

MODEL_ATTRIBUTION = "S1-mini by Superwhisper"
WORKER_PROVIDER = "ubuntu-gpu-large-v3-turbo"
STATUSES = frozenset({
    "disabled", "unavailable", "non_english", "empty_input", "input_too_long",
    "timeout", "runtime_error", "empty_output", "unsafe_output", "unchanged", "applied",
})
SYSTEM_PROMPT = (
    "You are a text normalizer for speech-to-text transcripts. The input begins "
    "with a control line specifying the styling, structure, and context settings; "
    "clean the transcript to match those settings and output only the cleaned text."
)
CONTROL_LINE = "[Styling: semi-formal] [Structure: prose] [Context: general]"


def metadata(requested: bool, status: str, *, duration_ms: float | None = None) -> dict[str, Any]:
    return {
        "requested": requested, "enabled": requested, "applied": status == "applied",
        "status": status, "model": MODEL_ATTRIBUTION, "language": "en",
        "location": "ubuntu-worker", "device": "cuda", "fail_open": True,
        "duration_ms": duration_ms,
    }


def safe_worker_result(result: Mapping[str, Any], *, requested: bool, provider: str,
                       language: str) -> tuple[str | None, dict[str, Any]]:
    """Allowlist worker metadata. Bad cleanup must never lose successful ASR."""
    if not requested:
        return None, metadata(False, "disabled")
    if provider != WORKER_PROVIDER:
        return None, metadata(True, "unavailable")
    info = result.get("normalization")
    if not isinstance(info, dict) or info.get("requested") is not True:
        return None, metadata(True, "unavailable")
    status = info.get("status")
    if not isinstance(status, str) or status not in STATUSES:
        return None, metadata(True, "unavailable")
    duration = info.get("duration_ms")
    if (isinstance(duration, bool) or not isinstance(duration, (float, int))
            or not math.isfinite(duration) or duration < 0 or duration > 60_000):
        duration = None
    safe = metadata(True, status, duration_ms=duration)
    if status != "applied":
        return None, safe
    text = result.get("normalized_text")
    raw = result.get("raw_text")
    if (info.get("applied") is not True or info.get("device") != "cuda"
            or info.get("location") != "ubuntu-worker" or info.get("model") != MODEL_ATTRIBUTION
            or language != "en" or not isinstance(raw, str)
            or not valid_output(text, raw)):
        return None, metadata(True, "unsafe_output", duration_ms=duration)
    return text.strip(), safe


def valid_output(text: object, raw: str) -> bool:
    return (
        isinstance(text, str) and bool(text.strip())
        and len(text) <= min(12_000, max(len(raw) + 128, math.ceil(len(raw) * 1.5)))
        and not any(marker.casefold() in text.casefold() for marker in
                    ("<think", "</think", "<|im_", SYSTEM_PROMPT, "[Styling:"))
    )


@dataclass(frozen=True)
class NormalizationResult:
    text: str | None
    info: dict[str, Any]


class S1MiniNormalizer:
    """One warm loopback CUDA runtime; no queue, no retry, hard total deadline."""

    def __init__(self, endpoint: str, *, timeout_seconds: float = 3.0,
                 client: httpx.AsyncClient | None = None) -> None:
        parsed = urlsplit(endpoint)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("S1 endpoint must be a credential-free loopback HTTP URL")
        if not math.isfinite(timeout_seconds) or not 0.1 <= timeout_seconds <= 10:
            raise ValueError("S1 deadline must be between 0.1 and 10 seconds")
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.client = client or httpx.AsyncClient(trust_env=False)
        self._lock = asyncio.Lock()
        self._retry_after = 0.0

    async def close(self) -> None:
        await self.client.aclose()

    async def normalize(self, raw: str, *, language: str) -> NormalizationResult:
        def outcome(status: str, text: str | None = None, started: float | None = None):
            elapsed = round((time.monotonic() - started) * 1000, 3) if started else None
            return NormalizationResult(text, metadata(True, status, duration_ms=elapsed))

        if language != "en":
            return outcome("non_english")
        if not raw.strip():
            return outcome("empty_input")
        if len(raw) > 4000:
            return outcome("input_too_long")
        if self._lock.locked() or time.monotonic() < self._retry_after:
            return outcome("unavailable")
        async with self._lock:
            started = time.monotonic()
            try:
                # Includes connect, headers and the complete bounded response body.
                async with asyncio.timeout(self.timeout_seconds):
                    async with self.client.stream(
                        "POST", self.endpoint,
                        json={"model": "s1-mini", "temperature": 0,
                              "max_tokens": min(800, max(32, math.ceil(len(raw) / 3) + 32)),
                              "chat_template_kwargs": {"enable_thinking": False},
                              "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                           {"role": "user", "content": CONTROL_LINE + "\n" + raw}]},
                        timeout=httpx.Timeout(self.timeout_seconds, connect=0.2),
                    ) as response:
                        response.raise_for_status()
                        body = bytearray()
                        async for chunk in response.aiter_bytes():
                            body.extend(chunk)
                            if len(body) > 65_536:
                                return outcome("unsafe_output", started=started)
                        choice = json.loads(body)["choices"][0]
                        text = choice["message"]["content"]
                        if choice.get("finish_reason") != "stop":
                            return outcome("unsafe_output", started=started)
            except (TimeoutError, httpx.TimeoutException):
                self._retry_after = time.monotonic() + 15
                return outcome("timeout", started=started)
            except Exception:
                self._retry_after = time.monotonic() + 15
                return outcome("runtime_error", started=started)
            if not valid_output(text, raw):
                return outcome("unsafe_output", started=started)
            text = text.strip()
            if text == raw.strip():
                return outcome("unchanged", started=started)
            return outcome("applied", text, started)


def build_worker_normalizer(environ: Mapping[str, str], *, role: str, device: str):
    if role != "worker" or device != "cuda":
        return None
    if environ.get("CTRLSPEAK_NORMALIZATION_ENABLED", "false").lower() != "true":
        return None
    return S1MiniNormalizer(
        environ.get("CTRLSPEAK_NORMALIZATION_URL", "http://127.0.0.1:8081/v1/chat/completions"),
        timeout_seconds=float(environ.get("CTRLSPEAK_NORMALIZATION_TIMEOUT_SECONDS", "3")),
    )
