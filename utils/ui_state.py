"""Headless UI state and telemetry helpers for the Midnight Signal interface.

The GUI reads immutable snapshots from :class:`TranscriptionUiSession`.  This
module deliberately stores neither audio nor transcript text, and provider
metadata is reduced to a small public allowlist before it reaches a snapshot.
It is therefore safe for the tray and overlay to poll without accidentally
retaining transcription content or credentials.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass
from enum import Enum
import math
import re
import sys
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence


DBFS_FLOOR = -60.0


class UiPhase(str, Enum):
    """The user-visible lifecycle of one capture/transcription session."""

    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"


class ProviderState(str, Enum):
    """Normalized capability state used by provider cards."""

    READY = "ready"
    REQUIRES_KEY = "requires_key"
    STARTING = "starting"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class AttemptOutcome(str, Enum):
    """Normalized outcome for a provider routing attempt."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProviderProfile:
    display_name: str
    model_name: str | None


@dataclass(frozen=True)
class ProviderTelemetry:
    """Sanitized provider information suitable for display and logging."""

    provider_id: str
    display_name: str
    model_name: str | None = None
    device: str | None = None
    state: ProviderState = ProviderState.UNKNOWN
    state_label: str = "Unknown"
    latency_ms: float | None = None
    inference_ms: float | None = None
    probe_duration_ms: float | None = None
    probe_age_ms: float | None = None
    circuit_retry_after_ms: float | None = None
    active: bool = False

    @property
    def latency_label(self) -> str:
        return format_latency_ms(self.latency_ms)

    @property
    def inference_label(self) -> str:
        return format_latency_ms(self.inference_ms)


@dataclass(frozen=True)
class ProviderAttempt:
    """A sanitized route attempt; raw server messages are never retained."""

    provider_id: str
    display_name: str
    outcome: AttemptOutcome
    category: str | None = None
    duration_ms: float | None = None
    inference_ms: float | None = None

    @property
    def duration_label(self) -> str:
        return format_latency_ms(self.duration_ms)


@dataclass(frozen=True)
class UiSnapshot:
    """Immutable view model consumed by the overlay and tray flyout."""

    revision: int
    phase: UiPhase
    headline: str
    detail: str
    elapsed_ms: float
    elapsed_label: str
    level_dbfs: float
    level_label: str
    level_fraction: float
    provider: ProviderTelemetry | None
    attempts: tuple[ProviderAttempt, ...]
    degraded: bool
    can_cancel: bool
    error_category: str | None


_PROVIDER_PROFILES: dict[str, ProviderProfile] = {
    "ubuntu-gpu-large-v3-turbo": ProviderProfile(
        "Ubuntu GPU", "Whisper large-v3-turbo"
    ),
    "openai-gpt-transcribe": ProviderProfile("OpenAI", "GPT Transcribe"),
    "nova-tiny-whisper": ProviderProfile("Gateway Tiny", "Whisper tiny"),
    "gateway-tiny": ProviderProfile("Gateway Tiny", "Whisper tiny"),
    "local-whisper": ProviderProfile("Local", "Whisper"),
    "bundled": ProviderProfile("Embedded / local", "Bundled Whisper"),
}

_GENERIC_PROVIDER_WORDS = {
    "api": "API",
    "cpu": "CPU",
    "cuda": "CUDA",
    "gpu": "GPU",
    "gpt": "GPT",
    "openai": "OpenAI",
    "ubuntu": "Ubuntu",
}

_FAILURE_CATEGORY_LABELS = {
    "microphone_failed": "The microphone could not be read.",
    "worker_unavailable": "The Ubuntu GPU is unavailable; check the fallback route.",
    "worker_circuit_open": "The Ubuntu GPU is temporarily being bypassed.",
    "openai_invalid_key": "OpenAI rejected the configured API key.",
    "openai_quota_exhausted": "OpenAI credit or quota is unavailable.",
    "openai_unavailable": "OpenAI transcription is temporarily unavailable.",
    "providers_exhausted": "No configured transcription provider completed the request.",
    "language_policy": "The result did not match the selected output language.",
    "cancelled": "The transcription was cancelled.",
}

_SAFE_ERROR_CATEGORIES = frozenset(
    {
        *_FAILURE_CATEGORY_LABELS,
        "invalid_route",
        "language_policy_violation",
        "local_provider_failed",
        "openai_key_required",
        "unexpected_error",
        "worker_error",
        "worker_invalid_response",
    }
)


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def format_latency_ms(value: object) -> str:
    """Format latency compactly while preserving useful millisecond detail."""

    milliseconds = _finite_number(value)
    if milliseconds is None or milliseconds < 0:
        return "—"
    if milliseconds < 1:
        return "<1 ms"
    if milliseconds < 1000:
        return f"{milliseconds:.0f} ms"
    if milliseconds < 10_000:
        return f"{milliseconds / 1000:.2f} s"
    if milliseconds < 60_000:
        return f"{milliseconds / 1000:.1f} s"
    minutes, seconds = divmod(milliseconds / 1000, 60)
    return f"{int(minutes)}m {seconds:.0f}s"


def format_elapsed_ms(value: object) -> str:
    """Format a live timer for the compact recording/transcribing capsule."""

    milliseconds = _finite_number(value)
    if milliseconds is None or milliseconds < 0:
        milliseconds = 0.0
    total_seconds = milliseconds / 1000
    if total_seconds < 60:
        return f"{total_seconds:.1f}s"
    minutes, seconds = divmod(total_seconds, 60)
    return f"{int(minutes)}:{seconds:04.1f}"


def format_dbfs(value: object, *, floor: float = DBFS_FLOOR) -> str:
    dbfs = _finite_number(value)
    if dbfs is None:
        dbfs = floor
    dbfs = clamp(dbfs, floor, 0.0)
    return f"{dbfs:.1f} dBFS"


def dbfs_to_fraction(value: object, *, floor: float = DBFS_FLOOR) -> float:
    """Map dBFS to a linear 0..1 meter position for rendering."""

    if floor >= 0:
        raise ValueError("dBFS floor must be negative")
    dbfs = _finite_number(value)
    if dbfs is None:
        dbfs = floor
    return clamp((clamp(dbfs, floor, 0.0) - floor) / -floor, 0.0, 1.0)


def pcm16_dbfs(frames: bytes | bytearray | memoryview, *, floor: float = DBFS_FLOOR) -> float:
    """Return RMS dBFS for interleaved signed 16-bit PCM without NumPy.

    An incomplete trailing byte is ignored, which makes the helper resilient to
    reads from arbitrary stream chunk boundaries. Silence and empty input use
    ``floor`` rather than negative infinity so the UI remains straightforward.
    """

    if floor >= 0:
        raise ValueError("dBFS floor must be negative")
    raw = bytes(frames)
    raw = raw[: len(raw) - (len(raw) % 2)]
    if not raw:
        return float(floor)
    samples = array("h")
    samples.frombytes(raw)
    if sys.byteorder == "big":
        samples.byteswap()
    if not samples:
        return float(floor)
    mean_square = math.fsum(float(sample) * float(sample) for sample in samples) / len(samples)
    if mean_square <= 0:
        return float(floor)
    rms = math.sqrt(mean_square)
    dbfs = 20.0 * math.log10(rms / 32768.0)
    return clamp(dbfs, floor, 0.0)


class DbfsSmoother:
    """Attack/release smoothing for a responsive but calm microphone meter."""

    def __init__(
        self,
        *,
        floor: float = DBFS_FLOOR,
        attack: float = 0.58,
        release: float = 0.16,
    ) -> None:
        if floor >= 0:
            raise ValueError("dBFS floor must be negative")
        if not 0 < attack <= 1 or not 0 < release <= 1:
            raise ValueError("attack and release must be in the range (0, 1]")
        self.floor = float(floor)
        self.attack = float(attack)
        self.release = float(release)
        self._value = self.floor
        self._lock = threading.Lock()

    @property
    def value(self) -> float:
        with self._lock:
            return self._value

    def reset(self) -> float:
        with self._lock:
            self._value = self.floor
            return self._value

    def update(self, value: object) -> float:
        incoming = _finite_number(value)
        if incoming is None:
            incoming = self.floor
        incoming = clamp(incoming, self.floor, 0.0)
        with self._lock:
            coefficient = self.attack if incoming > self._value else self.release
            self._value += coefficient * (incoming - self._value)
            return self._value

    def update_pcm16(self, frames: bytes | bytearray | memoryview) -> float:
        return self.update(pcm16_dbfs(frames, floor=self.floor))


def normalize_provider_id(value: object) -> str:
    provider_id = str(value or "").strip().casefold()
    provider_id = re.sub(r"[^a-z0-9._-]+", "-", provider_id).strip("-._")
    return provider_id or "unknown"


def provider_profile(provider_id: object) -> ProviderProfile:
    normalized = normalize_provider_id(provider_id)
    known = _PROVIDER_PROFILES.get(normalized)
    if known:
        return known
    if normalized.startswith("ubuntu-gpu"):
        return ProviderProfile("Ubuntu GPU", "Whisper large-v3-turbo")
    if "openai" in normalized:
        return ProviderProfile("OpenAI", "GPT Transcribe")
    if "tiny" in normalized:
        return ProviderProfile("Gateway Tiny", "Whisper tiny")
    words = re.split(r"[-_.]+", normalized)
    display = " ".join(
        _GENERIC_PROVIDER_WORDS.get(word, word.capitalize()) for word in words if word
    )
    return ProviderProfile(display or "Unknown provider", None)


def normalize_provider_state(value: object) -> tuple[ProviderState, str]:
    status = str(value or "").strip().casefold().replace("-", "_")
    if status in {"ready", "available", "healthy", "succeeded", "online"}:
        return ProviderState.READY, "Ready"
    if status in {"available_with_key", "credential_required", "requires_key"}:
        return ProviderState.REQUIRES_KEY, "Available with key"
    if status in {"starting", "loading", "checking", "initializing"}:
        return ProviderState.STARTING, "Starting"
    if status in {
        "unavailable",
        "offline",
        "failed",
        "error",
        "circuit_open",
        "not_authorized",
        "disabled",
    }:
        return ProviderState.UNAVAILABLE, "Unavailable"
    return ProviderState.UNKNOWN, "Unknown"


def provider_from_capability(
    capability: Mapping[str, Any],
    *,
    active_provider_id: object | None = None,
    latency_ms: object | None = None,
) -> ProviderTelemetry:
    """Reduce one gateway capability object to safe display telemetry."""

    provider_id = normalize_provider_id(capability.get("id"))
    profile = provider_profile(provider_id)
    state, state_label = normalize_provider_state(capability.get("status"))
    model = str(capability.get("model") or "").strip() or profile.model_name
    device = str(capability.get("device") or "").strip().casefold() or None
    latency = _finite_number(latency_ms)
    raw_health = capability.get("health")
    health = raw_health if isinstance(raw_health, Mapping) else {}
    probe_duration = _finite_number(health.get("probe_duration_ms"))
    probe_age = _finite_number(health.get("probe_age_ms"))
    circuit_retry = _finite_number(health.get("circuit_retry_after_ms"))
    active_id = normalize_provider_id(active_provider_id) if active_provider_id else None
    return ProviderTelemetry(
        provider_id=provider_id,
        display_name=profile.display_name,
        model_name=model,
        device=device,
        state=state,
        state_label=state_label,
        latency_ms=latency if latency is not None and latency >= 0 else None,
        probe_duration_ms=(
            probe_duration if probe_duration is not None and probe_duration >= 0 else None
        ),
        probe_age_ms=probe_age if probe_age is not None and probe_age >= 0 else None,
        circuit_retry_after_ms=(
            circuit_retry if circuit_retry is not None and circuit_retry >= 0 else None
        ),
        active=provider_id == active_id,
    )


def providers_from_capabilities(
    payload: Mapping[str, Any],
    *,
    active_provider_id: object | None = None,
    latency_by_provider: Mapping[str, object] | None = None,
) -> tuple[ProviderTelemetry, ...]:
    """Normalize the provider list returned by ``/v1/capabilities``."""

    raw_providers = payload.get("providers")
    if not isinstance(raw_providers, Sequence) or isinstance(raw_providers, (str, bytes)):
        single = payload.get("provider")
        raw_providers = [single] if isinstance(single, Mapping) else []
    latencies = latency_by_provider or {}
    providers: list[ProviderTelemetry] = []
    for raw in raw_providers:
        if not isinstance(raw, Mapping):
            continue
        provider_id = normalize_provider_id(raw.get("id"))
        providers.append(
            provider_from_capability(
                raw,
                active_provider_id=active_provider_id,
                latency_ms=latencies.get(provider_id),
            )
        )
    return tuple(providers)


def normalize_attempts(value: object) -> tuple[ProviderAttempt, ...]:
    """Allowlist route-attempt data and discard messages/actions/credentials."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    attempts: list[ProviderAttempt] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            continue
        provider_id = normalize_provider_id(raw.get("provider"))
        raw_status = str(raw.get("status") or "").strip().casefold()
        if raw_status in {"succeeded", "success", "ready"}:
            outcome = AttemptOutcome.SUCCEEDED
        elif raw_status in {"not_authorized", "skipped", "disabled"}:
            outcome = AttemptOutcome.SKIPPED
        elif raw_status in {"failed", "unavailable", "error", "circuit_open"}:
            outcome = AttemptOutcome.FAILED
        else:
            # Provider failures currently expose their category with no status.
            outcome = AttemptOutcome.FAILED if raw.get("category") else AttemptOutcome.UNKNOWN
        raw_category = str(raw.get("category") or "").strip().casefold()
        normalized_category = re.sub(r"[^a-z0-9._-]+", "_", raw_category)[:80]
        # Categories are a closed public protocol vocabulary. Avoid retaining
        # arbitrary server strings in case a malformed response places a
        # credential or internal detail in this field.
        category = (
            normalized_category
            if normalized_category in _SAFE_ERROR_CATEGORIES
            else ("provider_error" if normalized_category else None)
        )
        duration = _finite_number(raw.get("duration_ms"))
        inference = _finite_number(raw.get("inference_duration_ms"))
        attempts.append(
            ProviderAttempt(
                provider_id,
                provider_profile(provider_id).display_name,
                outcome,
                category,
                duration if duration is not None and duration >= 0 else None,
                inference if inference is not None and inference >= 0 else None,
            )
        )
    return tuple(attempts)


def provider_from_result_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    latency_ms: object | None = None,
) -> ProviderTelemetry | None:
    if not isinstance(metadata, Mapping):
        return None
    raw_id = metadata.get("provider_used") or metadata.get("provider")
    if not raw_id:
        backend = str(metadata.get("backend") or "").strip().casefold()
        raw_id = "bundled" if backend == "bundled" else None
    if not raw_id:
        return None
    provider_id = normalize_provider_id(raw_id)
    profile = provider_profile(provider_id)
    attempts = normalize_attempts(metadata.get("attempts"))
    chosen_attempt = next(
        (
            attempt
            for attempt in reversed(attempts)
            if attempt.provider_id == provider_id
            and attempt.outcome is AttemptOutcome.SUCCEEDED
        ),
        None,
    )
    routing_duration = _finite_number(metadata.get("routing_duration_ms"))
    latency = (
        chosen_attempt.duration_ms
        if chosen_attempt and chosen_attempt.duration_ms is not None
        else (
            routing_duration
            if routing_duration is not None and routing_duration >= 0
            else _finite_number(latency_ms)
        )
    )
    inference = chosen_attempt.inference_ms if chosen_attempt else None
    return ProviderTelemetry(
        provider_id=provider_id,
        display_name=profile.display_name,
        model_name=profile.model_name,
        state=ProviderState.READY,
        state_label="Used",
        latency_ms=latency if latency is not None and latency >= 0 else None,
        inference_ms=inference,
        active=True,
    )


def failure_detail(category: object) -> str:
    normalized = str(category or "").strip().casefold()
    return _FAILURE_CATEGORY_LABELS.get(
        normalized,
        "The transcription did not complete. Open CtrlSpeak for details.",
    )


class InvalidUiTransition(RuntimeError):
    """Raised when integration code attempts an impossible session transition."""


class TranscriptionUiSession:
    """Thread-safe, in-memory state machine for the Midnight Signal surfaces."""

    _TRANSITIONS = {
        UiPhase.IDLE: {UiPhase.RECORDING},
        UiPhase.RECORDING: {UiPhase.PROCESSING, UiPhase.ERROR, UiPhase.CANCELLED},
        UiPhase.PROCESSING: {UiPhase.SUCCESS, UiPhase.ERROR, UiPhase.CANCELLED},
        UiPhase.SUCCESS: {UiPhase.IDLE, UiPhase.RECORDING},
        UiPhase.ERROR: {UiPhase.IDLE, UiPhase.RECORDING},
        UiPhase.CANCELLED: {UiPhase.IDLE, UiPhase.RECORDING},
    }

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        now = float(clock())
        self._lock = threading.RLock()
        self._phase = UiPhase.IDLE
        self._session_started = now
        self._phase_started = now
        self._revision = 0
        self._level = DbfsSmoother()
        self._provider: ProviderTelemetry | None = None
        self._attempts: tuple[ProviderAttempt, ...] = ()
        self._degraded = False
        self._error_category: str | None = None
        self._final_elapsed_ms: float | None = None

    @property
    def phase(self) -> UiPhase:
        with self._lock:
            return self._phase

    def _transition(self, target: UiPhase) -> None:
        allowed = self._TRANSITIONS[self._phase]
        if target not in allowed:
            raise InvalidUiTransition(f"cannot transition from {self._phase.value} to {target.value}")
        self._phase = target
        self._phase_started = float(self._clock())
        self._revision += 1

    def begin_recording(self) -> None:
        with self._lock:
            self._transition(UiPhase.RECORDING)
            self._session_started = self._phase_started
            self._level.reset()
            self._provider = None
            self._attempts = ()
            self._degraded = False
            self._error_category = None
            self._final_elapsed_ms = None

    def begin_processing(self) -> None:
        with self._lock:
            self._transition(UiPhase.PROCESSING)
            self._level.reset()

    def update_level_pcm16(self, frames: bytes | bytearray | memoryview) -> float:
        """Update recording level without retaining the supplied audio chunk."""

        with self._lock:
            if self._phase is not UiPhase.RECORDING:
                return self._level.value
            value = self._level.update_pcm16(frames)
            self._revision += 1
            return value

    def complete(
        self,
        metadata: Mapping[str, Any] | None = None,
        *,
        elapsed_ms: object | None = None,
    ) -> None:
        """Finish successfully using only allowlisted routing metadata.

        Transcript fields such as ``text`` and ``raw_text`` are intentionally
        ignored even when they are present in ``metadata``.
        """

        with self._lock:
            self._transition(UiPhase.SUCCESS)
            measured = _finite_number(elapsed_ms)
            if measured is None or measured < 0:
                measured = max(0.0, (self._phase_started - self._session_started) * 1000)
            self._final_elapsed_ms = measured
            self._provider = provider_from_result_metadata(metadata, latency_ms=measured)
            self._attempts = normalize_attempts(metadata.get("attempts") if metadata else None)
            self._degraded = bool(metadata.get("degraded")) if metadata else False
            self._error_category = None

    def fail(self, category: object = None, *, elapsed_ms: object | None = None) -> None:
        with self._lock:
            self._transition(UiPhase.ERROR)
            normalized = re.sub(
                r"[^a-z0-9._-]+", "_", str(category or "unexpected_error").strip().casefold()
            )[:80]
            self._error_category = (
                normalized if normalized in _SAFE_ERROR_CATEGORIES else "unexpected_error"
            )
            measured = _finite_number(elapsed_ms)
            self._final_elapsed_ms = (
                measured
                if measured is not None and measured >= 0
                else max(0.0, (self._phase_started - self._session_started) * 1000)
            )

    def cancel(self) -> None:
        with self._lock:
            self._transition(UiPhase.CANCELLED)
            self._error_category = "cancelled"
            self._final_elapsed_ms = max(
                0.0, (self._phase_started - self._session_started) * 1000
            )

    def reset(self) -> None:
        with self._lock:
            self._transition(UiPhase.IDLE)
            self._level.reset()
            self._provider = None
            self._attempts = ()
            self._degraded = False
            self._error_category = None
            self._final_elapsed_ms = None

    def snapshot(self) -> UiSnapshot:
        with self._lock:
            now = float(self._clock())
            if self._final_elapsed_ms is not None:
                elapsed_ms = self._final_elapsed_ms
            elif self._phase is UiPhase.IDLE:
                elapsed_ms = 0.0
            else:
                elapsed_ms = max(0.0, (now - self._session_started) * 1000)
            level = self._level.value
            headline, detail = self._copy_for_phase()
            return UiSnapshot(
                revision=self._revision,
                phase=self._phase,
                headline=headline,
                detail=detail,
                elapsed_ms=elapsed_ms,
                elapsed_label=format_elapsed_ms(elapsed_ms),
                level_dbfs=level,
                level_label=format_dbfs(level),
                level_fraction=dbfs_to_fraction(level),
                provider=self._provider,
                attempts=self._attempts,
                degraded=self._degraded,
                can_cancel=self._phase in {UiPhase.RECORDING, UiPhase.PROCESSING},
                error_category=self._error_category,
            )

    def _copy_for_phase(self) -> tuple[str, str]:
        if self._phase is UiPhase.IDLE:
            return "CtrlSpeak is ready", "Hold Right Ctrl to start speaking."
        if self._phase is UiPhase.RECORDING:
            return "Listening", "Release Right Ctrl to transcribe."
        if self._phase is UiPhase.PROCESSING:
            return "Transcribing", "Routing through your preferred provider…"
        if self._phase is UiPhase.SUCCESS:
            if self._provider:
                route = self._provider.display_name
                if self._degraded:
                    return "Transcription ready", f"Completed via fallback · {route}"
                return "Transcription ready", f"Completed with {route}"
            return "Transcription ready", "Text was inserted into the active field."
        if self._phase is UiPhase.CANCELLED:
            return "Cancelled", _FAILURE_CATEGORY_LABELS["cancelled"]
        return "Couldn’t transcribe", failure_detail(self._error_category)


def active_route_label(attempts: Iterable[ProviderAttempt]) -> str:
    """Return a compact, truthful path label such as ``Ubuntu GPU → OpenAI``."""

    names: list[str] = []
    for attempt in attempts:
        if not names or names[-1] != attempt.display_name:
            names.append(attempt.display_name)
    return " → ".join(names) if names else "—"
