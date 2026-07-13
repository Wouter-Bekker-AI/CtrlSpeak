"""Best-effort active-field edit feedback for one CtrlSpeak injection.

The Enter observer never suppresses, replays, or synthesizes the user's Enter.
The selected platform adapter briefly uses its supported Ctrl+A/C path and
restores the text clipboard before the original key callback returns.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from utils.config_paths import get_logger
from utils.transcription_backend import FeedbackTarget, TranscriptionResult


LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class FieldSnapshot:
    text: str


class FieldSnapshotProvider(Protocol):
    def snapshot(self) -> FieldSnapshot | None: ...


class ActiveFieldSnapshotProvider:
    """Adapt a platform capture callback to the field-snapshot protocol."""

    def __init__(self, capture: Callable[[], str | None]) -> None:
        self._capture = capture

    def snapshot(self) -> FieldSnapshot | None:
        text = self._capture()
        return FieldSnapshot(text) if isinstance(text, str) else None


@dataclass(frozen=True)
class KeyEvent:
    key: str
    action: str
    modifiers: frozenset[str] = frozenset()


@dataclass(frozen=True)
class KeyHandlingOutcome:
    feedback_scheduled: bool = False
    suppress_event: bool = False
    replay_event: bool = False


@dataclass(frozen=True)
class PendingInjection:
    transcription_id: str
    injected_text: str
    raw_text: str | None
    capture_method: str
    created_at: float
    response_metadata: Mapping[str, Any]
    feedback_target: FeedbackTarget


SubmitFeedback = Callable[[str, str, str, dict[str, object], FeedbackTarget], None]
Executor = Callable[[Callable[[], None]], None]


def _background_executor(callback: Callable[[], None]) -> None:
    threading.Thread(target=callback, name="ctrlspeak-feedback", daemon=True).start()


class FeedbackCaptureCoordinator:
    """Tie one confirmed edit to one pending backend transcription result."""

    def __init__(
        self,
        *,
        snapshot_provider: FieldSnapshotProvider,
        submit_feedback: SubmitFeedback,
        executor: Executor = _background_executor,
        now: Callable[[], float] = time.monotonic,
        pending_ttl_seconds: float = 600.0,
    ) -> None:
        self.snapshot_provider = snapshot_provider
        self.submit_feedback = submit_feedback
        self.executor = executor
        self.now = now
        self.pending_ttl_seconds = pending_ttl_seconds
        self.pending: PendingInjection | None = None
        self._lock = threading.Lock()

    def track_injection(self, result: TranscriptionResult, *, capture_method: str) -> None:
        """Replace any prior pending injection; disabled/unroutable results are ignored."""
        with self._lock:
            self.pending = None
            if (
                capture_method != "active_field_on_enter"
                or not result.transcription_id
                or result.feedback_target is None
            ):
                return
            self.pending = PendingInjection(
                transcription_id=result.transcription_id,
                injected_text=result.text,
                raw_text=result.raw_text,
                capture_method=capture_method,
                created_at=self.now(),
                response_metadata=dict(result.metadata or {}),
                feedback_target=result.feedback_target,
            )

    def handle_key_event(self, event: KeyEvent) -> KeyHandlingOutcome:
        """Observe a bare Enter press without ever consuming or replaying it."""
        if event.action != "press" or event.key.lower() not in {"enter", "return"} or event.modifiers:
            return KeyHandlingOutcome()

        with self._lock:
            pending = self.pending
            self.pending = None
        if pending is None:
            return KeyHandlingOutcome()
        if self.now() - pending.created_at > self.pending_ttl_seconds:
            return KeyHandlingOutcome()

        try:
            # Capture before returning from the non-suppressing Enter callback;
            # many send fields clear as soon as the original Enter is delivered.
            snapshot = self.snapshot_provider.snapshot()
        except Exception:
            LOGGER.exception("Failed to capture the active CtrlSpeak target field")
            snapshot = None

        metadata: dict[str, object] = {
            "original_corrected_text": pending.injected_text,
            "raw_text": pending.raw_text or "",
        }
        language = pending.response_metadata.get("language")
        if isinstance(language, str) and language:
            metadata["language"] = language

        def capture_and_submit() -> None:
            try:
                if snapshot is None or not snapshot.text.strip():
                    return
                final_text = snapshot.text
                if final_text == pending.injected_text:
                    return
                self.submit_feedback(
                    pending.transcription_id,
                    final_text,
                    pending.capture_method,
                    metadata,
                    pending.feedback_target,
                )
            except Exception:
                LOGGER.exception("Failed to submit confirmed CtrlSpeak edit feedback")

        self.executor(capture_and_submit)
        return KeyHandlingOutcome(feedback_scheduled=True)
