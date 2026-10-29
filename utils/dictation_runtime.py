"""Bounded continuous dictation with capture, request and delivery ownership.

No GUI, microphone or input APIs live here. Adapters are injected so every mute,
late-result and shutdown race can be tested without touching a user's desktop.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import threading
import time
from typing import Callable

from utils.dictation_modes import SpeechSegmenter


@dataclass(frozen=True)
class AudioJob:
    number: int
    pcm: bytes
    epoch: int


class DeliveryCancelled:
    def __init__(self, session: "DictationSession", epoch: int):
        self.session, self.epoch = session, epoch

    def is_set(self) -> bool:
        return not self.session.delivery_allowed(self.epoch)


class DictationSession:
    MAX_AUDIO_BYTES = 100 * 1024 * 1024
    MAX_SESSION_SECONDS = 1800
    MAX_QUEUED = 3
    MAX_PENDING_CHARS = 128_000
    EDIT_SETTLE_SECONDS = 0.6

    def __init__(self, mode: str, *, capture: Callable, transcribe: Callable,
                 deliver: Callable, target: Callable, changed: Callable = lambda: None,
                 level: Callable = lambda pcm: None, clock: Callable = time.monotonic):
        if mode not in {"hands_free", "live"}:
            raise ValueError("Unsupported continuous dictation mode")
        self.mode = mode
        self.capture, self.transcribe, self.deliver = capture, transcribe, deliver
        self.target, self.changed, self.level, self.clock = target, changed, level, clock
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.cancel = threading.Event()
        self.stop_capture = threading.Event()
        self.capture_complete = threading.Event()
        self.done = threading.Event()
        self.muted = False
        self.backlog_paused = False
        self.editing = False
        self.keys_held = False
        self.epoch = 0
        self.audio_epoch = 0
        self.last_edit = 0.0
        self.started = clock()
        self.expected_target = target()
        self.jobs: deque[AudioJob] = deque()
        self.pending: list[str] = []
        self.pending_revision = 0
        self.committed: list[str] = []
        self.last_metadata: dict = {}
        self.frames = bytearray()
        self.segmenter = SpeechSegmenter()
        self.number = 0
        self.processing = False
        self.stage = "listening"
        self.error: str | None = None
        self.finishing = False
        self.worker: threading.Thread | None = None
        self.reader: threading.Thread | None = None

    def start(self) -> None:
        self.worker = threading.Thread(target=self._work, name="CtrlSpeakDictationRequests", daemon=True)
        self.reader = threading.Thread(target=self._read, name="CtrlSpeakContinuousMic", daemon=True)
        try:
            self.worker.start()
            self.reader.start()
        except BaseException:
            self.cancel.set()
            self.stop_capture.set()
            self.capture_complete.set()
            self.wake.set()
            if self.worker.is_alive():
                self.worker.join(timeout=2)
            raise

    def status(self) -> dict:
        with self.lock:
            return {"mode": self.mode, "active": not self.done.is_set(),
                    "muted": self.muted, "editing": self.editing,
                    "pending_count": len(self.pending), "queued_count": len(self.jobs),
                    "stage": self.stage, "error": self.error,
                    "processing": self.processing, "finishing": self.finishing}

    def capture_token(self) -> tuple[int, bool]:
        with self.lock:
            return self.audio_epoch, self.muted

    def feed(self, pcm: bytes, token: tuple[int, bool] | None = None) -> None:
        """Drop a whole read if mute changed during it; never buffer muted audio."""
        with self.lock:
            if self.cancel.is_set() or self.stop_capture.is_set() or self.muted:
                return
            if token is not None and (token[0] != self.audio_epoch or token[1]):
                return
            if self.mode == "live" and self.editing:
                if self.keys_held or self.clock() - self.last_edit < self.EDIT_SETTLE_SECONDS:
                    return
                self.editing = False
                self.stage = "listening"
            if self.clock() - self.started > self.MAX_SESSION_SECONDS:
                self.error = "Session limit reached; finish or cancel this dictation."
                self._mute_locked()
                self.changed()
                return
            self.level(pcm)
            if self.mode == "hands_free":
                if len(self.frames) + len(pcm) > self.MAX_AUDIO_BYTES:
                    self.error = "Recording limit reached; finish this dictation."
                    self._mute_locked()
                else:
                    self.frames.extend(pcm)
            else:
                for segment in self.segmenter.feed(pcm):
                    self._enqueue_locked(segment)

    def _enqueue_locked(self, pcm: bytes, epoch: int | None = None) -> None:
        if not pcm:
            return
        # Capture automatically pauses when the third waiting phrase is queued.
        # This bound is checked before accepting any further microphone frames.
        if len(self.jobs) >= self.MAX_QUEUED:
            self.error = "Backlog full; capture paused. Finish or cancel before continuing."
            self._mute_locked()
            return
        self.number += 1
        self.jobs.append(AudioJob(self.number, pcm, self.epoch if epoch is None else epoch))
        if len(self.jobs) >= self.MAX_QUEUED:
            self.error = "Backlog paused capture. Earlier audio will finish into held text; copy/discard it before resuming."
            self._mute_locked()
            self.backlog_paused = True
        self.wake.set()

    def _flush_locked(self) -> None:
        if self.mode == "live":
            for segment in self.segmenter.flush():
                self._enqueue_locked(segment)
            self.segmenter.reset()

    def _mute_locked(self) -> None:
        self.muted = True
        self.audio_epoch += 1
        self.epoch += 1
        self.stage = "muted"

    def toggle_mute(self) -> bool:
        with self.lock:
            if self.done.is_set() or self.finishing:
                return self.muted
            if not self.muted:
                self._flush_locked()
                self._mute_locked()
                self.backlog_paused = False
            elif len(self.jobs) < self.MAX_QUEUED:
                if self.backlog_paused and (self.jobs or self.processing):
                    self.error = "Backlog is still draining into held text. Wait before resuming."
                    return True
                if self.pending:
                    self.error = "Copy or discard pending text before resuming."
                    return True
                self.muted = False
                self.backlog_paused = False
                self.audio_epoch += 1
                self.segmenter.reset()
                self.expected_target = self.target()
                self.stage = "listening"
                self.error = None
            self.wake.set()
        self.changed()
        return self.muted

    def physical_edit(self) -> None:
        if self.mode != "live":
            return
        with self.lock:
            if self.done.is_set():
                return
            if not self.editing:
                self._flush_locked()
            self.epoch += 1
            self.editing = True
            self.last_edit = self.clock()
            self.stage = "muted" if self.muted else "editing"
        self.changed()

    def set_keys_held(self, held: bool) -> None:
        with self.lock:
            self.keys_held = held

    def check_target(self) -> None:
        """A changed/unknown desktop target requires explicit unmute to re-arm."""
        if self.done.is_set() or self.mode != "live":
            return
        current = self.target()
        with self.lock:
            if not self.muted and (current is None or current != self.expected_target):
                self._flush_locked()
                self._mute_locked()
                self.error = "Target changed. Focus the intended field, then explicitly unmute."
                self.changed()

    def delivery_allowed(self, epoch: int) -> bool:
        with self.lock:
            return (not self.cancel.is_set() and not self.muted and not self.editing
                    and not self.keys_held and epoch == self.epoch and not self.pending
                    and self.expected_target is not None and self.target() == self.expected_target)

    def finish(self) -> None:
        with self.lock:
            self.finishing = True
            self.stop_capture.set()
            self.stage = "finishing"
        self.wake.set()
        self.changed()

    def abort(self) -> None:
        with self.lock:
            self.cancel.set()
            self.stop_capture.set()
            self.jobs.clear()
            self.frames.clear()
            self.segmenter.reset()
            self.stage = "cancelled"
        self.wake.set()
        self.changed()

    def pending_text(self) -> str:
        with self.lock:
            return "\n\n".join(self.pending)

    def pending_snapshot(self) -> tuple[int, tuple[str, ...]]:
        with self.lock:
            return self.pending_revision, tuple(self.pending)

    def acknowledge_copied(self, snapshot: tuple[int, tuple[str, ...]]) -> None:
        """Remove only copied entries, never a result arriving during clipboard IO."""
        revision, entries = snapshot
        with self.lock:
            if revision == self.pending_revision and tuple(self.pending[:len(entries)]) == entries:
                del self.pending[:len(entries)]
                self.pending_revision += 1
        self.changed()

    def discard_pending(self) -> None:
        with self.lock:
            self.pending.clear()
            self.pending_revision += 1
        self.changed()

    def _hold_result(self, text: str, *, uncertain: bool = False) -> None:
        with self.lock:
            remaining = self.MAX_PENDING_CHARS - sum(map(len, self.pending))
            if remaining > 0:
                self.pending.append(text[:remaining])
            self._mute_locked()
            self.error = ("Insertion interrupted; some text may already exist. Inspect before copying pending text."
                          if uncertain else "Text held safely. Copy or discard pending text before resuming.")
            if len(text) > remaining:
                self.error = "Pending text exceeded its memory limit and was truncated. Finish this session."
                self.finishing = True
                self.stop_capture.set()

    def _read(self) -> None:
        try:
            self.capture(self.stop_capture, self.feed, self.capture_token)
        except Exception:
            with self.lock:
                self.error = "Microphone failed. Session stopped; no automatic retry."
            self.cancel.set()
            self.stop_capture.set()
        finally:
            with self.lock:
                if not self.cancel.is_set():
                    if self.mode == "hands_free" and self.frames:
                        self._enqueue_locked(bytes(self.frames))
                    else:
                        self._flush_locked()
                self.frames.clear()
                self.segmenter.reset()
            self.capture_complete.set()
            self.wake.set()
            self.changed()

    def _work(self) -> None:
        try:
            while True:
                self.wake.wait(0.1)
                self.wake.clear()
                if self.cancel.is_set():
                    break
                with self.lock:
                    # Mute blocks new submissions, not a request already sent.
                    allowed = not self.muted or self.finishing or self.backlog_paused
                    job = self.jobs.popleft() if self.jobs and allowed else None
                    if job is None:
                        if self.capture_complete.is_set() and not self.jobs:
                            break
                        continue
                    self.processing = True
                try:
                    result = self.transcribe(job.pcm, self.cancel)
                    if self.cancel.is_set():
                        break
                    self.last_metadata = dict(result.metadata or {})
                    text = str(result.text or "").strip()
                    if not text:
                        continue
                    # Live chunks never reformat or replace previously committed text.
                    if self.mode == "live":
                        from utils.dictation_text import prepare_dictation_text
                        text = prepare_dictation_text(text, preserve_formatting=False)
                        result = replace(result, text=text, metadata={**(result.metadata or {}),
                                         "live_chunk": True})
                    guard = DeliveryCancelled(self, job.epoch)
                    if guard.is_set():
                        self._hold_result(text)
                    else:
                        try:
                            self.deliver(result, guard)
                            self.committed.append(text + (" " if self.mode == "live" else ""))
                        except Exception:
                            self._hold_result(text, uncertain=True)
                except Exception:
                    with self.lock:
                        self.error = "Transcription failed; session paused. No automatic retry was made. Later queued audio was discarded."
                        self._mute_locked()
                        # Never continue delivering later phrases across a missing phrase.
                        self.jobs.clear()
                finally:
                    with self.lock:
                        self.processing = False
                    self.changed()
        finally:
            # Do not release session ownership while native capture still owns the mic.
            self.stop_capture.set()
            if not self.capture_complete.wait(3.0):
                with self.lock:
                    self.error = "Microphone has not closed. Recording stays blocked until its owner exits."
                    self.stage = "stalled"
                self.changed()
                while not self.capture_complete.wait(0.5):
                    pass  # Keep ownership, never pretend a native stream has stopped.
            with self.lock:
                self.jobs.clear()
                self.frames.clear()
                self.stage = "cancelled" if self.cancel.is_set() else "finished"
            self.done.set()
            self.changed()
