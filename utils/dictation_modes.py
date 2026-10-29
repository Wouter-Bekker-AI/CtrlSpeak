"""Pure gesture recognition and lossless PCM phrase boundaries for dictation.

There are deliberately no microphone, GUI, network, keyboard-hook or injection
side effects here. A caller serializes keyboard events and calls ``tick`` while
the primary key is held. Its provisional microphone pre-roll can start on the
initial press, but must not be submitted until a start action is emitted.

The segmenter is energy-based endpoint detection, not speaker identification.
Only feed unmuted PCM. ``flush`` completes pre-mute speech and ``reset`` discards
remaining pre-roll; never buffer muted speech to replay after unmuting.
"""
from __future__ import annotations

from array import array
from collections import deque
from enum import Enum
import math
import sys
import time
from typing import Callable


class DictationMode(str, Enum):
    HOLD = "hold"
    HANDS_FREE = "hands_free"
    LIVE = "live"


class GestureAction(str, Enum):
    START_HOLD = "start_hold"
    START_HANDS_FREE = "start_hands_free"
    START_LIVE = "start_live"
    FINISH = "finish"
    TOGGLE_MUTE = "toggle_mute"
    CANCEL = "cancel"
    ABORT_HOLD = "abort_hold"


MODE_LABELS = {
    DictationMode.HOLD: "Hold to Talk",
    DictationMode.HANDS_FREE: "Hands-free Dictation",
    DictationMode.LIVE: "Live Dictation",
}

SUPPORTED_GESTURE_KEYS = ("ctrl_r", "ctrl_l", "f8", "f9", "f10")


def _key_name(key: object) -> str:
    char = getattr(key, "char", None)
    value = char if isinstance(char, str) else str(key)
    return value.strip().lower().removeprefix("key.")


class DictationGestures:
    """Recognize isolated taps without stealing normal editing chords.

    ``mode`` is the currently committed runtime mode, or None while idle (also
    while a provisional hold recording is waiting for the hold threshold).
    Events from injected input must be passed with ``injected=True``.

    Primary hold: tick emits START_HOLD; release emits FINISH. A short first
    tap does nothing; a second short tap emits START_HANDS_FREE on release.
    Primary + secondary, in either order: START_LIVE after both are released.
    In hands-free/live, isolated primary/secondary taps finish/toggle mute.
    All other key combinations invalidate the tap. If a held recording was
    already committed before an editing chord, ABORT_HOLD prevents submission.

    Call reset on lock/sleep, listener replacement, or explicit session cancel.
    This object creates no timers/threads and must be serialized by its caller.
    """

    def __init__(
        self,
        *,
        primary_key: str = "ctrl_r",
        secondary_key: str = "ctrl_l",
        hold_seconds: float = 0.18,
        double_tap_seconds: float = 0.30,
        tap_max_seconds: float = 0.35,
        cancel_key: str | None = "esc",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.primary_key = _key_name(primary_key)
        self.secondary_key = _key_name(secondary_key)
        if self.primary_key not in SUPPORTED_GESTURE_KEYS:
            raise ValueError("Unsupported primary dictation key")
        if self.secondary_key not in SUPPORTED_GESTURE_KEYS:
            raise ValueError("Unsupported secondary dictation key")
        if self.primary_key == self.secondary_key:
            raise ValueError("Dictation gesture keys must be distinct")
        for value in (hold_seconds, double_tap_seconds, tap_max_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Gesture timing must be finite and positive")
        self.hold_seconds = hold_seconds
        self.double_tap_seconds = double_tap_seconds
        self.tap_max_seconds = tap_max_seconds
        self.cancel_key = _key_name(cancel_key) if cancel_key else None
        self._clock = clock
        self.reset()

    @property
    def keys_down(self) -> frozenset[str]:
        return frozenset(self._down)

    def is_down(self, key: str) -> bool:
        return _key_name(key) in self._down

    def reset(self) -> None:
        self._down: set[str] = set()
        self._pressed_at: dict[str, float] = {}
        self._dirty: set[str] = set()
        self._first_tap_at: float | None = None
        self._second_tap = False
        self._hold_emitted = False
        self._live_chord = False
        self._live_blocked = False

    def _now(self, now: float | None) -> float:
        return self._clock() if now is None else now

    def _clear_taps(self) -> None:
        self._first_tap_at = None
        self._second_tap = False

    def press(
        self, key: object, *, mode: DictationMode | str | None = None,
        injected: bool = False, now: float | None = None,
    ) -> tuple[GestureAction, ...]:
        if injected:
            return ()
        key = _key_name(key)
        if key in self._down:  # Hardware autorepeat is not another gesture.
            return ()
        moment = self._now(now)
        controls = {self.primary_key, self.secondary_key}
        previous = set(self._down)
        previous_clean = not bool(previous & self._dirty)
        self._down.add(key)
        self._pressed_at[key] = moment
        self._dirty.discard(key)
        actions: list[GestureAction] = []

        if key == self.cancel_key and (mode is not None or self._hold_emitted):
            self._dirty.update(self._down)
            self._clear_taps()
            self._hold_emitted = False
            self._live_blocked = True
            return (GestureAction.CANCEL,)

        if key not in controls:
            self._dirty.update(self._down & controls)
            self._clear_taps()
            self._live_blocked = True
            if self._hold_emitted:
                self._hold_emitted = False
                actions.append(GestureAction.ABORT_HOLD)
            return tuple(actions)

        if previous:
            self._dirty.update(previous & controls)
            self._dirty.add(key)
            self._clear_taps()
            if previous_clean and previous == controls - {key} and mode in (None, DictationMode.HOLD):
                self._live_chord = True
                self._live_blocked = False
            else:
                self._live_blocked = True
            if self._hold_emitted:
                self._hold_emitted = False
                actions.append(GestureAction.ABORT_HOLD)
        elif key == self.primary_key:
            self._live_blocked = False
            self._second_tap = (
                mode is None and self._first_tap_at is not None
                and moment - self._first_tap_at <= self.double_tap_seconds
            )
            if not self._second_tap:
                self._first_tap_at = None
        else:
            self._clear_taps()
            self._live_blocked = False
        return tuple(actions)

    def tick(
        self, *, mode: DictationMode | str | None = None,
        now: float | None = None,
    ) -> tuple[GestureAction, ...]:
        moment = self._now(now)
        if (
            mode is None and self._down == {self.primary_key}
            and self.primary_key not in self._dirty
            and not self._hold_emitted and not self._live_chord
            and moment - self._pressed_at[self.primary_key] >= self.hold_seconds
        ):
            self._clear_taps()
            self._hold_emitted = True
            return (GestureAction.START_HOLD,)
        if (
            not self._down and self._first_tap_at is not None
            and moment - self._first_tap_at > self.double_tap_seconds
        ):
            self._clear_taps()
        return ()

    def release(
        self, key: object, *, mode: DictationMode | str | None = None,
        injected: bool = False, now: float | None = None,
    ) -> tuple[GestureAction, ...]:
        if injected:
            return ()
        key = _key_name(key)
        if key not in self._down:  # Lost or synthetic releases cannot toggle.
            return ()
        moment = self._now(now)
        duration = moment - self._pressed_at.pop(key)
        isolated = key not in self._dirty and self._down == {key}
        self._down.remove(key)
        self._dirty.discard(key)
        controls = {self.primary_key, self.secondary_key}

        if self._live_chord:
            if not (self._down & controls):
                should_start = (
                    not self._live_blocked and not self._down
                    and mode in (None, DictationMode.HOLD)
                )
                self._live_chord = False
                self._live_blocked = False
                self._clear_taps()
                return (GestureAction.START_LIVE,) if should_start else ()
            return ()

        if key == self.primary_key and self._hold_emitted:
            self._hold_emitted = False
            self._clear_taps()
            if mode not in (None, DictationMode.HOLD):
                return ()
            return (GestureAction.FINISH if isolated else GestureAction.ABORT_HOLD,)

        if not isolated:
            if key in controls:
                self._clear_taps()
            return ()

        if mode in (DictationMode.HANDS_FREE, DictationMode.LIVE):
            self._clear_taps()
            if duration <= self.tap_max_seconds:
                if key == self.primary_key:
                    return (GestureAction.FINISH,)
                if key == self.secondary_key:
                    return (GestureAction.TOGGLE_MUTE,)
            return ()

        if mode is None and key == self.primary_key and duration < self.hold_seconds:
            if self._second_tap:
                self._clear_taps()
                return (GestureAction.START_HANDS_FREE,)
            self._first_tap_at = moment
            self._second_tap = False
        elif key in controls:
            self._clear_taps()
        return ()


class SpeechSegmenter:
    """Emit bounded, non-overlapping PCM16LE mono phrases, without resampling.

    Output contains exactly the original bytes of each retained sample. Leading
    silence beyond pre-roll is omitted. Silence alone never creates a request.
    Long speech is cut at a bounded analysis-frame boundary; this is phrase
    chunking for a batch provider, not token streaming or speaker recognition.

    The maximum duration is rounded down to a whole analysis frame. Callers
    own queue bounds, cancellation, privacy gates, and WAV container creation.
    """

    def __init__(
        self, *, sample_rate: int = 44100, threshold_dbfs: float = -38.0,
        silence_seconds: float = 0.65, min_speech_seconds: float = 0.25,
        max_chunk_seconds: float = 8.0, pre_roll_seconds: float = 0.20,
        frame_seconds: float = 0.02,
    ) -> None:
        if not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 192000:
            raise ValueError("Unsupported mono PCM sample rate")
        for value in (silence_seconds, min_speech_seconds, max_chunk_seconds, frame_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Segment timing must be finite and positive")
        if not math.isfinite(pre_roll_seconds) or pre_roll_seconds < 0:
            raise ValueError("Pre-roll must be finite and non-negative")
        if not math.isfinite(threshold_dbfs) or not -80 <= threshold_dbfs <= 0:
            raise ValueError("Invalid speech energy threshold")
        if max_chunk_seconds < max(frame_seconds, min_speech_seconds):
            raise ValueError("Maximum chunk must accommodate a speech frame")
        if pre_roll_seconds + min_speech_seconds > max_chunk_seconds:
            raise ValueError("Maximum chunk must accommodate pre-roll and speech")
        self.sample_rate = sample_rate
        self.threshold_dbfs = threshold_dbfs
        self._frame_samples = max(1, int(sample_rate * frame_seconds))
        self._frame_bytes = self._frame_samples * 2
        self._silence_samples = max(1, math.ceil(sample_rate * silence_seconds))
        self._min_speech_samples = max(1, math.ceil(sample_rate * min_speech_seconds))
        self._max_bytes = max(
            self._frame_bytes,
            int(sample_rate * max_chunk_seconds) // self._frame_samples * self._frame_bytes,
        )
        self._pre_roll_bytes = int(sample_rate * pre_roll_seconds) * 2
        self._threshold_squared = (32768.0 * 10.0 ** (threshold_dbfs / 20.0)) ** 2
        self.reset()

    @property
    def speech_active(self) -> bool:
        return bool(self._active)

    @property
    def buffered_bytes(self) -> int:
        return len(self._input) + len(self._active) + self._pre_bytes

    def reset(self) -> None:
        self._input = bytearray()
        self._active = bytearray()
        self._pre: deque[bytes] = deque()
        self._pre_bytes = 0
        self._speech_samples = 0
        self._silence_run = 0
        self._continuation = False
        self._accept_short = False

    def _voiced(self, frame: bytes) -> bool:
        samples = array("h")
        samples.frombytes(frame)
        if sys.byteorder != "little":
            samples.byteswap()
        return bool(samples) and sum(sample * sample for sample in samples) / len(samples) >= self._threshold_squared

    def _retain_pre_roll(self, frame: bytes) -> None:
        if not self._pre_roll_bytes:
            return
        self._pre.append(frame)
        self._pre_bytes += len(frame)
        while self._pre_bytes > self._pre_roll_bytes:
            excess = self._pre_bytes - self._pre_roll_bytes
            oldest = self._pre.popleft()
            if len(oldest) <= excess:
                self._pre_bytes -= len(oldest)
            else:
                self._pre.appendleft(oldest[excess:])
                self._pre_bytes -= excess

    def _finish_active(self, *, continuation: bool = False) -> list[bytes]:
        accepted = self._speech_samples >= self._min_speech_samples or (
            self._accept_short and self._speech_samples > 0
        )
        output = [bytes(self._active)] if self._active and accepted else []
        self._active.clear()
        self._speech_samples = 0
        self._silence_run = 0
        self._pre.clear()
        self._pre_bytes = 0
        self._continuation = continuation and accepted
        self._accept_short = False
        return output

    def _frame(self, frame: bytes) -> list[bytes]:
        count = len(frame) // 2
        voiced = self._voiced(frame)
        output: list[bytes] = []
        if not self._active:
            if not voiced:
                self._silence_run += count
                if self._silence_run >= self._silence_samples:
                    self._continuation = False
                self._retain_pre_roll(frame)
                return output
            self._active.extend(b"".join(self._pre))
            self._pre.clear()
            self._pre_bytes = 0
            self._accept_short = self._continuation
            self._continuation = False
        # Pre-roll can end mid analysis frame. Split precisely at the cap;
        # remainder bytes are new audio, never overlapping the previous output.
        available = self._max_bytes - len(self._active)
        first, remainder = frame[:available], frame[available:]
        self._active.extend(first)
        if voiced:
            self._speech_samples += len(first) // 2
            self._silence_run = 0
        else:
            self._silence_run += len(first) // 2
        if len(self._active) >= self._max_bytes:
            silence_at_cap = self._silence_run
            # A word can cross the cap after a brief inter-word pause. It is
            # still part of the accepted utterance, even when the exact cap
            # happens to land in silence, and its short tail must not vanish.
            output.extend(self._finish_active(
                continuation=silence_at_cap < self._silence_samples,
            ))
            self._silence_run = silence_at_cap
            if remainder:
                output.extend(self._frame(remainder))
        elif not voiced and self._silence_run >= self._silence_samples:
            output.extend(self._finish_active())
        return output

    def feed(self, pcm16: bytes | bytearray | memoryview) -> list[bytes]:
        """Accept arbitrary byte chunks; preserve an incomplete sample until next feed."""
        self._input.extend(pcm16)
        output: list[bytes] = []
        # Copy once, not repeated front-deletions of a potentially large buffer.
        complete = len(self._input) // self._frame_bytes * self._frame_bytes
        frames = bytes(self._input[:complete])
        del self._input[:complete]
        for offset in range(0, complete, self._frame_bytes):
            output.extend(self._frame(frames[offset:offset + self._frame_bytes]))
        return output

    def flush(self) -> list[bytes]:
        """Finish retained pre-boundary speech and erase all pre-roll/partial state."""
        if len(self._input) % 2:
            self.reset()
            raise ValueError("Incomplete PCM16 sample at recording boundary")
        output: list[bytes] = []
        if self._input:
            output.extend(self._frame(bytes(self._input)))
            self._input.clear()
        output.extend(self._finish_active())
        self.reset()
        return output
