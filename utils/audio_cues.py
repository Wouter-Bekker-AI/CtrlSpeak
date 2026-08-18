"""Headphone-safe, dependency-free audio cues for Midnight Signal.

This module prepares small in-memory PCM clips and delegates actual playback to
an injected sink.  Importing it never initializes an audio device, which keeps
headless tests and Linux gateway processes unaffected.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass, replace
from enum import Enum
import logging
import math
from pathlib import Path
import sys
import threading
import wave
from typing import Callable, Protocol


LOGGER = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 44_100
DEFAULT_PEAK_CEILING = 0.25  # -12.04 dBFS
MAX_CUE_SECONDS = 3.0
MAX_SAMPLE_RATE = 96_000


class CueKind(str, Enum):
    RECORDING_STARTED = "recording_started"
    PROCESSING_STARTED = "processing_started"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Pcm16Cue:
    """A bounded 16-bit PCM clip; frames are hidden from repr diagnostics."""

    frames: bytes
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = 1
    kind: CueKind | None = None

    def __repr__(self) -> str:
        return (
            f"Pcm16Cue(frames=<{len(self.frames)} bytes>, "
            f"sample_rate={self.sample_rate}, channels={self.channels}, kind={self.kind!r})"
        )

    @property
    def frame_count(self) -> int:
        return len(self.frames) // (2 * self.channels)

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.sample_rate


class CueSink(Protocol):
    def __call__(self, cue: Pcm16Cue) -> None: ...


def _validate_cue(cue: Pcm16Cue) -> None:
    if cue.channels not in {1, 2}:
        raise ValueError("cue must be mono or stereo")
    if not 8_000 <= cue.sample_rate <= MAX_SAMPLE_RATE:
        raise ValueError("cue sample rate must be between 8000 and 96000 Hz")
    if len(cue.frames) % (2 * cue.channels):
        raise ValueError("cue frames are not aligned 16-bit PCM")
    if cue.duration_seconds > MAX_CUE_SECONDS:
        raise ValueError(f"cue must not exceed {MAX_CUE_SECONDS:.1f} seconds")


def _pcm16_samples(frames: bytes) -> array:
    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder == "big":
        samples.byteswap()
    return samples


def peak_fraction(cue: Pcm16Cue) -> float:
    _validate_cue(cue)
    samples = _pcm16_samples(cue.frames)
    return (max((abs(sample) for sample in samples), default=0) / 32768.0)


def prepare_cue(
    cue: Pcm16Cue,
    *,
    volume: float = 1.0,
    peak_ceiling: float = DEFAULT_PEAK_CEILING,
    fade_ms: float = 8.0,
) -> Pcm16Cue:
    """Apply volume, transparent peak limiting, and click-preventing fades."""

    _validate_cue(cue)
    if not math.isfinite(volume) or not 0 <= volume <= 1:
        raise ValueError("cue volume must be in the range [0, 1]")
    if not math.isfinite(peak_ceiling) or not 0 < peak_ceiling <= 1:
        raise ValueError("peak ceiling must be in the range (0, 1]")
    if not math.isfinite(fade_ms) or fade_ms < 0:
        raise ValueError("fade duration must be non-negative")

    samples = _pcm16_samples(cue.frames)
    if not samples:
        return cue
    source_peak = max(abs(sample) for sample in samples)
    ceiling_sample = max(1, round(32767 * peak_ceiling))
    gain = volume
    if source_peak and source_peak * gain > ceiling_sample:
        gain = ceiling_sample / source_peak

    frame_count = cue.frame_count
    fade_frames = min(frame_count // 2, round(cue.sample_rate * fade_ms / 1000))
    output = array("h")
    for index, sample in enumerate(samples):
        frame_index = index // cue.channels
        envelope = 1.0
        if fade_frames:
            if frame_index < fade_frames:
                envelope = frame_index / fade_frames
            elif frame_index >= frame_count - fade_frames:
                envelope = max(0.0, (frame_count - 1 - frame_index) / fade_frames)
        scaled = round(sample * gain * envelope)
        output.append(max(-32768, min(32767, scaled)))
    if sys.byteorder == "big":
        output.byteswap()
    return replace(cue, frames=output.tobytes())


_CUE_NOTES: dict[CueKind, tuple[tuple[float, float], ...]] = {
    CueKind.RECORDING_STARTED: ((330.0, 0.06), (440.0, 0.07)),
    CueKind.PROCESSING_STARTED: ((294.0, 0.08), (370.0, 0.08)),
    CueKind.SUCCESS: ((392.0, 0.07), (523.25, 0.11)),
    CueKind.ERROR: ((246.94, 0.09), (196.0, 0.13)),
    CueKind.CANCELLED: ((293.66, 0.07), (246.94, 0.09)),
}


def synthesize_cue(
    kind: CueKind,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    peak_ceiling: float = DEFAULT_PEAK_CEILING,
) -> Pcm16Cue:
    """Create a restrained low/mid-frequency cue with no harsh high-pitch spike."""

    if not 8_000 <= sample_rate <= MAX_SAMPLE_RATE:
        raise ValueError("cue sample rate must be between 8000 and 96000 Hz")
    samples = array("h")
    gap_frames = round(sample_rate * 0.018)
    for note_index, (frequency, duration) in enumerate(_CUE_NOTES[kind]):
        note_frames = max(1, round(sample_rate * duration))
        fade_frames = max(1, min(note_frames // 3, round(sample_rate * 0.012)))
        for frame in range(note_frames):
            attack = min(1.0, frame / fade_frames)
            release = min(1.0, (note_frames - 1 - frame) / fade_frames)
            envelope = max(0.0, min(attack, release))
            # A quiet fundamental plus very small second harmonic sounds warm
            # while remaining legible on laptop speakers.
            phase = 2 * math.pi * frequency * frame / sample_rate
            value = (math.sin(phase) + 0.08 * math.sin(phase * 2)) / 1.08
            samples.append(round(value * envelope * 32767 * peak_ceiling))
        if note_index < len(_CUE_NOTES[kind]) - 1:
            samples.extend([0] * gap_frames)
    if sys.byteorder == "big":
        samples.byteswap()
    cue = Pcm16Cue(samples.tobytes(), sample_rate, 1, kind)
    _validate_cue(cue)
    return cue


def load_wav_cue(path: str | Path, *, kind: CueKind | None = None) -> Pcm16Cue:
    """Load one small PCM WAV cue after validating format and duration bounds."""

    source = Path(path)
    with wave.open(str(source), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frame_count = handle.getnframes()
        if sample_width != 2:
            raise ValueError("cue WAV must contain 16-bit PCM audio")
        duration = frame_count / sample_rate if sample_rate else math.inf
        provisional = Pcm16Cue(b"", sample_rate, channels, kind)
        _validate_cue(provisional)
        if duration > MAX_CUE_SECONDS:
            raise ValueError(f"cue must not exceed {MAX_CUE_SECONDS:.1f} seconds")
        frames = handle.readframes(frame_count)
    cue = Pcm16Cue(frames, sample_rate, channels, kind)
    _validate_cue(cue)
    return cue


class CuePlayer:
    """Small safe wrapper around an injected OS/audio-library playback sink."""

    def __init__(
        self,
        sink: CueSink,
        *,
        enabled: bool = True,
        volume: float = 0.65,
        peak_ceiling: float = DEFAULT_PEAK_CEILING,
    ) -> None:
        if not callable(sink):
            raise TypeError("cue sink must be callable")
        # Validate controls through prepare_cue without touching an audio device.
        prepare_cue(Pcm16Cue(b""), volume=volume, peak_ceiling=peak_ceiling)
        self._sink = sink
        self.enabled = bool(enabled)
        self.volume = float(volume)
        self.peak_ceiling = float(peak_ceiling)

    def play(self, kind: CueKind, *, background: bool = True) -> bool:
        if not self.enabled:
            return False
        cue = prepare_cue(
            synthesize_cue(kind, peak_ceiling=self.peak_ceiling),
            volume=self.volume,
            peak_ceiling=self.peak_ceiling,
        )

        def deliver() -> None:
            try:
                self._sink(cue)
            except Exception:
                # Never let a missing/disconnected output device break capture.
                LOGGER.exception("CtrlSpeak UI cue playback failed")

        if background:
            threading.Thread(
                target=deliver,
                name=f"CtrlSpeakCue-{kind.value}",
                daemon=True,
            ).start()
        else:
            deliver()
        return True
