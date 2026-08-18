from __future__ import annotations

from array import array
import io
import math
import sys
import wave

import pytest

from utils.audio_cues import (
    CueKind,
    CuePlayer,
    DEFAULT_PEAK_CEILING,
    MAX_CUE_SECONDS,
    Pcm16Cue,
    cue_to_wav_bytes,
    load_wav_cue,
    peak_fraction,
    prepare_cue,
    synthesize_cue,
)


pytestmark = pytest.mark.core_headless


def _pcm16(values: list[int]) -> bytes:
    samples = array("h", values)
    if sys.byteorder == "big":
        samples.byteswap()
    return samples.tobytes()


def _decode(frames: bytes) -> list[int]:
    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder == "big":
        samples.byteswap()
    return list(samples)


@pytest.mark.parametrize("kind", list(CueKind))
def test_synthesized_midnight_signal_cues_are_short_and_headphone_safe(kind: CueKind) -> None:
    cue = synthesize_cue(kind)
    assert cue.kind is kind
    assert 0 < cue.duration_seconds < 0.5
    assert peak_fraction(cue) <= DEFAULT_PEAK_CEILING
    assert max(abs(sample) for sample in _decode(cue.frames)) > 0


def test_prepare_cue_applies_limit_volume_and_click_preventing_fades() -> None:
    source = Pcm16Cue(_pcm16([30_000] * 1000), sample_rate=10_000)
    prepared = prepare_cue(source, volume=0.8, peak_ceiling=0.2, fade_ms=10)
    samples = _decode(prepared.frames)

    assert source.frames != prepared.frames
    assert samples[0] == 0
    assert samples[-1] == 0
    assert max(abs(sample) for sample in samples) <= round(32767 * 0.2)
    assert prepared.duration_seconds == source.duration_seconds


def test_wav_loader_accepts_small_pcm16_and_rejects_oversized_clip(tmp_path) -> None:
    valid = tmp_path / "valid.wav"
    with wave.open(str(valid), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8_000)
        handle.writeframes(_pcm16([1000] * 800))

    loaded = load_wav_cue(valid, kind=CueKind.SUCCESS)
    assert math.isclose(loaded.duration_seconds, 0.1)
    assert loaded.kind is CueKind.SUCCESS

    too_long = tmp_path / "long.wav"
    with wave.open(str(too_long), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8_000)
        handle.writeframes(_pcm16([0] * (int(MAX_CUE_SECONDS * 8_000) + 1)))
    with pytest.raises(ValueError, match="must not exceed"):
        load_wav_cue(too_long)


def test_wav_loader_rejects_non_pcm16_width(tmp_path) -> None:
    source = tmp_path / "eight-bit.wav"
    with wave.open(str(source), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(1)
        handle.setframerate(8_000)
        handle.writeframes(b"\x80" * 80)
    with pytest.raises(ValueError, match="16-bit"):
        load_wav_cue(source)


def test_cue_to_wav_bytes_preserves_prepared_pcm_and_format() -> None:
    cue = Pcm16Cue(
        _pcm16([0, 1200, -1200, 0, 500, -500]),
        sample_rate=16_000,
        channels=2,
        kind=CueKind.RECORDING_STARTED,
    )

    payload = cue_to_wav_bytes(cue)

    assert payload[:4] == b"RIFF"
    assert payload[8:12] == b"WAVE"
    with wave.open(io.BytesIO(payload), "rb") as handle:
        assert handle.getnchannels() == 2
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 16_000
        assert handle.getnframes() == cue.frame_count
        assert handle.readframes(handle.getnframes()) == cue.frames


def test_cue_player_obeys_enabled_setting_and_isolates_sink_failure(caplog) -> None:
    delivered: list[Pcm16Cue] = []
    player = CuePlayer(delivered.append, volume=0.5)
    assert player.play(CueKind.SUCCESS, background=False) is True
    assert len(delivered) == 1
    assert peak_fraction(delivered[0]) <= DEFAULT_PEAK_CEILING

    player.enabled = False
    assert player.play(CueKind.ERROR, background=False) is False
    assert len(delivered) == 1

    def broken_sink(cue: Pcm16Cue) -> None:
        raise RuntimeError("device unavailable")

    broken = CuePlayer(broken_sink)
    assert broken.play(CueKind.ERROR, background=False) is True
    assert "cue playback failed" in caplog.text.casefold()


def test_cue_player_reports_background_thread_start_failure_without_wedging(
    monkeypatch, caplog
) -> None:
    from utils import audio_cues

    delivered: list[Pcm16Cue] = []

    class RefusedThread:
        def __init__(self, *, target, **_kwargs) -> None:
            self.target = target

        def start(self) -> None:
            raise RuntimeError("interpreter is shutting down")

    monkeypatch.setattr(audio_cues.threading, "Thread", RefusedThread)
    player = CuePlayer(delivered.append)

    assert player.play(CueKind.CANCELLED, background=True) is False
    assert delivered == []
    assert "cue worker failed to start" in caplog.text.casefold()

    # Cancellation/release code can continue, and a later synchronous cue is
    # still usable; CuePlayer retains no pending/started lifecycle state.
    assert player.play(CueKind.SUCCESS, background=False) is True
    assert [cue.kind for cue in delivered] == [CueKind.SUCCESS]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"volume": -0.1},
        {"volume": 1.1},
        {"peak_ceiling": 0.0},
        {"peak_ceiling": 1.1},
        {"fade_ms": -1},
    ],
)
def test_prepare_cue_rejects_unsafe_controls(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        prepare_cue(Pcm16Cue(b""), **kwargs)
