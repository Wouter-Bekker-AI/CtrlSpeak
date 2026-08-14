from __future__ import annotations

from array import array
import sys

import pytest

from utils import system


pytestmark = pytest.mark.core_headless


def _pcm16(values: list[int]) -> bytes:
    samples = array("h", values)
    if sys.byteorder == "big":
        samples.byteswap()
    return samples.tobytes()


def _decode_pcm16(frames: bytes) -> list[int]:
    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder == "big":
        samples.byteswap()
    return list(samples)


def test_processing_sound_peak_is_reduced_without_hard_clipping() -> None:
    source = [-32768, -24000, -12000, 0, 12000, 24000, 32767]

    processed = _decode_pcm16(
        system.limit_processing_sound_peak(
            _pcm16(source),
            {"channels": 1, "rate": 44100, "width": 2},
        )
    )

    peak = max(abs(sample) for sample in processed)
    expected_ceiling = round(32767 * system.PROCESSING_SOUND_MAX_PEAK)
    assert peak <= expected_ceiling
    assert all(left < right for left, right in zip(processed, processed[1:]))
    assert len(set(processed)) == len(processed)


def test_quiet_processing_sound_is_left_bit_for_bit_unchanged() -> None:
    source = _pcm16([-4000, -1000, 0, 1000, 4000])

    assert system.limit_processing_sound_peak(
        source,
        {"channels": 1, "rate": 44100, "width": 2},
    ) == source


def test_packaged_processing_chime_respects_headphone_safe_peak_ceiling(monkeypatch) -> None:
    monkeypatch.setattr(system, "processing_sound_data", None)
    monkeypatch.setattr(system, "processing_sound_settings", None)

    frames, settings_audio = system.load_processing_sound()
    samples = _decode_pcm16(frames)
    peak = max(abs(sample) for sample in samples)

    assert settings_audio == {"channels": 2, "rate": 44100, "width": 2}
    assert peak <= round(32767 * system.PROCESSING_SOUND_MAX_PEAK)
