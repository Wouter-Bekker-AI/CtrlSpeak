from __future__ import annotations

from array import array
import math
import sys

import pytest

from utils.ui_state import (
    AttemptOutcome,
    DbfsSmoother,
    InvalidUiTransition,
    ProviderState,
    TranscriptionUiSession,
    UiPhase,
    active_route_label,
    format_elapsed_ms,
    format_latency_ms,
    pcm16_dbfs,
    providers_from_capabilities,
)


pytestmark = pytest.mark.core_headless


def _pcm16(values: list[int]) -> bytes:
    samples = array("h", values)
    if sys.byteorder == "big":
        samples.byteswap()
    return samples.tobytes()


class FakeClock:
    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_pcm16_dbfs_handles_silence_full_scale_and_partial_frame() -> None:
    assert pcm16_dbfs(b"") == -60.0
    assert pcm16_dbfs(_pcm16([0, 0, 0])) == -60.0
    assert math.isclose(pcm16_dbfs(_pcm16([32767, -32768])), 0.0, abs_tol=0.01)
    assert math.isclose(
        pcm16_dbfs(_pcm16([16384, -16384]) + b"\xff"), -6.02, abs_tol=0.02
    )


def test_dbfs_smoother_uses_fast_attack_and_gentle_release() -> None:
    meter = DbfsSmoother(attack=0.5, release=0.1)
    assert math.isclose(meter.update(-10.0), -35.0)
    assert math.isclose(meter.update(-60.0), -37.5)
    assert meter.reset() == -60.0


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "—"),
        (0.4, "<1 ms"),
        (248.7, "249 ms"),
        (1250, "1.25 s"),
        (12_540, "12.5 s"),
        (65_000, "1m 5s"),
    ],
)
def test_latency_formatting(value: object, expected: str) -> None:
    assert format_latency_ms(value) == expected


def test_elapsed_formatting_is_suitable_for_compact_capsule() -> None:
    assert format_elapsed_ms(12_345) == "12.3s"
    assert format_elapsed_ms(75_250) == "1:15.2"


def test_capabilities_are_normalized_without_secret_or_transport_fields() -> None:
    capabilities = {
        "providers": [
            {
                "id": "ubuntu-gpu-large-v3-turbo",
                "model": "large-v3-turbo",
                "device": "CUDA",
                "status": "ready",
                "health": {
                    "probe_duration_ms": 12.5,
                    "probe_age_ms": 230.0,
                    "circuit_retry_after_ms": 0.0,
                    "private_detail": "must-not-survive-either",
                },
                "private_token": "must-not-survive",
            },
            {
                "id": "openai-gpt-transcribe",
                "model": "gpt-4o-transcribe",
                "status": "available_with_key",
                "credential_header": "X-CtrlSpeak-OpenAI-Key",
            },
            {"id": "nova-tiny-whisper", "status": "starting"},
        ]
    }

    providers = providers_from_capabilities(
        capabilities,
        active_provider_id="ubuntu-gpu-large-v3-turbo",
        latency_by_provider={"ubuntu-gpu-large-v3-turbo": 184.2},
    )

    assert [item.display_name for item in providers] == [
        "Ubuntu GPU",
        "OpenAI",
        "Gateway Tiny",
    ]
    assert providers[0].active is True
    assert providers[0].device == "cuda"
    assert providers[0].latency_ms == 184.2
    assert providers[0].probe_duration_ms == 12.5
    assert providers[0].probe_age_ms == 230.0
    assert providers[0].circuit_retry_after_ms == 0.0
    assert providers[1].state is ProviderState.REQUIRES_KEY
    assert providers[2].state is ProviderState.STARTING
    assert "must-not-survive" not in repr(providers)
    assert "must-not-survive-either" not in repr(providers)
    assert "X-CtrlSpeak-OpenAI-Key" not in repr(providers)


def test_ui_session_tracks_full_lifecycle_without_retaining_transcript_or_key() -> None:
    clock = FakeClock()
    session = TranscriptionUiSession(clock=clock)

    idle = session.snapshot()
    assert idle.phase is UiPhase.IDLE
    assert idle.can_cancel is False

    session.begin_recording()
    session.update_level_pcm16(_pcm16([16384, -16384] * 50))
    clock.advance(1.2)
    recording = session.snapshot()
    assert recording.phase is UiPhase.RECORDING
    assert math.isclose(recording.elapsed_ms, 1200)
    assert recording.can_cancel is True
    assert recording.level_dbfs > -60

    session.begin_processing()
    clock.advance(0.8)
    session.complete(
        {
            "provider_used": "openai-gpt-transcribe",
            "degraded": True,
            "attempts": [
                {
                    "provider": "ubuntu-gpu-large-v3-turbo",
                    "category": "worker_unavailable",
                    "message": "internal network detail",
                    "duration_ms": 85.4,
                },
                {
                    "provider": "openai-gpt-transcribe",
                    "status": "succeeded",
                    "duration_ms": 1732.6,
                    "inference_duration_ms": 1410.0,
                },
                {"provider": "another-provider", "category": "sk-secret-category"},
            ],
            "routing_duration_ms": 1820.0,
            "text": "private transcript",
            "raw_text": "private raw transcript",
            "openai_api_key": "sk-secret",
        },
        elapsed_ms=1984,
    )
    complete = session.snapshot()

    assert complete.phase is UiPhase.SUCCESS
    assert complete.elapsed_ms == 1984
    assert complete.provider is not None
    assert complete.provider.display_name == "OpenAI"
    assert complete.provider.latency_ms == 1732.6
    assert complete.provider.inference_ms == 1410.0
    assert complete.degraded is True
    assert [attempt.outcome for attempt in complete.attempts] == [
        AttemptOutcome.FAILED,
        AttemptOutcome.SUCCEEDED,
        AttemptOutcome.FAILED,
    ]
    assert active_route_label(complete.attempts) == "Ubuntu GPU → OpenAI → Another Provider"
    assert complete.attempts[0].duration_label == "85 ms"
    assert "private transcript" not in repr(complete)
    assert "sk-secret" not in repr(complete)


def test_ui_session_failure_and_cancel_have_safe_user_copy() -> None:
    failure_clock = FakeClock()
    failed = TranscriptionUiSession(clock=failure_clock)
    failed.begin_recording()
    failed.begin_processing()
    failed.fail("openai_quota_exhausted")
    snapshot = failed.snapshot()
    assert snapshot.phase is UiPhase.ERROR
    assert "quota" in snapshot.detail.casefold()

    cancelled = TranscriptionUiSession(clock=FakeClock())
    cancelled.begin_recording()
    cancelled.cancel()
    assert cancelled.snapshot().phase is UiPhase.CANCELLED
    assert cancelled.snapshot().headline == "Cancelled"


def test_ui_session_rejects_impossible_transition() -> None:
    session = TranscriptionUiSession(clock=FakeClock())
    with pytest.raises(InvalidUiTransition, match="idle to processing"):
        session.begin_processing()
