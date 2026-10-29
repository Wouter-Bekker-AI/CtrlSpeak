"""No-display acceptance of persistent dictation state and mute affordances."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from utils.ui_state import dictation_error_notice, dictation_presentation


pytestmark = pytest.mark.core_headless


@pytest.mark.parametrize("mode,label", [("hands_free", "HANDS-FREE"), ("live", "LIVE")])
def test_persistent_modes_are_named_and_have_stop_and_mute_hints(mode, label):
    view = dictation_presentation({"mode": mode, "active": True})
    assert view.headline == f"{label} · LISTENING"
    assert "Left Ctrl: mute" in view.hint
    assert "Right Ctrl: finish" in view.hint
    assert not view.freeze_waveform


def test_manual_mute_takes_precedence_over_editing_and_network_activity():
    view = dictation_presentation({
        "mode": "live", "active": True, "muted": True, "editing": True,
        "stage": "transcribing", "queued_count": 2, "pending_count": 1,
    })
    assert view.headline == "LIVE · MUTED"
    assert "No new audio retained or sent" in view.detail
    assert "1 result(s) held safely" in view.detail
    assert "2 queued" in view.detail
    assert "resume" in view.hint
    assert view.freeze_waveform


def test_editing_has_a_distinct_frozen_indicator():
    view = dictation_presentation({"mode": "live", "active": True, "editing": True})
    assert view.headline == "LIVE · PAUSED FOR EDITING"
    assert "keyboard takes priority" in view.detail
    assert view.freeze_waveform
    assert not view.muted


def test_inactive_pending_results_are_not_presented_as_listening():
    view = dictation_presentation({"mode": "live", "active": False, "pending_count": 2})
    assert view.headline == "TEXT WAITING"
    assert "nothing is auto-inserted" in view.detail
    assert view.hint == "Copy pending · Discard pending"
    assert view.freeze_waveform


def test_remapped_key_labels_follow_the_snapshot():
    view = dictation_presentation({"mode": "hands_free", "active": True, "muted": True,
                                   "primary_label": "F8", "secondary_label": "F9"})
    assert view.hint == "F9: resume · F8: finish"


def test_presentation_does_not_leak_arbitrary_runtime_strings():
    view = dictation_presentation({"mode": "live", "active": True,
        "primary_label": "secret primary", "secondary_label": "secret secondary",
        "stage": "secret stage", "error": "secret error", "transcript": "secret transcript",
        "queued_count": -10, "pending_count": "secret count"})
    assert "secret" not in repr(view)
    assert view.pending_count == view.queued_count == 0


def test_finished_session_is_not_still_marked_muted():
    view = dictation_presentation({"active": False, "muted": True, "editing": True})
    assert view.headline == "READY"
    assert not view.muted and not view.editing


def test_remapped_hold_mode_shows_release_not_mute():
    view = dictation_presentation({"mode": "hold", "active": True, "primary_label": "F8"})
    assert view.hint == "Release F8 to transcribe"
    assert "mute" not in view.hint


def test_partial_insertion_warning_precedes_counts_and_survives_mute_state():
    view = dictation_presentation({
        "mode": "live", "active": True, "muted": True, "pending_count": 1,
        "error": "Insertion interrupted; some text may already exist. Inspect before copying pending text.",
    })
    assert view.headline == "LIVE · MUTED"
    assert view.detail.startswith("Partial insertion: inspect existing text before copying pending text.")
    assert view.notice == "Partial insertion: inspect existing text before copying pending text."
    assert view.freeze_waveform


@pytest.mark.parametrize("raw,expected", [
    ("Target changed. Focus the intended field, then explicitly unmute.", "Target changed:"),
    ("Backlog paused capture. Earlier audio will finish into held text; copy/discard it before resuming.", "Backlog paused capture:"),
    ("Transcription failed; session paused. No automatic retry was made.", "Transcription failed:"),
    ("Microphone has not closed. Recording stays blocked until its owner exits.", "Microphone is still closing:"),
    ("Pending text exceeded its memory limit and was truncated. Finish this session.", "Pending text was truncated"),
])
def test_runtime_reasons_use_fixed_clear_copy(raw, expected):
    assert dictation_error_notice(raw).startswith(expected)


@pytest.mark.parametrize("value", ["RuntimeError: bearer=secret transcript=private", {"private": "payload"}, 123])
def test_unknown_runtime_errors_are_generic_not_echoed(value):
    notice = dictation_error_notice(value)
    assert notice == "Attention required: inspect this session before continuing."


@pytest.mark.parametrize("status", [{"processing": True}, {"stage": "transcribing"}])
def test_live_processing_keeps_listening_visible_with_processing_subtitle(status):
    view = dictation_presentation({"mode": "live", "active": True, **status})
    assert view.headline == "LIVE · LISTENING"
    assert view.detail == "Processing phrase · microphone is still listening"
    assert not view.freeze_waveform


def test_live_processing_does_not_override_editing_pause():
    view = dictation_presentation({"mode": "live", "active": True,
                                   "processing": True, "editing": True})
    assert view.headline == "LIVE · PAUSED FOR EDITING"
    assert "microphone is still listening" not in view.detail
    assert view.freeze_waveform


def test_finished_capture_processing_never_claims_mic_is_listening():
    view = dictation_presentation({"mode": "live", "active": True,
                                   "processing": True, "finishing": True})
    assert view.headline == "LIVE · FINISHING"
    assert "Listening stopped" in view.hint
    assert "microphone is still listening" not in view.detail
    assert view.freeze_waveform


class FakeCanvas:
    def __init__(self):
        self.items = []

    def create_line(self, *args, **kwargs):
        self.items.append(("line", args, kwargs))

    def create_oval(self, *args, **kwargs):
        self.items.append(("oval", args, kwargs))

    def create_text(self, *args, **kwargs):
        self.items.append(("text", args, kwargs))


@pytest.mark.parametrize("width", [320, 548])
def test_muted_overlay_draws_crossed_mic_and_never_reads_audio(monkeypatch, width):
    from utils import midnight_overlay as overlay_module

    monkeypatch.setattr(overlay_module.tk, "ROUND", "round", raising=False)
    overlay = object.__new__(overlay_module.MidnightSignalOverlay)
    overlay._draw_microphone = lambda *_args: None
    overlay.waveform_provider = lambda: pytest.fail("Muted overlay must not request microphone samples")
    canvas = FakeCanvas()
    view = dictation_presentation({"mode": "live", "active": True, "muted": True})
    overlay._draw_dictation_session(canvas, width, 88, SimpleNamespace(elapsed_label="00:12"), view)
    tags = [tag for _kind, _args, kwargs in canvas.items for tag in kwargs.get("tags", ())]
    texts = [kwargs["text"] for kind, _args, kwargs in canvas.items if kind == "text"]
    assert "dictation-muted-microphone" in tags
    assert "dictation-waveform-frozen" in tags
    assert "dictation-listening-indicator" not in tags
    assert "LIVE · MUTED" in texts
    assert any("resume" in text for text in texts)


def test_processing_live_overlay_keeps_a_real_listening_waveform(monkeypatch):
    from utils import midnight_overlay as overlay_module

    monkeypatch.setattr(overlay_module.tk, "ROUND", "round", raising=False)
    overlay = object.__new__(overlay_module.MidnightSignalOverlay)
    overlay._draw_microphone = lambda *_args: None
    reads = []
    overlay.waveform_provider = lambda: reads.append(True) or [0.05, 0.2, 0.6, 0.2, 0.05]
    canvas = FakeCanvas()
    view = dictation_presentation({"mode": "live", "active": True, "processing": True})
    overlay._draw_dictation_session(canvas, 548, 88, SimpleNamespace(elapsed_label="00:12"), view)
    tags = [tag for _kind, _args, kwargs in canvas.items for tag in kwargs.get("tags", ())]
    texts = [kwargs["text"] for kind, _args, kwargs in canvas.items if kind == "text"]
    assert reads == [True]
    assert tags.count("dictation-waveform") == 5
    assert "dictation-waveform-frozen" not in tags
    assert "LIVE · LISTENING" in texts
    assert any("Processing phrase" in text for text in texts)


@pytest.mark.parametrize("paused", [{"editing": True}, {"finishing": True}])
def test_paused_overlay_does_not_read_audio_during_phrase_processing(monkeypatch, paused):
    from utils import midnight_overlay as overlay_module

    monkeypatch.setattr(overlay_module.tk, "ROUND", "round", raising=False)
    overlay = object.__new__(overlay_module.MidnightSignalOverlay)
    overlay._draw_microphone = lambda *_args: None
    overlay.waveform_provider = lambda: pytest.fail("Paused capture must not show a live waveform")
    canvas = FakeCanvas()
    view = dictation_presentation({"mode": "live", "active": True, "processing": True, **paused})
    overlay._draw_dictation_session(canvas, 548, 88, SimpleNamespace(elapsed_label="00:12"), view)
    tags = [tag for _kind, _args, kwargs in canvas.items for tag in kwargs.get("tags", ())]
    assert "dictation-waveform-frozen" in tags
    assert "dictation-listening-indicator" not in tags
