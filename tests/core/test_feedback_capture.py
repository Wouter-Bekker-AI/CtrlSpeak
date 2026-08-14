from __future__ import annotations

from dataclasses import dataclass
import sys

import pytest

from utils.feedback_capture import (
    ActiveFieldSnapshotProvider,
    FeedbackCaptureCoordinator,
    FieldSnapshot,
    KeyEvent,
)
from utils.transcription_backend import FeedbackTarget, TranscriptionResult
from utils import system
from utils import winio


pytestmark = pytest.mark.core_headless


@dataclass
class MutableSnapshots:
    text: str | None

    def snapshot(self) -> FieldSnapshot | None:
        return FieldSnapshot(self.text) if self.text is not None else None


def test_changed_active_field_snapshot_on_enter_reports_only_the_pending_injection() -> None:
    snapshots = MutableSnapshots(None)
    submitted: list[tuple[str, str, str, dict[str, object], FeedbackTarget]] = []
    coordinator = FeedbackCaptureCoordinator(
        snapshot_provider=snapshots,
        submit_feedback=lambda tx_id, text, method, metadata, target: submitted.append(
            (tx_id, text, method, metadata, target)
        ),
        executor=lambda callback: callback(),
    )
    result = TranscriptionResult(
        text="injected words",
        transcription_id="tx-123",
        raw_text="raw words",
        corrected_text="injected words",
        metadata={"language": "en"},
        feedback_target=FeedbackTarget("api", "https://whisper.example.test", "secret"),
    )
    coordinator.track_injection(result, capture_method="active_field_on_enter")
    snapshots.text = "user-confirmed edited words"

    outcome = coordinator.handle_key_event(KeyEvent("enter", "press"))

    assert outcome.feedback_scheduled is True
    assert outcome.suppress_event is False
    assert outcome.replay_event is False
    assert submitted == [(
        "tx-123",
        "user-confirmed edited words",
        "active_field_on_enter",
        {"original_corrected_text": "injected words", "raw_text": "raw words", "language": "en"},
        FeedbackTarget("api", "https://whisper.example.test", "secret"),
    )]
    assert coordinator.pending is None


def test_unchanged_or_unrelated_events_never_report_and_enter_consumes_pending_send() -> None:
    snapshots = MutableSnapshots("before")
    submitted: list[str] = []
    coordinator = FeedbackCaptureCoordinator(
        snapshot_provider=snapshots,
        submit_feedback=lambda _tx, text, _method, _metadata, _target: submitted.append(text),
        executor=lambda callback: callback(),
    )
    result = TranscriptionResult(
        text="same",
        transcription_id="tx-1",
        raw_text="raw",
        feedback_target=FeedbackTarget("api", "http://127.0.0.1:8765"),
    )
    coordinator.track_injection(result, capture_method="active_field_on_enter")

    snapshots.text = "edited"
    assert coordinator.handle_key_event(KeyEvent("a", "press")).feedback_scheduled is False
    assert coordinator.pending is not None
    snapshots.text = "same"
    outcome = coordinator.handle_key_event(KeyEvent("enter", "press"))

    assert outcome.feedback_scheduled is True
    assert outcome.suppress_event is False
    assert submitted == []
    assert coordinator.pending is None


def test_enter_captures_field_before_scheduling_background_submission() -> None:
    snapshots = MutableSnapshots("edited")
    snapshot_calls = 0
    original_snapshot = snapshots.snapshot

    def counted_snapshot() -> FieldSnapshot | None:
        nonlocal snapshot_calls
        snapshot_calls += 1
        return original_snapshot()

    snapshots.snapshot = counted_snapshot  # type: ignore[method-assign]
    callbacks: list[object] = []
    submitted: list[str] = []
    coordinator = FeedbackCaptureCoordinator(
        snapshot_provider=snapshots,
        submit_feedback=lambda _tx, text, _method, _metadata, _target: submitted.append(text),
        executor=callbacks.append,
    )
    coordinator.track_injection(
        TranscriptionResult(
            text="injected",
            transcription_id="tx-1",
            feedback_target=FeedbackTarget("api", "http://127.0.0.1:8765"),
        ),
        capture_method="active_field_on_enter",
    )
    assert snapshot_calls == 0

    outcome = coordinator.handle_key_event(KeyEvent("enter", "press"))

    assert outcome.feedback_scheduled is True
    assert outcome.suppress_event is False
    assert outcome.replay_event is False
    assert snapshot_calls == 1
    assert len(callbacks) == 1
    callback = callbacks[0]
    assert callable(callback)
    callback()
    assert snapshot_calls == 1
    assert submitted == ["edited"]


def test_modified_enter_is_ignored_then_bare_enter_is_observed_once() -> None:
    snapshots = MutableSnapshots("edited")
    submitted: list[str] = []
    coordinator = FeedbackCaptureCoordinator(
        snapshot_provider=snapshots,
        submit_feedback=lambda _tx, text, _method, _metadata, _target: submitted.append(text),
        executor=lambda callback: callback(),
    )
    coordinator.track_injection(
        TranscriptionResult(
            text="injected",
            transcription_id="tx-1",
            feedback_target=FeedbackTarget("api", "http://127.0.0.1:8765"),
        ),
        capture_method="active_field_on_enter",
    )

    modified = coordinator.handle_key_event(
        KeyEvent("enter", "press", frozenset({"shift"}))
    )
    assert modified.feedback_scheduled is False
    assert coordinator.pending is not None

    bare = coordinator.handle_key_event(KeyEvent("enter", "press"))
    assert bare.feedback_scheduled is True
    assert bare.suppress_event is False
    assert bare.replay_event is False
    assert submitted == ["edited"]


def test_capture_can_be_disabled_and_requires_a_feedback_target() -> None:
    snapshots = MutableSnapshots("before")
    coordinator = FeedbackCaptureCoordinator(
        snapshot_provider=snapshots,
        submit_feedback=lambda *_args: (_ for _ in ()).throw(AssertionError("must not submit")),
        executor=lambda callback: callback(),
    )

    coordinator.track_injection(
        TranscriptionResult(text="bundled text", transcription_id=None),
        capture_method="active_field_on_enter",
    )
    assert coordinator.pending is None
    coordinator.track_injection(
        TranscriptionResult(
            text="api text",
            transcription_id="tx-2",
            feedback_target=FeedbackTarget("api", "http://127.0.0.1:8765"),
        ),
        capture_method="disabled",
    )
    assert coordinator.pending is None


def test_bundled_result_uses_its_local_feedback_target() -> None:
    snapshots = MutableSnapshots("before")
    targets: list[FeedbackTarget] = []
    coordinator = FeedbackCaptureCoordinator(
        snapshot_provider=snapshots,
        submit_feedback=lambda _tx, _text, _method, _metadata, target: targets.append(target),
        executor=lambda callback: callback(),
    )
    coordinator.track_injection(
        TranscriptionResult(
            text="local result",
            transcription_id="local-tx",
            raw_text="local raw",
            feedback_target=FeedbackTarget("bundled"),
        ),
        capture_method="active_field_on_enter",
    )
    snapshots.text = "approved local edit"

    outcome = coordinator.handle_key_event(KeyEvent("enter", "press"))

    assert outcome.feedback_scheduled is True
    assert targets == [FeedbackTarget("bundled")]


def test_expired_pending_injection_is_not_reported() -> None:
    now = [100.0]
    snapshots = MutableSnapshots("before")
    submitted: list[str] = []
    coordinator = FeedbackCaptureCoordinator(
        snapshot_provider=snapshots,
        submit_feedback=lambda _tx, text, _method, _metadata, _target: submitted.append(text),
        executor=lambda callback: callback(),
        now=lambda: now[0],
        pending_ttl_seconds=60,
    )
    coordinator.track_injection(
        TranscriptionResult(
            text="api text",
            transcription_id="tx-2",
            feedback_target=FeedbackTarget("api", "http://127.0.0.1:8765"),
        ),
        capture_method="active_field_on_enter",
    )
    snapshots.text = "edited"
    now[0] = 161.0

    assert coordinator.handle_key_event(KeyEvent("enter", "press")).feedback_scheduled is False
    assert submitted == []


def test_active_field_snapshot_provider_returns_the_capture_result() -> None:
    provider = ActiveFieldSnapshotProvider(lambda: "complete edited field")

    assert provider.snapshot() == FieldSnapshot("complete edited field")


@pytest.mark.skipif(not sys.platform.startswith("win"), reason="Win32 adapter test")
def test_windows_active_field_snapshot_selects_copies_and_restores_clipboard(monkeypatch) -> None:
    from utils import windows_input

    clipboard = {"text": "keep me"}
    hotkeys: list[tuple[str, ...]] = []
    restored: list[str | None] = []
    monkeypatch.setattr(windows_input, "get_clipboard_text", lambda: clipboard["text"])
    monkeypatch.setattr(
        windows_input,
        "set_clipboard_text",
        lambda text: clipboard.update(text=text) is None,
    )
    monkeypatch.setattr(windows_input, "clipboard_contains_non_text_data", lambda: False)

    def send_hotkey(*keys: str) -> None:
        hotkeys.append(tuple(keys))
        if keys == ("ctrl", "c"):
            clipboard["text"] = "complete edited field"

    monkeypatch.setattr(
        windows_input.pyautogui,
        "hotkey",
        send_hotkey,
        raising=False,
    )
    monkeypatch.setattr(windows_input, "restore_clipboard_text", restored.append)

    captured = windows_input.snapshot_active_text_field(copy_wait_seconds=0)

    assert captured == "complete edited field"
    assert hotkeys == [("ctrl", "a"), ("ctrl", "c")]
    assert restored == ["keep me"]


@pytest.mark.skipif(not sys.platform.startswith("win"), reason="Win32 adapter test")
def test_windows_active_field_snapshot_rejects_a_failed_copy(monkeypatch) -> None:
    from utils import windows_input

    clipboard = {"text": "unrelated old clipboard"}
    restored: list[str | None] = []
    monkeypatch.setattr(windows_input, "get_clipboard_text", lambda: clipboard["text"])
    monkeypatch.setattr(
        windows_input,
        "set_clipboard_text",
        lambda text: clipboard.update(text=text) is None,
    )
    monkeypatch.setattr(windows_input, "clipboard_contains_non_text_data", lambda: False)
    monkeypatch.setattr(windows_input.pyautogui, "hotkey", lambda *_keys: None, raising=False)
    monkeypatch.setattr(windows_input, "restore_clipboard_text", restored.append)

    assert windows_input.snapshot_active_text_field(copy_wait_seconds=0) is None
    assert restored == ["unrelated old clipboard"]


def test_system_hotkey_hooks_only_observe_events_and_track_api_result(monkeypatch) -> None:
    observed: list[KeyEvent] = []
    tracked: list[tuple[TranscriptionResult, str]] = []

    class FakeCoordinator:
        def handle_key_event(self, event: KeyEvent):
            observed.append(event)

        def track_injection(self, result: TranscriptionResult, *, capture_method: str):
            tracked.append((result, capture_method))

    monkeypatch.setattr(system, "_feedback_coordinator", FakeCoordinator())
    with system.settings_lock:
        system.settings["feedback_capture_method"] = "active_field_on_enter"
    result = TranscriptionResult(text="injected", transcription_id="tx-9")

    assert system.observe_feedback_key_event("enter", "press") is None
    system.track_feedback_injection(result)

    assert observed == [KeyEvent("enter", "press")]
    assert tracked == [(result, "active_field_on_enter")]


def test_actual_injection_tracks_feedback_only_after_success(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    result = TranscriptionResult(text="insert this", transcription_id="tx-10")
    monkeypatch.setattr(system, "_last_transcript", None)
    monkeypatch.setattr(
        system,
        "insert_text_into_focus",
        lambda text: calls.append(("insert", text)),
    )
    monkeypatch.setattr(
        system,
        "track_feedback_injection",
        lambda tracked: calls.append(("track", tracked)),
    )

    system.inject_transcription_result(result)

    assert calls == [("insert", "insert this"), ("track", result)]
    assert system.get_last_transcript() == "insert this"


def test_failed_injection_never_tracks_feedback(monkeypatch) -> None:
    tracked: list[TranscriptionResult] = []
    result = TranscriptionResult(text="insert this", transcription_id="tx-11")
    monkeypatch.setattr(system, "_last_transcript", None)

    def fail_insertion(_text: str) -> None:
        raise RuntimeError("insertion failed")

    monkeypatch.setattr(system, "insert_text_into_focus", fail_insertion)
    monkeypatch.setattr(system, "track_feedback_injection", tracked.append)

    with pytest.raises(RuntimeError, match="insertion failed"):
        system.inject_transcription_result(result)

    assert tracked == []
    assert system.get_last_transcript() == "insert this"


def test_copy_last_transcript_is_memory_only_and_reports_outcome(monkeypatch) -> None:
    copied: list[str] = []
    notifications: list[tuple[str, str]] = []
    tray_refreshes: list[bool] = []
    tray_icon = type("TrayIcon", (), {"update_menu": lambda self: tray_refreshes.append(True)})()
    monkeypatch.setattr(system, "_last_transcript", None)
    monkeypatch.setattr(system, "_tray_icon", tray_icon)
    monkeypatch.setattr(system, "set_clipboard_text", lambda text: copied.append(text) is None)
    monkeypatch.setattr(
        system,
        "notify",
        lambda message, *, title="CtrlSpeak": notifications.append((title, message)),
    )

    assert system.has_last_transcript() is False
    assert system.copy_last_transcript_from_tray() is False
    assert copied == []

    assert system.remember_last_transcript("Recovered dictated text") is True
    assert system.has_last_transcript() is True
    assert tray_refreshes == [True]
    assert system.copy_last_transcript_from_tray() is True
    assert copied == ["Recovered dictated text"]
    assert notifications == [
        ("CtrlSpeak", "No successful transcription is available yet."),
        ("CtrlSpeak", "Last transcript copied to the clipboard."),
    ]


def test_empty_transcript_never_replaces_last_recoverable_text(monkeypatch) -> None:
    monkeypatch.setattr(system, "_last_transcript", "keep this")

    assert system.remember_last_transcript("  \n") is False
    assert system.get_last_transcript() == "keep this"


def test_system_submits_bundled_feedback_to_the_local_library(monkeypatch) -> None:
    from utils import local_corrections

    approvals: list[tuple[str, str, str, dict[str, object]]] = []

    class FakeLibrary:
        def approve_exact_override(
            self,
            transcription_id: str,
            *,
            confirmed_text: str,
            capture_method: str,
            client_metadata: dict[str, object],
        ) -> object:
            approvals.append(
                (transcription_id, confirmed_text, capture_method, client_metadata)
            )
            return object()

    monkeypatch.setattr(
        local_corrections,
        "get_local_correction_library",
        lambda: FakeLibrary(),
    )
    with system.settings_lock:
        system.settings["transcription_backend"] = "api"

    system._submit_confirmed_feedback(
        "local-tx",
        "approved local text",
        "active_field_on_enter",
        {"raw_text": "local raw"},
        FeedbackTarget("bundled"),
    )

    assert approvals == [(
        "local-tx",
        "approved local text",
        "active_field_on_enter",
        {"client": "CtrlSpeak", "version": system.APP_VERSION, "raw_text": "local raw"},
    )]


def test_system_submits_api_feedback_to_the_original_endpoint_after_settings_change(
    monkeypatch,
) -> None:
    from utils import transcription_backend

    calls: list[tuple[object, str, str, str, dict[str, object]]] = []

    class FakeApiClient:
        def __init__(self, config) -> None:
            self.config = config

        def submit_feedback(
            self,
            transcription_id: str,
            *,
            final_text: str,
            capture_method: str,
            client_metadata: dict[str, object],
        ) -> None:
            calls.append(
                (
                    self.config,
                    transcription_id,
                    final_text,
                    capture_method,
                    client_metadata,
                )
            )

    monkeypatch.setattr(transcription_backend, "ApiTranscriptionClient", FakeApiClient)
    with system.settings_lock:
        system.settings["transcription_backend"] = "bundled"
        system.settings["api_url"] = "http://changed.invalid"

    system._submit_confirmed_feedback(
        "remote-tx",
        "approved remote text",
        "active_field_on_enter",
        {"raw_text": "remote raw"},
        FeedbackTarget("api", "https://original.example.test/base", "original-token"),
    )

    config, transcription_id, final_text, capture_method, metadata = calls[0]
    assert config.backend == "api"
    assert config.api_url == "https://original.example.test/base"
    assert config.api_token == "original-token"
    assert transcription_id == "remote-tx"
    assert final_text == "approved remote text"
    assert capture_method == "active_field_on_enter"
    assert metadata == {
        "client": "CtrlSpeak",
        "version": system.APP_VERSION,
        "raw_text": "remote raw",
    }
