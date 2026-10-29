"""System integration with fake clipboard and cancellation, never OS input."""
import threading
from types import SimpleNamespace

import pytest

from utils.dictation_runtime import DictationSession

pytestmark = pytest.mark.core_headless


def test_copy_pending_only_acknowledges_the_clipboard_snapshot(monkeypatch):
    from utils import system
    session = DictationSession("live", capture=lambda *a: None,
                               transcribe=lambda *a: None, deliver=lambda *a: None,
                               target=lambda: 1)
    session._hold_result("copied phrase")
    monkeypatch.setattr(system, "_dictation_controller", SimpleNamespace(session=session))
    copied = []

    def copy(text):
        copied.append(text)
        session._hold_result("arrived during copy")
        return True

    monkeypatch.setattr(system, "copy_to_clipboard", copy)
    assert system.copy_pending_dictation()
    assert copied == ["copied phrase"]
    assert session.pending_text() == "arrived during copy"


def test_copy_failure_preserves_held_text(monkeypatch):
    from utils import system
    session = DictationSession("live", capture=lambda *a: None,
                               transcribe=lambda *a: None, deliver=lambda *a: None,
                               target=lambda: 1)
    session._hold_result("keep me")
    monkeypatch.setattr(system, "_dictation_controller", SimpleNamespace(session=session))
    monkeypatch.setattr(system, "copy_to_clipboard", lambda text: False)
    assert not system.copy_pending_dictation()
    assert session.pending_text() == "keep me"


def test_shutdown_wait_does_not_forget_continuous_owner(monkeypatch):
    from utils import system
    done = threading.Event()
    monkeypatch.setattr(system, "_dictation_controller", SimpleNamespace(session=SimpleNamespace(done=done)))
    monkeypatch.setattr(system, "cancel_active_transcription", lambda: True)
    monkeypatch.setattr(system, "_clear_waveform_buffers", lambda: None)
    for name in ("recording_thread", "transcription_thread", "recording_file_path"):
        monkeypatch.setattr(system, name, None)
    assert not system.cancel_and_wait_for_active_transcription(0)
    done.set()
    assert system.cancel_and_wait_for_active_transcription(0)


def test_listener_restart_refuses_unreleased_previous_controller(monkeypatch):
    from utils import system
    monkeypatch.setattr(system, "listener", None)
    previous = SimpleNamespace(shutdown=lambda **kwargs: False)
    monkeypatch.setattr(system, "_dictation_controller", previous)
    messages = []
    monkeypatch.setattr(system, "notify", messages.append)
    system.start_client_listener()
    assert system._dictation_controller is previous
    assert system.listener is None
    assert messages
