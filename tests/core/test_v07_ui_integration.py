from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
import time
import types

import pytest

from utils.audio_cues import CueKind
from utils.transcription_backend import TranscriptionResult
from utils.ui_state import TranscriptionUiSession, UiPhase


pytestmark = pytest.mark.core_headless


class FakeThread:
    def __init__(self, alive: bool = True) -> None:
        self.alive = alive
        self.join_calls: list[float | None] = []

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)


def _processing_session(system) -> TranscriptionUiSession:
    session = TranscriptionUiSession()
    session.begin_recording()
    session.begin_processing()
    system.transcription_ui_session = session
    system.transcription_cancel_event.clear()
    return session


def _install_fake_models(monkeypatch, transcriber) -> None:
    fake_models = types.ModuleType("utils.models")
    fake_models.transcribe_audio_result = transcriber
    monkeypatch.setitem(sys.modules, "utils.models", fake_models)


def _isolate_worker_side_effects(monkeypatch, system) -> dict[str, list]:
    observed: dict[str, list] = {
        "injected": [],
        "cues": [],
        "hide_delays": [],
        "notifications": [],
        "tray_refreshes": [],
        "management_refreshes": [],
    }
    monkeypatch.setattr(
        system, "inject_transcription_result", observed["injected"].append
    )
    monkeypatch.setattr(
        system,
        "play_ui_cue",
        lambda kind, **_kwargs: observed["cues"].append(kind) or True,
    )
    monkeypatch.setattr(
        system, "_set_terminal_overlay_hide", observed["hide_delays"].append
    )
    monkeypatch.setattr(
        system,
        "notify_error",
        lambda context, detail: observed["notifications"].append((context, detail)),
    )
    monkeypatch.setattr(
        system,
        "_refresh_tray_menu",
        lambda: observed["tray_refreshes"].append(True),
    )
    monkeypatch.setattr(
        system,
        "schedule_management_refresh",
        lambda *_args, **_kwargs: observed["management_refreshes"].append(True),
    )
    return observed


def _usable_recording(path: Path) -> Path:
    path.write_bytes(b"RIFF" + (b"\0" * 96))
    return path


def test_processing_feedback_is_one_finite_cue_and_never_starts_loop_thread(
    monkeypatch,
) -> None:
    from utils import system

    played: list[tuple[CueKind, bool]] = []
    monkeypatch.setattr(
        system,
        "play_ui_cue",
        lambda kind, *, background=True: played.append((kind, background)) or True,
    )
    system.processing_sound_thread = None
    system.processing_sound_stop_event.clear()

    system.start_processing_feedback()

    assert played == [(CueKind.PROCESSING_STARTED, True)]
    assert system.processing_sound_thread is None
    system.stop_processing_feedback()
    assert system.processing_sound_stop_event.is_set()
    assert system.processing_sound_thread is None


def test_notification_error_logs_and_notifies_without_touching_clipboard(
    monkeypatch,
) -> None:
    from utils import system

    logs: list[tuple[str, str]] = []
    notices: list[tuple[str, str]] = []
    monkeypatch.setattr(system, "write_error_log", lambda context, detail: logs.append((context, detail)))
    monkeypatch.setattr(
        system,
        "notify",
        lambda message, title="CtrlSpeak": notices.append((title, message)),
    )
    monkeypatch.setattr(
        system,
        "copy_to_clipboard",
        lambda _text: pytest.fail("notify_error must never overwrite the clipboard"),
    )

    system.notify_error("Remote transcription failed", "secret technical detail")

    assert logs == [("Remote transcription failed", "secret technical detail")]
    assert notices == [(
        "CtrlSpeak",
        "Remote transcription failed. Open CtrlSpeak or the log folder for details.",
    )]
    assert "secret technical detail" not in notices[0][1]


def test_busy_state_covers_capture_worker_and_ui_lifecycle(monkeypatch) -> None:
    from utils import system

    system.recording = False
    system.recording_thread = None
    system.transcription_thread = None
    system.transcription_ui_session = TranscriptionUiSession()
    assert system.is_transcription_busy() is False

    system.recording = True
    assert system.is_transcription_busy() is True
    system.recording = False

    system.recording_thread = FakeThread()
    assert system.is_transcription_busy() is True
    system.recording_thread = None

    system.transcription_thread = FakeThread()
    assert system.is_transcription_busy() is True
    system.transcription_thread = None

    system.transcription_ui_session.begin_recording()
    assert system.is_transcription_busy() is True
    system.transcription_ui_session.begin_processing()
    assert system.is_transcription_busy() is True

    events = _isolate_worker_side_effects(monkeypatch, system)
    assert system.cancel_active_transcription() is True
    assert system.transcription_cancel_event.is_set()
    assert system.transcription_ui_session.phase is UiPhase.CANCELLED
    assert events["cues"] == [CueKind.CANCELLED]
    assert events["hide_delays"] == [1200]
    assert system.is_transcription_busy() is False
    assert system.cancel_active_transcription() is False


def test_background_transcription_success_inserts_once_and_cleans_temp_file(
    monkeypatch, tmp_path
) -> None:
    from utils import system

    session = _processing_session(system)
    result = TranscriptionResult(
        text="insert exactly once",
        transcription_id="tx-v07-success",
        raw_text="insert exactly once",
        corrected_text="insert exactly once",
        metadata={
            "provider_used": "ubuntu-gpu-large-v3-turbo",
            "routing_duration_ms": 210.0,
            "attempts": [
                {
                    "provider": "ubuntu-gpu-large-v3-turbo",
                    "status": "succeeded",
                    "duration_ms": 205.0,
                }
            ],
        },
    )
    _install_fake_models(monkeypatch, lambda *_args, **_kwargs: result)
    observed = _isolate_worker_side_effects(monkeypatch, system)
    recording = _usable_recording(tmp_path / "success.wav")
    system.recording_file_path = recording
    system.transcription_thread = FakeThread()

    system._transcribe_recording_worker(recording, system.time.monotonic() - 0.25)

    snapshot = session.snapshot()
    assert snapshot.phase is UiPhase.SUCCESS
    assert snapshot.provider is not None
    assert snapshot.provider.display_name == "Ubuntu GPU"
    assert snapshot.provider.latency_ms == 205.0
    assert observed["injected"] == [result]
    assert observed["cues"] == [CueKind.SUCCESS]
    assert observed["hide_delays"] == [1600]
    assert observed["notifications"] == []
    assert not recording.exists()
    assert system.recording_file_path is None
    assert system.transcription_thread is None
    assert observed["tray_refreshes"] == [True]
    assert observed["management_refreshes"] == [True]


def test_background_transcription_error_is_terminal_and_cleans_temp_file(
    monkeypatch, tmp_path
) -> None:
    from utils import system

    session = _processing_session(system)

    def fail_transcription(*_args, **_kwargs):
        raise RuntimeError("gateway connection unavailable")

    _install_fake_models(monkeypatch, fail_transcription)
    observed = _isolate_worker_side_effects(monkeypatch, system)
    recording = _usable_recording(tmp_path / "error.wav")
    system.recording_file_path = recording
    system.transcription_thread = FakeThread()

    system._transcribe_recording_worker(recording, system.time.monotonic() - 0.1)

    snapshot = session.snapshot()
    assert snapshot.phase is UiPhase.ERROR
    assert snapshot.error_category == "providers_exhausted"
    assert observed["injected"] == []
    assert observed["cues"] == [CueKind.ERROR]
    assert observed["hide_delays"] == [3800]
    assert observed["notifications"]
    assert observed["notifications"][0][0] == "Transcription failed"
    assert not recording.exists()
    assert system.recording_file_path is None
    assert system.transcription_thread is None


def test_cancellation_during_background_request_suppresses_text_insertion_and_cleans_temp(
    monkeypatch, tmp_path
) -> None:
    from utils import system

    session = _processing_session(system)
    result = TranscriptionResult(
        text="must never be inserted",
        transcription_id="tx-v07-cancelled",
        metadata={"provider_used": "openai-gpt-transcribe"},
    )

    def finish_after_cancel(*_args, **_kwargs):
        system.transcription_cancel_event.set()
        return result

    _install_fake_models(monkeypatch, finish_after_cancel)
    observed = _isolate_worker_side_effects(monkeypatch, system)
    recording = _usable_recording(tmp_path / "cancelled.wav")
    system.recording_file_path = recording
    system.transcription_thread = FakeThread()

    system._transcribe_recording_worker(recording, system.time.monotonic() - 0.1)

    assert session.snapshot().phase is UiPhase.CANCELLED
    assert observed["injected"] == []
    assert observed["cues"] == [CueKind.CANCELLED]
    assert observed["hide_delays"] == [1200]
    assert observed["notifications"] == []
    assert not recording.exists()
    assert system.recording_file_path is None
    assert system.transcription_thread is None


def test_cancelled_live_recorder_stays_busy_and_blocks_a_new_generation(
    monkeypatch, tmp_path
) -> None:
    from utils import system

    observed = _isolate_worker_side_effects(monkeypatch, system)
    session = TranscriptionUiSession()
    session.begin_recording()
    cancel_event = threading.Event()
    stop_event = threading.Event()
    failure_event = threading.Event()
    allow_exit = threading.Event()
    recorder_started = threading.Event()
    recording = _usable_recording(tmp_path / "live-cancel.wav")

    def lingering_recorder() -> None:
        recorder_started.set()
        assert stop_event.wait(1.0)
        assert allow_exit.wait(2.0)

    recorder = threading.Thread(target=lingering_recorder, daemon=True)
    recorder.start()
    assert recorder_started.wait(1.0)

    monkeypatch.setattr(system, "transcription_ui_session", session)
    monkeypatch.setattr(system, "transcription_cancel_event", cancel_event)
    monkeypatch.setattr(system, "_recording_stop_event", stop_event)
    monkeypatch.setattr(system, "_recording_failed_event", failure_event)
    monkeypatch.setattr(system, "_transcription_generation", 40)
    monkeypatch.setattr(system, "_active_transcription_generation", 40)
    monkeypatch.setattr(system, "recording", True)
    monkeypatch.setattr(system, "recording_thread", recorder)
    monkeypatch.setattr(system, "transcription_thread", None)
    monkeypatch.setattr(system, "recording_file_path", recording)
    monkeypatch.setattr(system, "client_enabled", True)
    monkeypatch.setattr(system, "is_right_control", lambda _key: True)
    monkeypatch.setattr(system, "_observe_pynput_press", lambda _key: None)

    assert system.cancel_active_transcription() is True
    reaper = system.transcription_thread
    assert isinstance(reaper, threading.Thread)
    assert recorder.is_alive()
    assert system.recording_thread is recorder
    assert system.is_transcription_busy() is True

    # A repeated hotkey cannot clear/reuse this generation's cancellation Event.
    system.on_press(object())
    assert system._active_transcription_generation == 40
    assert system.transcription_cancel_event is cancel_event
    assert cancel_event.is_set()

    allow_exit.set()
    recorder.join(1.0)
    reaper.join(1.0)
    assert not recorder.is_alive()
    assert not reaper.is_alive()
    assert system.recording_thread is None
    assert system.transcription_thread is None
    assert system._active_transcription_generation is None
    assert system.is_transcription_busy() is False
    assert not recording.exists()
    assert observed["cues"] == [CueKind.CANCELLED]


def test_hotkey_cannot_start_recorder_after_listener_shutdown(
    monkeypatch, tmp_path
) -> None:
    from utils import system

    availability_entered = threading.Event()
    allow_availability_check_to_finish = threading.Event()
    recording_path = tmp_path / "must-not-be-created.wav"
    created_paths: list[Path] = []
    stop_waits: list[bool] = []

    def delayed_hotkey_check() -> bool:
        availability_entered.set()
        assert allow_availability_check_to_finish.wait(2.0)
        return True

    monkeypatch.setattr(system, "client_enabled", True)
    monkeypatch.setattr(system, "listener", None)
    monkeypatch.setattr(system, "recording", False)
    monkeypatch.setattr(system, "recording_thread", None)
    monkeypatch.setattr(system, "transcription_thread", None)
    monkeypatch.setattr(system, "recording_file_path", None)
    monkeypatch.setattr(system, "_transcription_generation", 70)
    monkeypatch.setattr(system, "_active_transcription_generation", None)
    monkeypatch.setattr(system, "transcription_ui_session", TranscriptionUiSession())
    monkeypatch.setattr(system, "is_right_control", lambda _key: True)
    monkeypatch.setattr(system, "_observe_pynput_press", lambda _key: None)
    monkeypatch.setattr(system, "_client_hotkey_available", delayed_hotkey_check)
    monkeypatch.setattr(
        system,
        "create_recording_file_path",
        lambda: created_paths.append(recording_path) or recording_path,
    )
    monkeypatch.setattr(
        system,
        "cancel_and_wait_for_active_transcription",
        lambda: stop_waits.append(True) or True,
    )
    monkeypatch.setattr(system, "schedule_management_refresh", lambda: None)

    callback = threading.Thread(target=system.on_press, args=(object(),), daemon=True)
    callback.start()
    assert availability_entered.wait(1.0)

    # Shutdown wins while the callback is outside both lifecycle locks.  The
    # callback must perform an authoritative, synchronized client-state check
    # before it is allowed to create or publish a new recording generation.
    system.stop_client_listener()
    assert system.client_enabled is False
    allow_availability_check_to_finish.set()
    callback.join(1.0)

    assert not callback.is_alive()
    assert stop_waits == [True]
    assert created_paths == []
    assert system._transcription_generation == 70
    assert system._active_transcription_generation is None
    assert system.recording is False
    assert system.recording_thread is None
    assert system.recording_file_path is None
    assert system.transcription_ui_session.phase is UiPhase.IDLE


def test_release_never_detaches_recorder_before_wav_flush(monkeypatch, tmp_path) -> None:
    from utils import system

    session = TranscriptionUiSession()
    session.begin_recording()
    cancel_event = threading.Event()
    stop_event = threading.Event()
    failure_event = threading.Event()
    allow_flush = threading.Event()
    recorder_started = threading.Event()
    recording = tmp_path / "delayed-flush.wav"
    result = TranscriptionResult(
        text="flushed safely",
        transcription_id="tx-delayed-flush",
        metadata={"provider_used": "ubuntu-gpu-large-v3-turbo"},
    )
    _install_fake_models(monkeypatch, lambda *_args, **_kwargs: result)
    observed = _isolate_worker_side_effects(monkeypatch, system)

    def delayed_flush() -> None:
        recorder_started.set()
        assert stop_event.wait(1.0)
        assert allow_flush.wait(2.0)
        _usable_recording(recording)

    recorder = threading.Thread(target=delayed_flush, daemon=True)
    recorder.start()
    assert recorder_started.wait(1.0)

    monkeypatch.setattr(system, "transcription_ui_session", session)
    monkeypatch.setattr(system, "transcription_cancel_event", cancel_event)
    monkeypatch.setattr(system, "_recording_stop_event", stop_event)
    monkeypatch.setattr(system, "_recording_failed_event", failure_event)
    monkeypatch.setattr(system, "_transcription_generation", 41)
    monkeypatch.setattr(system, "_active_transcription_generation", 41)
    monkeypatch.setattr(system, "recording", True)
    monkeypatch.setattr(system, "recording_thread", recorder)
    monkeypatch.setattr(system, "transcription_thread", None)
    monkeypatch.setattr(system, "recording_file_path", recording)
    monkeypatch.setattr(system, "is_right_control", lambda _key: True)
    monkeypatch.setattr(system, "_observe_pynput_release", lambda _key: None)

    system.on_release(object())
    coordinator = system.transcription_thread
    assert isinstance(coordinator, threading.Thread)
    assert recorder.is_alive()
    assert system.recording_thread is recorder
    assert system.is_transcription_busy() is True
    assert session.phase is UiPhase.PROCESSING

    allow_flush.set()
    recorder.join(1.0)
    coordinator.join(2.0)
    assert session.phase is UiPhase.SUCCESS
    assert observed["injected"] == [result]
    assert system.recording_thread is None
    assert system.transcription_thread is None
    assert system._active_transcription_generation is None
    assert not recording.exists()


def test_cancellation_and_text_insertion_have_one_atomic_commit(monkeypatch, tmp_path) -> None:
    from utils import system

    session = TranscriptionUiSession()
    session.begin_recording()
    session.begin_processing()
    cancel_event = threading.Event()
    recording = _usable_recording(tmp_path / "atomic-insert.wav")
    result = TranscriptionResult(
        text="committed text",
        transcription_id="tx-atomic",
        metadata={"provider_used": "openai-gpt-transcribe"},
    )
    _install_fake_models(monkeypatch, lambda *_args, **_kwargs: result)
    observed = _isolate_worker_side_effects(monkeypatch, system)
    injection_started = threading.Event()
    allow_injection = threading.Event()

    def blocking_injection(value) -> None:
        injection_started.set()
        assert allow_injection.wait(2.0)
        observed["injected"].append(value)

    monkeypatch.setattr(system, "inject_transcription_result", blocking_injection)
    monkeypatch.setattr(system, "transcription_ui_session", session)
    monkeypatch.setattr(system, "transcription_cancel_event", cancel_event)
    monkeypatch.setattr(system, "_recording_stop_event", threading.Event())
    monkeypatch.setattr(system, "_transcription_generation", 42)
    monkeypatch.setattr(system, "_active_transcription_generation", 42)
    monkeypatch.setattr(system, "recording", False)
    monkeypatch.setattr(system, "recording_thread", None)
    monkeypatch.setattr(system, "recording_file_path", recording)

    worker = threading.Thread(
        target=system._transcribe_recording_worker,
        args=(recording, time.monotonic()),
        kwargs={"generation": 42, "cancel_event": cancel_event},
        daemon=True,
    )
    monkeypatch.setattr(system, "transcription_thread", worker)
    worker.start()
    assert injection_started.wait(1.0)

    cancel_finished = threading.Event()
    cancel_result: list[bool] = []

    def cancel() -> None:
        cancel_result.append(system.cancel_active_transcription())
        cancel_finished.set()

    cancelling = threading.Thread(target=cancel, daemon=True)
    cancelling.start()
    assert not cancel_finished.wait(0.1)
    allow_injection.set()
    worker.join(2.0)
    cancelling.join(1.0)

    assert observed["injected"] == [result]
    assert session.phase is UiPhase.SUCCESS
    assert cancel_result == [False]
    assert not cancel_event.is_set()


def test_stale_generation_finalizer_cannot_clear_newer_session(monkeypatch, tmp_path) -> None:
    from utils import system

    old_path = _usable_recording(tmp_path / "old.wav")
    new_path = _usable_recording(tmp_path / "new.wav")
    current_recorder = FakeThread(alive=True)
    current_transcriber = FakeThread(alive=True)
    stale_owner = FakeThread(alive=False)
    monkeypatch.setattr(system, "_active_transcription_generation", 52)
    monkeypatch.setattr(system, "recording", True)
    monkeypatch.setattr(system, "recording_file_path", new_path)
    monkeypatch.setattr(system, "recording_thread", current_recorder)
    monkeypatch.setattr(system, "transcription_thread", current_transcriber)

    system._finalize_transcription_session(
        51,
        old_path,
        owner_thread=stale_owner,
        recorder=FakeThread(alive=False),
    )

    assert not old_path.exists()
    assert new_path.exists()
    assert system._active_transcription_generation == 52
    assert system.recording is True
    assert system.recording_file_path == new_path
    assert system.recording_thread is current_recorder
    assert system.transcription_thread is current_transcriber


def test_shutdown_wait_is_bounded_and_reconciles_only_stopped_workers(
    monkeypatch, tmp_path
) -> None:
    from utils import system

    session = TranscriptionUiSession()
    session.begin_recording()
    session.begin_processing()
    cancel_event = threading.Event()
    allow_exit = threading.Event()
    worker_started = threading.Event()
    recording = _usable_recording(tmp_path / "shutdown.wav")

    def blocked_request() -> None:
        worker_started.set()
        assert cancel_event.wait(1.0)
        assert allow_exit.wait(2.0)

    worker = threading.Thread(target=blocked_request, daemon=True)
    worker.start()
    assert worker_started.wait(1.0)
    monkeypatch.setattr(system, "transcription_ui_session", session)
    monkeypatch.setattr(system, "transcription_cancel_event", cancel_event)
    monkeypatch.setattr(system, "_recording_stop_event", threading.Event())
    monkeypatch.setattr(system, "_active_transcription_generation", 60)
    monkeypatch.setattr(system, "recording", False)
    monkeypatch.setattr(system, "recording_thread", None)
    monkeypatch.setattr(system, "transcription_thread", worker)
    monkeypatch.setattr(system, "recording_file_path", recording)
    _isolate_worker_side_effects(monkeypatch, system)

    started = time.monotonic()
    assert system.cancel_and_wait_for_active_transcription(0.05) is False
    assert time.monotonic() - started < 0.5
    assert worker.is_alive()
    assert system.transcription_thread is worker
    assert not recording.exists()

    allow_exit.set()
    worker.join(1.0)
    assert system.cancel_and_wait_for_active_transcription(0.2) is True
    assert system.transcription_thread is None
    assert system._active_transcription_generation is None
    assert not recording.exists()


def test_settings_schema_v4_migration_preserves_valid_ui_preferences(tmp_path) -> None:
    from utils import config_paths

    settings_file = config_paths.get_config_file_path()
    settings_file.write_text(
        json.dumps(
            {
                "settings_schema_version": 3,
                "mode": "client",
                "overlay_enabled": False,
                "reduced_motion": True,
                "audio_cues_enabled": False,
                "audio_cue_volume": 72,
                "future_ui_field": {"preserve": True},
            }
        ),
        encoding="utf-8",
    )

    loaded = config_paths.load_settings()

    assert loaded["settings_schema_version"] == 4
    assert loaded["overlay_enabled"] is False
    assert loaded["reduced_motion"] is True
    assert loaded["audio_cues_enabled"] is False
    assert loaded["audio_cue_volume"] == 72
    assert loaded["future_ui_field"] == {"preserve": True}
    assert len(list(settings_file.parent.glob("settings.pre-migration-v2.*.json"))) == 1
    persisted = json.loads(settings_file.read_text("utf-8"))
    assert persisted["settings_schema_version"] == 4
    assert persisted["audio_cue_volume"] == 72


def test_settings_schema_v4_salvages_invalid_ui_preferences_independently() -> None:
    from utils import config_paths

    settings_file = config_paths.get_config_file_path()
    settings_file.write_text(
        json.dumps(
            {
                "settings_schema_version": 4,
                "mode": "client_server",
                "overlay_enabled": "yes",
                "reduced_motion": 1,
                "audio_cues_enabled": None,
                "audio_cue_volume": 101,
            }
        ),
        encoding="utf-8",
    )

    loaded = config_paths.load_settings()

    assert loaded["mode"] == "client_server"
    assert loaded["overlay_enabled"] is True
    assert loaded["reduced_motion"] is False
    assert loaded["audio_cues_enabled"] is True
    assert loaded["audio_cue_volume"] == 30
    assert json.loads(settings_file.read_text("utf-8"))["audio_cue_volume"] == 30


def test_overlay_geometry_helpers_are_headless_and_monitor_safe(monkeypatch) -> None:
    from utils import midnight_overlay

    class FakeRoot:
        def winfo_screenwidth(self) -> int:
            return 2560

        def winfo_screenheight(self) -> int:
            return 1440

    class FakeCanvas:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[float, ...], dict[str, object]]] = []

        def create_polygon(self, points, **kwargs) -> int:
            self.calls.append((tuple(points), kwargs))
            return 17

    monkeypatch.setattr(midnight_overlay.sys, "platform", "linux")
    bounds = midnight_overlay.active_monitor_bounds(FakeRoot())
    assert bounds == midnight_overlay.MonitorBounds(0, 0, 2560, 1440)
    assert bounds.width == 2560
    assert bounds.height == 1440
    assert midnight_overlay.MonitorBounds(10, 20, 5, 15).width == 1
    assert midnight_overlay.MonitorBounds(10, 20, 5, 15).height == 1

    canvas = FakeCanvas()
    result = midnight_overlay._rounded_rectangle(
        canvas, 0, 0, 10, 6, 100, fill="#123456"
    )
    assert result == 17
    points, options = canvas.calls[0]
    assert min(points[0::2]) >= 0
    assert max(points[0::2]) <= 10
    assert min(points[1::2]) >= 0
    assert max(points[1::2]) <= 6
    assert options == {"smooth": True, "splinesteps": 24, "fill": "#123456"}


def test_overlay_reuse_revokes_pending_terminal_dismissal() -> None:
    from utils.midnight_overlay import MidnightSignalOverlay

    class FakeWindow:
        def __init__(self) -> None:
            self.jobs: dict[str, object] = {}
            self.cancelled: list[str] = []

        def winfo_exists(self) -> bool:
            return True

        def after(self, _delay: int, callback) -> str:
            job = f"job-{len(self.jobs) + 1}"
            self.jobs[job] = callback
            return job

        def after_cancel(self, job: str) -> None:
            self.cancelled.append(job)
            self.jobs.pop(job, None)

    overlay = object.__new__(MidnightSignalOverlay)
    overlay.window = FakeWindow()
    overlay.canvas = None
    overlay._job = None
    overlay._close_job = None
    overlay._closing = False
    overlay._started = 0.0
    overlay.waveform_provider = None

    overlay.close(delay_ms=1600)
    pending = overlay._close_job
    assert pending is not None
    assert pending in overlay.window.jobs

    replacement = lambda: ()
    overlay.set_waveform_provider(replacement)
    assert overlay.waveform_provider is replacement
    assert overlay._close_job is None
    assert overlay.window.cancelled == [pending]
    assert overlay.window.jobs == {}


def test_windows_toplevel_handle_resolution_walks_from_tk_child_to_wrapper() -> None:
    from utils.midnight_overlay import _resolve_windows_toplevel_hwnd

    class FakeWindow:
        @staticmethod
        def winfo_id() -> int:
            return 101

    assert _resolve_windows_toplevel_hwnd(
        FakeWindow(), get_root=lambda hwnd: 202 if hwnd == 101 else 0
    ) == 202
    # A failed GA_ROOT lookup safely keeps the Tk-reported handle.
    assert _resolve_windows_toplevel_hwnd(
        FakeWindow(), get_root=lambda _hwnd: 0
    ) == 101


@pytest.mark.parametrize(
    ("bounds", "scale", "expects_scroll"),
    [
        ((0, 0, 1920, 1080), 1.0, False),
        ((0, 0, 1920, 1080), 2.0, True),
        ((0, 0, 1366, 768), 1.5, True),
        ((-1920, 0, 0, 1080), 2.0, True),
    ],
)
def test_flyout_geometry_is_fully_inside_scaled_monitor_work_area(
    bounds, scale, expects_scroll
) -> None:
    from utils.midnight_overlay import MonitorBounds, flyout_geometry

    work = MonitorBounds(*bounds)
    x, y, width, height, scroll = flyout_geometry(work, scale)
    assert width > 0 and height > 0
    assert work.left <= x
    assert work.top <= y
    assert x + width <= work.right
    assert y + height <= work.bottom
    assert scroll is expects_scroll


def test_legacy_theme_helper_uses_windows_safe_braced_font_descriptors(
    monkeypatch,
) -> None:
    from utils import ui_theme

    class FakeRoot:
        def __init__(self) -> None:
            self.configured: dict[str, object] = {}
            self.options: list[tuple[str, object]] = []

        def configure(self, **kwargs) -> None:
            self.configured.update(kwargs)

        def option_add(self, pattern: str, value: object) -> None:
            self.options.append((pattern, value))

    class FakeTop:
        pass

    class FakeStyle:
        def __init__(self) -> None:
            self.theme: str | None = None
            self.configurations: list[tuple[str, dict[str, object]]] = []
            self.maps: list[tuple[str, dict[str, object]]] = []

        def theme_use(self, name: str) -> None:
            self.theme = name

        def configure(self, name: str, **kwargs) -> None:
            self.configurations.append((name, kwargs))

        def map(self, name: str, **kwargs) -> None:
            self.maps.append((name, kwargs))

    root = FakeRoot()
    style = FakeStyle()
    monkeypatch.setattr(ui_theme.tk, "Tk", FakeRoot)
    monkeypatch.setattr(ui_theme.tk, "Toplevel", FakeTop, raising=False)
    monkeypatch.setattr(ui_theme.ttk, "Style", lambda master: style)

    returned = ui_theme.apply_modern_theme(root)

    assert returned is style
    assert style.theme == "clam"
    assert root.configured["bg"] == ui_theme.BACKGROUND
    assert ("*Font", "{Segoe UI} 10") in root.options
    assert ("*Label.Font", "{Segoe UI} 10") in root.options
    configured_names = {name for name, _values in style.configurations}
    assert {"Modern.TFrame", "Accent.TButton", "Modern.TCombobox"} <= configured_names
    assert any(name == "Accent.TButton" for name, _values in style.maps)
