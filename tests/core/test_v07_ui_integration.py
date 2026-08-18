from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import threading
import time
import types
import wave

import pytest

from utils.audio_cues import CueKind, synthesize_cue
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


def _seed_waveform(system) -> None:
    """Populate the PCM ring without relying on the headless NumPy stub."""

    with system._waveform_lock:
        system._waveform_buffers.clear()
        system._waveform_buffers.append(object())
        system._waveform_samples = 1


def _waveform_is_empty(system) -> bool:
    with system._waveform_lock:
        return not system._waveform_buffers and system._waveform_samples == 0


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


def test_windows_cue_sink_uses_valid_memory_wav_without_pyaudio(monkeypatch) -> None:
    from utils import system

    played: list[tuple[bytes, int]] = []
    fake_winsound = types.SimpleNamespace(
        SND_MEMORY=0x0004,
        SND_NODEFAULT=0x0002,
        PlaySound=lambda payload, flags: played.append((payload, flags)),
    )
    monkeypatch.setitem(sys.modules, "winsound", fake_winsound)
    monkeypatch.setattr(system.sys, "platform", "win32")
    monkeypatch.setattr(
        system.pyaudio,
        "PyAudio",
        lambda: pytest.fail("Windows UI cues must never initialize PortAudio"),
    )
    cue = synthesize_cue(CueKind.RECORDING_STARTED)

    system._play_pcm_cue(cue)

    assert len(played) == 1
    payload, flags = played[0]
    assert flags == fake_winsound.SND_MEMORY | fake_winsound.SND_NODEFAULT
    with wave.open(io.BytesIO(payload), "rb") as handle:
        assert handle.getnchannels() == cue.channels
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == cue.sample_rate
        assert handle.readframes(handle.getnframes()) == cue.frames


def test_hotkey_capture_and_windows_cue_overlap_without_second_portaudio_session(
    monkeypatch, tmp_path
) -> None:
    from utils import system

    recorder_entered = threading.Event()
    cue_played = threading.Event()
    overlaps: list[bool] = []
    pyaudio_initializations: list[bool] = []
    recording_path = tmp_path / "hotkey-cue.wav"

    def fake_record_audio(_path: Path, stop_event: threading.Event) -> None:
        recorder_entered.set()
        assert stop_event.wait(2.0)

    def fake_play_sound(_payload: bytes, _flags: int) -> None:
        assert recorder_entered.wait(1.0)
        recorder = system.recording_thread
        overlaps.append(bool(recorder is not None and recorder.is_alive()))
        cue_played.set()

    def forbidden_pyaudio():
        pyaudio_initializations.append(True)
        raise AssertionError("the Windows cue path reached PortAudio")

    fake_winsound = types.SimpleNamespace(
        SND_MEMORY=0x0004,
        SND_NODEFAULT=0x0002,
        PlaySound=fake_play_sound,
    )
    monkeypatch.setitem(sys.modules, "winsound", fake_winsound)
    monkeypatch.setattr(system.sys, "platform", "win32")
    monkeypatch.setattr(system.pyaudio, "PyAudio", forbidden_pyaudio)
    monkeypatch.setattr(system, "record_audio", fake_record_audio)
    monkeypatch.setattr(system, "create_recording_file_path", lambda: recording_path)
    monkeypatch.setattr(system, "is_right_control", lambda _key: True)
    monkeypatch.setattr(system, "_observe_pynput_press", lambda _key: None)
    monkeypatch.setattr(system, "_client_hotkey_available", lambda: True)
    monkeypatch.setattr(system, "_show_recording_overlay", lambda: None)
    monkeypatch.setattr(system, "_set_terminal_overlay_hide", lambda _delay: None)
    monkeypatch.setattr(system, "_refresh_tray_menu", lambda: None)
    monkeypatch.setattr(system, "schedule_management_refresh", lambda: None)
    monkeypatch.setitem(system.settings, "audio_cues_enabled", True)
    monkeypatch.setitem(system.settings, "audio_cue_volume", 30)
    monkeypatch.setattr(system, "client_enabled", True)
    monkeypatch.setattr(system, "recording", False)
    monkeypatch.setattr(system, "recording_thread", None)
    monkeypatch.setattr(system, "transcription_thread", None)
    monkeypatch.setattr(system, "recording_file_path", None)
    monkeypatch.setattr(system, "_active_transcription_generation", None)
    monkeypatch.setattr(system, "transcription_ui_session", TranscriptionUiSession())

    system.on_press(object())

    assert cue_played.wait(1.0)
    assert overlaps == [True]
    assert pyaudio_initializations == []
    assert system.cancel_active_transcription() is True
    reaper = system.transcription_thread
    if isinstance(reaper, threading.Thread):
        reaper.join(2.0)
        assert not reaper.is_alive()
    deadline = time.monotonic() + 2.0
    while system.is_transcription_busy() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert system.is_transcription_busy() is False


def test_guarded_capture_stays_exclusive_while_async_start_and_cancel_cues_play(
    monkeypatch, tmp_path
) -> None:
    from utils import system

    capture_read_entered = threading.Event()
    release_capture_read = threading.Event()
    start_cue_delivered = threading.Event()
    cancel_cue_delivered = threading.Event()
    lifecycle_guard = threading.Lock()
    lifecycle: list[str] = []
    cue_kinds: list[CueKind | None] = []
    background_errors: list[BaseException] = []
    active_instances = 0
    maximum_active_instances = 0
    initializations = 0
    terminations = 0

    class GuardedStream:
        closed = False

        def read(self, frame_count: int) -> bytes:
            with lifecycle_guard:
                lifecycle.append("capture-read")
            capture_read_entered.set()
            if not release_capture_read.wait(2.0):
                raise AssertionError("test did not release the guarded capture")
            return b"\0" * (frame_count * 2)

        def stop_stream(self) -> None:
            with lifecycle_guard:
                lifecycle.append("stream-stop")

        def close(self) -> None:
            self.closed = True
            with lifecycle_guard:
                lifecycle.append("stream-close")

    class GuardedPyAudio:
        def __init__(self) -> None:
            nonlocal active_instances, maximum_active_instances, initializations
            self.stream: GuardedStream | None = None
            with lifecycle_guard:
                initializations += 1
                active_instances += 1
                maximum_active_instances = max(maximum_active_instances, active_instances)
                lifecycle.append("pyaudio-initialize")
                if active_instances != 1:
                    background_errors.append(
                        AssertionError("overlapping PyAudio initialization")
                    )

        def open(self, **_kwargs) -> GuardedStream:
            self.stream = GuardedStream()
            with lifecycle_guard:
                lifecycle.append("stream-open")
            return self.stream

        @staticmethod
        def get_sample_size(_audio_format: int) -> int:
            return 2

        def terminate(self) -> None:
            nonlocal active_instances, terminations
            with lifecycle_guard:
                if self.stream is not None and not self.stream.closed:
                    background_errors.append(
                        AssertionError("PyAudio terminated before its stream closed")
                    )
                if active_instances != 1:
                    background_errors.append(
                        AssertionError("overlapping or duplicate PyAudio termination")
                    )
                active_instances -= 1
                terminations += 1
                lifecycle.append("pyaudio-terminate")

    fake_winsound = types.SimpleNamespace(
        SND_MEMORY=0x0004,
        SND_NODEFAULT=0x0002,
        PlaySound=lambda _payload, _flags: None,
    )
    original_cue_sink = system._play_pcm_cue

    def observing_cue_sink(cue) -> None:
        try:
            original_cue_sink(cue)
            with lifecycle_guard:
                cue_kinds.append(cue.kind)
                lifecycle.append(f"cue-{cue.kind.value}")
            if cue.kind is CueKind.RECORDING_STARTED:
                start_cue_delivered.set()
            elif cue.kind is CueKind.CANCELLED:
                cancel_cue_delivered.set()
        except BaseException as exc:
            background_errors.append(exc)
            raise

    recording_path = tmp_path / "guarded-overlap.wav"
    monkeypatch.setitem(sys.modules, "winsound", fake_winsound)
    monkeypatch.setattr(system.sys, "platform", "win32")
    monkeypatch.setattr(system.pyaudio, "PyAudio", GuardedPyAudio)
    monkeypatch.setattr(system, "_pyaudio_session_lock", threading.Lock())
    monkeypatch.setattr(system, "_pyaudio_priority_lock", threading.Lock())
    monkeypatch.setattr(system, "_pyaudio_capture_pending", threading.Event())
    monkeypatch.setattr(system, "_play_pcm_cue", observing_cue_sink)
    monkeypatch.setattr(system, "create_recording_file_path", lambda: recording_path)
    monkeypatch.setattr(system, "is_right_control", lambda _key: True)
    monkeypatch.setattr(system, "_observe_pynput_press", lambda _key: None)
    monkeypatch.setattr(system, "_client_hotkey_available", lambda: True)
    monkeypatch.setattr(system, "_show_recording_overlay", lambda: None)
    monkeypatch.setattr(system, "_set_terminal_overlay_hide", lambda _delay: None)
    monkeypatch.setattr(system, "_refresh_tray_menu", lambda: None)
    monkeypatch.setattr(system, "schedule_management_refresh", lambda: None)
    monkeypatch.setitem(system.settings, "input_device", None)
    monkeypatch.setitem(system.settings, "audio_cues_enabled", True)
    monkeypatch.setitem(system.settings, "audio_cue_volume", 30)
    monkeypatch.setattr(system, "client_enabled", True)
    monkeypatch.setattr(system, "recording", False)
    monkeypatch.setattr(system, "recording_thread", None)
    monkeypatch.setattr(system, "transcription_thread", None)
    monkeypatch.setattr(system, "recording_file_path", None)
    monkeypatch.setattr(system, "_active_transcription_generation", None)
    monkeypatch.setattr(system, "transcription_ui_session", TranscriptionUiSession())

    system.on_press(object())

    assert capture_read_entered.wait(1.0)
    assert start_cue_delivered.wait(1.0)
    assert system.cancel_active_transcription() is True
    assert cancel_cue_delivered.wait(1.0)
    with lifecycle_guard:
        assert active_instances == 1
        assert initializations == 1
        assert terminations == 0
        assert cue_kinds == [CueKind.RECORDING_STARTED, CueKind.CANCELLED]
        assert background_errors == []

    release_capture_read.set()
    coordinator = system.transcription_thread
    if isinstance(coordinator, threading.Thread):
        coordinator.join(2.0)
        assert not coordinator.is_alive()

    assert system.is_transcription_busy() is False
    assert initializations == 1
    assert terminations == 1
    assert active_instances == 0
    assert maximum_active_instances == 1
    assert background_errors == []
    assert lifecycle.index("stream-close") < lifecycle.index("pyaudio-terminate")


def test_cancel_lifecycle_completes_when_background_cue_thread_cannot_start(
    monkeypatch, tmp_path, caplog
) -> None:
    from utils import audio_cues, system

    class RefusedThread:
        def __init__(self, **_kwargs) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("thread creation refused")

    # Replace only audio_cues' module reference; the cancellation reaper still
    # uses the real threading module from utils.system.
    monkeypatch.setattr(
        audio_cues,
        "threading",
        types.SimpleNamespace(Thread=RefusedThread),
    )
    monkeypatch.setattr(system.sys, "platform", "win32")
    monkeypatch.setitem(system.settings, "audio_cues_enabled", True)
    monkeypatch.setitem(system.settings, "audio_cue_volume", 30)
    session = TranscriptionUiSession()
    session.begin_recording()
    monkeypatch.setattr(system, "transcription_ui_session", session)
    monkeypatch.setattr(system, "transcription_cancel_event", threading.Event())
    monkeypatch.setattr(system, "_recording_stop_event", threading.Event())
    monkeypatch.setattr(system, "recording", True)
    monkeypatch.setattr(system, "recording_thread", None)
    monkeypatch.setattr(system, "transcription_thread", None)
    monkeypatch.setattr(system, "recording_file_path", tmp_path / "never-created.wav")
    monkeypatch.setattr(system, "_active_transcription_generation", 499)
    monkeypatch.setattr(system, "_set_terminal_overlay_hide", lambda _delay: None)
    monkeypatch.setattr(system, "_refresh_tray_menu", lambda: None)
    monkeypatch.setattr(system, "schedule_management_refresh", lambda: None)

    assert system.cancel_active_transcription() is True
    deadline = time.monotonic() + 1.0
    while system.is_transcription_busy() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert system.transcription_thread is None
    assert system._active_transcription_generation is None
    assert system.is_transcription_busy() is False
    assert session.phase is UiPhase.CANCELLED
    assert "cue worker failed to start" in caplog.text.casefold()


def test_cancel_reaper_start_failure_rolls_back_unstarted_worker(
    monkeypatch, tmp_path, caplog
) -> None:
    from utils import system

    class RefusedReaper:
        def __init__(self, **_kwargs) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("cleanup worker creation refused")

    recording_path = tmp_path / "cancelled-before-capture.wav"
    cleanups: list[Path | None] = []
    session = TranscriptionUiSession()
    session.begin_recording()
    monkeypatch.setattr(
        system,
        "threading",
        types.SimpleNamespace(Thread=RefusedReaper),
    )
    monkeypatch.setattr(system, "transcription_ui_session", session)
    monkeypatch.setattr(system, "transcription_cancel_event", threading.Event())
    monkeypatch.setattr(system, "_recording_stop_event", threading.Event())
    monkeypatch.setattr(system, "recording", True)
    monkeypatch.setattr(system, "recording_thread", None)
    monkeypatch.setattr(system, "transcription_thread", None)
    monkeypatch.setattr(system, "recording_file_path", recording_path)
    monkeypatch.setattr(system, "_active_transcription_generation", 4991)
    monkeypatch.setattr(system, "cleanup_recording_file", cleanups.append)
    monkeypatch.setattr(system, "play_ui_cue", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(system, "_set_terminal_overlay_hide", lambda _delay: None)
    monkeypatch.setattr(system, "_refresh_tray_menu", lambda: None)
    monkeypatch.setattr(system, "schedule_management_refresh", lambda: None)
    _seed_waveform(system)

    assert system.cancel_active_transcription() is True

    assert system.recording is False
    assert system.recording_thread is None
    assert system.transcription_thread is None
    assert system.recording_file_path is None
    assert system._active_transcription_generation is None
    assert system.is_transcription_busy() is False
    assert session.phase is UiPhase.CANCELLED
    assert system.transcription_cancel_event.is_set()
    assert system._recording_stop_event.is_set()
    assert cleanups == [recording_path]
    assert _waveform_is_empty(system)
    assert "failed to start the cancelled-recording cleanup worker" in caplog.text.casefold()


def test_cancel_reaper_start_failure_leaves_only_live_recorder_until_it_exits(
    monkeypatch, tmp_path
) -> None:
    from utils import system

    class RefusedReaper:
        def __init__(self, **_kwargs) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("cleanup worker creation refused")

    entered_capture = threading.Event()
    release_capture = threading.Event()
    recording_path = tmp_path / "live-cancelled-capture.wav"
    cleanups: list[Path | None] = []
    stop_event = threading.Event()
    failure_event = threading.Event()
    cancel_event = threading.Event()
    generation = 4992

    def fake_record_audio(_path: Path, worker_stop: threading.Event) -> None:
        entered_capture.set()
        assert worker_stop.wait(1.0)
        assert release_capture.wait(1.0)

    session = TranscriptionUiSession()
    session.begin_recording()
    monkeypatch.setattr(system, "record_audio", fake_record_audio)
    monkeypatch.setattr(system, "transcription_ui_session", session)
    monkeypatch.setattr(system, "transcription_cancel_event", cancel_event)
    monkeypatch.setattr(system, "_recording_stop_event", stop_event)
    monkeypatch.setattr(system, "_recording_failed_event", failure_event)
    monkeypatch.setattr(system, "recording", True)
    monkeypatch.setattr(system, "transcription_thread", None)
    monkeypatch.setattr(system, "recording_file_path", recording_path)
    monkeypatch.setattr(system, "_active_transcription_generation", generation)
    monkeypatch.setattr(system, "cleanup_recording_file", cleanups.append)
    monkeypatch.setattr(system, "play_ui_cue", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(system, "_set_terminal_overlay_hide", lambda _delay: None)
    monkeypatch.setattr(system, "_refresh_tray_menu", lambda: None)
    monkeypatch.setattr(system, "schedule_management_refresh", lambda: None)

    recorder = threading.Thread(
        target=system._record_audio_worker,
        args=(recording_path, generation, stop_event, failure_event, cancel_event),
        daemon=True,
    )
    monkeypatch.setattr(system, "recording_thread", recorder)
    recorder.start()
    assert entered_capture.wait(1.0)
    monkeypatch.setattr(
        system,
        "threading",
        types.SimpleNamespace(
            Thread=RefusedReaper,
            current_thread=threading.current_thread,
        ),
    )

    assert system.cancel_active_transcription() is True
    assert system.transcription_thread is None
    assert system.recording_thread is recorder
    assert system._active_transcription_generation == generation
    assert system.is_transcription_busy() is True

    release_capture.set()
    recorder.join(1.0)

    assert not recorder.is_alive()
    assert system.recording_thread is None
    assert system.transcription_thread is None
    assert system.recording_file_path is None
    assert system._active_transcription_generation is None
    assert system.is_transcription_busy() is False
    assert cleanups == [recording_path]


def test_recorder_thread_start_failure_rolls_back_atomically(monkeypatch, tmp_path) -> None:
    from utils import system

    recording_path = tmp_path / "recorder-never-started.wav"
    cleanups: list[Path | None] = []
    notifications: list[tuple[str, str]] = []
    cues: list[CueKind] = []

    class RefusedWorker:
        def __init__(self, **_kwargs) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("recorder start refused")

        def is_alive(self) -> bool:
            return False

    fake_threading = types.SimpleNamespace(
        Event=threading.Event,
        Thread=RefusedWorker,
        current_thread=threading.current_thread,
    )
    monkeypatch.setattr(system, "threading", fake_threading)
    monkeypatch.setattr(system, "client_enabled", True)
    monkeypatch.setattr(system, "recording", False)
    monkeypatch.setattr(system, "recording_thread", None)
    monkeypatch.setattr(system, "transcription_thread", None)
    monkeypatch.setattr(system, "recording_file_path", None)
    monkeypatch.setattr(system, "_active_transcription_generation", None)
    monkeypatch.setattr(system, "transcription_cancel_event", threading.Event())
    monkeypatch.setattr(system, "_recording_stop_event", threading.Event())
    monkeypatch.setattr(system, "_recording_failed_event", threading.Event())
    monkeypatch.setattr(system, "_pyaudio_capture_pending", threading.Event())
    monkeypatch.setattr(system, "transcription_ui_session", TranscriptionUiSession())
    monkeypatch.setattr(system, "is_right_control", lambda _key: True)
    monkeypatch.setattr(system, "_observe_pynput_press", lambda _key: None)
    monkeypatch.setattr(system, "_client_hotkey_available", lambda: True)
    monkeypatch.setattr(system, "create_recording_file_path", lambda: recording_path)
    monkeypatch.setattr(system, "cleanup_recording_file", cleanups.append)
    monkeypatch.setattr(
        system,
        "notify_error",
        lambda context, detail: notifications.append((context, detail)),
    )
    monkeypatch.setattr(system, "play_ui_cue", lambda kind, **_kwargs: cues.append(kind) or True)
    monkeypatch.setattr(
        system,
        "_show_recording_overlay",
        lambda: pytest.fail("failed recorder must not show a recording overlay"),
    )
    monkeypatch.setattr(system, "_set_terminal_overlay_hide", lambda _delay: None)
    monkeypatch.setattr(system, "_refresh_tray_menu", lambda: None)
    monkeypatch.setattr(system, "schedule_management_refresh", lambda: None)
    _seed_waveform(system)

    system.on_press(object())

    assert system.recording is False
    assert system.recording_thread is None
    assert system.transcription_thread is None
    assert system.recording_file_path is None
    assert system._active_transcription_generation is None
    assert system.is_transcription_busy() is False
    assert system.transcription_ui_session.phase is UiPhase.ERROR
    assert system.transcription_cancel_event.is_set()
    assert system._recording_stop_event.is_set()
    assert system._recording_failed_event.is_set()
    assert not system._pyaudio_capture_pending.is_set()
    assert cleanups == [recording_path]
    assert notifications and notifications[0][0] == "Microphone recording failed to start"
    assert cues == [CueKind.ERROR]
    assert _waveform_is_empty(system)


def test_transcriber_thread_start_failure_rolls_back_atomically(
    monkeypatch, tmp_path
) -> None:
    from utils import system

    recording_path = _usable_recording(tmp_path / "coordinator-never-started.wav")
    recorder = FakeThread(alive=False)
    cleanups: list[Path | None] = []
    notifications: list[tuple[str, str]] = []
    cues: list[CueKind] = []

    class RefusedWorker:
        def __init__(self, **_kwargs) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("coordinator start refused")

    fake_threading = types.SimpleNamespace(
        Thread=RefusedWorker,
        current_thread=threading.current_thread,
    )
    session = TranscriptionUiSession()
    session.begin_recording()
    monkeypatch.setattr(system, "threading", fake_threading)
    monkeypatch.setattr(system, "transcription_ui_session", session)
    monkeypatch.setattr(system, "transcription_cancel_event", threading.Event())
    monkeypatch.setattr(system, "_recording_stop_event", threading.Event())
    monkeypatch.setattr(system, "_recording_failed_event", threading.Event())
    monkeypatch.setattr(system, "recording", True)
    monkeypatch.setattr(system, "recording_thread", recorder)
    monkeypatch.setattr(system, "transcription_thread", None)
    monkeypatch.setattr(system, "recording_file_path", recording_path)
    monkeypatch.setattr(system, "_active_transcription_generation", 500)
    monkeypatch.setattr(system, "is_right_control", lambda _key: True)
    monkeypatch.setattr(system, "_observe_pynput_release", lambda _key: None)
    monkeypatch.setattr(system, "_show_processing_overlay", lambda: None)
    monkeypatch.setattr(system, "start_processing_feedback", lambda: None)
    monkeypatch.setattr(system, "cleanup_recording_file", cleanups.append)
    monkeypatch.setattr(
        system,
        "notify_error",
        lambda context, detail: notifications.append((context, detail)),
    )
    monkeypatch.setattr(system, "play_ui_cue", lambda kind, **_kwargs: cues.append(kind) or True)
    monkeypatch.setattr(system, "_set_terminal_overlay_hide", lambda _delay: None)
    monkeypatch.setattr(system, "_refresh_tray_menu", lambda: None)
    monkeypatch.setattr(system, "schedule_management_refresh", lambda: None)
    _seed_waveform(system)

    system.on_release(object())

    assert recorder.join_calls == [2.5]
    assert system.recording is False
    assert system.recording_thread is None
    assert system.transcription_thread is None
    assert system.recording_file_path is None
    assert system._active_transcription_generation is None
    assert system.is_transcription_busy() is False
    assert session.phase is UiPhase.ERROR
    assert system.transcription_cancel_event.is_set()
    assert system._recording_stop_event.is_set()
    assert system._recording_failed_event.is_set()
    assert cleanups == [recording_path]
    assert notifications and notifications[0][0] == "Transcription worker failed to start"
    assert cues == [CueKind.ERROR]
    assert _waveform_is_empty(system)


def test_optional_pyaudio_declines_on_capture_marker_even_with_free_lock(
    monkeypatch,
) -> None:
    from utils import system

    created: list[object] = []
    terminated: list[object] = []

    class FakePyAudio:
        def terminate(self) -> None:
            terminated.append(self)

    def factory() -> FakePyAudio:
        instance = FakePyAudio()
        created.append(instance)
        return instance

    monkeypatch.setattr(system, "_pyaudio_session_lock", threading.Lock())
    capture_pending = threading.Event()
    capture_pending.set()
    monkeypatch.setattr(system, "_pyaudio_capture_pending", capture_pending)
    monkeypatch.setattr(system.pyaudio, "PyAudio", factory)

    with system._managed_pyaudio(blocking=False) as optional_runtime:
        assert optional_runtime is None
    assert created == []

    # The marker reserves priority for capture; capture itself remains allowed
    # to initialize and owns the guarded runtime through termination.
    with system._managed_pyaudio(blocking=True) as capture_runtime:
        assert capture_runtime is created[0]
    assert terminated == created


def test_capture_publication_wins_optional_pyaudio_interleaving(monkeypatch) -> None:
    from utils import system

    priority = threading.Lock()
    marker = threading.Event()
    optional_attempting = threading.Event()
    optional_result: list[object | None] = []
    created: list[object] = []

    class FakePyAudio:
        def __init__(self) -> None:
            created.append(self)

        def terminate(self) -> None:
            pass

    monkeypatch.setattr(system, "_pyaudio_priority_lock", priority)
    monkeypatch.setattr(system, "_pyaudio_session_lock", threading.Lock())
    monkeypatch.setattr(system, "_pyaudio_capture_pending", marker)
    monkeypatch.setattr(system.pyaudio, "PyAudio", FakePyAudio)

    def optional_work() -> None:
        optional_attempting.set()
        with system._managed_pyaudio(blocking=False) as runtime:
            optional_result.append(runtime)

    # Model on_press already owning arbitration while it publishes capture.
    priority.acquire()
    worker = threading.Thread(target=optional_work, daemon=True)
    worker.start()
    assert optional_attempting.wait(1.0)
    marker.set()
    priority.release()
    worker.join(1.0)

    assert not worker.is_alive()
    assert optional_result == [None]
    assert created == []


def test_idle_non_windows_cue_uses_native_helper_without_pyaudio(monkeypatch) -> None:
    from utils import system

    calls: list[tuple[list[str], bytes]] = []

    def fake_which(name: str) -> str | None:
        return "/usr/bin/aplay" if name == "aplay" else None

    def fake_run(command, *, input, **_kwargs):
        calls.append((command, input))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(system.sys, "platform", "linux")
    monkeypatch.setattr(system.shutil, "which", fake_which)
    monkeypatch.setattr(system.subprocess, "run", fake_run)
    monkeypatch.setattr(
        system.pyaudio,
        "PyAudio",
        lambda: pytest.fail("idle native cue must not initialize PortAudio"),
    )
    monkeypatch.setattr(system, "_pyaudio_capture_pending", threading.Event())
    monkeypatch.setattr(system, "recording", False)
    monkeypatch.setattr(system, "recording_thread", None)
    monkeypatch.setattr(system, "transcription_thread", None)
    monkeypatch.setattr(system, "_active_transcription_generation", None)
    monkeypatch.setattr(system, "transcription_ui_session", TranscriptionUiSession())
    cue = synthesize_cue(CueKind.SUCCESS)

    system._play_pcm_cue(cue)

    assert len(calls) == 1
    command, payload = calls[0]
    assert command == ["/usr/bin/aplay", "--quiet"]
    assert payload[:4] == b"RIFF"
    assert payload[8:12] == b"WAVE"


def test_non_windows_lifecycle_cues_supersede_stale_pending_feedback(
    monkeypatch,
) -> None:
    from utils import system

    entered_first = threading.Event()
    release_first = threading.Event()
    latest_delivered = threading.Event()
    calls: list[tuple[str, CueKind | None]] = []

    def native_helper(cue) -> bool:
        calls.append(("start", cue.kind))
        if cue.kind is CueKind.RECORDING_STARTED:
            if not entered_first.is_set():
                entered_first.set()
                assert release_first.wait(2.0)
        calls.append(("done", cue.kind))
        if len(calls) == 4:
            latest_delivered.set()
        return True

    monkeypatch.setattr(system.sys, "platform", "linux")
    monkeypatch.setattr(system, "_play_native_non_windows_cue", native_helper)
    monkeypatch.setattr(
        system.pyaudio,
        "PyAudio",
        lambda: pytest.fail("non-Windows cues must never initialize PyAudio"),
    )
    monkeypatch.setitem(system.settings, "audio_cues_enabled", True)
    monkeypatch.setitem(system.settings, "audio_cue_volume", 30)

    assert system.play_ui_cue(CueKind.RECORDING_STARTED) is True
    assert entered_first.wait(1.0)
    # These lifecycle transitions happen while the first helper is blocked.
    # Only the newest pending state (the next generation's start) may survive.
    assert system.play_ui_cue(CueKind.PROCESSING_STARTED) is True
    assert system.play_ui_cue(CueKind.CANCELLED) is True
    assert system.play_ui_cue(CueKind.RECORDING_STARTED) is True
    assert calls == [("start", CueKind.RECORDING_STARTED)]

    release_first.set()
    assert latest_delivered.wait(2.0)
    assert calls == [
        ("start", CueKind.RECORDING_STARTED),
        ("done", CueKind.RECORDING_STARTED),
        ("start", CueKind.RECORDING_STARTED),
        ("done", CueKind.RECORDING_STARTED),
    ]

    # Once the worker catches up, a later cue is delivered normally rather
    # than replaying either superseded state.
    delivered_next = threading.Event()

    def next_helper(cue) -> bool:
        calls.append(("next", cue.kind))
        delivered_next.set()
        return True

    monkeypatch.setattr(system, "_play_native_non_windows_cue", next_helper)
    assert system.play_ui_cue(CueKind.SUCCESS) is True
    assert delivered_next.wait(1.0)
    assert calls[-1] == ("next", CueKind.SUCCESS)
    assert not any(kind is CueKind.PROCESSING_STARTED for _phase, kind in calls)
    assert not any(kind is CueKind.CANCELLED for _phase, kind in calls)


def test_non_windows_native_helper_timeout_never_blocks_caller(
    monkeypatch, caplog
) -> None:
    from utils import system

    helper_entered = threading.Event()
    allow_timeout = threading.Event()
    helper_finished = threading.Event()
    timeouts: list[float] = []

    monkeypatch.setattr(
        system.shutil,
        "which",
        lambda name: "/usr/bin/aplay" if name == "aplay" else None,
    )

    def timed_out_run(_command, *, timeout: float, **_kwargs):
        timeouts.append(timeout)
        helper_entered.set()
        assert allow_timeout.wait(2.0)
        helper_finished.set()
        raise system.subprocess.TimeoutExpired("aplay", timeout)

    monkeypatch.setattr(system.subprocess, "run", timed_out_run)
    monkeypatch.setattr(system.sys, "platform", "linux")
    monkeypatch.setattr(
        system.pyaudio,
        "PyAudio",
        lambda: pytest.fail("native cue timeout must never touch PyAudio"),
    )
    monkeypatch.setitem(system.settings, "audio_cues_enabled", True)
    monkeypatch.setitem(system.settings, "audio_cue_volume", 30)

    started = time.monotonic()
    assert system.play_ui_cue(CueKind.ERROR) is True
    assert time.monotonic() - started < 0.1
    assert helper_entered.wait(1.0)
    allow_timeout.set()
    assert helper_finished.wait(1.0)
    deadline = time.monotonic() + 1.0
    while "native ui cue helper" not in caplog.text.casefold() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert timeouts == [3.0]
    assert "native ui cue helper aplay timed out" in caplog.text.casefold()


def test_busy_audio_device_refresh_uses_cache_and_preserves_saved_preference(
    monkeypatch,
) -> None:
    from utils import system

    created: list[object] = []

    class DevicePyAudio:
        def __init__(self) -> None:
            created.append(self)

        @staticmethod
        def get_host_api_count() -> int:
            return 1

        @staticmethod
        def get_host_api_info_by_index(_index: int) -> dict[str, object]:
            return {"name": "PipeWire"}

        @staticmethod
        def get_device_count() -> int:
            return 1

        @staticmethod
        def get_device_info_by_index(_index: int) -> dict[str, object]:
            return {"name": "Studio Mic", "maxInputChannels": 2, "hostApi": 0}

        @staticmethod
        def terminate() -> None:
            pass

    monkeypatch.setattr(system.pyaudio, "PyAudio", DevicePyAudio)
    monkeypatch.setattr(system, "_pyaudio_session_lock", threading.Lock())
    monkeypatch.setattr(system, "_input_device_cache_lock", threading.Lock())
    monkeypatch.setattr(system, "_input_device_cache", ())
    monkeypatch.setitem(system.settings, "input_device", "Studio Mic")

    expected = [("Studio Mic", "Studio Mic · PipeWire")]
    assert system.list_input_audio_devices() == expected
    assert len(created) == 1

    with system._managed_pyaudio(blocking=True):
        assert system.list_input_audio_devices() == expected
        assert system.get_input_device_preference() == "Studio Mic"
        assert len(created) == 2

        # Even before any successful scan, a busy refresh represents the saved
        # selection provisionally instead of resetting the combo to default.
        system._input_device_cache = ()
        provisional = system.list_input_audio_devices()
        assert provisional == [
            ("Studio Mic", "Studio Mic · saved preference (scan deferred)")
        ]
        assert system.get_input_device_preference() == "Studio Mic"


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
    _seed_waveform(system)

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
    assert _waveform_is_empty(system)


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
    _seed_waveform(system)

    assert system.cancel_active_transcription() is True
    reaper = system.transcription_thread
    assert isinstance(reaper, threading.Thread)
    assert recorder.is_alive()
    assert system.recording_thread is recorder
    assert system.is_transcription_busy() is True
    assert _waveform_is_empty(system)

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
    _seed_waveform(system)

    system.on_release(object())
    coordinator = system.transcription_thread
    assert isinstance(coordinator, threading.Thread)
    assert recorder.is_alive()
    assert system.recording_thread is recorder
    assert system.is_transcription_busy() is True
    assert session.phase is UiPhase.PROCESSING
    assert _waveform_is_empty(system)

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
    system._clear_waveform_buffers()
    _seed_waveform(system)

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
    assert not _waveform_is_empty(system)
    system._clear_waveform_buffers()


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
    _seed_waveform(system)

    started = time.monotonic()
    assert system.cancel_and_wait_for_active_transcription(0.05) is False
    assert time.monotonic() - started < 0.5
    assert worker.is_alive()
    assert system.transcription_thread is worker
    assert not recording.exists()
    assert _waveform_is_empty(system)

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


def test_overlay_monitor_follows_foreground_window_before_cursor(monkeypatch) -> None:
    import ctypes
    from types import SimpleNamespace

    from utils import midnight_overlay

    class FakeRoot:
        def winfo_screenwidth(self) -> int:
            return 800

        def winfo_screenheight(self) -> int:
            return 600

    class FakeUser32:
        def GetForegroundWindow(self) -> int:
            return 101

        def MonitorFromWindow(self, window: int, flags: int) -> int:
            assert (window, flags) == (101, 2)
            return 202

        def GetCursorPos(self, _point) -> int:
            raise AssertionError("cursor fallback must not run for a foreground window")

        def MonitorFromPoint(self, _point, _flags: int) -> int:
            raise AssertionError("cursor monitor must not replace foreground monitor")

        def GetMonitorInfoW(self, monitor: int, info_pointer) -> int:
            assert monitor == 202
            work = info_pointer._obj.rcWork
            work.left, work.top, work.right, work.bottom = (-1920, 24, 0, 1080)
            return 1

    monkeypatch.setattr(midnight_overlay.sys, "platform", "win32")
    monkeypatch.setattr(
        ctypes,
        "windll",
        SimpleNamespace(user32=FakeUser32()),
    )

    assert midnight_overlay.active_monitor_bounds(FakeRoot()) == (
        midnight_overlay.MonitorBounds(-1920, 24, 0, 1080)
    )


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
