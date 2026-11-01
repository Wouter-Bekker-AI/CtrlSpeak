"""Controller acceptance without keyboard hooks, microphone, GUI or network.

The real gesture recognizer and real session privacy/delivery gates are wired
to an inert host. Thread objects are never started in these tests.
"""
from __future__ import annotations

from contextlib import nullcontext
import threading
from types import SimpleNamespace

import pytest

from utils import dictation_integration as integration
from utils.dictation_runtime import DeliveryCancelled, DictationSession
from utils.request_control import current_request_control


pytestmark = pytest.mark.core_headless


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class Host:
    def __init__(self, primary="ctrl_r", secondary="ctrl_l"):
        self.settings = {
            "dictation_primary_key": primary,
            "dictation_secondary_key": secondary,
            "dictation_live_enabled": True,
        }
        self.client_enabled = True
        self.recording = False
        self.cleanup_pending = False
        self._hold_gesture_tentative = False
        self.controller = None
        self.presses = []
        self.releases = []
        self.observed_presses = []
        self.observed_releases = []
        self.cancels = 0
        self.overlays = 0
        self.cues = []
        self.notifications = []
        self.failures = []
        self.capture_priority = []
        self.injected = []
        self.buffer_clears = 0
        self.CueKind = SimpleNamespace(RECORDING_STARTED="recording_started")
        self.logger = SimpleNamespace(warning=lambda *_args: None, exception=lambda *_args: None)
        self._failed_recording = SimpleNamespace(discard=lambda: None)
        self.transcription_ui_session = SimpleNamespace(fail=self.failures.append)

    def _observe_pynput_press(self, key):
        self.observed_presses.append(key)

    def _observe_pynput_release(self, key):
        self.observed_releases.append(key)

    def is_transcription_busy(self):
        return self.recording or self.cleanup_pending or bool(self.controller and self.controller.busy())

    def _client_hotkey_available(self):
        return True

    def on_press(self, key):
        self.presses.append((key, self._hold_gesture_tentative))
        self.recording = True

    def on_release(self, key):
        self.releases.append(key)
        self.recording = False
        self.cleanup_pending = True

    def cancel_active_transcription(self):
        self.cancels += 1
        if self.controller and self.controller.cancel():
            return True
        if self.recording:
            self.recording = False
            self.cleanup_pending = True
        return True

    def _show_recording_overlay(self):
        self.overlays += 1

    def play_ui_cue(self, cue):
        self.cues.append(cue)

    def _clear_waveform_buffers(self):
        self.buffer_clears += 1

    def _prepare_ui_session_for_recording(self):
        pass

    def _set_pyaudio_capture_pending(self, value):
        self.capture_priority.append(value)

    def _refresh_tray_menu(self):
        pass

    def schedule_management_refresh(self):
        pass

    def notify(self, text):
        self.notifications.append(text)

    def inject_transcription_result(self, result):
        self.injected.append(result)


class InertStartedSession(DictationSession):
    """Real session gates; starting it does not create a mic/request thread."""

    def start(self):
        self.start_called = True


@pytest.fixture
def build_controller(monkeypatch):
    # This is the Windows integration contract, not a simulated live feature
    # claim on platforms without reliable physical/injected input separation.
    monkeypatch.setattr(integration.sys, "platform", "win32")
    monkeypatch.setattr(integration, "desktop_target", lambda: "test-blank-editor")
    monkeypatch.setattr(integration, "DictationSession", InertStartedSession)

    def build(primary="ctrl_r", secondary="ctrl_l"):
        host, clock = Host(primary, secondary), Clock()
        controller = integration.DesktopDictationController(host)
        host.controller = controller
        controller.gestures._clock = clock
        controller.mouse_ready = True
        return controller, host, clock

    return build


def tap(controller, clock, key, duration=0.04):
    controller.press(key)
    clock.advance(duration)
    controller.release(key)


def commit_hold(controller, clock):
    clock.advance(0.20)
    controller._actions(controller.gestures.tick(mode=controller._mode()))


def test_first_short_tap_captures_preroll_but_never_submits_or_plays_recording_cue(build_controller):
    controller, host, clock = build_controller()
    tap(controller, clock, "ctrl_r")
    assert host.presses == [("Key.ctrl_r", True)]
    assert host.releases == []
    assert host.cancels == 1
    assert host.cleanup_pending
    assert host.overlays == 0 and host.cues == []
    assert controller.session is None and controller.pending_start is None
    assert not controller.provisional and not controller.hold_committed


def test_normal_hold_commits_once_and_uses_existing_release_pipeline(build_controller):
    controller, host, clock = build_controller()
    controller.press("ctrl_r")
    commit_hold(controller, clock)
    controller.press("ctrl_r")  # Hardware repeat must not create another capture.
    clock.advance(1.0)
    controller.release("ctrl_r")
    assert len(host.presses) == 1
    assert host.releases == ["Key.ctrl_r"]
    assert host.overlays == 1
    assert host.cues == ["recording_started"]
    assert host.cancels == 0
    assert not controller.hold_committed


@pytest.mark.parametrize("before,after,vk", [("\x03","c",67),("!","1",49),("é","e",69)])
def test_character_changes_on_release_never_leave_phantom_keys_blocking_ctrl(build_controller,before,after,vk):
    controller,host,clock=build_controller()
    controller.press("ctrl_l")
    controller.press(SimpleNamespace(char=before,vk=vk))
    controller.release("ctrl_l")
    controller.release(SimpleNamespace(char=after,vk=vk))
    assert controller.gestures.keys_down==frozenset()
    controller.press("ctrl_r")
    commit_hold(controller,clock)
    assert host.recording and controller.hold_committed
    controller.release("ctrl_r")
    assert host.releases==["Key.ctrl_r"]


def test_blocked_hold_reports_busy_once_without_auto_starting_later(build_controller):
    controller,host,clock=build_controller()
    host.cleanup_pending=True
    controller.press("ctrl_r")
    commit_hold(controller,clock)
    assert not host.recording and not controller.hold_committed
    assert "still finishing" in host.notifications[-1]
    host.cleanup_pending=False
    commit_hold(controller,clock)
    assert len(host.notifications)==1 and not host.recording
    controller.release("ctrl_r")
    controller.press("ctrl_r");commit_hold(controller,clock)
    assert host.recording and controller.hold_committed


def test_double_tap_queues_hands_free_while_first_capture_is_being_reaped(build_controller):
    controller, host, clock = build_controller()
    tap(controller, clock, "ctrl_r")
    clock.advance(0.05)
    tap(controller, clock, "ctrl_r")
    assert len(host.presses) == 1  # Second tap does not start a second PortAudio owner.
    assert host.releases == []
    assert controller.pending_start[0] == "hands_free"
    assert controller.session is None

    class MonitorSteps:
        def __init__(self):
            self.calls = 0

        def wait(self, _timeout):
            self.calls += 1
            if self.calls == 2:
                assert controller.session is None
                host.cleanup_pending = False
            return self.calls > 2

    controller.stop = MonitorSteps()
    controller._monitor()
    assert controller.pending_start is None
    assert controller.session.mode == "hands_free"
    assert controller.session.start_called
    assert host.releases == []
    assert host.capture_priority == [True]


@pytest.mark.parametrize("first,second", [("ctrl_l", "ctrl_r"), ("ctrl_r", "ctrl_l")])
def test_live_chord_waits_for_both_releases_and_does_not_toggle_mute(build_controller, first, second):
    controller, host, clock = build_controller()
    controller.press(first)
    clock.advance(0.03)
    controller.press(second)
    controller.release(second)
    assert controller.pending_start is None
    controller.release(first)
    assert controller.pending_start[0] == "live"
    assert host.releases == []
    host.cleanup_pending = False
    assert controller.start_mode("live")
    assert not controller.session.muted
    assert controller.session.epoch == 0


@pytest.mark.parametrize("left", ["Key.ctrl", "Key.ctrl_l"])
@pytest.mark.parametrize("right_first", [False, True])
@pytest.mark.parametrize("right_release_first", [False, True])
@pytest.mark.parametrize("committed_hold", [False, True])
def test_live_shortcut_activates_through_real_controller_and_monitor(
    build_controller, left, right_first, right_release_first, committed_hold,
):
    controller, host, clock = build_controller()
    first, second = ("Key.ctrl_r", left) if right_first else (left, "Key.ctrl_r")
    controller.press(first)
    if committed_hold:
        commit_hold(controller, clock)
    controller.press(second)
    assert controller.pending_start is None
    releases = ("Key.ctrl_r", left) if right_release_first else (left, "Key.ctrl_r")
    controller.release(releases[0])
    assert controller.pending_start is None
    controller.release(releases[1])
    assert controller.pending_start[0] == "live"
    assert not controller.gestures.keys_down
    assert not host.releases  # Never submit provisional/aborted Hold audio.
    host.cleanup_pending = False

    class OneStep:
        calls = 0

        def wait(self, _timeout):
            self.calls += 1
            return self.calls > 1

    controller.stop = OneStep()
    controller._monitor()
    assert controller.pending_start is None
    assert controller.session.mode == "live"
    assert controller.session.start_called
    assert not controller.session.muted
    assert not controller.session.finishing


@pytest.mark.parametrize("modifier", ["ctrl_r", "ctrl_l"])
def test_active_control_c_is_editing_not_finish_or_mute(build_controller, modifier):
    controller, _host, clock = build_controller()
    assert controller.start_mode("live")
    session = controller.session
    controller.press(modifier)
    controller.press("c")
    controller.release("c")
    controller.release(modifier)
    assert session.editing
    assert not session.finishing and not session.muted
    assert not session.keys_held
    assert not session.stop_capture.is_set()


def test_physical_left_control_mutes_even_during_windows_insertion(build_controller, monkeypatch):
    monkeypatch.setattr(integration.sys, "platform", "win32")
    controller, host, clock = build_controller()
    assert controller.start_mode("live")
    session = controller.session
    guard = DeliveryCancelled(session, session.epoch)
    observations = []

    def inserting(_result):
        assert controller.sending.is_set()
        assert current_request_control().cancel is guard
        assert not guard.is_set()
        controller.press("ctrl_l")
        # Even before tap recognition, a held physical modifier gates insertion.
        observations.append(guard.is_set())
        clock.advance(0.04)
        controller.release("ctrl_l")
        observations.append(guard.is_set())

    host.inject_transcription_result = inserting
    controller._deliver(SimpleNamespace(text="harmless sample"), guard)
    assert observations == [True, True]
    assert session.muted
    assert not session.keys_held
    assert host.buffer_clears == 1
    assert not controller.sending.is_set()
    assert current_request_control() is None


def test_left_control_unmutes_only_after_isolated_tap_release(build_controller):
    controller, _host, clock = build_controller()
    assert controller.start_mode("hands_free")
    tap(controller, clock, "ctrl_l")
    assert controller.session.muted
    controller.press("ctrl_l")
    assert controller.session.muted
    clock.advance(0.04)
    controller.release("ctrl_l")
    assert not controller.session.muted


def test_finish_tap_does_not_unmute_a_muted_session(build_controller):
    controller, _host, clock = build_controller()
    assert controller.start_mode("hands_free")
    tap(controller, clock, "ctrl_l")
    tap(controller, clock, "ctrl_r")
    assert controller.session.finishing and controller.session.muted
    assert controller.session.stop_capture.is_set()


def test_hold_followed_by_editing_chord_is_aborted_not_submitted(build_controller):
    controller, host, clock = build_controller()
    controller.press("ctrl_r")
    commit_hold(controller, clock)
    controller.press("c")
    controller.release("c")
    controller.release("ctrl_r")
    assert host.cancels == 1
    assert host.releases == []
    assert not controller.hold_committed


def test_remapped_function_keys_route_hold_live_and_mute_without_literal_ctrl_toggles(build_controller):
    controller, host, clock = build_controller("f8", "f9")
    assert controller.status()["primary_label"] == "F8"
    assert controller.status()["secondary_label"] == "F9"
    tap(controller, clock, "ctrl_r")
    assert host.presses == []
    controller.press("f8")
    commit_hold(controller, clock)
    controller.release("f8")
    assert host.releases == ["Key.ctrl_r"]  # Existing recorder entry point is intentionally reused.
    host.cleanup_pending = False
    controller.press("f9")
    controller.press("f8")
    controller.release("f8")
    controller.release("f9")
    assert controller.pending_start[0] == "live"
    assert controller.start_mode("live")
    tap(controller, clock, "f9")
    assert controller.session.muted


def test_live_start_requires_opt_in_and_working_mouse_observer(build_controller):
    controller, host, _clock = build_controller()
    host.settings["dictation_live_enabled"] = False
    assert not controller.start_mode("live")
    host.settings["dictation_live_enabled"] = True
    controller.mouse_ready = False
    assert not controller.start_mode("live")
    assert controller.session is None
    assert not host.capture_priority


def test_linux_live_mode_is_explicitly_unavailable_but_hands_free_can_start(build_controller, monkeypatch):
    controller, host, _clock = build_controller()
    monkeypatch.setattr(integration.sys, "platform", "linux")
    assert not controller.start_mode("live")
    assert controller.session is None
    assert "requires Windows" in host.notifications[-1]
    assert controller.start_mode("hands_free")


def test_unknown_target_and_disabled_client_never_start_capture(build_controller, monkeypatch):
    controller, host, _clock = build_controller()
    monkeypatch.setattr(integration, "desktop_target", lambda: None)
    assert not controller.start_mode("hands_free")
    monkeypatch.setattr(integration, "desktop_target", lambda: "test-blank-editor")
    host.client_enabled = False
    assert not controller.start_mode("hands_free")
    assert controller.session is None
    assert not host.capture_priority


def test_pending_text_blocks_replacing_a_finished_session(build_controller):
    controller, host, _clock = build_controller()
    assert controller.start_mode("live")
    previous = controller.session
    previous.pending.append("Keep this harmless pending phrase")
    previous.done.set()
    assert not controller.start_mode("hands_free")
    assert controller.session is previous
    assert previous.pending_text() == "Keep this harmless pending phrase"
    assert "Copy or discard" in host.notifications[-1]


def test_cancel_clears_a_queued_start_without_starting_capture(build_controller):
    controller, host, clock = build_controller()
    tap(controller, clock, "ctrl_r")
    clock.advance(0.05)
    tap(controller, clock, "ctrl_r")
    assert controller.pending_start is not None
    controller.cancel()
    assert controller.pending_start is None
    assert controller.session is None
    assert host.releases == []


def test_explicit_cancel_resets_tap_window_and_held_keys(build_controller):
    controller, host, clock = build_controller()
    tap(controller, clock, "ctrl_r")
    controller.cancel()
    host.cleanup_pending = False
    clock.advance(0.05)
    tap(controller, clock, "ctrl_r")
    assert controller.pending_start is None
    controller.press("ctrl_l")
    controller.cancel()
    assert not controller.gestures.keys_down
    assert not controller.provisional and not controller.hold_committed


class FakeOwner:
    def __init__(self, alive=True, *, stuck=False):
        self.alive, self.stuck = alive, stuck
        self.joins = []
        self.stops = 0

    def is_alive(self):
        return self.alive

    def stop(self):
        self.stops += 1

    def join(self, timeout):
        self.joins.append(timeout)
        if timeout > 0 and not self.stuck:
            self.alive = False


def test_shutdown_zero_is_nonblocking_and_retains_live_owner_gate(build_controller):
    controller, host, _clock = build_controller()
    controller.monitor = FakeOwner()
    controller.mouse = FakeOwner()
    assert controller.shutdown(timeout=0) is False
    assert controller.monitor.joins == [0]
    assert controller.mouse.joins == [0]
    assert not controller.mouse_ready
    assert controller.start_mode("hands_free") is False
    controller.press("ctrl_r")
    assert host.presses == []
    assert controller.shutdown(timeout=0.1) is True
    with pytest.raises(RuntimeError, match="cannot restart"):
        controller.start()


def test_shutdown_checks_reader_even_after_session_done(build_controller):
    controller, _host, _clock = build_controller()
    assert controller.start_mode("hands_free")
    session = controller.session
    session.done.set()
    session.reader = FakeOwner(stuck=True)
    session.worker = FakeOwner(alive=False)
    assert controller.shutdown(timeout=0.01) is False
    assert session.reader.joins[0] <= 0.01
    session.reader.alive = False
    assert controller.shutdown(timeout=0) is True


@pytest.mark.parametrize("rapid", [False, True])
def test_capture_closes_and_reopens_stream_after_mute_epoch_without_buffer_leak(build_controller, rapid):
    controller, host, _clock = build_controller()
    host.FORMAT, host.CHANNELS, host.RATE, host.CHUNKSIZE = 8, 1, 16000, 2
    host._resolve_input_device_index = lambda _pa: None
    session = DictationSession("hands_free", capture=lambda *_args: None,
                               transcribe=lambda *_args: None, deliver=lambda *_args: None,
                               target=lambda: "editor")
    streams, events, feeds = [], [], []

    class Stop:
        stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, timeout):
            assert session.muted
            assert streams[0].closed  # No device capture while muted.
            assert timeout == 0.02
            events.append("muted_wait")
            session.toggle_mute()
            return False

    stop = Stop()

    class Stream:
        closed = False

        def __init__(self, number):
            self.number = number

        def read(self, size):
            assert size == 2 and not self.closed
            if self.number == 0:
                session.toggle_mute()  # Mute during a blocking read.
                if rapid:
                    session.toggle_mute()  # Both transitions between reads.
                return b"old!"
            return b"new!"

        def stop_stream(self):
            events.append(("stop", self.number))

        def close(self):
            self.closed = True
            events.append(("close", self.number))

    def open_stream(**_options):
        if streams:
            assert streams[-1].closed
        stream = Stream(len(streams))
        streams.append(stream)
        return stream

    host._managed_pyaudio = lambda **_kwargs: nullcontext(SimpleNamespace(open=open_stream))

    def feed(pcm, token):
        feeds.append(token)
        session.feed(pcm, token)
        if len(feeds) == 2:
            stop.stopped = True

    controller._capture(stop, feed, session.capture_token)
    assert bytes(session.frames) == b"new!"
    assert feeds == [(0, False), (2, False)]
    assert len(streams) == 2 and all(stream.closed for stream in streams)
    assert events.count(("close", 0)) == events.count(("close", 1)) == 1
    assert ("muted_wait" in events) is not rapid
    assert host.capture_priority == [False]


def test_capture_closes_stream_and_releases_priority_when_stop_stream_raises(build_controller):
    controller, host, _clock = build_controller()
    host.FORMAT, host.CHANNELS, host.RATE, host.CHUNKSIZE = 8, 1, 16000, 2
    host._resolve_input_device_index = lambda _pa: None
    stop, closed = threading.Event(), []

    def read(_size):
        stop.set()
        return b"pcm!"

    def failed_stop():
        raise RuntimeError("device stopped unexpectedly")

    stream = SimpleNamespace(read=read, stop_stream=failed_stop, close=lambda: closed.append(True))
    host._managed_pyaudio = lambda **_kwargs: nullcontext(SimpleNamespace(open=lambda **_options: stream))
    with pytest.raises(RuntimeError, match="device stopped unexpectedly"):
        controller._capture(stop, lambda *_args: None, lambda: (0, False))
    assert closed == [True]
    assert host.capture_priority == [False]


def test_monitor_keeps_fast_gestures_but_throttles_desktop_queries(build_controller, monkeypatch):
    controller, _host, clock = build_controller()
    assert controller.start_mode("live")
    queries, ticks = [], []
    monkeypatch.setattr(integration.time, "monotonic", clock)
    monkeypatch.setattr(integration, "desktop_target", lambda: queries.append(clock()) or "editor")
    controller.session.target = integration.desktop_target
    controller.session.expected_target = "editor"
    controller.gestures.tick = lambda **_kwargs: ticks.append(clock()) or []

    class Steps:
        def wait(self, timeout):
            clock.advance(timeout)
            return clock.now > 100.51

    controller.stop = Steps()
    controller._monitor()
    assert 24 <= len(ticks) <= 26
    assert 8 <= len(queries) <= 10  # Two identity reads per 100 ms safety poll.


def test_monitor_aborts_live_capture_if_physical_mouse_observer_dies(build_controller):
    controller, host, _clock = build_controller()
    assert controller.start_mode("live")
    controller.mouse = FakeOwner(alive=False)

    class OneStep:
        calls = 0

        def wait(self, _timeout):
            self.calls += 1
            return self.calls > 1

    controller.stop = OneStep()
    controller._monitor()
    assert controller.session.cancel.is_set()
    assert controller.session.stop_capture.is_set()
    assert not controller.mouse_ready
    assert len(host.notifications) == 1 and "observer disconnected" in host.notifications[0]


@pytest.mark.parametrize("mode", ["hands_free", "live"])
def test_monitor_aborts_once_if_keyboard_observer_dies(build_controller, monkeypatch, mode):
    controller, host, clock = build_controller()
    assert controller.start_mode(mode)
    host.listener = FakeOwner(alive=False)
    monkeypatch.setattr(integration.time, "monotonic", clock)

    class SeveralSteps:
        calls = 0

        def wait(self, _timeout):
            self.calls += 1
            clock.advance(0.11)
            return self.calls > 3

    controller.stop = SeveralSteps()
    controller._monitor()
    assert controller.session.cancel.is_set()
    assert controller.session.stop_capture.is_set()
    assert not controller.mouse_ready
    assert len(host.notifications) == 1
    assert "keyboard observer disconnected" in host.notifications[0]
