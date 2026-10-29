"""Desktop wiring for the 0.8 dictation controller; no side effects on import."""
from __future__ import annotations

import ctypes
import sys
import threading
import time
import wave

from utils.dictation_modes import DictationGestures, DictationMode, GestureAction
from utils.dictation_runtime import DictationSession
from utils.hotkeys import key_name
from utils.request_control import request_control


KEY_LABELS = {"ctrl_r": "Right Ctrl", "ctrl_l": "Left Ctrl", "f8": "F8", "f9": "F9", "f10": "F10"}
TARGET_POLL_SECONDS = 0.1


def desktop_target():
    """Read target identity only. Never activates a window or moves the caret."""
    if sys.platform.startswith("win"):
        from utils.windows_input import get_focused_control
        user32 = ctypes.windll.user32
        user32.OpenInputDesktop.restype = ctypes.c_void_p
        user32.CloseDesktop.argtypes = [ctypes.c_void_p]
        desktop = user32.OpenInputDesktop(0, False, 0x0001)
        if not desktop:
            return None
        try:
            name = ctypes.create_unicode_buffer(128)
            needed = ctypes.c_ulong()
            user32.GetUserObjectInformationW.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                       ctypes.c_void_p, ctypes.c_ulong,
                                                       ctypes.POINTER(ctypes.c_ulong)]
            if not user32.GetUserObjectInformationW(desktop, 2, name, ctypes.sizeof(name), ctypes.byref(needed)):
                return None
            if name.value.lower() != "default":
                return None
            return get_focused_control() or None
        finally:
            user32.CloseDesktop(desktop)
    if sys.platform.startswith("linux"):
        try:
            from Xlib.display import Display
            connection = Display()
            try:
                focus = connection.get_input_focus().focus
                return getattr(focus, "id", None)
            finally:
                connection.close()
        except Exception:
            return None
    return None


class DesktopDictationController:
    def __init__(self, host):
        self.host = host
        self.lock = threading.RLock()
        try:
            self.gestures = DictationGestures(
                primary_key=str(host.settings.get("dictation_primary_key", "ctrl_r")),
                secondary_key=str(host.settings.get("dictation_secondary_key", "ctrl_l")))
        except ValueError:
            self.gestures = DictationGestures()
        self.session: DictationSession | None = None
        self.hold_committed = False
        self.provisional = False
        self.pending_start: tuple[str, float] | None = None
        self.stop = threading.Event()
        self.mouse = None
        self.mouse_ready = False
        self.finished_notified = False
        self.sending = threading.Event()
        self._closed = False
        self._started = False
        self._preserve_gesture_cancel = False
        self.monitor = threading.Thread(target=self._monitor, name="CtrlSpeakDictationControls", daemon=True)

    def start(self):
        with self.lock:
            if self._closed:
                raise RuntimeError("A stopped dictation controller cannot restart")
            if self._started:
                return
            self._started = True
        try:
            from pynput import mouse
            self.mouse = mouse.Listener(on_click=self._click, on_scroll=self._scroll,
                                        **({"win32_event_filter": lambda msg, data: not (data.flags & 1)}
                                           if sys.platform.startswith("win") else {}))
            self.mouse.start()
            self.mouse_ready = True
        except Exception:
            self.mouse_ready = False
            if self.mouse is not None:
                try:
                    self.mouse.stop()
                    if self.mouse is not threading.current_thread():
                        self.mouse.join(timeout=0.25)
                except Exception:
                    self.host.logger.warning("Could not finish closing a failed mouse observer")
            self.host.logger.warning("Live dictation unavailable: physical mouse observer could not start")
        try:
            self.monitor.start()
        except Exception:
            self.shutdown(timeout=0.5)
            raise

    def busy(self):
        return self.session is not None and not self.session.done.is_set()

    def status(self):
        state = self.session.status() if self.session else {
            "mode": "hold", "active": False, "muted": False, "editing": False,
            "pending_count": 0, "queued_count": 0, "stage": "idle", "error": None}
        state["primary_label"] = KEY_LABELS[self.gestures.primary_key]
        state["secondary_label"] = KEY_LABELS[self.gestures.secondary_key]
        return state

    def _mode(self):
        if self.busy():
            return self.session.mode
        return "hold" if self.hold_committed and self.host.recording else None

    def press(self, key):
        if self._closed:
            return
        if self.sending.is_set() and not sys.platform.startswith("win"):
            return
        self.host._observe_pynput_press(key)
        name = key_name(key)
        with self.lock:
            if self._closed or not self.host.client_enabled:
                return
            actions = self.gestures.press(name, mode=self._mode())
            if self.busy():
                self.session.set_keys_held(True)
                if name not in {self.gestures.primary_key, self.gestures.secondary_key, "esc"}:
                    self.session.physical_edit()
            # Pre-roll: capture immediately, but don't show/sound until a hold
            # is recognised. Short taps and ordinary Ctrl shortcuts are discarded.
            if (name == self.gestures.primary_key and self._mode() is None
                    and len(self.gestures.keys_down) == 1 and not self.host.is_transcription_busy()):
                self.provisional = True
                self.host._hold_gesture_tentative = True
                self.host.on_press("Key.ctrl_r")
            if self.provisional and name != self.gestures.primary_key:
                self._abort_hold()
            self._actions(actions)

    def release(self, key):
        if self._closed:
            return
        if self.sending.is_set() and not sys.platform.startswith("win"):
            return
        self.host._observe_pynput_release(key)
        name = key_name(key)
        with self.lock:
            if self._closed:
                return
            actions = self.gestures.release(name, mode=self._mode())
            if self.busy():
                self.session.set_keys_held(bool(self.gestures.keys_down))
            if name == self.gestures.primary_key and self.provisional and not self.hold_committed:
                self._abort_hold()
            self._actions(actions)

    def _abort_hold(self):
        self.provisional = False
        self.hold_committed = False
        if self.host.recording:
            # A short first tap cancels only provisional microphone capture;
            # retain the recognizer's tap window for a possible double-tap.
            self._preserve_gesture_cancel = True
            try:
                self.host.cancel_active_transcription()
            finally:
                self._preserve_gesture_cancel = False

    def _actions(self, actions):
        for action in actions:
            if action == GestureAction.START_HOLD:
                if self.host.recording:
                    self.hold_committed = True
                    self.provisional = False
                    self.host._hold_gesture_tentative = False
                    self.host._show_recording_overlay()
                    self.host.play_ui_cue(self.host.CueKind.RECORDING_STARTED)
            elif action in {GestureAction.START_HANDS_FREE, GestureAction.START_LIVE}:
                mode = "live" if action == GestureAction.START_LIVE else "hands_free"
                self.pending_start = (mode, time.monotonic() + 1.0)
            elif action == GestureAction.FINISH:
                if self.busy():
                    self.session.finish()
                elif self.hold_committed:
                    self.host.on_release("Key.ctrl_r")
                self.hold_committed = False
            elif action == GestureAction.TOGGLE_MUTE:
                if self.busy():
                    self.session.toggle_mute()
                    self.host._clear_waveform_buffers()
            elif action == GestureAction.ABORT_HOLD:
                self._abort_hold()
            elif action == GestureAction.CANCEL:
                self.cancel()
                self.host.cancel_active_transcription()

    def _click(self, x, y, button, pressed):
        if not self._closed and pressed and self.busy():
            self.session.physical_edit()

    def _scroll(self, *args):
        if not self._closed and self.busy():
            self.session.physical_edit()

    def _changed(self):
        self.host._refresh_tray_menu()
        self.host.schedule_management_refresh()

    def start_mode(self, mode):
        with self.lock:
            if self._closed:
                return False
            if mode not in {"hands_free", "live"}:
                raise ValueError("Unknown dictation mode")
            if self.busy() or self.host.is_transcription_busy():
                return False
            if self.session and self.session.pending_text():
                self.host.notify("Copy or discard the pending transcript before starting another session.")
                return False
            if not self.host.client_enabled or not self.host._client_hotkey_available():
                return False
            if mode == "live" and (not self.host.settings.get("dictation_live_enabled", False) or not self.mouse_ready):
                self.host.notify("Enable Live Dictation on the Capture page first. A working keyboard/mouse observer is required.")
                return False
            if mode == "live" and not sys.platform.startswith("win"):
                self.host.notify("Live Dictation currently requires Windows physical-input guards. Hold and hands-free remain available on X11.")
                return False
            if desktop_target() is None:
                self.host.notify("Cannot identify the active desktop target. Focus an editor and try again.")
                return False
            self.host._hold_gesture_tentative = False
            self.host._prepare_ui_session_for_recording()
            self.finished_notified = False
            self.session = DictationSession(mode, capture=self._capture, transcribe=self._transcribe,
                                            deliver=self._deliver, target=desktop_target,
                                            changed=self._changed, level=self._level)
            self.host._failed_recording.discard()
            self.host._set_pyaudio_capture_pending(True)
            try:
                self.session.start()
            except Exception:
                self.host._set_pyaudio_capture_pending(False)
                self.session = None
                self.host.transcription_ui_session.fail("microphone_failed")
                raise
            self.host._show_recording_overlay()
            self.host.play_ui_cue(self.host.CueKind.RECORDING_STARTED)
            self._changed()
            return True

    def _level(self, pcm):
        self.host._push_waveform_bytes(pcm)
        self.host.transcription_ui_session.update_level_pcm16(pcm)

    def _capture(self, stop, feed, token):
        h = self.host
        try:
            with h._managed_pyaudio(blocking=True) as pa:
                options = dict(format=h.FORMAT, channels=h.CHANNELS, rate=h.RATE,
                               input=True, frames_per_buffer=h.CHUNKSIZE)
                device = h._resolve_input_device_index(pa)
                if device is not None:
                    options["input_device_index"] = device
                stream = None
                stream_epoch = None
                try:
                    while not stop.is_set():
                        before_read = token()
                        # Close the input stream while muted. Reopen on every
                        # mute generation, including a rapid mute/unmute between
                        # loop iterations, so old device buffers cannot leak.
                        if stream is not None and (before_read[1] or stream_epoch != before_read[0]):
                            closing_stream = stream
                            stream = None
                            try:
                                closing_stream.stop_stream()
                            finally:
                                closing_stream.close()
                        if before_read[1]:
                            stop.wait(0.02)
                            continue
                        if stream is None:
                            try:
                                stream = pa.open(**options)
                            except Exception:
                                if "input_device_index" not in options:
                                    raise
                                options.pop("input_device_index")
                                stream = pa.open(**options)
                            stream_epoch = before_read[0]
                        pcm = stream.read(h.CHUNKSIZE)
                        feed(pcm, before_read)
                finally:
                    if stream is not None:
                        try:
                            stream.stop_stream()
                        finally:
                            stream.close()
        finally:
            h._set_pyaudio_capture_pending(False)

    def _transcribe(self, pcm, cancel):
        from utils.models import transcribe_audio_result
        h = self.host
        path = h.create_recording_file_path()
        try:
            with wave.open(str(path), "wb") as audio:
                audio.setnchannels(h.CHANNELS)
                audio.setsampwidth(2)
                audio.setframerate(h.RATE)
                audio.writeframes(pcm)
            with request_control(cancel):
                result = transcribe_audio_result(str(path), play_feedback=False)
            if result is None:
                raise RuntimeError("No transcription result")
            return result
        except Exception as exc:
            if getattr(exc, "retryable", False) and not cancel.is_set():
                h._failed_recording.retain(path)
            h.logger.warning("Continuous transcription failed: %s", h._safe_error_category(exc))
            raise
        finally:
            h.cleanup_recording_file(path)

    def _deliver(self, result, guard):
        self.sending.set()
        try:
            with request_control(guard):
                self.host.inject_transcription_result(result)
        finally:
            self.sending.clear()

    def cancel(self):
        with self.lock:
            self.pending_start = None
            if not self._preserve_gesture_cancel:
                self.gestures.reset()
                self.hold_committed = False
                self.provisional = False
            if self.busy():
                self.session.abort()
                return True
            return False

    def shutdown(self, timeout=2.5):
        deadline = time.monotonic() + max(0.0, float(timeout))
        self._closed = True
        self.stop.set()
        self.cancel()
        self.mouse_ready = False
        if self.mouse is not None:
            try:
                self.mouse.stop()
            except Exception:
                self.host.logger.warning("Could not stop the physical mouse observer")
        session_done = True
        if self.session:
            session_done = self.session.done.wait(max(0.0, deadline - time.monotonic()))
        threads = [self.monitor, self.mouse]
        if self.session:
            threads.extend((self.session.reader, self.session.worker))
        remaining_owners = False
        for thread in threads:
            if thread is None:
                continue
            if thread is threading.current_thread():
                remaining_owners = True
                continue
            try:
                if thread.is_alive():
                    thread.join(timeout=max(0.0, deadline - time.monotonic()))
                remaining_owners = remaining_owners or thread.is_alive()
            except RuntimeError:
                # A constructed but unstarted Thread has no live ownership.
                remaining_owners = remaining_owners or thread.is_alive()
            except Exception:
                remaining_owners = True
                self.host.logger.warning("A dictation worker could not be confirmed stopped")
        complete = session_done and not remaining_owners
        if not complete:
            self.host.logger.warning("Continuous dictation is still releasing its workers")
        return complete

    def _monitor(self):
        last = time.monotonic()
        next_target_poll = last
        while not self.stop.wait(0.02):
            try:
                now = time.monotonic()
                with self.lock:
                    if now - last > 5:
                        self.cancel()
                        self.gestures.reset()
                    last = now
                    self._actions(self.gestures.tick(mode=self._mode()))
                    if self.pending_start:
                        mode, deadline = self.pending_start
                        if not self.host.is_transcription_busy() and not self.gestures.keys_down:
                            self.pending_start = None
                            self.start_mode(mode)
                        elif now > deadline:
                            self.pending_start = None
                            self.host.notify("Microphone is still closing; try the dictation shortcut again.")
                    session = self.session
                    if session and not session.done.is_set():
                        if now >= next_target_poll and not session.cancel.is_set():
                            next_target_poll = now + TARGET_POLL_SECONDS
                            # Gesture recognition still ticks at 20 ms; desktop
                            # identity is a more expensive, bounded 10 Hz poll.
                            # Delivery independently rechecks its target per batch.
                            keyboard_observer = getattr(self.host, "listener", None)
                            if keyboard_observer is not None and not keyboard_observer.is_alive():
                                self.mouse_ready = False
                                session.abort()
                                self.host.notify("Dictation stopped because its keyboard observer disconnected. Restart CtrlSpeak before dictating again.")
                            elif session.mode == "live" and self.mouse is not None and not self.mouse.is_alive():
                                self.mouse_ready = False
                                session.abort()
                                self.host.notify("Live Dictation stopped because its physical mouse observer disconnected. Restart CtrlSpeak before using Live Dictation again.")
                            elif desktop_target() is None:
                                session.abort()
                            else:
                                session.check_target()
                    elif session and not self.finished_notified:
                        self.finished_notified = True
                        ui = self.host.transcription_ui_session
                        if session.cancel.is_set():
                            ui.cancel()
                        elif session.error:
                            ui.fail("unexpected_error")
                        else:
                            if ui.phase == self.host.UiPhase.RECORDING:
                                ui.begin_processing()
                            ui.complete(session.last_metadata, elapsed_ms=(now - session.started) * 1000)
                        if session.committed:
                            self.host.remember_last_transcript("".join(session.committed))
                        self.host._set_terminal_overlay_hide(4000 if session.pending else 1600)
                        self._changed()
            except Exception:
                self.host.logger.exception("Dictation controls failed; stopping active capture")
                self.cancel()
                self.gestures.reset()
