"""Correction-dialog geometry and asynchronous lifecycle regression tests."""
from __future__ import annotations

import sys
import subprocess
import threading
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import tkinter as tk


if not hasattr(tk, "messagebox"):
    tk.messagebox = types.SimpleNamespace()  # type: ignore[attr-defined]
    sys.modules.setdefault("tkinter.messagebox", tk.messagebox)  # type: ignore[arg-type, attr-defined]

from utils import gui
from utils.transcription_backend import BackendConfig


class _ThreadGuard:
    def __init__(self) -> None:
        self.owner = threading.get_ident()

    def check(self) -> None:
        assert threading.get_ident() == self.owner, "worker thread touched Tk state"


class _GuardedVar(_ThreadGuard):
    def __init__(self, value: object = "") -> None:
        super().__init__()
        self.value = value

    def get(self):
        self.check()
        return self.value

    def set(self, value: object) -> None:
        self.check()
        self.value = value


class _GuardedWidget(_ThreadGuard):
    def __init__(self) -> None:
        super().__init__()
        self.disabled = False
        self.mapped = False
        self.focus_calls = 0
        self.options: dict[str, object] = {}

    def state(self, specifications) -> None:
        self.check()
        for specification in specifications:
            if specification == "disabled":
                self.disabled = True
            elif specification == "!disabled":
                self.disabled = False

    def instate(self, specifications) -> bool:
        self.check()
        return all(
            self.disabled if specification == "disabled" else not self.disabled
            for specification in specifications
        )

    def configure(self, **options) -> None:
        self.check()
        self.options.update(options)

    def focus_set(self) -> None:
        self.check()
        self.focus_calls += 1

    def winfo_ismapped(self) -> bool:
        self.check()
        return self.mapped

    def pack(self, **_options) -> None:
        self.check()
        self.mapped = True

    def pack_forget(self) -> None:
        self.check()
        self.mapped = False

    def start(self, _interval: int) -> None:
        self.check()

    def stop(self) -> None:
        self.check()


class _GuardedWindow(_ThreadGuard):
    def __init__(self) -> None:
        super().__init__()
        self.exists = True
        self.window_state = "normal"
        self.bells = 0
        self.cursor = ""
        self.lifts = 0
        self.forced_focus = 0

    def winfo_exists(self) -> bool:
        self.check()
        return self.exists

    def state(self) -> str:
        self.check()
        return self.window_state

    def withdraw(self) -> None:
        self.check()
        self.window_state = "withdrawn"

    def deiconify(self) -> None:
        self.check()
        self.window_state = "normal"

    def lift(self) -> None:
        self.check()
        self.lifts += 1

    def focus_force(self) -> None:
        self.check()
        self.forced_focus += 1

    def configure(self, **options) -> None:
        self.check()
        self.cursor = str(options.get("cursor", self.cursor))

    def bell(self) -> None:
        self.check()
        self.bells += 1


def _headless_dialog() -> gui.CorrectionSubmissionDialog:
    dialog = object.__new__(gui.CorrectionSubmissionDialog)
    dialog._config = BackendConfig(
        backend="api",
        api_url="https://gateway.example.test",
        api_token=None,
        feedback_capture_method="disabled",
    )
    dialog._pending_config = None
    dialog._submitting = False
    dialog._compact_layout = False
    dialog._submission_generation = 0
    dialog._status_kind = "ready"
    dialog.window = _GuardedWindow()
    dialog.source_var = _GuardedVar("control speak")
    dialog.replacement_var = _GuardedVar("CtrlSpeak")
    dialog.global_scope_var = _GuardedVar(False)
    dialog.status_var = _GuardedVar("")
    dialog.gateway_var = _GuardedVar("https://gateway.example.test")
    dialog.source_entry = _GuardedWidget()
    dialog.replacement_entry = _GuardedWidget()
    dialog.global_scope_check = _GuardedWidget()
    dialog.submit_button = _GuardedWidget()
    dialog._status_label = _GuardedWidget()
    dialog._progress = _GuardedWidget()
    dialog._form_controls = (
        dialog.source_entry,
        dialog.replacement_entry,
        dialog.global_scope_check,
    )
    return dialog


@pytest.mark.core_headless
@pytest.mark.parametrize(
    ("screen_width", "screen_height", "expected"),
    [
        (640, 480, (52, 40, 536, 400, 536, 400)),
        (800, 600, (52, 40, 696, 520, 696, 520)),
    ],
)
def test_correction_dialog_geometry_remains_inside_constrained_200_percent_work_area(
    screen_width: int,
    screen_height: int,
    expected: tuple[int, int, int, int, int, int],
) -> None:
    bounds = SimpleNamespace(
        left=0,
        top=0,
        width=screen_width,
        height=screen_height,
    )

    geometry = gui.correction_dialog_geometry(bounds, 2.0)

    assert geometry == expected
    x, y, width, height, minimum_width, minimum_height = geometry
    assert 0 <= x <= screen_width - width
    assert 0 <= y <= screen_height - height
    assert minimum_width <= width
    assert minimum_height <= height


@pytest.mark.core_headless
def test_negative_monitor_position_is_converted_to_tk_edge_offsets() -> None:
    spec = gui.correction_dialog_geometry_spec(
        -1800,
        -900,
        500,
        400,
        reference_right=2560,
        reference_bottom=1440,
    )

    assert spec == "500x400-3860-1940"
    # Tk resolves the two negative offsets from the far edges.  Reconstructing
    # the absolute coordinates must return the requested left/above monitor.
    assert 2560 - 500 - 3860 == -1800
    assert 1440 - 400 - 1940 == -900


@pytest.mark.core_headless
def test_submit_is_single_flight_worker_only_enqueues_tk_completion_and_hidden_result_reopens(
    monkeypatch,
) -> None:
    dialog = _headless_dialog()
    main_thread = threading.get_ident()
    completion_ready = threading.Event()
    queued: list[tuple[object, tuple[object, ...], int]] = []
    calls: list[tuple[str, str, str]] = []

    class Client:
        def __init__(self, config, *, timeout_seconds) -> None:
            assert threading.get_ident() != main_thread
            assert config.api_url == "https://gateway.example.test"
            assert timeout_seconds == 20.0

        def create_correction(self, source, replacement, *, scope):
            assert threading.get_ident() != main_thread
            calls.append((source, replacement, scope))
            return {"id": "rule-123"}

    def enqueue(callback, *args) -> None:
        queued.append((callback, args, threading.get_ident()))
        completion_ready.set()

    monkeypatch.setattr(gui, "ApiTranscriptionClient", Client)
    monkeypatch.setattr(gui, "enqueue_management_task", enqueue)

    dialog.submit()
    dialog.submit()
    assert completion_ready.wait(2.0)
    assert calls == [("control speak", "CtrlSpeak", "user")]
    assert len(queued) == 1
    assert queued[0][2] != main_thread

    new_config = BackendConfig(
        backend="api",
        api_url="https://new-gateway.example.test",
        api_token=None,
        feedback_capture_method="disabled",
    )
    dialog.set_config(new_config)
    assert dialog._config.api_url == "https://gateway.example.test"
    assert dialog.gateway_var.get() == "https://gateway.example.test"

    dialog.hide()
    assert dialog.window.state() == "withdrawn"
    callback, args, _worker_thread = queued.pop()
    callback(*args)

    assert dialog._submitting is False
    assert dialog._status_kind == "success"
    assert dialog.submit_button.instate(["!disabled"])
    assert dialog.source_var.get() == ""
    assert dialog.replacement_var.get() == ""
    assert dialog.source_entry.focus_calls == 0
    assert dialog._config == new_config
    assert dialog.gateway_var.get() == "https://new-gateway.example.test"

    dialog.bring_to_front()
    assert dialog.window.state() == "normal"
    assert dialog.is_visible()
    assert dialog.window.lifts == 1
    assert dialog.window.forced_focus == 1


@pytest.mark.core_headless
def test_thread_start_failure_restores_form_and_exposes_bounded_inline_error(
    monkeypatch,
) -> None:
    dialog = _headless_dialog()
    monkeypatch.setattr(
        gui,
        "logger",
        SimpleNamespace(error=lambda *_args, **_kwargs: None, warning=lambda *_args, **_kwargs: None),
    )

    class BrokenThread:
        def __init__(self, **_kwargs) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("worker could not start " + ("x" * 400))

    monkeypatch.setattr(gui.threading, "Thread", BrokenThread)

    dialog.submit()

    assert dialog._submitting is False
    assert dialog._status_kind == "error"
    assert dialog.submit_button.instate(["!disabled"])
    assert dialog.source_var.get() == "control speak"
    assert dialog.replacement_var.get() == "CtrlSpeak"
    assert len(str(dialog.status_var.get())) <= 75
    assert dialog.window.bells == 1
    assert dialog.replacement_entry.focus_calls == 1


@pytest.mark.core_headless
def test_latest_runtime_config_wins_when_settings_change_back_during_submission() -> None:
    dialog = _headless_dialog()
    original_config = dialog._config
    temporary_config = BackendConfig(
        backend="api",
        api_url="https://temporary-gateway.example.test",
        api_token=None,
        feedback_capture_method="disabled",
    )
    dialog._submitting = True

    dialog.set_config(temporary_config)
    assert dialog._pending_config == temporary_config

    dialog.set_config(original_config)

    assert dialog._pending_config is None
    assert dialog._config == original_config
    assert dialog.gateway_var.get() == "https://gateway.example.test"


@pytest.mark.core_headless
def test_gateway_error_never_displays_or_logs_submitted_phrases_or_tokens(
    monkeypatch,
) -> None:
    dialog = _headless_dialog()
    secret_source = "private source phrase"
    secret_replacement = "private replacement phrase"
    secret_token = "secret-bearer-token"
    dialog.source_var.set(secret_source)
    dialog.replacement_var.set(secret_replacement)
    completion_ready = threading.Event()
    queued: list[tuple[object, tuple[object, ...]]] = []
    recorded_logs: list[str] = []

    class RecordingLogger:
        @staticmethod
        def warning(message, *args, **_kwargs) -> None:
            recorded_logs.append(str(message) % args if args else str(message))

    class Client:
        def __init__(self, _config, *, timeout_seconds) -> None:
            assert timeout_seconds == 20.0

        def create_correction(self, _source, _replacement, *, scope):
            assert scope == "user"
            raise gui.ApiBackendError(
                "HTTP 422 validation input="
                f"{secret_source}|{secret_replacement}|{secret_token}"
            )

    def enqueue(callback, *args) -> None:
        queued.append((callback, args))
        completion_ready.set()

    monkeypatch.setattr(gui, "ApiTranscriptionClient", Client)
    monkeypatch.setattr(gui, "enqueue_management_task", enqueue)
    monkeypatch.setattr(gui, "logger", RecordingLogger())

    dialog.submit()
    assert completion_ready.wait(2.0)
    callback, args = queued.pop()
    callback(*args)

    visible_and_logged = f"{dialog.status_var.get()}\n{' '.join(recorded_logs)}"
    assert "Gateway rejected" in visible_and_logged
    assert secret_source not in visible_and_logged
    assert secret_replacement not in visible_and_logged
    assert secret_token not in visible_and_logged


@pytest.mark.core_headless
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            "Whisper API returned HTTP 500: user input says bearer timeout validation duplicate",
            "The gateway could not save this correction. See the log for its error category.",
        ),
        (
            "Whisper API returned HTTP 422: user input says bearer token",
            "Gateway rejected the correction. Shorten both phrases and try again.",
        ),
        (
            "Whisper API returned HTTP 401: user input says validation",
            "Gateway authentication failed. Check the saved gateway token.",
        ),
    ],
)
def test_error_classification_uses_transport_status_not_untrusted_detail(
    error: str,
    expected: str,
) -> None:
    assert gui.CorrectionSubmissionDialog._safe_error_summary(error) == expected


@pytest.mark.core_headless
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ("HTTP 422: long input", "Not saved · Shorten both phrases"),
        ("HTTP 401: denied", "Not saved · Check gateway token"),
        ("could not reach gateway", "Not saved · Gateway unreachable"),
        ("HTTP 500: failure", "Not saved · See CtrlSpeak log"),
    ],
)
def test_compact_error_copy_remains_actionable_within_status_budget(
    error: str,
    expected: str,
) -> None:
    dialog = object.__new__(gui.CorrectionSubmissionDialog)
    dialog._compact_layout = True

    rendered = dialog._error_status_text(error)

    assert rendered == expected
    assert len(rendered) <= 42


@pytest.mark.full_gui
@pytest.mark.parametrize("screen_width,screen_height", [(640, 480), (800, 600)])
def test_real_tk_pinned_actions_stay_mapped_with_long_error_at_200_percent(
    screen_width: int,
    screen_height: int,
) -> None:
    if not sys.platform.startswith("win"):
        pytest.skip("native Windows Tk geometry check")
    script = textwrap.dedent(
        f"""
        import tkinter as tk
        from types import SimpleNamespace
        from utils import gui, midnight_overlay
        from utils.transcription_backend import BackendConfig

        root = tk.Tk()
        root.withdraw()
        root.tk.call("tk", "scaling", 96 * 2 / 72)
        gui.tk_root = root
        gui.management_window = None
        gui._set_window_icon = lambda _window: None
        midnight_overlay.display_scale = lambda _window: 2.0
        midnight_overlay.active_monitor_bounds = lambda _window: SimpleNamespace(
            left=0,
            top=0,
            width={screen_width},
            height={screen_height},
        )

        dialog = gui.CorrectionSubmissionDialog(
            None,
            BackendConfig(
                backend="api",
                api_url="https://gateway.example.test/" + ("long-path/" * 20),
                api_token=None,
                feedback_capture_method="disabled",
            ),
        )
        dialog._set_status("error", "Gateway failure " + ("long detail " * 100))
        root.update()
        window_width = dialog.window.winfo_width()
        window_height = dialog.window.winfo_height()
        for widget in (dialog._status_label, dialog.hide_button, dialog.submit_button):
            x = widget.winfo_rootx() - dialog.window.winfo_rootx()
            y = widget.winfo_rooty() - dialog.window.winfo_rooty()
            assert widget.winfo_ismapped()
            assert 0 <= x < window_width
            assert 0 <= y < window_height
            assert x + widget.winfo_width() <= window_width
            assert y + widget.winfo_height() <= window_height

        canvas = dialog._body_canvas
        assert canvas.winfo_ismapped()
        assert canvas.winfo_height() >= 80
        content_bounds = canvas.bbox("all")
        assert content_bounds and content_bounds[3] > canvas.winfo_height()
        body = dialog.source_entry.master.master
        content_height = content_bounds[3] - content_bounds[1]
        for entry in (dialog.source_entry, dialog.replacement_entry):
            entry_y = entry.winfo_rooty() - body.winfo_rooty()
            fraction = max(0.0, min(1.0, (entry_y - 8) / content_height))
            canvas.yview_moveto(fraction)
            root.update()
            visible_y = entry.winfo_rooty() - canvas.winfo_rooty()
            assert visible_y < canvas.winfo_height()
            assert visible_y + entry.winfo_height() > 0

        dialog.window.destroy()
        root.destroy()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.full_gui
def test_native_windows_negative_coordinate_placement_is_absolute() -> None:
    if not sys.platform.startswith("win"):
        pytest.skip("native Windows Tk geometry check")
    script = textwrap.dedent(
        """
        import tkinter as tk
        from utils.gui import (
            _place_windows_toplevel_absolute,
            _tk_geometry_reference_edges,
            correction_dialog_geometry_spec,
        )

        root = tk.Tk()
        root.withdraw()
        window = tk.Toplevel(root)
        window.attributes("-alpha", 0.0)
        reference_right, reference_bottom = _tk_geometry_reference_edges(window)
        window.geometry(
            correction_dialog_geometry_spec(
                -100,
                100,
                300,
                200,
                reference_right=reference_right,
                reference_bottom=reference_bottom,
            )
        )
        root.update()
        assert _place_windows_toplevel_absolute(window, -100, 100)
        root.update()
        assert window.winfo_x() == -100
        assert window.winfo_y() == 100
        window.destroy()
        root.destroy()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
