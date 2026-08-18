"""Deterministic lifecycle coverage for CtrlSpeak's tray-owned windows.

These tests deliberately use small Tk stand-ins.  They exercise the real
window/controller lifecycle without opening a desktop window or depending on
focus timing in the host operating system.
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import tkinter as tk


# The core-headless suite provides only the tkinter symbols needed by most
# modules.  CtrlSpeak's GUI imports messagebox at collection time, so add the
# inert namespace needed to collect these lifecycle-only tests.
if not hasattr(tk, "messagebox"):
    tk.messagebox = types.SimpleNamespace()  # type: ignore[attr-defined]
    sys.modules.setdefault("tkinter.messagebox", tk.messagebox)  # type: ignore[arg-type, attr-defined]

from utils import gui
from utils import midnight_signal_ui as midnight_ui


pytestmark = pytest.mark.core_headless


class _FakeVariable:
    def __init__(self, value=None, **_kwargs) -> None:
        self.value = value

    def get(self):
        return self.value

    def set(self, value) -> None:
        self.value = value


class _FakeWidget:
    def __init__(self, master=None, *args, **kwargs) -> None:
        del args
        self.master = master
        self.options = dict(kwargs)
        self.exists = True
        self.destroyed = False
        self.bindings: dict[str, object] = {}
        self.cancelled_jobs: list[str] = []
        self.scheduled: list[tuple[int, object, str]] = []
        self.focused = None
        self._job_number = 0

    def pack(self, *args, **kwargs) -> None:
        del args, kwargs

    def pack_forget(self) -> None:
        pass

    def grid(self, *args, **kwargs) -> None:
        del args, kwargs

    def rowconfigure(self, *args, **kwargs) -> None:
        del args, kwargs

    def columnconfigure(self, *args, **kwargs) -> None:
        del args, kwargs

    def configure(self, *args, **kwargs) -> None:
        del args, kwargs

    config = configure

    def bind(self, sequence, callback, add=None) -> None:
        del add
        self.bindings[sequence] = callback

    def create_window(self, *args, **kwargs) -> int:
        del args, kwargs
        return 1

    def create_oval(self, *args, **kwargs) -> int:
        del args, kwargs
        return 1

    def create_text(self, *args, **kwargs) -> int:
        del args, kwargs
        return 1

    def delete(self, *args, **kwargs) -> None:
        del args, kwargs

    def itemconfigure(self, *args, **kwargs) -> None:
        del args, kwargs

    def bbox(self, *args, **kwargs):
        del args, kwargs
        return (0, 0, 320, 480)

    def yview(self, *args, **kwargs) -> None:
        del args, kwargs

    def yview_scroll(self, *args, **kwargs) -> None:
        del args, kwargs

    def state(self, *args, **kwargs) -> None:
        del args, kwargs

    def set(self, *args, **kwargs) -> None:
        del args, kwargs

    def after(self, delay, callback) -> str:
        self._job_number += 1
        job = f"job-{self._job_number}"
        self.scheduled.append((delay, callback, job))
        return job

    def after_cancel(self, job: str) -> None:
        self.cancelled_jobs.append(job)

    def focus_get(self):
        return self.focused

    def focus_force(self) -> None:
        pass

    def lift(self) -> None:
        pass

    def winfo_exists(self) -> bool:
        return self.exists

    def destroy(self) -> None:
        self.exists = False
        self.destroyed = True


class _FakeWindow(_FakeWidget):
    def __init__(self, master=None, *args, **kwargs) -> None:
        super().__init__(master, *args, **kwargs)
        self.protocols: dict[str, object] = {}

    def title(self, _value: str) -> None:
        pass

    def overrideredirect(self, _value: bool) -> None:
        pass

    def attributes(self, *args) -> None:
        del args

    def protocol(self, name: str, callback) -> None:
        self.protocols[name] = callback

    def geometry(self, _value: str) -> None:
        pass


def _make_headless_flyout(monkeypatch):
    """Build the real flyout against deterministic, no-display widgets."""

    buttons: list[_FakeWidget] = []

    def make_button(master=None, *args, **kwargs):
        button = _FakeWidget(master, *args, **kwargs)
        buttons.append(button)
        return button

    for widget_name in ("Frame", "Label", "Progressbar", "Scrollbar"):
        monkeypatch.setattr(midnight_ui.ttk, widget_name, _FakeWidget, raising=False)
    monkeypatch.setattr(midnight_ui.ttk, "Button", make_button, raising=False)
    monkeypatch.setattr(midnight_ui.tk, "Toplevel", _FakeWindow, raising=False)
    monkeypatch.setattr(midnight_ui.tk, "Canvas", _FakeWidget, raising=False)
    monkeypatch.setattr(midnight_ui.tk, "StringVar", _FakeVariable, raising=False)
    monkeypatch.setattr(midnight_ui.tk, "DoubleVar", _FakeVariable, raising=False)
    for constant in ("LEFT", "RIGHT", "X", "Y", "W", "E", "VERTICAL"):
        monkeypatch.setattr(midnight_ui.tk, constant, constant.lower(), raising=False)
    monkeypatch.setattr(midnight_ui, "apply_midnight_signal_theme", lambda _window: None)
    monkeypatch.setattr(midnight_ui, "display_scale", lambda _window: 1.0)
    monkeypatch.setattr(
        midnight_ui,
        "active_monitor_bounds",
        lambda _window: SimpleNamespace(left=0, top=0, width=1280, height=800),
    )
    monkeypatch.setattr(
        midnight_ui,
        "flyout_geometry",
        lambda _bounds, _scale: (820, 160, 420, 600, False),
    )
    monkeypatch.setattr(midnight_ui.sysmod, "get_input_device_preference", lambda: None)

    quit_calls: list[bool] = []
    cancel_calls: list[bool] = []
    monkeypatch.setattr(
        midnight_ui.sysmod,
        "cancel_active_transcription",
        lambda: cancel_calls.append(True),
    )

    flyout = midnight_ui.MidnightTrayFlyout(
        _FakeWidget(),
        object(),
        open_control=lambda: None,
        open_corrections=lambda: None,
        check_updates=lambda: None,
        quit_app=lambda: quit_calls.append(True),
    )
    # Keep this test about window ownership, not telemetry or gateway workers.
    flyout._poll = lambda: None
    flyout._refresh_capabilities = lambda: None
    flyout._apply_route_strategy_visuals = lambda: None
    return flyout, buttons, quit_calls, cancel_calls


def test_quick_panel_escape_close_protocol_and_toggle_all_hide_without_exit(
    monkeypatch,
) -> None:
    flyout, buttons, quit_calls, cancel_calls = _make_headless_flyout(monkeypatch)

    flyout.toggle()
    first_window = flyout.window
    assert isinstance(first_window, _FakeWindow)
    hide_button = next(
        button for button in buttons if button.options.get("text") == "Hide panel"
    )
    hide_button.options["command"]()
    assert first_window.destroyed is True
    assert flyout.window is None

    flyout.toggle()
    second_window = flyout.window
    assert isinstance(second_window, _FakeWindow)
    assert second_window is not first_window
    second_window.bindings["<Escape>"](None)
    assert second_window.destroyed is True
    assert flyout.window is None

    flyout.toggle()
    third_window = flyout.window
    assert isinstance(third_window, _FakeWindow)
    third_window.protocols["WM_DELETE_WINDOW"]()
    assert third_window.destroyed is True
    assert flyout.window is None

    flyout.toggle()
    fourth_window = flyout.window
    assert isinstance(fourth_window, _FakeWindow)
    flyout.toggle()
    assert fourth_window.destroyed is True
    assert flyout.window is None

    assert quit_calls == []
    assert cancel_calls == []


def test_control_centre_reuses_live_controller_and_replaces_closed_one(
    monkeypatch,
) -> None:
    created: list[object] = []

    class FakeManagementController:
        def __init__(self, icon) -> None:
            self.icon = icon
            self.open = True
            self.front_calls = 0
            self.refresh_calls = 0
            created.append(self)

        def is_open(self) -> bool:
            return self.open

        def bring_to_front(self) -> None:
            self.front_calls += 1

        def refresh_status(self) -> None:
            self.refresh_calls += 1

    monkeypatch.setattr(gui, "ManagementWindow", FakeManagementController)
    monkeypatch.setattr(gui, "management_window", None)
    icon = object()

    gui._show_management_window(icon)
    first = gui.management_window
    gui._show_management_window(icon)

    assert len(created) == 1
    assert gui.management_window is first
    assert first.front_calls == 1
    assert first.refresh_calls == 1

    # Even if a toolkit close leaves a stale reference briefly, reopening must
    # replace it instead of treating the destroyed controller as live.
    first.open = False
    gui._show_management_window(icon)
    assert len(created) == 2
    assert gui.management_window is created[1]
    assert gui.management_window is not first


def test_management_close_and_ui_teardown_reset_ownership_without_quitting(
    monkeypatch,
) -> None:
    quit_calls: list[bool] = []
    cancel_calls: list[bool] = []
    monkeypatch.setattr(
        midnight_ui.sysmod,
        "cancel_active_transcription",
        lambda: cancel_calls.append(True),
    )

    visible = _FakeWindow()
    legacy = _FakeWindow()
    controller = object.__new__(midnight_ui.MidnightSignalManagementMixin)
    controller._icon = SimpleNamespace(stop=lambda: quit_calls.append(True))
    controller.window = visible
    controller._legacy_window = legacy
    controller._capability_generation = 0
    controller._corrections_generation = 0
    controller._correction_mutation_generation = 0
    controller._correction_mutations_pending = 1
    controller._correction_mutation_errors = ["stale"]
    controller._correction_refresh_notice = "stale"
    controller._update_coordinator = None
    controller._shell_poll_job = None
    monkeypatch.setattr(gui, "management_window", controller)

    controller.close()

    assert visible.destroyed is True
    assert legacy.destroyed is True
    assert gui.management_window is None
    assert quit_calls == []
    assert cancel_calls == []

    # Teardown must destroy a currently visible quick panel without quitting.
    root = _FakeWindow()
    panel_window = _FakeWindow()
    flyout = object.__new__(midnight_ui.MidnightTrayFlyout)
    flyout.window = panel_window
    flyout._poll_job = None
    flyout._capability_generation = 0
    monkeypatch.setattr(gui, "tk_root", root)
    monkeypatch.setattr(gui, "tray_flyout", flyout)
    monkeypatch.setattr(gui, "management_window", object())
    monkeypatch.setattr(gui, "_management_queue_job", "queue-job")
    monkeypatch.setattr(gui, "_management_thread_ident", 123)
    gui._management_thread_ready.set()
    monkeypatch.setattr(gui.sysmod, "_tray_icon", None)

    gui._teardown_management_ui()

    assert panel_window.destroyed is True
    assert root.destroyed is True
    assert root.cancelled_jobs == ["queue-job"]
    assert gui.tk_root is None
    assert gui.management_window is None
    assert gui.tray_flyout is None
    assert gui._management_thread_ident is None
    assert gui._management_thread_ready.is_set() is False
    assert quit_calls == []
    assert cancel_calls == []
