from __future__ import annotations

import sys
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.core_headless


class FakeTkClipboardRoot:
    """Small X11 clipboard double that records Tk ownership lifecycle calls."""

    def __init__(self, text: str, events: list[tuple[str, object]]) -> None:
        self.text = text
        self.events = events
        self.tk = SimpleNamespace(call=self._tk_call, splitlist=self._splitlist)
        self.events.append(("create", None))

    def _tk_call(self, *args: object) -> tuple[str, ...]:
        self.events.append(("tk_call", args))
        if args[-1] == "TARGETS":
            return ("TARGETS", "UTF8_STRING", "STRING", "text/plain")
        raise AssertionError(f"Unexpected Tk call: {args!r}")

    @staticmethod
    def _splitlist(value: object) -> tuple[str, ...]:
        return tuple(value) if isinstance(value, tuple) else tuple(str(value).split())

    def withdraw(self) -> None:
        self.events.append(("withdraw", None))

    def clipboard_get(self, **_kwargs: object) -> str:
        self.events.append(("get", self.text))
        return self.text

    def clipboard_clear(self) -> None:
        self.events.append(("clear", self.text))
        self.text = ""

    def clipboard_append(self, text: str) -> None:
        self.events.append(("append", text))
        self.text += text

    def update_idletasks(self) -> None:
        self.events.append(("update_idletasks", None))

    def update(self) -> None:
        self.events.append(("update", None))

    def destroy(self) -> None:
        self.events.append(("destroy", None))


def test_input_router_selects_linux_without_importing_windows_apis() -> None:
    from utils import winio

    sys.modules.pop("utils.windows_input", None)

    assert winio.platform_adapter_module_name("linux") == "utils.linux_input"
    assert winio.platform_adapter_module_name("win32") == "utils.windows_input"
    assert "utils.windows_input" not in sys.modules


def test_windows_adapter_remains_statically_discoverable_but_linux_excludes_it() -> None:
    root = Path(__file__).resolve().parents[2]
    router = (root / "utils" / "winio.py").read_text("utf-8")
    linux_spec = (root / "packaging" / "CtrlSpeak_v0.4.spec").read_text("utf-8")

    assert "from utils import windows_input" in router
    assert "if sys.platform.startswith(\"win\")" in router
    assert "'utils.windows_input'" in linux_spec


def test_system_import_defers_display_bound_tray_backend(tmp_path) -> None:
    environment = dict(os.environ)
    environment.update(
        DISPLAY="",
        XDG_SESSION_TYPE="",
        XDG_CONFIG_HOME=str(tmp_path / "config"),
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, types; "
                "p=types.ModuleType('pyaudio'); p.paInt16=8; "
                "p.PyAudio=type('PyAudio', (), {}); sys.modules['pyaudio']=p; "
                "pil=types.ModuleType('PIL'); image=types.ModuleType('PIL.Image'); "
                "pil.Image=image; sys.modules['PIL']=pil; sys.modules['PIL.Image']=image; "
                "import utils.system; assert 'pystray' not in sys.modules"
            ),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_x11_session_is_supported_and_wayland_is_actionable() -> None:
    from utils.hotkeys import (
        DesktopSessionError,
        ensure_desktop_automation_supported,
        get_desktop_session_status,
    )

    x11 = get_desktop_session_status(
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":1"},
        platform_name="linux",
    )
    assert x11.supported is True
    assert x11.session_type == "x11"

    wayland_environment = {
        "XDG_SESSION_TYPE": "wayland",
        "WAYLAND_DISPLAY": "wayland-0",
        "DISPLAY": ":0",
    }
    wayland = get_desktop_session_status(
        environ=wayland_environment,
        platform_name="linux",
    )
    assert wayland.supported is False
    assert wayland.session_type == "wayland"
    assert "global hotkeys" in wayland.detail

    with pytest.raises(DesktopSessionError, match="Ubuntu on Xorg"):
        ensure_desktop_automation_supported(
            environ=wayland_environment,
            platform_name="linux",
        )


def test_linux_listener_is_observer_only() -> None:
    from utils.hotkeys import create_global_listener

    listener_kwargs: dict[str, object] = {}
    fake_listener = object()

    def listener_factory(**kwargs):
        listener_kwargs.update(kwargs)
        return fake_listener

    on_press = lambda key: key
    on_release = lambda key: key
    listener = create_global_listener(
        on_press=on_press,
        on_release=on_release,
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":1"},
        platform_name="linux",
        listener_factory=listener_factory,
    )

    assert listener is fake_listener
    assert listener_kwargs == {
        "on_press": on_press,
        "on_release": on_release,
        "suppress": False,
    }


def test_linux_injection_pastes_once_and_restores_text_clipboard(monkeypatch) -> None:
    from utils import linux_input

    actions: list[tuple[str, object]] = []
    automation = SimpleNamespace(
        hotkey=lambda *keys: actions.append(("hotkey", keys)),
        write=lambda text, **_kwargs: actions.append(("write", text)),
    )
    monkeypatch.setattr(linux_input, "ensure_desktop_automation_supported", lambda: None)
    monkeypatch.setattr(linux_input, "clipboard_tool_available", lambda: True)
    monkeypatch.setattr(linux_input, "clipboard_contains_non_text_data", lambda: False)
    monkeypatch.setattr(linux_input, "get_clipboard_text", lambda: "keep me")
    monkeypatch.setattr(
        linux_input,
        "set_clipboard_text",
        lambda text: actions.append(("set", text)) or True,
    )
    monkeypatch.setattr(
        linux_input,
        "restore_clipboard_text",
        lambda text: actions.append(("restore", text)),
    )
    monkeypatch.setattr(linux_input, "get_pyautogui", lambda: automation)
    monkeypatch.setattr(linux_input.time, "sleep", lambda _seconds: None)

    linux_input.insert_text_into_focus("dictated words")

    assert actions == [
        ("set", "dictated words"),
        ("hotkey", ("ctrl", "v")),
        ("restore", "keep me"),
    ]
    assert all(action != ("hotkey", ("enter",)) for action in actions)


def test_linux_injection_uses_hidden_tk_fallback_for_unicode_and_multiline(
    monkeypatch,
) -> None:
    from utils import linux_input

    previous = "Żółty schowek\n第二行"
    transcript = "Dzień dobry 🌍\nDruga linia"
    events: list[tuple[str, object]] = []
    root = FakeTkClipboardRoot(previous, events)
    pasted: list[str] = []

    def hotkey(*keys: str) -> None:
        events.append(("hotkey", keys))
        if keys == ("ctrl", "v"):
            pasted.append(root.text)

    monkeypatch.setattr(linux_input, "ensure_desktop_automation_supported", lambda: None)
    monkeypatch.setattr(linux_input, "_xclip_path", lambda: None)
    monkeypatch.setattr(linux_input, "_create_tk_clipboard_root", lambda: root, raising=False)
    monkeypatch.setattr(
        linux_input,
        "get_pyautogui",
        lambda: SimpleNamespace(hotkey=hotkey),
    )
    monkeypatch.setattr(linux_input.time, "sleep", lambda _seconds: None)

    linux_input.insert_text_into_focus(transcript)

    assert pasted == [transcript]
    assert root.text == previous
    assert events[0:2] == [("create", None), ("withdraw", None)]
    assert events[-1] == ("destroy", None)
    staged_at = events.index(("append", transcript))
    paste_at = events.index(("hotkey", ("ctrl", "v")))
    restored_at = events.index(("append", previous))
    assert staged_at < paste_at < restored_at
    assert ("update", None) in events[staged_at:paste_at]
    assert ("update", None) in events[paste_at:restored_at]
    assert ("update", None) in events[restored_at:-1]


def test_linux_feedback_snapshot_restores_clipboard_without_sending_enter(monkeypatch) -> None:
    from utils import linux_input

    clipboard = {"text": "keep me"}
    hotkeys: list[tuple[str, ...]] = []
    restored: list[str | None] = []

    def hotkey(*keys: str) -> None:
        hotkeys.append(keys)
        if keys == ("ctrl", "c"):
            clipboard["text"] = "edited active field"

    monkeypatch.setattr(linux_input, "ensure_desktop_automation_supported", lambda: None)
    monkeypatch.setattr(linux_input, "clipboard_tool_available", lambda: True)
    monkeypatch.setattr(linux_input, "clipboard_contains_non_text_data", lambda: False)
    monkeypatch.setattr(linux_input, "get_clipboard_text", lambda: clipboard["text"])
    monkeypatch.setattr(
        linux_input,
        "set_clipboard_text",
        lambda text: clipboard.update(text=text) is None,
    )
    monkeypatch.setattr(linux_input, "restore_clipboard_text", restored.append)
    monkeypatch.setattr(
        linux_input,
        "get_pyautogui",
        lambda: SimpleNamespace(hotkey=hotkey),
    )

    captured = linux_input.snapshot_active_text_field(copy_wait_seconds=0)

    assert captured == "edited active field"
    assert hotkeys == [("ctrl", "a"), ("ctrl", "c")]
    assert restored == ["keep me"]
    assert ("enter",) not in hotkeys


def test_linux_feedback_snapshot_uses_tk_fallback_and_restores_unicode(
    monkeypatch,
) -> None:
    from utils import linux_input

    previous = "保留する\nclipboard 🌱"
    edited = "Zażółć gęślą\n第二行 ✅"
    events: list[tuple[str, object]] = []
    root = FakeTkClipboardRoot(previous, events)
    hotkeys: list[tuple[str, ...]] = []

    def hotkey(*keys: str) -> None:
        hotkeys.append(keys)
        events.append(("hotkey", keys))
        if keys == ("ctrl", "c"):
            root.text = edited

    monkeypatch.setattr(linux_input, "ensure_desktop_automation_supported", lambda: None)
    monkeypatch.setattr(linux_input, "_xclip_path", lambda: None)
    monkeypatch.setattr(linux_input, "_create_tk_clipboard_root", lambda: root, raising=False)
    monkeypatch.setattr(
        linux_input,
        "get_pyautogui",
        lambda: SimpleNamespace(hotkey=hotkey),
    )

    captured = linux_input.snapshot_active_text_field(copy_wait_seconds=0)

    assert captured == edited
    assert root.text == previous
    assert hotkeys == [("ctrl", "a"), ("ctrl", "c")]
    assert events[0:2] == [("create", None), ("withdraw", None)]
    assert events[-1] == ("destroy", None)
    marker_at = next(
        index
        for index, event in enumerate(events)
        if event[0] == "append" and str(event[1]).startswith("CtrlSpeak-field-capture-")
    )
    copy_at = events.index(("hotkey", ("ctrl", "c")))
    restored_at = events.index(("append", previous))
    assert marker_at < copy_at < restored_at
    assert ("update", None) in events[marker_at:copy_at]
    assert ("update", None) in events[copy_at:restored_at]
    assert ("update", None) in events[restored_at:-1]
    assert ("enter",) not in hotkeys


def test_tk_fallback_reports_actionable_unavailable_display(monkeypatch) -> None:
    from utils import linux_input
    from utils.hotkeys import DesktopSessionError

    monkeypatch.setattr(linux_input, "ensure_desktop_automation_supported", lambda: None)
    monkeypatch.setattr(linux_input, "_xclip_path", lambda: None)
    monkeypatch.setattr(
        sys.modules["tkinter"],
        "Tk",
        lambda: (_ for _ in ()).throw(RuntimeError("couldn't connect to display :99")),
    )

    with pytest.raises(DesktopSessionError) as error:
        linux_input.insert_text_into_focus("Unicode Ω\nmultiline")

    message = str(error.value)
    assert "xclip" in message
    assert "Tk" in message
    assert "DISPLAY" in message


def test_listener_start_failure_keeps_management_path_alive(monkeypatch) -> None:
    from utils import system
    from utils.hotkeys import DesktopSessionError

    notifications: list[tuple[str, str]] = []
    monkeypatch.setattr(
        system,
        "create_global_listener",
        lambda **_kwargs: (_ for _ in ()).throw(
            DesktopSessionError("Wayland is unsupported; choose Ubuntu on Xorg.")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        system,
        "notify",
        lambda message, title="CtrlSpeak": notifications.append((title, message)),
    )
    system.listener = None
    system.client_enabled = True

    system.start_client_listener()

    assert system.listener is None
    assert system.client_enabled is False
    assert notifications == [(
        "CtrlSpeak desktop support",
        "Wayland is unsupported; choose Ubuntu on Xorg.",
    )]


def test_recording_worker_reports_microphone_failure(monkeypatch, tmp_path) -> None:
    from utils import system

    failures: list[tuple[str, str]] = []
    monkeypatch.setattr(
        system,
        "record_audio",
        lambda _path: (_ for _ in ()).throw(RuntimeError("PortAudio unavailable")),
    )
    monkeypatch.setattr(
        system,
        "notify_error",
        lambda context, detail: failures.append((context, detail)),
    )

    system._record_audio_worker(tmp_path / "recording.wav")

    assert failures
    assert failures[0][0] == "Microphone recording failed"
    assert "PortAudio unavailable" in failures[0][1]
