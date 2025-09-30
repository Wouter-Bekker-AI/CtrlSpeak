"""Thread-safety regression tests for utils.gui helpers."""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("tkinter")
pytest.importorskip("tkinter.ttk")

import utils.gui as gui


pytestmark = pytest.mark.core_headless


class _DummyRoot:
    def __init__(self) -> None:
        self.quit_called = False

    def winfo_exists(self) -> bool:
        raise RuntimeError("winfo_exists should not be called from worker threads")

    def after(self, _delay: int, callback):
        callback()

    def quit(self) -> None:
        self.quit_called = True


def test_management_shutdown_helpers_skip_cross_thread_winfo(monkeypatch):
    dummy = _DummyRoot()

    previous_root = gui.tk_root
    previous_ready = gui._management_thread_ready.is_set()
    previous_ident = gui._management_thread_ident

    gui.tk_root = dummy
    gui._management_thread_ready.set()
    gui._management_thread_ident = threading.get_ident()

    results: list[object] = []

    def worker() -> None:
        results.append(gui._management_root_if_ready())
        gui.request_management_ui_shutdown()

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive(), "worker thread did not finish"

    assert results == [dummy]
    assert dummy.quit_called, "request_management_ui_shutdown did not invoke quit()"

    gui.tk_root = previous_root
    gui._management_thread_ident = previous_ident
    if previous_ready:
        gui._management_thread_ready.set()
    else:
        gui._management_thread_ready.clear()

