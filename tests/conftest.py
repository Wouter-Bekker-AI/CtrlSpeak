"""Shared pytest fixtures for CtrlSpeak tests."""
from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

import pytest

from utils.config_paths import get_logger


logger = get_logger(__name__)

# Ensure configuration paths stay inside a temporary directory so tests remain
# hermetic even when they exercise the real helpers.
_TEST_CONFIG_ROOT = Path(tempfile.mkdtemp(prefix="ctrlspeak-tests-"))
if sys.platform.startswith("win"):
    os.environ.setdefault("APPDATA", str(_TEST_CONFIG_ROOT))
else:
    os.environ.setdefault("XDG_CONFIG_HOME", str(_TEST_CONFIG_ROOT))


def _install_stub_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    sys.modules[name] = module
    return module


# Several modules pull in optional GUI/audio libraries when imported.  The core
# headless suite avoids exercising those code paths, but importing the modules
# still requires the symbols to exist.  Lightweight stubs keep the imports from
# failing without providing any behaviour.
def _install_runtime_stubs() -> None:
    if "pystray" not in sys.modules:
        pystray = _install_stub_module("pystray")
        pystray.Icon = type("Icon", (), {"__init__": lambda self, *a, **k: None, "run": lambda self: None, "stop": lambda self: None})
        pystray.MenuItem = lambda *a, **k: object()
        pystray.Menu = lambda *a, **k: object()
    if "pyautogui" not in sys.modules:
        pyautogui = _install_stub_module("pyautogui")
        pyautogui.confirm = lambda *a, **k: "Install Server"
    if "pyaudio" not in sys.modules:
        pyaudio = _install_stub_module("pyaudio")
        pyaudio.PyAudio = type("PyAudio", (), {})
        pyaudio.paInt16 = 8
    if "pynput" not in sys.modules:
        pynput = _install_stub_module("pynput")
        keyboard_mod = types.ModuleType("pynput.keyboard")
        keyboard_mod.Listener = type("Listener", (), {"__init__": lambda self, *a, **k: None, "start": lambda self: None, "stop": lambda self: None})
        pynput.keyboard = keyboard_mod
        sys.modules["pynput.keyboard"] = keyboard_mod
    if "tkinter" not in sys.modules:
        tkinter = _install_stub_module("tkinter")

        class _StubTk:
            def __init__(self, *args, **kwargs):
                self._exists = True
                self._visible = True

            def title(self, *_args, **_kwargs):  # pragma: no cover - stub
                return None

            def withdraw(self):  # pragma: no cover - stub
                self._visible = False

            def deiconify(self):  # pragma: no cover - stub
                self._visible = True

            def protocol(self, *_args, **_kwargs):  # pragma: no cover - stub
                return None

            def destroy(self):  # pragma: no cover - stub
                self._exists = False

            def winfo_exists(self) -> bool:  # pragma: no cover - stub
                return self._exists

            def after(self, *_args, **_kwargs):  # pragma: no cover - stub
                return None

            def after_cancel(self, *_args, **_kwargs):  # pragma: no cover - stub
                return None

            def update(self):  # pragma: no cover - stub
                return None

            def update_idletasks(self):  # pragma: no cover - stub
                return None

            def mainloop(self):  # pragma: no cover - stub
                return None

        tkinter.Tk = _StubTk
        ttk_mod = types.ModuleType("tkinter.ttk")
        ttk_mod.Frame = type("Frame", (), {"__init__": lambda self, *a, **k: None})
        messagebox_mod = types.ModuleType("tkinter.messagebox")
        messagebox_mod.showinfo = lambda *a, **k: None
        messagebox_mod.showwarning = lambda *a, **k: None
        tkinter.ttk = ttk_mod
        tkinter.messagebox = messagebox_mod
        sys.modules["tkinter.ttk"] = ttk_mod
        sys.modules["tkinter.messagebox"] = messagebox_mod
    if "utils.bot_integration" not in sys.modules:
        bot_stub = types.ModuleType("utils.bot_integration")
        bot_stub.get_active_identity = lambda: None
        bot_stub.is_bot_running = lambda: False
        bot_stub.start_bot = lambda **_kwargs: True
        bot_stub.stop_bot = lambda: None
        bot_stub.request_goodbye = lambda **_kwargs: True
        bot_stub.clear_identity_memory = lambda *_args, **_kwargs: None
        bot_stub.update_bot_theme = lambda *_args, **_kwargs: True
        bot_stub.list_available_identities = lambda: []
        bot_stub.get_identity_settings = lambda identity: {
            "store_vector_memory": True,
            "store_screenshots": True,
            "retrieval_top_k": 5,
            "retrieval_threshold": 0.75,
            "max_vector_items": 5000,
            "vector_ttl_days": None,
            "pii_redaction": False,
        }
        sys.modules["utils.bot_integration"] = bot_stub
    if "PIL" not in sys.modules:
        pil = _install_stub_module("PIL")
        image_mod = types.ModuleType("PIL.Image")
        image_mod.open = lambda *a, **k: None
        image_mod.new = lambda *a, **k: None
        pil.Image = image_mod
        imagetk_mod = types.ModuleType("PIL.ImageTk")
        imagetk_mod.PhotoImage = lambda *a, **k: None
        pil.ImageTk = imagetk_mod
        sys.modules["PIL.Image"] = image_mod
        sys.modules["PIL.ImageTk"] = imagetk_mod
    if "numpy" not in sys.modules:
        import importlib.util

        spec = importlib.util.find_spec("numpy")
        if spec is not None:
            module = importlib.import_module("numpy")
            sys.modules["numpy"] = module
        else:
            numpy = _install_stub_module("numpy")
            numpy.ndarray = list  # type: ignore[assignment]
            numpy.float32 = float
            numpy.int16 = int
            numpy.pi = 3.141592653589793
            numpy.zeros = lambda *a, **k: []
            numpy.concatenate = lambda *a, **k: []
            numpy.frombuffer = lambda *a, **k: []
            numpy.linspace = lambda *a, **k: []
            numpy.exp = lambda *a, **k: []
            numpy.sin = lambda *a, **k: []
            numpy.clip = lambda *a, **k: []
            numpy.sqrt = lambda *a, **k: 0.0
            numpy.mean = lambda *a, **k: 0.0


_install_runtime_stubs()


@pytest.fixture(autouse=True)
def reset_settings(tmp_path, monkeypatch):
    """Reset persisted settings and isolate config/temp directories per test."""
    config_home = tmp_path / "config"
    config_home.mkdir()
    if sys.platform.startswith("win"):
        monkeypatch.setenv("APPDATA", str(config_home))
    else:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

    # Reload config_paths so the globals (e.g., MODEL_ROOT_PATH) pick up the new
    # directory for this specific test.
    import importlib
    from utils import config_paths

    importlib.reload(config_paths)

    # system imports selected helpers directly, so reload it as well to ensure
    # it references the isolated config helpers.
    import utils.system as system

    importlib.reload(system)

    # Ensure callers start from default settings each time.
    config_paths.load_settings()
    yield

    # Clean up server/listener threads if a test left them running.
    try:
        system.shutdown_server()
    except Exception:
        logger.exception("Failed to shut down transcription server during test cleanup")
    try:
        system.stop_discovery_listener()
    except Exception:
        logger.exception("Failed to stop discovery listener during test cleanup")

