import base64
import sys
import types

import pytest

from tools import vision

pytestmark = pytest.mark.core_headless


class _DummyImage:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def save(self, buffer, format: str) -> None:  # noqa: A003 - format is Pillow compatible
        buffer.write(self._payload)


def _install_fake_pyautogui(monkeypatch, image_bytes: bytes) -> None:
    module = types.SimpleNamespace()
    module.screenshot = lambda: _DummyImage(image_bytes)
    monkeypatch.setitem(sys.modules, "pyautogui", module)


def _install_fake_pil(monkeypatch, image_bytes: bytes, *, empty: bool = False) -> None:
    pil_package = types.ModuleType("PIL")

    class FakeImage(_DummyImage):
        pass

    image_module = types.ModuleType("Image")
    image_module.Image = FakeImage

    def open(path):  # pragma: no cover - exercised via clipboard fallback only
        return FakeImage(image_bytes)

    image_module.open = open

    imagegrab_module = types.ModuleType("ImageGrab")

    def grabclipboard():
        if empty:
            return None
        return FakeImage(image_bytes)

    imagegrab_module.grabclipboard = grabclipboard

    pil_package.Image = image_module
    pil_package.ImageGrab = imagegrab_module

    monkeypatch.setitem(sys.modules, "PIL", pil_package)
    monkeypatch.setitem(sys.modules, "PIL.Image", image_module)
    monkeypatch.setitem(sys.modules, "PIL.ImageGrab", imagegrab_module)


@pytest.fixture(autouse=True)
def _no_camera_sound(monkeypatch):
    monkeypatch.setattr(vision, "_play_camera_shutter", lambda: None)


@pytest.fixture(autouse=True)
def _stable_timestamp(monkeypatch):
    monkeypatch.setattr(vision, "_timestamp", lambda: "20240101T000000000000Z")


def test_capture_screenshot_succeeds(tmp_path, monkeypatch):
    payload = b"PNGDATA"
    _install_fake_pyautogui(monkeypatch, payload)

    result = vision.capture_screenshot(tmp_path)
    assert result.success
    assert result.source == "screen"
    assert result.saved_path == tmp_path / "screenshot_20240101T000000000000Z.png"
    assert result.saved_path.read_bytes() == payload
    assert base64.b64decode(result.image_b64.encode("ascii")) == payload


def test_capture_screenshot_handles_failure(monkeypatch):
    class Broken:
        def screenshot(self):
            raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "pyautogui", Broken())

    result = vision.capture_screenshot()
    assert not result.success
    assert result.error
    assert result.source == "screen"


def test_capture_clipboard_image_succeeds(tmp_path, monkeypatch):
    payload = b"CLIP"
    _install_fake_pil(monkeypatch, payload)

    result = vision.capture_clipboard_image(tmp_path)
    assert result.success
    assert result.source == "clipboard"
    expected_path = tmp_path / "clipboard_20240101T000000000000Z.png"
    assert result.saved_path == expected_path
    assert expected_path.read_bytes() == payload
    assert base64.b64decode(result.image_b64.encode("ascii")) == payload


def test_capture_clipboard_image_empty(monkeypatch):
    _install_fake_pil(monkeypatch, b"", empty=True)

    result = vision.capture_clipboard_image()
    assert not result.success
    assert result.source == "clipboard"
    assert "clipboard" in (result.error or "").lower()
