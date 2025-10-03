
"""Vision-related tooling shared across CtrlSpeak components."""

from __future__ import annotations

import base64
import io
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from utils.config_paths import get_logger

logger = get_logger(__name__)

_CAMERA_SOUND_LOCK = threading.Lock()
_CAMERA_SOUND = None
_CAMERA_SOUND_FAILED = False


@dataclass(slots=True)
class VisionCapture:
    """Result object returned by the vision helpers."""

    image_b64: Optional[str]
    saved_path: Optional[Path]
    source: str
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.image_b64 is not None and self.error is None


def capture_screenshot(
    target_dir: Optional[Path] = None,
    *,
    filename_prefix: str = "screenshot",
) -> VisionCapture:
    """Capture the current desktop using PyAutoGUI."""

    try:
        import pyautogui  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - import failure path
        message = f"Screenshot capture unavailable: {exc}"
        logger.warning(message)
        return VisionCapture(None, None, "screen", error=message)

    try:
        image = pyautogui.screenshot()
    except Exception as exc:
        logger.exception("Failed to capture screenshot")
        return VisionCapture(None, None, "screen", error=str(exc))

    return _finalize_capture(image, target_dir, filename_prefix, source="screen")


def capture_clipboard_image(
    target_dir: Optional[Path] = None,
    *,
    filename_prefix: str = "clipboard",
) -> VisionCapture:
    """Capture the newest image from the system clipboard."""

    try:
        from PIL import ImageGrab  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - import failure path
        message = f"Clipboard capture unavailable: {exc}"
        logger.warning(message)
        return VisionCapture(None, None, "clipboard", error=message)

    try:
        payload = ImageGrab.grabclipboard()
    except Exception as exc:
        logger.exception("Failed to read clipboard image")
        return VisionCapture(None, None, "clipboard", error=str(exc))

    image = _resolve_clipboard_image(payload)
    if image is None:
        message = "Clipboard does not contain an image payload."
        logger.info(message)
        return VisionCapture(None, None, "clipboard", error=message)

    return _finalize_capture(image, target_dir, filename_prefix, source="clipboard")


def _resolve_clipboard_image(payload):
    """Extract a Pillow image object from the clipboard payload."""

    try:
        from PIL import Image  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - guarded by outer import
        return None

    if payload is None:
        return None
    if isinstance(payload, Image.Image):
        return payload
    if isinstance(payload, (list, tuple)):
        for item in payload:
            resolved = _resolve_clipboard_image(item)
            if resolved is not None:
                return resolved
        return None
    if isinstance(payload, str):
        path = Path(payload)
        if not path.exists():
            return None
        try:
            with Image.open(path) as img:
                return img.copy()
        except Exception:
            return None
    return None


def _finalize_capture(image, target_dir: Optional[Path], filename_prefix: str, *, source: str) -> VisionCapture:
    buffer = io.BytesIO()
    try:
        image.save(buffer, format="PNG")
    except Exception as exc:
        logger.exception("Failed to encode %s capture", source)
        return VisionCapture(None, None, source, error=str(exc))

    image_bytes = buffer.getvalue()
    image_b64 = base64.b64encode(image_bytes).decode("ascii")

    saved_path: Optional[Path] = None
    if target_dir:
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            saved_path = target_dir / f"{filename_prefix}_{_timestamp()}.png"
            saved_path.write_bytes(image_bytes)
        except Exception as exc:
            logger.exception("Failed to persist %s capture", source)
            saved_path = None

    logger.info("Captured %s image for vision tooling.", source)
    _play_camera_shutter()
    return VisionCapture(image_b64, saved_path, source)


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")


def _play_camera_shutter() -> None:
    global _CAMERA_SOUND, _CAMERA_SOUND_FAILED

    if _CAMERA_SOUND_FAILED:
        return

    with _CAMERA_SOUND_LOCK:
        if _CAMERA_SOUND_FAILED:
            return
        try:
            import pygame  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - pygame optional
            logger.debug("Camera sound unavailable: %s", exc)
            _CAMERA_SOUND_FAILED = True
            return

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
        except Exception as exc:
            logger.debug("Failed to initialize pygame mixer: %s", exc)
            _CAMERA_SOUND_FAILED = True
            return

        if _CAMERA_SOUND is None:
            sound_path = Path(__file__).resolve().parents[1] / "assets" / "camera.wav"
            try:
                _CAMERA_SOUND = pygame.mixer.Sound(str(sound_path))
            except Exception as exc:
                logger.debug("Failed to load camera sound: %s", exc)
                _CAMERA_SOUND_FAILED = True
                return

        try:
            _CAMERA_SOUND.play()  # type: ignore[union-attr]
        except Exception as exc:
            logger.debug("Failed to play camera sound: %s", exc)
            _CAMERA_SOUND_FAILED = True


__all__ = [
    "VisionCapture",
    "capture_clipboard_image",
    "capture_screenshot",
]
