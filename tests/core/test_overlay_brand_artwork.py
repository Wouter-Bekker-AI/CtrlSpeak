from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import tkinter as tk

from utils import midnight_overlay
from utils.config_paths import app_icon_path


pytestmark = pytest.mark.core_headless

# Keep this file standalone under the repository's minimal headless Tk stub;
# drawing tests must not depend on another test module having populated these
# normal tkinter constants first.
if not hasattr(tk, "ROUND"):
    tk.ROUND = "round"  # type: ignore[attr-defined]
if not hasattr(tk, "ARC"):
    tk.ARC = "arc"  # type: ignore[attr-defined]


class RecordingCanvas:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def _record(self, name: str, args: tuple[object, ...], kwargs: dict[str, object]) -> int:
        self.calls.append((name, args, kwargs))
        return len(self.calls)

    def create_arc(self, *args, **kwargs) -> int:
        return self._record("arc", args, kwargs)

    def create_image(self, *args, **kwargs) -> int:
        return self._record("image", args, kwargs)

    def create_line(self, *args, **kwargs) -> int:
        return self._record("line", args, kwargs)

    def create_oval(self, *args, **kwargs) -> int:
        return self._record("oval", args, kwargs)

    def create_text(self, *args, **kwargs) -> int:
        return self._record("text", args, kwargs)


@pytest.mark.parametrize(
    ("scale", "expected"),
    [(0.25, 12), (1.0, 44), (1.25, 55), (1.5, 66), (2.0, 88), (3.0, 132)],
)
def test_brand_microphone_bitmap_tracks_physical_dpi(scale: float, expected: int) -> None:
    assert midnight_overlay.brand_microphone_pixel_size(scale) == expected


def test_packaged_brand_microphone_has_transparency() -> None:
    path = app_icon_path()
    data = path.read_bytes()

    assert path.is_file()
    if path.suffix.lower() == ".png":
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert int.from_bytes(data[16:20], "big") == 128
        assert int.from_bytes(data[20:24], "big") == 128
        assert data[25] == 6  # PNG truecolour with alpha
    else:
        assert data.startswith(b"\x00\x00\x01\x00")
        assert int.from_bytes(data[12:14], "little") == 32


def test_loader_preserves_alpha_and_uses_lanczos_at_physical_size(monkeypatch) -> None:
    image_module = sys.modules["PIL.Image"]
    image_tk_module = sys.modules["PIL.ImageTk"]
    events: list[tuple[object, ...]] = []
    resized = SimpleNamespace(size=(88, 88), marker="resized")

    class Artwork:
        size = (128, 128)

        def resize(self, size, *, resample):
            events.append(("resize", size, resample))
            return resized

    class Source:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def convert(self, mode):
            events.append(("convert", mode))
            return Artwork()

    expected_photo = object()
    lanczos = object()
    canvas = object()
    monkeypatch.setattr(image_module, "Resampling", SimpleNamespace(LANCZOS=lanczos), raising=False)
    monkeypatch.setattr(
        image_module,
        "open",
        lambda path: events.append(("open", Path(path).name)) or Source(),
    )
    monkeypatch.setattr(
        image_tk_module,
        "PhotoImage",
        lambda image, *, master: events.append(("photo", image, master)) or expected_photo,
    )
    monkeypatch.setattr(midnight_overlay, "app_icon_path", lambda: Path("icon.ico"))

    photo = midnight_overlay._load_brand_microphone_photo(canvas, render_scale=2.0)

    assert photo is expected_photo
    assert events == [
        ("open", "icon.ico"),
        ("convert", "RGBA"),
        ("resize", (88, 88), lanczos),
        ("photo", resized, canvas),
    ]


def test_loader_failure_is_safe_and_returns_no_tk_image(monkeypatch) -> None:
    image_module = sys.modules["PIL.Image"]
    monkeypatch.setattr(
        image_module,
        "open",
        lambda _path: (_ for _ in ()).throw(OSError("undecodable icon")),
    )

    assert midnight_overlay._load_brand_microphone_photo(
        object(),
        render_scale=1.0,
    ) is None


def test_recording_and_processing_use_the_same_loaded_ctrlspeak_artwork(monkeypatch) -> None:
    artwork = object()
    overlay = object.__new__(midnight_overlay.MidnightSignalOverlay)
    overlay._brand_microphone_photo = artwork
    overlay.waveform_provider = lambda: [0.25] * 20
    overlay.reduced_motion = True
    overlay._started = 0.0
    monkeypatch.setattr(
        midnight_overlay.MidnightSignalOverlay,
        "_processing_strategy_label",
        staticmethod(lambda _state: "GPU preferred"),
    )

    recording = RecordingCanvas()
    overlay._draw_recording(
        recording,
        548,
        88,
        SimpleNamespace(
            elapsed_label="1.0s",
            level_fraction=0.5,
            level_label="-24.0 dBFS",
        ),
    )
    processing = RecordingCanvas()
    overlay._draw_processing(
        processing,
        548,
        88,
        SimpleNamespace(elapsed_label="1.4s"),
    )

    for canvas in (recording, processing):
        image_calls = [call for call in canvas.calls if call[0] == "image"]
        assert len(image_calls) == 1
        assert image_calls[0][2]["image"] is artwork
        assert image_calls[0][2]["anchor"] == "center"
        assert image_calls[0][2]["tags"] == ("brand-microphone-artwork",)
        assert not any(
            "brand-microphone-fallback" in call[2].get("tags", ())
            for call in canvas.calls
        )


def test_recording_and_processing_fail_safe_to_brand_silhouette(monkeypatch) -> None:
    overlay = object.__new__(midnight_overlay.MidnightSignalOverlay)
    overlay._brand_microphone_photo = None
    overlay.waveform_provider = lambda: [0.25] * 20
    overlay.reduced_motion = True
    overlay._started = 0.0
    monkeypatch.setattr(
        midnight_overlay.MidnightSignalOverlay,
        "_processing_strategy_label",
        staticmethod(lambda _state: "GPU preferred"),
    )

    canvases = (RecordingCanvas(), RecordingCanvas())
    overlay._draw_recording(
        canvases[0],
        548,
        88,
        SimpleNamespace(
            elapsed_label="1.0s",
            level_fraction=0.5,
            level_label="-24.0 dBFS",
        ),
    )
    overlay._draw_processing(
        canvases[1],
        548,
        88,
        SimpleNamespace(elapsed_label="1.4s"),
    )

    for canvas in canvases:
        assert not any(call[0] == "image" for call in canvas.calls)
        assert any(
            "brand-microphone-fallback" in call[2].get("tags", ())
            for call in canvas.calls
        )
