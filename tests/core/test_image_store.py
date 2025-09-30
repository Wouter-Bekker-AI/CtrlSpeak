import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.core_headless


_PIXEL_A = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAoMBgJRMnL0AAAAASUVORK5CYII="
)
_PIXEL_B = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8HwQACfsD/QyDrE8AAAAASUVORK5CYII="
)


def _prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import utils.config_paths as cfg

    data_home = tmp_path / "data"
    config_home = tmp_path / "cfg"
    data_home.mkdir()
    config_home.mkdir()

    if cfg.sys.platform.startswith("win"):
        monkeypatch.setenv("APPDATA", str(data_home))
    else:
        monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

    modules = {}
    for name in [
        "utils.config_paths",
        "utils.memory_paths",
        "utils.image_store",
    ]:
        modules[name] = importlib.reload(importlib.import_module(name))
    return modules


def test_write_load_and_clear_identity_image(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    image_store = modules["utils.image_store"]

    record = image_store.write_identity_image_from_base64("Sample", _PIXEL_A, source="screen")
    assert record is not None
    assert record.path.exists()
    assert record.source == "screen"
    assert record.image_b64 == _PIXEL_A

    # Replacing the image keeps a single PNG on disk
    updated = image_store.write_identity_image_from_base64("Sample", _PIXEL_B, source="clipboard")
    assert updated is not None
    assert updated.path.exists()
    png_files = list(updated.path.parent.glob("*.png"))
    assert png_files == [updated.path]
    assert updated.image_b64 == _PIXEL_B
    assert image_store.is_image_request("what's in the screenshot?")
    assert not image_store.is_image_request("tell me a joke")

    loaded = image_store.load_identity_image("Sample")
    assert loaded is not None
    assert loaded.image_b64 == _PIXEL_B
    assert loaded.source == "clipboard"

    image_store.clear_identity_image("Sample")
    assert not updated.path.exists()
    meta_path = updated.path.parent / "metadata.json"
    assert not meta_path.exists()
