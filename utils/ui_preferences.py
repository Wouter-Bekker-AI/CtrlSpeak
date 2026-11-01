"""Shared live display preferences; no audio, provider or input operations."""
from utils import config_paths
from utils.ui_state import format_microphone_level


def microphone_level_label(dbfs: object) -> str:
    # Read the current module's settings, not a cached dict: profile reloads
    # and a checkbox change must reach existing windows/overlays immediately.
    with config_paths.settings_lock:
        technical = config_paths.settings.get("microphone_level_dbfs") is True
    return format_microphone_level(dbfs, show_dbfs=technical)


def microphone_meter_self_test() -> dict[str, object]:
    """Check packaged formatters without reading or changing user settings."""
    expected = ((-60, "0 / 100"), (-30, "50 / 100"), (0, "100 / 100"))
    if any(format_microphone_level(value) != label for value, label in expected):
        raise RuntimeError("Microphone-level display self-test failed")
    if format_microphone_level(-30, show_dbfs=True) != "-30.0 dBFS":
        raise RuntimeError("Technical microphone-level display self-test failed")
    return {"ok": True, "default_scale": "0-100", "dbfs_optional": True}
