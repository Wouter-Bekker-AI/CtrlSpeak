"""0.8.1 regressions: no microphone, global hook, input injection or visible UI."""
from pathlib import Path
import subprocess
import sys

import pytest

from utils import midnight_overlay as overlay_module
from utils.hotkeys import key_name, is_right_control

pytestmark = pytest.mark.core_headless


@pytest.mark.parametrize("name", ["ctrl", "Key.ctrl", "ctrl_l", "Key.ctrl_l"])
def test_generic_and_explicit_left_control_have_one_gesture_identity(name):
    assert key_name(name) == "ctrl_l"
    assert not is_right_control(name)


def test_native_windows_pynput_control_objects_without_starting_a_listener():
    if not sys.platform.startswith("win"):
        pytest.skip("Windows backend enumeration, without hooks")
    script = """
from pynput.keyboard import Key
from utils.hotkeys import key_name
from utils.dictation_modes import DictationGestures, GestureAction
assert key_name(Key.ctrl) == key_name(Key.ctrl_l) == 'ctrl_l'
assert key_name(Key.ctrl_r) == 'ctrl_r'
for left in (Key.ctrl, Key.ctrl_l):
    g = DictationGestures()
    g.press(left, now=0)
    g.press(Key.ctrl_r, now=.1)
    assert g.release(Key.ctrl_r, now=.2) == ()
    assert g.release(left, now=.3) == (GestureAction.START_LIVE,)
"""
    result = subprocess.run([sys.executable, "-c", script],
                            cwd=Path(__file__).resolve().parents[2],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("bounds", [
    overlay_module.MonitorBounds(0, 0, 1920, 1040),
    overlay_module.MonitorBounds(-1920, -1080, 0, -40),
    overlay_module.MonitorBounds(1920, 48, 4480, 1440),
    overlay_module.MonitorBounds(0, 40, 400, 240),
])
@pytest.mark.parametrize("scale", [1.0, 1.5, 2.0, 3.0])
def test_live_geometry_is_top_centred_inside_monitor_work_area(bounds, scale):
    live = overlay_module.overlay_geometry(bounds, scale, live=True)
    normal = overlay_module.overlay_geometry(bounds, scale)
    assert bounds.left <= live.x <= bounds.right - live.width
    assert bounds.top <= live.y <= bounds.bottom - live.height
    assert live.width == normal.width and live.height == normal.height
    assert live.x == normal.x
    assert live.y <= normal.y
    bottom_gap = bounds.bottom - normal.y - normal.height
    if bounds.height - normal.height >= 2 * bottom_gap:
        assert live.y - bounds.top == bottom_gap
    else:
        assert live.y == bounds.top  # Do not push Live down on tiny work areas.


class Window:
    def __init__(self, *_args, **_kwargs):
        self.geometries = []

    def winfo_exists(self):
        return True

    def geometry(self, value):
        self.geometries.append(value)

    def __getattr__(self, name):
        return lambda *_args, **_kwargs: None


class Canvas:
    def __init__(self, *_args, **_kwargs):
        pass

    def pack(self, **_kwargs):
        pass


def make_overlay(monkeypatch, status, *, shown=False):
    overlay = object.__new__(overlay_module.MidnightSignalOverlay)
    overlay.root = object()
    overlay.window = Window() if shown else None
    overlay.canvas = None
    overlay.scale = overlay._render_scale = 1.0
    overlay._live_position = False
    overlay._close_job = None
    overlay._closing = False
    overlay.dictation_status_provider = lambda: status
    overlay._tick = lambda: None
    placements = []
    bounds = overlay_module.MonitorBounds(-1920, -1080, 0, -40)
    monkeypatch.setattr(overlay_module, "_active_monitor_bounds", lambda _root: bounds)
    monkeypatch.setattr(overlay_module, "_make_focus_safe", lambda _window: 7301)
    monkeypatch.setattr(overlay_module, "_place_windows_overlay_absolute",
                        lambda window, x, y, **kwargs: placements.append((x, y)) or True)
    monkeypatch.setattr(overlay_module, "_load_brand_microphone_photo", lambda *a, **k: None)
    monkeypatch.setattr(overlay_module.tk, "Toplevel", Window, raising=False)
    monkeypatch.setattr(overlay_module.tk, "Canvas", Canvas)
    return overlay, placements, bounds


@pytest.mark.parametrize("mode,active,top", [
    ("live", True, True), ("hands_free", True, False),
    ("hold", True, False), ("live", False, False),
])
def test_initial_overlay_placement_is_specific_to_active_live(monkeypatch, mode, active, top):
    overlay, placements, bounds = make_overlay(monkeypatch, {"mode": mode, "active": active})
    overlay.show()
    expected = overlay_module.overlay_geometry(bounds, 1, live=top)
    assert overlay.window.geometries == [overlay_module.overlay_geometry_spec(expected)]
    assert placements == [(expected.x, expected.y)]
    assert overlay._live_position is top


def test_reused_capsule_moves_to_top_for_live_and_back_for_next_hold(monkeypatch):
    status = {"active": True, "mode": "live"}
    overlay, placements, bounds = make_overlay(monkeypatch, status, shown=True)
    overlay.set_waveform_provider(None)
    top = overlay_module.overlay_geometry(bounds, 1, live=True)
    assert placements == [(top.x, top.y)]
    status.update(muted=True, editing=True, finishing=True)
    overlay.set_waveform_provider(None)
    assert placements == [(top.x, top.y)]  # No jumps while muted/finishing.
    # End-of-session drawing does not re-anchor; keep the result out of the way.
    status.update(active=False)
    assert overlay._live_position is True
    overlay.set_waveform_provider(None)  # A new legacy Hold capture reuses it.
    bottom = overlay_module.overlay_geometry(bounds, 1)
    assert placements == [(top.x, top.y), (bottom.x, bottom.y)]
    assert overlay._live_position is False
