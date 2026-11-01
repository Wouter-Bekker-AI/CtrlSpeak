from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
import pytest
from utils.keyboard_guide import (GuidePlayback, KEY_LABELS, build_lesson, guide_self_test)
from utils.dictation_modes import DictationMode, DictationGestures, GestureAction, SUPPORTED_GESTURE_KEYS

pytestmark = pytest.mark.core_headless


@pytest.mark.parametrize("keys",[("bad","ctrl_l"),("f8","f8"),("ctrl_r","bad"),("f9","f10")])
def test_saved_key_fallback_matches_runtime(keys):
    from utils.keyboard_guide import effective_keys
    try:
        gestures=DictationGestures(primary_key=keys[0],secondary_key=keys[1])
    except ValueError:
        gestures=DictationGestures()
    assert effective_keys(*keys)==(gestures.primary_key,gestures.secondary_key)


@pytest.mark.parametrize("mode", [m.value for m in DictationMode])
@pytest.mark.parametrize("keys", [(a,b) for a in SUPPORTED_GESTURE_KEYS for b in SUPPORTED_GESTURE_KEYS if a!=b])
def test_lesson_describes_actual_configured_keys_and_all_states(mode, keys):
    lesson = build_lesson(mode,*keys)
    assert KEY_LABELS[keys[0]] in lesson.shortcut
    assert all(step.duration_ms > 0 and step.keys_down <= set(keys) for step in lesson.steps)
    assert lesson.steps[0].state == "ready" and lesson.steps[-1].state == "done"
    if mode != "hold":
        assert KEY_LABELS[keys[1]] in lesson.shortcut
        assert {"muted","recording","processing","done"} <= {s.state for s in lesson.steps}
    assert "Ctrl+C/V" in lesson.safety or mode == "hold"


@pytest.mark.parametrize("keys", [("bad","ctrl_l"),("ctrl_r","ctrl_r"),("ctrl_r","bad")])
def test_rejects_invalid_or_duplicate_keys(keys):
    with pytest.raises(ValueError):build_lesson("hold",*keys)


@pytest.mark.parametrize("mode", [m.value for m in DictationMode])
def test_one_cycle_pause_seek_replay_and_finite_completion(mode):
    lesson = build_lesson(mode)
    p = GuidePlayback(lesson)
    p.advance(999999)
    assert p.index == 0 and p.elapsed_ms == 0
    p.restart()
    p.advance(lesson.steps[0].duration_ms+23)
    assert p.index == 1 and p.elapsed_ms == 23
    p.seek(-999)
    assert not p.playing and p.index == 0
    p.seek(999)
    assert p.index == len(lesson.steps)-1
    p.restart()
    p.advance(sum(s.duration_ms for s in lesson.steps)+1)
    assert not p.playing and p.progress == 1 and p.step.state == "done"
    p.restart(play=False)
    assert not p.playing and p.index == 0


@pytest.mark.parametrize("value", [-1,float("nan"),float("inf")])
def test_rejects_invalid_playback_intervals(value):
    with pytest.raises(ValueError):GuidePlayback(build_lesson("hold")).advance(value)


@pytest.mark.parametrize("windows,enabled,expected", [(False,True,"requires Windows"),
    (True,False,"Live is off"),(True,True,"Live is enabled")])
def test_live_availability_is_truthful(windows,enabled,expected):
    lesson = build_lesson("live",windows=windows,live_enabled=enabled)
    assert expected in lesson.availability
    assert "release both" in lesson.shortcut
    assert any("after both" in s.detail for s in lesson.steps)


def test_demo_tap_durations_match_real_gesture_recognizer():
    lesson=build_lesson("hands_free")
    g=DictationGestures()
    clock=0.0
    actions=[]
    down=set()
    mode=None
    for step in lesson.steps:
        for key in down-step.keys_down:actions.extend(g.release(key,mode=mode,now=clock))
        for key in step.keys_down-down:actions.extend(g.press(key,mode=mode,now=clock))
        if GestureAction.START_HANDS_FREE in actions:mode=DictationMode.HANDS_FREE
        clock+=step.duration_ms/1000
        down=set(step.keys_down)
    assert GestureAction.START_HANDS_FREE in actions
    assert actions.count(GestureAction.TOGGLE_MUTE)==2
    assert GestureAction.FINISH in actions
    assert GestureAction.START_HOLD not in actions


def _fake_widget():
    from utils.keyboard_guide_ui import KeyboardGuide
    widget = object.__new__(KeyboardGuide)
    widget.playback = GuidePlayback(build_lesson("hold"))
    widget.primary,widget.secondary="ctrl_r","ctrl_l"
    widget.windows,widget.live_enabled=True,False
    widget.reduced_motion,widget._visible,widget._closed=False,False,False
    widget._seen,widget._job=False,None
    widget._render=lambda:None
    widget.cancelled=[]
    widget.after_cancel=lambda job:widget.cancelled.append(job)
    widget.after=lambda delay,fn:"timer"
    return widget


def test_widget_owns_one_timer_and_tab_hiding_pauses():
    w=_fake_widget()
    w.set_visible(True)
    assert w.playback.playing and w._job=="timer"
    w._schedule()
    assert w.cancelled==["timer"]
    w.set_visible(False)
    assert not w.playback.playing and w._job is None
    w.set_visible(True)
    assert not w.playback.playing # returning does not unexpectedly resume
    w.toggle_play()
    w.shutdown()
    assert w._closed and w._job is None and not w.playback.playing


def test_reduced_motion_is_still_navigable_without_timers():
    w=_fake_widget();w.reduced_motion=True
    w.set_visible(True);w.toggle_play();w.replay()
    assert not w.playback.playing and w._job is None
    w.seek(1)
    assert w.playback.index==1
    w.select_mode("live")
    assert not w.playback.playing


def test_remap_resets_and_updates_instructions_without_starting_capture():
    w=_fake_widget();w.set_visible(True)
    w.configure_keys("f8","f9",live_enabled=True)
    assert w.playback.index==0 and "F8" in w.playback.lesson.shortcut
    w.select_mode("live")
    assert "F9 + F8" in w.playback.lesson.shortcut
    assert w.playback.playing
    w.configure_keys("f8","f9",reduced_motion=True)
    assert not w.playback.playing and w._job is None
    with pytest.raises(ValueError):w.configure_keys("f8","f8")
    assert w.secondary=="f9"


def test_guide_imports_and_commands_have_no_recording_input_or_network_adapters():
    from utils import keyboard_guide,keyboard_guide_ui
    forbidden={"utils.system","utils.dictation_integration","requests","pynput","pyautogui","pyaudio","subprocess","ctypes"}
    for module in (keyboard_guide,keyboard_guide_ui):
        tree=ast.parse(inspect.getsource(module))
        imports={node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)}
        imports|={alias.name for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names}
        assert not imports&forbidden
    assert guide_self_test()=={"ok":True,"modes":3,"simulation_only":True}


def test_capture_entry_point_visibility_sync_and_shutdown_are_wired():
    source=(Path(__file__).resolve().parents[2]/"utils/midnight_signal_ui.py").read_text("utf-8")
    assert '"Capture", "Guide"' in source
    assert 'Animated keyboard guide' in source
    assert 'self._build_keyboard_guide_page(self.ms_pages["Guide"])' in source
    assert '"<<NotebookTabChanged>>", self._keyboard_guide_visibility' in source
    assert "guide.shutdown()" in source
