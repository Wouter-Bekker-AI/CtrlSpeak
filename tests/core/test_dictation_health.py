"""Release smoke tests never access a real microphone, target or provider."""
from pathlib import Path
import threading

import pytest

from utils.dictation_health import dictation_health_self_test
from utils.dictation_modes import DictationGestures
from utils.dictation_runtime import DictationSession


pytestmark = pytest.mark.core_headless


def test_packaged_dictation_smoke_has_no_runtime_side_effects(monkeypatch):
    from utils import dictation_integration

    def forbidden(*_args, **_kwargs):
        pytest.fail("Pure packaged health check used a real runtime adapter")

    monkeypatch.setattr(threading.Thread, "start", forbidden)
    monkeypatch.setattr(DictationSession, "start", forbidden)
    monkeypatch.setattr(dictation_integration, "desktop_target", forbidden)
    result = dictation_health_self_test()
    assert result == {"schema": 1, "ok": True, "integration_import": True,
                      "gesture_recognizer": True, "pcm_bit_perfect": True,
                      "mute_capture_gate": True, "delivery_epoch_guard": True}


def test_packaged_smoke_detects_broken_gesture_recognition(monkeypatch):
    monkeypatch.setattr(DictationGestures, "tick", lambda *args, **kwargs: ())
    with pytest.raises(RuntimeError, match="hold activation"):
        dictation_health_self_test()


def test_packaged_smoke_detects_broken_delivery_guard(monkeypatch):
    monkeypatch.setattr(DictationSession, "delivery_allowed", lambda *args: True)
    with pytest.raises(RuntimeError, match="old delivery epoch rejected"):
        dictation_health_self_test()


def test_maintained_spec_explicitly_bundles_dictation_modules():
    root = Path(__file__).resolve().parents[2]
    source = (root / "packaging" / "CtrlSpeak_v0.7.spec").read_text(encoding="utf-8")
    for name in ("dictation_modes", "dictation_runtime", "dictation_integration", "dictation_health"):
        assert f"'utils.{name}'" in source


def test_release_requires_dictation_health_on_both_platforms():
    root = Path(__file__).resolve().parents[2]
    source = (root / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "$health.dictation_modes.ok" in source
    assert "['dictation_modes']; assert p['ok']" in source
    assert "delivery_epoch_guard" in source
