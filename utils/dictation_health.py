"""Bounded packaged-runtime checks without input, microphone or provider access."""
from __future__ import annotations

from array import array
import sys

from utils.dictation_modes import DictationGestures, DictationMode, GestureAction, SpeechSegmenter
from utils.dictation_runtime import DictationSession, DeliveryCancelled


def _require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError("Packaged dictation self-test failed: " + label)


def _forbidden_adapter(*_args, **_kwargs):
    raise RuntimeError("Packaged dictation self-test attempted a side-effecting adapter")


def dictation_health_self_test() -> dict[str, object]:
    """Exercise release imports, gestures, PCM boundaries and privacy gates.

    This function uses synthetic in-memory PCM and explicit timestamps. It
    starts no thread or timer, never creates a controller/listener, and has no
    access to a real target, microphone, filesystem output or provider client.
    The calling packaged CLI writes the resulting non-sensitive booleans.
    It proves bundling/pure logic, not actual desktop or AnyDesk acceptance.
    """
    # Deliberately import the wiring module too. Missing PyInstaller inclusion
    # must fail the build rather than first appearing when a user presses Ctrl.
    from utils.dictation_integration import DesktopDictationController, KEY_LABELS
    from utils.hotkeys import key_name

    _require(callable(DesktopDictationController), "controller import")
    _require(KEY_LABELS.get("ctrl_r") == "Right Ctrl", "key labels")
    _require(key_name("Key.ctrl") == "ctrl_l" and key_name("Key.ctrl_r") == "ctrl_r",
             "generic Control normalization")

    gestures = DictationGestures()
    _require(gestures.press("ctrl_r", now=0) == (), "provisional hold")
    _require(gestures.tick(now=0.2) == (GestureAction.START_HOLD,), "hold activation")
    _require(gestures.release("ctrl_r", mode=DictationMode.HOLD, now=0.3)
             == (GestureAction.FINISH,), "hold finish")
    gestures.reset()
    gestures.press("ctrl_r", now=1)
    _require(gestures.release("ctrl_r", now=1.05) == (), "first tap must not submit")
    gestures.press("ctrl_r", now=1.10)
    _require(gestures.release("ctrl_r", now=1.15)
             == (GestureAction.START_HANDS_FREE,), "hands-free activation")
    gestures.reset()
    gestures.press("Key.ctrl", now=2)
    gestures.press("ctrl_r", now=2.01)
    _require(gestures.release("ctrl_r", now=2.05) == (), "chord release barrier")
    _require(gestures.release("Key.ctrl", now=2.06)
             == (GestureAction.START_LIVE,), "live activation")
    gestures.press("ctrl_l", mode=DictationMode.LIVE, now=3)
    _require(gestures.release("ctrl_l", mode=DictationMode.LIVE, now=3.05)
             == (GestureAction.TOGGLE_MUTE,), "mute gesture")
    _require(gestures.press("ctrl_l", mode=DictationMode.LIVE, injected=True, now=4)
             == () and not gestures.keys_down, "injected input ignored")

    samples = array("h", [2000] * 44982)  # Exactly 1.02 s of PCM16 at 44.1 kHz.
    if sys.byteorder != "little":
        samples.byteswap()
    pcm = samples.tobytes()
    segmenter = SpeechSegmenter(max_chunk_seconds=0.5, min_speech_seconds=0.04,
                                pre_roll_seconds=0)
    segments = segmenter.feed(pcm) + segmenter.flush()
    _require(len(segments) == 3 and b"".join(segments) == pcm, "non-overlapping PCM")
    _require(all(len(part) <= 44100 for part in segments), "bounded PCM chunks")
    _require(segmenter.feed(b"\x00" * 8820) + segmenter.flush() == [], "silence filter")

    target = [101]
    session = DictationSession(
        "hands_free", capture=_forbidden_adapter, transcribe=_forbidden_adapter,
        deliver=_forbidden_adapter, target=lambda: target[0], clock=lambda: 10.0,
    )
    try:
        fragment = pcm[:8820]
        old_token = session.capture_token()
        session.feed(fragment, old_token)
        _require(bytes(session.frames) == fragment, "capture before mute")
        session.toggle_mute()
        muted_token = session.capture_token()
        session.feed(fragment, old_token)
        session.feed(fragment, muted_token)
        _require(bytes(session.frames) == fragment, "muted capture rejected")
        session.toggle_mute()
        session.feed(fragment, muted_token)
        session.feed(fragment, old_token)
        _require(bytes(session.frames) == fragment, "straddling capture rejected")
        session.feed(fragment, session.capture_token())
        _require(bytes(session.frames) == fragment * 2, "explicit capture resume")
        _require(DeliveryCancelled(session, 0).is_set(), "old delivery epoch rejected")
        current_guard = DeliveryCancelled(session, session.epoch)
        _require(not current_guard.is_set(), "valid delivery epoch")
        target[0] = 202
        _require(current_guard.is_set(), "changed target rejected")
    finally:
        session.abort()
    _require(not session.frames and session.worker is None and session.reader is None,
             "no capture or request worker")
    return {"schema": 1, "ok": True, "integration_import": True,
            "gesture_recognizer": True, "pcm_bit_perfect": True,
            "mute_capture_gate": True, "delivery_epoch_guard": True}
