"""Pure mode tests: no microphone, UI, OS input, network, or provider calls."""
from array import array
import sys

import pytest

from utils.dictation_modes import (
    DictationGestures, DictationMode as Mode, GestureAction as Action,
    SpeechSegmenter,
)


pytestmark = pytest.mark.core_headless


class Clock:
    def __init__(self):
        self.value = 10.0

    def __call__(self):
        return self.value

    def step(self, seconds=0.05):
        self.value += seconds


@pytest.fixture
def keys():
    clock = Clock()
    return DictationGestures(clock=clock), clock


def tap(keys, clock, key="ctrl_r", mode=None):
    pressed = keys.press(key, mode=mode)
    clock.step()
    return pressed + keys.release(key, mode=mode)


def test_hold_commits_once_after_threshold_and_finishes_on_release(keys):
    g, clock = keys
    assert g.press("ctrl_r") == ()
    clock.step(0.17)
    assert g.tick() == ()
    clock.step(0.02)
    assert g.tick() == (Action.START_HOLD,)
    assert g.tick() == ()
    assert g.tick(mode=Mode.HOLD) == ()
    assert g.release("ctrl_r", mode=Mode.HOLD) == (Action.FINISH,)


def test_first_short_tap_never_submits_or_starts(keys):
    g, clock = keys
    assert tap(g, clock) == ()
    clock.step(1)
    assert g.tick() == ()
    assert not g.keys_down


def test_double_tap_starts_hands_free_only_after_second_release(keys):
    g, clock = keys
    assert tap(g, clock) == ()
    clock.step(0.20)
    assert g.press("ctrl_r") == ()
    # Window is measured to the second press, not its eventual release.
    clock.step(0.15)
    assert g.release("ctrl_r") == (Action.START_HANDS_FREE,)
    assert tap(g, clock, mode=Mode.HANDS_FREE) == (Action.FINISH,)


def test_second_tap_too_late_does_not_latch(keys):
    g, clock = keys
    assert tap(g, clock) == ()
    clock.step(0.31)
    assert tap(g, clock) == ()


def test_second_tap_becoming_hold_starts_hold_not_hands_free(keys):
    g, clock = keys
    tap(g, clock)
    g.press("ctrl_r")
    clock.step(0.19)
    assert g.tick() == (Action.START_HOLD,)
    assert g.release("ctrl_r", mode=Mode.HOLD) == (Action.FINISH,)


@pytest.mark.parametrize("press_order", [("ctrl_l", "ctrl_r"), ("ctrl_r", "ctrl_l")])
@pytest.mark.parametrize("release_order", [("ctrl_l", "ctrl_r"), ("ctrl_r", "ctrl_l")])
def test_live_chord_waits_for_both_releases(keys, press_order, release_order):
    g, clock = keys
    assert g.press(press_order[0]) == ()
    clock.step()
    assert g.press(press_order[1]) == ()
    clock.step(0.4)
    assert g.tick() == ()
    assert g.release(release_order[0]) == ()
    assert g.release(release_order[1]) == (Action.START_LIVE,)


def test_live_chord_aborts_hold_without_submitting_it(keys):
    g, clock = keys
    g.press("ctrl_r")
    clock.step(0.2)
    assert g.tick() == (Action.START_HOLD,)
    assert g.press("ctrl_l", mode=Mode.HOLD) == (Action.ABORT_HOLD,)
    assert g.release("ctrl_l") == ()
    assert g.release("ctrl_r") == (Action.START_LIVE,)


@pytest.mark.parametrize("mode", [Mode.HANDS_FREE, Mode.LIVE])
def test_isolated_secondary_tap_toggles_mute_only_while_active(keys, mode):
    g, clock = keys
    assert tap(g, clock, "ctrl_l") == ()
    assert tap(g, clock, "ctrl_l", mode) == (Action.TOGGLE_MUTE,)
    assert tap(g, clock, "ctrl_l", mode) == (Action.TOGGLE_MUTE,)


@pytest.mark.parametrize("modifier", ["ctrl_l", "ctrl_r"])
@pytest.mark.parametrize("edit_key", ["c", "v", "z", "left", "shift_l", "alt_r"])
def test_normal_editing_chords_never_toggle_or_finish(keys, modifier, edit_key):
    g, clock = keys
    assert g.press(modifier, mode=Mode.LIVE) == ()
    assert g.press(edit_key, mode=Mode.LIVE) == ()
    clock.step()
    assert g.release(edit_key, mode=Mode.LIVE) == ()
    assert g.release(modifier, mode=Mode.LIVE) == ()


def test_other_key_held_before_control_invalidates_gesture(keys):
    g, clock = keys
    g.press("shift_l")
    g.press("ctrl_r")
    g.release("shift_l")
    clock.step(0.5)
    assert g.tick() == ()
    assert g.release("ctrl_r") == ()


def test_control_used_for_editing_cannot_become_live_chord(keys):
    g, clock = keys
    g.press("ctrl_l")
    g.press("c")
    g.release("c")
    g.press("ctrl_r")
    assert g.release("ctrl_r") == ()
    assert g.release("ctrl_l") == ()


def test_third_key_invalidates_live_chord(keys):
    g, clock = keys
    g.press("ctrl_l")
    g.press("ctrl_r")
    g.press("c")
    g.release("c")
    assert g.release("ctrl_l") == ()
    assert g.release("ctrl_r") == ()


def test_editing_after_committed_hold_aborts_without_finish(keys):
    g, clock = keys
    g.press("ctrl_r")
    clock.step(0.2)
    assert g.tick() == (Action.START_HOLD,)
    assert g.press("c", mode=Mode.HOLD) == (Action.ABORT_HOLD,)
    g.release("c")
    assert g.release("ctrl_r") == ()


def test_autorepeat_and_injected_input_are_not_gestures(keys):
    g, clock = keys
    g.press("ctrl_l", mode=Mode.LIVE)
    for _ in range(5):
        assert g.press("ctrl_l", mode=Mode.LIVE) == ()
        assert g.press("v", mode=Mode.LIVE, injected=True) == ()
        assert g.release("v", mode=Mode.LIVE, injected=True) == ()
    clock.step()
    assert g.release("ctrl_l", mode=Mode.LIVE) == (Action.TOGGLE_MUTE,)
    assert g.release("ctrl_l", mode=Mode.LIVE) == ()


def test_injected_release_does_not_release_real_key(keys):
    g, clock = keys
    g.press("ctrl_r")
    assert g.release("ctrl_r", injected=True) == ()
    assert g.is_down("ctrl_r")


def test_cancel_only_active_and_reset_clears_lost_keys(keys):
    g, clock = keys
    assert g.press("esc") == ()
    g.release("esc")
    g.press("ctrl_l", mode=Mode.LIVE)
    assert g.press("esc", mode=Mode.LIVE) == (Action.CANCEL,)
    g.reset()
    assert not g.keys_down
    assert g.release("ctrl_l", mode=Mode.LIVE) == ()
    assert g.release("esc", mode=Mode.LIVE) == ()


def test_cancel_can_be_disabled():
    g = DictationGestures(cancel_key=None)
    assert g.press("esc", mode=Mode.LIVE) == ()


def test_active_long_control_press_does_not_toggle(keys):
    g, clock = keys
    g.press("ctrl_l", mode=Mode.LIVE)
    clock.step(1)
    assert g.release("ctrl_l", mode=Mode.LIVE) == ()


def test_remapped_function_keys_support_all_gestures():
    clock = Clock()
    g = DictationGestures(primary_key="f8", secondary_key="f9", clock=clock)
    assert tap(g, clock, "f8") == ()
    assert tap(g, clock, "f8") == (Action.START_HANDS_FREE,)
    assert tap(g, clock, "f9", Mode.HANDS_FREE) == (Action.TOGGLE_MUTE,)
    g.reset()
    g.press("f9")
    g.press("f8")
    g.release("f9")
    assert g.release("f8") == (Action.START_LIVE,)


@pytest.mark.parametrize("kwargs", [
    {"primary_key": "ctrl_l", "secondary_key": "ctrl_l"},
    {"primary_key": "a"}, {"secondary_key": "alt_r"},
    {"hold_seconds": 0}, {"double_tap_seconds": float("nan")},
])
def test_invalid_gesture_config_rejected(kwargs):
    with pytest.raises(ValueError):
        DictationGestures(**kwargs)


def pcm(seconds, value=2000, rate=8000):
    result = array("h", [value] * round(rate * seconds))
    if sys.byteorder != "little":
        result.byteswap()
    return result.tobytes()


def test_silence_alone_never_creates_requests_and_memory_is_bounded():
    s = SpeechSegmenter(sample_rate=8000)
    for _ in range(50):
        assert s.feed(pcm(1, 0)) == []
        assert s.buffered_bytes <= 8000 * 2 * 0.22
    assert s.flush() == []
    assert s.buffered_bytes == 0


def test_speech_is_preserved_byte_for_byte():
    s = SpeechSegmenter(sample_rate=8000)
    audio = pcm(1)
    assert s.feed(audio) == []
    assert s.flush() == [audio]


def test_silence_endpoint_emits_without_waiting_for_finish():
    s = SpeechSegmenter(sample_rate=8000)
    voice = pcm(0.5)
    assert s.feed(voice) == []
    assert s.feed(pcm(0.64, 0)) == []
    output = s.feed(pcm(0.02, 0))
    assert output == [voice + pcm(0.66, 0)]
    assert s.flush() == []


def test_short_noise_is_filtered():
    s = SpeechSegmenter(sample_rate=8000)
    assert s.feed(pcm(0.1)) == []
    assert s.feed(pcm(0.7, 0)) == []
    assert s.flush() == []


def test_quiet_samples_below_threshold_are_not_transcribed():
    s = SpeechSegmenter(sample_rate=8000)
    assert s.feed(pcm(1, 50)) == []
    assert s.flush() == []


def test_preroll_retains_only_recent_silence():
    s = SpeechSegmenter(sample_rate=8000)
    assert s.feed(pcm(2, 0)) == []
    voice = pcm(0.5)
    assert s.feed(voice) == []
    assert s.flush() == [pcm(0.2, 0) + voice]


def test_long_speech_has_no_overlap_or_dropped_short_continuation():
    s = SpeechSegmenter(sample_rate=8000)
    # Distinct sample values expose any repeated frame at chunk boundaries.
    audio = b"".join(pcm(0.1, 2000 + n) for n in range(161))
    output = s.feed(audio) + s.flush()
    assert len(output) == 3
    assert all(len(chunk) <= 8 * 8000 * 2 for chunk in output)
    assert b"".join(output) == audio


def test_short_last_word_after_cap_in_interword_silence_is_not_lost():
    s = SpeechSegmenter(sample_rate=8000)
    audio = pcm(7.96) + pcm(0.08, 0) + pcm(0.10, 3000)
    output = s.feed(audio) + s.flush()
    assert b"".join(output) == audio


def test_continuation_short_noise_after_full_silence_is_not_accepted():
    s = SpeechSegmenter(sample_rate=8000)
    first = pcm(7.96) + pcm(0.04, 0)
    assert s.feed(first) == [first]
    assert s.feed(pcm(0.70, 0) + pcm(0.10, 3000)) == []
    assert s.flush() == []


def test_unaligned_preroll_cap_preserves_sample_order():
    s = SpeechSegmenter(sample_rate=8000, pre_roll_seconds=0.193)
    silence = pcm(0.4, 0)
    voice = pcm(9.1)
    output = s.feed(silence + voice) + s.flush()
    assert all(len(chunk) <= 8 * 8000 * 2 for chunk in output)
    assert b"".join(output) == pcm(0.193, 0) + voice


def test_arbitrary_odd_sized_transport_chunks_preserve_pcm():
    s = SpeechSegmenter(sample_rate=8000)
    audio = pcm(1)
    output = []
    for offset in range(0, len(audio), 113):
        output.extend(s.feed(audio[offset:offset + 113]))
    output.extend(s.flush())
    assert b"".join(output) == audio


def test_incomplete_pcm_sample_rejected_and_cleared():
    s = SpeechSegmenter(sample_rate=8000)
    s.feed(pcm(0.5) + b"x")
    with pytest.raises(ValueError, match="Incomplete PCM16"):
        s.flush()
    assert s.buffered_bytes == 0


def test_mute_boundary_flush_and_reset_does_not_replay_old_audio():
    s = SpeechSegmenter(sample_rate=8000)
    first = pcm(0.5, 2000)
    second = pcm(0.5, 3000)
    assert s.feed(first) == []
    assert s.flush() == [first]
    # Caller does not feed the segmenter at all while muted.
    assert s.feed(second) == []
    assert s.flush() == [second]


def test_reset_discards_everything_without_transcription():
    s = SpeechSegmenter(sample_rate=8000)
    s.feed(pcm(0.5))
    assert s.speech_active
    s.reset()
    assert not s.speech_active
    assert s.buffered_bytes == 0
    assert s.flush() == []


@pytest.mark.parametrize("kwargs", [
    {"sample_rate": 0}, {"threshold_dbfs": float("nan")},
    {"silence_seconds": 0}, {"max_chunk_seconds": 0.1},
    {"pre_roll_seconds": -1}, {"pre_roll_seconds": 8},
])
def test_invalid_segmenter_config_rejected(kwargs):
    with pytest.raises(ValueError):
        SpeechSegmenter(**kwargs)
