"""Continuous-session tests use fake adapters: never record, type, or call APIs."""
from array import array
from dataclasses import dataclass, field
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from utils.dictation_runtime import DeliveryCancelled, DictationSession


pytestmark = pytest.mark.core_headless


@dataclass(frozen=True)
class Result:
    text: str
    metadata: dict = field(default_factory=dict)


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def step(self, seconds):
        self.value += seconds


def pcm(seconds=0.3, value=2000):
    samples = array("h", [value] * round(44100 * seconds))
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes()


def until(predicate, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate(), "Timed out waiting for the fake session to settle"


def enqueue(session, data, *, epoch=None):
    with session.lock:
        session._enqueue_locked(data, epoch)


@pytest.fixture
def make_session():
    sessions = []

    def make(mode="live", *, transcribe=None, deliver=None):
        clock = Clock()
        calls, delivered, levels = [], [], []
        target = [101]
        capture_started = threading.Event()

        def capture(stop, feed, token):
            capture_started.set()
            while not stop.wait(0.01):
                pass

        def fake_transcribe(data, cancel):
            calls.append(data)
            if transcribe is not None:
                return transcribe(data, cancel)
            return Result(data.decode("ascii", errors="replace"))

        def fake_deliver(result, guard):
            assert not guard.is_set(), "Runtime delivered into an invalidated target"
            if deliver is not None:
                deliver(result, guard)
            delivered.append(result)

        session = DictationSession(
            mode, capture=capture, transcribe=fake_transcribe,
            deliver=fake_deliver, target=lambda: target[0],
            level=levels.append, clock=clock,
        )
        sessions.append(session)
        return SimpleNamespace(session=session, clock=clock, calls=calls,
                               delivered=delivered, levels=levels, target=target,
                               capture_started=capture_started)

    yield make
    # Every fake child is cancelled, joined and asserted reaped even on failure.
    for session in sessions:
        session.abort()
    for session in sessions:
        for thread in (session.reader, session.worker):
            if thread is not None:
                thread.join(timeout=2.0)
                assert not thread.is_alive(), "Fake dictation worker leaked"


def test_hands_free_mute_excludes_audio_but_keeps_prior_recording(make_session):
    h = make_session("hands_free")
    s = h.session
    before = pcm(value=1000)
    after = pcm(value=3000)
    s.feed(before, s.capture_token())
    pre_mute_read = s.capture_token()
    assert s.toggle_mute() is True
    s.feed(pcm(value=9000), pre_mute_read)
    s.feed(pcm(value=10000), s.capture_token())
    assert bytes(s.frames) == before
    assert h.levels == [before]
    muted_read = s.capture_token()
    assert s.toggle_mute() is False
    # Even after unmute, a read that began while muted must be rejected.
    s.feed(pcm(value=11000), muted_read)
    s.feed(after, s.capture_token())
    assert bytes(s.frames) == before + after
    assert h.levels == [before, after]


def test_read_spanning_mute_and_unmute_is_rejected_by_audio_epoch(make_session):
    h = make_session("hands_free")
    token = h.session.capture_token()
    h.session.toggle_mute()
    h.session.toggle_mute()
    assert h.session.muted is False
    h.session.feed(pcm(), token)
    assert not h.session.frames
    assert not h.levels


def test_live_mute_flushes_only_pre_mute_speech_and_clears_preroll(make_session):
    h = make_session()
    s = h.session
    audio = pcm()
    s.feed(audio, s.capture_token())
    assert s.segmenter.speech_active
    s.toggle_mute()
    assert len(s.jobs) == 1
    assert s.jobs[0].pcm == audio
    assert s.segmenter.buffered_bytes == 0
    s.feed(pcm(value=9000), s.capture_token())
    assert len(s.jobs) == 1
    assert s.segmenter.buffered_bytes == 0


def test_manual_mute_never_auto_resumes_after_keyboard_or_silence(make_session):
    h = make_session()
    s = h.session
    s.toggle_mute()
    s.physical_edit()
    h.clock.step(60)
    s.set_keys_held(False)
    s.feed(pcm(), s.capture_token())
    s.check_target()
    assert s.muted is True
    assert not s.jobs
    assert s.segmenter.buffered_bytes == 0
    assert not h.levels


def test_editing_pauses_capture_until_settled_and_resumes_same_target(make_session):
    h = make_session()
    s = h.session
    initial_epoch = s.epoch
    s.physical_edit()
    assert s.editing and s.epoch > initial_epoch
    h.clock.step(0.59)
    s.feed(pcm(), s.capture_token())
    assert not h.levels
    h.clock.step(0.02)
    s.feed(pcm(), s.capture_token())
    assert not s.editing
    assert len(h.levels) == 1
    assert s.segmenter.speech_active


def test_held_physical_key_blocks_insertion_and_capture_resume(make_session):
    h = make_session()
    s = h.session
    s.physical_edit()
    s.set_keys_held(True)
    h.clock.step(5)
    s.feed(pcm(), s.capture_token())
    assert not h.levels
    assert not s.delivery_allowed(s.epoch)
    s.set_keys_held(False)
    s.feed(pcm(), s.capture_token())
    assert h.levels


def test_editing_during_finish_still_invalidates_inflight_result(make_session):
    h = make_session()
    s = h.session
    old_epoch = s.epoch
    s.finish()
    s.physical_edit()
    assert s.epoch > old_epoch
    assert not s.delivery_allowed(old_epoch)


def test_changed_or_unknown_target_mutes_until_explicit_rearm(make_session):
    h = make_session()
    s = h.session
    old_epoch = s.epoch
    h.target[0] = 202
    s.check_target()
    assert s.muted
    assert not s.delivery_allowed(old_epoch)
    assert s.expected_target == 101
    assert s.toggle_mute() is False
    assert s.expected_target == 202
    assert s.delivery_allowed(s.epoch)
    h.target[0] = None
    s.check_target()
    assert s.muted
    assert not s.delivery_allowed(s.epoch)


def test_delivery_guard_observes_mute_edit_and_cancel_immediately(make_session):
    h = make_session()
    s = h.session
    guard = DeliveryCancelled(s, s.epoch)
    assert not guard.is_set()
    s.toggle_mute()
    assert guard.is_set()
    s.toggle_mute()
    assert guard.is_set()  # Old delivery never revives after unmute.
    current = DeliveryCancelled(s, s.epoch)
    assert not current.is_set()
    s.physical_edit()
    assert current.is_set()
    s.abort()
    assert DeliveryCancelled(s, s.epoch).is_set()


def test_requests_are_serial_and_output_remains_ordered(make_session):
    in_call = 0
    maximum = 0

    def transcribe(data, cancel):
        nonlocal in_call, maximum
        in_call += 1
        maximum = max(maximum, in_call)
        try:
            time.sleep(0.01)
            return Result(data.decode())
        finally:
            in_call -= 1

    h = make_session(transcribe=transcribe)
    s = h.session
    s.start()
    enqueue(s, b"first")
    enqueue(s, b"second")
    until(lambda: len(h.delivered) == 2)
    s.finish()
    assert s.done.wait(2)
    assert maximum == 1
    assert h.calls == [b"first", b"second"]
    assert [result.text for result in h.delivered] == ["first", "second"]
    assert "".join(s.committed) == "first second "


def test_result_arriving_after_edit_is_held_not_inserted_at_new_cursor(make_session):
    entered, release = threading.Event(), threading.Event()

    def transcribe(data, cancel):
        entered.set()
        while not release.wait(0.01):
            if cancel.is_set():
                raise RuntimeError("Fake request cancelled")
        return Result("earlier phrase")

    h = make_session(transcribe=transcribe)
    s = h.session
    s.start()
    enqueue(s, b"old")
    assert entered.wait(1)
    s.physical_edit()
    h.clock.step(1)
    s.feed(b"", s.capture_token())
    assert not s.editing  # Even after editing settles, the old epoch stays invalid.
    release.set()
    until(lambda: bool(s.pending))
    assert not h.delivered
    assert s.pending_text() == "earlier phrase"
    assert s.muted


def test_pre_and_post_mute_jobs_do_not_cross_delivery_epochs(make_session):
    entered, release = threading.Event(), threading.Event()

    def transcribe(data, cancel):
        if data == b"before":
            entered.set()
            while not release.wait(0.01):
                if cancel.is_set():
                    raise RuntimeError("Fake request cancelled")
        return Result(data.decode())

    h = make_session(transcribe=transcribe)
    s = h.session
    s.start()
    enqueue(s, b"before")
    assert entered.wait(1)
    s.toggle_mute()
    release.set()
    until(lambda: s.pending_text() == "before")
    assert not h.delivered
    assert s.toggle_mute() is True  # Explicit disposition required for old text.
    s.discard_pending()
    assert s.toggle_mute() is False
    enqueue(s, b"after")
    until(lambda: len(h.delivered) == 1)
    assert h.calls == [b"before", b"after"]
    assert h.delivered[0].text == "after"


def test_manual_mute_blocks_new_request_dispatch(make_session):
    called = threading.Event()

    def transcribe(data, cancel):
        called.set()
        return Result(data.decode())

    h = make_session(transcribe=transcribe)
    s = h.session
    enqueue(s, b"queued before mute")
    s.toggle_mute()
    s.start()
    assert h.capture_started.wait(1)
    assert not called.wait(0.15)
    assert len(s.jobs) == 1
    assert s.toggle_mute() is False
    assert called.wait(1)
    until(lambda: bool(s.pending))
    assert not h.delivered  # Pre-mute epoch must not be sent to the resumed caret.


def test_backpressure_drains_old_jobs_without_recording_or_delivering(make_session):
    h = make_session()
    s = h.session
    for text in (b"one", b"two", b"three"):
        enqueue(s, text)
    assert len(s.jobs) == s.MAX_QUEUED == 3
    assert s.muted and s.backlog_paused
    s.feed(pcm(value=9000), s.capture_token())
    assert not h.levels
    s.start()
    until(lambda: len(s.pending) == 3)
    assert not s.jobs
    assert h.calls == [b"one", b"two", b"three"]
    assert s.pending == ["one", "two", "three"]
    assert not h.delivered
    assert s.toggle_mute() is True
    s.discard_pending()
    assert s.toggle_mute() is False
    enqueue(s, b"new")
    until(lambda: len(h.delivered) == 1)
    assert h.delivered[0].text == "new"


def test_queue_capacity_is_hard_bounded(make_session):
    h = make_session()
    for number in range(10):
        enqueue(h.session, str(number).encode())
    assert len(h.session.jobs) == 3
    assert h.session.number == 3
    assert h.session.muted


def test_finish_while_muted_holds_result_without_unmuting_or_typing(make_session):
    h = make_session("hands_free", transcribe=lambda data, cancel: Result("kept dictation"))
    s = h.session
    s.start()
    s.feed(pcm(), s.capture_token())
    s.toggle_mute()
    s.finish()
    assert s.done.wait(2)
    assert len(h.calls) == 1
    assert not h.delivered
    assert s.pending_text() == "kept dictation"
    assert s.muted


def test_cancel_before_start_never_calls_provider_or_insertion(make_session):
    h = make_session()
    s = h.session
    enqueue(s, b"never send")
    s.abort()
    s.start()
    assert s.done.wait(2)
    assert h.calls == []
    assert h.delivered == []
    assert not s.jobs
    assert not s.frames


def test_cancel_inflight_never_replays_or_delivers_late_result(make_session):
    entered, release = threading.Event(), threading.Event()

    def transcribe(data, cancel):
        entered.set()
        while not release.wait(0.01):
            if cancel.is_set():
                return Result("late result")
        return Result("late result")

    h = make_session(transcribe=transcribe)
    s = h.session
    s.start()
    enqueue(s, b"one")
    assert entered.wait(1)
    enqueue(s, b"two")
    s.abort()
    release.set()
    assert s.done.wait(2)
    assert h.calls == [b"one"]
    assert not h.delivered
    assert not s.pending


def test_finish_discards_late_microphone_reads(make_session):
    h = make_session("hands_free")
    token = h.session.capture_token()
    h.session.finish()
    h.session.feed(pcm(), token)
    assert not h.session.frames
    assert not h.levels


def test_transcription_failure_pauses_without_retrying_or_skipping_phrase(make_session):
    def fail(data, cancel):
        raise RuntimeError("Fake provider failure")

    h = make_session(transcribe=fail)
    s = h.session
    enqueue(s, b"one")
    enqueue(s, b"two")
    s.start()
    until(lambda: bool(s.error))
    assert s.muted
    assert not s.jobs
    assert h.calls == [b"one"]
    assert not h.delivered
    assert "No automatic retry" in s.error


def test_uncertain_insertion_is_held_never_replayed(make_session):
    def partial(result, guard):
        raise RuntimeError("Simulated partial OS insertion")

    h = make_session(deliver=partial)
    s = h.session
    s.start()
    enqueue(s, b"one")
    until(lambda: bool(s.pending))
    assert h.calls == [b"one"]
    assert not h.delivered
    assert "some text may already exist" in s.error
    assert s.pending_text() == "one"
    assert not s.committed


def test_live_flattening_preserves_truthful_s1_metadata(make_session):
    h = make_session(transcribe=lambda data, cancel: Result(
        "first\nsecond\tthird", {"client_cleanup_applied": True, "cleanup_model": "S1 Mini"}))
    s = h.session
    s.start()
    enqueue(s, b"one")
    until(lambda: bool(h.delivered))
    result = h.delivered[0]
    assert result.text == "first second third"
    assert result.metadata["client_cleanup_applied"] is True
    assert result.metadata["live_chunk"] is True


def test_recording_size_limit_mutes_before_accepting_excess_audio(make_session):
    h = make_session("hands_free")
    s = h.session
    s.MAX_AUDIO_BYTES = 10
    s.feed(b"a" * 8, s.capture_token())
    s.feed(b"b" * 8, s.capture_token())
    assert bytes(s.frames) == b"a" * 8
    assert s.muted
    assert "limit reached" in s.error


def test_session_time_limit_cannot_be_extended_by_muting(make_session):
    h = make_session("hands_free")
    s = h.session
    s.toggle_mute()
    h.clock.step(s.MAX_SESSION_SECONDS + 1)
    s.toggle_mute()
    s.feed(pcm(), s.capture_token())
    assert s.muted
    assert not s.frames
    assert "Session limit reached" in s.error


def test_pending_memory_cap_is_explicit_not_silent_data_loss(make_session):
    h = make_session()
    s = h.session
    s.MAX_PENDING_CHARS = 10
    s._hold_result("12345678")
    s._hold_result("abcdef")
    assert sum(map(len, s.pending)) == 10
    assert "truncated" in s.error
    assert s.finishing
    assert s.stop_capture.is_set()


def test_copy_acknowledgement_keeps_results_arriving_during_clipboard_io(make_session):
    s = make_session().session
    s._hold_result("first")
    snapshot = s.pending_snapshot()
    s._hold_result("late")
    s.acknowledge_copied(snapshot)
    assert s.pending_text() == "late"


def test_old_copy_acknowledgement_cannot_clear_new_identical_result(make_session):
    s = make_session().session
    s._hold_result("same")
    snapshot = s.pending_snapshot()
    s.discard_pending()
    s._hold_result("same")
    s.acknowledge_copied(snapshot)
    assert s.pending_text() == "same"


def test_backlog_cannot_resume_while_older_jobs_are_draining(make_session):
    s = make_session().session
    s.backlog_paused = s.muted = s.processing = True
    assert s.toggle_mute() is True
    assert "draining" in s.error
