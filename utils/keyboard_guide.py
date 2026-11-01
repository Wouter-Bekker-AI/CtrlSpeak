"""Side-effect-free keyboard-guide content and deterministic playback.

The guide describes gestures; it never listens for them or invokes a runtime
action. All transcript strings are fixed examples, not user transcript data.
"""
from __future__ import annotations

from dataclasses import dataclass
from utils.dictation_modes import DictationMode, MODE_LABELS, SUPPORTED_GESTURE_KEYS

KEY_LABELS = {"ctrl_r": "Right Ctrl", "ctrl_l": "Left Ctrl", "f8": "F8", "f9": "F9", "f10": "F10"}
EXAMPLE = "Let's make this simple."


def effective_keys(primary: str, secondary: str) -> tuple[str, str]:
    """Match the runtime's default fallback for malformed saved key pairs."""
    if primary not in SUPPORTED_GESTURE_KEYS or secondary not in SUPPORTED_GESTURE_KEYS or primary == secondary:
        return "ctrl_r", "ctrl_l"
    return primary, secondary


@dataclass(frozen=True)
class GuideStep:
    title: str
    detail: str
    keys_down: frozenset[str] = frozenset()
    state: str = "ready"
    text: str = ""
    duration_ms: int = 1600


@dataclass(frozen=True)
class GuideLesson:
    mode: str
    title: str
    summary: str
    shortcut: str
    steps: tuple[GuideStep, ...]
    safety: str
    availability: str


def build_lesson(mode: str, primary: str = "ctrl_r", secondary: str = "ctrl_l",
                 *, windows: bool = True, live_enabled: bool = False) -> GuideLesson:
    selected = DictationMode(mode)
    if primary not in SUPPORTED_GESTURE_KEYS or secondary not in SUPPORTED_GESTURE_KEYS or primary == secondary:
        raise ValueError("Guide keys must be supported and distinct")
    p, s = KEY_LABELS[primary], KEY_LABELS[secondary]
    def step(title, detail, down=(), state="ready", text="", duration=1600):
        return GuideStep(title, detail, frozenset(down), state, text, duration)
    if selected == DictationMode.HOLD:
        steps = (
            step("Focus your destination", "Click the text field or terminal prompt where the result should go."),
            step(f"Hold {p} and speak", "Keep the key down for your whole message.", (primary,), "recording", duration=2600),
            step(f"Release {p}", "Recording stops. CtrlSpeak transcribes the complete recording.", state="processing"),
            step("Your words arrive", "The finished transcript is inserted into the destination.", state="done", text=EXAMPLE),
        )
        return GuideLesson(selected.value, MODE_LABELS[selected], "One message. Hold, speak, release.",
            f"Hold {p} → release to transcribe", steps,
            "No mute toggle in Hold mode. Release to finish, or use Cancel active in the panel. "
            "Do not combine the recording key with editing shortcuts.",
            "Available on Windows and supported X11 desktops.")
    start = (step("Focus your destination", "Choose a blank editor for your first test."),)
    if selected == DictationMode.HANDS_FREE:
        start += (
            step(f"Double-tap {p} · first tap", "Two quick, isolated taps; do not hold either tap.", (primary,), duration=120),
            step("Release between taps", "Tap again promptly (within 300 ms).", duration=100),
            step(f"Double-tap {p} · second tap", "Release the second tap to latch recording on.", (primary,), duration=120),
            step("Hands free. Keep speaking.", "Recording continues without holding a key. Text arrives only when you finish.", state="recording", duration=2400),
        )
    else:
        start += (
            step(f"Press {s} + {p}", "Press both keys together, with no other keys. Either press order works.", (secondary, primary), duration=1200),
            step("Release BOTH keys", "Live starts only after both are released. Enable Live in Capture first.", state="recording", duration=1400),
            step("Speak naturally", "Stable phrases arrive after speech pauses, not word by word. The status capsule stays at the top.", state="recording", text="Let's make", duration=2000),
            step("Phrases arrive in place", "The selected route handles each phrase. OpenAI routes may incur repeated API charges.", state="recording", text=EXAMPLE, duration=1800),
        )
    steps = start + (
        step(f"Tap {s} to mute", "An isolated tap pauses microphone retention and text delivery.", (secondary,), "recording", duration=120),
        step("MUTED · safe to talk aside", "The amber, crossed-microphone state means paused. Muted speech is not replayed later.", state="muted", text=EXAMPLE if selected == DictationMode.LIVE else "", duration=2200),
        step(f"Tap {s} again", "Release the tap to resume the same session.", (secondary,), "muted", duration=120),
        step("Listening again", "Continue your message. The microphone indicator returns to listening.", state="recording", text=EXAMPLE if selected == DictationMode.LIVE else "", duration=1800),
    )
    if selected == DictationMode.LIVE:
        steps += (step("Edit safely", "Physical editing pauses automatic delivery. Late results can be held; review Copy pending / Discard pending in Capture.", state="editing", text=EXAMPLE, duration=2200),)
    steps += (
        step(f"Tap {p} to finish", "One short, isolated tap ends this hands-free/live session.", (primary,), "recording", duration=120),
        step("Finish pending processing", "Listening stops. Already queued phrases finish safely; inspect any held text before pasting.", state="processing"),
        step("Finished", "Start a new session when you are ready. CtrlSpeak does not press Enter for live dictation.", state="done", text=EXAMPLE),
    )
    availability = "Available on Windows and supported X11 desktops."
    if selected == DictationMode.LIVE:
        availability = ("Live is enabled on this Windows client." if windows and live_enabled else
                        "Live is off. Enable it in Capture after reading its safety notice." if windows else
                        "Live currently requires Windows. This is an instructional preview only.")
    return GuideLesson(selected.value, MODE_LABELS[selected],
        "Latch on, pause when needed, finish when ready." if selected == DictationMode.HANDS_FREE else
        "Stable phrases, right where you are typing.",
        f"Double-tap {p} · tap {s} to mute · tap {p} to finish" if selected == DictationMode.HANDS_FREE else
        f"{s} + {p}, release both · tap {s} to mute · tap {p} to finish", steps,
        "Use isolated taps. Ctrl+C/V and other editing chords are not mute/finish gestures. "
        "Esc cancels hands-free/live; it does not undo text already inserted. "
        "AnyDesk cannot expose every remote caret change. Mute before editing and inspect held text.", availability)


class GuidePlayback:
    """One finite demonstration. No timer/thread, real keys or runtime adapters."""
    def __init__(self, lesson: GuideLesson):
        self.lesson = lesson
        self.index = 0
        self.elapsed_ms = 0.0
        self.playing = False

    @property
    def step(self) -> GuideStep:
        return self.lesson.steps[self.index]

    @property
    def progress(self) -> float:
        return min(1.0, self.elapsed_ms / self.step.duration_ms)

    def restart(self, *, play: bool = True) -> None:
        self.index, self.elapsed_ms, self.playing = 0, 0.0, play

    def seek(self, index: int) -> None:
        self.index = max(0, min(len(self.lesson.steps)-1, index))
        self.elapsed_ms, self.playing = 0.0, False

    def advance(self, milliseconds: float) -> None:
        import math
        if not math.isfinite(milliseconds) or milliseconds < 0:
            raise ValueError("Playback interval must be finite and non-negative")
        if not self.playing:
            return
        self.elapsed_ms += milliseconds
        while self.elapsed_ms >= self.step.duration_ms:
            if self.index == len(self.lesson.steps)-1:
                self.elapsed_ms, self.playing = float(self.step.duration_ms), False
                break
            self.elapsed_ms -= self.step.duration_ms
            self.index += 1


def guide_self_test() -> dict[str, object]:
    # Exercise the packaged widget import too, without creating a Tk display.
    from utils.keyboard_guide_ui import KeyboardGuide
    if not callable(KeyboardGuide):
        raise RuntimeError("Keyboard guide widget was not packaged")
    for mode in DictationMode:
        lesson = build_lesson(mode.value)
        playback = GuidePlayback(lesson)
        playback.restart()
        playback.advance(sum(s.duration_ms for s in lesson.steps))
        if playback.playing or playback.step.state != "done":
            raise RuntimeError("Keyboard guide playback failed")
    return {"ok": True, "modes": 3, "simulation_only": True}
