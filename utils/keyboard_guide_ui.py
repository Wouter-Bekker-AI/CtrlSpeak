"""Native vector keyboard tutorial; no capture, hooks or synthetic input."""
from __future__ import annotations

import math
import sys
import time
import tkinter as tk
from tkinter import ttk
from utils.keyboard_guide import GuidePlayback, KEY_LABELS, build_lesson

INK, CARD, ALT, OUTLINE = "#0B1117", "#17222C", "#1C2934", "#30424F"
TEXT, MUTED, CYAN, MINT, AMBER = "#F3F7FA", "#91A4B1", "#49D7E8", "#71E6BA", "#F5C56B"
MODE_TITLES = (("hold", "01  Hold to Talk"), ("hands_free", "02  Hands-free"), ("live", "03  Live Dictation"))


class KeyboardGuide(ttk.Frame):
    def __init__(self, parent, *, primary="ctrl_r", secondary="ctrl_l", reduced_motion=False,
                 live_enabled=False, windows=None):
        super().__init__(parent, style="MS.Surface.TFrame")
        self.primary, self.secondary = primary, secondary
        self.reduced_motion = bool(reduced_motion)
        self.live_enabled = bool(live_enabled)
        self.windows = sys.platform.startswith("win") if windows is None else windows
        self.playback = GuidePlayback(build_lesson("hold", primary, secondary, windows=self.windows,
                                                 live_enabled=self.live_enabled))
        self._visible = False
        self._job = None
        self._last_tick = time.monotonic()
        self._last_identity = None
        self._closed = False
        self._seen = False
        self.mode_buttons = {}
        ttk.Label(self, text="Your keys. Your flow.", style="MS.Title.TLabel", background="#111A22").pack(anchor="w")
        ttk.Label(self, text="Learn the gestures without recording a word.", style="MS.Surface.TLabel",
                  foreground=MUTED).pack(anchor="w", pady=(5, 16))
        modes = ttk.Frame(self, style="MS.Surface.TFrame")
        modes.pack(fill="x", pady=(0, 14))
        for column, (mode, title) in enumerate(MODE_TITLES):
            modes.columnconfigure(column, weight=1, uniform="guide-modes")
            button = ttk.Button(modes, text=title, style="MS.TButton", command=lambda m=mode: self.select_mode(m))
            button.grid(row=0, column=column, sticky="ew", padx=(0 if column==0 else 5, 0))
            self.mode_buttons[mode] = button
        card = ttk.Frame(self, style="MS.OutlinedCard.TFrame", padding=(20, 16))
        card.pack(fill="both", expand=True)
        self.title_var, self.summary_var, self.shortcut_var = (tk.StringVar() for _ in range(3))
        self.step_var, self.detail_var, self.counter_var = (tk.StringVar() for _ in range(3))
        self.availability_var, self.safety_var = tk.StringVar(), tk.StringVar()
        self.title_label = ttk.Label(card, textvariable=self.title_var, style="MS.Section.TLabel")
        self.title_label.pack(anchor="w")
        self.summary_label = ttk.Label(card, textvariable=self.summary_var, style="MS.CardMuted.TLabel")
        self.summary_label.pack(anchor="w", pady=(3, 10))
        self.shortcut_label = ttk.Label(card, textvariable=self.shortcut_var, style="MS.Card.TLabel", foreground=CYAN)
        self.shortcut_label.pack(anchor="w")
        self.canvas = tk.Canvas(card, width=620, height=252, bg=CARD, highlightthickness=0,
                                borderwidth=0, takefocus=False)
        self.canvas.pack(fill="x", pady=(8, 8))
        self.canvas.bind("<Configure>", self._resize)
        row = ttk.Frame(card, style="MS.Card.TFrame")
        row.pack(fill="x")
        self.step_label = ttk.Label(row, textvariable=self.step_var, style="MS.Section.TLabel")
        self.step_label.pack(side="left", fill="x", expand=True)
        ttk.Label(row, textvariable=self.counter_var, style="MS.CardMuted.TLabel").pack(side="right", padx=(12, 0))
        self.detail_label = ttk.Label(card, textvariable=self.detail_var, style="MS.Card.TLabel", justify="left")
        self.detail_label.pack(anchor="w", pady=(8, 10))
        controls = ttk.Frame(card, style="MS.Card.TFrame")
        controls.pack(fill="x")
        self.previous_button = ttk.Button(controls, text="← Previous", style="MS.Compact.TButton", command=lambda: self.seek(-1))
        self.previous_button.pack(side="left")
        self.play_button = ttk.Button(controls, text="Play demo", style="MS.Primary.TButton", command=self.toggle_play)
        self.play_button.pack(side="left", padx=8)
        self.next_button = ttk.Button(controls, text="Next →", style="MS.Compact.TButton", command=lambda: self.seek(1))
        self.next_button.pack(side="left")
        ttk.Button(controls, text="Replay", style="MS.Compact.TButton", command=self.replay).pack(side="right")
        # Put instructions and playback controls above the illustration: they
        # remain reachable on short displays, with safety notes scrollable below.
        row.pack_configure(before=self.canvas, pady=(12, 0))
        self.detail_label.pack_configure(before=self.canvas)
        controls.pack_configure(before=self.canvas, pady=(0, 4))
        self.motion_label = ttk.Label(card, style="MS.CardMuted.TLabel")
        self.motion_label.pack(anchor="w", pady=(10, 0))
        self.availability_label = ttk.Label(self, textvariable=self.availability_var, style="MS.Surface.TLabel", foreground=AMBER, justify="left")
        self.availability_label.pack(anchor="w", pady=(12, 6))
        ttk.Label(self, text="SAFE SHORTCUTS", style="MS.Surface.TLabel", foreground=CYAN).pack(anchor="w", pady=(4, 4))
        self.safety_label = ttk.Label(self, textvariable=self.safety_var, style="MS.Surface.TLabel", foreground=MUTED, justify="left")
        self.safety_label.pack(anchor="w")
        self.bind("<Destroy>", self._destroyed, add="+")
        self._render()

    def configure_keys(self, primary, secondary, *, reduced_motion=None, live_enabled=None):
        reduced = self.reduced_motion if reduced_motion is None else bool(reduced_motion)
        enabled = self.live_enabled if live_enabled is None else bool(live_enabled)
        # Validate before changing an existing working preview.
        lesson = build_lesson(self.playback.lesson.mode, primary, secondary, windows=self.windows, live_enabled=enabled)
        changed = (primary, secondary, enabled) != (self.primary, self.secondary, self.live_enabled)
        self.primary, self.secondary, self.live_enabled = primary, secondary, enabled
        self.reduced_motion = reduced
        if changed:
            self.playback = GuidePlayback(lesson)
            self._last_identity = None
        if reduced:
            self.playback.playing = False
        self._render()
        self._schedule()

    def select_mode(self, mode):
        lesson = build_lesson(mode, self.primary, self.secondary, windows=self.windows, live_enabled=self.live_enabled)
        self.playback = GuidePlayback(lesson)
        self._last_identity = None
        self._last_tick = time.monotonic()
        self.playback.playing = self._visible and not self.reduced_motion
        self._render()
        self._schedule()

    def set_visible(self, visible):
        self._visible = bool(visible)
        if visible and not self._seen:
            self._seen = True
            self.playback.restart(play=not self.reduced_motion)
        if not visible:
            self.playback.playing = False
        self._last_tick = time.monotonic()
        self._render()
        self._schedule()

    def toggle_play(self):
        if self.reduced_motion:
            return
        if self.playback.progress == 1 and self.playback.index == len(self.playback.lesson.steps)-1:
            self.playback.restart()
        else:
            self.playback.playing = not self.playback.playing
        self._last_tick = time.monotonic()
        self._render()
        self._schedule()

    def replay(self):
        self.playback.restart(play=not self.reduced_motion)
        self._last_tick = time.monotonic()
        self._last_identity = None
        self._render()
        self._schedule()

    def seek(self, delta):
        self.playback.seek(self.playback.index+delta)
        self._render()
        self._schedule()

    def _schedule(self):
        if self._job is not None:
            self.after_cancel(self._job)
            self._job = None
        if not self._closed and self._visible and self.playback.playing and not self.reduced_motion:
            self._job = self.after(50, self._tick)

    def _tick(self):
        self._job = None
        if self._closed or not self._visible:
            return
        now = time.monotonic()
        self.playback.advance((now-self._last_tick)*1000)
        self._last_tick = now
        self._render()
        self._schedule()

    def _destroyed(self, event):
        if event.widget is self:
            self.shutdown()

    def shutdown(self):
        self._closed, self._visible, self.playback.playing = True, False, False
        if self._job is not None:
            try:
                self.after_cancel(self._job)
            except tk.TclError:
                pass
            self._job = None

    def _resize(self, event):
        if self._closed:
            return
        width = max(240, event.width-8)
        for label in (self.summary_label, self.shortcut_label, self.detail_label, self.safety_label,
                      self.availability_label):
            label.configure(wraplength=width)
        self.step_label.configure(wraplength=max(180, width-90))
        self._draw()

    def _render(self):
        if self._closed:
            return
        lesson, step = self.playback.lesson, self.playback.step
        identity = (lesson, self.playback.index)
        if identity != self._last_identity:
            self._last_identity = identity
            self.title_var.set(lesson.title)
            self.summary_var.set(lesson.summary)
            self.shortcut_var.set(lesson.shortcut)
            self.step_var.set(step.title)
            self.detail_var.set(step.detail)
            self.counter_var.set(f"{self.playback.index+1:02d} / {len(lesson.steps):02d}")
            self.availability_var.set(lesson.availability)
            self.safety_var.set(lesson.safety)
        for mode, button in self.mode_buttons.items():
            button.configure(style="MS.Primary.TButton" if mode == lesson.mode else "MS.TButton")
        self.play_button.configure(text="Pause" if self.playback.playing else "Play demo",
                                   state="disabled" if self.reduced_motion else "normal")
        self.previous_button.configure(state="disabled" if self.playback.index == 0 else "normal")
        self.next_button.configure(state="disabled" if self.playback.index == len(lesson.steps)-1 else "normal")
        self.motion_label.configure(text="Reduced motion · use Previous / Next for still frames." if self.reduced_motion else
                                    "DEMONSTRATION ONLY · no microphone, no key presses, no API requests.")
        self._draw()

    def _draw(self):
        c, step = self.canvas, self.playback.step
        c.delete("all")
        width = max(320, c.winfo_width())
        scale = min(1.35, width/720)
        font = max(8, round(10*scale))
        left, right = 4, width-4
        colour = AMBER if step.state in {"muted", "editing"} else MINT if step.state == "done" else CYAN
        c.create_rectangle(left, 0, right, 77, fill=INK, outline=OUTLINE)
        state_text = {"ready":"READY", "recording":"LISTENING", "muted":"MUTED", "editing":"PAUSED FOR EDITING",
                      "processing":"PROCESSING", "done":"FINISHED"}[step.state]
        c.create_oval(17, 16, 25, 24, fill=colour, outline="")
        c.create_text(35, 20, anchor="w", text=state_text, fill=colour, font=("Segoe UI Semibold", 9))
        # Crossed mic in paused states, drawn in vectors rather than a font glyph.
        x = right-22
        c.create_oval(x-4, 13, x+4, 27, outline=colour, width=2)
        c.create_arc(x-8, 19, x+8, 33, start=180, extent=180, style="arc", outline=colour, width=2)
        c.create_line(x, 33, x, 38, x-5, 38, x+5, 38, fill=colour, width=2)
        if step.state in {"muted", "editing"}:
            c.create_line(x-10, 10, x+10, 37, fill=AMBER, width=2, tags="guide-muted-mic")
        elif step.state == "recording":
            phase = self.playback.elapsed_ms/180 if self.playback.playing else 0
            for i in range(24):
                amplitude = 3+10*abs(math.sin(i*.63+phase)*math.cos(i*.21))
                pos = width*.35+i*width*.012
                c.create_line(pos, 25-amplitude, pos, 25+amplitude, fill=CYAN, width=2, tags="guide-demo-wave")
        sample = step.text or ("Recording stays private until you finish…" if step.state=="recording" and self.playback.lesson.mode!="live" else
                               "Speech is not retained while muted." if step.state=="muted" else "Example destination · not a real text field")
        c.create_text(17, 55, anchor="w", text=sample, fill=TEXT if step.text else MUTED,
                      font=("Segoe UI", 10), width=max(200, width-35))
        # Schematic keyboard rows retain the actual left/right Ctrl positions and
        # function-key alternatives, while avoiding unreadable miniature keys.
        top = (("esc","Esc",1), (None,"",.6), ("f8","F8",1), ("f9","F9",1), ("f10","F10",1))
        bottom = (("ctrl_l","Left Ctrl",1.6), (None,"Alt",1), (None,"Space",3.8),
                  (None,"Alt",1), ("ctrl_r","Right Ctrl",1.6))
        for y, row in ((93, top), (151, bottom)):
            total = sum(k[2] for k in row)
            unit = (width-16-6*(len(row)-1))/total
            pos = 8
            for key, label, factor in row:
                key_width = factor*unit
                if label:
                    down = key in step.keys_down
                    relevant = key in (self.primary, self.secondary)
                    fill = colour if down else ALT
                    outline = colour if down else OUTLINE
                    c.create_rectangle(pos, y+(3 if down else 0), pos+key_width, y+42,
                        fill=fill, outline=outline, width=2 if relevant else 1, tags=("guide-key",key or "other"))
                    c.create_text(pos+key_width/2, y+21, text=label, fill=INK if down else TEXT if relevant or key=="esc" else MUTED,
                                  font=("Segoe UI Semibold", font))
                    if relevant:
                        c.create_rectangle(pos+6, y+36, pos+key_width-6, y+38, fill=INK if down else CYAN, outline="")
                pos += key_width+6
        c.create_text(8, 216, anchor="w", text=f"RECORD / FINISH  {KEY_LABELS[self.primary]}    ·    MUTE / MODIFIER  {KEY_LABELS[self.secondary]}",
                      fill=MUTED, font=("Segoe UI", max(8, font-1)), width=width-16)
        total = len(self.playback.lesson.steps)
        segment = (width-16)/total
        for i in range(total):
            c.create_rectangle(8+i*segment, 241, 8+(i+1)*segment-3, 245,
                fill=colour if i<=self.playback.index else OUTLINE, outline="", tags="guide-timeline")
