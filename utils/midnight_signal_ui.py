"""Midnight Signal control centre and custom tray flyout for CtrlSpeak 0.7."""
from __future__ import annotations

import hashlib
import math
import queue
import threading
import time
from typing import Callable, Mapping, Optional, Sequence

import tkinter as tk
from tkinter import messagebox, ttk

from utils import system as sysmod
from utils.config_paths import settings, settings_lock, save_settings
from utils.languages import language_choices
from utils.midnight_overlay import active_monitor_bounds, display_scale, flyout_geometry
from utils.transcription_backend import (
    ApiBackendError,
    ApiTranscriptionClient,
    backend_display_name,
    get_backend_config,
    get_runtime_backend_config,
    get_session_openai_api_key,
    secure_storage_available,
)
from utils.ui_state import (
    ProviderTelemetry,
    UiPhase,
    active_route_label,
    format_latency_ms,
    providers_from_capabilities,
)
from utils.audio_cues import CueKind


INK = "#0B1117"
SURFACE = "#111A22"
CARD = "#17222C"
CARD_ALT = "#1C2934"
OUTLINE = "#30424F"
TEXT = "#F3F7FA"
MUTED = "#91A4B1"
CYAN = "#49D7E8"
MINT = "#71E6BA"
AMBER = "#F5C56B"
CORAL = "#FF7A79"

ROUTING_COLUMN_WEIGHTS = (3, 2)
PROVIDER_CARD_SPECS: tuple[tuple[str, str, str, str], ...] = (
    (
        "ubuntu-gpu-large-v3-turbo",
        "Ubuntu GPU",
        "Whisper large-v3-turbo",
        "Local · GPU accelerated",
    ),
    (
        "openai-gpt-transcribe",
        "OpenAI",
        "GPT Transcribe",
        "Cloud · caller-supplied key",
    ),
    (
        "gateway-tiny",
        "Gateway Tiny",
        "Whisper tiny",
        "Gateway · emergency fallback",
    ),
)


STRATEGY_LABELS: dict[str, str] = {
    "server-default": "Gateway default",
    "ubuntu-gpu-preferred": "GPU preferred",
    "openai-preferred": "OpenAI preferred",
    "ubuntu-gpu-only": "Ubuntu GPU only",
    "openai-only": "OpenAI only",
    "gateway-tiny-only": "Emergency tiny only",
}
STRATEGY_IDS = {label: value for value, label in STRATEGY_LABELS.items()}

_STATIC_STRATEGY_ORDER: dict[str, tuple[str, ...]] = {
    "ubuntu-gpu-preferred": (
        "ubuntu-gpu-large-v3-turbo",
        "openai-gpt-transcribe",
        "gateway-tiny",
    ),
    "openai-preferred": (
        "openai-gpt-transcribe",
        "ubuntu-gpu-large-v3-turbo",
        "gateway-tiny",
    ),
    "ubuntu-gpu-only": ("ubuntu-gpu-large-v3-turbo",),
    "openai-only": ("openai-gpt-transcribe",),
    "gateway-tiny-only": ("gateway-tiny",),
}
_KNOWN_PROVIDER_IDS = frozenset(spec[0] for spec in PROVIDER_CARD_SPECS)


def _canonical_card_provider_id(value: object) -> str:
    provider_id = str(value or "").strip()
    return "gateway-tiny" if provider_id == "nova-tiny-whisper" else provider_id


def provider_order_for_strategy(
    strategy_id: str,
    capabilities: Mapping[str, object] | None = None,
) -> tuple[str, ...] | None:
    """Return a truthful displayed chain, or ``None`` for unresolved default.

    ``server-default`` is a client pseudo-choice.  It is deliberately left
    unnumbered until the authenticated gateway identifies its default strategy
    and provider chain.
    """

    payload = capabilities or {}
    selected = str(strategy_id or "server-default").strip() or "server-default"
    advertised_id = selected
    if selected == "server-default":
        advertised_id = str(payload.get("default_strategy") or "").strip()
        if not advertised_id:
            return None
    raw_strategies = payload.get("strategies")
    if isinstance(raw_strategies, Sequence) and not isinstance(raw_strategies, (str, bytes)):
        for raw in raw_strategies:
            if not isinstance(raw, Mapping) or str(raw.get("id") or "") != advertised_id:
                continue
            raw_chain = raw.get("providers")
            if isinstance(raw_chain, Sequence) and not isinstance(raw_chain, (str, bytes)):
                ordered: list[str] = []
                for item in raw_chain:
                    provider_id = _canonical_card_provider_id(item)
                    if provider_id in _KNOWN_PROVIDER_IDS and provider_id not in ordered:
                        ordered.append(provider_id)
                if ordered:
                    return tuple(ordered)
            break
    if selected == "server-default":
        return None
    return _STATIC_STRATEGY_ORDER.get(selected)


def strategy_choices_from_capabilities(
    capabilities: Mapping[str, object],
) -> tuple[tuple[str, str], ...]:
    """Preserve the client-side Gateway default choice before advertised routes."""

    choices: list[tuple[str, str]] = [("server-default", STRATEGY_LABELS["server-default"])]
    raw_strategies = capabilities.get("strategies")
    if isinstance(raw_strategies, Sequence) and not isinstance(raw_strategies, (str, bytes)):
        for raw in raw_strategies:
            if not isinstance(raw, Mapping) or not raw.get("id"):
                continue
            strategy_id = str(raw["id"])
            label = STRATEGY_LABELS.get(
                strategy_id,
                strategy_id.replace("-", " ").title(),
            )
            if strategy_id not in {item[0] for item in choices}:
                choices.append((strategy_id, label))
    return tuple(choices)


def capability_config_identity(config: object) -> tuple[object, ...]:
    """Return a secret-safe identity for rejecting stale capability results."""

    token = getattr(config, "api_token", None)
    token_digest = hashlib.sha256(str(token).encode("utf-8")).digest() if token else None
    return (
        getattr(config, "backend", None),
        getattr(config, "api_url", None),
        token_digest,
        getattr(config, "provider_strategy", None),
        tuple(getattr(config, "allowed_output_languages", ()) or ()),
    )


def management_scroll_units(delta: object) -> int:
    """Normalize platform wheel deltas to a bounded canvas scroll step."""

    try:
        value = int(delta)
    except (TypeError, ValueError):
        return 0
    return -3 if value > 0 else 3 if value < 0 else 0


def management_viewport_requires_scroll(content_height: int, viewport_height: int) -> bool:
    return max(0, int(content_height)) > max(0, int(viewport_height))


def management_content_width(requested_width: int, viewport_width: int) -> int:
    """Keep narrow pages fluid while making wide controls horizontally reachable."""

    return max(1, int(requested_width), int(viewport_width))


def _paint_route_badge(canvas: tk.Canvas, number: int | None, *, size: int) -> None:
    canvas.delete("route-badge")
    inset = 2
    canvas.create_oval(
        inset,
        inset,
        size - inset,
        size - inset,
        outline=CYAN if number is not None else OUTLINE,
        width=1,
        tags=("route-badge",),
    )
    canvas.create_text(
        size / 2,
        size / 2,
        text=str(number) if number is not None else "—",
        fill=TEXT if number is not None else MUTED,
        font=("Segoe UI Semibold", 9 if size >= 28 else 8),
        tags=("route-badge",),
    )


def provider_signal_levels(seed: int, count: int = 17) -> tuple[float, ...]:
    """Return a deterministic instrument-like signal used by provider cards."""

    count = max(3, int(count))
    return tuple(
        0.16
        + 0.64
        * abs(
            math.sin((index + 1) * (0.68 + (seed % 5) * 0.07))
            * math.cos((index + seed + 2) * 0.31)
        )
        for index in range(count)
    )


def language_summary(codes: tuple[str, ...]) -> str:
    """Compact, non-secret label for the configured output-language policy."""

    normalized = tuple(code.strip().upper() for code in codes if code.strip())
    return "Automatic" if not normalized else " · ".join(normalized)


def _paint_provider_signal(
    canvas: tk.Canvas,
    *,
    seed: int,
    colour: str = OUTLINE,
) -> None:
    """Paint a tiny deterministic status trace; it never represents live audio."""

    canvas.delete("signal")
    levels = provider_signal_levels(seed)
    width = max(90, int(float(canvas.cget("width"))))
    height = max(20, int(float(canvas.cget("height"))))
    step = (width - 8) / max(1, len(levels) - 1)
    centre = height / 2
    for index, level in enumerate(levels):
        x = 4 + index * step
        half = 2 + level * (height * 0.34)
        canvas.create_line(
            x, centre - half, x, centre + half,
            fill=colour, width=1, tags=("signal",),
        )


def apply_midnight_signal_theme(window: tk.Misc) -> None:
    style = ttk.Style(window)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    window.configure(background=INK)
    style.configure("MS.Root.TFrame", background=INK)
    style.configure("MS.Surface.TFrame", background=SURFACE)
    style.configure("MS.Card.TFrame", background=CARD, relief="flat")
    style.configure(
        "MS.OutlinedCard.TFrame",
        background=CARD,
        relief="solid",
        borderwidth=1,
        bordercolor=OUTLINE,
        lightcolor=OUTLINE,
        darkcolor=OUTLINE,
    )
    style.configure(
        "MS.CardAlt.TFrame",
        background=CARD_ALT,
        relief="solid",
        borderwidth=1,
        bordercolor="#263946",
        lightcolor="#263946",
        darkcolor="#263946",
    )
    style.configure("MS.TLabel", background=INK, foreground=TEXT, font=("Segoe UI", 10))
    style.configure("MS.Surface.TLabel", background=SURFACE, foreground=TEXT, font=("Segoe UI", 10))
    style.configure("MS.Card.TLabel", background=CARD, foreground=TEXT, font=("Segoe UI", 10))
    style.configure("MS.CardMuted.TLabel", background=CARD, foreground=MUTED, font=("Segoe UI", 9))
    style.configure("MS.Hero.TLabel", background=INK, foreground=TEXT, font=("Segoe UI Semibold", 22))
    style.configure("MS.Title.TLabel", background=INK, foreground=TEXT, font=("Segoe UI Semibold", 15))
    style.configure("MS.Section.TLabel", background=CARD, foreground=TEXT, font=("Segoe UI Semibold", 12))
    style.configure("MS.Metric.TLabel", background=CARD, foreground=CYAN, font=("Segoe UI Semibold", 16))
    style.configure("MS.CompactMetric.TLabel", background=CARD, foreground=CYAN, font=("Segoe UI Semibold", 10))
    style.configure("MS.Number.TLabel", background=CARD, foreground=CYAN, font=("Segoe UI Semibold", 12))
    style.configure("MS.Ready.TLabel", background=INK, foreground=MINT, font=("Segoe UI Semibold", 9))
    style.configure("MS.Warn.TLabel", background=CARD, foreground=AMBER, font=("Segoe UI Semibold", 9))
    style.configure("MS.Error.TLabel", background=CARD, foreground=CORAL, font=("Segoe UI Semibold", 9))
    style.configure("MS.TButton", background=CARD_ALT, foreground=TEXT, bordercolor=OUTLINE, lightcolor=OUTLINE, darkcolor=OUTLINE, focuscolor=OUTLINE, padding=(14, 9), font=("Segoe UI Semibold", 9))
    style.map("MS.TButton", background=[("active", "#263744"), ("pressed", "#20303B")])
    style.configure("MS.Primary.TButton", background=CYAN, foreground=INK, bordercolor=CYAN, lightcolor=CYAN, darkcolor=CYAN, focuscolor=CYAN, padding=(16, 9), font=("Segoe UI Semibold", 9))
    style.map("MS.Primary.TButton", background=[("active", "#78E5F0"), ("pressed", "#34C1D2")])
    style.configure("MS.Danger.TButton", background=CARD_ALT, foreground=CORAL, bordercolor=OUTLINE, padding=(14, 9), font=("Segoe UI Semibold", 9))
    style.configure("MS.Compact.TButton", background=CARD_ALT, foreground=TEXT, bordercolor=OUTLINE, lightcolor=OUTLINE, darkcolor=OUTLINE, focuscolor=OUTLINE, padding=(10, 6), font=("Segoe UI Semibold", 8))
    style.map("MS.Compact.TButton", background=[("active", "#263744"), ("pressed", "#20303B")])
    style.configure("MS.TCheckbutton", background=CARD, foreground=TEXT, font=("Segoe UI", 9))
    style.map("MS.TCheckbutton", background=[("active", CARD)], foreground=[("disabled", MUTED)])
    style.configure("MS.TRadiobutton", background=CARD, foreground=TEXT, font=("Segoe UI", 9))
    style.configure("MS.TNotebook", background=INK, bordercolor=OUTLINE, lightcolor=OUTLINE, darkcolor=OUTLINE, borderwidth=1, tabmargins=(0, 8, 0, 0))
    style.configure("MS.TNotebook.Tab", background=INK, foreground=MUTED, bordercolor=OUTLINE, lightcolor=OUTLINE, darkcolor=OUTLINE, focuscolor=OUTLINE, borderwidth=1, padding=(17, 10), font=("Segoe UI Semibold", 9))
    style.map("MS.TNotebook.Tab", background=[("selected", SURFACE), ("active", CARD)], foreground=[("selected", CYAN), ("active", TEXT)], bordercolor=[("selected", OUTLINE), ("active", OUTLINE)], lightcolor=[("selected", OUTLINE), ("active", OUTLINE)], darkcolor=[("selected", OUTLINE), ("active", OUTLINE)])
    style.configure("MS.Treeview", background=CARD, fieldbackground=CARD, foreground=TEXT, rowheight=32, bordercolor=OUTLINE, font=("Segoe UI", 9))
    style.configure("MS.Treeview.Heading", background=CARD_ALT, foreground=MUTED, relief="flat", font=("Segoe UI Semibold", 9))
    style.map("MS.Treeview", background=[("selected", "#244752")], foreground=[("selected", TEXT)])
    style.configure("MS.Horizontal.TProgressbar", troughcolor=CARD_ALT, background=CYAN, bordercolor=CARD_ALT, lightcolor=CYAN, darkcolor=CYAN)
    style.configure("MS.TEntry", fieldbackground=CARD_ALT, foreground=TEXT, insertcolor=TEXT, bordercolor=OUTLINE, padding=7)
    style.configure("MS.TCombobox", fieldbackground=CARD_ALT, background=CARD_ALT, foreground=TEXT, arrowcolor=CYAN, bordercolor=OUTLINE, padding=6)


def _card(parent: tk.Misc, *, padding=(20, 18)) -> ttk.Frame:
    return ttk.Frame(parent, style="MS.OutlinedCard.TFrame", padding=padding)


def _label(parent: tk.Misc, text: str, *, muted: bool = False, **kwargs) -> ttk.Label:
    style = "MS.CardMuted.TLabel" if muted else "MS.Card.TLabel"
    return ttk.Label(parent, text=text, style=style, **kwargs)


class MidnightSignalManagementMixin:
    """Replace the visible legacy window while reusing its proven controllers."""

    def __init__(self, icon) -> None:
        super().__init__(icon)
        self._legacy_window = self.window
        self._legacy_window.withdraw()
        self.window = tk.Toplevel(self._legacy_window.master, class_="CtrlSpeak")
        self.window.title(f"CtrlSpeak {sysmod.APP_VERSION} · Midnight Signal")
        scale = display_scale(self.window)
        bounds = active_monitor_bounds(self.window)
        width = min(int(round(1120 * scale)), int(bounds.width * 0.94))
        height = min(int(round(760 * scale)), int(bounds.height * 0.92))
        x = bounds.left + (bounds.width - width) // 2
        y = bounds.top + (bounds.height - height) // 2
        self.window.geometry(f"{width}x{height}+{x}+{y}")
        self.window.minsize(min(int(960 * scale), width), min(int(650 * scale), height))
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.bind("<Escape>", lambda _event: self.close())
        apply_midnight_signal_theme(self.window)
        self._provider_profiles: tuple[ProviderTelemetry, ...] = ()
        self._capabilities: dict[str, object] = {}
        self._correction_rules: list[dict[str, object]] = []
        self._corrections_loading = False
        self._capability_generation = 0
        self._corrections_generation = 0
        self._correction_mutation_generation = 0
        self._correction_mutation_queue: queue.Queue[
            tuple[int, str, Callable[[ApiTranscriptionClient], object]]
        ] = queue.Queue()
        self._correction_mutation_worker_active = False
        self._correction_mutations_pending = 0
        self._correction_mutation_errors: list[str] = []
        self._correction_refresh_notice: str | None = None
        self._shell_poll_job: str | None = None
        self._build_shell()
        self.window.after(100, self._poll_midnight_state)
        self.window.after(180, self._refresh_gateway_capabilities)
        self.window.after(240, self._refresh_corrections)
        self.bring_to_front()

    def _build_shell(self) -> None:
        root = ttk.Frame(self.window, style="MS.Root.TFrame", padding=(28, 22))
        root.pack(fill=tk.BOTH, expand=True)
        header = ttk.Frame(root, style="MS.Root.TFrame")
        header.pack(fill=tk.X, pady=(0, 12))
        left = ttk.Frame(header, style="MS.Root.TFrame")
        left.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(left, text="CTRLSPEAK", style="MS.Hero.TLabel").pack(side=tk.LEFT)
        ttk.Label(left, text=f"  {sysmod.APP_VERSION}  ·  MIDNIGHT SIGNAL", style="MS.TLabel", foreground=MUTED).pack(side=tk.LEFT, pady=(7, 0))
        ttk.Button(
            header,
            text="Hide to tray",
            style="MS.Compact.TButton",
            command=self.close,
        ).pack(side=tk.RIGHT, padx=(10, 0))
        self.ms_ready_var = tk.StringVar(value="● READY")
        ttk.Label(header, textvariable=self.ms_ready_var, style="MS.Ready.TLabel").pack(side=tk.RIGHT, pady=(8, 0))

        self.ms_notebook = ttk.Notebook(root, style="MS.TNotebook")
        self.ms_notebook.pack(fill=tk.BOTH, expand=True)
        self.ms_pages: dict[str, ttk.Frame] = {}
        self.ms_page_tabs: dict[str, ttk.Frame] = {}
        self.ms_page_canvases: dict[str, tk.Canvas] = {}
        self.ms_page_horizontal_scrollbars: dict[str, ttk.Scrollbar] = {}
        for name in ("Capture", "Transcription", "Routing", "Corrections", "Updates", "System"):
            tab = ttk.Frame(self.ms_notebook, style="MS.Surface.TFrame")
            tab.rowconfigure(0, weight=1)
            tab.columnconfigure(0, weight=1)
            canvas = tk.Canvas(
                tab,
                background=SURFACE,
                highlightthickness=0,
                borderwidth=0,
            )
            scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
            horizontal = ttk.Scrollbar(tab, orient=tk.HORIZONTAL, command=canvas.xview)
            canvas.configure(
                yscrollcommand=scrollbar.set,
                xscrollcommand=horizontal.set,
            )
            canvas.grid(row=0, column=0, sticky="nsew")
            scrollbar.grid(row=0, column=1, sticky="ns")
            horizontal.grid(row=1, column=0, sticky="ew")
            page = ttk.Frame(canvas, style="MS.Surface.TFrame", padding=(22, 20))
            content_id = canvas.create_window((0, 0), window=page, anchor="nw")

            def sync_scroll_region(_event=None, *, target=canvas) -> None:
                target.configure(scrollregion=target.bbox("all"))

            def sync_content_width(
                event,
                *,
                target=canvas,
                item=content_id,
                content=page,
            ) -> None:
                target.itemconfigure(
                    item,
                    width=management_content_width(
                        content.winfo_reqwidth(), int(event.width)
                    ),
                )

            page.bind("<Configure>", sync_scroll_region)
            canvas.bind("<Configure>", sync_content_width)
            self.ms_pages[name] = page
            self.ms_page_tabs[name] = tab
            self.ms_page_canvases[name] = canvas
            self.ms_page_horizontal_scrollbars[name] = horizontal
            self.ms_notebook.add(tab, text=name.upper())
        self._build_capture_page(self.ms_pages["Capture"])
        self._build_transcription_page(self.ms_pages["Transcription"])
        self._build_routing_page(self.ms_pages["Routing"])
        self._build_corrections_page(self.ms_pages["Corrections"])
        self._build_updates_page(self.ms_pages["Updates"])
        self._build_system_page(self.ms_pages["System"])
        self.window.bind("<MouseWheel>", self._scroll_management_wheel, add="+")
        self.window.bind("<Shift-MouseWheel>", self._scroll_management_horizontal, add="+")
        self.window.bind("<Button-4>", lambda _event: self._scroll_active_management_page(-3), add="+")
        self.window.bind("<Button-5>", lambda _event: self._scroll_active_management_page(3), add="+")
        self.window.bind("<Prior>", lambda _event: self._scroll_active_management_page(-1, pages=True), add="+")
        self.window.bind("<Next>", lambda _event: self._scroll_active_management_page(1, pages=True), add="+")
        self.window.bind("<Home>", lambda _event: self._move_active_management_page(0.0), add="+")
        self.window.bind("<End>", lambda _event: self._move_active_management_page(1.0), add="+")

    def _active_management_canvas(self) -> tk.Canvas | None:
        selected = self.ms_notebook.select()
        for name, tab in self.ms_page_tabs.items():
            if str(tab) == selected:
                return self.ms_page_canvases[name]
        return None

    def _scroll_active_management_page(self, amount: int, *, pages: bool = False) -> str | None:
        canvas = self._active_management_canvas()
        if canvas is None or amount == 0:
            return None
        canvas.yview_scroll(int(amount), "pages" if pages else "units")
        return "break"

    def _move_active_management_page(self, fraction: float) -> str | None:
        canvas = self._active_management_canvas()
        if canvas is None:
            return None
        canvas.yview_moveto(max(0.0, min(1.0, float(fraction))))
        return "break"

    def _scroll_management_wheel(self, event) -> str | None:
        interactive_types = (tk.Listbox, ttk.Combobox, ttk.Treeview, ttk.Scale)
        if isinstance(getattr(event, "widget", None), interactive_types):
            return None
        return self._scroll_active_management_page(
            management_scroll_units(getattr(event, "delta", 0))
        )

    def _scroll_management_horizontal(self, event) -> str | None:
        canvas = self._active_management_canvas()
        amount = management_scroll_units(getattr(event, "delta", 0))
        if canvas is None or amount == 0:
            return None
        canvas.xview_scroll(amount, "units")
        return "break"

    def _build_capture_page(self, page: ttk.Frame) -> None:
        hero = _card(page, padding=(26, 22))
        hero.pack(fill=tk.X)
        ttk.Label(hero, text="Hold Right Ctrl to speak", style="MS.Section.TLabel").pack(anchor=tk.W)
        self.ms_capture_detail_var = tk.StringVar(value="Ready · release the key when you finish")
        ttk.Label(hero, textvariable=self.ms_capture_detail_var, style="MS.CardMuted.TLabel").pack(anchor=tk.W, pady=(6, 0))
        meter_row = ttk.Frame(hero, style="MS.Card.TFrame")
        meter_row.pack(fill=tk.X, pady=(18, 0))
        self.ms_level_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(meter_row, variable=self.ms_level_var, maximum=100, style="MS.Horizontal.TProgressbar").pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.ms_dbfs_var = tk.StringVar(value="−60.0 dBFS")
        ttk.Label(meter_row, textvariable=self.ms_dbfs_var, style="MS.Card.TLabel", width=13, anchor=tk.E).pack(side=tk.RIGHT, padx=(16, 0))

        columns = ttk.Frame(page, style="MS.Surface.TFrame")
        columns.pack(fill=tk.BOTH, expand=True, pady=(16, 0))
        left = _card(columns)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        right = _card(columns)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))
        ttk.Label(left, text="Last transcription", style="MS.Section.TLabel").pack(anchor=tk.W)
        self.ms_last_provider_var = tk.StringVar(value="No transcription in this session")
        ttk.Label(left, textvariable=self.ms_last_provider_var, style="MS.Metric.TLabel", wraplength=410, justify=tk.LEFT).pack(anchor=tk.W, pady=(16, 4))
        self.ms_last_route_var = tk.StringVar(value="Provider and route telemetry will appear here.")
        ttk.Label(left, textvariable=self.ms_last_route_var, style="MS.CardMuted.TLabel", wraplength=410, justify=tk.LEFT).pack(anchor=tk.W)
        actions = ttk.Frame(left, style="MS.Card.TFrame")
        actions.pack(fill=tk.X, side=tk.BOTTOM, pady=(20, 0))
        ttk.Button(actions, text="Copy last", style="MS.TButton", command=sysmod.copy_last_transcript_from_tray).pack(side=tk.LEFT)
        ttk.Button(actions, text="Cancel active", style="MS.Danger.TButton", command=sysmod.cancel_active_transcription).pack(side=tk.LEFT, padx=(10, 0))

        ttk.Label(right, text="Capture feedback", style="MS.Section.TLabel").pack(anchor=tk.W)
        with settings_lock:
            cues = bool(settings.get("audio_cues_enabled", True))
            volume = int(settings.get("audio_cue_volume", 30))
            overlay = bool(settings.get("overlay_enabled", True))
            reduced = bool(settings.get("reduced_motion", False))
        self.ms_cues_enabled_var = tk.BooleanVar(value=cues)
        self.ms_cue_volume_var = tk.IntVar(value=volume)
        self.ms_overlay_enabled_var = tk.BooleanVar(value=overlay)
        self.ms_reduced_motion_var = tk.BooleanVar(value=reduced)
        ttk.Checkbutton(right, text="Show the recording capsule", variable=self.ms_overlay_enabled_var, style="MS.TCheckbutton", command=self._save_experience_preferences).pack(anchor=tk.W, pady=(14, 6))
        ttk.Checkbutton(right, text="Use reduced motion", variable=self.ms_reduced_motion_var, style="MS.TCheckbutton", command=self._save_experience_preferences).pack(anchor=tk.W, pady=6)
        ttk.Checkbutton(right, text="Play short audio cues", variable=self.ms_cues_enabled_var, style="MS.TCheckbutton", command=self._save_experience_preferences).pack(anchor=tk.W, pady=6)
        _label(right, "Cue volume", muted=True).pack(anchor=tk.W, pady=(16, 3))
        scale = ttk.Scale(right, from_=0, to=100, variable=self.ms_cue_volume_var)
        scale.pack(fill=tk.X)
        scale.bind("<ButtonRelease-1>", lambda _event: self._save_experience_preferences())
        scale.bind("<KeyRelease>", lambda _event: self._save_experience_preferences())
        buttons = ttk.Frame(right, style="MS.Card.TFrame")
        buttons.pack(fill=tk.X, pady=(14, 0))
        ttk.Button(buttons, text="Preview", style="MS.TButton", command=lambda: sysmod.play_ui_cue(CueKind.SUCCESS)).pack(side=tk.LEFT)
        _label(right, "Cues are low/mid frequency, peak-limited to −12 dBFS, and never loop.", muted=True, wraplength=410, justify=tk.LEFT).pack(anchor=tk.W, pady=(14, 0))

    def _build_transcription_page(self, page: ttk.Frame) -> None:
        top = _card(page)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Transcription preferences", style="MS.Section.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")
        _label(top, "Backend", muted=True).grid(row=1, column=0, sticky="w", pady=(18, 5))
        backend_combo = ttk.Combobox(top, textvariable=self.backend_var, values=("Embedded / local", "Remote API"), state="readonly", style="MS.TCombobox", width=24)
        backend_combo.grid(row=2, column=0, sticky="ew", padx=(0, 12))
        _label(top, "Provider route", muted=True).grid(row=1, column=1, sticky="w", pady=(18, 5))
        self.ms_strategy_display_var = tk.StringVar(value=STRATEGY_LABELS.get(self.provider_strategy_var.get(), self.provider_strategy_var.get()))
        self.ms_strategy_combo = ttk.Combobox(top, textvariable=self.ms_strategy_display_var, values=tuple(STRATEGY_IDS), state="readonly", style="MS.TCombobox", width=24)
        self.ms_strategy_combo.grid(row=2, column=1, sticky="ew", padx=(0, 12))
        ttk.Button(top, text="Save preferences", style="MS.Primary.TButton", command=self._save_backend_midnight).grid(row=2, column=2, sticky="ew")
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)

        body = ttk.Frame(page, style="MS.Surface.TFrame")
        body.pack(fill=tk.BOTH, expand=True, pady=(16, 0))
        languages = _card(body)
        languages.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        ttk.Label(languages, text="Allowed output languages", style="MS.Section.TLabel").pack(anchor=tk.W)
        _label(languages, "Choose none for automatic detection, one to force it, or up to five allowed languages.", muted=True, wraplength=410, justify=tk.LEFT).pack(anchor=tk.W, pady=(5, 12))
        choices = language_choices()
        self._output_language_codes = tuple(code for code, _name in choices)
        self.ms_output_language_list = tk.Listbox(
            languages, selectmode=tk.MULTIPLE, exportselection=False, height=12,
            background=CARD_ALT, foreground=TEXT, selectbackground="#244752",
            selectforeground=TEXT, highlightbackground=OUTLINE, highlightcolor=CYAN,
            borderwidth=1, relief=tk.FLAT, font=("Segoe UI", 10),
        )
        for code, name in choices:
            self.ms_output_language_list.insert(tk.END, f"{name} ({code})")
        self.ms_output_language_list.pack(fill=tk.BOTH, expand=True)
        self._set_output_language_selection(get_backend_config().allowed_output_languages)
        quick = ttk.Frame(languages, style="MS.Card.TFrame")
        quick.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(quick, text="English only", style="MS.TButton", command=lambda: self._set_output_language_selection(("en",))).pack(side=tk.LEFT)
        ttk.Button(quick, text="Automatic", style="MS.TButton", command=lambda: self._set_output_language_selection(())).pack(side=tk.LEFT, padx=(10, 0))

        behaviour = _card(body)
        behaviour.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))
        ttk.Label(behaviour, text="Text and corrections", style="MS.Section.TLabel").pack(anchor=tk.W)
        _label(behaviour, "Automatic edit feedback", muted=True).pack(anchor=tk.W, pady=(18, 5))
        ttk.Combobox(behaviour, textvariable=self.feedback_capture_var, values=("active_field_on_enter", "disabled"), state="readonly", style="MS.TCombobox").pack(fill=tk.X)
        _label(behaviour, "After insertion, bare Enter can submit confirmed edits without retaining transcript history on the desktop.", muted=True, wraplength=410, justify=tk.LEFT).pack(anchor=tk.W, pady=(8, 16))
        ttk.Button(behaviour, text="Open corrections", style="MS.Primary.TButton", command=lambda: self.ms_notebook.select(self.ms_pages["Corrections"])).pack(anchor=tk.W)
        self.ms_backend_status_var = tk.StringVar(value=self.backend_status_var.get())
        ttk.Label(behaviour, textvariable=self.ms_backend_status_var, style="MS.CardMuted.TLabel", wraplength=410, justify=tk.LEFT).pack(anchor=tk.W, pady=(18, 0))

    def _build_routing_page(self, page: ttk.Frame) -> None:
        header = ttk.Frame(page, style="MS.Surface.TFrame")
        header.pack(fill=tk.X, pady=(0, 12))
        title = ttk.Frame(header, style="MS.Surface.TFrame")
        title.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(title, text="Provider routing", style="MS.Surface.TLabel", font=("Segoe UI Semibold", 13)).pack(anchor=tk.W)
        self.ms_route_summary_var = tk.StringVar(value="Checking the configured gateway…")
        ttk.Label(title, textvariable=self.ms_route_summary_var, style="MS.Surface.TLabel", foreground=MUTED).pack(anchor=tk.W, pady=(4, 0))
        self.ms_route_order_var = tk.StringVar(value="Gateway default · server decides provider order")
        ttk.Label(title, textvariable=self.ms_route_order_var, style="MS.Surface.TLabel", foreground=CYAN).pack(anchor=tk.W, pady=(2, 0))
        controls = ttk.Frame(header, style="MS.Surface.TFrame")
        controls.pack(side=tk.RIGHT)
        ttk.Label(controls, text="Preset", style="MS.Surface.TLabel", foreground=MUTED).pack(side=tk.LEFT, padx=(0, 7))
        self.ms_route_strategy_combo = ttk.Combobox(
            controls,
            textvariable=self.ms_strategy_display_var,
            values=tuple(STRATEGY_IDS),
            state="readonly",
            style="MS.TCombobox",
            width=18,
        )
        self.ms_route_strategy_combo.pack(side=tk.LEFT)
        self.ms_route_strategy_combo.bind("<<ComboboxSelected>>", lambda _event: self._save_backend_midnight())
        ttk.Button(controls, text="Refresh", style="MS.Primary.TButton", command=self._refresh_gateway_capabilities).pack(side=tk.LEFT, padx=(8, 0))

        dashboard = ttk.Frame(page, style="MS.Surface.TFrame")
        dashboard.pack(fill=tk.BOTH, expand=True)
        dashboard.columnconfigure(0, weight=ROUTING_COLUMN_WEIGHTS[0], uniform="routing")
        dashboard.columnconfigure(1, weight=ROUTING_COLUMN_WEIGHTS[1], uniform="routing")
        dashboard.rowconfigure(0, weight=1)
        providers = ttk.Frame(dashboard, style="MS.Surface.TFrame")
        providers.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        sidebar = ttk.Frame(dashboard, style="MS.Surface.TFrame")
        sidebar.grid(row=0, column=1, sticky="nsew", padx=(7, 0))

        self.ms_provider_vars: dict[
            str, tuple[tk.StringVar, tk.StringVar, tk.StringVar, tk.StringVar]
        ] = {}
        self.ms_provider_signals: dict[str, tk.Canvas] = {}
        self.ms_provider_cards: dict[str, ttk.Frame] = {}
        self.ms_provider_badges: dict[str, tk.Canvas] = {}
        for index, (provider_id, provider_title, model, location) in enumerate(PROVIDER_CARD_SPECS, start=1):
            card = _card(providers, padding=(14, 11))
            card.pack(fill=tk.X, pady=(0, 8))
            top = ttk.Frame(card, style="MS.Card.TFrame")
            top.pack(fill=tk.X)
            number = tk.Canvas(top, width=28, height=28, background=CARD, highlightthickness=0, borderwidth=0)
            number.pack(side=tk.LEFT, padx=(0, 10))
            _paint_route_badge(number, None, size=28)
            identity = ttk.Frame(top, style="MS.Card.TFrame")
            identity.pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Label(identity, text=provider_title, style="MS.Section.TLabel").pack(anchor=tk.W)
            _label(identity, f"{model}  ·  {location}", muted=True).pack(anchor=tk.W, pady=(2, 0))
            state_var = tk.StringVar(value="UNKNOWN")
            ttk.Label(top, textvariable=state_var, style="MS.CompactMetric.TLabel").pack(side=tk.RIGHT, anchor=tk.N, padx=(10, 0))
            detail_row = ttk.Frame(card, style="MS.Card.TFrame")
            detail_row.pack(fill=tk.X, pady=(8, 0))
            metrics = ttk.Frame(detail_row, style="MS.Card.TFrame")
            metrics.pack(side=tk.LEFT, fill=tk.X, expand=True)
            timing_var = tk.StringVar(value="Worker probe —")
            detail_var = tk.StringVar(value="Health not checked")
            last_var = tk.StringVar(value="Last request —")
            ttk.Label(metrics, textvariable=timing_var, style="MS.Card.TLabel").pack(anchor=tk.W)
            ttk.Label(metrics, textvariable=detail_var, style="MS.CardMuted.TLabel").pack(anchor=tk.W, pady=(2, 0))
            ttk.Label(metrics, textvariable=last_var, style="MS.CardMuted.TLabel").pack(anchor=tk.W, pady=(2, 0))
            signal = tk.Canvas(detail_row, width=118, height=30, background=CARD, highlightthickness=0, borderwidth=0)
            signal.pack(side=tk.RIGHT, padx=(10, 0))
            _paint_provider_signal(signal, seed=index, colour=OUTLINE)
            self.ms_provider_signals[provider_id] = signal
            self.ms_provider_vars[provider_id] = (state_var, timing_var, detail_var, last_var)
            self.ms_provider_cards[provider_id] = card
            self.ms_provider_badges[provider_id] = number

        telemetry = _card(providers, padding=(14, 10))
        telemetry.pack(fill=tk.BOTH, expand=True)
        self.ms_route_telemetry_card = telemetry
        ttk.Label(telemetry, text="Route attempts", style="MS.Section.TLabel").pack(anchor=tk.W)
        self.ms_attempts_var = tk.StringVar(value="Attempt timings appear only after a completed request.")
        ttk.Label(telemetry, textvariable=self.ms_attempts_var, style="MS.CardMuted.TLabel", wraplength=560, justify=tk.LEFT).pack(anchor=tk.W, pady=(5, 0))

        corrections = _card(sidebar, padding=(13, 10))
        corrections.pack(fill=tk.X, pady=(0, 8))
        correction_header = ttk.Frame(corrections, style="MS.Card.TFrame")
        correction_header.pack(fill=tk.X)
        ttk.Label(correction_header, text="Corrections", style="MS.Section.TLabel").pack(side=tk.LEFT)
        ttk.Button(correction_header, text="＋ Add", style="MS.Compact.TButton", command=lambda: self.ms_notebook.select(self.ms_pages["Corrections"])).pack(side=tk.RIGHT)
        self.ms_route_correction_preview_var = tk.StringVar(value="Loading known-word rules…")
        ttk.Label(corrections, textvariable=self.ms_route_correction_preview_var, style="MS.CardMuted.TLabel", wraplength=330, justify=tk.LEFT).pack(anchor=tk.W, pady=(7, 0))

        microphone = _card(sidebar, padding=(13, 10))
        microphone.pack(fill=tk.X, pady=(0, 8))
        mic_header = ttk.Frame(microphone, style="MS.Card.TFrame")
        mic_header.pack(fill=tk.X)
        mic_name = sysmod.get_input_device_preference() or "System default"
        self.ms_route_mic_name_var = tk.StringVar(value=mic_name)
        ttk.Label(mic_header, text="Microphone", style="MS.Section.TLabel").pack(side=tk.LEFT)
        ttk.Label(mic_header, textvariable=self.ms_route_mic_name_var, style="MS.CardMuted.TLabel").pack(side=tk.RIGHT)
        meter = ttk.Frame(microphone, style="MS.Card.TFrame")
        meter.pack(fill=tk.X, pady=(8, 0))
        self.ms_route_level_var = tk.DoubleVar(value=0.0)
        self.ms_route_dbfs_var = tk.StringVar(value="−60.0 dBFS")
        ttk.Progressbar(meter, variable=self.ms_route_level_var, maximum=100, style="MS.Horizontal.TProgressbar").pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(meter, textvariable=self.ms_route_dbfs_var, style="MS.CardMuted.TLabel", width=12, anchor=tk.E).pack(side=tk.RIGHT, padx=(8, 0))

        preferences = _card(sidebar, padding=(13, 9))
        preferences.pack(fill=tk.X, pady=(0, 8))
        pref_row = ttk.Frame(preferences, style="MS.Card.TFrame")
        pref_row.pack(fill=tk.X)
        self.ms_route_language_var = tk.StringVar(value=language_summary(get_backend_config().allowed_output_languages))
        with settings_lock:
            cue_enabled = bool(settings.get("audio_cues_enabled", True))
            cue_volume = int(settings.get("audio_cue_volume", 30))
        self.ms_route_cue_var = tk.StringVar(value=f"{cue_volume}%" if cue_enabled else "Muted")
        ttk.Label(pref_row, text="Language", style="MS.CardMuted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(pref_row, text="Cue volume", style="MS.CardMuted.TLabel").grid(row=0, column=2, sticky="w", padx=(20, 0))
        ttk.Label(pref_row, textvariable=self.ms_route_language_var, style="MS.Card.TLabel").grid(row=1, column=0, sticky="w", pady=(3, 0))
        ttk.Separator(pref_row, orient=tk.VERTICAL).grid(row=0, column=1, rowspan=2, sticky="ns", padx=(16, 0))
        ttk.Label(pref_row, textvariable=self.ms_route_cue_var, style="MS.Card.TLabel").grid(row=1, column=2, sticky="w", padx=(20, 0), pady=(3, 0))

        result = _card(sidebar, padding=(13, 10))
        result.pack(fill=tk.X, pady=(0, 8))
        result_header = ttk.Frame(result, style="MS.Card.TFrame")
        result_header.pack(fill=tk.X)
        ttk.Label(result_header, text="Last result", style="MS.Section.TLabel").pack(side=tk.LEFT)
        ttk.Button(result_header, text="Copy", style="MS.Compact.TButton", command=sysmod.copy_last_transcript_from_tray).pack(side=tk.RIGHT)
        self.ms_routing_result_var = tk.StringVar(value="No result in this session")
        ttk.Label(result, textvariable=self.ms_routing_result_var, style="MS.CompactMetric.TLabel", wraplength=330, justify=tk.LEFT).pack(anchor=tk.W, pady=(7, 0))

        updates = _card(sidebar, padding=(13, 10))
        updates.pack(fill=tk.X)
        updates_header = ttk.Frame(updates, style="MS.Card.TFrame")
        updates_header.pack(fill=tk.X)
        ttk.Label(updates_header, text=f"Updates · {sysmod.APP_VERSION}", style="MS.Section.TLabel").pack(side=tk.LEFT)
        ttk.Button(updates_header, text="Check", style="MS.Compact.TButton", command=self.check_for_updates).pack(side=tk.RIGHT)
        ttk.Label(updates, textvariable=self.update_status_var, style="MS.CardMuted.TLabel", wraplength=330, justify=tk.LEFT).pack(anchor=tk.W, pady=(7, 0))
        self._apply_route_strategy_visuals()

    def _selected_route_strategy_id(self) -> str:
        display = self.ms_strategy_display_var.get()
        return STRATEGY_IDS.get(display, self.provider_strategy_var.get() or "server-default")

    def _apply_route_strategy_visuals(self) -> None:
        if not hasattr(self, "ms_provider_cards"):
            return
        strategy_id = self._selected_route_strategy_id()
        order = provider_order_for_strategy(strategy_id, self._capabilities)
        if order is None:
            visible = tuple(spec[0] for spec in PROVIDER_CARD_SPECS)
            note = "Gateway default · server decides provider order"
        else:
            visible = order
            names = {provider_id: title for provider_id, title, _model, _location in PROVIDER_CARD_SPECS}
            chain = " → ".join(names[provider_id] for provider_id in visible)
            if strategy_id == "server-default":
                note = f"Gateway default · {chain}"
            else:
                note = f"Selected route · {chain}"
        self.ms_route_order_var.set(note)
        self.ms_route_telemetry_card.pack_forget()
        for card in self.ms_provider_cards.values():
            card.pack_forget()
        for position, provider_id in enumerate(visible, start=1):
            self.ms_provider_cards[provider_id].pack(fill="x", pady=(0, 8))
            _paint_route_badge(
                self.ms_provider_badges[provider_id],
                position if order is not None else None,
                size=28,
            )
        self.ms_route_telemetry_card.pack(fill=tk.BOTH, expand=True)

    def _build_corrections_page(self, page: ttk.Frame) -> None:
        editor = _card(page)
        editor.pack(fill=tk.X)
        ttk.Label(editor, text="Add or edit a known-word correction", style="MS.Section.TLabel").grid(row=0, column=0, columnspan=5, sticky="w")
        self.ms_correction_source_var = tk.StringVar()
        self.ms_correction_replacement_var = tk.StringVar()
        self.ms_selected_rule_id: str | None = None
        _label(editor, "When CtrlSpeak hears", muted=True).grid(row=1, column=0, sticky="w", pady=(14, 4))
        _label(editor, "Replace with", muted=True).grid(row=1, column=1, sticky="w", pady=(14, 4), padx=(12, 0))
        ttk.Entry(editor, textvariable=self.ms_correction_source_var, style="MS.TEntry").grid(row=2, column=0, sticky="ew")
        ttk.Entry(editor, textvariable=self.ms_correction_replacement_var, style="MS.TEntry").grid(row=2, column=1, sticky="ew", padx=(12, 0))
        ttk.Button(editor, text="Add", style="MS.Primary.TButton", command=self._add_correction).grid(row=2, column=2, padx=(12, 0))
        ttk.Button(editor, text="Save selected", style="MS.TButton", command=self._save_selected_correction).grid(row=2, column=3, padx=(8, 0))
        ttk.Button(editor, text="Clear", style="MS.TButton", command=self._clear_correction_editor).grid(row=2, column=4, padx=(8, 0))
        editor.columnconfigure(0, weight=1)
        editor.columnconfigure(1, weight=1)

        toolbar = ttk.Frame(page, style="MS.Surface.TFrame")
        toolbar.pack(fill=tk.X, pady=(14, 8))
        self.ms_correction_search_var = tk.StringVar()
        self.ms_correction_search_var.trace_add("write", lambda *_args: self._render_corrections())
        ttk.Label(toolbar, text="Search", style="MS.Surface.TLabel", foreground=MUTED).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Entry(toolbar, textvariable=self.ms_correction_search_var, style="MS.TEntry", width=34).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Refresh", style="MS.TButton", command=self._refresh_corrections).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(toolbar, text="Enable / disable", style="MS.TButton", command=self._toggle_selected_correction).pack(side=tk.RIGHT)
        ttk.Button(toolbar, text="Delete", style="MS.Danger.TButton", command=self._delete_selected_correction).pack(side=tk.RIGHT, padx=(0, 8))

        columns = ("heard", "output", "scope", "status", "priority")
        tree_frame = ttk.Frame(page, style="MS.Surface.TFrame")
        tree_frame.pack(fill=tk.BOTH, expand=True)
        self.ms_correction_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", style="MS.Treeview", selectmode="browse")
        for name, title, width in (("heard", "HEARD", 260), ("output", "OUTPUT", 260), ("scope", "SCOPE", 90), ("status", "STATUS", 90), ("priority", "PRIORITY", 80)):
            self.ms_correction_tree.heading(name, text=title)
            self.ms_correction_tree.column(name, width=width, minwidth=60, stretch=name in {"heard", "output"})
        correction_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.ms_correction_tree.yview)
        self.ms_correction_tree.configure(yscrollcommand=correction_scroll.set)
        self.ms_correction_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        correction_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.ms_correction_tree.bind("<<TreeviewSelect>>", self._correction_selected)
        self.ms_correction_status_var = tk.StringVar(value="Corrections have not been loaded.")
        ttk.Label(page, textvariable=self.ms_correction_status_var, style="MS.Surface.TLabel", foreground=MUTED).pack(anchor=tk.W, pady=(8, 0))

    def _build_updates_page(self, page: ttk.Frame) -> None:
        card = _card(page, padding=(26, 24))
        card.pack(fill=tk.BOTH, expand=True)
        ttk.Label(card, text="Application updates", style="MS.Section.TLabel").pack(anchor=tk.W)
        ttk.Label(card, text=f"CtrlSpeak {sysmod.APP_VERSION} · stable channel", style="MS.CardMuted.TLabel").pack(anchor=tk.W, pady=(5, 22))
        ttk.Label(card, textvariable=self.update_status_var, style="MS.Metric.TLabel", wraplength=920, justify=tk.LEFT).pack(anchor=tk.W)
        ttk.Label(card, textvariable=self.update_last_checked_var, style="MS.CardMuted.TLabel").pack(anchor=tk.W, pady=(10, 0))
        ttk.Progressbar(card, variable=self.update_progress_var, maximum=100, style="MS.Horizontal.TProgressbar").pack(fill=tk.X, pady=(24, 4))
        ttk.Label(card, textvariable=self.update_progress_text_var, style="MS.CardMuted.TLabel").pack(anchor=tk.W)
        buttons = ttk.Frame(card, style="MS.Card.TFrame")
        buttons.pack(fill=tk.X, pady=(24, 0))
        self.ms_check_update_btn = ttk.Button(buttons, text="Check for updates", style="MS.Primary.TButton", command=self.check_for_updates)
        self.ms_check_update_btn.pack(side=tk.LEFT)
        self.ms_install_update_btn = ttk.Button(buttons, text="Download and install", style="MS.TButton", command=self._download_or_install_update)
        self.ms_install_update_btn.pack(side=tk.LEFT, padx=(10, 0))
        self.ms_cancel_update_btn = ttk.Button(buttons, text="Cancel", style="MS.TButton", command=self._cancel_update)
        self.ms_cancel_update_btn.pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(buttons, text="View release", style="MS.TButton", command=self._view_update_release).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Copy diagnostics", style="MS.TButton", command=self._copy_update_diagnostics).pack(side=tk.RIGHT, padx=(0, 10))
        _label(card, "Downloads are signed, SHA-256 verified, installed by a separate helper, and automatically rolled back if startup health fails.", muted=True, wraplength=920, justify=tk.LEFT).pack(anchor=tk.W, side=tk.BOTTOM)

    def _build_system_page(self, page: ttk.Frame) -> None:
        columns = ttk.Frame(page, style="MS.Surface.TFrame")
        columns.pack(fill=tk.BOTH, expand=True)
        left = _card(columns)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        right = _card(columns)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))
        ttk.Label(left, text="Audio and local engine", style="MS.Section.TLabel").pack(anchor=tk.W)
        _label(left, "Input microphone", muted=True).pack(anchor=tk.W, pady=(16, 5))
        ttk.Combobox(left, textvariable=self.audio_device_var, values=self.audio_device_combo.cget("values"), state="readonly", style="MS.TCombobox").pack(fill=tk.X)
        ttk.Button(left, text="Apply input device", style="MS.TButton", command=self._apply_audio_device).pack(anchor=tk.W, pady=(10, 0))
        _label(left, "Embedded device", muted=True).pack(anchor=tk.W, pady=(20, 5))
        device_row = ttk.Frame(left, style="MS.Card.TFrame")
        device_row.pack(fill=tk.X)
        ttk.Radiobutton(device_row, text="CPU", variable=self.device_var, value="cpu", style="MS.TRadiobutton").pack(side=tk.LEFT)
        ttk.Radiobutton(device_row, text="CUDA GPU", variable=self.device_var, value="cuda", style="MS.TRadiobutton").pack(side=tk.LEFT, padx=(18, 0))
        ttk.Button(left, text="Apply device", style="MS.TButton", command=self._apply_device).pack(anchor=tk.W, pady=(10, 0))
        _label(left, "Embedded model", muted=True).pack(anchor=tk.W, pady=(20, 5))
        ttk.Combobox(left, textvariable=self.model_var, values=("small", "large-v3"), state="readonly", style="MS.TCombobox").pack(fill=tk.X)
        model_buttons = ttk.Frame(left, style="MS.Card.TFrame")
        model_buttons.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(model_buttons, text="Activate", style="MS.TButton", command=self._apply_model).pack(side=tk.LEFT)
        ttk.Button(model_buttons, text="Download", style="MS.TButton", command=self._download_model).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(right, text="Connection and credentials", style="MS.Section.TLabel").pack(anchor=tk.W)
        _label(right, "Gateway URL", muted=True).pack(anchor=tk.W, pady=(16, 4))
        ttk.Entry(right, textvariable=self.api_url_var, style="MS.TEntry").pack(fill=tk.X)
        _label(right, "Gateway bearer token", muted=True).pack(anchor=tk.W, pady=(12, 4))
        ttk.Entry(right, textvariable=self.api_token_var, show="•", style="MS.TEntry").pack(fill=tk.X)
        _label(right, "OpenAI API key", muted=True).pack(anchor=tk.W, pady=(12, 4))
        ttk.Entry(right, textvariable=self.openai_api_key_var, show="•", style="MS.TEntry").pack(fill=tk.X)
        ttk.Checkbutton(right, text="Remember securely on this computer", variable=self.remember_openai_key_var, style="MS.TCheckbutton").pack(anchor=tk.W, pady=(8, 0))
        connection_buttons = ttk.Frame(right, style="MS.Card.TFrame")
        connection_buttons.pack(fill=tk.X, pady=(16, 0))
        ttk.Button(connection_buttons, text="Save connection", style="MS.Primary.TButton", command=self._save_backend_midnight).pack(side=tk.LEFT)
        ttk.Button(connection_buttons, text="Forget key", style="MS.TButton", command=self._forget_openai_key).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Separator(right).pack(fill=tk.X, pady=20)
        ttk.Label(right, text="Runtime controls", style="MS.Section.TLabel").pack(anchor=tk.W)
        runtime = ttk.Frame(right, style="MS.Card.TFrame")
        runtime.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(runtime, text="Start client", style="MS.TButton", command=self.start_client).pack(side=tk.LEFT)
        ttk.Button(runtime, text="Stop client", style="MS.TButton", command=self.stop_client).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(runtime, text="Refresh servers", style="MS.TButton", command=self.refresh_servers).pack(side=tk.LEFT, padx=(8, 0))
        footer = ttk.Frame(right, style="MS.Card.TFrame")
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Button(footer, text="Hide to tray", style="MS.TButton", command=self.close).pack(side=tk.RIGHT)
        ttk.Button(footer, text="Quit CtrlSpeak", style="MS.Danger.TButton", command=self.stop_everything).pack(side=tk.RIGHT, padx=(0, 8))

    def _save_experience_preferences(self) -> None:
        with settings_lock:
            settings["overlay_enabled"] = bool(self.ms_overlay_enabled_var.get())
            settings["reduced_motion"] = bool(self.ms_reduced_motion_var.get())
            settings["audio_cues_enabled"] = bool(self.ms_cues_enabled_var.get())
            settings["audio_cue_volume"] = max(0, min(100, int(round(self.ms_cue_volume_var.get()))))
        save_settings()
        if hasattr(self, "ms_route_cue_var"):
            enabled = bool(self.ms_cues_enabled_var.get())
            volume = max(0, min(100, int(round(self.ms_cue_volume_var.get()))))
            self.ms_route_cue_var.set(f"{volume}%" if enabled else "Muted")

    def _selected_output_languages(self) -> tuple[str, ...]:
        widget = getattr(self, "ms_output_language_list", None)
        if widget is None:
            return super()._selected_output_languages()
        return tuple(self._output_language_codes[int(index)] for index in widget.curselection())

    def _set_output_language_selection(self, codes: tuple[str, ...]) -> None:
        widget = getattr(self, "ms_output_language_list", None)
        if widget is None:
            return super()._set_output_language_selection(codes)
        selected = set(codes)
        widget.selection_clear(0, tk.END)
        for index, code in enumerate(self._output_language_codes):
            if code in selected:
                widget.selection_set(index)

    def _save_backend_midnight(self) -> None:
        self.provider_strategy_var.set(STRATEGY_IDS.get(self.ms_strategy_display_var.get(), "server-default"))
        self._apply_backend()
        self._capability_generation += 1
        self._corrections_generation += 1
        self._correction_mutation_generation += 1
        self._correction_mutations_pending = 0
        self._correction_mutation_errors.clear()
        self._correction_refresh_notice = None
        self._corrections_loading = False
        # Capability chains are authenticated observations of one exact
        # gateway/config identity.  Never carry their numbering across a save.
        self._capabilities = {}
        self._provider_profiles = ()
        self.ms_backend_status_var.set(self.backend_status_var.get())
        if hasattr(self, "ms_route_language_var"):
            self.ms_route_language_var.set(language_summary(get_backend_config().allowed_output_languages))
        self._apply_route_strategy_visuals()
        self.window.after(120, self._refresh_gateway_capabilities)

    def _refresh_gateway_capabilities(self) -> None:
        if not hasattr(self, "ms_route_summary_var"):
            return
        config = get_backend_config()
        self._capability_generation += 1
        request_generation = self._capability_generation
        config_identity = capability_config_identity(config)
        if config.backend != "api":
            self._capabilities = {}
            self._provider_profiles = ()
            self.ms_route_summary_var.set("The embedded backend is selected; gateway telemetry is inactive.")
            self._apply_route_strategy_visuals()
            return
        self.ms_route_summary_var.set("Checking gateway health and provider capabilities…")

        def worker() -> None:
            started = time.monotonic()
            try:
                payload = ApiTranscriptionClient(config, timeout_seconds=10.0).get_capabilities()
                rtt = (time.monotonic() - started) * 1000.0
            except Exception:
                sysmod.enqueue_management_task(
                    self._finish_capabilities_error,
                    request_generation,
                    config_identity,
                )
                return
            safe_payload = dict(payload)
            sysmod.enqueue_management_task(
                self._finish_capabilities,
                safe_payload,
                rtt,
                request_generation,
                config_identity,
            )

        threading.Thread(target=worker, name="CtrlSpeakCapabilityCards", daemon=True).start()

    def _capability_request_is_current(
        self,
        request_generation: int,
        config_identity: tuple[object, ...],
    ) -> bool:
        if not self.is_open() or request_generation != self._capability_generation:
            return False
        try:
            return capability_config_identity(get_backend_config()) == config_identity
        except Exception:
            return False

    def _finish_capabilities_error(
        self,
        request_generation: int,
        config_identity: tuple[object, ...],
    ) -> None:
        if not self._capability_request_is_current(request_generation, config_identity):
            return
        self._capabilities = {}
        self._provider_profiles = ()
        self.ms_route_summary_var.set("Gateway unavailable · check the URL, WireGuard link, and token")
        for state_var, timing_var, detail_var, last_var in self.ms_provider_vars.values():
            state_var.set("UNAVAILABLE")
            timing_var.set("Gateway RTT —")
            detail_var.set("No authenticated capability telemetry was accepted")
            last_var.set("Last request —")
        for index, (provider_id, _title, _model, _location) in enumerate(PROVIDER_CARD_SPECS, start=1):
            signal = self.ms_provider_signals.get(provider_id)
            if signal is not None:
                _paint_provider_signal(signal, seed=index, colour=CORAL)
        self._apply_route_strategy_visuals()

    def _finish_capabilities(
        self,
        payload: Mapping[str, object],
        gateway_rtt: float,
        request_generation: int,
        config_identity: tuple[object, ...],
    ) -> None:
        if not self._capability_request_is_current(request_generation, config_identity):
            return
        self._capabilities = dict(payload)
        snapshot = sysmod.transcription_ui_session.snapshot()
        active_id = snapshot.provider.provider_id if snapshot.provider else None
        profiles = providers_from_capabilities(payload, active_provider_id=active_id)
        self._provider_profiles = profiles
        choices = strategy_choices_from_capabilities(payload)
        labels: list[str] = []
        for strategy_id, label in choices:
            STRATEGY_LABELS.setdefault(strategy_id, label)
            STRATEGY_IDS[label] = strategy_id
            labels.append(label)
        self.ms_strategy_combo.configure(values=tuple(labels))
        self.ms_route_strategy_combo.configure(values=tuple(labels))
        self.ms_route_summary_var.set(
            f"Gateway {payload.get('version', 'unknown')} · RTT {format_latency_ms(gateway_rtt)} · "
            f"{STRATEGY_LABELS.get(self.provider_strategy_var.get(), self.provider_strategy_var.get())}"
        )
        by_id = {profile.provider_id: profile for profile in profiles}
        aliases = {
            "gateway-tiny": by_id.get("nova-tiny-whisper") or by_id.get("gateway-tiny"),
            "ubuntu-gpu-large-v3-turbo": by_id.get("ubuntu-gpu-large-v3-turbo"),
            "openai-gpt-transcribe": by_id.get("openai-gpt-transcribe"),
        }
        for index, (provider_id, _title, _model, _location) in enumerate(PROVIDER_CARD_SPECS, start=1):
            variables = self.ms_provider_vars[provider_id]
            state_var, timing_var, detail_var, _last_var = variables
            profile = aliases.get(provider_id)
            signal = self.ms_provider_signals.get(provider_id)
            if profile is None:
                state_var.set("UNKNOWN")
                timing_var.set("Gateway telemetry —")
                detail_var.set("Provider not advertised")
                if signal is not None:
                    _paint_provider_signal(signal, seed=index, colour=OUTLINE)
                continue
            state_var.set(profile.state_label.upper())
            if profile.probe_duration_ms is not None:
                timing_var.set(f"Worker probe {format_latency_ms(profile.probe_duration_ms)}")
            else:
                timing_var.set(f"Gateway RTT {format_latency_ms(gateway_rtt)}")
            if provider_id == "openai-gpt-transcribe":
                key = "Key configured" if get_session_openai_api_key() else "Key required"
                detail_var.set(f"{key} · availability is verified only by a request")
            elif profile.circuit_retry_after_ms:
                detail_var.set(f"Circuit retry in {format_latency_ms(profile.circuit_retry_after_ms)}")
            else:
                device = profile.device.upper() if profile.device else None
                detail_var.set(" · ".join(part for part in (profile.model_name, device) if part) or "Advertised by gateway")
            if signal is not None:
                state_name = profile.state_label.casefold()
                colour = (
                    MINT
                    if state_name in {"ready", "available with key"}
                    else AMBER
                    if state_name == "starting"
                    else CORAL
                    if state_name == "unavailable"
                    else OUTLINE
                )
                _paint_provider_signal(signal, seed=index, colour=colour)
        self._apply_route_strategy_visuals()

    def _correction_client(self) -> ApiTranscriptionClient:
        config = get_backend_config()
        if config.backend != "api":
            raise ValueError("Select Remote API before managing gateway corrections")
        return ApiTranscriptionClient(config, timeout_seconds=20.0)

    def _refresh_corrections(self) -> None:
        if not hasattr(self, "ms_correction_status_var") or self._corrections_loading:
            return
        self._corrections_loading = True
        self._corrections_generation += 1
        request_generation = self._corrections_generation
        self.ms_correction_status_var.set("Loading correction rules…")

        def worker() -> None:
            try:
                rules = self._correction_client().list_corrections()
            except Exception as exc:
                error_text = str(exc)
                sysmod.enqueue_management_task(
                    self._finish_corrections_request,
                    request_generation,
                    (),
                    error_text,
                )
                return
            safe_rules = tuple(dict(rule) for rule in rules)
            sysmod.enqueue_management_task(
                self._finish_corrections_request,
                request_generation,
                safe_rules,
                None,
            )

        threading.Thread(target=worker, name="CtrlSpeakCorrectionsList", daemon=True).start()

    def _finish_corrections_request(
        self,
        request_generation: int,
        rules: Sequence[Mapping[str, object]],
        error: str | None,
    ) -> None:
        if not self.is_open() or request_generation != self._corrections_generation:
            return
        self._finish_corrections([dict(rule) for rule in rules], error)

    def _finish_corrections(self, rules: list[dict[str, object]], error: str | None) -> None:
        if not self.is_open():
            return
        self._corrections_loading = False
        notice = getattr(self, "_correction_refresh_notice", None)
        self._correction_refresh_notice = None
        if error:
            status = f"Corrections unavailable: {error}"
            if notice:
                status += f" · {notice}"
            self.ms_correction_status_var.set(status)
            if hasattr(self, "ms_route_correction_preview_var"):
                self.ms_route_correction_preview_var.set("Corrections unavailable")
            return
        self._correction_rules = rules
        status = f"{len(rules)} visible correction rule{'s' if len(rules) != 1 else ''}"
        if notice:
            status += f" · {notice}"
        self.ms_correction_status_var.set(status)
        if hasattr(self, "ms_route_correction_preview_var"):
            enabled = [rule for rule in rules if rule.get("enabled")]
            if enabled:
                first = enabled[0]
                source = str(first.get("source_phrase") or "").strip()
                replacement = str(first.get("replacement_phrase") or "").strip()
                preview = f"{source}  →  {replacement}" if source and replacement else f"{len(enabled)} enabled rules"
                remaining = len(enabled) - 1
                if remaining > 0:
                    preview += f"  ·  +{remaining} more"
            else:
                preview = "No enabled correction rules"
            self.ms_route_correction_preview_var.set(preview)
        self._render_corrections()

    def _render_corrections(self) -> None:
        tree = getattr(self, "ms_correction_tree", None)
        if tree is None:
            return
        query = self.ms_correction_search_var.get().strip().casefold()
        tree.delete(*tree.get_children())
        for rule in self._correction_rules:
            source = str(rule.get("source_phrase") or "")
            replacement = str(rule.get("replacement_phrase") or "")
            if query and query not in source.casefold() and query not in replacement.casefold():
                continue
            rule_id = str(rule.get("id") or "")
            tree.insert("", tk.END, iid=rule_id, values=(source, replacement, str(rule.get("scope") or "user"), "Enabled" if rule.get("enabled") else "Disabled", str(rule.get("priority", 0))))

    def _selected_rule(self) -> dict[str, object] | None:
        selection = self.ms_correction_tree.selection()
        if not selection:
            return None
        selected = selection[0]
        return next((rule for rule in self._correction_rules if str(rule.get("id")) == selected), None)

    def _correction_selected(self, _event=None) -> None:
        rule = self._selected_rule()
        if not rule:
            return
        self.ms_selected_rule_id = str(rule.get("id"))
        self.ms_correction_source_var.set(str(rule.get("source_phrase") or ""))
        self.ms_correction_replacement_var.set(str(rule.get("replacement_phrase") or ""))

    def _clear_correction_editor(self) -> None:
        self.ms_selected_rule_id = None
        self.ms_correction_source_var.set("")
        self.ms_correction_replacement_var.set("")
        self.ms_correction_tree.selection_remove(*self.ms_correction_tree.selection())

    def _run_correction_mutation(self, label: str, action: Callable[[ApiTranscriptionClient], object]) -> None:
        self._corrections_generation += 1
        self._corrections_loading = False
        request_generation = self._correction_mutation_generation
        mutation_queue = getattr(self, "_correction_mutation_queue", None)
        if mutation_queue is None:
            mutation_queue = queue.Queue()
            self._correction_mutation_queue = mutation_queue
            self._correction_mutation_worker_active = False
        self._correction_mutations_pending = (
            int(getattr(self, "_correction_mutations_pending", 0)) + 1
        )
        if not hasattr(self, "_correction_mutation_errors"):
            self._correction_mutation_errors = []
        self.ms_correction_status_var.set(
            f"{label}… · {self._correction_mutations_pending} queued"
        )
        mutation_queue.put((request_generation, str(label), action))
        self._start_correction_mutation_worker()

    def _start_correction_mutation_worker(self) -> None:
        if self._correction_mutation_worker_active:
            return
        self._correction_mutation_worker_active = True

        def worker() -> None:
            # One FIFO daemon consumes every queued operation, preserving the
            # order in which toolbar actions were submitted without blocking Tk.
            while True:
                try:
                    request_generation, label, action = (
                        self._correction_mutation_queue.get_nowait()
                    )
                except queue.Empty:
                    break
                error_text: str | None = None
                if request_generation == self._correction_mutation_generation:
                    try:
                        action(self._correction_client())
                    except Exception as exc:
                        error_text = str(exc)
                    sysmod.enqueue_management_task(
                        self._finish_correction_mutation,
                        request_generation,
                        label,
                        error_text,
                    )
                self._correction_mutation_queue.task_done()
            sysmod.enqueue_management_task(self._finish_correction_mutation_worker)

        thread = threading.Thread(
            target=worker,
            name="CtrlSpeakCorrectionMutation",
            daemon=True,
        )
        try:
            thread.start()
        except Exception as exc:
            self._correction_mutation_worker_active = False
            error_text = str(exc)
            while True:
                try:
                    request_generation, queued_label, _action = (
                        self._correction_mutation_queue.get_nowait()
                    )
                except queue.Empty:
                    break
                self._correction_mutation_queue.task_done()
                sysmod.enqueue_management_task(
                    self._finish_correction_mutation,
                    request_generation,
                    queued_label,
                    error_text,
                )

    def _finish_correction_mutation_worker(self) -> None:
        self._correction_mutation_worker_active = False
        if self.is_open() and not self._correction_mutation_queue.empty():
            # Covers the narrow race where Tk queued another operation after
            # the daemon observed Empty but before this completion callback.
            self._start_correction_mutation_worker()

    def _finish_correction_mutation(
        self,
        request_generation: int,
        label: str,
        error: str | None,
    ) -> None:
        if not self.is_open() or request_generation != self._correction_mutation_generation:
            return
        self._correction_mutations_pending = max(
            0, int(getattr(self, "_correction_mutations_pending", 1)) - 1
        )
        if error:
            self._correction_mutation_errors.append(f"{label} failed: {error}")
        if self._correction_mutations_pending:
            suffix = (
                f" · latest issue: {self._correction_mutation_errors[-1]}"
                if self._correction_mutation_errors
                else ""
            )
            self.ms_correction_status_var.set(
                f"{self._correction_mutations_pending} correction change(s) queued{suffix}"
            )
            return
        if self._correction_mutation_errors:
            self._correction_refresh_notice = "Mutation issue: " + "; ".join(
                self._correction_mutation_errors
            )
            self._correction_mutation_errors.clear()
        # Invalidate any list request that started between mutations, then take
        # one authoritative snapshot after every serialized commit has settled.
        self._corrections_generation += 1
        self._corrections_loading = False
        self._refresh_corrections()

    def _validated_correction_fields(self) -> tuple[str, str] | None:
        source = self.ms_correction_source_var.get().strip()
        replacement = self.ms_correction_replacement_var.get().strip()
        if not source or not replacement or source == replacement:
            messagebox.showwarning("Correction required", "Enter two different, non-empty phrases.", parent=self.window)
            return None
        return source, replacement

    def _add_correction(self) -> None:
        values = self._validated_correction_fields()
        if not values:
            return
        source, replacement = values
        self._run_correction_mutation("Adding correction", lambda client: client.create_correction(source, replacement, scope="user"))
        self._clear_correction_editor()

    def _save_selected_correction(self) -> None:
        rule_id = self.ms_selected_rule_id
        values = self._validated_correction_fields()
        if not rule_id or not values:
            return
        source, replacement = values
        self._run_correction_mutation("Saving correction", lambda client: client.update_correction(rule_id, source_phrase=source, replacement_phrase=replacement))

    def _toggle_selected_correction(self) -> None:
        rule = self._selected_rule()
        if not rule:
            return
        rule_id = str(rule.get("id"))
        enabled = not bool(rule.get("enabled"))
        self._run_correction_mutation("Updating correction", lambda client: client.update_correction(rule_id, enabled=enabled))

    def _delete_selected_correction(self) -> None:
        rule = self._selected_rule()
        if not rule:
            return
        if not messagebox.askyesno("Delete correction", f"Delete the rule “{rule.get('source_phrase')}” → “{rule.get('replacement_phrase')}”?", parent=self.window):
            return
        rule_id = str(rule.get("id"))
        self._run_correction_mutation("Deleting correction", lambda client: client.delete_correction(rule_id))
        self._clear_correction_editor()

    def _poll_midnight_state(self) -> None:
        if not self.is_open():
            return
        snapshot = sysmod.transcription_ui_session.snapshot()
        ready = snapshot.phase in {UiPhase.IDLE, UiPhase.SUCCESS}
        self.ms_ready_var.set("● READY" if ready else f"● {snapshot.phase.value.upper()}")
        self.ms_capture_detail_var.set(f"{snapshot.headline} · {snapshot.detail}")
        self.ms_level_var.set(snapshot.level_fraction * 100)
        self.ms_dbfs_var.set(snapshot.level_label)
        self.ms_route_level_var.set(snapshot.level_fraction * 100)
        self.ms_route_dbfs_var.set(snapshot.level_label)
        if snapshot.provider:
            provider = snapshot.provider
            self.ms_last_provider_var.set(f"{provider.display_name} · {provider.latency_label}")
            route = active_route_label(snapshot.attempts)
            self.ms_last_route_var.set(f"{route}{' · fallback used' if snapshot.degraded else ''}")
            self.ms_routing_result_var.set(f"{provider.display_name} · {provider.latency_label}")
            attempt_text = "  ·  ".join(f"{attempt.display_name}: {attempt.duration_label} ({attempt.outcome.value})" for attempt in snapshot.attempts)
            self.ms_attempts_var.set(attempt_text or "No per-provider attempts were reported.")
            for attempt in snapshot.attempts:
                provider_id = "gateway-tiny" if attempt.provider_id == "nova-tiny-whisper" else attempt.provider_id
                variables = self.ms_provider_vars.get(provider_id)
                if variables:
                    variables[3].set(f"Last request {attempt.duration_label} · {attempt.outcome.value}")
        self.ms_backend_status_var.set(self.backend_status_var.get())
        self._shell_poll_job = self.window.after(200, self._poll_midnight_state)

    def _apply_update_event(self, event) -> None:
        super()._apply_update_event(event)
        if not hasattr(self, "ms_check_update_btn"):
            return
        busy = event.state in {"checking", "downloading", "verifying", "launching_updater"}
        self._set_button_enabled(self.ms_check_update_btn, not busy)
        self._set_button_enabled(self.ms_cancel_update_btn, event.state == "downloading")
        install_eligible = self._update_runtime == "packaged_user_writable"
        if event.state == "ready_to_install" and install_eligible:
            self.ms_install_update_btn.configure(text="Restart and install")
            self._set_button_enabled(self.ms_install_update_btn, True)
        elif event.state == "available" and install_eligible:
            self.ms_install_update_btn.configure(text="Download and install")
            self._set_button_enabled(self.ms_install_update_btn, True)
        else:
            self.ms_install_update_btn.configure(text="Download and install")
            self._set_button_enabled(self.ms_install_update_btn, False)

    def refresh_status(self) -> None:
        super().refresh_status()
        if hasattr(self, "ms_backend_status_var"):
            self.ms_backend_status_var.set(self.backend_status_var.get())

    def stop_everything(self) -> None:
        sysmod.stop_client_listener()
        sysmod.shutdown_server()
        root = getattr(self.window, "master", None)
        try:
            if root is not None:
                root.after(120, self._icon.stop)
            else:
                self._icon.stop()
        except Exception:
            pass
        self.close()

    def close(self) -> None:
        from utils import gui as gui_module

        self._capability_generation += 1
        self._corrections_generation += 1
        self._correction_mutation_generation += 1
        self._correction_mutations_pending = 0
        self._correction_mutation_errors.clear()
        self._correction_refresh_notice = None
        coordinator = getattr(self, "_update_coordinator", None)
        if coordinator is not None:
            try:
                coordinator.remove_listener(self._on_update_event)
            except Exception:
                pass
        if self._shell_poll_job and self.is_open():
            try:
                self.window.after_cancel(self._shell_poll_job)
            except tk.TclError:
                pass
        for candidate in (getattr(self, "window", None), getattr(self, "_legacy_window", None)):
            try:
                if candidate is not None and candidate.winfo_exists():
                    candidate.destroy()
            except tk.TclError:
                pass
        gui_module.management_window = None


class MidnightTrayFlyout:
    """Rich left-click tray surface; native right-click menu remains the fallback."""

    def __init__(
        self,
        root: tk.Misc,
        icon,
        *,
        open_control: Callable[[], None],
        open_corrections: Callable[[], None],
        check_updates: Callable[[], None],
        quit_app: Callable[[], None],
    ) -> None:
        self.root = root
        self.icon = icon
        self.open_control = open_control
        self.open_corrections = open_corrections
        self.check_updates = check_updates
        self.quit_app = quit_app
        self.window: tk.Toplevel | None = None
        self._poll_job: str | None = None
        self._capability_generation = 0
        self._capabilities: dict[str, object] = {}
        self._provider_rows: dict[str, tuple[tk.StringVar, tk.StringVar]] = {}
        self._provider_cards: dict[str, ttk.Frame] = {}
        self._provider_badges: dict[str, tk.Canvas] = {}
        self._closing = False

    def is_open(self) -> bool:
        try:
            return bool(self.window and self.window.winfo_exists())
        except Exception:
            return False

    def toggle(self) -> None:
        if self.is_open():
            self.close()
        else:
            self.show()

    def show(self) -> None:
        if self.is_open():
            self.window.lift()
            return
        # A reopened panel must not number Gateway default with an observation
        # made before a config/token change while the flyout was closed.
        self._capabilities = {}
        win = tk.Toplevel(self.root, class_="CtrlSpeakFlyout")
        win.title("CtrlSpeak quick panel")
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.protocol("WM_DELETE_WINDOW", self.close)
        apply_midnight_signal_theme(win)
        bounds = active_monitor_bounds(self.root)
        scale = display_scale(self.root)
        x, y, width, height, needs_scroll = flyout_geometry(bounds, scale)
        win.geometry(f"{width}x{height}+{x}+{y}")
        win.bind("<Escape>", lambda _event: self.close())

        viewport = ttk.Frame(win, style="MS.Root.TFrame")
        viewport.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(
            viewport,
            background=INK,
            highlightthickness=0,
            borderwidth=0,
        )
        scrollbar = ttk.Scrollbar(viewport, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        if needs_scroll:
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        frame = ttk.Frame(canvas, style="MS.Root.TFrame", padding=(18, 16))
        content_id = canvas.create_window((0, 0), window=frame, anchor="nw")

        def sync_scroll_region(_event=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def sync_content_width(event) -> None:
            canvas.itemconfigure(content_id, width=max(1, int(event.width)))

        def mousewheel(event) -> str:
            delta = int(getattr(event, "delta", 0))
            direction = -1 if delta > 0 else 1
            canvas.yview_scroll(direction * 3, "units")
            return "break"

        frame.bind("<Configure>", sync_scroll_region)
        canvas.bind("<Configure>", sync_content_width)
        win.bind("<MouseWheel>", mousewheel)
        win.bind("<Button-4>", lambda _event: canvas.yview_scroll(-3, "units"))
        win.bind("<Button-5>", lambda _event: canvas.yview_scroll(3, "units"))
        win.bind("<Prior>", lambda _event: canvas.yview_scroll(-1, "pages"))
        win.bind("<Next>", lambda _event: canvas.yview_scroll(1, "pages"))
        header = ttk.Frame(frame, style="MS.Root.TFrame")
        header.pack(fill=tk.X)
        ttk.Label(header, text=f"CTRLSPEAK  {sysmod.APP_VERSION}", style="MS.Title.TLabel").pack(side=tk.LEFT)
        ttk.Button(
            header,
            text="Hide panel",
            style="MS.Compact.TButton",
            command=self.close,
        ).pack(side=tk.RIGHT)
        self.status_var = tk.StringVar(value="● READY")
        ttk.Label(header, textvariable=self.status_var, style="MS.Ready.TLabel").pack(side=tk.RIGHT, padx=(0, 9))
        self.route_var = tk.StringVar(value="Hold Right Ctrl to speak")
        ttk.Label(frame, textvariable=self.route_var, style="MS.TLabel", foreground=MUTED).pack(anchor=tk.W, pady=(4, 10))

        mic = _card(frame, padding=(13, 10))
        mic.pack(fill=tk.X)
        self.level_var = tk.DoubleVar(value=0)
        self.level_label_var = tk.StringVar(value="−60.0 dBFS")
        mic_header = ttk.Frame(mic, style="MS.Card.TFrame")
        mic_header.pack(fill=tk.X)
        mic_name = sysmod.get_input_device_preference() or "System default"
        if len(mic_name) > 28:
            mic_name = mic_name[:27] + "…"
        ttk.Label(mic_header, text=f"◉  {mic_name}", style="MS.Card.TLabel").pack(side=tk.LEFT)
        with settings_lock:
            cue_volume = int(settings.get("audio_cue_volume", 30))
            cue_enabled = bool(settings.get("audio_cues_enabled", True))
        audio_label = f"Cues {cue_volume}%" if cue_enabled else "Cues muted"
        ttk.Button(mic_header, text=audio_label, style="MS.Compact.TButton", command=self._open_control).pack(side=tk.RIGHT)
        row = ttk.Frame(mic, style="MS.Card.TFrame")
        row.pack(fill=tk.X, pady=(6, 0))
        ttk.Progressbar(row, variable=self.level_var, maximum=100, style="MS.Horizontal.TProgressbar").pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(row, textvariable=self.level_label_var, style="MS.Card.TLabel", width=12, anchor=tk.E).pack(side=tk.RIGHT, padx=(10, 0))

        hero = ttk.Frame(frame, style="MS.Root.TFrame")
        hero.pack(pady=(11, 9))
        ttk.Label(hero, text="Hold", style="MS.TLabel", font=("Segoe UI Semibold", 14)).pack(side=tk.LEFT)
        ttk.Label(hero, text=" Right Ctrl ", style="MS.TLabel", foreground=CYAN, font=("Segoe UI Semibold", 14)).pack(side=tk.LEFT)
        ttk.Label(hero, text="to speak", style="MS.TLabel", font=("Segoe UI Semibold", 14)).pack(side=tk.LEFT)

        provider_header = ttk.Frame(frame, style="MS.Root.TFrame")
        provider_header.pack(fill=tk.X, pady=(0, 4))
        self.provider_route_label_var = tk.StringVar(value="PROVIDER ROUTE · GATEWAY DEFAULT")
        ttk.Label(provider_header, textvariable=self.provider_route_label_var, style="MS.TLabel", foreground=MUTED, font=("Segoe UI Semibold", 8)).pack(side=tk.LEFT)
        ttk.Button(provider_header, text="Refresh", style="MS.Compact.TButton", command=self._refresh_capabilities).pack(side=tk.RIGHT)
        for index, (provider_id, label, model, _location) in enumerate(PROVIDER_CARD_SPECS, start=1):
            provider = _card(frame, padding=(10, 7))
            provider.pack(fill=tk.X, pady=3)
            number = tk.Canvas(provider, width=24, height=24, background=CARD, highlightthickness=0, borderwidth=0)
            number.pack(side=tk.LEFT, padx=(0, 9))
            _paint_route_badge(number, None, size=24)
            identity = ttk.Frame(provider, style="MS.Card.TFrame")
            identity.pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Label(identity, text=label, style="MS.Card.TLabel", font=("Segoe UI Semibold", 10)).pack(anchor=tk.W)
            ttk.Label(identity, text=model, style="MS.CardMuted.TLabel").pack(anchor=tk.W, pady=(1, 0))
            metrics = ttk.Frame(provider, style="MS.Card.TFrame")
            metrics.pack(side=tk.RIGHT)
            state_var = tk.StringVar(value="Unknown")
            timing_var = tk.StringVar(value="—")
            ttk.Label(metrics, textvariable=state_var, style="MS.CompactMetric.TLabel", anchor=tk.E).pack(anchor=tk.E)
            ttk.Label(metrics, textvariable=timing_var, style="MS.CardMuted.TLabel", anchor=tk.E).pack(anchor=tk.E, pady=(1, 0))
            self._provider_rows[provider_id] = (state_var, timing_var)
            self._provider_cards[provider_id] = provider
            self._provider_badges[provider_id] = number

        last = _card(frame, padding=(12, 9))
        last.pack(fill=tk.X, pady=(8, 0))
        self._provider_following_widget = last
        last_header = ttk.Frame(last, style="MS.Card.TFrame")
        last_header.pack(fill=tk.X)
        ttk.Label(last_header, text="LAST RESULT", style="MS.CardMuted.TLabel").pack(side=tk.LEFT)
        ttk.Label(last_header, text="This session", style="MS.CardMuted.TLabel").pack(side=tk.RIGHT)
        self.last_var = tk.StringVar(value="Nothing transcribed in this session")
        result_row = ttk.Frame(last, style="MS.Card.TFrame")
        result_row.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(result_row, textvariable=self.last_var, style="MS.Card.TLabel", wraplength=270, justify=tk.LEFT).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(result_row, text="Copy", style="MS.Compact.TButton", command=sysmod.copy_last_transcript_from_tray).pack(side=tk.RIGHT, padx=(8, 0))

        actions = ttk.Frame(frame, style="MS.Root.TFrame")
        actions.pack(fill=tk.X, pady=(8, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        ttk.Button(actions, text="＋  Add correction", style="MS.Primary.TButton", command=self._open_corrections).grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=(0, 4))
        ttk.Button(actions, text="Routing", style="MS.TButton", command=self._open_control).grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=(0, 4))
        ttk.Button(actions, text="Audio controls", style="MS.TButton", command=self._open_control).grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(4, 0))
        ttk.Button(actions, text="Open CtrlSpeak", style="MS.TButton", command=self._open_control).grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(4, 0))

        footer = ttk.Frame(frame, style="MS.Root.TFrame")
        footer.pack(fill=tk.X, pady=(9, 0))
        ttk.Label(footer, text=f"●  {sysmod.APP_VERSION}", style="MS.Ready.TLabel").pack(side=tk.LEFT)
        self.cancel_button = ttk.Button(footer, text="Cancel", style="MS.Danger.TButton", command=sysmod.cancel_active_transcription)
        self.cancel_button.pack(side=tk.LEFT, padx=(9, 0))
        ttk.Button(footer, text="Quit", style="MS.Danger.TButton", command=self._quit).pack(side=tk.RIGHT)
        ttk.Button(footer, text="Check updates", style="MS.Compact.TButton", command=self._check_updates).pack(side=tk.RIGHT, padx=(0, 7))
        self._apply_route_strategy_visuals()
        self.window = win
        win.lift()
        win.focus_force()
        self._poll()
        self._refresh_capabilities()

    def _open_control(self) -> None:
        self.close()
        self.open_control()

    def _open_corrections(self) -> None:
        self.close()
        self.open_corrections()

    def _check_updates(self) -> None:
        self.close()
        self.check_updates()

    def _quit(self) -> None:
        self.close()
        self.quit_app()

    def _apply_route_strategy_visuals(self) -> None:
        if not self._provider_cards or not hasattr(self, "_provider_following_widget"):
            return
        strategy_id = get_runtime_backend_config().provider_strategy
        order = provider_order_for_strategy(strategy_id, self._capabilities)
        if order is None:
            visible = tuple(spec[0] for spec in PROVIDER_CARD_SPECS)
            label = "PROVIDER ROUTE · GATEWAY DEFAULT (SERVER ORDER)"
        else:
            visible = order
            names = {provider_id: title for provider_id, title, _model, _location in PROVIDER_CARD_SPECS}
            label = "PROVIDER ROUTE · " + " → ".join(names[item].upper() for item in visible)
        self.provider_route_label_var.set(label)
        for card in self._provider_cards.values():
            card.pack_forget()
        for position, provider_id in enumerate(visible, start=1):
            self._provider_cards[provider_id].pack(
                fill="x",
                pady=3,
                before=self._provider_following_widget,
            )
            _paint_route_badge(
                self._provider_badges[provider_id],
                position if order is not None else None,
                size=24,
            )

    def _poll(self) -> None:
        if not self.is_open():
            return
        snapshot = sysmod.transcription_ui_session.snapshot()
        self.status_var.set(f"● {snapshot.phase.value.upper() if snapshot.phase is not UiPhase.IDLE else 'READY'}")
        if snapshot.phase is UiPhase.IDLE:
            strategy = STRATEGY_LABELS.get(
                get_runtime_backend_config().provider_strategy,
                "Gateway default",
            )
            self.route_var.set(f"{strategy} · Hold Right Ctrl to speak")
        else:
            self.route_var.set(f"{snapshot.headline} · {snapshot.detail}")
        self.level_var.set(snapshot.level_fraction * 100)
        self.level_label_var.set(snapshot.level_label)
        if snapshot.provider:
            self.last_var.set(f"{snapshot.provider.display_name} · {snapshot.provider.latency_label}{' · fallback' if snapshot.degraded else ''}")
        self.cancel_button.state(["!disabled"] if snapshot.can_cancel else ["disabled"])
        self._poll_job = self.window.after(200, self._poll)

    def _refresh_capabilities(self) -> None:
        config = get_runtime_backend_config()
        self._capability_generation += 1
        request_generation = self._capability_generation
        config_identity = capability_config_identity(config)
        if config.backend != "api":
            self._capabilities = {}
            self._apply_route_strategy_visuals()
            return

        def worker() -> None:
            started = time.monotonic()
            try:
                payload = ApiTranscriptionClient(config, timeout_seconds=6.0).get_capabilities()
                rtt = (time.monotonic() - started) * 1000
            except Exception:
                sysmod.enqueue_management_task(
                    self._apply_capabilities_error,
                    request_generation,
                    config_identity,
                )
                return
            sysmod.enqueue_management_task(
                self._apply_capabilities,
                dict(payload),
                rtt,
                request_generation,
                config_identity,
            )

        threading.Thread(target=worker, name="CtrlSpeakFlyoutCapabilities", daemon=True).start()

    def _flyout_capability_request_is_current(
        self,
        request_generation: int,
        config_identity: tuple[object, ...],
    ) -> bool:
        if not self.is_open() or request_generation != self._capability_generation:
            return False
        try:
            return capability_config_identity(get_runtime_backend_config()) == config_identity
        except Exception:
            return False

    def _apply_capabilities_error(
        self,
        request_generation: int,
        config_identity: tuple[object, ...],
    ) -> None:
        if not self._flyout_capability_request_is_current(request_generation, config_identity):
            return
        self._capabilities = {}
        for provider_id, (state_var, timing_var) in self._provider_rows.items():
            if provider_id == "openai-gpt-transcribe":
                key_label = "configured" if get_session_openai_api_key() else "key required"
                state_var.set(f"Unavailable · {key_label}")
            else:
                state_var.set("Unavailable")
            timing_var.set("—")
        self._apply_route_strategy_visuals()

    def _apply_capabilities(
        self,
        payload: Mapping[str, object],
        rtt: float,
        request_generation: int,
        config_identity: tuple[object, ...],
    ) -> None:
        if not self._flyout_capability_request_is_current(request_generation, config_identity):
            return
        self._capabilities = dict(payload)
        profiles = providers_from_capabilities(payload)
        by_id = {profile.provider_id: profile for profile in profiles}
        aliases = {"gateway-tiny": by_id.get("gateway-tiny") or by_id.get("nova-tiny-whisper"), **by_id}
        for provider_id, (state_var, timing_var) in self._provider_rows.items():
            profile = aliases.get(provider_id)
            if profile:
                if provider_id == "openai-gpt-transcribe":
                    key_label = "configured" if get_session_openai_api_key() else "key required"
                    capability_label = profile.state_label
                    if capability_label.casefold() in {"ready", "available with key"}:
                        capability_label = "Available with key"
                    state_var.set(f"{capability_label} · {key_label}")
                else:
                    state_var.set(profile.state_label)
                timing = profile.probe_duration_ms if profile.probe_duration_ms is not None else rtt
                prefix = "Probe" if profile.probe_duration_ms is not None else "Gateway"
                timing_var.set(f"{prefix} {format_latency_ms(timing)}")
            else:
                if provider_id == "openai-gpt-transcribe":
                    key_label = "configured" if get_session_openai_api_key() else "key required"
                    state_var.set(f"Not advertised · {key_label}")
                else:
                    state_var.set("Not advertised")
                timing_var.set("—")
        self._apply_route_strategy_visuals()

    def close(self) -> None:
        if getattr(self, "_closing", False):
            return
        if not self.is_open():
            self.window = None
            self._poll_job = None
            return
        self._closing = True
        try:
            self._capability_generation += 1
            if self._poll_job:
                try:
                    self.window.after_cancel(self._poll_job)
                except tk.TclError:
                    pass
            try:
                self.window.destroy()
            except tk.TclError:
                pass
        finally:
            self._poll_job = None
            self.window = None
            self._closing = False


__all__ = [
    "MidnightSignalManagementMixin",
    "MidnightTrayFlyout",
    "PROVIDER_CARD_SPECS",
    "ROUTING_COLUMN_WEIGHTS",
    "apply_midnight_signal_theme",
    "flyout_geometry",
    "language_summary",
    "provider_signal_levels",
]
