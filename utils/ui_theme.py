from __future__ import annotations

import tkinter as tk
from tkinter import ttk

# NOTE ABOUT FONTS
# -----------------
# Windows parses Tk font descriptors token-by-token, so multi-word font
# families such as "Segoe UI" need to be wrapped in braces when provided as a
# string literal.  The previous implementation passed values like "Segoe UI 10"
# into option_add / style.configure, which works on macOS/Linux but raises a
# TclError on Windows because Tk tries to treat "UI" as the font size.  By
# explicitly bracing the family portion we keep the value intact and prevent
# "expected integer but got 'UI'" crashes when creating themed widgets.

# Core palette
BACKGROUND = "#040913"
SURFACE = "#0d1a2b"
ELEVATED_SURFACE = "#122238"
ACCENT = "#3ad6ff"
ACCENT_HOVER = "#5ae1ff"
ACCENT_PRESSED = "#2fbfe3"
TEXT_PRIMARY = "#e7f2ff"
TEXT_SECONDARY = "#8fa4c4"
OUTLINE = "#1b2c44"
DANGER = "#ff4d6d"
DANGER_HOVER = "#ff6f88"
DANGER_PRESSED = "#f03a5a"
MUTED_BUTTON = "#162a43"
MUTED_BUTTON_HOVER = "#1f3554"
MUTED_BUTTON_DISABLED = "#0f1b2c"


def apply_modern_theme(root: tk.Misc) -> ttk.Style:
    """Apply a modern dark style to the provided Tk widget tree."""
    style = ttk.Style(master=root)
    try:
        style.theme_use("clam")
    except Exception:
        # Fall back silently if the theme isn't available
        pass

    # Global widget defaults
    if isinstance(root, tk.Tk) or isinstance(root, tk.Toplevel):
        root.configure(bg=BACKGROUND)
    try:
        root.option_add("*Font", "{Segoe UI} 10")
        root.option_add("*Label.Font", "{Segoe UI} 10")
        root.option_add("*Background", BACKGROUND)
        root.option_add("*Foreground", TEXT_PRIMARY)
        root.option_add("*Entry*Foreground", TEXT_PRIMARY)
        root.option_add("*Entry*Background", ELEVATED_SURFACE)
        root.option_add("*Entry*InsertBackground", ACCENT)
        root.option_add("*TCombobox*Listbox*Background", ELEVATED_SURFACE)
        root.option_add("*TCombobox*Listbox*Foreground", TEXT_PRIMARY)
    except Exception:
        pass

    # Base frames/labels
    style.configure("Modern.TFrame", background=BACKGROUND)
    style.configure("ModernCard.TFrame", background=ELEVATED_SURFACE, relief="flat")
    style.configure("ModernCardInner.TFrame", background=ELEVATED_SURFACE, relief="flat")
    style.configure("Modern.TLabel", background=BACKGROUND, foreground=TEXT_PRIMARY)
    style.configure("Body.TLabel", background=ELEVATED_SURFACE, foreground=TEXT_PRIMARY,
                    font=("{Segoe UI}", 10))
    style.configure("Title.TLabel", background=ELEVATED_SURFACE, foreground=TEXT_PRIMARY,
                    font=("{Segoe UI Semibold}", 18))
    style.configure("Subtitle.TLabel", background=ELEVATED_SURFACE, foreground=TEXT_SECONDARY,
                    font=("{Segoe UI}", 11))
    style.configure("SectionHeading.TLabel", background=ELEVATED_SURFACE, foreground=ACCENT,
                    font=("{Segoe UI Semibold}", 12))
    style.configure("Caption.TLabel", background=ELEVATED_SURFACE, foreground=TEXT_SECONDARY,
                    font=("{Segoe UI}", 9))

    # Buttons
    style.configure("Accent.TButton", background=ACCENT, foreground=BACKGROUND,
                    borderwidth=0, focusthickness=1, focuscolor=ACCENT,
                    padding=(14, 8))
    style.map(
        "Accent.TButton",
        background=[("disabled", MUTED_BUTTON_DISABLED), ("pressed", ACCENT_PRESSED), ("active", ACCENT_HOVER)],
        foreground=[("disabled", TEXT_SECONDARY)],
    )

    style.configure("Subtle.TButton", background=MUTED_BUTTON, foreground=TEXT_PRIMARY,
                    borderwidth=0, focusthickness=1, focuscolor=OUTLINE,
                    padding=(14, 8))
    style.map(
        "Subtle.TButton",
        background=[("disabled", MUTED_BUTTON_DISABLED), ("pressed", MUTED_BUTTON), ("active", MUTED_BUTTON_HOVER)],
        foreground=[("disabled", TEXT_SECONDARY)],
    )

    style.configure("Danger.TButton", background=DANGER, foreground=BACKGROUND,
                    borderwidth=0, focusthickness=1, focuscolor=DANGER,
                    padding=(14, 8))
    style.map(
        "Danger.TButton",
        background=[("disabled", "#5a3240"), ("pressed", DANGER_PRESSED), ("active", DANGER_HOVER)],
        foreground=[("disabled", "#b37b8a")],
    )

    # Radio buttons / checkbuttons
    style.configure("Modern.TRadiobutton", background=ELEVATED_SURFACE, foreground=TEXT_PRIMARY,
                    focuscolor=ACCENT)
    style.map(
        "Modern.TRadiobutton",
        foreground=[("disabled", TEXT_SECONDARY)],
        indicatorcolor=[("selected", ACCENT), ("!selected", OUTLINE)],
    )

    # Combobox
    style.configure("Modern.TCombobox", fieldbackground=ELEVATED_SURFACE, foreground=TEXT_PRIMARY,
                    background=ELEVATED_SURFACE, bordercolor=OUTLINE, lightcolor=ELEVATED_SURFACE,
                    darkcolor=ELEVATED_SURFACE, arrowcolor=ACCENT)
    style.map("Modern.TCombobox",
              fieldbackground=[("readonly", ELEVATED_SURFACE)],
              foreground=[("disabled", TEXT_SECONDARY)])

    # Progressbar and separators
    style.configure("Modern.Horizontal.TProgressbar", troughcolor=MUTED_BUTTON_DISABLED,
                    background=ACCENT, bordercolor=MUTED_BUTTON_DISABLED,
                    lightcolor=ACCENT, darkcolor=ACCENT)
    style.configure("Modern.Horizontal.TSeparator", background=OUTLINE)

    # Accent line helper
    style.configure("AccentLine.TFrame", background=ACCENT)

    return style
