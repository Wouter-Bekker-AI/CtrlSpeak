"""Open the v0.7 Midnight Signal shell for manual/visual QA.

This harness does not start the hotkey listener or transcription engine.  It
uses the normal saved CtrlSpeak configuration so read-only gateway cards and
correction listings can be inspected without competing with an installed app.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_paths import load_settings
from utils import gui


class _SmokeIcon:
    title = "CtrlSpeak 0.7 UI smoke"

    @staticmethod
    def stop() -> None:
        gui.request_management_ui_shutdown()

    @staticmethod
    def update_menu() -> None:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page", default="Capture")
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument(
        "--overlay",
        choices=("recording", "processing", "success", "error"),
    )
    parser.add_argument("--flyout", action="store_true")
    args = parser.parse_args()
    load_settings()
    gui.ensure_management_ui_thread()
    gui._show_management_window(_SmokeIcon())
    if gui.management_window is not None:
        pages = getattr(gui.management_window, "ms_pages", {})
        page = pages.get(args.page)
        if page is not None:
            gui.management_window.ms_notebook.select(page)
    if args.overlay:
        session = gui.sysmod.transcription_ui_session
        if session.phase.value != "idle":
            session.reset()
        session.begin_recording()
        pcm_sample = int(32767 * 0.12).to_bytes(2, "little", signed=True)
        session.update_level_pcm16(pcm_sample * 1024)
        waveform = np.sin(np.linspace(0, np.pi * 8, 1024)).astype(np.float32) * 0.32
        gui.show_waveform_overlay(lambda: waveform)
        if args.overlay in {"processing", "success", "error"}:
            session.begin_processing()
            gui.set_waveform_processing()
        if args.overlay == "success":
            session.complete(
                {
                    "provider_used": "ubuntu-gpu-large-v3-turbo",
                    "attempts": [
                        {
                            "provider": "ubuntu-gpu-large-v3-turbo",
                            "status": "succeeded",
                            "duration_ms": 842,
                            "inference_duration_ms": 711,
                        }
                    ],
                    "routing_duration_ms": 842,
                },
                elapsed_ms=1040,
            )
        elif args.overlay == "error":
            session.fail("providers_exhausted", elapsed_ms=1320)
    if args.flyout:
        gui._show_tray_flyout(_SmokeIcon())
    if gui.tk_root is not None and args.duration > 0:
        gui.tk_root.after(args.duration * 1000, gui.request_management_ui_shutdown)
    gui.run_management_ui_loop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
