"""Isolated native guide preview/QA: no user settings, hooks, audio or API calls."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify",action="store_true")
    parser.add_argument("--mode",choices=("hold","hands_free","live"),default="hold")
    parser.add_argument("--duration",type=int,default=180)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="ctrlspeak-guide-qa-") as config:
        os.environ["APPDATA"]=os.environ["XDG_CONFIG_HOME"]=config
        import tkinter as tk
        from utils.midnight_signal_ui import apply_midnight_signal_theme
        from utils.keyboard_guide_ui import KeyboardGuide
        root=tk.Tk(className="CtrlSpeakGuidePreview")
        root.title("CtrlSpeak 0.8.3 · Keyboard guide preview · isolated")
        root.geometry("1000x820")
        apply_midnight_signal_theme(root)
        guide=KeyboardGuide(root)
        guide.pack(fill="both",expand=True,padx=24,pady=20)
        guide.select_mode(args.mode)
        root.update()
        guide.set_visible(True)
        try:
            if args.verify:
                for mode in ("hold","hands_free","live"):
                    guide.select_mode(mode)
                    for index,step in enumerate(guide.playback.lesson.steps):
                        guide.playback.seek(index);guide._render();root.update()
                        assert guide.step_var.get()==step.title
                        assert len(guide.canvas.find_withtag("guide-key"))==9
                        if step.state=="muted":assert guide.canvas.find_withtag("guide-muted-mic")
                guide.configure_keys("f8","f9",reduced_motion=True)
                guide.replay();guide.seek(1);root.update()
                assert guide._job is None and guide.playback.index==1
                assert "F9 + F8" in guide.shortcut_var.get()
                guide.configure_keys("ctrl_r","ctrl_l",reduced_motion=False)
                guide.select_mode("hands_free")
                deadline=time.monotonic()+.35
                while time.monotonic()<deadline:root.update();time.sleep(.01)
                assert guide.playback.elapsed_ms>100
                guide.set_visible(False)
                assert guide._job is None and not guide.playback.playing
                guide.set_visible(True);guide.toggle_play();root.update()
                for geometry in ("1000x820","760x650"):
                    root.geometry(geometry);root.update()
                    for button in (guide.play_button,guide.next_button,guide.previous_button):
                        assert button.winfo_viewable()
                        assert button.winfo_rooty()+button.winfo_height() <= root.winfo_rooty()+root.winfo_height()
                guide.shutdown();assert guide._job is None
                print(json.dumps({"ok":True,"native_tk":True,"all_mode_frames":True,
                    "remapped_keys":True,"reduced_motion":True,"timer_cleanup":True,"simulation_only":True}))
            else:
                root.after(max(1,min(args.duration,600))*1000,root.destroy)
                root.mainloop()
        finally:
            guide.shutdown()
            try:root.destroy()
            except tk.TclError:pass
            # The isolated theme import owns a rotating log under this temporary
            # profile. Close it before TemporaryDirectory cleanup on Windows.
            import logging
            logging.shutdown()
    return 0


if __name__=="__main__":raise SystemExit(main())
