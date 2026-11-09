"""Exercise actual Tk mapping and update controls without using the installed app."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import logging
import os
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    with tempfile.TemporaryDirectory(prefix="ctrlspeak-control-centre-") as folder:
        os.environ["APPDATA"] = os.environ["XDG_CONFIG_HOME"] = folder
        import tkinter as tk
        from tkinter import ttk
        from utils import gui
        from utils.config_paths import load_settings
        from utils.update_manager import UpdateEvent

        load_settings()
        gui.ensure_management_ui_thread()
        root = gui.tk_root
        constructed, mapped, raised = [], [], []
        original_toplevel = tk.Toplevel

        class ObservedToplevel(original_toplevel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                constructed.append(self)

            def deiconify(self):
                raised.append(self)
                return super().deiconify()

        def on_map(event):
            if isinstance(event.widget, original_toplevel):
                mapped.append(event.widget)

        root.bind_all("<Map>", on_map, add="+")
        icon = SimpleNamespace(title="QA", update_menu=lambda: None)
        try:
            with (
                patch.object(tk, "Toplevel", ObservedToplevel),
                patch.object(gui, "classify_runtime", return_value="packaged_user_writable"),
                patch.object(gui, "list_input_audio_devices", return_value=[]),
                patch.object(gui.ManagementWindow, "_refresh_gateway_capabilities"),
                patch.object(gui.ManagementWindow, "_refresh_corrections"),
            ):
                for _ in range(2):
                    constructed.clear()
                    mapped.clear()
                    raised.clear()
                    gui._show_management_page(icon, "Updates")
                    owner = gui.management_window
                    root.update()
                    assert owner.window.winfo_ismapped()
                    assert constructed == [owner.window]
                    assert mapped == [owner.window] and raised == [owner.window]
                    assert owner.install_update_btn.instate(["disabled"])
                    assert owner.view_release_btn.instate(["disabled"])

                    release = SimpleNamespace(version="9.0.0")
                    owner._apply_update_event(UpdateEvent(1, "available", "Verified update", release=release))
                    assert owner.install_update_btn.instate(["!disabled"])
                    assert owner.view_release_btn.instate(["!disabled"])
                    owner._apply_update_event(UpdateEvent(2, "checking", "Checking…"))
                    owner._apply_update_event(UpdateEvent(1, "available", "Stale", release=release))
                    assert owner._update_event.state == "checking"
                    assert owner.install_update_btn.instate(["disabled"])
                    owner._apply_update_event(UpdateEvent(2, "failed", "Check failed"))
                    assert owner.install_update_btn.instate(["disabled"])
                    assert owner.view_release_btn.instate(["disabled"])
                    assert owner._update_release is None

                    updates = owner.ms_pages["Updates"]
                    assert owner.check_update_btn.master is not owner.view_release_btn.master
                    entry = ttk.Entry(updates)
                    assert owner._scroll_management_key(SimpleNamespace(widget=entry, keysym="Home")) is None
                    entry.destroy()
                    assert owner._scroll_management_key(SimpleNamespace(widget=updates, keysym="End")) == "break"
                    assert not hasattr(owner, "ms_rollback_btn")
                    owner._set_output_language_selection(("en",))
                    assert owner._selected_output_languages() == ("en",)
                    assert owner.audio_device_combo.cget("values")
                    assert owner.audio_device_status.get()
                    owner.close()
                    root.update()
            print("Native Tk: only Midnight Signal mapped; reopen, update controls and keyboard navigation passed.")
        finally:
            gui._teardown_management_ui()
            logging.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
