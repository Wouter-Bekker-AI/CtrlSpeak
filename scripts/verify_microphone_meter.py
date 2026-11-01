"""Native Tk readout/checkbox QA, isolated from user settings and hardware."""
import json,logging,os,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    with tempfile.TemporaryDirectory(prefix="ctrlspeak-meter-qa-") as folder:
        os.environ['APPDATA']=os.environ['XDG_CONFIG_HOME']=folder
        import tkinter as tk
        from tkinter import ttk
        from utils import config_paths
        from utils.midnight_signal_ui import MidnightSignalManagementMixin,apply_midnight_signal_theme
        from utils.midnight_overlay import MidnightSignalOverlay
        from utils.ui_state import TranscriptionUiSession
        from utils.ui_preferences import microphone_level_label
        config_paths.load_settings()
        root=tk.Tk();root.withdraw();apply_midnight_signal_theme(root)
        try:
            variable=tk.BooleanVar(value=False)
            owner=SimpleNamespace(ms_meter_dbfs_var=variable,window=root)
            checkbox=ttk.Checkbutton(root,variable=variable,
                command=lambda:MidnightSignalManagementMixin._save_meter_preference(owner))
            overlay=object.__new__(MidnightSignalOverlay)
            overlay._draw_microphone=lambda *a:None
            overlay.waveform_provider=lambda:[.1]*100
            snapshot=replace(TranscriptionUiSession().snapshot(),level_dbfs=-30,level_fraction=.5)
            for technical,expected in ((False,'50 / 100'),(True,'-30.0 dBFS'),(False,'50 / 100')):
                if variable.get()!=technical:checkbox.invoke()
                assert config_paths.load_settings()['microphone_level_dbfs'] is technical
                assert microphone_level_label(-30)==expected
                for width in (360,548):
                    canvas=tk.Canvas(root,width=width,height=88)
                    overlay._draw_recording(canvas,width,88,snapshot)
                    items=[i for i in canvas.find_all() if canvas.type(i)=='text' and canvas.itemcget(i,'text')==expected]
                    assert len(items)==1
                    root.update_idletasks()
                    x1,_,x2,_=canvas.bbox(items[0]);assert 0<=x1<x2<=width
                    canvas.destroy()
            print(json.dumps({'ok':True,'native_checkbox_persistence':True,'both_overlay_layouts':True,'no_audio_or_input':True}))
        finally:
            root.destroy();logging.shutdown()
    return 0


if __name__=='__main__':raise SystemExit(main())
