from types import SimpleNamespace
from dataclasses import replace
import json
import pytest
from utils.ui_state import format_microphone_level,TranscriptionUiSession
from utils.ui_preferences import microphone_level_label,microphone_meter_self_test
from utils import config_paths

pytestmark=pytest.mark.core_headless


@pytest.fixture(autouse=True)
def _headless_tk_contract(monkeypatch):
    import tkinter as tk
    if not hasattr(tk,"messagebox"):
        monkeypatch.setattr(tk,"messagebox",SimpleNamespace(),raising=False)
    monkeypatch.setattr(tk,"ROUND","round",raising=False)


@pytest.mark.parametrize("value,expected",[(-90,0),(-60,0),(-45,25),(-30,50),(-20,67),(-18,70),(-6,90),(0,100),(10,100)])
def test_default_is_bounded_zero_to_one_hundred(value,expected):
    assert format_microphone_level(value)==f"{expected} / 100"


@pytest.mark.parametrize("value",[None,False,True,"invalid",float("nan"),float("inf"),float("-inf")])
def test_invalid_values_are_safe_quiet_readouts(value):
    assert format_microphone_level(value)=="0 / 100"
    assert format_microphone_level(value,show_dbfs=True)=="-60.0 dBFS"


def test_technical_readout_and_dynamic_switch_preserve_measured_snapshot():
    state=replace(TranscriptionUiSession().snapshot(),level_dbfs=-18,level_fraction=.7,level_label="-18.0 dBFS")
    assert microphone_level_label(state.level_dbfs)=="70 / 100"
    config_paths.settings["microphone_level_dbfs"]=True
    assert microphone_level_label(state.level_dbfs)=="-18.0 dBFS"
    config_paths.settings["microphone_level_dbfs"]=False
    assert microphone_level_label(state.level_dbfs)=="70 / 100"
    assert state.level_fraction==.7 and state.level_dbfs==-18 and state.level_label=="-18.0 dBFS"


@pytest.mark.parametrize("saved",[True,False])
def test_preference_survives_save_and_reload(saved):
    config_paths.settings["microphone_level_dbfs"]=saved
    assert config_paths.save_settings()
    config_paths.settings.clear()
    assert config_paths.load_settings()["microphone_level_dbfs"] is saved
    assert microphone_level_label(-30)==("-30.0 dBFS" if saved else "50 / 100")


@pytest.mark.parametrize("saved",[None,"true",1,[],{}])
def test_invalid_persisted_preference_falls_back_without_losing_other_fields(saved):
    path=config_paths.get_config_file_path()
    path.write_text(json.dumps({"settings_schema_version":4,"microphone_level_dbfs":saved,
                               "audio_cue_volume":37,"future_field":{"keep":True}}))
    loaded=config_paths.load_settings()
    assert loaded["microphone_level_dbfs"] is False
    assert loaded["audio_cue_volume"]==37 and loaded["future_field"]=={"keep":True}


class Variable:
    def __init__(self,value):self.value=value
    def get(self):return self.value
    def set(self,value):self.value=value


@pytest.mark.parametrize("save_ok",[True,False])
@pytest.mark.parametrize("previous",[True,False])
def test_checkbox_save_and_failed_save_restore_previous_choice(monkeypatch,save_ok,previous):
    from utils import midnight_signal_ui as ui
    monkeypatch.setattr(ui,"settings",config_paths.settings)
    monkeypatch.setattr(ui,"settings_lock",config_paths.settings_lock)
    monkeypatch.setattr(ui,"save_settings",lambda:save_ok)
    notices=[]
    monkeypatch.setattr(ui.messagebox,"showerror",lambda *a,**k:notices.append(a),raising=False)
    config_paths.settings["microphone_level_dbfs"]=previous
    obj=SimpleNamespace(ms_meter_dbfs_var=Variable(not previous),window=object())
    ui.MidnightSignalManagementMixin._save_meter_preference(obj)
    expected=not previous if save_ok else previous
    assert config_paths.settings["microphone_level_dbfs"] is expected
    assert obj.ms_meter_dbfs_var.get() is expected
    assert bool(notices) is not save_ok


@pytest.mark.parametrize("width",[360,548])
@pytest.mark.parametrize("technical",[False,True])
def test_real_overlay_renderer_uses_preference_in_both_layouts(width,technical):
    from utils.midnight_overlay import MidnightSignalOverlay
    config_paths.settings["microphone_level_dbfs"]=technical
    class Canvas:
        def __init__(self):self.text=[]
        def create_line(self,*a,**k):pass
        def create_text(self,*a,**k):self.text.append(k["text"])
    overlay=object.__new__(MidnightSignalOverlay)
    overlay._draw_microphone=lambda *a:None
    overlay.waveform_provider=lambda:[0.1]*100
    state=replace(TranscriptionUiSession().snapshot(),level_dbfs=-18,elapsed_label="1.0s")
    canvas=Canvas()
    overlay._draw_recording(canvas,width,88,state)
    assert ("-18.0 dBFS" if technical else "70 / 100") in canvas.text
    assert ("70 / 100" if technical else "-18.0 dBFS") not in canvas.text


def test_all_numeric_ui_surfaces_use_the_same_live_formatter():
    from pathlib import Path
    source=(Path(__file__).resolve().parents[2]/"utils/midnight_signal_ui.py").read_text("utf-8")
    for name in ("ms_dbfs_var","ms_route_dbfs_var","level_label_var"):
        assert f"self.{name}.set(microphone_level_label(snapshot.level_dbfs))" in source
    assert "Show technical mic level (RMS dBFS)" in source
    assert microphone_meter_self_test()=={"ok":True,"default_scale":"0-100","dbfs_optional":True}
