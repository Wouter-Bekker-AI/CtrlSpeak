# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, Optional
import tempfile
import threading

CONFIG_FILENAME = "settings.json"
DEFAULT_SETTINGS: Dict[str, object] = {
    "mode": None,                    # "client" | "client_server"
    "server_port": 65432,
    "discovery_port": 54330,
    "preferred_server_host": None,
    "preferred_server_port": None,
    "device_preference": "cpu",
    "model_name": "small",
}

settings_lock = threading.RLock()
settings: Dict[str, object] = {}

def get_config_dir() -> Path:
    """
    All persistent data goes here:
      %APPDATA%/CtrlSpeak (Windows)
      $XDG_CONFIG_HOME/CtrlSpeak or ~/.config/CtrlSpeak (others)
    Subfolders used:
      models/  cuda/  temp/  (plus settings.json + logs)
    """
    if sys.platform.startswith("win"):
        base_dir = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base_dir = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    config_dir = base_dir / "CtrlSpeak"
    config_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("models", "cuda", "temp"):
        (config_dir / sub).mkdir(parents=True, exist_ok=True)
    return config_dir

def get_config_file_path() -> Path:
    return get_config_dir() / CONFIG_FILENAME

def get_temp_dir() -> Path:
    temp_dir = get_config_dir() / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir

def create_recording_file_path() -> Path:
    return get_temp_dir() / f"recording-{uuid.uuid4().hex}.wav"

def cleanup_recording_file(path: Optional[Path]) -> None:
    if not path:
        return
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass

def load_settings() -> Dict[str, object]:
    path = get_config_file_path()
    loaded = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text("utf-8-sig"))
        except Exception:
            pass
    with settings_lock:
        settings.clear()
        settings.update(DEFAULT_SETTINGS)
        settings.update(loaded)
        return dict(settings)

def save_settings() -> None:
    path = get_config_file_path()
    with settings_lock:
        snapshot = dict(settings)
    try:
        path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    except Exception:
        pass

def resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS  # pyinstaller
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)
