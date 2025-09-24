# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Optional
import tempfile
import threading

CONFIG_FILENAME = "settings.json"
LOG_DIR_NAME = "logs"
ASSETS_DIR_NAME = "assets"

DEFAULT_SETTINGS: Dict[str, object] = {
    "mode": None,                    # "client" | "client_server"
    "server_port": 65432,
    "discovery_port": 54330,
    "preferred_server_host": None,
    "preferred_server_port": None,
    "device_preference": "cpu",
    "input_device": None,
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
    for sub in ("models", "cuda", "temp", LOG_DIR_NAME):
        (config_dir / sub).mkdir(parents=True, exist_ok=True)
    return config_dir


def get_logs_dir() -> Path:
    logs_dir = get_config_dir() / LOG_DIR_NAME
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def get_assets_dir() -> Path:
    base_dir = get_app_base_dir()
    return base_dir / ASSETS_DIR_NAME

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
        logger = get_logger()
        logger.exception("Failed to remove temporary recording file: %s", path)

def load_settings() -> Dict[str, object]:
    path = get_config_file_path()
    loaded = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text("utf-8-sig"))
        except Exception:
            get_logger().exception("Unable to read settings from %s", path)
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
        get_logger().exception("Unable to save settings to %s", path)


def get_app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        # Running from a bundled executable (PyInstaller)
        base_path = getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent)
        return Path(base_path)
    return Path(__file__).resolve().parent.parent


def resource_path(relative_path: str) -> str:
    return str(get_app_base_dir() / relative_path)


def asset_path(relative_name: str) -> Path:
    return get_assets_dir() / relative_name


LOGGER_NAME = "ctrlspeak"
_LOG_HANDLER: Optional[RotatingFileHandler] = None
_CONSOLE_HANDLER: Optional[logging.Handler] = None
_LOG_CONFIG_LOCK = threading.Lock()


def _configure_logging() -> logging.Logger:
    global _LOG_HANDLER, _CONSOLE_HANDLER

    with _LOG_CONFIG_LOCK:
        ctrl_logger = logging.getLogger(LOGGER_NAME)
        if _LOG_HANDLER is None:
            logs_dir = get_logs_dir()
            handler = RotatingFileHandler(
                logs_dir / "ctrlspeak.log",
                maxBytes=1_048_576,
                backupCount=5,
                encoding="utf-8",
            )
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            handler.setFormatter(formatter)
            handler.setLevel(logging.DEBUG)
            _LOG_HANDLER = handler

            root_logger = logging.getLogger()
            root_logger.addHandler(handler)
            if root_logger.level == logging.NOTSET or root_logger.level > logging.INFO:
                root_logger.setLevel(logging.INFO)

            if _CONSOLE_HANDLER is None:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.WARNING)
                console_handler.setFormatter(formatter)
                root_logger.addHandler(console_handler)
                _CONSOLE_HANDLER = console_handler

            logging.captureWarnings(True)

        ctrl_logger.setLevel(logging.INFO)
        ctrl_logger.propagate = True
        return ctrl_logger


_CONFIGURED_LOGGER = _configure_logging()


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    logger = _CONFIGURED_LOGGER if name == LOGGER_NAME else logging.getLogger(name)
    if logger is not _CONFIGURED_LOGGER:
        _configure_logging()
    return logger
