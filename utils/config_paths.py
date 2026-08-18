# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Dict, Optional
import tempfile
import threading
from urllib.parse import urlsplit

from utils.version import APP_VERSION
from utils.languages import is_valid_allowed_output_languages_setting

CONFIG_FILENAME = "settings.json"
LOG_DIR_NAME = "logs"
ASSETS_DIR_NAME = "assets"
SETTINGS_SCHEMA_VERSION = 4
SETTINGS_BACKUP_PREFIX = "settings.pre-migration-v2"

DEFAULT_SETTINGS: Dict[str, object] = {
    "settings_schema_version": SETTINGS_SCHEMA_VERSION,
    "mode": None,                    # "client" | "client_server"
    "server_port": 65432,
    "discovery_port": 54363,
    "preferred_server_host": None,
    "preferred_server_port": None,
    "device_preference": "cpu",
    "input_device": None,
    "model_name": "small",
    "model_auto_install_complete": False,
    "transcription_backend": "bundled",
    "api_url": "http://127.0.0.1:8765",
    "api_token": None,
    "feedback_capture_method": "active_field_on_enter",
    "allowed_output_languages": [],
    "provider_strategy": "server-default",
    "update_channel": "stable",
    "last_update_check_at": None,
    "show_whats_new_on_update": True,
    "whats_new_last_seen_version": APP_VERSION,
    "overlay_enabled": True,
    "reduced_motion": False,
    "audio_cues_enabled": True,
    "audio_cue_volume": 30,
}

settings_lock = threading.RLock()
settings: Dict[str, object] = {}


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_optional_string(value: object) -> bool:
    return value is None or isinstance(value, str)


def _is_optional_nonempty_string(value: object) -> bool:
    return value is None or (isinstance(value, str) and bool(value.strip()))


def _is_port(value: object) -> bool:
    return _is_plain_int(value) and 1 <= int(value) <= 65535


def _is_http_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return False
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


_SETTING_VALIDATORS: Dict[str, Callable[[object], bool]] = {
    "settings_schema_version": lambda value: _is_plain_int(value) and int(value) >= 1,
    "mode": lambda value: value is None or value in {"client", "client_server"},
    "server_port": _is_port,
    "discovery_port": _is_port,
    "preferred_server_host": _is_optional_nonempty_string,
    "preferred_server_port": lambda value: value is None or _is_port(value),
    "device_preference": lambda value: value in {"cpu", "cuda"},
    "input_device": _is_optional_string,
    "model_name": lambda value: value in {"small", "large-v3"},
    "model_auto_install_complete": lambda value: isinstance(value, bool),
    "transcription_backend": lambda value: value in {"bundled", "api"},
    "api_url": _is_http_url,
    "api_token": _is_optional_string,
    "feedback_capture_method": lambda value: value in {"active_field_on_enter", "disabled"},
    "allowed_output_languages": is_valid_allowed_output_languages_setting,
    "provider_strategy": lambda value: isinstance(value, str) and bool(value.strip()),
    "update_channel": lambda value: value == "stable",
    "last_update_check_at": _is_optional_string,
    "show_whats_new_on_update": lambda value: isinstance(value, bool),
    "whats_new_last_seen_version": lambda value: isinstance(value, str),
    "overlay_enabled": lambda value: isinstance(value, bool),
    "reduced_motion": lambda value: isinstance(value, bool),
    "audio_cues_enabled": lambda value: isinstance(value, bool),
    "audio_cue_volume": lambda value: _is_plain_int(value) and 0 <= int(value) <= 100,
}

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


def cleanup_stale_recordings() -> int:
    """Remove only CtrlSpeak-generated recording WAVs after single-instance lock.

    Callers must hold the application's single-instance lock so no live desktop
    generation can own a matching path.  The exact filename pattern and direct
    parent check keep cleanup bounded to CtrlSpeak's private temp directory.
    """

    temp_dir = get_temp_dir().resolve()
    removed = 0
    for candidate in temp_dir.glob("recording-*.wav"):
        try:
            if candidate.parent.resolve() != temp_dir:
                continue
            candidate.unlink(missing_ok=True)
            removed += 1
        except Exception:
            get_logger().exception("Failed to remove stale CtrlSpeak recording")
    return removed


def _settings_backup_path(path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return path.with_name(
        f"{SETTINGS_BACKUP_PREFIX}.{timestamp}.{uuid.uuid4().hex[:8]}.json"
    )


def _backup_settings_file(path: Path) -> Optional[Path]:
    if not path.is_file():
        return None
    backup_path = _settings_backup_path(path)
    try:
        shutil.copy2(path, backup_path)
        if not sys.platform.startswith("win"):
            backup_path.chmod(0o600)
        return backup_path
    except Exception:
        get_logger().exception("Unable to back up settings before migration: %s", path)
        return None


def _atomic_write_settings(path: Path, snapshot: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        if not sys.platform.startswith("win"):
            os.chmod(temporary_path, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        if not sys.platform.startswith("win"):
            path.chmod(0o600)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _salvage_settings(loaded: Dict[str, object]) -> tuple[Dict[str, object], list[str]]:
    salvaged = dict(DEFAULT_SETTINGS)
    invalid_fields: list[str] = []

    # Unknown fields are retained for rollback/forward compatibility. Known
    # fields are accepted independently so one malformed value cannot erase
    # unrelated credentials or user preferences.
    for key, value in loaded.items():
        validator = _SETTING_VALIDATORS.get(key)
        if validator is None:
            salvaged[key] = value
            continue
        try:
            valid = validator(value)
        except Exception:
            valid = False
        if valid:
            salvaged[key] = value
        else:
            invalid_fields.append(key)

    stored_schema = loaded.get("settings_schema_version", 0)
    if not _is_plain_int(stored_schema) or int(stored_schema) < SETTINGS_SCHEMA_VERSION:
        salvaged["settings_schema_version"] = SETTINGS_SCHEMA_VERSION
    return salvaged, invalid_fields


def load_settings() -> Dict[str, object]:
    path = get_config_file_path()
    loaded: Dict[str, object] = {}
    needs_migration = False
    if path.exists():
        try:
            decoded = json.loads(path.read_text("utf-8-sig"))
            if not isinstance(decoded, dict):
                raise ValueError("settings root must be a JSON object")
            loaded = decoded
        except Exception:
            get_logger().exception("Unable to read settings from %s", path)
            needs_migration = True

    salvaged, invalid_fields = _salvage_settings(loaded)
    stored_schema = loaded.get("settings_schema_version", 0)
    if path.exists() and (
        not _is_plain_int(stored_schema)
        or int(stored_schema) < SETTINGS_SCHEMA_VERSION
        or bool(invalid_fields)
    ):
        needs_migration = True

    if invalid_fields:
        get_logger().warning(
            "Invalid settings fields fell back to safe defaults: %s",
            ", ".join(sorted(invalid_fields)),
        )

    with settings_lock:
        settings.clear()
        settings.update(salvaged)
        snapshot = dict(settings)

    if needs_migration:
        backup_path = _backup_settings_file(path)
        if path.exists() and backup_path is None:
            get_logger().error("Settings migration was not written because backup creation failed")
        else:
            try:
                _atomic_write_settings(path, snapshot)
                get_logger().info(
                    "Settings migrated to schema %s; backup=%s",
                    SETTINGS_SCHEMA_VERSION,
                    backup_path,
                )
            except Exception:
                get_logger().exception("Unable to write migrated settings to %s", path)
    return snapshot

def save_settings() -> bool:
    path = get_config_file_path()
    with settings_lock:
        snapshot = dict(settings)
    try:
        snapshot["settings_schema_version"] = SETTINGS_SCHEMA_VERSION
        _atomic_write_settings(path, snapshot)
        with settings_lock:
            settings["settings_schema_version"] = SETTINGS_SCHEMA_VERSION
        return True
    except Exception:
        get_logger().exception("Unable to save settings to %s", path)
        return False


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


def app_icon_path() -> Path:
    """Return the native application icon while preserving Windows packaging."""
    return asset_path("icon.ico" if sys.platform.startswith("win") else "icon.png")


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
