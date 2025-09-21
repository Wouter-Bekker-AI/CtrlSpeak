"""Quick developer smoke test for CtrlSpeak subsystems.

Run from the project root with::

    python -m utils.sanity_check
"""
from __future__ import annotations
import json
import time
import urllib.request
from utils.system import (
    load_settings, save_settings, settings, settings_lock,
    get_config_dir, get_config_file_path, get_temp_dir,
    create_recording_file_path, cleanup_recording_file,
    start_server, shutdown_server, start_discovery_listener,
    manual_discovery_refresh, get_best_server,
)

def assert_true(cond, msg):
    if not cond:
        raise AssertionError(msg)

def main():
    # settings round-trip
    load_settings()
    with settings_lock:
        settings["_sanity_probe"] = "ok"
    save_settings()
    p = get_config_file_path()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert_true(data.get("_sanity_probe") == "ok", "Settings did not persist")

    # dirs + temp recording file paths
    cfg = get_config_dir()
    tmpdir = get_temp_dir()
    assert_true(cfg.exists() and tmpdir.exists(), "Config/temp dir missing")
    rec = create_recording_file_path()
    rec.write_bytes(b"WAV")  # stub
    assert_true(rec.exists(), "Recording temp file not created")
    cleanup_recording_file(rec)
    assert_true(not rec.exists(), "Recording temp file not removed")

    # discovery helpers (no server needed; just ensure they run)
    start_discovery_listener()
    _ = manual_discovery_refresh(wait_time=0.6)
    _ = get_best_server()

    # start server and ping it
    start_server()
    time.sleep(0.3)
    # The default port is 65432 unless overridden in settings
    port = int(settings.get("server_port", 65432))
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/ping", timeout=2) as resp:
        body = resp.read().decode("utf-8")
        assert_true('"status": "ok"' in body, "/ping did not return ok")

    # cleanup
    shutdown_server()
    print("✅ sanity OK")

if __name__ == "__main__":
    main()
