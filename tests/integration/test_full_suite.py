from __future__ import annotations

import os
import time
import urllib.request

import pytest

pytestmark = pytest.mark.full_gui

if os.environ.get("CTRLSPEAK_RUN_FULL_TESTS") != "1":
    pytest.skip("Full GUI tests require CTRLSPEAK_RUN_FULL_TESTS=1", allow_module_level=True)

pytest.importorskip("ctranslate2")
pytest.importorskip("faster_whisper")

from utils import config_paths
from utils import system


def test_full_stack_transcription_and_server_cycle():
    # Reload models after the config fixtures have isolated the filesystem.
    import importlib
    from utils import models

    importlib.reload(models)

    config_paths.load_settings()
    with config_paths.settings_lock:
        config_paths.settings["mode"] = "client_server"
    config_paths.save_settings()

    model = models.initialize_transcriber(force=True)
    assert model is not None

    audio_path = config_paths.asset_path("test.wav")
    assert audio_path.exists(), "assets/test.wav must exist for the integration test"

    transcript = models.transcribe_local(str(audio_path), play_feedback=False)
    assert transcript is not None and transcript.strip(), "Transcription returned empty output"

    system.start_server()
    time.sleep(0.5)

    port = int(system.settings.get("server_port", 65432))
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/ping", timeout=5) as response:
        body = response.read().decode("utf-8")
        assert '"status": "ok"' in body

    system.start_discovery_listener()
    discovered = system.manual_discovery_refresh(wait_time=0.6)
    assert discovered is not None

    best = system.get_best_server()
    assert best is not None
    assert best.host
