from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_CONFIG = ROOT / "scripts" / "runtime-config"


def _run_shell(command: str, **environment: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    for key in (
        "WHISPER_BIND_HOST",
        "WHISPER_HOST",
        "WHISPER_BEARER_TOKEN",
        "WHISPER_MODEL_NAME",
        "WHISPER_DEVICE",
        "WHISPER_COMPUTE_TYPE",
        "WHISPER_CPU_THREADS",
        "WHISPER_NUM_WORKERS",
    ):
        env.pop(key, None)
    env.update(environment)
    return subprocess.run(
        ["bash", "-c", f"source {RUNTIME_CONFIG!s}; {command}"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_runtime_binding_defaults_to_loopback() -> None:
    result = _run_shell("whisper_bind_host")
    assert result.returncode == 0
    assert result.stdout.strip() == "127.0.0.1"


def test_non_loopback_binding_requires_bearer_token() -> None:
    blocked = _run_shell("whisper_validate_runtime_config", WHISPER_BIND_HOST="0.0.0.0")
    allowed = _run_shell(
        "whisper_validate_runtime_config",
        WHISPER_BIND_HOST="0.0.0.0",
        WHISPER_BEARER_TOKEN="configured-outside-source",
    )
    assert blocked.returncode != 0
    assert "WHISPER_BEARER_TOKEN" in blocked.stderr
    assert allowed.returncode == 0


def test_hostname_prefixed_with_127_is_not_treated_as_loopback() -> None:
    result = _run_shell(
        "whisper_validate_runtime_config",
        WHISPER_BIND_HOST="127.example.test",
    )

    assert result.returncode != 0
    assert "WHISPER_BEARER_TOKEN" in result.stderr


def test_runtime_summary_reports_explicit_cpu_configuration() -> None:
    result = _run_shell(
        "whisper_print_runtime_config",
        WHISPER_DEVICE="cpu",
        WHISPER_COMPUTE_TYPE="int8",
        WHISPER_CPU_THREADS="3",
        WHISPER_NUM_WORKERS="2",
    )

    assert result.returncode == 0
    assert "device=cpu" in result.stdout
    assert "compute_type=int8" in result.stdout
    assert "cpu_threads=3" in result.stdout
    assert "model_workers=2" in result.stdout
