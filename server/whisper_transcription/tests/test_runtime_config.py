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
        "CTRLSPEAK_SERVICE_ROLE",
        "CTRLSPEAK_CLIENTS_JSON",
        "CTRLSPEAK_WORKER_URL",
        "CTRLSPEAK_WORKER_TOKEN",
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


def test_worker_non_loopback_binding_requires_dedicated_worker_token() -> None:
    blocked = _run_shell(
        "whisper_validate_runtime_config",
        CTRLSPEAK_SERVICE_ROLE="worker",
        WHISPER_BIND_HOST="10.83.233.2",
        WHISPER_BEARER_TOKEN="client-token-is-not-a-worker-token",
    )
    allowed = _run_shell(
        "whisper_validate_runtime_config",
        CTRLSPEAK_SERVICE_ROLE="worker",
        WHISPER_BIND_HOST="10.83.233.2",
        CTRLSPEAK_WORKER_TOKEN="dedicated-worker-token",
    )

    assert blocked.returncode != 0
    assert "CTRLSPEAK_WORKER_TOKEN" in blocked.stderr
    assert allowed.returncode == 0


def test_gateway_requires_paired_worker_url_and_token() -> None:
    missing_token = _run_shell(
        "whisper_validate_runtime_config",
        CTRLSPEAK_SERVICE_ROLE="gateway",
        WHISPER_BIND_HOST="0.0.0.0",
        WHISPER_BEARER_TOKEN="client-token",
        CTRLSPEAK_WORKER_URL="http://10.83.233.2:8765",
    )
    configured = _run_shell(
        "whisper_validate_runtime_config",
        CTRLSPEAK_SERVICE_ROLE="gateway",
        WHISPER_BIND_HOST="0.0.0.0",
        WHISPER_BEARER_TOKEN="client-token",
        CTRLSPEAK_WORKER_URL="http://10.83.233.2:8765",
        CTRLSPEAK_WORKER_TOKEN="worker-token",
    )

    assert missing_token.returncode != 0
    assert "CTRLSPEAK_WORKER_TOKEN" in missing_token.stderr
    assert configured.returncode == 0


def test_runtime_summary_reports_secret_presence_without_values() -> None:
    result = _run_shell(
        "whisper_print_runtime_config",
        CTRLSPEAK_SERVICE_ROLE="gateway",
        CTRLSPEAK_CLIENTS_JSON='{"alice":{"token":"never-print-client-secret"}}',
        CTRLSPEAK_WORKER_URL="http://10.83.233.2:8765",
        CTRLSPEAK_WORKER_TOKEN="never-print-worker-secret",
    )

    assert result.returncode == 0
    assert "role=gateway" in result.stdout
    assert "client_identities=configured" in result.stdout
    assert "worker_token=configured" in result.stdout
    assert "worker_url=configured" in result.stdout
    assert "never-print" not in result.stdout
