from __future__ import annotations

from pathlib import Path

import certifi
import pytest

from utils import system, update_helper


pytestmark = pytest.mark.core_headless


def _ca_file(root: Path) -> Path:
    path = root / "certifi" / "cacert.pem"
    path.parent.mkdir(parents=True)
    path.write_text("test CA", encoding="ascii")
    return path


def test_bootstrap_replaces_an_existing_ca_path_from_the_previous_bundle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    previous_ca = _ca_file(tmp_path / "_MEIprevious")
    current_ca = _ca_file(tmp_path / "_MEIcurrent")
    monkeypatch.setenv("SSL_CERT_FILE", str(previous_ca))
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(previous_ca))
    monkeypatch.setattr(certifi, "where", lambda: str(current_ca))
    monkeypatch.setattr(system, "get_config_dir", lambda: tmp_path / "config")

    system._bootstrap_runtime_environment()

    assert system.os.environ["SSL_CERT_FILE"] == str(current_ca)
    assert system.os.environ["REQUESTS_CA_BUNDLE"] == str(current_ca)


def test_bootstrap_preserves_an_explicit_existing_non_bundle_ca(
    tmp_path: Path,
    monkeypatch,
) -> None:
    custom_ca = _ca_file(tmp_path / "company-ca")
    bundled_ca = _ca_file(tmp_path / "_MEIcurrent")
    monkeypatch.setenv("SSL_CERT_FILE", str(custom_ca))
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(custom_ca))
    monkeypatch.setattr(certifi, "where", lambda: str(bundled_ca))
    monkeypatch.setattr(system, "get_config_dir", lambda: tmp_path / "config")

    system._bootstrap_runtime_environment()

    assert system.os.environ["SSL_CERT_FILE"] == str(custom_ca)
    assert system.os.environ["REQUESTS_CA_BUNDLE"] == str(custom_ca)


def test_detached_updater_launch_drops_only_temporary_bundle_ca_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    temporary_ca = _ca_file(tmp_path / "_MEIold")
    custom_ca = _ca_file(tmp_path / "company-ca")
    monkeypatch.setenv("SSL_CERT_FILE", str(temporary_ca))
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(custom_ca))

    kwargs = update_helper._detached_process_kwargs()
    environment = kwargs["env"]

    assert isinstance(environment, dict)
    assert "SSL_CERT_FILE" not in environment
    assert environment["REQUESTS_CA_BUNDLE"] == str(custom_ca)
