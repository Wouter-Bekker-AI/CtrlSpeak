from __future__ import annotations

import base64
import os
from argparse import Namespace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption

from scripts import release
from utils.version import APP_VERSION


pytestmark = pytest.mark.core_headless


def test_release_version_gate_matches_current_release():
    assert APP_VERSION == "0.8.3"
    assert release.read_app_version() == APP_VERSION
    assert release.read_server_versions() == (APP_VERSION, APP_VERSION)
    assert release.check_version(f"v{APP_VERSION}") == APP_VERSION
    with pytest.raises(SystemExit, match="tag/version mismatch"):
        release.check_version("v0.5.0")


def test_midnight_signal_visible_identity_uses_version_source():
    source = (release.ROOT / "utils" / "midnight_signal_ui.py").read_text("utf-8")

    assert "0.7.0  ·  MIDNIGHT SIGNAL" not in source
    assert "APP_VERSION" in source


def test_generate_and_verify_complete_release_set(tmp_path, monkeypatch):
    windows = tmp_path / "CtrlSpeak-windows-x86_64.exe"
    linux = tmp_path / "CtrlSpeak-linux-x86_64"
    windows.write_bytes(b"windows artifact")
    linux.write_bytes(b"linux artifact")
    if os.name != "nt":
        linux.chmod(0o755)
    private_key = Ed25519PrivateKey.generate()
    raw = private_key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    monkeypatch.setenv("CTRLSPEAK_UPDATE_SIGNING_KEY", base64.b64encode(raw).decode("ascii"))

    # The release verifier intentionally uses the production public key. For
    # this fixture, pass through a small wrapper that supplies the test key.
    original_verify = release.verify_signed_manifest
    public_raw = private_key.public_key().public_bytes_raw()

    def verify_with_test_key(*args, **kwargs):
        kwargs["public_key_base64"] = base64.b64encode(public_raw).decode("ascii")
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(release, "verify_signed_manifest", verify_with_test_key)
    release.generate(
        Namespace(
            tag=f"v{APP_VERSION}",
            windows=str(windows),
            linux=str(linux),
            output_dir=str(tmp_path),
            published_at="2026-08-18T12:00:00Z",
        )
    )
    release.verify(Namespace(directory=str(tmp_path), tag=f"v{APP_VERSION}"))

    assert (tmp_path / "update-manifest.json").is_file()
    assert (tmp_path / "update-manifest.sig").is_file()
    assert (tmp_path / "SHA256SUMS").is_file()
