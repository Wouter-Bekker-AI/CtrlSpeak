"""Explicit rollback coverage; never launches an executable or uses the network."""
from __future__ import annotations

import base64
import hashlib
import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from utils import update_helper as helper, update_manager as updater

pytestmark = pytest.mark.core_headless


class Response:
    status_code = 200
    headers = {}

    def __init__(self, payload, url):
        self.payload, self.url = payload, url

    def json(self):
        return self.payload

    def iter_content(self, chunk_size):
        yield self.payload

    def close(self):
        pass

    def raise_for_status(self):
        pass


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers, self.calls = {}, []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        assert self.responses, "Unexpected extra network request"
        return self.responses.pop(0)


@pytest.fixture
def signed(monkeypatch):
    artifact = b"harmless fake 0.7.7 executable bytes"
    private = Ed25519PrivateKey.generate()
    public = base64.b64encode(private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()
    tag = "v0.7.7"
    root = updater.GITHUB_RELEASE_ROOT
    payload = {
        "schema_version": 1, "product": "ctrlspeak", "channel": "stable",
        "version": "0.7.7", "tag": tag, "published_at": "2026-10-22T00:00:00Z",
        "release_url": f"{root}/releases/tag/{tag}", "minimum_updater_version": "0.5.0",
        "assets": [{"platform": "windows", "architecture": "x86_64", "variant": "standard",
                    "name": "CtrlSpeak-windows-x86_64.exe",
                    "url": f"{root}/releases/download/{tag}/CtrlSpeak-windows-x86_64.exe",
                    "size": len(artifact), "sha256": hashlib.sha256(artifact).hexdigest()}],
    }
    manifest = updater.canonical_manifest_bytes(payload)
    signature = base64.b64encode(private.sign(manifest))
    verify = updater.verify_signed_manifest

    def verify_test_key(*args, **kwargs):
        return verify(*args, **kwargs, public_key_base64=public)

    monkeypatch.setattr(updater, "verify_signed_manifest", verify_test_key)
    monkeypatch.setattr(helper, "verify_signed_manifest", verify_test_key)
    metadata = {
        "draft": False, "prerelease": False, "tag_name": tag, "name": "CtrlSpeak 0.7.7",
        "html_url": payload["release_url"], "body": "Release notes",
        "assets": [{"name": name, "browser_download_url": f"{root}/releases/download/{tag}/{name}"}
                   for name in (updater.MANIFEST_ASSET_NAME, updater.SIGNATURE_ASSET_NAME)],
    }

    def session(metadata_override=None, manifest_override=None, signature_override=None):
        return Session([
            Response(metadata if metadata_override is None else metadata_override, "https://api.github.com"),
            Response(manifest if manifest_override is None else manifest_override, metadata["assets"][0]["browser_download_url"]),
            Response(signature if signature_override is None else signature_override, metadata["assets"][1]["browser_download_url"]),
        ])

    return SimpleNamespace(artifact=artifact, metadata=metadata, manifest=manifest,
                           signature=signature, session=session)


def discover(signed):
    return updater.discover_rollback_077("0.8.0", session=signed.session(),
                                         platform_name="windows", architecture="AMD64").release


def test_rollback_checks_exact_tag_and_still_verifies_signature(signed):
    session = signed.session()
    result = updater.discover_rollback_077("0.8.0", session=session,
                                         platform_name="windows", architecture="AMD64")
    assert result.status == "available"
    assert result.release.operation == "rollback_077"
    assert result.release.tag == "v0.7.7"
    assert session.calls[0][0].endswith("/releases/tags/v0.7.7")
    assert session.calls[0][1]["allow_redirects"] is False
    with pytest.raises(updater.UpdateError, match="invalid_signature"):
        updater.discover_rollback_077("0.8.0", session=signed.session(manifest_override=signed.manifest + b" "),
                                     platform_name="windows", architecture="AMD64")


@pytest.mark.parametrize("field,value,category", [
    ("tag_name", "v0.7.6", "wrong_rollback_target"),
    ("draft", True, "unstable_release"), ("prerelease", True, "unstable_release"),
])
def test_rollback_rejects_wrong_or_unpublished_release(signed, field, value, category):
    metadata = dict(signed.metadata, **{field: value})
    with pytest.raises(updater.UpdateError, match=category):
        updater.discover_rollback_077("0.8.0", session=signed.session(metadata_override=metadata),
                                     platform_name="windows", architecture="AMD64")


@pytest.mark.parametrize("version", ["0.7.7", "0.7.6"])
def test_rollback_cannot_be_selected_on_same_or_older_version(version):
    with pytest.raises(updater.UpdateError, match="rollback_not_applicable"):
        updater.discover_rollback_077(version)
    with pytest.raises(updater.UpdateError, match="rollback_not_applicable"):
        updater.UpdateCoordinator(version).rollback_to_077()


def test_normal_check_still_refuses_downgrade_and_download(signed):
    session = signed.session()
    result = updater.discover_update("0.8.0", session=session, platform_name="windows", architecture="AMD64")
    assert session.calls[0][0] == updater.GITHUB_API_URL
    assert result.status == "newer_than_release"
    assert result.release.operation == "update"
    coordinator = updater.UpdateCoordinator("0.8.0")
    coordinator._release = result.release
    with pytest.raises(updater.UpdateError, match="downgrade_not_allowed"):
        coordinator.download_async()


def test_rollback_intent_is_persisted_and_checked(tmp_path, signed):
    release = discover(signed)
    with pytest.raises(updater.UpdateError, match="wrong_rollback_target"):
        updater.create_transaction(release, root=tmp_path)
    transaction = updater.create_transaction(release, root=tmp_path, source_version="0.8.0")
    journal = updater.read_transaction_journal(transaction)
    assert journal["operation"] == "rollback_077"
    assert journal["source_version"] == "0.8.0"
    assert journal["explicit_rollback_target"] == "0.7.7"
    helper._validate_transaction_identity(journal)
    helper._verify_rollback_manifest(transaction, journal)
    for mutation in ({"version": "0.7.6"}, {"explicit_rollback_target": None}, {"source_version": "0.7.7"}):
        with pytest.raises(updater.UpdateError, match="wrong_rollback_target"):
            helper._validate_transaction_identity(dict(journal, **mutation))
    with pytest.raises(updater.UpdateError, match="rollback_manifest_mismatch"):
        helper._verify_rollback_manifest(transaction, dict(journal, asset_sha256="0" * 64))
    transaction.signature_path.write_bytes(base64.b64encode(b"x" * 64))
    with pytest.raises(updater.UpdateError, match="invalid_signature"):
        helper._verify_rollback_manifest(transaction, journal)


def test_normal_transaction_cannot_be_disguised_downgrade(tmp_path, signed):
    normal = replace(discover(signed), operation="update")
    with pytest.raises(updater.UpdateError, match="downgrade_not_allowed"):
        updater.create_transaction(normal, root=tmp_path, source_version="0.8.0")
    with pytest.raises(updater.UpdateError, match="downgrade_not_allowed"):
        helper._validate_transaction_identity({"product": "ctrlspeak", "channel": "stable",
                                              "operation": "update", "version": "0.7.7", "source_version": "0.8.0"})


def test_selection_change_clears_stale_rollback_candidate(signed, monkeypatch):
    # Inline worker makes state transitions deterministic without another process.
    class InlineThread:
        def __init__(self, target, **kwargs):
            self.target = target
        def start(self):
            self.target()

    monkeypatch.setattr(updater.threading, "Thread", InlineThread)
    release = discover(signed)
    coordinator = updater.UpdateCoordinator(
        "0.8.0", rollback_discoverer=lambda _: updater.UpdateCheckResult("available", "rollback", "now", release),
        discoverer=lambda _: updater.UpdateCheckResult("up_to_date", "current", "now"),
    )
    coordinator.rollback_to_077()
    assert coordinator.release.operation == "rollback_077"
    coordinator._transaction = object()
    coordinator.check_async()
    assert coordinator.state == "up_to_date"
    assert coordinator.release is None and coordinator.transaction is None
    with pytest.raises(updater.UpdateError, match="no_update"):
        coordinator.download_async()


def test_rollback_handoff_preserves_installed_binary_and_credentials(tmp_path, signed, monkeypatch):
    installed = tmp_path / "install" / "CtrlSpeak.exe"
    installed.parent.mkdir()
    installed.write_bytes(b"original 0.8.0 binary")
    settings = installed.parent / "settings.json"
    settings.write_text('{"saved_key_reference":"unchanged"}')
    release = discover(signed)
    transaction = updater.create_transaction(release, installation_path=installed,
                                             root=tmp_path, source_version="0.8.0")
    transaction.candidate_path.write_bytes(signed.artifact)
    updater.update_transaction_journal(transaction, state="ready_to_install")
    monkeypatch.setattr(helper.sys, "frozen", True, raising=False)
    monkeypatch.setattr(helper.sys, "platform", "win32")
    helper.prepare_update_handoff(transaction, executable=installed, original_pid=1234,
                                  launch_process=lambda *args, **kwargs: SimpleNamespace(pid=4321))
    assert installed.read_bytes() == b"original 0.8.0 binary"
    assert installed.with_name("CtrlSpeak.exe.new").read_bytes() == signed.artifact
    journal = updater.read_transaction_journal(transaction)
    assert journal["operation"] == "rollback_077"
    assert settings.read_text() == '{"saved_key_reference":"unchanged"}'
    # Old 0.7.7 consumes schema/version/path, so added intent fields do not
    # require editing or re-signing that immutable historical binary.
    request = json.loads(transaction.health_request_path.read_text())
    assert request["schema_version"] == 1 and request["expected_version"] == "0.7.7"
