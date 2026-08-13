from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from utils import update_helper, update_manager


pytestmark = pytest.mark.core_headless


def _release(artifact: bytes) -> update_manager.VerifiedRelease:
    private_key = Ed25519PrivateKey.generate()
    public_key = base64.b64encode(
        private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    ).decode("ascii")
    asset = {
        "platform": "windows",
        "architecture": "x86_64",
        "variant": "standard",
        "name": "CtrlSpeak-windows-x86_64.exe",
        "url": (
            "https://github.com/Wouter-Bekker-AI/CtrlSpeak/releases/download/"
            "v0.5.1/CtrlSpeak-windows-x86_64.exe"
        ),
        "size": len(artifact),
        "sha256": hashlib.sha256(artifact).hexdigest(),
    }
    payload = {
        "schema_version": 1,
        "product": "ctrlspeak",
        "channel": "stable",
        "version": "0.5.1",
        "tag": "v0.5.1",
        "published_at": "2026-08-14T12:00:00Z",
        "release_url": "https://github.com/Wouter-Bekker-AI/CtrlSpeak/releases/tag/v0.5.1",
        "minimum_updater_version": "0.5.0",
        "assets": [asset],
    }
    manifest_bytes = update_manager.canonical_manifest_bytes(payload)
    signature_bytes = base64.b64encode(private_key.sign(manifest_bytes))
    manifest, selected, version, minimum = update_manager.verify_signed_manifest(
        manifest_bytes,
        signature_bytes,
        platform_name="windows",
        architecture="x86_64",
        public_key_base64=public_key,
    )
    return update_manager.VerifiedRelease(
        version=version,
        tag=str(manifest["tag"]),
        title="Test release",
        notes="Safe notes",
        published_at=str(manifest["published_at"]),
        release_url=str(manifest["release_url"]),
        minimum_updater_version=minimum,
        asset=selected,
        manifest_bytes=manifest_bytes,
        signature_bytes=signature_bytes,
    )


def _ready_transaction(tmp_path: Path, installed: Path, artifact: bytes):
    release = _release(artifact)
    transaction = update_manager.create_transaction(
        release,
        installation_path=installed,
        root=tmp_path,
    )
    transaction.candidate_path.write_bytes(artifact)
    update_manager.update_transaction_journal(transaction, state="ready_to_install")
    return release, transaction


def test_prepare_handoff_copies_verified_candidate_and_old_helper(tmp_path, monkeypatch):
    installed = tmp_path / "install" / "CtrlSpeak.exe"
    installed.parent.mkdir()
    installed.write_bytes(b"old executable")
    artifact = b"new executable" * 20
    release, transaction = _ready_transaction(tmp_path, installed, artifact)
    launches: list[tuple[list[str], dict[str, object]]] = []

    def fake_launch(args, **kwargs):
        launches.append((args, kwargs))
        return SimpleNamespace(pid=4321)

    monkeypatch.setattr(update_helper.sys, "frozen", True, raising=False)
    monkeypatch.setattr(update_helper.sys, "platform", "win32")

    helper_pid = update_helper.prepare_update_handoff(
        transaction,
        executable=installed,
        original_pid=1234,
        launch_process=fake_launch,
    )

    assert helper_pid == 4321
    assert installed.with_name("CtrlSpeak.exe.new").read_bytes() == artifact
    journal = update_manager.read_transaction_journal(transaction)
    helper_path = Path(str(journal["helper_path"]))
    assert helper_path.read_bytes() == b"old executable"
    assert journal["state"] == "awaiting_original_exit"
    assert journal["original_pid"] == 1234
    assert launches[0][0][1:3] == ["--apply-update", str(transaction.journal_path)]
    assert release.asset.sha256 == hashlib.sha256(artifact).hexdigest()


class _RunningProcess:
    def __init__(self, pid: int):
        self.pid = pid
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 1

    def kill(self):
        self.returncode = 1

    def wait(self, timeout=None):
        del timeout
        return self.returncode


class _ExitedProcess(_RunningProcess):
    def __init__(self, pid: int):
        super().__init__(pid)
        self.returncode = 1


def test_apply_update_replaces_and_accepts_matching_health(tmp_path, monkeypatch):
    installed = tmp_path / "install" / "CtrlSpeak.exe"
    installed.parent.mkdir()
    original = b"old executable" * 20
    artifact = b"healthy executable" * 20
    installed.write_bytes(original)
    _, transaction = _ready_transaction(tmp_path, installed, artifact)
    monkeypatch.setattr(update_helper.sys, "frozen", True, raising=False)
    monkeypatch.setattr(update_helper.sys, "platform", "win32")
    update_helper.prepare_update_handoff(
        transaction,
        executable=installed,
        original_pid=1234,
        launch_process=lambda *_args, **_kwargs: SimpleNamespace(pid=4321),
    )
    monkeypatch.setattr(update_helper, "get_updates_dir", lambda: tmp_path)
    monkeypatch.setattr(
        update_helper,
        "resolve_transaction",
        lambda transaction_id: transaction
        if transaction_id == transaction.transaction_id
        else (_ for _ in ()).throw(AssertionError("wrong transaction")),
    )

    def launch_new(args, **_kwargs):
        assert args[1:] == ["--post-update", transaction.transaction_id]
        transaction.health_response_path.write_text(
            json.dumps(
                {
                    "transaction_id": transaction.transaction_id,
                    "version": "0.5.1",
                    "executable": str(installed.resolve()),
                    "pid": 9876,
                }
            ),
            encoding="utf-8",
        )
        return _RunningProcess(9876)

    result = update_helper.apply_update_transaction(
        str(transaction.journal_path),
        original_exit_timeout=0,
        health_timeout=0.2,
        health_grace=0,
        process_exists=lambda _pid: False,
        launch_process=launch_new,
    )

    assert result == 0
    assert installed.read_bytes() == artifact
    journal = update_manager.read_transaction_journal(transaction)
    assert journal["state"] == "installed"
    assert Path(str(journal["backup_path"])).read_bytes() == original


def test_failed_health_rolls_back_and_relaunches_previous_version(tmp_path, monkeypatch):
    installed = tmp_path / "install" / "CtrlSpeak.exe"
    installed.parent.mkdir()
    original = b"old executable" * 20
    artifact = b"broken executable" * 20
    installed.write_bytes(original)
    _, transaction = _ready_transaction(tmp_path, installed, artifact)
    monkeypatch.setattr(update_helper.sys, "frozen", True, raising=False)
    monkeypatch.setattr(update_helper.sys, "platform", "win32")
    update_helper.prepare_update_handoff(
        transaction,
        executable=installed,
        original_pid=1234,
        launch_process=lambda *_args, **_kwargs: SimpleNamespace(pid=4321),
    )
    monkeypatch.setattr(update_helper, "get_updates_dir", lambda: tmp_path)
    monkeypatch.setattr(
        update_helper,
        "resolve_transaction",
        lambda transaction_id: transaction
        if transaction_id == transaction.transaction_id
        else (_ for _ in ()).throw(AssertionError("wrong transaction")),
    )
    launches: list[list[str]] = []

    def launch(args, **_kwargs):
        launches.append(args)
        if "--post-update" in args:
            return _ExitedProcess(5555)
        return _RunningProcess(6666)

    result = update_helper.apply_update_transaction(
        str(transaction.journal_path),
        original_exit_timeout=0,
        health_timeout=0.01,
        health_grace=0,
        process_exists=lambda _pid: False,
        launch_process=launch,
    )

    assert result == 1
    assert installed.read_bytes() == original
    assert launches[-1][1:] == ["--rollback-notice", transaction.transaction_id]
    assert update_manager.read_transaction_journal(transaction)["state"] == "rolled_back"


def test_post_update_health_rejects_version_mismatch(tmp_path, monkeypatch):
    installed = tmp_path / "CtrlSpeak.exe"
    installed.write_bytes(b"new")
    _, transaction = _ready_transaction(tmp_path, installed, b"new")
    monkeypatch.setattr(update_helper, "resolve_transaction", lambda _transaction_id: transaction)
    monkeypatch.setattr(update_helper.sys, "executable", str(installed))

    with pytest.raises(update_manager.UpdateError, match="health_version_mismatch"):
        update_helper.write_post_update_health(transaction.transaction_id, "9.9.9")
