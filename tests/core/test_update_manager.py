from __future__ import annotations

import base64
import hashlib
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
import requests
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from utils import update_manager


pytestmark = pytest.mark.core_headless


def _key_pair() -> tuple[Ed25519PrivateKey, str]:
    private_key = Ed25519PrivateKey.generate()
    public_raw = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return private_key, base64.b64encode(public_raw).decode("ascii")


def _manifest_payload(
    artifact: bytes,
    *,
    version: str = "0.5.1",
    product: str = update_manager.PRODUCT_ID,
    duplicate: bool = False,
) -> dict[str, object]:
    tag = f"v{version}"
    asset = {
        "platform": "windows",
        "architecture": "x86_64",
        "variant": "standard",
        "name": "CtrlSpeak-windows-x86_64.exe",
        "url": (
            "https://github.com/Wouter-Bekker-AI/CtrlSpeak/releases/download/"
            f"{tag}/CtrlSpeak-windows-x86_64.exe"
        ),
        "size": len(artifact),
        "sha256": hashlib.sha256(artifact).hexdigest(),
    }
    return {
        "schema_version": 1,
        "product": product,
        "channel": "stable",
        "version": version,
        "tag": tag,
        "published_at": "2026-08-14T12:00:00Z",
        "release_url": f"https://github.com/Wouter-Bekker-AI/CtrlSpeak/releases/tag/{tag}",
        "minimum_updater_version": "0.5.0",
        "assets": [asset, dict(asset)] if duplicate else [asset],
    }


def _signed_release(artifact: bytes) -> tuple[update_manager.VerifiedRelease, str]:
    private_key, public_key = _key_pair()
    manifest_bytes = update_manager.canonical_manifest_bytes(_manifest_payload(artifact))
    signature = base64.b64encode(private_key.sign(manifest_bytes))
    manifest, asset, version, minimum = update_manager.verify_signed_manifest(
        manifest_bytes,
        signature,
        platform_name="win32",
        architecture="AMD64",
        public_key_base64=public_key,
    )
    return (
        update_manager.VerifiedRelease(
            version=version,
            tag=str(manifest["tag"]),
            title="CtrlSpeak test release",
            notes="Test notes",
            published_at=str(manifest["published_at"]),
            release_url=str(manifest["release_url"]),
            minimum_updater_version=minimum,
            asset=asset,
            manifest_bytes=manifest_bytes,
            signature_bytes=signature,
        ),
        public_key,
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0.5.0", (0, 5, 0)),
        ("10.20.30", (10, 20, 30)),
    ],
)
def test_semantic_version_is_strict(value, expected):
    parsed = update_manager.SemanticVersion.parse(value)
    assert (parsed.major, parsed.minor, parsed.patch) == expected


@pytest.mark.parametrize("value", ["v0.5.0", "0.5", "0.5.0-beta", "01.2.3", "latest", None])
def test_semantic_version_rejects_non_release_values(value):
    with pytest.raises(update_manager.UpdateError, match="invalid_version"):
        update_manager.SemanticVersion.parse(value)


def test_signed_manifest_selects_exact_target():
    artifact = b"signed executable bytes"
    release, _ = _signed_release(artifact)

    assert release.version == update_manager.SemanticVersion(0, 5, 1)
    assert release.asset.name == "CtrlSpeak-windows-x86_64.exe"
    assert release.asset.size == len(artifact)


def test_signed_manifest_rejects_tampering_wrong_product_and_duplicates():
    artifact = b"artifact"
    private_key, public_key = _key_pair()

    valid_bytes = update_manager.canonical_manifest_bytes(_manifest_payload(artifact))
    valid_signature = base64.b64encode(private_key.sign(valid_bytes))
    with pytest.raises(update_manager.UpdateError, match="invalid_signature"):
        update_manager.verify_signed_manifest(
            valid_bytes + b" ",
            valid_signature,
            platform_name="windows",
            architecture="x86_64",
            public_key_base64=public_key,
        )

    wrong_product = update_manager.canonical_manifest_bytes(
        _manifest_payload(artifact, product="watcher")
    )
    with pytest.raises(update_manager.UpdateError, match="wrong_product"):
        update_manager.verify_signed_manifest(
            wrong_product,
            base64.b64encode(private_key.sign(wrong_product)),
            platform_name="windows",
            architecture="x86_64",
            public_key_base64=public_key,
        )

    duplicate = update_manager.canonical_manifest_bytes(
        _manifest_payload(artifact, duplicate=True)
    )
    with pytest.raises(update_manager.UpdateError, match="ambiguous_asset"):
        update_manager.verify_signed_manifest(
            duplicate,
            base64.b64encode(private_key.sign(duplicate)),
            platform_name="windows",
            architecture="x86_64",
            public_key_base64=public_key,
        )


def test_signed_manifest_rejects_noncanonical_json():
    artifact = b"artifact"
    private_key, public_key = _key_pair()
    canonical = update_manager.canonical_manifest_bytes(_manifest_payload(artifact))
    noncanonical = canonical.replace(b'"assets":', b'"assets": ')

    with pytest.raises(update_manager.UpdateError, match="noncanonical_manifest"):
        update_manager.verify_signed_manifest(
            noncanonical,
            base64.b64encode(private_key.sign(noncanonical)),
            platform_name="windows",
            architecture="x86_64",
            public_key_base64=public_key,
        )


def test_runtime_classification_distinguishes_source_and_packaged(tmp_path):
    executable = tmp_path / "CtrlSpeak.exe"
    executable.write_bytes(b"exe")

    assert update_manager.classify_runtime(frozen=False, executable=executable) == "source_checkout"
    assert (
        update_manager.classify_runtime(frozen=True, executable=executable)
        == "packaged_user_writable"
    )


class _FakeResponse:
    def __init__(self, status_code: int, chunks: list[bytes], *, headers=None, url=None):
        self.status_code = status_code
        self._chunks = chunks
        self.headers = headers or {}
        self.url = url or "https://github.com/Wouter-Bekker-AI/CtrlSpeak/releases/download/v0.5.1/CtrlSpeak-windows-x86_64.exe"
        self.closed = False

    def iter_content(self, chunk_size: int):
        del chunk_size
        yield from self._chunks

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def close(self):
        self.closed = True


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]):
        self.responses = list(responses)
        self.headers: dict[str, str] = {}
        self.calls: list[dict[str, object]] = []

    def get(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        if not self.responses:
            raise AssertionError("unexpected request")
        return self.responses.pop(0)

    def close(self):
        pass


def test_download_streams_verifies_and_finalizes_atomically(tmp_path):
    artifact = b"A" * 4096 + b"B" * 3072
    release, _ = _signed_release(artifact)
    transaction = update_manager.create_transaction(release, root=tmp_path)
    progress: list[tuple[int, int]] = []
    session = _FakeSession(
        [
            _FakeResponse(
                200,
                [artifact[:4096], artifact[4096:]],
                headers={"Content-Length": str(len(artifact))},
            )
        ]
    )

    candidate = update_manager.download_release(
        release,
        transaction,
        progress=lambda downloaded, total: progress.append((downloaded, total)),
        session=session,
    )

    assert candidate.read_bytes() == artifact
    assert not transaction.partial_path.exists()
    assert progress[-1] == (len(artifact), len(artifact))
    assert update_manager.read_transaction_journal(transaction)["state"] == "ready_to_install"


def test_resume_restarts_cleanly_when_server_ignores_range(tmp_path):
    artifact = b"0123456789" * 100
    release, _ = _signed_release(artifact)
    transaction = update_manager.create_transaction(release, root=tmp_path)
    transaction.partial_path.write_bytes(artifact[:100])
    session = _FakeSession(
        [
            _FakeResponse(200, [artifact]),
            _FakeResponse(200, [artifact], headers={"Content-Length": str(len(artifact))}),
        ]
    )

    candidate = update_manager.download_release(release, transaction, session=session)

    assert candidate.read_bytes() == artifact
    assert session.calls[0]["headers"] == {"Range": "bytes=100-"}
    assert session.calls[1]["headers"] is None


def test_resume_rejects_wrong_content_range(tmp_path):
    artifact = b"0123456789" * 100
    release, _ = _signed_release(artifact)
    transaction = update_manager.create_transaction(release, root=tmp_path)
    transaction.partial_path.write_bytes(artifact[:100])
    session = _FakeSession(
        [
            _FakeResponse(
                206,
                [artifact[100:]],
                headers={"Content-Range": f"bytes 99-{len(artifact)-1}/{len(artifact)}"},
            )
        ]
    )

    with pytest.raises(update_manager.UpdateError, match="invalid_range"):
        update_manager.download_release(release, transaction, session=session)


def test_oversize_body_is_removed(tmp_path):
    artifact = b"safe artifact"
    release, _ = _signed_release(artifact)
    transaction = update_manager.create_transaction(release, root=tmp_path)
    session = _FakeSession([_FakeResponse(200, [artifact + b"tainted"])])

    with pytest.raises(update_manager.UpdateError, match="oversize_download"):
        update_manager.download_release(release, transaction, session=session)

    assert not transaction.partial_path.exists()


def test_transaction_rejects_path_traversal(tmp_path):
    with pytest.raises(update_manager.UpdateError, match="invalid_transaction"):
        update_manager.resolve_transaction("../outside", root=tmp_path)


def test_coordinator_ignores_late_result_from_superseded_generation():
    first_started = threading.Event()
    release_first = threading.Event()
    call_count = 0

    def discoverer(_current_version):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            first_started.set()
            assert release_first.wait(2)
            return update_manager.UpdateCheckResult(
                "available", "stale update", update_manager.utc_now_iso()
            )
        return update_manager.UpdateCheckResult(
            "up_to_date", "current result", update_manager.utc_now_iso()
        )

    coordinator = update_manager.UpdateCoordinator("0.5.0", discoverer=discoverer)
    events: list[update_manager.UpdateEvent] = []
    coordinator.add_listener(events.append)

    coordinator.check_async()
    assert first_started.wait(2)
    latest_generation = coordinator.check_async()
    deadline = time.monotonic() + 2
    while coordinator.state != "up_to_date" and time.monotonic() < deadline:
        time.sleep(0.01)
    release_first.set()
    time.sleep(0.05)

    assert coordinator.generation == latest_generation
    assert coordinator.state == "up_to_date"
    assert not any(event.message == "stale update" for event in events)
