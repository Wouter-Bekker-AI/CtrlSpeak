# -*- coding: utf-8 -*-
"""Signed GitHub Release discovery and verified update staging for CtrlSpeak."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional
from urllib.parse import quote, urlsplit

import requests
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from utils.config_paths import get_config_dir, get_logger


logger = get_logger(__name__)

GITHUB_OWNER = "Wouter-Bekker-AI"
GITHUB_REPOSITORY = "CtrlSpeak"
PRODUCT_ID = "ctrlspeak"
UPDATE_CHANNEL = "stable"
UPDATE_VARIANT = "standard"
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_ASSET_NAME = "update-manifest.json"
SIGNATURE_ASSET_NAME = "update-manifest.sig"
CHECKSUMS_ASSET_NAME = "SHA256SUMS"
GITHUB_API_URL = (
    f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPOSITORY}/releases/latest"
)
GITHUB_RELEASE_ROOT = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPOSITORY}"
ED25519_PUBLIC_KEY_BASE64 = "2pgW2LL2C+2DdM0L/fwgmjYsEiiFG2f6eIL2bbKLEvI="

CONNECT_TIMEOUT_SECONDS = 8
READ_TIMEOUT_SECONDS = 30
MANIFEST_MAX_BYTES = 512 * 1024
SIGNATURE_MAX_BYTES = 8 * 1024
ARTIFACT_MAX_BYTES = 1024 * 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
PROGRESS_INTERVAL_SECONDS = 0.1
APPROVED_DOWNLOAD_HOSTS = {
    "github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
}

_SEMVER_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CONTENT_RANGE_PATTERN = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$", re.IGNORECASE)
_TRANSACTION_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class UpdateError(RuntimeError):
    """Finite, user-displayable updater failure without secret-bearing details."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.user_message = message
        super().__init__(f"{code}: {message}")


class UpdateCancelled(UpdateError):
    def __init__(self):
        super().__init__("cancelled", "The update download was cancelled.")


@dataclass(frozen=True, order=True)
class SemanticVersion:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, value: object) -> "SemanticVersion":
        if not isinstance(value, str):
            raise UpdateError("invalid_version", "The release version is not a string.")
        match = _SEMVER_PATTERN.fullmatch(value.strip())
        if match is None:
            raise UpdateError(
                "invalid_version",
                f"The release version {value!r} is not strict MAJOR.MINOR.PATCH.",
            )
        return cls(*(int(part) for part in match.groups()))

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass(frozen=True)
class UpdateAsset:
    platform: str
    architecture: str
    variant: str
    name: str
    url: str
    size: int
    sha256: str


@dataclass(frozen=True)
class VerifiedRelease:
    version: SemanticVersion
    tag: str
    title: str
    notes: str
    published_at: str
    release_url: str
    minimum_updater_version: SemanticVersion
    asset: UpdateAsset
    manifest_bytes: bytes
    signature_bytes: bytes

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(self.manifest_bytes).hexdigest()


@dataclass(frozen=True)
class UpdateCheckResult:
    status: str
    message: str
    checked_at: str
    release: Optional[VerifiedRelease] = None


@dataclass(frozen=True)
class UpdateTransaction:
    transaction_id: str
    directory: Path
    journal_path: Path
    manifest_path: Path
    signature_path: Path
    partial_path: Path
    candidate_path: Path
    release_metadata_path: Path
    health_request_path: Path
    health_response_path: Path
    updater_log_path: Path


@dataclass(frozen=True)
class UpdateEvent:
    generation: int
    state: str
    message: str
    downloaded: int = 0
    total: int = 0
    release: Optional[VerifiedRelease] = None
    transaction: Optional[UpdateTransaction] = None
    error_code: Optional[str] = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_manifest_bytes(payload: Mapping[str, object]) -> bytes:
    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise UpdateError("invalid_manifest", "The update manifest cannot be canonicalized.") from exc
    return (text + "\n").encode("utf-8")


def normalize_platform(value: Optional[str] = None) -> str:
    target = (value or sys.platform).lower()
    if target.startswith("win"):
        return "windows"
    if target.startswith("linux"):
        return "linux"
    raise UpdateError("unsupported_platform", f"CtrlSpeak updates do not support {target!r}.")


def normalize_architecture(value: Optional[str] = None) -> str:
    target = (value or platform.machine()).lower().replace("-", "_")
    if target in {"amd64", "x86_64", "x64"}:
        return "x86_64"
    raise UpdateError(
        "unsupported_architecture",
        f"CtrlSpeak updates do not support architecture {target!r}.",
    )


def expected_asset_name(platform_name: str) -> str:
    if platform_name == "windows":
        return "CtrlSpeak-windows-x86_64.exe"
    if platform_name == "linux":
        return "CtrlSpeak-linux-x86_64"
    raise UpdateError("unsupported_platform", f"Unsupported update platform {platform_name!r}.")


def classify_runtime(
    *,
    frozen: Optional[bool] = None,
    executable: Optional[Path] = None,
) -> str:
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    if not is_frozen:
        return "source_checkout"
    target = Path(executable or sys.executable).resolve()
    parent = target.parent
    if target.is_file() and os.access(target, os.R_OK) and os.access(parent, os.W_OK):
        return "packaged_user_writable"
    return "packaged_manual_install_required"


def sanitize_release_notes(value: object, *, limit: int = 4000) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = "".join(
        character
        for character in value
        if character in "\n\t" or ord(character) >= 32
    )
    cleaned = re.sub(r"<[^>]{0,500}>", "", cleaned)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(cleaned) > limit:
        cleaned = cleaned[: max(0, limit - 1)].rstrip() + "…"
    return cleaned


def _decode_signature(value: bytes) -> bytes:
    try:
        decoded = base64.b64decode(value.strip(), validate=True)
    except Exception as exc:
        raise UpdateError("invalid_signature", "The update signature is not valid base64.") from exc
    if len(decoded) != 64:
        raise UpdateError("invalid_signature", "The update signature has an invalid length.")
    return decoded


def _require_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise UpdateError("invalid_manifest", f"The update manifest has an invalid {key!r} field.")
    return value.strip()


def _validate_release_url(value: str, tag: str) -> None:
    expected = f"{GITHUB_RELEASE_ROOT}/releases/tag/{quote(tag, safe='')}"
    if value != expected:
        raise UpdateError("wrong_release_url", "The signed release URL does not match this product and tag.")


def _validate_asset_url(value: str, tag: str, asset_name: str) -> None:
    expected = (
        f"{GITHUB_RELEASE_ROOT}/releases/download/"
        f"{quote(tag, safe='')}/{quote(asset_name, safe='._-')}"
    )
    if value != expected:
        raise UpdateError("wrong_asset_url", "The signed asset URL does not match this product and tag.")


def _parse_asset(value: object, *, tag: str) -> UpdateAsset:
    if not isinstance(value, dict):
        raise UpdateError("invalid_manifest", "An update asset entry is not a JSON object.")
    platform_name = _require_string(value, "platform")
    architecture = _require_string(value, "architecture")
    variant = _require_string(value, "variant")
    name = _require_string(value, "name")
    url = _require_string(value, "url")
    size = value.get("size")
    sha256 = value.get("sha256")
    if not isinstance(size, int) or isinstance(size, bool) or not 1 <= size <= ARTIFACT_MAX_BYTES:
        raise UpdateError("invalid_manifest", "An update asset has an invalid or excessive size.")
    if not isinstance(sha256, str) or _SHA256_PATTERN.fullmatch(sha256) is None:
        raise UpdateError("invalid_manifest", "An update asset has an invalid SHA-256 value.")
    _validate_asset_url(url, tag, name)
    return UpdateAsset(platform_name, architecture, variant, name, url, size, sha256)


def verify_signed_manifest(
    manifest_bytes: bytes,
    signature_bytes: bytes,
    *,
    platform_name: Optional[str] = None,
    architecture: Optional[str] = None,
    public_key_base64: str = ED25519_PUBLIC_KEY_BASE64,
) -> tuple[dict[str, object], UpdateAsset, SemanticVersion, SemanticVersion]:
    if not manifest_bytes or len(manifest_bytes) > MANIFEST_MAX_BYTES:
        raise UpdateError("invalid_manifest", "The update manifest is missing or too large.")
    if not signature_bytes or len(signature_bytes) > SIGNATURE_MAX_BYTES:
        raise UpdateError("invalid_signature", "The update signature is missing or too large.")
    try:
        public_key_raw = base64.b64decode(public_key_base64, validate=True)
        public_key = Ed25519PublicKey.from_public_bytes(public_key_raw)
        public_key.verify(_decode_signature(signature_bytes), manifest_bytes)
    except UpdateError:
        raise
    except (InvalidSignature, ValueError) as exc:
        raise UpdateError("invalid_signature", "The update manifest signature is invalid.") from exc

    try:
        decoded = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UpdateError("invalid_manifest", "The signed update manifest is not valid UTF-8 JSON.") from exc
    if not isinstance(decoded, dict):
        raise UpdateError("invalid_manifest", "The signed update manifest root is not an object.")
    if canonical_manifest_bytes(decoded) != manifest_bytes:
        raise UpdateError("noncanonical_manifest", "The signed update manifest is not canonical JSON.")
    if decoded.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise UpdateError("unsupported_manifest", "This update manifest schema is not supported.")
    if decoded.get("product") != PRODUCT_ID:
        raise UpdateError("wrong_product", "The update belongs to a different product.")
    if decoded.get("channel") != UPDATE_CHANNEL:
        raise UpdateError("wrong_channel", "The update is not from the stable channel.")

    version = SemanticVersion.parse(decoded.get("version"))
    tag = _require_string(decoded, "tag")
    if tag != f"v{version}":
        raise UpdateError("wrong_tag", "The manifest version and release tag do not match.")
    release_url = _require_string(decoded, "release_url")
    _validate_release_url(release_url, tag)
    minimum_updater = SemanticVersion.parse(decoded.get("minimum_updater_version"))
    _require_string(decoded, "published_at")

    values = decoded.get("assets")
    if not isinstance(values, list) or not values:
        raise UpdateError("missing_asset", "The update manifest has no release assets.")
    assets = [_parse_asset(value, tag=tag) for value in values]
    wanted_platform = normalize_platform(platform_name)
    wanted_architecture = normalize_architecture(architecture)
    matches = [
        asset
        for asset in assets
        if asset.platform == wanted_platform
        and asset.architecture == wanted_architecture
        and asset.variant == UPDATE_VARIANT
    ]
    if len(matches) != 1:
        raise UpdateError(
            "missing_asset" if not matches else "ambiguous_asset",
            "The manifest does not contain exactly one compatible CtrlSpeak artifact.",
        )
    selected = matches[0]
    if selected.name != expected_asset_name(wanted_platform):
        raise UpdateError("wrong_asset", "The compatible update artifact has an unexpected name.")
    return decoded, selected, version, minimum_updater


def _validate_initial_release_asset_url(url: object, tag: str, asset_name: str) -> str:
    if not isinstance(url, str):
        raise UpdateError("invalid_release", f"GitHub did not provide {asset_name}.")
    _validate_asset_url(url, tag, asset_name)
    return url


def _validate_download_response(response: requests.Response) -> None:
    parsed = urlsplit(response.url)
    if parsed.scheme.lower() != "https" or (parsed.hostname or "").lower() not in APPROVED_DOWNLOAD_HOSTS:
        raise UpdateError("unsafe_redirect", "GitHub redirected the update to an unapproved host.")


def _request(
    session: requests.Session,
    url: str,
    *,
    stream: bool,
    headers: Optional[dict[str, str]] = None,
) -> requests.Response:
    try:
        response = session.get(
            url,
            headers=headers,
            stream=stream,
            timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
            allow_redirects=True,
        )
    except requests.exceptions.Timeout as exc:
        raise UpdateError("network_timeout", "The update server did not respond in time.") from exc
    except requests.exceptions.SSLError as exc:
        raise UpdateError("tls_failure", "A secure connection to GitHub could not be verified.") from exc
    except OSError as exc:
        raise UpdateError(
            "tls_failure",
            "CtrlSpeak could not load its secure certificate bundle. Restart CtrlSpeak and try again.",
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise UpdateError("network_error", "CtrlSpeak could not reach GitHub to check for updates.") from exc
    _validate_download_response(response)
    return response


def _read_bounded_response(response: requests.Response, maximum: int, label: str) -> bytes:
    try:
        response.raise_for_status()
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) > maximum:
            raise UpdateError("oversize_metadata", f"The {label} is larger than allowed.")
        output = bytearray()
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            output.extend(chunk)
            if len(output) > maximum:
                raise UpdateError("oversize_metadata", f"The {label} is larger than allowed.")
        return bytes(output)
    except UpdateError:
        raise
    except (requests.exceptions.RequestException, ValueError) as exc:
        raise UpdateError("invalid_release", f"CtrlSpeak could not download the {label}.") from exc
    finally:
        response.close()


def _find_release_asset(payload: Mapping[str, object], name: str, tag: str) -> str:
    assets = payload.get("assets")
    if not isinstance(assets, list):
        raise UpdateError("invalid_release", "The GitHub Release has no asset list.")
    matches = [item for item in assets if isinstance(item, dict) and item.get("name") == name]
    if len(matches) != 1:
        raise UpdateError(
            "missing_metadata" if not matches else "ambiguous_metadata",
            f"The GitHub Release does not contain exactly one {name} asset.",
        )
    return _validate_initial_release_asset_url(matches[0].get("browser_download_url"), tag, name)


def discover_update(
    current_version: str,
    *,
    session: Optional[requests.Session] = None,
    platform_name: Optional[str] = None,
    architecture: Optional[str] = None,
) -> UpdateCheckResult:
    current = SemanticVersion.parse(current_version)
    client = session or requests.Session()
    client.headers.setdefault("User-Agent", f"CtrlSpeak/{current_version} updater")
    client.headers.setdefault("Accept", "application/vnd.github+json")
    client.headers.setdefault("X-GitHub-Api-Version", "2022-11-28")
    try:
        try:
            response = client.get(
                GITHUB_API_URL,
                timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
                allow_redirects=False,
            )
        except requests.exceptions.Timeout as exc:
            raise UpdateError("network_timeout", "GitHub did not respond before the update check timed out.") from exc
        except requests.exceptions.SSLError as exc:
            raise UpdateError("tls_failure", "A secure connection to GitHub could not be verified.") from exc
        except OSError as exc:
            raise UpdateError(
                "tls_failure",
                "CtrlSpeak could not load its secure certificate bundle. Restart CtrlSpeak and try again.",
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise UpdateError("network_error", "CtrlSpeak could not reach GitHub to check for updates.") from exc

        if response.status_code == 403:
            raise UpdateError("rate_limited", "GitHub temporarily refused the update check; try again later.")
        if response.status_code == 404:
            raise UpdateError("no_release", "No stable CtrlSpeak release is currently published.")
        if response.status_code != 200:
            raise UpdateError("github_error", f"GitHub returned HTTP {response.status_code} during the update check.")
        try:
            release_payload = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise UpdateError("invalid_release", "GitHub returned invalid release metadata.") from exc
        if not isinstance(release_payload, dict):
            raise UpdateError("invalid_release", "GitHub returned invalid release metadata.")
        if release_payload.get("draft") is not False or release_payload.get("prerelease") is not False:
            raise UpdateError("unstable_release", "GitHub's latest release is not a stable published release.")

        tag = _require_string(release_payload, "tag_name")
        release_version = SemanticVersion.parse(tag[1:] if tag.startswith("v") else "")
        manifest_url = _find_release_asset(release_payload, MANIFEST_ASSET_NAME, tag)
        signature_url = _find_release_asset(release_payload, SIGNATURE_ASSET_NAME, tag)
        manifest_response = _request(client, manifest_url, stream=True)
        manifest_bytes = _read_bounded_response(manifest_response, MANIFEST_MAX_BYTES, "update manifest")
        signature_response = _request(client, signature_url, stream=True)
        signature_bytes = _read_bounded_response(signature_response, SIGNATURE_MAX_BYTES, "update signature")
        manifest, asset, manifest_version, minimum_updater = verify_signed_manifest(
            manifest_bytes,
            signature_bytes,
            platform_name=platform_name,
            architecture=architecture,
        )
        if manifest_version != release_version or manifest.get("tag") != tag:
            raise UpdateError("release_mismatch", "GitHub and the signed manifest identify different releases.")

        release_url = _require_string(release_payload, "html_url")
        if release_url != manifest.get("release_url"):
            raise UpdateError("release_mismatch", "GitHub and the signed manifest have different release URLs.")
        release = VerifiedRelease(
            version=manifest_version,
            tag=tag,
            title=str(release_payload.get("name") or tag),
            notes=sanitize_release_notes(release_payload.get("body")),
            published_at=_require_string(manifest, "published_at"),
            release_url=release_url,
            minimum_updater_version=minimum_updater,
            asset=asset,
            manifest_bytes=manifest_bytes,
            signature_bytes=signature_bytes,
        )
        checked_at = utc_now_iso()
        if current < minimum_updater:
            return UpdateCheckResult(
                "manual_install_required",
                "This release requires a newer updater. Open the verified GitHub Release to install it manually.",
                checked_at,
                release,
            )
        if release_version > current:
            return UpdateCheckResult("available", f"CtrlSpeak {release_version} is available.", checked_at, release)
        if release_version == current:
            return UpdateCheckResult("up_to_date", "CtrlSpeak is up to date.", checked_at, release)
        return UpdateCheckResult(
            "newer_than_release",
            "This CtrlSpeak build is newer than the latest stable release; no downgrade is offered.",
            checked_at,
            release,
        )
    finally:
        if session is None:
            client.close()


def get_updates_dir() -> Path:
    path = get_config_dir() / "updates"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_write_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _transaction_paths(transaction_id: str, root: Optional[Path] = None) -> UpdateTransaction:
    if _TRANSACTION_ID_PATTERN.fullmatch(transaction_id) is None:
        raise UpdateError("invalid_transaction", "The update transaction identifier is invalid.")
    updates_root = Path(root or get_updates_dir()).resolve()
    directory = (updates_root / transaction_id).resolve()
    if directory.parent != updates_root:
        raise UpdateError("unsafe_path", "The update transaction path escaped the update directory.")
    return UpdateTransaction(
        transaction_id=transaction_id,
        directory=directory,
        journal_path=directory / "transaction.json",
        manifest_path=directory / MANIFEST_ASSET_NAME,
        signature_path=directory / SIGNATURE_ASSET_NAME,
        partial_path=directory / "artifact.partial",
        candidate_path=directory / "verified-candidate",
        release_metadata_path=directory / "release.json",
        health_request_path=directory / "health-request.json",
        health_response_path=directory / "health-response.json",
        updater_log_path=directory / "update.log",
    )


def resolve_transaction(transaction_id: str, *, root: Optional[Path] = None) -> UpdateTransaction:
    transaction = _transaction_paths(transaction_id, root)
    if not transaction.directory.is_dir():
        raise UpdateError("missing_transaction", "The update transaction directory does not exist.")
    return transaction


def create_transaction(
    release: VerifiedRelease,
    *,
    installation_path: Optional[Path] = None,
    root: Optional[Path] = None,
) -> UpdateTransaction:
    transaction = _transaction_paths(uuid.uuid4().hex, root)
    transaction.directory.mkdir(parents=False, exist_ok=False)
    _atomic_write_bytes(transaction.manifest_path, release.manifest_bytes)
    _atomic_write_bytes(transaction.signature_path, release.signature_bytes)
    release_metadata = {
        "version": str(release.version),
        "tag": release.tag,
        "title": release.title,
        "notes": release.notes,
        "published_at": release.published_at,
        "release_url": release.release_url,
    }
    _atomic_write_json(transaction.release_metadata_path, release_metadata)
    journal = {
        "schema_version": 1,
        "transaction_id": transaction.transaction_id,
        "state": "created",
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "product": PRODUCT_ID,
        "channel": UPDATE_CHANNEL,
        "variant": UPDATE_VARIANT,
        "version": str(release.version),
        "tag": release.tag,
        "manifest_sha256": release.manifest_sha256,
        "platform": release.asset.platform,
        "architecture": release.asset.architecture,
        "asset_name": release.asset.name,
        "asset_url": release.asset.url,
        "asset_size": release.asset.size,
        "asset_sha256": release.asset.sha256,
        "installation_path": str(Path(installation_path or sys.executable).resolve()),
    }
    _atomic_write_json(transaction.journal_path, journal)
    return transaction


def read_transaction_journal(transaction: UpdateTransaction) -> dict[str, object]:
    try:
        payload = json.loads(transaction.journal_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise UpdateError("invalid_transaction", "The update transaction journal cannot be read.") from exc
    if not isinstance(payload, dict) or payload.get("transaction_id") != transaction.transaction_id:
        raise UpdateError("invalid_transaction", "The update transaction journal does not match its directory.")
    return payload


def update_transaction_journal(transaction: UpdateTransaction, **changes: object) -> dict[str, object]:
    journal = read_transaction_journal(transaction)
    journal.update(changes)
    journal["updated_at"] = utc_now_iso()
    _atomic_write_json(transaction.journal_path, journal)
    return journal


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, asset: UpdateAsset) -> None:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise UpdateError("missing_candidate", "The downloaded update file is missing.") from exc
    if size != asset.size:
        raise UpdateError(
            "size_mismatch",
            f"The downloaded update size is invalid (expected {asset.size} bytes, received {size}).",
        )
    if sha256_file(path) != asset.sha256:
        raise UpdateError("hash_mismatch", "The downloaded update failed SHA-256 verification.")


def _content_length(response: requests.Response) -> Optional[int]:
    value = response.headers.get("Content-Length")
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise UpdateError("invalid_response", "The update server returned an invalid Content-Length.") from exc
    if parsed < 0:
        raise UpdateError("invalid_response", "The update server returned an invalid Content-Length.")
    return parsed


def _download_response(
    session: requests.Session,
    asset: UpdateAsset,
    *,
    resume_from: int,
) -> requests.Response:
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else None
    return _request(session, asset.url, stream=True, headers=headers)


def download_release(
    release: VerifiedRelease,
    transaction: UpdateTransaction,
    *,
    cancel_event: Optional[threading.Event] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    session: Optional[requests.Session] = None,
) -> Path:
    journal = read_transaction_journal(transaction)
    expected_binding = {
        "product": PRODUCT_ID,
        "version": str(release.version),
        "tag": release.tag,
        "manifest_sha256": release.manifest_sha256,
        "asset_name": release.asset.name,
        "asset_url": release.asset.url,
        "asset_size": release.asset.size,
        "asset_sha256": release.asset.sha256,
    }
    if any(journal.get(key) != value for key, value in expected_binding.items()):
        raise UpdateError("stale_partial", "The partial download belongs to a different release.")

    client = session or requests.Session()
    client.headers.setdefault("User-Agent", f"CtrlSpeak/{release.version} updater")
    partial = transaction.partial_path
    candidate = transaction.candidate_path
    candidate.unlink(missing_ok=True)
    partial_size = partial.stat().st_size if partial.exists() else 0
    if partial_size > release.asset.size:
        partial.unlink(missing_ok=True)
        partial_size = 0
    if partial_size == release.asset.size:
        try:
            verify_file(partial, release.asset)
            os.replace(partial, candidate)
            update_transaction_journal(transaction, state="ready_to_install", downloaded=release.asset.size)
            if progress:
                progress(release.asset.size, release.asset.size)
            return candidate
        except UpdateError:
            partial.unlink(missing_ok=True)
            partial_size = 0

    update_transaction_journal(transaction, state="downloading", downloaded=partial_size)
    response: Optional[requests.Response] = None
    try:
        response = _download_response(client, release.asset, resume_from=partial_size)
        if partial_size:
            if response.status_code == 200:
                response.close()
                partial.unlink(missing_ok=True)
                partial_size = 0
                response = _download_response(client, release.asset, resume_from=0)
            elif response.status_code == 416:
                response.close()
                try:
                    verify_file(partial, release.asset)
                    os.replace(partial, candidate)
                    update_transaction_journal(
                        transaction,
                        state="ready_to_install",
                        downloaded=release.asset.size,
                    )
                    if progress:
                        progress(release.asset.size, release.asset.size)
                    return candidate
                except UpdateError:
                    partial.unlink(missing_ok=True)
                    partial_size = 0
                    response = _download_response(client, release.asset, resume_from=0)
            elif response.status_code == 206:
                content_range = response.headers.get("Content-Range", "")
                match = _CONTENT_RANGE_PATTERN.fullmatch(content_range.strip())
                if (
                    match is None
                    or int(match.group(1)) != partial_size
                    or int(match.group(3)) != release.asset.size
                ):
                    raise UpdateError("invalid_range", "The update server returned an invalid resume range.")
            else:
                raise UpdateError(
                    "download_http_error",
                    f"The update download returned HTTP {response.status_code}.",
                )
        elif response.status_code != 200:
            raise UpdateError(
                "download_http_error",
                f"The update download returned HTTP {response.status_code}.",
            )

        remaining = release.asset.size - partial_size
        advertised = _content_length(response)
        if advertised is not None and advertised > remaining:
            partial.unlink(missing_ok=True)
            raise UpdateError("oversize_download", "The update server advertised more data than the signed size.")

        mode = "ab" if partial_size else "wb"
        downloaded = partial_size
        last_progress_at = 0.0
        with partial.open(mode) as stream:
            if progress:
                progress(downloaded, release.asset.size)
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                if cancel_event is not None and cancel_event.is_set():
                    stream.flush()
                    os.fsync(stream.fileno())
                    update_transaction_journal(transaction, state="cancelled", downloaded=downloaded)
                    raise UpdateCancelled()
                if not chunk:
                    continue
                if downloaded + len(chunk) > release.asset.size:
                    stream.close()
                    partial.unlink(missing_ok=True)
                    raise UpdateError("oversize_download", "The update server sent more than the signed size.")
                stream.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if progress and now - last_progress_at >= PROGRESS_INTERVAL_SECONDS:
                    progress(downloaded, release.asset.size)
                    last_progress_at = now
            stream.flush()
            os.fsync(stream.fileno())

        if downloaded != release.asset.size:
            raise UpdateError(
                "incomplete_download",
                f"The update download ended at {downloaded} of {release.asset.size} bytes.",
            )
        update_transaction_journal(transaction, state="verifying", downloaded=downloaded)
        verify_file(partial, release.asset)
        os.replace(partial, candidate)
        update_transaction_journal(transaction, state="ready_to_install", downloaded=downloaded)
        if progress:
            progress(downloaded, release.asset.size)
        logger.info(
            "Verified update transaction %s: version=%s tag=%s platform=%s architecture=%s bytes=%s sha256=%s",
            transaction.transaction_id,
            release.version,
            release.tag,
            release.asset.platform,
            release.asset.architecture,
            downloaded,
            release.asset.sha256,
        )
        return candidate
    except UpdateError:
        raise
    except requests.exceptions.Timeout as exc:
        raise UpdateError("download_stalled", "The update transfer stalled before completion.") from exc
    except requests.exceptions.RequestException as exc:
        raise UpdateError("network_error", "The update download was interrupted by a network error.") from exc
    except OSError as exc:
        raise UpdateError("disk_error", "CtrlSpeak could not write the update to its application-data directory.") from exc
    finally:
        if response is not None:
            response.close()
        if session is None:
            client.close()


class UpdateCoordinator:
    """Own one process-wide update operation and reject stale worker events."""

    def __init__(
        self,
        current_version: str,
        *,
        discoverer: Callable[..., UpdateCheckResult] = discover_update,
        downloader: Callable[..., Path] = download_release,
        transaction_factory: Callable[..., UpdateTransaction] = create_transaction,
    ):
        self.current_version = current_version
        self._discoverer = discoverer
        self._downloader = downloader
        self._transaction_factory = transaction_factory
        self._lock = threading.RLock()
        self._listeners: list[Callable[[UpdateEvent], None]] = []
        self._generation = 0
        self._state = "idle"
        self._message = "Updates have not been checked in this session."
        self._downloaded = 0
        self._total = 0
        self._error_code: Optional[str] = None
        self._release: Optional[VerifiedRelease] = None
        self._transaction: Optional[UpdateTransaction] = None
        self._cancel_event: Optional[threading.Event] = None

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def release(self) -> Optional[VerifiedRelease]:
        with self._lock:
            return self._release

    @property
    def transaction(self) -> Optional[UpdateTransaction]:
        with self._lock:
            return self._transaction

    def add_listener(self, listener: Callable[[UpdateEvent], None]) -> None:
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[UpdateEvent], None]) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def snapshot(self) -> UpdateEvent:
        with self._lock:
            return UpdateEvent(
                self._generation,
                self._state,
                self._message,
                downloaded=self._downloaded,
                total=self._total,
                release=self._release,
                transaction=self._transaction,
                error_code=self._error_code,
            )

    def _next_generation(self, state: str) -> int:
        with self._lock:
            if self._cancel_event is not None:
                self._cancel_event.set()
            self._generation += 1
            self._state = state
            self._cancel_event = threading.Event()
            return self._generation

    def _publish(self, event: UpdateEvent) -> bool:
        with self._lock:
            if event.generation != self._generation:
                logger.debug(
                    "Ignored stale update event generation=%s active=%s state=%s",
                    event.generation,
                    self._generation,
                    event.state,
                )
                return False
            self._state = event.state
            self._message = event.message
            self._downloaded = event.downloaded
            self._total = event.total
            self._error_code = event.error_code
            if event.release is not None:
                self._release = event.release
            if event.transaction is not None:
                self._transaction = event.transaction
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(event)
            except Exception:
                logger.exception("Update listener failed while handling %s", event.state)
        return True

    def check_async(self) -> int:
        generation = self._next_generation("checking")
        self._publish(UpdateEvent(generation, "checking", "Checking GitHub for a signed stable release…"))

        def worker() -> None:
            try:
                result = self._discoverer(self.current_version)
                self._publish(
                    UpdateEvent(
                        generation,
                        result.status,
                        result.message,
                        release=result.release,
                    )
                )
            except UpdateError as exc:
                logger.warning("Update check failed [%s]: %s", exc.code, exc.user_message)
                self._publish(
                    UpdateEvent(
                        generation,
                        "failed",
                        exc.user_message,
                        error_code=exc.code,
                    )
                )
            except Exception:
                logger.exception("Unexpected update check failure")
                self._publish(
                    UpdateEvent(
                        generation,
                        "failed",
                        "The update check failed unexpectedly. See the CtrlSpeak log for details.",
                        error_code="unexpected_error",
                    )
                )

        threading.Thread(target=worker, name=f"CtrlSpeakUpdateCheck-{generation}", daemon=True).start()
        return generation

    def download_async(self, *, installation_path: Optional[Path] = None) -> int:
        with self._lock:
            release = self._release
            existing_transaction = self._transaction
        if release is None:
            raise UpdateError("no_update", "No verified update is ready to download.")
        generation = self._next_generation("downloading")
        cancel_event = self._cancel_event
        self._publish(
            UpdateEvent(
                generation,
                "downloading",
                f"Preparing CtrlSpeak {release.version}…",
                total=release.asset.size,
                release=release,
            )
        )

        def worker() -> None:
            try:
                transaction = existing_transaction
                if transaction is None:
                    transaction = self._transaction_factory(
                        release,
                        installation_path=installation_path,
                    )
                if not self._publish(
                    UpdateEvent(
                        generation,
                        "downloading",
                        f"Downloading CtrlSpeak {release.version}…",
                        total=release.asset.size,
                        release=release,
                        transaction=transaction,
                    )
                ):
                    return

                def on_progress(downloaded: int, total: int) -> None:
                    self._publish(
                        UpdateEvent(
                            generation,
                            "downloading",
                            f"Downloading CtrlSpeak {release.version}…",
                            downloaded=downloaded,
                            total=total,
                            release=release,
                            transaction=transaction,
                        )
                    )

                candidate = self._downloader(
                    release,
                    transaction,
                    cancel_event=cancel_event,
                    progress=on_progress,
                )
                self._publish(
                    UpdateEvent(
                        generation,
                        "verifying",
                        "Verifying the signed update…",
                        downloaded=release.asset.size,
                        total=release.asset.size,
                        release=release,
                        transaction=transaction,
                    )
                )
                verify_file(candidate, release.asset)
                self._publish(
                    UpdateEvent(
                        generation,
                        "ready_to_install",
                        f"CtrlSpeak {release.version} is verified and ready to install.",
                        downloaded=release.asset.size,
                        total=release.asset.size,
                        release=release,
                        transaction=transaction,
                    )
                )
            except UpdateCancelled:
                self._publish(
                    UpdateEvent(
                        generation,
                        "available",
                        "Download cancelled. The compatible partial is retained for retry.",
                        release=release,
                        transaction=self._transaction,
                    )
                )
            except UpdateError as exc:
                logger.warning("Update download failed [%s]: %s", exc.code, exc.user_message)
                self._publish(
                    UpdateEvent(
                        generation,
                        "failed",
                        exc.user_message,
                        release=release,
                        transaction=self._transaction,
                        error_code=exc.code,
                    )
                )
            except Exception:
                logger.exception("Unexpected update download failure")
                self._publish(
                    UpdateEvent(
                        generation,
                        "failed",
                        "The update download failed unexpectedly. See the CtrlSpeak log for details.",
                        release=release,
                        transaction=self._transaction,
                        error_code="unexpected_error",
                    )
                )

        threading.Thread(target=worker, name=f"CtrlSpeakUpdateDownload-{generation}", daemon=True).start()
        return generation

    def cancel(self) -> None:
        with self._lock:
            if self._cancel_event is not None:
                self._cancel_event.set()
            self._generation += 1
            generation = self._generation
            self._state = "available" if self._release is not None else "idle"
            release = self._release
            transaction = self._transaction
        self._publish(
            UpdateEvent(
                generation,
                self._state,
                "Update operation cancelled.",
                release=release,
                transaction=transaction,
            )
        )


_COORDINATOR: Optional[UpdateCoordinator] = None
_COORDINATOR_LOCK = threading.Lock()


def get_update_coordinator(current_version: str) -> UpdateCoordinator:
    global _COORDINATOR
    with _COORDINATOR_LOCK:
        if _COORDINATOR is None or _COORDINATOR.current_version != current_version:
            _COORDINATOR = UpdateCoordinator(current_version)
        return _COORDINATOR
