# -*- coding: utf-8 -*-
"""Signed GitHub Release discovery and verified update staging for CtrlSpeak."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional
from urllib.parse import quote, urlsplit

import requests
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from utils.config_paths import get_config_dir, get_logger
from utils.binary_delta import ALGORITHM, MAX_PATCH_BYTES, MAX_PATCH_RATIO, PatchCancelled, PatchError, apply_patch_file
from utils.version import APP_VERSION


logger = get_logger(__name__)

GITHUB_OWNER = "Wouter-Bekker-AI"
GITHUB_REPOSITORY = "CtrlSpeak"
PRODUCT_ID = "ctrlspeak"
UPDATE_CHANNEL = "stable"
UPDATE_VARIANT = "standard"
ROLLBACK_VERSION = "0.7.7"
ROLLBACK_OPERATION = "rollback_077"
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
class DeltaAsset:
    asset: UpdateAsset
    base_version: str
    base_size: int
    base_sha256: str
    target_size: int
    target_sha256: str
    minimum_updater_version: str = "0.8.2"


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
    # Local, explicit user intent; never supplied by unsigned GitHub metadata.
    operation: str = "update"
    deltas: tuple[DeltaAsset, ...] = ()
    selected_delta: Optional[DeltaAsset] = None

    @property
    def download_size(self) -> int:
        return self.selected_delta.asset.size if self.selected_delta else self.asset.size

    @property
    def transfer_description(self) -> str:
        if self.selected_delta:
            return f"Differential patch from {self.selected_delta.base_version}"
        return "Full application download"

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


def delta_asset_name(platform_name: str, base_version: str) -> str:
    return f"CtrlSpeak-{platform_name}-x86_64-from-{base_version}.vcdiff"


def parse_deltas(manifest: Mapping[str, object], target: UpdateAsset) -> tuple[DeltaAsset, ...]:
    values = manifest.get("deltas", [])
    if not isinstance(values, list) or len(values) > 12:
        raise UpdateError("invalid_delta", "The signed patch list is invalid or excessive.")
    result, seen = [], set()
    for entry in values:
        if not isinstance(entry, dict):
            raise UpdateError("invalid_delta", "A signed patch entry is invalid.")
        if entry.get("algorithm") != ALGORITHM:
            continue  # Future algorithms fall back to the normal signed full asset.
        if (entry.get("platform"), entry.get("architecture"), entry.get("variant")) != (target.platform, target.architecture, target.variant):
            continue
        asset = _parse_asset(entry, tag=str(manifest["tag"]))
        base_version = str(SemanticVersion.parse(entry.get("base_version")))
        minimum = str(SemanticVersion.parse(entry.get("minimum_updater_version")))
        base_size, base_hash = entry.get("base_size"), entry.get("base_sha256")
        if (SemanticVersion.parse(base_version) >= SemanticVersion.parse(manifest["version"])
                or asset.name != delta_asset_name(target.platform, base_version)
                or not isinstance(base_size, int) or isinstance(base_size, bool)
                or not 1 <= base_size <= ARTIFACT_MAX_BYTES
                or not isinstance(base_hash, str) or not _SHA256_PATTERN.fullmatch(base_hash)
                or entry.get("target_sha256") != target.sha256
                or entry.get("target_size") != target.size
                or asset.size > min(MAX_PATCH_BYTES, int(target.size * MAX_PATCH_RATIO))
                or base_version in seen):
            raise UpdateError("invalid_delta", "A signed patch is not bound to a valid base and target.")
        seen.add(base_version)
        result.append(DeltaAsset(asset, base_version, base_size, base_hash, target.size, target.sha256, minimum))
    return tuple(result)


def select_delta(release: VerifiedRelease, current_version: str | None, installation_path: Path) -> VerifiedRelease:
    """Select by exact version, size AND SHA-256, not version text alone."""
    selected = None
    if release.operation == "update" and current_version:
        path = Path(installation_path)
        for delta in release.deltas:
            if delta.base_version != current_version or SemanticVersion.parse(APP_VERSION) < SemanticVersion.parse(delta.minimum_updater_version):
                continue
            try:
                if path.is_file() and path.stat().st_size == delta.base_size and sha256_file(path) == delta.base_sha256:
                    selected = delta
            except OSError:
                pass
    return replace(release, selected_delta=selected)


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
    parse_deltas(decoded, selected)
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
    """Check latest stable only; this path never offers a downgrade."""
    return _discover_release(
        current_version, session=session, platform_name=platform_name,
        architecture=architecture, rollback_077=False,
    )


def discover_rollback_077(
    current_version: str,
    *,
    session: Optional[requests.Session] = None,
    platform_name: Optional[str] = None,
    architecture: Optional[str] = None,
) -> UpdateCheckResult:
    """Discover only the explicitly selected, signed 0.7.7 rollback target."""
    if SemanticVersion.parse(current_version) <= SemanticVersion.parse(ROLLBACK_VERSION):
        raise UpdateError("rollback_not_applicable", "Rollback requires a version newer than 0.7.7.")
    return _discover_release(
        current_version, session=session, platform_name=platform_name,
        architecture=architecture, rollback_077=True,
    )


def _discover_release(
    current_version: str, *, session: Optional[requests.Session],
    platform_name: Optional[str], architecture: Optional[str], rollback_077: bool,
) -> UpdateCheckResult:
    current = SemanticVersion.parse(current_version)
    api_url = (
        f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPOSITORY}/releases/tags/v{ROLLBACK_VERSION}"
        if rollback_077 else GITHUB_API_URL
    )
    client = session or requests.Session()
    client.headers.setdefault("User-Agent", f"CtrlSpeak/{current_version} updater")
    client.headers.setdefault("Accept", "application/vnd.github+json")
    client.headers.setdefault("X-GitHub-Api-Version", "2022-11-28")
    try:
        try:
            response = client.get(
                api_url,
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
        if rollback_077 and tag != f"v{ROLLBACK_VERSION}":
            raise UpdateError("wrong_rollback_target", "GitHub did not return the requested 0.7.7 release.")
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
            operation=ROLLBACK_OPERATION if rollback_077 else "update",
            deltas=parse_deltas(manifest, asset),
        )
        checked_at = utc_now_iso()
        if current < minimum_updater:
            return UpdateCheckResult(
                "manual_install_required",
                "This release requires a newer updater. Open the verified GitHub Release to install it manually.",
                checked_at,
                release,
            )
        if rollback_077:
            return UpdateCheckResult(
                "available", "Signed CtrlSpeak 0.7.7 is available for explicit rollback. Settings and saved credentials are retained.",
                checked_at, release,
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
    source_version: Optional[str] = None,
) -> UpdateTransaction:
    _validate_version_operation(release, source_version)
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
        "operation": release.operation,
        "source_version": source_version,
        "explicit_rollback_target": ROLLBACK_VERSION if release.operation == ROLLBACK_OPERATION else None,
    }
    _atomic_write_json(transaction.journal_path, journal)
    return transaction


def _validate_version_operation(release: VerifiedRelease, source_version: Optional[str]) -> None:
    if release.operation not in {"update", ROLLBACK_OPERATION}:
        raise UpdateError("invalid_operation", "The update operation is not recognised.")
    if release.operation == ROLLBACK_OPERATION:
        if (str(release.version) != ROLLBACK_VERSION or release.tag != f"v{ROLLBACK_VERSION}"
                or source_version is None
                or SemanticVersion.parse(source_version) <= release.version):
            raise UpdateError("wrong_rollback_target", "Only an explicit rollback from a newer version to 0.7.7 is permitted.")
    elif source_version is not None and release.version <= SemanticVersion.parse(source_version):
        raise UpdateError("downgrade_not_allowed", "Check for updates does not install an older or identical version.")


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


def prune_completed_update_payloads(*, root: Optional[Path] = None, keep: int = 2,
                                    protected: tuple[str, ...] = ()) -> dict[str, int]:
    """Bound successful-update caches, keeping metadata and recovery states.

    No recursive deletion, no glob removal, no directory removal, no user
    settings/models. Failed, active, unrecognised and symlinked entries stay.
    Keep the most recent explicit 0.7.7 rollback in addition to recent installs.
    """
    directory = Path(root or get_updates_dir()).resolve()
    if not directory.is_dir():
        return {"files": 0, "bytes": 0}
    completed = []
    for index, child in enumerate(directory.iterdir()):
        if index >= 512:
            break  # Bounded metadata inspection, not an unbounded housekeeping job.
        if child.is_symlink() or not child.is_dir() or not _TRANSACTION_ID_PATTERN.fullmatch(child.name):
            continue
        if child.resolve().parent != directory:
            continue
        try:
            journal_path = child / "transaction.json"
            if journal_path.is_symlink() or journal_path.stat().st_size > 65536:
                continue
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
            if (journal.get("transaction_id") == child.name and journal.get("product") == PRODUCT_ID
                    and journal.get("state") == "installed" and journal.get("health_confirmed") is True):
                completed.append((str(journal.get("installed_at", journal.get("updated_at", ""))), child, journal))
        except (OSError, ValueError, AttributeError):
            continue
    completed.sort(key=lambda item: item[0], reverse=True)
    retained = {item[1].name for item in completed[:max(2, keep)]} | set(protected)
    last_rollback = next((item[1].name for item in completed if item[2].get("operation") == ROLLBACK_OPERATION), None)
    if last_rollback:
        retained.add(last_rollback)
    stats = {"files": 0, "bytes": 0}
    names = ("artifact.partial", "verified-candidate", "previous-executable",
             "ctrlspeak-updater-helper.exe", "ctrlspeak-updater-helper", "delta.partial", "delta.vcdiff")
    for _, child, _ in completed:
        if child.name in retained:
            continue
        for name in names:
            path = child / name
            try:
                if path.is_file() and not path.is_symlink() and path.resolve().parent == child.resolve():
                    size = path.stat().st_size
                    path.unlink()
                    stats["files"] += 1
                    stats["bytes"] += size
            except OSError:
                continue  # A locked payload remains recoverable; never force-delete it.
    return stats


def latest_completed_transfer(*, root: Optional[Path] = None) -> dict[str, object] | None:
    """Redacted, bounded receipt for proving a patch was used after restart."""
    directory = Path(root or get_updates_dir()).resolve()
    if not directory.is_dir():
        return None
    candidates = []
    for index, child in enumerate(directory.iterdir()):
        if index >= 512:
            break
        if child.is_symlink() or not child.is_dir() or not _TRANSACTION_ID_PATTERN.fullmatch(child.name):
            continue
        try:
            path = child / "transaction.json"
            if path.is_symlink() or path.stat().st_size > 65536:
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("state") != "installed" or row.get("health_confirmed") is not True or row.get("product") != PRODUCT_ID or row.get("transaction_id") != child.name:
                continue
            version = str(SemanticVersion.parse(row.get("version")))
            full, downloaded = row.get("asset_size"), row.get("transfer_bytes", row.get("asset_size"))
            mode = row.get("transfer_mode", "full")
            if (mode not in {"full", "delta"} or not isinstance(full, int) or isinstance(full, bool)
                    or not isinstance(downloaded, int) or isinstance(downloaded, bool)
                    or not 1 <= downloaded <= full <= ARTIFACT_MAX_BYTES):
                continue
            candidates.append((str(row.get("installed_at", row.get("updated_at", ""))),
                {"version": version, "transfer_mode": mode, "download_bytes": downloaded, "full_bytes": full}))
        except (OSError, ValueError, AttributeError, UpdateError):
            continue
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


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


def _download_full_release(
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
        "operation": release.operation,
    }
    if any(journal.get(key) != value for key, value in expected_binding.items()):
        raise UpdateError("stale_partial", "The partial download belongs to a different release.")

    client = session or requests.Session()
    client.headers.setdefault("User-Agent", f"CtrlSpeak/{release.version} updater")
    partial = transaction.partial_path
    candidate = transaction.candidate_path
    if candidate.is_file():
        try:
            verify_file(candidate, release.asset)
            update_transaction_journal(transaction, state="ready_to_install", downloaded=release.asset.size)
            if progress:
                progress(release.asset.size, release.asset.size)
            if session is None:
                client.close()
            return candidate
        except UpdateError:
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


def download_release(
    release: VerifiedRelease, transaction: UpdateTransaction, *,
    cancel_event: Optional[threading.Event] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    session: Optional[requests.Session] = None,
    phase: Optional[Callable[[str, str], None]] = None,
) -> Path:
    """Reconstruct an exact signed candidate, or download that full candidate.

    Installation still uses the existing external helper/health/rollback flow.
    Patches are never applied to the running executable or user configuration.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise UpdateCancelled()
    journal = read_transaction_journal(transaction)
    expected = {"version": str(release.version), "tag": release.tag,
                "manifest_sha256": release.manifest_sha256,
                "asset_name": release.asset.name, "asset_size": release.asset.size,
                "asset_sha256": release.asset.sha256, "asset_url": release.asset.url,
                "product": PRODUCT_ID, "operation": release.operation}
    if any(journal.get(k) != v for k, v in expected.items()):
        raise UpdateError("stale_partial", "The staged update does not match the selected signed release.")
    selected = None
    base = Path(str(journal.get("installation_path", "")))
    if release.deltas and release.operation == "update":
        # Revalidate the signed bytes at use time, not a mutable dataclass or
        # transaction's proposed patch/hash. A full candidate remains mandatory.
        signed, asset, version, _ = verify_signed_manifest(
            release.manifest_bytes, release.signature_bytes,
            platform_name=release.asset.platform, architecture=release.asset.architecture)
        if asset != release.asset or version != release.version:
            raise UpdateError("release_mismatch", "The patch target differs from the signed release.")
        selected = select_delta(replace(release, deltas=parse_deltas(signed, asset)),
                                journal.get("source_version"), base).selected_delta
    # Smaller network transfer does not remove candidate/sibling/rollback or
    # one-file extraction space. Reserve conservatively on both relevant drives.
    base_size = base.stat().st_size if base.is_file() else release.asset.size
    required = (2 * (base_size + release.asset.size) + 4 * max(base_size, release.asset.size)
                + (selected.asset.size if selected else 0) + 128 * 1024 * 1024)
    locations = {transaction.directory.resolve(), base.parent.resolve()}
    if any(shutil.disk_usage(location).free < required for location in locations):
        raise UpdateError("insufficient_space",
            f"Free at least {required / (1024*1024):.0f} MB on the application-data and installation drives before updating. Small patches still need space for reconstruction, startup and rollback.")
    if selected is not None:
        delta = selected
        patch_transaction = replace(transaction,
            journal_path=transaction.directory / "delta-journal.json",
            partial_path=transaction.directory / "delta.partial",
            candidate_path=transaction.directory / "delta.vcdiff")
        if not patch_transaction.journal_path.exists():
            _atomic_write_json(patch_transaction.journal_path, {
                **journal, "asset_name": delta.asset.name, "asset_url": delta.asset.url,
                "asset_size": delta.asset.size, "asset_sha256": delta.asset.sha256})
        try:
            if phase:
                phase("downloading", f"Downloading differential patch from {delta.base_version}…")
            update_transaction_journal(transaction, state="downloading", transfer_mode="delta",
                                       delta_name=delta.asset.name, transfer_bytes=delta.asset.size)
            patch_path = _download_full_release(replace(release, asset=delta.asset), patch_transaction,
                cancel_event=cancel_event, progress=progress, session=session)
            if cancel_event is not None and cancel_event.is_set():
                raise UpdateCancelled()
            if base.stat().st_size != delta.base_size or sha256_file(base) != delta.base_sha256:
                raise PatchError("Patch base changed")
            update_transaction_journal(transaction, state="applying_patch")
            if phase:
                phase("applying_patch", "Reconstructing the new executable locally; the running app is untouched…")
            apply_patch_file(base, patch_path, transaction.candidate_path, release.asset.size, cancel=cancel_event)
            verify_file(transaction.candidate_path, release.asset)
            update_transaction_journal(transaction, state="ready_to_install", downloaded=delta.asset.size,
                transfer_mode="delta", transfer_bytes=delta.asset.size, reconstructed_size=release.asset.size)
            return transaction.candidate_path
        except (UpdateCancelled, PatchCancelled) as exc:
            raise UpdateCancelled() from exc
        except (UpdateError, PatchError, OSError, RuntimeError) as exc:
            if isinstance(exc, UpdateError) and exc.code in {"insufficient_space", "disk_error"}:
                raise
            if cancel_event is not None and cancel_event.is_set():
                raise UpdateCancelled() from exc
            transaction.candidate_path.unlink(missing_ok=True)
            reason = exc.code if isinstance(exc, UpdateError) else "patch_failed"
            update_transaction_journal(transaction, transfer_mode="full", delta_fallback_reason=reason)
            logger.warning("Differential update fallback [%s]; using signed full artifact", reason)
            if phase:
                phase("falling_back", "Patch unavailable or failed verification; switching to the full signed download…")
    elif release.selected_delta is not None and phase:
        phase("falling_back", "Installed base no longer matches the patch; using the full signed download…")
    if cancel_event is not None and cancel_event.is_set():
        raise UpdateCancelled()
    candidate = _download_full_release(release, transaction, cancel_event=cancel_event,
                                       progress=progress, session=session)
    update_transaction_journal(transaction, transfer_mode="full", transfer_bytes=release.asset.size)
    return candidate


class UpdateCoordinator:
    """Own one process-wide update operation and reject stale worker events."""

    def __init__(
        self,
        current_version: str,
        *,
        discoverer: Callable[..., UpdateCheckResult] = discover_update,
        downloader: Callable[..., Path] = download_release,
        transaction_factory: Callable[..., UpdateTransaction] = create_transaction,
        rollback_discoverer: Callable[..., UpdateCheckResult] = discover_rollback_077,
    ):
        self.current_version = current_version
        self._discoverer = discoverer
        self._downloader = downloader
        self._transaction_factory = transaction_factory
        self._rollback_discoverer = rollback_discoverer
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
        self._download_worker: Optional[threading.Thread] = None

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
        return self._check_async(rollback_077=False)

    def rollback_to_077(self) -> int:
        """Select the rollback target only; download/install still need user actions."""
        if SemanticVersion.parse(self.current_version) <= SemanticVersion.parse(ROLLBACK_VERSION):
            raise UpdateError("rollback_not_applicable", "Rollback requires a version newer than 0.7.7.")
        return self._check_async(rollback_077=True)

    def _check_async(self, *, rollback_077: bool) -> int:
        with self._lock:
            if self._state in {"downloading", "applying_patch", "falling_back", "verifying"} or (self._download_worker is not None and self._download_worker.is_alive()):
                raise UpdateError("update_busy", "Cancel the active download and wait for its patch worker to close before checking again.")
            generation = self._next_generation("checking")
            # A newly selected release must never reuse an earlier candidate or
            # leave it installable after a failed/superseded check.
            self._release = None
            self._transaction = None
        message = ("Checking the signed 0.7.7 rollback release…" if rollback_077
                   else "Checking GitHub for a signed stable release…")
        self._publish(UpdateEvent(generation, "checking", message))

        def worker() -> None:
            try:
                discoverer = self._rollback_discoverer if rollback_077 else self._discoverer
                result = discoverer(self.current_version)
                if result.release is not None and result.status == "available":
                    planned = select_delta(result.release, self.current_version, Path(sys.executable))
                    result = replace(result, release=planned,
                        message=result.message + f" {planned.transfer_description}: {planned.download_size / (1024*1024):.1f} MB.")
                if result.release is not None and rollback_077:
                    _validate_version_operation(result.release, self.current_version)
                    if result.release.operation != ROLLBACK_OPERATION:
                        raise UpdateError("wrong_rollback_target", "The selected release is not an explicit rollback.")
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
            if self._state in {"downloading", "applying_patch", "falling_back", "verifying"} or (self._download_worker is not None and self._download_worker.is_alive()):
                raise UpdateError("update_busy", "The previous update worker is still closing; retry shortly.")
            release = self._release
            existing_transaction = self._transaction
            if release is None:
                raise UpdateError("no_update", "No verified update is ready to download.")
            _validate_version_operation(release, self.current_version)
            release = select_delta(release, self.current_version, Path(installation_path or sys.executable))
            generation = self._next_generation("downloading")
            cancel_event = self._cancel_event
        self._publish(
            UpdateEvent(
                generation,
                "downloading",
                f"Preparing CtrlSpeak {release.version}…",
                total=release.download_size,
                release=release,
            )
        )

        def worker() -> None:
            transferred = [0, release.download_size]
            transfer_message = [f"Downloading CtrlSpeak {release.version} ({release.transfer_description.lower()})…"]
            try:
                transaction = existing_transaction
                if transaction is None:
                    transaction = self._transaction_factory(
                        release,
                        installation_path=installation_path,
                        source_version=self.current_version,
                    )
                if not self._publish(
                    UpdateEvent(
                        generation,
                        "downloading",
                        transfer_message[0],
                        total=release.download_size,
                        release=release,
                        transaction=transaction,
                    )
                ):
                    return

                def on_progress(downloaded: int, total: int) -> None:
                    transferred[:] = [downloaded, total]
                    self._publish(
                        UpdateEvent(
                            generation,
                            "downloading",
                            transfer_message[0],
                            downloaded=downloaded,
                            total=total,
                            release=release,
                            transaction=transaction,
                        )
                    )

                def on_phase(state: str, message: str) -> None:
                    if state == "falling_back":
                        transfer_message[0] = f"Downloading CtrlSpeak {release.version} (full-download fallback)…"
                        transferred[:] = [0, release.asset.size]
                    self._publish(UpdateEvent(generation, state, message,
                        downloaded=transferred[0], total=transferred[1], release=release, transaction=transaction))

                extra = {"phase": on_phase} if self._downloader is download_release else {}
                candidate = self._downloader(
                    release,
                    transaction,
                    cancel_event=cancel_event,
                    progress=on_progress,
                    **extra,
                )
                self._publish(
                    UpdateEvent(
                        generation,
                        "verifying",
                        "Verifying the signed update…",
                        downloaded=transferred[0],
                        total=transferred[1],
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
                        downloaded=transferred[0],
                        total=transferred[1],
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

        with self._lock:
            self._download_worker = threading.Thread(target=worker, name=f"CtrlSpeakUpdateDownload-{generation}", daemon=True)
            try:
                self._download_worker.start()
            except Exception:
                self._download_worker = None
                self._state = "failed"
                raise
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
