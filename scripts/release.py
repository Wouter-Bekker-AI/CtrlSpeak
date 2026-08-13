#!/usr/bin/env python3
"""Create and independently audit CtrlSpeak signed release metadata."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.update_manager import (  # noqa: E402
    CHECKSUMS_ASSET_NAME,
    GITHUB_RELEASE_ROOT,
    MANIFEST_ASSET_NAME,
    SIGNATURE_ASSET_NAME,
    SemanticVersion,
    canonical_manifest_bytes,
    sha256_file,
    verify_signed_manifest,
)


VERSION_PATTERN = re.compile(r'^APP_VERSION\s*=\s*"([^"]+)"\s*$', re.MULTILINE)
RELEASE_FILENAMES = {
    "windows": "CtrlSpeak-windows-x86_64.exe",
    "linux": "CtrlSpeak-linux-x86_64",
}


def read_app_version() -> str:
    source = (ROOT / "utils" / "version.py").read_text(encoding="utf-8")
    matches = VERSION_PATTERN.findall(source)
    if len(matches) != 1:
        raise SystemExit("utils/version.py must contain exactly one literal APP_VERSION assignment")
    version = matches[0]
    SemanticVersion.parse(version)
    return version


def check_version(tag: str) -> str:
    version = read_app_version()
    if tag != f"v{version}":
        raise SystemExit(f"tag/version mismatch: tag={tag!r}, APP_VERSION={version!r}")
    return version


def _published_at(value: str | None) -> str:
    if value:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise SystemExit("--published-at must include a timezone")
        return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _asset_entry(path: Path, platform_name: str, tag: str) -> dict[str, object]:
    expected_name = RELEASE_FILENAMES[platform_name]
    if path.name != expected_name:
        raise SystemExit(f"expected {expected_name}, received {path.name}")
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"release artifact is missing or empty: {path}")
    return {
        "platform": platform_name,
        "architecture": "x86_64",
        "variant": "standard",
        "name": expected_name,
        "url": f"{GITHUB_RELEASE_ROOT}/releases/download/{tag}/{expected_name}",
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _load_private_key() -> Ed25519PrivateKey:
    encoded = os.environ.get("CTRLSPEAK_UPDATE_SIGNING_KEY", "").strip()
    if not encoded:
        raise SystemExit("CTRLSPEAK_UPDATE_SIGNING_KEY is not configured")
    try:
        raw = base64.b64decode(encoded, validate=True)
        key = Ed25519PrivateKey.from_private_bytes(raw)
    except Exception as exc:
        raise SystemExit("CTRLSPEAK_UPDATE_SIGNING_KEY is not a valid raw Ed25519 private key") from exc
    return key


def generate(args: argparse.Namespace) -> None:
    version = check_version(args.tag)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    windows_path = Path(args.windows).resolve()
    linux_path = Path(args.linux).resolve()
    payload = {
        "schema_version": 1,
        "product": "ctrlspeak",
        "channel": "stable",
        "version": version,
        "tag": args.tag,
        "published_at": _published_at(args.published_at),
        "release_url": f"{GITHUB_RELEASE_ROOT}/releases/tag/{args.tag}",
        "minimum_updater_version": "0.5.0",
        "assets": [
            _asset_entry(windows_path, "windows", args.tag),
            _asset_entry(linux_path, "linux", args.tag),
        ],
    }
    manifest_bytes = canonical_manifest_bytes(payload)
    signature_bytes = base64.b64encode(_load_private_key().sign(manifest_bytes)) + b"\n"
    manifest_path = output_dir / MANIFEST_ASSET_NAME
    signature_path = output_dir / SIGNATURE_ASSET_NAME
    manifest_path.write_bytes(manifest_bytes)
    signature_path.write_bytes(signature_bytes)

    checksum_paths = [windows_path, linux_path, manifest_path, signature_path]
    checksums = "".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_paths)
    (output_dir / CHECKSUMS_ASSET_NAME).write_text(checksums, encoding="ascii", newline="\n")
    print(f"Generated signed CtrlSpeak {version} manifest in {output_dir}")


def _read_checksums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._-]+)", line)
        if match is None or match.group(2) in result:
            raise SystemExit(f"invalid checksum line: {line!r}")
        result[match.group(2)] = match.group(1)
    return result


def verify(args: argparse.Namespace) -> None:
    directory = Path(args.directory).resolve()
    manifest_path = directory / MANIFEST_ASSET_NAME
    signature_path = directory / SIGNATURE_ASSET_NAME
    manifest_bytes = manifest_path.read_bytes()
    signature_bytes = signature_path.read_bytes()
    windows_manifest, windows_asset, version, _ = verify_signed_manifest(
        manifest_bytes,
        signature_bytes,
        platform_name="windows",
        architecture="x86_64",
    )
    linux_manifest, linux_asset, linux_version, _ = verify_signed_manifest(
        manifest_bytes,
        signature_bytes,
        platform_name="linux",
        architecture="x86_64",
    )
    if windows_manifest != linux_manifest or version != linux_version:
        raise SystemExit("platform manifest verification produced inconsistent results")
    if args.tag and str(windows_manifest.get("tag")) != args.tag:
        raise SystemExit("audited manifest tag does not match the requested release tag")
    if str(version) != read_app_version():
        raise SystemExit("audited manifest version does not match APP_VERSION")

    for asset in (windows_asset, linux_asset):
        path = directory / asset.name
        if not path.is_file():
            raise SystemExit(f"missing release artifact: {asset.name}")
        if path.stat().st_size != asset.size or sha256_file(path) != asset.sha256:
            raise SystemExit(f"release artifact failed manifest verification: {asset.name}")
        if asset.platform == "linux" and os.name != "nt" and not os.access(path, os.X_OK):
            raise SystemExit("Linux release artifact is not executable")

    checksum_path = directory / CHECKSUMS_ASSET_NAME
    checksums = _read_checksums(checksum_path)
    expected_names = {
        windows_asset.name,
        linux_asset.name,
        MANIFEST_ASSET_NAME,
        SIGNATURE_ASSET_NAME,
    }
    if set(checksums) != expected_names:
        raise SystemExit("SHA256SUMS does not name exactly the four signed release inputs")
    for name, expected_hash in checksums.items():
        if sha256_file(directory / name) != expected_hash:
            raise SystemExit(f"SHA256SUMS verification failed: {name}")
    print(f"Verified complete CtrlSpeak {version} release set in {directory}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check-version")
    check.add_argument("--tag", required=True)

    create = subparsers.add_parser("generate")
    create.add_argument("--tag", required=True)
    create.add_argument("--windows", required=True)
    create.add_argument("--linux", required=True)
    create.add_argument("--output-dir", required=True)
    create.add_argument("--published-at")

    audit = subparsers.add_parser("verify")
    audit.add_argument("--directory", required=True)
    audit.add_argument("--tag")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "check-version":
        version = check_version(args.tag)
        print(version)
    elif args.command == "generate":
        generate(args)
    elif args.command == "verify":
        verify(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
