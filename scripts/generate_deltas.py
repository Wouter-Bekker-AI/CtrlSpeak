#!/usr/bin/env python3
"""Generate and roundtrip-test direct patches from recent signed releases."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_patch_tool import ensure_tool
from scripts.release import read_app_version, RELEASE_FILENAMES
from utils.binary_delta import ALGORITHM, MAX_PATCH_RATIO, MAX_PATCH_BYTES, PatchTooLarge, create_patch, apply_patch_file
from utils.update_manager import (SemanticVersion, GITHUB_RELEASE_ROOT, verify_signed_manifest,
    _request, _read_bounded_response, sha256_file, delta_asset_name, verify_file)


def recent_bases(target_version, *, limit=3):
    headers = {"Accept": "application/vnd.github+json"}
    if os.environ.get("GH_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["GH_TOKEN"]
    response = requests.get("https://api.github.com/repos/Wouter-Bekker-AI/CtrlSpeak/releases?per_page=30",
                            headers=headers, timeout=(8,30))
    response.raise_for_status()
    versions = []
    for release in response.json():
        if release.get("draft") or release.get("prerelease"):
            continue
        try:
            version = SemanticVersion.parse(str(release.get("tag_name", ""))[1:])
        except Exception:
            continue
        if SemanticVersion.parse("0.8.2") <= version < SemanticVersion.parse(target_version):
            versions.append(version)
    # Bootstrap 0.8.2's pipeline against 0.8.1, despite old clients not yet
    # knowing how to apply it. Subsequent releases target patch-aware bases.
    if not versions and target_version == "0.8.2":
        versions.append(SemanticVersion.parse("0.8.1"))
    return [str(v) for v in sorted(set(versions), reverse=True)[:limit]]


def signed_base(directory, version, platform_name):
    root = f"{GITHUB_RELEASE_ROOT}/releases/download/v{version}/"
    with requests.Session() as session:
        manifest = _read_bounded_response(_request(session, root+"update-manifest.json", stream=True), 512*1024, "base manifest")
        signature = _read_bounded_response(_request(session, root+"update-manifest.sig", stream=True), 8192, "base signature")
        _, asset, observed, _ = verify_signed_manifest(manifest, signature,
                platform_name=platform_name, architecture="x86_64")
        if str(observed) != version:
            raise RuntimeError("Signed patch base version mismatch")
        path = directory / ".delta-bases" / version / asset.name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            verify_file(path,asset)
            return path, asset
        deadline = time.monotonic()+600
        with _request(session, asset.url, stream=True) as response, path.open("wb") as out:
            response.raise_for_status()
            size=0
            for chunk in response.iter_content(1024*1024):
                size+=len(chunk)
                if size>asset.size or time.monotonic()>deadline:
                    raise RuntimeError("Signed base download exceeded its size/deadline")
                out.write(chunk)
        verify_file(path,asset)
        return path, asset


def generate(directory, bases=None):
    directory=Path(directory)
    version=read_app_version()
    ensure_tool()
    bases=recent_bases(version) if bases is None else bases
    entries, evidence = [], []
    for base_version in bases:
        if SemanticVersion.parse(base_version)>=SemanticVersion.parse(version):
            raise RuntimeError("Patch base must precede target")
        for platform_name,name in RELEASE_FILENAMES.items():
            target=directory/name
            base,asset=signed_base(directory,base_version,platform_name)
            patch=directory/delta_asset_name(platform_name,base_version)
            reconstructed=base.parent/(platform_name+"-reconstructed")
            started=time.monotonic()
            try:
                try:
                    create_patch(base,target,patch)
                except PatchTooLarge:
                    evidence.append({"base":base_version,"platform":platform_name,"status":"full_download_patch_budget_exceeded"})
                    continue
                if patch.stat().st_size>min(MAX_PATCH_BYTES,int(target.stat().st_size*MAX_PATCH_RATIO)):
                    patch.unlink()
                    evidence.append({"base":base_version,"platform":platform_name,"status":"full_download_more_economical"})
                    continue
                apply_patch_file(base,patch,reconstructed,target.stat().st_size)
                if sha256_file(reconstructed)!=sha256_file(target):
                    raise RuntimeError("Release patch reconstruction mismatch")
                entries.append({"algorithm":ALGORITHM,"platform":platform_name,"architecture":"x86_64",
                    "variant":"standard","base_version":base_version,"base_size":asset.size,
                    "base_sha256":asset.sha256,"target_size":target.stat().st_size,"target_sha256":sha256_file(target),
                    "minimum_updater_version":"0.8.2","name":patch.name,
                    "url":f"{GITHUB_RELEASE_ROOT}/releases/download/v{version}/{patch.name}",
                    "size":patch.stat().st_size,"sha256":sha256_file(patch)})
                evidence.append({"base":base_version,"platform":platform_name,"status":"byte_identical",
                    "full_bytes":target.stat().st_size,"patch_bytes":patch.stat().st_size,
                    "seconds":round(time.monotonic()-started,2)})
            finally:
                reconstructed.unlink(missing_ok=True)
    (directory/"deltas.json").write_text(json.dumps(entries,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    (directory/"delta-evidence.json").write_text(json.dumps(evidence,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(evidence),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",default="release")
    parser.add_argument("--bases",nargs="*")
    args=parser.parse_args()
    generate(args.directory,args.bases)
