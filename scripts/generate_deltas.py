#!/usr/bin/env python3
"""Generate and roundtrip-test direct patches from recent signed releases."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
from urllib.parse import urlsplit
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_patch_tool import ensure_tool
from scripts.release import read_app_version, RELEASE_FILENAMES
from utils.binary_delta import ALGORITHM, MAX_PATCH_RATIO, MAX_PATCH_BYTES, PatchTooLarge, create_patch, apply_patch_file
from utils.update_manager import (SemanticVersion, GITHUB_RELEASE_ROOT, APPROVED_DOWNLOAD_HOSTS, verify_signed_manifest,
    _request, _read_bounded_response, sha256_file, delta_asset_name, verify_file)

RELEASES_API = "https://api.github.com/repos/Wouter-Bekker-AI/CtrlSpeak/releases"


def _api_headers(accept):
    headers = {"Accept": accept}
    if os.environ.get("GH_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["GH_TOKEN"]
    return headers


def recent_bases(target_version, *, limit=3):
    response = requests.get(RELEASES_API + "?per_page=30",
                            headers=_api_headers("application/vnd.github+json"), timeout=(8,30))
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


def _github_response(session, url, accept, *, asset_redirect=False):
    # Authorization is scoped to the GitHub API. The asset CDN gets no token.
    response = session.get(url, headers=_api_headers(accept), stream=True,
                           timeout=(8, 30), allow_redirects=False)
    if asset_redirect and response.is_redirect:
        location = response.headers.get("Location", "")
        response.close()
        destination = urlsplit(location)
        if destination.scheme != "https" or destination.hostname not in APPROVED_DOWNLOAD_HOSTS:
            raise RuntimeError("GitHub redirected the base asset to an unapproved host")
        response = _request(session, location, stream=True)
    if response.status_code != 200:
        status = response.status_code
        response.close()
        raise RuntimeError(f"GitHub base asset request failed (HTTP {status})")
    return response


def _base_release_assets(session, version):
    tag = f"v{SemanticVersion.parse(version)}"
    response = _github_response(session, f"{RELEASES_API}/tags/{tag}", "application/vnd.github+json")
    metadata = json.loads(_read_bounded_response(response, 512*1024, "base release"))
    if metadata["tag_name"] != tag or metadata["draft"] or metadata["prerelease"]:
        raise RuntimeError("Patch base must be the exact published stable release")
    return metadata["assets"]


def _base_asset_response(session, assets, name):
    matches = [asset["id"] for asset in assets if asset["name"] == name]
    if len(matches) != 1 or type(matches[0]) is not int or matches[0] < 1:
        raise RuntimeError(f"Missing or ambiguous patch base asset: {name}")
    # Use GitHub's asset API instead of the browser download route. The signed
    # manifest still describes the exact canonical browser URL for clients.
    return _github_response(session, f"{RELEASES_API}/assets/{matches[0]}?download=1",
                            "application/octet-stream", asset_redirect=True)


def signed_base(directory, version, platform_name, assets):
    with requests.Session() as session:
        manifest = _read_bounded_response(_base_asset_response(session, assets, "update-manifest.json"),
                                          512*1024, "base manifest")
        signature = _read_bounded_response(_base_asset_response(session, assets, "update-manifest.sig"),
                                           8192, "base signature")
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
        partial = path.with_suffix(".part")
        try:
            with _base_asset_response(session, assets, asset.name) as response, partial.open("wb") as out:
                response.raise_for_status()
                size=0
                for chunk in response.iter_content(1024*1024):
                    size+=len(chunk)
                    if size>asset.size or time.monotonic()>deadline:
                        raise RuntimeError("Signed base download exceeded its size/deadline")
                    out.write(chunk)
            verify_file(partial,asset)
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
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
        with requests.Session() as session:
            assets = _base_release_assets(session, base_version)
        for platform_name,name in RELEASE_FILENAMES.items():
            target=directory/name
            base,asset=signed_base(directory,base_version,platform_name,assets)
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
