"""Release signing and bounded patch publication, with ephemeral test keys."""
import base64
import hashlib
import json
import os
from argparse import Namespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding,PrivateFormat,NoEncryption

from scripts import release
from utils import update_manager as u
from utils.binary_delta import ALGORITHM
from utils.version import APP_VERSION

pytestmark=pytest.mark.core_headless


def test_signed_release_set_includes_exact_patches_and_rejects_extras(tmp_path,monkeypatch):
    windows=tmp_path/'CtrlSpeak-windows-x86_64.exe';windows.write_bytes(b'w'*10000)
    linux=tmp_path/'CtrlSpeak-linux-x86_64';linux.write_bytes(b'l'*10000)
    linux.chmod(0o755)
    patch=tmp_path/u.delta_asset_name('windows','0.8.1');patch.write_bytes(b'patch bytes')
    entries=[{'algorithm':ALGORITHM,'base_version':'0.8.1','base_size':10000,'base_sha256':'1'*64,
        'target_size':10000,'target_sha256':hashlib.sha256(windows.read_bytes()).hexdigest(),
        'minimum_updater_version':'0.8.2','platform':'windows','architecture':'x86_64','variant':'standard',
        'name':patch.name,'url':u.GITHUB_RELEASE_ROOT+'/releases/download/v'+APP_VERSION+'/'+patch.name,
        'size':patch.stat().st_size,'sha256':hashlib.sha256(patch.read_bytes()).hexdigest()}]
    input_file=tmp_path/'deltas.json';input_file.write_text(json.dumps(entries))
    key=Ed25519PrivateKey.generate()
    monkeypatch.setenv('CTRLSPEAK_UPDATE_SIGNING_KEY',base64.b64encode(key.private_bytes(Encoding.Raw,PrivateFormat.Raw,NoEncryption())).decode())
    original=release.verify_signed_manifest
    monkeypatch.setattr(release,'verify_signed_manifest',lambda *args,**kw:original(*args,**{**kw,'public_key_base64':base64.b64encode(key.public_key().public_bytes_raw()).decode()}))
    release.generate(Namespace(tag='v'+APP_VERSION,windows=str(windows),linux=str(linux),output_dir=str(tmp_path),published_at=None,deltas_file=str(input_file)))
    manifest=json.loads((tmp_path/'update-manifest.json').read_text())
    assert manifest['schema_version']==1 and len(manifest['assets'])==2 and manifest['deltas']==entries
    assert patch.name in (tmp_path/'SHA256SUMS').read_text()
    release.verify(Namespace(directory=str(tmp_path),tag='v'+APP_VERSION,exact_assets=False))
    input_file.unlink()  # Build input is not part of the published asset set.
    release.verify(Namespace(directory=str(tmp_path),tag='v'+APP_VERSION,exact_assets=True))
    (tmp_path/'unexpected.txt').write_text('must not publish')
    with pytest.raises(SystemExit,match='exactly'):release.verify(Namespace(directory=str(tmp_path),tag='v'+APP_VERSION,exact_assets=True))


def test_native_tool_is_pinned_and_bundled_with_license():
    from scripts.release import ROOT
    from utils.delta_tools import verified_tool,tool_path
    assert verified_tool().is_file()
    assert (tool_path().parent/'LICENSE').is_file()
    spec=(ROOT/'packaging/CtrlSpeak_v0.7.spec').read_text()
    assert "'tools/xdelta3'" in spec and "'LICENSE'" in spec and "'utils.binary_delta'" in spec


@pytest.mark.parametrize("platform_name", ("linux", "windows"))
def test_pinned_upstream_archive_can_be_staged_from_an_empty_cache(
    tmp_path, monkeypatch, platform_name,
):
    from scripts import fetch_patch_tool as stage

    spec = stage.TOOLS[platform_name]
    expected = tmp_path / platform_name / spec["binary"]
    monkeypatch.setattr(stage, "tool_path", lambda _platform: expected)

    assert stage.ensure_tool(platform_name) == expected
    assert hashlib.sha256(expected.read_bytes()).hexdigest() == spec["binary_sha256"]
    assert hashlib.sha256((expected.parent / "LICENSE").read_bytes()).hexdigest() == stage.LICENSE_SHA256


def test_tampered_build_archive_is_rejected_before_staging(tmp_path, monkeypatch):
    from scripts import fetch_patch_tool as stage

    vendor = tmp_path / "vendor"
    vendor.mkdir()
    spec = stage.TOOLS["windows"]
    (vendor / spec["archive"]).write_bytes(b"not the pinned archive")
    target = tmp_path / "cache" / spec["binary"]
    monkeypatch.setattr(stage, "VENDOR", vendor)
    monkeypatch.setattr(stage, "tool_path", lambda _platform: target)

    with pytest.raises(RuntimeError, match="pinned SHA-256"):
        stage.ensure_tool("windows")
    assert not target.exists()


def test_future_patch_generation_chooses_recent_patch_aware_bases_only(monkeypatch):
    from scripts import generate_deltas as generator
    class Response:
        def raise_for_status(self):pass
        def json(self):return [
            {'tag_name':'v0.8.1','draft':False,'prerelease':False},
            {'tag_name':'v0.8.2','draft':False,'prerelease':False},
            {'tag_name':'v0.8.3','draft':False,'prerelease':False},
            {'tag_name':'v0.8.4','draft':True,'prerelease':False},
            {'tag_name':'v1.0.0','draft':False,'prerelease':True}]
    monkeypatch.setattr(generator.requests,'get',lambda *a,**k:Response())
    assert generator.recent_bases('0.8.2')==['0.8.1']
    assert generator.recent_bases('0.8.4')==['0.8.3','0.8.2']
    assert generator.recent_bases('0.8.4',limit=1)==['0.8.3']


def test_build_reads_exact_asset_id_through_github_api(monkeypatch):
    from scripts import generate_deltas as generator

    monkeypatch.delenv("GH_TOKEN", raising=False)
    calls = []

    def request(session, url, accept, *, asset_redirect):
        calls.append((url, accept, asset_redirect))
        return "response"

    monkeypatch.setattr(generator, "_github_response", request)
    base_assets = [{"name": "update-manifest.sig", "id": 42}]
    assert generator._base_asset_response(None, base_assets, "update-manifest.sig") == "response"
    assert calls == [
        (f"{generator.RELEASES_API}/assets/42?download=1", "application/octet-stream", True)
    ]
    with pytest.raises(RuntimeError, match="ambiguous"):
        generator._base_asset_response(None, base_assets * 2, "update-manifest.sig")


def test_build_checks_redirect_before_contacting_asset_host(monkeypatch):
    from types import SimpleNamespace
    from scripts import generate_deltas as generator

    redirect = SimpleNamespace(is_redirect=True, headers={"Location": "https://example.test/file"},
                               close=lambda: None)
    session = SimpleNamespace(get=lambda *args, **kwargs: redirect)
    monkeypatch.setattr(generator, "_request", lambda *args, **kwargs: pytest.fail("unsafe redirect followed"))
    with pytest.raises(RuntimeError, match="unapproved"):
        generator._github_response(session, generator.RELEASES_API, "application/octet-stream",
                                   asset_redirect=True)
