"""Signed differential transfer tests, with no installed app/input/provider access."""
from dataclasses import replace
import base64
import hashlib
import json
from pathlib import Path
import threading

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from utils import update_manager as u
from utils import binary_delta as binary

pytestmark = pytest.mark.core_headless


class Response:
    def __init__(self, data, url, status=200):
        self.data, self.url, self.status_code = data, url, status
        self.headers = {"Content-Length":str(len(data))}

    def iter_content(self, chunk_size):
        for offset in range(0,len(self.data),chunk_size):
            yield self.data[offset:offset+chunk_size]

    def close(self):
        pass


class Session:
    def __init__(self, data):
        self.data, self.headers, self.calls = data, {}, []

    def get(self,url,**kwargs):
        self.calls.append(url)
        return Response(self.data[url],url)


@pytest.fixture
def signed_delta(tmp_path,monkeypatch):
    base=tmp_path/'CtrlSpeak.exe'
    # Native diff, not a fake decoder. Entropy isn't needed for transfer guards.
    base.write_bytes(bytes(range(256))*8192)
    target=base.read_bytes()[:80000]+b'changed application code'+base.read_bytes()[80000:]
    target_path=tmp_path/'target.exe';target_path.write_bytes(target)
    patch=tmp_path/'patch.vcdiff'
    binary.create_patch(base,target_path,patch)
    key=Ed25519PrivateKey.generate()
    public=base64.b64encode(key.public_key().public_bytes_raw()).decode()
    full=u.UpdateAsset('windows','x86_64','standard','CtrlSpeak-windows-x86_64.exe',
        u.GITHUB_RELEASE_ROOT+'/releases/download/v0.8.3/CtrlSpeak-windows-x86_64.exe',len(target),hashlib.sha256(target).hexdigest())
    name=u.delta_asset_name('windows','0.8.2')
    entry={'algorithm':binary.ALGORITHM,'base_version':'0.8.2','base_size':base.stat().st_size,
        'base_sha256':u.sha256_file(base),'target_size':full.size,'target_sha256':full.sha256,
        'minimum_updater_version':'0.8.2','platform':'windows','architecture':'x86_64','variant':'standard',
        'name':name,'url':u.GITHUB_RELEASE_ROOT+'/releases/download/v0.8.3/'+name,
        'size':patch.stat().st_size,'sha256':u.sha256_file(patch)}
    manifest={'schema_version':1,'product':'ctrlspeak','channel':'stable','version':'0.8.3','tag':'v0.8.3',
        'release_url':u.GITHUB_RELEASE_ROOT+'/releases/tag/v0.8.3','minimum_updater_version':'0.5.0',
        'published_at':'2026-10-29T00:00:00Z','assets':[full.__dict__],'deltas':[entry]}
    original=u.verify_signed_manifest
    monkeypatch.setattr(u,'verify_signed_manifest',lambda *args,**kwargs:original(*args,**{**kwargs,'public_key_base64':public}))

    def sign(payload=manifest):
        raw=u.canonical_manifest_bytes(payload)
        signature=base64.b64encode(key.sign(raw))
        decoded,asset,version,minimum=u.verify_signed_manifest(raw,signature,platform_name='windows',architecture='x86_64')
        return u.VerifiedRelease(version,'v0.8.3','Test','',payload['published_at'],payload['release_url'],minimum,
            asset,raw,signature,deltas=u.parse_deltas(decoded,asset))

    release=sign()
    transaction=u.create_transaction(release,root=tmp_path,installation_path=base,source_version='0.8.2')
    session=Session({full.url:target,entry['url']:patch.read_bytes()})
    return type('Fixture',(),dict(base=base,target=target,patch=patch,entry=entry,manifest=manifest,
        release=release,transaction=transaction,session=session,sign=staticmethod(sign)))()


def test_native_delta_downloads_only_patch_and_reconstructs_exact_target(signed_delta):
    f=signed_delta;progress=[];phases=[]
    original=f.base.read_bytes()
    candidate=u.download_release(f.release,f.transaction,session=f.session,
        progress=lambda n,t:progress.append((n,t)),phase=lambda s,m:phases.append(s))
    assert candidate.read_bytes()==f.target
    assert f.base.read_bytes()==original
    assert f.session.calls==[f.entry['url']]
    assert progress[-1]==(f.entry['size'],f.entry['size'])
    assert 'applying_patch' in phases
    journal=u.read_transaction_journal(f.transaction)
    assert journal['transfer_mode']=='delta' and journal['transfer_bytes']==f.entry['size']
    assert journal['asset_size']==len(f.target) and journal['asset_sha256']==hashlib.sha256(f.target).hexdigest()
    assert journal['state']=='ready_to_install'


@pytest.mark.parametrize('change',['hash','size','version'])
def test_wrong_base_uses_full_without_requesting_patch(signed_delta,change):
    f=signed_delta
    if change=='hash':f.base.write_bytes(b'x'*f.base.stat().st_size)
    elif change=='size':f.base.write_bytes(b'x')
    else:u.update_transaction_journal(f.transaction,source_version='0.8.1')
    result=u.download_release(f.release,f.transaction,session=f.session)
    assert result.read_bytes()==f.target and f.session.calls==[f.release.asset.url]
    assert u.read_transaction_journal(f.transaction)['transfer_mode']=='full'


def test_corrupt_patch_falls_back_once_to_signed_full_file(signed_delta):
    f=signed_delta
    f.session.data[f.entry['url']]=b'x'*f.entry['size']
    result=u.download_release(f.release,f.transaction,session=f.session)
    assert result.read_bytes()==f.target
    assert f.session.calls==[f.entry['url'],f.release.asset.url]
    assert u.read_transaction_journal(f.transaction)['delta_fallback_reason']=='hash_mismatch'


def test_wrong_decoder_output_is_never_installable_and_falls_back(signed_delta,monkeypatch):
    f=signed_delta
    monkeypatch.setattr(u,'apply_patch_file',lambda base,patch,output,*args,**kwargs:output.write_bytes(b'not executable'))
    result=u.download_release(f.release,f.transaction,session=f.session)
    assert result.read_bytes()==f.target
    assert f.session.calls==[f.entry['url'],f.release.asset.url]


def test_cancel_during_reconstruction_does_not_fallback_or_touch_base(signed_delta,monkeypatch):
    f=signed_delta;cancel=threading.Event();before=f.base.read_bytes()
    def cancelled(base,patch,output,*args,**kwargs):
        cancel.set();raise binary.PatchCancelled('cancelled')
    monkeypatch.setattr(u,'apply_patch_file',cancelled)
    with pytest.raises(u.UpdateCancelled):u.download_release(f.release,f.transaction,session=f.session,cancel_event=cancel)
    assert f.session.calls==[f.entry['url']]
    assert f.base.read_bytes()==before
    assert u.read_transaction_journal(f.transaction)['state']!='ready_to_install'


@pytest.mark.parametrize('field,value',[
    ('base_sha256','bad'),('base_size',True),('base_version','0.8.3'),('target_sha256','0'*64),
    ('target_size',1),('name','../escape.vcdiff'),('url','http://attacker.invalid/patch'),
    ('size',1024*1024*1024),('minimum_updater_version','invalid')])
def test_invalid_signed_delta_metadata_is_rejected(signed_delta,field,value):
    f=signed_delta;payload=json.loads(json.dumps(f.manifest));payload['deltas'][0][field]=value
    with pytest.raises(u.UpdateError):f.sign(payload)


def test_unknown_patch_algorithm_keeps_full_update_compatible(signed_delta):
    f=signed_delta;payload=json.loads(json.dumps(f.manifest));payload['deltas'][0]['algorithm']='future-algorithm'
    assert f.sign(payload).deltas==()


def test_duplicate_signed_base_is_rejected(signed_delta):
    f=signed_delta;payload=json.loads(json.dumps(f.manifest));payload['deltas'].append(dict(payload['deltas'][0]))
    with pytest.raises(u.UpdateError,match='invalid_delta'):f.sign(payload)


def test_tampered_transaction_cannot_select_its_own_target(signed_delta):
    f=signed_delta;u.update_transaction_journal(f.transaction,asset_sha256='0'*64)
    with pytest.raises(u.UpdateError,match='stale_partial'):u.download_release(f.release,f.transaction,session=f.session)
    assert not f.session.calls


def test_signed_manifest_is_reverified_before_any_patch_transfer(signed_delta):
    f=signed_delta;changed=replace(f.release,signature_bytes=b'A'*88)
    with pytest.raises(u.UpdateError):u.download_release(changed,f.transaction,session=f.session)
    assert not f.session.calls


def test_native_tool_wont_overwrite_base_and_has_bounded_output(tmp_path):
    base=tmp_path/'base';base.write_bytes(b'abc'*1000)
    target=tmp_path/'target';target.write_bytes(base.read_bytes()+b'changed')
    patch=tmp_path/'patch'
    binary.create_patch(base,target,patch)
    with pytest.raises(binary.PatchError):binary.apply_patch_file(base,patch,base,target.stat().st_size)
    with pytest.raises(binary.PatchError):binary.apply_patch_file(base,patch,tmp_path/'output',1)
    assert not (tmp_path/'output').exists() and base.read_bytes()==b'abc'*1000


def test_native_tool_deadline_cleans_output_and_reaps(tmp_path):
    base=tmp_path/'base';base.write_bytes(b'abc'*1000)
    with pytest.raises(binary.PatchError):
        binary.run_tool(['-e','-s',base,base,tmp_path/'output'],output=tmp_path/'output',maximum_output=100000,timeout=0)
    assert not (tmp_path/'output').exists()


def test_verified_patch_is_reused_after_cancel_before_install(signed_delta):
    f=signed_delta
    u.download_release(f.release,f.transaction,session=f.session)
    f.transaction.candidate_path.unlink()
    f.session.calls.clear()
    assert u.download_release(f.release,f.transaction,session=f.session).read_bytes()==f.target
    assert f.session.calls==[]


def test_delta_handoff_installs_only_the_reconstructed_full_target(signed_delta,monkeypatch):
    from types import SimpleNamespace
    from utils import update_helper as helper
    f=signed_delta;before=f.base.read_bytes()
    u.download_release(f.release,f.transaction,session=f.session)
    monkeypatch.setattr(helper,'verify_signed_manifest',u.verify_signed_manifest)
    monkeypatch.setattr(helper.sys,'frozen',True,raising=False)
    monkeypatch.setattr(helper.sys,'platform','win32')
    helper.prepare_update_handoff(f.transaction,executable=f.base,original_pid=1234,
        launch_process=lambda *a,**k:SimpleNamespace(pid=4321))
    assert f.base.read_bytes()==before
    assert f.base.with_name(f.base.name+'.new').read_bytes()==f.target
    assert u.read_transaction_journal(f.transaction)['asset_name']=='CtrlSpeak-windows-x86_64.exe'


def test_completed_cache_pruning_preserves_active_failed_rollback_and_metadata(tmp_path):
    import uuid
    directories=[]
    for index,state in enumerate(['installed','installed','installed','installed','failed','downloading']):
        directory=tmp_path/uuid.uuid4().hex;directory.mkdir();directories.append(directory)
        journal={'transaction_id':directory.name,'product':'ctrlspeak','state':state,
            'health_confirmed':state=='installed','installed_at':f'2026-10-{index+1:02d}T00:00:00Z',
            'operation':'rollback_077' if index==0 else 'update'}
        (directory/'transaction.json').write_text(json.dumps(journal))
        (directory/'previous-executable').write_bytes(b'backup')
        (directory/'verified-candidate').write_bytes(b'candidate')
        (directory/'personal-note.txt').write_text('untouched')
    result=u.prune_completed_update_payloads(root=tmp_path)
    assert result=={'files':2,'bytes':15}
    assert not (directories[1]/'previous-executable').exists()
    for directory in directories:
        assert (directory/'transaction.json').exists() and (directory/'personal-note.txt').read_text()=='untouched'
    for index in (0,2,3,4,5):assert (directories[index]/'previous-executable').exists()


def test_active_update_owner_blocks_overlapping_check_or_download(signed_delta):
    f=signed_delta;entered=threading.Event();release=threading.Event()
    def downloader(*a,**kwargs):
        entered.set()
        if not release.wait(2):raise RuntimeError('test deadline')
        raise u.UpdateCancelled()
    coordinator=u.UpdateCoordinator('0.8.2',downloader=downloader)
    coordinator._release=f.release
    try:
        coordinator.download_async(installation_path=f.base)
        assert entered.wait(1)
        with pytest.raises(u.UpdateError,match='update_busy'):coordinator.download_async(installation_path=f.base)
        with pytest.raises(u.UpdateError,match='update_busy'):coordinator.check_async()
        coordinator.cancel()
        with pytest.raises(u.UpdateError,match='update_busy'):coordinator.download_async(installation_path=f.base)
    finally:
        release.set()
        coordinator._download_worker.join(2)
        assert not coordinator._download_worker.is_alive()


def test_already_cancelled_operation_makes_no_network_request(signed_delta):
    f=signed_delta;cancel=threading.Event();cancel.set()
    with pytest.raises(u.UpdateCancelled):u.download_release(f.release,f.transaction,session=f.session,cancel_event=cancel)
    assert f.session.calls==[]


def test_receipt_survives_restart_without_exposing_paths_or_notes(signed_delta):
    f=signed_delta
    u.download_release(f.release,f.transaction,session=f.session)
    u.update_transaction_journal(f.transaction,state='installed',health_confirmed=True,
        installed_at='2026-10-29T01:00:00Z',private_note='must not be returned')
    result=u.latest_completed_transfer(root=f.transaction.directory.parent)
    assert result=={'version':'0.8.3','transfer_mode':'delta','download_bytes':f.entry['size'],'full_bytes':len(f.target)}
    assert 'private_note' not in json.dumps(result) and str(f.base) not in json.dumps(result)


def test_legacy_full_install_receipt_remains_readable(signed_delta):
    f=signed_delta
    u.update_transaction_journal(f.transaction,state='installed',health_confirmed=True)
    assert u.latest_completed_transfer(root=f.transaction.directory.parent)['transfer_mode']=='full'


def test_future_updater_requirement_uses_full_instead_of_unsupported_delta(signed_delta):
    f=signed_delta;payload=json.loads(json.dumps(f.manifest));payload['deltas'][0]['minimum_updater_version']='9.0.0'
    release=f.sign(payload)
    assert u.select_delta(release,'0.8.2',f.base).selected_delta is None


def test_cache_pruning_never_follows_a_directory_link(tmp_path):
    import os
    outside=tmp_path/'outside';outside.mkdir();(outside/'transaction.json').write_text('{}')
    root=tmp_path/'updates';root.mkdir()
    try:os.symlink(outside,root/('f'*32),target_is_directory=True)
    except OSError:pytest.skip('Creating a directory symlink is unavailable in this test account')
    assert u.prune_completed_update_payloads(root=root)=={'files':0,'bytes':0}
    assert (outside/'transaction.json').read_text()=='{}'


def test_insufficient_space_stops_before_any_download_or_replacement(signed_delta,monkeypatch):
    from types import SimpleNamespace
    f=signed_delta;before=f.base.read_bytes()
    monkeypatch.setattr(u.shutil,'disk_usage',lambda path:SimpleNamespace(free=0))
    with pytest.raises(u.UpdateError,match='insufficient_space'):
        u.download_release(f.release,f.transaction,session=f.session)
    assert not f.session.calls and f.base.read_bytes()==before
