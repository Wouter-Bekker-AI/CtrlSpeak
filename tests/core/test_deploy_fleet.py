import base64
import json
from pathlib import Path
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
from scripts.deploy_host import verify_request, atomic, Host
from scripts.deploy_fleet import transaction

pytestmark = pytest.mark.core_headless


def signed_request():
    key = Ed25519PrivateKey.generate()
    manifest = {'product': 'ctrlspeak', 'channel': 'stable', 'version': '0.7.5', 'tag': 'v0.7.5',
                'deployment': {'schema': 1, 'transport_protocol': 1, 'revision': 'a' * 40,
                               'roles': dict.fromkeys(('worker', 'gateway', 'helper'), '0.7.5')}}
    raw = json.dumps(manifest).encode()
    public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return {'action': 'stage', 'manifest': base64.b64encode(raw).decode(),
            'signature': base64.b64encode(key.sign(raw)).decode()}, base64.b64encode(public).decode()


def test_signature_and_actions_fail_closed():
    request, public = signed_request()
    assert verify_request(request, public)['version'] == '0.7.5'
    with pytest.raises(Exception):
        verify_request(dict(request, action='shell'), public)
    raw = base64.b64decode(request['manifest']).replace(b'0.7.5', b'0.7.6')
    with pytest.raises(Exception):
        verify_request(dict(request, manifest=base64.b64encode(raw).decode()), public)


@pytest.mark.parametrize('failed_role', ['worker', 'gateway', 'helper'])
def test_stage_failure_never_activates(failed_role):
    calls = []
    def call(role, action):
        calls.append((role, action))
        if role == failed_role:
            raise OSError('offline')
    with pytest.raises(OSError):
        transaction(call)
    assert all(action == 'stage' for _, action in calls)


@pytest.mark.parametrize('failed_role', ['worker', 'gateway', 'helper'])
def test_partial_activation_rolls_back_including_uncertain_host(failed_role):
    calls = []
    def call(role, action):
        calls.append((role, action))
        if role == failed_role and action == 'activate':
            raise TimeoutError()
    with pytest.raises(TimeoutError):
        transaction(call)
    activated = [role for role, action in calls if action == 'activate']
    assert [role for role, action in calls if action == 'rollback'] == activated[::-1]


def test_verified_transaction_has_all_staging_first():
    calls = []
    result = transaction(lambda role, action: calls.append((role, action)))
    assert calls[:3] == [(r, 'stage') for r in ('worker', 'gateway', 'helper')]
    assert result['status'] == 'verified' and len(calls) == 10


def test_atomic_replacement(tmp_path):
    target = tmp_path / 'selector'
    atomic(target, b'old')
    atomic(target, b'new')
    assert target.read_bytes() == b'new'
    assert not list(tmp_path.glob('.selector*'))
