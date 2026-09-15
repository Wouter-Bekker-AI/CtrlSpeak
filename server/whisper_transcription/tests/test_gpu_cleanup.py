import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.corrections import CorrectionStore
from app.main import create_app
from app.normalization import (
    MODEL_ATTRIBUTION, NormalizationResult, S1MiniNormalizer, build_worker_normalizer,
    metadata, safe_worker_result,
)
from app.providers import ProviderContext, ProviderRouter, RemoteWorkerProvider


class Backend:
    name, device, compute_type = 'large-v3-turbo', 'cuda', 'float16'

    def load(self):
        pass

    def transcribe(self, *args):
        return {'raw_text': 'hello acme', 'language': 'en', 'segments': []}


class Normalizer:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    async def close(self):
        pass

    async def normalize(self, raw, *, language):
        self.calls += 1
        if self.fail:
            raise RuntimeError('secret-internal-error')
        return NormalizationResult('Hello acme.', metadata(True, 'applied', duration_ms=12))


def upload(**data):
    return {'files': {'audio': ('test.wav', b'RIFF-test', 'audio/wav')}, 'data': data}


@pytest.mark.parametrize('cleanup,expected', [(None, 0), ('false', 0), ('true', 1)])
def test_worker_cleanup_opt_in(tmp_path, cleanup, expected):
    normalizer = Normalizer()
    app = create_app(backend=Backend(), normalizer=normalizer, service_role='worker',
                     worker_token='worker', environ={}, temp_dir=tmp_path/'uploads')
    with TestClient(app, client=('10.1.1.1', 123)) as client:
        assert client.post('/v1/worker/transcribe', **upload(cleanup='true')).status_code == 401
        data = {} if cleanup is None else {'cleanup': cleanup}
        r = client.post('/v1/worker/transcribe', headers={'Authorization': 'Bearer worker'},
                        **upload(**data))
    assert r.status_code == 200
    assert normalizer.calls == expected
    assert r.json()['raw_text'] == 'hello acme'
    assert r.json()['normalization']['applied'] is bool(expected)
    assert r.json()['normalized_text'] == ('Hello acme.' if expected else None)
    assert not list((tmp_path/'uploads').iterdir())


def test_worker_cleanup_failure_keeps_asr(tmp_path):
    app = create_app(backend=Backend(), normalizer=Normalizer(True), service_role='worker',
                     environ={}, temp_dir=tmp_path/'uploads')
    with TestClient(app, client=('127.0.0.1', 123)) as client:
        r = client.post('/v1/worker/transcribe', **upload(cleanup='true'))
    assert r.status_code == 200
    assert r.json()['raw_text'] == 'hello acme'
    assert r.json()['normalization']['status'] == 'runtime_error'
    assert 'secret-internal-error' not in r.text


@pytest.mark.parametrize('provider,requested,applied', [
    ('ubuntu-gpu-large-v3-turbo', True, True),
    ('ubuntu-gpu-large-v3-turbo', False, False),
    ('openai-gpt-transcribe', True, False),
    ('nova-tiny-whisper', True, False),
])
def test_gateway_selects_only_gpu_cleanup_and_preserves_corrections(tmp_path, provider, requested, applied):
    class Provider:
        id = provider
        calls = []

        def describe(self):
            return {'id': self.id, 'status': 'ready'}

        def transcribe(self, path, context):
            self.calls.append(context)
            return {**Backend().transcribe(), 'normalized_text': 'Hello acme.',
                    'normalization': metadata(True, 'applied', duration_ms=5)}

    p = Provider()
    store = CorrectionStore(tmp_path/'corrections.sqlite3')
    rule = store.create_rule(source_phrase='acme', replacement_phrase='ACME', scope='global')
    n = Normalizer(True)
    app = create_app(backend=Backend(), store=store, normalizer=n, service_role='gateway',
                     router=ProviderRouter([p], {'route': (provider,)}, default_strategy='route'),
                     environ={'CTRLSPEAK_NORMALIZATION_ENABLED': 'true'}, temp_dir=tmp_path/'uploads')
    with TestClient(app, client=('127.0.0.1', 123)) as client:
        r = client.post('/v1/transcribe', **upload(cleanup=str(requested).lower()))
        assert client.get('/v1/capabilities').json()['text_normalization']['gateway_cpu_cleanup'] is False
    assert r.status_code == 200
    result = r.json()
    assert result['raw_text'] == 'hello acme'
    assert result['text'] == 'hello ACME'
    assert result['normalized_text'] == ('Hello ACME.' if applied else None)
    assert result['normalization']['applied'] is applied
    assert p.calls[0].cleanup is requested
    assert n.calls == 0
    assert store.get_rule(rule['id'])['use_count'] == 1


def test_remote_transport_cleanup_and_no_openai_key(tmp_path):
    def handle(request):
        if request.method == 'GET':
            return httpx.Response(200, json={'status':'ready','role':'worker','device':'cuda'})
        body = request.read()
        assert b'name="cleanup"\r\n\r\ntrue' in body
        assert b'openai-secret' not in body
        assert 'x-ctrlspeak-openai-key' not in request.headers
        return httpx.Response(200, json=Backend().transcribe())
    path = tmp_path/'audio.wav'
    path.write_bytes(b'RIFF')
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        provider = RemoteWorkerProvider('gpu', 'http://worker', 'worker', client=client)
        provider.transcribe(path, ProviderContext(('en',),None,(),False,'openai-secret',cleanup=True))


@pytest.mark.parametrize('language,text,status', [
    ('af','hello','non_english'), ('en','','empty_input'), ('en','a'*4001,'input_too_long')])
def test_normalizer_bypasses_without_requests(language, text, status):
    async def run():
        def handle(request):
            pytest.fail('No request should be made')
        n=S1MiniNormalizer('http://127.0.0.1:8081/v1/chat/completions',
                           client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        result=await n.normalize(text,language=language)
        await n.close()
        return result
    assert asyncio.run(run()).info['status'] == status


@pytest.mark.parametrize('text,finish,status', [
    ('Hello acme.','stop','applied'), ('hello acme','stop','unchanged'),
    ('<think>secret</think>','stop','unsafe_output'), ('','stop','unsafe_output'),
    ('truncated','length','unsafe_output'), ('x'*1000,'stop','unsafe_output')])
def test_normalizer_validates_output(text,finish,status):
    async def run():
        def handle(request):
            body=json.loads(request.content)
            assert body['chat_template_kwargs']['enable_thinking'] is False
            return httpx.Response(200,json={'choices':[{'message':{'content':text},'finish_reason':finish}]})
        n=S1MiniNormalizer('http://127.0.0.1:8081/v1/chat/completions',
                           client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        result=await n.normalize('hello acme',language='en')
        await n.close()
        return result
    assert asyncio.run(run()).info['status'] == status


def test_hard_deadline_cancels_and_circuit_skips_next_request():
    async def run():
        calls=[]
        async def handle(request):
            calls.append(True)
            await asyncio.sleep(5)
            pytest.fail('deadline did not cancel transport')
        n=S1MiniNormalizer('http://localhost:8081/v1/chat/completions', timeout_seconds=.1,
                           client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        start=time.monotonic()
        result=await n.normalize('hello acme',language='en')
        assert time.monotonic()-start < .6
        assert result.info['status']=='timeout'
        assert (await n.normalize('hello acme',language='en')).info['status']=='unavailable'
        assert len(calls)==1
        await n.close()
    asyncio.run(run())


@pytest.mark.parametrize('role,device', [('gateway','cuda'),('worker','cpu'),('standalone','cuda')])
def test_only_cuda_worker_can_enable_normalizer(role,device):
    assert build_worker_normalizer({'CTRLSPEAK_NORMALIZATION_ENABLED':'true'},role=role,device=device) is None


@pytest.mark.parametrize('change', [
    {'applied':'true'}, {'device':'cpu'}, {'model':'unknown'}, {'location':'gateway'}, {'status':{}},
])
def test_gateway_rejects_untrusted_cleanup_metadata(change):
    info=metadata(True,'applied')
    info.update(change)
    text,meta=safe_worker_result({'raw_text':'hello','normalized_text':'Hello.','normalization':info},
                                requested=True,provider='ubuntu-gpu-large-v3-turbo',language='en')
    assert text is None
    assert meta['applied'] is False


def test_busy_normalizer_skips_queue_and_releases_lock_after_cancel():
    async def run():
        started = asyncio.Event()
        async def handle(request):
            started.set()
            await asyncio.sleep(30)
        n = S1MiniNormalizer('http://localhost:8081/v1/chat/completions',
                            client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        task = asyncio.create_task(n.normalize('hello', language='en'))
        try:
            await asyncio.wait_for(started.wait(), .5)
            assert (await n.normalize('second', language='en')).info['status'] == 'unavailable'
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await n.close()
        assert not n._lock.locked()
    asyncio.run(run())


def test_response_size_limit():
    async def run():
        n = S1MiniNormalizer('http://localhost:8081/v1/chat/completions',
            client=httpx.AsyncClient(transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=b'x' * 65_537))))
        try:
            assert (await n.normalize('hello', language='en')).info['status'] == 'unsafe_output'
        finally:
            await n.close()
    asyncio.run(run())


def test_gateway_exact_override_beats_cleanup(tmp_path):
    class Provider:
        id = 'ubuntu-gpu-large-v3-turbo'
        def describe(self):
            return {'id': self.id, 'status': 'ready'}
        def transcribe(self, path, context):
            return {**Backend().transcribe(), 'normalized_text': 'Hello acme.',
                    'normalization': metadata(True, 'applied')}
    app = create_app(backend=Backend(), service_role='gateway',
        store=CorrectionStore(tmp_path/'rules.sqlite3'), environ={}, temp_dir=tmp_path/'uploads',
        router=ProviderRouter([Provider()], {'route': (Provider.id,)}, default_strategy='route'))
    with TestClient(app, client=('127.0.0.1', 123)) as client:
        first = client.post('/v1/transcribe', **upload(cleanup='true')).json()
        response = client.post(f"/v1/transcriptions/{first['id']}/feedback",
                               json={'confirmed_text': 'My confirmed version', 'rule_ids': []})
        assert response.status_code == 200
        result = client.post('/v1/transcribe', **upload(cleanup='true')).json()
    assert result['text'] == 'My confirmed version'
    assert result['normalized_text'] is None
    assert result['normalization']['status'] == 'exact_override'
    assert result['normalization']['applied'] is False


def test_old_worker_is_backward_compatible():
    text, info = safe_worker_result(Backend().transcribe(), requested=True,
                                   provider='ubuntu-gpu-large-v3-turbo', language='en')
    assert text is None
    assert info['status'] == 'unavailable'
