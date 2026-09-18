import io
import json
from pathlib import Path
import wave

import httpx
import numpy as np
import pytest

from app.audio_transport import prepare_audio, decode_bounded, AudioLimitError
from app.providers import RemoteWorkerProvider, OpenAITranscriptionProvider, ProviderContext
from app.stt_client import transcribe
from app.stt_response import write_result


def make_audio(path):
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        wav.writeframes((np.sin(np.arange(44100) * .07) * 10000).astype('<i2').tobytes())


@pytest.mark.parametrize('provider', ['worker', 'openai'])
def test_compressed_provider_multipart_preserves_file_bytes(tmp_path, provider):
    path = tmp_path / 'sample.wav'
    make_audio(path)
    seen = []

    def handle(request):
        if request.method == 'GET':
            return httpx.Response(200, json={'status': 'ready', 'role': 'worker'})
        seen.append(request.read())
        return httpx.Response(200, json={'text': 'hello', 'raw_text': 'hello', 'language': 'en'})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        target = (RemoteWorkerProvider('gpu', 'http://worker', 'test-worker-token', client=client)
                  if provider == 'worker' else OpenAITranscriptionProvider(client=client))
        with prepare_audio(path) as audio:
            raw = audio.path.read_bytes()
            target.transcribe(audio.path, ProviderContext(('en',), None, (), False, 'test-key'))
    assert len(seen) == 1
    assert raw in seen[0]
    assert b'audio/flac' in seen[0]
    assert b'recording.flac' in seen[0]
    if provider == 'worker':
        assert b'test-key' not in seen[0]


def test_helper_preserves_strategy_and_schema_without_hermes(tmp_path, monkeypatch):
    original = tmp_path / 'sample.wav'
    make_audio(original)
    output = tmp_path / 'transcript.txt'
    calls = []

    class Reply:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def iter_content(self, _size):
            yield json.dumps({'id': 'fixture', 'text': 'corrected', 'raw_text': 'raw',
                              'provider_used': 'openai-gpt-transcribe'}).encode()

    class Session:
        def post(self, _url, **kwargs):
            calls.append(kwargs)
            assert kwargs['files']['audio'][1].read(4) == b'fLaC'
            return Reply()

    monkeypatch.setenv('CTRLSPEAK_GATEWAY_URL', 'http://gateway.test')
    monkeypatch.delenv('CTRLSPEAK_OPENAI_KEY_FILE', raising=False)
    transcribe(original, 'en', output, session=Session())
    assert calls[0]['data'] == {'strategy': 'ubuntu-gpu-preferred', 'cleanup': 'true', 'allowed_languages': 'en'}
    assert calls[0]['allow_redirects'] is False
    assert output.read_text().strip() == 'corrected'
    sidecar = json.loads(Path(str(output) + '.ctrlspeak.json').read_text())
    assert sidecar['schema_version'] == 2
    assert sidecar['ctrlspeak_feedback']['id'] == 'fixture'
    assert sidecar['normalization_applied'] is False


def test_helper_only_selects_proven_gpu_cleanup(tmp_path):
    payload = {'text': 'corrected', 'raw_text': 'raw', 'normalized_text': 'clean',
               'language': 'en', 'provider_used': 'ubuntu-gpu-large-v3-turbo',
               'normalization': {'requested': True, 'applied': True, 'status': 'applied',
                                 'model': 'S1-mini by Superwhisper', 'location': 'ubuntu-worker', 'device': 'cuda'}}
    output = tmp_path / 'result.txt'
    write_result(payload, output)
    assert output.read_text().strip() == 'clean'
    payload['exact_override_id'] = 'override'
    write_result(payload, output)
    assert output.read_text().strip() == 'corrected'


def test_decoded_limits_apply_to_actual_samples(tmp_path):
    path = tmp_path / 'sample.wav'
    make_audio(path)
    with prepare_audio(path) as prepared:
        with pytest.raises(AudioLimitError):
            decode_bounded(prepared.path, max_pcm_bytes=1000)
