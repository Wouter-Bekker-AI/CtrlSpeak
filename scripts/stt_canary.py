#!/usr/bin/env python3
"""Non-personal release canaries, run only on Nova using its protected env."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import time

import requests
from app.audio_transport import prepare_audio
from app.stt_client import transcribe


def main():
    root = Path(__file__).resolve().parents[1]
    fixture = root / 'tests/fixtures/transport-speech.wav'
    headers = {'Authorization': 'Bearer ' + os.environ['CTRLSPEAK_CLIENT_TOKEN'],
               'X-CtrlSpeak-OpenAI-Key': Path(os.environ['CTRLSPEAK_OPENAI_KEY_FILE']).read_text().strip()}
    base = os.environ['CTRLSPEAK_GATEWAY_URL'].rstrip('/')
    evidence = []
    with tempfile.TemporaryDirectory(prefix='ctrlspeak-canary-') as temp:
        temp = Path(temp)
        ogg = temp / 'synthetic.ogg'
        # This Opus fixture tests Telegram input compatibility. It is NOT a new
        # lossy encoding step in production (real Telegram clips pass unchanged).
        subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
            '-i', str(fixture), '-c:a', 'libopus', '-b:a', '32k', str(ogg)],
            check=True, timeout=20, capture_output=True)
        with prepare_audio(fixture) as flac:
            assert flac.outcome == 'lossless_flac'
            for provider, audio in [('ubuntu-gpu-large-v3-turbo', flac.path),
                                    ('ubuntu-gpu-large-v3-turbo', ogg),
                                    ('openai-gpt-transcribe', flac.path),
                                    ('openai-gpt-transcribe', ogg),
                                    ('nova-tiny-whisper', flac.path)]:
                started = time.monotonic()
                with audio.open('rb') as handle:
                    response = requests.post(base + '/v1/transcribe', headers=headers,
                        data={'provider': provider, 'allowed_languages': 'en', 'cleanup': 'true'},
                        files={'audio': (audio.name, handle, 'audio/ogg' if audio == ogg else 'audio/flac')},
                        timeout=(5, 150), allow_redirects=False)
                assert response.status_code == 200, provider + ' canary failed HTTP ' + str(response.status_code)
                payload = response.json()
                assert payload['provider_used'] == provider and payload['text'].strip()
                assert payload['language'] == 'en'
                normalization = payload.get('normalization', {})
                assert 's1_cleaned_text' in payload, 'gateway audit response field missing'
                if normalization.get('applied'):
                    assert isinstance(payload['s1_cleaned_text'], str) and payload['s1_cleaned_text']
                    assert isinstance(payload.get('normalized_text'), str) and payload['normalized_text']
                if provider != 'ubuntu-gpu-large-v3-turbo':
                    assert not normalization.get('applied')
                evidence.append({'provider': provider, 'format': audio.suffix,
                    'transcription_id': payload['id'],
                    's1_sha256': hashlib.sha256(payload['s1_cleaned_text'].encode()).hexdigest() if payload.get('s1_cleaned_text') is not None else None,
                    'normalized_sha256': hashlib.sha256(payload['normalized_text'].encode()).hexdigest() if payload.get('normalized_text') is not None else None,
                    'seconds': round(time.monotonic() - started, 2),
                    'cleanup_status': normalization.get('status'), 'text_present': True})
            selected = temp / 'result.txt'
            transcribe(ogg, 'en', selected)
            sidecar = json.loads(Path(str(selected) + '.ctrlspeak.json').read_text())
            assert selected.read_text().strip() and sidecar['schema_version'] == 2
            evidence.append({'external_helper': True, 'sidecar_schema': 2,
                             'original_bytes': flac.original_bytes, 'flac_bytes': flac.upload_bytes})
    print(json.dumps(evidence), flush=True)


if __name__ == '__main__':
    main()
