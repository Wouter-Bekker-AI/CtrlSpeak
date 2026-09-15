"""Stage pinned, hash-verified S1 CUDA assets on the Ubuntu worker. No service changes."""
from pathlib import Path
import hashlib
import subprocess
import tarfile

ROOT = Path.home() / '.config' / 'CtrlSpeak'
STAGE = ROOT / 'temp' / 'v074-s1-assets'
RUNTIME = ROOT / 'cuda' / 's1-mini-b10985'
MODEL = ROOT / 'models' / 's1-mini'
ASSETS = [
    ('https://github.com/ggml-org/llama.cpp/releases/download/b10985/llama-b10985-bin-ubuntu-cuda-12.8-x64.tar.gz',
     '61089382d5c6529f03a6bbb734f634606693416284856fcd55dd230debda1206'),
    ('https://github.com/ggml-org/llama.cpp/releases/download/b10985/cudart-llama-b10985-bin-ubuntu-cuda-12.8-x64.tar.gz',
     'f24e240da2c80fe858ec51678a93cd05d05f96ce79c35fd2b52e704a94cba8de'),
    ('https://huggingface.co/superwhisper/s1-mini-GGUF/resolve/main/s1-mini-q4_k_m.gguf',
     '3b41ebe2502cbd03e811d5d16b022f5ab551eda58d62597d152f89535003c634'),
]


def main():
    for directory in (STAGE, RUNTIME, MODEL):
        directory.mkdir(parents=True, exist_ok=True)
    for url, digest in ASSETS:
        name = url.rsplit('/', 1)[-1]
        dest = (MODEL if name.endswith('.gguf') else STAGE) / name
        def verified():
            if not dest.is_file():
                return False
            with dest.open('rb') as stream:
                value = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    value.update(chunk)
                return value.hexdigest() == digest
        if not verified():
            print('Downloading', name, flush=True)
            subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                            '--connect-timeout', '10', '--max-time', '900',
                            '--output', str(dest), url], check=True, timeout=920)
        if not verified():
            raise RuntimeError('Asset digest mismatch: ' + name)
        print('Verified', name, flush=True)
        if name.endswith('.tar.gz'):
            with tarfile.open(dest) as archive:
                archive.extractall(RUNTIME, filter='data')
    for name in ('README.md', 'LICENSE'):
        subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                        '--max-time', '30', '--output', str(MODEL / name),
                        'https://huggingface.co/superwhisper/s1-mini-GGUF/raw/main/' + name],
                       check=True, timeout=35)
    print('Runtime binaries:', [str(p) for p in RUNTIME.rglob('llama-server')], flush=True)


if __name__ == '__main__':
    main()
